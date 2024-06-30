# `D:\src\scipysrc\scikit-learn\sklearn\ensemble\_weight_boosting.py`

```
"""
Weight Boosting.

This module contains weight boosting estimators for both classification and
regression.

The module structure is the following:

- The `BaseWeightBoosting` base class implements a common ``fit`` method
  for all the estimators in the module. Regression and classification
  only differ from each other in the loss function that is optimized.

- :class:`~sklearn.ensemble.AdaBoostClassifier` implements adaptive boosting
  (AdaBoost-SAMME) for classification problems.

- :class:`~sklearn.ensemble.AdaBoostRegressor` implements adaptive boosting
  (AdaBoost.R2) for regression problems.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real

import numpy as np
from scipy.special import xlogy

from ..base import (
    ClassifierMixin,
    RegressorMixin,
    _fit_context,
    is_classifier,
    is_regressor,
)
from ..metrics import accuracy_score, r2_score
from ..tree import DecisionTreeClassifier, DecisionTreeRegressor
from ..utils import _safe_indexing, check_random_state
from ..utils._param_validation import HasMethods, Interval, StrOptions
from ..utils.extmath import softmax, stable_cumsum
from ..utils.metadata_routing import (
    _raise_for_unsupported_routing,
    _RoutingNotSupportedMixin,
)
from ..utils.validation import (
    _check_sample_weight,
    _num_samples,
    check_is_fitted,
    has_fit_parameter,
)
from ._base import BaseEnsemble

__all__ = [
    "AdaBoostClassifier",
    "AdaBoostRegressor",
]


class BaseWeightBoosting(BaseEnsemble, metaclass=ABCMeta):
    """Base class for AdaBoost estimators.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    _parameter_constraints: dict = {
        "estimator": [HasMethods(["fit", "predict"]), None],
        "n_estimators": [Interval(Integral, 1, None, closed="left")],
        "learning_rate": [Interval(Real, 0, None, closed="neither")],
        "random_state": ["random_state"],
    }

    @abstractmethod
    def __init__(
        self,
        estimator=None,
        *,
        n_estimators=50,
        estimator_params=tuple(),
        learning_rate=1.0,
        random_state=None,
    ):
        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
        )

        self.learning_rate = learning_rate
        self.random_state = random_state

    def _check_X(self, X):
        # Only called to validate X in non-fit methods, therefore reset=False
        return self._validate_data(
            X,
            accept_sparse=["csr", "csc"],
            ensure_2d=True,
            allow_nd=True,
            dtype=None,
            reset=False,
        )

    @_fit_context(
        # AdaBoost*.estimator is not validated yet
        prefer_skip_nested_validation=False
    )
    @abstractmethod
    def fit(self, X, y, sample_weight=None):
        """
        Fit the boosting model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        y : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in regression).

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.

        Returns
        -------
        self : object
            Returns self.
        """
        pass  # Placeholder for fit method, to be implemented in derived classes
    def _boost(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost.

        Warning: This method needs to be overridden by subclasses.

        Parameters
        ----------
        iboost : int
            The index of the current boost iteration.

        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        y : array-like of shape (n_samples,)
            The target values (class labels).

        sample_weight : array-like of shape (n_samples,)
            The current sample weights.

        random_state : RandomState
            The current random number generator

        Returns
        -------
        sample_weight : array-like of shape (n_samples,) or None
            The reweighted sample weights.
            If None then boosting has terminated early.

        estimator_weight : float
            The weight for the current boost.
            If None then boosting has terminated early.

        error : float
            The classification error for the current boost.
            If None then boosting has terminated early.
        """
        pass

    def staged_score(self, X, y, sample_weight=None):
        """Return staged scores for X, y.

        This generator method yields the ensemble score after each iteration of
        boosting and therefore allows monitoring, such as to determine the
        score on a test set after each boost.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        y : array-like of shape (n_samples,)
            Labels for X.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Yields
        ------
        z : float
            Yielded value representing the score after each boosting iteration.
        """
        # Ensure X is in the correct format for boosting
        X = self._check_X(X)

        # Iterate through staged predictions and yield scores
        for y_pred in self.staged_predict(X):
            # Check if the current object is a classifier
            if is_classifier(self):
                # Yield accuracy score for classifiers
                yield accuracy_score(y, y_pred, sample_weight=sample_weight)
            else:
                # Yield R^2 score for regressors
                yield r2_score(y, y_pred, sample_weight=sample_weight)

    @property


注释：
    def feature_importances_(self):
        """
        The impurity-based feature importances.

        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

        Returns
        -------
        feature_importances_ : ndarray of shape (n_features,)
            The feature importances.
        """
        # 检查是否已经拟合了估计器，如果未拟合则引发值错误
        if self.estimators_ is None or len(self.estimators_) == 0:
            raise ValueError(
                "Estimator not fitted, call `fit` before `feature_importances_`."
            )

        try:
            # 计算估计器权重的总和
            norm = self.estimator_weights_.sum()
            # 计算特征重要性作为加权平均的结果
            return (
                sum(
                    weight * clf.feature_importances_
                    for weight, clf in zip(self.estimator_weights_, self.estimators_)
                )
                / norm
            )

        except AttributeError as e:
            # 如果估计器没有 feature_importances_ 属性，则引发属性错误
            raise AttributeError(
                "Unable to compute feature importances "
                "since estimator does not have a "
                "feature_importances_ attribute"
            ) from e
def _samme_proba(estimator, n_classes, X):
    """Calculate algorithm 4, step 2, equation c) of Zhu et al [1].

    References
    ----------
    .. [1] J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost", 2009.

    """
    # 使用给定的分类器预测输入数据 X 的类别概率
    proba = estimator.predict_proba(X)

    # 将概率值限制在一个极小的正数上，确保可以对其取对数
    # 同时修正可能出现的负样本权重导致的负概率元素
    np.clip(proba, np.finfo(proba.dtype).eps, None, out=proba)
    # 对概率值取对数，得到对数概率
    log_proba = np.log(proba)

    # 计算 SAMME 算法中的概率权重
    return (n_classes - 1) * (
        log_proba - (1.0 / n_classes) * log_proba.sum(axis=1)[:, np.newaxis]
    )


class AdaBoostClassifier(
    _RoutingNotSupportedMixin, ClassifierMixin, BaseWeightBoosting
):
    """An AdaBoost classifier.

    An AdaBoost [1]_ classifier is a meta-estimator that begins by fitting a
    classifier on the original dataset and then fits additional copies of the
    classifier on the same dataset but where the weights of incorrectly
    classified instances are adjusted such that subsequent classifiers focus
    more on difficult cases.

    This class implements the algorithm based on [2]_.

    Read more in the :ref:`User Guide <adaboost>`.

    .. versionadded:: 0.14

    Parameters
    ----------
    estimator : object, default=None
        The base estimator from which the boosted ensemble is built.
        Support for sample weighting is required, as well as proper
        ``classes_`` and ``n_classes_`` attributes. If ``None``, then
        the base estimator is :class:`~sklearn.tree.DecisionTreeClassifier`
        initialized with `max_depth=1`.

        .. versionadded:: 1.2
           `base_estimator` was renamed to `estimator`.

    n_estimators : int, default=50
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.
        Values must be in the range `[1, inf)`.

    learning_rate : float, default=1.0
        Weight applied to each classifier at each boosting iteration. A higher
        learning rate increases the contribution of each classifier. There is
        a trade-off between the `learning_rate` and `n_estimators` parameters.
        Values must be in the range `(0.0, inf)`.

    algorithm : {'SAMME', 'SAMME.R'}, default='SAMME.R'
        If 'SAMME.R' then use the SAMME.R real boosting algorithm.
        ``estimator`` must support calculation of class probabilities.
        If 'SAMME' then use the SAMME discrete boosting algorithm.
        The SAMME.R algorithm typically converges faster than SAMME,
        achieving a lower test error with fewer boosting iterations.

        .. deprecated:: 1.4
            `"SAMME.R"` is deprecated and will be removed in version 1.6.
            '"SAMME"' will become the default.
    random_state : int, RandomState instance or None, default=None
        控制每个 `estimator` 在每次 boosting 迭代时给定的随机种子。
        只有当 `estimator` 支持 `random_state` 时才会使用此参数。
        传递一个整数以确保在多次函数调用中产生可重复的输出。
        参见 :term:`Glossary <random_state>`。

    Attributes
    ----------
    estimator_ : estimator
        用于生成集成学习器的基础估计器。

        .. versionadded:: 1.2
           `base_estimator_` 已重命名为 `estimator_`。

    estimators_ : list of classifiers
        拟合后的子估计器的集合。

    classes_ : ndarray of shape (n_classes,)
        类别标签。

    n_classes_ : int
        类别的数量。

    estimator_weights_ : ndarray of floats
        提升集成中每个估计器的权重。

    estimator_errors_ : ndarray of floats
        提升集成中每个估计器的分类错误率。

    feature_importances_ : ndarray of shape (n_features,)
        如果 `estimator` 支持基于不纯度的特征重要性，则显示相应的特征重要性（基于决策树）。

        警告：基于不纯度的特征重要性可能对高基数特征（具有许多唯一值）产生误导。
        可以参考 :func:`sklearn.inspection.permutation_importance` 作为替代方法。

    n_features_in_ : int
        在 :term:`fit` 过程中看到的特征数量。

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        在 :term:`fit` 过程中看到的特征名称。仅在 `X` 具有所有字符串类型的特征名称时定义。

        .. versionadded:: 1.0

    See Also
    --------
    AdaBoostRegressor : AdaBoost 回归器，首先在原始数据集上拟合回归器，然后在调整了实例权重的相同数据集上拟合额外的回归器，根据当前预测的错误调整实例权重。

    GradientBoostingClassifier : GB 以逐步向前的方式构建加法模型。回归树在二项或多项偏差损失函数的负梯度上拟合。二元分类是一个特例，其中仅诱导一个回归树。

    sklearn.tree.DecisionTreeClassifier : 用于分类的非参数监督学习方法。通过从数据特征推断出简单的决策规则来创建预测目标变量的模型。

    References
    ----------
    .. [1] Y. Freund, R. Schapire, "A Decision-Theoretic Generalization of
           on-Line Learning and an Application to Boosting", 1995.

    .. [2] :doi:`J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class adaboost."
           Statistics and its Interface 2.3 (2009): 349-360.
           <10.4310/SII.2009.v2.n3.a8>`
    # TODO(1.6): Modify _parameter_constraints for "algorithm" to only check
    # for "SAMME"
    # 定义参数约束字典，限制 "algorithm" 只能为 {"SAMME", "SAMME.R"} 中的一个
    _parameter_constraints: dict = {
        **BaseWeightBoosting._parameter_constraints,
        "algorithm": [
            StrOptions({"SAMME", "SAMME.R"}),
        ],
    }
    
    # TODO(1.6): Change default "algorithm" value to "SAMME"
    # 定义 AdaBoostClassifier 类，初始化方法设置默认参数，其中 "algorithm" 默认为 "SAMME.R"
    class AdaBoostClassifier(BaseWeightBoosting):
        def __init__(
            self,
            estimator=None,
            *,
            n_estimators=50,
            learning_rate=1.0,
            algorithm="SAMME.R",
            random_state=None,
        ):
            super().__init__(
                estimator=estimator,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=random_state,
            )
    
            # 将传入的 "algorithm" 参数值赋给实例变量 self.algorithm
            self.algorithm = algorithm
    # 校验分类器并设置 estimator_ 属性
    def _validate_estimator(self):
        """Check the estimator and set the estimator_ attribute."""
        # 调用父类方法，验证并设置默认的决策树分类器
        super()._validate_estimator(default=DecisionTreeClassifier(max_depth=1))

        # TODO(1.6): 移除，因为在版本 1.6 中，"algorithm" 参数的 "SAMME.R" 值将被移除
        # SAMME-R 算法需要支持 predict_proba 的基学习器
        if self.algorithm != "SAMME":
            # 发出未来警告，提示 SAMME.R 算法即将移除，建议使用 SAMME 算法
            warnings.warn(
                (
                    "The SAMME.R algorithm (the default) is deprecated and will be"
                    " removed in 1.6. Use the SAMME algorithm to circumvent this"
                    " warning."
                ),
                FutureWarning,
            )
            # 如果基学习器不支持 predict_proba 方法，则抛出类型错误
            if not hasattr(self.estimator_, "predict_proba"):
                raise TypeError(
                    "AdaBoostClassifier with algorithm='SAMME.R' requires "
                    "that the weak learner supports the calculation of class "
                    "probabilities with a predict_proba method.\n"
                    "Please change the base estimator or set "
                    "algorithm='SAMME' instead."
                )

        # 检查基学习器是否支持 sample_weight 参数
        if not has_fit_parameter(self.estimator_, "sample_weight"):
            raise ValueError(
                f"{self.estimator.__class__.__name__} doesn't support sample_weight."
            )

    # TODO(1.6): 重新定义 "_boost" 和 "_boost_discrete" 函数的作用域，使其与版本 1.6 中
    # "algorithm" 参数的默认值 SAMME 相同。因此，不再需要区分这两个函数。（或者在此处调整代码，
    # 如果将使用 SAMME.R 之外的其他算法。）
    def _boost(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost.

        Perform a single boost according to the real multi-class SAMME.R
        algorithm or to the discrete SAMME algorithm and return the updated
        sample weights.

        Parameters
        ----------
        iboost : int
            The index of the current boost iteration.

        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values (class labels).

        sample_weight : array-like of shape (n_samples,)
            The current sample weights.

        random_state : RandomState instance
            The RandomState instance used if the base estimator accepts a
            `random_state` attribute.

        Returns
        -------
        sample_weight : array-like of shape (n_samples,) or None
            The reweighted sample weights.
            If None then boosting has terminated early.

        estimator_weight : float
            The weight for the current boost.
            If None then boosting has terminated early.

        estimator_error : float
            The classification error for the current boost.
            If None then boosting has terminated early.
        """
        # 如果算法选择为 "SAMME.R"，调用 _boost_real 方法执行真实多类 SAMME.R 算法
        if self.algorithm == "SAMME.R":
            return self._boost_real(iboost, X, y, sample_weight, random_state)
        
        else:  # 否则，选择 "SAMME" 算法，调用 _boost_discrete 方法执行离散 SAMME 算法
            return self._boost_discrete(iboost, X, y, sample_weight, random_state)

    # TODO(1.6): 移除此函数。因为 SAMME.R 算法将在 1.6 版本中被弃用，_boost_real 函数将不再使用。
    # 实现使用 SAMME.R 真实算法的单次增强过程
    def _boost_real(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost using the SAMME.R real algorithm."""
        
        # 创建一个基础估算器对象
        estimator = self._make_estimator(random_state=random_state)
        
        # 使用样本加权训练基础估算器
        estimator.fit(X, y, sample_weight=sample_weight)
        
        # 预测样本的类别概率
        y_predict_proba = estimator.predict_proba(X)
        
        # 如果是第一次增强
        if iboost == 0:
            # 设置类别属性为基础估算器的类别属性
            self.classes_ = getattr(estimator, "classes_", None)
            # 记录类别数目
            self.n_classes_ = len(self.classes_)
        
        # 根据类别概率确定预测类别
        y_predict = self.classes_.take(np.argmax(y_predict_proba, axis=1), axis=0)
        
        # 被错误分类的实例
        incorrect = y_predict != y
        
        # 计算估算器的误差分数
        estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))
        
        # 如果估算器的误差小于等于零，则返回权重样本、完美分数和零
        if estimator_error <= 0:
            return sample_weight, 1.0, 0.0
        
        # 构建 y 编码，如 Zhu 等人描述的那样
        n_classes = self.n_classes_
        classes = self.classes_
        y_codes = np.array([-1.0 / (n_classes - 1), 1.0])
        y_coding = y_codes.take(classes == y[:, np.newaxis])
        
        # 调整概率值，确保对数函数定义
        proba = y_predict_proba  # 为提升可读性而设置的别名
        np.clip(proba, np.finfo(proba.dtype).eps, None, out=proba)
        
        # 使用多类别 AdaBoost SAMME.R 算法计算增强权重
        estimator_weight = (
            -1.0
            * self.learning_rate
            * ((n_classes - 1.0) / n_classes)
            * xlogy(y_coding, y_predict_proba).sum(axis=1)
        )
        
        # 如果不是最后一次增强
        if not iboost == self.n_estimators - 1:
            # 只增强正权重样本
            sample_weight *= np.exp(
                estimator_weight * ((sample_weight > 0) | (estimator_weight < 0))
            )
        
        # 返回更新后的样本权重、完美分数和估算器的误差
        return sample_weight, 1.0, estimator_error
    # 实现 SAMME 离散算法的单次增强过程
    def _boost_discrete(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost using the SAMME discrete algorithm."""
        # 使用随机状态创建一个基础估算器
        estimator = self._make_estimator(random_state=random_state)

        # 使用样本权重拟合估算器
        estimator.fit(X, y, sample_weight=sample_weight)

        # 预测结果
        y_predict = estimator.predict(X)

        # 若为第一次增强，设置类别和类别数
        if iboost == 0:
            self.classes_ = getattr(estimator, "classes_", None)
            self.n_classes_ = len(self.classes_)

        # 错误分类的实例
        incorrect = y_predict != y

        # 估算器误差率
        estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))

        # 若分类完全正确，返回结果
        if estimator_error <= 0:
            return sample_weight, 1.0, 0.0

        n_classes = self.n_classes_

        # 若误差率至少与随机猜测一样糟糕，停止增强
        if estimator_error >= 1.0 - (1.0 / n_classes):
            self.estimators_.pop(-1)
            if len(self.estimators_) == 0:
                raise ValueError(
                    "BaseClassifier in AdaBoostClassifier "
                    "ensemble is worse than random, ensemble "
                    "can not be fit."
                )
            return None, None, None

        # 使用多类别 AdaBoost SAMME 算法增强权重
        estimator_weight = self.learning_rate * (
            np.log((1.0 - estimator_error) / estimator_error) + np.log(n_classes - 1.0)
        )

        # 只有在不是最后一次增强时才增加权重
        if not iboost == self.n_estimators - 1:
            # 只增强正权重
            sample_weight = np.exp(
                np.log(sample_weight)
                + estimator_weight * incorrect * (sample_weight > 0)
            )

        return sample_weight, estimator_weight, estimator_error

    # 预测函数，预测输入样本的类别
    def predict(self, X):
        """Predict classes for X.

        The predicted class of an input sample is computed as the weighted mean
        prediction of the classifiers in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes.
        """
        # 调用决策函数获取预测结果
        pred = self.decision_function(X)

        # 若只有两类，则返回预测结果
        if self.n_classes_ == 2:
            return self.classes_.take(pred > 0, axis=0)

        # 返回预测结果中概率最高的类别
        return self.classes_.take(np.argmax(pred, axis=1), axis=0)
    def staged_predict(self, X):
        """Return staged predictions for X.

        The predicted class of an input sample is computed as the weighted mean
        prediction of the classifiers in the ensemble.

        This generator method yields the ensemble prediction after each
        iteration of boosting and therefore allows monitoring, such as to
        determine the prediction on a test set after each boost.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        Yields
        ------
        y : generator of ndarray of shape (n_samples,)
            The predicted classes.
        """
        # Ensure X is valid for prediction
        X = self._check_X(X)

        # Determine the number of classes and retrieve class labels
        n_classes = self.n_classes_
        classes = self.classes_

        # Perform staged prediction based on the number of classes
        if n_classes == 2:
            # For binary classification, compute predictions
            for pred in self.staged_decision_function(X):
                # Yield predicted classes based on decision function
                yield np.array(classes.take(pred > 0, axis=0))
        else:
            # For multi-class classification, compute predictions
            for pred in self.staged_decision_function(X):
                # Yield predicted classes based on argmax of decision function
                yield np.array(classes.take(np.argmax(pred, axis=1), axis=0))

    def decision_function(self, X):
        """Compute the decision function of ``X``.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        Returns
        -------
        score : ndarray of shape of (n_samples, k)
            The decision function of the input samples. The order of
            outputs is the same as that of the :term:`classes_` attribute.
            Binary classification is a special case with ``k == 1``,
            otherwise ``k == n_classes``. For binary classification,
            values closer to -1 or 1 mean more like the first or second
            class in ``classes_``, respectively.
        """
        # Ensure the estimator is fitted
        check_is_fitted(self)
        
        # Validate and prepare X for prediction
        X = self._check_X(X)

        # Determine the number of classes and retrieve class labels
        n_classes = self.n_classes_
        classes = self.classes_[:, np.newaxis]

        # Compute prediction based on the chosen boosting algorithm
        if self.algorithm == "SAMME.R":
            # For SAMME.R algorithm, weights are uniform
            pred = sum(
                _samme_proba(estimator, n_classes, X) for estimator in self.estimators_
            )
        else:  # self.algorithm == "SAMME"
            # For SAMME algorithm, compute weighted predictions
            pred = sum(
                np.where(
                    (estimator.predict(X) == classes).T,
                    w,
                    -1 / (n_classes - 1) * w,
                )
                for estimator, w in zip(self.estimators_, self.estimator_weights_)
            )

        # Normalize predictions by the sum of estimator weights
        pred /= self.estimator_weights_.sum()

        # Adjust predictions for binary classification
        if n_classes == 2:
            pred[:, 0] *= -1
            return pred.sum(axis=1)
        
        # Return the computed decision function
        return pred
    def staged_decision_function(self, X):
        """Compute decision function of ``X`` for each boosting iteration.

        This method allows monitoring (i.e. determine error on testing set)
        after each boosting iteration.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        Yields
        ------
        score : generator of ndarray of shape (n_samples, k)
            The decision function of the input samples. The order of
            outputs is the same of that of the :term:`classes_` attribute.
            Binary classification is a special case with ``k == 1``,
            otherwise ``k == n_classes``. For binary classification,
            values closer to -1 or 1 mean more like the first or second
            class in ``classes_``, respectively.
        """
        # 确保模型已经拟合
        check_is_fitted(self)
        # 对输入数据进行检查和预处理
        X = self._check_X(X)

        # 获取类别数和类别标签
        n_classes = self.n_classes_
        classes = self.classes_[:, np.newaxis]
        pred = None
        norm = 0.0

        # 遍历每个基础估算器及其权重
        for weight, estimator in zip(self.estimator_weights_, self.estimators_):
            norm += weight

            # 根据算法类型进行预测
            if self.algorithm == "SAMME.R":
                # 对于 SAMME.R 算法，所有权重都为 1
                current_pred = _samme_proba(estimator, n_classes, X)
            else:  # elif self.algorithm == "SAMME":
                # 对于 SAMME 算法，根据预测结果计算当前预测值
                current_pred = np.where(
                    (estimator.predict(X) == classes).T,
                    weight,
                    -1 / (n_classes - 1) * weight,
                )

            # 合并当前预测值到总预测值中
            if pred is None:
                pred = current_pred
            else:
                pred += current_pred

            # 根据类别数返回预测结果
            if n_classes == 2:
                # 对二元分类进行特殊处理，生成临时预测值
                tmp_pred = np.copy(pred)
                tmp_pred[:, 0] *= -1
                yield (tmp_pred / norm).sum(axis=1)
            else:
                # 对多类分类直接返回归一化后的预测值
                yield pred / norm

    @staticmethod
    def _compute_proba_from_decision(decision, n_classes):
        """Compute probabilities from the decision function.

        This is based eq. (15) of [1] where:
            p(y=c|X) = exp((1 / K-1) f_c(X)) / sum_k(exp((1 / K-1) f_k(X)))
                     = softmax((1 / K-1) * f(X))

        References
        ----------
        .. [1] J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost",
               2009.
        """
        # 根据决策函数计算概率分布
        if n_classes == 2:
            # 对二元分类进行特殊处理，计算softmax值
            decision = np.vstack([-decision, decision]).T / 2
        else:
            # 对多类分类计算softmax值
            decision /= n_classes - 1
        return softmax(decision, copy=False)
    def predict_proba(self, X):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is computed as
        the weighted mean predicted class probabilities of the classifiers
        in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of
            outputs is the same of that of the :term:`classes_` attribute.
        """
        # 检查模型是否已经拟合
        check_is_fitted(self)
        # 获取分类的数量
        n_classes = self.n_classes_

        # 如果只有一个类别，返回每个样本的概率都为1
        if n_classes == 1:
            return np.ones((_num_samples(X), 1))

        # 调用决策函数来计算决策结果
        decision = self.decision_function(X)
        # 使用决策结果计算类别概率并返回
        return self._compute_proba_from_decision(decision, n_classes)

    def staged_predict_proba(self, X):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is computed as
        the weighted mean predicted class probabilities of the classifiers
        in the ensemble.

        This generator method yields the ensemble predicted class probabilities
        after each iteration of boosting and therefore allows monitoring, such
        as to determine the predicted class probabilities on a test set after
        each boost.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        Yields
        ------
        p : generator of ndarray of shape (n_samples,)
            The class probabilities of the input samples. The order of
            outputs is the same of that of the :term:`classes_` attribute.
        """

        # 获取分类的数量
        n_classes = self.n_classes_

        # 遍历每个 boosting 迭代，生成每个迭代后的类别概率
        for decision in self.staged_decision_function(X):
            yield self._compute_proba_from_decision(decision, n_classes)
    # 预测输入样本 X 的类别对数概率。

    # 计算输入样本的预测类别对数概率，这是基于集成中各个分类器预测类别对数概率的加权平均值。

    # Parameters 参数
    # ----------
    # X : {array-like, sparse matrix} of shape (n_samples, n_features)
    #     训练输入样本。稀疏矩阵可以是 CSC、CSR、COO、DOK 或 LIL 格式。COO、DOK 和 LIL 格式将被转换为 CSR 格式。

    # Returns 返回
    # -------
    # p : ndarray of shape (n_samples, n_classes)
    #     输入样本的类别概率。输出的顺序与 :term:`classes_` 属性的顺序相同。
    def predict_log_proba(self, X):
        """Predict class log-probabilities for X.

        The predicted class log-probabilities of an input sample is computed as
        the weighted mean predicted class log-probabilities of the classifiers
        in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of
            outputs is the same of that of the :term:`classes_` attribute.
        """
        # 返回预测类别对数概率的自然对数
        return np.log(self.predict_proba(X))
# 定义 AdaBoostRegressor 类，继承自 _RoutingNotSupportedMixin, RegressorMixin 和 BaseWeightBoosting
class AdaBoostRegressor(_RoutingNotSupportedMixin, RegressorMixin, BaseWeightBoosting):
    """An AdaBoost regressor.

    An AdaBoost [1] regressor is a meta-estimator that begins by fitting a
    regressor on the original dataset and then fits additional copies of the
    regressor on the same dataset but where the weights of instances are
    adjusted according to the error of the current prediction. As such,
    subsequent regressors focus more on difficult cases.

    This class implements the algorithm known as AdaBoost.R2 [2].

    Read more in the :ref:`User Guide <adaboost>`.

    .. versionadded:: 0.14

    Parameters
    ----------
    estimator : object, default=None
        The base estimator from which the boosted ensemble is built.
        If ``None``, then the base estimator is
        :class:`~sklearn.tree.DecisionTreeRegressor` initialized with
        `max_depth=3`.

        .. versionadded:: 1.2
           `base_estimator` was renamed to `estimator`.

    n_estimators : int, default=50
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.
        Values must be in the range `[1, inf)`.

    learning_rate : float, default=1.0
        Weight applied to each regressor at each boosting iteration. A higher
        learning rate increases the contribution of each regressor. There is
        a trade-off between the `learning_rate` and `n_estimators` parameters.
        Values must be in the range `(0.0, inf)`.

    loss : {'linear', 'square', 'exponential'}, default='linear'
        The loss function to use when updating the weights after each
        boosting iteration.

    random_state : int, RandomState instance or None, default=None
        Controls the random seed given at each `estimator` at each
        boosting iteration.
        Thus, it is only used when `estimator` exposes a `random_state`.
        In addition, it controls the bootstrap of the weights used to train the
        `estimator` at each boosting iteration.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    estimator_ : estimator
        The base estimator from which the ensemble is grown.

        .. versionadded:: 1.2
           `base_estimator_` was renamed to `estimator_`.

    estimators_ : list of regressors
        The collection of fitted sub-estimators.

    estimator_weights_ : ndarray of floats
        Weights for each estimator in the boosted ensemble.

    estimator_errors_ : ndarray of floats
        Regression error for each estimator in the boosted ensemble.
    """
    # 初始化方法，定义 AdaBoostRegressor 对象的参数
    def __init__(self, estimator=None, n_estimators=50, learning_rate=1.0, loss='linear', random_state=None):
        # 调用父类的初始化方法
        super().__init__()
        # 设置基础估计器，如果未提供则使用决策树回归器，并指定最大深度为3
        self.estimator = estimator if estimator is not None else DecisionTreeRegressor(max_depth=3)
        # 设置最大的估计器数量
        self.n_estimators = n_estimators
        # 设置学习率，影响每个 boosting 迭代中每个估计器的权重
        self.learning_rate = learning_rate
        # 设置更新权重的损失函数类型
        self.loss = loss
        # 设置随机种子，影响每个估计器的随机性以及每次 boosting 迭代中权重的 bootstrap 过程
        self.random_state = random_state

    # 实现 fit 方法，用于训练 AdaBoostRegressor 模型
    def fit(self, X, y, sample_weight=None):
        # 初始化一些属性
        self.estimators_ = []  # 存储每个 boosting 迭代后的估计器
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)  # 每个估计器的权重
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)  # 每个估计器的误差

        # 获取数据集的样本数量
        n_samples = X.shape[0]

        # 初始化样本权重，如果未提供则初始化为均匀分布
        sample_weight = _check_sample_weight(sample_weight, X)

        # 开始 boosting 迭代
        for iboost in range(self.n_estimators):
            # 在第一次迭代或者在之后迭代中重新校正样本权重
            if iboost == 0:
                sample_weight, estimator_weight, estimator_error = self._boost(
                    iboost,
                    X, y,
                    sample_weight,
                    random_state)
            else:
                sample_weight, estimator_weight, estimator_error = self._boost(
                    iboost,
                    X, y,
                    sample_weight,
                    random_state,
                    self.estimators_[iboost - 1])

            # Early termination
            if sample_weight is None:
                break

            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            # 计算新的样本权重
            sample_weight *= np.exp(estimator_weight *
                                    np.where(y_pred != y, 1, -1))

            # 将估计器添加到列表中
            self.estimators_.append(estimator)

        return self

    # 辅助方法，用于计算每个 boosting 迭代的估计器权重和误差
    def _boost(self, iboost, X, y, sample_weight, random_state, estimator=None):
        if iboost == 0:
            if estimator is None:
                estimator = self.estimator
            estimator.fit(X, y, sample_weight=sample_weight)
        else:
            estimator.fit(X, y, sample_weight=sample_weight)

        # 计算预测值
        y_pred = estimator.predict(X)

        # 更新样本权重
        sample_weight, estimator_weight, estimator_error = (
            sample_weight,
            estimator_weight,
            estimator_error
        )

        return sample_weight, estimator_weight, estimator_error
    # 特征重要性数组，形状为 (n_features,)
    feature_importances_ : ndarray of shape (n_features,)
        如果 `estimator` 支持（基于决策树），则为基于不纯度的特征重要性。

        警告：基于不纯度的特征重要性对于高基数特征（具有许多唯一值）可能具有误导性。参见 :func:`sklearn.inspection.permutation_importance` 作为一个替代方法。

    # 在 `fit` 过程中看到的特征数量
    n_features_in_ : int
        在 `fit` 过程中看到的特征数目。

        .. versionadded:: 0.24

    # 在 `fit` 过程中看到的特征名称数组，形状为 (`n_features_in_`,)
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        在 `fit` 过程中看到的特征名称。仅在 `X` 具有全为字符串的特征名称时定义。

        .. versionadded:: 1.0

    See Also
    --------
    AdaBoostClassifier : 一个 AdaBoost 分类器。
    GradientBoostingRegressor : 梯度提升分类树。
    sklearn.tree.DecisionTreeRegressor : 决策树回归器。

    References
    ----------
    .. [1] Y. Freund, R. Schapire, "A Decision-Theoretic Generalization of
           on-Line Learning and an Application to Boosting", 1995.

    .. [2] H. Drucker, "Improving Regressors using Boosting Techniques", 1997.

    Examples
    --------
    >>> from sklearn.ensemble import AdaBoostRegressor
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_features=4, n_informative=2,
    ...                        random_state=0, shuffle=False)
    >>> regr = AdaBoostRegressor(random_state=0, n_estimators=100)
    >>> regr.fit(X, y)
    AdaBoostRegressor(n_estimators=100, random_state=0)
    >>> regr.predict([[0, 0, 0, 0]])
    array([4.7972...])
    >>> regr.score(X, y)
    0.9771...

    有关使用 :class:`~sklearn.ensemble.AdaBoostRegressor` 拟合一系列决策树作为弱学习器的详细示例，请参阅 :ref:`sphx_glr_auto_examples_ensemble_plot_adaboost_regression.py`。

    """

    # 参数约束字典，继承自 BaseWeightBoosting._parameter_constraints
    _parameter_constraints: dict = {
        **BaseWeightBoosting._parameter_constraints,
        "loss": [StrOptions({"linear", "square", "exponential"})],
    }

    def __init__(
        self,
        estimator=None,
        *,
        n_estimators=50,
        learning_rate=1.0,
        loss="linear",
        random_state=None,
    ):
        # 调用父类构造函数初始化
        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
        )

        # 设置 loss 属性
        self.loss = loss
        # 设置 random_state 属性
        self.random_state = random_state

    def _validate_estimator(self):
        """检查估计器并设置 estimator_ 属性。"""
        # 调用父类方法验证估计器，默认使用最大深度为 3 的决策树回归器
        super()._validate_estimator(default=DecisionTreeRegressor(max_depth=3))
    def _get_median_predict(self, X, limit):
        # Evaluate predictions of all estimators
        predictions = np.array([est.predict(X) for est in self.estimators_[:limit]]).T
        # 将前 limit 个估计器的预测结果组成一个二维数组，每一列代表一个估计器的预测结果

        # Sort the predictions
        sorted_idx = np.argsort(predictions, axis=1)
        # 对预测结果进行排序，返回排序后的索引，axis=1 表示沿着列进行排序

        # Find index of median prediction for each sample
        weight_cdf = stable_cumsum(self.estimator_weights_[sorted_idx], axis=1)
        # 计算加权累积分布函数，用于找到每个样本的中位数预测索引
        median_or_above = weight_cdf >= 0.5 * weight_cdf[:, -1][:, np.newaxis]
        # 找到每个样本中位数及以上的估计器索引
        median_idx = median_or_above.argmax(axis=1)
        # 找到每个样本的中位数预测所对应的估计器索引

        median_estimators = sorted_idx[np.arange(_num_samples(X)), median_idx]
        # 根据中位数预测的索引，从排序后的索引中找到对应的估计器索引

        # Return median predictions
        return predictions[np.arange(_num_samples(X)), median_estimators]
        # 返回每个样本的中位数预测值

    def predict(self, X):
        """Predict regression value for X.

        The predicted regression value of an input sample is computed
        as the weighted median prediction of the regressors in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted regression values.
        """
        check_is_fitted(self)
        # Ensure the estimator is fitted
        X = self._check_X(X)
        # Validate and convert input X

        return self._get_median_predict(X, len(self.estimators_))
        # 返回基于输入 X 的中位数预测回归值

    def staged_predict(self, X):
        """Return staged predictions for X.

        The predicted regression value of an input sample is computed
        as the weighted median prediction of the regressors in the ensemble.

        This generator method yields the ensemble prediction after each
        iteration of boosting and therefore allows monitoring, such as to
        determine the prediction on a test set after each boost.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.

        Yields
        ------
        y : generator of ndarray of shape (n_samples,)
            The predicted regression values.
        """
        check_is_fitted(self)
        # Ensure the estimator is fitted
        X = self._check_X(X)
        # Validate and convert input X

        for i, _ in enumerate(self.estimators_, 1):
            yield self._get_median_predict(X, limit=i)
        # 逐步生成每次增强后的集成预测值
```