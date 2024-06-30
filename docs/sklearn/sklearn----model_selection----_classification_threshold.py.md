# `D:\src\scipysrc\scikit-learn\sklearn\model_selection\_classification_threshold.py`

```
from collections.abc import MutableMapping  # 导入MutableMapping抽象基类，用于定义映射类的接口
from numbers import Integral, Real  # 导入Integral和Real类，用于数字类型检查

import numpy as np  # 导入NumPy库，用于数值计算

from ..base import (  # 导入BaseEstimator等类和函数
    BaseEstimator,
    ClassifierMixin,
    MetaEstimatorMixin,
    _fit_context,
    clone,
)
from ..exceptions import NotFittedError  # 导入NotFittedError异常类
from ..metrics import (  # 导入评估相关函数和类
    check_scoring,
    get_scorer_names,
)
from ..metrics._scorer import _BaseScorer  # 导入基础评分器类
from ..utils import _safe_indexing  # 导入安全索引函数
from ..utils._param_validation import HasMethods, Interval, RealNotInt, StrOptions  # 导入参数验证类
from ..utils._response import _get_response_values_binary  # 导入获取二进制响应值的函数
from ..utils.metadata_routing import (  # 导入元数据路由相关类和函数
    MetadataRouter,
    MethodMapping,
    _raise_for_params,
    process_routing,
)
from ..utils.metaestimators import available_if  # 导入条件激活装饰器
from ..utils.multiclass import type_of_target  # 导入目标类型检查函数
from ..utils.parallel import Parallel, delayed  # 导入并行计算相关类和函数
from ..utils.validation import (  # 导入验证函数和类
    _check_method_params,
    _num_samples,
    check_is_fitted,
    indexable,
)
from ._split import StratifiedShuffleSplit, check_cv  # 导入分割类和交叉验证函数


def _estimator_has(attr):
    """Check if we can delegate a method to the underlying estimator.

    First, we check the fitted estimator if available, otherwise we
    check the unfitted estimator.
    """
    
    def check(self):
        if hasattr(self, "estimator_"):
            getattr(self.estimator_, attr)  # 如果存在已拟合的estimator，调用其方法
        else:
            getattr(self.estimator, attr)  # 否则调用未拟合的estimator的方法
        return True

    return check


def _threshold_scores_to_class_labels(y_score, threshold, classes, pos_label):
    """Threshold `y_score` and return the associated class labels."""
    if pos_label is None:
        map_thresholded_score_to_label = np.array([0, 1])  # 如果没有指定pos_label，则默认类标签为0和1
    else:
        pos_label_idx = np.flatnonzero(classes == pos_label)[0]  # 获取正类标签的索引
        neg_label_idx = np.flatnonzero(classes != pos_label)[0]  # 获取负类标签的索引
        map_thresholded_score_to_label = np.array([neg_label_idx, pos_label_idx])  # 创建类标签映射数组

    return classes[map_thresholded_score_to_label[(y_score >= threshold).astype(int)]]  # 返回阈值处理后的类标签


class BaseThresholdClassifier(ClassifierMixin, MetaEstimatorMixin, BaseEstimator):
    """Base class for binary classifiers that set a non-default decision threshold.

    In this base class, we define the following interface:

    - the validation of common parameters in `fit`;
    - the different prediction methods that can be used with the classifier.

    .. versionadded:: 1.5

    Parameters
    ----------
    estimator : estimator instance
        The binary classifier, fitted or not, for which we want to optimize
        the decision threshold used during `predict`.
    """
    response_method : {"auto", "decision_function", "predict_proba"}, default="auto"
        分类器 `estimator` 对应的决策函数方法，用于找到阈值的方法。可以是以下之一：

        * 如果为 `"auto"`，则尝试依次调用每个分类器的 `"predict_proba"` 或 `"decision_function"`。
        * 否则，为 `"predict_proba"` 或 `"decision_function"` 中的一个。
          如果分类器未实现该方法，则会引发错误。
    """

    _required_parameters = ["estimator"]
    _parameter_constraints: dict = {
        "estimator": [
            HasMethods(["fit", "predict_proba"]),
            HasMethods(["fit", "decision_function"]),
        ],
        "response_method": [StrOptions({"auto", "predict_proba", "decision_function"})],
    }

    def __init__(self, estimator, *, response_method="auto"):
        self.estimator = estimator
        self.response_method = response_method

    def _get_response_method(self):
        """定义响应方法。"""
        if self.response_method == "auto":
            response_method = ["predict_proba", "decision_function"]
        else:
            response_method = self.response_method
        return response_method

    @_fit_context(
        # *ThresholdClassifier*.estimator is not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(self, X, y, **params):
        """拟合分类器。

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            训练数据。

        y : array-like of shape (n_samples,)
            目标值。

        **params : dict
            传递给底层分类器的 `fit` 方法的参数。

        Returns
        -------
        self : object
            返回 `self` 的实例。
        """
        _raise_for_params(params, self, None)

        X, y = indexable(X, y)

        y_type = type_of_target(y, input_name="y")
        if y_type != "binary":
            raise ValueError(
                f"Only binary classification is supported. Unknown label type: {y_type}"
            )

        self._fit(X, y, **params)

        if hasattr(self.estimator_, "n_features_in_"):
            self.n_features_in_ = self.estimator_.n_features_in_
        if hasattr(self.estimator_, "feature_names_in_"):
            self.feature_names_in_ = self.estimator_.feature_names_in_

        return self

    @property
    def classes_(self):
        """类标签。"""
        return self.estimator_.classes_

    @available_if(_estimator_has("predict_proba"))
    # 使用训练好的估算器预测输入数据 `X` 的类别概率。

    def predict_proba(self, X):
        """Predict class probabilities for `X` using the fitted estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            训练向量，其中 `n_samples` 是样本数，`n_features` 是特征数。

        Returns
        -------
        probabilities : ndarray of shape (n_samples, n_classes)
            输入样本的类别概率。
        """
        # 确保估算器已经拟合
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(X)

    @available_if(_estimator_has("predict_log_proba"))
    def predict_log_proba(self, X):
        """Predict logarithm class probabilities for `X` using the fitted estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            训练向量，其中 `n_samples` 是样本数，`n_features` 是特征数。

        Returns
        -------
        log_probabilities : ndarray of shape (n_samples, n_classes)
            输入样本的对数类别概率。
        """
        # 确保估算器已经拟合
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_log_proba(X)

    @available_if(_estimator_has("decision_function"))
    def decision_function(self, X):
        """Decision function for samples in `X` using the fitted estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            训练向量，其中 `n_samples` 是样本数，`n_features` 是特征数。

        Returns
        -------
        decisions : ndarray of shape (n_samples,)
            使用拟合的估算器计算的决策函数值。
        """
        # 确保估算器已经拟合
        check_is_fitted(self, "estimator_")
        return self.estimator_.decision_function(X)

    def _more_tags(self):
        return {
            "binary_only": True,
            "_xfail_checks": {
                "check_classifiers_train": "Threshold at probability 0.5 does not hold",
                "check_sample_weights_invariance": (
                    "Due to the cross-validation and sample ordering, removing a sample"
                    " is not strictly equal to putting is weight to zero. Specific unit"
                    " tests are added for TunedThresholdClassifierCV specifically."
                ),
            },
        }
class FixedThresholdClassifier(BaseThresholdClassifier):
    """Binary classifier that manually sets the decision threshold.

    This classifier allows to change the default decision threshold used for
    converting posterior probability estimates (i.e. output of `predict_proba`) or
    decision scores (i.e. output of `decision_function`) into a class label.

    Here, the threshold is not optimized and is set to a constant value.

    Read more in the :ref:`User Guide <FixedThresholdClassifier>`.

    .. versionadded:: 1.5

    Parameters
    ----------
    estimator : estimator instance
        The binary classifier, fitted or not, for which we want to optimize
        the decision threshold used during `predict`.

    threshold : {"auto"} or float, default="auto"
        The decision threshold to use when converting posterior probability estimates
        (i.e. output of `predict_proba`) or decision scores (i.e. output of
        `decision_function`) into a class label. When `"auto"`, the threshold is set
        to 0.5 if `predict_proba` is used as `response_method`, otherwise it is set to
        0 (i.e. the default threshold for `decision_function`).

    pos_label : int, float, bool or str, default=None
        The label of the positive class. Used to process the output of the
        `response_method` method. When `pos_label=None`, if `y_true` is in `{-1, 1}` or
        `{0, 1}`, `pos_label` is set to 1, otherwise an error will be raised.

    response_method : {"auto", "decision_function", "predict_proba"}, default="auto"
        Methods by the classifier `estimator` corresponding to the
        decision function for which we want to find a threshold. It can be:

        * if `"auto"`, it will try to invoke `"predict_proba"` or `"decision_function"`
          in that order.
        * otherwise, one of `"predict_proba"` or `"decision_function"`.
          If the method is not implemented by the classifier, it will raise an
          error.

    prefit : bool, default=False
        Whether a pre-fitted model is expected to be passed into the constructor
        directly or not. If `True`, `estimator` must be a fitted estimator. If `False`,
        `estimator` is fitted and updated by calling `fit`.

        .. versionadded:: 1.6

    Attributes
    ----------
    estimator_ : estimator instance
        The fitted classifier used when predicting.

    classes_ : ndarray of shape (n_classes,)
        The class labels.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

    See Also
    --------
    BaseThresholdClassifier : Base class for threshold classifiers.
    """

    def __init__(self, estimator, threshold="auto", pos_label=None,
                 response_method="auto", prefit=False):
        # 调用父类的构造函数，初始化基础阈值分类器
        super().__init__(estimator, threshold, pos_label, response_method, prefit)

    def fit(self, X, y):
        # 调用基础阈值分类器的fit方法来拟合模型
        self.estimator_.fit(X, y)
        # 返回自身实例，以支持方法链式调用
        return self

    def predict(self, X):
        # 使用基础阈值分类器的预测方法来预测类别
        # 根据设定的阈值，将预测的概率或决策分数转换为类别标签
        return self.estimator_.predict(X, threshold=self.threshold,
                                       pos_label=self.pos_label,
                                       response_method=self.response_method)

    def predict_proba(self, X):
        # 使用基础阈值分类器的预测概率方法来预测类别概率
        # 根据设定的阈值，将预测的概率或决策分数转换为类别标签
        return self.estimator_.predict_proba(X, threshold=self.threshold,
                                             pos_label=self.pos_label,
                                             response_method=self.response_method)

    def decision_function(self, X):
        # 使用基础阈值分类器的决策函数方法来获取决策分数
        # 根据设定的阈值，将预测的概率或决策分数转换为类别标签
        return self.estimator_.decision_function(X, threshold=self.threshold,
                                                 pos_label=self.pos_label,
                                                 response_method=self.response_method)
    _parameter_constraints: dict = {
        **BaseThresholdClassifier._parameter_constraints,
        "threshold": [StrOptions({"auto"}), Real],
        "pos_label": [Real, str, "boolean", None],
        "prefit": ["boolean"],
    }


# 定义参数约束字典，继承自基类 BaseThresholdClassifier 的参数约束，并添加特定于 TunedThresholdClassifierCV 的约束。
_parameter_constraints: dict = {
    **BaseThresholdClassifier._parameter_constraints,
    "threshold": [StrOptions({"auto"}), Real],  # 约束 threshold 参数可以是字符串选项 "auto" 或实数
    "pos_label": [Real, str, "boolean", None],  # 约束 pos_label 参数可以是实数、字符串、布尔值或 None
    "prefit": ["boolean"],  # 约束 prefit 参数为布尔值
}



    def __init__(
        self,
        estimator,
        *,
        threshold="auto",
        pos_label=None,
        response_method="auto",
        prefit=False,
    ):
        super().__init__(estimator=estimator, response_method=response_method)
        self.pos_label = pos_label
        self.threshold = threshold
        self.prefit = prefit


# 初始化 TunedThresholdClassifierCV 类的实例

def __init__(
    self,
    estimator,
    *,
    threshold="auto",        # 设置默认的 threshold 参数为 "auto"
    pos_label=None,          # 设置默认的 pos_label 参数为 None
    response_method="auto",  # 设置默认的 response_method 参数为 "auto"
    prefit=False,            # 设置默认的 prefit 参数为 False
):
    # 调用父类的初始化方法，传入 estimator 和 response_method 参数
    super().__init__(estimator=estimator, response_method=response_method)
    self.pos_label = pos_label  # 将传入的 pos_label 参数赋值给实例变量
    self.threshold = threshold  # 将传入的 threshold 参数赋值给实例变量
    self.prefit = prefit        # 将传入的 prefit 参数赋值给实例变量



    def _fit(self, X, y, **params):
        """Fit the classifier.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        **params : dict
            Parameters to pass to the `fit` method of the underlying
            classifier.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        routed_params = process_routing(self, "fit", **params)
        if self.prefit:
            check_is_fitted(self.estimator)
            self.estimator_ = self.estimator
        else:
            self.estimator_ = clone(self.estimator).fit(
                X, y, **routed_params.estimator.fit
            )
        return self


# 定义 TunedThresholdClassifierCV 类的 _fit 方法，用于训练分类器

def _fit(self, X, y, **params):
    """Fit the classifier.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        训练数据。

    y : array-like of shape (n_samples,)
        目标值。

    **params : dict
        传递给底层分类器的 fit 方法的参数。

    Returns
    -------
    self : object
        返回自身的实例。
    """
    routed_params = process_routing(self, "fit", **params)  # 路由处理参数
    if self.prefit:
        check_is_fitted(self.estimator)  # 检查估计器是否已经拟合
        self.estimator_ = self.estimator  # 直接使用传入的估计器
    else:
        self.estimator_ = clone(self.estimator).fit(
            X, y, **routed_params.estimator.fit
        )  # 使用克隆的估计器拟合数据
    return self
    # 预测新样本的目标类别。

    def predict(self, X):
        """Predict the target of new samples.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The samples, as accepted by `estimator.predict`.

        Returns
        -------
        class_labels : ndarray of shape (n_samples,)
            The predicted class.
        """
        # 检查模型是否已经拟合
        check_is_fitted(self, "estimator_")
        
        # 获取预测分数、响应方法及其使用的方法
        y_score, _, response_method_used = _get_response_values_binary(
            self.estimator_,
            X,
            self._get_response_method(),
            pos_label=self.pos_label,
            return_response_method_used=True,
        )

        # 根据阈值设定决策阈值
        if self.threshold == "auto":
            decision_threshold = 0.5 if response_method_used == "predict_proba" else 0.0
        else:
            decision_threshold = self.threshold

        # 将分数转换为类别标签
        return _threshold_scores_to_class_labels(
            y_score, decision_threshold, self.classes_, self.pos_label
        )

    # 获取此对象的元数据路由信息。

    def get_metadata_routing(self):
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        Returns
        -------
        routing : MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        # 创建并返回元数据路由器对象，包含估算器和方法映射信息
        router = MetadataRouter(owner=self.__class__.__name__).add(
            estimator=self.estimator,
            method_mapping=MethodMapping().add(callee="fit", caller="fit"),
        )
        return router
class _CurveScorer(_BaseScorer):
    """Scorer taking a continuous response and output a score for each threshold.

    Parameters
    ----------
    score_func : callable
        The score function to use. It will be called as
        `score_func(y_true, y_pred, **kwargs)`.

    sign : int
        Either 1 or -1 to returns the score with `sign * score_func(estimator, X, y)`.
        Thus, `sign` defined if higher scores are better or worse.

    kwargs : dict
        Additional parameters to pass to the score function.

    thresholds : int or array-like
        Related to the number of decision thresholds for which we want to compute the
        score. If an integer, it will be used to generate `thresholds` thresholds
        uniformly distributed between the minimum and maximum predicted scores. If an
        array-like, it will be used as the thresholds.

    response_method : str
        The method to call on the estimator to get the response values.
    """

    def __init__(self, score_func, sign, kwargs, thresholds, response_method):
        # 调用父类构造函数初始化
        super().__init__(
            score_func=score_func,
            sign=sign,
            kwargs=kwargs,
            response_method=response_method,
        )
        # 设置阈值属性
        self._thresholds = thresholds

    @classmethod
    def from_scorer(cls, scorer, response_method, thresholds):
        """Create a continuous scorer from a normal scorer."""
        # 使用给定的评分器创建一个连续评分器实例
        instance = cls(
            score_func=scorer._score_func,
            sign=scorer._sign,
            response_method=response_method,
            thresholds=thresholds,
            kwargs=scorer._kwargs,
        )
        # 转移元数据请求
        instance._metadata_request = scorer._get_metadata_request()
        return instance
    def _score(self, method_caller, estimator, X, y_true, **kwargs):
        """Evaluate predicted target values for X relative to y_true.

        Parameters
        ----------
        method_caller : callable
            返回预测值的方法，给定估算器、方法名称和其他参数，可能会缓存结果。

        estimator : object
            已训练的估算器，用于评分。

        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            将输入估算器.predict 的测试数据。

        y_true : array-like of shape (n_samples,)
            X 的标准目标值。

        **kwargs : dict
            传递给评分器的其他参数。有关详细信息，请参阅 set_score_request 函数。

        Returns
        -------
        scores : ndarray of shape (thresholds,)
            与每个阈值相关联的分数。

        potential_thresholds : ndarray of shape (thresholds,)
            用于计算分数的潜在阈值。
        """
        pos_label = self._get_pos_label()
        # 调用 method_caller 获取 y_score，该方法会返回预测分数
        y_score = method_caller(
            estimator, self._response_method, X, pos_label=pos_label
        )

        scoring_kwargs = {**self._kwargs, **kwargs}
        if isinstance(self._thresholds, Integral):
            # 如果阈值为整数，则在最小和最大分数之间生成等间隔的潜在阈值
            potential_thresholds = np.linspace(
                np.min(y_score), np.max(y_score), self._thresholds
            )
        else:
            # 否则，将阈值转换为数组
            potential_thresholds = np.asarray(self._thresholds)
        
        # 对每个潜在阈值计算分数
        score_thresholds = [
            self._sign
            * self._score_func(
                y_true,
                _threshold_scores_to_class_labels(
                    y_score, th, estimator.classes_, pos_label
                ),
                **scoring_kwargs,
            )
            for th in potential_thresholds
        ]
        # 返回计算的分数数组和潜在阈值数组
        return np.array(score_thresholds), potential_thresholds
def _fit_and_score_over_thresholds(
    classifier,
    X,
    y,
    *,
    fit_params,
    train_idx,
    val_idx,
    curve_scorer,
    score_params,
):
    """Fit a classifier and compute the scores for different decision thresholds.

    Parameters
    ----------
    classifier : estimator instance
        The classifier to fit and use for scoring. If `classifier` is already fitted,
        it will be used as is.

    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The entire dataset.

    y : array-like of shape (n_samples,)
        The entire target vector.

    fit_params : dict
        Parameters to pass to the `fit` method of the underlying classifier.

    train_idx : ndarray of shape (n_train_samples,) or None
        The indices of the training set. If `None`, `classifier` is expected to be
        already fitted.

    val_idx : ndarray of shape (n_val_samples,)
        The indices of the validation set used to score `classifier`. If `train_idx`,
        the entire set will be used.

    curve_scorer : scorer instance
        The scorer taking `classifier` and the validation set as input and outputting
        decision thresholds and scores as a curve. Note that this is different from
        the usual scorer that outputs a single score value:

        * when `score_method` is one of the four constraint metrics, the curve scorer
          will output a curve of two scores parametrized by the decision threshold, e.g.
          TPR/TNR or precision/recall curves for each threshold;
        * otherwise, the curve scorer will output a single score value for each
          threshold.

    score_params : dict
        Parameters to pass to the `score` method of the underlying scorer.

    Returns
    -------
    scores : ndarray of shape (thresholds,) or tuple of such arrays
        The scores computed for each decision threshold. When TPR/TNR or precision/
        recall are computed, `scores` is a tuple of two arrays.

    potential_thresholds : ndarray of shape (thresholds,)
        The decision thresholds used to compute the scores. They are returned in
        ascending order.
    """
    if train_idx is not None:
        # Select training and validation sets based on indices
        X_train, X_val = _safe_indexing(X, train_idx), _safe_indexing(X, val_idx)
        y_train, y_val = _safe_indexing(y, train_idx), _safe_indexing(y, val_idx)
        # Prepare fit parameters specific to training indices
        fit_params_train = _check_method_params(X, fit_params, indices=train_idx)
        # Prepare score parameters specific to validation indices
        score_params_val = _check_method_params(X, score_params, indices=val_idx)
        # Fit the classifier using the training data
        classifier.fit(X_train, y_train, **fit_params_train)
    else:
        # If no training indices are provided, assume the classifier is pre-fitted
        X_val, y_val, score_params_val = X, y, score_params

    # Return scores and potential thresholds computed by the curve scorer
    return curve_scorer(classifier, X_val, y_val, **score_params_val)


def _mean_interpolated_score(target_thresholds, cv_thresholds, cv_scores):
    """Compute the mean interpolated score across folds by defining common thresholds.

    Parameters
    ----------
    target_thresholds : ndarray of shape (n_thresholds,)
        The target thresholds for interpolation.

    cv_thresholds : list of ndarrays
        List of arrays where each array contains the decision thresholds per fold.

    cv_scores : list of ndarrays
        List of arrays where each array contains the scores per fold corresponding
        to the `cv_thresholds`.

    Returns
    -------
    mean_scores : ndarray of shape (n_thresholds,)
        The mean interpolated scores across folds corresponding to `target_thresholds`.
    """
    # 计算每个目标阈值的平均分数
    target_thresholds : ndarray of shape (thresholds,)
        # 目标阈值，用于计算平均分数的阈值数组

    cv_thresholds : ndarray of shape (n_folds, thresholds_fold)
        # 用于每个折叠计算分数的阈值数组，每行代表一个折叠的阈值列表

    cv_scores : ndarray of shape (n_folds, thresholds_fold)
        # 每个折叠对应的每个阈值的分数数组

    Returns
    -------
    mean_score : ndarray of shape (thresholds,)
        # 每个目标阈值在所有折叠中的平均分数数组
    """
    # 使用插值方法计算每个折叠的阈值和分数，然后计算每个目标阈值的平均值
    return np.mean(
        [
            np.interp(target_thresholds, split_thresholds, split_score)
            for split_thresholds, split_score in zip(cv_thresholds, cv_scores)
        ],
        axis=0,
    )
# 定义一个基于交叉验证后调整决策阈值的分类器
class TunedThresholdClassifierCV(BaseThresholdClassifier):
    """Classifier that post-tunes the decision threshold using cross-validation.
    
    This estimator post-tunes the decision threshold (cut-off point) that is
    used for converting posterior probability estimates (i.e. output of
    `predict_proba`) or decision scores (i.e. output of `decision_function`)
    into a class label. The tuning is done by optimizing a binary metric,
    potentially constrained by a another metric.
    
    Read more in the :ref:`User Guide <TunedThresholdClassifierCV>`.
    
    .. versionadded:: 1.5

    Parameters
    ----------
    estimator : estimator instance
        The classifier, fitted or not, for which we want to optimize
        the decision threshold used during `predict`.
        
    scoring : str or callable, default="balanced_accuracy"
        The objective metric to be optimized. Can be one of:
        
        * a string associated to a scoring function for binary classification
          (see :ref:`scoring_parameter`);
        * a scorer callable object created with :func:`~sklearn.metrics.make_scorer`;
          
    response_method : {"auto", "decision_function", "predict_proba"}, default="auto"
        Methods by the classifier `estimator` corresponding to the
        decision function for which we want to find a threshold. It can be:
        
        * if `"auto"`, it will try to invoke, for each classifier,
          `"predict_proba"` or `"decision_function"` in that order.
        * otherwise, one of `"predict_proba"` or `"decision_function"`.
          If the method is not implemented by the classifier, it will raise an
          error.
    
    thresholds : int or array-like, default=100
        The number of decision threshold to use when discretizing the output of the
        classifier `method`. Pass an array-like to manually specify the thresholds
        to use.
    cv : int, float, cross-validation generator, iterable or "prefit", default=None
        # 用于确定交叉验证的分割策略以训练分类器。
        # 可接受的输入为：
        # * `None`，使用默认的5折分层K折交叉验证；
        # * 整数，指定分层K折中的折数；
        # * 浮点数，指定单次随机分割。浮点数应在(0, 1)之间，表示验证集的大小；
        # * 用作交叉验证生成器的对象；
        # * 生成训练和测试分割的可迭代对象；
        # * `"prefit"`，用于跳过交叉验证过程。
        # 
        # 参考 :ref:`用户指南 <cross_validation>` 查看可用的交叉验证策略。

        .. warning::
            # 使用 `cv="prefit"` 并在同一数据集上同时拟合 `estimator` 和调整截断点会导致不良的过拟合。您可以参考 :ref:`TunedThresholdClassifierCV_no_cv` 获取示例。
            # 
            # 此选项应仅在用于拟合 `estimator` 的集合与用于通过调用 :meth:`TunedThresholdClassifierCV.fit` 调整截断点的集合不同时使用。

    refit : bool, default=True
        # 是否在找到决策阈值后重新在整个训练集上拟合分类器。
        # 注意，对于具有多个分割的交叉验证，强制 `refit=False` 将引发错误。同样，`refit=True` 与 `cv="prefit"` 结合使用也会引发错误。

    n_jobs : int, default=None
        # 并行运行的作业数。当 `cv` 表示交叉验证策略时，对每个数据分割的拟合和评分将并行进行。
        # `None` 表示除非在 :obj:`joblib.parallel_backend` 上下文中，否则为1。`-1` 表示使用所有处理器。详细信息请参见 :term:`术语表 <n_jobs>`。

    random_state : int, RandomState instance or None, default=None
        # 当 `cv` 是浮点数时，控制交叉验证的随机性。
        # 请参见 :term:`术语表 <random_state>`。

    store_cv_results : bool, default=False
        # 是否存储交叉验证过程中计算的所有分数和阈值。

    Attributes
    ----------
    estimator_ : estimator instance
        # 用于预测的已拟合分类器实例。

    best_threshold_ : float
        # 新的决策阈值。

    best_score_ : float or None
        # 在 `best_threshold_` 处评估的优化目标度量的最佳分数。

    cv_results_ : dict or None
        # 包含交叉验证过程中计算的分数和阈值的字典。仅在 `store_cv_results=True` 时存在。键为 `"thresholds"` 和 `"scores"`。

    classes_ : ndarray of shape (n_classes,)
        # 类标签。
    # n_features_in_ : int
    #     Number of features seen during :term:`fit`. Only defined if the
    #     underlying estimator exposes such an attribute when fit.

    # feature_names_in_ : ndarray of shape (`n_features_in_`,)
    #     Names of features seen during :term:`fit`. Only defined if the
    #     underlying estimator exposes such an attribute when fit.

    # See Also
    # --------
    # sklearn.model_selection.FixedThresholdClassifier : Classifier that uses a
    #     constant threshold.
    # sklearn.calibration.CalibratedClassifierCV : Estimator that calibrates
    #     probabilities.

    # Examples
    # --------
    # >>> from sklearn.datasets import make_classification
    # >>> from sklearn.ensemble import RandomForestClassifier
    # >>> from sklearn.metrics import classification_report
    # >>> from sklearn.model_selection import TunedThresholdClassifierCV, train_test_split
    # >>> X, y = make_classification(
    # ...     n_samples=1_000, weights=[0.9, 0.1], class_sep=0.8, random_state=42
    # ... )
    # >>> X_train, X_test, y_train, y_test = train_test_split(
    # ...     X, y, stratify=y, random_state=42
    # ... )
    # >>> classifier = RandomForestClassifier(random_state=0).fit(X_train, y_train)
    # >>> print(classification_report(y_test, classifier.predict(X_test)))
    #               precision    recall  f1-score   support
    # <BLANKLINE>
    #            0       0.94      0.99      0.96       224
    #            1       0.80      0.46      0.59        26
    # <BLANKLINE>
    #     accuracy                           0.93       250
    #    macro avg       0.87      0.72      0.77       250
    # weighted avg       0.93      0.93      0.92       250
    # <BLANKLINE>
    # >>> classifier_tuned = TunedThresholdClassifierCV(
    # ...     classifier, scoring="balanced_accuracy"
    # ... ).fit(X_train, y_train)
    # >>> print(
    # ...     f"Cut-off point found at {classifier_tuned.best_threshold_:.3f}"
    # ... )
    # Cut-off point found at 0.342
    # >>> print(classification_report(y_test, classifier_tuned.predict(X_test)))
    #               precision    recall  f1-score   support
    # <BLANKLINE>
    #            0       0.96      0.95      0.96       224
    #            1       0.61      0.65      0.63        26
    # <BLANKLINE>
    #     accuracy                           0.92       250
    #    macro avg       0.78      0.80      0.79       250
    # weighted avg       0.92      0.92      0.92       250
    # <BLANKLINE>
    # 定义参数约束字典，继承自 `BaseThresholdClassifier._parameter_constraints`
    _parameter_constraints: dict = {
        **BaseThresholdClassifier._parameter_constraints,
        # "scoring" 参数要求是以下类型之一：字符串选项集合，可调用对象，可变映射
        "scoring": [
            StrOptions(set(get_scorer_names())),  # 字符串选项集合，使用 `get_scorer_names()` 函数获取
            callable,  # 可调用对象
            MutableMapping,  # 可变映射
        ],
        # "thresholds" 参数要求是以下类型之一：整数区间，左闭右开的数组样式对象
        "thresholds": [Interval(Integral, 1, None, closed="left"), "array-like"],
        # "cv" 参数要求是以下类型之一：CV 对象，字符串选项集合（包含 "prefit"），实数非整数区间 (0.0, 1.0) 的闭区间
        "cv": [
            "cv_object",  # CV 对象
            StrOptions({"prefit"}),  # 字符串选项集合，包含 "prefit"
            Interval(RealNotInt, 0.0, 1.0, closed="neither"),  # 实数非整数区间 (0.0, 1.0) 的闭区间
        ],
        # "refit" 参数要求是布尔值
        "refit": ["boolean"],
        # "n_jobs" 参数要求是整数或者 None
        "n_jobs": [Integral, None],
        # "random_state" 参数要求是 "random_state"
        "random_state": ["random_state"],
        # "store_cv_results" 参数要求是布尔值
        "store_cv_results": ["boolean"],
    }

    # 初始化方法，接受多个参数以配置分类器
    def __init__(
        self,
        estimator,
        *,
        scoring="balanced_accuracy",  # 评分方法，默认为 "balanced_accuracy"
        response_method="auto",  # 响应方法，默认为 "auto"
        thresholds=100,  # 阈值，默认为 100
        cv=None,  # 交叉验证方法，默认为 None
        refit=True,  # 是否重新拟合，默认为 True
        n_jobs=None,  # 并行作业数，默认为 None
        random_state=None,  # 随机状态，默认为 None
        store_cv_results=False,  # 是否存储交叉验证结果，默认为 False
    ):
        # 调用父类的初始化方法，设置估计器和响应方法
        super().__init__(estimator=estimator, response_method=response_method)
        # 设置评分方法
        self.scoring = scoring
        # 设置阈值
        self.thresholds = thresholds
        # 设置交叉验证方法
        self.cv = cv
        # 设置是否重新拟合
        self.refit = refit
        # 设置并行作业数
        self.n_jobs = n_jobs
        # 设置随机状态
        self.random_state = random_state
        # 设置是否存储交叉验证结果
        self.store_cv_results = store_cv_results

    # 预测方法，用于预测新样本的目标值
    def predict(self, X):
        """Predict the target of new samples.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The samples, as accepted by `estimator.predict`.

        Returns
        -------
        class_labels : ndarray of shape (n_samples,)
            The predicted class.
        """
        # 检查模型是否已拟合
        check_is_fitted(self, "estimator_")
        # 获取正例标签
        pos_label = self._curve_scorer._get_pos_label()
        # 获取响应值和分数
        y_score, _ = _get_response_values_binary(
            self.estimator_,
            X,
            self._get_response_method(),
            pos_label=pos_label,
        )

        # 将分数转换为类标签
        return _threshold_scores_to_class_labels(
            y_score, self.best_threshold_, self.classes_, pos_label
        )

    # 获取元数据路由方法，返回元数据路由对象
    def get_metadata_routing(self):
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        Returns
        -------
        routing : MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        # 创建元数据路由对象
        router = (
            MetadataRouter(owner=self.__class__.__name__)
            .add(
                estimator=self.estimator,
                method_mapping=MethodMapping().add(callee="fit", caller="fit"),
            )
            .add(
                splitter=self.cv,
                method_mapping=MethodMapping().add(callee="split", caller="fit"),
            )
            .add(
                scorer=self._get_curve_scorer(),
                method_mapping=MethodMapping().add(callee="score", caller="fit"),
            )
        )
        return router
    # 定义一个方法 `_get_curve_scorer`，用于获取基于所使用的目标度量的曲线评分器
    def _get_curve_scorer(self):
        """Get the curve scorer based on the objective metric used."""
        # 调用 `check_scoring` 函数，传入估计器和评分方式参数，返回评分对象 `scoring`
        scoring = check_scoring(self.estimator, scoring=self.scoring)
        # 使用 `_CurveScorer.from_scorer` 方法，基于评分对象 `scoring`、响应方法和阈值，创建曲线评分器
        curve_scorer = _CurveScorer.from_scorer(
            scoring, self._get_response_method(), self.thresholds
        )
        # 返回创建的曲线评分器对象
        return curve_scorer
```