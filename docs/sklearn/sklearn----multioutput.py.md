# `D:\src\scipysrc\scikit-learn\sklearn\multioutput.py`

```
# 多输出回归和分类。

# 提供的估计器是元估计器：它们在构造函数中需要提供一个基础估计器。元估计器将单输出估计器扩展为多输出估计器。

# 作者：scikit-learn 开发者
# SPDX-License-Identifier: BSD-3-Clause

# 导入必要的库和模块
from abc import ABCMeta, abstractmethod  # 导入抽象基类和抽象方法
from numbers import Integral  # 导入整数类型

import numpy as np  # 导入 NumPy 库
import scipy.sparse as sp  # 导入 SciPy 稀疏矩阵模块

# 导入基类、分类器混合类、元估计器混合类、回归器混合类和其他工具函数
from .base import (
    BaseEstimator,
    ClassifierMixin,
    MetaEstimatorMixin,
    RegressorMixin,
    _fit_context,
    clone,
    is_classifier,
)
# 导入交叉验证预测函数
from .model_selection import cross_val_predict
# 导入 Bunch 类和检查随机状态函数
from .utils import Bunch, check_random_state
# 导入方法验证和字符串选项验证工具函数
from .utils._param_validation import HasMethods, StrOptions
# 导入响应值获取函数
from .utils._response import _get_response_values
# 导入用户界面时间打印函数
from .utils._user_interface import _print_elapsed_time
# 导入元数据路由器、方法映射、参数异常抛出函数、路由器启用检查函数和路由处理函数
from .utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    _raise_for_params,
    _routing_enabled,
    process_routing,
)
# 导入基于条件可用性装饰器
from .utils.metaestimators import available_if
# 导入多类别检查目标函数
from .utils.multiclass import check_classification_targets
# 导入并行处理模块和延迟函数
from .utils.parallel import Parallel, delayed
# 导入验证工具函数
from .utils.validation import (
    _check_method_params,
    _check_response_method,
    check_is_fitted,
    has_fit_parameter,
)

# 声明可以导出的模块列表
__all__ = [
    "MultiOutputRegressor",
    "MultiOutputClassifier",
    "ClassifierChain",
    "RegressorChain",
]

# 定义函数：复制并拟合估计器
def _fit_estimator(estimator, X, y, sample_weight=None, **fit_params):
    estimator = clone(estimator)  # 克隆给定的估计器对象
    if sample_weight is not None:
        estimator.fit(X, y, sample_weight=sample_weight, **fit_params)  # 使用样本权重拟合估计器
    else:
        estimator.fit(X, y, **fit_params)  # 拟合估计器
    return estimator  # 返回拟合后的估计器对象

# 定义函数：部分拟合估计器
def _partial_fit_estimator(
    estimator, X, y, classes=None, partial_fit_params=None, first_time=True
):
    partial_fit_params = {} if partial_fit_params is None else partial_fit_params  # 处理部分拟合参数
    if first_time:
        estimator = clone(estimator)  # 如果是第一次拟合，则克隆估计器对象

    if classes is not None:
        estimator.partial_fit(X, y, classes=classes, **partial_fit_params)  # 使用给定的类别部分拟合估计器
    else:
        estimator.partial_fit(X, y, **partial_fit_params)  # 部分拟合估计器
    return estimator  # 返回部分拟合后的估计器对象

# 定义函数：如果估计器具有指定属性，则返回可用性的函数
def _available_if_estimator_has(attr):
    """Return a function to check if the sub-estimator(s) has(have) `attr`.

    Helper for Chain implementations.
    """

    def _check(self):
        if hasattr(self, "estimators_"):
            return all(hasattr(est, attr) for est in self.estimators_)

        if hasattr(self.estimator, attr):
            return True

        return False

    return available_if(_check)

# 定义抽象类：多输出估计器
class _MultiOutputEstimator(MetaEstimatorMixin, BaseEstimator, metaclass=ABCMeta):
    _parameter_constraints: dict = {
        "estimator": [HasMethods(["fit", "predict"])],  # 估计器必须具有 fit 和 predict 方法
        "n_jobs": [Integral, None],  # n_jobs 参数必须是整数或 None
    }

    @abstractmethod
    def __init__(self, estimator, *, n_jobs=None):
        self.estimator = estimator  # 设置估计器属性
        self.n_jobs = n_jobs  # 设置 n_jobs 属性

    @_available_if_estimator_has("partial_fit")  # 如果估计器具有 partial_fit 属性，可用性装饰器
    @_fit_context(
        # MultiOutput*.estimator is not validated yet
        prefer_skip_nested_validation=False
    )
    @_fit_context(
        # MultiOutput*.estimator is not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(self, X, y, sample_weight=None, **fit_params):
        """Fit the model to data, separately for each output variable.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        y : {array-like, sparse matrix} of shape (n_samples, n_outputs)
            Multi-output targets. An indicator matrix turns on multilabel
            estimation.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If `None`, then samples are equally weighted.
            Only supported if the underlying regressor supports sample
            weights.

        **fit_params : dict of string -> object
            Parameters passed to the ``estimator.fit`` method of each step.

            .. versionadded:: 0.23

        Returns
        -------
        self : object
            Returns a fitted instance.
        """
        # 检查是否有适合的估计器来执行拟合方法
        if not hasattr(self.estimator, "fit"):
            raise ValueError("The base estimator should implement a fit method")

        # 验证数据，不进行X的验证，验证多输出的y
        y = self._validate_data(X="no_validation", y=y, multi_output=True)

        # 如果是分类器，则检查分类目标y的有效性
        if is_classifier(self):
            check_classification_targets(y)

        # 如果y的维度为1，则抛出异常
        if y.ndim == 1:
            raise ValueError(
                "y must have at least two dimensions for "
                "multi-output regression but has only one."
            )

        # 如果启用了路由功能，则处理路由参数
        if _routing_enabled():
            # 如果存在样本权重，则将其添加到fit_params中
            if sample_weight is not None:
                fit_params["sample_weight"] = sample_weight
            # 处理路由参数
            routed_params = process_routing(
                self,
                "fit",
                **fit_params,
            )
        else:
            # 如果存在样本权重且基础估计器不支持样本权重，则引发异常
            if sample_weight is not None and not has_fit_parameter(
                self.estimator, "sample_weight"
            ):
                raise ValueError(
                    "Underlying estimator does not support sample weights."
                )

            # 检查方法参数X和fit_params的有效性
            fit_params_validated = _check_method_params(X, params=fit_params)
            # 创建路由参数Bunch对象
            routed_params = Bunch(estimator=Bunch(fit=fit_params_validated))
            # 如果存在样本权重，则将其添加到fit参数中
            if sample_weight is not None:
                routed_params.estimator.fit["sample_weight"] = sample_weight

        # 使用并行处理拟合过程，返回拟合后的估计器实例列表
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_estimator)(
                self.estimator, X, y[:, i], **routed_params.estimator.fit
            )
            for i in range(y.shape[1])
        )

        # 如果估计器具有'n_features_in_'属性，则设置self.n_features_in_
        if hasattr(self.estimators_[0], "n_features_in_"):
            self.n_features_in_ = self.estimators_[0].n_features_in_
        # 如果估计器具有'feature_names_in_'属性，则设置self.feature_names_in_
        if hasattr(self.estimators_[0], "feature_names_in_"):
            self.feature_names_in_ = self.estimators_[0].feature_names_in_

        # 返回拟合后的实例self
        return self
    def predict(self, X):
        """Predict multi-output variable using model for each target variable.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y : {array-like, sparse matrix} of shape (n_samples, n_outputs)
            Multi-output targets predicted across multiple predictors.
            Note: Separate models are generated for each predictor.
        """
        # 检查模型是否已经拟合（适合）
        check_is_fitted(self)
        # 检查基础估计器是否实现了预测方法
        if not hasattr(self.estimators_[0], "predict"):
            raise ValueError("The base estimator should implement a predict method")

        # 使用并行计算预测结果，每个估计器（模型）都进行预测
        y = Parallel(n_jobs=self.n_jobs)(
            delayed(e.predict)(X) for e in self.estimators_
        )

        # 将预测结果转换为 NumPy 数组并返回，进行转置以匹配多输出的格式
        return np.asarray(y).T

    def _more_tags(self):
        """Returns additional tags related to this estimator."""
        return {"multioutput_only": True}

    def get_metadata_routing(self):
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        .. versionadded:: 1.3

        Returns
        -------
        routing : MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        # 创建并配置元数据路由对象，指定所有者和方法映射
        router = MetadataRouter(owner=self.__class__.__name__).add(
            estimator=self.estimator,
            method_mapping=MethodMapping()
            .add(caller="partial_fit", callee="partial_fit")
            .add(caller="fit", callee="fit"),
        )
        # 返回包含路由信息的 MetadataRouter 对象
        return router
class MultiOutputRegressor(RegressorMixin, _MultiOutputEstimator):
    """Multi target regression.

    This strategy consists of fitting one regressor per target. This is a
    simple strategy for extending regressors that do not natively support
    multi-target regression.

    .. versionadded:: 0.18

    Parameters
    ----------
    estimator : estimator object
        An estimator object implementing :term:`fit` and :term:`predict`.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel.
        :meth:`fit`, :meth:`predict` and :meth:`partial_fit` (if supported
        by the passed estimator) will be parallelized for each target.

        When individual estimators are fast to train or predict,
        using ``n_jobs > 1`` can result in slower performance due
        to the parallelism overhead.

        ``None`` means `1` unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all available processes / threads.
        See :term:`Glossary <n_jobs>` for more details.

        .. versionchanged:: 0.20
            `n_jobs` default changed from `1` to `None`.

    Attributes
    ----------
    estimators_ : list of ``n_output`` estimators
        Estimators used for predictions.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying `estimator` exposes such an attribute when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if the
        underlying estimators expose such an attribute when fit.

        .. versionadded:: 1.0

    See Also
    --------
    RegressorChain : A multi-label model that arranges regressions into a
        chain.
    MultiOutputClassifier : Classifies each output independently rather than
        chaining.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import load_linnerud
    >>> from sklearn.multioutput import MultiOutputRegressor
    >>> from sklearn.linear_model import Ridge
    >>> X, y = load_linnerud(return_X_y=True)
    >>> regr = MultiOutputRegressor(Ridge(random_state=123)).fit(X, y)
    >>> regr.predict(X[[0]])
    array([[176..., 35..., 57...]])
    """

    def __init__(self, estimator, *, n_jobs=None):
        # 调用父类的初始化方法，传递给定的estimator和n_jobs参数
        super().__init__(estimator, n_jobs=n_jobs)

    @_available_if_estimator_has("partial_fit")
    def partial_fit(self, X, y, sample_weight=None, **partial_fit_params):
        """
        Incrementally fit the model to data, for each output variable.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        y : {array-like, sparse matrix} of shape (n_samples, n_outputs)
            Multi-output targets.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If `None`, then samples are equally weighted.
            Only supported if the underlying regressor supports sample
            weights.

        **partial_fit_params : dict of str -> object
            Parameters passed to the ``estimator.partial_fit`` method of each
            sub-estimator.

            Only available if `enable_metadata_routing=True`. See the
            :ref:`User Guide <metadata_routing>`.

            .. versionadded:: 1.3

        Returns
        -------
        self : object
            Returns a fitted instance.
        """
        # 调用父类的 partial_fit 方法，将输入数据和参数传递给基础估计器进行增量拟合
        super().partial_fit(X, y, sample_weight=sample_weight, **partial_fit_params)
# 定义一个多目标分类器类，继承自 ClassifierMixin 和 _MultiOutputEstimator
class MultiOutputClassifier(ClassifierMixin, _MultiOutputEstimator):
    """Multi target classification.

    This strategy consists of fitting one classifier per target. This is a
    simple strategy for extending classifiers that do not natively support
    multi-target classification.

    Parameters
    ----------
    estimator : estimator object
        An estimator object implementing :term:`fit` and :term:`predict`.
        A :term:`predict_proba` method will be exposed only if `estimator` implements
        it.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel.
        :meth:`fit`, :meth:`predict` and :meth:`partial_fit` (if supported
        by the passed estimator) will be parallelized for each target.

        When individual estimators are fast to train or predict,
        using ``n_jobs > 1`` can result in slower performance due
        to the parallelism overhead.

        ``None`` means `1` unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all available processes / threads.
        See :term:`Glossary <n_jobs>` for more details.

        .. versionchanged:: 0.20
            `n_jobs` default changed from `1` to `None`.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels.

    estimators_ : list of ``n_output`` estimators
        Estimators used for predictions.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying `estimator` exposes such an attribute when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if the
        underlying estimators expose such an attribute when fit.

        .. versionadded:: 1.0

    See Also
    --------
    ClassifierChain : A multi-label model that arranges binary classifiers
        into a chain.
    MultiOutputRegressor : Fits one regressor per target variable.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import make_multilabel_classification
    >>> from sklearn.multioutput import MultiOutputClassifier
    >>> from sklearn.linear_model import LogisticRegression
    >>> X, y = make_multilabel_classification(n_classes=3, random_state=0)
    >>> clf = MultiOutputClassifier(LogisticRegression()).fit(X, y)
    >>> clf.predict(X[-2:])
    array([[1, 1, 1],
           [1, 0, 1]])
    """

    # 初始化方法，接受一个估算器对象和一个可选的 n_jobs 参数
    def __init__(self, estimator, *, n_jobs=None):
        # 调用父类 _MultiOutputEstimator 的初始化方法
        super().__init__(estimator, n_jobs=n_jobs)
    # 将模型适配到数据矩阵 X 和目标值 Y 上。
    def fit(self, X, Y, sample_weight=None, **fit_params):
        """Fit the model to data matrix X and targets Y.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Y : array-like of shape (n_samples, n_classes)
            The target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If `None`, then samples are equally weighted.
            Only supported if the underlying classifier supports sample
            weights.

        **fit_params : dict of string -> object
            Parameters passed to the ``estimator.fit`` method of each step.

            .. versionadded:: 0.23

        Returns
        -------
        self : object
            Returns a fitted instance.
        """
        # 调用父类的 fit 方法来适配模型
        super().fit(X, Y, sample_weight=sample_weight, **fit_params)
        # 从每个步骤的 estimator 中提取类别信息，存入 self.classes_
        self.classes_ = [estimator.classes_ for estimator in self.estimators_]
        return self

    # 检查每个 estimator 是否具有 predict_proba 方法
    def _check_predict_proba(self):
        if hasattr(self, "estimators_"):
            # 如果任一 estimator 缺少 predict_proba 方法，抛出 AttributeError
            [getattr(est, "predict_proba") for est in self.estimators_]
            return True
        # 如果未适配的 estimator 缺少 predict_proba 方法，抛出 AttributeError
        getattr(self.estimator, "predict_proba")
        return True

    # 只有在 _check_predict_proba 方法返回 True 时才可用
    @available_if(_check_predict_proba)
    def predict_proba(self, X):
        """Return prediction probabilities for each class of each output.

        This method will raise a ``ValueError`` if any of the
        estimators do not have ``predict_proba``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        p : array of shape (n_samples, n_classes), or a list of n_outputs \
                such arrays if n_outputs > 1.
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.

            .. versionchanged:: 0.19
                This function now returns a list of arrays where the length of
                the list is ``n_outputs``, and each array is (``n_samples``,
                ``n_classes``) for that particular output.
        """
        # 检查模型是否已经适配
        check_is_fitted(self)
        # 对每个 estimator 预测输入数据 X 的类别概率
        results = [estimator.predict_proba(X) for estimator in self.estimators_]
        return results
    # 计算模型在给定测试数据和标签上的平均准确率。

    # 确保模型已经拟合（训练完成）
    check_is_fitted(self)

    # 获取模型中的输出数量
    n_outputs_ = len(self.estimators_)

    # 如果标签 y 的维度为 1，则引发异常，因为多目标分类需要至少二维数组
    if y.ndim == 1:
        raise ValueError(
            "y must have at least two dimensions for "
            "multi target classification but has only one"
        )

    # 如果 y 的第二个维度与模型中的输出数量不匹配，则引发异常
    if y.shape[1] != n_outputs_:
        raise ValueError(
            "The number of outputs of Y for fit {0} and"
            " score {1} should be same".format(n_outputs_, y.shape[1])
        )

    # 使用模型对测试数据 X 进行预测
    y_pred = self.predict(X)

    # 返回预测准确率的平均值
    return np.mean(np.all(y == y_pred, axis=1))


    # 返回一个包含 "_skip_test" 键的字典，值为 True
    def _more_tags(self):
        # FIXME
        return {"_skip_test": True}
def _available_if_base_estimator_has(attr):
    """Return a function to check if `base_estimator` or `estimators_` has `attr`.

    Helper for Chain implementations.
    """

    def _check(self):
        # 检查 `base_estimator` 是否具有 `attr` 属性，或者所有 `estimators_` 都具有该属性
        return hasattr(self.base_estimator, attr) or all(
            hasattr(est, attr) for est in self.estimators_
        )

    return available_if(_check)


class _BaseChain(BaseEstimator, metaclass=ABCMeta):
    _parameter_constraints: dict = {
        "base_estimator": [HasMethods(["fit", "predict"])],
        "order": ["array-like", StrOptions({"random"}), None],
        "cv": ["cv_object", StrOptions({"prefit"})],
        "random_state": ["random_state"],
        "verbose": ["boolean"],
    }

    def __init__(
        self, base_estimator, *, order=None, cv=None, random_state=None, verbose=False
    ):
        # 初始化 `_BaseChain` 类的实例，设置基础估计器、顺序、交叉验证、随机状态和详细模式
        self.base_estimator = base_estimator
        self.order = order
        self.cv = cv
        self.random_state = random_state
        self.verbose = verbose

    def _log_message(self, *, estimator_idx, n_estimators, processing_msg):
        # 如果不是详细模式，返回空；否则返回格式化的处理消息字符串
        if not self.verbose:
            return None
        return f"({estimator_idx} of {n_estimators}) {processing_msg}"

    def _get_predictions(self, X, *, output_method):
        """Get predictions for each model in the chain."""
        # 确保模型已经拟合
        check_is_fitted(self)
        # 验证输入数据 `X`，接受稀疏矩阵，不重置数据
        X = self._validate_data(X, accept_sparse=True, reset=False)
        # 初始化特征链和输出链的预测数组
        Y_output_chain = np.zeros((X.shape[0], len(self.estimators_)))
        Y_feature_chain = np.zeros((X.shape[0], len(self.estimators_)))

        # 如果 `RegressorChain` 没有 `chain_method_` 参数，默认使用 "predict"
        chain_method = getattr(self, "chain_method_", "predict")
        # 根据 `X` 的类型选择水平堆叠方法
        hstack = sp.hstack if sp.issparse(X) else np.hstack
        for chain_idx, estimator in enumerate(self.estimators_):
            # 获取先前链中的预测值
            previous_predictions = Y_feature_chain[:, :chain_idx]
            # 如果 `X` 是稀疏的 dok_array，转换为 coo_array 格式以加快速度
            if sp.issparse(X) and not sp.isspmatrix(X) and X.format == "dok":
                X = sp.coo_array(X)
            # 执行特征扩展
            X_aug = hstack((X, previous_predictions))

            # 获取特征预测和输出预测值
            feature_predictions, _ = _get_response_values(
                estimator,
                X_aug,
                response_method=chain_method,
            )
            Y_feature_chain[:, chain_idx] = feature_predictions

            output_predictions, _ = _get_response_values(
                estimator,
                X_aug,
                response_method=output_method,
            )
            Y_output_chain[:, chain_idx] = output_predictions

        # 构建逆序的输出链
        inv_order = np.empty_like(self.order_)
        inv_order[self.order_] = np.arange(len(self.order_))
        Y_output = Y_output_chain[:, inv_order]

        return Y_output

    @abstractmethod
    def predict(self, X):
        """使用 ClassifierChain 模型对数据矩阵 X 进行预测。

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            输入数据矩阵。

        Returns
        -------
        Y_pred : array-like of shape (n_samples, n_classes)
            预测的值。
        """
        # 调用内部方法 _get_predictions，使用预测模式进行预测
        return self._get_predictions(X, output_method="predict")
class ClassifierChain(MetaEstimatorMixin, ClassifierMixin, _BaseChain):
    """A multi-label model that arranges binary classifiers into a chain.

    Each model makes a prediction in the order specified by the chain using
    all of the available features provided to the model plus the predictions
    of models that are earlier in the chain.

    For an example of how to use ``ClassifierChain`` and benefit from its
    ensemble, see
    :ref:`ClassifierChain on a yeast dataset
    <sphx_glr_auto_examples_multioutput_plot_classifier_chain_yeast.py>` example.

    Read more in the :ref:`User Guide <classifierchain>`.

    .. versionadded:: 0.19

    Parameters
    ----------
    base_estimator : estimator
        The base estimator from which the classifier chain is built.

    order : array-like of shape (n_outputs,) or 'random', default=None
        If `None`, the order will be determined by the order of columns in
        the label matrix Y.::

            order = [0, 1, 2, ..., Y.shape[1] - 1]

        The order of the chain can be explicitly set by providing a list of
        integers. For example, for a chain of length 5.::

            order = [1, 3, 2, 4, 0]

        means that the first model in the chain will make predictions for
        column 1 in the Y matrix, the second model will make predictions
        for column 3, etc.

        If order is `random` a random ordering will be used.

    cv : int, cross-validation generator or an iterable, default=None
        Determines whether to use cross validated predictions or true
        labels for the results of previous estimators in the chain.
        Possible inputs for cv are:

        - None, to use true labels when fitting,
        - integer, to specify the number of folds in a (Stratified)KFold,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

    chain_method : {'predict', 'predict_proba', 'predict_log_proba', \
            'decision_function'} or list of such str's, default='predict'

        Prediction method to be used by estimators in the chain for
        the 'prediction' features of previous estimators in the chain.

        - if `str`, name of the method;
        - if a list of `str`, provides the method names in order of
          preference. The method used corresponds to the first method in
          the list that is implemented by `base_estimator`.

        .. versionadded:: 1.5

    random_state : int, RandomState instance or None, optional (default=None)
        If ``order='random'``, determines random number generation for the
        chain order.
        In addition, it controls the random seed given at each `base_estimator`
        at each chaining iteration. Thus, it is only used when `base_estimator`
        exposes a `random_state`.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
    """
    # verbose : bool, default=False
    #     控制是否输出链式进度信息，每完成一个模型就输出一次。
    #     如果设为 True，则输出每个模型完成的进度信息。
    
    # .. versionadded:: 1.2
    #     版本新增功能说明，指出此参数是从版本 1.2 开始添加的。
    
    Attributes
    ----------
    classes_ : list
        包含链式模型中每个估计器的类标签数组列表，长度为 `len(estimators_)`。
    
    estimators_ : list
        base_estimator 的克隆列表。
    
    order_ : list
        分类器链中标签的顺序。
    
    chain_method_ : str
        用于链式模型中估计器进行特征预测的方法。
    
    n_features_in_ : int
        在拟合期间观察到的特征数量。只有在基础 `base_estimator` 在拟合时暴露了这样的属性时才定义。
    
        .. versionadded:: 0.24
            版本新增功能说明，指出此属性是从版本 0.24 开始添加的。
    
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        在拟合期间看到的特征名称。仅当 `X` 的特征名称全为字符串时才定义。
    
        .. versionadded:: 1.0
            版本新增功能说明，指出此属性是从版本 1.0 开始添加的。
    
    See Also
    --------
    RegressorChain : 用于回归问题的等效链式模型。
    MultiOutputClassifier : 对每个输出进行独立分类，而不是进行链式操作。
    
    References
    ----------
    Jesse Read, Bernhard Pfahringer, Geoff Holmes, Eibe Frank, "Classifier
    Chains for Multi-label Classification", 2009.
        多标签分类的分类器链理论参考文献，详细描述了这种分类方法的原理和应用。
    
    Examples
    --------
    >>> from sklearn.datasets import make_multilabel_classification
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.multioutput import ClassifierChain
    >>> X, Y = make_multilabel_classification(
    ...    n_samples=12, n_classes=3, random_state=0
    ... )
    >>> X_train, X_test, Y_train, Y_test = train_test_split(
    ...    X, Y, random_state=0
    ... )
    >>> base_lr = LogisticRegression(solver='lbfgs', random_state=0)
    >>> chain = ClassifierChain(base_lr, order='random', random_state=0)
    >>> chain.fit(X_train, Y_train).predict(X_test)
    array([[1., 1., 0.],
           [1., 0., 0.],
           [0., 1., 0.]])
    >>> chain.predict_proba(X_test)
    array([[0.8387..., 0.9431..., 0.4576...],
           [0.8878..., 0.3684..., 0.2640...],
           [0.0321..., 0.9935..., 0.0626...]])
    """
    
    _parameter_constraints: dict = {
        **_BaseChain._parameter_constraints,
        "chain_method": [
            list,
            tuple,
            StrOptions(
                {"predict", "predict_proba", "predict_log_proba", "decision_function"}
            ),
        ],
    }
    
    def __init__(
        self,
        base_estimator,
        *,
        order=None,
        cv=None,
        chain_method="predict",
        random_state=None,
        verbose=False,
    ):
        super().__init__(
            base_estimator,
            order=order,
            cv=cv,
            random_state=random_state,
            verbose=verbose,
        )
        self.chain_method = chain_method
    @_fit_context(
        # 根据参数说明，这里指出 ClassifierChain.base_estimator 尚未验证
        prefer_skip_nested_validation=False
    )
    # 定义模型拟合方法，将模型拟合到数据矩阵 X 和目标值 Y 上
    def fit(self, X, Y, **fit_params):
        """Fit the model to data matrix X and targets Y.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            输入数据。

        Y : array-like of shape (n_samples, n_classes)
            目标值。

        **fit_params : dict of string -> object
            传递给每个步骤的 `fit` 方法的参数。

            只有在 `enable_metadata_routing=True` 时才可用。参见
            :ref:`用户指南 <metadata_routing>`。

            .. versionadded:: 1.3

        Returns
        -------
        self : object
            类实例。
        """
        # 检查参数的有效性
        _raise_for_params(fit_params, self, "fit")

        # 调用父类的 fit 方法，传递 X、Y 和额外的 fit_params
        super().fit(X, Y, **fit_params)
        # 设置模型的 classes_ 属性为每个基本估计器的 classes_
        self.classes_ = [estimator.classes_ for estimator in self.estimators_]
        return self

    @_available_if_base_estimator_has("predict_proba")
    # 如果基本估计器具有 "predict_proba" 方法，则可用
    def predict_proba(self, X):
        """Predict probability estimates.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            输入数据。

        Returns
        -------
        Y_prob : array-like of shape (n_samples, n_classes)
            预测的概率估计。
        """
        # 调用内部方法 _get_predictions，使用 "predict_proba" 输出方法
        return self._get_predictions(X, output_method="predict_proba")

    # 预测概率的对数值
    def predict_log_proba(self, X):
        """Predict logarithm of probability estimates.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            输入数据。

        Returns
        -------
        Y_log_prob : array-like of shape (n_samples, n_classes)
            预测的概率对数值。
        """
        # 调用 predict_proba 方法预测概率，然后对结果取对数
        return np.log(self.predict_proba(X))

    @_available_if_base_estimator_has("decision_function")
    # 如果基本估计器具有 "decision_function" 方法，则可用
    def decision_function(self, X):
        """Evaluate the decision_function of the models in the chain.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            输入数据。

        Returns
        -------
        Y_decision : array-like of shape (n_samples, n_classes)
            返回链中每个模型的样本的决策函数。
        """
        # 调用内部方法 _get_predictions，使用 "decision_function" 输出方法
        return self._get_predictions(X, output_method="decision_function")
    def get_metadata_routing(self):
        """获取此对象的元数据路由。

        请参阅 :ref:`用户指南 <metadata_routing>` 了解路由机制的工作原理。

        .. versionadded:: 1.3

        Returns
        -------
        routing : MetadataRouter
            一个 :class:`~sklearn.utils.metadata_routing.MetadataRouter` 对象，
            封装了路由信息。
        """
        # 创建一个 MetadataRouter 对象，设置所有者为当前对象的类名，并添加路由信息
        router = MetadataRouter(owner=self.__class__.__name__).add(
            estimator=self.base_estimator,
            method_mapping=MethodMapping().add(caller="fit", callee="fit"),
        )
        # 返回创建的路由对象
        return router

    def _more_tags(self):
        # 返回一个包含额外标签的字典，用于指示跳过测试和仅支持多输出的信息
        return {"_skip_test": True, "multioutput_only": True}
class RegressorChain(MetaEstimatorMixin, RegressorMixin, _BaseChain):
    """A multi-label model that arranges regressions into a chain.

    Each model makes a prediction in the order specified by the chain using
    all of the available features provided to the model plus the predictions
    of models that are earlier in the chain.

    Read more in the :ref:`User Guide <regressorchain>`.

    .. versionadded:: 0.20

    Parameters
    ----------
    base_estimator : estimator
        The base estimator from which the regressor chain is built.

    order : array-like of shape (n_outputs,) or 'random', default=None
        If `None`, the order will be determined by the order of columns in
        the label matrix Y.::

            order = [0, 1, 2, ..., Y.shape[1] - 1]

        The order of the chain can be explicitly set by providing a list of
        integers. For example, for a chain of length 5.::

            order = [1, 3, 2, 4, 0]

        means that the first model in the chain will make predictions for
        column 1 in the Y matrix, the second model will make predictions
        for column 3, etc.

        If order is 'random' a random ordering will be used.

    cv : int, cross-validation generator or an iterable, default=None
        Determines whether to use cross validated predictions or true
        labels for the results of previous estimators in the chain.
        Possible inputs for cv are:

        - None, to use true labels when fitting,
        - integer, to specify the number of folds in a (Stratified)KFold,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

    random_state : int, RandomState instance or None, optional (default=None)
        If ``order='random'``, determines random number generation for the
        chain order.
        In addition, it controls the random seed given at each `base_estimator`
        at each chaining iteration. Thus, it is only used when `base_estimator`
        exposes a `random_state`.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    verbose : bool, default=False
        If True, chain progress is output as each model is completed.

        .. versionadded:: 1.2

    Attributes
    ----------
    estimators_ : list
        A list of clones of base_estimator.

    order_ : list
        The order of labels in the classifier chain.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying `base_estimator` exposes such an attribute when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    ClassifierChain : Equivalent for classification.
    """
    """RegressorChain类，用于将回归模型按顺序排列成链式结构。

    每个模型按照链条中指定的顺序进行预测，利用提供给模型的所有特征以及链条中较早模型的预测结果。

    详细信息请参阅用户指南中的:ref:`RegressorChain <regressorchain>`。

    .. versionadded:: 0.20

    Parameters
    ----------
    base_estimator : estimator
        用于构建回归链的基础估计器。

    order : array-like of shape (n_outputs,) or 'random', default=None
        如果为`None`，则顺序将由标签矩阵Y的列顺序决定。::

            order = [0, 1, 2, ..., Y.shape[1] - 1]

        链的顺序可以通过提供整数列表来显式设置。例如，对于长度为5的链条。::

            order = [1, 3, 2, 4, 0]

        表示链中的第一个模型将为Y矩阵的第1列进行预测，第二个模型将为第3列进行预测，依此类推。

        如果order为'random'，将使用随机顺序。

    cv : int、交叉验证生成器或可迭代对象，默认为None
        确定是否使用交叉验证预测或链中先前估计器的真实标签结果。cv的可能输入为：

        - None，表示在拟合时使用真实标签，
        - 整数，指定（分层）KFold中的折叠数，
        - :term:`CV splitter`，
        - 作为索引数组的(train, test)分割的可迭代对象。

    random_state : int、RandomState实例或None，可选（默认为None）
        如果 ``order='random'``，则确定链的顺序的随机数生成。
        此外，它还控制每个`base_estimator`在每个链接迭代时给定的随机种子。因此，只有当`base_estimator`暴露出`random_state`时才会使用它。
        为了在多次函数调用中获得可重复的输出，请传递一个整数。
        参见 :term:`Glossary <random_state>`。

    verbose : bool，默认为False
        如果为True，则在每个模型完成时输出链的进展情况。

        .. versionadded:: 1.2

    Attributes
    ----------
    estimators_ : list
        基础估计器的克隆列表。

    order_ : list
        分类器链中标签的顺序。

    n_features_in_ : int
        在 :term:`fit` 过程中观察到的特征数。仅在底层的`base_estimator`在拟合时暴露了这样的属性时才定义。

        .. versionadded:: 0.24

    feature_names_in_ : 形状为 (`n_features_in_`,) 的ndarray
        在 :term:`fit` 过程中观察到的特征名称。仅当`X`具有全为字符串的特征名称时才定义。

        .. versionadded:: 1.0

    See Also
    --------
    ClassifierChain : 用于分类的等效模型链。
    """
    # MultiOutputRegressor: 学习每个输出独立而不是串联。

    Examples
    --------
    >>> from sklearn.multioutput import RegressorChain
    >>> from sklearn.linear_model import LogisticRegression
    >>> logreg = LogisticRegression(solver='lbfgs')
    >>> X, Y = [[1, 0], [0, 1], [1, 1]], [[0, 2], [1, 1], [2, 0]]
    >>> chain = RegressorChain(base_estimator=logreg, order=[0, 1]).fit(X, Y)
    >>> chain.predict(X)
    array([[0., 2.],
           [1., 1.],
           [2., 0.]])
    """

    @_fit_context(
        # RegressorChain.base_estimator is not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(self, X, Y, **fit_params):
        """将模型拟合到数据矩阵 X 和目标 Y 上。

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            输入数据。

        Y : array-like of shape (n_samples, n_classes)
            目标值。

        **fit_params : dict of string -> object
            传递给每个步骤的 `fit` 方法的参数。

            .. versionadded:: 0.23

        Returns
        -------
        self : object
            返回拟合后的实例。
        """
        super().fit(X, Y, **fit_params)
        return self

    def get_metadata_routing(self):
        """获取此对象的元数据路由信息。

        请参阅 :ref:`User Guide <metadata_routing>` 了解路由机制的工作方式。

        .. versionadded:: 1.3

        Returns
        -------
        routing : MetadataRouter
            包含路由信息的 :class:`~sklearn.utils.metadata_routing.MetadataRouter` 实例。
        """
        router = MetadataRouter(owner=self.__class__.__name__).add(
            estimator=self.base_estimator,
            method_mapping=MethodMapping().add(caller="fit", callee="fit"),
        )
        return router

    def _more_tags(self):
        return {"multioutput_only": True}
```