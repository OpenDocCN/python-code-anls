# `D:\src\scipysrc\scikit-learn\sklearn\ensemble\_stacking.py`

```
"""Stacking classifier and regressor."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABCMeta, abstractmethod
from copy import deepcopy
from numbers import Integral

import numpy as np
import scipy.sparse as sparse

# 导入基类和工具函数
from ..base import (
    ClassifierMixin,
    RegressorMixin,
    TransformerMixin,
    _fit_context,
    clone,
    is_classifier,
    is_regressor,
)
# 导入异常类
from ..exceptions import NotFittedError
# 导入线性模型
from ..linear_model import LogisticRegression, RidgeCV
# 导入模型选择和预处理模块
from ..model_selection import check_cv, cross_val_predict
from ..preprocessing import LabelEncoder
# 导入工具类和帮助函数
from ..utils import Bunch
from ..utils._estimator_html_repr import _VisualBlock
from ..utils._param_validation import HasMethods, StrOptions
from ..utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    _raise_for_params,
    _routing_enabled,
    process_routing,
)
from ..utils.metaestimators import available_if
from ..utils.multiclass import check_classification_targets, type_of_target
from ..utils.parallel import Parallel, delayed
from ..utils.validation import (
    _check_feature_names_in,
    _check_response_method,
    _deprecate_positional_args,
    check_is_fitted,
    column_or_1d,
)
# 导入基类和辅助函数
from ._base import _BaseHeterogeneousEnsemble, _fit_single_estimator


def _estimator_has(attr):
    """Check if we can delegate a method to the underlying estimator.

    First, we check the fitted `final_estimator_` if available, otherwise we check the
    unfitted `final_estimator`. We raise the original `AttributeError` if `attr` does
    not exist. This function is used together with `available_if`.
    """

    def check(self):
        if hasattr(self, "final_estimator_"):
            # 如果存在已拟合的 `final_estimator_`，则调用其属性 `attr`
            getattr(self.final_estimator_, attr)
        else:
            # 否则，调用未拟合的 `final_estimator` 的属性 `attr`
            getattr(self.final_estimator, attr)

        return True

    return check


class _BaseStacking(TransformerMixin, _BaseHeterogeneousEnsemble, metaclass=ABCMeta):
    """Base class for stacking method."""

    # 参数约束字典，指定各参数的类型或可选值
    _parameter_constraints: dict = {
        "estimators": [list],
        "final_estimator": [None, HasMethods("fit")],
        "cv": ["cv_object", StrOptions({"prefit"})],
        "n_jobs": [None, Integral],
        "passthrough": ["boolean"],
        "verbose": ["verbose"],
    }

    @abstractmethod
    def __init__(
        self,
        estimators,
        final_estimator=None,
        *,
        cv=None,
        stack_method="auto",
        n_jobs=None,
        verbose=0,
        passthrough=False,
    ):
        # 调用父类的初始化方法，传入参数 `estimators`
        super().__init__(estimators=estimators)
        # 初始化属性
        self.final_estimator = final_estimator  # 最终预测器
        self.cv = cv  # 交叉验证方法或对象
        self.stack_method = stack_method  # 堆叠方法
        self.n_jobs = n_jobs  # 并行任务数
        self.verbose = verbose  # 控制输出详细程度
        self.passthrough = passthrough  # 是否保留原始特征

    def _clone_final_estimator(self, default):
        # 如果 `final_estimator` 存在，则克隆为 `final_estimator_`，否则克隆 `default`
        if self.final_estimator is not None:
            self.final_estimator_ = clone(self.final_estimator)
        else:
            self.final_estimator_ = clone(default)
    def _concatenate_predictions(self, X, predictions):
        """Concatenate the predictions of each first layer learner and
        possibly the input dataset `X`.

        If `X` is sparse and `self.passthrough` is False, the output of
        `transform` will be dense (the predictions). If `X` is sparse
        and `self.passthrough` is True, the output of `transform` will
        be sparse.

        This helper is in charge of ensuring the predictions are 2D arrays and
        it will drop one of the probability column when using probabilities
        in the binary case. Indeed, the p(y|c=0) = 1 - p(y|c=1)

        When `y` type is `"multilabel-indicator"`` and the method used is
        `predict_proba`, `preds` can be either a `ndarray` of shape
        `(n_samples, n_class)` or for some estimators a list of `ndarray`.
        This function will drop one of the probability column in this situation as well.
        """
        X_meta = []  # 初始化一个空列表，用于存储处理后的预测结果
        for est_idx, preds in enumerate(predictions):
            if isinstance(preds, list):
                # 如果 `preds` 是一个列表，表示包含多个预测结果
                # 每个预测结果是一个 `n_targets` 个 `n_classes` 列的 2D ndarray
                # 我们需要去掉每个预测结果的第一列，因为概率和为一
                for pred in preds:
                    X_meta.append(pred[:, 1:])
            elif preds.ndim == 1:
                # 如果 `preds` 是一个1维数组，需要将其转换成2维数组
                X_meta.append(preds.reshape(-1, 1))
            elif (
                self.stack_method_[est_idx] == "predict_proba"
                and len(self.classes_) == 2
            ):
                # 在二分类问题中，当使用概率时，移除第一列
                # 因为两个特征 `preds` 是完全共线的
                X_meta.append(preds[:, 1:])
            else:
                # 其他情况下直接将 `preds` 添加到 `X_meta`
                X_meta.append(preds)

        self._n_feature_outs = [pred.shape[1] for pred in X_meta]  # 记录每个预测结果的特征数量
        if self.passthrough:
            X_meta.append(X)  # 如果 `self.passthrough` 为真，将输入数据 `X` 添加到 `X_meta`
            if sparse.issparse(X):
                return sparse.hstack(X_meta, format=X.format)  # 如果 `X` 是稀疏矩阵，则使用稀疏矩阵的方法合并 `X_meta`

        return np.hstack(X_meta)  # 合并处理后的结果为一个数组

    @staticmethod
    def _method_name(name, estimator, method):
        if estimator == "drop":
            return None  # 如果 `estimator` 是 "drop"，则返回 None
        if method == "auto":
            method = ["predict_proba", "decision_function", "predict"]  # 如果 `method` 是 "auto"，则定义默认方法列表
        try:
            method_name = _check_response_method(estimator, method).__name__  # 获取方法名称
        except AttributeError as e:
            raise ValueError(
                f"Underlying estimator {name} does not implement the method {method}."
            ) from e  # 如果方法不存在，抛出 ValueError 异常

        return method_name  # 返回方法名称
    @_fit_context(
        # estimators in Stacking*.estimators are not validated yet
        prefer_skip_nested_validation=False
    )


# 使用装饰器定义 `_fit_context` 方法，并设置参数 `prefer_skip_nested_validation` 为 False，
# 这个装饰器用于上下文管理，用于某些操作的环境设置。



    @property
    def n_features_in_(self):
        """Number of features seen during :term:`fit`."""


# 定义属性方法 `n_features_in_`，用于返回在拟合过程中观察到的特征数量。



        try:
            check_is_fitted(self)
        except NotFittedError as nfe:
            raise AttributeError(
                f"{self.__class__.__name__} object has no attribute n_features_in_"
            ) from nfe
        return self.estimators_[0].n_features_in_


# 在 `n_features_in_` 方法中，首先检查对象是否已经拟合（fit）。
# 如果未拟合，则抛出 `AttributeError` 异常，说明对象没有 `n_features_in_` 属性。
# 如果已拟合，则返回第一个估算器（estimator）的特征数量 `n_features_in_`。



    def _transform(self, X):
        """Concatenate and return the predictions of the estimators."""
        check_is_fitted(self)
        predictions = [
            getattr(est, meth)(X)
            for est, meth in zip(self.estimators_, self.stack_method_)
            if est != "drop"
        ]
        return self._concatenate_predictions(X, predictions)


# 定义 `_transform` 方法，用于对输入数据 `X` 进行转换。
# 首先检查对象是否已拟合。
# 使用 `getattr` 方法调用每个估算器（estimator）的指定方法 `meth` 进行预测，并将结果存储在 `predictions` 中。
# 最后调用 `_concatenate_predictions` 方法将所有预测结果拼接并返回。



    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features. The input feature names are only used when `passthrough` is
            `True`.

            - If `input_features` is `None`, then `feature_names_in_` is
              used as feature names in. If `feature_names_in_` is not defined,
              then names are generated: `[x0, x1, ..., x(n_features_in_ - 1)]`.
            - If `input_features` is an array-like, then `input_features` must
              match `feature_names_in_` if `feature_names_in_` is defined.

            If `passthrough` is `False`, then only the names of `estimators` are used
            to generate the output feature names.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        check_is_fitted(self, "n_features_in_")
        input_features = _check_feature_names_in(
            self, input_features, generate_names=self.passthrough
        )

        class_name = self.__class__.__name__.lower()
        non_dropped_estimators = (
            name for name, est in self.estimators if est != "drop"
        )
        meta_names = []
        for est, n_features_out in zip(non_dropped_estimators, self._n_feature_outs):
            if n_features_out == 1:
                meta_names.append(f"{class_name}_{est}")
            else:
                meta_names.extend(
                    f"{class_name}_{est}{i}" for i in range(n_features_out)
                )

        if self.passthrough:
            return np.concatenate((meta_names, input_features))

        return np.asarray(meta_names, dtype=object)


# 定义 `get_feature_names_out` 方法，用于获取转换后的输出特征名。
# 首先检查对象是否已拟合，特别是检查是否有 `n_features_in_` 属性。
# 使用 `_check_feature_names_in` 函数验证并处理输入特征名称 `input_features`。
# 根据 `passthrough` 参数的设置，决定如何生成输出特征名：
# - 如果 `passthrough` 是 `True`，则将 `meta_names` 和 `input_features` 拼接起来返回。
# - 如果 `passthrough` 是 `False`，则只使用估算器（estimators）的名称生成输出特征名。
# 最后返回转换后的特征名数组 `feature_names_out`。



    @available_if(_estimator_has("predict"))


# 使用 `@available_if` 装饰器，并指定 `_estimator_has("predict")` 作为条件。
# 这个装饰器通常用于指定方法或属性是否可用，依赖于估算器是否具有 `predict` 方法。
    def predict(self, X, **predict_params):
        """Predict target for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        **predict_params : dict of str -> obj
            Parameters to the `predict` called by the `final_estimator`. Note
            that this may be used to return uncertainties from some estimators
            with `return_std` or `return_cov`. Be aware that it will only
            account for uncertainty in the final estimator.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_output)
            Predicted targets.
        """
        
        # 检查模型是否已经拟合
        check_is_fitted(self)
        
        # 调用最终评估器（final_estimator）的预测方法来进行预测，并传递额外的预测参数
        return self.final_estimator_.predict(self.transform(X), **predict_params)

    def _sk_visual_block_with_final_estimator(self, final_estimator):
        names, estimators = zip(*self.estimators)
        
        # 创建一个并行块（parallel block），展示多个估计器的视觉块
        parallel = _VisualBlock("parallel", estimators, names=names, dash_wrapped=False)

        # 最终评估器被包装在一个并行块中，以显示标签 'final_estimator' 在 HTML 表示中
        final_block = _VisualBlock(
            "parallel", [final_estimator], names=["final_estimator"], dash_wrapped=False
        )
        
        # 返回一个串行块（serial block），包含并行块和最终评估器的并行块
        return _VisualBlock("serial", (parallel, final_block), dash_wrapped=False)

    def get_metadata_routing(self):
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        .. versionadded:: 1.6

        Returns
        -------
        routing : MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        
        # 创建一个 MetadataRouter 对象，将所有估计器和最终评估器添加到路由中
        router = MetadataRouter(owner=self.__class__.__name__)

        # self.estimators 是一个包含 (name, estimator) 元组的列表
        for name, estimator in self.estimators:
            router.add(
                **{name: estimator},
                method_mapping=MethodMapping().add(callee="fit", caller="fit"),
            )

        try:
            final_estimator_ = self.final_estimator_
        except AttributeError:
            final_estimator_ = self.final_estimator

        # 将最终评估器添加到路由中，允许映射 'predict' 方法
        router.add(
            final_estimator_=final_estimator_,
            method_mapping=MethodMapping().add(caller="predict", callee="predict"),
        )

        return router
class StackingClassifier(ClassifierMixin, _BaseStacking):
    """Stack of estimators with a final classifier.

    Stacked generalization consists in stacking the output of individual
    estimator and use a classifier to compute the final prediction. Stacking
    allows to use the strength of each individual estimator by using their
    output as input of a final estimator.

    Note that `estimators_` are fitted on the full `X` while `final_estimator_`
    is trained using cross-validated predictions of the base estimators using
    `cross_val_predict`.

    Read more in the :ref:`User Guide <stacking>`.

    .. versionadded:: 0.22

    Parameters
    ----------
    estimators : list of (str, estimator)
        Base estimators which will be stacked together. Each element of the
        list is defined as a tuple of string (i.e. name) and an estimator
        instance. An estimator can be set to 'drop' using `set_params`.

        The type of estimator is generally expected to be a classifier.
        However, one can pass a regressor for some use case (e.g. ordinal
        regression).

    final_estimator : estimator, default=None
        A classifier which will be used to combine the base estimators.
        The default classifier is a
        :class:`~sklearn.linear_model.LogisticRegression`.
    """


**注释：**

# 定义一个堆叠分类器类，继承自ClassifierMixin和_BaseStacking
class StackingClassifier(ClassifierMixin, _BaseStacking):
    """Stack of estimators with a final classifier.

    Stacked generalization consists in stacking the output of individual
    estimator and use a classifier to compute the final prediction. Stacking
    allows to use the strength of each individual estimator by using their
    output as input of a final estimator.

    Note that `estimators_` are fitted on the full `X` while `final_estimator_`
    is trained using cross-validated predictions of the base estimators using
    `cross_val_predict`.

    Read more in the :ref:`User Guide <stacking>`.

    .. versionadded:: 0.22

    Parameters
    ----------
    estimators : list of (str, estimator)
        Base estimators which will be stacked together. Each element of the
        list is defined as a tuple of string (i.e. name) and an estimator
        instance. An estimator can be set to 'drop' using `set_params`.

        The type of estimator is generally expected to be a classifier.
        However, one can pass a regressor for some use case (e.g. ordinal
        regression).

    final_estimator : estimator, default=None
        A classifier which will be used to combine the base estimators.
        The default classifier is a
        :class:`~sklearn.linear_model.LogisticRegression`.
    """
    cv : int, cross-validation generator, iterable, or "prefit", default=None
        确定在 `cross_val_predict` 中用于训练 `final_estimator` 的交叉验证分割策略。`cv` 可能的输入包括：

        * None，使用默认的 5 折交叉验证，
        * 整数，指定 (分层) KFold 中的折数，
        * 用作交叉验证生成器的对象，
        * 生成训练-测试拆分的可迭代对象，
        * `"prefit"`，假设 `estimators` 已经被预先拟合。在这种情况下，估计器将不会被重新拟合。

        对于整数/None的输入，如果估计器是分类器且y是二元或多类别的，
        将使用 :class:`~sklearn.model_selection.StratifiedKFold`。
        在所有其他情况下，将使用 :class:`~sklearn.model_selection.KFold`。
        这些分割器使用 `shuffle=False` 实例化，因此调用时的拆分将保持一致。

        参见 :ref:`用户指南 <cross_validation>` 获取可以在此处使用的各种交叉验证策略。

        如果传递了 "prefit"，则假定所有 `estimators` 已经被拟合。
        `final_estimator_` 是在 `estimators` 在完整训练集上的预测上训练的，
        **而不是** 经过交叉验证的预测。请注意，如果模型已经在相同数据上训练来训练堆叠模型，
        则存在过拟合的高风险。

        .. versionadded:: 1.1
            1.1 版本中添加了 "prefit" 选项。

        .. note::
           如果训练样本数量足够大，则增加拆分数量将不会带来任何好处。实际上，训练时间会增加。
           ``cv`` 不用于模型评估，而是用于预测。

    stack_method : {'auto', 'predict_proba', 'decision_function', 'predict'}, \
            default='auto'
        每个基础估计器调用的方法。可以是：

        * 如果为 'auto'，它将尝试依次调用每个估计器的 `'predict_proba'`、`'decision_function'` 或 `'predict'`。
        * 否则，可以是 `'predict_proba'`、`'decision_function'` 或 `'predict'` 中的一个。如果估计器未实现该方法，则会引发错误。

    n_jobs : int, default=None
        `fit` 所有 `estimators` 时并行运行的作业数。`None` 表示 1，除非在 `joblib.parallel_backend` 上下文中。-1 表示使用所有处理器。有关详细信息，请参见术语表。

    passthrough : bool, default=False
        当为 False 时，仅使用估计器的预测作为 `final_estimator` 的训练数据。当为 True 时，
        `final_estimator` 将在预测以及原始训练数据上进行训练。

    verbose : int, default=0
        详细程度级别。

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,) or list of ndarray if `y` \
        is of type `"multilabel-indicator"`.
        Class labels.
    
    estimators_ : list of estimators
        The elements of the `estimators` parameter, having been fitted on the
        training data. If an estimator has been set to `'drop'`, it
        will not appear in `estimators_`. When `cv="prefit"`, `estimators_`
        is set to `estimators` and is not fitted again.
    
    named_estimators_ : :class:`~sklearn.utils.Bunch`
        Attribute to access any fitted sub-estimators by name.
    
    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying classifier exposes such an attribute when fit.
        
        .. versionadded:: 0.24
    
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if the
        underlying estimators expose such an attribute when fit.
        
        .. versionadded:: 1.0
    
    final_estimator_ : estimator
        The classifier which predicts given the output of `estimators_`.
    
    stack_method_ : list of str
        The method used by each base estimator.
    
    See Also
    --------
    StackingRegressor : Stack of estimators with a final regressor.
    
    Notes
    -----
    When `predict_proba` is used by each estimator (i.e. most of the time for
    `stack_method='auto'` or specifically for `stack_method='predict_proba'`),
    the first column predicted by each estimator will be dropped in the case
    of a binary classification problem. Indeed, both feature will be perfectly
    collinear.
    
    In some cases (e.g. ordinal regression), one can pass regressors as the
    first layer of the :class:`StackingClassifier`. However, note that `y` will
    be internally encoded in a numerically increasing order or lexicographic
    order. If this ordering is not adequate, one should manually numerically
    encode the classes in the desired order.
    
    References
    ----------
    .. [1] Wolpert, David H. "Stacked generalization." Neural networks 5.2
       (1992): 241-259.
    
    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.svm import LinearSVC
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.ensemble import StackingClassifier
    >>> X, y = load_iris(return_X_y=True)
    >>> estimators = [
    ...     ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
    ...     ('svr', make_pipeline(StandardScaler(),
    ...                           LinearSVC(random_state=42)))
    ... ]
    >>> clf = StackingClassifier(
    ...     estimators=estimators, final_estimator=LogisticRegression()
    ... )
    >>> from sklearn.model_selection import train_test_split
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, stratify=y, random_state=42
    ... )
    >>> clf.fit(X_train, y_train).score(X_test, y_test)
    0.9...
    """
    进行数据集拆分，分为训练集和测试集，并指定了随机种子和分层抽样
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, stratify=y, random_state=42
    ... )
    使用分类器 `clf` 对训练集进行拟合，然后计算在测试集上的预测准确率得分

    _parameter_constraints: dict = {
        **_BaseStacking._parameter_constraints,
        "stack_method": [
            StrOptions({"auto", "predict_proba", "decision_function", "predict"})
        ],
    }
    初始化参数限制字典 `_parameter_constraints`，继承自 `_BaseStacking` 的限制，并加入了 `stack_method` 的选项限制

    def __init__(
        self,
        estimators,
        final_estimator=None,
        *,
        cv=None,
        stack_method="auto",
        n_jobs=None,
        passthrough=False,
        verbose=0,
    ):
        调用父类初始化方法，设置集成模型的各种参数和超参数

    def _validate_final_estimator(self):
        对最终估计器进行验证，确保其是分类器类型，默认使用 LogisticRegression() 作为默认值

    def _validate_estimators(self):
        """重载 `_BaseHeterogeneousEnsemble` 的方法以更宽松地接受 `estimators` 的类型。
        
        可以接受回归器的情况，例如序数回归。
        """
        验证集成器列表的有效性，确保至少有一个估计器不是 "drop"，并检查列表不能为空

    # TODO(1.7): remove `sample_weight` from the signature after deprecation
    # cycle; pop it from `fit_params` before the `_raise_for_params` check and
    # reinsert afterwards, for backwards compatibility
    @_deprecate_positional_args(version="1.7")
    标记 `sample_weight` 参数在后续版本中将被弃用，建议在检查参数之前从 `fit_params` 中移除，并在检查后重新插入，以确保向后兼容
    # 定义一个方法用于拟合（训练）估算器（estimators）。
    def fit(self, X, y, *, sample_weight=None, **fit_params):
        """Fit the estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values. Note that `y` will be internally encoded in
            numerically increasing order or lexicographic order. If the order
            matter (e.g. for ordinal regression), one should numerically encode
            the target `y` before calling :term:`fit`.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if all underlying estimators
            support sample weights.

        **fit_params : dict
            Parameters to pass to the underlying estimators.

            .. versionadded:: 1.6

                Only available if `enable_metadata_routing=True`, which can be
                set by using ``sklearn.set_config(enable_metadata_routing=True)``.
                See :ref:`Metadata Routing User Guide <metadata_routing>` for
                more details.

        Returns
        -------
        self : object
            Returns a fitted instance of estimator.
        """
        # 检查参数是否有效，如果不合规会引发异常
        _raise_for_params(fit_params, self, "fit")
        # 检查分类目标y是否有效
        check_classification_targets(y)
        
        # 如果y的类型是多标签指示器（multilabel-indicator）
        if type_of_target(y) == "multilabel-indicator":
            # 为每个标签列（每个y的列）创建一个LabelEncoder对象，并拟合
            self._label_encoder = [LabelEncoder().fit(yk) for yk in y.T]
            # 记录每个LabelEncoder对象的类别
            self.classes_ = [le.classes_ for le in self._label_encoder]
            # 对y进行编码，转换成数字表示
            y_encoded = np.array(
                [
                    self._label_encoder[target_idx].transform(target)
                    for target_idx, target in enumerate(y.T)
                ]
            ).T
        else:
            # 创建一个LabelEncoder对象，并拟合y
            self._label_encoder = LabelEncoder().fit(y)
            # 记录LabelEncoder对象的类别
            self.classes_ = self._label_encoder.classes_
            # 对y进行编码，转换成数字表示
            y_encoded = self._label_encoder.transform(y)

        # 如果有样本权重，则将其添加到fit_params中
        if sample_weight is not None:
            fit_params["sample_weight"] = sample_weight
        
        # 调用父类的fit方法来拟合（训练）模型，传入X、y_encoded以及fit_params
        return super().fit(X, y_encoded, **fit_params)
    
    @available_if(_estimator_has("predict"))
    def predict(self, X, **predict_params):
        """Predict target for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        **predict_params : dict of str -> obj
            Parameters to the `predict` called by the `final_estimator`. Note
            that this may be used to return uncertainties from some estimators
            with `return_std` or `return_cov`. Be aware that it will only
            account for uncertainty in the final estimator.

            - If `enable_metadata_routing=False` (default):
              Parameters directly passed to the `predict` method of the
              `final_estimator`.

            - If `enable_metadata_routing=True`: Parameters safely routed to
              the `predict` method of the `final_estimator`. See :ref:`Metadata
              Routing User Guide <metadata_routing>` for more details.

            .. versionchanged:: 1.6
                `**predict_params` can be routed via metadata routing API.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_output)
            Predicted targets.
        """
        # Check if routing is enabled
        if _routing_enabled():
            # Process routing using current object and predict_params
            routed_params = process_routing(self, "predict", **predict_params)
        else:
            # Temporary handling for backward compatibility
            # TODO(SLEP6): remove when metadata routing cannot be disabled.
            routed_params = Bunch()
            # Simulate final estimator prediction parameters
            routed_params.final_estimator_ = Bunch(predict={})
            routed_params.final_estimator_.predict = predict_params

        # Perform prediction using superclass's predict method and routed parameters
        y_pred = super().predict(X, **routed_params.final_estimator_["predict"])

        # Adjust prediction based on label encoder type
        if isinstance(self._label_encoder, list):
            # Handle multilabel-indicator case by inverse transforming each target
            y_pred = np.array(
                [
                    self._label_encoder[target_idx].inverse_transform(target)
                    for target_idx, target in enumerate(y_pred.T)
                ]
            ).T
        else:
            # Inverse transform predictions using label encoder
            y_pred = self._label_encoder.inverse_transform(y_pred)

        # Return the predicted targets
        return y_pred

    @available_if(_estimator_has("predict_proba"))


注释：
    def predict_proba(self, X):
        """Predict class probabilities for `X` using the final estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        probabilities : ndarray of shape (n_samples, n_classes) or \
            list of ndarray of shape (n_output,)
            The class probabilities of the input samples.
        """
        # 检查模型是否已经拟合
        check_is_fitted(self)
        # 使用最终估计器预测样本 X 的类别概率
        y_pred = self.final_estimator_.predict_proba(self.transform(X))

        if isinstance(self._label_encoder, list):
            # 处理多标签指示器情况
            # 将每个预测结果的第一列组成新的数组
            y_pred = np.array([preds[:, 0] for preds in y_pred]).T
        return y_pred

    @available_if(_estimator_has("decision_function"))
    def decision_function(self, X):
        """Decision function for samples in `X` using the final estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        decisions : ndarray of shape (n_samples,), (n_samples, n_classes), \
            or (n_samples, n_classes * (n_classes-1) / 2)
            The decision function computed the final estimator.
        """
        # 检查模型是否已经拟合
        check_is_fitted(self)
        # 返回使用最终估计器计算的决策函数结果
        return self.final_estimator_.decision_function(self.transform(X))

    def transform(self, X):
        """Return class labels or probabilities for X for each estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        y_preds : ndarray of shape (n_samples, n_estimators) or \
                (n_samples, n_classes * n_estimators)
            Prediction outputs for each estimator.
        """
        # 调用内部方法 _transform 进行转换操作
        return self._transform(X)

    def _sk_visual_block_(self):
        # 如果 final_estimator 未设置，则使用 LogisticRegression 作为默认值
        if self.final_estimator is None:
            final_estimator = LogisticRegression()
        else:
            final_estimator = self.final_estimator
        # 调用父类方法，并传入最终估计器
        return super()._sk_visual_block_with_final_estimator(final_estimator)
class StackingRegressor(RegressorMixin, _BaseStacking):
    """Stack of estimators with a final regressor.

    Stacked generalization consists in stacking the output of individual
    estimator and use a regressor to compute the final prediction. Stacking
    allows to use the strength of each individual estimator by using their
    output as input of a final estimator.

    Note that `estimators_` are fitted on the full `X` while `final_estimator_`
    is trained using cross-validated predictions of the base estimators using
    `cross_val_predict`.

    Read more in the :ref:`User Guide <stacking>`.

    .. versionadded:: 0.22

    Parameters
    ----------
    estimators : list of (str, estimator)
        Base estimators which will be stacked together. Each element of the
        list is defined as a tuple of string (i.e. name) and an estimator
        instance. An estimator can be set to 'drop' using `set_params`.

    final_estimator : estimator, default=None
        A regressor which will be used to combine the base estimators.
        The default regressor is a :class:`~sklearn.linear_model.RidgeCV`.

    cv : int, cross-validation generator, iterable, or "prefit", default=None
        Determines the cross-validation splitting strategy used in
        `cross_val_predict` to train `final_estimator`. Possible inputs for
        cv are:

        * None, to use the default 5-fold cross validation,
        * integer, to specify the number of folds in a (Stratified) KFold,
        * An object to be used as a cross-validation generator,
        * An iterable yielding train, test splits.
        * "prefit" to assume the `estimators` are prefit, and skip cross validation

        For integer/None inputs, if the estimator is a classifier and y is
        either binary or multiclass,
        :class:`~sklearn.model_selection.StratifiedKFold` is used.
        In all other cases, :class:`~sklearn.model_selection.KFold` is used.
        These splitters are instantiated with `shuffle=False` so the splits
        will be the same across calls.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        If "prefit" is passed, it is assumed that all `estimators` have
        been fitted already. The `final_estimator_` is trained on the `estimators`
        predictions on the full training set and are **not** cross validated
        predictions. Please note that if the models have been trained on the same
        data to train the stacking model, there is a very high risk of overfitting.

        .. versionadded:: 1.1
            The 'prefit' option was added in 1.1

        .. note::
           A larger number of split will provide no benefits if the number
           of training samples is large enough. Indeed, the training time
           will increase. ``cv`` is not used for model evaluation but for
           prediction.
"""
    n_jobs : int, default=None
        # 控制并行运行的作业数量，用于`fit`所有`estimators`。
        # 默认为`None`，除非在`joblib.parallel_backend`上下文中，否则为1。
        # `-1`表示使用所有处理器。详见术语表中的说明。

    passthrough : bool, default=False
        # 当为`False`时，仅使用estimators的预测作为`final_estimator`的训练数据。
        # 当为`True`时，`final_estimator`会在预测数据和原始训练数据上进行训练。

    verbose : int, default=0
        # 决定详细程度的级别。

    Attributes
    ----------
    estimators_ : list of estimator
        # 经过拟合的`estimators`参数的元素，用于训练数据。
        # 如果某个estimator被设置为'drop'，它将不会出现在`estimators_`中。
        # 当`cv="prefit"`时，`estimators_`被设置为`estimators`，不会再次进行拟合。

    named_estimators_ : :class:`~sklearn.utils.Bunch`
        # 通过名称访问任何经过拟合的子estimator的属性。

    n_features_in_ : int
        # 在`fit`过程中看到的特征数量。
        # 只有在基础回归器在fit时公开了这样的属性时才定义。

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        # 在`fit`过程中看到的特征名称。
        # 只有在基础estimators在fit时公开了这样的属性时才定义。

        .. versionadded:: 1.0

    final_estimator_ : estimator
        # 用于堆叠基础estimators拟合的回归器。

    stack_method_ : list of str
        # 每个基础estimator使用的方法。

    See Also
    --------
    StackingClassifier : 带有最终分类器的estimators堆叠。

    References
    ----------
    .. [1] Wolpert, David H. "Stacked generalization." Neural networks 5.2
       (1992): 241-259.

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.linear_model import RidgeCV
    >>> from sklearn.svm import LinearSVR
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.ensemble import StackingRegressor
    >>> X, y = load_diabetes(return_X_y=True)
    >>> estimators = [
    ...     ('lr', RidgeCV()),
    ...     ('svr', LinearSVR(random_state=42))
    ... ]
    >>> reg = StackingRegressor(
    ...     estimators=estimators,
    ...     final_estimator=RandomForestRegressor(n_estimators=10,
    ...                                           random_state=42)
    ... )
    >>> from sklearn.model_selection import train_test_split
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, random_state=42
    ... )
    >>> reg.fit(X_train, y_train).score(X_test, y_test)
    0.3...
    ):
        super().__init__(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=cv,
            stack_method="predict",
            n_jobs=n_jobs,
            passthrough=passthrough,
            verbose=verbose,
        )

    def _validate_final_estimator(self):
        self._clone_final_estimator(default=RidgeCV())
        if not is_regressor(self.final_estimator_):
            raise ValueError(
                "'final_estimator' parameter should be a regressor. Got {}".format(
                    self.final_estimator_
                )
            )

    # TODO(1.7): remove `sample_weight` from the signature after deprecation
    # cycle; pop it from `fit_params` before the `_raise_for_params` check and
    # reinsert afterwards, for backwards compatibility
    @_deprecate_positional_args(version="1.7")
    def fit(self, X, y, *, sample_weight=None, **fit_params):
        """Fit the estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if all underlying estimators
            support sample weights.

        **fit_params : dict
            Parameters to pass to the underlying estimators.

            .. versionadded:: 1.6

                Only available if `enable_metadata_routing=True`, which can be
                set by using ``sklearn.set_config(enable_metadata_routing=True)``.
                See :ref:`Metadata Routing User Guide <metadata_routing>` for
                more details.

        Returns
        -------
        self : object
            Returns a fitted instance.
        """
        _raise_for_params(fit_params, self, "fit")  # 检查并确保传递的参数有效
        y = column_or_1d(y, warn=True)  # 将目标值确保为一维数组或一列
        if sample_weight is not None:
            fit_params["sample_weight"] = sample_weight  # 如果存在样本权重，则将其加入到fit_params中
        return super().fit(X, y, **fit_params)  # 调用父类的fit方法进行拟合

    def transform(self, X):
        """Return the predictions for X for each estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        y_preds : ndarray of shape (n_samples, n_estimators)
            Prediction outputs for each estimator.
        """
        return self._transform(X)  # 调用内部方法_transform进行转换

    # TODO(1.7): remove `sample_weight` from the signature after deprecation
    # cycle; pop it from `fit_params` before the `_raise_for_params` check and
    # reinsert afterwards, for backwards compatibility
    @_deprecate_positional_args(version="1.7")  # 装饰器，标记该方法在特定版本中已弃用
    def fit_transform(self, X, y, *, sample_weight=None, **fit_params):
        """Fit the estimators and return the predictions for X for each estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if all underlying estimators
            support sample weights.

        **fit_params : dict
            Parameters to pass to the underlying estimators.

            .. versionadded:: 1.6

                Only available if `enable_metadata_routing=True`, which can be
                set by using ``sklearn.set_config(enable_metadata_routing=True)``.
                See :ref:`Metadata Routing User Guide <metadata_routing>` for
                more details.

        Returns
        -------
        y_preds : ndarray of shape (n_samples, n_estimators)
            Prediction outputs for each estimator.
        """
        # 检查 fit_params 中的参数，抛出异常
        _raise_for_params(fit_params, self, "fit")
        # 如果提供了样本权重，将其添加到 fit_params 中
        if sample_weight is not None:
            fit_params["sample_weight"] = sample_weight
        # 调用父类的 fit_transform 方法，传入数据和参数，返回结果
        return super().fit_transform(X, y, **fit_params)

    @available_if(_estimator_has("predict"))  # 装饰器，确保 estimator 支持 predict 方法
    def predict(self, X, **predict_params):
        """Predict target for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        **predict_params : dict of str -> obj
            Parameters to the `predict` called by the `final_estimator`. Note
            that this may be used to return uncertainties from some estimators
            with `return_std` or `return_cov`. Be aware that it will only
            account for uncertainty in the final estimator.

            - If `enable_metadata_routing=False` (default):
              Parameters directly passed to the `predict` method of the
              `final_estimator`.

            - If `enable_metadata_routing=True`: Parameters safely routed to
              the `predict` method of the `final_estimator`. See :ref:`Metadata
              Routing User Guide <metadata_routing>` for more details.

            .. versionchanged:: 1.6
                `**predict_params` can be routed via metadata routing API.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_output)
            Predicted targets.
        """
        # Check if routing is enabled
        if _routing_enabled():
            # Process routing for predict_params using self and "predict" method
            routed_params = process_routing(self, "predict", **predict_params)
        else:
            # Metadata routing disabled, use default routing
            routed_params = Bunch()
            routed_params.final_estimator_ = Bunch(predict={})
            routed_params.final_estimator_.predict = predict_params

        # Call superclass's predict method with final_estimator's predict_params
        y_pred = super().predict(X, **routed_params.final_estimator_["predict"])

        return y_pred

    def _sk_visual_block_(self):
        # If final_estimator is not specified, default to RidgeCV
        if self.final_estimator is None:
            final_estimator = RidgeCV()
        else:
            final_estimator = self.final_estimator

        # Return result of _sk_visual_block_with_final_estimator using final_estimator
        return super()._sk_visual_block_with_final_estimator(final_estimator)
```