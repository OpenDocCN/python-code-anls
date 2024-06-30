# `D:\src\scipysrc\scikit-learn\sklearn\feature_selection\_from_model.py`

```
# 从scikit-learn开发人员导入所需的库
# 使用BSD-3-Clause许可证

from copy import deepcopy  # 导入深拷贝函数
from numbers import Integral, Real  # 导入整数和实数类型的支持

import numpy as np  # 导入NumPy库

from ..base import BaseEstimator, MetaEstimatorMixin, _fit_context, clone  # 导入基础估计器、元估计器混合类、拟合上下文、克隆函数
from ..exceptions import NotFittedError  # 导入未拟合错误类
from ..utils._param_validation import HasMethods, Interval, Options  # 导入参数验证相关类
from ..utils._tags import _safe_tags  # 导入安全标签工具函数
from ..utils.metadata_routing import (  # 导入元数据路由相关类和函数
    MetadataRouter,
    MethodMapping,
    _routing_enabled,
    process_routing,
)
from ..utils.metaestimators import available_if  # 导入可用条件元估计器函数
from ..utils.validation import _num_features, check_is_fitted, check_scalar  # 导入验证相关函数
from ._base import SelectorMixin, _get_feature_importances  # 导入选择混合类、获取特征重要性函数


def _calculate_threshold(estimator, importances, threshold):
    """解释阈值的含义"""

    if threshold is None:
        # 如果未指定阈值，则根据估计器确定默认值
        est_name = estimator.__class__.__name__
        is_l1_penalized = hasattr(estimator, "penalty") and estimator.penalty == "l1"
        is_lasso = "Lasso" in est_name
        is_elasticnet_l1_penalized = "ElasticNet" in est_name and (
            (hasattr(estimator, "l1_ratio_") and np.isclose(estimator.l1_ratio_, 1.0))
            or (hasattr(estimator, "l1_ratio") and np.isclose(estimator.l1_ratio, 1.0))
        )
        if is_l1_penalized or is_lasso or is_elasticnet_l1_penalized:
            # 如果使用了l1惩罚，则自然的默认阈值为0
            threshold = 1e-5
        else:
            threshold = "mean"

    if isinstance(threshold, str):
        # 如果阈值是字符串类型
        if "*" in threshold:
            scale, reference = threshold.split("*")
            scale = float(scale.strip())
            reference = reference.strip()

            if reference == "median":
                reference = np.median(importances)
            elif reference == "mean":
                reference = np.mean(importances)
            else:
                raise ValueError("Unknown reference: " + reference)

            threshold = scale * reference

        elif threshold == "median":
            threshold = np.median(importances)

        elif threshold == "mean":
            threshold = np.mean(importances)

        else:
            raise ValueError(
                "Expected threshold='mean' or threshold='median' got %s" % threshold
            )

    else:
        threshold = float(threshold)

    return threshold


def _estimator_has(attr):
    """检查是否可以委托方法给底层估计器。

    首先检查拟合过的`estimator_`（如果可用），否则检查未拟合的`estimator`。
    如果`attr`不存在，则抛出原始的`AttributeError`。
    该函数通常与`available_if`一起使用。
    """

    def check(self):
        if hasattr(self, "estimator_"):
            getattr(self.estimator_, attr)
        else:
            getattr(self.estimator, attr)

        return True

    return check


class SelectFromModel(MetaEstimatorMixin, SelectorMixin, BaseEstimator):
    # 从MetaEstimatorMixin、SelectorMixin和BaseEstimator继承的选择器类
    """Meta-transformer for selecting features based on importance weights.

    .. versionadded:: 0.17

    Read more in the :ref:`User Guide <select_from_model>`.

    Parameters
    ----------
    estimator : object
        The base estimator from which the transformer is built.
        This can be both a fitted (if ``prefit`` is set to True)
        or a non-fitted estimator. The estimator should have a
        ``feature_importances_`` or ``coef_`` attribute after fitting.
        Otherwise, the ``importance_getter`` parameter should be used.

    threshold : str or float, default=None
        The threshold value to use for feature selection. Features whose
        absolute importance value is greater or equal are kept while the others
        are discarded. If "median" (resp. "mean"), then the ``threshold`` value
        is the median (resp. the mean) of the feature importances. A scaling
        factor (e.g., "1.25*mean") may also be used. If None and if the
        estimator has a parameter penalty set to l1, either explicitly
        or implicitly (e.g, Lasso), the threshold used is 1e-5.
        Otherwise, "mean" is used by default.

    prefit : bool, default=False
        Whether a prefit model is expected to be passed into the constructor
        directly or not.
        If `True`, `estimator` must be a fitted estimator.
        If `False`, `estimator` is fitted and updated by calling
        `fit` and `partial_fit`, respectively.

    norm_order : non-zero int, inf, -inf, default=1
        Order of the norm used to filter the vectors of coefficients below
        ``threshold`` in the case where the ``coef_`` attribute of the
        estimator is of dimension 2.

    max_features : int, callable, default=None
        The maximum number of features to select.

        - If an integer, then it specifies the maximum number of features to
          allow.
        - If a callable, then it specifies how to calculate the maximum number of
          features allowed by using the output of `max_features(X)`.
        - If `None`, then all features are kept.

        To only select based on ``max_features``, set ``threshold=-np.inf``.

        .. versionadded:: 0.20
        .. versionchanged:: 1.1
           `max_features` accepts a callable.


    """
    # importance_getter 参数，用于指定如何获取特征重要性信息，默认为'auto'
    importance_getter : str or callable, default='auto'
        If 'auto', uses the feature importance either through a ``coef_``
        attribute or ``feature_importances_`` attribute of estimator.

        # 如果设置为'auto'，则根据 estimator 的 coef_ 属性或 feature_importances_ 属性获取特征重要性信息

        Also accepts a string that specifies an attribute name/path
        for extracting feature importance (implemented with `attrgetter`).
        For example, give `regressor_.coef_` in case of
        :class:`~sklearn.compose.TransformedTargetRegressor`  or
        `named_steps.clf.feature_importances_` in case of
        :class:`~sklearn.pipeline.Pipeline` with its last step named `clf`.

        # 还接受字符串，用于指定从哪个属性名/路径提取特征重要性信息，可用 `attrgetter` 实现
        # 例如，在 :class:`~sklearn.compose.TransformedTargetRegressor` 中，可以使用 `regressor_.coef_`
        # 或者在 :class:`~sklearn.pipeline.Pipeline` 中，如果最后一步命名为 `clf`，可以使用 `named_steps.clf.feature_importances_`

        If `callable`, overrides the default feature importance getter.
        The callable is passed with the fitted estimator and it should
        return importance for each feature.

        # 如果是可调用对象，则覆盖默认的特征重要性获取器
        # 可调用对象将接收已拟合的 estimator，应返回每个特征的重要性信息

        .. versionadded:: 0.24

    Attributes
    ----------
    estimator_ : estimator
        The base estimator from which the transformer is built. This attribute
        exist only when `fit` has been called.

        # transformer 构建的基础 estimator。此属性仅在调用 `fit` 后存在

        - If `prefit=True`, it is a deep copy of `estimator`.
        - If `prefit=False`, it is a clone of `estimator` and fit on the data
          passed to `fit` or `partial_fit`.

          # 如果 `prefit=True`，则为 estimator 的深拷贝
          # 如果 `prefit=False`，则为 estimator 的克隆，拟合于传递给 `fit` 或 `partial_fit` 的数据上

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

        # 在 `fit` 过程中观察到的特征数量。仅在底层 estimator 在拟合时公开此类属性时定义

        .. versionadded:: 0.24

    max_features_ : int
        Maximum number of features calculated during :term:`fit`. Only defined
        if the ``max_features`` is not `None`.

        # 在 `fit` 过程中计算的最大特征数量。仅在 `max_features` 不为 `None` 时定义

        - If `max_features` is an `int`, then `max_features_ = max_features`.
        - If `max_features` is a callable, then `max_features_ = max_features(X)`.

          # 如果 `max_features` 是整数，则 `max_features_ = max_features`
          # 如果 `max_features` 是可调用对象，则 `max_features_ = max_features(X)`

        .. versionadded:: 1.1

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        # 在 `fit` 过程中观察到的特征名称。仅在 `X` 具有所有字符串特征名称时定义

        .. versionadded:: 1.0

    threshold_ : float
        The threshold value used for feature selection.

        # 用于特征选择的阈值值

    See Also
    --------
    RFE : Recursive feature elimination based on importance weights.
    RFECV : Recursive feature elimination with built-in cross-validated
        selection of the best number of features.
    SequentialFeatureSelector : Sequential cross-validation based feature
        selection. Does not rely on importance weights.

        # 基于重要性权重的递归特征消除
        # 基于内置交叉验证选择最佳特征数量的递归特征消除
        # 基于顺序交叉验证的特征选择，不依赖于重要性权重

    Notes
    -----
    Allows NaN/Inf in the input if the underlying estimator does as well.

    # 如果底层 estimator 允许输入中存在 NaN/Inf，则也允许它们

    Examples
    --------
    >>> from sklearn.feature_selection import SelectFromModel
    >>> from sklearn.linear_model import LogisticRegression
    >>> X = [[ 0.87, -1.34,  0.31 ],
    ...      [-2.79, -0.02, -0.85 ],
    ...      [-1.34, -0.48, -2.55 ],
    ...      [ 1.92,  1.48,  0.65 ]]
    >>> y = [0, 1, 0, 1]
    >>> selector = SelectFromModel(estimator=LogisticRegression()).fit(X, y)
    >>> selector.estimator_.coef_
    array([[-0.3252...,  0.8345...,  0.4976...]])
    >>> selector.threshold_
    0.55249...

        # 示例
    # 定义参数约束字典，用于描述各参数的类型和限制条件
    _parameter_constraints: dict = {
        "estimator": [HasMethods("fit")],  # estimator参数必须具有"fit"方法
        "threshold": [Interval(Real, None, None, closed="both"), str, None],
        # threshold参数可以是实数区间，字符串或None
        "prefit": ["boolean"],  # prefit参数必须是布尔值
        "norm_order": [
            Interval(Integral, None, -1, closed="right"),  # norm_order参数为右闭整数区间
            Interval(Integral, 1, None, closed="left"),   # 或左闭整数区间
            Options(Real, {np.inf, -np.inf}),  # 或者为实数且取值为无穷大或负无穷大
        ],
        "max_features": [
            Interval(Integral, 0, None, closed="left"),  # max_features参数为左闭整数区间
            callable,  # 或者为可调用对象
            None,  # 或者为None
        ],
        "importance_getter": [str, callable],  # importance_getter参数可以是字符串或可调用对象
    }
    
    # 定义类的构造函数，初始化各参数
    def __init__(
        self,
        estimator,
        *,
        threshold=None,
        prefit=False,
        norm_order=1,
        max_features=None,
        importance_getter="auto",
    ):
        self.estimator = estimator  # 将estimator参数赋给对象的estimator属性
        self.threshold = threshold  # 将threshold参数赋给对象的threshold属性
        self.prefit = prefit  # 将prefit参数赋给对象的prefit属性
        self.importance_getter = importance_getter  # 将importance_getter参数赋给对象的importance_getter属性
        self.norm_order = norm_order  # 将norm_order参数赋给对象的norm_order属性
        self.max_features = max_features  # 将max_features参数赋给对象的max_features属性
    # 获取支持掩码，用于选择特征的子集
    def _get_support_mask(self):
        # 获取估计器对象，优先使用已经拟合的属性`estimator_`，否则使用`estimator`
        estimator = getattr(self, "estimator_", self.estimator)
        # 获取最大特征数，优先使用`max_features_`，否则使用`max_features`
        max_features = getattr(self, "max_features_", self.max_features)

        # 如果预先拟合属性为True，则验证估计器是否已拟合
        if self.prefit:
            try:
                check_is_fitted(self.estimator)
            except NotFittedError as exc:
                # 如果估计器未拟合且预先拟合属性为True，抛出异常
                raise NotFittedError(
                    "When `prefit=True`, `estimator` is expected to be a fitted "
                    "estimator."
                ) from exc

        # 如果`max_features`是可调用的，则要求在调用`transform`之前先调用`fit`
        if callable(max_features):
            raise NotFittedError(
                "When `prefit=True` and `max_features` is a callable, call `fit` "
                "before calling `transform`."
            )
        # 如果`max_features`不为None且不是整数类型，则抛出值错误异常
        elif max_features is not None and not isinstance(max_features, Integral):
            raise ValueError(
                f"`max_features` must be an integer. Got `max_features={max_features}` "
                "instead."
            )

        # 获取特征重要性得分
        scores = _get_feature_importances(
            estimator=estimator,
            getter=self.importance_getter,
            transform_func="norm",
            norm_order=self.norm_order,
        )

        # 计算阈值
        threshold = _calculate_threshold(estimator, scores, self.threshold)

        # 如果`max_features`不为None，则创建布尔掩码，选择最重要的特征
        if self.max_features is not None:
            mask = np.zeros_like(scores, dtype=bool)
            candidate_indices = np.argsort(-scores, kind="mergesort")[:max_features]
            mask[candidate_indices] = True
        else:
            # 否则，掩码全为True，表示所有特征都被选择
            mask = np.ones_like(scores, dtype=bool)

        # 根据阈值，将得分低于阈值的特征设置为False
        mask[scores < threshold] = False

        # 返回最终的特征支持掩码
        return mask

    # 检查最大特征数是否合法，并将其存储在`max_features_`属性中
    def _check_max_features(self, X):
        if self.max_features is not None:
            # 获取输入数据的特征数量
            n_features = _num_features(X)

            # 如果`max_features`是可调用的，则根据输入数据计算最大特征数
            if callable(self.max_features):
                max_features = self.max_features(X)
            else:  # 否则，直接使用设定的整数值
                max_features = self.max_features

            # 检查最大特征数是否为整数类型，并处于合法范围内
            check_scalar(
                max_features,
                "max_features",
                Integral,
                min_val=0,
                max_val=n_features,
            )

            # 将最大特征数存储在对象的`max_features_`属性中
            self.max_features_ = max_features

    # 用于装饰`SelectFromModel`的内部方法，在此处注释说明未进行嵌套验证
    @_fit_context(
        prefer_skip_nested_validation=False
    )
    def fit(self, X, y=None, **fit_params):
        """Fit the SelectFromModel meta-transformer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            训练输入样本。

        y : array-like of shape (n_samples,), default=None
            目标值（在分类中为整数，对应于类别；在回归中为实数）。

        **fit_params : dict
            - 如果 `enable_metadata_routing=False`（默认情况下）：

                直接传递给子评估器的 `fit` 方法的参数。如果 `prefit=True`，则会被忽略。

            - 如果 `enable_metadata_routing=True`：

                安全地路由到子评估器的 `fit` 方法的参数。如果 `prefit=True`，则会被忽略。

                .. versionchanged:: 1.4
                    更多详细信息，请参见：:ref:`Metadata Routing User Guide <metadata_routing>`。

        Returns
        -------
        self : object
            已拟合的估计器。
        """
        self._check_max_features(X)

        if self.prefit:
            try:
                check_is_fitted(self.estimator)
            except NotFittedError as exc:
                raise NotFittedError(
                    "When `prefit=True`, `estimator` is expected to be a fitted "
                    "estimator."
                ) from exc
            self.estimator_ = deepcopy(self.estimator)
        else:
            if _routing_enabled():
                routed_params = process_routing(self, "fit", **fit_params)
                self.estimator_ = clone(self.estimator)
                self.estimator_.fit(X, y, **routed_params.estimator.fit)
            else:
                # TODO(SLEP6): remove when metadata routing cannot be disabled.
                self.estimator_ = clone(self.estimator)
                self.estimator_.fit(X, y, **fit_params)

        if hasattr(self.estimator_, "feature_names_in_"):
            self.feature_names_in_ = self.estimator_.feature_names_in_
        else:
            self._check_feature_names(X, reset=True)

        return self

    @property
    def threshold_(self):
        """Threshold value used for feature selection."""
        scores = _get_feature_importances(
            estimator=self.estimator_,
            getter=self.importance_getter,
            transform_func="norm",
            norm_order=self.norm_order,
        )
        return _calculate_threshold(self.estimator, scores, self.threshold)

    @available_if(_estimator_has("partial_fit"))
    @_fit_context(
        # SelectFromModel.estimator is not validated yet
        prefer_skip_nested_validation=False
    )
    def partial_fit(self, X, y=None, **partial_fit_params):
        """Fit the SelectFromModel meta-transformer only once.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            训练输入样本。

        y : array-like of shape (n_samples,), default=None
            目标值（在分类中对应类别的整数，在回归中为实数）。

        **partial_fit_params : dict
            - 如果 `enable_metadata_routing=False`（默认）：

                直接传递给子估计器的 `partial_fit` 方法的参数。

            - 如果 `enable_metadata_routing=True`：

                传递给子估计器的 `partial_fit` 方法的参数。如果 `prefit=True`，则忽略这些参数。

                .. versionchanged:: 1.4
                    如果通过 :func:`~sklearn.set_config` 设置了 `enable_metadata_routing=True`，
                    则 `**partial_fit_params` 将被路由到子估计器，允许使用别名。

                有关更多详细信息，请参见 :ref:`Metadata Routing User Guide <metadata_routing>`。

        Returns
        -------
        self : object
            拟合后的估计器。
        """
        first_call = not hasattr(self, "estimator_")

        # 如果是第一次调用，则执行以下操作
        if first_call:
            # 检查最大特征数限制
            self._check_max_features(X)

        # 如果已经预先拟合，则直接返回
        if self.prefit:
            if first_call:
                try:
                    # 检查是否已经拟合
                    check_is_fitted(self.estimator)
                except NotFittedError as exc:
                    raise NotFittedError(
                        "When `prefit=True`, `estimator` is expected to be a fitted "
                        "estimator."
                    ) from exc
                # 深拷贝估计器对象
                self.estimator_ = deepcopy(self.estimator)
            return self

        # 如果是第一次调用，则克隆估计器对象
        if first_call:
            self.estimator_ = clone(self.estimator)

        # 如果启用了元数据路由
        if _routing_enabled():
            # 处理路由后的参数
            routed_params = process_routing(self, "partial_fit", **partial_fit_params)
            # 克隆估计器对象
            self.estimator_ = clone(self.estimator)
            # 调用子估计器的 partial_fit 方法
            self.estimator_.partial_fit(X, y, **routed_params.estimator.partial_fit)
        else:
            # 否则直接调用子估计器的 partial_fit 方法，传递原始参数
            self.estimator_.partial_fit(X, y, **partial_fit_params)

        # 如果估计器对象具有 feature_names_in_ 属性，则使用它
        if hasattr(self.estimator_, "feature_names_in_"):
            self.feature_names_in_ = self.estimator_.feature_names_in_
        else:
            # 否则检查特征名，如果是第一次调用则重置
            self._check_feature_names(X, reset=first_call)

        return self
    def n_features_in_(self):
        """
        Number of features seen during `fit`.

        """
        # 检查是否已经拟合（fit），如果未拟合，则引发 AttributeError
        try:
            check_is_fitted(self)
        except NotFittedError as nfe:
            raise AttributeError(
                "{} object has no n_features_in_ attribute.".format(
                    self.__class__.__name__
                )
            ) from nfe

        # 返回估算器（estimator）的 n_features_in_ 属性
        return self.estimator_.n_features_in_

    def get_metadata_routing(self):
        """
        Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        .. versionadded:: 1.4

        Returns
        -------
        routing : MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        # 创建一个 MetadataRouter 对象，包含当前对象的元数据路由信息
        router = MetadataRouter(owner=self.__class__.__name__).add(
            estimator=self.estimator,
            method_mapping=MethodMapping()
            .add(caller="partial_fit", callee="partial_fit")
            .add(caller="fit", callee="fit"),
        )
        # 返回创建的路由器对象
        return router

    def _more_tags(self):
        """
        Return additional tags for the estimator.

        These tags include information like whether NaNs are allowed,
        based on the estimator's behavior.

        Returns
        -------
        dict
            A dictionary containing additional tags for the estimator.
        """
        # 返回一个字典，包含估算器（estimator）的额外标签，如 NaN 是否允许等
        return {"allow_nan": _safe_tags(self.estimator, key="allow_nan")}
```