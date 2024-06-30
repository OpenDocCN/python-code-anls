# `D:\src\scipysrc\scikit-learn\sklearn\ensemble\_voting.py`

```
"""
Soft Voting/Majority Rule classifier and Voting regressor.

This module contains:
 - A Soft Voting/Majority Rule classifier for classification estimators.
 - A Voting regressor for regression estimators.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from abc import abstractmethod  # 导入抽象基类模块中的抽象方法
from numbers import Integral  # 导入整数类型

import numpy as np  # 导入NumPy库

from ..base import (  # 导入自定义的基础类和函数
    ClassifierMixin,  # 分类器Mixin类
    RegressorMixin,  # 回归器Mixin类
    TransformerMixin,  # 转换器Mixin类
    _fit_context,  # 拟合上下文
    clone,  # 克隆函数
)
from ..exceptions import NotFittedError  # 导入未拟合错误异常类
from ..preprocessing import LabelEncoder  # 导入标签编码器
from ..utils import Bunch  # 导入Bunch对象
from ..utils._estimator_html_repr import _VisualBlock  # 导入用于HTML表示的视觉块
from ..utils._param_validation import StrOptions  # 导入字符串选项验证模块
from ..utils.metadata_routing import (  # 导入元数据路由相关模块
    MetadataRouter,  # 元数据路由器
    MethodMapping,  # 方法映射
    _raise_for_params,  # 参数错误时的异常处理
    _routing_enabled,  # 路由是否启用
    process_routing,  # 处理路由
)
from ..utils.metaestimators import available_if  # 导入可用的元估计器
from ..utils.multiclass import type_of_target  # 导入目标类型判断函数
from ..utils.parallel import Parallel, delayed  # 导入并行计算相关函数
from ..utils.validation import (  # 导入验证相关函数
    _check_feature_names_in,  # 检查特征名称是否在列表中
    _deprecate_positional_args,  # 废弃位置参数
    check_is_fitted,  # 检查是否已拟合
    column_or_1d,  # 检查是否为一维数组或列向量
)
from ._base import _BaseHeterogeneousEnsemble, _fit_single_estimator  # 导入基础异质集成相关函数


class _BaseVoting(TransformerMixin, _BaseHeterogeneousEnsemble):
    """Base class for voting.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    _parameter_constraints: dict = {
        "estimators": [list],  # 参数约束：estimators必须是列表
        "weights": ["array-like", None],  # 参数约束：weights可以是数组样式或None
        "n_jobs": [None, Integral],  # 参数约束：n_jobs可以是None或整数
        "verbose": ["verbose"],  # 参数约束：verbose参数必须为verbose
    }

    def _log_message(self, name, idx, total):
        if not self.verbose:  # 如果verbose为False，则不输出日志信息
            return None
        return f"({idx} of {total}) Processing {name}"  # 返回日志信息字符串

    @property
    def _weights_not_none(self):
        """Get the weights of not `None` estimators."""
        if self.weights is None:  # 如果weights为None，则返回None
            return None
        return [w for est, w in zip(self.estimators, self.weights) if est[1] != "drop"]  # 返回非"drop"评估器的权重列表

    def _predict(self, X):
        """Collect results from clf.predict calls."""
        return np.asarray([est.predict(X) for est in self.estimators_]).T  # 收集每个评估器的预测结果并返回转置后的NumPy数组

    @abstractmethod
    def fit(self, X, y, **fit_params):
        """Get common fit operations."""
        # 验证并获取所有分类器的名称和实例
        names, clfs = self._validate_estimators()

        # 如果设定了权重且权重数量与分类器数量不匹配，抛出数值错误异常
        if self.weights is not None and len(self.weights) != len(self.estimators):
            raise ValueError(
                "Number of `estimators` and weights must be equal; got"
                f" {len(self.weights)} weights, {len(self.estimators)} estimators"
            )

        # 如果启用了路由功能，则处理路由参数
        if _routing_enabled():
            routed_params = process_routing(self, "fit", **fit_params)
        else:
            # 否则创建空的路由参数容器，并为每个分类器名称创建一个包含适应阶段参数的Bunch对象
            routed_params = Bunch()
            for name in names:
                routed_params[name] = Bunch(fit={})
                if "sample_weight" in fit_params:
                    routed_params[name].fit["sample_weight"] = fit_params[
                        "sample_weight"
                    ]

        # 并行拟合每个分类器，使用路由参数中的适应阶段参数
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_single_estimator)(
                clone(clf),
                X,
                y,
                fit_params=routed_params[name]["fit"],
                message_clsname="Voting",
                message=self._log_message(name, idx + 1, len(clfs)),
            )
            for idx, (name, clf) in enumerate(zip(names, clfs))
            if clf != "drop"
        )

        # 初始化命名分类器的容器
        self.named_estimators_ = Bunch()

        # 使用'drop'作为被删除分类器的占位符
        est_iter = iter(self.estimators_)
        for name, est in zip(names, self.estimators):
            current_est = est if est == "drop" else next(est_iter)
            self.named_estimators_[name] = current_est

            # 如果当前分类器具有'feature_names_in_'属性，则将其设置为模型的输入特征名称
            if hasattr(current_est, "feature_names_in_"):
                self.feature_names_in_ = current_est.feature_names_in_

        # 返回自身对象
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Return class labels or probabilities for each estimator.

        Return predictions for X for each estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix, dataframe} of shape \
                (n_samples, n_features)
            Input samples.

        y : ndarray of shape (n_samples,), default=None
            Target values (None for unsupervised transformations).

        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        X_new : ndarray array of shape (n_samples, n_features_new)
            Transformed array.
        """
        # 调用父类的fit_transform方法，返回转换后的数据
        return super().fit_transform(X, y, **fit_params)

    @property
    # 返回已拟合对象的特征数
    def n_features_in_(self):
        """Number of features seen during :term:`fit`."""
        # 检查是否已经拟合，如果未拟合则抛出 AttributeError，以便 hasattr() 失败
        try:
            check_is_fitted(self)
        except NotFittedError as nfe:
            raise AttributeError(
                "{} object has no n_features_in_ attribute.".format(
                    self.__class__.__name__
                )
            ) from nfe

        # 返回第一个估算器（estimator）的特征数
        return self.estimators_[0].n_features_in_

    # 创建一个 VisualBlock 对象，用于并行估算器（estimators）的可视化
    def _sk_visual_block_(self):
        names, estimators = zip(*self.estimators)
        return _VisualBlock("parallel", estimators, names=names)

    # 获取对象的元数据路由信息
    def get_metadata_routing(self):
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        .. versionadded:: 1.5

        Returns
        -------
        routing : MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        # 创建一个 MetadataRouter 对象，使用当前类名作为所有者
        router = MetadataRouter(owner=self.__class__.__name__)

        # `self.estimators` 是一个 (name, estimator) 元组的列表
        # 将每个估算器（estimator）及其名称添加到路由器中，并指定方法映射为 'fit' 的调用关系
        for name, estimator in self.estimators:
            router.add(
                **{name: estimator},
                method_mapping=MethodMapping().add(callee="fit", caller="fit"),
            )
        return router
class VotingClassifier(ClassifierMixin, _BaseVoting):
    """Soft Voting/Majority Rule classifier for unfitted estimators.

    Read more in the :ref:`User Guide <voting_classifier>`.

    .. versionadded:: 0.17

    Parameters
    ----------
    estimators : list of (str, estimator) tuples
        Invoking the ``fit`` method on the ``VotingClassifier`` will fit clones
        of those original estimators that will be stored in the class attribute
        ``self.estimators_``. An estimator can be set to ``'drop'`` using
        :meth:`set_params`.

        .. versionchanged:: 0.21
            ``'drop'`` is accepted. Using None was deprecated in 0.22 and
            support was removed in 0.24.

    voting : {'hard', 'soft'}, default='hard'
        If 'hard', uses predicted class labels for majority rule voting.
        Else if 'soft', predicts the class label based on the argmax of
        the sums of the predicted probabilities, which is recommended for
        an ensemble of well-calibrated classifiers.

    weights : array-like of shape (n_classifiers,), default=None
        Sequence of weights (`float` or `int`) to weight the occurrences of
        predicted class labels (`hard` voting) or class probabilities
        before averaging (`soft` voting). Uses uniform weights if `None`.

    n_jobs : int, default=None
        The number of jobs to run in parallel for ``fit``.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

        .. versionadded:: 0.18

    flatten_transform : bool, default=True
        Affects shape of transform output only when voting='soft'
        If voting='soft' and flatten_transform=True, transform method returns
        matrix with shape (n_samples, n_classifiers * n_classes). If
        flatten_transform=False, it returns
        (n_classifiers, n_samples, n_classes).

    verbose : bool, default=False
        If True, the time elapsed while fitting will be printed as it
        is completed.

        .. versionadded:: 0.23

    Attributes
    ----------
    estimators_ : list of classifiers
        The collection of fitted sub-estimators as defined in ``estimators``
        that are not 'drop'.

    named_estimators_ : :class:`~sklearn.utils.Bunch`
        Attribute to access any fitted sub-estimators by name.

        .. versionadded:: 0.20

    le_ : :class:`~sklearn.preprocessing.LabelEncoder`
        Transformer used to encode the labels during fit and decode during
        prediction.

    classes_ : ndarray of shape (n_classes,)
        The classes labels.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying classifier exposes such an attribute when fit.

        .. versionadded:: 0.24
    """

    def __init__(self, estimators, voting='hard', weights=None, n_jobs=None,
                 flatten_transform=True, verbose=False):
        # 继承父类 ClassifierMixin 和 _BaseVoting，并初始化 VotingClassifier
        super().__init__(estimators=estimators)
        
        # 设定投票方式（硬投票或软投票）
        self.voting = voting
        
        # 设定权重数组
        self.weights = weights
        
        # 设定并行处理的作业数
        self.n_jobs = n_jobs
        
        # 设定是否展平转换输出（仅影响 voting='soft' 时）
        self.flatten_transform = flatten_transform
        
        # 设定是否打印拟合过程中的时间信息
        self.verbose = verbose
        
        # 初始化时没有拟合的子估算器
        self.estimators_ = []

        # 初始化命名的子估算器集合为空
        self.named_estimators_ = Bunch()

        # 初始化标签编码器为空
        self.le_ = LabelEncoder()

        # 初始化类别标签为空数组
        self.classes_ = []

        # 初始化特征数量为 0
        self.n_features_in_ = 0
    """
    _parameter_constraints: dict = {
        **_BaseVoting._parameter_constraints,
        "voting": [StrOptions({"hard", "soft"})],
        "flatten_transform": ["boolean"],
    }
    """

    # 定义参数约束字典，继承自 _BaseVoting 的参数约束，并添加 VotingClassifier 特有的约束
    _parameter_constraints: dict = {
        **_BaseVoting._parameter_constraints,
        "voting": [StrOptions({"hard", "soft"})],  # voting 参数只能为 "hard" 或 "soft"
        "flatten_transform": ["boolean"],  # flatten_transform 参数必须为布尔类型
    }

    """
    def __init__(
        self,
        estimators,
        *,
        voting="hard",
        weights=None,
        n_jobs=None,
        flatten_transform=True,
        verbose=False,
    ):
        super().__init__(estimators=estimators)
        self.voting = voting
        self.weights = weights
        self.n_jobs = n_jobs
        self.flatten_transform = flatten_transform
        self.verbose = verbose
    """

    # VotingClassifier 的构造函数，初始化各个参数
    def __init__(
        self,
        estimators,  # 使用的基础估算器列表
        *,
        voting="hard",  # 投票策略，默认为 "hard"
        weights=None,  # 各个估算器的权重，默认为 None
        n_jobs=None,  # 并行运行的作业数，默认为 None
        flatten_transform=True,  # 是否平铺 transform 的输出，默认为 True
        verbose=False,  # 是否输出详细信息，默认为 False
    ):
        super().__init__(estimators=estimators)  # 调用父类 _BaseVoting 的构造函数初始化基础估算器
        self.voting = voting  # 设置投票策略
        self.weights = weights  # 设置各个估算器的权重
        self.n_jobs = n_jobs  # 设置并行作业数
        self.flatten_transform = flatten_transform  # 设置是否平铺 transform 的输出
        self.verbose = verbose  # 设置是否输出详细信息

    """
    @_fit_context(
        # estimators in VotingClassifier.estimators are not validated yet
        prefer_skip_nested_validation=False
    )
    # TODO(1.7): remove `sample_weight` from the signature after deprecation
    # cycle; pop it from `fit_params` before the `_raise_for_params` check and
    """

    # 使用 _fit_context 装饰器，控制 fit 方法的上下文
    @_fit_context(
        # estimators in VotingClassifier.estimators are not validated yet
        prefer_skip_nested_validation=False  # 不优先跳过嵌套验证
    )
    # TODO(1.7): remove `sample_weight` from the signature after deprecation
    # cycle; pop it from `fit_params` before the `_raise_for_params` check and
    # TODO(1.7): 在废弃周期后从签名中删除 `sample_weight`；在 `_raise_for_params` 检查之前从 `fit_params` 弹出
    # 将来重新插入，以确保向后兼容性
    @_deprecate_positional_args(version="1.7")
    def fit(self, X, y, *, sample_weight=None, **fit_params):
        """Fit the estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            训练向量，其中 `n_samples` 是样本数量，`n_features` 是特征数量。

        y : array-like of shape (n_samples,)
            目标值。

        sample_weight : array-like of shape (n_samples,), default=None
            样本权重。如果为 None，则所有样本权重相等。
            注意，仅当所有底层估算器都支持样本权重时才支持。

            .. versionadded:: 0.18

        **fit_params : dict
            传递给底层估算器的参数。

            .. versionadded:: 1.5

                仅在 `enable_metadata_routing=True` 时可用，
                可通过 ``sklearn.set_config(enable_metadata_routing=True)`` 设置。
                更多详情请参见 :ref:`Metadata Routing User Guide <metadata_routing>`。

        Returns
        -------
        self : object
            返回实例本身。
        """
        _raise_for_params(fit_params, self, "fit")  # 检查参数的有效性
        y_type = type_of_target(y, input_name="y")  # 确定目标类型
        if y_type in ("unknown", "continuous"):
            # 针对非分类任务引发特定的 ValueError
            raise ValueError(
                f"Unknown label type: {y_type}. Maybe you are trying to fit a "
                "classifier, which expects discrete classes on a "
                "regression target with continuous values."
            )
        elif y_type not in ("binary", "multiclass"):
            # 针对不支持的分类任务，引发 NotImplementedError，以保持向后兼容性
            raise NotImplementedError(
                f"{self.__class__.__name__} only supports binary or multiclass "
                "classification. Multilabel and multi-output classification are not "
                "supported."
            )

        self.le_ = LabelEncoder().fit(y)  # 使用 LabelEncoder 对目标值进行编码
        self.classes_ = self.le_.classes_  # 保存编码后的类别
        transformed_y = self.le_.transform(y)  # 对目标值进行转换

        if sample_weight is not None:
            fit_params["sample_weight"] = sample_weight  # 如果有样本权重，则加入 fit_params 中

        return super().fit(X, transformed_y, **fit_params)  # 调用父类的 fit 方法进行模型拟合
    def predict(self, X):
        """Predict class labels for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        maj : array-like of shape (n_samples,)
            Predicted class labels.
        """
        # 检查模型是否已拟合
        check_is_fitted(self)
        
        # 如果投票策略是软投票
        if self.voting == "soft":
            # 预测每个样本的类别标签
            maj = np.argmax(self.predict_proba(X), axis=1)
        
        else:  # 'hard' 投票
            # 获取每个分类器的预测结果
            predictions = self._predict(X)
            # 应用 np.apply_along_axis 函数，对每个样本计算加权不为 None 的预测结果中的最大值
            maj = np.apply_along_axis(
                lambda x: np.argmax(np.bincount(x, weights=self._weights_not_none)),
                axis=1,
                arr=predictions,
            )
        
        # 将预测的类别标签反转为原始编码
        maj = self.le_.inverse_transform(maj)

        # 返回预测的类别标签
        return maj

    def _collect_probas(self, X):
        """Collect results from clf.predict calls."""
        # 返回所有分类器对 X 的预测概率结果
        return np.asarray([clf.predict_proba(X) for clf in self.estimators_])

    def _check_voting(self):
        # 如果投票策略是硬投票，则抛出属性错误
        if self.voting == "hard":
            raise AttributeError(
                f"predict_proba is not available when voting={repr(self.voting)}"
            )
        # 投票策略是软投票则返回 True
        return True

    @available_if(_check_voting)
    def predict_proba(self, X):
        """Compute probabilities of possible outcomes for samples in X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        avg : array-like of shape (n_samples, n_classes)
            Weighted average probability for each class per sample.
        """
        # 检查模型是否已拟合
        check_is_fitted(self)
        
        # 计算加权平均概率结果
        avg = np.average(
            self._collect_probas(X), axis=0, weights=self._weights_not_none
        )
        
        # 返回加权平均概率结果
        return avg

    def transform(self, X):
        """Return class labels or probabilities for X for each estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        probabilities_or_labels
            If `voting='soft'` and `flatten_transform=True`:
                returns ndarray of shape (n_samples, n_classifiers * n_classes),
                being class probabilities calculated by each classifier.
            If `voting='soft' and `flatten_transform=False`:
                ndarray of shape (n_classifiers, n_samples, n_classes)
            If `voting='hard'`:
                ndarray of shape (n_samples, n_classifiers), being
                class labels predicted by each classifier.
        """
        # 检查模型是否已拟合
        check_is_fitted(self)
        
        # 如果投票策略是软投票
        if self.voting == "soft":
            # 收集每个分类器对 X 的预测概率结果
            probas = self._collect_probas(X)
            # 如果 flatten_transform=True，则返回展平后的概率结果
            if not self.flatten_transform:
                return probas
            return np.hstack(probas)

        else:
            # 如果投票策略是硬投票，则直接返回预测的类别标签
            return self._predict(X)
    # 获取转换后的特征名称列表，用于输出

    # 检查模型是否已拟合，确保 n_features_in_ 属性存在
    check_is_fitted(self, "n_features_in_")

    # 如果 voting 设置为 "soft" 并且 flatten_transform 设置为 False，则抛出数值错误
    if self.voting == "soft" and not self.flatten_transform:
        raise ValueError(
            "get_feature_names_out is not supported when `voting='soft'` and "
            "`flatten_transform=False`"
        )

    # 检查输入特征名称的有效性，但不生成新名称
    _check_feature_names_in(self, input_features, generate_names=False)

    # 获取当前类名的小写形式
    class_name = self.__class__.__name__.lower()

    # 获取所有非 "drop" 估计器的活跃特征名称
    active_names = [name for name, est in self.estimators if est != "drop"]

    # 如果 voting 设置为 "hard"，返回一个特征名称的 ndarray 数组
    if self.voting == "hard":
        return np.asarray(
            [f"{class_name}_{name}" for name in active_names], dtype=object
        )

    # 如果 voting 设置为 "soft"
    # 获取类别数量
    n_classes = len(self.classes_)

    # 生成带有类别索引后缀的特征名称列表
    names_out = [
        f"{class_name}_{name}{i}" for name in active_names for i in range(n_classes)
    ]

    # 返回特征名称列表的 ndarray 数组
    return np.asarray(names_out, dtype=object)
class VotingRegressor(RegressorMixin, _BaseVoting):
    """Prediction voting regressor for unfitted estimators.

    A voting regressor is an ensemble meta-estimator that fits several base
    regressors, each on the whole dataset. Then it averages the individual
    predictions to form a final prediction.

    Read more in the :ref:`User Guide <voting_regressor>`.

    .. versionadded:: 0.21

    Parameters
    ----------
    estimators : list of (str, estimator) tuples
        Invoking the ``fit`` method on the ``VotingRegressor`` will fit clones
        of those original estimators that will be stored in the class attribute
        ``self.estimators_``. An estimator can be set to ``'drop'`` using
        :meth:`set_params`.

        .. versionchanged:: 0.21
            ``'drop'`` is accepted. Using None was deprecated in 0.22 and
            support was removed in 0.24.

    weights : array-like of shape (n_regressors,), default=None
        Sequence of weights (`float` or `int`) to weight the occurrences of
        predicted values before averaging. Uses uniform weights if `None`.

    n_jobs : int, default=None
        The number of jobs to run in parallel for ``fit``.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    verbose : bool, default=False
        If True, the time elapsed while fitting will be printed as it
        is completed.

        .. versionadded:: 0.23

    Attributes
    ----------
    estimators_ : list of regressors
        The collection of fitted sub-estimators as defined in ``estimators``
        that are not 'drop'.

    named_estimators_ : :class:`~sklearn.utils.Bunch`
        Attribute to access any fitted sub-estimators by name.

        .. versionadded:: 0.20

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying regressor exposes such an attribute when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if the
        underlying estimators expose such an attribute when fit.

        .. versionadded:: 1.0

    See Also
    --------
    VotingClassifier : Soft Voting/Majority Rule classifier.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.ensemble import VotingRegressor
    >>> from sklearn.neighbors import KNeighborsRegressor
    >>> r1 = LinearRegression()
    >>> r2 = RandomForestRegressor(n_estimators=10, random_state=1)
    >>> r3 = KNeighborsRegressor()
    >>> X = np.array([[1, 1], [2, 4], [3, 9], [4, 16], [5, 25], [6, 36]])
    >>> y = np.array([2, 6, 12, 20, 30, 42])
    >>> er = VotingRegressor([('lr', r1), ('rf', r2), ('r3', r3)])

    """

    def __init__(self, estimators, weights=None, n_jobs=None, verbose=False):
        # Initialize the VotingRegressor with provided parameters
        super().__init__(estimators=estimators)
        # Set weights for averaging predictions if provided
        self.weights = weights
        # Number of parallel jobs to run during fitting
        self.n_jobs = n_jobs
        # Whether to print progress during fitting
        self.verbose = verbose
        # Initialize other attributes
        self.estimators_ = []
        self.named_estimators_ = None
        self.n_features_in_ = None
        self.feature_names_in_ = None

    def fit(self, X, y, sample_weight=None):
        # Fit the VotingRegressor with the provided data
        self.estimators_ = []
        # Iterate over provided estimators and fit each on the data
        for name, estimator in self.estimators:
            estimator.fit(X, y, sample_weight=sample_weight)
            self.estimators_.append(estimator)
        # Set attributes related to fitting
        self.named_estimators_ = None
        self.n_features_in_ = None
        self.feature_names_in_ = None
        # Return the fitted VotingRegressor
        return self
    >>> print(er.fit(X, y).predict(X))
    [ 6.8...  8.4... 12.5... 17.8... 26...  34...]

    In the following example, we drop the `'lr'` estimator with
    :meth:`~VotingRegressor.set_params` and fit the remaining two estimators:

    >>> er = er.set_params(lr='drop')
    >>> er = er.fit(X, y)
    >>> len(er.estimators_)
    2
    """


    def __init__(self, estimators, *, weights=None, n_jobs=None, verbose=False):
        super().__init__(estimators=estimators)
        self.weights = weights
        self.n_jobs = n_jobs
        self.verbose = verbose


    @_fit_context(
        # estimators in VotingRegressor.estimators are not validated yet
        prefer_skip_nested_validation=False
    )
    # TODO(1.7): remove `sample_weight` from the signature after deprecation cycle;
    # pop it from `fit_params` before the `_raise_for_params` check and reinsert later,
    # for backwards compatibility
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

            .. versionadded:: 1.5

                Only available if `enable_metadata_routing=True`,
                which can be set by using
                ``sklearn.set_config(enable_metadata_routing=True)``.
                See :ref:`Metadata Routing User Guide <metadata_routing>` for
                more details.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        _raise_for_params(fit_params, self, "fit")
        y = column_or_1d(y, warn=True)
        if sample_weight is not None:
            fit_params["sample_weight"] = sample_weight
        return super().fit(X, y, **fit_params)


    def predict(self, X):
        """Predict regression target for X.

        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the estimators in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted values.
        """
        check_is_fitted(self)
        return np.average(self._predict(X), axis=1, weights=self._weights_not_none)
    def transform(self, X):
        """
        Return predictions for X for each estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        predictions : ndarray of shape (n_samples, n_classifiers)
            Values predicted by each regressor.
        """
        # 确保模型已经拟合，否则引发异常
        check_is_fitted(self)
        # 调用内部方法 _predict 进行预测
        return self._predict(X)

    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Not used, present here for API consistency by convention.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        # 确保模型已经拟合，否则引发异常
        check_is_fitted(self, "n_features_in_")
        # 检查输入特征名称是否合法，生成的特征名称
        _check_feature_names_in(self, input_features, generate_names=False)
        # 获取当前类的小写形式，作为特征名的前缀
        class_name = self.__class__.__name__.lower()
        # 生成输出的特征名数组，每个特征名由类名前缀和特征名组成
        return np.asarray(
            [f"{class_name}_{name}" for name, est in self.estimators if est != "drop"],
            dtype=object,
        )
```