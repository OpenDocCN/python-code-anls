# `D:\src\scipysrc\scikit-learn\sklearn\ensemble\_bagging.py`

```
"""Bagging meta-estimator."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause


import itertools  # 导入 itertools 模块，用于高效循环和迭代操作
import numbers  # 导入 numbers 模块，用于数值相关的工具函数
from abc import ABCMeta, abstractmethod  # 从 abc 模块导入 ABCMeta 和 abstractmethod，用于定义抽象基类
from functools import partial  # 导入 functools 模块中的 partial 函数，用于创建 partial 函数应用
from numbers import Integral  # 从 numbers 模块导入 Integral，用于整数类型的检查
from warnings import warn  # 导入 warn 函数，用于发出警告

import numpy as np  # 导入 NumPy 库，用于数值计算

from ..base import ClassifierMixin, RegressorMixin, _fit_context  # 从上级包中导入分类器和回归器的基类和相关函数
from ..metrics import accuracy_score, r2_score  # 从上级包中导入准确率和 R2 分数评估函数
from ..tree import DecisionTreeClassifier, DecisionTreeRegressor  # 从上级包中导入决策树分类器和回归器
from ..utils import (  # 从上级包中导入各种工具函数和类
    Bunch,
    _safe_indexing,
    check_random_state,
    column_or_1d,
)
from ..utils._mask import indices_to_mask  # 从上级包中导入用于生成掩码的函数
from ..utils._param_validation import HasMethods, Interval, RealNotInt  # 从上级包中导入参数验证相关函数
from ..utils._tags import _safe_tags  # 从上级包中导入安全标签函数
from ..utils.metadata_routing import (  # 从上级包中导入元数据路由相关函数
    MetadataRouter,
    MethodMapping,
    _raise_for_params,
    _routing_enabled,
    get_routing_for_object,
    process_routing,
)
from ..utils.metaestimators import available_if  # 从上级包中导入元估计器相关函数
from ..utils.multiclass import check_classification_targets  # 从上级包中导入检查分类目标的函数
from ..utils.parallel import Parallel, delayed  # 从上级包中导入并行处理相关函数
from ..utils.random import sample_without_replacement  # 从上级包中导入无重复抽样函数
from ..utils.validation import (  # 从上级包中导入验证相关函数
    _check_method_params,
    _check_sample_weight,
    _deprecate_positional_args,
    check_is_fitted,
    has_fit_parameter,
)
from ._base import BaseEnsemble, _partition_estimators  # 从当前包中导入集成学习基类和分割估计器函数

__all__ = ["BaggingClassifier", "BaggingRegressor"]  # 导出的公共接口

MAX_INT = np.iinfo(np.int32).max  # 设置最大整数值为 32 位整数的最大值


def _generate_indices(random_state, bootstrap, n_population, n_samples):
    """Draw randomly sampled indices."""
    # Draw sample indices
    if bootstrap:
        indices = random_state.randint(0, n_population, n_samples)  # 如果使用自助法，从总体中随机抽取样本索引
    else:
        indices = sample_without_replacement(
            n_population, n_samples, random_state=random_state
        )  # 如果不使用自助法，则无重复抽样生成样本索引

    return indices  # 返回生成的样本索引


def _generate_bagging_indices(
    random_state,
    bootstrap_features,
    bootstrap_samples,
    n_features,
    n_samples,
    max_features,
    max_samples,
):
    """Randomly draw feature and sample indices."""
    # Get valid random state
    random_state = check_random_state(random_state)  # 检查并获取有效的随机状态对象

    # Draw indices
    feature_indices = _generate_indices(
        random_state, bootstrap_features, n_features, max_features
    )  # 生成特征索引，可以选择是否使用自助法
    sample_indices = _generate_indices(
        random_state, bootstrap_samples, n_samples, max_samples
    )  # 生成样本索引，可以选择是否使用自助法

    return feature_indices, sample_indices  # 返回生成的特征和样本索引


def _parallel_build_estimators(
    n_estimators,
    ensemble,
    X,
    y,
    seeds,
    total_n_estimators,
    verbose,
    check_input,
    fit_params,
):
    """Private function used to build a batch of estimators within a job."""
    # Retrieve settings
    n_samples, n_features = X.shape  # 获取样本数和特征数
    max_features = ensemble._max_features  # 获取集成模型允许的最大特征数
    max_samples = ensemble._max_samples  # 获取集成模型允许的最大样本数
    bootstrap = ensemble.bootstrap  # 获取是否使用自助法抽样的标志
    bootstrap_features = ensemble.bootstrap_features  # 获取是否使用自助法抽样特征的标志
    has_check_input = has_fit_parameter(ensemble.estimator_, "check_input")  # 检查基本估计器是否支持 check_input 参数
    requires_feature_indexing = bootstrap_features or max_features != n_features  # 判断是否需要对特征进行索引
    # 创建空列表用于存储估算器（estimators）
    estimators = []
    # 创建空列表用于存储估算器的特征（estimators_features）
    estimators_features = []
    
    # TODO: (slep6) 如果未设置路由时，检查是否需要移除样本权重的条件。当元数据路由无法禁用时执行。
    # 检查基础估算器是否支持样本权重
    support_sample_weight = has_fit_parameter(ensemble.estimator_, "sample_weight")
    # 如果未启用路由并且基础估算器不支持样本权重，同时fit_params中传递了样本权重，则引发异常。
    if not _routing_enabled() and (
        not support_sample_weight and fit_params.get("sample_weight") is not None
    ):
        raise ValueError(
            "The base estimator doesn't support sample weight, but sample_weight is "
            "passed to the fit method."
        )
    # 对于每个估计器进行迭代，构建集合中的每一个估计器
    for i in range(n_estimators):
        # 如果设置了详细输出级别大于1，打印当前并行运行中构建估计器的信息
        if verbose > 1:
            print(
                "Building estimator %d of %d for this parallel run (total %d)..."
                % (i + 1, n_estimators, total_n_estimators)
            )

        # 为当前估计器选择随机种子
        random_state = seeds[i]
        # 创建一个新的估计器实例
        estimator = ensemble._make_estimator(append=False, random_state=random_state)

        # 如果支持输入检查，设置部分拟合方法
        if has_check_input:
            estimator_fit = partial(estimator.fit, check_input=check_input)
        else:
            estimator_fit = estimator.fit

        # 生成用于 Bagging 的随机特征和样本索引
        features, indices = _generate_bagging_indices(
            random_state,
            bootstrap_features,
            bootstrap,
            n_features,
            n_samples,
            max_features,
            max_samples,
        )

        # 复制拟合参数
        fit_params_ = fit_params.copy()

        # TODO(SLEP6): 当 metadata 路由不能被禁用时，移除条件
        # 如果路由被启用，获取适用于 ensemble.estimator_ 的路由器
        if _routing_enabled():
            request_or_router = get_routing_for_object(ensemble.estimator_)
            # 检查路由器是否接受样本权重作为输入
            consumes_sample_weight = request_or_router.consumes(
                "fit", ("sample_weight",)
            )
        else:
            # 否则，检查基础估计器是否支持样本权重
            consumes_sample_weight = support_sample_weight

        # 如果支持样本权重
        if consumes_sample_weight:
            # 获取当前样本权重，并进行检查
            curr_sample_weight = _check_sample_weight(
                fit_params_.pop("sample_weight", None), X
            ).copy()

            # 如果使用自助法（bootstrap），计算样本的频次并调整当前样本权重
            if bootstrap:
                sample_counts = np.bincount(indices, minlength=n_samples)
                curr_sample_weight *= sample_counts
            else:
                # 否则，将不在索引中的样本权重置为0
                not_indices_mask = ~indices_to_mask(indices, n_samples)
                curr_sample_weight[not_indices_mask] = 0

            # 更新拟合参数中的样本权重
            fit_params_["sample_weight"] = curr_sample_weight
            # 如果需要进行特征索引，将 X_ 限制为选定的特征列
            X_ = X[:, features] if requires_feature_indexing else X
            # 使用调整后的样本权重拟合估计器
            estimator_fit(X_, y, **fit_params_)
        else:
            # 如果不支持样本权重，则通过索引方式处理
            y_ = _safe_indexing(y, indices)
            X_ = _safe_indexing(X, indices)
            # 检查方法参数，并进行适当的调整
            fit_params_ = _check_method_params(X, params=fit_params_, indices=indices)
            # 如果需要进行特征索引，将 X_ 限制为选定的特征列
            if requires_feature_indexing:
                X_ = X_[:, features]
            # 使用索引后的数据拟合估计器
            estimator_fit(X_, y_, **fit_params_)

        # 将当前估计器添加到估计器列表中
        estimators.append(estimator)
        # 将当前特征集合添加到特征列表中
        estimators_features.append(features)

    # 返回构建的所有估计器及其对应的特征集合
    return estimators, estimators_features
# 私有函数，用于在一个作业中计算概率预测或预测
def _parallel_predict_proba(estimators, estimators_features, X, n_classes):
    # 获取样本数量
    n_samples = X.shape[0]
    # 创建一个全零的概率矩阵，形状为 (样本数, 类别数)
    proba = np.zeros((n_samples, n_classes))

    # 遍历每个估算器及其对应的特征集合
    for estimator, features in zip(estimators, estimators_features):
        # 检查估算器是否具有 predict_proba 方法
        if hasattr(estimator, "predict_proba"):
            # 使用估算器的 predict_proba 方法得到预测概率
            proba_estimator = estimator.predict_proba(X[:, features])

            # 如果类别数与估算器的类别数相同，直接累加预测概率
            if n_classes == len(estimator.classes_):
                proba += proba_estimator
            else:
                # 否则，根据估算器的类别重新累加预测概率
                proba[:, estimator.classes_] += proba_estimator[
                    :, range(len(estimator.classes_))
                ]

        else:
            # 如果估算器没有 predict_proba 方法，采用投票方式进行预测
            predictions = estimator.predict(X[:, features])

            # 对每个样本的预测结果进行累加
            for i in range(n_samples):
                proba[i, predictions[i]] += 1

    return proba


# 私有函数，用于在一个作业中计算对数概率
def _parallel_predict_log_proba(estimators, estimators_features, X, n_classes):
    # 获取样本数量
    n_samples = X.shape[0]
    # 创建一个填满负无穷的对数概率矩阵，形状为 (样本数, 类别数)
    log_proba = np.empty((n_samples, n_classes))
    log_proba.fill(-np.inf)
    # 创建一个包含所有类别的数组
    all_classes = np.arange(n_classes, dtype=int)

    # 遍历每个估算器及其对应的特征集合
    for estimator, features in zip(estimators, estimators_features):
        # 使用估算器的 predict_log_proba 方法得到对数概率
        log_proba_estimator = estimator.predict_log_proba(X[:, features])

        # 如果类别数与估算器的类别数相同，使用 logaddexp 函数累加对数概率
        if n_classes == len(estimator.classes_):
            log_proba = np.logaddexp(log_proba, log_proba_estimator)

        else:
            # 否则，根据估算器的类别重新累加对数概率
            log_proba[:, estimator.classes_] = np.logaddexp(
                log_proba[:, estimator.classes_],
                log_proba_estimator[:, range(len(estimator.classes_))],
            )
            # 找出缺失的类别，将对应位置的对数概率设为负无穷
            missing = np.setdiff1d(all_classes, estimator.classes_)
            log_proba[:, missing] = np.logaddexp(log_proba[:, missing], -np.inf)

    return log_proba


# 私有函数，用于在一个作业中计算决策函数
def _parallel_decision_function(estimators, estimators_features, X):
    # 返回所有估算器在特征集合上的决策函数值之和
    return sum(
        estimator.decision_function(X[:, features])
        for estimator, features in zip(estimators, estimators_features)
    )


# 私有函数，用于在一个作业中进行回归预测
def _parallel_predict_regression(estimators, estimators_features, X):
    # 返回所有估算器在特征集合上的预测值之和
    return sum(
        estimator.predict(X[:, features])
        for estimator, features in zip(estimators, estimators_features)
    )


# 返回一个函数，检查是否可以委托一个方法给基础估算器
def _estimator_has(attr):
    """Check if we can delegate a method to the underlying estimator.

    First, we check the first fitted estimator if available, otherwise we
    check the estimator attribute.
    """
    def check(self):
        # 如果有 estimators_ 属性，则检查第一个已拟合的估算器是否具有 attr 属性
        if hasattr(self, "estimators_"):
            return hasattr(self.estimators_[0], attr)
        else:  # 否则，检查 estimator 属性是否存在且具有 attr 属性
            return hasattr(self.estimator, attr)

    return check


class BaseBagging(BaseEnsemble, metaclass=ABCMeta):
    """Base class for Bagging meta-estimator."""
    # 警告：不应直接使用此类。请使用派生类。
    """
    
    # 参数约束字典，定义了Bagging模型的参数约束条件
    _parameter_constraints: dict = {
        "estimator": [HasMethods(["fit", "predict"]), None],  # estimator参数要求具有fit和predict方法，或者为None
        "n_estimators": [Interval(Integral, 1, None, closed="left")],  # n_estimators参数为大于等于1的整数
        "max_samples": [
            Interval(Integral, 1, None, closed="left"),  # max_samples参数为大于等于1的整数
            Interval(RealNotInt, 0, 1, closed="right"),  # max_samples参数为0到1之间的实数，右闭区间
        ],
        "max_features": [
            Interval(Integral, 1, None, closed="left"),  # max_features参数为大于等于1的整数
            Interval(RealNotInt, 0, 1, closed="right"),  # max_features参数为0到1之间的实数，右闭区间
        ],
        "bootstrap": ["boolean"],  # bootstrap参数为布尔值
        "bootstrap_features": ["boolean"],  # bootstrap_features参数为布尔值
        "oob_score": ["boolean"],  # oob_score参数为布尔值
        "warm_start": ["boolean"],  # warm_start参数为布尔值
        "n_jobs": [None, Integral],  # n_jobs参数为None或整数
        "random_state": ["random_state"],  # random_state参数为random_state对象
        "verbose": ["verbose"],  # verbose参数为verbose对象
    }
    
    # 抽象方法，初始化Bagging模型的参数
    @abstractmethod
    def __init__(
        self,
        estimator=None,
        n_estimators=10,
        *,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=True,
        bootstrap_features=False,
        oob_score=False,
        warm_start=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
    ):
        # 调用父类初始化方法，设置estimator和n_estimators参数
        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
        )
        # 设置Bagging模型的其他参数
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.oob_score = oob_score
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
    
    # TODO（1.7）：在废弃周期后，从签名中删除`sample_weight`；在`_raise_for_params`检查之前从`fit_params`中弹出它，并以向后兼容的方式重新插入
    # 标记为废弃的位置参数，版本为1.7
    @_deprecate_positional_args(version="1.7")
    # 进入拟合上下文，BaseBagging.estimator尚未验证
    @_fit_context(
        prefer_skip_nested_validation=False
    )
    def fit(self, X, y, *, sample_weight=None, **fit_params):
        """Build a Bagging ensemble of estimators from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        y : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if the base estimator supports
            sample weighting.

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

        # Convert data (X is required to be 2d and indexable)
        X, y = self._validate_data(
            X,
            y,
            accept_sparse=["csr", "csc"],  # Accept sparse formats 'csr' and 'csc'
            dtype=None,  # Allow any data type
            force_all_finite=False,  # Do not force all data to be finite
            multi_output=True,  # Support multi-output format
        )

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=None)
            fit_params["sample_weight"] = sample_weight

        # Call the internal fitting method with specified parameters
        return self._fit(X, y, max_samples=self.max_samples, **fit_params)

    def _parallel_args(self):
        """Return empty dictionary for parallel execution arguments."""
        return {}

    def _fit(
        self,
        X,
        y,
        max_samples=None,
        max_depth=None,
        check_input=True,
        **fit_params,
    ):
        """Internal fitting method for the Bagging ensemble."""
        # This method is expected to be implemented in derived classes
        raise NotImplementedError("Method '_fit' must be implemented in subclass")

    @abstractmethod
    def _set_oob_score(self, X, y):
        """Calculate out of bag predictions and score."""
        # This method is intended to calculate and set out-of-bag scores
        # It must be implemented in derived classes

    def _validate_y(self, y):
        """Validate target labels y and ensure compatibility."""
        if len(y.shape) == 1 or y.shape[1] == 1:
            return column_or_1d(y, warn=True)  # Ensure y is 1-dimensional
        return y

    def _get_estimators_indices(self):
        """Generate indices for sampling both features and samples."""
        # Iterate over random seeds to generate indices for bagging
        for seed in self._seeds:
            # Obtain indices for both features and samples
            # Ensure these operations match those in `_parallel_build_estimators()`
            feature_indices, sample_indices = _generate_bagging_indices(
                seed,
                self.bootstrap_features,
                self.bootstrap,
                self.n_features_in_,
                self._n_samples,
                self._max_features,
                self._max_samples,
            )

            yield feature_indices, sample_indices

    @property
    def estimators_samples_(self):
        """
        The subset of drawn samples for each base estimator.

        Returns a dynamically generated list of indices identifying
        the samples used for fitting each member of the ensemble, i.e.,
        the in-bag samples.

        Note: the list is re-created at each call to the property in order
        to reduce the object memory footprint by not storing the sampling
        data. Thus fetching the property may be slower than expected.
        """
        # 返回一个动态生成的列表，包含每个基础估计器使用的样本索引
        return [sample_indices for _, sample_indices in self._get_estimators_indices()]

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
        # 创建一个 MetadataRouter 对象来存储元数据路由信息
        router = MetadataRouter(owner=self.__class__.__name__)
        # 将估计器和方法映射信息添加到路由器中
        router.add(
            estimator=self._get_estimator(),
            method_mapping=MethodMapping().add(callee="fit", caller="fit"),
        )
        return router

    @abstractmethod
    def _get_estimator(self):
        """Resolve which estimator to return."""
        # 抽象方法：解析并返回哪个估计器对象

    def _more_tags(self):
        # 返回一个字典，包含有关估计器的额外标签信息
        return {"allow_nan": _safe_tags(self._get_estimator(), "allow_nan")}
class BaggingClassifier(ClassifierMixin, BaseBagging):
    """A Bagging classifier.

    A Bagging classifier is an ensemble meta-estimator that fits base
    classifiers each on random subsets of the original dataset and then
    aggregate their individual predictions (either by voting or by averaging)
    to form a final prediction. Such a meta-estimator can typically be used as
    a way to reduce the variance of a black-box estimator (e.g., a decision
    tree), by introducing randomization into its construction procedure and
    then making an ensemble out of it.

    This algorithm encompasses several works from the literature. When random
    subsets of the dataset are drawn as random subsets of the samples, then
    this algorithm is known as Pasting [1]_. If samples are drawn with
    replacement, then the method is known as Bagging [2]_. When random subsets
    of the dataset are drawn as random subsets of the features, then the method
    is known as Random Subspaces [3]_. Finally, when base estimators are built
    on subsets of both samples and features, then the method is known as
    Random Patches [4]_.

    Read more in the :ref:`User Guide <bagging>`.

    .. versionadded:: 0.15

    Parameters
    ----------
    estimator : object, default=None
        The base estimator to fit on random subsets of the dataset.
        If None, then the base estimator is a
        :class:`~sklearn.tree.DecisionTreeClassifier`.

        .. versionadded:: 1.2
           `base_estimator` was renamed to `estimator`.

    n_estimators : int, default=10
        The number of base estimators in the ensemble.

    max_samples : int or float, default=1.0
        The number of samples to draw from X to train each base estimator (with
        replacement by default, see `bootstrap` for more details).

        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples.

    max_features : int or float, default=1.0
        The number of features to draw from X to train each base estimator (
        without replacement by default, see `bootstrap_features` for more
        details).

        - If int, then draw `max_features` features.
        - If float, then draw `max(1, int(max_features * n_features_in_))` features.

    bootstrap : bool, default=True
        Whether samples are drawn with replacement. If False, sampling
        without replacement is performed.

    bootstrap_features : bool, default=False
        Whether features are drawn with replacement.

    oob_score : bool, default=False
        Whether to use out-of-bag samples to estimate
        the generalization error. Only available if bootstrap=True.
    """
    warm_start : bool, default=False
        是否启用热启动模式。若设置为True，则会复用上次调用fit方法的解决方案，并添加更多的估计器到集成中；否则，将会重新拟合一个全新的集成模型。详见术语表中的“热启动”。

        .. versionadded:: 0.17
           *warm_start* 构造参数。

    n_jobs : int, default=None
        并行运行的作业数量，用于 :meth:`fit` 和 :meth:`predict` 方法。``None`` 表示使用1个作业，除非在 :obj:`joblib.parallel_backend` 上下文中。``-1`` 表示使用所有处理器。详见术语表中的“作业数量”。

    random_state : int, RandomState instance or None, default=None
        控制对原始数据集的随机重采样（样本和特征方向）。
        如果基础估计器接受 `random_state` 属性，则为集成中的每个实例生成不同的种子。
        传递一个整数可以保证在多次函数调用中获得可重现的输出。详见术语表中的“随机种子”。

    verbose : int, default=0
        控制拟合和预测时的详细程度。

    Attributes
    ----------
    estimator_ : estimator
        用于生成集成的基础估计器。

        .. versionadded:: 1.2
           将 `base_estimator_` 更名为 `estimator_`。

    n_features_in_ : int
        :term:`fit` 过程中观察到的特征数。

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        在 `X` 具有全为字符串的特征名称时定义的特征名称。

        .. versionadded:: 1.0

    estimators_ : list of estimators
        拟合的基础估计器的集合。

    estimators_samples_ : list of arrays
        每个基础估计器的抽取样本子集（即袋外样本）。每个子集由选定的索引数组定义。

    estimators_features_ : list of arrays
        每个基础估计器的抽取特征子集。

    classes_ : ndarray of shape (n_classes,)
        类别标签。

    n_classes_ : int or list
        类别数量。

    oob_score_ : float
        使用袋外估计得到的训练数据集分数。仅当 ``oob_score`` 为True时存在此属性。

    oob_decision_function_ : ndarray of shape (n_samples, n_classes)
        使用袋外估计在训练集上计算的决策函数。如果 n_estimators 较小，可能会出现在自举过程中从未有数据点被排除的情况下，`oob_decision_function_` 可能包含NaN。仅当 ``oob_score`` 为True时存在此属性。

    See Also
    --------
    BaggingRegressor : 一个 Bagging 回归器。

    References
    ----------
    def __init__(
        self,
        estimator=None,
        n_estimators=10,
        *,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=True,
        bootstrap_features=False,
        oob_score=False,
        warm_start=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
    ):
        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )


# 初始化方法，用于创建 BaggingClassifier 实例
# 参数说明：
#   - estimator: 基础分类器，默认为 None
#   - n_estimators: 子分类器的数量，默认为 10
#   - max_samples: 每个子分类器的样本数目比例，默认为 1.0（使用所有样本）
#   - max_features: 每个子分类器的特征数目比例，默认为 1.0（使用所有特征）
#   - bootstrap: 是否使用自助法（bootstrap），默认为 True
#   - bootstrap_features: 是否在特征上使用自助法， 默认为 False
#   - oob_score: 是否计算 out-of-bag 得分，默认为 False
#   - warm_start: 是否热启动，即保留前次调用后的子分类器，默认为 False
#   - n_jobs: 并行运行的作业数目，默认为 None（不并行）
#   - random_state: 随机数种子，默认为 None
#   - verbose: 控制详细程度，默认为 0（不输出详细信息）
super().__init__(
    estimator=estimator,
    n_estimators=n_estimators,
    max_samples=max_samples,
    max_features=max_features,
    bootstrap=bootstrap,
    bootstrap_features=bootstrap_features,
    oob_score=oob_score,
    warm_start=warm_start,
    n_jobs=n_jobs,
    random_state=random_state,
    verbose=verbose,
)


    def _get_estimator(self):
        """Resolve which estimator to return (default is DecisionTreeClassifier)"""
        if self.estimator is None:
            # 如果没有指定基础分类器，则返回默认的 DecisionTreeClassifier
            return DecisionTreeClassifier()
        # 否则返回指定的基础分类器
        return self.estimator
    # 设置Out-of-Bag（OOB）评分的私有方法，用于计算随机森林的预测结果
    def _set_oob_score(self, X, y):
        # 获取样本数
        n_samples = y.shape[0]
        # 获取类别数
        n_classes_ = self.n_classes_

        # 初始化预测矩阵，全为零
        predictions = np.zeros((n_samples, n_classes_))

        # 遍历每棵树的估计器、采样样本索引、特征索引
        for estimator, samples, features in zip(
            self.estimators_, self.estimators_samples_, self.estimators_features_
        ):
            # 创建Out-of-Bag样本的掩码
            mask = ~indices_to_mask(samples, n_samples)

            # 如果估计器有“predict_proba”方法，使用预测概率累加到predictions
            if hasattr(estimator, "predict_proba"):
                predictions[mask, :] += estimator.predict_proba(
                    (X[mask, :])[:, features]
                )
            # 否则，使用预测结果进行投票
            else:
                p = estimator.predict((X[mask, :])[:, features])
                j = 0
                for i in range(n_samples):
                    if mask[i]:
                        predictions[i, p[j]] += 1
                        j += 1

        # 如果有样本的预测总和为零，发出警告
        if (predictions.sum(axis=1) == 0).any():
            warn(
                "Some inputs do not have OOB scores. "
                "This probably means too few estimators were used "
                "to compute any reliable oob estimates."
            )

        # 计算OOB决策函数
        oob_decision_function = predictions / predictions.sum(axis=1)[:, np.newaxis]
        # 计算OOB评分
        oob_score = accuracy_score(y, np.argmax(predictions, axis=1))

        # 设置模型的OOB决策函数和OOB评分
        self.oob_decision_function_ = oob_decision_function
        self.oob_score_ = oob_score

    # 验证目标变量y的有效性并转换为数值编码
    def _validate_y(self, y):
        # 将y转换为一维数组或列向量
        y = column_or_1d(y, warn=True)
        # 检查分类目标的有效性
        check_classification_targets(y)
        # 获取唯一的类别并返回y的数值编码
        self.classes_, y = np.unique(y, return_inverse=True)
        # 记录类别数
        self.n_classes_ = len(self.classes_)

        return y

    # 预测输入样本X的类别
    def predict(self, X):
        """Predict class for X.

        The predicted class of an input sample is computed as the class with
        the highest mean predicted probability. If base estimators do not
        implement a ``predict_proba`` method, then it resorts to voting.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes.
        """
        # 预测每个样本的类别概率
        predicted_probabilitiy = self.predict_proba(X)
        # 返回具有最高平均预测概率的类别
        return self.classes_.take((np.argmax(predicted_probabilitiy, axis=1)), axis=0)
    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        The predicted class probabilities of an input sample is computed as
        the mean predicted class probabilities of the base estimators in the
        ensemble. If base estimators do not implement a ``predict_proba``
        method, then it resorts to voting and the predicted class probabilities
        of an input sample represents the proportion of estimators predicting
        each class.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        # Ensure the estimator is fitted before prediction
        check_is_fitted(self)

        # Validate and preprocess input data
        X = self._validate_data(
            X,
            accept_sparse=["csr", "csc"],  # Accept Compressed Sparse Row or Column format
            dtype=None,
            force_all_finite=False,
            reset=False,
        )

        # Partition estimators for parallel processing
        n_jobs, _, starts = _partition_estimators(self.n_estimators, self.n_jobs)

        # Perform parallel prediction of probabilities
        all_proba = Parallel(
            n_jobs=n_jobs, verbose=self.verbose, **self._parallel_args()
        )(
            delayed(_parallel_predict_proba)(
                self.estimators_[starts[i] : starts[i + 1]],  # Subset of estimators for this job
                self.estimators_features_[starts[i] : starts[i + 1]],  # Subset of estimator features
                X,  # Input data to predict probabilities for
                self.n_classes_,  # Number of classes for prediction
            )
            for i in range(n_jobs)
        )

        # Aggregate predictions from all jobs and compute mean probabilities
        proba = sum(all_proba) / self.n_estimators

        return proba
    def predict_log_proba(self, X):
        """Predict class log-probabilities for X.

        The predicted class log-probabilities of an input sample is computed as
        the log of the mean predicted class probabilities of the base
        estimators in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class log-probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        # Ensure that the estimator is fitted before prediction
        check_is_fitted(self)
        
        # Check if the estimator supports log-probability prediction
        if hasattr(self.estimator_, "predict_log_proba"):
            # Validate and preprocess input data X
            X = self._validate_data(
                X,
                accept_sparse=["csr", "csc"],
                dtype=None,
                force_all_finite=False,
                reset=False,
            )

            # Determine number of jobs and split the estimators for parallel execution
            n_jobs, _, starts = _partition_estimators(self.n_estimators, self.n_jobs)

            # Parallel computation of log-probabilities using joblib
            all_log_proba = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
                delayed(_parallel_predict_log_proba)(
                    self.estimators_[starts[i] : starts[i + 1]],
                    self.estimators_features_[starts[i] : starts[i + 1]],
                    X,
                    self.n_classes_,
                )
                for i in range(n_jobs)
            )

            # Reduce the computed log-probabilities
            log_proba = all_log_proba[0]
            for j in range(1, len(all_log_proba)):
                log_proba = np.logaddexp(log_proba, all_log_proba[j])

            # Normalize by the number of estimators
            log_proba -= np.log(self.n_estimators)

        else:
            # Fall back to using predict_proba and then compute log-probabilities
            log_proba = np.log(self.predict_proba(X))

        return log_proba

    @available_if(_estimator_has("decision_function"))
    def decision_function(self, X):
        """
        Average of the decision functions of the base classifiers.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        score : ndarray of shape (n_samples, k)
            The decision function of the input samples. The columns correspond
            to the classes in sorted order, as they appear in the attribute
            ``classes_``. Regression and binary classification are special
            cases with ``k == 1``, otherwise ``k==n_classes``.
        """
        # 检查模型是否已拟合
        check_is_fitted(self)

        # 验证输入数据 X，并确保其格式符合要求
        X = self._validate_data(
            X,
            accept_sparse=["csr", "csc"],
            dtype=None,
            force_all_finite=False,
            reset=False,
        )

        # 并行处理每个基分类器的决策函数
        n_jobs, _, starts = _partition_estimators(self.n_estimators, self.n_jobs)

        all_decisions = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(_parallel_decision_function)(
                self.estimators_[starts[i] : starts[i + 1]],
                self.estimators_features_[starts[i] : starts[i + 1]],
                X,
            )
            for i in range(n_jobs)
        )

        # 对所有基分类器的决策函数取平均
        decisions = sum(all_decisions) / self.n_estimators

        return decisions
class BaggingRegressor(RegressorMixin, BaseBagging):
    """A Bagging regressor.

    A Bagging regressor is an ensemble meta-estimator that fits base
    regressors each on random subsets of the original dataset and then
    aggregate their individual predictions (either by voting or by averaging)
    to form a final prediction. Such a meta-estimator can typically be used as
    a way to reduce the variance of a black-box estimator (e.g., a decision
    tree), by introducing randomization into its construction procedure and
    then making an ensemble out of it.

    This algorithm encompasses several works from the literature. When random
    subsets of the dataset are drawn as random subsets of the samples, then
    this algorithm is known as Pasting [1]_. If samples are drawn with
    replacement, then the method is known as Bagging [2]_. When random subsets
    of the dataset are drawn as random subsets of the features, then the method
    is known as Random Subspaces [3]_. Finally, when base estimators are built
    on subsets of both samples and features, then the method is known as
    Random Patches [4]_.

    Read more in the :ref:`User Guide <bagging>`.

    .. versionadded:: 0.15

    Parameters
    ----------
    estimator : object, default=None
        The base estimator to fit on random subsets of the dataset.
        If None, then the base estimator is a
        :class:`~sklearn.tree.DecisionTreeRegressor`.

        .. versionadded:: 1.2
           `base_estimator` was renamed to `estimator`.

    n_estimators : int, default=10
        The number of base estimators in the ensemble.

    max_samples : int or float, default=1.0
        The number of samples to draw from X to train each base estimator (with
        replacement by default, see `bootstrap` for more details).

        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples.

    max_features : int or float, default=1.0
        The number of features to draw from X to train each base estimator (
        without replacement by default, see `bootstrap_features` for more
        details).

        - If int, then draw `max_features` features.
        - If float, then draw `max(1, int(max_features * n_features_in_))` features.

    bootstrap : bool, default=True
        Whether samples are drawn with replacement. If False, sampling
        without replacement is performed.

    bootstrap_features : bool, default=False
        Whether features are drawn with replacement.

    oob_score : bool, default=False
        Whether to use out-of-bag samples to estimate
        the generalization error. Only available if bootstrap=True.

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit
        a whole new ensemble. See :term:`the Glossary <warm_start>`.

    """
    n_jobs : int, default=None
        控制并行运行的作业数量，用于 :meth:`fit` 和 :meth:`predict` 方法。如果没有指定，通常为1，除非在 :obj:`joblib.parallel_backend` 上下文中，此时为全部处理器。使用 :term:`Glossary <n_jobs>` 查看更多细节。

    random_state : int, RandomState instance or None, default=None
        控制对原始数据集进行随机重采样（逐样本和逐特征）。如果基础估算器接受 `random_state` 属性，每个集成实例会生成不同的种子。传递整数以便多次函数调用时产生可重现的输出。参见 :term:`Glossary <random_state>`。

    verbose : int, default=0
        控制拟合和预测时的详细程度。

    Attributes
    ----------
    estimator_ : estimator
        用于生长集成的基础估算器。

        .. versionadded:: 1.2
           `base_estimator_` 已更名为 `estimator_`。

    n_features_in_ : int
        在 :term:`fit` 过程中观察到的特征数量。

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        在 :term:`fit` 过程中观察到的特征名称。仅当 `X` 的特征名称全部为字符串时定义。

        .. versionadded:: 1.0

    estimators_ : list of estimators
        安装好的子估算器集合。

    estimators_samples_ : list of arrays
        每个基础估算器的抽取样本子集（即袋外样本）。每个子集由选定的索引数组定义。

    estimators_features_ : list of arrays
        每个基础估算器的抽取特征子集。

    oob_score_ : float
        使用袋外估计得分的训练数据集得分。仅当 ``oob_score`` 为 True 时存在此属性。

    oob_prediction_ : ndarray of shape (n_samples,)
        使用训练集的袋外估计进行预测。如果 n_estimators 较小，可能存在某个数据点在自助法期间从未被排除的情况。在这种情况下，`oob_prediction_` 可能包含 NaN。仅当 ``oob_score`` 为 True 时存在此属性。

    See Also
    --------
    BaggingClassifier : 一个 Bagging 分类器。

    References
    ----------

    .. [1] L. Breiman, "Pasting small votes for classification in large
           databases and on-line", Machine Learning, 36(1), 85-103, 1999.

    .. [2] L. Breiman, "Bagging predictors", Machine Learning, 24(2), 123-140,
           1996.

    .. [3] T. Ho, "The random subspace method for constructing decision
           forests", Pattern Analysis and Machine Intelligence, 20(8), 832-844,
           1998.

    .. [4] G. Louppe and P. Geurts, "Ensembles on Random Patches", Machine
           Learning and Knowledge Discovery in Databases, 346-361, 2012.

    Examples
    --------
    >>> from sklearn.svm import SVR
    >>> from sklearn.ensemble import BaggingRegressor
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=100, n_features=4,
    ...                        n_informative=2, n_targets=1,
    ...                        random_state=0, shuffle=False)
    >>> regr = BaggingRegressor(estimator=SVR(),
    ...                         n_estimators=10, random_state=0).fit(X, y)
    >>> regr.predict([[0, 0, 0, 0]])
    array([-2.8720...])
    """

    # BaggingRegressor 类的构造函数
    def __init__(
        self,
        estimator=None,
        n_estimators=10,
        *,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=True,
        bootstrap_features=False,
        oob_score=False,
        warm_start=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
    ):
        # 调用父类的构造函数
        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )

    # 预测函数，用于预测回归目标值
    def predict(self, X):
        """Predict regression target for X.

        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the estimators in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted values.
        """
        # 检查模型是否已拟合
        check_is_fitted(self)
        # 验证数据
        X = self._validate_data(
            X,
            accept_sparse=["csr", "csc"],
            dtype=None,
            force_all_finite=False,
            reset=False,
        )

        # 并行循环
        n_jobs, _, starts = _partition_estimators(self.n_estimators, self.n_jobs)

        all_y_hat = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(_parallel_predict_regression)(
                self.estimators_[starts[i] : starts[i + 1]],
                self.estimators_features_[starts[i] : starts[i + 1]],
                X,
            )
            for i in range(n_jobs)
        )

        # 汇总结果
        y_hat = sum(all_y_hat) / self.n_estimators

        return y_hat
    # 设置 Out-of-Bag（OOB）评分
    def _set_oob_score(self, X, y):
        # 获取样本数量
        n_samples = y.shape[0]

        # 初始化预测和预测次数数组
        predictions = np.zeros((n_samples,))
        n_predictions = np.zeros((n_samples,))

        # 遍历每个基础估计器、其样本索引和特征索引
        for estimator, samples, features in zip(
            self.estimators_, self.estimators_samples_, self.estimators_features_
        ):
            # 创建 OOB 样本的掩码
            mask = ~indices_to_mask(samples, n_samples)

            # 对 OOB 样本进行预测并累加预测值
            predictions[mask] += estimator.predict((X[mask, :])[:, features])
            # 记录对每个样本的预测次数
            n_predictions[mask] += 1

        # 如果存在没有预测次数的情况，发出警告
        if (n_predictions == 0).any():
            warn(
                "Some inputs do not have OOB scores. "
                "This probably means too few estimators were used "
                "to compute any reliable oob estimates."
            )
            # 避免除以零，将没有预测次数的位置设为1
            n_predictions[n_predictions == 0] = 1

        # 计算每个样本的平均预测值
        predictions /= n_predictions

        # 设置 OOB 预测值属性
        self.oob_prediction_ = predictions
        # 计算并设置 OOB 分数属性
        self.oob_score_ = r2_score(y, predictions)

    # 获取基础估计器
    def _get_estimator(self):
        """Resolve which estimator to return (default is DecisionTreeClassifier)"""
        # 如果未指定估计器，则返回默认的决策树回归器
        if self.estimator is None:
            return DecisionTreeRegressor()
        # 否则返回指定的估计器
        return self.estimator
```