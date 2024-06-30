# `D:\src\scipysrc\scikit-learn\sklearn\ensemble\_base.py`

```
"""Base class for ensemble-based estimators."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 引入抽象基类（Abstract Base Class，ABC）和抽象方法的装饰器
from abc import ABCMeta, abstractmethod
# 引入类型提示，指定参数和返回类型
from typing import List

# 引入NumPy库，用于处理数组和矩阵数据
import numpy as np
# 引入joblib库中的effective_n_jobs函数，用于确定有效的作业数
from joblib import effective_n_jobs

# 从scikit-learn库中引入基本估计器类、元估计器混合类、克隆函数、分类器检查函数、回归器检查函数
from ..base import BaseEstimator, MetaEstimatorMixin, clone, is_classifier, is_regressor
# 引入Bunch类，用于将任意数据打包成一个对象，检查随机状态的函数
from ..utils import Bunch, check_random_state
# 引入标签安全检查模块
from ..utils._tags import _safe_tags
# 引入用户界面函数，用于打印经过的时间
from ..utils._user_interface import _print_elapsed_time
# 引入元数据路由模块，用于判断是否启用路由
from ..utils.metadata_routing import _routing_enabled
# 引入元估计器模块中的基本组合类
from ..utils.metaestimators import _BaseComposition


def _fit_single_estimator(
    estimator, X, y, fit_params, message_clsname=None, message=None
):
    """Private function used to fit an estimator within a job."""
    # TODO(SLEP6): remove if-condition for unrouted sample_weight when metadata
    # routing can't be disabled.
    # 如果未启用元数据路由且fit_params中含有'sample_weight'参数，则执行以下逻辑
    if not _routing_enabled() and "sample_weight" in fit_params:
        try:
            # 使用_print_elapsed_time函数打印消息类名和消息内容，执行估计器的拟合过程
            with _print_elapsed_time(message_clsname, message):
                estimator.fit(X, y, sample_weight=fit_params["sample_weight"])
        except TypeError as exc:
            # 如果出现类型错误，检查是否是不支持'sample_weight'参数
            if "unexpected keyword argument 'sample_weight'" in str(exc):
                raise TypeError(
                    "Underlying estimator {} does not support sample weights.".format(
                        estimator.__class__.__name__
                    )
                ) from exc
            raise
    else:
        # 否则，使用_print_elapsed_time函数打印消息类名和消息内容，执行估计器的拟合过程
        with _print_elapsed_time(message_clsname, message):
            estimator.fit(X, y, **fit_params)
    return estimator


def _set_random_states(estimator, random_state=None):
    """Set fixed random_state parameters for an estimator.

    Finds all parameters ending ``random_state`` and sets them to integers
    derived from ``random_state``.

    Parameters
    ----------
    estimator : estimator supporting get/set_params
        Estimator with potential randomness managed by random_state
        parameters.

    random_state : int, RandomState instance or None, default=None
        Pseudo-random number generator to control the generation of the random
        integers. Pass an int for reproducible output across multiple function
        calls.
        See :term:`Glossary <random_state>`.

    Notes
    -----
    This does not necessarily set *all* ``random_state`` attributes that
    control an estimator's randomness, only those accessible through
    ``estimator.get_params()``.  ``random_state``s not controlled include
    those belonging to:

        * cross-validation splitters
        * ``scipy.stats`` rvs
    """
    # 将random_state参数转换为RandomState实例
    random_state = check_random_state(random_state)
    # 初始化一个空字典用于存储要设置的参数
    to_set = {}
    # 遍历所有深度参数，找到以'random_state'结尾的参数并设置为从random_state派生的整数
    for key in sorted(estimator.get_params(deep=True)):
        if key == "random_state" or key.endswith("__random_state"):
            to_set[key] = random_state.randint(np.iinfo(np.int32).max)

    # 如果有要设置的参数，则调用estimator的set_params方法进行设置
    if to_set:
        estimator.set_params(**to_set)


class BaseEnsemble(MetaEstimatorMixin, BaseEstimator, metaclass=ABCMeta):
    """
    Base class for all ensemble classes.

    Warning: This class should not be used directly. Use derived classes
    instead.

    Parameters
    ----------
    estimator : object
        The base estimator from which the ensemble is built.

    n_estimators : int, default=10
        The number of estimators in the ensemble.

    estimator_params : list of str, default=tuple()
        The list of attributes to use as parameters when instantiating a
        new base estimator. If none are given, default parameters are used.

    Attributes
    ----------
    estimator_ : estimator
        The base estimator from which the ensemble is grown.

    estimators_ : list of estimators
        The collection of fitted base estimators.
    """

    # overwrite _required_parameters from MetaEstimatorMixin
    _required_parameters: List[str] = []

    @abstractmethod
    def __init__(
        self,
        estimator=None,
        *,
        n_estimators=10,
        estimator_params=tuple(),
    ):
        # Set parameters
        self.estimator = estimator  # 设置基础估计器对象
        self.n_estimators = n_estimators  # 设置集成中的估计器数量
        self.estimator_params = estimator_params  # 设置用于实例化新基础估计器的参数列表

        # Don't instantiate estimators now! Parameters of estimator might
        # still change. Eg., when grid-searching with the nested object syntax.
        # self.estimators_ needs to be filled by the derived classes in fit.
        # 不要立即实例化估计器！因为估计器的参数可能会改变，例如在使用嵌套对象语法进行网格搜索时。
        # self.estimators_ 需要在派生类的 fit 方法中填充。

    def _validate_estimator(self, default=None):
        """Check the base estimator.

        Sets the `estimator_` attributes.
        """
        if self.estimator is not None:
            self.estimator_ = self.estimator  # 设置 estimator_ 属性为给定的基础估计器对象
        else:
            self.estimator_ = default  # 如果未提供基础估计器，则使用默认值

    def _make_estimator(self, append=True, random_state=None):
        """Make and configure a copy of the `estimator_` attribute.

        Warning: This method should be used to properly instantiate new
        sub-estimators.
        """
        estimator = clone(self.estimator_)  # 克隆当前的 estimator_
        estimator.set_params(**{p: getattr(self, p) for p in self.estimator_params})  # 根据参数设置 estimator 的参数

        if random_state is not None:
            _set_random_states(estimator, random_state)  # 设置估计器的随机状态

        if append:
            self.estimators_.append(estimator)  # 将新创建的估计器添加到 estimators_ 列表中

        return estimator  # 返回新创建的估计器对象

    def __len__(self):
        """Return the number of estimators in the ensemble."""
        return len(self.estimators_)  # 返回集成中估计器的数量

    def __getitem__(self, index):
        """Return the index'th estimator in the ensemble."""
        return self.estimators_[index]  # 返回集成中索引为 index 的估计器对象

    def __iter__(self):
        """Return iterator over estimators in the ensemble."""
        return iter(self.estimators_)  # 返回一个迭代器，用于遍历集成中的估计器
def _partition_estimators(n_estimators, n_jobs):
    """Private function used to partition estimators between jobs."""
    # 计算并发任务数，取有效的任务数和估算器数量的较小值
    n_jobs = min(effective_n_jobs(n_jobs), n_estimators)

    # 将估算器均匀分配到各个任务中
    n_estimators_per_job = np.full(n_jobs, n_estimators // n_jobs, dtype=int)
    n_estimators_per_job[: n_estimators % n_jobs] += 1
    starts = np.cumsum(n_estimators_per_job)

    return n_jobs, n_estimators_per_job.tolist(), [0] + starts.tolist()


class _BaseHeterogeneousEnsemble(
    MetaEstimatorMixin, _BaseComposition, metaclass=ABCMeta
):
    """Base class for heterogeneous ensemble of learners.

    Parameters
    ----------
    estimators : list of (str, estimator) tuples
        The ensemble of estimators to use in the ensemble. Each element of the
        list is defined as a tuple of string (i.e. name of the estimator) and
        an estimator instance. An estimator can be set to `'drop'` using
        `set_params`.

    Attributes
    ----------
    estimators_ : list of estimators
        The elements of the estimators parameter, having been fitted on the
        training data. If an estimator has been set to `'drop'`, it will not
        appear in `estimators_`.
    """

    _required_parameters = ["estimators"]

    @property
    def named_estimators(self):
        """Dictionary to access any fitted sub-estimators by name.

        Returns
        -------
        :class:`~sklearn.utils.Bunch`
        """
        # 返回一个字典，用于通过名称访问任何已拟合的子估算器
        return Bunch(**dict(self.estimators))

    @abstractmethod
    def __init__(self, estimators):
        self.estimators = estimators

    def _validate_estimators(self):
        if len(self.estimators) == 0:
            raise ValueError(
                "Invalid 'estimators' attribute, 'estimators' should be a "
                "non-empty list of (string, estimator) tuples."
            )
        names, estimators = zip(*self.estimators)
        # 由 MetaEstimatorMixin 定义
        self._validate_names(names)

        has_estimator = any(est != "drop" for est in estimators)
        if not has_estimator:
            raise ValueError(
                "All estimators are dropped. At least one is required "
                "to be an estimator."
            )

        is_estimator_type = is_classifier if is_classifier(self) else is_regressor

        for est in estimators:
            if est != "drop" and not is_estimator_type(est):
                raise ValueError(
                    "The estimator {} should be a {}.".format(
                        est.__class__.__name__, is_estimator_type.__name__[3:]
                    )
                )

        return names, estimators
    def set_params(self, **params):
        """
        Set the parameters of an estimator from the ensemble.

        Valid parameter keys can be listed with `get_params()`. Note that you
        can directly set the parameters of the estimators contained in
        `estimators`.

        Parameters
        ----------
        **params : keyword arguments
            Specific parameters using e.g.
            `set_params(parameter_name=new_value)`. In addition, to setting the
            parameters of the estimator, the individual estimator of the
            estimators can also be set, or can be removed by setting them to
            'drop'.

        Returns
        -------
        self : object
            Estimator instance.
        """
        # 调用父类方法 `_set_params` 来设置 `estimators` 参数的值
        super()._set_params("estimators", **params)
        return self

    def get_params(self, deep=True):
        """
        Get the parameters of an estimator from the ensemble.

        Returns the parameters given in the constructor as well as the
        estimators contained within the `estimators` parameter.

        Parameters
        ----------
        deep : bool, default=True
            Setting it to True gets the various estimators and the parameters
            of the estimators as well.

        Returns
        -------
        params : dict
            Parameter and estimator names mapped to their values or parameter
            names mapped to their values.
        """
        # 调用父类方法 `_get_params` 来获取 `estimators` 参数的值
        return super()._get_params("estimators", deep=deep)

    def _more_tags(self):
        try:
            # 检查所有的子估计器，确认它们是否允许 NaN 值
            allow_nan = all(
                _safe_tags(est[1])["allow_nan"] if est[1] != "drop" else True
                for est in self.estimators
            )
        except Exception:
            # 如果 `estimators` 不符合我们的 API（应为元组列表），则假设不允许 NaN 值
            allow_nan = False
        # 返回额外的标签，指示数据类型是否保持不变以及是否允许 NaN 值
        return {"preserves_dtype": [], "allow_nan": allow_nan}
```