# `D:\src\scipysrc\scikit-learn\sklearn\metrics\_scorer.py`

```
# 导入必要的模块和函数
import copy  # 导入深拷贝函数
import warnings  # 导入警告模块
from collections import Counter  # 导入计数器
from functools import partial  # 导入函数偏函数化模块
from inspect import signature  # 导入签名模块
from traceback import format_exc  # 导入异常跟踪模块

# 导入所需的基础模块和功能
from ..base import is_regressor  # 从基础模块导入是否为回归器函数
from ..utils import Bunch  # 从工具模块导入 Bunch 类
from ..utils._param_validation import HasMethods, Hidden, StrOptions, validate_params  # 导入参数验证相关函数
from ..utils._response import _get_response_values  # 导入获取响应值函数
from ..utils.metadata_routing import (
    MetadataRequest, MetadataRouter, MethodMapping, _MetadataRequester,
    _raise_for_params, _routing_enabled, get_routing_for_object, process_routing
)  # 导入元数据路由相关函数和类
from ..utils.validation import _check_response_method  # 导入响应方法验证函数
from . import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, class_likelihood_ratios, d2_absolute_error_score,
    explained_variance_score, f1_score, jaccard_score, log_loss,
    matthews_corrcoef, max_error, mean_absolute_error,
    mean_absolute_percentage_error, mean_gamma_deviance,
    mean_poisson_deviance, mean_squared_error, mean_squared_log_error,
    median_absolute_error, precision_score, r2_score, recall_score,
    roc_auc_score, root_mean_squared_error, root_mean_squared_log_error,
    top_k_accuracy_score
)  # 导入各种评分函数
from .cluster import (
    adjusted_mutual_info_score, adjusted_rand_score, completeness_score,
    fowlkes_mallows_score, homogeneity_score, mutual_info_score,
    normalized_mutual_info_score, rand_score, v_measure_score
)  # 导入聚类相关评分函数


def _cached_call(cache, estimator, response_method, *args, **kwargs):
    """调用评估器的特定响应方法，并可选择使用缓存加速。

    Parameters:
    - cache: 可选的缓存字典，用于存储已计算的结果
    - estimator: 要评估的模型
    - response_method: 响应方法的名称
    - *args: 位置参数传递给评估器的方法
    - **kwargs: 关键字参数传递给评估器的方法

    Returns:
    - result: 调用评估器方法后得到的结果

    Notes:
    如果缓存不为 None 并且已经存储了 response_method 对应的结果，则直接从缓存中获取并返回，
    否则调用 _get_response_values 函数获取结果，并根据需要将结果存入缓存。
    """
    if cache is not None and response_method in cache:
        return cache[response_method]

    result, _ = _get_response_values(
        estimator, *args, response_method=response_method, **kwargs
    )

    if cache is not None:
        cache[response_method] = result

    return result


class _MultimetricScorer:
    """多指标评分的可调用对象，用于避免重复调用 `predict_proba`, `predict`, 和 `decision_function`。

    `_MultimetricScorer` 将返回一个字典，其中包含与字典中的评分函数相对应的分数。
    注意，`_MultimetricScorer` 可以被
    """
    created with a dictionary with one key  (i.e. only one actual scorer).

    Parameters
    ----------
    scorers : dict
        Dictionary mapping names to callable scorers.

    raise_exc : bool, default=True
        Whether to raise the exception in `__call__` or not. If set to `False`
        a formatted string of the exception details is passed as result of
        the failing scorer.
    """



# 初始化函数，接收评分器字典和是否抛出异常的标志位
def __init__(self, *, scorers, raise_exc=True):
    # 将评分器字典保存在实例变量中
    self._scorers = scorers
    # 是否在评分器调用中抛出异常的标志位
    self._raise_exc = raise_exc



    def __call__(self, estimator, *args, **kwargs):
        """Evaluate predicted target values."""
        # 初始化评分结果字典
        scores = {}
        # 如果缓存可用，则初始化缓存字典；否则设为None
        cache = {} if self._use_cache(estimator) else None
        # 创建一个局部函数，用于缓存评分器调用结果
        cached_call = partial(_cached_call, cache)

        # 如果路由功能启用，则处理路由参数
        if _routing_enabled():
            routed_params = process_routing(self, "score", **kwargs)
        else:
            # 否则，每个评分器都使用相同的参数args和kwargs
            routed_params = Bunch(
                **{name: Bunch(score=kwargs) for name in self._scorers}
            )

        # 遍历评分器字典中的每个评分器
        for name, scorer in self._scorers.items():
            try:
                # 如果评分器是_BaseScorer的实例
                if isinstance(scorer, _BaseScorer):
                    # 调用评分器的_score方法进行评分
                    score = scorer._score(
                        cached_call, estimator, *args, **routed_params.get(name).score
                    )
                else:
                    # 否则，直接调用评分器进行评分
                    score = scorer(estimator, *args, **routed_params.get(name).score)
                # 将评分结果存入scores字典
                scores[name] = score
            except Exception as e:
                # 如果设定为抛出异常，则抛出当前异常
                if self._raise_exc:
                    raise e
                # 否则，将异常信息格式化后作为评分结果存入scores字典
                else:
                    scores[name] = format_exc()
        # 返回所有评分结果的字典
        return scores



    def __repr__(self):
        # 返回当前对象的字符串表示，包含所有评分器的名称
        scorers = ", ".join([f'"{s}"' for s in self._scorers])
        return f"MultiMetricScorer({scorers})"



    def _use_cache(self, estimator):
        """Return True if using a cache is beneficial, thus when a response method will
        be called several time.
        """
        # 如果只有一个评分器，则不使用缓存
        if len(self._scorers) == 1:  # Only one scorer
            return False

        # 统计所有评分器中具有相同响应方法的个数
        counter = Counter(
            [
                _check_response_method(estimator, scorer._response_method).__name__
                for scorer in self._scorers.values()
                if isinstance(scorer, _BaseScorer)
            ]
        )
        # 如果有任何响应方法被调用超过一次，则使用缓存
        if any(val > 1 for val in counter.values()):
            return True

        # 否则不使用缓存
        return False
    def get_metadata_routing(self):
        """
        Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        .. versionadded:: 1.3

        Returns
        -------
        routing : MetadataRouter
            A :class:`~utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        # 创建一个 MetadataRouter 对象，使用当前对象的类名作为所有者
        return MetadataRouter(owner=self.__class__.__name__).add(
            # 向 MetadataRouter 添加一组评分器 (scorers)
            **self._scorers,
            # 设置方法映射，将 'score' 方法映射到 'score' 方法上
            method_mapping=MethodMapping().add(caller="score", callee="score"),
        )
class _BaseScorer(_MetadataRequester):
    """Base scorer that is used as `scorer(estimator, X, y_true)`.

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

    response_method : str
        The method to call on the estimator to get the response values.
    """

    def __init__(self, score_func, sign, kwargs, response_method="predict"):
        # 初始化方法，保存评分函数、符号、额外参数和响应方法
        self._score_func = score_func
        self._sign = sign
        self._kwargs = kwargs
        self._response_method = response_method

    def _get_pos_label(self):
        # 获取正标签参数，优先使用传入的参数，其次是评分函数默认参数
        if "pos_label" in self._kwargs:
            return self._kwargs["pos_label"]
        score_func_params = signature(self._score_func).parameters
        if "pos_label" in score_func_params:
            return score_func_params["pos_label"].default
        return None

    def __repr__(self):
        # 返回对象的字符串表示形式，包括评分函数名、符号、响应方法和其他参数
        sign_string = "" if self._sign > 0 else ", greater_is_better=False"
        response_method_string = f", response_method={self._response_method!r}"
        kwargs_string = "".join([f", {k}={v}" for k, v in self._kwargs.items()])
        return (
            f"make_scorer({self._score_func.__name__}{sign_string}"
            f"{response_method_string}{kwargs_string})"
        )

    def __call__(self, estimator, X, y_true, sample_weight=None, **kwargs):
        """Evaluate predicted target values for X relative to y_true.

        Parameters
        ----------
        estimator : object
            Trained estimator to use for scoring. Must have a predict_proba
            method; the output of that is used to compute the score.

        X : {array-like, sparse matrix}
            Test data that will be fed to estimator.predict.

        y_true : array-like
            Gold standard target values for X.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        **kwargs : dict
            Other parameters passed to the scorer. Refer to
            :func:`set_score_request` for more details.

            Only available if `enable_metadata_routing=True`. See the
            :ref:`User Guide <metadata_routing>`.

            .. versionadded:: 1.3

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """
        # 检查并触发参数异常
        _raise_for_params(kwargs, self, None)

        # 深拷贝参数，处理样本权重
        _kwargs = copy.deepcopy(kwargs)
        if sample_weight is not None:
            _kwargs["sample_weight"] = sample_weight

        # 调用评分方法并返回评分结果
        return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
    def _warn_overlap(self, message, kwargs):
        """Warn if there is any overlap between ``self._kwargs`` and ``kwargs``.

        This method is intended to be used to check for overlap between
        ``self._kwargs`` and ``kwargs`` passed as metadata.
        """
        # 将 self._kwargs 的键转换为集合，如果为 None 则创建空集合
        _kwargs = set() if self._kwargs is None else set(self._kwargs.keys())
        # 找到 self._kwargs 和 kwargs 的键集合的交集，即存在重叠的参数
        overlap = _kwargs.intersection(kwargs.keys())
        # 如果存在重叠的参数，则发出警告
        if overlap:
            warnings.warn(
                f"{message} Overlapping parameters are: {overlap}", UserWarning
            )

    def set_score_request(self, **kwargs):
        """Set requested parameters by the scorer.

        Please see :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        .. versionadded:: 1.3

        Parameters
        ----------
        kwargs : dict
            Arguments should be of the form ``param_name=alias``, and `alias`
            can be one of ``{True, False, None, str}``.
        """
        # 检查是否启用了元数据路由，如果未启用则抛出运行时错误
        if not _routing_enabled():
            raise RuntimeError(
                "This method is only available when metadata routing is enabled."
                " You can enable it using"
                " sklearn.set_config(enable_metadata_routing=True)."
            )

        # 调用 _warn_overlap 方法检查重叠参数
        self._warn_overlap(
            message=(
                "You are setting metadata request for parameters which are "
                "already set as kwargs for this metric. These set values will be "
                "overridden by passed metadata if provided. Please pass them either "
                "as metadata or kwargs to `make_scorer`."
            ),
            kwargs=kwargs,
        )
        # 创建 MetadataRequest 对象，并设置所有参数请求
        self._metadata_request = MetadataRequest(owner=self.__class__.__name__)
        for param, alias in kwargs.items():
            self._metadata_request.score.add_request(param=param, alias=alias)
        # 返回当前对象
        return self
class _Scorer(_BaseScorer):
    def _score(self, method_caller, estimator, X, y_true, **kwargs):
        """Evaluate the response method of `estimator` on `X` and `y_true`.

        Parameters
        ----------
        method_caller : callable
            Returns predictions given an estimator, method name, and other
            arguments, potentially caching results.

        estimator : object
            Trained estimator to use for scoring.

        X : {array-like, sparse matrix}
            Test data that will be fed to clf.decision_function or
            clf.predict_proba.

        y_true : array-like
            Gold standard target values for X. These must be class labels,
            not decision function values.

        **kwargs : dict
            Other parameters passed to the scorer. Refer to
            :func:`set_score_request` for more details.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """
        # 提示重叠问题，如果使用了相同的参数键名，建议将它们作为 `make_scorer` 的参数或元数据的一部分传递，但不要同时使用两者。
        self._warn_overlap(
            message=(
                "There is an overlap between set kwargs of this scorer instance and"
                " passed metadata. Please pass them either as kwargs to `make_scorer`"
                " or metadata, but not both."
            ),
            kwargs=kwargs,
        )

        # 如果是回归器，则不指定正类标签；否则获取正类标签
        pos_label = None if is_regressor(estimator) else self._get_pos_label()
        # 确定响应方法
        response_method = _check_response_method(estimator, self._response_method)
        # 调用方法获取预测值
        y_pred = method_caller(
            estimator, response_method.__name__, X, pos_label=pos_label
        )

        # 合并默认评分参数和传入参数
        scoring_kwargs = {**self._kwargs, **kwargs}
        # 计算评分函数应用于预测结果的分数
        return self._sign * self._score_func(y_true, y_pred, **scoring_kwargs)


@validate_params(
    {
        "scoring": [str, callable, None],
    },
    prefer_skip_nested_validation=True,
)
def get_scorer(scoring):
    """Get a scorer from string.

    Read more in the :ref:`User Guide <scoring_parameter>`.
    :func:`~sklearn.metrics.get_scorer_names` can be used to retrieve the names
    of all available scorers.

    Parameters
    ----------
    scoring : str, callable or None
        Scoring method as string. If callable it is returned as is.
        If None, returns None.

    Returns
    -------
    scorer : callable
        The scorer.

    Notes
    -----
    When passed a string, this function always returns a copy of the scorer
    object. Calling `get_scorer` twice for the same scorer results in two
    separate scorer objects.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.dummy import DummyClassifier
    >>> from sklearn.metrics import get_scorer
    >>> X = np.reshape([0, 1, -1, -0.5, 2], (-1, 1))
    >>> y = np.array([0, 1, 1, 0, 1])
    >>> classifier = DummyClassifier(strategy="constant", constant=0).fit(X, y)
    >>> accuracy = get_scorer("accuracy")
    >>> accuracy(classifier, X, y)
    0.4
    """
    # 从字符串中获取评分器，返回评分器对象的副本
    # 更多详情请参阅用户指南中的评分参数部分
    # 可以使用 `sklearn.metrics.get_scorer_names` 获取所有可用评分器的名称
    pass
    # 如果 `scoring` 是字符串类型，则进行以下操作
    if isinstance(scoring, str):
        # 尝试深拷贝 `_SCORERS` 字典中 `scoring` 键对应的值
        try:
            scorer = copy.deepcopy(_SCORERS[scoring])
        # 如果 `scoring` 不在 `_SCORERS` 字典中，则引发 ValueError 异常
        except KeyError:
            raise ValueError(
                "%r is not a valid scoring value. "
                "Use sklearn.metrics.get_scorer_names() "
                "to get valid options." % scoring
            )
    # 如果 `scoring` 不是字符串类型，则直接将其赋值给 `scorer`
    else:
        scorer = scoring
    # 返回 `scorer` 变量作为结果
    return scorer
class _PassthroughScorer(_MetadataRequester):
    # Passes scoring of estimator's `score` method back to estimator if scoring
    # is `None`.

    def __init__(self, estimator):
        # 初始化方法，接收一个 estimator 对象作为参数
        self._estimator = estimator

        # 创建一个 MetadataRequest 对象，并尝试复制 estimator 的 score 请求参数
        requests = MetadataRequest(owner=self.__class__.__name__)
        try:
            requests.score = copy.deepcopy(estimator._metadata_request.score)
        except AttributeError:
            # 如果 estimator 没有 _metadata_request.score 属性，则尝试获取默认的请求参数
            try:
                requests.score = copy.deepcopy(estimator._get_default_requests().score)
            except AttributeError:
                # 如果都失败，则不进行任何操作
                pass

        self._metadata_request = requests

    def __call__(self, estimator, *args, **kwargs):
        """Method that wraps estimator.score"""
        # 调用对象时执行的方法，调用 estimator 的 score 方法并传递参数
        return estimator.score(*args, **kwargs)

    def __repr__(self):
        # 返回对象的字符串表示形式，显示 estimator 的类名后跟 .score
        return f"{self._estimator.__class__}.score"

    def get_metadata_routing(self):
        """Get requested data properties.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        .. versionadded:: 1.3

        Returns
        -------
        routing : MetadataRouter
            A :class:`~utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        # 获取元数据路由信息的方法，返回一个 MetadataRouter 对象
        return get_routing_for_object(self._metadata_request)

    def set_score_request(self, **kwargs):
        """Set requested parameters by the scorer.

        Please see :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        .. versionadded:: 1.5

        Parameters
        ----------
        kwargs : dict
            Arguments should be of the form ``param_name=alias``, and `alias`
            can be one of ``{True, False, None, str}``.
        """
        # 设置评分器请求的参数的方法

        # 检查是否启用了元数据路由
        if not _routing_enabled():
            raise RuntimeError(
                "This method is only available when metadata routing is enabled."
                " You can enable it using"
                " sklearn.set_config(enable_metadata_routing=True)."
            )

        # 遍历参数 kwargs 中的键值对，将请求参数添加到 _metadata_request 的 score 属性中
        for param, alias in kwargs.items():
            self._metadata_request.score.add_request(param=param, alias=alias)
        return self


def _check_multimetric_scoring(estimator, scoring):
    """Check the scoring parameter in cases when multiple metrics are allowed.

    In addition, multimetric scoring leverages a caching mechanism to not call the same
    estimator response method multiple times. Hence, the scorer is modified to only use
    a single response method given a list of response methods and the estimator.

    Parameters
    ----------
    estimator : sklearn estimator instance
        The estimator for which the scoring will be applied.
    """
    # 检查评分参数在允许多个指标的情况下的处理方式的方法
    # 多指标评分利用缓存机制，避免多次调用相同的估计器响应方法。因此，根据响应方法列表和估计器，修改评分器以仅使用单个响应方法。
    # 参数 estimator: sklearn 估计器实例，将应用评分的估计器。
    # 定义函数返回的评分字典，用于评估交叉验证模型在测试集上的性能
    scoring : list, tuple or dict
        # 评分策略，用于评估交叉验证模型在测试集上的表现。

        可能的取值有：

        - 独特字符串组成的列表或元组；
        - 返回字典的可调用对象，其中键是指标名称，值是指标得分；
        - 以指标名称为键，以可调用对象为值的字典。

        参见 :ref:`multimetric_grid_search` 以查看示例。

    Returns
    -------
    scorers_dict : dict
        # 映射每个评分器名称到其验证过的评分器的字典。
    """
    # 定义通用错误消息，用于处理无效的评分策略
    err_msg_generic = (
        f"scoring is invalid (got {scoring!r}). Refer to the "
        "scoring glossary for details: "
        "https://scikit-learn.org/stable/glossary.html#term-scoring"
    )

    # 根据评分策略的类型进行分支处理
    if isinstance(scoring, (list, tuple, set)):
        # 如果评分策略是列表、元组或集合
        err_msg = (
            "The list/tuple elements must be unique strings of predefined scorers. "
        )
        try:
            keys = set(scoring)
        except TypeError as e:
            raise ValueError(err_msg) from e

        # 检查列表元素是否唯一
        if len(keys) != len(scoring):
            raise ValueError(
                f"{err_msg} Duplicate elements were found in"
                f" the given list. {scoring!r}"
            )
        elif len(keys) > 0:
            # 如果列表中有元素
            if not all(isinstance(k, str) for k in keys):
                if any(callable(k) for k in keys):
                    # 如果列表中有可调用对象
                    raise ValueError(
                        f"{err_msg} One or more of the elements "
                        "were callables. Use a dict of score "
                        "name mapped to the scorer callable. "
                        f"Got {scoring!r}"
                    )
                else:
                    # 如果列表中有非字符串类型的元素
                    raise ValueError(
                        f"{err_msg} Non-string types were found "
                        f"in the given list. Got {scoring!r}"
                    )
            # 为每个评分器调用 check_scoring 函数
            scorers = {
                scorer: check_scoring(estimator, scoring=scorer) for scorer in scoring
            }
        else:
            # 如果列表为空
            raise ValueError(f"{err_msg} Empty list was given. {scoring!r}")

    elif isinstance(scoring, dict):
        # 如果评分策略是字典
        keys = set(scoring)
        if not all(isinstance(k, str) for k in keys):
            # 检查字典键是否全部为字符串类型
            raise ValueError(
                "Non-string types were found in the keys of "
                f"the given dict. scoring={scoring!r}"
            )
        if len(keys) == 0:
            # 检查字典是否为空
            raise ValueError(f"An empty dict was passed. {scoring!r}")
        # 为每个评分器调用 check_scoring 函数
        scorers = {
            key: check_scoring(estimator, scoring=scorer)
            for key, scorer in scoring.items()
        }
    else:
        # 如果评分策略类型无法识别，则抛出通用错误消息
        raise ValueError(err_msg_generic)

    # 返回处理后的评分器字典
    return scorers
# 定义一个函数，处理 `response_method` 参数，同时处理 `needs_threshold` 和 `needs_proba` 的弃用情况
def _get_response_method(response_method, needs_threshold, needs_proba):
    # 检查是否提供了 `needs_threshold` 参数，如果提供了，则不为 "deprecated"
    needs_threshold_provided = needs_threshold != "deprecated"
    # 检查是否提供了 `needs_proba` 参数，如果提供了，则不为 "deprecated"
    needs_proba_provided = needs_proba != "deprecated"
    # 检查是否提供了 `response_method` 参数
    response_method_provided = response_method is not None

    # 将 `needs_threshold` 参数设置为 False，如果它被标记为 "deprecated"
    needs_threshold = False if needs_threshold == "deprecated" else needs_threshold
    # 将 `needs_proba` 参数设置为 False，如果它被标记为 "deprecated"
    needs_proba = False if needs_proba == "deprecated" else needs_proba

    # 如果提供了 `response_method` 参数，并且同时提供了 `needs_proba` 或 `needs_threshold` 参数，则抛出错误
    if response_method_provided and (needs_proba_provided or needs_threshold_provided):
        raise ValueError(
            "You cannot set both `response_method` and `needs_proba` or "
            "`needs_threshold` at the same time. Only use `response_method` since "
            "the other two are deprecated in version 1.4 and will be removed in 1.6."
        )

    # 如果提供了 `needs_proba` 或 `needs_threshold` 参数，则发出警告，指示它们已弃用
    if needs_proba_provided or needs_threshold_provided:
        warnings.warn(
            (
                "The `needs_threshold` and `needs_proba` parameter are deprecated in "
                "version 1.4 and will be removed in 1.6. You can either let "
                "`response_method` be `None` or set it to `predict` to preserve the "
                "same behaviour."
            ),
            FutureWarning,
        )

    # 如果提供了 `response_method` 参数，则直接返回该参数
    if response_method_provided:
        return response_method

    # 如果需要概率 (`needs_proba` 为 True)，则设置 `response_method` 为 "predict_proba"
    if needs_proba is True:
        response_method = "predict_proba"
    # 如果需要阈值 (`needs_threshold` 为 True)，则设置 `response_method` 为 ("decision_function", "predict_proba")
    elif needs_threshold is True:
        response_method = ("decision_function", "predict_proba")
    # 否则，默认设置 `response_method` 为 "predict"
    else:
        response_method = "predict"

    # 返回最终确定的 `response_method`
    return response_method
    The parameter `response_method` allows to specify which method of the estimator
    should be used to feed the scoring/loss function.

    Read more in the :ref:`User Guide <scoring>`.

    Parameters
    ----------
    score_func : callable
        Score function (or loss function) with signature
        ``score_func(y, y_pred, **kwargs)``.

    response_method : {"predict_proba", "decision_function", "predict"} or \
            list/tuple of such str, default=None
        指定从估计器获取预测的方法（即 :term:`predict_proba`, :term:`decision_function` 或 :term:`predict`）。
        可能的选择有：

        - 如果是 `str`，则对应于要返回的方法的名称；
        - 如果是 `str` 的列表或元组，则按照优先顺序提供方法名称。返回的方法对应于列表中第一个由 `estimator` 实现的方法。
        - 如果是 `None`，等效于 `"predict"`。

        .. versionadded:: 1.4

    greater_is_better : bool, default=True
        Whether `score_func` is a score function (default), meaning high is
        good, or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the `score_func`.

    needs_proba : bool, default=False
        Whether `score_func` requires `predict_proba` to get probability
        estimates out of a classifier.

        If True, for binary `y_true`, the score function is supposed to accept
        a 1D `y_pred` (i.e., probability of the positive class, shape
        `(n_samples,)`).

        .. deprecated:: 1.4
           `needs_proba` is deprecated in version 1.4 and will be removed in
           1.6. Use `response_method="predict_proba"` instead.

    needs_threshold : bool, default=False
        Whether `score_func` takes a continuous decision certainty.
        This only works for binary classification using estimators that
        have either a `decision_function` or `predict_proba` method.

        If True, for binary `y_true`, the score function is supposed to accept
        a 1D `y_pred` (i.e., probability of the positive class or the decision
        function, shape `(n_samples,)`).

        For example `average_precision` or the area under the roc curve
        can not be computed using discrete predictions alone.

        .. deprecated:: 1.4
           `needs_threshold` is deprecated in version 1.4 and will be removed
           in 1.6. Use `response_method=("decision_function", "predict_proba")`
           instead to preserve the same behaviour.

    **kwargs : additional arguments
        Additional parameters to be passed to `score_func`.

    Returns
    -------
    scorer : callable
        Callable object that returns a scalar score; greater is better.

    Examples
    --------
    >>> from sklearn.metrics import fbeta_score, make_scorer
    >>> ftwo_scorer = make_scorer(fbeta_score, beta=2)
    # 调用 make_scorer 函数创建一个评分器，用于计算 F-beta 分数，其中响应方法为 'predict'，beta 值为 2
    ftwo_scorer = make_scorer(fbeta_score, response_method='predict', beta=2)
    
    # 导入 GridSearchCV 类，用于进行网格搜索交叉验证
    from sklearn.model_selection import GridSearchCV
    
    # 导入 LinearSVC 类，支持向量机的线性分类器
    from sklearn.svm import LinearSVC
    
    # 创建 GridSearchCV 对象，使用 LinearSVC() 作为基础分类器，参数网格为 {'C': [1, 10]}，评分器为 ftwo_scorer
    grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]},
                        scoring=ftwo_scorer)
# 创建解释方差得分的评分器
explained_variance_scorer = make_scorer(explained_variance_score)

# 创建R^2得分的评分器
r2_scorer = make_scorer(r2_score)

# 创建最大误差得分的评分器，需要最小化
max_error_scorer = make_scorer(max_error, greater_is_better=False)

# 创建负均方误差得分的评分器，需要最小化
neg_mean_squared_error_scorer = make_scorer(mean_squared_error, greater_is_better=False)

# 创建负均方对数误差得分的评分器，需要最小化
neg_mean_squared_log_error_scorer = make_scorer(mean_squared_log_error, greater_is_better=False)

# 创建负平均绝对误差得分的评分器，需要最小化
neg_mean_absolute_error_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

# 创建负平均绝对百分比误差得分的评分器，需要最小化
neg_mean_absolute_percentage_error_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)

# 创建负中位数绝对误差得分的评分器，需要最小化
neg_median_absolute_error_scorer = make_scorer(median_absolute_error, greater_is_better=False)

# 创建负平方根均方误差得分的评分器，需要最小化
neg_root_mean_squared_error_scorer = make_scorer(root_mean_squared_error, greater_is_better=False)

# 创建负平方根均方对数误差得分的评分器，需要最小化
neg_root_mean_squared_log_error_scorer = make_scorer(root_mean_squared_log_error, greater_is_better=False)

# 创建负泊松偏差得分的评分器，需要最小化
neg_mean_poisson_deviance_scorer = make_scorer(mean_poisson_deviance, greater_is_better=False)

# 创建负伽马偏差得分的评分器，需要最小化
neg_mean_gamma_deviance_scorer = make_scorer(mean_gamma_deviance, greater_is_better=False)

# 创建D2绝对误差得分的评分器
d2_absolute_error_scorer = make_scorer(d2_absolute_error_score)

# 创建准确率得分的评分器
accuracy_scorer = make_scorer(accuracy_score)

# 创建平衡准确率得分的评分器
balanced_accuracy_scorer = make_scorer(balanced_accuracy_score)

# 创建马修斯相关系数得分的评分器
matthews_corrcoef_scorer = make_scorer(matthews_corrcoef)

# 创建正类似然比得分的评分器
positive_likelihood_ratio_scorer = make_scorer(positive_likelihood_ratio)

# 创建负类似然比得分的评分器，需要最小化
neg_negative_likelihood_ratio_scorer = make_scorer(negative_likelihood_ratio, greater_is_better=False)

# 创建Top-k准确率得分的评分器，根据决策值预测，越大越好
top_k_accuracy_scorer = make_scorer(top_k_accuracy_score, greater_is_better=True, response_method=("decision_function", "predict_proba"))

# 创建ROC曲线下面积得分的评分器，根据决策值预测，越大越好
roc_auc_scorer = make_scorer(roc_auc_score, greater_is_better=True, response_method=("decision_function", "predict_proba"))

# 创建平均精度得分的评分器，根据决策值预测
average_precision_scorer = make_scorer(average_precision_score, response_method=("decision_function", "predict_proba"))

# 创建多类别ROC曲线下面积得分的评分器，使用OvO策略
roc_auc_ovo_scorer = make_scorer(roc_auc_score, response_method="predict_proba", multi_class="ovo")

# 创建加权多类别ROC曲线下面积得分的评分器，使用OvO策略
roc_auc_ovo_weighted_scorer = make_scorer(roc_auc_score, response_method="predict_proba", multi_class="ovo", average="weighted")

# 创建多类别ROC曲线下面积得分的评分器，使用OvR策略
roc_auc_ovr_scorer = make_scorer(roc_auc_score, response_method="predict_proba", multi_class="ovr")

# 创建加权多类别ROC曲线下面积得分的评分器，使用OvR策略
roc_auc_ovr_weighted_scorer = make_scorer(roc_auc_score, response_method="predict_proba", multi_class="ovr", average="weighted")

# 创建负对数损失得分的评分器，需要最小化，使用概率预测
neg_log_loss_scorer = make_scorer(log_loss, greater_is_better=False, response_method="predict_proba")

# 创建负布里尔得分的评分器，需要最小化
neg_brier_score_scorer = make_scorer(
    # 定义变量 brier_score_loss，并设置其默认值为 False
    brier_score_loss, greater_is_better=False, response_method="predict_proba"
)
# 创建一个基于 brier_score_loss 函数的评分器，设置参数 greater_is_better=False 和 response_method="predict_proba"
brier_score_loss_scorer = make_scorer(
    brier_score_loss, greater_is_better=False, response_method="predict_proba"
)

# Clustering scores

# 创建 adjusted_rand_score 的评分器
adjusted_rand_scorer = make_scorer(adjusted_rand_score)
# 创建 rand_score 的评分器
rand_scorer = make_scorer(rand_score)
# 创建 homogeneity_score 的评分器
homogeneity_scorer = make_scorer(homogeneity_score)
# 创建 completeness_score 的评分器
completeness_scorer = make_scorer(completeness_score)
# 创建 v_measure_score 的评分器
v_measure_scorer = make_scorer(v_measure_score)
# 创建 mutual_info_score 的评分器
mutual_info_scorer = make_scorer(mutual_info_score)
# 创建 adjusted_mutual_info_score 的评分器
adjusted_mutual_info_scorer = make_scorer(adjusted_mutual_info_score)
# 创建 normalized_mutual_info_score 的评分器
normalized_mutual_info_scorer = make_scorer(normalized_mutual_info_score)
# 创建 fowlkes_mallows_score 的评分器
fowlkes_mallows_scorer = make_scorer(fowlkes_mallows_score)

# 将所有评分器以名称和对应的评分函数映射成字典
_SCORERS = dict(
    explained_variance=explained_variance_scorer,
    r2=r2_scorer,
    max_error=max_error_scorer,
    matthews_corrcoef=matthews_corrcoef_scorer,
    neg_median_absolute_error=neg_median_absolute_error_scorer,
    neg_mean_absolute_error=neg_mean_absolute_error_scorer,
    neg_mean_absolute_percentage_error=neg_mean_absolute_percentage_error_scorer,
    neg_mean_squared_error=neg_mean_squared_error_scorer,
    neg_mean_squared_log_error=neg_mean_squared_log_error_scorer,
    neg_root_mean_squared_error=neg_root_mean_squared_error_scorer,
    neg_root_mean_squared_log_error=neg_root_mean_squared_log_error_scorer,
    neg_mean_poisson_deviance=neg_mean_poisson_deviance_scorer,
    neg_mean_gamma_deviance=neg_mean_gamma_deviance_scorer,
    d2_absolute_error_score=d2_absolute_error_scorer,
    accuracy=accuracy_scorer,
    top_k_accuracy=top_k_accuracy_scorer,
    roc_auc=roc_auc_scorer,
    roc_auc_ovr=roc_auc_ovr_scorer,
    roc_auc_ovo=roc_auc_ovo_scorer,
    roc_auc_ovr_weighted=roc_auc_ovr_weighted_scorer,
    roc_auc_ovo_weighted=roc_auc_ovo_weighted_scorer,
    balanced_accuracy=balanced_accuracy_scorer,
    average_precision=average_precision_scorer,
    neg_log_loss=neg_log_loss_scorer,
    neg_brier_score=neg_brier_score_scorer,
    positive_likelihood_ratio=positive_likelihood_ratio_scorer,
    neg_negative_likelihood_ratio=neg_negative_likelihood_ratio_scorer,
    # 包含使用监督评估的聚类度量
    adjusted_rand_score=adjusted_rand_scorer,
    rand_score=rand_scorer,
    homogeneity_score=homogeneity_scorer,
    completeness_score=completeness_scorer,
    v_measure_score=v_measure_scorer,
    mutual_info_score=mutual_info_scorer,
    adjusted_mutual_info_score=adjusted_mutual_info_scorer,
    normalized_mutual_info_score=normalized_mutual_info_scorer,
    fowlkes_mallows_score=fowlkes_mallows_scorer,
)

def get_scorer_names():
    """获取所有可用评分器的名称。

    这些名称可以传递给 :func:`~sklearn.metrics.get_scorer` 函数以获取评分器对象。

    Returns
    -------
    list of str
        所有可用评分器的名称列表。
    """
    # 导入 get_scorer_names 函数后，可以获取所有评分器的名称列表
    from sklearn.metrics import get_scorer_names
    all_scorers = get_scorer_names()
    # 返回评分器名称列表
    return all_scorers
    # 返回一个排序后的所有评分函数名称列表
    def all_estimators():
        # 创建一个包含评分函数名称的列表
        ['accuracy', 'adjusted_mutual_info_score', 'adjusted_rand_score']
        # 检查字符串 "roc_auc" 是否在所有评分函数名称列表中
        >>> "roc_auc" in all_scorers
        # 返回 True，表示 "roc_auc" 存在于所有评分函数名称列表中
        True
        """
        # 返回排序后的评分函数名称列表，基于全局变量 _SCORERS 的键
        return sorted(_SCORERS.keys())
# 遍历给定的评估指标及其对应的评分函数，并生成相应的评分器
for name, metric in [
    ("precision", precision_score),
    ("recall", recall_score),
    ("f1", f1_score),
    ("jaccard", jaccard_score),
]:
    # 将评估指标名称与评分函数关联，创建一个二元评分器并存储到全局变量 `_SCORERS` 中
    _SCORERS[name] = make_scorer(metric, average="binary")
    
    # 针对每个评估指标，根据不同的平均类型（macro、micro、samples、weighted），生成对应的合格名称并存储到 `_SCORERS` 中
    for average in ["macro", "micro", "samples", "weighted"]:
        qualified_name = "{0}_{1}".format(name, average)
        _SCORERS[qualified_name] = make_scorer(metric, pos_label=None, average=average)


@validate_params(
    {
        "estimator": [HasMethods("fit"), None],
        "scoring": [
            StrOptions(set(get_scorer_names())),
            callable,
            list,
            set,
            tuple,
            dict,
            None,
        ],
        "allow_none": ["boolean"],
        "raise_exc": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
# 校验评分参数的有效性，并确定合适的评分器
def check_scoring(estimator=None, scoring=None, *, allow_none=False, raise_exc=True):
    """Determine scorer from user options.

    A TypeError will be thrown if the estimator cannot be scored.

    Parameters
    ----------
    estimator : estimator object implementing 'fit' or None, default=None
        The object to use to fit the data. If `None`, then this function may error
        depending on `allow_none`.

    scoring : str, callable, list, tuple, set, or dict, default=None
        Scorer to use. If `scoring` represents a single score, one can use:

        - a single string (see :ref:`scoring_parameter`);
        - a callable (see :ref:`scoring`) that returns a single value.

        If `scoring` represents multiple scores, one can use:

        - a list, tuple or set of unique strings;
        - a callable returning a dictionary where the keys are the metric names and the
          values are the metric scorers;
        - a dictionary with metric names as keys and callables a values. The callables
          need to have the signature `callable(estimator, X, y)`.

        If None, the provided estimator object's `score` method is used.

    allow_none : bool, default=False
        Whether to return None or raise an error if no `scoring` is specified and the
        estimator has no `score` method.

    raise_exc : bool, default=True
        Whether to raise an exception (if a subset of the scorers in multimetric scoring
        fails) or to return an error code.

        - If set to `True`, raises the failing scorer's exception.
        - If set to `False`, a formatted string of the exception details is passed as
          result of the failing scorer(s).

        This applies if `scoring` is list, tuple, set, or dict. Ignored if `scoring` is
        a str or a callable.

        .. versionadded:: 1.6

    Returns
    -------
    scoring : callable
        A scorer callable object / function with signature ``scorer(estimator, X, y)``.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.metrics import check_scoring
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> X, y = load_iris(return_X_y=True)
    # 创建一个最大深度为2的决策树分类器，并使用输入数据集 X 和标签 y 进行拟合
    >>> classifier = DecisionTreeClassifier(max_depth=2).fit(X, y)
    
    # 根据给定的分类器和指定的评分方法（这里是准确率），获取评分函数
    >>> scorer = check_scoring(classifier, scoring='accuracy')
    
    # 使用评分函数对分类器在数据集 X 和标签 y 上进行评分，返回准确率得分
    >>> scorer(classifier, X, y)
    0.96...

    # 从 sklearn.metrics 中导入所需的评分函数和数据集加载函数
    >>> from sklearn.metrics import make_scorer, accuracy_score, mean_squared_log_error
    >>> X, y = load_iris(return_X_y=True)
    
    # 将标签 y 中的所有元素取反
    >>> y *= -1
    
    # 创建一个默认参数的决策树分类器，并使用加载的 iris 数据集 X 和处理后的标签 y 进行拟合
    >>> clf = DecisionTreeClassifier().fit(X, y)
    
    # 定义多个评分方法的字典
    >>> scoring = {
    ...     "accuracy": make_scorer(accuracy_score),
    ...     "mean_squared_log_error": make_scorer(mean_squared_log_error),
    ... }
    
    # 检查评分方法的调用，并指定是否要抛出异常
    >>> scoring_call = check_scoring(estimator=clf, scoring=scoring, raise_exc=False)
    
    # 使用 scoring_call 对分类器 clf 在数据集 X 和标签 y 上进行评分，返回包含多个评分结果的字典
    >>> scores = scoring_call(clf, X, y)
    
    # 输出评分结果字典
    >>> scores
    {'accuracy': 1.0, 'mean_squared_log_error': 'Traceback ...'}
    """
    # 根据 scoring 的类型进行不同的处理分支，返回相应的评分函数或评分器对象
    if isinstance(scoring, str):
        return get_scorer(scoring)
    if callable(scoring):
        # 启发式检查，确保用户未传递一个指标函数而非评分器
        module = getattr(scoring, "__module__", None)
        if (
            hasattr(module, "startswith")
            and module.startswith("sklearn.metrics.")
            and not module.startswith("sklearn.metrics._scorer")
            and not module.startswith("sklearn.metrics.tests.")
        ):
            raise ValueError(
                "scoring value %r looks like it is a metric "
                "function rather than a scorer. A scorer should "
                "require an estimator as its first parameter. "
                "Please use `make_scorer` to convert a metric "
                "to a scorer." % scoring
            )
        return get_scorer(scoring)
    if isinstance(scoring, (list, tuple, set, dict)):
        # 检查并处理多评分指标情况
        scorers = _check_multimetric_scoring(estimator, scoring=scoring)
        return _MultimetricScorer(scorers=scorers, raise_exc=raise_exc)
    if scoring is None:
        if hasattr(estimator, "score"):
            # 如果没有指定评分方法，但估计器具有 score 方法，则返回 _PassthroughScorer 类的实例
            return _PassthroughScorer(estimator)
        elif allow_none:
            # 如果允许 scoring 为 None，则返回 None
            return None
        else:
            # 否则，抛出类型错误，要求估计器具有 score 方法
            raise TypeError(
                "If no scoring is specified, the estimator passed should "
                "have a 'score' method. The estimator %r does not." % estimator
            )
```