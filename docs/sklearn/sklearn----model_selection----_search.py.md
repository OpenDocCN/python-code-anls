# `D:\src\scipysrc\scikit-learn\sklearn\model_selection\_search.py`

```
"""
The :class:`ParameterGrid` class generates a grid of parameters with discrete values for each parameter.

It allows iterating over combinations of parameter values and is commonly used in grid search procedures.

Read more in the :ref:`User Guide <grid_search>`.

Attributes
----------
param_grid : dict of str to sequence, or sequence of such
    The parameter grid to explore, as a dictionary mapping estimator
    parameters to sequences of allowed values.

    An empty dict signifies default parameters.

Methods
-------
__init__(self, param_grid)
    Initializes the ParameterGrid object with the provided parameter grid.

Examples
--------
>>> from sklearn.model_selection import ParameterGrid
>>> param_grid = {'a': [1, 2], 'b': [True, False]}
>>> list(ParameterGrid(param_grid)) == (
...    [{'a': 1, 'b': True}, {'a': 1, 'b': False},
...     {'a': 2, 'b': True}, {'a': 2, 'b': False}])
True

>>> grid = [{'kernel': ['linear']}, {'kernel': ['rbf'], 'gamma': [1, 10]}]
"""

class ParameterGrid:
    """Grid of parameters with a discrete number of values for each.

    Can be used to iterate over parameter value combinations with the
    Python built-in function iter.
    The order of the generated parameter combinations is deterministic.

    Read more in the :ref:`User Guide <grid_search>`.

    Parameters
    ----------
    param_grid : dict of str to sequence, or sequence of such
        The parameter grid to explore, as a dictionary mapping estimator
        parameters to sequences of allowed values.

        An empty dict signifies default parameters.

        A sequence of dicts signifies a sequence of grids to search, and is
        useful to avoid exploring parameter combinations that make no sense
        or have no effect. See the examples below.

    Examples
    --------
    >>> from sklearn.model_selection import ParameterGrid
    >>> param_grid = {'a': [1, 2], 'b': [True, False]}
    >>> list(ParameterGrid(param_grid)) == (
    ...    [{'a': 1, 'b': True}, {'a': 1, 'b': False},
    ...     {'a': 2, 'b': True}, {'a': 2, 'b': False}])
    True

    >>> grid = [{'kernel': ['linear']}, {'kernel': ['rbf'], 'gamma': [1, 10]}]
    """

    def __init__(self, param_grid):
        """
        Initialize ParameterGrid with the provided parameter grid.

        Parameters
        ----------
        param_grid : dict of str to sequence, or sequence of such
            The parameter grid to explore, as a dictionary mapping estimator
            parameters to sequences of allowed values.
        """
        pass
    >>> list(ParameterGrid(grid)) == [{'kernel': 'linear'},
    ...                               {'kernel': 'rbf', 'gamma': 1},
    ...                               {'kernel': 'rbf', 'gamma': 10}]
    True
    >>> ParameterGrid(grid)[1] == {'kernel': 'rbf', 'gamma': 1}
    True

    See Also
    --------
    GridSearchCV : Uses :class:`ParameterGrid` to perform a full parallelized
        parameter search.
    """

    # 定义参数网格类 ParameterGrid
    def __init__(self, param_grid):
        # 检查 param_grid 是否为字典或可迭代对象
        if not isinstance(param_grid, (Mapping, Iterable)):
            raise TypeError(
                f"Parameter grid should be a dict or a list, got: {param_grid!r} of"
                f" type {type(param_grid).__name__}"
            )

        if isinstance(param_grid, Mapping):
            # 如果 param_grid 是字典，则将其包装在一个列表中以支持 dict 或 list of dicts
            param_grid = [param_grid]

        # 检查所有条目是否为字典的列表
        for grid in param_grid:
            if not isinstance(grid, dict):
                raise TypeError(f"Parameter grid is not a dict ({grid!r})")
            for key, value in grid.items():
                if isinstance(value, np.ndarray) and value.ndim > 1:
                    raise ValueError(
                        f"Parameter array for {key!r} should be one-dimensional, got:"
                        f" {value!r} with shape {value.shape}"
                    )
                if isinstance(value, str) or not isinstance(
                    value, (np.ndarray, Sequence)
                ):
                    raise TypeError(
                        f"Parameter grid for parameter {key!r} needs to be a list or a"
                        f" numpy array, but got {value!r} (of type "
                        f"{type(value).__name__}) instead. Single values "
                        "need to be wrapped in a list with one element."
                    )
                if len(value) == 0:
                    raise ValueError(
                        f"Parameter grid for parameter {key!r} need "
                        f"to be a non-empty sequence, got: {value!r}"
                    )

        # 将参数网格保存在实例变量中
        self.param_grid = param_grid

    def __iter__(self):
        """Iterate over the points in the grid.

        Returns
        -------
        params : iterator over dict of str to any
            Yields dictionaries mapping each estimator parameter to one of its
            allowed values.
        """
        # 遍历参数网格中的每一个点
        for p in self.param_grid:
            # 总是对字典的键进行排序，以确保可复现性
            items = sorted(p.items())
            if not items:
                yield {}
            else:
                keys, values = zip(*items)
                # 对每个参数的取值进行笛卡尔积操作
                for v in product(*values):
                    params = dict(zip(keys, v))
                    yield params
    def __len__(self):
        """Number of points on the grid."""
        # 定义一个能处理可迭代对象的乘积函数（np.prod 不能处理）
        product = partial(reduce, operator.mul)
        # 返回所有子网格中参数数目的总和
        return sum(
            product(len(v) for v in p.values()) if p else 1 for p in self.param_grid
        )

    def __getitem__(self, ind):
        """Get the parameters that would be ``ind``th in iteration

        Parameters
        ----------
        ind : int
            迭代的索引

        Returns
        -------
        params : dict of str to any
            等同于 list(self)[ind] 的参数
        """
        # 用于在没有替换的情况下进行高效离散抽样
        for sub_grid in self.param_grid:
            # XXX: 可以缓存此处使用的信息
            if not sub_grid:
                if ind == 0:
                    return {}
                else:
                    ind -= 1
                    continue

            # 反转子网格，使得最频繁循环的参数首先出现
            keys, values_lists = zip(*sorted(sub_grid.items())[::-1])
            sizes = [len(v_list) for v_list in values_lists]
            total = np.prod(sizes)

            if ind >= total:
                # 尝试下一个网格
                ind -= total
            else:
                out = {}
                for key, v_list, n in zip(keys, values_lists, sizes):
                    ind, offset = divmod(ind, n)
                    out[key] = v_list[offset]
                return out

        # 如果索引超出范围则引发异常
        raise IndexError("ParameterGrid index out of range")
class ParameterSampler:
    """Generator on parameters sampled from given distributions.

    Non-deterministic iterable over random candidate combinations for hyper-
    parameter search. If all parameters are presented as a list,
    sampling without replacement is performed. If at least one parameter
    is given as a distribution, sampling with replacement is used.
    It is highly recommended to use continuous distributions for continuous
    parameters.

    Read more in the :ref:`User Guide <grid_search>`.

    Parameters
    ----------
    param_distributions : dict
        Dictionary with parameters names (`str`) as keys and distributions
        or lists of parameters to try. Distributions must provide a ``rvs``
        method for sampling (such as those from scipy.stats.distributions).
        If a list is given, it is sampled uniformly.
        If a list of dicts is given, first a dict is sampled uniformly, and
        then a parameter is sampled using that dict as above.

    n_iter : int
        Number of parameter settings that are produced.

    random_state : int, RandomState instance or None, default=None
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
        Pass an int for reproducible output across multiple
        function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    params : dict of str to any
        **Yields** dictionaries mapping each estimator parameter to
        as sampled value.
    """

    # Initialize the ParameterSampler class with the given parameters
    def __init__(self, param_distributions, n_iter, random_state=None):
        # Store the dictionary of parameter distributions for sampling
        self.param_distributions = param_distributions
        # Store the number of iterations for parameter sampling
        self.n_iter = n_iter
        # Store the random state for reproducible random sampling
        self.random_state = random_state

    # Implementing the iterable behavior using __iter__ method
    def __iter__(self):
        # Initialize the random state for sampling
        rng = check_random_state(self.random_state)
        
        # Loop over the specified number of iterations
        for i in range(self.n_iter):
            # Sample parameters according to the distributions
            # Create a dictionary to store sampled parameters
            params = {}
            for name, distribution in self.param_distributions.items():
                # If the distribution is a list, sample uniformly
                if isinstance(distribution, (list, np.ndarray)):
                    params[name] = rng.choice(distribution)
                else:
                    # Sample from the distribution using its rvs method
                    params[name] = distribution.rvs(random_state=rng)

            # Yield the sampled parameter set
            yield params
    def __init__(self, param_distributions, n_iter, *, random_state=None):
        # 检查 param_distributions 是否为 Mapping 或 Iterable 类型
        if not isinstance(param_distributions, (Mapping, Iterable)):
            raise TypeError(
                "Parameter distribution is not a dict or a list,"
                f" got: {param_distributions!r} of type "
                f"{type(param_distributions).__name__}"
            )

        if isinstance(param_distributions, Mapping):
            # 如果是 Mapping 类型，则将其包装在一个单元素列表中，以支持 dict 或 dict 列表的输入
            param_distributions = [param_distributions]

        for dist in param_distributions:
            if not isinstance(dist, dict):
                # 检查每个分布是否为 dict 类型
                raise TypeError(
                    "Parameter distribution is not a dict ({!r})".format(dist)
                )
            for key in dist:
                if not isinstance(dist[key], Iterable) and not hasattr(
                    dist[key], "rvs"
                ):
                    # 检查每个参数值是否可迭代或者具有 rvs 方法（用于随机变量的生成）
                    raise TypeError(
                        f"Parameter grid for parameter {key!r} is not iterable "
                        f"or a distribution (value={dist[key]})"
                    )
        # 初始化对象属性
        self.n_iter = n_iter
        self.random_state = random_state
        self.param_distributions = param_distributions

    def _is_all_lists(self):
        # 检查 param_distributions 中的所有 dict 是否都不包含具有 rvs 方法的值
        return all(
            all(not hasattr(v, "rvs") for v in dist.values())
            for dist in self.param_distributions
        )

    def __iter__(self):
        # 检查随机数生成器是否为空，如果为空则使用默认随机数生成器
        rng = check_random_state(self.random_state)

        # 如果所有的分布都是列表形式，采样时不进行替换
        if self._is_all_lists():
            # 根据 param_distributions 创建参数网格
            param_grid = ParameterGrid(self.param_distributions)
            grid_size = len(param_grid)
            n_iter = self.n_iter

            if grid_size < n_iter:
                # 如果参数网格大小小于 n_iter，则发出警告并调整 n_iter
                warnings.warn(
                    "The total space of parameters %d is smaller "
                    "than n_iter=%d. Running %d iterations. For exhaustive "
                    "searches, use GridSearchCV." % (grid_size, self.n_iter, grid_size),
                    UserWarning,
                )
                n_iter = grid_size
            # 对参数网格进行不重复抽样
            for i in sample_without_replacement(grid_size, n_iter, random_state=rng):
                yield param_grid[i]

        else:
            # 如果分布不全为列表，则按照 n_iter 迭代次数进行参数抽样
            for _ in range(self.n_iter):
                dist = rng.choice(self.param_distributions)
                # 对字典的键进行排序，确保结果的可重复性
                items = sorted(dist.items())
                params = dict()
                for k, v in items:
                    if hasattr(v, "rvs"):
                        # 如果 v 具有 rvs 方法，则使用该方法生成随机变量
                        params[k] = v.rvs(random_state=rng)
                    else:
                        # 否则从 v 中随机选择一个值作为参数
                        params[k] = v[rng.randint(len(v))]
                yield params
    def __len__(self):
        """返回将被采样的点的数量。"""
        # 如果所有参数都是列表，计算参数网格的大小
        if self._is_all_lists():
            grid_size = len(ParameterGrid(self.param_distributions))
            # 返回采样次数和网格大小的较小值
            return min(self.n_iter, grid_size)
        else:
            # 如果参数不全是列表，则直接返回指定的迭代次数
            return self.n_iter
# 检查是否需要重新拟合（refit），如果不需要，则抛出 AttributeError
def _check_refit(search_cv, attr):
    if not search_cv.refit:
        raise AttributeError(
            f"This {type(search_cv).__name__} instance was initialized with "
            f"`refit=False`. {attr} is available only after refitting on the best "
            "parameters. You can refit an estimator manually using the "
            "`best_params_` attribute"
        )


def _estimator_has(attr):
    """返回一个函数，用于检查是否可以将方法委托给基础估算器。

    只有在 `refit=True` 的情况下才能调用预测方法。在这种情况下，我们首先检查最佳已拟合估算器。
    如果它尚未拟合，则检查未拟合的估算器。

    在调用 `fit` 之前，检查未拟合的估算器允许在 `SearchCV` 实例上使用 `hasattr`。
    """
    
    def check(self):
        _check_refit(self, attr)
        if hasattr(self, "best_estimator_"):
            # 如果 `attr` 不存在，则抛出 AttributeError
            getattr(self.best_estimator_, attr)
            return True
        # 如果 `attr` 不存在，则抛出 AttributeError
        getattr(self.estimator, attr)
        return True

    return check


class BaseSearchCV(MetaEstimatorMixin, BaseEstimator, metaclass=ABCMeta):
    """用于带交叉验证的超参数搜索的抽象基类。"""

    _parameter_constraints: dict = {
        "estimator": [HasMethods(["fit"])],
        "scoring": [
            StrOptions(set(get_scorer_names())),
            callable,
            list,
            tuple,
            dict,
            None,
        ],
        "n_jobs": [numbers.Integral, None],
        "refit": ["boolean", str, callable],
        "cv": ["cv_object"],
        "verbose": ["verbose"],
        "pre_dispatch": [numbers.Integral, str],
        "error_score": [StrOptions({"raise"}), numbers.Real],
        "return_train_score": ["boolean"],
    }

    @abstractmethod
    def __init__(
        self,
        estimator,
        *,
        scoring=None,
        n_jobs=None,
        refit=True,
        cv=None,
        verbose=0,
        pre_dispatch="2*n_jobs",
        error_score=np.nan,
        return_train_score=True,
    ):
        self.scoring = scoring
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.refit = refit
        self.cv = cv
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        self.error_score = error_score
        self.return_train_score = return_train_score

    @property
    def _estimator_type(self):
        # 返回基础估算器的估算器类型
        return self.estimator._estimator_type

    def _more_tags(self):
        # 允许交叉验证查看 'precomputed' 指标
        return {
            "pairwise": _safe_tags(self.estimator, "pairwise"),
            "_xfail_checks": {
                "check_supervised_y_2d": "DataConversionWarning not caught"
            },
            "array_api_support": _safe_tags(self.estimator, "array_api_support"),
        }
    # 定义评分方法，用于返回在给定数据上的评分，前提是评估器已经重新拟合过。
    # 如果提供了“scoring”，则使用其定义的评分；否则使用“best_estimator_.score”方法。
    def score(self, X, y=None, **params):
        _check_refit(self, "score")  # 检查评估器是否已经重新拟合
        check_is_fitted(self)  # 检查评估器是否已经拟合

        _raise_for_params(params, self, "score")  # 检查参数是否符合要求

        if _routing_enabled():  # 如果启用了路由
            score_params = process_routing(self, "score", **params).scorer["score"]  # 处理评分参数
        else:
            score_params = dict()  # 如果未启用路由，则参数为空字典

        if self.scorer_ is None:
            raise ValueError(
                "No score function explicitly defined, "
                "and the estimator doesn't provide one %s" % self.best_estimator_
            )  # 如果没有明确定义评分函数且评估器也没有提供评分函数，则引发错误

        if isinstance(self.scorer_, dict):
            if self.multimetric_:
                scorer = self.scorer_[self.refit]  # 如果是多指标评分，选择适合的评分器
            else:
                scorer = self.scorer_  # 否则选择默认的评分器
            return scorer(self.best_estimator_, X, y, **score_params)  # 返回评分结果

        # 如果评分器是可调用的
        score = self.scorer_(self.best_estimator_, X, y, **score_params)
        if self.multimetric_:
            score = score[self.refit]  # 如果是多指标评分，则选择适合的指标
        return score  # 返回评分结果

    @available_if(_estimator_has("score_samples"))
    # 如果评估器支持“score_samples”，则定义方法
    def score_samples(self, X):
        """Call score_samples on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``score_samples``.

        .. versionadded:: 0.24

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements
            of the underlying estimator.

        Returns
        -------
        y_score : ndarray of shape (n_samples,)
            The ``best_estimator_.score_samples`` method.
        """
        check_is_fitted(self)  # 检查评估器是否已经拟合
        return self.best_estimator_.score_samples(X)  # 调用最佳参数的评分样本方法

    @available_if(_estimator_has("predict"))
    # 如果评估器支持“predict”，则定义方法
    # 使用最佳参数调用估算器的预测函数。
    # 只有当 `refit=True` 且基础估算器支持 `predict` 时才可用。
    def predict(self, X):
        """Call predict on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``predict``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted labels or values for `X` based on the estimator with
            the best found parameters.
        """
        # 检查模型是否已拟合
        check_is_fitted(self)
        # 调用最佳估算器的预测方法，并返回预测结果
        return self.best_estimator_.predict(X)

    @available_if(_estimator_has("predict_proba"))
    # 只有当 `refit=True` 且基础估算器支持 `predict_proba` 时才可用。
    def predict_proba(self, X):
        """Call predict_proba on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``predict_proba``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_classes)
            Predicted class probabilities for `X` based on the estimator with
            the best found parameters. The order of the classes corresponds
            to that in the fitted attribute :term:`classes_`.
        """
        # 检查模型是否已拟合
        check_is_fitted(self)
        # 调用最佳估算器的 `predict_proba` 方法，并返回概率预测结果
        return self.best_estimator_.predict_proba(X)

    @available_if(_estimator_has("predict_log_proba"))
    # 只有当 `refit=True` 且基础估算器支持 `predict_log_proba` 时才可用。
    def predict_log_proba(self, X):
        """Call predict_log_proba on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``predict_log_proba``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_classes)
            Predicted class log-probabilities for `X` based on the estimator
            with the best found parameters. The order of the classes
            corresponds to that in the fitted attribute :term:`classes_`.
        """
        # 检查模型是否已拟合
        check_is_fitted(self)
        # 调用最佳估算器的 `predict_log_proba` 方法，并返回对数概率预测结果
        return self.best_estimator_.predict_log_proba(X)

    @available_if(_estimator_has("decision_function"))
    def decision_function(self, X):
        """Call decision_function on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``decision_function``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        Returns
        -------
        y_score : ndarray of shape (n_samples,) or (n_samples, n_classes) \
                or (n_samples, n_classes * (n_classes-1) / 2)
            Result of the decision function for `X` based on the estimator with
            the best found parameters.
        """
        # 检查模型是否已经拟合
        check_is_fitted(self)
        # 调用最佳估计器的 decision_function 方法，返回对输入 X 的决策函数值
        return self.best_estimator_.decision_function(X)

    @available_if(_estimator_has("transform"))
    def transform(self, X):
        """Call transform on the estimator with the best found parameters.

        Only available if the underlying estimator supports ``transform`` and
        ``refit=True``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        Returns
        -------
        Xt : {ndarray, sparse matrix} of shape (n_samples, n_features)
            `X` transformed in the new space based on the estimator with
            the best found parameters.
        """
        # 检查模型是否已经拟合
        check_is_fitted(self)
        # 调用最佳估计器的 transform 方法，返回对输入 X 进行转换后的结果
        return self.best_estimator_.transform(X)

    @available_if(_estimator_has("inverse_transform"))
    def inverse_transform(self, X=None, Xt=None):
        """Call inverse_transform on the estimator with the best found params.

        Only available if the underlying estimator implements
        ``inverse_transform`` and ``refit=True``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        Xt : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

            .. deprecated:: 1.5
                `Xt` was deprecated in 1.5 and will be removed in 1.7. Use `X` instead.

        Returns
        -------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Result of the `inverse_transform` function for `Xt` based on the
            estimator with the best found parameters.
        """
        # 在逆转换前，将 Xt 转为 X，用于兼容旧版本
        X = _deprecate_Xt_in_inverse_transform(X, Xt)
        # 检查模型是否已经拟合
        check_is_fitted(self)
        # 调用最佳估计器的 inverse_transform 方法，返回对输入 X 进行逆转换后的结果
        return self.best_estimator_.inverse_transform(X)
    def n_features_in_(self):
        """Number of features seen during :term:`fit`.

        Only available when `refit=True`.
        """
        # 检查是否已经拟合，若未拟合则引发 AttributeError，使得 hasattr() 失败
        try:
            check_is_fitted(self)
        except NotFittedError as nfe:
            raise AttributeError(
                "{} object has no n_features_in_ attribute.".format(
                    self.__class__.__name__
                )
            ) from nfe

        # 返回最佳估计器的 n_features_in_ 属性
        return self.best_estimator_.n_features_in_

    @property
    def classes_(self):
        """Class labels.

        Only available when `refit=True` and the estimator is a classifier.
        """
        # 确保估计器具有 classes_ 属性，如果没有将引发异常
        _estimator_has("classes_")(self)
        # 返回最佳估计器的 classes_ 属性
        return self.best_estimator_.classes_

    def _check_refit_for_multimetric(self, scores):
        """Check `refit` is compatible with `scores` is valid"""
        # 检查多度量评分情况下 refit 参数的有效性
        multimetric_refit_msg = (
            "For multi-metric scoring, the parameter refit must be set to a "
            "scorer key or a callable to refit an estimator with the best "
            "parameter setting on the whole data and make the best_* "
            "attributes available for that metric. If this is not needed, "
            f"refit should be set to False explicitly. {self.refit!r} was "
            "passed."
        )

        # 检查 refit 参数的类型和内容，确保其在多度量评分场景下的有效性
        valid_refit_dict = isinstance(self.refit, str) and self.refit in scores

        if (
            self.refit is not False
            and not valid_refit_dict
            and not callable(self.refit)
        ):
            # 若 refit 参数不合法，引发 ValueError
            raise ValueError(multimetric_refit_msg)

    @staticmethod
    def _select_best_index(refit, refit_metric, results):
        """Select index of the best combination of hyperparemeters."""
        # 根据 refit 参数的类型选择最佳超参数组合的索引
        if callable(refit):
            # 若 refit 是可调用对象，则期望其返回最佳参数集的索引
            best_index = refit(results)
            # 确保返回的最佳索引是整数类型
            if not isinstance(best_index, numbers.Integral):
                raise TypeError("best_index_ returned is not an integer")
            # 确保返回的最佳索引在结果参数列表的有效范围内
            if best_index < 0 or best_index >= len(results["params"]):
                raise IndexError("best_index_ index out of range")
        else:
            # 若 refit 是字符串，则根据对应的 refit_metric 选择排名最低的索引
            best_index = results[f"rank_test_{refit_metric}"].argmin()
        return best_index
    def _get_scorers(self):
        """Get the scorer(s) to be used.

        This is used in ``fit`` and ``get_metadata_routing``.

        Returns
        -------
        scorers, refit_metric
        """
        refit_metric = "score"  # 设定默认的重拟合指标为 "score"

        if callable(self.scoring):  # 如果 self.scoring 是可调用对象（函数等）
            scorers = self.scoring  # 则将 scorers 设置为 self.scoring
        elif self.scoring is None or isinstance(self.scoring, str):  # 如果 self.scoring 是 None 或者字符串
            scorers = check_scoring(self.estimator, self.scoring)  # 则使用 check_scoring 函数获取 scorers
        else:  # 否则，假定 self.scoring 是多指标的情况
            scorers = _check_multimetric_scoring(self.estimator, self.scoring)  # 使用 _check_multimetric_scoring 获取 scorers
            self._check_refit_for_multimetric(scorers)  # 检查是否需要重拟合
            refit_metric = self.refit  # 设置重拟合指标为 self.refit
            scorers = _MultimetricScorer(
                scorers=scorers, raise_exc=(self.error_score == "raise")
            )  # 封装成 _MultimetricScorer 对象

        return scorers, refit_metric  # 返回 scorers 和 refit_metric

    def _get_routed_params_for_fit(self, params):
        """Get the parameters to be used for routing.

        This is a method instead of a snippet in ``fit`` since it's used twice,
        here in ``fit``, and in ``HalvingRandomSearchCV.fit``.
        """
        if _routing_enabled():  # 如果路由功能已启用
            routed_params = process_routing(self, "fit", **params)  # 则使用 process_routing 处理路由
        else:  # 否则
            params = params.copy()  # 复制参数
            groups = params.pop("groups", None)  # 弹出参数中的 "groups" 键
            routed_params = Bunch(
                estimator=Bunch(fit=params),
                splitter=Bunch(split={"groups": groups}),
                scorer=Bunch(score={}),
            )  # 创建 Bunch 对象，包含估计器、分割器和评分器

        return routed_params  # 返回路由后的参数集合

    @_fit_context(
        # *SearchCV.estimator is not validated yet
        prefer_skip_nested_validation=False
    )
    def get_metadata_routing(self):
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        .. versionadded:: 1.4

        Returns
        -------
        routing : MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        router = MetadataRouter(owner=self.__class__.__name__)  # 创建 MetadataRouter 对象，设置所有者名称

        router.add(
            estimator=self.estimator,
            method_mapping=MethodMapping().add(caller="fit", callee="fit"),
        )  # 将估计器和 fit 方法映射添加到路由器中

        scorer, _ = self._get_scorers()  # 获取评分器
        router.add(
            scorer=scorer,
            method_mapping=MethodMapping()
            .add(caller="score", callee="score")
            .add(caller="fit", callee="score"),
        )  # 将评分器和 score、fit 方法映射添加到路由器中

        router.add(
            splitter=self.cv,
            method_mapping=MethodMapping().add(caller="fit", callee="split"),
        )  # 将分割器和 fit 方法映射添加到路由器中

        return router  # 返回包含路由信息的 MetadataRouter 对象
    # 定义一个方法 `_sk_visual_block_`，用于生成一个视觉块对象
    def _sk_visual_block_(self):
        # 检查当前对象是否具有属性 "best_estimator_"，如果有则使用它作为关键词和估算器
        if hasattr(self, "best_estimator_"):
            key, estimator = "best_estimator_", self.best_estimator_
        else:
            # 如果没有 "best_estimator_" 属性，则使用 "estimator" 属性作为关键词和估算器
            key, estimator = "estimator", self.estimator

        # 返回一个 _VisualBlock 对象
        return _VisualBlock(
            "parallel",  # 使用 "parallel" 模式创建视觉块
            [estimator],  # 将估算器作为列表传递给视觉块
            names=[f"{key}: {estimator.__class__.__name__}"],  # 使用关键词和估算器类名创建名称列表
            name_details=[str(estimator)],  # 使用估算器的字符串表示作为名称细节列表的元素
        )
class GridSearchCV(BaseSearchCV):
    """Exhaustive search over specified parameter values for an estimator.

    Important members are fit, predict.

    GridSearchCV implements a "fit" and a "score" method.
    It also implements "score_samples", "predict", "predict_proba",
    "decision_function", "transform" and "inverse_transform" if they are
    implemented in the estimator used.

    The parameters of the estimator used to apply these methods are optimized
    by cross-validated grid-search over a parameter grid.

    Read more in the :ref:`User Guide <grid_search>`.

    Parameters
    ----------
    estimator : estimator object
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    param_grid : dict or list of dictionaries
        Dictionary with parameters names (`str`) as keys and lists of
        parameter settings to try as values, or a list of such
        dictionaries, in which case the grids spanned by each dictionary
        in the list are explored. This enables searching over any sequence
        of parameter settings.

    scoring : str, callable, list, tuple or dict, default=None
        Strategy to evaluate the performance of the cross-validated model on
        the test set.

        If `scoring` represents a single score, one can use:

        - a single string (see :ref:`scoring_parameter`);
        - a callable (see :ref:`scoring`) that returns a single value.

        If `scoring` represents multiple scores, one can use:

        - a list or tuple of unique strings;
        - a callable returning a dictionary where the keys are the metric
          names and the values are the metric scores;
        - a dictionary with metric names as keys and callables a values.

        See :ref:`multimetric_grid_search` for an example.

    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

        .. versionchanged:: v0.20
           `n_jobs` default changed from 1 to None
    refit : bool, str, or callable, default=True
        # refit参数可以是布尔值、字符串或可调用对象，默认为True
        Refit an estimator using the best found parameters on the whole
        dataset.
        # 使用找到的最佳参数对整个数据集上的估算器进行重新拟合

        For multiple metric evaluation, this needs to be a `str` denoting the
        scorer that would be used to find the best parameters for refitting
        the estimator at the end.
        # 对于多指标评估，如果需要重新拟合最佳参数，这应为一个字符串，指定用于在最后重新拟合估算器的评分器

        Where there are considerations other than maximum score in
        choosing a best estimator, ``refit`` can be set to a function which
        returns the selected ``best_index_`` given ``cv_results_``. In that
        case, the ``best_estimator_`` and ``best_params_`` will be set
        according to the returned ``best_index_`` while the ``best_score_``
        attribute will not be available.
        # 如果在选择最佳估算器时考虑除了最大分数以外的其他因素，则可以将“refit”设置为一个函数，该函数根据给定的“cv_results_”返回选定的“best_index_”。在这种情况下，“best_estimator_”和“best_params_”将根据返回的“best_index_”设置，而“best_score_”属性将不可用。

        The refitted estimator is made available at the ``best_estimator_``
        attribute and permits using ``predict`` directly on this
        ``GridSearchCV`` instance.
        # 重新拟合后的估算器可以通过“best_estimator_”属性访问，并允许在此GridSearchCV实例上直接使用“predict”。

        Also for multiple metric evaluation, the attributes ``best_index_``,
        ``best_score_`` and ``best_params_`` will only be available if
        ``refit`` is set and all of them will be determined w.r.t this specific
        scorer.
        # 对于多指标评估，只有在设置了“refit”并且所有这些属性都将基于特定的评分器确定时，“best_index_”、“best_score_”和“best_params_”属性才可用。

        See ``scoring`` parameter to know more about multiple metric
        evaluation.
        # 查看“scoring”参数以了解更多关于多指标评估的信息。

        See :ref:`sphx_glr_auto_examples_model_selection_plot_grid_search_digits.py`
        to see how to design a custom selection strategy using a callable
        via `refit`.
        # 查看:ref:`sphx_glr_auto_examples_model_selection_plot_grid_search_digits.py`以了解如何使用可调用对象通过“refit”设计自定义选择策略。

        .. versionchanged:: 0.20
            Support for callable added.
        # 版本更改：0.20 添加了对可调用对象的支持。

    cv : int, cross-validation generator or an iterable, default=None
        # cv参数可以是整数、交叉验证生成器或可迭代对象，默认为None
        Determines the cross-validation splitting strategy.
        # 确定交叉验证的分割策略。
        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        # None，使用默认的5折交叉验证，

        - integer, to specify the number of folds in a `(Stratified)KFold`,
        # 整数，指定(Stratified)KFold中的折数，

        - :term:`CV splitter`,
        # CV分割器，

        - An iterable yielding (train, test) splits as arrays of indices.
        # 生成一个可迭代对象，产生(train, test)的分割数组作为索引。

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.
        # 对于整数/None输入，如果估算器是分类器且“y”是二进制或多类别，则使用:class:`StratifiedKFold`。在所有其他情况下，使用:class:`KFold`。这些分割器使用“shuffle=False”进行实例化，因此分割在每次调用时保持相同。

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.
        # 参考:ref:`User Guide <cross_validation>`以了解可在此处使用的各种交叉验证策略。

        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.
        # 版本更改：0.22 如果为None，“cv”的默认值从3折变为5折。

    verbose : int
        # verbose参数为整数，控制详细程度，数字越高输出的信息越多。
        Controls the verbosity: the higher, the more messages.
        - >1 : the computation time for each fold and parameter candidate is
          displayed;
        # >1：显示每个折叠和参数候选的计算时间；

        - >2 : the score is also displayed;
        # >2：还显示分数；

        - >3 : the fold and candidate parameter indexes are also displayed
          together with the starting time of the computation.
        # >3：还显示折数和候选参数索引，以及计算开始时间。
    pre_dispatch : int, or str, default='2*n_jobs'
        # 控制并行执行过程中分派的作业数量。当分派的作业超过 CPU 处理能力时，减少此数值有助于避免内存消耗的激增。
        # 可选参数:
        # - None: 所有作业立即创建并启动。适用于轻量且运行速度快的作业，避免由于按需生成作业而导致的延迟。
        # - 整数: 指定要启动的总作业数量。
        # - 字符串: 作为 n_jobs 的函数表达式，例如 '2*n_jobs'。

    error_score : 'raise' or numeric, default=np.nan
        # 如果在拟合估计器时发生错误，用于赋值给分数的值。
        # - 如果设置为 'raise'，则会抛出错误。
        # - 如果给出数值，则会引发 FitFailedWarning。
        # 此参数不影响重新拟合步骤，该步骤总是会引发错误。

    return_train_score : bool, default=False
        # 如果为 ``False``，则 ``cv_results_`` 属性将不包括训练分数。
        # 计算训练分数用于了解不同参数设置对过拟合/欠拟合的影响。
        # 然而，在训练集上计算分数可能计算开销很大，且不是严格要求选择产生最佳泛化性能的参数。

        # .. versionadded:: 0.19
        # .. versionchanged:: 0.21
        #    默认值从 ``True`` 更改为 ``False``

    Attributes
    ----------
    best_estimator_ : estimator
        # 由搜索选择的估计器，即在留出数据上获得最高分数（或最小损失，如果指定）的估计器。
        # 如果 ``refit=False``，则不可用。有关允许值的更多信息，请参阅 ``refit`` 参数。

    best_score_ : float
        # 最佳估计器的平均交叉验证分数。
        # 对于多指标评估，仅在指定了 ``refit`` 时存在。

    best_params_ : dict
        # 在留出数据上获得最佳结果的参数设置。
        # 对于多指标评估，仅在指定了 ``refit`` 时存在。

    best_index_ : int
        # 对应于最佳候选参数设置的 ``cv_results_`` 数组的索引。
        # ``search.cv_results_['params'][search.best_index_]`` 中的字典给出了给出最高均分的最佳模型的参数设置。
        # 对于多指标评估，仅在指定了 ``refit`` 时存在。
    # scorer_ : function or a dict
    #   评分函数或字典。用于在留出数据上评估模型以选择最佳参数。

    # n_splits_ : int
    #   交叉验证分割的数量（折数/迭代次数）。

    # refit_time_ : float
    #   在整个数据集上重新拟合最佳模型所用的秒数。
    #   仅在 `refit` 参数不为 False 时存在。
    #   
    #   .. versionadded:: 0.20

    # multimetric_ : bool
    #   评分器是否计算多个指标。

    # classes_ : ndarray of shape (n_classes,)
    #   类别标签。仅在指定了 `refit` 参数且基础估计器是分类器时存在。

    # n_features_in_ : int
    #   在拟合过程中看到的特征数量。仅在 `best_estimator_` 定义了并且在拟合时 `best_estimator_` 公开了 `n_features_in_` 时才定义。
    #   
    #   .. versionadded:: 0.24

    # feature_names_in_ : ndarray of shape (`n_features_in_`,)
    #   在拟合过程中看到的特征名称。仅在 `best_estimator_` 定义了并且在拟合时 `best_estimator_` 公开了 `feature_names_in_` 时才定义。
    #   
    #   .. versionadded:: 1.0

    # See Also
    # --------
    # ParameterGrid : 生成超参数网格的所有组合。
    # train_test_split : 将数据拆分为开发集（用于拟合 GridSearchCV 实例）和评估集（用于最终评估）的实用函数。
    # sklearn.metrics.make_scorer : 从性能指标或损失函数创建评分器。

    # Notes
    # -----
    # 所选参数是最大化留出数据得分的参数，除非显式传递了评分，否则将使用该评分。

    # 如果 `n_jobs` 设置为高于一的值，则为网格中的每个点复制数据（而不是 `n_jobs` 次）。
    # 如果每个作业时间很短，出于效率考虑会这样做，但如果数据集很大且内存不足，则可能会引发错误。
    # 在这种情况下的一种解决方法是设置 `pre_dispatch`。
    # 然后，内存仅复制 `pre_dispatch` 次。`pre_dispatch` 的合理值是 `2 * n_jobs`。

    # Examples
    # --------
    # >>> from sklearn import svm, datasets
    # >>> from sklearn.model_selection import GridSearchCV
    # >>> iris = datasets.load_iris()
    # >>> parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    # >>> svc = svm.SVC()
    # >>> clf = GridSearchCV(svc, parameters)
    # >>> clf.fit(iris.data, iris.target)
    # GridSearchCV(estimator=SVC(),
    #              param_grid={'C': [1, 10], 'kernel': ('linear', 'rbf')})
    # >>> sorted(clf.cv_results_.keys())
    ['mean_fit_time', 'mean_score_time', 'mean_test_score',...
     'param_C', 'param_kernel', 'params',...
     'rank_test_score', 'split0_test_score',...
     'split2_test_score', ...
     'std_fit_time', 'std_score_time', 'std_test_score']
    """
    
    _required_parameters = ["estimator", "param_grid"]

    _parameter_constraints: dict = {
        **BaseSearchCV._parameter_constraints,
        "param_grid": [dict, list],
    }

    def __init__(
        self,
        estimator,
        param_grid,
        *,
        scoring=None,
        n_jobs=None,
        refit=True,
        cv=None,
        verbose=0,
        pre_dispatch="2*n_jobs",
        error_score=np.nan,
        return_train_score=False,
    ):
        # 调用父类的构造方法，初始化基类的属性
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )
        # 将参数网格存储在实例中
        self.param_grid = param_grid

    def _run_search(self, evaluate_candidates):
        """执行搜索，评估参数网格中的所有候选项"""
        # 调用evaluate_candidates函数，对param_grid中的所有参数组合进行搜索
        evaluate_candidates(ParameterGrid(self.param_grid))
# RandomizedSearchCV 类，用于在超参数上进行随机搜索。

class RandomizedSearchCV(BaseSearchCV):
    """Randomized search on hyper parameters.

    RandomizedSearchCV 实现了 "fit" 和 "score" 方法。
    它还实现了 "score_samples"、"predict"、"predict_proba"、
    "decision_function"、"transform" 和 "inverse_transform" 方法，
    如果在使用的估计器中实现了这些方法的话。

    The parameters of the estimator used to apply these methods are optimized
    by cross-validated search over parameter settings.
    使用交叉验证搜索优化应用这些方法的估计器的参数设置。

    In contrast to GridSearchCV, not all parameter values are tried out, but
    rather a fixed number of parameter settings is sampled from the specified
    distributions. The number of parameter settings that are tried is
    given by n_iter.
    与 GridSearchCV 不同，不是尝试所有参数值，而是从指定的分布中随机抽取固定数量的参数设置。
    尝试的参数设置数量由 n_iter 给出。

    If all parameters are presented as a list,
    sampling without replacement is performed. If at least one parameter
    is given as a distribution, sampling with replacement is used.
    如果所有参数都以列表形式给出，则执行无替换抽样。如果至少有一个参数以分布形式给出，则使用有替换抽样。
    强烈建议对连续参数使用连续分布。

    Read more in the :ref:`User Guide <randomized_parameter_search>`.

    .. versionadded:: 0.14

    Parameters
    ----------
    estimator : estimator object
        用于每个网格点实例化的对象。假设其实现了 scikit-learn 的估计器接口。
        估计器需要提供一个 "score" 方法，或者必须传递 "scoring" 参数。

    param_distributions : dict or list of dicts
        参数名（`str`）作为键，分布或要尝试的参数列表作为值的字典。
        分布必须提供一个用于抽样的 `rvs` 方法（例如来自 scipy.stats.distributions）。
        如果给出一个列表，则进行均匀抽样。
        如果给出一个字典列表，则首先均匀抽样一个字典，然后使用上述字典进行参数抽样。

    n_iter : int, default=10
        抽样的参数设置数量。n_iter 在运行时间与解决方案质量之间进行交换。

    scoring : str, callable, list, tuple or dict, default=None
        评估交叉验证模型在测试集上性能的策略。

        如果 `scoring` 表示单个分数，则可以使用：

        - 单个字符串（参见 :ref:`scoring_parameter`）；
        - 可调用函数（参见 :ref:`scoring`），返回单个值。

        如果 `scoring` 表示多个分数，则可以使用：

        - 独特字符串的列表或元组；
        - 返回字典的可调用函数，其中键是度量名称，值是度量分数；
        - 具有度量名称作为键和可调用函数作为值的字典。

        请参阅 :ref:`multimetric_grid_search` 以获取示例。

        如果为 None，则使用估计器的 score 方法。
    """
    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
        
        .. versionchanged:: v0.20
           `n_jobs` default changed from 1 to None

        # 并行作业数，可以设置为整数
        # None 表示默认为 1，除非在 :obj:`joblib.parallel_backend` 上下文中
        # -1 表示使用所有处理器。详细信息请参见 :term:`Glossary <n_jobs>`

    refit : bool, str, or callable, default=True
        Refit an estimator using the best found parameters on the whole
        dataset.

        For multiple metric evaluation, this needs to be a `str` denoting the
        scorer that would be used to find the best parameters for refitting
        the estimator at the end.

        Where there are considerations other than maximum score in
        choosing a best estimator, ``refit`` can be set to a function which
        returns the selected ``best_index_`` given the ``cv_results``. In that
        case, the ``best_estimator_`` and ``best_params_`` will be set
        according to the returned ``best_index_`` while the ``best_score_``
        attribute will not be available.

        The refitted estimator is made available at the ``best_estimator_``
        attribute and permits using ``predict`` directly on this
        ``RandomizedSearchCV`` instance.

        Also for multiple metric evaluation, the attributes ``best_index_``,
        ``best_score_`` and ``best_params_`` will only be available if
        ``refit`` is set and all of them will be determined w.r.t this specific
        scorer.

        See ``scoring`` parameter to know more about multiple metric
        evaluation.

        .. versionchanged:: 0.20
            Support for callable added.

        # 是否重新拟合估算器，使用在整个数据集上找到的最佳参数
        # 如果进行多指标评估，需要将其设为字符串，表示用于在最后重新拟合估算器的评分器
        # 如果在选择最佳估算器时有除最大分数之外的其他考虑，可以将 ``refit`` 设置为一个函数，
        # 该函数根据 ``cv_results`` 返回选定的 ``best_index_``
        # 重新拟合后的估算器在 ``best_estimator_`` 属性中可用，并允许直接在此实例上使用 ``predict``
        # 对于多指标评估，只有当 ``refit`` 被设置并且所有这些属性将根据特定评分器确定时，
        # ``best_index_``, ``best_score_`` 和 ``best_params_`` 属性才可用

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.

        # 确定交叉验证分割策略
        # 可用的 cv 输入有：
        # - None，使用默认的 5 折交叉验证
        # - 整数，指定 `(Stratified)KFold` 中的折数
        # - :term:`CV splitter`
        # - 作为索引数组的可迭代对象，生成 (train, test) 分割
        # 如果估算器是分类器且 ``y`` 是二进制或多类别，则使用 :class:`StratifiedKFold`，否则使用 :class:`KFold`
        # 这些分割器使用 `shuffle=False` 实例化，因此调用时的分割将保持一致

    verbose : int
        Controls the verbosity: the higher, the more messages.

        - >1 : the computation time for each fold and parameter candidate is
          displayed;
        - >2 : the score is also displayed;
        - >3 : the fold and candidate parameter indexes are also displayed
          together with the starting time of the computation.

        # 控制详细程度：越高，显示的消息越多
        # - >1：显示每个折叠和参数候选的计算时间
        # - >2：还显示评分
        # - >3：还显示折叠和候选参数索引，以及计算的开始时间
    pre_dispatch : int, or str, default='2*n_jobs'
        # 控制并行执行期间分派的作业数量。减少此数字可避免在分派的作业超过CPU可处理的数量时内存消耗激增。
        # 此参数可以是以下之一：

        # - None，此时所有作业立即创建和启动。对于轻量且运行速度快的作业，避免由于按需生成作业而导致的延迟。

        # - 一个整数，表示要生成的总作业数。

        # - 一个字符串，表示作为 n_jobs 函数的表达式，例如 '2*n_jobs'
        
    random_state : int, RandomState instance or None, default=None
        # 用于从可能值列表中进行随机均匀抽样的伪随机数生成器状态，而不是使用 scipy.stats 分布。
        # 传递一个整数以确保在多次函数调用中获得可重复的输出。
        # 参见 :term:`Glossary <random_state>`。

    error_score : 'raise' or numeric, default=np.nan
        # 如果在估计器拟合过程中出现错误，用于分配给得分的值。
        # 如果设置为 'raise'，则会引发错误。如果给定一个数值，则会引发 FitFailedWarning。
        # 此参数不影响 refit 步骤，后者总会引发错误。

    return_train_score : bool, default=False
        # 如果为 ``False``，``cv_results_`` 属性将不包括训练分数。
        # 计算训练分数用于了解不同参数设置如何影响过拟合/欠拟合的权衡。
        # 然而，在训练集上计算分数可能计算开销很大，并且不严格要求选择具有最佳泛化性能的参数。

        # .. versionadded:: 0.19

        # .. versionchanged:: 0.21
        #    默认值从 ``True`` 更改为 ``False``
    cv_results_ : dict of numpy (masked) ndarrays
        A dictionary containing various evaluation results from cross-validation.

        Each key represents a different metric or parameter, and its corresponding value
        is an array or masked array storing the computed scores or statistics.

        For instance, it includes:
        - 'param_kernel': Kernel types used in parameter combinations
        - 'param_gamma': Gamma values used in parameter combinations
        - 'split0_test_score': Scores for the first split in cross-validation
        - 'mean_test_score': Mean scores across all splits
        - 'std_test_score': Standard deviation of scores across all splits
        - 'rank_test_score': Rank of each parameter combination based on scores

        It also includes timing information:
        - 'mean_fit_time': Mean time taken for fitting across all splits
        - 'std_fit_time': Standard deviation of fit times across all splits
        - 'mean_score_time': Mean time taken for scoring
        - 'std_score_time': Standard deviation of score times

        Additional metrics like precision or other scorers have their scores
        under keys ending with their respective names ('_precision', '_recall', etc.).

        The key 'params' holds a list of dictionaries, each containing a set of
        parameter settings used in the evaluation.

        NOTE:
        - The format supports multi-metric evaluation.
        - Time-related metrics ('fit_time' and 'score_time') are in seconds.

    best_estimator_ : estimator
        The best estimator chosen by the search based on the highest score (or smallest loss).

        This attribute is populated only if 'refit=True' was specified during the search.

        See the 'refit' parameter documentation for details on its values and behavior.
    best_score_ : float
        # 存储最佳估计器的交叉验证得分的平均值。

        # 对于多指标评估，如果“refit”为“False”，则此值不可用。有关更多信息，请参阅“refit”参数。

        # 如果“refit”是一个函数，则此属性不可用。

    best_params_ : dict
        # 给出在留出数据上获得最佳结果的参数设置。

        # 对于多指标评估，如果“refit”为“False”，则此值不可用。有关更多信息，请参阅“refit”参数。

    best_index_ : int
        # 对应于最佳候选参数设置的“cv_results_”数组的索引。

        # “search.cv_results_['params'][search.best_index_]”中的字典给出了最佳模型的参数设置，该设置给出了最高的平均分数（“search.best_score_”）。

        # 对于多指标评估，如果“refit”为“False”，则此值不可用。有关更多信息，请参阅“refit”参数。

    scorer_ : function or a dict
        # 用于在留出数据上选择最佳模型参数的评分函数。

        # 对于多指标评估，此属性保存经过验证的“scoring”字典，该字典将评分器键映射到评分器可调用对象。

    n_splits_ : int
        # 交叉验证拆分（折叠/迭代）的数量。

    refit_time_ : float
        # 用于在整个数据集上重新拟合最佳模型所用的秒数。

        # 仅当“refit”不为False时才存在。

        # .. versionadded:: 0.20

    multimetric_ : bool
        # 评分器是否计算多个指标。

    classes_ : ndarray of shape (n_classes,)
        # 类别标签。仅当指定了“refit”并且基础估计器是分类器时才存在。

    n_features_in_ : int
        # 在拟合期间看到的特征数量。仅在定义了“best_estimator_”（有关“refit”参数的更多详细信息，请参阅文档）且“best_estimator_”在拟合时公开了“n_features_in_”时才定义。

        # .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        # 在拟合期间看到的特征名称。仅在定义了“best_estimator_”（有关“refit”参数的更多详细信息，请参阅文档）且“best_estimator_”在拟合时公开了“feature_names_in_”时才定义。

        # .. versionadded:: 1.0

    See Also
    --------
    GridSearchCV : 对参数网格执行详尽搜索。
    ParameterSampler : 从param_distributions构造的参数设置生成器。

    Notes
    -----
    # 选择的参数是使留出数据的得分最大化的参数。

    # 如果“n_jobs”设置为高于1的值，则为每个参数设置复制数据（而不是“n_jobs”次）。这样做是为了效率
    reasons if individual jobs take very little time, but may raise errors if
    the dataset is large and not enough memory is available.  A workaround in
    this case is to set `pre_dispatch`. Then, the memory is copied only
    `pre_dispatch` many times. A reasonable value for `pre_dispatch` is `2 *
    n_jobs`.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.model_selection import RandomizedSearchCV
    >>> from scipy.stats import uniform
    >>> iris = load_iris()
    >>> logistic = LogisticRegression(solver='saga', tol=1e-2, max_iter=200,
    ...                               random_state=0)
    >>> distributions = dict(C=uniform(loc=0, scale=4),
    ...                      penalty=['l2', 'l1'])
    >>> clf = RandomizedSearchCV(logistic, distributions, random_state=0)
    >>> search = clf.fit(iris.data, iris.target)
    >>> search.best_params_
    {'C': 2..., 'penalty': 'l1'}
    """

# 定义类的必需参数列表
_required_parameters = ["estimator", "param_distributions"]

# 定义参数约束字典，继承自BaseSearchCV的约束，并添加新的约束条件
_parameter_constraints: dict = {
    **BaseSearchCV._parameter_constraints,
    "param_distributions": [dict, list],
    "n_iter": [Interval(numbers.Integral, 1, None, closed="left")],
    "random_state": ["random_state"],
}

def __init__(
    self,
    estimator,
    param_distributions,
    *,
    n_iter=10,
    scoring=None,
    n_jobs=None,
    refit=True,
    cv=None,
    verbose=0,
    pre_dispatch="2*n_jobs",
    random_state=None,
    error_score=np.nan,
    return_train_score=False,
):
    # 初始化随机搜索对象
    self.param_distributions = param_distributions
    self.n_iter = n_iter
    self.random_state = random_state
    # 调用父类BaseSearchCV的初始化方法
    super().__init__(
        estimator=estimator,
        scoring=scoring,
        n_jobs=n_jobs,
        refit=refit,
        cv=cv,
        verbose=verbose,
        pre_dispatch=pre_dispatch,
        error_score=error_score,
        return_train_score=return_train_score,
    )

def _run_search(self, evaluate_candidates):
    """Search n_iter candidates from param_distributions"""
    # 对参数分布中的n_iter个候选项进行评估
    evaluate_candidates(
        ParameterSampler(
            self.param_distributions, self.n_iter, random_state=self.random_state
        )
    )
```