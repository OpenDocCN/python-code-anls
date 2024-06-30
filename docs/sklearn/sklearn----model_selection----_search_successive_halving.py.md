# `D:\src\scipysrc\scikit-learn\sklearn\model_selection\_search_successive_halving.py`

```
    from abc import abstractmethod
    from copy import deepcopy
    from math import ceil, floor, log
    from numbers import Integral, Real
    
    import numpy as np
    
    from ..base import _fit_context, is_classifier
    from ..metrics._scorer import get_scorer_names
    from ..utils import resample
    from ..utils._param_validation import Interval, StrOptions
    from ..utils.multiclass import check_classification_targets
    from ..utils.validation import _num_samples
    from . import ParameterGrid, ParameterSampler
    from ._search import BaseSearchCV
    from ._split import _yields_constant_splits, check_cv
    
    __all__ = ["HalvingGridSearchCV", "HalvingRandomSearchCV"]
    
    class _SubsampleMetaSplitter:
        """Splitter that subsamples a given fraction of the dataset"""
    
        def __init__(self, *, base_cv, fraction, subsample_test, random_state):
            self.base_cv = base_cv
            self.fraction = fraction
            self.subsample_test = subsample_test
            self.random_state = random_state
    
        def split(self, X, y, **kwargs):
            # Iterate over base cross-validation splits
            for train_idx, test_idx in self.base_cv.split(X, y, **kwargs):
                # Resample training indices based on fraction
                train_idx = resample(
                    train_idx,
                    replace=False,
                    random_state=self.random_state,
                    n_samples=int(self.fraction * len(train_idx)),
                )
                # Optionally resample test indices based on fraction
                if self.subsample_test:
                    test_idx = resample(
                        test_idx,
                        replace=False,
                        random_state=self.random_state,
                        n_samples=int(self.fraction * len(test_idx)),
                    )
                yield train_idx, test_idx
    
    def _top_k(results, k, itr):
        # Return the best candidates of a given iteration
        iteration, mean_test_score, params = (
            np.asarray(a)
            for a in (results["iter"], results["mean_test_score"], results["params"])
        )
        # Find indices of results corresponding to current iteration
        iter_indices = np.flatnonzero(iteration == itr)
        # Extract scores for current iteration
        scores = mean_test_score[iter_indices]
        # Sort indices based on scores, handling NaNs properly
        sorted_indices = np.roll(np.argsort(scores), np.count_nonzero(np.isnan(scores)))
        # Return top k parameter sets with highest scores
        return np.array(params[iter_indices][sorted_indices[-k:]])
    
    class BaseSuccessiveHalving(BaseSearchCV):
        """Implements successive halving.
    
        Ref:
        Almost optimal exploration in multi-armed bandits, ICML 13
        Zohar Karnin, Tomer Koren, Oren Somekh
        """
    _parameter_constraints: dict = {
        **BaseSearchCV._parameter_constraints,
        # 覆盖 `scoring`，因为不支持多个评分指标
        "scoring": [StrOptions(set(get_scorer_names())), callable, None],
        "random_state": ["random_state"],  # 随机数种子
        "max_resources": [
            Interval(Integral, 0, None, closed="neither"),  # 最大资源的约束条件
            StrOptions({"auto"}),  # 支持的字符串选项
        ],
        "min_resources": [
            Interval(Integral, 0, None, closed="neither"),  # 最小资源的约束条件
            StrOptions({"exhaust", "smallest"}),  # 支持的字符串选项
        ],
        "resource": [str],  # 资源类型的约束条件
        "factor": [Interval(Real, 0, None, closed="neither")],  # 因子的约束条件
        "aggressive_elimination": ["boolean"],  # 是否进行激进的消除
    }
    _parameter_constraints.pop("pre_dispatch")  # 不在本类中使用的参数，移除

    def __init__(
        self,
        estimator,
        *,
        scoring=None,  # 评分方法，可以为 None
        n_jobs=None,  # 并行工作的任务数
        refit=True,  # 是否在找到最佳参数后重新拟合估算器
        cv=5,  # 交叉验证折数
        verbose=0,  # 详细程度
        random_state=None,  # 随机数种子
        error_score=np.nan,  # 出错时的分数
        return_train_score=True,  # 是否返回训练集上的得分
        max_resources="auto",  # 最大资源，默认自动
        min_resources="exhaust",  # 最小资源，默认用尽
        resource="n_samples",  # 资源类型，默认样本数
        factor=3,  # 因子，默认为 3
        aggressive_elimination=False,  # 是否进行激进的消除，默认否
    ):
        super().__init__(
            estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            error_score=error_score,
            return_train_score=return_train_score,
        )

        self.random_state = random_state  # 设置对象的随机数种子属性
        self.max_resources = max_resources  # 设置对象的最大资源属性
        self.resource = resource  # 设置对象的资源类型属性
        self.factor = factor  # 设置对象的因子属性
        self.min_resources = min_resources  # 设置对象的最小资源属性
        self.aggressive_elimination = aggressive_elimination  # 设置对象的激进消除属性
    # 检查输入参数的有效性，确保 CV 分割是稳定的
    def _check_input_parameters(self, X, y, split_params):
        if not _yields_constant_splits(self._checked_cv_orig):
            # 如果 CV 不稳定，则抛出数值错误
            raise ValueError(
                "The cv parameter must yield consistent folds across "
                "calls to split(). Set its random_state to an int, or set "
                "shuffle=False."
            )

        if (
            self.resource != "n_samples"
            and self.resource not in self.estimator.get_params()
        ):
            # 如果资源不是 n_samples 且不在估计器参数中，则抛出数值错误
            raise ValueError(
                f"Cannot use resource={self.resource} which is not supported "
                f"by estimator {self.estimator.__class__.__name__}"
            )

        if isinstance(self, HalvingRandomSearchCV):
            if self.min_resources == self.n_candidates == "exhaust":
                # 如果同时设置 n_candidates 和 min_resources 为 'exhaust'，则抛出数值错误
                raise ValueError(
                    "n_candidates and min_resources cannot be both set to 'exhaust'."
                )

        # 设置最小资源量为初始值
        self.min_resources_ = self.min_resources
        if self.min_resources_ in ("smallest", "exhaust"):
            if self.resource == "n_samples":
                # 如果资源是 n_samples，则根据 CV 分割的数量和类别数调整最小资源量
                n_splits = self._checked_cv_orig.get_n_splits(X, y, **split_params)
                # 请参见 https://gph.is/1KjihQe 以获取相关理由
                magic_factor = 2
                self.min_resources_ = n_splits * magic_factor
                if is_classifier(self.estimator):
                    # 如果是分类器，则验证目标数据并检查分类目标
                    y = self._validate_data(X="no_validation", y=y)
                    check_classification_targets(y)
                    n_classes = np.unique(y).shape[0]
                    self.min_resources_ *= n_classes
            else:
                # 否则设置最小资源量为 1
                self.min_resources_ = 1
            # 如果设置为 'exhaust'，则最小资源量可能在 _run_search 中被重新设置为更高的值
            # in _run_search 中可能会将 min_resources_ 设置为更高的值

        # 设置最大资源量为初始值
        self.max_resources_ = self.max_resources
        if self.max_resources_ == "auto":
            # 如果最大资源量设置为 'auto'
            if not self.resource == "n_samples":
                # 如果资源不是 n_samples，则抛出数值错误
                raise ValueError(
                    "resource can only be 'n_samples' when max_resources='auto'"
                )
            # 否则根据输入数据 X 计算最大资源量
            self.max_resources_ = _num_samples(X)

        # 检查最小资源量是否大于最大资源量，若是则抛出数值错误
        if self.min_resources_ > self.max_resources_:
            raise ValueError(
                f"min_resources_={self.min_resources_} is greater "
                f"than max_resources_={self.max_resources}."
            )

        # 如果最小资源量为 0，则抛出数值错误
        if self.min_resources_ == 0:
            raise ValueError(
                f"min_resources_={self.min_resources_}: you might have passed "
                "an empty dataset X."
            )
    def _select_best_index(refit, refit_metric, results):
        """Custom refit callable to return the index of the best candidate.

        We want the best candidate out of the last iteration. By default
        BaseSearchCV would return the best candidate out of all iterations.

        Currently, we only support for a single metric thus `refit` and
        `refit_metric` are not required.
        """
        # 获取最后一次迭代的最大值
        last_iter = np.max(results["iter"])
        # 找到最后一次迭代中对应的索引
        last_iter_indices = np.flatnonzero(results["iter"] == last_iter)

        # 获取最后一次迭代中的测试分数
        test_scores = results["mean_test_score"][last_iter_indices]
        # 如果所有分数都是 NaN，则无法选择最佳，将第一个条目作为最佳条目
        if np.isnan(test_scores).all():
            best_idx = 0
        else:
            # 找到最大测试分数对应的索引
            best_idx = np.nanargmax(test_scores)

        return last_iter_indices[best_idx]

    @_fit_context(
        # Halving*SearchCV.estimator is not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(self, X, y=None, **params):
        """Run fit with all sets of parameters.

        Parameters
        ----------

        X : array-like, shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like, shape (n_samples,) or (n_samples, n_output), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        **params : dict of string -> object
            Parameters passed to the ``fit`` method of the estimator.

        Returns
        -------
        self : object
            Instance of fitted estimator.
        """
        # 检查交叉验证方法
        self._checked_cv_orig = check_cv(
            self.cv, y, classifier=is_classifier(self.estimator)
        )

        # 获取适用于拟合的参数
        routed_params = self._get_routed_params_for_fit(params)
        # 检查输入参数
        self._check_input_parameters(
            X=X, y=y, split_params=routed_params.splitter.split
        )

        # 记录原始样本数
        self._n_samples_orig = _num_samples(X)

        # 调用超类的拟合方法
        super().fit(X, y=y, **params)

        # 设置 best_score_: 因为 refit 是一个可调用对象，BaseSearchCV 不会设置它
        self.best_score_ = self.cv_results_["mean_test_score"][self.best_index_]

        return self

    @abstractmethod
    def _generate_candidate_params(self):
        pass

    def _more_tags(self):
        tags = deepcopy(super()._more_tags())
        tags["_xfail_checks"].update(
            {
                "check_fit2d_1sample": (
                    "Fail during parameter check since min/max resources requires"
                    " more samples"
                ),
            }
        )
        return tags
# 定义一个新的类 HalvingGridSearchCV，继承自 BaseSuccessiveHalving
class HalvingGridSearchCV(BaseSuccessiveHalving):
    """Search over specified parameter values with successive halving.

    The search strategy starts evaluating all the candidates with a small
    amount of resources and iteratively selects the best candidates, using
    more and more resources.

    Read more in the :ref:`User guide <successive_halving_user_guide>`.

    .. note::

      This estimator is still **experimental** for now: the predictions
      and the API might change without any deprecation cycle. To use it,
      you need to explicitly import ``enable_halving_search_cv``::

        >>> # explicitly require this experimental feature
        >>> from sklearn.experimental import enable_halving_search_cv # noqa
        >>> # now you can import normally from model_selection
        >>> from sklearn.model_selection import HalvingGridSearchCV

    Parameters
    ----------
    estimator : estimator object
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    param_grid : dict or list of dictionaries
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values, or a list of such
        dictionaries, in which case the grids spanned by each dictionary
        in the list are explored. This enables searching over any sequence
        of parameter settings.

    factor : int or float, default=3
        The 'halving' parameter, which determines the proportion of candidates
        that are selected for each subsequent iteration. For example,
        ``factor=3`` means that only one third of the candidates are selected.

    resource : ``'n_samples'`` or str, default='n_samples'
        Defines the resource that increases with each iteration. By default,
        the resource is the number of samples. It can also be set to any
        parameter of the base estimator that accepts positive integer
        values, e.g. 'n_iterations' or 'n_estimators' for a gradient
        boosting estimator. In this case ``max_resources`` cannot be 'auto'
        and must be set explicitly.

    max_resources : int, default='auto'
        The maximum amount of resource that any candidate is allowed to use
        for a given iteration. By default, this is set to ``n_samples`` when
        ``resource='n_samples'`` (default), else an error is raised.
    """
    # 定义最小资源量，可以是 {'exhaust', 'smallest'} 或整数，默认为 'exhaust'
    min_resources : {'exhaust', 'smallest'} or int, default='exhaust'
        The minimum amount of resource that any candidate is allowed to use
        for a given iteration. Equivalently, this defines the amount of
        resources `r0` that are allocated for each candidate at the first
        iteration.

        - 'smallest' is a heuristic that sets `r0` to a small value:

            - ``n_splits * 2`` when ``resource='n_samples'`` for a regression
              problem
            - ``n_classes * n_splits * 2`` when ``resource='n_samples'`` for a
              classification problem
            - ``1`` when ``resource != 'n_samples'``

        - 'exhaust' will set `r0` such that the **last** iteration uses as
          much resources as possible. Namely, the last iteration will use the
          highest value smaller than ``max_resources`` that is a multiple of
          both ``min_resources`` and ``factor``. In general, using 'exhaust'
          leads to a more accurate estimator, but is slightly more time
          consuming.

        Note that the amount of resources used at each iteration is always a
        multiple of ``min_resources``.

    # 是否启用激进的淘汰策略，布尔型，默认为 False
    aggressive_elimination : bool, default=False
        This is only relevant in cases where there isn't enough resources to
        reduce the remaining candidates to at most `factor` after the last
        iteration. If ``True``, then the search process will 'replay' the
        first iteration for as long as needed until the number of candidates
        is small enough. This is ``False`` by default, which means that the
        last iteration may evaluate more than ``factor`` candidates. See
        :ref:`aggressive_elimination` for more details.

    # 交叉验证的折数或生成器，默认为 5
    cv : int, cross-validation generator or iterable, default=5
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. note::
            Due to implementation details, the folds produced by `cv` must be
            the same across multiple calls to `cv.split()`. For
            built-in `scikit-learn` iterators, this can be achieved by
            deactivating shuffling (`shuffle=False`), or by setting the
            `cv`'s `random_state` parameter to an integer.
    # 控制评分方式的参数，可以是字符串或可调用对象，用于评估测试集的预测结果
    scoring : str, callable, or None, default=None
        A single string (see :ref:`scoring_parameter`) or a callable
        (see :ref:`scoring`) to evaluate the predictions on the test set.
        If None, the estimator's score method is used.

    # 是否在找到最佳参数后使用整个数据集重新训练估计器
    refit : bool, default=True
        If True, refit an estimator using the best found parameters on the
        whole dataset.

        The refitted estimator is made available at the ``best_estimator_``
        attribute and permits using ``predict`` directly on this
        ``HalvingGridSearchCV`` instance.

    # 当估计器拟合过程中出现错误时，分配给得分的值
    error_score : 'raise' or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error. Default is ``np.nan``.

    # 是否在 ``cv_results_`` 属性中包含训练分数
    return_train_score : bool, default=False
        If ``False``, the ``cv_results_`` attribute will not include training
        scores.
        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.

    # 用于子采样数据集的伪随机数生成器状态
    random_state : int, RandomState instance or None, default=None
        Pseudo random number generator state used for subsampling the dataset
        when `resources != 'n_samples'`. Ignored otherwise.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    # 并行运行的作业数量
    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    # 控制输出信息的详细程度，值越高输出的消息越多
    verbose : int
        Controls the verbosity: the higher, the more messages.

    # 以下是属性列表

    # 每次迭代中使用的资源数量列表
    Attributes
    ----------
    n_resources_ : list of int
        The amount of resources used at each iteration.

    # 每次迭代中评估的候选参数数量列表
    n_candidates_ : list of int
        The number of candidate parameters that were evaluated at each
        iteration.

    # 最后一次迭代后剩余的候选参数数量
    n_remaining_candidates_ : int
        The number of candidate parameters that are left after the last
        iteration. It corresponds to `ceil(n_candidates[-1] / factor)`

    # 每次迭代中任何候选者允许使用的最大资源数
    max_resources_ : int
        The maximum number of resources that any candidate is allowed to use
        for a given iteration. Note that since the number of resources used
        at each iteration must be a multiple of ``min_resources_``, the
        actual number of resources used at the last iteration may be smaller
        than ``max_resources_``.

    # 第一次迭代中为每个候选者分配的资源数量
    min_resources_ : int
        The amount of resources that are allocated for each candidate at the
        first iteration.
    n_iterations_ : int
        # 存储实际运行的迭代次数。如果 aggressive_elimination 为 True，则等于 n_required_iterations_。
        # 否则，等于 min(n_possible_iterations_, n_required_iterations_)。
        The actual number of iterations that were run. This is equal to
        ``n_required_iterations_`` if ``aggressive_elimination`` is ``True``.
        Else, this is equal to ``min(n_possible_iterations_,
        n_required_iterations_)``.

    n_possible_iterations_ : int
        # 存储使用 min_resources_ 资源但不超过 max_resources_ 资源时可能的最大迭代次数。
        The number of iterations that are possible starting with
        ``min_resources_`` resources and without exceeding
        ``max_resources_``.

    n_required_iterations_ : int
        # 存储在最后一次迭代时，希望保留少于 factor 个候选项所需的迭代次数。
        # 使用 min_resources_ 资源进行起始计算。当资源不足时，该值将小于 n_possible_iterations_。
        The number of iterations that are required to end up with less than
        ``factor`` candidates at the last iteration, starting with
        ``min_resources_`` resources. This will be smaller than
        ``n_possible_iterations_`` when there isn't enough resources.

    cv_results_ : dict of numpy (masked) ndarrays
        # 存储交叉验证结果的字典，可以导入到 pandas 的 DataFrame 中进行分析。
        # 包含大量分析搜索结果的信息。详细内容请参阅用户指南中的 successive_halving_cv_results 部分。
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``. It contains lots of information
        for analysing the results of a search.
        Please refer to the :ref:`User guide<successive_halving_cv_results>`
        for details.

    best_estimator_ : estimator or dict
        # 存储搜索过程中选择的最佳评估器（estimator）。
        # 如果 refit=False，则此项不可用。
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if ``refit=False``.

    best_score_ : float
        # 存储 best_estimator_ 的平均交叉验证得分。
        Mean cross-validated score of the best_estimator.

    best_params_ : dict
        # 存储在 hold out 数据上表现最佳的参数设置。
        Parameter setting that gave the best results on the hold out data.

    best_index_ : int
        # 存储对应于最佳候选参数设置的索引（在 cv_results_ 数组中）。
        # 可以通过 search.cv_results_['params'][search.best_index_] 获取最佳模型的参数设置，
        # 这个模型给出了最高的平均得分（search.best_score_）。
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.

        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).

    scorer_ : function or a dict
        # 存储用于在保留数据上选择最佳参数的评分函数。
        Scorer function used on the held out data to choose the best
        parameters for the model.

    n_splits_ : int
        # 存储交叉验证中的折数（分割数）。
        The number of cross-validation splits (folds/iterations).

    refit_time_ : float
        # 存储用于在整个数据集上重新拟合最佳模型所用的秒数。
        # 仅在 refit 不为 False 时出现。
        Seconds used for refitting the best model on the whole dataset.

        This is present only if ``refit`` is not False.

    multimetric_ : bool
        # 标记评分器是否计算多个指标。
        Whether or not the scorers compute several metrics.

    classes_ : ndarray of shape (n_classes,)
        # 存储类别标签。仅在指定 refit 且底层评估器为分类器时出现。
        The classes labels. This is present only if ``refit`` is specified and
        the underlying estimator is a classifier.

    n_features_in_ : int
        # 记录在拟合过程中观察到的特征数。
        # 仅在 best_estimator_ 已定义且 best_estimator_ 在拟合时公开了 n_features_in_ 时定义。
        # 更多信息请参阅 refit 参数的文档。
        Number of features seen during :term:`fit`. Only defined if
        `best_estimator_` is defined (see the documentation for the `refit`
        parameter for more details) and that `best_estimator_` exposes
        `n_features_in_` when fit.

        .. versionadded:: 0.24
    # `_required_parameters`是一个类级别的私有属性，定义了在初始化时必须提供的参数列表
    _required_parameters = ["estimator", "param_grid"]

    # `_parameter_constraints`是一个类级别的私有属性，定义了参数的约束条件，包括继承的基类约束和特定的`param_grid`类型约束
    _parameter_constraints: dict = {
        **BaseSuccessiveHalving._parameter_constraints,
        "param_grid": [dict, list],
    }

    # 初始化方法，用于创建一个`HalvingGridSearchCV`对象
    def __init__(
        self,
        estimator,
        param_grid,
        *,
        factor=3,
        resource="n_samples",
        max_resources="auto",
        min_resources="exhaust",
        aggressive_elimination=False,
        cv=5,
        scoring=None,
        refit=True,
        error_score=np.nan,
        return_train_score=True,
        random_state=None,
        n_jobs=None,
        verbose=0,
    ):
        # 调用父类的初始化方法，设置基本的超参数和评分相关的参数
        super().__init__(
            estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            verbose=verbose,
            cv=cv,
            random_state=random_state,
            error_score=error_score,
            return_train_score=return_train_score,
            max_resources=max_resources,
            resource=resource,
            factor=factor,
            min_resources=min_resources,
            aggressive_elimination=aggressive_elimination,
        )
        # 将传入的`param_grid`参数保存在对象的`param_grid`属性中
        self.param_grid = param_grid

    # 私有方法，用于生成候选参数的网格，基于传入的`param_grid`
    def _generate_candidate_params(self):
        return ParameterGrid(self.param_grid)
class HalvingRandomSearchCV(BaseSuccessiveHalving):
    """Randomized search on hyper parameters.

    The search strategy starts evaluating all the candidates with a small
    amount of resources and iteratively selects the best candidates, using more
    and more resources.

    The candidates are sampled at random from the parameter space and the
    number of sampled candidates is determined by ``n_candidates``.

    Read more in the :ref:`User guide<successive_halving_user_guide>`.

    .. note::

      This estimator is still **experimental** for now: the predictions
      and the API might change without any deprecation cycle. To use it,
      you need to explicitly import ``enable_halving_search_cv``::

        >>> # explicitly require this experimental feature
        >>> from sklearn.experimental import enable_halving_search_cv # noqa
        >>> # now you can import normally from model_selection
        >>> from sklearn.model_selection import HalvingRandomSearchCV

    Parameters
    ----------
    estimator : estimator object
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    param_distributions : dict or list of dicts
        Dictionary with parameters names (`str`) as keys and distributions
        or lists of parameters to try. Distributions must provide a ``rvs``
        method for sampling (such as those from scipy.stats.distributions).
        If a list is given, it is sampled uniformly.
        If a list of dicts is given, first a dict is sampled uniformly, and
        then a parameter is sampled using that dict as above.

    n_candidates : "exhaust" or int, default="exhaust"
        The number of candidate parameters to sample, at the first
        iteration. Using 'exhaust' will sample enough candidates so that the
        last iteration uses as many resources as possible, based on
        `min_resources`, `max_resources` and `factor`. In this case,
        `min_resources` cannot be 'exhaust'.

    factor : int or float, default=3
        The 'halving' parameter, which determines the proportion of candidates
        that are selected for each subsequent iteration. For example,
        ``factor=3`` means that only one third of the candidates are selected.

    resource : ``'n_samples'`` or str, default='n_samples'
        Defines the resource that increases with each iteration. By default,
        the resource is the number of samples. It can also be set to any
        parameter of the base estimator that accepts positive integer
        values, e.g. 'n_iterations' or 'n_estimators' for a gradient
        boosting estimator. In this case ``max_resources`` cannot be 'auto'
        and must be set explicitly.
    """
    # 最大资源限制：整数，默认为'auto'
    max_resources : int, default='auto'
        # 对于每一次迭代，候选者可以使用的最大资源数量限制。
        # 当 resource='n_samples'（默认）时，默认设置为 n_samples。
        # 否则会引发错误。

    # 最小资源限制：{'exhaust', 'smallest'} 或 整数，默认为'smallest'
    min_resources : {'exhaust', 'smallest'} or int, default='smallest'
        # 对于每一次迭代，每个候选者被允许使用的最小资源量。
        # 同时也定义了第一次迭代为每个候选者分配的资源量 r0。

        # 'smallest' 是一个启发式方法，将 r0 设置为一个较小的值：
        #   - 当 resource='n_samples' 时，用于回归问题的 n_splits * 2
        #   - 当 resource='n_samples' 时，用于分类问题的 n_classes * n_splits * 2
        #   - 当 resource != 'n_samples' 时，设置为 1

        # 'exhaust' 将设置 r0，以便最后一次迭代尽可能使用更多资源。
        # 特别地，最后一次迭代将使用小于 max_resources 的最高值，
        # 且该值是 min_resources 和 factor 的倍数。通常，使用'exhaust'会导致更精确的估计器，
        # 但会略微增加时间消耗。当 n_candidates='exhaust' 时，'exhaust' 不可用。

        # 注意：每次迭代使用的资源量始终是 min_resources 的倍数。

    # 强制淘汰：布尔值，默认为 False
    aggressive_elimination : bool, default=False
        # 仅在剩余的候选者无法在最后一次迭代后减少至最多因子个数时才相关。
        # 如果为 True，则搜索过程将“重播”第一次迭代，直到候选者数量足够小为止。
        # 默认为 False，这意味着最后一次迭代可能评估超过因子个数的候选者。
        # 更多详细信息请参阅 :ref:`aggressive_elimination`。
    # 定义交叉验证生成器或可迭代器，默认为5折交叉验证
    cv : int, cross-validation generator or an iterable, default=5
        确定交叉验证的分割策略。
        可选的输入方式包括：
        
        - 整数，用于指定`(Stratified)KFold`中的折数，
        - `CV splitter`，
        - 作为索引数组生成的(train, test)分割的可迭代对象。

        对于整数/None的输入，如果估计器是分类器且`y`是二进制或多类，则使用`StratifiedKFold`。对于其他情况，使用`KFold`。这些分割器会使用`shuffle=False`来确保调用时的分割方式相同。

        可参考 :ref:`User Guide <cross_validation>` 查看可以使用的各种交叉验证策略。

        .. note::
            由于实现细节，由`cv`产生的折叠在多次调用`cv.split()`时必须相同。对于内置的`scikit-learn`迭代器，可以通过禁用洗牌（`shuffle=False`）或将`cv`的`random_state`参数设置为整数来实现此目标。

    # 评分函数，用于评估在测试集上的预测表现，可以是字符串或可调用对象，默认为None
    scoring : str, callable, or None, default=None
        可以是单个字符串（参见 :ref:`scoring_parameter`）或可调用对象（参见 :ref:`scoring`），用于评估在测试集上的预测结果。
        如果为None，则使用估计器的score方法。

    # 是否在整个数据集上使用最佳参数重新拟合估计器，默认为True
    refit : bool, default=True
        如果为True，在整个数据集上使用找到的最佳参数重新拟合估计器。

        重新拟合后的估计器可以在`HalvingRandomSearchCV`实例上直接使用`predict`。

    # 如果在估计器拟合过程中发生错误，分配给分数的值。如果设置为'raise'，则引发错误；如果给出数字值，则引发FitFailedWarning。此参数不影响重新拟合步骤，该步骤始终会引发错误。默认为`np.nan`。
    error_score : 'raise' or numeric
        如果在估计器拟合过程中发生错误，则分配给分数的值。
        如果设置为'raise'，则引发错误。如果给出数字值，则引发FitFailedWarning。
        此参数不影响重新拟合步骤，该步骤始终会引发错误。默认为`np.nan`。

    # 如果为False，则`cv_results_`属性将不包含训练分数。
    # 计算训练分数用于洞察如何在不同参数设置下平衡过拟合/欠拟合的折中。但是，在训练集上计算分数可能会非常耗时，且不严格要求选择能够产生最佳泛化性能的参数。
    return_train_score : bool, default=False
        如果为False，则`cv_results_`属性将不包含训练分数。
        计算训练分数用于获取不同参数设置对过拟合/欠拟合权衡的影响。
        然而，在训练集上计算分数可能会非常昂贵，且不是严格必需的，以选择能够产生最佳泛化性能的参数。

    # 伪随机数生成器的状态，用于在`resources != 'n_samples'`时对数据集进行子采样，也用于从可能值列表中的随机均匀采样而不是scipy.stats分布。
    # 传递一个整数以确保在多次函数调用中输出是可重现的。参见 :term:`Glossary <random_state>`。
    random_state : int, RandomState instance or None, default=None
        用于在`resources != 'n_samples'`时对数据集进行子采样的伪随机数生成器状态。
        也用于从可能值列表中进行随机均匀采样而不是使用scipy.stats分布。
        传递一个整数以确保在多次函数调用中输出是可重现的。
        参见 :term:`Glossary <random_state>`。
    # 并行运行的作业数量，可以是整数或者None，默认为None
    n_jobs : int or None, default=None
        # 控制并行运行的作业数量，如果为None，则默认为1，除非在joblib.parallel_backend上下文中
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    # 控制输出的详细程度，数字越高，输出信息越多
    verbose : int
        Controls the verbosity: the higher, the more messages.

    # 每次迭代使用的资源量列表
    Attributes
    ----------
    n_resources_ : list of int
        The amount of resources used at each iteration.

    # 每次迭代评估的候选参数数量列表
    n_candidates_ : list of int
        The number of candidate parameters that were evaluated at each
        iteration.

    # 最后一次迭代后剩余的候选参数数量，计算方式为ceil(n_candidates[-1] / factor)
    n_remaining_candidates_ : int
        The number of candidate parameters that are left after the last
        iteration. It corresponds to `ceil(n_candidates[-1] / factor)`

    # 每次迭代中任何候选者允许使用的最大资源数
    max_resources_ : int
        The maximum number of resources that any candidate is allowed to use
        for a given iteration. Note that since the number of resources used at
        each iteration must be a multiple of ``min_resources_``, the actual
        number of resources used at the last iteration may be smaller than
        ``max_resources_``.

    # 第一次迭代为每个候选者分配的最小资源量
    min_resources_ : int
        The amount of resources that are allocated for each candidate at the
        first iteration.

    # 实际运行的迭代次数，如果`aggressive_elimination`为True，则等于`n_required_iterations_`。
    # 否则，等于`min(n_possible_iterations_, n_required_iterations_)`
    n_iterations_ : int
        The actual number of iterations that were run. This is equal to
        ``n_required_iterations_`` if ``aggressive_elimination`` is ``True``.
        Else, this is equal to ``min(n_possible_iterations_,
        n_required_iterations_)``.

    # 在不超过`max_resources_`的情况下，从`min_resources_`资源开始的可能迭代次数
    n_possible_iterations_ : int
        The number of iterations that are possible starting with
        ``min_resources_`` resources and without exceeding
        ``max_resources_``.

    # 从`min_resources_`资源开始，最后一个迭代结束时剩余少于`factor`候选者所需的迭代次数
    n_required_iterations_ : int
        The number of iterations that are required to end up with less than
        ``factor`` candidates at the last iteration, starting with
        ``min_resources_`` resources. This will be smaller than
        ``n_possible_iterations_`` when there isn't enough resources.

    # 交叉验证结果的字典，可以导入到pandas的DataFrame中进行分析
    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``. It contains lots of information
        for analysing the results of a search.
        Please refer to the :ref:`User guide<successive_halving_cv_results>`
        for details.

    # 搜索选择的最佳估算器，即在留出数据上给出最高分数（或最小损失）的估算器
    # 如果`refit=False`，则不可用
    best_estimator_ : estimator or dict
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if ``refit=False``.

    # `best_estimator_`的平均交叉验证分数
    best_score_ : float
        Mean cross-validated score of the best_estimator.

    # 在留出数据上给出最佳结果的参数设置
    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.
    best_index_ : int
        # `best_index_` 是一个整数，指示与最佳候选参数设置对应的 `cv_results_` 数组的索引。
        # 在 `search.cv_results_['params'][search.best_index_]` 字典中可以找到最佳模型的参数设置，
        # 它产生了最高的平均分数 (`search.best_score_`)。

    scorer_ : function or a dict
        # 用于在留出数据上选择最佳参数的评分函数。

    n_splits_ : int
        # 交叉验证的折数（fold/iteration）的数量。

    refit_time_ : float
        # 用于在整个数据集上重新拟合最佳模型所用的秒数。
        # 只有当 `refit` 不是 False 时才会出现。

    multimetric_ : bool
        # 表示评分器是否计算多个指标。

    classes_ : ndarray of shape (n_classes,)
        # 类别标签数组。只有在指定了 `refit` 且基础估计器是分类器时才会出现。

    n_features_in_ : int
        # 在拟合期间观察到的特征数量。只有当 `best_estimator_` 被定义并且 `best_estimator_`
        # 在拟合时公开了 `n_features_in_` 时才会定义。
        # 查看 `refit` 参数的文档以获取更多细节。

        .. versionadded:: 0.24
            # 添加版本说明：从版本 0.24 开始可用。

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        # 在拟合期间观察到的特征名称数组。只有当 `best_estimator_` 被定义并且 `best_estimator_`
        # 在拟合时公开了 `feature_names_in_` 时才会定义。
        # 查看 `refit` 参数的文档以获取更多细节。

        .. versionadded:: 1.0
            # 添加版本说明：从版本 1.0 开始可用。

    See Also
    --------
    :class:`HalvingGridSearchCV`:
        # 使用连续减半搜索网格参数。

    Notes
    -----
    # 所选的参数是根据评分参数最大化留出数据的分数来选择的。

    # 所有得分为 NaN 的参数组合将共享最低的排名。

    Examples
    --------

    >>> from sklearn.datasets import load_iris
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.experimental import enable_halving_search_cv  # noqa
    >>> from sklearn.model_selection import HalvingRandomSearchCV
    >>> from scipy.stats import randint
    >>> import numpy as np
    ...
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = RandomForestClassifier(random_state=0)
    >>> np.random.seed(0)
    ...
    >>> param_distributions = {"max_depth": [3, None],
    ...                        "min_samples_split": randint(2, 11)}
    >>> search = HalvingRandomSearchCV(clf, param_distributions,
    ...                                resource='n_estimators',
    ...                                max_resources=10,
    ...                                random_state=0).fit(X, y)
    >>> search.best_params_  # doctest: +SKIP
    {'max_depth': None, 'min_samples_split': 10, 'n_estimators': 9}
    """
    # 定义类变量，指定必需的参数列表
    _required_parameters = ["estimator", "param_distributions"]

    # 定义参数约束的字典，继承自BaseSuccessiveHalving类的参数约束，并添加新的约束条件
    _parameter_constraints: dict = {
        **BaseSuccessiveHalving._parameter_constraints,
        "param_distributions": [dict, list],  # param_distributions必须是字典或列表类型
        "n_candidates": [
            Interval(Integral, 0, None, closed="neither"),  # n_candidates必须是大于0的整数或"exhaust"
            StrOptions({"exhaust"}),  # 或者是字符串"exhaust"
        ],
    }

    # 初始化方法，接收多个参数并设置对象的初始状态
    def __init__(
        self,
        estimator,
        param_distributions,
        *,
        n_candidates="exhaust",  # 可选参数，控制候选参数生成的方式，默认为"exhaust"
        factor=3,  # 控制资源减少比例的因子，默认为3
        resource="n_samples",  # 指定用于资源评估的属性，默认为样本数量
        max_resources="auto",  # 最大资源数量，默认自动设置
        min_resources="smallest",  # 最小资源数量，默认为最小的可能值
        aggressive_elimination=False,  # 是否采用激进淘汰策略，默认关闭
        cv=5,  # 交叉验证折数，默认为5
        scoring=None,  # 评分方法，默认为None
        refit=True,  # 是否在训练后用最佳参数重新拟合，默认为True
        error_score=np.nan,  # 在评分过程中遇到错误时返回的值，默认为NaN
        return_train_score=True,  # 是否返回训练集上的得分，默认为True
        random_state=None,  # 随机数种子，默认为None
        n_jobs=None,  # 并行作业的数量，默认为None，即不并行
        verbose=0,  # 控制输出详细程度的参数，默认为0，即不输出额外信息
    ):
        # 调用父类的初始化方法，设置基础参数
        super().__init__(
            estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            verbose=verbose,
            cv=cv,
            random_state=random_state,
            error_score=error_score,
            return_train_score=return_train_score,
            max_resources=max_resources,
            resource=resource,
            factor=factor,
            min_resources=min_resources,
            aggressive_elimination=aggressive_elimination,
        )
        # 设置实例的param_distributions属性，用于存储参数分布信息
        self.param_distributions = param_distributions
        # 设置实例的n_candidates属性，用于控制候选参数的数量或方式
        self.n_candidates = n_candidates

    # 内部方法，生成候选参数集合
    def _generate_candidate_params(self):
        # 根据n_candidates的设置确定首次迭代的候选参数数量
        n_candidates_first_iter = self.n_candidates
        if n_candidates_first_iter == "exhaust":
            # 如果n_candidates为"exhaust"，根据资源的最大和最小设置生成尽可能多的候选参数
            n_candidates_first_iter = self.max_resources_ // self.min_resources_
        # 使用ParameterSampler生成候选参数，返回生成器对象
        return ParameterSampler(
            self.param_distributions,
            n_candidates_first_iter,
            random_state=self.random_state,
        )
```