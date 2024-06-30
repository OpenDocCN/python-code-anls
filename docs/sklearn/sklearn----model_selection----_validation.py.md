# `D:\src\scipysrc\scikit-learn\sklearn\model_selection\_validation.py`

```
def _check_params_groups_deprecation(fit_params, params, groups, version):
    """A helper function to check deprecations on `groups` and `fit_params`.
    
    This function checks the usage of `fit_params` and `params` and issues warnings or errors based on their presence and deprecation status.

    Parameters:
    - fit_params : dict or None
        Dictionary of parameters passed to the `fit` method.
    - params : dict or None
        Dictionary of parameters.
    - groups : object or None
        Group labels for the samples used while splitting the data into train/test set.
    - version : str
        Version number indicating when the deprecation will take effect.

    Returns:
    - params : dict
        Updated dictionary of parameters after handling deprecations.

    Raises:
    - ValueError:
        If both `params` and `fit_params` are provided simultaneously.
        If `groups` are provided when metadata routing is enabled.

    Notes:
    - Deprecation warning is issued if `fit_params` are used instead of `params`.
    - `fit_params` will be removed in future versions in favor of `params`.
    """
    if params is not None and fit_params is not None:
        raise ValueError(
            "`params` and `fit_params` cannot both be provided. Pass parameters "
            "via `params`. `fit_params` is deprecated and will be removed in "
            f"version {version}."
        )
    elif fit_params is not None:
        warnings.warn(
            (
                "`fit_params` is deprecated and will be removed in version {version}. "
                "Pass parameters via `params` instead."
            ),
            FutureWarning,
        )
        params = fit_params

    params = {} if params is None else params

    if groups is not None and _routing_enabled():
        raise ValueError(
            "`groups` can only be passed if metadata routing is not enabled via"
            " `sklearn.set_config(enable_metadata_routing=True)`. When routing is"
            " enabled, pass `groups` alongside other metadata via the `params` argument"
            " instead."
        )

    return params
    {
        "estimator": [HasMethods("fit")],  
        # 要求 "estimator" 属性必须具有 "fit" 方法
    
        "X": ["array-like", "sparse matrix"],  
        # "X" 应为类似数组或稀疏矩阵类型
    
        "y": ["array-like", None],  
        # "y" 应为类似数组类型或可以为空
    
        "groups": ["array-like", None],  
        # "groups" 应为类似数组类型或可以为空
    
        "scoring": [
            StrOptions(set(get_scorer_names())),  
            # "scoring" 应为字符串选项集合，其中选项来自于可用的评分器名称集合
    
            callable,  
            # "scoring" 可以是可调用对象
    
            list,  
            # "scoring" 可以是列表类型
    
            tuple,  
            # "scoring" 可以是元组类型
    
            dict,  
            # "scoring" 可以是字典类型
    
            None,  
            # "scoring" 可以为空
        ],
    
        "cv": ["cv_object"],  
        # "cv" 应为交叉验证对象类型
    
        "n_jobs": [Integral, None],  
        # "n_jobs" 应为整数或可以为空
    
        "verbose": ["verbose"],  
        # "verbose" 应为详细程度设置
    
        "fit_params": [dict, None],  
        # "fit_params" 应为字典类型或可以为空
    
        "params": [dict, None],  
        # "params" 应为字典类型或可以为空
    
        "pre_dispatch": [Integral, str],  
        # "pre_dispatch" 应为整数或字符串类型
    
        "return_train_score": ["boolean"],  
        # "return_train_score" 应为布尔值
    
        "return_estimator": ["boolean"],  
        # "return_estimator" 应为布尔值
    
        "return_indices": ["boolean"],  
        # "return_indices" 应为布尔值
    
        "error_score": [StrOptions({"raise"}), Real],  
        # "error_score" 应为字符串选项集合 {"raise"} 或实数类型
    },
    prefer_skip_nested_validation=False,  
    # prefer_skip_nested_validation 设置为 False，表示尚未验证估算器
)
def cross_validate(
    estimator,
    X,
    y=None,
    *,
    groups=None,
    scoring=None,
    cv=None,
    n_jobs=None,
    verbose=0,
    fit_params=None,
    params=None,
    pre_dispatch="2*n_jobs",
    return_train_score=False,
    return_estimator=False,
    return_indices=False,
    error_score=np.nan,
):
    """Evaluate metric(s) by cross-validation and also record fit/score times.

    Read more in the :ref:`User Guide <multimetric_cross_validation>`.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data to fit. Can be for example a list, or an array.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
        The target variable to try to predict in the case of
        supervised learning.

    groups : array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting the dataset into
        train/test set. Only used in conjunction with a "Group" :term:`cv`
        instance (e.g., :class:`GroupKFold`).

        .. versionchanged:: 1.4
            ``groups`` can only be passed if metadata routing is not enabled
            via ``sklearn.set_config(enable_metadata_routing=True)``. When routing
            is enabled, pass ``groups`` alongside other metadata via the ``params``
            argument instead. E.g.:
            ``cross_validate(..., params={'groups': groups})``.

    scoring : str, callable, list, tuple, or dict, default=None
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

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross-validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int, default=None
        Number of jobs to run in parallel. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    verbose : int, default=0
        The verbosity level.

    fit_params : dict, default=None
        Parameters to pass to the `estimator`'s `fit` method.

    params : dict, default=None
        Parameters to pass to the `estimator` and the `scorer`.

    pre_dispatch : int, or str, default="2*n_jobs"
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

        - None, in which case all the jobs are immediately created and
          spawned. Use this for lightweight and fast-running jobs, to
          avoid delays due to on-demand spawning of the jobs
        - An int, giving the exact number of total jobs that are
          spawned
        - A str, giving an expression as a function of n_jobs, as in
          '2*n_jobs'

    return_train_score : bool, default=False
        If `False`, the `cv_results_` attribute will not include training
        scores.

    return_estimator : bool, default=False
        Whether to return the fitted estimators. If False, the estimators
        will be refitted on the whole dataset after fitting on the folds.

    return_indices : bool, default=False
        Whether to return the indices that can be used to reconstruct
        the training and testing sets for each fold.

    error_score : "raise" or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to "raise", the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.

    Returns
    -------
    scores : dict of float arrays of shape (n_splits,)
        Array of scores of the estimator for each run of the cross validation.

    fit_time : array of shape (n_splits,)
        Array of time taken for the fitting of each cross-validated estimator.

    score_time : array of shape (n_splits,)
        Array of time taken for the scoring of each cross-validated estimator.
    """
    cv : int, cross-validation generator or an iterable, default=None
        # 参数 cv 控制交叉验证的生成器或可迭代对象，默认为 None
        Determines the cross-validation splitting strategy.
        # 确定交叉验证的分割策略

        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        # 如果为 None，则使用默认的 5 折交叉验证

        - int, to specify the number of folds in a `(Stratified)KFold`,
        # 如果为整数，则指定在 `(Stratified)KFold` 中的折数

        - :term:`CV splitter`,
        # CV 分割器的术语

        - An iterable yielding (train, test) splits as arrays of indices.
        # 一个可迭代对象，生成 (train, test) 拆分的索引数组

        For int/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.
        # 对于 int/None 输入，如果估计器是分类器且 ``y`` 是二元或多类，则使用 :class:`StratifiedKFold`。
        # 在所有其他情况下，使用 :class:`KFold`。这些分割器实例化时 `shuffle=False`，因此跨调用的分割将保持相同。

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.
        # 参考 :ref:`User Guide <cross_validation>` 查看可以在此处使用的各种交叉验证策略。

        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.
            # 如果为 None，默认值从 3 折改为 5 折交叉验证。

    n_jobs : int, default=None
        # 参数 n_jobs 指定并行运行的作业数，默认为 None
        Number of jobs to run in parallel. Training the estimator and computing
        the score are parallelized over the cross-validation splits.
        # 并行运行的作业数。估计器的训练和得分计算在交叉验证拆分上并行化。

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        # 如果为 ``None``，除非在 :obj:`joblib.parallel_backend` 上下文中，否则默认为 1。

        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
        # ``-1`` 表示使用所有处理器。详细信息请参阅 :term:`Glossary <n_jobs>`。

    verbose : int, default=0
        # 参数 verbose 控制详细程度，默认为 0
        The verbosity level.
        # 详细程度级别。

    fit_params : dict, default=None
        # 参数 fit_params 指定传递给估计器的 fit 方法的参数，默认为 None
        Parameters to pass to the fit method of the estimator.
        # 传递给估计器的 fit 方法的参数。

        .. deprecated:: 1.4
            This parameter is deprecated and will be removed in version 1.6. Use
            ``params`` instead.
            # 此参数已被弃用，并将在版本 1.6 中移除。请改用 ``params``。

    params : dict, default=None
        # 参数 params 指定传递给底层估计器的 ``fit``、得分器和 CV 分割器的参数，默认为 None
        Parameters to pass to the underlying estimator's ``fit``, the scorer,
        and the CV splitter.
        # 传递给底层估计器的 ``fit``、得分器和 CV 分割器的参数。

        .. versionadded:: 1.4

    pre_dispatch : int or str, default='2*n_jobs'
        # 参数 pre_dispatch 控制并行执行时分派的作业数，默认为 '2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process.
        # 控制并行执行时分派的作业数。减少此数值可以避免内存消耗的激增，当分派的作业超过处理器能力时特别有用。

        This parameter can be:

            - An int, giving the exact number of total jobs that are
              spawned
            # 整数，指定生成的总作业数

            - A str, giving an expression as a function of n_jobs,
              as in '2*n_jobs'
            # 字符串，表达为作为 n_jobs 的函数，例如 '2*n_jobs'

    return_train_score : bool, default=False
        # 参数 return_train_score 控制是否包括训练分数，默认为 False
        Whether to include train scores.
        # 是否包括训练分数。

        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.
        # 计算训练分数用于了解不同参数设置对过拟合/欠拟合权衡的影响。
        # 然而，在训练集上计算分数可能会很耗时，且在严格意义上并不需要选择产生最佳泛化性能的参数。

        .. versionadded:: 0.19

        .. versionchanged:: 0.21
            Default value was changed from ``True`` to ``False``
            # 默认值从 ``True`` 改为 ``False``

    return_estimator : bool, default=False
        # 参数 return_estimator 控制是否返回每个拆分上拟合的估计器，默认为 False
        Whether to return the estimators fitted on each split.
        # 是否返回每个拆分上拟合的估计器。

        .. versionadded:: 0.20
    return_indices : bool, default=False
        是否返回每次拆分所选的训练-测试索引。

        .. versionadded:: 1.3
            版本新增功能说明

    error_score : 'raise' or numeric, default=np.nan
        如果在估算器拟合过程中发生错误，则为得分指定的值。
        如果设置为'raise'，则会引发错误。
        如果给出了数值，则会引发FitFailedWarning。

        .. versionadded:: 0.20
            版本新增功能说明

    Returns
    -------
    scores : dict of float arrays of shape (n_splits,)
        估算器每次交叉验证运行的得分数组。

        返回一个包含每个评分器的得分/时间数组的字典。此字典的可能键为：

            ``test_score``
                每个交叉验证拆分的测试得分数组。
                如果在scoring参数中有多个评分指标，则``test_score``中的``_score``后缀会更改为特定指标，如``test_r2``或``test_auc``。
            ``train_score``
                每个交叉验证拆分的训练得分数组。
                如果在scoring参数中有多个评分指标，则``train_score``中的``_score``后缀会更改为特定指标，如``train_r2``或``train_auc``。
                仅在``return_train_score=True``时才可用。
            ``fit_time``
                拟合估算器在每个交叉验证拆分的训练集上所用时间。
            ``score_time``
                在每个交叉验证拆分的测试集上为估算器评分所用时间。
                （注意，即使``return_train_score=True``，也不包括在训练集上评分的时间）
            ``estimator``
                每个交叉验证拆分的估算器对象。
                仅当``return_estimator=True``时才可用。
            ``indices``
                每个交叉验证拆分的训练/测试位置索引。返回一个字典，其中键为`"train"`或`"test"`，对应的值是一个整数类型的NumPy数组列表。
                仅在`return_indices=True`时可用。

    See Also
    --------
    cross_val_score : 运行单一指标评估的交叉验证。

    cross_val_predict : 用于诊断目的，获取交叉验证每个拆分的预测结果。

    sklearn.metrics.make_scorer : 从性能指标或损失函数创建评分器。

    Examples
    --------
    >>> from sklearn import datasets, linear_model
    >>> from sklearn.model_selection import cross_validate
    >>> from sklearn.metrics import make_scorer
    >>> from sklearn.metrics import confusion_matrix
    >>> from sklearn.svm import LinearSVC
    >>> diabetes = datasets.load_diabetes()
    >>> X = diabetes.data[:150]
    >>> y = diabetes.target[:150]
    >>> lasso = linear_model.Lasso()


    # 创建特征矩阵 X，包含 diabetes 数据集的前 150 行数据
    X = diabetes.data[:150]
    # 创建目标向量 y，包含 diabetes 数据集的前 150 行目标值
    y = diabetes.target[:150]
    # 初始化 Lasso 回归模型
    lasso = linear_model.Lasso()


    Single metric evaluation using ``cross_validate``

    # 使用 ``cross_validate`` 进行单个评估指标的评估
    >>> cv_results = cross_validate(lasso, X, y, cv=3)
    # 输出评估结果中的键列表，包括拟合时间、评分时间和测试分数
    >>> sorted(cv_results.keys())
    ['fit_time', 'score_time', 'test_score']
    # 输出评估结果中的测试分数数组
    >>> cv_results['test_score']
    array([0.3315057 , 0.08022103, 0.03531816])


    Multiple metric evaluation using ``cross_validate``
    (please refer the ``scoring`` parameter doc for more information)

    # 使用 ``cross_validate`` 进行多个评估指标的评估，需要参考 ``scoring`` 参数文档获取更多信息
    >>> scores = cross_validate(lasso, X, y, cv=3,
    ...                         scoring=('r2', 'neg_mean_squared_error'),
    ...                         return_train_score=True)
    # 打印出测试集上的负均方误差
    >>> print(scores['test_neg_mean_squared_error'])
    [-3635.5... -3573.3... -6114.7...]
    # 打印出训练集上的 R^2 分数
    >>> print(scores['train_r2'])
    [0.28009951 0.3908844  0.22784907]
    """
    # 检查参数组和组参数的过时提示
    params = _check_params_groups_deprecation(fit_params, params, groups, "1.6")


    X, y = indexable(X, y)

    # 检查交叉验证生成器，根据标签 y 是否为分类器来确定分类器类型
    cv = check_cv(cv, y, classifier=is_classifier(estimator))

    # 检查评分方法，根据评分参数和错误分数是否为 "raise" 来确定是否抛出异常
    scorers = check_scoring(
        estimator, scoring=scoring, raise_exc=(error_score == "raise")
    )
    if _routing_enabled():
        # 如果启用了路由功能，则创建 MetadataRouter 对象来处理元数据路由
        # 在 `get_metadata_routing` 方法中为评估器创建 MetadataRouter 对象。
        # 对于这些路由方法，我们创建路由器以在其上使用 `process_routing`。
        router = (
            MetadataRouter(owner="cross_validate")
            .add(
                splitter=cv,
                method_mapping=MethodMapping().add(caller="fit", callee="split"),
            )
            .add(
                estimator=estimator,
                # TODO(SLEP6): 是否也将元数据传递给预测方法以进行评分？
                method_mapping=MethodMapping().add(caller="fit", callee="fit"),
            )
            .add(
                scorer=scorers,
                method_mapping=MethodMapping().add(caller="fit", callee="score"),
            )
        )
        try:
            # 使用 `process_routing` 处理路由器对象，调用 `fit` 方法并传入参数 `params`
            routed_params = process_routing(router, "fit", **params)
        except UnsetMetadataPassedError as e:
            # 默认的异常消息会提及 `fit`，因为在上面的 `process_routing` 代码中，我们将 `fit` 作为调用者传递。
            # 然而，用户并没有直接调用 `fit`，因此我们改变消息以更适合此情况。
            unrequested_params = sorted(e.unrequested_params)
            raise UnsetMetadataPassedError(
                message=(
                    f"{unrequested_params} are passed to cross validation but are not"
                    " explicitly set as requested or not requested for cross_validate's"
                    f" estimator: {estimator.__class__.__name__}. Call"
                    " `.set_fit_request({{metadata}}=True)` on the estimator for"
                    f" each metadata in {unrequested_params} that you"
                    " want to use and `metadata=False` for not using it. See the"
                    " Metadata Routing User guide"
                    " <https://scikit-learn.org/stable/metadata_routing.html> for more"
                    " information."
                ),
                unrequested_params=e.unrequested_params,
                routed_params=e.routed_params,
            )
    else:
        # 如果未启用路由功能，则创建一个空的 Bunch 对象作为 routed_params
        routed_params = Bunch()
        routed_params.splitter = Bunch(split={"groups": groups})
        routed_params.estimator = Bunch(fit=params)
        routed_params.scorer = Bunch(score={})

    # 使用交叉验证对象 `cv` 的 `split` 方法拆分数据集 `X` 和标签 `y`，根据 `routed_params.splitter.split` 参数
    indices = cv.split(X, y, **routed_params.splitter.split)
    if return_indices:
        # 如果需要返回拆分的索引，则将生成器转换为列表
        indices = list(indices)

    # 克隆评估器以确保所有折叠是独立的，并且它可以被序列化
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)
    # 并行执行模型拟合和评分过程，对每个数据集进行拟合和评分
    results = parallel(
        delayed(_fit_and_score)(
            clone(estimator),  # 克隆估算器对象
            X,  # 特征数据
            y,  # 目标数据
            scorer=scorers,  # 评分器
            train=train,  # 训练数据集的索引
            test=test,  # 测试数据集的索引
            verbose=verbose,
            parameters=None,  # 拟合参数为None，表示使用默认参数
            fit_params=routed_params.estimator.fit,  # 拟合参数，从路由参数中获取
            score_params=routed_params.scorer.score,  # 评分参数，从路由参数中获取
            return_train_score=return_train_score,  # 是否返回训练集分数
            return_times=True,  # 是否返回拟合和评分时间
            return_estimator=return_estimator,  # 是否返回估算器对象
            error_score=error_score,  # 错误得分
        )
        for train, test in indices  # 遍历交叉验证的训练集和测试集索引
    )

    # 警告或抛出关于拟合失败的警告
    _warn_or_raise_about_fit_failures(results, error_score)

    # 对于可调用的评分函数，只有调用后才知道返回类型。如果返回类型是字典，
    # 现在可以插入正确的错误得分。
    if callable(scoring):
        _insert_error_scores(results, error_score)

    # 聚合所有评分结果字典
    results = _aggregate_score_dicts(results)

    # 构造返回结果字典
    ret = {}
    ret["fit_time"] = results["fit_time"]  # 拟合时间
    ret["score_time"] = results["score_time"]  # 评分时间

    if return_estimator:
        ret["estimator"] = results["estimator"]  # 如果需要返回估算器对象，则加入结果字典中

    if return_indices:
        # 如果需要返回索引信息，则构造包含训练集和测试集索引的字典
        ret["indices"] = {}
        ret["indices"]["train"], ret["indices"]["test"] = zip(*indices)

    # 标准化测试集的评分结果字典
    test_scores_dict = _normalize_score_results(results["test_scores"])
    if return_train_score:
        # 如果需要返回训练集的评分结果，则也标准化训练集的评分结果字典
        train_scores_dict = _normalize_score_results(results["train_scores"])

    # 将标准化后的测试集评分结果加入返回结果字典中
    for name in test_scores_dict:
        ret["test_%s" % name] = test_scores_dict[name]
        if return_train_score:
            # 如果需要返回训练集的评分结果，则也加入到返回结果字典中
            key = "train_%s" % name
            ret[key] = train_scores_dict[name]

    # 返回最终构造的结果字典
    return ret
# 将错误分数 `error_score` 插入到 `results` 中，替换掉原有的分数，仅适用于多指标评分，因为 `_fit_and_score` 会处理单指标情况。
def _insert_error_scores(results, error_score):
    # 初始化成功的评分为 None
    successful_score = None
    # 存储失败的索引列表
    failed_indices = []
    
    # 遍历结果列表，检查每个结果
    for i, result in enumerate(results):
        # 如果有适合错误，则记录失败的索引
        if result["fit_error"] is not None:
            failed_indices.append(i)
        # 如果成功的评分还未设置，则使用当前结果的测试评分作为成功的评分
        elif successful_score is None:
            successful_score = result["test_scores"]
    
    # 如果成功的评分是字典类型，则格式化错误分数
    if isinstance(successful_score, dict):
        formatted_error = {name: error_score for name in successful_score}
        # 对于每个失败的索引，将测试分数和训练分数（如果存在）设置为格式化的错误分数副本
        for i in failed_indices:
            results[i]["test_scores"] = formatted_error.copy()
            if "train_scores" in results[i]:
                results[i]["train_scores"] = formatted_error.copy()


# 根据 `scores` 类型创建评分字典
def _normalize_score_results(scores, scaler_score_key="score"):
    if isinstance(scores[0], dict):
        # 如果是多指标评分，则聚合评分字典
        return _aggregate_score_dicts(scores)
    else:
        # 否则，返回单一评分字典，使用默认的缩放器分数键名
        return {scaler_score_key: scores}


# 警告或引发关于适合失败的警告信息
def _warn_or_raise_about_fit_failures(results, error_score):
    # 收集所有适合错误
    fit_errors = [result["fit_error"] for result in results if result["fit_error"] is not None]
    
    if fit_errors:
        # 统计失败的适合次数及总次数
        num_failed_fits = len(fit_errors)
        num_fits = len(results)
        # 统计不同错误的次数
        fit_errors_counter = Counter(fit_errors)
        # 分隔线
        delimiter = "-" * 80 + "\n"
        # 错误摘要信息，包括错误及其次数
        fit_errors_summary = "\n".join(
            f"{delimiter}{n} fits failed with the following error:\n{error}"
            for error, n in fit_errors_counter.items()
        )

        if num_failed_fits == num_fits:
            # 如果所有适合都失败了，引发值错误，并提供详细错误信息摘要
            all_fits_failed_message = (
                f"\nAll the {num_fits} fits failed.\n"
                "It is very likely that your model is misconfigured.\n"
                "You can try to debug the error by setting error_score='raise'.\n\n"
                f"Below are more details about the failures:\n{fit_errors_summary}"
            )
            raise ValueError(all_fits_failed_message)

        else:
            # 如果只有部分适合失败，发出警告，并提供详细错误信息摘要
            some_fits_failed_message = (
                f"\n{num_failed_fits} fits failed out of a total of {num_fits}.\n"
                "The score on these train-test partitions for these parameters"
                f" will be set to {error_score}.\n"
                "If these failures are not expected, you can try to debug them "
                "by setting error_score='raise'.\n\n"
                f"Below are more details about the failures:\n{fit_errors_summary}"
            )
            warnings.warn(some_fits_failed_message, FitFailedWarning)
    {
        # estimator: 期望一个具有 'fit' 方法的对象，通常是一个机器学习模型
        "estimator": [HasMethods("fit")],
        # X: 输入数据，可以是数组或稀疏矩阵
        "X": ["array-like", "sparse matrix"],
        # y: 目标数据，可以是数组，如果不适用于模型，则为 None
        "y": ["array-like", None],
        # groups: 分组数据，可以是数组，如果不适用于模型，则为 None
        "groups": ["array-like", None],
        # scoring: 评分方法，可以是预定义的字符串选项集合或可调用对象，或者为 None
        "scoring": [StrOptions(set(get_scorer_names())), callable, None],
        # cv: 交叉验证对象
        "cv": ["cv_object"],
        # n_jobs: 并行运行的作业数，可以是整数或 None
        "n_jobs": [Integral, None],
        # verbose: 输出详细程度，可以是一个整数
        "verbose": ["verbose"],
        # fit_params: fit 方法的附加参数，期望一个字典，如果不使用则为 None
        "fit_params": [dict, None],
        # params: 模型的参数，期望一个字典，如果没有特定参数则为 None
        "params": [dict, None],
        # pre_dispatch: 预调度的作业数，可以是整数、字符串或 None
        "pre_dispatch": [Integral, str, None],
        # error_score: 发生错误时的处理方式，期望一个预定义字符串选项或实数
        "error_score": [StrOptions({"raise"}), Real],
    },
    # prefer_skip_nested_validation: 是否优先跳过嵌套验证，默认为 False，表示尚未验证 estimator
    prefer_skip_nested_validation=False,
)
def cross_val_score(
    estimator,
    X,
    y=None,
    *,
    groups=None,
    scoring=None,
    cv=None,
    n_jobs=None,
    verbose=0,
    fit_params=None,
    params=None,
    pre_dispatch="2*n_jobs",
    error_score=np.nan,
):
    """Evaluate a score by cross-validation.

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data to fit. Can be for example a list, or an array.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs), \
            default=None
        The target variable to try to predict in the case of
        supervised learning.

    groups : array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting the dataset into
        train/test set. Only used in conjunction with a "Group" :term:`cv`
        instance (e.g., :class:`GroupKFold`).

        .. versionchanged:: 1.4
            ``groups`` can only be passed if metadata routing is not enabled
            via ``sklearn.set_config(enable_metadata_routing=True)``. When routing
            is enabled, pass ``groups`` alongside other metadata via the ``params``
            argument instead. E.g.:
            ``cross_val_score(..., params={'groups': groups})``.

    scoring : str or callable, default=None
        A str (see :ref:`scoring_parameter`) or a scorer callable object / function with
        signature ``scorer(estimator, X, y)`` which should return only a single value.

        Similar to :func:`cross_validate`
        but only a single metric is permitted.

        If `None`, the estimator's default scorer (if available) is used.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - `None`, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable that generates (train, test) splits as arrays of indices.

        For `int`/`None` inputs, if the estimator is a classifier and `y` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.22
            `cv` default value if `None` changed from 3-fold to 5-fold.
    # 设置并行运行的作业数，用于交叉验证过程中并行训练估算器和计算得分。
    # 如果未在 joblib.parallel_backend 上下文中，则默认为 1。
    # 如果设为 -1，则使用所有处理器。详见术语表中的 "n_jobs"。
    n_jobs : int, default=None

    # 决定是否输出详细信息的级别。
    verbose : int, default=0

    # 将参数传递给估算器的 fit 方法。
    # 
    # .. deprecated:: 1.4
    #     该参数已弃用，并将在 1.6 版本中移除。请改用 "params"。
    fit_params : dict, default=None

    # 将参数传递给底层估算器的 fit 方法、评分器和 CV 分割器。
    # 
    # .. versionadded:: 1.4
    params : dict, default=None

    # 控制并行执行过程中分派的作业数。
    # 减少此数字可避免在分派的作业超过 CPU 处理能力时内存消耗激增。
    # 可选值包括：
    # 
    # - ``None``：此时所有作业立即创建并启动。适用于轻量且运行速度快的作业，
    #   可避免由于按需启动作业而导致的延迟。
    # 
    # - 整数：指定生成的总作业数。
    # 
    # - 字符串：作为 n_jobs 函数的表达式，如 '2*n_jobs'。
    pre_dispatch : int or str, default='2*n_jobs'

    # 如果在估算器拟合过程中发生错误，用于分配给分数的值。
    # 如果设为 'raise'，则会抛出错误。
    # 如果指定为数值，则会引发 FitFailedWarning。
    # 
    # .. versionadded:: 0.20
    error_score : 'raise' or numeric, default=np.nan

Returns
-------
scores : ndarray of float of shape=(len(list(cv)),)
    # 交叉验证每次运行的估算器得分数组。

See Also
--------
cross_validate : 用于多个度量运行交叉验证，并返回训练分数、拟合时间和评分时间。
    
cross_val_predict : 用于诊断目的获取交叉验证每次分割的预测结果。
    
sklearn.metrics.make_scorer : 从性能度量或损失函数创建评分器。

Examples
--------
>>> from sklearn import datasets, linear_model
>>> from sklearn.model_selection import cross_val_score
>>> diabetes = datasets.load_diabetes()
>>> X = diabetes.data[:150]
>>> y = diabetes.target[:150]
>>> lasso = linear_model.Lasso()
>>> print(cross_val_score(lasso, X, y, cv=3))
[0.3315057  0.08022103 0.03531816]
"""
# 确保不支持多指标格式
scorer = check_scoring(estimator, scoring=scoring)
    # 使用交叉验证评估器对模型进行评估
    cv_results = cross_validate(
        estimator=estimator,  # 指定要评估的模型估计器
        X=X,  # 特征数据
        y=y,  # 目标数据
        groups=groups,  # 样本组标识，用于分组交叉验证
        scoring={"score": scorer},  # 指定评分方法，这里是一个字典，包含要计算的分数类型
        cv=cv,  # 指定交叉验证的折数或生成器
        n_jobs=n_jobs,  # 指定并行运行的作业数量
        verbose=verbose,  # 控制输出详细程度的标志
        fit_params=fit_params,  # 传递给拟合方法的参数
        params=params,  # 模型的超参数
        pre_dispatch=pre_dispatch,  # 控制分发作业之前的批量数量或使用的批量数
        error_score=error_score,  # 在评估期间遇到错误时使用的替代分数
    )
    # 返回交叉验证结果中的测试分数
    return cv_results["test_score"]
# 定义了一个函数 _fit_and_score，用于拟合评估器并计算给定数据集拆分的分数。

def _fit_and_score(
    estimator,
    X,
    y,
    *,
    scorer,
    train,
    test,
    verbose,
    parameters,
    fit_params,
    score_params,
    return_train_score=False,
    return_parameters=False,
    return_n_test_samples=False,
    return_times=False,
    return_estimator=False,
    split_progress=None,
    candidate_progress=None,
    error_score=np.nan,
):
    """Fit estimator and compute scores for a given dataset split.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like of shape (n_samples, n_features)
        The data to fit.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs) or None
        The target variable to try to predict in the case of
        supervised learning.

    scorer : A single callable or dict mapping scorer name to the callable
        If it is a single callable, the return value for ``train_scores`` and
        ``test_scores`` is a single float.

        For a dict, it should be one mapping the scorer name to the scorer
        callable object / function.

        The callable object / fn should have signature
        ``scorer(estimator, X, y)``.

    train : array-like of shape (n_train_samples,)
        Indices of training samples.

    test : array-like of shape (n_test_samples,)
        Indices of test samples.

    verbose : int
        The verbosity level.

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised.
        If a numeric value is given, FitFailedWarning is raised.

    parameters : dict or None
        Parameters to be set on the estimator.

    fit_params : dict or None
        Parameters that will be passed to ``estimator.fit``.

    score_params : dict or None
        Parameters that will be passed to the scorer.

    return_train_score : bool, default=False
        Compute and return score on training set.

    return_parameters : bool, default=False
        Return parameters that has been used for the estimator.

    split_progress : {list, tuple} of int, default=None
        A list or tuple of format (<current_split_id>, <total_num_of_splits>).

    candidate_progress : {list, tuple} of int, default=None
        A list or tuple of format
        (<current_candidate_id>, <total_number_of_candidates>).

    return_n_test_samples : bool, default=False
        Whether to return the ``n_test_samples``.

    return_times : bool, default=False
        Whether to return the fit/score times.

    return_estimator : bool, default=False
        Whether to return the fitted estimator.

    Returns
    -------
    ```
    result : dict with the following attributes
        train_scores : dict of scorer name -> float
            Score on training set (for all the scorers),
            returned only if `return_train_score` is `True`.
        test_scores : dict of scorer name -> float
            Score on testing set (for all the scorers).
        n_test_samples : int
            Number of test samples.
        fit_time : float
            Time spent for fitting in seconds.
        score_time : float
            Time spent for scoring in seconds.
        parameters : dict or None
            The parameters that have been evaluated.
        estimator : estimator object
            The fitted estimator.
        fit_error : str or None
            Traceback str if the fit failed, None if the fit succeeded.
    """
    xp, _ = get_namespace(X)
    X_device = device(X)

    # Make sure that we can fancy index X even if train and test are provided
    # as NumPy arrays by NumPy only cross-validation splitters.
    # 将输入数据 X 转换为适当的设备（如 GPU）上的 NumPy 数组
    train, test = xp.asarray(train, device=X_device), xp.asarray(test, device=X_device)

    if not isinstance(error_score, numbers.Number) and error_score != "raise":
        # 如果 error_score 不是数字且不是字符串 "raise"，则抛出值错误
        raise ValueError(
            "error_score must be the string 'raise' or a numeric value. "
            "(Hint: if using 'raise', please make sure that it has been "
            "spelled correctly.)"
        )

    progress_msg = ""
    if verbose > 2:
        if split_progress is not None:
            progress_msg = f" {split_progress[0]+1}/{split_progress[1]}"
        if candidate_progress and verbose > 9:
            progress_msg += f"; {candidate_progress[0]+1}/{candidate_progress[1]}"

    if verbose > 1:
        if parameters is None:
            params_msg = ""
        else:
            sorted_keys = sorted(parameters)  # 确保输出结果的键是有序的
            params_msg = ", ".join(f"{k}={parameters[k]}" for k in sorted_keys)
    
    if verbose > 9:
        # 如果详细程度超过 9，输出交叉验证的详细进度信息
        start_msg = f"[CV{progress_msg}] START {params_msg}"
        print(f"{start_msg}{(80 - len(start_msg)) * '.'}")

    # 调整样本权重的长度
    # 对拟合和评分的参数进行检查，确保其适用于指定的训练数据集
    fit_params = fit_params if fit_params is not None else {}
    fit_params = _check_method_params(X, params=fit_params, indices=train)
    score_params = score_params if score_params is not None else {}
    score_params_train = _check_method_params(X, params=score_params, indices=train)
    score_params_test = _check_method_params(X, params=score_params, indices=test)

    if parameters is not None:
        # 克隆参数，因为有时参数本身可能是估算器，例如在管道中搜索不同的估算器时
        # 参考：https://github.com/scikit-learn/scikit-learn/pull/26786
        estimator = estimator.set_params(**clone(parameters, safe=False))

    start_time = time.time()

    # 将数据集分割为训练集和测试集，返回拟合和评分所需的 X 和 y
    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)
    # 初始化一个空字典用于存储结果
    result = {}
    try:
        # 尝试执行以下操作，捕获可能出现的异常
        if y_train is None:
            # 如果 y_train 为空，使用 X_train 进行模型拟合，传入额外的参数 fit_params
            estimator.fit(X_train, **fit_params)
        else:
            # 如果 y_train 不为空，使用 X_train 和 y_train 进行模型拟合，传入额外的参数 fit_params
            estimator.fit(X_train, y_train, **fit_params)

    except Exception:
        # 如果发生异常，则记录拟合过程中出错时的时间
        fit_time = time.time() - start_time
        score_time = 0.0
        if error_score == "raise":
            # 如果 error_score 设置为 "raise"，则继续抛出异常
            raise
        elif isinstance(error_score, numbers.Number):
            # 如果 error_score 是一个数字类型
            if isinstance(scorer, _MultimetricScorer):
                # 如果 scorer 是 _MultimetricScorer 的实例
                # 为每个评分器设置相同的错误分数
                test_scores = {name: error_score for name in scorer._scorers}
                if return_train_score:
                    # 如果需要返回训练分数，复制测试分数到训练分数
                    train_scores = test_scores.copy()
            else:
                # 如果 scorer 不是 _MultimetricScorer 的实例，则使用相同的错误分数
                test_scores = error_score
                if return_train_score:
                    # 如果需要返回训练分数，使用相同的错误分数
                    train_scores = error_score
        # 将拟合时的异常信息格式化并存储到结果字典中的 "fit_error" 键下
        result["fit_error"] = format_exc()
    else:
        # 如果没有异常发生，则将 "fit_error" 设置为 None
        result["fit_error"] = None

        # 计算拟合时间
        fit_time = time.time() - start_time
        # 计算测试分数，调用 _score 函数计算
        test_scores = _score(
            estimator, X_test, y_test, scorer, score_params_test, error_score
        )
        # 计算评分时间
        score_time = time.time() - start_time - fit_time
        if return_train_score:
            # 如果需要返回训练分数，调用 _score 函数计算
            train_scores = _score(
                estimator, X_train, y_train, scorer, score_params_train, error_score
            )

    if verbose > 1:
        # 如果 verbose 大于 1，则输出详细信息
        total_time = score_time + fit_time
        end_msg = f"[CV{progress_msg}] END "
        result_msg = params_msg + (";" if params_msg else "")
        if verbose > 2:
            if isinstance(test_scores, dict):
                # 如果测试分数是一个字典，则按字母顺序输出每个评分器的分数
                for scorer_name in sorted(test_scores):
                    result_msg += f" {scorer_name}: ("
                    if return_train_score:
                        # 如果需要返回训练分数，获取并添加训练分数
                        scorer_scores = train_scores[scorer_name]
                        result_msg += f"train={scorer_scores:.3f}, "
                    # 添加测试分数
                    result_msg += f"test={test_scores[scorer_name]:.3f})"
            else:
                # 如果测试分数不是字典，则添加一个总体分数
                result_msg += ", score="
                if return_train_score:
                    # 如果需要返回训练分数，添加训练分数
                    result_msg += f"(train={train_scores:.3f}, test={test_scores:.3f})"
                else:
                    # 否则只添加测试分数
                    result_msg += f"{test_scores:.3f}"
        # 添加总时间
        result_msg += f" total time={logger.short_format_time(total_time)}"

        # 右对齐 result_msg
        end_msg += "." * (80 - len(end_msg) - len(result_msg))
        end_msg += result_msg
        # 打印结束消息
        print(end_msg)

    # 将测试分数存储到结果字典中的 "test_scores" 键下
    result["test_scores"] = test_scores
    if return_train_score:
        # 如果需要返回训练分数，将训练分数存储到结果字典中的 "train_scores" 键下
        result["train_scores"] = train_scores
    if return_n_test_samples:
        # 如果需要返回测试样本数，将其存储到结果字典中的 "n_test_samples" 键下
        result["n_test_samples"] = _num_samples(X_test)
    if return_times:
        # 如果需要返回时间信息，将拟合时间和评分时间存储到结果字典中
        result["fit_time"] = fit_time
        result["score_time"] = score_time
    if return_parameters:
        # 如果需要返回参数信息，将参数信息存储到结果字典中的 "parameters" 键下
        result["parameters"] = parameters
    if return_estimator:
        # 如果需要返回估计器，将其存储到结果字典中的 "estimator" 键下
        result["estimator"] = estimator
    # 返回最终的结果字典
    return result
def _score(estimator, X_test, y_test, scorer, score_params, error_score="raise"):
    """Compute the score(s) of an estimator on a given test set.

    Will return a dict of floats if `scorer` is a _MultiMetricScorer, otherwise a single
    float is returned.
    """
    # 如果 score_params 为 None，则置为一个空字典
    score_params = {} if score_params is None else score_params

    try:
        # 尝试计算评分
        if y_test is None:
            # 如果 y_test 为 None，则调用 scorer 评估器计算 X_test 的分数
            scores = scorer(estimator, X_test, **score_params)
        else:
            # 否则，使用 scorer 评估器计算 X_test 和 y_test 的分数
            scores = scorer(estimator, X_test, y_test, **score_params)
    except Exception:
        if isinstance(scorer, _MultimetricScorer):
            # 如果是 _MultimetricScorer 抛出异常，则根据 error_score 参数处理
            raise
        else:
            if error_score == "raise":
                # 如果 error_score 为 "raise"，则重新抛出异常
                raise
            else:
                # 否则，将 scores 设为 error_score，并发出警告
                scores = error_score
                warnings.warn(
                    (
                        "Scoring failed. The score on this train-test partition for "
                        f"these parameters will be set to {error_score}. Details: \n"
                        f"{format_exc()}"
                    ),
                    UserWarning,
                )

    # 检查 `_MultimetricScorer` 抛出的非 "raise" 错误信息
    if isinstance(scorer, _MultimetricScorer):
        # 找出所有返回字符串类型的异常消息
        exception_messages = [
            (name, str_e) for name, str_e in scores.items() if isinstance(str_e, str)
        ]
        if exception_messages:
            # 如果有异常消息，对每个异常消息的分数设为 error_score，并发出警告
            for name, str_e in exception_messages:
                scores[name] = error_score
                warnings.warn(
                    (
                        "Scoring failed. The score on this train-test partition for "
                        f"these parameters will be set to {error_score}. Details: \n"
                        f"{str_e}"
                    ),
                    UserWarning,
                )

    # 检查分数是否为数值类型
    error_msg = "scoring must return a number, got %s (%s) instead. (scorer=%s)"
    if isinstance(scores, dict):
        # 如果 scores 是字典，则对每个分数检查其类型是否为数值
        for name, score in scores.items():
            if hasattr(score, "item"):
                with suppress(ValueError):
                    # 尝试解包 memmapped scalar
                    score = score.item()
            if not isinstance(score, numbers.Number):
                # 如果不是数值类型，则抛出 ValueError 异常
                raise ValueError(error_msg % (score, type(score), name))
            scores[name] = score
    else:  # scalar
        # 如果 scores 是标量，则检查其类型是否为数值
        if hasattr(scores, "item"):
            with suppress(ValueError):
                # 尝试解包 memmapped scalar
                scores = scores.item()
        if not isinstance(scores, numbers.Number):
            # 如果不是数值类型，则抛出 ValueError 异常
            raise ValueError(error_msg % (scores, type(scores), scorer))
    return scores
    {
        # `estimator`参数要求具有`fit`和`predict`方法的对象
        "estimator": [HasMethods(["fit", "predict"])],
        # `X`参数可以是类数组或稀疏矩阵
        "X": ["array-like", "sparse matrix"],
        # `y`参数可以是类数组、稀疏矩阵或者为None
        "y": ["array-like", "sparse matrix", None],
        # `groups`参数可以是类数组或者为None
        "groups": ["array-like", None],
        # `cv`参数是交叉验证对象
        "cv": ["cv_object"],
        # `n_jobs`参数可以是整数或者为None
        "n_jobs": [Integral, None],
        # `verbose`参数是一个布尔值或者整数
        "verbose": ["verbose"],
        # `fit_params`参数可以是字典或者为None
        "fit_params": [dict, None],
        # `params`参数可以是字典或者为None
        "params": [dict, None],
        # `pre_dispatch`参数可以是整数、字符串或者为None
        "pre_dispatch": [Integral, str, None],
        # `method`参数指定预测方法，可以是以下字符串之一：
        # "predict", "predict_proba", "predict_log_proba", "decision_function"
        "method": [
            StrOptions(
                {
                    "predict",
                    "predict_proba",
                    "predict_log_proba",
                    "decision_function",
                }
            )
        ],
    },
    # `prefer_skip_nested_validation`为False，表示未对estimator进行嵌套验证
    prefer_skip_nested_validation=False,
# 定义交叉验证预测函数
def cross_val_predict(
    estimator,
    X,
    y=None,
    *,
    groups=None,
    cv=None,
    n_jobs=None,
    verbose=0,
    fit_params=None,
    params=None,
    pre_dispatch="2*n_jobs",
    method="predict",
):
    """生成每个输入数据点的交叉验证估计值。

    根据参数cv对数据进行拆分。每个样本属于唯一的测试集，并使用在对应训练集上拟合的估计器进行预测。

    将这些预测结果传递给评估度量可能不是衡量泛化性能的有效方式。结果可能与cross_validate和cross_val_score不同，除非所有测试集大小相等，并且度量值在样本上分解。

    详细信息请参阅 :ref:`用户指南 <cross_validation>`。

    参数
    ----------
    estimator : estimator
        用于拟合数据的估计器实例。必须实现`fit`方法和`method`参数指定的方法。

    X : {array-like, sparse matrix}，形状为 (n_samples, n_features)
        要拟合的数据。可以是列表或至少二维数组等。

    y : {array-like, sparse matrix}，形状为 (n_samples,) 或 (n_samples, n_outputs)，默认为None
        在监督学习中要预测的目标变量。

    groups : 形状为 (n_samples,) 的数组，默认为None
        在将数据集拆分为训练/测试集时使用的样本组标签。仅与“Group” :term:`cv` 实例（例如 :class:`GroupKFold`）一起使用。

        .. versionchanged:: 1.4
            如果启用了元数据路由（通过 ``sklearn.set_config(enable_metadata_routing=True)``），则只能通过 ``params`` 参数一起传递 ``groups``。
            例如：``cross_val_predict(..., params={'groups': groups})``。

    cv : int、交叉验证生成器或可迭代对象，默认为None
        确定交叉验证拆分策略。
        cv 的可能输入有：

        - None，使用默认的 5 折交叉验证，
        - int，指定 `(Stratified)KFold` 中的折数，
        - :term:`CV splitter`，
        - 生成(train, test)拆分索引数组的可迭代对象。

        对于 int/None 输入，如果估计器是分类器且 `y` 是二进制或多类别，则使用 :class:`StratifiedKFold`。
        在所有其他情况下，使用 :class:`KFold`。这些分割器将使用 `shuffle=False` 实例化，因此调用时的分割将保持一致。

        参见 :ref:`用户指南 <cross_validation>`，了解可以在此处使用的各种交叉验证策略。

        .. versionchanged:: 0.22
            如果为 None，则 `cv` 的默认值从 3 折更改为 5 折。
    n_jobs : int, default=None
        Number of jobs to run in parallel. Training the estimator and
        predicting are parallelized over the cross-validation splits.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
        
    verbose : int, default=0
        The verbosity level.
        
    fit_params : dict, default=None
        Parameters to pass to the fit method of the estimator.

        .. deprecated:: 1.4
            This parameter is deprecated and will be removed in version 1.6. Use
            ``params`` instead.
            
    params : dict, default=None
        Parameters to pass to the underlying estimator's ``fit`` and the CV
        splitter.

        .. versionadded:: 1.4
        
    pre_dispatch : int or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:
        
            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs
              
            - An int, giving the exact number of total jobs that are
              spawned
              
            - A str, giving an expression as a function of n_jobs,
              as in '2*n_jobs'
              
    method : {'predict', 'predict_proba', 'predict_log_proba', \
              'decision_function'}, default='predict'
        The method to be invoked by `estimator`.
        
    Returns
    -------
    predictions : ndarray
        This is the result of calling `method`. Shape:
        
            - When `method` is 'predict' and in special case where `method` is
              'decision_function' and the target is binary: (n_samples,)
            - When `method` is one of {'predict_proba', 'predict_log_proba',
              'decision_function'} (unless special case above):
              (n_samples, n_classes)
            - If `estimator` is :term:`multioutput`, an extra dimension
              'n_outputs' is added to the end of each shape above.
              
    See Also
    --------
    cross_val_score : Calculate score for each CV split.
    cross_validate : Calculate one or more scores and timings for each CV
        split.
        
    Notes
    -----
    In the case that one or more classes are absent in a training portion, a
    default score needs to be assigned to all instances for that class if
    ``method`` produces columns per class, as in {'decision_function',
    'predict_proba', 'predict_log_proba'}.  For ``predict_proba`` this value is
    0.  In order to ensure finite output, we approximate negative infinity by
    the minimum finite float value for the dtype in other cases.
    
    Examples
    --------
    params = _check_params_groups_deprecation(fit_params, params, groups, "1.6")
    """
    # 检查和处理参数中的分组依赖性，根据指定的版本号"1.6"
    params = _check_params_groups_deprecation(fit_params, params, groups, "1.6")
    
    X, y = indexable(X, y)
    # 确保 X 和 y 可以索引化，即确保它们可以用作索引操作

    if _routing_enabled():
        # 如果启用了路由功能
        
        # 创建一个 MetadataRouter 实例，并设置其所有者为 "cross_validate"
        router = (
            MetadataRouter(owner="cross_validate")
            # 添加一个分割器，使用给定的交叉验证策略 cv，方法映射为 fit -> split
            .add(
                splitter=cv,
                method_mapping=MethodMapping().add(caller="fit", callee="split"),
            )
            # 添加一个评估器，方法映射为 fit -> fit
            .add(
                estimator=estimator,
                # TODO(SLEP6): 也需要传递预测方法的元数据。
                method_mapping=MethodMapping().add(caller="fit", callee="fit"),
            )
        )
        
        try:
            # 对 router 进行路由处理，调用 process_routing 函数进行处理
            routed_params = process_routing(router, "fit", **params)
        
        except UnsetMetadataPassedError as e:
            # 如果捕获到未设置元数据传递错误
            
            # 默认的异常信息会提到 `fit`，但实际上用户并未直接调用 `fit` 方法，
            # 因此修改消息以使其更适合这种情况。
            unrequested_params = sorted(e.unrequested_params)
            raise UnsetMetadataPassedError(
                message=(
                    f"{unrequested_params} are passed to `cross_val_predict` but are"
                    " not explicitly set as requested or not requested for"
                    f" cross_validate's estimator: {estimator.__class__.__name__} Call"
                    " `.set_fit_request({{metadata}}=True)` on the estimator for"
                    f" each metadata in {unrequested_params} that you want to use and"
                    " `metadata=False` for not using it. See the Metadata Routing User"
                    " guide <https://scikit-learn.org/stable/metadata_routing.html>"
                    " for more information."
                ),
                unrequested_params=e.unrequested_params,
                routed_params=e.routed_params,
            )
    
    else:
        # 如果未启用路由功能
        
        # 创建一个 Bunch 对象，用于保存路由参数
        routed_params = Bunch()
        routed_params.splitter = Bunch(split={"groups": groups})
        routed_params.estimator = Bunch(fit=params)

    # 检查并确认交叉验证策略，确保其可用性，如果是分类器，则指定为分类器
    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    
    # 根据路由参数 splits 对数据集 X, y 进行拆分
    splits = list(cv.split(X, y, **routed_params.splitter.split))

    # 获取测试数据集的索引，将所有分割后的测试集索引合并为一个数组
    test_indices = np.concatenate([test for _, test in splits])
    
    # 检查测试数据集索引是否为数据集 X 的一个排列，否则抛出异常
    if not _check_is_permutation(test_indices, _num_samples(X)):
        raise ValueError("cross_val_predict only works for partitions")

    # 如果分类方法生成多列输出，
    # 我们需要手动编码类以确保列的顺序一致。
    encode = (
        method in ["decision_function", "predict_proba", "predict_log_proba"]
        and y is not None
    )
    # 如果需要编码，则将目标变量 y 转换为 NumPy 数组
    if encode:
        y = np.asarray(y)
        # 如果 y 是一维数组，则使用 LabelEncoder 进行编码
        if y.ndim == 1:
            le = LabelEncoder()
            y = le.fit_transform(y)
        # 如果 y 是二维数组，则对每列分别使用 LabelEncoder 进行编码
        elif y.ndim == 2:
            y_enc = np.zeros_like(y, dtype=int)
            for i_label in range(y.shape[1]):
                y_enc[:, i_label] = LabelEncoder().fit_transform(y[:, i_label])
            y = y_enc
    
    # 克隆评估器以确保所有折叠（folds）都是独立的，并且它可以被序列化（pickle）
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)
    # 使用并行处理并行计算每个分割（split）上的预测结果
    predictions = parallel(
        delayed(_fit_and_predict)(
            clone(estimator),
            X,
            y,
            train,
            test,
            routed_params.estimator.fit,
            method,
        )
        for train, test in splits
    )
    
    # 创建一个逆索引数组以便将测试集索引映射回原始顺序
    inv_test_indices = np.empty(len(test_indices), dtype=int)
    inv_test_indices[test_indices] = np.arange(len(test_indices))
    
    # 如果预测结果是稀疏矩阵，则进行堆叠操作
    if sp.issparse(predictions[0]):
        predictions = sp.vstack(predictions, format=predictions[0].format)
    # 如果需要编码并且预测结果是列表，则处理多输出多类别任务
    elif encode and isinstance(predictions[0], list):
        # `predictions` 是每个折叠（fold）中方法输出的列表。
        # 如果每个输出也是一个列表，则将其视为多输出多类别任务。
        # 我们需要将每个标签的方法输出分别连接成一个长度为 `n_labels` 的列表。
        n_labels = y.shape[1]
        concat_pred = []
        for i_label in range(n_labels):
            label_preds = np.concatenate([p[i_label] for p in predictions])
            concat_pred.append(label_preds)
        predictions = concat_pred
    else:
        # 否则，将预测结果列表连接成一个数组
        predictions = np.concatenate(predictions)
    
    # 如果预测结果是列表，则按照逆索引数组重新排列每个预测列表
    if isinstance(predictions, list):
        return [p[inv_test_indices] for p in predictions]
    else:
        # 否则，按照逆索引数组重新排列预测结果数组
        return predictions[inv_test_indices]
# 定义一个函数用于拟合估算器并预测给定数据集拆分的值
def _fit_and_predict(estimator, X, y, train, test, fit_params, method):
    """Fit estimator and predict values for a given dataset split.

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    estimator : estimator object implementing 'fit' and 'predict'
        The object to use to fit the data.

    X : array-like of shape (n_samples, n_features)
        The data to fit.

        .. versionchanged:: 0.20
            X is only required to be an object with finite length or shape now

    y : array-like of shape (n_samples,) or (n_samples, n_outputs) or None
        The target variable to try to predict in the case of
        supervised learning.

    train : array-like of shape (n_train_samples,)
        Indices of training samples.

    test : array-like of shape (n_test_samples,)
        Indices of test samples.

    fit_params : dict or None
        Parameters that will be passed to ``estimator.fit``.

    method : str
        Invokes the passed method name of the passed estimator.

    Returns
    -------
    predictions : sequence
        Result of calling 'estimator.method'
    """
    # 调整样本权重的长度
    fit_params = fit_params if fit_params is not None else {}
    fit_params = _check_method_params(X, params=fit_params, indices=train)

    # 将数据集分割为训练集和测试集
    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, _ = _safe_split(estimator, X, y, test, train)

    # 根据是否有目标变量选择调用estimator.fit的方式
    if y_train is None:
        estimator.fit(X_train, **fit_params)
    else:
        estimator.fit(X_train, y_train, **fit_params)
    
    # 获取estimator对象中指定方法的函数
    func = getattr(estimator, method)
    # 使用测试集进行预测
    predictions = func(X_test)

    # 对预测结果进行编码，确保预测顺序正确
    encode = (
        method in ["decision_function", "predict_proba", "predict_log_proba"]
        and y is not None
    )

    if encode:
        if isinstance(predictions, list):
            # 对于列表类型的预测结果，分别确保预测顺序正确
            predictions = [
                _enforce_prediction_order(
                    estimator.classes_[i_label],
                    predictions[i_label],
                    n_classes=len(set(y[:, i_label])),
                    method=method,
                )
                for i_label in range(len(predictions))
            ]
        else:
            # 对于二维的y数组，确保二元标签指示矩阵的预测顺序正确
            n_classes = len(set(y)) if y.ndim == 1 else y.shape[1]
            predictions = _enforce_prediction_order(
                estimator.classes_, predictions, n_classes, method
            )
    
    # 返回预测结果
    return predictions


def _enforce_prediction_order(classes, predictions, n_classes, method):
    """Ensure that prediction arrays have correct column order

    When doing cross-validation, if one or more classes are
    not present in the subset of data used for training,
    then the output prediction array might not have the same
    columns as other folds. Use the list of class names
    (assumed to be ints) to enforce the correct column order.

    Note that `classes` is the list of classes in this fold
    """
    # 确保预测数组具有正确的列顺序
    # 在交叉验证中，如果某些类别在训练数据的子集中不存在，
    # 那么输出的预测数组可能与其他折叠的列不同。使用类名列表来确保正确的列顺序。
    pass
    # 如果训练集中的类别数量与整个训练集的类别数量不一致，发出警告
    if n_classes != len(classes):
        # 提出建议以解决问题：使用交叉验证技术生成适当的分层折叠
        recommendation = (
            "To fix this, use a cross-validation "
            "technique resulting in properly "
            "stratified folds"
        )
        # 发出运行时警告，说明类别数量不匹配的后果及解决建议
        warnings.warn(
            "Number of classes in training fold ({}) does "
            "not match total number of classes ({}). "
            "Results may not be appropriate for your use case. "
            "{}".format(len(classes), n_classes, recommendation),
            RuntimeWarning,
        )
        # 如果方法是 decision_function，则检查预测结果的维度与类别数的匹配情况
        if method == "decision_function":
            if predictions.ndim == 2 and predictions.shape[1] != len(classes):
                # 处理当预测结果的形状与用于训练的类别数不匹配的情况，
                # 例如 sklearn.svm.SVC 设置为 `decision_function_shape='ovo'` 的情况
                raise ValueError(
                    "Output shape {} of {} does not match "
                    "number of classes ({}) in fold. "
                    "Irregular decision_function outputs "
                    "are not currently supported by "
                    "cross_val_predict".format(predictions.shape, method, len(classes))
                )
            # 特殊情况下，当类别数小于等于2时，预测结果应该是1维数组
            if len(classes) <= 2:
                raise ValueError(
                    "Only {} class/es in training fold, but {} "
                    "in overall dataset. This "
                    "is not supported for decision_function "
                    "with imbalanced folds. {}".format(
                        len(classes), n_classes, recommendation
                    )
                )

        # 获取 predictions 数据类型的最小浮点数值
        float_min = np.finfo(predictions.dtype).min
        # 默认值字典，根据不同的 method 设置不同的默认值
        default_values = {
            "decision_function": float_min,
            "predict_log_proba": float_min,
            "predict_proba": 0,
        }
        # 创建一个全零数组，形状为（预测样本数，整个训练集类别数），并填充默认值
        predictions_for_all_classes = np.full(
            (_num_samples(predictions), n_classes),
            default_values[method],
            dtype=predictions.dtype,
        )
        # 将预测结果填充到对应的类别位置上
        predictions_for_all_classes[:, classes] = predictions
        # 更新预测结果为包含所有类别的完整预测值数组
        predictions = predictions_for_all_classes
    # 返回更新后的预测结果
    return predictions
# 定义一个函数，用于检查给定的索引数组是否是 np.arange(n_samples) 的重新排序
def _check_is_permutation(indices, n_samples):
    # 如果索引数组的长度不等于预期的样本数，返回 False
    if len(indices) != n_samples:
        return False
    # 创建一个长度为 n_samples 的布尔类型的数组 hit，初始值为 False
    hit = np.zeros(n_samples, dtype=bool)
    # 将 hit 中对应索引位置设为 True
    hit[indices] = True
    # 如果 hit 中存在 False，说明索引数组不是 np.arange(n_samples) 的重新排序，返回 False
    if not np.all(hit):
        return False
    # 否则返回 True，表示索引数组是 np.arange(n_samples) 的重新排序
    return True


# 使用装饰器 validate_params 对 permutation_test_score 函数进行参数验证
@validate_params(
    {
        "estimator": [HasMethods("fit")],  # estimator 参数要求具有 fit 方法
        "X": ["array-like", "sparse matrix"],  # X 参数可以是数组或稀疏矩阵
        "y": ["array-like", None],  # y 参数可以是数组或者为空
        "groups": ["array-like", None],  # groups 参数可以是数组或者为空
        "cv": ["cv_object"],  # cv 参数要求是交叉验证对象
        "n_permutations": [Interval(Integral, 1, None, closed="left")],  # n_permutations 参数要求是大于等于1的整数
        "n_jobs": [Integral, None],  # n_jobs 参数要求是整数或者为空
        "random_state": ["random_state"],  # random_state 参数要求是随机数生成器或者种子
        "verbose": ["verbose"],  # verbose 参数要求是详细程度控制参数
        "scoring": [StrOptions(set(get_scorer_names())), callable, None],  # scoring 参数要求是评分器名称集合中的字符串、可调用对象或者为空
        "fit_params": [dict, None],  # fit_params 参数要求是字典或者为空
    },
    prefer_skip_nested_validation=False,  # estimator 参数尚未验证
)
# 定义一个函数，用于计算带置换检验的交叉验证得分的显著性
def permutation_test_score(
    estimator,
    X,
    y,
    *,
    groups=None,
    cv=None,
    n_permutations=100,
    n_jobs=None,
    random_state=0,
    verbose=0,
    scoring=None,
    fit_params=None,
):
    """Evaluate the significance of a cross-validated score with permutations.

    Permutes targets to generate 'randomized data' and compute the empirical
    p-value against the null hypothesis that features and targets are
    independent.

    The p-value represents the fraction of randomized data sets where the
    estimator performed as well or better than in the original data. A small
    p-value suggests that there is a real dependency between features and
    targets which has been used by the estimator to give good predictions.
    A large p-value may be due to lack of real dependency between features
    and targets or the estimator was not able to use the dependency to
    give good predictions.

    Read more in the :ref:`User Guide <permutation_test_score>`.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like of shape at least 2D
        The data to fit.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs) or None
        The target variable to try to predict in the case of
        supervised learning.
    """
    # groups参数：用于限制在组内进行置换的标签，即将“y”值在具有相同组标识符的样本中进行置换。
    # 如果未指定，则“y”值将在所有样本中进行置换。

    # cv参数：交叉验证分割策略的确定。
    # cv的可能输入包括：
    # - `None`，使用默认的5折交叉验证，
    # - int，指定`(Stratified)KFold`中的折数，
    # - :term:`CV splitter`，
    # - 生成(train, test)分割的可迭代对象作为索引数组。

    # 对于`int`/`None`的输入，如果估计器是分类器并且`y`是二进制或多类别的，将使用:class:`StratifiedKFold`。
    # 在其他所有情况下，将使用:class:`KFold`。这些分割器将以`shuffle=False`实例化，因此分割在每次调用时都相同。

    # 参考 :ref:`用户指南 <cross_validation>` 查看可用的各种交叉验证策略。

    # .. versionchanged:: 0.22
    #    如果`cv`为`None`，默认值从3折更改为5折。

    # n_permutations参数：对“y”进行置换的次数，默认为100次。

    # n_jobs参数：并行运行的作业数。训练估计器和计算交叉验证分数在置换过程中并行化。
    # 如果未在:obj:`joblib.parallel_backend`上下文中，`None`表示1。`-1`表示使用所有处理器。
    # 更多细节请参阅 :term:`术语表 <n_jobs>`。

    # random_state参数：传递一个整数以便在样本中对“y”值进行重复置换以产生可重复的输出。
    # 参见 :term:`术语表 <random_state>`。

    # verbose参数：详细程度级别。

    # scoring参数：用于评估测试集预测的单个字符串（参见 :ref:`scoring_parameter`）或可调用函数（参见 :ref:`scoring`）。
    # 如果为`None`，则使用估计器的评分方法。

    # fit_params参数：传递给估计器的fit方法的参数。

    # .. versionadded:: 0.24

    # 返回：
    # score : float
    #     不对目标进行置换时的真实分数。
    # permutation_scores : array of shape (n_permutations,)
    #     每个置换得到的分数。
    # 将输入数据 X, y, groups 转换为可索引的格式
    X, y, groups = indexable(X, y, groups)

    # 检查并确保交叉验证策略 cv 的合法性，并根据分类器是否是评估器来进行检查
    cv = check_cv(cv, y, classifier=is_classifier(estimator))

    # 确定评分函数，确保评估器可以使用该评分函数进行评估
    scorer = check_scoring(estimator, scoring=scoring)

    # 检查并确保随机数生成器 random_state 的合法性
    random_state = check_random_state(random_state)

    # 克隆评估器，以确保每个折叠都是独立的，并且评估器可以被序列化
    score = _permutation_test_score(
        clone(estimator), X, y, groups, cv, scorer, fit_params=fit_params
    )

    # 使用并行处理生成 n_permutations 次排列测试的分数
    permutation_scores = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_permutation_test_score)(
            clone(estimator),
            X,
            _shuffle(y, groups, random_state),
            groups,
            cv,
            scorer,
            fit_params=fit_params,
        )
        for _ in range(n_permutations)
    )

    # 将生成的排列分数转换为 NumPy 数组
    permutation_scores = np.array(permutation_scores)

    # 计算 p-value，表示排列测试中分数大于等于真实分数的比例的近似概率
    pvalue = (np.sum(permutation_scores >= score) + 1.0) / (n_permutations + 1)

    # 返回原始得分、排列分数数组和计算得到的 p-value
    return score, permutation_scores, pvalue
    {
        "estimator": [HasMethods(["fit"])],  # 确保 estimator 参数有 fit 方法
        "X": ["array-like", "sparse matrix"],  # 特征数据，可以是数组或稀疏矩阵
        "y": ["array-like", None],  # 目标数据，可以是数组或为空
        "groups": ["array-like", None],  # 分组数据，可以是数组或为空
        "train_sizes": ["array-like"],  # 训练集大小的列表
        "cv": ["cv_object"],  # 交叉验证生成器对象
        "scoring": [StrOptions(set(get_scorer_names())), callable, None],  # 评分方法的选择，可以是预定义名称集合、可调用对象或为空
        "exploit_incremental_learning": ["boolean"],  # 是否利用增量学习的布尔值
        "n_jobs": [Integral, None],  # 并行作业数量或空
        "pre_dispatch": [Integral, str],  # 预分配作业或其字符串表示
        "verbose": ["verbose"],  # 冗长模式控制
        "shuffle": ["boolean"],  # 是否洗牌数据集的布尔值
        "random_state": ["random_state"],  # 随机状态对象
        "error_score": [StrOptions({"raise"}), Real],  # 错误得分处理方式，可以是"raise"或实数
        "return_times": ["boolean"],  # 是否返回时间的布尔值
        "fit_params": [dict, None],  # fit 方法的附加参数字典或空
        "params": [dict, None],  # 其他参数字典或空
    },
    prefer_skip_nested_validation=False,  # 不偏向跳过嵌套验证
)
def learning_curve(
    estimator,
    X,
    y,
    *,
    groups=None,
    train_sizes=np.linspace(0.1, 1.0, 5),  # 训练集大小的范围，默认从0.1到1.0均匀分布5个点
    cv=None,  # 交叉验证生成器对象
    scoring=None,  # 评分方法，默认为空
    exploit_incremental_learning=False,  # 是否利用增量学习，默认为False
    n_jobs=None,  # 并行作业数量或空
    pre_dispatch="all",  # 预分配作业或"all"
    verbose=0,  # 冗长模式控制，默认为0
    shuffle=False,  # 是否洗牌数据集，默认为False
    random_state=None,  # 随机状态对象，默认为空
    error_score=np.nan,  # 错误得分处理方式，默认为NaN
    return_times=False,  # 是否返回时间，默认为False
    fit_params=None,  # fit 方法的附加参数字典或空
    params=None,  # 其他参数字典或空
):
    """Learning curve.

    Determines cross-validated training and test scores for different training
    set sizes.

    A cross-validation generator splits the whole dataset k times in training
    and test data. Subsets of the training set with varying sizes will be used
    to train the estimator and a score for each training subset size and the
    test set will be computed. Afterwards, the scores will be averaged over
    all k runs for each training subset size.

    Read more in the :ref:`User Guide <learning_curve>`.

    Parameters
    ----------
    estimator : object type that implements the "fit" method
        An object of that type which is cloned for each validation. It must
        also implement "predict" unless `scoring` is a callable that doesn't
        rely on "predict" to compute a score.

    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training vector, where `n_samples` is the number of samples and
        `n_features` is the number of features.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs) or None
        Target relative to X for classification or regression;
        None for unsupervised learning.

    groups : array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting the dataset into
        train/test set. Only used in conjunction with a "Group" :term:`cv`
        instance (e.g., :class:`GroupKFold`).

        .. versionchanged:: 1.6
            ``groups`` can only be passed if metadata routing is not enabled
            via ``sklearn.set_config(enable_metadata_routing=True)``. When routing
            is enabled, pass ``groups`` alongside other metadata via the ``params``
            argument instead. E.g.:
            ``learning_curve(..., params={'groups': groups})``.

    train_sizes : array-like of shape (n_ticks,), \
            default=np.linspace(0.1, 1.0, 5)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually has to
        be big enough to contain at least one sample from each class.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For int/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.
    scoring : str or callable, default=None
        # 评分标准，可以是字符串（参见:ref:`scoring_parameter`），也可以是一个可调用的评分函数，其签名为 ``scorer(estimator, X, y)``.

    exploit_incremental_learning : bool, default=False
        # 如果估计器支持增量学习，设置为 True 可加快不同训练集大小的拟合速度。

    n_jobs : int, default=None
        # 并行运行的作业数。估计器的训练和评分在不同的训练和测试集上并行执行。``None`` 表示使用 1 个作业，除非在 :obj:`joblib.parallel_backend` 上下文中。``-1`` 表示使用所有处理器。详见 :term:`Glossary <n_jobs>`。

    pre_dispatch : int or str, default='all'
        # 预派发的作业数量以便并行执行（默认为所有）。该选项可以减少分配的内存。字符串可以是一个表达式，如 '2*n_jobs'。

    verbose : int, default=0
        # 控制详细程度：数值越高，输出的消息越多。

    shuffle : bool, default=False
        # 是否在基于 ``train_sizes`` 取前缀之前对训练数据进行洗牌。

    random_state : int, RandomState instance or None, default=None
        # 当 ``shuffle`` 为 True 时使用。传递一个整数以在多次函数调用中获得可重现的输出。详见 :term:`Glossary <random_state>`。

    error_score : 'raise' or numeric, default=np.nan
        # 如果在估计器拟合时发生错误，将为分数赋予一个值。如果设置为 'raise'，则会引发错误。如果给定一个数值，则会引发 FitFailedWarning。

        .. versionadded:: 0.20
            # 添加于版本 0.20

    return_times : bool, default=False
        # 是否返回拟合和评分时间。

    fit_params : dict, default=None
        # 传递给估计器的 `fit` 方法的参数。

        .. deprecated:: 1.6
            # 从版本 1.6 开始，此参数已被弃用，并将在将来的版本中移除。请改用 ``params``。

    params : dict, default=None
        # 传递给估计器的 `fit` 方法和评分函数的参数。

            - 如果 `enable_metadata_routing=False`（默认）：
              直接传递给估计器的 `fit` 方法的参数。

            - 如果 `enable_metadata_routing=True`：
              安全地传递给估计器的 `fit` 方法的参数。详见 :ref:`Metadata Routing User Guide <metadata_routing>` 获取更多细节。

            .. versionadded:: 1.6
                # 添加于版本 1.6

    Returns
    -------
    train_sizes_abs : array of shape (n_unique_ticks,)
        # 用于生成学习曲线的训练示例数量。请注意，ticks 的数量可能少于 n_ticks，因为重复的条目将被删除。

    train_scores : array of shape (n_ticks, n_cv_folds)
        # 训练集的得分。
    # test_scores : array of shape (n_ticks, n_cv_folds)
    #     测试集上的评分结果，形状为 (n_ticks, n_cv_folds)，即不同训练集大小和交叉验证折数下的测试集评分
    
    # fit_times : array of shape (n_ticks, n_cv_folds)
    #     拟合时间，单位为秒，形状为 (n_ticks, n_cv_folds)
    #     只有在 `return_times` 参数为 True 时才存在
    
    # score_times : array of shape (n_ticks, n_cv_folds)
    #     评分时间，单位为秒，形状为 (n_ticks, n_cv_folds)
    #     只有在 `return_times` 参数为 True 时才存在
    
    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from sklearn.model_selection import learning_curve
    >>> X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    >>> tree = DecisionTreeClassifier(max_depth=4, random_state=42)
    >>> train_size_abs, train_scores, test_scores = learning_curve(
    ...     tree, X, y, train_sizes=[0.3, 0.6, 0.9]
    ... )
    >>> for train_size, cv_train_scores, cv_test_scores in zip(
    ...     train_size_abs, train_scores, test_scores
    ... ):
    ...     print(f"{train_size} samples were used to train the model")
    ...     print(f"The average train accuracy is {cv_train_scores.mean():.2f}")
    ...     print(f"The average test accuracy is {cv_test_scores.mean():.2f}")
    24 samples were used to train the model
    The average train accuracy is 1.00
    The average test accuracy is 0.85
    48 samples were used to train the model
    The average train accuracy is 1.00
    The average test accuracy is 0.90
    72 samples were used to train the model
    The average train accuracy is 1.00
    The average test accuracy is 0.93
    """
    
    # 如果开启增量学习并且评估器没有 `partial_fit` 方法，则抛出 ValueError 异常
    if exploit_incremental_learning and not hasattr(estimator, "partial_fit"):
        raise ValueError(
            "An estimator must support the partial_fit interface "
            "to exploit incremental learning"
        )
    
    # 检查并更新参数组，用于适应 sklearn 版本 1.8 之后的变化
    params = _check_params_groups_deprecation(fit_params, params, groups, "1.8")
    
    # 确保输入数据 X, y, groups 可索引化
    X, y, groups = indexable(X, y, groups)
    
    # 检查并返回交叉验证生成器
    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    
    # 检查并返回评分器
    scorer = check_scoring(estimator, scoring=scoring)
    # 如果启用了路由功能
    if _routing_enabled():
        # 创建一个 MetadataRouter 实例，并指定所有者为 "learning_curve"
        router = (
            MetadataRouter(owner="learning_curve")
            .add(
                estimator=estimator,
                # TODO(SLEP6): also pass metadata to the predict method for
                # scoring?
                # 将方法映射添加到 MethodMapping 对象中，caller 为 "fit"，callee 为 "fit"
                method_mapping=MethodMapping()
                .add(caller="fit", callee="fit")
                .add(caller="fit", callee="partial_fit"),
            )
            .add(
                splitter=cv,
                # 将方法映射添加到 MethodMapping 对象中，caller 为 "fit"，callee 为 "split"
                method_mapping=MethodMapping().add(caller="fit", callee="split"),
            )
            .add(
                scorer=scorer,
                # 将方法映射添加到 MethodMapping 对象中，caller 为 "fit"，callee 为 "score"
                method_mapping=MethodMapping().add(caller="fit", callee="score"),
            )
        )

        try:
            # 尝试对路由后的参数进行处理，调用 process_routing 函数
            routed_params = process_routing(router, "fit", **params)
        except UnsetMetadataPassedError as e:
            # 如果捕获到未设置元数据传递错误 UnsetMetadataPassedError 异常
            # 定制错误消息，以适应当前情况，说明未请求的参数
            unrequested_params = sorted(e.unrequested_params)
            raise UnsetMetadataPassedError(
                message=(
                    f"{unrequested_params} are passed to `learning_curve` but are not"
                    " explicitly set as requested or not requested for learning_curve's"
                    f" estimator: {estimator.__class__.__name__}. Call"
                    " `.set_fit_request({{metadata}}=True)` on the estimator for"
                    f" each metadata in {unrequested_params} that you"
                    " want to use and `metadata=False` for not using it. See the"
                    " Metadata Routing User guide"
                    " <https://scikit-learn.org/stable/metadata_routing.html> for more"
                    " information."
                ),
                unrequested_params=e.unrequested_params,
                routed_params=e.routed_params,
            )

    else:
        # 如果未启用路由功能，则创建一个空的 Bunch 对象来存储参数
        routed_params = Bunch()
        routed_params.estimator = Bunch(fit=params, partial_fit=params)
        routed_params.splitter = Bunch(split={"groups": groups})
        routed_params.scorer = Bunch(score={})

    # 将交叉验证对象 cv 转换为列表，因为后续会多次迭代这个列表
    cv_iter = list(cv.split(X, y, **routed_params.splitter.split))

    # 计算交叉验证的第一个分组的训练样本数量
    n_max_training_samples = len(cv_iter[0][0])
    
    # 将指定的训练集大小转换为绝对数值
    train_sizes_abs = _translate_train_sizes(train_sizes, n_max_training_samples)
    
    # 计算训练集大小的唯一值数量
    n_unique_ticks = train_sizes_abs.shape[0]
    
    # 如果 verbose 大于 0，则打印训练集大小信息
    if verbose > 0:
        print("[learning_curve] Training set sizes: " + str(train_sizes_abs))
    
    # 创建并返回一个 Parallel 对象，用于并行计算
    parallel = Parallel(n_jobs=n_jobs, pre_dispatch=pre_dispatch, verbose=verbose)
    # 如果 shuffle 参数为 True，则进行数据集的随机化操作
    if shuffle:
        # 根据 random_state 参数创建随机数生成器 rng
        rng = check_random_state(random_state)
        # 使用生成器 rng 对交叉验证迭代器 cv_iter 进行随机排列
        cv_iter = ((rng.permutation(train), test) for train, test in cv_iter)

    # 如果 exploit_incremental_learning 参数为 True，则执行增量学习的策略
    if exploit_incremental_learning:
        # 如果 estimator 是分类器，则获取其唯一的类别
        classes = np.unique(y) if is_classifier(estimator) else None
        # 使用并行处理方式，对每个交叉验证折进行增量学习拟合
        out = parallel(
            delayed(_incremental_fit_estimator)(
                clone(estimator),
                X,
                y,
                classes,
                train,
                test,
                train_sizes_abs,
                scorer,
                return_times,
                error_score=error_score,
                fit_params=routed_params.estimator.partial_fit,
                score_params=routed_params.scorer.score,
            )
            for train, test in cv_iter
        )
        # 将输出转换为 NumPy 数组并调整维度
        out = np.asarray(out).transpose((2, 1, 0))
    else:
        # 如果不进行增量学习，则创建 train_test_proportions 列表用于存储所有训练-测试样本组合
        train_test_proportions = []
        for train, test in cv_iter:
            for n_train_samples in train_sizes_abs:
                train_test_proportions.append((train[:n_train_samples], test))

        # 使用并行处理方式，对每个训练-测试样本组合进行拟合和评分
        results = parallel(
            delayed(_fit_and_score)(
                clone(estimator),
                X,
                y,
                scorer=scorer,
                train=train,
                test=test,
                verbose=verbose,
                parameters=None,
                fit_params=routed_params.estimator.fit,
                score_params=routed_params.scorer.score,
                return_train_score=True,
                error_score=error_score,
                return_times=return_times,
            )
            for train, test in train_test_proportions
        )
        # 提示或者引发关于拟合失败的警告
        _warn_or_raise_about_fit_failures(results, error_score)
        # 聚合评分字典中的结果
        results = _aggregate_score_dicts(results)
        # 调整训练分数和测试分数的形状
        train_scores = results["train_scores"].reshape(-1, n_unique_ticks).T
        test_scores = results["test_scores"].reshape(-1, n_unique_ticks).T
        # 将训练分数和测试分数存储在 out 列表中
        out = [train_scores, test_scores]

        # 如果 return_times 参数为 True，则还需返回拟合时间和评分时间
        if return_times:
            fit_times = results["fit_time"].reshape(-1, n_unique_ticks).T
            score_times = results["score_time"].reshape(-1, n_unique_ticks).T
            out.extend([fit_times, score_times])

    # 将训练集样本数、训练分数、测试分数打包成结果 ret
    ret = train_sizes_abs, out[0], out[1]

    # 如果 return_times 参数为 True，则在 ret 中增加拟合时间和评分时间
    if return_times:
        ret = ret + (out[2], out[3])

    # 返回最终的结果 ret
    return ret
# 在给定的训练集大小和最大训练样本数之间转换并验证训练集大小
def _translate_train_sizes(train_sizes, n_max_training_samples):
    # 将训练集大小转换为NumPy数组
    train_sizes_abs = np.asarray(train_sizes)
    # 获取训练集大小的数量
    n_ticks = train_sizes_abs.shape[0]
    # 计算最小和最大所需样本数
    n_min_required_samples = np.min(train_sizes_abs)
    n_max_required_samples = np.max(train_sizes_abs)
    
    # 如果训练集大小是浮点数类型，则将其视为最大训练样本数的分数
    if np.issubdtype(train_sizes_abs.dtype, np.floating):
        # 如果最小值小于等于0或最大值大于1，引发值错误
        if n_min_required_samples <= 0.0 or n_max_required_samples > 1.0:
            raise ValueError(
                "train_sizes has been interpreted as fractions "
                "of the maximum number of training samples and "
                "must be within (0, 1], but is within [%f, %f]."
                % (n_min_required_samples, n_max_required_samples)
            )
        # 将训练集大小转换为绝对整数值，并限制在1到最大训练样本数之间
        train_sizes_abs = (train_sizes_abs * n_max_training_samples).astype(
            dtype=int, copy=False
        )
        train_sizes_abs = np.clip(train_sizes_abs, 1, n_max_training_samples)
    else:
        # 如果训练集大小是整数类型，则需在1到最大训练样本数之间
        if (
            n_min_required_samples <= 0
            or n_max_required_samples > n_max_training_samples
        ):
            raise ValueError(
                "train_sizes has been interpreted as absolute "
                "numbers of training samples and must be within "
                "(0, %d], but is within [%d, %d]."
                % (
                    n_max_training_samples,
                    n_min_required_samples,
                    n_max_required_samples,
                )
            )

    # 去除训练集大小中的重复项并返回唯一值数组
    train_sizes_abs = np.unique(train_sizes_abs)
    if n_ticks > train_sizes_abs.shape[0]:
        # 如果有重复项，发出警告
        warnings.warn(
            "Removed duplicate entries from 'train_sizes'. Number "
            "of ticks will be less than the size of "
            "'train_sizes': %d instead of %d." % (train_sizes_abs.shape[0], n_ticks),
            RuntimeWarning,
        )

    # 返回转换后的训练集大小数组
    return train_sizes_abs


# 逐步训练评估器模型并计算评分
def _incremental_fit_estimator(
    estimator,
    X,
    y,
    classes,
    train,
    test,
    train_sizes,
    scorer,
    return_times,
    error_score,
    fit_params,
    score_params,
):
    # 初始化四个空列表，用于存储训练得分、测试得分、拟合时间和评分时间
    train_scores, test_scores, fit_times, score_times = [], [], [], []
    
    # 将训练集的大小和训练数据分割成多个部分，并创建一个迭代器对象 partitions
    partitions = zip(train_sizes, np.split(train, train_sizes)[:-1])
    
    # 如果 fit_params 参数为 None，则将其设为空字典
    if fit_params is None:
        fit_params = {}
    
    # 如果 classes 参数为 None，则使用 partial 函数创建一个局部函数 partial_fit_func，
    # 该函数调用 estimator 的 partial_fit 方法，传入 fit_params 参数
    if classes is None:
        partial_fit_func = partial(estimator.partial_fit, **fit_params)
    else:
        # 否则，使用 partial 函数创建 partial_fit_func，
        # 该函数调用 estimator 的 partial_fit 方法，传入 classes 和 fit_params 参数
        partial_fit_func = partial(estimator.partial_fit, classes=classes, **fit_params)
    
    # 如果 score_params 不为 None，则将其赋值给 score_params；否则 score_params 为一个空字典
    score_params = score_params if score_params is not None else {}
    
    # 根据训练集的索引，检查并获取 score_params 参数的适用参数，存储在 score_params_train 中
    score_params_train = _check_method_params(X, params=score_params, indices=train)
    
    # 根据测试集的索引，检查并获取 score_params 参数的适用参数，存储在 score_params_test 中
    score_params_test = _check_method_params(X, params=score_params, indices=test)
    
    # 遍历 partitions 中的每个 n_train_samples（训练样本数）和 partial_train（部分训练数据）
    for n_train_samples, partial_train in partitions:
        # 从完整的训练集 train 中提取一个子集 train_subset
        train_subset = train[:n_train_samples]
        
        # 使用 _safe_split 函数安全地分割 estimator、X 和 y，得到训练集的子集 X_train 和 y_train
        X_train, y_train = _safe_split(estimator, X, y, train_subset)
        
        # 使用 _safe_split 函数安全地分割 estimator、X 和 y，得到部分训练集的子集 X_partial_train 和 y_partial_train
        X_partial_train, y_partial_train = _safe_split(estimator, X, y, partial_train)
        
        # 使用 _safe_split 函数安全地分割 estimator、X 和 y，得到测试集的子集 X_test 和 y_test
        X_test, y_test = _safe_split(estimator, X, y, test, train_subset)
        
        # 记录当前开始拟合时间
        start_fit = time.time()
        
        # 根据是否存在 y_partial_train，调用 partial_fit_func 函数进行拟合
        if y_partial_train is None:
            partial_fit_func(X_partial_train)
        else:
            partial_fit_func(X_partial_train, y_partial_train)
        
        # 计算拟合所花费的时间，并将其加入 fit_times 列表中
        fit_time = time.time() - start_fit
        fit_times.append(fit_time)
        
        # 记录当前开始评分时间
        start_score = time.time()
        
        # 调用 _score 函数计算并记录测试集的评分，将评分结果存入 test_scores 列表中
        test_scores.append(
            _score(
                estimator,
                X_test,
                y_test,
                scorer,
                score_params=score_params_test,
                error_score=error_score,
            )
        )
        
        # 调用 _score 函数计算并记录训练集的评分，将评分结果存入 train_scores 列表中
        train_scores.append(
            _score(
                estimator,
                X_train,
                y_train,
                scorer,
                score_params=score_params_train,
                error_score=error_score,
            )
        )
        
        # 计算评分所花费的时间，并将其加入 score_times 列表中
        score_time = time.time() - start_score
        score_times.append(score_time)
    
    # 根据 return_times 的值，决定返回的结果 ret 是包含所有时间信息的元组，还是仅包含得分信息的元组
    ret = (
        (train_scores, test_scores, fit_times, score_times)
        if return_times
        else (train_scores, test_scores)
    )
    
    # 返回包含结果的 numpy 数组，转置后的结果 ret
    return np.array(ret).T
# 使用 @validate_params 装饰器验证函数参数的有效性
@validate_params(
    {
        "estimator": [HasMethods(["fit"])],  # estimator 参数必须具有 fit 方法
        "X": ["array-like", "sparse matrix"],  # X 参数可以是数组或稀疏矩阵
        "y": ["array-like", None],  # y 参数可以是数组或者 None（用于无监督学习）
        "param_name": [str],  # param_name 参数必须是字符串
        "param_range": ["array-like"],  # param_range 参数必须是数组类型
        "groups": ["array-like", None],  # groups 参数可以是数组或者 None
        "cv": ["cv_object"],  # cv 参数必须是交叉验证对象
        "scoring": [StrOptions(set(get_scorer_names())), callable, None],  # scoring 参数可以是预定义的评分器名集合、可调用对象或者 None
        "n_jobs": [Integral, None],  # n_jobs 参数可以是整数或者 None
        "pre_dispatch": [Integral, str],  # pre_dispatch 参数可以是整数或者字符串
        "verbose": ["verbose"],  # verbose 参数必须是 verbose 类型
        "error_score": [StrOptions({"raise"}), Real],  # error_score 参数可以是 "raise" 或实数
        "fit_params": [dict, None],  # fit_params 参数可以是字典或者 None
    },
    prefer_skip_nested_validation=False,  # 不跳过嵌套验证
)
# 定义 validation_curve 函数，用于计算不同参数取值下的训练和测试得分
def validation_curve(
    estimator,
    X,
    y,
    *,
    param_name,
    param_range,
    groups=None,
    cv=None,
    scoring=None,
    n_jobs=None,
    pre_dispatch="all",
    verbose=0,
    error_score=np.nan,
    fit_params=None,
):
    """Validation curve.

    Determine training and test scores for varying parameter values.

    Compute scores for an estimator with different values of a specified
    parameter. This is similar to grid search with one parameter. However, this
    will also compute training scores and is merely a utility for plotting the
    results.

    Read more in the :ref:`User Guide <validation_curve>`.

    Parameters
    ----------
    estimator : object type that implements the "fit" method
        An object of that type which is cloned for each validation. It must
        also implement "predict" unless `scoring` is a callable that doesn't
        rely on "predict" to compute a score.

    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training vector, where `n_samples` is the number of samples and
        `n_features` is the number of features.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs) or None
        Target relative to X for classification or regression;
        None for unsupervised learning.

    param_name : str
        Name of the parameter that will be varied.

    param_range : array-like of shape (n_values,)
        The values of the parameter that will be evaluated.

    groups : array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting the dataset into
        train/test set. Only used in conjunction with a "Group" :term:`cv`
        instance (e.g., :class:`GroupKFold`).
    cv : int, cross-validation generator or an iterable, default=None
        # 定义交叉验证的参数，可以是整数、交叉验证生成器或可迭代对象，默认为None
        Determines the cross-validation splitting strategy.
        # 确定交叉验证的分割策略

        Possible inputs for cv are:
        # 可以使用以下选项作为cv的输入:

        - None, to use the default 5-fold cross validation,
        # None，使用默认的5折交叉验证

        - int, to specify the number of folds in a `(Stratified)KFold`,
        # 整数，指定在`(Stratified)KFold`中的折数

        - :term:`CV splitter`,
        # :term:`CV splitter`

        - An iterable yielding (train, test) splits as arrays of indices.
        # 产生(train, test)拆分的可迭代对象，作为索引数组

        For int/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.
        # 对于整数或None作为输入，如果评估器是分类器且`y`是二进制或多类别，则使用:class:`StratifiedKFold`。在其他情况下，使用:class:`KFold`。这些分割器被实例化为`shuffle=False`，因此在调用时拆分将保持一致。

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.
        # 参考 :ref:`用户指南 <cross_validation>` 获取这里可以使用的各种交叉验证策略的详细信息。

        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.
            # 如果`cv`为None，则默认值从3折更改为5折。

    scoring : str or callable, default=None
        # 评分参数，可以是字符串或可调用的评分函数，或具有`scorer(estimator, X, y)`签名的对象/函数，默认为None

    n_jobs : int, default=None
        # 并行运行的作业数，默认为None
        Number of jobs to run in parallel. Training the estimator and computing
        the score are parallelized over the combinations of each parameter
        value and each cross-validation split.
        # 并行运行的作业数。通过每个参数值和每个交叉验证拆分的组合并行训练评估器和计算得分。

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        # `None` 表示1，除非在 :obj:`joblib.parallel_backend` 上下文中

        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
        # `-1` 表示使用所有处理器。更多详情请见 :term:`术语表 <n_jobs>`

    pre_dispatch : int or str, default='all'
        # 预派遣的作业数，用于并行执行的数量，默认为'all'。此选项可以减少分配的内存。字符串可以是类似于'2*n_jobs'的表达式。

    verbose : int, default=0
        # 控制详细程度的参数：数值越高，消息越多。

    error_score : 'raise' or numeric, default=np.nan
        # 如果在评估器拟合时发生错误，分数的值。如果设置为'raise'，则会引发错误。如果给出数值，则会引发FitFailedWarning。

        .. versionadded:: 0.20
            # 添加于版本0.20

    fit_params : dict, default=None
        # 传递给评估器的fit方法的参数。

        .. versionadded:: 0.24
            # 添加于版本0.24

    Returns
    -------
    train_scores : array of shape (n_ticks, n_cv_folds)
        # 训练集上的分数数组，形状为(n_ticks, n_cv_folds)

    test_scores : array of shape (n_ticks, n_cv_folds)
        # 测试集上的分数数组，形状为(n_ticks, n_cv_folds)

    Notes
    -----
    # 注意事项
    See :ref:`sphx_glr_auto_examples_model_selection_plot_validation_curve.py`
    # 参见 :ref:`sphx_glr_auto_examples_model_selection_plot_validation_curve.py` 的示例

    Examples
    --------
    # 示例
    >>> import numpy as np
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import validation_curve
    >>> from sklearn.linear_model import LogisticRegression
    >>> X, y = make_classification(n_samples=1_000, random_state=0)
    >>> logistic_regression = LogisticRegression()
    >>> param_name, param_range = "C", np.logspace(-8, 3, 10)
    # 调用 validation_curve 函数进行交叉验证曲线计算，获取训练集和测试集的得分
    train_scores, test_scores = validation_curve(
        logistic_regression, X, y, param_name=param_name, param_range=param_range
    )
    
    # 打印训练集的平均准确率
    print(f"The average train accuracy is {train_scores.mean():.2f}")
    
    # 打印测试集的平均准确率
    print(f"The average test accuracy is {test_scores.mean():.2f}")
    """
    X, y, groups = indexable(X, y, groups)
    
    # 确保数据集可以被索引，返回索引后的数据集和分组信息
    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    
    # 检查并返回适用于估计器的评分器
    scorer = check_scoring(estimator, scoring=scoring)
    
    # 并行计算，根据指定参数创建并行对象
    parallel = Parallel(n_jobs=n_jobs, pre_dispatch=pre_dispatch, verbose=verbose)
    
    # 并行执行拟合和评分，并返回结果
    results = parallel(
        delayed(_fit_and_score)(
            clone(estimator),
            X,
            y,
            scorer=scorer,
            train=train,
            test=test,
            verbose=verbose,
            parameters={param_name: v},
            fit_params=fit_params,
            # TODO(SLEP6): 在这里支持评分参数
            score_params=None,
            return_train_score=True,
            error_score=error_score,
        )
        # 注意：不改变迭代顺序以允许单次 CV 分割
        for train, test in cv.split(X, y, groups)  # 在数据集上划分训练集和测试集
        for v in param_range  # 遍历参数范围
    )
    n_params = len(param_range)
    
    # 聚合评分结果字典
    results = _aggregate_score_dicts(results)
    
    # 重新组织训练集和测试集的得分矩阵
    train_scores = results["train_scores"].reshape(-1, n_params).T
    test_scores = results["test_scores"].reshape(-1, n_params).T
    
    # 返回训练集和测试集的得分矩阵
    return train_scores, test_scores
def _aggregate_score_dicts(scores):
    """Aggregate the list of dict to dict of np ndarray

    The aggregated output of _aggregate_score_dicts will be a list of dict
    of form [{'prec': 0.1, 'acc':1.0}, {'prec': 0.1, 'acc':1.0}, ...]
    Convert it to a dict of array {'prec': np.array([0.1 ...]), ...}

    Parameters
    ----------

    scores : list of dict
        List of dicts of the scores for all scorers. This is a flat list,
        assumed originally to be of row major order.

    Example
    -------

    >>> scores = [{'a': 1, 'b':10}, {'a': 2, 'b':2}, {'a': 3, 'b':3},
    ...           {'a': 10, 'b': 10}]                         # doctest: +SKIP
    >>> _aggregate_score_dicts(scores)                        # doctest: +SKIP
    {'a': array([1, 2, 3, 10]),
     'b': array([10, 2, 3, 10])}
    """
    # 返回一个字典，其键为输入字典列表中的每个键，值为该键对应的所有值组成的 ndarray 或列表
    return {
        key: (
            np.asarray([score[key] for score in scores])  # 如果值是数值型，则转换为 ndarray
            if isinstance(scores[0][key], numbers.Number)  # 检查第一个字典的键对应的值是否是数值型
            else [score[key] for score in scores]  # 如果不是数值型，则保持为列表
        )
        for key in scores[0]  # 遍历第一个字典的所有键
    }
```