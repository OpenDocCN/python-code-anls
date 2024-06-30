# `D:\src\scipysrc\scikit-learn\sklearn\linear_model\_ransac.py`

```
    """RANSAC (RANdom SAmple Consensus) 算法。

    RANSAC 是一种迭代算法，用于从完整数据集的子集中稳健地估计参数。

    在 :ref:`用户指南 <ransac_regression>` 中可以详细了解。

    Parameters
    ----------
    base_estimator : object, optional, default=None
        用于拟合数据的基本估计器对象。如果为 None，则默认为线性回归 (LinearRegression)。

    min_samples : int, float, optional, default=None
        每次迭代时从完整数据集中选择的最小样本数。如果是 int，则直接使用此数量的样本；如果是 float，它表示样本的比例。

    residual_threshold : float, optional, default=None
        用于将数据点分为内点和外点的残差阈值。残差小于此阈值的点将被视为内点。

    max_trials : int, optional, default=None
        最大迭代次数。如果设置，算法将尝试 max_trials 次采样。

    max_skips : int, optional, default=inf
        在没有找到更好模型之前允许的最大连续跳过次数。

    stop_n_inliers : int, optional, default=None
        当内点的数量达到此值时，算法将停止迭代。

    stop_score : float, optional, default=None
        当模型得分（如 R^2 分数）达到此值时，算法将停止迭代。

    residual_metric : {'absolute_loss', 'squared_loss'}, default='absolute_loss'
        用于计算残差的度量方式。'absolute_loss' 使用绝对误差，'squared_loss' 使用平方误差。

    random_state : int, RandomState instance or None, optional, default=None
        控制随机性的种子值或随机状态实例。

    min_inliers_ratio : float, optional, default=None
        在每次迭代中至少选择的内点比例。

    stop_probability : float, optional, default=0.99
        停止迭代的概率（置信度）。如果在 max_trials 次迭代中未找到更好的模型，则根据此概率停止。

    residual_threshold_mode : {'percentile', 'absolute'}, default='percentile'
        残差阈值的模式。'percentile' 根据残差的分位数确定阈值，'absolute' 使用绝对值阈值。

    loss : {'absolute_loss', 'squared_loss'}, default='absolute_loss'
        用于优化的损失函数。'absolute_loss' 使用绝对损失，'squared_loss' 使用平方损失。

    max_skips_no_better : int, optional, default=inf
        在没有找到更好模型之前允许的最大连续跳过次数。

    is_data_valid : callable, optional, default=None
        用于验证数据点是否有效的函数。

    stop_score_factor : float, optional, default=1.0
        停止评分的因子。用于乘以当前模型的评分以确定是否停止迭代。

    Attributes
    ----------
    estimator_ : object
        最佳模型估计器对象。

    inlier_mask_ : bool array
        指示每个样本是否为内点的布尔掩码。

    n_trials_ : int
        完成的迭代次数。

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import RANSACRegressor
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=100, n_features=1, noise=4.0, random_state=0)
    >>> reg = RANSACRegressor(random_state=0).fit(X, y)
    >>> reg.score(X, y)
    0.8406...

    References
    ----------
    Fischler, M.A. and Bolles, R.C., 1981. Random sample consensus: A paradigm for model fitting with applications to image analysis and automated cartography. Communications of the ACM, 24(6), pp.381-395.

    """
    estimator : object, default=None
        # 基础估算器对象，需实现以下方法：

         * `fit(X, y)`: 将模型拟合给定的训练数据和目标值。
         * `score(X, y)`: 返回给定测试数据的平均准确率，用于由 `stop_score` 定义的停止准则。
           此外，该分数用于决定两个大小相等的共识集中哪一个更好。
         * `predict(X)`: 使用线性模型返回预测值，用于使用损失函数计算残差误差。

        如果 `estimator` 为 None，则默认使用 :class:`~sklearn.linear_model.LinearRegression` 用于
        目标值类型为浮点数。

        注意，当前实现仅支持回归估算器。

    min_samples : int (>= 1) or float ([0, 1]), default=None
        # 从原始数据中随机选择的最小样本数。对于 `min_samples >= 1`，作为样本的绝对数目处理，
        对于 `min_samples < 1`，作为 `ceil(min_samples * X.shape[0])` 的相对数目处理。
        这通常被选择为估算给定 `estimator` 所需的最小样本数。默认情况下，假定使用 :class:`~sklearn.linear_model.LinearRegression` 估算器，并选择 `min_samples` 为 ``X.shape[1] + 1``。
        此参数高度依赖于模型，因此如果使用除 :class:`~sklearn.linear_model.LinearRegression` 之外的 `estimator`，用户必须提供一个值。

    residual_threshold : float, default=None
        # 数据样本被分类为内点的最大残差。默认情况下，阈值被选择为目标值 `y` 的中位数绝对偏差（MAD）。
        其残差严格等于阈值的点被视为内点。

    is_data_valid : callable, default=None
        # 在模型拟合之前，随机选择的数据调用此函数：`is_data_valid(X, y)`。
        如果其返回值为 False，则跳过当前随机选择的子样本。

    is_model_valid : callable, default=None
        # 使用估算的模型和随机选择的数据调用此函数：`is_model_valid(model, X, y)`。
        如果其返回值为 False，则跳过当前随机选择的子样本。
        与 `is_data_valid` 相比，使用此函数拒绝样本的计算成本更高。
        因此，只有在需要使用估算的模型来进行拒绝决策时才应使用 `is_model_valid`。

    max_trials : int, default=100
        # 随机样本选择的最大迭代次数。
    max_skips : int, default=np.inf
        # 最大迭代跳过次数，可以跳过因找到零内点或由 ``is_data_valid`` 或 ``is_model_valid`` 定义的无效数据或模型所导致的迭代次数。
        # 默认值为无穷大（np.inf）。

        .. versionadded:: 0.19
        # 添加于版本 0.19。

    stop_n_inliers : int, default=np.inf
        # 如果找到至少这么多的内点，则停止迭代。

    stop_score : float, default=np.inf
        # 如果得分大于等于此阈值，则停止迭代。

    stop_probability : float in range [0, 1], default=0.99
        # 如果在 RANSAC 中采样到至少一个无异常值的训练数据集，则停止迭代。
        # 这需要生成至少 N 个样本（迭代次数）：
        # N >= log(1 - probability) / log(1 - e**m)
        # 其中概率（置信度）通常设置为高值，如 0.99（默认值），e 是当前内点的分数与总样本数的比率。

    loss : str, callable, default='absolute_error'
        # 支持字符串输入 'absolute_error' 和 'squared_error'，分别计算每个样本的绝对误差和平方误差。
        # 如果 ``loss`` 是可调用的，则应该是一个接受两个数组作为输入的函数，真实值和预测值，并返回一个 1-D 数组，其中第 i 个值对应于 ``X[i]`` 的损失值。
        # 如果样本的损失大于 ``residual_threshold``，则将其分类为异常值。

        .. versionadded:: 0.18
        # 添加于版本 0.18。

    random_state : int, RandomState instance, default=None
        # 用于初始化中心的随机数生成器。
        # 传递一个整数以在多次函数调用之间获得可重复的输出。
        # 参见 :term:`Glossary <random_state>`。

    Attributes
    ----------
    estimator_ : object
        # 最佳拟合的模型对象的副本。

    n_trials_ : int
        # 随机选择试验次数，直到满足一个停止条件为止。
        # 总是 ``<= max_trials``。

    inlier_mask_ : bool array of shape [n_samples]
        # 内点的布尔掩码，分类为 ``True``。

    n_skips_no_inliers_ : int
        # 因找到零内点而跳过的迭代次数。

        .. versionadded:: 0.19
        # 添加于版本 0.19。

    n_skips_invalid_data_ : int
        # 因 ``is_data_valid`` 定义的无效数据而跳过的迭代次数。

        .. versionadded:: 0.19
        # 添加于版本 0.19。

    n_skips_invalid_model_ : int
        # 因 ``is_model_valid`` 定义的无效模型而跳过的迭代次数。

        .. versionadded:: 0.19
        # 添加于版本 0.19。

    n_features_in_ : int
        # 在 :term:`fit` 过程中看到的特征数。

        .. versionadded:: 0.24
        # 添加于版本 0.24。

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        # 在 :term:`fit` 过程中看到的特征名称。
        # 仅当 `X` 的特征名称全部为字符串时定义。

        .. versionadded:: 1.0
        # 添加于版本 1.0。

    See Also
    --------
    # 定义参数约束字典，用于描述 RANSACRegressor 的参数限制和类型要求
    _parameter_constraints: dict = {
        # 估计器参数要求：应具有 fit、score、predict 方法，或者为 None
        "estimator": [HasMethods(["fit", "score", "predict"]), None],
        # 最小样本数要求：整数类型，至少为 1
        "min_samples": [
            Interval(Integral, 1, None, closed="left"),
            # 或者为非整数实数，范围在 0 到 1 之间
            Interval(RealNotInt, 0, 1, closed="both"),
            None,
        ],
        # 残差阈值要求：实数类型，至少为 0
        "residual_threshold": [Interval(Real, 0, None, closed="left"), None],
        # 数据有效性检查函数要求：可调用对象或者为 None
        "is_data_valid": [callable, None],
        # 模型有效性检查函数要求：可调用对象或者为 None
        "is_model_valid": [callable, None],
        # 最大迭代次数要求：整数类型，至少为 0
        "max_trials": [
            Interval(Integral, 0, None, closed="left"),
            # 或者为实数，可以是 np.inf
            Options(Real, {np.inf}),
        ],
        # 最大跳过次数要求：整数类型，至少为 0
        "max_skips": [
            Interval(Integral, 0, None, closed="left"),
            # 或者为实数，可以是 np.inf
            Options(Real, {np.inf}),
        ],
        # 停止时的最小内点数要求：整数类型，至少为 0
        "stop_n_inliers": [
            Interval(Integral, 0, None, closed="left"),
            # 或者为实数，可以是 np.inf
            Options(Real, {np.inf}),
        ],
        # 停止时的分数阈值要求：实数类型，无特定范围要求
        "stop_score": [Interval(Real, None, None, closed="both")],
        # 停止时的概率阈值要求：实数类型，在 0 到 1 之间
        "stop_probability": [Interval(Real, 0, 1, closed="both")],
        # 损失函数类型要求：字符串，可以是 "absolute_error" 或 "squared_error"，或者是可调用对象
        "loss": [StrOptions({"absolute_error", "squared_error"}), callable],
        # 随机数种子要求：随机数种子对象或者 "random_state"
        "random_state": ["random_state"],
    }
    
    def __init__(
        self,
        estimator=None,
        *,
        min_samples=None,
        residual_threshold=None,
        is_data_valid=None,
        is_model_valid=None,
        max_trials=100,
        max_skips=np.inf,
        stop_n_inliers=np.inf,
        stop_score=np.inf,
        stop_probability=0.99,
        loss="absolute_error",
        random_state=None,
    ):
        # 初始化 RANSACRegressor 对象的参数
        self.estimator = estimator
        self.min_samples = min_samples
        self.residual_threshold = residual_threshold
        self.is_data_valid = is_data_valid
        self.is_model_valid = is_model_valid
        self.max_trials = max_trials
        self.max_skips = max_skips
        self.stop_n_inliers = stop_n_inliers
        self.stop_score = stop_score
        self.stop_probability = stop_probability
        self.random_state = random_state
        self.loss = loss
    
    # 应用装饰器函数 _fit_context，用于 RANSACRegressor 的拟合过程
    @_fit_context(
        prefer_skip_nested_validation=False
    )
    # TODO(1.7): remove `sample_weight` from the signature after deprecation
    # cycle; for backwards compatibility: pop it from `fit_params` before the
    # `_raise_for_params` check and reinsert it after the check
    @_deprecate_positional_args(version="1.7")
    # 使用装饰器将下面的方法标记为在版本1.7中已弃用的位置参数
    def predict(self, X, **params):
        """Predict using the estimated model.

        This is a wrapper for `estimator_.predict(X)`.

        Parameters
        ----------
        X : {array-like or sparse matrix} of shape (n_samples, n_features)
            Input data.

        **params : dict
            Parameters routed to the `predict` method of the sub-estimator via
            the metadata routing API.

            .. versionadded:: 1.5

                Only available if
                `sklearn.set_config(enable_metadata_routing=True)` is set. See
                :ref:`Metadata Routing User Guide <metadata_routing>` for more
                details.

        Returns
        -------
        y : array, shape = [n_samples] or [n_samples, n_targets]
            Returns predicted values.
        """
        # 检查模型是否已拟合
        check_is_fitted(self)
        # 验证输入数据是否有效，并根据参数设置进行验证
        X = self._validate_data(
            X,
            force_all_finite=False,
            accept_sparse=True,
            reset=False,
        )

        # 检查参数是否有效
        _raise_for_params(params, self, "predict")

        # 如果启用了路由功能，则处理路由参数，否则为空字典
        if _routing_enabled():
            predict_params = process_routing(self, "predict", **params).estimator[
                "predict"
            ]
        else:
            predict_params = {}

        # 调用估算器的预测方法，并传递验证后的数据和处理后的参数
        return self.estimator_.predict(X, **predict_params)

    def score(self, X, y, **params):
        """Return the score of the prediction.

        This is a wrapper for `estimator_.score(X, y)`.

        Parameters
        ----------
        X : (array-like or sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        **params : dict
            Parameters routed to the `score` method of the sub-estimator via
            the metadata routing API.

            .. versionadded:: 1.5

                Only available if
                `sklearn.set_config(enable_metadata_routing=True)` is set. See
                :ref:`Metadata Routing User Guide <metadata_routing>` for more
                details.

        Returns
        -------
        z : float
            Score of the prediction.
        """
        # 检查模型是否已拟合
        check_is_fitted(self)
        # 验证输入数据是否有效，并根据参数设置进行验证
        X = self._validate_data(
            X,
            force_all_finite=False,
            accept_sparse=True,
            reset=False,
        )

        # 检查参数是否有效
        _raise_for_params(params, self, "score")

        # 如果启用了路由功能，则处理路由参数，否则为空字典
        if _routing_enabled():
            score_params = process_routing(self, "score", **params).estimator["score"]
        else:
            score_params = {}

        # 调用估算器的评分方法，并传递验证后的数据、目标值和处理后的参数
        return self.estimator_.score(X, y, **score_params)
    def get_metadata_routing(self):
        """
        Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        .. versionadded:: 1.5

        Returns
        -------
        routing : MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        # 创建一个 MetadataRouter 对象，用于处理元数据路由
        router = MetadataRouter(owner=self.__class__.__name__).add(
            # 向路由器添加不同方法的调用映射
            estimator=self.estimator,
            method_mapping=MethodMapping()
            .add(caller="fit", callee="fit")
            .add(caller="fit", callee="score")
            .add(caller="score", callee="score")
            .add(caller="predict", callee="predict"),
        )
        # 返回路由器对象，其中包含了设置的元数据路由信息
        return router

    def _more_tags(self):
        # 返回一个字典，该字典包含了一些额外的标签信息，用于测试和验证
        return {
            "_xfail_checks": {
                "check_sample_weights_invariance": (
                    "zero sample_weight is not equivalent to removing samples"
                ),
            }
        }
```