# `D:\src\scipysrc\scikit-learn\sklearn\isotonic.py`

```
@validate_params(
    {
        "x": ["array-like"],
        "y": ["array-like"],
    },
    prefer_skip_nested_validation=True,
)
"""验证输入参数x和y的格式是否正确，并允许跳过嵌套验证。

Parameters
----------
x : array-like of shape (n_samples,)
        训练数据。

y : array-like of shape (n_samples,)
    训练目标。

Returns
-------
increasing_bool : boolean
    表示关系是增加还是减少的布尔值。

Notes
-----
从数据中估计Spearman相关系数，并使用结果的符号作为返回值。

如果基于Fisher变换的95%置信区间跨越零，将发出警告。

References
----------
Fisher transformation. Wikipedia.
https://en.wikipedia.org/wiki/Fisher_transformation
"""
def check_increasing(x, y):
    """Determine whether y is monotonically correlated with x.

    y is found increasing or decreasing with respect to x based on a Spearman
    correlation test.

    Parameters
    ----------
    x : array-like of shape (n_samples,)
            Training data.

    y : array-like of shape (n_samples,)
        Training target.

    Returns
    -------
    increasing_bool : boolean
        Whether the relationship is increasing or decreasing.

    Notes
    -----
    The Spearman correlation coefficient is estimated from the data, and the
    sign of the resulting estimate is used as the result.

    In the event that the 95% confidence interval based on Fisher transform
    spans zero, a warning is raised.

    References
    ----------
    Fisher transformation. Wikipedia.
    https://en.wikipedia.org/wiki/Fisher_transformation

    Examples
    --------
    >>> from sklearn.isotonic import check_increasing
    >>> x, y = [1, 2, 3, 4, 5], [2, 4, 6, 8, 10]
    >>> check_increasing(x, y)
    True
    >>> y = [10, 8, 6, 4, 2]
    >>> check_increasing(x, y)
    False
    """

    # Calculate Spearman rho estimate and set return accordingly.
    rho, _ = spearmanr(x, y)
    increasing_bool = rho >= 0

    # Run Fisher transform to get the rho CI, but handle rho=+/-1
    if rho not in [-1.0, 1.0] and len(x) > 3:
        F = 0.5 * math.log((1.0 + rho) / (1.0 - rho))
        F_se = 1 / math.sqrt(len(x) - 3)

        # Use a 95% CI, i.e., +/-1.96 S.E.
        # https://en.wikipedia.org/wiki/Fisher_transformation
        rho_0 = math.tanh(F - 1.96 * F_se)
        rho_1 = math.tanh(F + 1.96 * F_se)

        # Warn if the CI spans zero.
        if np.sign(rho_0) != np.sign(rho_1):
            warnings.warn(
                "Confidence interval of the Spearman "
                "correlation coefficient spans zero. "
                "Determination of ``increasing`` may be "
                "suspect."
            )

    return increasing_bool
    {
        # "y": 参数，接受类数组对象作为输入
        "y": ["array-like"],
        # "sample_weight": 可选参数，接受类数组对象或者 None 作为输入
        "sample_weight": ["array-like", None],
        # "y_min": 可选参数，接受实数域范围的闭区间作为输入，左右都包含
        "y_min": [Interval(Real, None, None, closed="both"), None],
        # "y_max": 可选参数，接受实数域范围的闭区间作为输入，左右都包含
        "y_max": [Interval(Real, None, None, closed="both"), None],
        # "increasing": 参数，接受布尔值作为输入
        "increasing": ["boolean"],
    },
    # prefer_skip_nested_validation=True，偏好跳过嵌套验证设置为真
    prefer_skip_nested_validation=True,
)
def isotonic_regression(
    y, *, sample_weight=None, y_min=None, y_max=None, increasing=True
):
    """Solve the isotonic regression model.

    Read more in the :ref:`User Guide <isotonic>`.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        The data.

    sample_weight : array-like of shape (n_samples,), default=None
        Weights on each point of the regression.
        If None, weight is set to 1 (equal weights).

    y_min : float, default=None
        Lower bound on the lowest predicted value (the minimum value may
        still be higher). If not set, defaults to -inf.

    y_max : float, default=None
        Upper bound on the highest predicted value (the maximum may still be
        lower). If not set, defaults to +inf.

    increasing : bool, default=True
        Whether to compute ``y_`` is increasing (if set to True) or decreasing
        (if set to False).

    Returns
    -------
    y_ : ndarray of shape (n_samples,)
        Isotonic fit of y.

    References
    ----------
    "Active set algorithms for isotonic regression; A unifying framework"
    by Michael J. Best and Nilotpal Chakravarti, section 3.

    Examples
    --------
    >>> from sklearn.isotonic import isotonic_regression
    >>> isotonic_regression([5, 3, 1, 2, 8, 10, 7, 9, 6, 4])
    array([2.75   , 2.75   , 2.75   , 2.75   , 7.33...,
           7.33..., 7.33..., 7.33..., 7.33..., 7.33...])
    """
    # 使用 check_array 函数确保 y 是正确的数据类型和形状
    y = check_array(y, ensure_2d=False, input_name="y", dtype=[np.float64, np.float32])
    # 检查 SciPy 的版本是否符合要求
    if sp_base_version >= parse_version("1.12.0"):
        # 使用 optimize.isotonic_regression 函数进行等保回归计算
        res = optimize.isotonic_regression(
            y=y, weights=sample_weight, increasing=increasing
        )
        # 将结果转换为与 y 相同类型的 ndarray
        y = np.asarray(res.x, dtype=y.dtype)
    else:
        # 如果 SciPy 版本不符合要求，则使用旧版本的处理方式
        # 根据 increasing 参数决定升序还是降序排序
        order = np.s_[:] if increasing else np.s_[::-1]
        y = np.array(y[order], dtype=y.dtype)
        # 检查和复制样本权重数据
        sample_weight = _check_sample_weight(sample_weight, y, dtype=y.dtype, copy=True)
        # 转换样本权重数据为连续的 ndarray
        sample_weight = np.ascontiguousarray(sample_weight[order])
        # 使用 _inplace_contiguous_isotonic_regression 函数进行等保回归计算
        _inplace_contiguous_isotonic_regression(y, sample_weight)
        # 恢复原始顺序
        y = y[order]

    # 如果指定了 y_min 或 y_max，则对 y 进行截断处理
    if y_min is not None or y_max is not None:
        # 对 y 进行截断处理，使用 np.clip 函数
        # 旧版本的 np.clip 不接受 None 作为边界，所以使用 np.inf 代替
        if y_min is None:
            y_min = -np.inf
        if y_max is None:
            y_max = np.inf
        np.clip(y, y_min, y_max, y)
    
    # 返回经过等保回归处理后的 y
    return y


class IsotonicRegression(RegressorMixin, TransformerMixin, BaseEstimator):
    """Isotonic regression model.

    Read more in the :ref:`User Guide <isotonic>`.

    .. versionadded:: 0.13

    Parameters
    ----------
    y_min : float, default=None
        Lower bound on the lowest predicted value (the minimum value may
        still be higher). If not set, defaults to -inf.
    # y_max : float, default=None
    #     Upper bound on the highest predicted value (the maximum may still be
    #     lower). If not set, defaults to +inf.

    # increasing : bool or 'auto', default=True
    #     Determines whether the predictions should be constrained to increase
    #     or decrease with `X`. 'auto' will decide based on the Spearman
    #     correlation estimate's sign.

    # out_of_bounds : {'nan', 'clip', 'raise'}, default='nan'
    #     Handles how `X` values outside of the training domain are handled
    #     during prediction.
    #
    #     - 'nan', predictions will be NaN.
    #     - 'clip', predictions will be set to the value corresponding to
    #       the nearest train interval endpoint.
    #     - 'raise', a `ValueError` is raised.

    # Attributes
    # ----------
    # X_min_ : float
    #     Minimum value of input array `X_` for left bound.

    # X_max_ : float
    #     Maximum value of input array `X_` for right bound.

    # X_thresholds_ : ndarray of shape (n_thresholds,)
    #     Unique ascending `X` values used to interpolate
    #     the y = f(X) monotonic function.
    #
    #     .. versionadded:: 0.24

    # y_thresholds_ : ndarray of shape (n_thresholds,)
    #     De-duplicated `y` values suitable to interpolate the y = f(X)
    #     monotonic function.
    #
    #     .. versionadded:: 0.24

    # f_ : function
    #     The stepwise interpolating function that covers the input domain ``X``.

    # increasing_ : bool
    #     Inferred value for ``increasing``.

    # See Also
    # --------
    # sklearn.linear_model.LinearRegression : Ordinary least squares Linear
    #     Regression.
    # sklearn.ensemble.HistGradientBoostingRegressor : Gradient boosting that
    #     is a non-parametric model accepting monotonicity constraints.
    # isotonic_regression : Function to solve the isotonic regression model.

    # Notes
    # -----
    # Ties are broken using the secondary method from de Leeuw, 1977.

    # References
    # ----------
    # Isotonic Median Regression: A Linear Programming Approach
    # Nilotpal Chakravarti
    # Mathematics of Operations Research
    # Vol. 14, No. 2 (May, 1989), pp. 303-308

    # Isotone Optimization in R : Pool-Adjacent-Violators
    # Algorithm (PAVA) and Active Set Methods
    # de Leeuw, Hornik, Mair
    # Journal of Statistical Software 2009

    # Correctness of Kruskal's algorithms for monotone regression with ties
    # de Leeuw, Psychometrica, 1977

    # Examples
    # --------
    # >>> from sklearn.datasets import make_regression
    # >>> from sklearn.isotonic import IsotonicRegression
    # >>> X, y = make_regression(n_samples=10, n_features=1, random_state=41)
    # >>> iso_reg = IsotonicRegression().fit(X, y)
    # >>> iso_reg.predict([.1, .2])
    # array([1.8628..., 3.7256...])
    # 定义参数约束字典，指定每个参数的类型约束和默认值
    _parameter_constraints: dict = {
        "y_min": [Interval(Real, None, None, closed="both"), None],
        "y_max": [Interval(Real, None, None, closed="both"), None],
        "increasing": ["boolean", StrOptions({"auto"})],
        "out_of_bounds": [StrOptions({"nan", "clip", "raise"})],
    }

    # 初始化函数，设置类的初始属性
    def __init__(self, *, y_min=None, y_max=None, increasing=True, out_of_bounds="nan"):
        self.y_min = y_min  # 设置最小值属性
        self.y_max = y_max  # 设置最大值属性
        self.increasing = increasing  # 设置是否递增属性
        self.out_of_bounds = out_of_bounds  # 设置越界处理方式属性

    # 检查输入数据形状的私有方法
    def _check_input_data_shape(self, X):
        # 如果输入数据 X 不是一维数组，也不是二维数组且第二维长度不为1，则抛出数值错误
        if not (X.ndim == 1 or (X.ndim == 2 and X.shape[1] == 1)):
            msg = (
                "Isotonic regression input X should be a 1d array or "
                "2d array with 1 feature"
            )
            raise ValueError(msg)

    # 构建 f_ interp1d 函数的私有方法
    def _build_f(self, X, y):
        """Build the f_ interp1d function."""
        
        # 根据参数设置是否在越界时抛出错误
        bounds_error = self.out_of_bounds == "raise"
        
        # 如果 y 的长度为1，则使用常数预测函数
        if len(y) == 1:
            self.f_ = lambda x: y.repeat(x.shape)  # 设置 f_ 为返回常数的 lambda 函数
        else:
            # 否则使用线性插值创建 f_
            self.f_ = interpolate.interp1d(
                X, y, kind="linear", bounds_error=bounds_error
            )
    def _build_y(self, X, y, sample_weight, trim_duplicates=True):
        """Build the y_ IsotonicRegression."""
        # 检查输入数据的形状是否符合要求
        self._check_input_data_shape(X)
        # 将输入数据 X 转换为一维数组视图
        X = X.reshape(-1)  # use 1d view

        # 如果需要自动确定是否递增
        if self.increasing == "auto":
            # 检查并确定是否递增
            self.increasing_ = check_increasing(X, y)
        else:
            self.increasing_ = self.increasing

        # 如果传入了样本权重，移除权重为零的数据点并进行排序
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        mask = sample_weight > 0
        X, y, sample_weight = X[mask], y[mask], sample_weight[mask]

        # 根据 y 和 X 的排序顺序重新排列 X, y, sample_weight
        order = np.lexsort((y, X))
        X, y, sample_weight = [array[order] for array in [X, y, sample_weight]]
        # 去除重复的 X, y, sample_weight
        unique_X, unique_y, unique_sample_weight = _make_unique(X, y, sample_weight)

        X = unique_X
        # 使用保序回归方法计算 y
        y = isotonic_regression(
            unique_y,
            sample_weight=unique_sample_weight,
            y_min=self.y_min,
            y_max=self.y_max,
            increasing=self.increasing_,
        )

        # 处理 X 的左右边界
        self.X_min_, self.X_max_ = np.min(X), np.max(X)

        if trim_duplicates:
            # 移除不必要的点以加快预测速度
            keep_data = np.ones((len(y),), dtype=bool)
            # 除了第一个和最后一个点外，移除那些其 y 值等于前后两个点的点
            keep_data[1:-1] = np.logical_or(
                np.not_equal(y[1:-1], y[:-2]), np.not_equal(y[1:-1], y[2:])
            )
            return X[keep_data], y[keep_data]
        else:
            # 可以选择关闭 trim_duplicates，用于更轻松地进行单元测试，验证删除 y 中的重复点不会对插值函数产生影响（除了预测速度）
            return X, y

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None):
        """Fit the model using X, y as training data.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, 1)
            Training data.

            .. versionchanged:: 0.24
               Also accepts 2d array with 1 feature.

        y : array-like of shape (n_samples,)
            Training target.

        sample_weight : array-like of shape (n_samples,), default=None
            Weights. If set to None, all weights will be set to 1 (equal
            weights).

        Returns
        -------
        self : object
            Returns an instance of self.

        Notes
        -----
        X is stored for future use, as :meth:`transform` needs X to interpolate
        new input data.
        """
        # 定义参数检查的字典
        check_params = dict(accept_sparse=False, ensure_2d=False)
        # 对输入的特征数据 X 进行参数检查和转换，确保数据类型和形状正确
        X = check_array(
            X, input_name="X", dtype=[np.float64, np.float32], **check_params
        )
        # 对目标数据 y 进行参数检查和转换，确保数据类型和形状正确
        y = check_array(y, input_name="y", dtype=X.dtype, **check_params)
        # 检查特征数据 X、目标数据 y 和样本权重 sample_weight 的长度是否一致
        check_consistent_length(X, y, sample_weight)

        # 使用内部方法 _build_y 对特征数据 X 和目标数据 y 进行预处理
        # 运行保序回归算法，并相应地转换特征数据 X 和目标数据 y
        X, y = self._build_y(X, y, sample_weight)

        # 为了支持模型持久化，需要存储训练集的非冗余部分
        # 使用 pickle 模块存储对象时，由于 scipy.interp1d 对象无法直接序列化，
        # 需要存储模型的非冗余部分
        self.X_thresholds_, self.y_thresholds_ = X, y

        # 构建插值函数
        self._build_f(X, y)
        return self

    def _transform(self, T):
        """`_transform` is called by both `transform` and `predict` methods.

        Since `transform` is wrapped to output arrays of specific types (e.g.
        NumPy arrays, pandas DataFrame), we cannot make `predict` call `transform`
        directly.

        The above behaviour could be changed in the future, if we decide to output
        other type of arrays when calling `predict`.
        """
        # 如果模型已经拟合过，获取特征数据 X 的数据类型
        if hasattr(self, "X_thresholds_"):
            dtype = self.X_thresholds_.dtype
        else:
            dtype = np.float64

        # 对输入数据 T 进行类型检查和可能的转换，确保数据类型正确
        T = check_array(T, dtype=dtype, ensure_2d=False)

        # 检查输入数据 T 的形状是否符合预期
        self._check_input_data_shape(T)
        # 将输入数据 T 转换为一维数组的视图
        T = T.reshape(-1)

        # 如果模型设置了超出边界后的处理方式为剪切，则对数据进行剪切处理
        if self.out_of_bounds == "clip":
            T = np.clip(T, self.X_min_, self.X_max_)

        # 使用保存的插值函数对数据进行转换
        res = self.f_(T)

        # 对于 scipy 版本为 0.17，interp1d 函数返回的结果类型可能会自动提升为 float64，这里进行类型转换
        res = res.astype(T.dtype)

        return res
    def transform(self, T):
        """Transform new data by linear interpolation.

        Parameters
        ----------
        T : array-like of shape (n_samples,) or (n_samples, 1)
            Data to transform.

            .. versionchanged:: 0.24
               Also accepts 2d array with 1 feature.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The transformed data.
        """
        return self._transform(T)



    def predict(self, T):
        """Predict new data by linear interpolation.

        Parameters
        ----------
        T : array-like of shape (n_samples,) or (n_samples, 1)
            Data to transform.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Transformed data.
        """
        return self._transform(T)



    # We implement get_feature_names_out here instead of using
    # `ClassNamePrefixFeaturesOutMixin`` because `input_features` are ignored.
    # `input_features` are ignored because `IsotonicRegression` accepts 1d
    # arrays and the semantics of `feature_names_in_` are not clear for 1d arrays.
    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Ignored.

        Returns
        -------
        feature_names_out : ndarray of str objects
            An ndarray with one string i.e. ["isotonicregression0"].
        """
        check_is_fitted(self, "f_")
        class_name = self.__class__.__name__.lower()
        return np.asarray([f"{class_name}0"], dtype=object)



    def __getstate__(self):
        """Pickle-protocol - return state of the estimator."""
        state = super().__getstate__()
        # remove interpolation method
        state.pop("f_", None)
        return state



    def __setstate__(self, state):
        """Pickle-protocol - set state of the estimator.

        We need to rebuild the interpolation function.
        """
        super().__setstate__(state)
        if hasattr(self, "X_thresholds_") and hasattr(self, "y_thresholds_"):
            self._build_f(self.X_thresholds_, self.y_thresholds_)



    def _more_tags(self):
        return {"X_types": ["1darray"]}
```