# `D:\src\scipysrc\scikit-learn\sklearn\linear_model\_theil_sen.py`

```
def _spatial_median(X, max_iter=300, tol=1.0e-3):
    """Spatial median (L1 median).

    The spatial median is member of a class of so-called M-estimators which
    are defined by an optimization problem. Given a number of p points in an
    n-dimensional space, the point x minimizing the sum of all distances to the
    p other points is called spatial median.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training vector, where `n_samples` is the number of samples and
        `n_features` is the number of features.

    max_iter : int, default=300
        Maximum number of iterations.

    tol : float, default=1.e-3
        Stop the algorithm if spatial_median has converged.

    Returns
    -------
    x_opt : ndarray of shape (n_features,)
        Estimated spatial median.

    Notes
    -----
    This function implements the Theil-Sen estimator for robust linear
    regression using the spatial median as a central point. It iteratively
    applies the modified Weiszfeld algorithm to find the spatial median.

    References
    ----------
    - On Computation of Spatial Median for Robust Data Mining, 2005
      T. Kärkkäinen and S. Äyrämö
      http://users.jyu.fi/~samiayr/pdf/ayramo_eurogen05.pdf
    """
    x_old = np.median(X, axis=0)
    
    for _ in range(max_iter):
        x_new = _modified_weiszfeld_step(X, x_old)
        if np.sum(np.abs(x_old - x_new)) < tol:
            return x_new
        x_old = x_new
    
    warnings.warn("Spatial median did not converge.", ConvergenceWarning)
    return x_old
    # 如果输入数据 X 的列数为 1，则直接返回 1 和 X 中位数的结果
    if X.shape[1] == 1:
        return 1, np.median(X.ravel(), keepdims=True)

    # 将容差 tol 的平方，用于计算平方范数
    tol **= 2  # We are computing the tol on the squared norm

    # 初始化空间中位数的初始值为 X 的列均值
    spatial_median_old = np.mean(X, axis=0)

    # 开始迭代求解空间中位数
    for n_iter in range(max_iter):
        # 使用修改的 Weiszfeld 算法进行迭代计算新的空间中位数
        spatial_median = _modified_weiszfeld_step(X, spatial_median_old)
        
        # 判断是否收敛：如果前后两次迭代的空间中位数差的平方和小于容差 tol，则停止迭代
        if np.sum((spatial_median_old - spatial_median) ** 2) < tol:
            break
        else:
            spatial_median_old = spatial_median  # 更新上一次的空间中位数为当前值

    # 如果迭代达到最大次数仍未收敛，则发出警告
    else:
        warnings.warn(
            "Maximum number of iterations {max_iter} reached in "
            "spatial median for TheilSen regressor."
            "".format(max_iter=max_iter),
            ConvergenceWarning,
        )

    # 返回迭代次数和最终计算得到的空间中位数
    return n_iter, spatial_median
# 计算断点（breakdown point）的近似值。

def _breakdown_point(n_samples, n_subsamples):
    # 参数 n_samples：样本数量
    # 参数 n_subsamples：要考虑的子样本数量

    # 返回值 breakdown_point：断点的近似值
    return (
        1
        - (
            # 计算公式的分子部分
            0.5 ** (1 / n_subsamples) * (n_samples - n_subsamples + 1)
            + n_subsamples
            - 1
        )
        / n_samples
    )


# TheilSenRegressor 类的最小二乘估计器。

def _lstsq(X, y, indices, fit_intercept):
    # 参数 X：形状为 (n_samples, n_features) 的设计矩阵，其中 n_samples 是样本数量，n_features 是特征数量。
    # 参数 y：形状为 (n_samples,) 的目标向量，其中 n_samples 是样本数量。
    # 参数 indices：形状为 (n_subpopulation, n_subsamples) 的数组，指定了所选子群体的所有子样本的索引。
    # 参数 fit_intercept：布尔值，指示是否拟合截距。

    fit_intercept = int(fit_intercept)
    n_features = X.shape[1] + fit_intercept
    n_subsamples = indices.shape[1]
    # 初始化权重数组，形状为 (n_subpopulation, n_features)
    weights = np.empty((indices.shape[0], n_features))
    # 创建形状为 (n_subsamples, n_features) 的全一矩阵
    X_subpopulation = np.ones((n_subsamples, n_features))
    # 创建形状为 (max(n_subsamples, n_features)) 的全零矩阵，用于存储 y_subpopulation
    y_subpopulation = np.zeros((max(n_subsamples, n_features)))
    # 获取 lstsq 函数，用于解决 X_subpopulation 和 y_subpopulation 的最小二乘问题
    (lstsq,) = get_lapack_funcs(("gelss",), (X_subpopulation, y_subpopulation))

    # 遍历索引数组 indices 中的每个子集，并计算对应的最小二乘解
    for index, subset in enumerate(indices):
        # 将 X 中对应子集的数据放入 X_subpopulation
        X_subpopulation[:, fit_intercept:] = X[subset, :]
        # 将 y 中对应子集的数据放入 y_subpopulation
        y_subpopulation[:n_subsamples] = y[subset]
        # 使用 lstsq 函数计算最小二乘解，并将结果存入 weights 中
        weights[index] = lstsq(X_subpopulation, y_subpopulation)[1][:n_features]

    # 返回权重数组
    return weights


class TheilSenRegressor(RegressorMixin, LinearModel):
    """Theil-Sen Estimator: robust multivariate regression model.

    The algorithm calculates least square solutions on subsets with size
    n_subsamples of the samples in X. Any value of n_subsamples between the
    number of features and samples leads to an estimator with a compromise
    between robustness and efficiency. Since the number of least square
    solutions is "n_samples choose n_subsamples", it can be extremely large
    and can therefore be limited with max_subpopulation. If this limit is
    reached, the subsets are chosen randomly. In a final step, the spatial
    median (or L1 median) is calculated of all least square solutions.

    Read more in the :ref:`User Guide <theil_sen_regression>`.
    """

    # Theil-Sen 估计器：鲁棒的多变量回归模型。
    # 该算法在 X 中大小为 n_subsamples 的子样本上计算最小二乘解。n_subsamples 的任何值介于特征数和样本数之间，会导致在鲁棒性和效率之间达成平衡的估计器。
    # 由于最小二乘解的数量为 "n_samples choose n_subsamples"，可能会非常大，因此可以通过 max_subpopulation 进行限制。如果达到此限制，则随机选择子集。在最后一步，计算所有最小二乘解的空间中位数（或 L1 中位数）。

    # 详细信息请参阅用户指南 :ref:`User Guide <theil_sen_regression>`.
    Parameters
    ----------
    fit_intercept : bool, default=True
        是否计算该模型的截距。如果设置为 False，则计算过程中不使用截距。

    copy_X : bool, default=True
        如果为 True，则复制 X；否则可能会被覆盖。

        .. deprecated:: 1.6
            `copy_X` 在 1.6 版本中已弃用，并将在 1.8 版本中移除。
            由于总是会进行复制，所以此参数已经没有影响。

    max_subpopulation : int, default=1e4
        如果 'n choose k' 的计算结果大于 max_subpopulation，则考虑给定最大大小的随机子群体，
        而不是对 'n choose k' 进行计算，其中 n 是样本数，k 是子样本数（至少等于特征数）。
        对于大问题，此参数将影响内存使用和运行时间。注意数据类型应为 int，但也可以接受如 1e4 这样的浮点数。

    n_subsamples : int, default=None
        用于计算参数的样本数。至少是特征数（如果 fit_intercept=True，则再加 1）和样本数的最大值。
        较低的值会导致较高的断点和低效率，而较高的值会导致较低的断点和高效率。
        如果为 None，则采用最小数量的子样本以达到最大的鲁棒性。
        如果 n_subsamples 设为 n_samples，则 Theil-Sen 回归与最小二乘回归相同。

    max_iter : int, default=300
        计算空间中位数时的最大迭代次数。

    tol : float, default=1e-3
        计算空间中位数时的容差。

    random_state : int, RandomState instance or None, default=None
        用于定义随机排列生成器状态的随机数生成器实例。
        传递一个整数可以在多次函数调用中产生可重现的输出。
        参见 :term:`术语表 <random_state>`。

    n_jobs : int, default=None
        在交叉验证期间要使用的 CPU 数量。
        ``None`` 表示使用 1 个处理器，除非在 :obj:`joblib.parallel_backend` 上下文中。
        ``-1`` 表示使用所有处理器。有关更多详细信息，请参见 :term:`术语表 <n_jobs>`。

    verbose : bool, default=False
        拟合模型时的详细模式。

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        回归模型的系数（分布的中位数）。

    intercept_ : float
        回归模型的估计截距。

    breakdown_ : float
        近似的断点。

    n_iter_ : int
        计算空间中位数所需的迭代次数。

    n_subpopulation_ : int
        从 'n choose k' 中考虑的组合数量，其中 n 是样本数，k 是子样本数。
    # 定义一个私有属性 `_parameter_constraints`，这是一个字典，用于描述参数的约束条件
    _parameter_constraints: dict = {
        "fit_intercept": ["boolean"],  # 参数 fit_intercept 应为布尔类型
        "copy_X": ["boolean", Hidden(StrOptions({"deprecated"}))],  # 参数 copy_X 应为布尔类型，同时具有一个隐藏选项 "deprecated"
        # 参数 max_subpopulation 应为大于等于 1 的实数，左闭区间
        "max_subpopulation": [Interval(Real, 1, None, closed="left")],
        "n_subsamples": [None, Integral],  # 参数 n_subsamples 可以为 None 或整数类型
        # 参数 max_iter 应为大于等于 0 的整数，左闭区间
        "max_iter": [Interval(Integral, 0, None, closed="left")],
        # 参数 tol 应为大于等于 0.0 的实数，左闭区间
        "tol": [Interval(Real, 0.0, None, closed="left")],
        "random_state": ["random_state"],  # 参数 random_state 应为随机状态对象
        "n_jobs": [None, Integral],  # 参数 n_jobs 可以为 None 或整数类型
        "verbose": ["verbose"],  # 参数 verbose 应为布尔类型
    }
    
    # 定义类的构造函数 __init__，用于初始化 TheilSenRegressor 类的对象
    def __init__(
        self,
        *,
        fit_intercept=True,  # 是否拟合截距，默认为 True
        copy_X="deprecated",  # 是否复制输入数据，默认为 "deprecated"
        max_subpopulation=1e4,  # 最大子群体大小，默认为 10000
        n_subsamples=None,  # 子样本数量，默认为 None
        max_iter=300,  # 最大迭代次数，默认为 300
        tol=1.0e-3,  # 收敛容差，默认为 0.001
        random_state=None,  # 随机状态，默认为 None
        n_jobs=None,  # 并行工作数量，默认为 None
        verbose=False,  # 是否输出详细信息，默认为 False
    ):
        self.fit_intercept = fit_intercept  # 初始化对象的 fit_intercept 属性
        self.copy_X = copy_X  # 初始化对象的 copy_X 属性
        self.max_subpopulation = max_subpopulation  # 初始化对象的 max_subpopulation 属性
        self.n_subsamples = n_subsamples  # 初始化对象的 n_subsamples 属性
        self.max_iter = max_iter  # 初始化对象的 max_iter 属性
        self.tol = tol  # 初始化对象的 tol 属性
        self.random_state = random_state  # 初始化对象的 random_state 属性
        self.n_jobs = n_jobs  # 初始化对象的 n_jobs 属性
        self.verbose = verbose  # 初始化对象的 verbose 属性
    # 检查子参数的有效性，确保模型参数设置正确
    def _check_subparams(self, n_samples, n_features):
        # 获取子样本数
        n_subsamples = self.n_subsamples

        # 如果需要拟合截距项
        if self.fit_intercept:
            # 计算特征维度，考虑了截距项
            n_dim = n_features + 1
        else:
            # 计算特征维度，不考虑截距项
            n_dim = n_features

        # 如果指定了子样本数
        if n_subsamples is not None:
            # 检查子样本数是否超过总样本数，如果是则报错
            if n_subsamples > n_samples:
                raise ValueError(
                    "Invalid parameter since n_subsamples > "
                    "n_samples ({0} > {1}).".format(n_subsamples, n_samples)
                )
            # 如果总样本数大于等于特征数
            if n_samples >= n_features:
                # 检查特征维度是否大于子样本数，如果是则报错
                if n_dim > n_subsamples:
                    plus_1 = "+1" if self.fit_intercept else ""
                    raise ValueError(
                        "Invalid parameter since n_features{0} "
                        "> n_subsamples ({1} > {2})."
                        "".format(plus_1, n_dim, n_subsamples)
                    )
            else:  # 如果总样本数小于特征数
                # 检查子样本数是否等于总样本数，如果不等则报错
                if n_subsamples != n_samples:
                    raise ValueError(
                        "Invalid parameter since n_subsamples != "
                        "n_samples ({0} != {1}) while n_samples "
                        "< n_features.".format(n_subsamples, n_samples)
                    )
        else:
            # 如果未指定子样本数，默认为特征维度和总样本数的最小值
            n_subsamples = min(n_dim, n_samples)

        # 计算所有可能的子样本组合数
        all_combinations = max(1, np.rint(binom(n_samples, n_subsamples)))
        # 限制子种群的最大数量
        n_subpopulation = int(min(self.max_subpopulation, all_combinations))

        # 返回有效的子样本数和子种群数量
        return n_subsamples, n_subpopulation

    # 在拟合过程中的上下文环境中，优先跳过嵌套验证
    @_fit_context(prefer_skip_nested_validation=True)
    # 定义一个方法，用于拟合线性模型
    def fit(self, X, y):
        """Fit linear model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            训练数据，包含n_samples个样本和n_features个特征。
        y : ndarray of shape (n_samples,)
            目标值，包含n_samples个样本的目标数值。

        Returns
        -------
        self : returns an instance of self.
            拟合后的`TheilSenRegressor`估计器实例。
        """
        # 如果self.copy_X不等于"deprecated"，发出警告信息
        if self.copy_X != "deprecated":
            warnings.warn(
                "`copy_X` was deprecated in 1.6 and will be removed in 1.8 since it "
                "has no effect internally. Simply leave this parameter to its default "
                "value to avoid this warning.",
                FutureWarning,
            )

        # 检查随机状态并处理数据验证
        random_state = check_random_state(self.random_state)
        X, y = self._validate_data(X, y, y_numeric=True)
        n_samples, n_features = X.shape
        # 检查子样本数量和子群体参数
        n_subsamples, self.n_subpopulation_ = self._check_subparams(
            n_samples, n_features
        )
        # 计算断点
        self.breakdown_ = _breakdown_point(n_samples, n_subsamples)

        # 如果设置了verbose标志，输出一些调试信息
        if self.verbose:
            print("Breakdown point: {0}".format(self.breakdown_))
            print("Number of samples: {0}".format(n_samples))
            tol_outliers = int(self.breakdown_ * n_samples)
            print("Tolerable outliers: {0}".format(tol_outliers))
            print("Number of subpopulations: {0}".format(self.n_subpopulation_))

        # 确定子群体的索引
        if np.rint(binom(n_samples, n_subsamples)) <= self.max_subpopulation:
            indices = list(combinations(range(n_samples), n_subsamples))
        else:
            indices = [
                random_state.choice(n_samples, size=n_subsamples, replace=False)
                for _ in range(self.n_subpopulation_)
            ]

        # 计算有效的n_jobs数目
        n_jobs = effective_n_jobs(self.n_jobs)
        # 将索引数组分割成多个子数组
        index_list = np.array_split(indices, n_jobs)
        # 并行计算每个子数组的最小二乘回归权重
        weights = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(_lstsq)(X, y, index_list[job], self.fit_intercept)
            for job in range(n_jobs)
        )
        weights = np.vstack(weights)
        # 计算空间中位数估计和迭代次数
        self.n_iter_, coefs = _spatial_median(
            weights, max_iter=self.max_iter, tol=self.tol
        )

        # 根据fit_intercept属性确定截距和系数
        if self.fit_intercept:
            self.intercept_ = coefs[0]
            self.coef_ = coefs[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = coefs

        # 返回拟合后的实例对象
        return self
```