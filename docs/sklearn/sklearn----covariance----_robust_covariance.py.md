# `D:\src\scipysrc\scikit-learn\sklearn\covariance\_robust_covariance.py`

```
"""
Robust location and covariance estimators.

Here are implemented estimators that are resistant to outliers.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import warnings  # 导入警告模块，用于处理警告信息
from numbers import Integral, Real  # 导入整数和实数的类型检查工具

import numpy as np  # 导入NumPy库，用于数值计算
from scipy import linalg  # 导入SciPy线性代数模块，用于求解线性方程
from scipy.stats import chi2  # 导入SciPy统计模块中的卡方分布

from ..base import _fit_context  # 从当前包中导入_fit_context基础模块
from ..utils import check_array, check_random_state  # 从当前包中导入数组检查和随机状态检查工具
from ..utils._param_validation import Interval  # 从当前包中导入参数验证中的Interval类
from ..utils.extmath import fast_logdet  # 从当前包中导入扩展数学模块中的快速对数行列式计算函数
from ._empirical_covariance import EmpiricalCovariance, empirical_covariance  # 导入经验协方差模块中的类和函数


# Minimum Covariance Determinant
#   Implementing of an algorithm by Rousseeuw & Van Driessen described in
#   (A Fast Algorithm for the Minimum Covariance Determinant Estimator,
#   1999, American Statistical Association and the American Society
#   for Quality, TECHNOMETRICS)
# XXX Is this really a public function? It's not listed in the docs or
# exported by sklearn.covariance. Deprecate?
def c_step(
    X,
    n_support,
    remaining_iterations=30,
    initial_estimates=None,
    verbose=False,
    cov_computation_method=empirical_covariance,
    random_state=None,
):
    """C_step procedure described in [Rouseeuw1984]_ aiming at computing MCD.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data set in which we look for the n_support observations whose
        scatter matrix has minimum determinant.

    n_support : int
        Number of observations to compute the robust estimates of location
        and covariance from. This parameter must be greater than
        `n_samples / 2`.

    remaining_iterations : int, default=30
        Number of iterations to perform.
        According to [Rouseeuw1999]_, two iterations are sufficient to get
        close to the minimum, and we never need more than 30 to reach
        convergence.

    initial_estimates : tuple of shape (2,), default=None
        Initial estimates of location and shape from which to run the c_step
        procedure:
        - initial_estimates[0]: an initial location estimate
        - initial_estimates[1]: an initial covariance estimate

    verbose : bool, default=False
        Verbose mode.

    cov_computation_method : callable, \
            default=:func:`sklearn.covariance.empirical_covariance`
        The function which will be used to compute the covariance.
        Must return array of shape (n_features, n_features).

    random_state : int, RandomState instance or None, default=None
        Determines the pseudo random number generator for shuffling the data.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    location : ndarray of shape (n_features,)
        Robust location estimates.

    covariance : ndarray of shape (n_features, n_features)
        Robust covariance estimates.
    """
    # 将输入数据 X 转换为 NumPy 数组，确保 X 是可操作的数组类型
    X = np.asarray(X)
    # 确保 random_state 是一个有效的随机数生成器对象
    random_state = check_random_state(random_state)
    # 调用 _c_step 函数进行 C 步骤的执行，返回最小行列式确定性估计器的结果
    return _c_step(
        X,
        n_support,
        remaining_iterations=remaining_iterations,
        initial_estimates=initial_estimates,
        verbose=verbose,
        cov_computation_method=cov_computation_method,
        random_state=random_state,
    )
def _c_step(
    X,
    n_support,
    random_state,
    remaining_iterations=30,
    initial_estimates=None,
    verbose=False,
    cov_computation_method=empirical_covariance,
):
    # 获取样本数量和特征数量
    n_samples, n_features = X.shape
    # 初始距离设置为无穷大
    dist = np.inf

    # 初始化
    support = np.zeros(n_samples, dtype=bool)
    if initial_estimates is None:
        # 如果未提供初始估计值，则从随机子集中计算初始稳健估计
        support[random_state.permutation(n_samples)[:n_support]] = True
    else:
        # 否则，从函数参数中获取初始稳健估计值
        location = initial_estimates[0]
        covariance = initial_estimates[1]
        # 针对这种情况运行特殊迭代（以获取初始支持）
        precision = linalg.pinvh(covariance)
        X_centered = X - location
        dist = (np.dot(X_centered, precision) * X_centered).sum(1)
        # 计算新的估计值
        support[np.argsort(dist)[:n_support]] = True

    # 提取支持数据集
    X_support = X[support]
    # 计算支持数据集的均值作为新的位置估计
    location = X_support.mean(0)
    # 使用指定方法计算支持数据集的协方差矩阵
    covariance = cov_computation_method(X_support)

    # 最小协方差行列式计算的迭代过程
    det = fast_logdet(covariance)
    # 如果数据已经具有奇异协方差，计算精度，因为下面的循环不会执行
    if np.isinf(det):
        precision = linalg.pinvh(covariance)

    previous_det = np.inf
    while det < previous_det and remaining_iterations > 0 and not np.isinf(det):
        # 保存旧的估计值
        previous_location = location
        previous_covariance = covariance
        previous_det = det
        previous_support = support
        # 计算全数据集的马氏距离，以获取新的支持
        precision = linalg.pinvh(covariance)
        X_centered = X - location
        dist = (np.dot(X_centered, precision) * X_centered).sum(axis=1)
        # 计算新的支持集合
        support = np.zeros(n_samples, dtype=bool)
        support[np.argsort(dist)[:n_support]] = True
        X_support = X[support]
        # 计算新的位置估计
        location = X_support.mean(axis=0)
        # 使用指定方法计算新的协方差矩阵
        covariance = cov_computation_method(X_support)
        # 计算新的行列式值
        det = fast_logdet(covariance)
        # 更新剩余迭代次数，以便进行早停止
        remaining_iterations -= 1

    # 计算上一次迭代的距离
    previous_dist = dist
    dist = (np.dot(X - location, precision) * (X - location)).sum(axis=1)
    # 检查是否已经找到最佳拟合（det >= 0，logdet <= -inf）
    if np.isinf(det):
        results = location, covariance, det, support, dist
    # 检查收敛性
    if np.allclose(det, previous_det):
        # c_step 过程已收敛
        if verbose:
            print(
                "Optimal couple (location, covariance) found before"
                " ending iterations (%d left)" % (remaining_iterations)
            )
        results = location, covariance, det, support, dist
    # 如果当前的行列式值（det）大于之前的行列式值（previous_det）
    elif det > previous_det:
        # 发出警告，指出行列式值增加了（通常不应该发生）
        warnings.warn(
            "Determinant has increased; this should not happen: "
            "log(det) > log(previous_det) (%.15f > %.15f). "
            "You may want to try with a higher value of "
            "support_fraction (current value: %.3f)."
            % (det, previous_det, n_support / n_samples),
            RuntimeWarning,
        )
        # 返回之前保存的结果元组，因为行列式值不应增加
        results = (
            previous_location,
            previous_covariance,
            previous_det,
            previous_support,
            previous_dist,
        )

    # 检查是否达到早停条件
    if remaining_iterations == 0:
        # 如果启用了详细输出
        if verbose:
            # 打印消息，说明已达到最大迭代次数
            print("Maximum number of iterations reached")
        # 返回当前的计算结果元组
        results = location, covariance, det, support, dist

    # 返回最终的结果元组（在某些条件下，可能会是早停时的结果）
    return results
# 定义函数 select_candidates，用于从数据集 X 中选择最纯净的观测子集来计算 MCD（Minimum Covariance Determinant）。

def select_candidates(
    X,
    n_support,
    n_trials,
    select=1,
    n_iter=30,
    verbose=False,
    cov_computation_method=empirical_covariance,
    random_state=None,
):
    """Finds the best pure subset of observations to compute MCD from it.
    
    根据 MCD 的计算要求，找到最适合计算的 n_support 个观测样本，以最小化它们的协方差矩阵行列式。等价地，移除 n_samples - n_support 个观测，构建不包含异常值的纯净数据集。
    纯净数据集的观测列表被称为“支持”。

    Starting from a random support, the pure data set is found by the
    c_step procedure introduced by Rousseeuw and Van Driessen in
    [RV]_.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        数据集，从中寻找包含 n_support 个纯净观测的数据子集。

    n_support : int
        纯净数据集中必须包含的样本数。
        此参数必须在范围 `[(n + p + 1)/2] < n_support < n` 内。

    n_trials : int or tuple of shape (2,)
        从中运行算法的不同初始观测集合的数量。此参数应为严格正整数。
        可以提供初始估计值列表，用于迭代运行 c_step 程序。
        - n_trials[0]: array-like, shape (n_trials, n_features)
          为 `n_trials` 个初始位置估计的列表
        - n_trials[1]: array-like, shape (n_trials, n_features, n_features)
          为 `n_trials` 个初始协方差估计的列表

    select : int, default=1
        返回的最佳候选结果数。此参数必须为严格正整数。

    n_iter : int, default=30
        c_step 程序的最大迭代次数。
        （2 足以接近最终解，通常不超过 20）。
        此参数必须为严格正整数。

    verbose : bool, default=False
        控制输出的详细程度。

    cov_computation_method : callable, \
            default=:func:`sklearn.covariance.empirical_covariance`
        用于计算协方差的函数。
        必须返回形状为 (n_features, n_features) 的数组。

    random_state : int, RandomState instance or None, default=None
        确定用于对数据进行洗牌的伪随机数生成器。
        传递一个整数以便跨多次函数调用产生可复现的结果。
        参见 :term:`Glossary <random_state>`。

    See Also
    ---------
    c_step

    Returns
    -------
    best_locations : ndarray of shape (select, n_features)
        从数据集（`X`）中找到的 `select` 个最佳支持所计算的 `select` 个位置估计。

    """
    # best_covariances 是一个形状为 (select, n_features, n_features) 的 ndarray
    # 包含从数据集 `X` 中找到的 `select` 个最佳支持下计算的协方差估计值
    best_covariances : ndarray of shape (select, n_features, n_features)

    # best_supports 是一个形状为 (select, n_samples) 的 ndarray
    # 包含从数据集 `X` 中找到的 `select` 个最佳支持
    best_supports : ndarray of shape (select, n_samples)

    References
    ----------
    .. [RV] A Fast Algorithm for the Minimum Covariance Determinant
        Estimator, 1999, American Statistical Association and the American
        Society for Quality, TECHNOMETRICS
    """
    # 使用给定的随机状态检查和设置随机数生成器
    random_state = check_random_state(random_state)

    # 根据 `n_trials` 的类型进行不同的处理分支
    if isinstance(n_trials, Integral):
        # 如果 `n_trials` 是整数，则直接使用初始估计运行
        run_from_estimates = False
    elif isinstance(n_trials, tuple):
        # 如果 `n_trials` 是元组，则使用给定的初始估计运行
        run_from_estimates = True
        estimates_list = n_trials
        n_trials = estimates_list[0].shape[0]
    else:
        # 如果 `n_trials` 类型不符合预期，则抛出类型错误异常
        raise TypeError(
            "Invalid 'n_trials' parameter, expected tuple or  integer, got %s (%s)"
            % (n_trials, type(n_trials))
        )

    # 计算在子集中的 `n_trials` 位置和形状估计候选项
    all_estimates = []
    if not run_from_estimates:
        # 如果不是从给定的估计运行，则从随机初始支持开始进行 `n_trials` 次计算
        for j in range(n_trials):
            all_estimates.append(
                _c_step(
                    X,
                    n_support,
                    remaining_iterations=n_iter,
                    verbose=verbose,
                    cov_computation_method=cov_computation_method,
                    random_state=random_state,
                )
            )
    else:
        # 如果是从给定的估计运行，则从每个给定的初始估计开始进行计算
        for j in range(n_trials):
            initial_estimates = (estimates_list[0][j], estimates_list[1][j])
            all_estimates.append(
                _c_step(
                    X,
                    n_support,
                    remaining_iterations=n_iter,
                    initial_estimates=initial_estimates,
                    verbose=verbose,
                    cov_computation_method=cov_computation_method,
                    random_state=random_state,
                )
            )

    # 将所有估计的结果解压缩成不同的元组
    all_locs_sub, all_covs_sub, all_dets_sub, all_supports_sub, all_ds_sub = zip(
        *all_estimates
    )

    # 从 `n_trials` 中找到最佳的 `n_best` 个结果
    index_best = np.argsort(all_dets_sub)[:select]
    best_locations = np.asarray(all_locs_sub)[index_best]
    best_covariances = np.asarray(all_covs_sub)[index_best]
    best_supports = np.asarray(all_supports_sub)[index_best]
    best_ds = np.asarray(all_ds_sub)[index_best]

    # 返回最佳位置、协方差、支持和 ds 的结果数组
    return best_locations, best_covariances, best_supports, best_ds
    # 确保随机状态参数为一个 RandomState 实例，如果为 None，则使用默认的随机数生成器
    random_state = check_random_state(random_state)

    # 检查输入数据 X 是否为数组，并确保至少有两个样本，适用于最小协方差行列式估计
    X = check_array(X, ensure_min_samples=2, estimator="fast_mcd")
    n_samples, n_features = X.shape

    # 最小破坏值
    # 如果支持分数未指定，计算支持样本数
    if support_fraction is None:
        n_support = int(np.ceil(0.5 * (n_samples + n_features + 1)))
    else:
        # 根据支持分数计算支持样本数
        n_support = int(support_fraction * n_samples)

    # 对于一维情况的快速计算
    # (参考文献：Rousseeuw, P. J. and Leroy, A. M. (2005) Robust Regression and Outlier Detection, John Wiley & Sons, 第4章)
    if n_features == 1:
        if n_support < n_samples:
            # 找到样本中最短的两半部分
            X_sorted = np.sort(np.ravel(X))
            diff = X_sorted[n_support:] - X_sorted[: (n_samples - n_support)]
            halves_start = np.where(diff == np.min(diff))[0]
            # 取中间点的平均值作为稳健位置估计
            location = (
                0.5
                * (X_sorted[n_support + halves_start] + X_sorted[halves_start]).mean()
            )
            support = np.zeros(n_samples, dtype=bool)
            X_centered = X - location
            # 根据绝对值排序后的索引选择支持样本
            support[np.argsort(np.abs(X_centered), 0)[:n_support]] = True
            covariance = np.asarray([[np.var(X[support])]])
            location = np.array([location])
            # 通过优化的方式获取精度矩阵
            precision = linalg.pinvh(covariance)
            dist = (np.dot(X_centered, precision) * (X_centered)).sum(axis=1)
        else:
            # 所有样本都被支持的情况
            support = np.ones(n_samples, dtype=bool)
            covariance = np.asarray([[np.var(X)]])
            location = np.asarray([np.mean(X)])
            X_centered = X - location
            # 通过优化的方式获取精度矩阵
            precision = linalg.pinvh(covariance)
            dist = (np.dot(X_centered, precision) * (X_centered)).sum(axis=1)
    # 对于多维情况，启动 FastMCD 算法
    elif n_features > 1:
        # 1. 找到最佳的十对（位置，协方差）
        # 考虑两次迭代
        n_trials = 30
        n_best = 10
        locations_best, covariances_best, _, _ = select_candidates(
            X,
            n_support,
            n_trials=n_trials,
            select=n_best,
            n_iter=2,
            cov_computation_method=cov_computation_method,
            random_state=random_state,
        )
        # 2. 在十对中选择整个数据集上的最佳一对
        locations_full, covariances_full, supports_full, d = select_candidates(
            X,
            n_support,
            n_trials=(locations_best, covariances_best),
            select=1,
            cov_computation_method=cov_computation_method,
            random_state=random_state,
        )
        location = locations_full[0]
        covariance = covariances_full[0]
        support = supports_full[0]
        dist = d[0]

    return location, covariance, support, dist
# 继承 EmpiricalCovariance 类，实现最小协方差行列式（MCD）估计器，用于鲁棒性协方差估计。

class MinCovDet(EmpiricalCovariance):
    """Minimum Covariance Determinant (MCD): robust estimator of covariance.

    The Minimum Covariance Determinant covariance estimator is to be applied
    on Gaussian-distributed data, but could still be relevant on data
    drawn from a unimodal, symmetric distribution. It is not meant to be used
    with multi-modal data (the algorithm used to fit a MinCovDet object is
    likely to fail in such a case).
    One should consider projection pursuit methods to deal with multi-modal
    datasets.

    Read more in the :ref:`User Guide <robust_covariance>`.

    Parameters
    ----------
    store_precision : bool, default=True
        Specify if the estimated precision is stored.

    assume_centered : bool, default=False
        If True, the support of the robust location and the covariance
        estimates is computed, and a covariance estimate is recomputed from
        it, without centering the data.
        Useful to work with data whose mean is significantly equal to
        zero but is not exactly zero.
        If False, the robust location and covariance are directly computed
        with the FastMCD algorithm without additional treatment.

    support_fraction : float, default=None
        The proportion of points to be included in the support of the raw
        MCD estimate. Default is None, which implies that the minimum
        value of support_fraction will be used within the algorithm:
        `(n_samples + n_features + 1) / 2 * n_samples`. The parameter must be
        in the range (0, 1].

    random_state : int, RandomState instance or None, default=None
        Determines the pseudo random number generator for shuffling the data.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    raw_location_ : ndarray of shape (n_features,)
        The raw robust estimated location before correction and re-weighting.

    raw_covariance_ : ndarray of shape (n_features, n_features)
        The raw robust estimated covariance before correction and re-weighting.

    raw_support_ : ndarray of shape (n_samples,)
        A mask of the observations that have been used to compute
        the raw robust estimates of location and shape, before correction
        and re-weighting.

    location_ : ndarray of shape (n_features,)
        Estimated robust location.

    covariance_ : ndarray of shape (n_features, n_features)
        Estimated robust covariance matrix.

    precision_ : ndarray of shape (n_features, n_features)
        Estimated pseudo inverse matrix.
        (stored only if store_precision is True)

    support_ : ndarray of shape (n_samples,)
        A mask of the observations that have been used to compute
        the robust estimates of location and shape.
    """
    # Mahalanobis距离的训练集的数组，形状为(n_samples,)
    dist_ : ndarray of shape (n_samples,)
        Mahalanobis distances of the training set (on which :meth:`fit` is
        called) observations.
    
    # 在拟合过程中观察到的特征数量
    n_features_in_ : int
        Number of features seen during :term:`fit`.
    
        .. versionadded:: 0.24
    
    # 在拟合过程中观察到的特征名称数组，仅当`X`的特征名称都是字符串时定义
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
    
        .. versionadded:: 1.0
    
    # 相关类的参考信息
    
    See Also
    --------
    EllipticEnvelope : An object for detecting outliers in
        a Gaussian distributed dataset.
    EmpiricalCovariance : Maximum likelihood covariance estimator.
    GraphicalLasso : Sparse inverse covariance estimation
        with an l1-penalized estimator.
    GraphicalLassoCV : Sparse inverse covariance with cross-validated
        choice of the l1 penalty.
    LedoitWolf : LedoitWolf Estimator.
    OAS : Oracle Approximating Shrinkage Estimator.
    ShrunkCovariance : Covariance estimator with shrinkage.
    
    # 引用信息
    
    References
    ----------
    .. [Rouseeuw1984] P. J. Rousseeuw. Least median of squares regression.
        J. Am Stat Ass, 79:871, 1984.
    .. [Rousseeuw] A Fast Algorithm for the Minimum Covariance Determinant
        Estimator, 1999, American Statistical Association and the American
        Society for Quality, TECHNOMETRICS
    .. [ButlerDavies] R. W. Butler, P. L. Davies and M. Jhun,
        Asymptotics For The Minimum Covariance Determinant Estimator,
        The Annals of Statistics, 1993, Vol. 21, No. 3, 1385-1400
    
    # 示例
    
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.covariance import MinCovDet
    >>> from sklearn.datasets import make_gaussian_quantiles
    >>> real_cov = np.array([[.8, .3],
    ...                      [.3, .4]])
    >>> rng = np.random.RandomState(0)
    >>> X = rng.multivariate_normal(mean=[0, 0],
    ...                                   cov=real_cov,
    ...                                   size=500)
    >>> cov = MinCovDet(random_state=0).fit(X)
    >>> cov.covariance_
    array([[0.7411..., 0.2535...],
           [0.2535..., 0.3053...]])
    >>> cov.location_
    array([0.0813... , 0.0427...])
    
    """
    
    # _parameter_constraints的字典，包含参数约束信息
    _parameter_constraints: dict = {
        **EmpiricalCovariance._parameter_constraints,
        "support_fraction": [Interval(Real, 0, 1, closed="right"), None],
        "random_state": ["random_state"],
    }
    
    # _nonrobust_covariance是empirical_covariance的静态方法
    _nonrobust_covariance = staticmethod(empirical_covariance)
    
    # 初始化函数，设定属性值
    def __init__(
        self,
        *,
        store_precision=True,
        assume_centered=False,
        support_fraction=None,
        random_state=None,
    ):
        self.store_precision = store_precision
        self.assume_centered = assume_centered
        self.support_fraction = support_fraction
        self.random_state = random_state
    
    # _fit_context装饰器，设置prefer_skip_nested_validation=True
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Fit a Minimum Covariance Determinant with the FastMCD algorithm.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Validate and preprocess input data X
        X = self._validate_data(X, ensure_min_samples=2, estimator="MinCovDet")

        # Initialize random state for consistent random number generation
        random_state = check_random_state(self.random_state)
        
        # Get the number of samples (n_samples) and features (n_features) from X
        n_samples, n_features = X.shape
        
        # Check if the empirical covariance matrix is full rank
        if (linalg.svdvals(np.dot(X.T, X)) > 1e-8).sum() != n_features:
            # Issue a warning if the covariance matrix is not full rank
            warnings.warn(
                "The covariance matrix associated to your dataset is not full rank"
            )
        
        # Compute raw estimates using the FastMCD algorithm
        raw_location, raw_covariance, raw_support, raw_dist = fast_mcd(
            X,
            support_fraction=self.support_fraction,
            cov_computation_method=self._nonrobust_covariance,
            random_state=random_state,
        )
        
        # Adjust raw estimates if assume_centered is True
        if self.assume_centered:
            raw_location = np.zeros(n_features)  # Set raw_location to zeros
            # Compute raw_covariance based on selected support
            raw_covariance = self._nonrobust_covariance(
                X[raw_support], assume_centered=True
            )
            # Compute precision matrix (inverse of covariance matrix) using pinvh
            precision = linalg.pinvh(raw_covariance)
            # Compute raw_dist using the precision matrix
            raw_dist = np.sum(np.dot(X, precision) * X, 1)
        
        # Assign computed values to instance attributes
        self.raw_location_ = raw_location
        self.raw_covariance_ = raw_covariance
        self.raw_support_ = raw_support
        self.location_ = raw_location
        self.support_ = raw_support
        self.dist_ = raw_dist
        
        # Ensure consistency with normal models for covariance
        self.correct_covariance(X)
        
        # Reweight the covariance estimator
        self.reweight_covariance(X)

        # Return the fitted instance of the object
        return self
    def correct_covariance(self, data):
        """
        Apply a correction to raw Minimum Covariance Determinant estimates.

        Correction using the empirical correction factor suggested
        by Rousseeuw and Van Driessen in [RVD]_.

        Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            The data matrix, with p features and n samples.
            The data set must be the one which was used to compute
            the raw estimates.

        Returns
        -------
        covariance_corrected : ndarray of shape (n_features, n_features)
            Corrected robust covariance estimate.

        References
        ----------
        .. [RVD] A Fast Algorithm for the Minimum Covariance
            Determinant Estimator, 1999, American Statistical Association
            and the American Society for Quality, TECHNOMETRICS
        """

        # Check that the covariance of the support data is not equal to 0.
        # Otherwise self.dist_ = 0 and thus correction = 0.
        n_samples = len(self.dist_)  # 获取支持数据的样本数
        n_support = np.sum(self.support_)  # 计算支持数据的数量
        if n_support < n_samples and np.allclose(self.raw_covariance_, 0):
            raise ValueError(
                "The covariance matrix of the support data "
                "is equal to 0, try to increase support_fraction"
            )
        correction = np.median(self.dist_) / chi2(data.shape[1]).isf(0.5)  # 计算修正因子
        covariance_corrected = self.raw_covariance_ * correction  # 应用修正因子到原始协方差矩阵
        self.dist_ /= correction  # 调整支持数据的分布以反映修正因子的影响
        return covariance_corrected  # 返回修正后的协方差矩阵估计
    def reweight_covariance(self, data):
        """
        Re-weight raw Minimum Covariance Determinant estimates.

        Re-weight observations using Rousseeuw's method (equivalent to
        deleting outlying observations from the data set before
        computing location and covariance estimates) described
        in [RVDriessen]_.

        Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            The data matrix, with p features and n samples.
            The data set must be the one which was used to compute
            the raw estimates.

        Returns
        -------
        location_reweighted : ndarray of shape (n_features,)
            Re-weighted robust location estimate.

        covariance_reweighted : ndarray of shape (n_features, n_features)
            Re-weighted robust covariance estimate.

        support_reweighted : ndarray of shape (n_samples,), dtype=bool
            A mask of the observations that have been used to compute
            the re-weighted robust location and covariance estimates.

        References
        ----------
        .. [RVDriessen] A Fast Algorithm for the Minimum Covariance
            Determinant Estimator, 1999, American Statistical Association
            and the American Society for Quality, TECHNOMETRICS
        """
        # 获取数据矩阵的样本数和特征数
        n_samples, n_features = data.shape
        # 根据卡方分布的反函数计算阈值，生成异常值的掩码
        mask = self.dist_ < chi2(n_features).isf(0.025)
        
        # 如果假设数据已居中，重新加权的位置估计为零向量
        if self.assume_centered:
            location_reweighted = np.zeros(n_features)
        else:
            # 否则，使用掩码选择的数据计算均值作为位置估计
            location_reweighted = data[mask].mean(0)
        
        # 使用非鲁棒协方差方法计算重新加权的协方差估计
        covariance_reweighted = self._nonrobust_covariance(
            data[mask], assume_centered=self.assume_centered
        )
        
        # 创建用于支持估计的掩码
        support_reweighted = np.zeros(n_samples, dtype=bool)
        support_reweighted[mask] = True
        
        # 设置对象的协方差矩阵、位置估计和支持向量
        self._set_covariance(covariance_reweighted)
        self.location_ = location_reweighted
        self.support_ = support_reweighted
        
        # 计算中心化后的数据和精确度的乘积并求和，更新距离数组
        X_centered = data - self.location_
        self.dist_ = np.sum(np.dot(X_centered, self.get_precision()) * X_centered, 1)
        
        # 返回重新加权的位置估计、协方差估计和支持向量
        return location_reweighted, covariance_reweighted, support_reweighted
```