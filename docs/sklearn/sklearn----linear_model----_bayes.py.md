# `D:\src\scipysrc\scikit-learn\sklearn\linear_model\_bayes.py`

```
"""
Various bayesian regression
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 从 math 模块导入 log 函数
from math import log
# 从 numbers 模块导入 Integral（整数）和 Real（实数）类
from numbers import Integral, Real

# 导入 numpy 库，并使用 np 别名
import numpy as np
# 导入 scipy 中的 linalg 子模块
from scipy import linalg
# 从 scipy.linalg 中导入 pinvh 函数
from scipy.linalg import pinvh

# 从 ..base 模块导入 RegressorMixin 和 _fit_context
from ..base import RegressorMixin, _fit_context
# 从 ..utils 中导入 _safe_indexing 函数
from ..utils import _safe_indexing
# 从 ..utils._param_validation 中导入 Interval 类
from ..utils._param_validation import Interval
# 从 ..utils.extmath 中导入 fast_logdet 函数
from ..utils.extmath import fast_logdet
# 从 ..utils.validation 中导入 _check_sample_weight 函数
from ..utils.validation import _check_sample_weight
# 从 ._base 模块导入 LinearModel、_preprocess_data 和 _rescale_data 函数
from ._base import LinearModel, _preprocess_data, _rescale_data

###############################################################################
# BayesianRidge regression

# 定义 BayesianRidge 类，继承自 RegressorMixin 和 LinearModel
class BayesianRidge(RegressorMixin, LinearModel):
    """Bayesian ridge regression.

    Fit a Bayesian ridge model. See the Notes section for details on this
    implementation and the optimization of the regularization parameters
    lambda (precision of the weights) and alpha (precision of the noise).

    Read more in the :ref:`User Guide <bayesian_regression>`.
    For an intuitive visualization of how the sinusoid is approximated by
    a polynomial using different pairs of initial values, see
    :ref:`sphx_glr_auto_examples_linear_model_plot_bayesian_ridge_curvefit.py`.

    Parameters
    ----------
    max_iter : int, default=300
        Maximum number of iterations over the complete dataset before
        stopping independently of any early stopping criterion.

        .. versionchanged:: 1.3

    tol : float, default=1e-3
        Stop the algorithm if w has converged.

    alpha_1 : float, default=1e-6
        Hyper-parameter : shape parameter for the Gamma distribution prior
        over the alpha parameter.

    alpha_2 : float, default=1e-6
        Hyper-parameter : inverse scale parameter (rate parameter) for the
        Gamma distribution prior over the alpha parameter.

    lambda_1 : float, default=1e-6
        Hyper-parameter : shape parameter for the Gamma distribution prior
        over the lambda parameter.

    lambda_2 : float, default=1e-6
        Hyper-parameter : inverse scale parameter (rate parameter) for the
        Gamma distribution prior over the lambda parameter.

    alpha_init : float, default=None
        Initial value for alpha (precision of the noise).
        If not set, alpha_init is 1/Var(y).

            .. versionadded:: 0.22

    lambda_init : float, default=None
        Initial value for lambda (precision of the weights).
        If not set, lambda_init is 1.

            .. versionadded:: 0.22

    compute_score : bool, default=False
        If True, compute the log marginal likelihood at each iteration of the
        optimization.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
        The intercept is not treated as a probabilistic parameter
        and thus has no associated variance. If set
        to False, no intercept will be used in calculations
        (i.e. data is expected to be centered).
    """
    copy_X : bool, default=True
        # 是否复制输入数据 X；如果为 True，则将复制 X，否则可能会覆盖它。

    verbose : bool, default=False
        # 在拟合模型时是否启用详细模式。

    Attributes
    ----------
    coef_ : array-like of shape (n_features,)
        # 回归模型的系数（分布的均值）

    intercept_ : float
        # 决策函数中的独立项。如果 `fit_intercept = False`，则设置为 0.0。

    alpha_ : float
       # 噪声的估计精度。

    lambda_ : float
       # 权重的估计精度。

    sigma_ : array-like of shape (n_features, n_features)
        # 权重的估计方差-协方差矩阵。

    scores_ : array-like of shape (n_iter_+1,)
        # 如果 computed_score 为 True，在优化的每次迭代中的对数边际似然值（要最大化）。数组从初始 alpha 和 lambda 的值开始，以估计的 alpha 和 lambda 的值结束。

    n_iter_ : int
        # 达到停止标准所需的实际迭代次数。

    X_offset_ : ndarray of shape (n_features,)
        # 如果 `fit_intercept=True`，用于将数据居中到零均值的偏移量。否则设置为 np.zeros(n_features)。

    X_scale_ : ndarray of shape (n_features,)
        # 设置为 np.ones(n_features)。

    n_features_in_ : int
        # 在 `fit` 过程中观察到的特征数量。

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        # 在 `fit` 过程中观察到的特征名称。仅在 `X` 的特征名称全为字符串时定义。

        .. versionadded:: 1.0

    See Also
    --------
    ARDRegression : 贝叶斯自相关稀疏回归。

    Notes
    -----
    执行贝叶斯岭回归有几种策略。此实现基于(Tipping, 2001)附录 A 中描述的算法，其中按照(MacKay, 1992)建议更新正则化参数。请注意，根据《自动相关决定的新视图》(Wipf 和 Nagarajan, 2008)，这些更新规则不能保证在优化的两个连续迭代之间边际似然性是增加的。

    References
    ----------
    D. J. C. MacKay, Bayesian Interpolation, Computation and Neural Systems,
    Vol. 4, No. 3, 1992.

    M. E. Tipping, Sparse Bayesian Learning and the Relevance Vector Machine,
    Journal of Machine Learning Research, Vol. 1, 2001.

    Examples
    --------
    >>> from sklearn import linear_model
    >>> clf = linear_model.BayesianRidge()
    >>> clf.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])
    BayesianRidge()
    >>> clf.predict([[1, 1]])
    array([1.])
    """
    # 定义参数约束字典，指定每个参数的类型和取值范围
    _parameter_constraints: dict = {
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="neither")],
        "alpha_1": [Interval(Real, 0, None, closed="left")],
        "alpha_2": [Interval(Real, 0, None, closed="left")],
        "lambda_1": [Interval(Real, 0, None, closed="left")],
        "lambda_2": [Interval(Real, 0, None, closed="left")],
        "alpha_init": [None, Interval(Real, 0, None, closed="left")],
        "lambda_init": [None, Interval(Real, 0, None, closed="left")],
        "compute_score": ["boolean"],
        "fit_intercept": ["boolean"],
        "copy_X": ["boolean"],
        "verbose": ["verbose"],
    }

    # 初始化方法，设定模型参数的默认值
    def __init__(
        self,
        *,
        max_iter=300,
        tol=1.0e-3,
        alpha_1=1.0e-6,
        alpha_2=1.0e-6,
        lambda_1=1.0e-6,
        lambda_2=1.0e-6,
        alpha_init=None,
        lambda_init=None,
        compute_score=False,
        fit_intercept=True,
        copy_X=True,
        verbose=False,
    ):
        self.max_iter = max_iter  # 设置最大迭代次数
        self.tol = tol  # 设置收敛阈值
        self.alpha_1 = alpha_1  # 设置第一类正则化参数
        self.alpha_2 = alpha_2  # 设置第二类正则化参数
        self.lambda_1 = lambda_1  # 设置第一类精度参数
        self.lambda_2 = lambda_2  # 设置第二类精度参数
        self.alpha_init = alpha_init  # 设置初始正则化参数
        self.lambda_init = lambda_init  # 设置初始精度参数
        self.compute_score = compute_score  # 设置是否计算评分
        self.fit_intercept = fit_intercept  # 设置是否拟合截距
        self.copy_X = copy_X  # 设置是否复制输入数据
        self.verbose = verbose  # 设置是否输出详细信息

    # 预测方法装饰器，预测线性模型的输出
    @_fit_context(prefer_skip_nested_validation=True)
    def predict(self, X, return_std=False):
        """Predict using the linear model.

        In addition to the mean of the predictive distribution, also its
        standard deviation can be returned.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Samples.

        return_std : bool, default=False
            Whether to return the standard deviation of posterior prediction.

        Returns
        -------
        y_mean : array-like of shape (n_samples,)
            Mean of predictive distribution of query points.

        y_std : array-like of shape (n_samples,)
            Standard deviation of predictive distribution of query points.
        """
        # 计算预测的均值
        y_mean = self._decision_function(X)
        if not return_std:
            return y_mean  # 如果不需要标准差，只返回均值
        else:
            # 计算预测的标准差
            sigmas_squared_data = (np.dot(X, self.sigma_) * X).sum(axis=1)
            y_std = np.sqrt(sigmas_squared_data + (1.0 / self.alpha_))
            return y_mean, y_std  # 返回均值和标准差

    # 更新模型系数的方法
    def _update_coef_(
        self, X, y, n_samples, n_features, XT_y, U, Vh, eigen_vals_, alpha_, lambda_
    ):
        """
        Update posterior mean and compute corresponding rmse.

        Posterior mean is given by coef_ = scaled_sigma_ * X.T * y where
        scaled_sigma_ = (lambda_/alpha_ * np.eye(n_features)
                         + np.dot(X.T, X))^-1
        """

        if n_samples > n_features:
            # Compute coef_ for n_samples > n_features case
            coef_ = np.linalg.multi_dot(
                [Vh.T, Vh / (eigen_vals_ + lambda_ / alpha_)[:, np.newaxis], XT_y]
            )
        else:
            # Compute coef_ for n_samples <= n_features case
            coef_ = np.linalg.multi_dot(
                [X.T, U / (eigen_vals_ + lambda_ / alpha_)[None, :], U.T, y]
            )

        # Compute rmse_
        rmse_ = np.sum((y - np.dot(X, coef_)) ** 2)

        # Return posterior mean (coef_) and root mean square error (rmse_)
        return coef_, rmse_

    def _log_marginal_likelihood(
        self, n_samples, n_features, eigen_vals, alpha_, lambda_, coef, rmse
    ):
        """
        Log marginal likelihood.

        Computes the score based on the log of the determinant of the posterior covariance.
        Posterior covariance is given by:
        sigma = (lambda_ * np.eye(n_features) + alpha_ * np.dot(X.T, X))^-1
        """
        alpha_1 = self.alpha_1
        alpha_2 = self.alpha_2
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2

        if n_samples > n_features:
            # Compute log of the determinant of sigma for n_samples > n_features case
            logdet_sigma = -np.sum(np.log(lambda_ + alpha_ * eigen_vals))
        else:
            # Compute log of the determinant of sigma for n_samples <= n_features case
            logdet_sigma = np.full(n_features, lambda_, dtype=np.array(lambda_).dtype)
            logdet_sigma[:n_samples] += alpha_ * eigen_vals
            logdet_sigma = -np.sum(np.log(logdet_sigma))

        # Compute score based on log marginal likelihood
        score = lambda_1 * np.log(lambda_) - lambda_2 * lambda_
        score += alpha_1 * np.log(alpha_) - alpha_2 * alpha_
        score += 0.5 * (
            n_features * np.log(lambda_)
            + n_samples * np.log(alpha_)
            - alpha_ * rmse
            - lambda_ * np.sum(coef**2)
            + logdet_sigma
            - n_samples * np.log(2 * np.pi)
        )

        return score
###############################################################################
# ARD (Automatic Relevance Determination) regression

# 定义 ARDRegression 类，继承自 RegressorMixin 和 LinearModel
class ARDRegression(RegressorMixin, LinearModel):
    """Bayesian ARD regression.

    Fit the weights of a regression model, using an ARD prior. The weights of
    the regression model are assumed to be in Gaussian distributions.
    Also estimate the parameters lambda (precisions of the distributions of the
    weights) and alpha (precision of the distribution of the noise).
    The estimation is done by an iterative procedures (Evidence Maximization)

    Read more in the :ref:`User Guide <bayesian_regression>`.

    Parameters
    ----------
    max_iter : int, default=300
        Maximum number of iterations.

        .. versionchanged:: 1.3

    tol : float, default=1e-3
        Stop the algorithm if w has converged.

    alpha_1 : float, default=1e-6
        Hyper-parameter : shape parameter for the Gamma distribution prior
        over the alpha parameter.

    alpha_2 : float, default=1e-6
        Hyper-parameter : inverse scale parameter (rate parameter) for the
        Gamma distribution prior over the alpha parameter.

    lambda_1 : float, default=1e-6
        Hyper-parameter : shape parameter for the Gamma distribution prior
        over the lambda parameter.

    lambda_2 : float, default=1e-6
        Hyper-parameter : inverse scale parameter (rate parameter) for the
        Gamma distribution prior over the lambda parameter.

    compute_score : bool, default=False
        If True, compute the objective function at each step of the model.

    threshold_lambda : float, default=10 000
        Threshold for removing (pruning) weights with high precision from
        the computation.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.

    verbose : bool, default=False
        Verbose mode when fitting the model.

    Attributes
    ----------
    coef_ : array-like of shape (n_features,)
        Coefficients of the regression model (mean of distribution)

    alpha_ : float
       estimated precision of the noise.

    lambda_ : array-like of shape (n_features,)
       estimated precisions of the weights.

    sigma_ : array-like of shape (n_features, n_features)
        estimated variance-covariance matrix of the weights

    scores_ : float
        if computed, value of the objective function (to be maximized)

    n_iter_ : int
        The actual number of iterations to reach the stopping criterion.

        .. versionadded:: 1.3

    intercept_ : float
        Independent term in decision function. Set to 0.0 if
        ``fit_intercept = False``.
    """
    # 定义一个浮点数，用于拟合截距时数据的偏移，以使数据均值为零。如果 `fit_intercept=True`，则设置为 `np.zeros(n_features)`。
    X_offset_ : float
    
    # 定义一个浮点数，初始化为 `np.ones(n_features)`，用于缩放数据。
    X_scale_ : float
    
    # 整数值，表示在 `fit` 过程中观察到的特征数量。
    n_features_in_ : int
    
    # 形状为 (`n_features_in_`,) 的 ndarray，仅在输入 `X` 具有全部为字符串的特征名称时定义。
    feature_names_in_ : ndarray
    
    # 引用的相关类：贝叶斯岭回归模型 `BayesianRidge`。
    See Also
    --------
    BayesianRidge : 贝叶斯岭回归模型。
    
    # 示例链接，展示如何使用该类的示例。
    Notes
    -----
    For an example, see :ref:`examples/linear_model/plot_ard.py
    <sphx_glr_auto_examples_linear_model_plot_ard.py>`.
    
    # 参考文献引用：
    References
    ----------
    D. J. C. MacKay, Bayesian nonlinear modeling for the prediction
    competition, ASHRAE Transactions, 1994.
    
    R. Salakhutdinov, Lecture notes on Statistical Machine Learning,
    http://www.utstat.toronto.edu/~rsalakhu/sta4273/notes/Lecture2.pdf#page=15
    Their beta is our ``self.alpha_``
    Their alpha is our ``self.lambda_``
    ARD is a little different than the slide: only dimensions/features for
    which ``self.lambda_ < self.threshold_lambda`` are kept and the rest are
    discarded.
    
    # 使用示例，展示如何实例化和使用该类。
    Examples
    --------
    >>> from sklearn import linear_model
    >>> clf = linear_model.ARDRegression()
    >>> clf.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])
    ARDRegression()
    >>> clf.predict([[1, 1]])
    array([1.])
    """
    
    # 参数约束字典，定义了各个参数的取值范围约束。
    _parameter_constraints: dict = {
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="left")],
        "alpha_1": [Interval(Real, 0, None, closed="left")],
        "alpha_2": [Interval(Real, 0, None, closed="left")],
        "lambda_1": [Interval(Real, 0, None, closed="left")],
        "lambda_2": [Interval(Real, 0, None, closed="left")],
        "compute_score": ["boolean"],
        "threshold_lambda": [Interval(Real, 0, None, closed="left")],
        "fit_intercept": ["boolean"],
        "copy_X": ["boolean"],
        "verbose": ["verbose"],
    }
    
    # 类的初始化方法，设置了各个参数的默认值。
    def __init__(
        self,
        *,
        max_iter=300,
        tol=1.0e-3,
        alpha_1=1.0e-6,
        alpha_2=1.0e-6,
        lambda_1=1.0e-6,
        lambda_2=1.0e-6,
        compute_score=False,
        threshold_lambda=1.0e4,
        fit_intercept=True,
        copy_X=True,
        verbose=False,
    ):
        self.max_iter = max_iter  # 最大迭代次数
        self.tol = tol  # 公差值
        self.fit_intercept = fit_intercept  # 是否拟合截距
        self.alpha_1 = alpha_1  # 超参数 alpha_1
        self.alpha_2 = alpha_2  # 超参数 alpha_2
        self.lambda_1 = lambda_1  # 超参数 lambda_1
        self.lambda_2 = lambda_2  # 超参数 lambda_2
        self.compute_score = compute_score  # 是否计算分数
        self.threshold_lambda = threshold_lambda  # lambda 阈值
        self.copy_X = copy_X  # 是否复制 X
        self.verbose = verbose  # 是否显示详细信息
    
    # 应用于类的修饰符，用于适应上下文并优先跳过嵌套验证。
    @_fit_context(prefer_skip_nested_validation=True)
    def _update_sigma_woodbury(self, X, alpha_, lambda_, keep_lambda):
        # 当 n_samples < n_features 时，使用 Woodbury 公式逆转矩阵
        # 参考幻灯片中所述
        # 这个函数用于逆转形状为 (n_samples, n_samples) 的矩阵，
        # 利用 Woodbury 矩阵恒等式：https://en.wikipedia.org/wiki/Woodbury_matrix_identity
        n_samples = X.shape[0]
        # 从 X 中仅保留指定列 keep_lambda 的数据
        X_keep = X[:, keep_lambda]
        # 计算 lambda_ 的倒数
        inv_lambda = 1 / lambda_[keep_lambda].reshape(1, -1)
        # 计算 sigma_，利用 Woodbury 公式
        sigma_ = pinvh(
            np.eye(n_samples, dtype=X.dtype) / alpha_
            + np.dot(X_keep * inv_lambda, X_keep.T)
        )
        # 再次利用 Woodbury 公式调整 sigma_
        sigma_ = np.dot(sigma_, X_keep * inv_lambda)
        sigma_ = -np.dot(inv_lambda.reshape(-1, 1) * X_keep.T, sigma_)
        # 对角线元素加上 lambda_ 的倒数
        sigma_[np.diag_indices(sigma_.shape[1])] += 1.0 / lambda_[keep_lambda]
        return sigma_

    def _update_sigma(self, X, alpha_, lambda_, keep_lambda):
        # 当 n_samples >= n_features 时，直接逆转形状为 (n_features, n_features) 的矩阵
        # 参考幻灯片中所述
        X_keep = X[:, keep_lambda]
        # 计算 Gram 矩阵
        gram = np.dot(X_keep.T, X_keep)
        eye = np.eye(gram.shape[0], dtype=X.dtype)
        # 构建逆转矩阵 sigma_inv
        sigma_inv = lambda_[keep_lambda] * eye + alpha_ * gram
        # 计算逆矩阵 sigma_
        sigma_ = pinvh(sigma_inv)
        return sigma_

    def predict(self, X, return_std=False):
        """使用线性模型进行预测。

        除了预测分布的均值外，还可以返回其标准差。

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            样本数据。

        return_std : bool, default=False
            是否返回后验预测的标准差。

        Returns
        -------
        y_mean : array-like of shape (n_samples,)
            查询点预测分布的均值。

        y_std : array-like of shape (n_samples,)
            查询点预测分布的标准差。
        """
        y_mean = self._decision_function(X)
        if return_std is False:
            return y_mean
        else:
            # 根据 lambda_ 和 threshold_lambda 进行列索引安全处理
            col_index = self.lambda_ < self.threshold_lambda
            X = _safe_indexing(X, indices=col_index, axis=1)
            # 计算预测标准差 y_std
            sigmas_squared_data = (np.dot(X, self.sigma_) * X).sum(axis=1)
            y_std = np.sqrt(sigmas_squared_data + (1.0 / self.alpha_))
            return y_mean, y_std
```