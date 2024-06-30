# `D:\src\scipysrc\scikit-learn\sklearn\linear_model\_huber.py`

```
# 从 numbers 模块导入 Integral 和 Real 类型
from numbers import Integral, Real

# 导入 numpy 库并用 np 别名引用
import numpy as np

# 从 scipy 库导入 optimize 模块
from scipy import optimize

# 从相对路径导入所需模块和类
from ..base import BaseEstimator, RegressorMixin, _fit_context
from ..utils._mask import axis0_safe_slice
from ..utils._param_validation import Interval
from ..utils.extmath import safe_sparse_dot
from ..utils.optimize import _check_optimize_result
from ..utils.validation import _check_sample_weight
from ._base import LinearModel


def _huber_loss_and_gradient(w, X, y, epsilon, alpha, sample_weight=None):
    """返回 Huber 损失和梯度。

    Parameters
    ----------
    w : ndarray, shape (n_features + 1,) or (n_features + 2,)
        特征向量。
        w[:n_features] 给出系数
        w[-1] 给出比例因子，如果拟合截距，w[-2] 给出截距因子。

    X : ndarray of shape (n_samples, n_features)
        输入数据。

    y : ndarray of shape (n_samples,)
        目标向量。

    epsilon : float
        Huber 估计器的鲁棒性。

    alpha : float
        正则化参数。

    sample_weight : ndarray of shape (n_samples,), default=None
        每个样本分配的权重。

    Returns
    -------
    loss : float
        Huber 损失。

    gradient : ndarray, shape (len(w))
        返回 Huber 损失对每个系数、截距和比例的导数。
    """
    _, n_features = X.shape
    # 检查是否拟合了截距
    fit_intercept = n_features + 2 == w.shape[0]
    if fit_intercept:
        intercept = w[-2]
    sigma = w[-1]
    w = w[:n_features]
    n_samples = np.sum(sample_weight)

    # 计算 |y - X'w -c / sigma| > epsilon 的值
    # 超过此阈值的值被视为异常值。
    linear_loss = y - safe_sparse_dot(X, w)
    if fit_intercept:
        linear_loss -= intercept
    abs_linear_loss = np.abs(linear_loss)
    outliers_mask = abs_linear_loss > epsilon * sigma

    # 计算由异常值引起的线性损失。
    # 这等于 (2 * M * |y - X'w -c / sigma| - M**2) * sigma
    outliers = abs_linear_loss[outliers_mask]
    num_outliers = np.count_nonzero(outliers_mask)
    n_non_outliers = X.shape[0] - num_outliers

    # n_sq_outliers 包括对异常值的加权，而 num_outliers 只是异常值的数量。
    outliers_sw = sample_weight[outliers_mask]
    n_sw_outliers = np.sum(outliers_sw)
    outlier_loss = (
        2.0 * epsilon * np.sum(outliers_sw * outliers)
        - sigma * n_sw_outliers * epsilon**2
    )

    # 计算由非异常值引起的二次损失。
    # 这等于 |(y - X'w - c)**2 / sigma**2| * sigma
    non_outliers = linear_loss[~outliers_mask]
    weighted_non_outliers = sample_weight[~outliers_mask] * non_outliers
    weighted_loss = np.dot(weighted_non_outliers.T, non_outliers)
    squared_loss = weighted_loss / sigma
    # 如果需要拟合截距，则初始化包括截距在内的梯度数组
    if fit_intercept:
        grad = np.zeros(n_features + 2)
    else:
        # 否则，只初始化不包括截距的梯度数组
        grad = np.zeros(n_features + 1)

    # 计算因平方损失而产生的梯度
    X_non_outliers = -axis0_safe_slice(X, ~outliers_mask, n_non_outliers)
    grad[:n_features] = (
        2.0 / sigma * safe_sparse_dot(weighted_non_outliers, X_non_outliers)
    )

    # 计算因线性损失而产生的梯度
    signed_outliers = np.ones_like(outliers)
    signed_outliers_mask = linear_loss[outliers_mask] < 0
    signed_outliers[signed_outliers_mask] = -1.0
    X_outliers = axis0_safe_slice(X, outliers_mask, num_outliers)
    sw_outliers = sample_weight[outliers_mask] * signed_outliers
    grad[:n_features] -= 2.0 * epsilon * (safe_sparse_dot(sw_outliers, X_outliers))

    # 计算因正则化惩罚项而产生的梯度
    grad[:n_features] += alpha * 2.0 * w

    # 计算因标准差参数 sigma 而产生的梯度
    grad[-1] = n_samples
    grad[-1] -= n_sw_outliers * epsilon**2
    grad[-1] -= squared_loss / sigma

    # 计算因截距而产生的梯度
    if fit_intercept:
        grad[-2] = -2.0 * np.sum(weighted_non_outliers) / sigma
        grad[-2] -= 2.0 * epsilon * np.sum(sw_outliers)

    # 计算损失函数值
    loss = n_samples * sigma + squared_loss + outlier_loss
    loss += alpha * np.dot(w, w)
    
    # 返回损失值和梯度向量
    return loss, grad
class HuberRegressor(LinearModel, RegressorMixin, BaseEstimator):
    """L2-regularized linear regression model that is robust to outliers.

    The Huber Regressor optimizes the squared loss for the samples where
    ``|(y - Xw - c) / sigma| < epsilon`` and the absolute loss for the samples
    where ``|(y - Xw - c) / sigma| > epsilon``, where the model coefficients
    ``w``, the intercept ``c`` and the scale ``sigma`` are parameters
    to be optimized. The parameter sigma makes sure that if y is scaled up
    or down by a certain factor, one does not need to rescale epsilon to
    achieve the same robustness. Note that this does not take into account
    the fact that the different features of X may be of different scales.

    The Huber loss function has the advantage of not being heavily influenced
    by the outliers while not completely ignoring their effect.

    Read more in the :ref:`User Guide <huber_regression>`

    .. versionadded:: 0.18

    Parameters
    ----------
    epsilon : float, default=1.35
        The parameter epsilon controls the number of samples that should be
        classified as outliers. The smaller the epsilon, the more robust it is
        to outliers. Epsilon must be in the range `[1, inf)`.

    max_iter : int, default=100
        Maximum number of iterations that
        ``scipy.optimize.minimize(method="L-BFGS-B")`` should run for.

    alpha : float, default=0.0001
        Strength of the squared L2 regularization. Note that the penalty is
        equal to ``alpha * ||w||^2``.
        Must be in the range `[0, inf)`.

    warm_start : bool, default=False
        This is useful if the stored attributes of a previously used model
        has to be reused. If set to False, then the coefficients will
        be rewritten for every call to fit.
        See :term:`the Glossary <warm_start>`.

    fit_intercept : bool, default=True
        Whether or not to fit the intercept. This can be set to False
        if the data is already centered around the origin.

    tol : float, default=1e-05
        The iteration will stop when
        ``max{|proj g_i | i = 1, ..., n}`` <= ``tol``
        where pg_i is the i-th component of the projected gradient.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        Features got by optimizing the L2-regularized Huber loss.

    intercept_ : float
        Bias.

    scale_ : float
        The value by which ``|y - Xw - c|`` is scaled down.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0
    """
    # 定义 HuberRegressor 类，继承自 LinearModel, RegressorMixin, BaseEstimator
    # 这是一个 L2 正则化的线性回归模型，对离群值具有较强的鲁棒性
    # 优化器同时考虑了绝对损失和平方损失，根据参数 epsilon 和 sigma 控制不同样本的损失计算
    # 包含了参数和属性说明，以及版本信息和参数默认值
    def __init__(self, epsilon=1.35, max_iter=100, alpha=0.0001,
                 warm_start=False, fit_intercept=True, tol=1e-05):
        # 初始化方法，设置 HuberRegressor 的各个参数
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.alpha = alpha
        self.warm_start = warm_start
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None
        self.scale_ = None
        self.n_features_in_ = None
        self.feature_names_in_ = None
        # 初始化属性，这些属性将在 fit 方法中被赋值

    def fit(self, X, y):
        # 拟合方法，根据输入的特征 X 和目标 y 训练模型参数
        # 实现对 Huber 损失函数的优化

    def predict(self, X):
        # 预测方法，根据输入的特征 X 预测目标值

    def score(self, X, y):
        # 评分方法，计算模型在给定数据集上的预测性能得分

    def _more_tags(self):
        # 返回额外的标签，用于测试和验证
    # 记录 ``scipy.optimize.minimize(method="L-BFGS-B")`` 运行的迭代次数
    n_iter_ : int
        Number of iterations that
        ``scipy.optimize.minimize(method="L-BFGS-B")`` has run for.

        .. versionchanged:: 0.20

            In SciPy <= 1.0.0 the number of lbfgs iterations may exceed
            ``max_iter``. ``n_iter_`` will now report at most ``max_iter``.

    # 标记异常值的布尔遮罩，为 True 表示被识别为异常值的样本
    outliers_ : array, shape (n_samples,)
        A boolean mask which is set to True where the samples are identified
        as outliers.

    # 相关算法参考
    See Also
    --------
    RANSACRegressor : RANSAC (RANdom SAmple Consensus) algorithm.
    TheilSenRegressor : Theil-Sen Estimator robust multivariate regression model.
    SGDRegressor : Fitted by minimizing a regularized empirical loss with SGD.

    # 引用文献
    References
    ----------
    .. [1] Peter J. Huber, Elvezio M. Ronchetti, Robust Statistics
           Concomitant scale estimates, pg 172
    .. [2] Art B. Owen (2006), A robust hybrid of lasso and ridge regression.
           https://statweb.stanford.edu/~owen/reports/hhu.pdf

    # 示例
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import HuberRegressor, LinearRegression
    >>> from sklearn.datasets import make_regression
    >>> rng = np.random.RandomState(0)
    >>> X, y, coef = make_regression(
    ...     n_samples=200, n_features=2, noise=4.0, coef=True, random_state=0)
    >>> X[:4] = rng.uniform(10, 20, (4, 2))
    >>> y[:4] = rng.uniform(10, 20, 4)
    >>> huber = HuberRegressor().fit(X, y)
    >>> huber.score(X, y)
    -7.284...
    >>> huber.predict(X[:1,])
    array([806.7200...])
    >>> linear = LinearRegression().fit(X, y)
    >>> print("True coefficients:", coef)
    True coefficients: [20.4923...  34.1698...]
    >>> print("Huber coefficients:", huber.coef_)
    Huber coefficients: [17.7906... 31.0106...]
    >>> print("Linear Regression coefficients:", linear.coef_)
    Linear Regression coefficients: [-1.9221...  7.0226...]

    # 参数约束字典，用于验证参数的类型和取值范围
    _parameter_constraints: dict = {
        "epsilon": [Interval(Real, 1.0, None, closed="left")],
        "max_iter": [Interval(Integral, 0, None, closed="left")],
        "alpha": [Interval(Real, 0, None, closed="left")],
        "warm_start": ["boolean"],
        "fit_intercept": ["boolean"],
        "tol": [Interval(Real, 0.0, None, closed="left")],
    }

    # 初始化方法，设置回归器的初始参数
    def __init__(
        self,
        *,
        epsilon=1.35,
        max_iter=100,
        alpha=0.0001,
        warm_start=False,
        fit_intercept=True,
        tol=1e-05,
    ):
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.alpha = alpha
        self.warm_start = warm_start
        self.fit_intercept = fit_intercept
        self.tol = tol

    # 标记用于拟合过程的上下文装饰器，优先跳过嵌套验证
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like, shape (n_samples,)
            Target vector relative to X.

        sample_weight : array-like, shape (n_samples,), optional
            Weight given to each sample. Defaults to None.

        Returns
        -------
        self : object
            Fitted `HuberRegressor` estimator.
        """
        # Validate input data, ensuring it is in the correct format
        X, y = self._validate_data(
            X,
            y,
            copy=False,
            accept_sparse=["csr"],  # Accept Compressed Sparse Row format
            y_numeric=True,  # Ensure y is numeric
            dtype=[np.float64, np.float32],  # Allow float64 or float32 data types
        )

        # Check and adjust sample weights if provided
        sample_weight = _check_sample_weight(sample_weight, X)

        # Initialize parameters based on warm start or fit options
        if self.warm_start and hasattr(self, "coef_"):
            parameters = np.concatenate((self.coef_, [self.intercept_, self.scale_]))
        else:
            if self.fit_intercept:
                parameters = np.zeros(X.shape[1] + 2)  # Initialize with intercept and scale
            else:
                parameters = np.zeros(X.shape[1] + 1)  # Initialize without intercept
            # Initialize scale parameter to a small positive value
            parameters[-1] = 1

        # Define bounds for optimization, ensuring scale is non-negative
        bounds = np.tile([-np.inf, np.inf], (parameters.shape[0], 1))
        bounds[-1][0] = np.finfo(np.float64).eps * 10

        # Perform optimization to fit the model using L-BFGS-B method
        opt_res = optimize.minimize(
            _huber_loss_and_gradient,
            parameters,
            method="L-BFGS-B",
            jac=True,
            args=(X, y, self.epsilon, self.alpha, sample_weight),
            options={"maxiter": self.max_iter, "gtol": self.tol, "iprint": -1},
            bounds=bounds,
        )

        # Retrieve optimized parameters from optimization result
        parameters = opt_res.x

        # Handle cases where optimization fails to converge
        if opt_res.status == 2:
            raise ValueError(
                "HuberRegressor convergence failed: L-BFGS-B solver terminated with %s"
                % opt_res.message
            )

        # Validate and store number of iterations used in optimization
        self.n_iter_ = _check_optimize_result("lbfgs", opt_res, self.max_iter)

        # Update model attributes with optimized parameters
        self.scale_ = parameters[-1]
        if self.fit_intercept:
            self.intercept_ = parameters[-2]
        else:
            self.intercept_ = 0.0
        self.coef_ = parameters[: X.shape[1]]

        # Identify outliers based on residuals and model parameters
        residual = np.abs(y - safe_sparse_dot(X, self.coef_) - self.intercept_)
        self.outliers_ = residual > self.scale_ * self.epsilon

        # Return the fitted estimator object
        return self
```