# `D:\src\scipysrc\scikit-learn\sklearn\linear_model\_glm\glm.py`

```
# 导入所需模块和类
from numbers import Integral, Real  # 从标准库导入整数和实数类型

import numpy as np  # 导入NumPy库，用于数值计算
import scipy.optimize  # 导入SciPy库中的优化模块

# 导入损失函数相关的模块和类
from ..._loss.loss import (
    HalfGammaLoss,  # 导入半Gamma损失函数类
    HalfPoissonLoss,  # 导入半泊松损失函数类
    HalfSquaredError,  # 导入半平方误差损失函数类
    HalfTweedieLoss,  # 导入半Tweedie损失函数类
    HalfTweedieLossIdentity,  # 导入带身份矩阵的半Tweedie损失函数类
)

# 导入基础估计器、回归器混合类以及拟合上下文相关的类
from ...base import BaseEstimator, RegressorMixin, _fit_context

# 导入数据检查函数
from ...utils import check_array

# 导入OpenMP相关的帮助函数
from ...utils._openmp_helpers import _openmp_effective_n_threads

# 导入参数验证相关的模块
from ...utils._param_validation import Hidden, Interval, StrOptions

# 导入优化相关的函数
from ...utils.optimize import _check_optimize_result

# 导入验证函数，用于检查样本权重和是否已拟合
from ...utils.validation import _check_sample_weight, check_is_fitted

# 导入线性模型损失相关的模块
from .._linear_loss import LinearModelLoss

# 导入牛顿法求解器相关的类
from ._newton_solver import NewtonCholeskySolver, NewtonSolver


class _GeneralizedLinearRegressor(RegressorMixin, BaseEstimator):
    """Regression via a penalized Generalized Linear Model (GLM).

    GLMs based on a reproductive Exponential Dispersion Model (EDM) aim at fitting and
    predicting the mean of the target y as y_pred=h(X*w) with coefficients w.
    Therefore, the fit minimizes the following objective function with L2 priors as
    regularizer::

        1/(2*sum(s_i)) * sum(s_i * deviance(y_i, h(x_i*w)) + 1/2 * alpha * ||w||_2^2

    with inverse link function h, s=sample_weight and per observation (unit) deviance
    deviance(y_i, h(x_i*w)). Note that for an EDM, 1/2 * deviance is the negative
    log-likelihood up to a constant (in w) term.
    The parameter ``alpha`` corresponds to the lambda parameter in glmnet.

    Instead of implementing the EDM family and a link function separately, we directly
    use the loss functions `from sklearn._loss` which have the link functions included
    in them for performance reasons. We pick the loss functions that implement
    (1/2 times) EDM deviances.

    Read more in the :ref:`User Guide <Generalized_linear_models>`.

    .. versionadded:: 0.23

    Parameters
    ----------
    alpha : float, default=1
        Constant that multiplies the penalty term and thus determines the
        regularization strength. ``alpha = 0`` is equivalent to unpenalized
        GLMs. In this case, the design matrix `X` must have full column rank
        (no collinearities).
        Values must be in the range `[0.0, inf)`.

    fit_intercept : bool, default=True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the linear predictor (X @ coef + intercept).
    """
    solver : {'lbfgs', 'newton-cholesky'}, default='lbfgs'
        # 选择优化问题中要使用的算法:
        #
        # 'lbfgs'
        #     调用 scipy 的 L-BFGS-B 优化器。
        #
        # 'newton-cholesky'
        #     使用牛顿-乔列斯基步骤（在等价于迭代重新加权最小二乘的任意精度算术中）
        #     使用基于Cholesky的内部求解器。这个求解器对于 `n_samples` >> `n_features` 特别适合，
        #     特别是在具有稀有类别的独热编码分类特征时。请注意，此求解器的内存使用量与 `n_features` 有二次依赖关系，
        #     因为它显式计算Hessian矩阵。
        #
        #     .. versionadded:: 1.2

    max_iter : int, default=100
        # 求解器的最大迭代次数。
        # 值必须在范围 `[1, inf)` 内。

    tol : float, default=1e-4
        # 停止标准。对于 lbfgs 求解器，
        # 当 ``max{|g_j|, j = 1, ..., d} <= tol`` 时，迭代将停止，
        # 其中 ``g_j`` 是目标函数的梯度（导数）的第 j 个分量。
        # 值必须在范围 `(0.0, inf)` 内。

    warm_start : bool, default=False
        # 如果设置为 ``True``，则重复利用上一次调用 ``fit`` 的解决方案
        # 作为 ``coef_`` 和 ``intercept_`` 的初始化值。

    verbose : int, default=0
        # 对于 lbfgs 求解器，设置 verbose 为任何正数以增加详细程度。
        # 值必须在范围 `[0, inf)` 内。

    Attributes
    ----------
    coef_ : array of shape (n_features,)
        # GLM 中线性预测器 (`X @ coef_ + intercept_`) 的估计系数。

    intercept_ : float
        # 添加到线性预测器的截距（也称为偏差）。

    n_iter_ : int
        # 求解器中实际使用的迭代次数。

    _base_loss : BaseLoss, default=HalfSquaredError()
        # 这在 `fit` 中通过 `self._get_loss()` 设置。
        # `_base_loss` 包含特定的损失函数和链接函数。
        # 要最小化的损失指定了 GLM 的分布假设，即来自 EDM 的分布。
        # 以下是一些示例:

        # =======================  ========  ==========================
        # _base_loss               Link      Target Domain
        # =======================  ========  ==========================
        # HalfSquaredError         identity  y 任意实数
        # HalfPoissonLoss          log       0 <= y
        # HalfGammaLoss            log       0 < y
        # HalfTweedieLoss          log       依赖于 Tweedie power
        # HalfTweedieLossIdentity  identity  依赖于 Tweedie power
        # =======================  ========  ==========================

        # GLM 的链接函数，即从线性预测器 `X @ coeff + intercept` 到预测 `y_pred` 的映射。
        # 例如，使用对数链接函数时，我们有 `y_pred = exp(X @ coeff + intercept)`。
    # 定义参数约束字典，用于验证输入参数的类型和取值范围
    _parameter_constraints: dict = {
        "alpha": [Interval(Real, 0.0, None, closed="left")],
        "fit_intercept": ["boolean"],
        "solver": [
            StrOptions({"lbfgs", "newton-cholesky"}),  # solver 参数可选取值为 "lbfgs" 或 "newton-cholesky"
            Hidden(type),  # solver 参数的类型为隐藏类型
        ],
        "max_iter": [Interval(Integral, 1, None, closed="left")],  # max_iter 参数为大于等于1的整数
        "tol": [Interval(Real, 0.0, None, closed="neither")],  # tol 参数为大于0的实数
        "warm_start": ["boolean"],  # warm_start 参数为布尔类型
        "verbose": ["verbose"],  # verbose 参数为 verbose 类型
    }

    def __init__(
        self,
        *,
        alpha=1.0,  # 正则化强度，默认为1.0
        fit_intercept=True,  # 是否拟合截距，默认为True
        solver="lbfgs",  # 优化算法选择，默认为 "lbfgs"
        max_iter=100,  # 最大迭代次数，默认为100
        tol=1e-4,  # 迭代收敛容限，默认为1e-4
        warm_start=False,  # 是否使用前次拟合结果继续训练，默认为False
        verbose=0,  # 控制输出详细程度，默认为0（不输出）
    ):
        self.alpha = alpha  # 初始化 alpha 参数
        self.fit_intercept = fit_intercept  # 初始化 fit_intercept 参数
        self.solver = solver  # 初始化 solver 参数
        self.max_iter = max_iter  # 初始化 max_iter 参数
        self.tol = tol  # 初始化 tol 参数
        self.warm_start = warm_start  # 初始化 warm_start 参数
        self.verbose = verbose  # 初始化 verbose 参数

    @_fit_context(prefer_skip_nested_validation=True)
    def _linear_predictor(self, X):
        """Compute the linear_predictor = `X @ coef_ + intercept_`.

        Note that we often use the term raw_prediction instead of linear predictor.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_pred : array of shape (n_samples,)
            Returns predicted values of linear predictor.
        """
        check_is_fitted(self)  # 检查模型是否已拟合
        X = self._validate_data(
            X,
            accept_sparse=["csr", "csc", "coo"],  # 接受稀疏矩阵的格式
            dtype=[np.float64, np.float32],  # 数据类型为浮点数
            ensure_2d=True,  # 确保输入是二维的
            allow_nd=False,  # 不允许多维数组输入
            reset=False,  # 不重置输入数据
        )
        return X @ self.coef_ + self.intercept_  # 返回线性预测器的预测值

    def predict(self, X):
        """Predict using GLM with feature matrix X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_pred : array of shape (n_samples,)
            Returns predicted values.
        """
        # check_array is done in _linear_predictor
        raw_prediction = self._linear_predictor(X)  # 获取原始预测值
        y_pred = self._base_loss.link.inverse(raw_prediction)  # 将原始预测值转换为最终预测值
        return y_pred  # 返回预测结果
    def score(self, X, y, sample_weight=None):
        """Compute D^2, the percentage of deviance explained.

        D^2 is a generalization of the coefficient of determination R^2.
        R^2 uses squared error and D^2 uses the deviance of this GLM, see the
        :ref:`User Guide <regression_metrics>`.

        D^2 is defined as
        :math:`D^2 = 1-\\frac{D(y_{true},y_{pred})}{D_{null}}`,
        :math:`D_{null}` is the null deviance, i.e. the deviance of a model
        with intercept alone, which corresponds to :math:`y_{pred} = \\bar{y}`.
        The mean :math:`\\bar{y}` is averaged by sample_weight.
        Best possible score is 1.0 and it can be negative (because the model
        can be arbitrarily worse).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,)
            True values of target.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            D^2 of self.predict(X) w.r.t. y.
        """
        # TODO: Adapt link to User Guide in the docstring, once
        # https://github.com/scikit-learn/scikit-learn/pull/22118 is merged.
        
        # Note, default score defined in RegressorMixin is R^2 score.
        # TODO: make D^2 a score function in module metrics (and thereby get
        #       input validation and so on)
        
        # Compute raw predictions using the internal method _linear_predictor, which validates X
        raw_prediction = self._linear_predictor(X)

        # Check if y is a valid array for the predictions
        y = check_array(y, dtype=raw_prediction.dtype, order="C", ensure_2d=False)

        # If sample weights are provided, validate and convert them
        if sample_weight is not None:
            # Ensure sample weights are valid using _check_sample_weight
            sample_weight = _check_sample_weight(sample_weight, X, dtype=y.dtype)

        # Get the base loss function used by the model
        base_loss = self._base_loss

        # Ensure all values in y are within the acceptable range for the loss function
        if not base_loss.in_y_true_range(y):
            raise ValueError(
                "Some value(s) of y are out of the valid range of the loss"
                f" {base_loss.__name__}."
            )

        # Calculate the constant term used in the deviance computation
        constant = np.average(
            base_loss.constant_to_optimal_zero(y_true=y, sample_weight=None),
            weights=sample_weight,
        )

        # Compute the deviance between true values y and predicted values
        deviance = base_loss(
            y_true=y,
            raw_prediction=raw_prediction,
            sample_weight=sample_weight,
            n_threads=1,
        )

        # Compute the mean of y weighted by sample weights and apply the link function
        y_mean = base_loss.link.link(np.average(y, weights=sample_weight))

        # Compute the null deviance using the mean predictions and the same loss function
        deviance_null = base_loss(
            y_true=y,
            raw_prediction=np.tile(y_mean, y.shape[0]),
            sample_weight=sample_weight,
            n_threads=1,
        )

        # Compute and return D^2 score based on the computed values
        return 1 - (deviance + constant) / (deviance_null + constant)
    def _more_tags(self):
        try:
            # 如果尚未调用 fit 方法，则创建 BaseLoss 的实例。这是必要的，因为 TweedieRegressor
            # 在 fit 过程中可能会设置与 self._base_loss 不同的损失函数。
            base_loss = self._get_loss()
            # 返回一个字典，指示是否需要 y 值为正的标签，这取决于 base_loss 是否不在 [-1.0, +∞) 范围内。
            return {"requires_positive_y": not base_loss.in_y_true_range(-1.0)}
        except (ValueError, AttributeError, TypeError):
            # 当 TweedieRegressor 的链接或功率参数无效时会发生这种情况。在这种情况下，
            # 我们会回退到默认的标签。
            return {}

    def _get_loss(self):
        """这仅仅是因为 TweedieRegressor 的链接和功率参数而需要的。

        注意，我们不需要将 sample_weight 传递给损失类，因为这仅在设置 loss.constant_hessian 时需要，
        而广义线性模型不依赖于它。
        """
        # 返回 HalfSquaredError 的实例作为损失函数
        return HalfSquaredError()
# 定义一个泊松回归器，继承自广义线性回归器
class PoissonRegressor(_GeneralizedLinearRegressor):
    """Generalized Linear Model with a Poisson distribution.
    
    This regressor uses the 'log' link function.

    Read more in the :ref:`User Guide <Generalized_linear_models>`.

    .. versionadded:: 0.23

    Parameters
    ----------
    alpha : float, default=1
        Constant that multiplies the L2 penalty term and determines the
        regularization strength. ``alpha = 0`` is equivalent to unpenalized
        GLMs. In this case, the design matrix `X` must have full column rank
        (no collinearities).
        Values of `alpha` must be in the range `[0.0, inf)`.

    fit_intercept : bool, default=True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the linear predictor (`X @ coef + intercept`).

    solver : {'lbfgs', 'newton-cholesky'}, default='lbfgs'
        Algorithm to use in the optimization problem:

        'lbfgs'
            Calls scipy's L-BFGS-B optimizer.

        'newton-cholesky'
            Uses Newton-Raphson steps (in arbitrary precision arithmetic equivalent to
            iterated reweighted least squares) with an inner Cholesky based solver.
            This solver is a good choice for `n_samples` >> `n_features`, especially
            with one-hot encoded categorical features with rare categories. Be aware
            that the memory usage of this solver has a quadratic dependency on
            `n_features` because it explicitly computes the Hessian matrix.

            .. versionadded:: 1.2

    max_iter : int, default=100
        The maximal number of iterations for the solver.
        Values must be in the range `[1, inf)`.

    tol : float, default=1e-4
        Stopping criterion. For the lbfgs solver,
        the iteration will stop when ``max{|g_j|, j = 1, ..., d} <= tol``
        where ``g_j`` is the j-th component of the gradient (derivative) of
        the objective function.
        Values must be in the range `(0.0, inf)`.

    warm_start : bool, default=False
        If set to ``True``, reuse the solution of the previous call to ``fit``
        as initialization for ``coef_`` and ``intercept_`` .

    verbose : int, default=0
        For the lbfgs solver set verbose to any positive number for verbosity.
        Values must be in the range `[0, inf)`.

    Attributes
    ----------
    coef_ : array of shape (n_features,)
        Estimated coefficients for the linear predictor (`X @ coef_ +
        intercept_`) in the GLM.

    intercept_ : float
        Intercept (a.k.a. bias) added to linear predictor.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_iter_ : int
        Actual number of iterations used in the solver.
    """
    # 参数约束字典，继承自通用线性回归器的参数约束
    _parameter_constraints: dict = {
        **_GeneralizedLinearRegressor._parameter_constraints
    }

    # 初始化方法，设置泊松回归器的参数
    def __init__(
        self,
        *,
        alpha=1.0,                  # 正则化强度，默认为1.0
        fit_intercept=True,         # 是否拟合截距，默认为True
        solver="lbfgs",             # 优化算法选择，默认为"lbfgs"
        max_iter=100,               # 最大迭代次数，默认为100
        tol=1e-4,                   # 迭代停止阈值，默认为1e-4
        warm_start=False,           # 是否热启动，默认为False
        verbose=0,                  # 冗余信息级别，默认为0
    ):
        super().__init__(
            alpha=alpha,             # 设置正则化强度
            fit_intercept=fit_intercept,  # 设置是否拟合截距
            solver=solver,           # 设置优化算法
            max_iter=max_iter,       # 设置最大迭代次数
            tol=tol,                 # 设置迭代停止阈值
            warm_start=warm_start,   # 设置是否热启动
            verbose=verbose,         # 设置冗余信息级别
        )

    # 获取损失函数对象的方法，返回半泊松损失函数
    def _get_loss(self):
        return HalfPoissonLoss()
class GammaRegressor(_GeneralizedLinearRegressor):
    """Generalized Linear Model with a Gamma distribution.

    This regressor uses the 'log' link function.

    Read more in the :ref:`User Guide <Generalized_linear_models>`.

    .. versionadded:: 0.23

    Parameters
    ----------
    alpha : float, default=1
        Constant that multiplies the L2 penalty term and determines the
        regularization strength. ``alpha = 0`` is equivalent to unpenalized
        GLMs. In this case, the design matrix `X` must have full column rank
        (no collinearities).
        Values of `alpha` must be in the range `[0.0, inf)`.

    fit_intercept : bool, default=True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the linear predictor `X @ coef_ + intercept_`.

    solver : {'lbfgs', 'newton-cholesky'}, default='lbfgs'
        Algorithm to use in the optimization problem:

        'lbfgs'
            Calls scipy's L-BFGS-B optimizer.

        'newton-cholesky'
            Uses Newton-Raphson steps (in arbitrary precision arithmetic equivalent to
            iterated reweighted least squares) with an inner Cholesky based solver.
            This solver is a good choice for `n_samples` >> `n_features`, especially
            with one-hot encoded categorical features with rare categories. Be aware
            that the memory usage of this solver has a quadratic dependency on
            `n_features` because it explicitly computes the Hessian matrix.

            .. versionadded:: 1.2

    max_iter : int, default=100
        The maximal number of iterations for the solver.
        Values must be in the range `[1, inf)`.

    tol : float, default=1e-4
        Stopping criterion. For the lbfgs solver,
        the iteration will stop when ``max{|g_j|, j = 1, ..., d} <= tol``
        where ``g_j`` is the j-th component of the gradient (derivative) of
        the objective function.
        Values must be in the range `(0.0, inf)`.

    warm_start : bool, default=False
        If set to ``True``, reuse the solution of the previous call to ``fit``
        as initialization for `coef_` and `intercept_`.

    verbose : int, default=0
        For the lbfgs solver set verbose to any positive number for verbosity.
        Values must be in the range `[0, inf)`.

    Attributes
    ----------
    coef_ : array of shape (n_features,)
        Estimated coefficients for the linear predictor (`X @ coef_ +
        intercept_`) in the GLM.

    intercept_ : float
        Intercept (a.k.a. bias) added to linear predictor.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    n_iter_ : int
        Actual number of iterations used in the solver.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    """


The above code defines a class `GammaRegressor` which extends `_GeneralizedLinearRegressor`. It represents a generalized linear model using a Gamma distribution with a logarithmic link function. It includes parameters for regularization (`alpha`), intercept fitting (`fit_intercept`), solver selection (`solver`), maximum iterations (`max_iter`), convergence tolerance (`tol`), warm start capability (`warm_start`), and verbosity (`verbose`). It also defines several attributes related to model coefficients, intercepts, iterations, and feature names.
    # 定义一个基于泊松分布的广义线性回归模型，使用泊松回归算法进行预测。
    PoissonRegressor : Generalized Linear Model with a Poisson distribution.
    # 定义一个基于 Tweedie 分布的广义线性回归模型，使用 Tweedie 分布进行预测。
    TweedieRegressor : Generalized Linear Model with a Tweedie distribution.
    
    Examples
    --------
    # 导入线性模型的库
    >>> from sklearn import linear_model
    # 创建一个 Gamma 回归器对象
    >>> clf = linear_model.GammaRegressor()
    # 定义特征数据 X 和目标数据 y
    >>> X = [[1, 2], [2, 3], [3, 4], [4, 3]]
    >>> y = [19, 26, 33, 30]
    # 使用 X 和 y 进行拟合
    >>> clf.fit(X, y)
    GammaRegressor()
    # 计算拟合模型的得分
    >>> clf.score(X, y)
    0.773...
    # 输出模型的系数
    >>> clf.coef_
    array([0.072..., 0.066...])
    # 输出模型的截距
    >>> clf.intercept_
    2.896...
    # 预测新的数据点的目标值
    >>> clf.predict([[1, 0], [2, 8]])
    array([19.483..., 35.795...])
    """
    
    # 定义一个参数约束字典，继承自 _GeneralizedLinearRegressor 的参数约束
    _parameter_constraints: dict = {
        **_GeneralizedLinearRegressor._parameter_constraints
    }
    
    def __init__(
        self,
        *,
        alpha=1.0,
        fit_intercept=True,
        solver="lbfgs",
        max_iter=100,
        tol=1e-4,
        warm_start=False,
        verbose=0,
    ):
        # 调用父类的初始化方法，设置模型的参数
        super().__init__(
            alpha=alpha,
            fit_intercept=fit_intercept,
            solver=solver,
            max_iter=max_iter,
            tol=tol,
            warm_start=warm_start,
            verbose=verbose,
        )
    
    def _get_loss(self):
        # 返回 HalfGammaLoss 的损失函数对象
        return HalfGammaLoss()
class TweedieRegressor(_GeneralizedLinearRegressor):
    """Generalized Linear Model with a Tweedie distribution.

    This estimator can be used to model different GLMs depending on the
    ``power`` parameter, which determines the underlying distribution.

    Read more in the :ref:`User Guide <Generalized_linear_models>`.

    .. versionadded:: 0.23

    Parameters
    ----------
    power : float, default=0
            The power determines the underlying target distribution according
            to the following table:

            +-------+------------------------+
            | Power | Distribution           |
            +=======+========================+
            | 0     | Normal                 |
            +-------+------------------------+
            | 1     | Poisson                |
            +-------+------------------------+
            | (1,2) | Compound Poisson Gamma |
            +-------+------------------------+
            | 2     | Gamma                  |
            +-------+------------------------+
            | 3     | Inverse Gaussian       |
            +-------+------------------------+

            For ``0 < power < 1``, no distribution exists.

    alpha : float, default=1
        Constant that multiplies the L2 penalty term and determines the
        regularization strength. ``alpha = 0`` is equivalent to unpenalized
        GLMs. In this case, the design matrix `X` must have full column rank
        (no collinearities).
        Values of `alpha` must be in the range `[0.0, inf)`.

    fit_intercept : bool, default=True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the linear predictor (`X @ coef + intercept`).

    link : {'auto', 'identity', 'log'}, default='auto'
        The link function of the GLM, i.e. mapping from linear predictor
        `X @ coeff + intercept` to prediction `y_pred`. Option 'auto' sets
        the link depending on the chosen `power` parameter as follows:

        - 'identity' for ``power <= 0``, e.g. for the Normal distribution
        - 'log' for ``power > 0``, e.g. for Poisson, Gamma and Inverse Gaussian
          distributions

    solver : {'lbfgs', 'newton-cholesky'}, default='lbfgs'
        Algorithm to use in the optimization problem:

        'lbfgs'
            Calls scipy's L-BFGS-B optimizer.

        'newton-cholesky'
            Uses Newton-Raphson steps (in arbitrary precision arithmetic equivalent to
            iterated reweighted least squares) with an inner Cholesky based solver.
            This solver is a good choice for `n_samples` >> `n_features`, especially
            with one-hot encoded categorical features with rare categories. Be aware
            that the memory usage of this solver has a quadratic dependency on
            `n_features` because it explicitly computes the Hessian matrix.

            .. versionadded:: 1.2
    """

    # 定义 TweedieRegressor 类，继承自 _GeneralizedLinearRegressor
    """Generalized Linear Model with a Tweedie distribution.

    This estimator can be used to model different GLMs depending on the
    ``power`` parameter, which determines the underlying distribution.

    Read more in the :ref:`User Guide <Generalized_linear_models>`.

    .. versionadded:: 0.23

    Parameters
    ----------
    power : float, default=0
            The power determines the underlying target distribution according
            to the following table:

            +-------+------------------------+
            | Power | Distribution           |
            +=======+========================+
            | 0     | Normal                 |
            +-------+------------------------+
            | 1     | Poisson                |
            +-------+------------------------+
            | (1,2) | Compound Poisson Gamma |
            +-------+------------------------+
            | 2     | Gamma                  |
            +-------+------------------------+
            | 3     | Inverse Gaussian       |
            +-------+------------------------+

            For ``0 < power < 1``, no distribution exists.

    alpha : float, default=1
        Constant that multiplies the L2 penalty term and determines the
        regularization strength. ``alpha = 0`` is equivalent to unpenalized
        GLMs. In this case, the design matrix `X` must have full column rank
        (no collinearities).
        Values of `alpha` must be in the range `[0.0, inf)`.

    fit_intercept : bool, default=True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the linear predictor (`X @ coef + intercept`).

    link : {'auto', 'identity', 'log'}, default='auto'
        The link function of the GLM, i.e. mapping from linear predictor
        `X @ coeff + intercept` to prediction `y_pred`. Option 'auto' sets
        the link depending on the chosen `power` parameter as follows:

        - 'identity' for ``power <= 0``, e.g. for the Normal distribution
        - 'log' for ``power > 0``, e.g. for Poisson, Gamma and Inverse Gaussian
          distributions

    solver : {'lbfgs', 'newton-cholesky'}, default='lbfgs'
        Algorithm to use in the optimization problem:

        'lbfgs'
            Calls scipy's L-BFGS-B optimizer.

        'newton-cholesky'
            Uses Newton-Raphson steps (in arbitrary precision arithmetic equivalent to
            iterated reweighted least squares) with an inner Cholesky based solver.
            This solver is a good choice for `n_samples` >> `n_features`, especially
            with one-hot encoded categorical features with rare categories. Be aware
            that the memory usage of this solver has a quadratic dependency on
            `n_features` because it explicitly computes the Hessian matrix.

            .. versionadded:: 1.2
    """

    def __init__(self, power=0, alpha=1, fit_intercept=True, link='auto', solver='lbfgs'):
        # 初始化 TweedieRegressor 对象

        # 调用父类 _GeneralizedLinearRegressor 的初始化方法
        super().__init__(power=power, alpha=alpha, fit_intercept=fit_intercept, link=link, solver=solver)

        # 设置 TweedieRegressor 类的 power 参数
        self.power = power

        # 设置 TweedieRegressor 类的 alpha 参数
        self.alpha = alpha

        # 设置 TweedieRegressor 类的 fit_intercept 参数
        self.fit_intercept = fit_intercept

        # 设置 TweedieRegressor 类的 link 参数
        self.link = link

        # 设置 TweedieRegressor 类的 solver 参数
        self.solver = solver
    _parameter_constraints: dict = {
        **_GeneralizedLinearRegressor._parameter_constraints,
        "power": [Interval(Real, None, None, closed="neither")],
        "link": [StrOptions({"auto", "identity", "log"})],
    }



# 定义模型参数的约束条件字典，继承自泛化线性回归器的参数约束
_parameter_constraints: dict = {
    **_GeneralizedLinearRegressor._parameter_constraints,
    # 设置对 'power' 参数的约束，必须为非封闭的实数区间
    "power": [Interval(Real, None, None, closed="neither")],
    # 设置对 'link' 参数的约束，必须是预定义的字符串选项之一
    "link": [StrOptions({"auto", "identity", "log"})],
}



    def __init__(
        self,
        *,
        power=0.0,
        alpha=1.0,
        fit_intercept=True,
        link="auto",
        solver="lbfgs",
        max_iter=100,
        tol=1e-4,
        warm_start=False,
        verbose=0,
    ):
        # 调用父类的构造函数初始化对象
        super().__init__(
            alpha=alpha,
            fit_intercept=fit_intercept,
            solver=solver,
            max_iter=max_iter,
            tol=tol,
            warm_start=warm_start,
            verbose=verbose,
        )
        # 设置模型特定的参数
        self.link = link  # 设置链接函数的类型
        self.power = power  # 设置分布的功率参数



    Attributes
    ----------
    coef_ : array of shape (n_features,)
        线性预测器的估计系数 (`X @ coef_ + intercept_`) 在广义线性模型中。

    intercept_ : float
        线性预测器中添加的截距（偏差）。

    n_iter_ : int
        在求解器中实际使用的迭代次数。

    n_features_in_ : int
        在拟合期间看到的特征数。

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        在拟合期间看到的特征名称。仅当 `X` 中的特征名称都是字符串时定义。

        .. versionadded:: 1.0

    See Also
    --------
    PoissonRegressor : 使用泊松分布的广义线性模型。
    GammaRegressor : 使用伽马分布的广义线性模型。

    Examples
    --------
    >>> from sklearn import linear_model
    >>> clf = linear_model.TweedieRegressor()
    >>> X = [[1, 2], [2, 3], [3, 4], [4, 3]]
    >>> y = [2, 3.5, 5, 5.5]
    >>> clf.fit(X, y)
    TweedieRegressor()
    >>> clf.score(X, y)
    0.839...
    >>> clf.coef_
    array([0.599..., 0.299...])
    >>> clf.intercept_
    1.600...
    >>> clf.predict([[1, 1], [3, 4]])
    array([2.500..., 4.599...])


这些注释完整地解释了每行代码的作用，符合给定的注意事项和示例格式要求。
    # 定义一个方法 _get_loss，用于根据链接函数类型返回相应的损失函数对象
    def _get_loss(self):
        # 如果链接函数为 "auto"
        if self.link == "auto":
            # 如果指数小于等于零，选择恒等链接
            if self.power <= 0:
                # 返回一个使用恒等链接的半 Tweedie 损失函数对象
                return HalfTweedieLossIdentity(power=self.power)
            else:
                # 否则，选择对数链接
                # 返回一个使用对数链接的半 Tweedie 损失函数对象
                return HalfTweedieLoss(power=self.power)

        # 如果链接函数为 "log"
        if self.link == "log":
            # 返回一个使用对数链接的半 Tweedie 损失函数对象
            return HalfTweedieLoss(power=self.power)

        # 如果链接函数为 "identity"
        if self.link == "identity":
            # 返回一个使用恒等链接的半 Tweedie 损失函数对象
            return HalfTweedieLossIdentity(power=self.power)
```