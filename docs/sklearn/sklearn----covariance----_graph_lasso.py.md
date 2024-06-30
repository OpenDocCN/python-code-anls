# `D:\src\scipysrc\scikit-learn\sklearn\covariance\_graph_lasso.py`

```
"""GraphicalLasso: sparse inverse covariance estimation with an l1-penalized
estimator.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 导入所需的库和模块
import operator  # 导入 operator 模块，用于操作符函数
import sys  # 导入 sys 模块，用于与 Python 解释器进行交互
import time  # 导入 time 模块，用于时间操作
import warnings  # 导入 warnings 模块，用于警告处理
from numbers import Integral, Real  # 从 numbers 模块导入 Integral 和 Real 类型

import numpy as np  # 导入 NumPy 库，并使用 np 别名
from scipy import linalg  # 从 SciPy 库导入 linalg 模块

from ..base import _fit_context  # 导入 _fit_context 模块
from ..exceptions import ConvergenceWarning  # 导入 ConvergenceWarning 异常类

# mypy error: Module 'sklearn.linear_model' has no attribute '_cd_fast'
from ..linear_model import _cd_fast as cd_fast  # type: ignore  # 导入 _cd_fast，并忽略 mypy 类型检查
from ..linear_model import lars_path_gram  # 导入 lars_path_gram 函数
from ..model_selection import check_cv, cross_val_score  # 导入 check_cv 和 cross_val_score 函数
from ..utils import Bunch  # 导入 Bunch 类
from ..utils._param_validation import Interval, StrOptions, validate_params  # 导入 Interval、StrOptions 和 validate_params
from ..utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    _raise_for_params,
    _routing_enabled,
    process_routing,
)  # 导入 metadata_routing 相关函数和类
from ..utils.parallel import Parallel, delayed  # 导入 Parallel 和 delayed 函数
from ..utils.validation import (
    _is_arraylike_not_scalar,
    check_random_state,
    check_scalar,
)  # 导入验证相关的函数

from . import EmpiricalCovariance, empirical_covariance, log_likelihood  # 从当前包导入相关函数和类


# Helper functions to compute the objective and dual objective functions
# of the l1-penalized estimator
def _objective(mle, precision_, alpha):
    """Evaluation of the graphical-lasso objective function

    the objective function is made of a shifted scaled version of the
    normalized log-likelihood (i.e. its empirical mean over the samples) and a
    penalisation term to promote sparsity
    """
    p = precision_.shape[0]
    # 计算图形 Lasso 目标函数的评估值
    cost = -2.0 * log_likelihood(mle, precision_) + p * np.log(2 * np.pi)
    cost += alpha * (np.abs(precision_).sum() - np.abs(np.diag(precision_)).sum())
    return cost


def _dual_gap(emp_cov, precision_, alpha):
    """Expression of the dual gap convergence criterion

    The specific definition is given in Duchi "Projected Subgradient Methods
    for Learning Sparse Gaussians".
    """
    # 表达对偶间隙收敛标准的表达式
    gap = np.sum(emp_cov * precision_)
    gap -= precision_.shape[0]
    gap += alpha * (np.abs(precision_).sum() - np.abs(np.diag(precision_)).sum())
    return gap


# The g-lasso algorithm
def _graphical_lasso(
    emp_cov,
    alpha,
    *,
    cov_init=None,
    mode="cd",
    tol=1e-4,
    enet_tol=1e-4,
    max_iter=100,
    verbose=False,
    eps=np.finfo(np.float64).eps,
):
    _, n_features = emp_cov.shape
    if alpha == 0:
        # Early return without regularization
        precision_ = linalg.inv(emp_cov)
        cost = -2.0 * log_likelihood(emp_cov, precision_)
        cost += n_features * np.log(2 * np.pi)
        d_gap = np.sum(emp_cov * precision_) - n_features
        return emp_cov, precision_, (cost, d_gap), 0

    if cov_init is None:
        covariance_ = emp_cov.copy()
    else:
        covariance_ = cov_init.copy()
    # As a trivial regularization (Tikhonov like), we scale down the
    # off-diagonal coefficients of our starting point: This is needed, as
    # in the cross-validation the cov_init can easily be
    # 将协方差矩阵乘以0.95，以加快收敛速度
    covariance_ *= 0.95

    # 从经验协方差展平数组中获取对角线元素
    diagonal = emp_cov.flat[:: n_features + 1]

    # 将协方差矩阵的对角线元素替换为经验协方差的对角线元素
    covariance_.flat[:: n_features + 1] = diagonal

    # 计算协方差矩阵的伪逆，得到精度矩阵
    precision_ = linalg.pinvh(covariance_)

    # 创建一个包含0到n_features-1的数组索引
    indices = np.arange(n_features)

    # 初始化计数器，即使 `max_iter=0` 时也能正常工作
    i = 0

    # 初始化成本列表
    costs = list()

    # 根据不同的模式选择不同的 l1 回归求解器，处理数值错误
    if mode == "cd":
        errors = dict(over="raise", invalid="ignore")
    else:
        errors = dict(invalid="raise")

    # 捕获浮点数错误，如果系统条件太差以至于无法使用该求解器
    except FloatingPointError as e:
        e.args = (e.args[0] + ". The system is too ill-conditioned for this solver",)
        raise e

    # 返回计算得到的协方差矩阵、精度矩阵、成本列表和计数器值加1
    return covariance_, precision_, costs, i + 1
# 定义函数 alpha_max，用于计算给定样本协方差矩阵的最大 alpha 值
def alpha_max(emp_cov):
    # 复制输入的样本协方差矩阵，避免修改原始数据
    A = np.copy(emp_cov)
    # 将矩阵 A 对角线上的元素设为零，这些元素通常是协方差矩阵的方差部分
    A.flat[:: A.shape[0] + 1] = 0
    # 返回矩阵 A 中所有元素的绝对值的最大值，即非对角线上的最大值
    return np.max(np.abs(A))


# 装饰器函数 validate_params 用于验证参数类型和返回类型
@validate_params(
    {
        "emp_cov": ["array-like"],
        "return_costs": ["boolean"],
        "return_n_iter": ["boolean"],
    },
    prefer_skip_nested_validation=False,
)
# 定义函数 graphical_lasso，实现 L1-正则化的协方差估计
def graphical_lasso(
    emp_cov,
    alpha,
    *,
    mode="cd",
    tol=1e-4,
    enet_tol=1e-4,
    max_iter=100,
    verbose=False,
    return_costs=False,
    eps=np.finfo(np.float64).eps,
    return_n_iter=False,
):
    """L1-penalized covariance estimator.

    Read more in the :ref:`User Guide <sparse_inverse_covariance>`.

    .. versionchanged:: v0.20
        graph_lasso has been renamed to graphical_lasso

    Parameters
    ----------
    emp_cov : array-like of shape (n_features, n_features)
        Empirical covariance from which to compute the covariance estimate.

    alpha : float
        The regularization parameter: the higher alpha, the more
        regularization, the sparser the inverse covariance.
        Range is (0, inf].

    mode : {'cd', 'lars'}, default='cd'
        The Lasso solver to use: coordinate descent or LARS. Use LARS for
        very sparse underlying graphs, where p > n. Elsewhere prefer cd
        which is more numerically stable.

    tol : float, default=1e-4
        The tolerance to declare convergence: if the dual gap goes below
        this value, iterations are stopped. Range is (0, inf].

    enet_tol : float, default=1e-4
        The tolerance for the elastic net solver used to calculate the descent
        direction. This parameter controls the accuracy of the search direction
        for a given column update, not of the overall parameter estimate. Only
        used for mode='cd'. Range is (0, inf].

    max_iter : int, default=100
        The maximum number of iterations.

    verbose : bool, default=False
        If verbose is True, the objective function and dual gap are
        printed at each iteration.

    return_costs : bool, default=False
        If return_costs is True, the objective function and dual gap
        at each iteration are returned.

    eps : float, default=eps
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems. Default is `np.finfo(np.float64).eps`.

    return_n_iter : bool, default=False
        Whether or not to return the number of iterations.

    Returns
    -------
    model = GraphicalLasso(
        alpha=alpha,
        mode=mode,
        covariance="precomputed",
        tol=tol,
        enet_tol=enet_tol,
        max_iter=max_iter,
        verbose=verbose,
        eps=eps,
        assume_centered=True,
    ).fit(emp_cov)


# 使用 GraphicalLasso 类创建模型对象，进行稀疏逆协方差估计
model = GraphicalLasso(
    alpha=alpha,             # 控制稀疏性的正则化参数
    mode=mode,               # 算法模式
    covariance="precomputed",# 使用预计算的协方差矩阵
    tol=tol,                 # 算法的收敛容限
    enet_tol=enet_tol,       # 弹性网惩罚路径算法的容限
    max_iter=max_iter,       # 最大迭代次数
    verbose=verbose,         # 是否输出详细信息
    eps=eps,                 # 对角线元素小量，用于数值稳定性
    assume_centered=True,    # 是否假设数据已中心化
).fit(emp_cov)


    output = [model.covariance_, model.precision_]
    if return_costs:
        output.append(model.costs_)
    if return_n_iter:
        output.append(model.n_iter_)
    return tuple(output)


# 构建返回的输出结果，包括估计的协方差矩阵和稀疏精度矩阵，以及可能的迭代代价和迭代次数
output = [model.covariance_, model.precision_]
if return_costs:  # 如果设置了返回代价信息
    output.append(model.costs_)  # 添加优化函数值和对偶间隙的列表
if return_n_iter:  # 如果设置了返回迭代次数
    output.append(model.n_iter_)  # 添加迭代次数
return tuple(output)  # 返回结果作为元组
class BaseGraphicalLasso(EmpiricalCovariance):
    # 定义参数约束字典，继承自EmpiricalCovariance类的参数约束
    _parameter_constraints: dict = {
        **EmpiricalCovariance._parameter_constraints,
        "tol": [Interval(Real, 0, None, closed="right")],  # 容忍度的约束条件
        "enet_tol": [Interval(Real, 0, None, closed="right")],  # 弹性网络容忍度的约束条件
        "max_iter": [Interval(Integral, 0, None, closed="left")],  # 最大迭代次数的约束条件
        "mode": [StrOptions({"cd", "lars"})],  # 使用的优化模式的约束条件，可以是'cd'或'lars'
        "verbose": ["verbose"],  # 是否详细输出的约束条件
        "eps": [Interval(Real, 0, None, closed="both")],  # 机器精度的约束条件
    }
    _parameter_constraints.pop("store_precision")  # 移除存储精度的约束条件

    def __init__(
        self,
        tol=1e-4,
        enet_tol=1e-4,
        max_iter=100,
        mode="cd",
        verbose=False,
        eps=np.finfo(np.float64).eps,
        assume_centered=False,
    ):
        super().__init__(assume_centered=assume_centered)  # 调用父类的构造函数
        self.tol = tol  # 设置容忍度
        self.enet_tol = enet_tol  # 设置弹性网络容忍度
        self.max_iter = max_iter  # 设置最大迭代次数
        self.mode = mode  # 设置优化模式
        self.verbose = verbose  # 设置是否详细输出
        self.eps = eps  # 设置机器精度


class GraphicalLasso(BaseGraphicalLasso):
    """Sparse inverse covariance estimation with an l1-penalized estimator.

    Read more in the :ref:`User Guide <sparse_inverse_covariance>`.

    .. versionchanged:: v0.20
        GraphLasso has been renamed to GraphicalLasso

    Parameters
    ----------
    alpha : float, default=0.01
        The regularization parameter: the higher alpha, the more
        regularization, the sparser the inverse covariance.
        Range is (0, inf].

    mode : {'cd', 'lars'}, default='cd'
        The Lasso solver to use: coordinate descent or LARS. Use LARS for
        very sparse underlying graphs, where p > n. Elsewhere prefer cd
        which is more numerically stable.

    covariance : "precomputed", default=None
        If covariance is "precomputed", the input data in `fit` is assumed
        to be the covariance matrix. If `None`, the empirical covariance
        is estimated from the data `X`.

        .. versionadded:: 1.3

    tol : float, default=1e-4
        The tolerance to declare convergence: if the dual gap goes below
        this value, iterations are stopped. Range is (0, inf].

    enet_tol : float, default=1e-4
        The tolerance for the elastic net solver used to calculate the descent
        direction. This parameter controls the accuracy of the search direction
        for a given column update, not of the overall parameter estimate. Only
        used for mode='cd'. Range is (0, inf].

    max_iter : int, default=100
        The maximum number of iterations.

    verbose : bool, default=False
        If verbose is True, the objective function and dual gap are
        plotted at each iteration.

    eps : float, default=eps
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems. Default is `np.finfo(np.float64).eps`.

        .. versionadded:: 1.3
    """
    """
    assume_centered : bool, default=False
        如果为 True，则在计算之前不对数据进行中心化。
        当处理平均值接近但不完全为零的数据时很有用。
        如果为 False，则在计算之前对数据进行中心化。

    Attributes
    ----------
    location_ : ndarray of shape (n_features,)
        估计的位置，即估计的均值。

    covariance_ : ndarray of shape (n_features, n_features)
        估计的协方差矩阵。

    precision_ : ndarray of shape (n_features, n_features)
        估计的伪逆矩阵。

    n_iter_ : int
        运行的迭代次数。

    costs_ : list of (objective, dual_gap) pairs
        每次迭代时目标函数值和对偶间隙的列表。仅在 return_costs 为 True 时返回。

        .. versionadded:: 1.3

    n_features_in_ : int
        在拟合期间看到的特征数量。

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        在拟合期间看到的特征名称。仅当 `X` 具有全部为字符串的特征名称时定义。

        .. versionadded:: 1.0

    See Also
    --------
    graphical_lasso : L1惩罚协方差估计器。
    GraphicalLassoCV : 稀疏逆协方差，带有交叉验证选择的 L1 惩罚。

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.covariance import GraphicalLasso
    >>> true_cov = np.array([[0.8, 0.0, 0.2, 0.0],
    ...                      [0.0, 0.4, 0.0, 0.0],
    ...                      [0.2, 0.0, 0.3, 0.1],
    ...                      [0.0, 0.0, 0.1, 0.7]])
    >>> np.random.seed(0)
    >>> X = np.random.multivariate_normal(mean=[0, 0, 0, 0],
    ...                                   cov=true_cov,
    ...                                   size=200)
    >>> cov = GraphicalLasso().fit(X)
    >>> np.around(cov.covariance_, decimals=3)
    array([[0.816, 0.049, 0.218, 0.019],
           [0.049, 0.364, 0.017, 0.034],
           [0.218, 0.017, 0.322, 0.093],
           [0.019, 0.034, 0.093, 0.69 ]])
    >>> np.around(cov.location_, decimals=3)
    array([0.073, 0.04 , 0.038, 0.143])
    """

    _parameter_constraints: dict = {
        **BaseGraphicalLasso._parameter_constraints,
        "alpha": [Interval(Real, 0, None, closed="both")],
        "covariance": [StrOptions({"precomputed"}), None],
    }

    def __init__(
        self,
        alpha=0.01,
        *,
        mode="cd",
        covariance=None,
        tol=1e-4,
        enet_tol=1e-4,
        max_iter=100,
        verbose=False,
        eps=np.finfo(np.float64).eps,
        assume_centered=False,
    ):
        """
        构造函数，初始化 GraphicalLasso 对象。

        Parameters
        ----------
        alpha : float, optional, default=0.01
            控制稀疏性的参数。
        mode : {'cd', 'lars', 'lars_path'}, default='cd'
            用于控制算法的求解模式。
        covariance : str or None, optional, default=None
            协方差估计方法，可以是 "precomputed" 或 None。
        tol : float, optional, default=1e-4
            算法收敛的容忍度。
        enet_tol : float, optional, default=1e-4
            弹性网络算法的容忍度。
        max_iter : int, optional, default=100
            最大迭代次数。
        verbose : bool, optional, default=False
            是否输出详细信息。
        eps : float, optional, default=np.finfo(np.float64).eps
            算法的机器精度。
        assume_centered : bool, optional, default=False
            如果为 True，则在计算之前不对数据进行中心化。

        """
        super().__init__(
            tol=tol,
            enet_tol=enet_tol,
            max_iter=max_iter,
            mode=mode,
            verbose=verbose,
            eps=eps,
            assume_centered=assume_centered,
        )
        self.alpha = alpha
        self.covariance = covariance
    # 应用装饰器 _fit_context，并设置参数 prefer_skip_nested_validation=True
    @_fit_context(prefer_skip_nested_validation=True)
    # 定义 fit 方法，用于将 GraphicalLasso 模型拟合到数据 X 上
    def fit(self, X, y=None):
        """Fit the GraphicalLasso model to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data from which to compute the covariance estimate.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # 对于单个特征，计算协方差没有意义，所以确保数据 X 至少有 2 个特征和 2 个样本
        X = self._validate_data(X, ensure_min_features=2, ensure_min_samples=2)

        # 如果 covariance 属性设置为 "precomputed"，则直接将 X 赋给 emp_cov，并初始化 location_ 为全零数组
        if self.covariance == "precomputed":
            emp_cov = X.copy()
            self.location_ = np.zeros(X.shape[1])
        else:
            # 否则，计算 X 的经验协方差矩阵，根据 assume_centered 的设置进行处理
            emp_cov = empirical_covariance(X, assume_centered=self.assume_centered)
            if self.assume_centered:
                self.location_ = np.zeros(X.shape[1])
            else:
                self.location_ = X.mean(0)

        # 调用 _graphical_lasso 函数进行图形化 Lasso 拟合，返回协方差矩阵、精度矩阵、代价值和迭代次数
        self.covariance_, self.precision_, self.costs_, self.n_iter_ = _graphical_lasso(
            emp_cov,
            alpha=self.alpha,
            cov_init=None,
            mode=self.mode,
            tol=self.tol,
            enet_tol=self.enet_tol,
            max_iter=self.max_iter,
            verbose=self.verbose,
            eps=self.eps,
        )
        # 返回实例本身
        return self
# 使用 GraphicalLasso 进行交叉验证

def graphical_lasso_path(
    X,
    alphas,
    cov_init=None,
    X_test=None,
    mode="cd",
    tol=1e-4,
    enet_tol=1e-4,
    max_iter=100,
    verbose=False,
    eps=np.finfo(np.float64).eps,
):
    """l1-penalized covariance estimator along a path of decreasing alphas
    
    Read more in the :ref:`User Guide <sparse_inverse_covariance>`.
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data from which to compute the covariance estimate.
        
    alphas : array-like of shape (n_alphas,)
        The list of regularization parameters, decreasing order.
        
    cov_init : array of shape (n_features, n_features), default=None
        The initial guess for the covariance.
        
    X_test : array of shape (n_test_samples, n_features), default=None
        Optional test matrix to measure generalisation error.
        
    mode : {'cd', 'lars'}, default='cd'
        The Lasso solver to use: coordinate descent or LARS. Use LARS for
        very sparse underlying graphs, where p > n. Elsewhere prefer cd
        which is more numerically stable.
        
    tol : float, default=1e-4
        The tolerance to declare convergence: if the dual gap goes below
        this value, iterations are stopped. The tolerance must be a positive
        number.
        
    enet_tol : float, default=1e-4
        The tolerance for the elastic net solver used to calculate the descent
        direction. This parameter controls the accuracy of the search direction
        for a given column update, not of the overall parameter estimate. Only
        used for mode='cd'. The tolerance must be a positive number.
        
    max_iter : int, default=100
        The maximum number of iterations. This parameter should be a strictly
        positive integer.
        
    verbose : int or bool, default=False
        The higher the verbosity flag, the more information is printed
        during the fitting.
        
    eps : float, default=eps
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems. Default is `np.finfo(np.float64).eps`.
        
        .. versionadded:: 1.3
        
    Returns
    -------
    covariances_ : list of shape (n_alphas,) of ndarray of shape \
            (n_features, n_features)
        The estimated covariance matrices.
        
    precisions_ : list of shape (n_alphas,) of ndarray of shape \
            (n_features, n_features)
        The estimated (sparse) precision matrices.
        
    scores_ : list of shape (n_alphas,), dtype=float
        The generalisation error (log-likelihood) on the test data.
        Returned only if test data is passed.
    """
    # 根据 verbose 参数设置内部的详细程度
    inner_verbose = max(0, verbose - 1)
    
    # 计算输入数据 X 的经验协方差矩阵
    emp_cov = empirical_covariance(X)
    
    # 如果未提供初始协方差矩阵，则将经验协方差矩阵复制给 covariance_
    if cov_init is None:
        covariance_ = emp_cov.copy()
    else:
        covariance_ = cov_init
    
    # 初始化用于存储结果的列表
    covariances_ = list()
    precisions_ = list()
    scores_ = list()
    # 如果给定了测试集 X_test，则计算测试集的经验协方差
    if X_test is not None:
        test_emp_cov = empirical_covariance(X_test)

    # 对每个正则化参数 alpha 进行迭代
    for alpha in alphas:
        try:
            # 调用 _graphical_lasso 函数进行图形化套索估计，捕获可能出现的错误并继续执行
            covariance_, precision_, _, _ = _graphical_lasso(
                emp_cov,
                alpha=alpha,
                cov_init=covariance_,
                mode=mode,
                tol=tol,
                enet_tol=enet_tol,
                max_iter=max_iter,
                verbose=inner_verbose,
                eps=eps,
            )
            # 将计算得到的协方差矩阵和精度矩阵添加到结果列表中
            covariances_.append(covariance_)
            precisions_.append(precision_)
            
            # 如果给定了测试集 X_test，则计算当前 alpha 值下的对数似然值
            if X_test is not None:
                this_score = log_likelihood(test_emp_cov, precision_)
        
        # 捕获浮点运算错误，将当前分数设为负无穷，并将协方差和精度矩阵列表中对应位置设为 NaN
        except FloatingPointError:
            this_score = -np.inf
            covariances_.append(np.nan)
            precisions_.append(np.nan)
        
        # 如果给定了测试集 X_test，则根据计算结果判断当前分数是否为有限值
        if X_test is not None:
            if not np.isfinite(this_score):
                this_score = -np.inf
            # 将当前分数添加到分数列表中
            scores_.append(this_score)
        
        # 如果 verbose 设置为 1，则输出一个点符号表示进度
        if verbose == 1:
            sys.stderr.write(".")
        
        # 如果 verbose 大于 1，则根据情况输出当前 alpha 值和对应分数信息
        elif verbose > 1:
            if X_test is not None:
                print(
                    "[graphical_lasso_path] alpha: %.2e, score: %.2e"
                    % (alpha, this_score)
                )
            else:
                print("[graphical_lasso_path] alpha: %.2e" % alpha)
    
    # 如果给定了测试集 X_test，则返回协方差矩阵列表、精度矩阵列表和分数列表
    if X_test is not None:
        return covariances_, precisions_, scores_
    # 如果未给定测试集 X_test，则仅返回协方差矩阵列表和精度矩阵列表
    return covariances_, precisions_
class GraphicalLassoCV(BaseGraphicalLasso):
    """Sparse inverse covariance w/ cross-validated choice of the l1 penalty.

    See glossary entry for :term:`cross-validation estimator`.

    Read more in the :ref:`User Guide <sparse_inverse_covariance>`.

    .. versionchanged:: v0.20
        GraphLassoCV has been renamed to GraphicalLassoCV

    Parameters
    ----------
    alphas : int or array-like of shape (n_alphas,), dtype=float, default=4
        如果给定整数，将确定要使用的 alpha 网格上的点数。如果给定列表，将给出要使用的网格。更多详细信息请参阅类文档字符串中的注释。
        整数时范围为 [1, inf)。
        浮点数组时范围为 (0, inf]。

    n_refinements : int, default=4
        网格细化的次数。如果传入明确的 alphas 值，则不使用此参数。
        范围为 [1, inf)。

    cv : int, cross-validation generator or iterable, default=None
        确定交叉验证的拆分策略。cv 的可能输入有：

        - None，使用默认的 5 折交叉验证，
        - 整数，指定折数。
        - CV 分割器，
        - 一个可迭代对象，产生作为索引数组的 (train, test) 拆分。

        对于整数/None 输入，默认使用 :class:`~sklearn.model_selection.KFold`。

        有关可用的交叉验证策略，请参阅 :ref:`User Guide <cross_validation>`。

        .. versionchanged:: 0.20
            如果 None，则 ``cv`` 的默认值从 3 折更改为 5 折。

    tol : float, default=1e-4
        声明收敛的容差：如果二元间隙低于此值，则停止迭代。
        范围为 (0, inf]。

    enet_tol : float, default=1e-4
        用于计算下降方向的弹性网络解算器的容差。
        此参数控制给定列更新的搜索方向的准确性，而不是整体参数估计的准确性。
        仅在 mode='cd' 时使用。
        范围为 (0, inf]。

    max_iter : int, default=100
        最大迭代次数。

    mode : {'cd', 'lars'}, default='cd'
        要使用的 Lasso 求解器：坐标下降或 LARS。
        对于非常稀疏的基础图形，特征数大于样本数的情况下使用 LARS。
        其他情况下，推荐使用 cd，因为它更加数值稳定。

    n_jobs : int, default=None
        并行运行的作业数。
        ``None`` 表示除非在 :obj:`joblib.parallel_backend` 上下文中，否则为 1。
        ``-1`` 表示使用所有处理器。
        有关更多详细信息，请参阅 :term:`Glossary <n_jobs>`。

        .. versionchanged:: v0.20
           `n_jobs` 的默认值从 1 更改为 None。

    verbose : bool, default=False
        如果 verbose 为 True，则在每次迭代时打印目标函数和对偶间隙。
    """
    eps : float, default=eps
        # Cholesky分解对角因子计算中的机器精度正则化。对于条件非常糟糕的系统，增加此值。
        # 默认为 `np.finfo(np.float64).eps`。

        .. versionadded:: 1.3

    assume_centered : bool, default=False
        # 如果为True，在计算之前数据不会被居中。
        # 在处理均值接近但不完全为零的数据时很有用。
        # 如果为False，在计算之前数据会被居中。

    Attributes
    ----------
    location_ : ndarray of shape (n_features,)
        # 估计的位置，即估计的均值。

    covariance_ : ndarray of shape (n_features, n_features)
        # 估计的协方差矩阵。

    precision_ : ndarray of shape (n_features, n_features)
        # 估计的精确度矩阵（协方差的逆）。

    costs_ : list of (objective, dual_gap) pairs
        # 每次迭代的目标函数值和对偶间隙的列表。
        # 仅在 return_costs 为 True 时返回。

        .. versionadded:: 1.3

    alpha_ : float
        # 选择的惩罚参数。

    cv_results_ : dict of ndarrays
        # 一个字典，包含以下键：

        alphas : ndarray of shape (n_alphas,)
            # 探索的所有惩罚参数。

        split(k)_test_score : ndarray of shape (n_alphas,)
            # 第 (k) 折的留出数据上的对数似然分数。

            .. versionadded:: 1.0

        mean_test_score : ndarray of shape (n_alphas,)
            # 所有折的分数的平均值。

            .. versionadded:: 1.0

        std_test_score : ndarray of shape (n_alphas,)
            # 所有折的分数的标准差。

            .. versionadded:: 1.0

    n_iter_ : int
        # 用于最优 alpha 的运行迭代次数。

    n_features_in_ : int
        # 在拟合期间观察到的特征数。

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        # 在 `fit` 期间观察到的特征的名称。
        # 仅当 `X` 具有所有字符串类型的特征名称时定义。

        .. versionadded:: 1.0

    See Also
    --------
    graphical_lasso : L1惩罚协方差估计器。
    GraphicalLasso : 使用L1惩罚估计稀疏逆协方差的估计器。

    Notes
    -----
    # 在一个迭代精细化的网格上搜索最优惩罚参数（`alpha`）：
    # 首先计算网格上的交叉验证分数，然后围绕最大值重新定义一个新的精细化网格，如此类推。

    # 面临的一个挑战是求解器可能无法收敛到一个条件良好的估计值。
    # 对应的 `alpha` 值可能会变成缺失值，但最优值可能接近这些缺失值。

    # 在 `fit` 中，一旦通过交叉验证找到最佳参数 `alpha`，就会再次使用整个训练集拟合模型。

    Examples
    # 定义参数约束字典，继承自 BaseGraphicalLasso 的参数约束，并添加额外约束
    _parameter_constraints: dict = {
        **BaseGraphicalLasso._parameter_constraints,  # 继承基类的参数约束
        "alphas": [Interval(Integral, 0, None, closed="left"), "array-like"],  # alpha 参数的约束条件
        "n_refinements": [Interval(Integral, 1, None, closed="left")],  # n_refinements 参数的约束条件
        "cv": ["cv_object"],  # cv 参数的约束条件，应为一个交叉验证对象
        "n_jobs": [Integral, None],  # n_jobs 参数的约束条件，为整数或 None
    }
    
    def __init__(
        self,
        *,
        alphas=4,
        n_refinements=4,
        cv=None,
        tol=1e-4,
        enet_tol=1e-4,
        max_iter=100,
        mode="cd",
        n_jobs=None,
        verbose=False,
        eps=np.finfo(np.float64).eps,
        assume_centered=False,
    ):
        # 调用父类的初始化方法，设置基本参数
        super().__init__(
            tol=tol,
            enet_tol=enet_tol,
            max_iter=max_iter,
            mode=mode,
            verbose=verbose,
            eps=eps,
            assume_centered=assume_centered,
        )
        # 设置当前类的特有参数
        self.alphas = alphas  # 设置 alphas 参数
        self.n_refinements = n_refinements  # 设置 n_refinements 参数
        self.cv = cv  # 设置 cv 参数
        self.n_jobs = n_jobs  # 设置 n_jobs 参数
    
    @_fit_context(prefer_skip_nested_validation=True)
    def get_metadata_routing(self):
        """获取对象的元数据路由信息。
    
        请查阅 :ref:`用户指南 <metadata_routing>` 了解路由机制的工作方式。
    
        .. versionadded:: 1.5
    
        Returns
        -------
        routing : MetadataRouter
            封装路由信息的 :class:`~sklearn.utils.metadata_routing.MetadataRouter` 对象。
        """
        router = MetadataRouter(owner=self.__class__.__name__).add(
            splitter=check_cv(self.cv),  # 使用 check_cv 方法检查并设置交叉验证分割器
            method_mapping=MethodMapping().add(callee="split", caller="fit"),  # 添加方法映射关系
        )
        return router  # 返回路由器对象
```