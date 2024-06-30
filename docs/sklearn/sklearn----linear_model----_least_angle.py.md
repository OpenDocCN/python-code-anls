# `D:\src\scipysrc\scikit-learn\sklearn\linear_model\_least_angle.py`

```
"""
Least Angle Regression algorithm. See the documentation on the
Generalized Linear Model for a complete discussion.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 导入系统模块和警告模块
import sys
import warnings
# 导入数学模块中的对数函数
from math import log
# 导入整数和实数类型判断模块
from numbers import Integral, Real

# 导入第三方库
import numpy as np
# 导入科学计算库中的插值和线性代数模块
from scipy import interpolate, linalg
# 导入线性代数库中的 LAPACK 函数接口
from scipy.linalg.lapack import get_lapack_funcs

# 导入 scikit-learn 中的模型和工具函数
from ..base import MultiOutputMixin, RegressorMixin, _fit_context
# 导入 scikit-learn 中的异常处理模块
from ..exceptions import ConvergenceWarning
# 导入 scikit-learn 中的交叉验证函数
from ..model_selection import check_cv

# 导入 scikit-learn 中的工具函数和元组处理模块
# mypy error: Module 'sklearn.utils' has no attribute 'arrayfuncs'
from ..utils import (  # type: ignore
    Bunch,
    arrayfuncs,
    as_float_array,
    check_random_state,
)
# 导入 scikit-learn 中的元数据和参数验证模块
from ..utils._metadata_requests import (
    MetadataRouter,
    MethodMapping,
    _raise_for_params,
    _routing_enabled,
    process_routing,
)
from ..utils._param_validation import Hidden, Interval, StrOptions, validate_params
# 导入 scikit-learn 中的并行计算模块
from ..utils.parallel import Parallel, delayed
# 导入 scikit-learn 中的线性模型基类和回归模型
from ._base import LinearModel, LinearRegression, _preprocess_data

# 定义解三角形方程时的参数设置
SOLVE_TRIANGULAR_ARGS = {"check_finite": False}

# 参数验证装饰器，用于验证输入参数的类型和取值范围
@validate_params(
    {
        "X": [np.ndarray, None],
        "y": [np.ndarray, None],
        "Xy": [np.ndarray, None],
        "Gram": [StrOptions({"auto"}), "boolean", np.ndarray, None],
        "max_iter": [Interval(Integral, 0, None, closed="left")],
        "alpha_min": [Interval(Real, 0, None, closed="left")],
        "method": [StrOptions({"lar", "lasso"})],
        "copy_X": ["boolean"],
        "eps": [Interval(Real, 0, None, closed="neither"), None],
        "copy_Gram": ["boolean"],
        "verbose": ["verbose"],
        "return_path": ["boolean"],
        "return_n_iter": ["boolean"],
        "positive": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
# 定义 LARS 算法主函数
def lars_path(
    X,
    y,
    Xy=None,
    *,
    Gram=None,
    max_iter=500,
    alpha_min=0,
    method="lar",
    copy_X=True,
    eps=np.finfo(float).eps,
    copy_Gram=True,
    verbose=0,
    return_path=True,
    return_n_iter=False,
    positive=False,
):
    """Compute Least Angle Regression or Lasso path using the LARS algorithm.

    The optimization objective for the case method='lasso' is::

    (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

    in the case of method='lar', the objective function is only known in
    the form of an implicit equation (see discussion in [1]_).

    Read more in the :ref:`User Guide <least_angle_regression>`.

    Parameters
    ----------
    X : None or ndarray of shape (n_samples, n_features)
        Input data. Note that if X is `None` then the Gram matrix must be
        specified, i.e., cannot be `None` or `False`.

    y : None or ndarray of shape (n_samples,)
        Input targets.

    Xy : array-like of shape (n_features,), default=None
        `Xy = X.T @ y` that can be precomputed. It is useful
        only when the Gram matrix is precomputed.
    Gram : None, 'auto', bool, ndarray of shape (n_features, n_features), \
            default=None
        # Gram 矩阵参数，可以是 None、'auto'、布尔型、形状为 (n_features, n_features) 的 ndarray
        # 如果为 'auto'，则从给定的 X 计算 Gram 矩阵，如果样本数大于特征数

    max_iter : int, default=500
        # 最大迭代次数，默认为 500
        # 设置为无穷大以取消迭代次数限制

    alpha_min : float, default=0
        # 路径中的最小相关性，对应于 Lasso 正则化参数 `alpha`

    method : {'lar', 'lasso'}, default='lar'
        # 指定返回的模型类型。选择 `'lar'` 表示最小角度回归（Least Angle Regression），选择 `'lasso'` 表示 Lasso 回归

    copy_X : bool, default=True
        # 如果为 `False`，则会覆盖 X

    eps : float, default=np.finfo(float).eps
        # 在计算 Cholesky 对角因子时的机器精度正则化。对于条件非常差的系统，增加此值
        # 与某些基于迭代优化算法中的 `tol` 参数不同，此参数不控制优化的容差

    copy_Gram : bool, default=True
        # 如果为 `False`，则会覆盖 Gram

    verbose : int, default=0
        # 控制输出详细程度

    return_path : bool, default=True
        # 如果为 `True`，则返回整个路径；否则仅返回路径的最后一个点

    return_n_iter : bool, default=False
        # 是否返回迭代次数

    positive : bool, default=False
        # 限制系数为非负数
        # 该选项仅允许与方法 'lasso' 一起使用。注意，对于较小的 alpha 值，模型系数将不会收敛到普通最小二乘解
        # 仅通过逐步 Lars-Lasso 算法达到的步骤中的系数通常与坐标下降 `lasso_path` 函数的解一致

    Returns
    -------
    alphas : ndarray of shape (n_alphas + 1,)
        # 每次迭代中的最大协方差（绝对值）。`n_alphas` 是 `max_iter`、`n_features` 或路径中具有 `alpha >= alpha_min` 的节点数中较小的那个

    active : ndarray of shape (n_alphas,)
        # 路径结束时的活动变量的索引

    coefs : ndarray of shape (n_features, n_alphas + 1)
        # 路径上的系数

    n_iter : int
        # 运行的迭代次数。仅在 `return_n_iter` 设置为 True 时返回

    See Also
    --------
    lars_path_gram : 在足够统计模式下计算 LARS 路径
    lasso_path : 使用坐标下降计算 Lasso 路径
    LassoLars : 使用最小角度回归（Lars）拟合的 Lasso 模型
    Lars : 最小角度回归（LAR）模型
    LassoLarsCV : 使用 LARS 算法进行交叉验证的 Lasso
    """
    LarsCV : 交叉验证的最小角度回归模型。
    sklearn.decomposition.sparse_encode : 稀疏编码。

    References
    ----------
    .. [1] "Least Angle Regression", Efron et al.
           http://statweb.stanford.edu/~tibs/ftp/lars.pdf

    .. [2] `Wikipedia entry on the Least-angle regression
           <https://en.wikipedia.org/wiki/Least-angle_regression>`_

    .. [3] `Wikipedia entry on the Lasso
           <https://en.wikipedia.org/wiki/Lasso_(statistics)>`_

    Examples
    --------
    >>> from sklearn.linear_model import lars_path
    >>> from sklearn.datasets import make_regression
    >>> X, y, true_coef = make_regression(
    ...    n_samples=100, n_features=5, n_informative=2, coef=True, random_state=0
    ... )
    >>> true_coef
    array([ 0.        ,  0.        ,  0.        , 97.9..., 45.7...])
    >>> alphas, _, estimated_coef = lars_path(X, y)
    >>> alphas.shape
    (3,)
    >>> estimated_coef
    array([[ 0.     ,  0.     ,  0.     ],
           [ 0.     ,  0.     ,  0.     ],
           [ 0.     ,  0.     ,  0.     ],
           [ 0.     , 46.96..., 97.99...],
           [ 0.     ,  0.     , 45.70...]])
    """
    if X is None and Gram is not None:
        raise ValueError(
            "X cannot be None if Gram is not None"
            "Use lars_path_gram to avoid passing X and y."
        )
    return _lars_path_solver(
        X=X,
        y=y,
        Xy=Xy,
        Gram=Gram,
        n_samples=None,
        max_iter=max_iter,
        alpha_min=alpha_min,
        method=method,
        copy_X=copy_X,
        eps=eps,
        copy_Gram=copy_Gram,
        verbose=verbose,
        return_path=return_path,
        return_n_iter=return_n_iter,
        positive=positive,
    )
# 使用装饰器 @validate_params 对函数进行参数验证，确保输入参数的类型和取值范围符合预期
@validate_params(
    {
        "Xy": [np.ndarray],  # 参数 Xy 应为 numpy 数组
        "Gram": [np.ndarray],  # 参数 Gram 应为 numpy 数组
        "n_samples": [Interval(Integral, 0, None, closed="left")],  # 参数 n_samples 应为大于等于零的整数
        "max_iter": [Interval(Integral, 0, None, closed="left")],  # 参数 max_iter 应为大于等于零的整数
        "alpha_min": [Interval(Real, 0, None, closed="left")],  # 参数 alpha_min 应为大于等于零的实数
        "method": [StrOptions({"lar", "lasso"})],  # 参数 method 只能取 "lar" 或 "lasso"
        "copy_X": ["boolean"],  # 参数 copy_X 应为布尔值
        "eps": [Interval(Real, 0, None, closed="neither"), None],  # 参数 eps 应为大于零的实数，或者可以为 None
        "copy_Gram": ["boolean"],  # 参数 copy_Gram 应为布尔值
        "verbose": ["verbose"],  # 参数 verbose 应符合输出详细程度的约定
        "return_path": ["boolean"],  # 参数 return_path 应为布尔值
        "return_n_iter": ["boolean"],  # 参数 return_n_iter 应为布尔值
        "positive": ["boolean"],  # 参数 positive 应为布尔值
    },
    prefer_skip_nested_validation=True,  # 优先跳过嵌套验证
)
def lars_path_gram(
    Xy,
    Gram,
    *,
    n_samples,
    max_iter=500,  # 默认最大迭代次数为 500
    alpha_min=0,  # 默认最小相关性为 0
    method="lar",  # 默认使用最小角回归方法
    copy_X=True,  # 默认复制 X
    eps=np.finfo(float).eps,  # 默认机器精度的正则化参数
    copy_Gram=True,  # 默认复制 Gram
    verbose=0,  # 默认输出详细程度为 0
    return_path=True,  # 默认返回整个路径
    return_n_iter=False,  # 默认不返回迭代次数
    positive=False,  # 默认非正数
):
    """The lars_path in the sufficient stats mode.

    The optimization objective for the case method='lasso' is::

    (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

    in the case of method='lar', the objective function is only known in
    the form of an implicit equation (see discussion in [1]_).

    Read more in the :ref:`User Guide <least_angle_regression>`.

    Parameters
    ----------
    Xy : ndarray of shape (n_features,)
        `Xy = X.T @ y`.

    Gram : ndarray of shape (n_features, n_features)
        `Gram = X.T @ X`.

    n_samples : int
        Equivalent size of sample.

    max_iter : int, default=500
        Maximum number of iterations to perform, set to infinity for no limit.

    alpha_min : float, default=0
        Minimum correlation along the path. It corresponds to the
        regularization parameter alpha parameter in the Lasso.

    method : {'lar', 'lasso'}, default='lar'
        Specifies the returned model. Select `'lar'` for Least Angle
        Regression, ``'lasso'`` for the Lasso.

    copy_X : bool, default=True
        If `False`, `X` is overwritten.

    eps : float, default=np.finfo(float).eps
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems. Unlike the `tol` parameter in some iterative
        optimization-based algorithms, this parameter does not control
        the tolerance of the optimization.

    copy_Gram : bool, default=True
        If `False`, `Gram` is overwritten.

    verbose : int, default=0
        Controls output verbosity.

    return_path : bool, default=True
        If `return_path==True` returns the entire path, else returns only the
        last point of the path.

    return_n_iter : bool, default=False
        Whether to return the number of iterations.
    """
    # positive : bool, default=False
    # 将系数限制为 >= 0。
    # 此选项仅适用于方法 'lasso'。注意，对于较小的 alpha 值，模型系数将不会收敛到普通最小二乘解。
    # 通常仅有通过逐步 Lars-Lasso 算法达到的最小 alpha 值（当 `fit_path=True` 时为 `alphas_[alphas_ > 0.].min()`）的系数
    # 与坐标下降 lasso_path 函数的解一致。

    # Returns
    # -------
    # alphas : ndarray of shape (n_alphas + 1,)
    # 每次迭代中最大协方差（绝对值）。
    # `n_alphas` 是 `max_iter`、`n_features` 或路径中 `alpha >= alpha_min` 的节点数，以较小者为准。

    # active : ndarray of shape (n_alphas,)
    # 路径末尾的活跃变量的索引。

    # coefs : ndarray of shape (n_features, n_alphas + 1)
    # 路径上的系数。

    # n_iter : int
    # 运行的迭代次数。仅在 `return_n_iter` 设置为 True 时返回。

    # See Also
    # --------
    # lars_path_gram : 计算 LARS 路径。
    # lasso_path : 使用坐标下降计算 Lasso 路径。
    # LassoLars : 使用最小角度回归（Lars）拟合的 Lasso 模型。
    # Lars : 使用最小角度回归（LAR）拟合的最小角度回归模型。
    # LassoLarsCV : 使用 LARS 算法进行交叉验证的 Lasso。
    # LarsCV : 交叉验证的最小角度回归模型。
    # sklearn.decomposition.sparse_encode : 稀疏编码。

    # References
    # ----------
    # .. [1] "Least Angle Regression", Efron et al.
    #        http://statweb.stanford.edu/~tibs/ftp/lars.pdf

    # .. [2] `Wikipedia entry on the Least-angle regression
    #        <https://en.wikipedia.org/wiki/Least-angle_regression>`

    # .. [3] `Wikipedia entry on the Lasso
    #        <https://en.wikipedia.org/wiki/Lasso_(statistics)>`

    # Examples
    # --------
    # >>> from sklearn.linear_model import lars_path_gram
    # >>> from sklearn.datasets import make_regression
    # >>> X, y, true_coef = make_regression(
    # ...    n_samples=100, n_features=5, n_informative=2, coef=True, random_state=0
    # ... )
    # >>> true_coef
    # array([ 0.        ,  0.        ,  0.        , 97.9..., 45.7...])
    # >>> alphas, _, estimated_coef = lars_path_gram(X.T @ y, X.T @ X, n_samples=100)
    # >>> alphas.shape
    # (3,)
    # >>> estimated_coef
    # array([[ 0.     ,  0.     ,  0.     ],
    #        [ 0.     ,  0.     ,  0.     ],
    #        [ 0.     ,  0.     ,  0.     ],
    #        [ 0.     , 46.96..., 97.99...],
    #        [ 0.     ,  0.     , 45.70...]])
    # 调用名为 _lars_path_solver 的函数，并传入多个命名参数作为参数
    return _lars_path_solver(
        X=None,  # X 参数设置为 None
        y=None,  # y 参数设置为 None
        Xy=Xy,  # Xy 参数设置为传入函数的参数 Xy
        Gram=Gram,  # Gram 参数设置为传入函数的参数 Gram
        n_samples=n_samples,  # n_samples 参数设置为传入函数的参数 n_samples
        max_iter=max_iter,  # max_iter 参数设置为传入函数的参数 max_iter
        alpha_min=alpha_min,  # alpha_min 参数设置为传入函数的参数 alpha_min
        method=method,  # method 参数设置为传入函数的参数 method
        copy_X=copy_X,  # copy_X 参数设置为传入函数的参数 copy_X
        eps=eps,  # eps 参数设置为传入函数的参数 eps
        copy_Gram=copy_Gram,  # copy_Gram 参数设置为传入函数的参数 copy_Gram
        verbose=verbose,  # verbose 参数设置为传入函数的参数 verbose
        return_path=return_path,  # return_path 参数设置为传入函数的参数 return_path
        return_n_iter=return_n_iter,  # return_n_iter 参数设置为传入函数的参数 return_n_iter
        positive=positive,  # positive 参数设置为传入函数的参数 positive
    )
# 定义一个函数来求解最小角回归（Least Angle Regression, LAR）或 Lasso 路径，使用 LARS 算法 [1]
def _lars_path_solver(
    X,
    y,
    Xy=None,
    Gram=None,
    n_samples=None,
    max_iter=500,
    alpha_min=0,
    method="lar",
    copy_X=True,
    eps=np.finfo(float).eps,
    copy_Gram=True,
    verbose=0,
    return_path=True,
    return_n_iter=False,
    positive=False,
):
    """Compute Least Angle Regression or Lasso path using LARS algorithm [1]

    The optimization objective for the case method='lasso' is::

    (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

    in the case of method='lar', the objective function is only known in
    the form of an implicit equation (see discussion in [1])

    Read more in the :ref:`User Guide <least_angle_regression>`.

    Parameters
    ----------
    X : None or ndarray of shape (n_samples, n_features)
        Input data. Note that if X is None then Gram must be specified,
        i.e., cannot be None or False.

    y : None or ndarray of shape (n_samples,)
        Input targets.

    Xy : array-like of shape (n_features,), default=None
        `Xy = np.dot(X.T, y)` that can be precomputed. It is useful
        only when the Gram matrix is precomputed.

    Gram : None, 'auto' or array-like of shape (n_features, n_features), \
            default=None
        Precomputed Gram matrix `(X' * X)`, if ``'auto'``, the Gram
        matrix is precomputed from the given X, if there are more samples
        than features.

    n_samples : int or float, default=None
        Equivalent size of sample. If `None`, it will be `n_samples`.

    max_iter : int, default=500
        Maximum number of iterations to perform, set to infinity for no limit.

    alpha_min : float, default=0
        Minimum correlation along the path. It corresponds to the
        regularization parameter alpha parameter in the Lasso.

    method : {'lar', 'lasso'}, default='lar'
        Specifies the returned model. Select ``'lar'`` for Least Angle
        Regression, ``'lasso'`` for the Lasso.

    copy_X : bool, default=True
        If ``False``, ``X`` is overwritten.

    eps : float, default=np.finfo(float).eps
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems. Unlike the ``tol`` parameter in some iterative
        optimization-based algorithms, this parameter does not control
        the tolerance of the optimization.

    copy_Gram : bool, default=True
        If ``False``, ``Gram`` is overwritten.

    verbose : int, default=0
        Controls output verbosity.

    return_path : bool, default=True
        If ``return_path==True`` returns the entire path, else returns only the
        last point of the path.

    return_n_iter : bool, default=False
        Whether to return the number of iterations.
    """
    positive : bool, default=False
        限制系数为 >= 0。
        此选项仅适用于方法为 'lasso' 时。请注意，对于较小的 alpha 值，模型系数将不会收敛到普通最小二乘解。通常只有通过逐步 Lars-Lasso 算法达到的最小 alpha 值（当 fit_path=True 时为 ``alphas_[alphas_ > 0.].min()``）的系数才与坐标下降 lasso_path 函数的解一致。

    Returns
    -------
    alphas : array-like of shape (n_alphas + 1,)
        每次迭代中协方差的最大值（绝对值）。
        ``n_alphas`` 可以是 ``max_iter``、``n_features`` 或具有 ``alpha >= alpha_min`` 路径中的节点数中的最小者。

    active : array-like of shape (n_alphas,)
        路径结束时活跃变量的索引。

    coefs : array-like of shape (n_features, n_alphas + 1)
        路径上的系数。

    n_iter : int
        运行的迭代次数。仅在 return_n_iter 设置为 True 时返回。

    See Also
    --------
    lasso_path
    LassoLars
    Lars
    LassoLarsCV
    LarsCV
    sklearn.decomposition.sparse_encode

    References
    ----------
    .. [1] "Least Angle Regression", Efron et al.
           http://statweb.stanford.edu/~tibs/ftp/lars.pdf

    .. [2] `Wikipedia entry on the Least-angle regression
           <https://en.wikipedia.org/wiki/Least-angle_regression>`_

    .. [3] `Wikipedia entry on the Lasso
           <https://en.wikipedia.org/wiki/Lasso_(statistics)>`_
    if len(dtypes) == 1:
        # 如果只有一个数据类型，则使用输入数据的精度水平
        return_dtype = next(iter(dtypes))
    else:
        # 否则，回退到双精度浮点数
        return_dtype = np.float64

    if return_path:
        # 如果需要返回路径
        coefs = np.zeros((max_features + 1, n_features), dtype=return_dtype)
        alphas = np.zeros(max_features + 1, dtype=return_dtype)
    else:
        # 否则，初始化系数和先前系数为全零数组
        coef, prev_coef = (
            np.zeros(n_features, dtype=return_dtype),
            np.zeros(n_features, dtype=return_dtype),
        )
        alpha, prev_alpha = (
            np.array([0.0], dtype=return_dtype),
            np.array([0.0], dtype=return_dtype),
        )
        # 上述更好的想法？

    n_iter, n_active = 0, 0
    active, indices = list(), np.arange(n_features)
    # 存储协方差的符号
    sign_active = np.empty(max_features, dtype=np.int8)
    drop = False

    # 将保存Cholesky分解的因子。只引用下三角部分。
    if Gram is None:
        L = np.empty((max_features, max_features), dtype=X.dtype)
        swap, nrm2 = linalg.get_blas_funcs(("swap", "nrm2"), (X,))
    else:
        L = np.empty((max_features, max_features), dtype=Gram.dtype)
        swap, nrm2 = linalg.get_blas_funcs(("swap", "nrm2"), (Cov,))
    (solve_cholesky,) = get_lapack_funcs(("potrs",), (L,))

    if verbose:
        if verbose > 1:
            # 如果详细输出大于1，则打印详细步骤信息
            print("Step\t\tAdded\t\tDropped\t\tActive set size\t\tC")
        else:
            # 否则，输出一个点表示进展
            sys.stdout.write(".")
            sys.stdout.flush()

    tiny32 = np.finfo(np.float32).tiny  # 避免除以0的警告
    cov_precision = np.finfo(Cov.dtype).precision
    equality_tolerance = np.finfo(np.float32).eps

    if Gram is not None:
        Gram_copy = Gram.copy()
        Cov_copy = Cov.copy()

    if return_path:
        # 在早停的情况下调整 coefs 的大小
        alphas = alphas[: n_iter + 1]
        coefs = coefs[: n_iter + 1]

        if return_n_iter:
            # 如果需要返回迭代次数
            return alphas, active, coefs.T, n_iter
        else:
            # 否则，返回 alpha、active 和 coefs.T
            return alphas, active, coefs.T
    else:
        if return_n_iter:
            # 如果需要返回迭代次数
            return alpha, active, coef, n_iter
        else:
            # 否则，返回 alpha、active 和 coef
            return alpha, active, coef
###############################################################################
# Estimator classes

class Lars(MultiOutputMixin, RegressorMixin, LinearModel):
    """Least Angle Regression model a.k.a. LAR.

    Read more in the :ref:`User Guide <least_angle_regression>`.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    verbose : bool or int, default=False
        Sets the verbosity amount.

    precompute : bool, 'auto' or array-like , default='auto'
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'`` let us decide. The Gram
        matrix can also be passed as argument.

    n_nonzero_coefs : int, default=500
        Target number of non-zero coefficients. Use ``np.inf`` for no limit.

    eps : float, default=np.finfo(float).eps
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems. Unlike the ``tol`` parameter in some iterative
        optimization-based algorithms, this parameter does not control
        the tolerance of the optimization.

    copy_X : bool, default=True
        If ``True``, X will be copied; else, it may be overwritten.

    fit_path : bool, default=True
        If True the full path is stored in the ``coef_path_`` attribute.
        If you compute the solution for a large problem or many targets,
        setting ``fit_path`` to ``False`` will lead to a speedup, especially
        with a small alpha.

    jitter : float, default=None
        Upper bound on a uniform noise parameter to be added to the
        `y` values, to satisfy the model's assumption of
        one-at-a-time computations. Might help with stability.

        .. versionadded:: 0.23

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for jittering. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`. Ignored if `jitter` is None.

        .. versionadded:: 0.23

    Attributes
    ----------
    alphas_ : array-like of shape (n_alphas + 1,) or list of such arrays
        Maximum of covariances (in absolute value) at each iteration.
        ``n_alphas`` is either ``max_iter``, ``n_features`` or the
        number of nodes in the path with ``alpha >= alpha_min``, whichever
        is smaller. If this is a list of array-like, the length of the outer
        list is `n_targets`.

    active_ : list of shape (n_alphas,) or list of such lists
        Indices of active variables at the end of the path.
        If this is a list of list, the length of the outer list is `n_targets`.
    """
    # Least Angle Regression 模型，继承自 MultiOutputMixin, RegressorMixin 和 LinearModel
    # 这是一个回归模型，用于拟合数据，实现了最小角回归算法
    # 参数说明如下所示，包括是否拟合截距、是否使用预计算的 Gram 矩阵等
    # 其中 jitter 和 random_state 是 0.23 版本新增的参数，用于提高稳定性和重现性
    pass
    # 存储各个 alpha 路径上系数的变化值的数组列表，形状为 (n_features, n_alphas + 1) 或这种数组的列表
    # 如果 fit_path 参数为 False，则不存在该属性
    coef_path_ : array-like of shape (n_features, n_alphas + 1) or list \
            of such arrays

    # 模型的参数向量，对应于公式中的 w
    coef_ : array-like of shape (n_features,) or (n_targets, n_features)

    # 决策函数中的独立项
    intercept_ : float or array-like of shape (n_targets,)

    # lars_path 寻找每个目标的 alpha 网格所花费的迭代次数
    n_iter_ : array-like or int

    # 在拟合期间观察到的特征数目
    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    # 在拟合期间观察到的特征名称数组，仅当输入数据 `X` 中的特征名称全为字符串时定义
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    # 相关链接
    # 计算最小角回归或 Lasso 路径的 LARS 算法
    # Compute Least Angle Regression or Lasso path using LARS algorithm.
    # 交叉验证的最小角回归模型
    # Cross-validated Least Angle Regression model.
    # sklearn.decomposition.sparse_encode: 稀疏编码
    # sklearn.decomposition.sparse_encode: Sparse coding.
    See Also
    --------
    lars_path: Compute Least Angle Regression or Lasso
        path using LARS algorithm.
    LarsCV : Cross-validated Least Angle Regression model.
    sklearn.decomposition.sparse_encode : Sparse coding.

    Examples
    --------
    >>> from sklearn import linear_model
    >>> reg = linear_model.Lars(n_nonzero_coefs=1)
    >>> reg.fit([[-1, 1], [0, 0], [1, 1]], [-1.1111, 0, -1.1111])
    Lars(n_nonzero_coefs=1)
    >>> print(reg.coef_)
    [ 0. -1.11...]

    # 参数约束字典，指定模型参数的类型和取值范围
    _parameter_constraints: dict = {
        "fit_intercept": ["boolean"],
        "verbose": ["verbose"],
        "precompute": ["boolean", StrOptions({"auto"}), np.ndarray, Hidden(None)],
        "n_nonzero_coefs": [Interval(Integral, 1, None, closed="left")],
        "eps": [Interval(Real, 0, None, closed="left")],
        "copy_X": ["boolean"],
        "fit_path": ["boolean"],
        "jitter": [Interval(Real, 0, None, closed="left"), None],
        "random_state": ["random_state"],
    }

    # 指定使用的算法名称
    method = "lar"

    # 是否强制系数为非负
    positive = False

    # 初始化方法，设置模型参数
    def __init__(
        self,
        *,
        fit_intercept=True,
        verbose=False,
        precompute="auto",
        n_nonzero_coefs=500,
        eps=np.finfo(float).eps,
        copy_X=True,
        fit_path=True,
        jitter=None,
        random_state=None,
    ):
        # 初始化对象时传入的参数
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.precompute = precompute
        self.n_nonzero_coefs = n_nonzero_coefs
        self.eps = eps
        self.copy_X = copy_X
        self.fit_path = fit_path
        self.jitter = jitter
        self.random_state = random_state

    @staticmethod
    # 获取 Gram 矩阵的静态方法，根据 precompute 参数不同返回不同的值
    def _get_gram(precompute, X, y):
        if (not hasattr(precompute, "__array__")) and (
            (precompute is True)
            or (precompute == "auto" and X.shape[0] > X.shape[1])
            or (precompute == "auto" and y.shape[1] > 1)
        ):
            precompute = np.dot(X.T, X)

        return precompute
    def _fit(self, X, y, max_iter, alpha, fit_path, Xy=None):
        """Auxiliary method to fit the model using X, y as training data"""
        # 获取特征数
        n_features = X.shape[1]

        # 数据预处理，包括拟合截距和复制数据
        X, y, X_offset, y_offset, X_scale = _preprocess_data(
            X, y, fit_intercept=self.fit_intercept, copy=self.copy_X
        )

        # 如果 y 是一维的，则转换为二维的
        if y.ndim == 1:
            y = y[:, np.newaxis]

        # 获取目标数
        n_targets = y.shape[1]

        # 获取 Gram 矩阵
        Gram = self._get_gram(self.precompute, X, y)

        # 初始化结果存储属性
        self.alphas_ = []
        self.n_iter_ = []
        self.coef_ = np.empty((n_targets, n_features), dtype=X.dtype)

        # 如果 fit_path 为 True，执行路径型 LARS
        if fit_path:
            self.active_ = []
            self.coef_path_ = []
            # 对每个目标变量执行 LARS 路径算法
            for k in range(n_targets):
                this_Xy = None if Xy is None else Xy[:, k]
                # 执行 LARS 路径算法，获取结果
                alphas, active, coef_path, n_iter_ = lars_path(
                    X,
                    y[:, k],
                    Gram=Gram,
                    Xy=this_Xy,
                    copy_X=self.copy_X,
                    copy_Gram=True,
                    alpha_min=alpha,
                    method=self.method,
                    verbose=max(0, self.verbose - 1),
                    max_iter=max_iter,
                    eps=self.eps,
                    return_path=True,
                    return_n_iter=True,
                    positive=self.positive,
                )
                # 将结果存储到对应属性中
                self.alphas_.append(alphas)
                self.active_.append(active)
                self.n_iter_.append(n_iter_)
                self.coef_path_.append(coef_path)
                self.coef_[k] = coef_path[:, -1]

            # 如果只有一个目标变量，将结果展平
            if n_targets == 1:
                self.alphas_, self.active_, self.coef_path_, self.coef_ = [
                    a[0]
                    for a in (self.alphas_, self.active_, self.coef_path_, self.coef_)
                ]
                self.n_iter_ = self.n_iter_[0]
        else:
            # 如果 fit_path 为 False，执行标准 LARS
            for k in range(n_targets):
                this_Xy = None if Xy is None else Xy[:, k]
                # 执行标准 LARS，获取结果
                alphas, _, self.coef_[k], n_iter_ = lars_path(
                    X,
                    y[:, k],
                    Gram=Gram,
                    Xy=this_Xy,
                    copy_X=self.copy_X,
                    copy_Gram=True,
                    alpha_min=alpha,
                    method=self.method,
                    verbose=max(0, self.verbose - 1),
                    max_iter=max_iter,
                    eps=self.eps,
                    return_path=False,
                    return_n_iter=True,
                    positive=self.positive,
                )
                # 将结果存储到对应属性中
                self.alphas_.append(alphas)
                self.n_iter_.append(n_iter_)
            # 如果只有一个目标变量，将结果展平
            if n_targets == 1:
                self.alphas_ = self.alphas_[0]
                self.n_iter_ = self.n_iter_[0]

        # 设置截距
        self._set_intercept(X_offset, y_offset, X_scale)
        # 返回自身对象
        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, Xy=None):
        """Fit the model using X, y as training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Xy : array-like of shape (n_features,) or (n_features, n_targets), \
                default=None
            Xy = np.dot(X.T, y) that can be precomputed. It is useful
            only when the Gram matrix is precomputed.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        # 验证并处理输入数据，确保符合模型要求
        X, y = self._validate_data(
            X, y, force_writeable=True, y_numeric=True, multi_output=True
        )

        # 从模型属性中获取 alpha 值，默认为 0.0
        alpha = getattr(self, "alpha", 0.0)
        # 如果模型有定义 n_nonzero_coefs 属性，则优先使用其值来设置 alpha
        if hasattr(self, "n_nonzero_coefs"):
            alpha = 0.0  # n_nonzero_coefs 参数化优先级更高
            max_iter = self.n_nonzero_coefs
        else:
            max_iter = self.max_iter

        # 如果定义了 jitter 参数，则加入噪声以防止过拟合
        if self.jitter is not None:
            rng = check_random_state(self.random_state)

            # 生成与目标值 y 维度相同的均匀分布的噪声
            noise = rng.uniform(high=self.jitter, size=len(y))
            y = y + noise

        # 调用内部方法 _fit 进行模型拟合
        self._fit(
            X,
            y,
            max_iter=max_iter,
            alpha=alpha,
            fit_path=self.fit_path,
            Xy=Xy,
        )

        # 返回模型实例本身
        return self
class LassoLars(Lars):
    """Lasso model fit with Least Angle Regression a.k.a. Lars.

    It is a Linear Model trained with an L1 prior as regularizer.

    The optimization objective for Lasso is::

    (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

    Read more in the :ref:`User Guide <least_angle_regression>`.

    Parameters
    ----------
    alpha : float, default=1.0
        Constant that multiplies the penalty term. Defaults to 1.0.
        ``alpha = 0`` is equivalent to an ordinary least square, solved
        by :class:`LinearRegression`. For numerical reasons, using
        ``alpha = 0`` with the LassoLars object is not advised and you
        should prefer the LinearRegression object.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    verbose : bool or int, default=False
        Sets the verbosity amount.

    precompute : bool, 'auto' or array-like, default='auto'
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'`` let us decide. The Gram
        matrix can also be passed as argument.

    max_iter : int, default=500
        Maximum number of iterations to perform.

    eps : float, default=np.finfo(float).eps
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems. Unlike the ``tol`` parameter in some iterative
        optimization-based algorithms, this parameter does not control
        the tolerance of the optimization.

    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.

    fit_path : bool, default=True
        If ``True`` the full path is stored in the ``coef_path_`` attribute.
        If you compute the solution for a large problem or many targets,
        setting ``fit_path`` to ``False`` will lead to a speedup, especially
        with a small alpha.

    positive : bool, default=False
        Restrict coefficients to be >= 0. Be aware that you might want to
        remove fit_intercept which is set True by default.
        Under the positive restriction the model coefficients will not converge
        to the ordinary-least-squares solution for small values of alpha.
        Only coefficients up to the smallest alpha value (``alphas_[alphas_ >
        0.].min()`` when fit_path=True) reached by the stepwise Lars-Lasso
        algorithm are typically in congruence with the solution of the
        coordinate descent Lasso estimator.

    jitter : float, default=None
        Upper bound on a uniform noise parameter to be added to the
        `y` values, to satisfy the model's assumption of
        one-at-a-time computations. Might help with stability.

        .. versionadded:: 0.23
    """
    # LassoLars 类，继承自 Lars 类，实现了 Lasso 模型拟合，即最小角回归（Lars）方法。

    def __init__(self, alpha=1.0, fit_intercept=True, verbose=False,
                 precompute='auto', max_iter=500, eps=np.finfo(float).eps,
                 copy_X=True, fit_path=True, positive=False, jitter=None):
        """
        Parameters
        ----------
        alpha : float, default=1.0
            正则化项的乘数常数。默认为 1.0。
            当 alpha = 0 时，相当于普通最小二乘法，由 LinearRegression 解决。
            由于数值原因，不建议在 LassoLars 对象中使用 alpha = 0，应优先选择 LinearRegression 对象。

        fit_intercept : bool, default=True
            是否计算该模型的截距。如果设置为 False，则计算过程中不使用截距
            （即假设数据已经居中）。

        verbose : bool or int, default=False
            设置详细程度。

        precompute : bool, 'auto' or array-like, default='auto'
            是否使用预先计算的 Gram 矩阵加快计算速度。
            如果设置为 'auto'，由程序自行决定。也可以将 Gram 矩阵作为参数传递进来。

        max_iter : int, default=500
            执行的最大迭代次数。

        eps : float, default=np.finfo(float).eps
            在计算 Cholesky 对角因子时的机器精度正则化。对于非常病态的系统，可以增加此值。
            与某些基于迭代优化的算法中的 tol 参数不同，此参数不控制优化的容差。

        copy_X : bool, default=True
            如果为 True，则复制 X；否则可能会被覆盖。

        fit_path : bool, default=True
            如果为 True，则完整路径存储在 coef_path_ 属性中。
            如果计算大问题或多目标的解，将 fit_path 设置为 False 将导致加速，特别是在小 alpha 的情况下。

        positive : bool, default=False
            限制系数为 >= 0。请注意，可能需要删除默认为 True 的 fit_intercept。
            在正限制条件下，模型系数不会收敛到小 alpha 值时的普通最小二乘解。
            仅当 fit_path=True 时，通过逐步 Lars-Lasso 算法达到的系数通常与坐标下降 Lasso 估计器的解一致。

        jitter : float, default=None
            添加到 y 值的均匀噪声参数的上限，以满足模型的一次计算假设。可能有助于稳定性。

            .. versionadded:: 0.23
        """
        # 初始化函数，设定 LassoLars 类的参数和默认值
        super().__init__(alpha=alpha, fit_intercept=fit_intercept,
                         verbose=verbose, precompute=precompute,
                         max_iter=max_iter, eps=eps, copy_X=copy_X,
                         fit_path=fit_path, positive=positive, jitter=jitter)
    random_state : int, RandomState instance or None, default=None
        # 参数 random_state 控制随机数生成以进行抖动。传入一个整数可确保多次函数调用产生可复现的输出。
        # 查看“术语表<random_state>”。如果 jitter 为 None，则此参数被忽略。
        .. versionadded:: 0.23

    Attributes
    ----------
    alphas_ : array-like of shape (n_alphas + 1,) or list of such arrays
        # 在每次迭代中，路径上每个 alpha 对应的最大协方差（绝对值）。n_alphas 可能是 max_iter、n_features 或 alpha >= alpha_min 时路径中节点的数量，取其中最小值。
        # 如果这是一个 array-like 的 list，则外部 list 的长度是 n_targets。
    
    active_ : list of length n_alphas or list of such lists
        # 路径结束时活跃变量的索引列表。如果这是一个 list of list，则外部 list 的长度是 n_targets。
    
    coef_path_ : array-like of shape (n_features, n_alphas + 1) or list \
            of such arrays
        # 系数沿路径变化的值。如果传入一个 list，则期望是 n_targets 个这样的 array。如果 fit_path 参数为 False，则此项不出现。
        # 如果这是一个 array-like 的 list，则外部 list 的长度是 n_targets。
    
    coef_ : array-like of shape (n_features,) or (n_targets, n_features)
        # 参数向量（在公式中的 w）。
    
    intercept_ : float or array-like of shape (n_targets,)
        # 决策函数中的独立项。
    
    n_iter_ : array-like or int
        # lars_path 寻找每个目标的 alpha 网格所花费的迭代次数。
    
    n_features_in_ : int
        # 在“fit”期间看到的特征数量。
        .. versionadded:: 0.24
    
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        # 在“fit”期间看到的特征名称。仅在 X 具有全部为字符串的特征名称时定义。
        .. versionadded:: 1.0
    
    See Also
    --------
    lars_path : 使用 LARS 算法计算最小角回归或 Lasso 路径。
    lasso_path : 使用坐标下降计算 Lasso 路径。
    Lasso : 使用 L1 先验作为正则化器训练的线性模型（即 Lasso）。
    LassoCV : 沿正则化路径进行迭代拟合的 Lasso 线性模型。
    LassoLarsCV: 使用 LARS 算法进行交叉验证的 Lasso。
    LassoLarsIC : 使用 BIC 或 AIC 进行模型选择的 Lasso 拟合。
    sklearn.decomposition.sparse_encode : 稀疏编码。

    Examples
    --------
    >>> from sklearn import linear_model
    >>> reg = linear_model.LassoLars(alpha=0.01)
    >>> reg.fit([[-1, 1], [0, 0], [1, 1]], [-1, 0, -1])
    LassoLars(alpha=0.01)
    >>> print(reg.coef_)
    [ 0.         -0.955...]
    # 定义参数约束字典，继承自Lars类的参数约束，并添加以下额外的约束条件：
    # - "alpha": 必须为大于等于0的实数，左闭区间
    # - "max_iter": 必须为大于等于0的整数，左闭区间
    # - "positive": 必须为布尔类型
    _parameter_constraints: dict = {
        **Lars._parameter_constraints,
        "alpha": [Interval(Real, 0, None, closed="left")],
        "max_iter": [Interval(Integral, 0, None, closed="left")],
        "positive": ["boolean"],
    }
    
    # 从参数约束字典中移除键为"n_nonzero_coefs"的约束条件
    _parameter_constraints.pop("n_nonzero_coefs")
    
    # 设置方法(method)为"lasso"
    method = "lasso"
    
    # 初始化函数，设定模型的初始参数
    def __init__(
        self,
        alpha=1.0,
        *,
        fit_intercept=True,
        verbose=False,
        precompute="auto",
        max_iter=500,
        eps=np.finfo(float).eps,
        copy_X=True,
        fit_path=True,
        positive=False,
        jitter=None,
        random_state=None,
    ):
        # 将参数赋值给对象的属性
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.verbose = verbose
        self.positive = positive
        self.precompute = precompute
        self.copy_X = copy_X
        self.eps = eps
        self.fit_path = fit_path
        self.jitter = jitter
        self.random_state = random_state
###############################################################################
# Cross-validated estimator classes


def _check_copy_and_writeable(array, copy=False):
    # 如果设置了复制标志或数组不可写，则复制数组
    if copy or not array.flags.writeable:
        return array.copy()
    # 否则返回原数组
    return array


def _lars_path_residues(
    X_train,
    y_train,
    X_test,
    y_test,
    Gram=None,
    copy=True,
    method="lar",
    verbose=False,
    fit_intercept=True,
    max_iter=500,
    eps=np.finfo(float).eps,
    positive=False,
):
    """Compute the residues on left-out data for a full LARS path

    Parameters
    -----------
    X_train : array-like of shape (n_samples, n_features)
        训练数据，用于拟合 LARS

    y_train : array-like of shape (n_samples,)
        目标变量，用于拟合 LARS

    X_test : array-like of shape (n_samples, n_features)
        用于计算残差的测试数据

    y_test : array-like of shape (n_samples,)
        用于计算残差的测试目标变量

    Gram : None, 'auto' or array-like of shape (n_features, n_features), \
            default=None
        预计算的 Gram 矩阵 (X' * X)，如果设置为 'auto'，则从给定的 X 预计算 Gram
        矩阵，如果样本数大于特征数

    copy : bool, default=True
        是否复制 X_train、X_test、y_train 和 y_test；
        如果为 False，它们可能会被覆盖。

    method : {'lar' , 'lasso'}, default='lar'
        指定返回的模型。选择 'lar' 表示最小角度回归，选择 'lasso' 表示 Lasso 回归。

    verbose : bool or int, default=False
        设置详细程度的参数

    fit_intercept : bool, default=True
        是否计算此模型的截距。如果设置为 False，则计算过程中不使用截距
        （即预期数据已经中心化）。

    positive : bool, default=False
        是否限制系数为非负。注意可能需要去除默认为 True 的 fit_intercept。
        参见 LassoLarsCV 和 LassoLarsIC 的文档，关于在 'lasso' 方法中使用此选项
        时预期 alpha 值较小的情况的注意事项。

    max_iter : int, default=500
        执行的最大迭代次数。

    eps : float, default=np.finfo(float).eps
        在计算 Cholesky 对角因子时的机器精度正则化。对于条件非常糟糕的系统，
        可以增加此值。与某些基于迭代优化算法中的 tol 参数不同，此参数不控制优化的
        容忍度。

    Returns
    --------
    alphas : array-like of shape (n_alphas,)
        每次迭代的最大协方差（绝对值）。n_alphas 取决于 max_iter 和 n_features 中较小的一个。

    active : list
        路径结束时活跃变量的索引列表。
    coefs : array-like of shape (n_features, n_alphas)
        Coefficients along the path

    residues : array-like of shape (n_alphas, n_samples)
        Residues of the prediction on the test data
    """
    # 确保 X_train 是可复制和可写的，并根据需要进行复制
    X_train = _check_copy_and_writeable(X_train, copy)
    # 确保 y_train 是可复制和可写的，并根据需要进行复制
    y_train = _check_copy_and_writeable(y_train, copy)
    # 确保 X_test 是可复制和可写的，并根据需要进行复制
    X_test = _check_copy_and_writeable(X_test, copy)
    # 确保 y_test 是可复制和可写的，并根据需要进行复制
    y_test = _check_copy_and_writeable(y_test, copy)

    # 如果需要拟合截距项
    if fit_intercept:
        # 计算 X_train 的均值
        X_mean = X_train.mean(axis=0)
        # 对 X_train 减去均值
        X_train -= X_mean
        # 对 X_test 减去相同的均值
        X_test -= X_mean
        # 计算 y_train 的均值
        y_mean = y_train.mean(axis=0)
        # 将 y_train 转换为浮点数组，并在原地减去均值
        y_train = as_float_array(y_train, copy=False)
        y_train -= y_mean
        # 将 y_test 转换为浮点数组，并在原地减去均值
        y_test = as_float_array(y_test, copy=False)
        y_test -= y_mean

    # 使用 LARS（Least Angle Regression）算法计算路径上的系数
    alphas, active, coefs = lars_path(
        X_train,
        y_train,
        Gram=Gram,  # 如果提供了 Gram 矩阵，则传入
        copy_X=False,  # 不复制 X 数据
        copy_Gram=False,  # 不复制 Gram 矩阵
        method=method,  # 使用的方法
        verbose=max(0, verbose - 1),  # 如果 verbose 大于 0，则调整为更低的值
        max_iter=max_iter,  # 最大迭代次数
        eps=eps,  # 控制算法终止的容差
        positive=positive,  # 是否强制系数为正
    )
    
    # 计算测试数据的残差
    residues = np.dot(X_test, coefs) - y_test[:, np.newaxis]
    # 返回结果：路径上的 alpha 值、活跃变量集、系数矩阵以及残差的转置
    return alphas, active, coefs, residues.T
class LarsCV(Lars):
    """Cross-validated Least Angle Regression model.

    See glossary entry for :term:`cross-validation estimator`.

    Read more in the :ref:`User Guide <least_angle_regression>`.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    verbose : bool or int, default=False
        Sets the verbosity amount.

    max_iter : int, default=500
        Maximum number of iterations to perform.

    precompute : bool, 'auto' or array-like , default='auto'
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'`` let us decide. The Gram matrix
        cannot be passed as argument since we will use only subsets of X.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, :class:`~sklearn.model_selection.KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.

    max_n_alphas : int, default=1000
        The maximum number of points on the path used to compute the
        residuals in the cross-validation.

    n_jobs : int or None, default=None
        Number of CPUs to use during the cross validation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    eps : float, default=np.finfo(float).eps
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems. Unlike the ``tol`` parameter in some iterative
        optimization-based algorithms, this parameter does not control
        the tolerance of the optimization.

    copy_X : bool, default=True
        If ``True``, X will be copied; else, it may be overwritten.

    Attributes
    ----------
    active_ : list of length n_alphas or list of such lists
        Indices of active variables at the end of the path.
        If this is a list of lists, the outer list length is `n_targets`.

    coef_ : array-like of shape (n_features,)
        parameter vector (w in the formulation formula)

    intercept_ : float
        independent term in decision function

    coef_path_ : array-like of shape (n_features, n_alphas)
        the varying values of the coefficients along the path
    """
    alpha_ : float
        # 存储估计的正则化参数 alpha

    alphas_ : array-like of shape (n_alphas,)
        # 存储沿着路径的不同 alpha 值

    cv_alphas_ : array-like of shape (n_cv_alphas,)
        # 存储不同交叉验证折叠中路径上的所有 alpha 值

    mse_path_ : array-like of shape (n_folds, n_cv_alphas)
        # 存储每个折叠中留出样本的均方误差，对应于路径上的 alpha 值（由 ``cv_alphas`` 给出）

    n_iter_ : array-like or int
        # 记录使用最佳 alpha 运行的 LARS 算法的迭代次数

    n_features_in_ : int
        # 在拟合过程中看到的特征数量

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        # 在拟合过程中看到的特征名称，仅当 `X` 具有全部为字符串的特征名称时定义

        .. versionadded:: 1.0

    See Also
    --------
    lars_path : 使用 LARS 算法计算最小角回归或 Lasso 路径。
    lasso_path : 使用坐标下降计算 Lasso 路径。
    Lasso : 使用 L1 正则化作为正则化器的线性模型（即 Lasso）。
    LassoCV : 在正则化路径上进行迭代拟合的 Lasso 线性模型。
    LassoLars : 使用最小角回归（Lars）拟合的 Lasso 模型。
    LassoLarsIC : 使用 BIC 或 AIC 进行模型选择的 Lasso 模型拟合。

    Notes
    -----
    在 `fit` 方法中，一旦通过交叉验证找到最佳参数 `alpha`，模型将再次使用整个训练集进行拟合。

    Examples
    --------
    >>> from sklearn.linear_model import LarsCV
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=200, noise=4.0, random_state=0)
    >>> reg = LarsCV(cv=5).fit(X, y)
    >>> reg.score(X, y)
    0.9996...
    >>> reg.alpha_
    0.2961...
    >>> reg.predict(X[:1,])
    array([154.3996...])
    """

    _parameter_constraints: dict = {
        **Lars._parameter_constraints,
        "max_iter": [Interval(Integral, 0, None, closed="left")],
        "cv": ["cv_object"],
        "max_n_alphas": [Interval(Integral, 1, None, closed="left")],
        "n_jobs": [Integral, None],
    }
    # 存储 `_parameter_constraints` 字典，包含与 LarsCV 类相关的参数约束

    for parameter in ["n_nonzero_coefs", "jitter", "fit_path", "random_state"]:
        _parameter_constraints.pop(parameter)
        # 从 `_parameter_constraints` 中移除特定的参数名称

    method = "lar"
    # 设置使用的方法为 "lar"

    def __init__(
        self,
        *,
        fit_intercept=True,
        verbose=False,
        max_iter=500,
        precompute="auto",
        cv=None,
        max_n_alphas=1000,
        n_jobs=None,
        eps=np.finfo(float).eps,
        copy_X=True,
        # 初始化 LarsCV 类的构造函数，设置默认参数和选项
    ):
        self.max_iter = max_iter
        self.cv = cv
        self.max_n_alphas = max_n_alphas
        self.n_jobs = n_jobs
        super().__init__(
            fit_intercept=fit_intercept,
            verbose=verbose,
            precompute=precompute,
            n_nonzero_coefs=500,
            eps=eps,
            copy_X=copy_X,
            fit_path=True,
        )


        # 调用父类初始化方法，设置线性模型的相关参数
        super().__init__(
            fit_intercept=fit_intercept,
            verbose=verbose,
            precompute=precompute,
            n_nonzero_coefs=500,
            eps=eps,
            copy_X=copy_X,
            fit_path=True,
        )



    def _more_tags(self):
        return {"multioutput": False}


    # 返回一个字典，指示此模型不支持多输出
    def _more_tags(self):
        return {"multioutput": False}



    @_fit_context(prefer_skip_nested_validation=True)
    def get_metadata_routing(self):
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        .. versionadded:: 1.4

        Returns
        -------
        routing : MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        router = MetadataRouter(owner=self.__class__.__name__).add(
            splitter=check_cv(self.cv),
            method_mapping=MethodMapping().add(caller="fit", callee="split"),
        )
        return router


    # 获取此对象的元数据路由
    def get_metadata_routing(self):
        """获取此对象的元数据路由。

        请查看 :ref:`用户指南 <metadata_routing>` 了解路由机制的工作原理。

        .. versionadded:: 1.4

        Returns
        -------
        routing : MetadataRouter
            一个 :class:`~sklearn.utils.metadata_routing.MetadataRouter` 对象，
            封装了路由信息。
        """
        # 创建一个 MetadataRouter 对象，设定其所有者为当前类名
        router = MetadataRouter(owner=self.__class__.__name__).add(
            # 使用给定的交叉验证策略作为拆分器
            splitter=check_cv(self.cv),
            # 设定方法映射，将 'fit' 方法映射到 'split' 方法
            method_mapping=MethodMapping().add(caller="fit", callee="split"),
        )
        return router
class LassoLarsCV(LarsCV):
    """Cross-validated Lasso, using the LARS algorithm.

    See glossary entry for :term:`cross-validation estimator`.

    The optimization objective for Lasso is::

    (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

    Read more in the :ref:`User Guide <least_angle_regression>`.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    verbose : bool or int, default=False
        Sets the verbosity amount.

    max_iter : int, default=500
        Maximum number of iterations to perform.

    precompute : bool or 'auto' , default='auto'
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'`` let us decide. The Gram matrix
        cannot be passed as argument since we will use only subsets of X.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, :class:`~sklearn.model_selection.KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.

    max_n_alphas : int, default=1000
        The maximum number of points on the path used to compute the
        residuals in the cross-validation.

    n_jobs : int or None, default=None
        Number of CPUs to use during the cross validation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    eps : float, default=np.finfo(float).eps
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems. Unlike the ``tol`` parameter in some iterative
        optimization-based algorithms, this parameter does not control
        the tolerance of the optimization.

    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.
    """
    positive : bool, default=False
        # 是否将系数限制为非负数。注意，可能需要移除默认为 True 的 fit_intercept 参数。
        # 在系数被限制为非负数时，对于较小的 alpha 值，模型系数不会收敛到普通最小二乘解。
        # 仅有在使用 stepwise Lars-Lasso 算法达到的最小 alpha 值（当 fit_path=True 时为 `alphas_[alphas_ > 0.].min()`）时，
        # 才能与坐标下降 Lasso 估计器的解一致。
        # 因此，仅在期望或已经获得稀疏解的问题中使用 LassoLarsCV 才有意义。

    Attributes
    ----------
    coef_ : array-like of shape (n_features,)
        # 参数向量（在公式中的 w）

    intercept_ : float
        # 决策函数中的独立项

    coef_path_ : array-like of shape (n_features, n_alphas)
        # 沿路径的系数变化值

    alpha_ : float
        # 估计的正则化参数 alpha

    alphas_ : array-like of shape (n_alphas,)
        # 沿路径的不同 alpha 值

    cv_alphas_ : array-like of shape (n_cv_alphas,)
        # 不同折叠中沿路径的所有 alpha 值

    mse_path_ : array-like of shape (n_folds, n_cv_alphas)
        # 沿路径的每个折叠中的左出均方误差（由 `cv_alphas` 给出的 alpha 值）

    n_iter_ : array-like or int
        # 使用最佳 alpha 值运行的 Lars 迭代次数

    active_ : list of int
        # 路径结束时活跃变量的索引列表

    n_features_in_ : int
        # 在拟合过程中看到的特征数量

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        # 在拟合过程中看到的特征名称。仅当 `X` 的特征名都是字符串时定义。

        .. versionadded:: 1.0

    See Also
    --------
    lars_path : 使用 LARS 算法计算最小角回归或 Lasso 路径。
    lasso_path : 使用坐标下降计算 Lasso 路径。
    Lasso : 使用 L1 先验作为正则化器（即 Lasso）的线性模型。
    LassoCV : 沿正则化路径进行迭代拟合的 Lasso 线性模型。
    LassoLars : 使用最小角回归（即 Lars）拟合的 Lasso 模型。
    LassoLarsIC : 使用 BIC 或 AIC 进行模型选择的 Lars 拟合的 Lasso 模型。
    sklearn.decomposition.sparse_encode : 稀疏编码。

    Notes
    -----
    # 该对象解决与 `sklearn.linear_model.LassoCV` 相同的问题。然而，不同于 `sklearn.linear_model.LassoCV`，
    # 它会自行找到相关的 alpha 值。一般来说，因为这个属性，它会更加稳定。但是，它对于多重共线性数据集更加脆弱。
    """
    _parameter_constraints 是一个字典，用于定义参数约束，继承自 LarsCV 的参数约束，并增加了 "positive" 参数的约束为布尔类型。
    
    method 是字符串 "lasso"，指定了当前模型的方法为 Lasso 回归。
    
    __init__ 是 LassoLarsCV 类的构造函数，用于初始化类的实例。参数包括：
    
    - fit_intercept: 是否拟合截距，默认为 True。
    - verbose: 是否输出详细信息，默认为 False。
    - max_iter: 最大迭代次数，默认为 500。
    - precompute: 是否预先计算，默认为 "auto"。
    - cv: 交叉验证的折数，默认为 None。
    - max_n_alphas: 最大的 alpha 参数数量，默认为 1000。
    - n_jobs: 并行运行的作业数量，默认为 None。
    - eps: 机器精度的浮点数精度，默认为 np.finfo(float).eps。
    - copy_X: 是否复制数据，默认为 True。
    - positive: 是否强制系数为正，默认为 False。
    
    注：代码中的 "# XXX : we don't use super().__init__" 是一个注释，表示在初始化过程中不调用父类的构造函数 super().__init__，以避免设置 n_nonzero_coefs 属性。
    """
class LassoLarsIC(LassoLars):
    """Lasso model fit with Lars using BIC or AIC for model selection.

    The optimization objective for Lasso is::

    (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

    AIC is the Akaike information criterion [2]_ and BIC is the Bayes
    Information criterion [3]_. Such criteria are useful to select the value
    of the regularization parameter by making a trade-off between the
    goodness of fit and the complexity of the model. A good model should
    explain well the data while being simple.

    Read more in the :ref:`User Guide <lasso_lars_ic>`.

    Parameters
    ----------
    criterion : {'aic', 'bic'}, default='aic'
        The type of criterion to use. 'aic' for Akaike Information Criterion,
        'bic' for Bayesian Information Criterion.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    verbose : bool or int, default=False
        Sets the verbosity amount. If True or non-zero integer, prints messages
        about the progress and results of the fitting process.

    precompute : bool, 'auto' or array-like, default='auto'
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'``, the algorithm decides based on
        conditions whether to use the Gram matrix. Alternatively, a specific
        Gram matrix can be provided.

    max_iter : int, default=500
        Maximum number of iterations to perform. Can be used for
        early stopping.

    eps : float, default=np.finfo(float).eps
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems. Unlike the ``tol`` parameter in some iterative
        optimization-based algorithms, this parameter does not control
        the tolerance of the optimization.

    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.

    positive : bool, default=False
        Restrict coefficients to be >= 0. When set to True, coefficients are
        constrained to be non-negative. It is recommended to also set
        `fit_intercept` to False when using positive=True.

        For very small values of alpha, coefficients may not converge to
        the ordinary-least-squares solution due to the nature of the LassoLars
        algorithm.

    noise_variance : float, default=None
        The estimated noise variance of the data. If `None`, an unbiased
        estimate is computed by an Ordinary Least Squares (OLS) model. This
        estimation is feasible only when `n_samples > n_features + fit_intercept`.

        .. versionadded:: 1.1

    Attributes
    ----------
    coef_ : array-like of shape (n_features,)
        Parameter vector (w in the formulation formula). This attribute stores
        the estimated coefficients of the model after fitting.
    """
    # 模型的截距项，在决策函数中独立存在。
    intercept_ : float

    # 信息准则选择的 alpha 参数。
    alpha_ : float

    # 形状为 (n_alphas + 1,) 的数组，或者这样的数组列表。
    # 每次迭代的最大协方差（绝对值），``n_alphas`` 是 ``max_iter``、``n_features`` 或具有 ``alpha >= alpha_min`` 的路径上节点数中较小的那个。
    # 如果是列表，则长度为 `n_targets`。
    alphas_ : array-like of shape (n_alphas + 1,) or list of such arrays

    # lars_path 运行的迭代次数，用于找到 alphas 的网格。
    n_iter_ : int

    # 形状为 (n_alphas,) 的数组，信息准则（'aic'、'bic'）的值跨越所有 alphas。
    # 选择具有最小信息准则的 alpha，如 [1]_ 所述。
    criterion_ : array-like of shape (n_alphas,)

    # 从用于计算准则的数据中估计的噪声方差。
    # .. versionadded:: 1.1
    noise_variance_ : float

    # 在 :term:`fit` 过程中看到的特征数。
    # .. versionadded:: 0.24
    n_features_in_ : int

    # 在 :term:`fit` 过程中看到的特征的名称数组。
    # 仅当 `X` 具有所有字符串特征名时定义。
    # .. versionadded:: 1.0
    feature_names_in_ : ndarray of shape (`n_features_in_`,)

    # 查看更多详细信息，请参考 :ref:`User Guide <lasso_lars_ic>` 中关于 AIC 和 BIC 准则的数学公式。
    # 计算自由度的数量如 [1]_ 所述。
    #
    # 参见
    # --------
    # lars_path : 使用 LARS 算法计算最小角度回归或 Lasso 路径。
    # lasso_path : 使用坐标下降计算 Lasso 路径。
    # Lasso : 使用 L1 先验作为正则化器训练的线性模型（即 Lasso）。
    # LassoCV : 在正则化路径上进行迭代拟合的 Lasso 线性模型。
    # LassoLars : 使用最小角度回归（即 Lars）拟合的 Lasso 模型。
    # LassoLarsCV: 使用 LARS 算法进行交叉验证的 Lasso。
    # sklearn.decomposition.sparse_encode : 稀疏编码。
    #
    # 参考
    # ----------
    # .. [1] :arxiv:`Zou, Hui, Trevor Hastie, and Robert Tibshirani.
    #         "On the degrees of freedom of the lasso."
    #         The Annals of Statistics 35.5 (2007): 2173-2192.
    #         <0712.0881>`
    #
    # .. [2] `Akaike 信息准则的维基百科条目
    #         <https://en.wikipedia.org/wiki/Akaike_information_criterion>`_
    #
    # .. [3] `贝叶斯信息准则的维基百科条目
    #         <https://en.wikipedia.org/wiki/Bayesian_information_criterion>`_

    # 示例
    # --------
    # >>> from sklearn import linear_model
    # >>> reg = linear_model.LassoLarsIC(criterion='bic')
    # >>> X = [[-2, 2], [-1, 1], [0, 0], [1, 1], [2, 2]]
    # >>> y = [-2.2222, -1.1111, 0, -1.1111, -2.2222]
    # >>> reg.fit(X, y)
    # LassoLarsIC(criterion='bic')
    # >>> print(reg.coef_)
    # [ 0.  -1.11...]
    # 定义参数约束字典，继承自 LassoLars 类的参数约束，并添加了额外的约束条件
    _parameter_constraints: dict = {
        **LassoLars._parameter_constraints,
        "criterion": [StrOptions({"aic", "bic"})],  # criterion 参数只能取 "aic" 或 "bic"
        "noise_variance": [Interval(Real, 0, None, closed="left"), None],  # noise_variance 参数为非负实数或 None
    }
    
    # 从参数约束字典中移除指定的参数
    for parameter in ["jitter", "fit_path", "alpha", "random_state"]:
        _parameter_constraints.pop(parameter)
    
    # 初始化函数，设置各参数的默认值
    def __init__(
        self,
        criterion="aic",  # 选择使用 AIC 还是 BIC 作为准则，默认为 "aic"
        *,
        fit_intercept=True,  # 是否拟合截距，默认为 True
        verbose=False,  # 是否显示详细输出信息，默认为 False
        precompute="auto",  # 是否预计算，默认为 "auto"
        max_iter=500,  # 最大迭代次数，默认为 500
        eps=np.finfo(float).eps,  # 机器精度，默认为 float 类型的机器精度
        copy_X=True,  # 是否复制输入数据，默认为 True
        positive=False,  # 是否施加正则化约束，默认为 False
        noise_variance=None,  # 噪声方差的初始设定，默认为 None
    ):
        # 将参数值赋给对象的属性
        self.criterion = criterion
        self.fit_intercept = fit_intercept
        self.positive = positive
        self.max_iter = max_iter
        self.verbose = verbose
        self.copy_X = copy_X
        self.precompute = precompute
        self.eps = eps
        self.fit_path = True  # 设置 fit_path 属性为 True
        self.noise_variance = noise_variance  # 设置 noise_variance 属性为传入的值或 None
    
    # 返回额外的标签信息，指定 multioutput 属性为 False
    def _more_tags(self):
        return {"multioutput": False}
    
    # 应用 _fit_context 装饰器，设置 prefer_skip_nested_validation 为 True
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, copy_X=None):
        """Fit the model using X, y as training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values. Will be cast to X's dtype if necessary.

        copy_X : bool, default=None
            If provided, this parameter will override the choice
            of copy_X made at instance creation.
            If ``True``, X will be copied; else, it may be overwritten.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        if copy_X is None:
            # 若未提供 copy_X 参数，则使用实例创建时的默认值
            copy_X = self.copy_X
        
        # 验证输入数据 X 和 y，确保可以进行写操作和 y 是数值型数据
        X, y = self._validate_data(X, y, force_writeable=True, y_numeric=True)

        # 预处理数据，包括中心化和标准化
        X, y, Xmean, ymean, Xstd = _preprocess_data(
            X, y, fit_intercept=self.fit_intercept, copy=copy_X
        )

        # 获取预先计算的 Gram 矩阵
        Gram = self.precompute

        # 使用 LARS 算法进行路径计算，获得系数路径和迭代次数
        alphas_, _, coef_path_, self.n_iter_ = lars_path(
            X,
            y,
            Gram=Gram,
            copy_X=copy_X,
            copy_Gram=True,
            alpha_min=0.0,
            method="lasso",
            verbose=self.verbose,
            max_iter=self.max_iter,
            eps=self.eps,
            return_n_iter=True,
            positive=self.positive,
        )

        # 样本数
        n_samples = X.shape[0]

        # 根据选择的准则设置相应的因子
        if self.criterion == "aic":
            criterion_factor = 2
        elif self.criterion == "bic":
            criterion_factor = np.log(n_samples)
        else:
            raise ValueError(
                f"criterion should be either bic or aic, got {self.criterion!r}"
            )

        # 计算残差
        residuals = y[:, np.newaxis] - np.dot(X, coef_path_)
        residuals_sum_squares = np.sum(residuals**2, axis=0)

        # 计算自由度
        degrees_of_freedom = np.zeros(coef_path_.shape[1], dtype=int)
        for k, coef in enumerate(coef_path_.T):
            mask = np.abs(coef) > np.finfo(coef.dtype).eps
            if not np.any(mask):
                continue
            # 计算自由度，即非零系数的数量
            degrees_of_freedom[k] = np.sum(mask)

        # 保存计算得到的 alphas_
        self.alphas_ = alphas_

        # 如果未提供噪声方差，则估计它
        if self.noise_variance is None:
            self.noise_variance_ = self._estimate_noise_variance(
                X, y, positive=self.positive
            )
        else:
            self.noise_variance_ = self.noise_variance

        # 计算准则值
        self.criterion_ = (
            n_samples * np.log(2 * np.pi * self.noise_variance_)
            + residuals_sum_squares / self.noise_variance_
            + criterion_factor * degrees_of_freedom
        )

        # 找到最小准则值对应的索引
        n_best = np.argmin(self.criterion_)

        # 保存最佳的 alpha 和对应的系数路径
        self.alpha_ = alphas_[n_best]
        self.coef_ = coef_path_[:, n_best]

        # 设置截距
        self._set_intercept(Xmean, ymean, Xstd)

        # 返回自身实例
        return self
    def _estimate_noise_variance(self, X, y, positive):
        """Compute an estimate of the variance with an OLS model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to be fitted by the OLS model. We expect the data to be
            centered.

        y : ndarray of shape (n_samples,)
            Associated target.

        positive : bool, default=False
            Restrict coefficients to be >= 0. This should be inline with
            the `positive` parameter from `LassoLarsIC`.

        Returns
        -------
        noise_variance : float
            An estimator of the noise variance of an OLS model.
        """
        # Check if the number of samples is sufficient for fitting
        if X.shape[0] <= X.shape[1] + self.fit_intercept:
            raise ValueError(
                f"You are using {self.__class__.__name__} in the case where the number "
                "of samples is smaller than the number of features. In this setting, "
                "getting a good estimate for the variance of the noise is not "
                "possible. Provide an estimate of the noise variance in the "
                "constructor."
            )
        # Create an OLS model without intercept and possibly positive coefficients constraint
        ols_model = LinearRegression(positive=positive, fit_intercept=False)
        # Fit the OLS model on the data and predict y values
        y_pred = ols_model.fit(X, y).predict(X)
        # Calculate the estimated noise variance using the residual sum of squares
        return np.sum((y - y_pred) ** 2) / (
            X.shape[0] - X.shape[1] - self.fit_intercept
        )
```