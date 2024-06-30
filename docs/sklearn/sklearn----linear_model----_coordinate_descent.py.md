# `D:\src\scipysrc\scikit-learn\sklearn\linear_model\_coordinate_descent.py`

```
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 导入必要的库和模块
import numbers  # 用于数字类型的判断
import sys  # 提供对 Python 解释器的访问
import warnings  # 用于警告处理
from abc import ABC, abstractmethod  # 提供抽象基类（ABC）和抽象方法（abstractmethod）的支持
from functools import partial  # 函数工具，用于创建偏函数
from numbers import Integral, Real  # 导入整数和实数的类型判断

import numpy as np  # 数值计算库
from joblib import effective_n_jobs  # 用于获取有效的并行作业数
from scipy import sparse  # 稀疏矩阵库

# 导入 scikit-learn 内部的类和函数
from ..base import MultiOutputMixin, RegressorMixin, _fit_context  # 基础模块中的混合类和函数
from ..model_selection import check_cv  # 模型选择模块中的交叉验证函数
from ..utils import Bunch, check_array, check_scalar  # 工具模块中的数据容器、数组和标量检查函数
from ..utils._metadata_requests import (  # 工具模块中的元数据请求相关函数和类
    MetadataRouter,
    MethodMapping,
    _raise_for_params,
    get_routing_for_object,
)
from ..utils._param_validation import Interval, StrOptions, validate_params  # 工具模块中的参数验证相关函数和类
from ..utils.extmath import safe_sparse_dot  # 工具模块中的安全稀疏点积函数
from ..utils.metadata_routing import (  # 工具模块中的元数据路由相关函数
    _routing_enabled,
    process_routing,
)
from ..utils.parallel import Parallel, delayed  # 工具模块中的并行处理函数和延迟执行函数
from ..utils.validation import (  # 工具模块中的验证相关函数
    _check_sample_weight,
    check_consistent_length,
    check_is_fitted,
    check_random_state,
    column_or_1d,
    has_fit_parameter,
)

# 导入自定义的线性模型类和函数
# mypy error: Module 'sklearn.linear_model' has no attribute '_cd_fast'
from . import _cd_fast as cd_fast  # type: ignore
from ._base import LinearModel, _pre_fit, _preprocess_data  # 导入基础模型类和预处理数据函数


def _set_order(X, y, order="C"):
    """Change the order of X and y if necessary.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data.

    y : ndarray of shape (n_samples,)
        Target values.

    order : {None, 'C', 'F'}
        If 'C', dense arrays are returned as C-ordered, sparse matrices in csr
        format. If 'F', dense arrays are return as F-ordered, sparse matrices
        in csc format.

    Returns
    -------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data with guaranteed order.

    y : ndarray of shape (n_samples,)
        Target values with guaranteed order.
    """
    # 检查 order 参数是否有效，若无效则引发 ValueError 异常
    if order not in [None, "C", "F"]:
        raise ValueError(
            "Unknown value for order. Got {} instead of None, 'C' or 'F'.".format(order)
        )
    
    # 检查输入的 X 和 y 是否为稀疏矩阵
    sparse_X = sparse.issparse(X)
    sparse_y = sparse.issparse(y)
    
    # 根据 order 参数重新排列 X 和 y 的顺序
    if order is not None:
        sparse_format = "csc" if order == "F" else "csr"
        if sparse_X:
            X = X.asformat(sparse_format, copy=False)
        else:
            X = np.asarray(X, order=order)
        if sparse_y:
            y = y.asformat(sparse_format)
        else:
            y = np.asarray(y, order=order)
    
    # 返回重新排列后的 X 和 y
    return X, y


###############################################################################
# Paths functions


def _alpha_grid(
    X,
    y,
    Xy=None,
    l1_ratio=1.0,
    fit_intercept=True,
    eps=1e-3,
    n_alphas=100,
    copy_X=True,
):
    """Compute the grid of alpha values for elastic net parameter search

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Input data.

    y : ndarray of shape (n_samples,)
        Target values.

    Xy : None or ndarray of shape (n_samples,)
        Xy = np.dot(X.T, y) if precomputed.

    l1_ratio : float, default=1.0
        The ElasticNet mixing parameter, with 0 <= l1_ratio <= 1.
        For l1_ratio = 0, the penalty is an L2 penalty. For l1_ratio = 1,
        it is an L1 penalty. For 0 < l1_ratio < 1, the penalty is a combination
        of L1 and L2.

    fit_intercept : bool, default=True
        Whether to fit an intercept term.

    eps : float, default=1e-3
        Length of the path. eps=1e-3 means that alpha_min / alpha_max = 1e-3.

    n_alphas : int, default=100
        Number of alphas along the regularization path.

    copy_X : bool, default=True
        Whether to copy X; if False, it may be overwritten.

    Returns
    -------
    alphas : ndarray of shape (n_alphas,)
        The grid of alpha values to search over.

    """
    # 计算弹性网格搜索的 alpha 参数网格
    # X : {array-like, sparse matrix} of shape (n_samples, n_features)
    #     Training data. Pass directly as Fortran-contiguous data to avoid
    #     unnecessary memory duplication

    # y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
    #     Target values

    # Xy : array-like of shape (n_features,) or (n_features, n_outputs),\
    #      default=None
    #     Xy = np.dot(X.T, y) that can be precomputed.

    # l1_ratio : float, default=1.0
    #     The elastic net mixing parameter, with ``0 < l1_ratio <= 1``.
    #     For ``l1_ratio = 0`` the penalty is an L2 penalty. (currently not
    #     supported) ``For l1_ratio = 1`` it is an L1 penalty. For
    #     ``0 < l1_ratio <1``, the penalty is a combination of L1 and L2.

    # eps : float, default=1e-3
    #     Length of the path. ``eps=1e-3`` means that
    #     ``alpha_min / alpha_max = 1e-3``

    # n_alphas : int, default=100
    #     Number of alphas along the regularization path

    # fit_intercept : bool, default=True
    #     Whether to fit an intercept or not

    # copy_X : bool, default=True
    #     If ``True``, X will be copied; else, it may be overwritten.

    if l1_ratio == 0:
        # 如果 l1_ratio 为 0，则抛出 ValueError
        raise ValueError(
            "Automatic alpha grid generation is not supported for"
            " l1_ratio=0. Please supply a grid by providing "
            "your estimator with the appropriate `alphas=` "
            "argument."
        )
    
    # 计算样本数目
    n_samples = len(y)

    # 初始化 sparse_center 标志为 False
    sparse_center = False

    # 如果 Xy 为 None，则进行以下处理
    if Xy is None:
        # 检查 X 是否是稀疏矩阵
        X_sparse = sparse.issparse(X)
        # 如果 X 是稀疏矩阵且需要拟合截距，则设置 sparse_center 为 True
        sparse_center = X_sparse and fit_intercept
        # 检查并准备 X，接受 CSC 格式的稀疏矩阵，可选地进行复制
        X = check_array(
            X, accept_sparse="csc", copy=(copy_X and fit_intercept and not X_sparse)
        )
        # 如果 X 不是稀疏矩阵，则调用 _preprocess_data 处理 X 和 y
        if not X_sparse:
            # 由于上述行的处理，X 可以就地修改
            X, y, _, _, _ = _preprocess_data(
                X, y, fit_intercept=fit_intercept, copy=False
            )
        # 计算 X.T 与 y 的稀疏点积，得到 Xy
        Xy = safe_sparse_dot(X.T, y, dense_output=True)

        # 如果需要拟合截距，并且 X 是稀疏矩阵，则进行以下处理
        if sparse_center:
            # 为了在稀疏矩阵中找到 alpha_max 的方法
            # 通过以下方式找到 alpha_max：不破坏稀疏矩阵的稀疏性
            _, _, X_offset, _, X_scale = _preprocess_data(
                X, y, fit_intercept=fit_intercept
            )
            mean_dot = X_offset * np.sum(y)

    # 如果 Xy 的维度为 1，则将其转换为二维
    if Xy.ndim == 1:
        Xy = Xy[:, np.newaxis]

    # 如果需要在稀疏矩阵中心化，则进行以下处理
    if sparse_center:
        if fit_intercept:
            Xy -= mean_dot[:, np.newaxis]

    # 计算 alpha_max 的值
    alpha_max = np.sqrt(np.sum(Xy**2, axis=1)).max() / (n_samples * l1_ratio)

    # 如果 alpha_max 小于等于 float 类型的最小分辨率，则返回一系列的 alphas
    if alpha_max <= np.finfo(float).resolution:
        alphas = np.empty(n_alphas)
        alphas.fill(np.finfo(float).resolution)
        return alphas

    # 返回在对数空间中从 alpha_max 到 alpha_max * eps 的等比数列
    return np.geomspace(alpha_max, alpha_max * eps, num=n_alphas)
# 使用 @validate_params 装饰器对 lasso_path 函数进行参数验证，确保输入参数符合指定的类型和条件
@validate_params(
    {
        "X": ["array-like", "sparse matrix"],  # X 参数应为 array-like 或 sparse matrix 类型
        "y": ["array-like", "sparse matrix"],  # y 参数应为 array-like 或 sparse matrix 类型
        "eps": [Interval(Real, 0, None, closed="neither")],  # eps 参数应为大于零的实数
        "n_alphas": [Interval(Integral, 1, None, closed="left")],  # n_alphas 参数应为大于等于 1 的整数
        "alphas": ["array-like", None],  # alphas 参数应为 array-like 或 None
        "precompute": [StrOptions({"auto"}), "boolean", "array-like"],  # precompute 参数可为 "auto", boolean 或 array-like
        "Xy": ["array-like", None],  # Xy 参数应为 array-like 或 None
        "copy_X": ["boolean"],  # copy_X 参数应为布尔值
        "coef_init": ["array-like", None],  # coef_init 参数应为 array-like 或 None
        "verbose": ["verbose"],  # verbose 参数应为 verbose 类型
        "return_n_iter": ["boolean"],  # return_n_iter 参数应为布尔值
        "positive": ["boolean"],  # positive 参数应为布尔值
    },
    prefer_skip_nested_validation=True,  # 首选跳过嵌套验证
)
def lasso_path(
    X,
    y,
    *,
    eps=1e-3,  # eps 默认值为 1e-3
    n_alphas=100,  # n_alphas 默认值为 100
    alphas=None,  # alphas 默认为 None
    precompute="auto",  # precompute 默认为 "auto"
    Xy=None,  # Xy 默认为 None
    copy_X=True,  # copy_X 默认为 True
    coef_init=None,  # coef_init 默认为 None
    verbose=False,  # verbose 默认为 False
    return_n_iter=False,  # return_n_iter 默认为 False
    positive=False,  # positive 默认为 False
    **params,  # 允许接收任意额外参数
):
    """Compute Lasso path with coordinate descent.

    The Lasso optimization function varies for mono and multi-outputs.

    For mono-output tasks it is::

        (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

    For multi-output tasks it is::

        (1 / (2 * n_samples)) * ||Y - XW||^2_Fro + alpha * ||W||_21

    Where::

        ||W||_21 = \\sum_i \\sqrt{\\sum_j w_{ij}^2}

    i.e. the sum of norm of each row.

    Read more in the :ref:`User Guide <lasso>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data. Pass directly as Fortran-contiguous data to avoid
        unnecessary memory duplication. If ``y`` is mono-output then ``X``
        can be sparse.

    y : {array-like, sparse matrix} of shape (n_samples,) or \
        (n_samples, n_targets)
        Target values.

    eps : float, default=1e-3
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``.

    n_alphas : int, default=100
        Number of alphas along the regularization path.

    alphas : array-like, default=None
        List of alphas where to compute the models.
        If ``None`` alphas are set automatically.

    precompute : 'auto', bool or array-like of shape \
            (n_features, n_features), default='auto'
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'`` let us decide. The Gram
        matrix can also be passed as argument.

    Xy : array-like of shape (n_features,) or (n_features, n_targets),\
         default=None
        Xy = np.dot(X.T, y) that can be precomputed. It is useful
        only when the Gram matrix is precomputed.

    copy_X : bool, default=True
        If ``True``, X will be copied; else, it may be overwritten.

    coef_init : array-like of shape (n_features, ), default=None
        The initial values of the coefficients.

    verbose : bool or int, default=False
        Amount of verbosity.

    return_n_iter : bool, default=False
        Whether to return the number of iterations or not.
    """
    positive : bool, default=False
        如果设置为True，强制系数为正数。
        （仅当 ``y.ndim == 1`` 时允许）。

    **params : kwargs
        传递给坐标下降求解器的关键字参数。

    Returns
    -------
    alphas : ndarray of shape (n_alphas,)
        计算模型的路径上的 alpha 值。

    coefs : ndarray of shape (n_features, n_alphas) or \
            (n_targets, n_features, n_alphas)
        路径上的系数。

    dual_gaps : ndarray of shape (n_alphas,)
        优化结束时每个 alpha 的对偶间隙。

    n_iters : list of int
        坐标下降优化器为达到每个 alpha 的指定容差而执行的迭代次数。

    See Also
    --------
    lars_path : 使用 LARS 算法计算最小角度回归或 Lasso 路径。
    Lasso : Lasso 是一个线性模型，估计稀疏系数。
    LassoLars : 使用最小角度回归（Lars）拟合的 Lasso 模型。
    LassoCV : Lasso 线性模型，沿着正则化路径进行迭代拟合。
    LassoLarsCV : 使用 LARS 算法进行交叉验证的 Lasso。

    sklearn.decomposition.sparse_encode : 估计器，用于将信号转换为固定原子的稀疏线性组合。

    Notes
    -----
    示例详见
    :ref:`examples/linear_model/plot_lasso_coordinate_descent_path.py
    <sphx_glr_auto_examples_linear_model_plot_lasso_coordinate_descent_path.py>`。

    为避免不必要的内存重复，fit 方法的 X 参数应直接作为 Fortran 连续的 numpy 数组传递。

    注意，在某些情况下，Lars 求解器可能比较快速地实现此功能。特别是，可以使用线性插值来检索 lars_path 输出的值之间的模型系数。

    Examples
    --------

    使用 lasso_path 和 lars_path 进行插值比较：

    >>> import numpy as np
    >>> from sklearn.linear_model import lasso_path
    >>> X = np.array([[1, 2, 3.1], [2.3, 5.4, 4.3]]).T
    >>> y = np.array([1, 2, 3.1])
    >>> # 使用 lasso_path 计算系数路径
    >>> _, coef_path, _ = lasso_path(X, y, alphas=[5., 1., .5])
    >>> print(coef_path)
    [[0.         0.         0.46874778]
     [0.2159048  0.4425765  0.23689075]]

    >>> # 现在使用 lars_path 和一维线性插值来计算相同的路径
    >>> from sklearn.linear_model import lars_path
    >>> alphas, active, coef_path_lars = lars_path(X, y, method='lasso')
    >>> from scipy import interpolate
    >>> coef_path_continuous = interpolate.interp1d(alphas[::-1],
    ...                                             coef_path_lars[:, ::-1])
    >>> print(coef_path_continuous([5., 1., .5]))
    [[0.         0.         0.46915237]
     [0.2159048  0.4425765  0.23668876]]
    # 调用 enet_path 函数，执行弹性网络路径计算，并返回计算结果
    return enet_path(
        X,                  # 输入特征 X
        y,                  # 目标变量 y
        l1_ratio=1.0,       # L1 比例参数，默认为 1.0，表示纯 L1 正则化
        eps=eps,            # 正则化路径的终止条件
        n_alphas=n_alphas,  # 正则化路径中的 alpha 值数量
        alphas=alphas,      # 正则化路径上的 alpha 值列表
        precompute=precompute,  # 是否预计算 Gram 矩阵
        Xy=Xy,              # 输入特征 X 与目标变量 y 的乘积
        copy_X=copy_X,      # 是否复制输入数据 X
        coef_init=coef_init,    # 系数的初始值
        verbose=verbose,    # 是否打印详细信息
        positive=positive,  # 是否强制系数为正
        return_n_iter=return_n_iter,  # 是否返回迭代次数
        **params,           # 其它参数，以关键字参数形式传递
    )
@validate_params(
    {
        "X": ["array-like", "sparse matrix"],  # X 参数应为数组或稀疏矩阵
        "y": ["array-like", "sparse matrix"],  # y 参数应为数组或稀疏矩阵
        "l1_ratio": [Interval(Real, 0.0, 1.0, closed="both")],  # l1_ratio 参数应为实数，范围在 [0.0, 1.0] 之间
        "eps": [Interval(Real, 0.0, None, closed="neither")],  # eps 参数应为正实数
        "n_alphas": [Interval(Integral, 1, None, closed="left")],  # n_alphas 参数应为大于等于 1 的整数
        "alphas": ["array-like", None],  # alphas 参数应为数组或者为空
        "precompute": [StrOptions({"auto"}), "boolean", "array-like"],  # precompute 参数应为 "auto"、布尔值或数组
        "Xy": ["array-like", None],  # Xy 参数应为数组或为空
        "copy_X": ["boolean"],  # copy_X 参数应为布尔值
        "coef_init": ["array-like", None],  # coef_init 参数应为数组或为空
        "verbose": ["verbose"],  # verbose 参数应为详细程度的描述
        "return_n_iter": ["boolean"],  # return_n_iter 参数应为布尔值
        "positive": ["boolean"],  # positive 参数应为布尔值
        "check_input": ["boolean"],  # check_input 参数应为布尔值
    },
    prefer_skip_nested_validation=True,  # 设置了 prefer_skip_nested_validation 参数为 True
)
def enet_path(
    X,
    y,
    *,
    l1_ratio=0.5,  # 默认 l1_ratio 为 0.5
    eps=1e-3,  # 默认 eps 为 0.001
    n_alphas=100,  # 默认 n_alphas 为 100
    alphas=None,  # 默认 alphas 为 None
    precompute="auto",  # 默认 precompute 为 "auto"
    Xy=None,  # 默认 Xy 为 None
    copy_X=True,  # 默认 copy_X 为 True
    coef_init=None,  # 默认 coef_init 为 None
    verbose=False,  # 默认 verbose 为 False
    return_n_iter=False,  # 默认 return_n_iter 为 False
    positive=False,  # 默认 positive 为 False
    check_input=True,  # 默认 check_input 为 True
    **params,
):
    """Compute elastic net path with coordinate descent.

    The elastic net optimization function varies for mono and multi-outputs.

    For mono-output tasks it is::

        1 / (2 * n_samples) * ||y - Xw||^2_2
        + alpha * l1_ratio * ||w||_1
        + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2

    For multi-output tasks it is::

        (1 / (2 * n_samples)) * ||Y - XW||_Fro^2
        + alpha * l1_ratio * ||W||_21
        + 0.5 * alpha * (1 - l1_ratio) * ||W||_Fro^2

    Where::

        ||W||_21 = \\sum_i \\sqrt{\\sum_j w_{ij}^2}

    i.e. the sum of norm of each row.

    Read more in the :ref:`User Guide <elastic_net>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data. Pass directly as Fortran-contiguous data to avoid
        unnecessary memory duplication. If ``y`` is mono-output then ``X``
        can be sparse.

    y : {array-like, sparse matrix} of shape (n_samples,) or \
        (n_samples, n_targets)
        Target values.

    l1_ratio : float, default=0.5
        Number between 0 and 1 passed to elastic net (scaling between
        l1 and l2 penalties). ``l1_ratio=1`` corresponds to the Lasso.

    eps : float, default=1e-3
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``.

    n_alphas : int, default=100
        Number of alphas along the regularization path.

    alphas : array-like, default=None
        List of alphas where to compute the models.
        If None alphas are set automatically.

    precompute : 'auto', bool or array-like of shape \
            (n_features, n_features), default='auto'
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'`` let us decide. The Gram
        matrix can also be passed as argument.
    Xy : array-like of shape (n_features,) or (n_features, n_targets),\
         default=None
        # Xy 是一个数组，形状可以是 (特征数,) 或 (特征数, 目标数)，默认为 None。
        # 它可以是预先计算的 X^T * y。只有在预先计算 Gram 矩阵时才有用。

    copy_X : bool, default=True
        # 如果为 True，将复制 X；否则可能会被覆盖。

    coef_init : array-like of shape (n_features, ), default=None
        # 系数的初始值数组，形状为 (特征数,)，默认为 None。

    verbose : bool or int, default=False
        # 冗余输出的量，可以是布尔值或整数，默认为 False。

    return_n_iter : bool, default=False
        # 是否返回迭代次数。

    positive : bool, default=False
        # 如果设置为 True，则强制系数为正数。
        # （仅在 y 的维度为 1 时允许使用。）

    check_input : bool, default=True
        # 如果设置为 False，将跳过输入验证检查（包括提供的 Gram 矩阵）。
        # 假定这些检查由调用方处理。

    **params : kwargs
        # 传递给坐标下降求解器的关键字参数。

    Returns
    -------
    alphas : ndarray of shape (n_alphas,)
        # 在路径上计算模型时的 alpha 值。

    coefs : ndarray of shape (n_features, n_alphas) or \
            (n_targets, n_features, n_alphas)
        # 路径上的系数。

    dual_gaps : ndarray of shape (n_alphas,)
        # 每个 alpha 值优化结束时的对偶间隙。

    n_iters : list of int
        # 每个 alpha 值达到指定容差所花费的迭代次数。
        # （当 return_n_iter 设置为 True 时返回。）

    See Also
    --------
    MultiTaskElasticNet : 使用混合 L1/L2 范数作为正则化器训练的多任务弹性网络模型。
    MultiTaskElasticNetCV : 带内置交叉验证的多任务 L1/L2 弹性网络。
    ElasticNet : 使用组合 L1 和 L2 先验作为正则化器的线性回归模型。
    ElasticNetCV : 沿着正则化路径进行迭代拟合的弹性网络模型。

    Notes
    -----
    有关示例，请参见
    :ref:`examples/linear_model/plot_lasso_coordinate_descent_path.py
    <sphx_glr_auto_examples_linear_model_plot_lasso_coordinate_descent_path.py>`.

    Examples
    --------
    >>> from sklearn.linear_model import enet_path
    >>> from sklearn.datasets import make_regression
    >>> X, y, true_coef = make_regression(
    ...    n_samples=100, n_features=5, n_informative=2, coef=True, random_state=0
    ... )
    >>> true_coef
    array([ 0.        ,  0.        ,  0.        , 97.9..., 45.7...])
    >>> alphas, estimated_coef, _ = enet_path(X, y, n_alphas=3)
    >>> alphas.shape
    (3,)
    >>> estimated_coef
     array([[ 0.        ,  0.78...,  0.56...],
            [ 0.        ,  1.12...,  0.61...],
            [-0.        , -2.12..., -1.12...],
            [ 0.        , 23.04..., 88.93...],
            [ 0.        , 10.63..., 41.56...]])
    """
    # 从参数中弹出 X_offset，如果存在则赋给 X_offset_param
    X_offset_param = params.pop("X_offset", None)
    # 从参数中弹出并获取 X_scale 参数，如果不存在则为 None
    X_scale_param = params.pop("X_scale", None)
    # 从参数中弹出并获取 sample_weight 参数，如果不存在则为 None
    sample_weight = params.pop("sample_weight", None)
    # 从参数中弹出并获取 tol 参数，如果不存在则为 1e-4
    tol = params.pop("tol", 1e-4)
    # 从参数中弹出并获取 max_iter 参数，如果不存在则为 1000
    max_iter = params.pop("max_iter", 1000)
    # 从参数中弹出并获取 random_state 参数，如果不存在则为 None
    random_state = params.pop("random_state", None)
    # 从参数中弹出并获取 selection 参数，如果不存在则为 "cyclic"
    selection = params.pop("selection", "cyclic")

    # 如果 params 还有未处理的参数，抛出 ValueError 异常
    if len(params) > 0:
        raise ValueError("Unexpected parameters in params", params.keys())

    # 如果需要检查输入数据，则对 X 和 y 进行检查和处理
    if check_input:
        # 检查并转换 X 为指定格式，接受稀疏格式 "csc"，浮点数类型为 np.float64 或 np.float32，F 表示 Fortran 排序
        X = check_array(
            X,
            accept_sparse="csc",
            dtype=[np.float64, np.float32],
            order="F",
            copy=copy_X,
        )
        # 检查并转换 y 为指定格式，接受稀疏格式 "csc"，数据类型与 X 相同，F 表示 Fortran 排序，不允许复制
        y = check_array(
            y,
            accept_sparse="csc",
            dtype=X.dtype.type,
            order="F",
            copy=False,
            ensure_2d=False,
        )
        # 如果 Xy 不为 None，则检查并转换 Xy 为指定格式，数据类型与 X 相同，C 表示 C 顺序，不允许复制
        if Xy is not None:
            Xy = check_array(
                Xy, dtype=X.dtype.type, order="C", copy=False, ensure_2d=False
            )

    # 获取样本数和特征数
    n_samples, n_features = X.shape

    # 判断是否为多输出情况，若 y 的维度不为 1，则为多输出
    multi_output = False
    if y.ndim != 1:
        multi_output = True
        n_targets = y.shape[1]

    # 若为多输出且指定 positive=True，则抛出异常
    if multi_output and positive:
        raise ValueError("positive=True is not allowed for multi-output (y.ndim != 1)")

    # 如果不支持稀疏矩阵且 X 是稀疏矩阵，则抛出异常
    if not multi_output and sparse.issparse(X):
        if X_offset_param is not None:
            # 计算稀疏矩阵的缩放参数，用于传递给 CD 求解器
            X_sparse_scaling = X_offset_param / X_scale_param
            X_sparse_scaling = np.asarray(X_sparse_scaling, dtype=X.dtype)
        else:
            # 如果没有偏移参数，则创建全零数组作为缩放参数
            X_sparse_scaling = np.zeros(n_features, dtype=X.dtype)

    # 如果需要检查输入数据，则调用 _pre_fit 函数对 X、y 进行预处理
    if check_input:
        X, y, _, _, _, precompute, Xy = _pre_fit(
            X,
            y,
            Xy,
            precompute,
            fit_intercept=False,
            copy=False,
            check_input=check_input,
        )

    # 如果 alphas 为 None，则计算一组合适的 alpha 值
    if alphas is None:
        alphas = _alpha_grid(
            X,
            y,
            Xy=Xy,
            l1_ratio=l1_ratio,
            fit_intercept=False,
            eps=eps,
            n_alphas=n_alphas,
            copy_X=False,
        )
    # 否则，如果 alpha 值个数大于 1，则对 alpha 值进行逆序排序
    elif len(alphas) > 1:
        alphas = np.sort(alphas)[::-1]  # 确保 alpha 值逆序排列

    # 获取 alpha 值的个数
    n_alphas = len(alphas)
    # 创建空数组，用于存储对偶间隙
    dual_gaps = np.empty(n_alphas)
    # 创建空列表，用于存储迭代次数
    n_iters = []

    # 检查随机数生成器的状态
    rng = check_random_state(random_state)
    # 如果 selection 参数不是 "random" 或 "cyclic"，则抛出异常
    if selection not in ["random", "cyclic"]:
        raise ValueError("selection should be either random or cyclic.")
    # 判断是否选择随机特征
    random = selection == "random"

    # 如果不是多输出情况，则创建空数组，用于存储系数
    if not multi_output:
        coefs = np.empty((n_features, n_alphas), dtype=X.dtype)
    # 如果 coef_init 为 None，则创建一个空的系数数组，其形状为 (n_targets, n_features, n_alphas)，数据类型与输入数据 X 相同，使用 Fortran 风格存储
    else:
        coefs = np.empty((n_targets, n_features, n_alphas), dtype=X.dtype)

    # 如果 coef_init 为 None，则初始化系数数组 coef_ 为全零数组，其形状为 coefs 的前两个维度，数据类型与输入数据 X 相同，使用 Fortran 风格存储
    if coef_init is None:
        coef_ = np.zeros(coefs.shape[:-1], dtype=X.dtype, order="F")
    else:
        # 否则，使用 coef_init 初始化 coef_，并确保数据类型与输入数据 X 相同，使用 Fortran 风格存储
        coef_ = np.asfortranarray(coef_init, dtype=X.dtype)

    # 遍历 alphas 中的每个元素及其索引 i
    for i, alpha in enumerate(alphas):
        # 计算 L1 正则化系数 l1_reg 和 L2 正则化系数 l2_reg
        # l1_reg 基于 alpha、l1_ratio 和样本数 n_samples 计算
        l1_reg = alpha * l1_ratio * n_samples
        # l2_reg 基于 alpha、(1 - l1_ratio) 和样本数 n_samples 计算
        l2_reg = alpha * (1.0 - l1_ratio) * n_samples

        # 如果不是多输出且输入 X 是稀疏矩阵
        if not multi_output and sparse.issparse(X):
            # 使用 cd_fast 中的稀疏弹性网络协调下降方法拟合模型
            model = cd_fast.sparse_enet_coordinate_descent(
                w=coef_,  # 系数数组
                alpha=l1_reg,  # L1 正则化系数
                beta=l2_reg,  # L2 正则化系数
                X_data=X.data,  # 稀疏矩阵的数据部分
                X_indices=X.indices,  # 稀疏矩阵的索引部分
                X_indptr=X.indptr,  # 稀疏矩阵的指针部分
                y=y,  # 目标值
                sample_weight=sample_weight,  # 样本权重
                X_mean=X_sparse_scaling,  # 稀疏矩阵的均值（用于缩放）
                max_iter=max_iter,  # 最大迭代次数
                tol=tol,  # 迭代收敛容差
                rng=rng,  # 随机数生成器
                random=random,  # 随机性参数
                positive=positive,  # 是否强制系数为正
            )
        # 如果是多输出问题
        elif multi_output:
            # 使用 cd_fast 中的多任务弹性网络协调下降方法拟合模型
            model = cd_fast.enet_coordinate_descent_multi_task(
                coef_,  # 系数数组
                l1_reg,  # L1 正则化系数
                l2_reg,  # L2 正则化系数
                X,  # 输入数据
                y,  # 目标值
                max_iter,  # 最大迭代次数
                tol,  # 迭代收敛容差
                rng,  # 随机数生成器
                random,  # 随机性参数
            )
        # 如果 precompute 是 np.ndarray 类型
        elif isinstance(precompute, np.ndarray):
            # 如果需要检查输入
            if check_input:
                # 检查并确保 precompute 是 C 风格存储的数组
                precompute = check_array(precompute, dtype=X.dtype.type, order="C")
            # 使用 cd_fast 中的 Gram 矩阵弹性网络协调下降方法拟合模型
            model = cd_fast.enet_coordinate_descent_gram(
                coef_,  # 系数数组
                l1_reg,  # L1 正则化系数
                l2_reg,  # L2 正则化系数
                precompute,  # 预计算的 Gram 矩阵
                Xy,  # 输入数据与目标值的乘积
                y,  # 目标值
                max_iter,  # 最大迭代次数
                tol,  # 迭代收敛容差
                rng,  # 随机数生成器
                random,  # 随机性参数
                positive,  # 是否强制系数为正
            )
        # 如果 precompute 是 False
        elif precompute is False:
            # 使用 cd_fast 中的弹性网络协调下降方法拟合模型
            model = cd_fast.enet_coordinate_descent(
                coef_,  # 系数数组
                l1_reg,  # L1 正则化系数
                l2_reg,  # L2 正则化系数
                X,  # 输入数据
                y,  # 目标值
                max_iter,  # 最大迭代次数
                tol,  # 迭代收敛容差
                rng,  # 随机数生成器
                random,  # 随机性参数
                positive,  # 是否强制系数为正
            )
        else:
            # 抛出异常，说明 precompute 参数类型不正确
            raise ValueError(
                "Precompute should be one of True, False, 'auto' or array-like. Got %r"
                % precompute
            )

        # 从模型中解包出系数数组 coef_、对偶间隙 dual_gap_、容差 eps_ 和迭代次数 n_iter_
        coef_, dual_gap_, eps_, n_iter_ = model
        # 将当前迭代的系数数组 coef_ 存储到 coefs 的第 i 列中
        coefs[..., i] = coef_
        # 根据文档字符串中的目标修正返回的对偶间隙 dual_gap
        dual_gaps[i] = dual_gap_ / n_samples
        # 将当前迭代的 n_iter_ 添加到 n_iters 列表中
        n_iters.append(n_iter_)

        # 如果设置了 verbose
        if verbose:
            # 如果 verbose 大于 2，打印完整的模型信息
            if verbose > 2:
                print(model)
            # 如果 verbose 大于 1，打印当前路径索引和总路径数
            elif verbose > 1:
                print("Path: %03i out of %03i" % (i, n_alphas))
            # 否则，在标准错误流中写入一个点，用于进度显示
            else:
                sys.stderr.write(".")

    # 如果需要返回迭代次数
    if return_n_iter:
        # 返回 alphas、系数数组 coefs、修正后的对偶间隙 dual_gaps 和迭代次数列表 n_iters
        return alphas, coefs, dual_gaps, n_iters
    # 否则，只返回 alphas、系数数组 coefs 和修正后的对偶间隙 dual_gaps
    return alphas, coefs, dual_gaps
###############################################################################
# ElasticNet model

# 弹性网络模型，结合了L1和L2正则化作为正则化项的线性回归模型。

class ElasticNet(MultiOutputMixin, RegressorMixin, LinearModel):
    """Linear regression with combined L1 and L2 priors as regularizer.

    Minimizes the objective function::

            1 / (2 * n_samples) * ||y - Xw||^2_2
            + alpha * l1_ratio * ||w||_1
            + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2

    If you are interested in controlling the L1 and L2 penalty
    separately, keep in mind that this is equivalent to::

            a * ||w||_1 + 0.5 * b * ||w||_2^2

    where::

            alpha = a + b and l1_ratio = a / (a + b)

    The parameter l1_ratio corresponds to alpha in the glmnet R package while
    alpha corresponds to the lambda parameter in glmnet. Specifically, l1_ratio
    = 1 is the lasso penalty. Currently, l1_ratio <= 0.01 is not reliable,
    unless you supply your own sequence of alpha.

    Read more in the :ref:`User Guide <elastic_net>`.

    Parameters
    ----------
    alpha : float, default=1.0
        Constant that multiplies the penalty terms. Defaults to 1.0.
        See the notes for the exact mathematical meaning of this
        parameter. ``alpha = 0`` is equivalent to an ordinary least square,
        solved by the :class:`LinearRegression` object. For numerical
        reasons, using ``alpha = 0`` with the ``Lasso`` object is not advised.
        Given this, you should use the :class:`LinearRegression` object.

    l1_ratio : float, default=0.5
        The ElasticNet mixing parameter, with ``0 <= l1_ratio <= 1``. For
        ``l1_ratio = 0`` the penalty is an L2 penalty. ``For l1_ratio = 1`` it
        is an L1 penalty.  For ``0 < l1_ratio < 1``, the penalty is a
        combination of L1 and L2.

    fit_intercept : bool, default=True
        Whether the intercept should be estimated or not. If ``False``, the
        data is assumed to be already centered.

    precompute : bool or array-like of shape (n_features, n_features),\
                 default=False
        Whether to use a precomputed Gram matrix to speed up
        calculations. The Gram matrix can also be passed as argument.
        For sparse input this option is always ``False`` to preserve sparsity.
        Check :ref:`an example on how to use a precomputed Gram Matrix in ElasticNet
        <sphx_glr_auto_examples_linear_model_plot_elastic_net_precomputed_gram_matrix_with_weighted_samples.py>`
        for details.

    max_iter : int, default=1000
        The maximum number of iterations.

    copy_X : bool, default=True
        If ``True``, X will be copied; else, it may be overwritten.

    tol : float, default=1e-4
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``, see Notes below.
    warm_start : bool, default=False
        # 是否启用热启动，即是否重用上一次调用 fit 方法的解作为初始化，否则会擦除上一次的解
        当设置为 ``True`` 时，将重用前一次调用 fit 方法的解作为初始化；否则，会擦除前一次的解。
        参见 :term:`术语表 <warm_start>`。

    positive : bool, default=False
        # 是否强制系数为正数
        当设置为 ``True`` 时，强制系数为正数。

    random_state : int, RandomState instance, default=None
        # 伪随机数生成器的种子，用于选择要更新的随机特征
        用于 'selection' == 'random' 时选择要更新的随机特征的伪随机数生成器的种子。
        传入一个整数以确保多次函数调用产生可重现的输出。
        参见 :term:`术语表 <random_state>`。

    selection : {'cyclic', 'random'}, default='cyclic'
        # 系数更新的选择方式
        如果设置为 'random'，则每次迭代随机更新一个系数，而不是默认顺序循环遍历特征。
        这种设置通常导致收敛速度显著加快，尤其是当 tol 大于 1e-4 时。

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,) or (n_targets, n_features)
        # 参数向量 (成本函数公式中的 w)

    sparse_coef_ : sparse matrix of shape (n_features,) or \
            (n_targets, n_features)
        # 稀疏表示的 `coef_`

    intercept_ : float or ndarray of shape (n_targets,)
        # 决策函数中的独立项

    n_iter_ : list of int
        # 坐标下降求解器运行的迭代次数，以达到指定的容差

    dual_gap_ : float or ndarray of shape (n_targets,)
        # 在优化结束时，给定参数 alpha 下的对偶间隙，与 y 的每个观测值具有相同的形状

    n_features_in_ : int
        # 在 `fit` 中看到的特征数目
        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        # 在 `fit` 中看到的特征名称，仅在 `X` 具有全部为字符串的特征名称时定义
        .. versionadded:: 1.0

    See Also
    --------
    ElasticNetCV : 通过交叉验证进行最佳模型选择的弹性网模型
    SGDRegressor : 实现带增量训练的弹性网回归
    SGDClassifier : 实现带弹性网惩罚的逻辑回归 (`SGDClassifier(loss="log_loss", penalty="elasticnet")`)

    Notes
    -----
    # 避免不必要的内存复制，`fit` 方法的 `X` 参数应直接传递为 Fortran 连续的 numpy 数组

    # 基于 `tol` 的精确停止准则如下：
    首先，检查最大坐标更新，即 :math:`\\max_j |w_j^{new} - w_j^{old}|` 是否小于 `tol` 倍的最大绝对系数，即 :math:`\\max_j |w_j|`。
    如果是，则另外检查对偶间隙是否小于 `tol` 倍的 :math:`||y||_2^2 / n_{\text{samples}}`。

    Examples
    --------
    # 导入 ElasticNet 模型
    >>> from sklearn.linear_model import ElasticNet
    # 导入 make_regression 函数用于生成回归样本数据
    >>> from sklearn.datasets import make_regression

    # 生成包含两个特征的回归样本数据 X 和目标变量 y，使用随机种子确保可重复性
    >>> X, y = make_regression(n_features=2, random_state=0)
    # 创建 ElasticNet 回归模型对象 regr，使用相同的随机种子以便可重复性
    >>> regr = ElasticNet(random_state=0)
    # 使用 X, y 训练 ElasticNet 模型
    >>> regr.fit(X, y)
    # 打印训练后的模型系数
    ElasticNet(random_state=0)
    # 打印训练后的模型截距
    >>> print(regr.intercept_)
    1.451...
    # 对新样本 [[0, 0]] 进行预测并打印预测结果
    >>> print(regr.predict([[0, 0]]))
    [1.451...]
    """

    # 定义参数约束字典 _parameter_constraints
    _parameter_constraints: dict = {
        "alpha": [Interval(Real, 0, None, closed="left")],
        "l1_ratio": [Interval(Real, 0, 1, closed="both")],
        "fit_intercept": ["boolean"],
        "precompute": ["boolean", "array-like"],
        "max_iter": [Interval(Integral, 1, None, closed="left"), None],
        "copy_X": ["boolean"],
        "tol": [Interval(Real, 0, None, closed="left")],
        "warm_start": ["boolean"],
        "positive": ["boolean"],
        "random_state": ["random_state"],
        "selection": [StrOptions({"cyclic", "random"})],
    }

    # 将 enet_path 函数作为静态方法 path
    path = staticmethod(enet_path)

    # ElasticNet 类的初始化方法
    def __init__(
        self,
        alpha=1.0,
        *,
        l1_ratio=0.5,
        fit_intercept=True,
        precompute=False,
        max_iter=1000,
        copy_X=True,
        tol=1e-4,
        warm_start=False,
        positive=False,
        random_state=None,
        selection="cyclic",
    ):
        # 设置 ElasticNet 模型的参数
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.precompute = precompute
        self.max_iter = max_iter
        self.copy_X = copy_X
        self.tol = tol
        self.warm_start = warm_start
        self.positive = positive
        self.random_state = random_state
        self.selection = selection

    # 装饰器，用于实现 _fit_context 方法
    @_fit_context(prefer_skip_nested_validation=True)
    # 返回稀疏表示的 coef_ 的属性方法 sparse_coef_
    @property
    def sparse_coef_(self):
        """Sparse representation of the fitted `coef_`."""
        return sparse.csr_matrix(self.coef_)

    # 内部方法，计算决策函数
    def _decision_function(self, X):
        """Decision function of the linear model.

        Parameters
        ----------
        X : numpy array or scipy.sparse matrix of shape (n_samples, n_features)

        Returns
        -------
        T : ndarray of shape (n_samples,)
            The predicted decision function.
        """
        # 检查模型是否已经拟合
        check_is_fitted(self)
        # 如果输入 X 是稀疏矩阵，则使用稀疏矩阵运算
        if sparse.issparse(X):
            return safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_
        else:
            # 否则使用默认的决策函数计算
            return super()._decision_function(X)
###############################################################################
# Lasso model

class Lasso(ElasticNet):
    """Linear Model trained with L1 prior as regularizer (aka the Lasso).

    The optimization objective for Lasso is::

        (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

    Technically the Lasso model is optimizing the same objective function as
    the Elastic Net with ``l1_ratio=1.0`` (no L2 penalty).

    Read more in the :ref:`User Guide <lasso>`.

    Parameters
    ----------
    alpha : float, default=1.0
        Constant that multiplies the L1 term, controlling regularization
        strength. `alpha` must be a non-negative float i.e. in `[0, inf)`.

        When `alpha = 0`, the objective is equivalent to ordinary least
        squares, solved by the :class:`LinearRegression` object. For numerical
        reasons, using `alpha = 0` with the `Lasso` object is not advised.
        Instead, you should use the :class:`LinearRegression` object.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to False, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    precompute : bool or array-like of shape (n_features, n_features),\
                 default=False
        Whether to use a precomputed Gram matrix to speed up
        calculations. The Gram matrix can also be passed as argument.
        For sparse input this option is always ``False`` to preserve sparsity.

    copy_X : bool, default=True
        If ``True``, X will be copied; else, it may be overwritten.

    max_iter : int, default=1000
        The maximum number of iterations.

    tol : float, default=1e-4
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``, see Notes below.

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        See :term:`the Glossary <warm_start>`.

    positive : bool, default=False
        When set to ``True``, forces the coefficients to be positive.

    random_state : int, RandomState instance, default=None
        The seed of the pseudo random number generator that selects a random
        feature to update. Used when ``selection`` == 'random'.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    selection : {'cyclic', 'random'}, default='cyclic'
        If set to 'random', a random coefficient is updated every iteration
        rather than looping over features sequentially by default. This
        (setting to 'random') often leads to significantly faster convergence
        especially when tol is higher than 1e-4.

    Attributes
    ----------
    """
    # 参数向量 (w) 在成本函数公式中的系数，形状为 (n_features,) 或 (n_targets, n_features)
    coef_ : ndarray of shape (n_features,) or (n_targets, n_features)

    # 在优化结束时，给定参数 alpha，每个观测 y 对应的对偶间隙，形状与 y 的每个观测一致
    dual_gap_ : float or ndarray of shape (n_targets,)

    # 从 ``coef_`` 派生的只读属性，稀疏矩阵，形状为 (n_features, 1) 或 (n_targets, n_features)
    sparse_coef_ : sparse matrix of shape (n_features, 1) or (n_targets, n_features)

    # 决策函数中的独立项，形状为 float 或 (n_targets,)
    intercept_ : float or ndarray of shape (n_targets,)

    # 达到指定容差时由坐标下降求解器运行的迭代次数，形状为 int 或 int 列表
    n_iter_ : int or list of int

    # 在 `fit` 过程中看到的特征数量
    n_features_in_ : int

        .. versionadded:: 0.24

    # 在 `fit` 过程中看到的特征名称数组，仅在 `X` 全为字符串特征名时定义
    feature_names_in_ : ndarray of shape (n_features_in_,)

        .. versionadded:: 1.0

    See Also
    --------
    # 使用 LARS 算法的正则化路径
    lars_path : Regularization path using LARS.
    # 使用 Lasso 算法的正则化路径
    lasso_path : Regularization path using Lasso.
    # 使用 LARS 算法的 Lasso 正则化路径
    LassoLars : Lasso Path along the regularization parameter using LARS algorithm.
    # 通过交叉验证确定 Lasso alpha 参数
    LassoCV : Lasso alpha parameter by cross-validation.
    # 通过交叉验证确定 Lasso 最小角度算法的参数
    LassoLarsCV : Lasso least angle parameter algorithm by cross-validation.
    # sklearn.decomposition.sparse_encode : 稀疏编码数组估计器

    Notes
    -----
    # 用于拟合模型的算法是坐标下降法
    The algorithm used to fit the model is coordinate descent.

    # 为了避免不必要的内存复制，`fit` 方法的 X 参数应直接传递为 Fortran 连续的 numpy 数组
    To avoid unnecessary memory duplication the X argument of the fit method
    should be directly passed as a Fortran-contiguous numpy array.

    # 正则化可以改善问题的条件性并减少估计的方差。较大的值表示更强的正则化。
    Regularization improves the conditioning of the problem and
    reduces the variance of the estimates. Larger values specify stronger
    regularization.

    # alpha 对应于其他线性模型（如 LogisticRegression 或 LinearSVC）中的 `1 / (2C)`
    Alpha corresponds to `1 / (2C)` in other linear
    models such as :class:`~sklearn.linear_model.LogisticRegression` or
    :class:`~sklearn.svm.LinearSVC`.

    # 如果传递了一个数组，则惩罚被假定为特定于目标。因此，它们必须在数量上对应。
    If an array is passed, penalties are
    assumed to be specific to the targets. Hence they must correspond in
    number.

    # 基于 `tol` 的精确停止标准如下：
    The precise stopping criteria based on `tol` are the following:

    # 首先，检查最大坐标更新，即 :math:`\\max_j |w_j^{new} - w_j^{old}|` 是否小于 `tol` 乘以最大绝对系数，即 :math:`\\max_j |w_j|`
    First, check that maximum coordinate update, i.e. :math:`\\max_j |w_j^{new} - w_j^{old}|`
    is smaller than `tol` times the maximum absolute coefficient, :math:`\\max_j |w_j|`.

    # 如果是，则另外检查对偶间隙是否小于 `tol` 倍的 :math:`||y||_2^2 / n_{\\text{samples}}`
    If so, then additionally check whether the dual gap is smaller than `tol` times
    :math:`||y||_2^2 / n_{\\text{samples}}`.

    # 目标可以是一个二维数组，导致优化以下目标：
    The target can be a 2-dimensional array, resulting in the optimization of the
    following objective::

        (1 / (2 * n_samples)) * ||Y - XW||^2_F + alpha * ||W||_11

    # 其中 :math:`||W||_{1,1}` 是矩阵系数的幅度之和。不应与 `MultiTaskLasso` 混淆，后者惩罚系数的 :math:`L_{2,1}` 范数，导致系数的逐行稀疏性。
    where :math:`||W||_{1,1}` is the sum of the magnitude of the matrix coefficients.
    It should not be confused with :class:`~sklearn.linear_model.MultiTaskLasso` which
    instead penalizes the :math:`L_{2,1}` norm of the coefficients, yielding row-wise
    sparsity in the coefficients.

    Examples
    --------
    """
    _parameter_constraints: dict = {
        **ElasticNet._parameter_constraints,
    }
    _parameter_constraints.pop("l1_ratio")
    """
    # 创建一个新的字典 _parameter_constraints，继承自 ElasticNet 类的 _parameter_constraints 字典
    # 移除新字典中的 "l1_ratio" 键，该键不再适用于 Lasso 模型的参数约束
    
    path = staticmethod(enet_path)
    """
    path = staticmethod(enet_path)
    """
    # 将 enet_path 方法转换为静态方法，并将其赋值给变量 path
    
    def __init__(
        self,
        alpha=1.0,
        *,
        fit_intercept=True,
        precompute=False,
        copy_X=True,
        max_iter=1000,
        tol=1e-4,
        warm_start=False,
        positive=False,
        random_state=None,
        selection="cyclic",
    ):
        """
        __init__(
            self,
            alpha=1.0,
            *,
            fit_intercept=True,
            precompute=False,
            copy_X=True,
            max_iter=1000,
            tol=1e-4,
            warm_start=False,
            positive=False,
            random_state=None,
            selection="cyclic",
        )
        """
        # Lasso 类的初始化方法，设置模型的各种参数
        super().__init__(
            alpha=alpha,
            l1_ratio=1.0,  # 在 Lasso 中，l1_ratio 固定为 1.0，即纯 L1 正则化
            fit_intercept=fit_intercept,
            precompute=precompute,
            copy_X=copy_X,
            max_iter=max_iter,
            tol=tol,
            warm_start=warm_start,
            positive=positive,
            random_state=random_state,
            selection=selection,
        )
    """
    super().__init__(
        alpha=alpha,
        l1_ratio=1.0,
        fit_intercept=fit_intercept,
        precompute=precompute,
        copy_X=copy_X,
        max_iter=max_iter,
        tol=tol,
        warm_start=warm_start,
        positive=positive,
        random_state=random_state,
        selection=selection,
    )
    """
    # 调用父类 ElasticNet 的初始化方法，传入 Lasso 模型特定的参数
    # 这里的 super() 指代 Lasso 类的父类 ElasticNet
    # 初始化过程中，设置了 alpha 和其余的参数，确保模型能正确运行
###############################################################################
# Functions for CV with paths functions

# 计算由路径函数 'path' 计算的模型的均方误差（MSE）
def _path_residuals(
    X,
    y,
    sample_weight,
    train,
    test,
    fit_intercept,
    path,
    path_params,
    alphas=None,
    l1_ratio=1,
    X_order=None,
    dtype=None,
):
    """Returns the MSE for the models computed by 'path'.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        训练数据。

    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        目标值。

    sample_weight : None or array-like of shape (n_samples,)
        样本权重。

    train : list of indices
        训练集的索引列表。

    test : list of indices
        测试集的索引列表。

    path : callable
        返回路径上模型列表的函数。参见 enet_path 的签名。

    path_params : dictionary
        传递给路径函数的参数。

    alphas : array-like, default=None
        用于交叉验证的浮点数数组。如果未提供，则使用 'path' 计算。

    l1_ratio : float, default=1
        介于 0 和 1 之间的浮点数，传递给 ElasticNet（L1 和 L2 惩罚的缩放比例）。
        当 'l1_ratio = 0' 时，是一个 L2 惩罚。当 'l1_ratio = 1' 时，是一个 L1 惩罚。
        当 '0 < l1_ratio < 1' 时，是 L1 和 L2 的组合惩罚。

    X_order : {'F', 'C'}, default=None
        路径函数期望的数组顺序，以避免内存复制。

    dtype : a numpy dtype, default=None
        路径函数期望的数组的数据类型，以避免内存复制。
    """
    X_train = X[train]  # 从整体数据中选择训练集数据
    y_train = y[train]  # 从整体目标数据中选择训练集目标
    X_test = X[test]    # 从整体数据中选择测试集数据
    y_test = y[test]    # 从整体目标数据中选择测试集目标

    if sample_weight is None:
        sw_train, sw_test = None, None
    else:
        sw_train = sample_weight[train]  # 从整体样本权重中选择训练集样本权重
        sw_test = sample_weight[test]    # 从整体样本权重中选择测试集样本权重
        n_samples = X_train.shape[0]
        # TLDR: 将训练集的 sw_train 重新缩放，使其总和为 n_samples。
        # 参见 ElasticNet.fit 中的 TLDR 和长评论。
        sw_train *= n_samples / np.sum(sw_train)
        # 注意：作为替代，我们也可以重新缩放 alpha 而不是 sample_weight：
        #
        #     alpha *= np.sum(sample_weight) / n_samples

    if not sparse.issparse(X):
        # 如果 X 不是稀疏矩阵
        for array, array_input in (
            (X_train, X),
            (y_train, y),
            (X_test, X),
            (y_test, y),
        ):
            if array.base is not array_input and not array.flags["WRITEABLE"]:
                # fancy indexing 应该创建可写副本，但对于只读 memmaps，它不会这样做（参见 numpy#14132）。
                array.setflags(write=True)

    if y.ndim == 1:
        precompute = path_params["precompute"]  # 预计算路径参数中的 precompute
    else:
        # 如果没有多任务的 Gram 变体存在
        # 回退到默认的 enet_multitask
        precompute = False

    # 进行预适应处理
    X_train, y_train, X_offset, y_offset, X_scale, precompute, Xy = _pre_fit(
        X_train,
        y_train,
        None,
        precompute,
        fit_intercept=fit_intercept,
        copy=False,
        sample_weight=sw_train,
    )

    # 复制路径参数，并设置相关属性
    path_params = path_params.copy()
    path_params["Xy"] = Xy
    path_params["X_offset"] = X_offset
    path_params["X_scale"] = X_scale
    path_params["precompute"] = precompute
    path_params["copy_X"] = False
    path_params["alphas"] = alphas
    # sparse cd solver 需要的参数
    path_params["sample_weight"] = sw_train

    # 如果路径参数中包含 "l1_ratio"，设置其值为 l1_ratio
    if "l1_ratio" in path_params:
        path_params["l1_ratio"] = l1_ratio

    # 在这里进行排序和类型转换，如果在路径中执行，
    # X 会被复制，并且这里保留了一个引用
    X_train = check_array(X_train, accept_sparse="csc", dtype=dtype, order=X_order)
    # 调用路径求解器，返回 alphas（正则化参数）、coefs（系数）和 _
    alphas, coefs, _ = path(X_train, y_train, **path_params)
    # 删除临时变量 X_train 和 y_train
    del X_train, y_train

    # 如果 y 的维度为 1
    if y.ndim == 1:
        # 这样做是为了使其与多输出保持一致
        coefs = coefs[np.newaxis, :, :]
        y_offset = np.atleast_1d(y_offset)
        y_test = y_test[:, np.newaxis]

    # 计算截距
    intercepts = y_offset[:, np.newaxis] - np.dot(X_offset, coefs)
    # 计算 X_test 与 coefs 的点乘结果
    X_test_coefs = safe_sparse_dot(X_test, coefs)
    # 计算残差
    residues = X_test_coefs - y_test[:, :, np.newaxis]
    residues += intercepts
    # 如果没有样本权重，则计算均方误差（MSE）
    if sample_weight is None:
        this_mse = (residues**2).mean(axis=0)
    else:
        this_mse = np.average(residues**2, weights=sw_test, axis=0)

    # 返回均方误差的均值
    return this_mse.mean(axis=0)
class LinearModelCV(MultiOutputMixin, LinearModel, ABC):
    """Base class for iterative model fitting along a regularization path."""

    # 定义参数约束字典，指定每个参数的类型和取值范围
    _parameter_constraints: dict = {
        "eps": [Interval(Real, 0, None, closed="neither")],  # 参数 eps 的取值约束
        "n_alphas": [Interval(Integral, 1, None, closed="left")],  # 参数 n_alphas 的取值约束
        "alphas": ["array-like", None],  # 参数 alphas 的类型约束
        "fit_intercept": ["boolean"],  # 参数 fit_intercept 的类型约束
        "precompute": [StrOptions({"auto"}), "array-like", "boolean"],  # 参数 precompute 的类型约束和取值约束
        "max_iter": [Interval(Integral, 1, None, closed="left")],  # 参数 max_iter 的取值约束
        "tol": [Interval(Real, 0, None, closed="left")],  # 参数 tol 的取值约束
        "copy_X": ["boolean"],  # 参数 copy_X 的类型约束
        "cv": ["cv_object"],  # 参数 cv 的类型约束
        "verbose": ["verbose"],  # 参数 verbose 的类型约束
        "n_jobs": [Integral, None],  # 参数 n_jobs 的类型约束
        "positive": ["boolean"],  # 参数 positive 的类型约束
        "random_state": ["random_state"],  # 参数 random_state 的类型约束
        "selection": [StrOptions({"cyclic", "random"})],  # 参数 selection 的取值约束
    }

    @abstractmethod
    def __init__(
        self,
        eps=1e-3,
        n_alphas=100,
        alphas=None,
        fit_intercept=True,
        precompute="auto",
        max_iter=1000,
        tol=1e-4,
        copy_X=True,
        cv=None,
        verbose=False,
        n_jobs=None,
        positive=False,
        random_state=None,
        selection="cyclic",
    ):
        # 初始化线性模型交叉验证的基本参数
        self.eps = eps
        self.n_alphas = n_alphas
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.precompute = precompute
        self.max_iter = max_iter
        self.tol = tol
        self.copy_X = copy_X
        self.cv = cv
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.positive = positive
        self.random_state = random_state
        self.selection = selection

    @abstractmethod
    def _get_estimator(self):
        """Model to be fitted after the best alpha has been determined."""

    @abstractmethod
    def _is_multitask(self):
        """Bool indicating if class is meant for multidimensional target."""

    @staticmethod
    @abstractmethod
    def path(X, y, **kwargs):
        """Compute path with coordinate descent."""

    @_fit_context(prefer_skip_nested_validation=True)
    def _more_tags(self):
        # 返回额外的标签信息，用于测试时的特定处理
        # 注意: check_sample_weights_invariance(kind='ones') 应该可以工作，但目前只能标记整个测试为 xfail.
        return {
            "_xfail_checks": {
                "check_sample_weights_invariance": (
                    "zero sample_weight is not equivalent to removing samples"
                ),
            }
        }
    def get_metadata_routing(self):
        """
        Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        .. versionadded:: 1.4

        Returns
        -------
        routing : MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        # 创建一个 MetadataRouter 对象，使用当前对象的类名作为所有者
        router = (
            MetadataRouter(owner=self.__class__.__name__)
            # 将当前对象添加到路由请求中
            .add_self_request(self)
            # 添加路由规则，使用 check_cv 函数检查的交叉验证分割器作为分割器
            .add(
                splitter=check_cv(self.cv),
                # 创建一个 MethodMapping 对象，将 'fit' 方法映射到 'split' 方法
                method_mapping=MethodMapping().add(caller="fit", callee="split"),
            )
        )
        # 返回构建好的 MetadataRouter 对象
        return router
class LassoCV(RegressorMixin, LinearModelCV):
    """Lasso linear model with iterative fitting along a regularization path.
    
    See glossary entry for :term:`cross-validation estimator`.

    The best model is selected by cross-validation.

    The optimization objective for Lasso is::

        (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

    Read more in the :ref:`User Guide <lasso>`.

    Parameters
    ----------
    eps : float, default=1e-3
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``.

    n_alphas : int, default=100
        Number of alphas along the regularization path.

    alphas : array-like, default=None
        List of alphas where to compute the models.
        If ``None`` alphas are set automatically.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    precompute : 'auto', bool or array-like of shape \
            (n_features, n_features), default='auto'
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'`` let us decide. The Gram
        matrix can also be passed as argument.

    max_iter : int, default=1000
        The maximum number of iterations.

    tol : float, default=1e-4
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.

    copy_X : bool, default=True
        If ``True``, X will be copied; else, it may be overwritten.

    cv : int, cross-validation generator or iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross-validation,
        - int, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For int/None inputs, :class:`~sklearn.model_selection.KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.

    verbose : bool or int, default=False
        Amount of verbosity.

    n_jobs : int, default=None
        Number of CPUs to use during the cross validation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    positive : bool, default=False
        If positive, restrict regression coefficients to be positive.
    """
    # LassoCV 类，继承自 RegressorMixin 和 LinearModelCV，实现了基于正则化路径的迭代拟合的 Lasso 线性模型

    def __init__(self, eps=1e-3, n_alphas=100, alphas=None, fit_intercept=True,
                 precompute='auto', max_iter=1000, tol=1e-4, copy_X=True,
                 cv=None, verbose=False, n_jobs=None, positive=False):
        # 初始化函数，设置 LassoCV 对象的各种参数

        # 调用父类的初始化方法，初始化 RegressorMixin 和 LinearModelCV 的参数
        super().__init__()

        self.eps = eps
        # 路径长度，用于确定正则化路径的范围，eps=1e-3 表示 alpha_min / alpha_max = 1e-3

        self.n_alphas = n_alphas
        # 正则化路径中的 alpha 数量，默认为 100

        self.alphas = alphas
        # 用于计算模型的 alpha 列表，如果为 None，则 alpha 会自动设置

        self.fit_intercept = fit_intercept
        # 是否计算模型的截距，默认为 True，如果为 False，则不使用截距（即数据预期已居中）

        self.precompute = precompute
        # 是否使用预先计算的 Gram 矩阵来加速计算，默认为 'auto'，可以根据情况自动决定是否使用 Gram 矩阵

        self.max_iter = max_iter
        # 最大迭代次数，默认为 1000

        self.tol = tol
        # 优化过程的容差，如果更新小于 tol，则优化代码检查对偶间隙是否足够小，并继续直到小于 tol

        self.copy_X = copy_X
        # 如果为 True，则复制 X；否则可能会覆盖 X

        self.cv = cv
        # 交叉验证的划分策略，默认为 None，即使用默认的 5 折交叉验证

        self.verbose = verbose
        # 冗余度级别，控制输出信息的数量

        self.n_jobs = n_jobs
        # 用于交叉验证期间使用的 CPU 数量，默认为 None，意味着在非 joblib.parallel_backend 上下文中使用 1 个 CPU

        self.positive = positive
        # 如果为 True，则限制回归系数为正数
    # random_state : int, RandomState instance, default=None
    #     伪随机数生成器的种子，用于选择要更新的随机特征。当 `selection` == 'random' 时使用。
    #     传递一个整数以便在多次函数调用中产生可重复的输出。参见 :term:`术语表 <random_state>`。

    # selection : {'cyclic', 'random'}, default='cyclic'
    #     如果设置为 'random'，则每次迭代更新一个随机系数，而不是默认的按特征顺序循环。
    #     这通常导致收敛速度显著更快，特别是当 tol 大于 1e-4 时。

    # Attributes
    # ----------
    # alpha_ : float
    #     通过交叉验证选择的惩罚量。

    # coef_ : ndarray of shape (n_features,) or (n_targets, n_features)
    #     参数向量（在成本函数公式中的 w）。

    # intercept_ : float or ndarray of shape (n_targets,)
    #     决策函数中的独立项。

    # mse_path_ : ndarray of shape (n_alphas, n_folds)
    #     每个折叠上测试集的均方误差，变化 alpha。

    # alphas_ : ndarray of shape (n_alphas,)
    #     用于拟合的 alpha 网格。

    # dual_gap_ : float or ndarray of shape (n_targets,)
    #     在达到最优 alpha（``alpha_``）的优化结束时的对偶间隙。

    # n_iter_ : int
    #     坐标下降求解器运行的迭代次数，以达到最优 alpha 的指定容差。

    # n_features_in_ : int
    #     在 :term:`fit` 过程中看到的特征数。

    #     .. versionadded:: 0.24

    # feature_names_in_ : ndarray of shape (`n_features_in_`,)
    #     在 :term:`fit` 过程中看到的特征名称。仅当 `X` 具有所有字符串类型的特征名称时定义。

    #     .. versionadded:: 1.0

    # See Also
    # --------
    # lars_path : 使用 LARS 算法计算最小角度回归或 Lasso 路径。
    # lasso_path : 使用坐标下降计算 Lasso 路径。
    # Lasso : Lasso 是一个估计稀疏系数的线性模型。
    # LassoLars : 使用最小角度回归（LARS）拟合的 Lasso 模型。
    # LassoCV : 沿着正则化路径进行迭代拟合的 Lasso 线性模型。
    # LassoLarsCV : 使用 LARS 算法交叉验证的 Lasso。

    # Notes
    # -----
    # 在 `fit` 中，一旦通过交叉验证找到最佳参数 `alpha`，则再次使用整个训练集拟合模型。

    # 为避免不必要的内存复制，`fit` 方法的 `X` 参数应直接传递为 Fortran 连续的 numpy 数组。

    # 有关示例，请参见
    # :ref:`examples/linear_model/plot_lasso_model_selection.py
    # <sphx_glr_auto_examples_linear_model_plot_lasso_model_selection.py>`。

    # :class:`LassoCV` 与使用 :class:`~sklearn.model_selection.GridSearchCV` 进行的超参数搜索
    # 导致不同的结果。
    # 导入 LassoCV 类
    from sklearn.linear_model import LassoCV
    # 导入 make_regression 函数
    from sklearn.datasets import make_regression
    # 创建一个具有噪声的回归数据集 X, y
    X, y = make_regression(noise=4, random_state=0)
    # 使用 LassoCV 拟合数据，cv=5 表示使用 5 折交叉验证，random_state=0 表示随机种子
    reg = LassoCV(cv=5, random_state=0).fit(X, y)
    # 打印拟合结果的得分
    reg.score(X, y)
    # 对 X 中的第一个样本进行预测
    reg.predict(X[:1,])
class ElasticNetCV(RegressorMixin, LinearModelCV):
    """Elastic Net model with iterative fitting along a regularization path.

    See glossary entry for :term:`cross-validation estimator`.

    Read more in the :ref:`User Guide <elastic_net>`.

    Parameters
    ----------
    l1_ratio : float or list of float, default=0.5
        Float between 0 and 1 passed to ElasticNet (scaling between
        l1 and l2 penalties). For ``l1_ratio = 0``
        the penalty is an L2 penalty. For ``l1_ratio = 1`` it is an L1 penalty.
        For ``0 < l1_ratio < 1``, the penalty is a combination of L1 and L2
        This parameter can be a list, in which case the different
        values are tested by cross-validation and the one giving the best
        prediction score is used. Note that a good choice of list of
        values for l1_ratio is often to put more values close to 1
        (i.e. Lasso) and less close to 0 (i.e. Ridge), as in ``[.1, .5, .7,
        .9, .95, .99, 1]``.

    eps : float, default=1e-3
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``.

    n_alphas : int, default=100
        Number of alphas along the regularization path, used for each l1_ratio.

    alphas : array-like, default=None
        List of alphas where to compute the models.
        If None alphas are set automatically.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    precompute : 'auto', bool or array-like of shape (n_features, n_features), default='auto'
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'`` let us decide. The Gram
        matrix can also be passed as argument.

    max_iter : int, default=1000
        The maximum number of iterations.

    tol : float, default=1e-4
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.

    cv : int, cross-validation generator or iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross-validation,
        - int, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For int/None inputs, :class:`~sklearn.model_selection.KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.

    copy_X : bool, default=True
        If ``True``, X will be copied; else, it may be overwritten.
    """
    # 是否显示详细信息，可以是布尔型或整数，默认为0
    verbose : bool or int, default=0
        Amount of verbosity.

    # 用于交叉验证过程中要使用的 CPU 核心数量，默认为 None
    n_jobs : int, default=None
        Number of CPUs to use during the cross validation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    # 是否强制系数为正数，默认为 False
    positive : bool, default=False
        When set to ``True``, forces the coefficients to be positive.

    # 随机数种子，用于选择要更新的随机特征，当 `selection` == 'random' 时使用
    random_state : int, RandomState instance, default=None
        The seed of the pseudo random number generator that selects a random
        feature to update. Used when ``selection`` == 'random'.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    # 系数更新顺序，默认为 'cyclic'，可以选择 'random'
    selection : {'cyclic', 'random'}, default='cyclic'
        If set to 'random', a random coefficient is updated every iteration
        rather than looping over features sequentially by default. This
        (setting to 'random') often leads to significantly faster convergence
        especially when tol is higher than 1e-4.

    # 模型属性
    Attributes
    ----------

    # 通过交叉验证选择的惩罚值
    alpha_ : float
        The amount of penalization chosen by cross validation.

    # 通过交叉验证选择的 l1 与 l2 惩罚的折中值
    l1_ratio_ : float
        The compromise between l1 and l2 penalization chosen by
        cross validation.

    # 参数向量，即成本函数公式中的 w
    coef_ : ndarray of shape (n_features,) or (n_targets, n_features)
        Parameter vector (w in the cost function formula).

    # 决策函数中的独立项
    intercept_ : float or ndarray of shape (n_targets, n_features)
        Independent term in the decision function.

    # 在每个折叠上测试集的均方误差，变化 l1_ratio 和 alpha
    mse_path_ : ndarray of shape (n_l1_ratio, n_alpha, n_folds)
        Mean square error for the test set on each fold, varying l1_ratio and
        alpha.

    # 用于拟合的 alpha 网格
    alphas_ : ndarray of shape (n_alphas,) or (n_l1_ratio, n_alphas)
        The grid of alphas used for fitting, for each l1_ratio.

    # 最优 alpha 的优化结束时的对偶间隙
    dual_gap_ : float
        The dual gaps at the end of the optimization for the optimal alpha.

    # 达到最优 alpha 所需的坐标下降求解器运行的迭代次数
    n_iter_ : int
        Number of iterations run by the coordinate descent solver to reach
        the specified tolerance for the optimal alpha.

    # 在 `fit` 过程中看到的特征数
    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    # 在 `fit` 过程中看到的特征的名称，仅当 `X` 具有全部为字符串的特征名时定义
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    # 参见
    # --------
    # enet_path : 使用坐标下降计算弹性网络路径
    # ElasticNet : 使用组合 L1 和 L2 先验作为正则化的线性回归模型

    # 注释
    # -----
    # 在 `fit` 方法中，一旦通过交叉验证找到最佳参数 `l1_ratio` 和 `alpha`，则再次使用整个训练集拟合模型。

    # 为避免不必要的内存复制，`fit` 方法的 `X` 参数应直接传递为 Fortran 连续的 numpy 数组。
    # 定义 `_parameter_constraints` 字典，继承自 `LinearModelCV._parameter_constraints`
    # 增加了 `l1_ratio` 参数的约束条件，要求其为实数数组，取值范围在 [0, 1] 闭区间内
    _parameter_constraints: dict = {
        **LinearModelCV._parameter_constraints,
        "l1_ratio": [Interval(Real, 0, 1, closed="both"), "array-like"],
    }

    # 将 `enet_path` 方法设置为静态方法 `path`
    path = staticmethod(enet_path)

    # 初始化 ElasticNetCV 类的构造函数
    def __init__(
        self,
        *,
        l1_ratio=0.5,  # L1 正则化项的比例参数，默认为 0.5
        eps=1e-3,  # 优化算法的收敛阈值，默认为 0.001
        n_alphas=100,  # 正则化路径中的 alpha 参数数量，默认为 100
        alphas=None,  # 指定的 alpha 参数列表，如果为 None 则自动生成
        fit_intercept=True,  # 是否计算截距，默认为 True
        precompute="auto",  # 是否预计算 Gram 矩阵，默认为 "auto"
        max_iter=1000,  # 最大迭代次数，默认为 1000
        tol=1e-4,  # 优化算法的数值精度，默认为 0.0001
        cv=None,  # 交叉验证折数，默认为 None，即使用默认的 5 折交叉验证
        copy_X=True,  # 是否复制输入数据 X，默认为 True
        verbose=0,  # 是否显示详细信息，默认为 0，即不显示
        n_jobs=None,  # 并行运行的作业数量，默认为 None，表示使用所有 CPU 核心
        positive=False,  # 是否强制系数为正，默认为 False
        random_state=None,  # 随机数种子，默认为 None
        selection="cyclic",  # 系数更新策略，默认为 "cyclic"，即循环更新
    ):
        # 初始化对象的各个属性
        self.l1_ratio = l1_ratio
        self.eps = eps
        self.n_alphas = n_alphas
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.precompute = precompute
        self.max_iter = max_iter
        self.tol = tol
        self.cv = cv
        self.copy_X = copy_X
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.positive = positive
        self.random_state = random_state
        self.selection = selection

    # 返回 ElasticNet 模型对象
    def _get_estimator(self):
        return ElasticNet()

    # 判断是否为多任务模型，返回 False
    def _is_multitask(self):
        return False

    # 返回更多的标签信息，表明该模型不支持多输出任务
    def _more_tags(self):
        return {"multioutput": False}
###############################################################################
# Multi Task ElasticNet and Lasso models (with joint feature selection)

# 定义一个多任务 ElasticNet 模型，同时使用 L1 和 L2 混合范数作为正则化项。

class MultiTaskElasticNet(Lasso):
    """Multi-task ElasticNet model trained with L1/L2 mixed-norm as regularizer.

    多任务 ElasticNet 模型，使用 L1/L2 混合范数作为正则化项进行训练。

    The optimization objective for MultiTaskElasticNet is::

        (1 / (2 * n_samples)) * ||Y - XW||_Fro^2
        + alpha * l1_ratio * ||W||_21
        + 0.5 * alpha * (1 - l1_ratio) * ||W||_Fro^2

    多任务 ElasticNet 的优化目标为::

        (1 / (2 * n_samples)) * ||Y - XW||_Fro^2
        + alpha * l1_ratio * ||W||_21
        + 0.5 * alpha * (1 - l1_ratio) * ||W||_Fro^2

    Where::

        ||W||_21 = sum_i sqrt(sum_j W_ij ^ 2)

    其中::

        ||W||_21 = sum_i sqrt(sum_j W_ij ^ 2)

    i.e. the sum of norms of each row.

    即每行的范数之和。

    Read more in the :ref:`User Guide <multi_task_elastic_net>`.

    详细内容请参阅 :ref:`用户指南 <multi_task_elastic_net>`。

    Parameters
    ----------
    alpha : float, default=1.0
        Constant that multiplies the L1/L2 term. Defaults to 1.0.
        乘以 L1/L2 项的常数，默认为 1.0.

    l1_ratio : float, default=0.5
        The ElasticNet mixing parameter, with 0 < l1_ratio <= 1.
        For l1_ratio = 1 the penalty is an L1/L2 penalty. For l1_ratio = 0 it
        is an L2 penalty.
        For ``0 < l1_ratio < 1``, the penalty is a combination of L1/L2 and L2.

        ElasticNet 的混合参数，0 < l1_ratio <= 1。
        当 l1_ratio = 1 时，惩罚项为 L1/L2。当 l1_ratio = 0 时，为 L2 惩罚项。
        对于 ``0 < l1_ratio < 1``，惩罚项是 L1/L2 和 L2 的组合。

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be centered).

        是否计算该模型的截距。如果设置为 False，则计算中不使用截距
        （即预期数据已经中心化）。

    copy_X : bool, default=True
        If ``True``, X will be copied; else, it may be overwritten.

        如果为 True，则复制 X；否则，可能会被覆盖。

    max_iter : int, default=1000
        The maximum number of iterations.

        最大迭代次数。

    tol : float, default=1e-4
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.

        优化过程中的容差：如果更新小于 ``tol``，优化代码会检查对偶间隙是否最优，
        并继续直到小于 ``tol``。

    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        See :term:`the Glossary <warm_start>`.

        当设置为 True 时，重用前一次调用 fit 的解作为初始化，否则，只是擦除先前的解。
        参见 :term:`术语表 <warm_start>`。

    random_state : int, RandomState instance, default=None
        The seed of the pseudo random number generator that selects a random
        feature to update. Used when ``selection`` == 'random'.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

        伪随机数生成器的种子，用于选择要更新的随机特征，当 ``selection`` == 'random' 时使用。
        传递一个整数以在多次函数调用中获得可重现的输出。
        参见 :term:`术语表 <random_state>`。

    selection : {'cyclic', 'random'}, default='cyclic'
        If set to 'random', a random coefficient is updated every iteration
        rather than looping over features sequentially by default. This
        (setting to 'random') often leads to significantly faster convergence
        especially when tol is higher than 1e-4.

        如果设置为 'random'，每次迭代更新一个随机系数，而不是默认顺序循环特征。
        这种设置通常导致收敛速度显著加快，特别是当 tol 高于 1e-4 时。

    Attributes
    ----------
    intercept_ : ndarray of shape (n_targets,)
        Independent term in decision function.

        决策函数中的独立项。

    coef_ : ndarray of shape (n_targets, n_features)
        Parameter vector (W in the cost function formula). If a 1D y is
        passed in at fit (non multi-task usage), ``coef_`` is then a 1D array.
        Note that ``coef_`` stores the transpose of ``W``, ``W.T``.

        参数向量（成本函数公式中的 W）。如果在 fit 中传入一个 1D 的 y（非多任务使用），
        则 ``coef_`` 是一个 1D 数组。注意，``coef_`` 存储了 ``W`` 的转置，即 ``W.T``。
    """
    n_iter_ : int
        # 运行坐标下降求解器达到指定容差的迭代次数。
        Number of iterations run by the coordinate descent solver to reach
        the specified tolerance.

    dual_gap_ : float
        # 优化结束时的对偶间隙。
        The dual gaps at the end of the optimization.

    eps_ : float
        # 通过目标 `y` 的方差缩放的容差。
        The tolerance scaled by the variance of the target `y`.

    sparse_coef_ : sparse matrix of shape (n_features,) or \
            (n_targets, n_features)
        # `coef_` 的稀疏表示。
        Sparse representation of the `coef_`.

    n_features_in_ : int
        # 在拟合过程中看到的特征数。
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        # 在 `fit` 过程中看到的特征名称。仅当 `X` 具有所有字符串特征名称时定义。

        .. versionadded:: 1.0

    See Also
    --------
    MultiTaskElasticNetCV : Multi-task L1/L2 ElasticNet with built-in
        cross-validation.
    ElasticNet : Linear regression with combined L1 and L2 priors as regularizer.
    MultiTaskLasso : Multi-task Lasso model trained with L1/L2
        mixed-norm as regularizer.

    Notes
    -----
    # 用于拟合模型的算法是坐标下降法。
    The algorithm used to fit the model is coordinate descent.

    # 为了避免不必要的内存复制，`fit` 方法的 `X` 和 `y` 参数应直接作为 Fortran 连续的 numpy 数组传递。
    To avoid unnecessary memory duplication the X and y arguments of the fit
    method should be directly passed as Fortran-contiguous numpy arrays.

    Examples
    --------
    >>> from sklearn import linear_model
    >>> clf = linear_model.MultiTaskElasticNet(alpha=0.1)
    >>> clf.fit([[0,0], [1, 1], [2, 2]], [[0, 0], [1, 1], [2, 2]])
    MultiTaskElasticNet(alpha=0.1)
    >>> print(clf.coef_)
    [[0.45663524 0.45612256]
     [0.45663524 0.45612256]]
    >>> print(clf.intercept_)
    [0.0872422 0.0872422]
    """

    _parameter_constraints: dict = {
        **ElasticNet._parameter_constraints,
    }
    for param in ("precompute", "positive"):
        _parameter_constraints.pop(param)

    def __init__(
        self,
        alpha=1.0,
        *,
        l1_ratio=0.5,
        fit_intercept=True,
        copy_X=True,
        max_iter=1000,
        tol=1e-4,
        warm_start=False,
        random_state=None,
        selection="cyclic",
    ):
        # 设置类的属性
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.copy_X = copy_X
        self.tol = tol
        self.warm_start = warm_start
        self.random_state = random_state
        self.selection = selection

    @_fit_context(prefer_skip_nested_validation=True)
    # 使用坐标下降算法拟合 MultiTaskElasticNet 模型

    """Fit MultiTaskElasticNet model with coordinate descent.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        数据集.
    y : ndarray of shape (n_samples, n_targets)
        目标值. 如果需要会被转换为与 X 相同的数据类型.

    Returns
    -------
    self : object
        拟合好的估计器.

    Notes
    -----
    坐标下降算法会逐列处理数据，因此如果需要，会自动将 X 转换为 Fortran 连续的 numpy 数组格式.

    为了避免内存重新分配，建议直接使用该格式预先分配内存存储初始数据.
    """
    
    # 需要单独验证这里的参数。
    # 不能设置 multi_output=True，因为这会允许 y 是 csr 格式的。
    check_X_params = dict(
        dtype=[np.float64, np.float32],  # 指定 X 的数据类型
        order="F",                       # 指定 X 的内存布局为 Fortran 连续
        force_writeable=True,            # 强制 X 是可写的
        copy=self.copy_X and self.fit_intercept,  # 根据条件决定是否拷贝 X
    )
    check_y_params = dict(ensure_2d=False, order="F")  # 指定 y 的参数，确保不是二维，内存布局为 Fortran 连续
    X, y = self._validate_data(
        X, y, validate_separately=(check_X_params, check_y_params)  # 验证数据 X 和 y
    )
    check_consistent_length(X, y)  # 检查 X 和 y 的长度是否一致
    y = y.astype(X.dtype)  # 将 y 转换为与 X 相同的数据类型

    if hasattr(self, "l1_ratio"):  # 如果模型有 l1_ratio 属性
        model_str = "ElasticNet"  # 使用 ElasticNet 模型
    else:
        model_str = "Lasso"  # 否则使用 Lasso 模型
    if y.ndim == 1:
        raise ValueError("For mono-task outputs, use %s" % model_str)  # 如果 y 是一维的，抛出错误

    n_samples, n_features = X.shape  # 获取样本数和特征数
    n_targets = y.shape[1]  # 获取目标值的数量

    X, y, X_offset, y_offset, X_scale = _preprocess_data(
        X, y, fit_intercept=self.fit_intercept, copy=False  # 预处理数据 X 和 y
    )

    if not self.warm_start or not hasattr(self, "coef_"):
        self.coef_ = np.zeros(
            (n_targets, n_features), dtype=X.dtype.type, order="F"  # 初始化系数矩阵为零
        )

    l1_reg = self.alpha * self.l1_ratio * n_samples  # 计算 L1 正则化系数
    l2_reg = self.alpha * (1.0 - self.l1_ratio) * n_samples  # 计算 L2 正则化系数

    self.coef_ = np.asfortranarray(self.coef_)  # 将系数矩阵转换为 Fortran 连续的数组

    random = self.selection == "random"  # 如果选择随机的方法

    (
        self.coef_,  # 更新后的系数
        self.dual_gap_,  # 对偶间隙
        self.eps_,  # 精度
        self.n_iter_,  # 迭代次数
    ) = cd_fast.enet_coordinate_descent_multi_task(
        self.coef_,  # 当前系数
        l1_reg,  # L1 正则化系数
        l2_reg,  # L2 正则化系数
        X,  # 数据 X
        y,  # 目标值 y
        self.max_iter,  # 最大迭代次数
        self.tol,  # 公差
        check_random_state(self.random_state),  # 随机数生成器的状态
        random,  # 是否使用随机方法
    )

    # 账户不同目标缩放在此处和 cd_fast 中的不同目标
    self.dual_gap_ /= n_samples  # 对偶间隙除以样本数

    self._set_intercept(X_offset, y_offset, X_scale)  # 设置截距

    # 返回 self 以便链式调用 fit 和 predict
    return self
class MultiTaskLasso(MultiTaskElasticNet):
    """Multi-task Lasso model trained with L1/L2 mixed-norm as regularizer.

    The optimization objective for Lasso is::

        (1 / (2 * n_samples)) * ||Y - XW||^2_Fro + alpha * ||W||_21

    Where::

        ||W||_21 = \\sum_i \\sqrt{\\sum_j w_{ij}^2}

    i.e. the sum of norm of each row.

    Read more in the :ref:`User Guide <multi_task_lasso>`.

    Parameters
    ----------
    alpha : float, default=1.0
        Constant that multiplies the L1/L2 term. Defaults to 1.0.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    copy_X : bool, default=True
        If ``True``, X will be copied; else, it may be overwritten.

    max_iter : int, default=1000
        The maximum number of iterations.

    tol : float, default=1e-4
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.

    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        See :term:`the Glossary <warm_start>`.

    random_state : int, RandomState instance, default=None
        The seed of the pseudo random number generator that selects a random
        feature to update. Used when ``selection`` == 'random'.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    selection : {'cyclic', 'random'}, default='cyclic'
        If set to 'random', a random coefficient is updated every iteration
        rather than looping over features sequentially by default. This
        (setting to 'random') often leads to significantly faster convergence
        especially when tol is higher than 1e-4.

    Attributes
    ----------
    coef_ : ndarray of shape (n_targets, n_features)
        Parameter vector (W in the cost function formula).
        Note that ``coef_`` stores the transpose of ``W``, ``W.T``.

    intercept_ : ndarray of shape (n_targets,)
        Independent term in decision function.

    n_iter_ : int
        Number of iterations run by the coordinate descent solver to reach
        the specified tolerance.

    dual_gap_ : ndarray of shape (n_alphas,)
        The dual gaps at the end of the optimization for each alpha.

    eps_ : float
        The tolerance scaled scaled by the variance of the target `y`.

    sparse_coef_ : sparse matrix of shape (n_features,) or \
            (n_targets, n_features)
        Sparse representation of the `coef_`.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24
    """
    _parameter_constraints: dict = {
        **MultiTaskElasticNet._parameter_constraints,
    }


# 创建一个参数约束字典，继承自MultiTaskElasticNet类的参数约束
_parameter_constraints: dict = {
    **MultiTaskElasticNet._parameter_constraints,
}



    _parameter_constraints.pop("l1_ratio")


# 从参数约束字典中移除键为"l1_ratio"的项
_parameter_constraints.pop("l1_ratio")



    def __init__(
        self,
        alpha=1.0,
        *,
        fit_intercept=True,
        copy_X=True,
        max_iter=1000,
        tol=1e-4,
        warm_start=False,
        random_state=None,
        selection="cyclic",
    ):


# 初始化方法，设置多任务Lasso模型的参数
def __init__(
    self,
    alpha=1.0,                 # 正则化参数
    *,
    fit_intercept=True,         # 是否拟合截距
    copy_X=True,                # 是否复制输入数据
    max_iter=1000,              # 最大迭代次数
    tol=1e-4,                   # 迭代收敛的容差
    warm_start=False,           # 是否热启动
    random_state=None,          # 随机数种子
    selection="cyclic",         # 特征更新的顺序
):


```  
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.copy_X = copy_X
        self.tol = tol
        self.warm_start = warm_start
        self.l1_ratio = 1.0   # 设置L1正则化和L2正则化的比率，默认为1.0，即纯L1正则化
        self.random_state = random_state
        self.selection = selection


# 将初始化方法中的参数赋值给对应的实例变量
self.alpha = alpha                    # 正则化参数
self.fit_intercept = fit_intercept    # 是否拟合截距
self.max_iter = max_iter              # 最大迭代次数
self.copy_X = copy_X                  # 是否复制输入数据
self.tol = tol                        # 迭代收敛的容差
self.warm_start = warm_start          # 是否热启动
self.l1_ratio = 1.0                   # 设置L1正则化和L2正则化的比率，默认为1.0，即纯L1正则化
self.random_state = random_state      # 随机数种子
self.selection = selection            # 特征更新的顺序
class MultiTaskElasticNetCV(RegressorMixin, LinearModelCV):
    """Multi-task L1/L2 ElasticNet with built-in cross-validation.

    See glossary entry for :term:`cross-validation estimator`.

    The optimization objective for MultiTaskElasticNet is::

        (1 / (2 * n_samples)) * ||Y - XW||^Fro_2
        + alpha * l1_ratio * ||W||_21
        + 0.5 * alpha * (1 - l1_ratio) * ||W||_Fro^2

    Where::

        ||W||_21 = \\sum_i \\sqrt{\\sum_j w_{ij}^2}

    i.e. the sum of norm of each row.

    Read more in the :ref:`User Guide <multi_task_elastic_net>`.

    .. versionadded:: 0.15

    Parameters
    ----------
    l1_ratio : float or list of float, default=0.5
        The ElasticNet mixing parameter, with 0 < l1_ratio <= 1.
        For l1_ratio = 1 the penalty is an L1/L2 penalty. For l1_ratio = 0 it
        is an L2 penalty.
        For ``0 < l1_ratio < 1``, the penalty is a combination of L1/L2 and L2.
        This parameter can be a list, in which case the different
        values are tested by cross-validation and the one giving the best
        prediction score is used. Note that a good choice of list of
        values for l1_ratio is often to put more values close to 1
        (i.e. Lasso) and less close to 0 (i.e. Ridge), as in ``[.1, .5, .7,
        .9, .95, .99, 1]``.

    eps : float, default=1e-3
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``.

    n_alphas : int, default=100
        Number of alphas along the regularization path.

    alphas : array-like, default=None
        List of alphas where to compute the models.
        If not provided, set automatically.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    max_iter : int, default=1000
        The maximum number of iterations.

    tol : float, default=1e-4
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.

    cv : int, cross-validation generator or iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross-validation,
        - int, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For int/None inputs, :class:`~sklearn.model_selection.KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.

    copy_X : bool, default=True
        If ``True``, X will be copied; else, it may be overwritten.
    """
    # 是否输出详细信息的标志，可以是布尔值或整数，默认为0
    verbose : bool or int, default=0
        Amount of verbosity.

    # 用于交叉验证的CPU数量，仅在提供多个l1_ratio值时使用
    # 如果未指定，将使用1个CPU，除非在joblib.parallel_backend上下文中
    # -1表示使用所有处理器。有关更多详细信息，请参见术语表中的'n_jobs'
    n_jobs : int, default=None
        Number of CPUs to use during the cross validation. Note that this is
        used only if multiple values for l1_ratio are given.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    # 随机种子，用于选择要更新的随机特征，仅在'selection'=='random'时使用
    # 传递一个整数以在多次函数调用中获得可重现的输出
    # 请参见术语表中的'random_state'
    random_state : int, RandomState instance, default=None
        The seed of the pseudo random number generator that selects a random
        feature to update. Used when ``selection`` == 'random'.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    # 选择更新系数的方式，可以是'cyclic'或'random'，默认为'cyclic'
    # 如果设置为'random'，则每次迭代随机更新一个系数，而不是按顺序遍历特征
    # 当tol大于1e-4时，通常会导致收敛速度显著加快
    selection : {'cyclic', 'random'}, default='cyclic'
        If set to 'random', a random coefficient is updated every iteration
        rather than looping over features sequentially by default. This
        (setting to 'random') often leads to significantly faster convergence
        especially when tol is higher than 1e-4.

    # 拟合后的属性
    Attributes
    ----------

    # 决策函数中的独立项，形状为(n_targets,)
    intercept_ : ndarray of shape (n_targets,)
        Independent term in decision function.

    # 参数向量（成本函数公式中的W），形状为(n_targets, n_features)
    # 注意，coef_存储的是W的转置，即W.T
    coef_ : ndarray of shape (n_targets, n_features)
        Parameter vector (W in the cost function formula).
        Note that ``coef_`` stores the transpose of ``W``, ``W.T``.

    # 由交叉验证选择的正则化量
    alpha_ : float
        The amount of penalization chosen by cross validation.

    # 每个折叠测试集上的均方误差，不同alpha值的变化
    mse_path_ : ndarray of shape (n_alphas, n_folds) or \
                (n_l1_ratio, n_alphas, n_folds)
        Mean square error for the test set on each fold, varying alpha.

    # 用于拟合的alpha网格，对于每个l1_ratio
    alphas_ : ndarray of shape (n_alphas,) or (n_l1_ratio, n_alphas)
        The grid of alphas used for fitting, for each l1_ratio.

    # 通过交叉验证获得的最佳l1_ratio
    l1_ratio_ : float
        Best l1_ratio obtained by cross-validation.

    # 达到指定容差的坐标下降求解器运行的迭代次数
    n_iter_ : int
        Number of iterations run by the coordinate descent solver to reach
        the specified tolerance for the optimal alpha.

    # 最佳alpha的优化结束时的对偶间隙
    dual_gap_ : float
        The dual gap at the end of the optimization for the optimal alpha.

    # 在'fit'期间看到的特征数
    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    # 在'fit'期间看到的特征的名称数组，仅当X具有所有字符串类型的特征名称时定义
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    # 另请参阅
    See Also
    --------
    MultiTaskElasticNet : Multi-task L1/L2 ElasticNet with built-in cross-validation.
    ElasticNetCV : Elastic net model with best model selection by
        cross-validation.
    MultiTaskLassoCV : Multi-task Lasso model trained with L1 norm
        as regularizer and built-in cross-validation.

    # 注意事项
    Notes
    -----
    # 用于拟合模型的算法是坐标下降法
    The algorithm used to fit the model is coordinate descent.

    # 在'fit'中，一旦通过交叉验证找到最佳参数'l1_ratio'和'alpha'，则再次使用整个训练集进行拟合
    In `fit`, once the best parameters `l1_ratio` and `alpha` are found through
    cross-validation, the model is fit again using the entire training set.
    To avoid unnecessary memory duplication the `X` and `y` arguments of the
    `fit` method should be directly passed as Fortran-contiguous numpy arrays.

    Examples
    --------
    >>> from sklearn import linear_model
    >>> clf = linear_model.MultiTaskElasticNetCV(cv=3)
    >>> clf.fit([[0,0], [1, 1], [2, 2]],
    ...         [[0, 0], [1, 1], [2, 2]])
    MultiTaskElasticNetCV(cv=3)
    >>> print(clf.coef_)
    [[0.52875032 0.46958558]
     [0.52875032 0.46958558]]
    >>> print(clf.intercept_)
    [0.00166409 0.00166409]
    """

    # 定义参数约束字典，继承自LinearModelCV的参数约束，并添加'l1_ratio'的约束
    _parameter_constraints: dict = {
        **LinearModelCV._parameter_constraints,
        "l1_ratio": [Interval(Real, 0, 1, closed="both"), "array-like"],
    }
    # 移除不需要的参数约束
    _parameter_constraints.pop("precompute")
    _parameter_constraints.pop("positive")

    # 设置静态方法path为enet_path
    path = staticmethod(enet_path)

    # 初始化方法，设置各种参数
    def __init__(
        self,
        *,
        l1_ratio=0.5,
        eps=1e-3,
        n_alphas=100,
        alphas=None,
        fit_intercept=True,
        max_iter=1000,
        tol=1e-4,
        cv=None,
        copy_X=True,
        verbose=0,
        n_jobs=None,
        random_state=None,
        selection="cyclic",
    ):
        self.l1_ratio = l1_ratio
        self.eps = eps
        self.n_alphas = n_alphas
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.cv = cv
        self.copy_X = copy_X
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.selection = selection

    # 获取评估器
    def _get_estimator(self):
        return MultiTaskElasticNet()

    # 判断是否为多任务
    def _is_multitask(self):
        return True

    # 返回更多标签
    def _more_tags(self):
        return {"multioutput_only": True}

    # 由于LinearModelCV现在支持sample_weight，而MultiTaskElasticNet还不支持，因此需要重写fit方法
    def fit(self, X, y, **params):
        """Fit MultiTaskElasticNet model with coordinate descent.

        Fit is on grid of alphas and best alpha estimated by cross-validation.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples, n_targets)
            Training target variable. Will be cast to X's dtype if necessary.

        **params : dict, default=None
            Parameters to be passed to the CV splitter.

            .. versionadded:: 1.4
                Only available if `enable_metadata_routing=True`,
                which can be set by using
                ``sklearn.set_config(enable_metadata_routing=True)``.
                See :ref:`Metadata Routing User Guide <metadata_routing>` for
                more details.

        Returns
        -------
        self : object
            Returns MultiTaskElasticNet instance.
        """
        # 调用父类的fit方法
        return super().fit(X, y, **params)
class MultiTaskLassoCV(RegressorMixin, LinearModelCV):
    """Multi-task Lasso model trained with L1/L2 mixed-norm as regularizer.

    See glossary entry for :term:`cross-validation estimator`.

    The optimization objective for MultiTaskLasso is::

        (1 / (2 * n_samples)) * ||Y - XW||^Fro_2 + alpha * ||W||_21

    Where::

        ||W||_21 = \\sum_i \\sqrt{\\sum_j w_{ij}^2}

    i.e. the sum of norm of each row.

    Read more in the :ref:`User Guide <multi_task_lasso>`.

    .. versionadded:: 0.15

    Parameters
    ----------
    eps : float, default=1e-3
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``.

    n_alphas : int, default=100
        Number of alphas along the regularization path.

    alphas : array-like, default=None
        List of alphas where to compute the models.
        If not provided, set automatically.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    max_iter : int, default=1000
        The maximum number of iterations.

    tol : float, default=1e-4
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.

    copy_X : bool, default=True
        If ``True``, X will be copied; else, it may be overwritten.

    cv : int, cross-validation generator or iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross-validation,
        - int, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For int/None inputs, :class:`~sklearn.model_selection.KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.

    verbose : bool or int, default=False
        Amount of verbosity.

    n_jobs : int, default=None
        Number of CPUs to use during the cross validation. Note that this is
        used only if multiple values for l1_ratio are given.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    random_state : int, RandomState instance, default=None
        The seed of the pseudo random number generator that selects a random
        feature to update. Used when ``selection`` == 'random'.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
    """
    # `_parameter_constraints` 是一个类属性，它继承自 `LinearModelCV` 的参数约束字典，并进行了一些修改。
    # 移除了键为 "precompute" 和 "positive" 的条目。
    _parameter_constraints: dict = {
        **LinearModelCV._parameter_constraints,
    }
    _parameter_constraints.pop("precompute")
    _parameter_constraints.pop("positive")
    
    # 将 `lasso_path` 方法设置为静态方法，并将其赋值给变量 `path`。
    path = staticmethod(lasso_path)
    # 调用父类的初始化方法，设置 MultiTaskLassoCV 的参数
    def __init__(
        self,
        *,
        eps=1e-3,
        n_alphas=100,
        alphas=None,
        fit_intercept=True,
        max_iter=1000,
        tol=1e-4,
        copy_X=True,
        cv=None,
        verbose=False,
        n_jobs=None,
        random_state=None,
        selection="cyclic",
    ):
        super().__init__(
            eps=eps,
            n_alphas=n_alphas,
            alphas=alphas,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            copy_X=copy_X,
            cv=cv,
            verbose=verbose,
            n_jobs=n_jobs,
            random_state=random_state,
            selection=selection,
        )

    # 返回一个 MultiTaskLasso 实例用于估计
    def _get_estimator(self):
        return MultiTaskLasso()

    # 指示该模型是一个多任务模型
    def _is_multitask(self):
        return True

    # 返回额外的标签，表明该模型仅支持多输出
    def _more_tags(self):
        return {"multioutput_only": True}

    # 由于 LinearModelCV 现在支持 sample_weight 而 MultiTaskElasticNet 还不支持，这个方法是必需的。
    # 使用坐标下降拟合 MultiTaskLasso 模型
    def fit(self, X, y, **params):
        """Fit MultiTaskLasso model with coordinate descent.

        Fit is on grid of alphas and best alpha estimated by cross-validation.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data.
        y : ndarray of shape (n_samples, n_targets)
            Target. Will be cast to X's dtype if necessary.

        **params : dict, default=None
            Parameters to be passed to the CV splitter.

            .. versionadded:: 1.4
                Only available if `enable_metadata_routing=True`,
                which can be set by using
                ``sklearn.set_config(enable_metadata_routing=True)``.
                See :ref:`Metadata Routing User Guide <metadata_routing>` for
                more details.

        Returns
        -------
        self : object
            Returns an instance of fitted model.
        """
        # 调用父类 LinearModelCV 的 fit 方法进行拟合
        return super().fit(X, y, **params)
```