# `D:\src\scipysrc\scikit-learn\sklearn\decomposition\_fastica.py`

```
"""
Python implementation of the fast ICA algorithms.

Reference: Tables 8.3 and 8.4 page 196 in the book:
Independent Component Analysis, by  Hyvarinen et al.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from numbers import Integral, Real

import numpy as np
from scipy import linalg

from ..base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    _fit_context,
)
from ..exceptions import ConvergenceWarning
from ..utils import as_float_array, check_array, check_random_state
from ..utils._param_validation import Interval, Options, StrOptions, validate_params
from ..utils.validation import check_is_fitted

__all__ = ["fastica", "FastICA"]


def _gs_decorrelation(w, W, j):
    """
    Orthonormalize w wrt the first j rows of W.

    Parameters
    ----------
    w : ndarray of shape (n,)
        Array to be orthogonalized

    W : ndarray of shape (p, n)
        Null space definition

    j : int < p
        The no of (from the first) rows of Null space W wrt which w is
        orthogonalized.

    Notes
    -----
    Assumes that W is orthogonal
    w changed in place
    """
    # Orthonormalize w with respect to the first j rows of W
    w -= np.linalg.multi_dot([w, W[:j].T, W[:j]])
    return w


def _sym_decorrelation(W):
    """
    Symmetric decorrelation
    i.e. W <- (W * W.T) ^{-1/2} * W
    """
    # Compute eigenvalues and eigenvectors of W * W.T
    s, u = linalg.eigh(np.dot(W, W.T))
    
    # Clip eigenvalues to avoid sqrt of negative values due to rounding errors
    s = np.clip(s, a_min=np.finfo(W.dtype).tiny, a_max=None)

    # Perform symmetric decorrelation
    return np.linalg.multi_dot([u * (1.0 / np.sqrt(s)), u.T, W])


def _ica_def(X, tol, g, fun_args, max_iter, w_init):
    """
    Deflationary FastICA using fun approx to neg-entropy function

    Used internally by FastICA.
    """

    n_components = w_init.shape[0]
    W = np.zeros((n_components, n_components), dtype=X.dtype)
    n_iter = []

    # j is the index of the extracted component
    for j in range(n_components):
        w = w_init[j, :].copy()
        w /= np.sqrt((w**2).sum())

        for i in range(max_iter):
            # Compute contrast function and its derivative
            gwtx, g_wtx = g(np.dot(w.T, X), fun_args)

            # Update w according to the FastICA algorithm
            w1 = (X * gwtx).mean(axis=1) - g_wtx.mean() * w

            # Decorrelate w1 with the previously estimated components
            _gs_decorrelation(w1, W, j)

            # Normalize w1 to unit length
            w1 /= np.sqrt((w1**2).sum())

            # Check convergence criterion
            lim = np.abs(np.abs((w1 * w).sum()) - 1)
            w = w1
            if lim < tol:
                break

        # Record the number of iterations for convergence
        n_iter.append(i + 1)
        W[j, :] = w

    return W, max(n_iter)


def _ica_par(X, tol, g, fun_args, max_iter, w_init):
    """
    Parallel FastICA.

    Used internally by FastICA --main loop
    """
    # Symmetric decorrelation of the initial guess of mixing matrix
    W = _sym_decorrelation(w_init)
    del w_init
    p_ = float(X.shape[1])
    # 迭代 FastICA 算法，最多执行 max_iter 次
    for ii in range(max_iter):
        # 计算 g 函数在 W*X 上的值及其导数
        gwtx, g_wtx = g(np.dot(W, X), fun_args)
        # 计算 W1，通过对 g(w^T*x) 进行装饰化得到的矩阵
        W1 = _sym_decorrelation(np.dot(gwtx, X.T) / p_ - g_wtx[:, np.newaxis] * W)
        # 清理中间变量
        del gwtx, g_wtx
        
        # 计算 W1 和 W 的乘积的每行向量与单位向量的内积的绝对值与1的差的绝对值的最大值
        # 使用内置的 max 和 abs 函数比 numpy 的更快。
        # np.einsum 允许最低的内存占用。
        # 它比 np.diag(np.dot(W1, W.T)) 更快。
        lim = max(abs(abs(np.einsum("ij,ij->i", W1, W)) - 1))
        
        # 更新 W
        W = W1
        
        # 如果 lim 小于设定的容差 tol，则跳出循环
        if lim < tol:
            break
    else:
        # 如果循环没有通过 break 跳出，则发出警告，提示 FastICA 算法未收敛
        warnings.warn(
            (
                "FastICA did not converge. Consider increasing "
                "tolerance or the maximum number of iterations."
            ),
            ConvergenceWarning,
        )

    # 返回计算得到的 W 矩阵及实际迭代次数 ii + 1
    return W, ii + 1
# Some standard non-linear functions.
# XXX: these should be optimized, as they can be a bottleneck.
def _logcosh(x, fun_args=None):
    # Retrieve alpha from fun_args dictionary or set default value 1.0
    alpha = fun_args.get("alpha", 1.0)  # comment it out?

    # Scale input x by alpha
    x *= alpha
    # Apply hyperbolic tangent function inplace on x
    gx = np.tanh(x, x)  # apply the tanh inplace
    # Initialize g_x array with the same shape and dtype as x
    g_x = np.empty(x.shape[0], dtype=x.dtype)
    # XXX compute in chunks to avoid extra allocation
    # Iterate over elements of gx and compute mean of derivative expression
    for i, gx_i in enumerate(gx):  # please don't vectorize.
        g_x[i] = (alpha * (1 - gx_i**2)).mean()
    # Return gx (result of hyperbolic tangent) and g_x (mean of derivatives)
    return gx, g_x


def _exp(x, fun_args):
    # Compute element-wise exponential function of -(x^2)/2
    exp = np.exp(-(x**2) / 2)
    # Compute gx as element-wise multiplication of x and exp
    gx = x * exp
    # Compute g_x as element-wise multiplication of (1 - x^2) and exp, then take mean along last axis
    g_x = (1 - x**2) * exp
    # Return gx (result of x * exp) and g_x (mean of (1 - x^2) * exp)
    return gx, g_x.mean(axis=-1)


def _cube(x, fun_args):
    # Compute cube of x element-wise and return it
    return x**3, (3 * x**2).mean(axis=-1)


@validate_params(
    {
        "X": ["array-like"],
        "return_X_mean": ["boolean"],
        "compute_sources": ["boolean"],
        "return_n_iter": ["boolean"],
    },
    prefer_skip_nested_validation=False,
)
def fastica(
    X,
    n_components=None,
    *,
    algorithm="parallel",
    whiten="unit-variance",
    fun="logcosh",
    fun_args=None,
    max_iter=200,
    tol=1e-04,
    w_init=None,
    whiten_solver="svd",
    random_state=None,
    return_X_mean=False,
    compute_sources=True,
    return_n_iter=False,
):
    """Perform Fast Independent Component Analysis.

    The implementation is based on [1]_.

    Read more in the :ref:`User Guide <ICA>`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training vector, where `n_samples` is the number of samples and
        `n_features` is the number of features.

    n_components : int, default=None
        Number of components to use. If None is passed, all are used.

    algorithm : {'parallel', 'deflation'}, default='parallel'
        Specify which algorithm to use for FastICA.

    whiten : str or bool, default='unit-variance'
        Specify the whitening strategy to use.

        - If 'arbitrary-variance', a whitening with variance
          arbitrary is used.
        - If 'unit-variance', the whitening matrix is rescaled to ensure that
          each recovered source has unit variance.
        - If False, the data is already considered to be whitened, and no
          whitening is performed.

        .. versionchanged:: 1.3
            The default value of `whiten` changed to 'unit-variance' in 1.3.

    fun : {'logcosh', 'exp', 'cube'} or callable, default='logcosh'
        The functional form of the G function used in the
        approximation to neg-entropy. Could be either 'logcosh', 'exp',
        or 'cube'.
        You can also provide your own function. It should return a tuple
        containing the value of the function, and of its derivative, in the
        point. The derivative should be averaged along its last dimension.
        Example::

            def my_g(x):
                return x ** 3, (3 * x ** 2).mean(axis=-1)
    fun_args : dict, default=None
        # 函数的参数字典，如果为空或为None，并且fun='logcosh'，则fun_args将取值{'alpha' : 1.0}。
        Arguments to send to the functional form.
        If empty or None and if fun='logcosh', fun_args will take value
        {'alpha' : 1.0}.

    max_iter : int, default=200
        # 最大迭代次数，默认为200。
        Maximum number of iterations to perform.

    tol : float, default=1e-4
        # 容差值，一个正的标量，表示未混合矩阵被认为已收敛的容差。
        A positive scalar giving the tolerance at which the
        un-mixing matrix is considered to have converged.

    w_init : ndarray of shape (n_components, n_components), default=None
        # 初始未混合数组。如果`w_init=None`，则使用从正态分布中抽取的值数组。
        Initial un-mixing array. If `w_init=None`, then an array of values
        drawn from a normal distribution is used.

    whiten_solver : {"eigh", "svd"}, default="svd"
        # 用于白化的求解器选择。

        - "svd"在问题退化时数值上更稳定，当`n_samples <= n_features`时通常更快。
        
        - "eigh"在`n_samples >= n_features`时通常更节省内存，并且在`n_samples >= 50 * n_features`时可能更快。

        .. versionadded:: 1.2

        The solver to use for whitening.

        - "svd" is more stable numerically if the problem is degenerate, and
          often faster when `n_samples <= n_features`.

        - "eigh" is generally more memory efficient when
          `n_samples >= n_features`, and can be faster when
          `n_samples >= 50 * n_features`.

    random_state : int, RandomState instance or None, default=None
        # 用于初始化`w_init`的随机状态，当未指定时使用正态分布。传递一个整数以实现多次函数调用的可重复结果。
        Used to initialize ``w_init`` when not specified, with a
        normal distribution. Pass an int, for reproducible results
        across multiple function calls.
        See :term:`Glossary <random_state>`.

    return_X_mean : bool, default=False
        # 如果为True，则也返回X_mean。
        If True, X_mean is returned too.

    compute_sources : bool, default=True
        # 如果为False，则不计算源数据，仅返回旋转矩阵。在处理大数据时可以节省内存。默认为True。
        If False, sources are not computed, but only the rotation matrix.
        This can save memory when working with big data. Defaults to True.

    return_n_iter : bool, default=False
        # 是否返回迭代次数。
        Whether or not to return the number of iterations.

    Returns
    -------
    K : ndarray of shape (n_components, n_features) or None
        # 如果whiten为'True'，则K是预白化矩阵，用于将数据投影到前n_components个主成分上。如果whiten为'False'，则K为'None'。
        If whiten is 'True', K is the pre-whitening matrix that projects data
        onto the first n_components principal components. If whiten is 'False',
        K is 'None'.

    W : ndarray of shape (n_components, n_components)
        # 解混合数据的方阵。如果K不为None，则混合矩阵是矩阵``W K``的伪逆，否则是W的逆。
        The square matrix that unmixes the data after whitening.
        The mixing matrix is the pseudo-inverse of matrix ``W K``
        if K is not None, else it is the inverse of W.

    S : ndarray of shape (n_samples, n_components) or None
        # 估计的源矩阵。
        Estimated source matrix.

    X_mean : ndarray of shape (n_features,)
        # 特征的均值。仅在return_X_mean为True时返回。
        The mean over features. Returned only if return_X_mean is True.

    n_iter : int
        # 如果算法是“deflation”，则n_iter是所有组件中运行的最大迭代次数。否则，它们只是收敛所用的迭代次数。仅在return_n_iter设置为`True`时返回。
        If the algorithm is "deflation", n_iter is the
        maximum number of iterations run across all components. Else
        they are just the number of iterations taken to converge. This is
        returned only when return_n_iter is set to `True`.

    Notes
    -----
    # 数据矩阵X被认为是非高斯（独立）分量的线性组合，即X = AS，其中S的列包含独立分量，A是线性混合矩阵。简言之，ICA试图通过估计一个
    The data matrix X is considered to be a linear combination of
    non-Gaussian (independent) components i.e. X = AS where columns of S
    contain the independent components and A is a linear mixing
    matrix. In short ICA attempts to `un-mix' the data by estimating an
    """
    un-mixing matrix W where ``S = W K X.``
    While FastICA was proposed to estimate as many sources
    as features, it is possible to estimate less by setting
    n_components < n_features. It this case K is not a square matrix
    and the estimated A is the pseudo-inverse of ``W K``.

    This implementation was originally made for data of shape
    [n_features, n_samples]. Now the input is transposed
    before the algorithm is applied. This makes it slightly
    faster for Fortran-ordered input.

    References
    ----------
    .. [1] A. Hyvarinen and E. Oja, "Fast Independent Component Analysis",
           Algorithms and Applications, Neural Networks, 13(4-5), 2000,
           pp. 411-430.

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.decomposition import fastica
    >>> X, _ = load_digits(return_X_y=True)
    >>> K, W, S = fastica(X, n_components=7, random_state=0, whiten='unit-variance')
    >>> K.shape
    (7, 64)
    >>> W.shape
    (7, 7)
    >>> S.shape
    (1797, 7)
    """
    # 创建 FastICA 对象，使用给定的参数初始化
    est = FastICA(
        n_components=n_components,
        algorithm=algorithm,
        whiten=whiten,
        fun=fun,
        fun_args=fun_args,
        max_iter=max_iter,
        tol=tol,
        w_init=w_init,
        whiten_solver=whiten_solver,
        random_state=random_state,
    )
    # 验证参数的有效性
    est._validate_params()
    # 应用 FastICA 算法，计算源信号 S
    S = est._fit_transform(X, compute_sources=compute_sources)

    # 根据 whiten 参数的不同选择性地获取白化矩阵 K 和均值 X_mean
    if est.whiten in ["unit-variance", "arbitrary-variance"]:
        K = est.whitening_
        X_mean = est.mean_
    else:
        K = None
        X_mean = None

    # 返回 K、解混矩阵 W、源信号 S 及可能的均值 X_mean
    returned_values = [K, est._unmixing, S]
    if return_X_mean:
        returned_values.append(X_mean)
    if return_n_iter:
        returned_values.append(est.n_iter_)

    return returned_values
# 定义 FastICA 类，继承 ClassNamePrefixFeaturesOutMixin、TransformerMixin 和 BaseEstimator
class FastICA(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """FastICA: a fast algorithm for Independent Component Analysis.

    The implementation is based on [1]_.

    Read more in the :ref:`User Guide <ICA>`.

    Parameters
    ----------
    n_components : int, default=None
        Number of components to use. If None is passed, all are used.

    algorithm : {'parallel', 'deflation'}, default='parallel'
        Specify which algorithm to use for FastICA.

    whiten : str or bool, default='unit-variance'
        Specify the whitening strategy to use.

        - If 'arbitrary-variance', a whitening with variance
          arbitrary is used.
        - If 'unit-variance', the whitening matrix is rescaled to ensure that
          each recovered source has unit variance.
        - If False, the data is already considered to be whitened, and no
          whitening is performed.

        .. versionchanged:: 1.3
            The default value of `whiten` changed to 'unit-variance' in 1.3.

    fun : {'logcosh', 'exp', 'cube'} or callable, default='logcosh'
        The functional form of the G function used in the
        approximation to neg-entropy. Could be either 'logcosh', 'exp',
        or 'cube'.
        You can also provide your own function. It should return a tuple
        containing the value of the function, and of its derivative, in the
        point. The derivative should be averaged along its last dimension.
        Example::

            def my_g(x):
                return x ** 3, (3 * x ** 2).mean(axis=-1)

    fun_args : dict, default=None
        Arguments to send to the functional form.
        If empty or None and if fun='logcosh', fun_args will take value
        {'alpha' : 1.0}.

    max_iter : int, default=200
        Maximum number of iterations during fit.

    tol : float, default=1e-4
        A positive scalar giving the tolerance at which the
        un-mixing matrix is considered to have converged.

    w_init : array-like of shape (n_components, n_components), default=None
        Initial un-mixing array. If `w_init=None`, then an array of values
        drawn from a normal distribution is used.

    whiten_solver : {"eigh", "svd"}, default="svd"
        The solver to use for whitening.

        - "svd" is more stable numerically if the problem is degenerate, and
          often faster when `n_samples <= n_features`.

        - "eigh" is generally more memory efficient when
          `n_samples >= n_features`, and can be faster when
          `n_samples >= 50 * n_features`.

        .. versionadded:: 1.2

    random_state : int, RandomState instance or None, default=None
        Used to initialize ``w_init`` when not specified, with a
        normal distribution. Pass an int, for reproducible results
        across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        # 独立成分分析后得到的线性操作符，作用于数据以得到独立源。当 `whiten` 为 False 时，等同于解混合矩阵；当 `whiten` 为 True 时，等同于 `np.dot(unmixing_matrix, self.whitening_)`。

    mixing_ : ndarray of shape (n_features, n_components)
        # `components_` 的伪逆矩阵。它是将独立源映射回数据的线性操作符。

    mean_ : ndarray of shape(n_features,)
        # 特征的均值。仅在 `self.whiten` 为 True 时设置。

    n_features_in_ : int
        # 在拟合过程中看到的特征数。

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        # 在 `X` 具有全部为字符串的特征名时定义的特征名列表。

        .. versionadded:: 1.0

    n_iter_ : int
        # 如果算法是 "deflation"，`n_iter` 是所有成分中运行的最大迭代次数。否则，它只是收敛所需的迭代次数。

    whitening_ : ndarray of shape (n_components, n_features)
        # 仅在 `whiten` 为 'True' 时设置。这是预白化矩阵，将数据投影到前 `n_components` 个主成分上。

    See Also
    --------
    PCA : 主成分分析（PCA）。
    IncrementalPCA : 增量主成分分析（IPCA）。
    KernelPCA : 核主成分分析（KPCA）。
    MiniBatchSparsePCA : 小批量稀疏主成分分析。
    SparsePCA : 稀疏主成分分析（SparsePCA）。

    References
    ----------
    .. [1] A. Hyvarinen and E. Oja, Independent Component Analysis:
           Algorithms and Applications, Neural Networks, 13(4-5), 2000,
           pp. 411-430.

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.decomposition import FastICA
    >>> X, _ = load_digits(return_X_y=True)
    >>> transformer = FastICA(n_components=7,
    ...         random_state=0,
    ...         whiten='unit-variance')
    >>> X_transformed = transformer.fit_transform(X)
    >>> X_transformed.shape
    (1797, 7)
    """

    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left"), None],
        "algorithm": [StrOptions({"parallel", "deflation"})],
        "whiten": [
            StrOptions({"arbitrary-variance", "unit-variance"}),
            Options(bool, {False}),
        ],
        "fun": [StrOptions({"logcosh", "exp", "cube"}), callable],
        "fun_args": [dict, None],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0.0, None, closed="left")],
        "w_init": ["array-like", None],
        "whiten_solver": [StrOptions({"eigh", "svd"})],
        "random_state": ["random_state"],
    }
    # 初始化方法，设置独立成分分析器的参数
    def __init__(
        self,
        n_components=None,                    # 独立成分的数量，默认为None
        *,
        algorithm="parallel",                 # 算法选择，默认为"parallel"
        whiten="unit-variance",               # 白化选项，默认为"unit-variance"
        fun="logcosh",                        # 用于计算 ICA 的损失函数，默认为"logcosh"
        fun_args=None,                        # 损失函数的参数，默认为None
        max_iter=200,                         # 最大迭代次数，默认为200
        tol=1e-4,                             # 迭代收敛的容差，默认为1e-4
        w_init=None,                          # 初始化 unmixing 矩阵的方式，默认为None
        whiten_solver="svd",                  # 白化过程中使用的解法，默认为"svd"
        random_state=None,                    # 随机数种子，默认为None
    ):
        super().__init__()                     # 调用父类的初始化方法

    @_fit_context(prefer_skip_nested_validation=True)
    # 拟合模型并从 X 中恢复源数据
    def fit_transform(self, X, y=None):
        """Fit the model and recover the sources from X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Estimated sources obtained by transforming the data with the
            estimated unmixing matrix.
        """
        return self._fit_transform(X, compute_sources=True)

    @_fit_context(prefer_skip_nested_validation=True)
    # 仅拟合模型到 X
    def fit(self, X, y=None):
        """Fit the model to X.

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
        self._fit_transform(X, compute_sources=False)
        return self

    # 对数据 X 进行转换，用估计的 unmixing 矩阵恢复源数据
    def transform(self, X, copy=True):
        """Recover the sources from X (apply the unmixing matrix).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        copy : bool, default=True
            If False, data passed to fit can be overwritten. Defaults to True.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Estimated sources obtained by transforming the data with the
            estimated unmixing matrix.
        """
        check_is_fitted(self)                   # 检查模型是否已经拟合过

        X = self._validate_data(                # 验证数据 X
            X, copy=(copy and self.whiten),     # 如果 copy 为 True 并且需要白化，则复制数据
            dtype=[np.float64, np.float32],     # 数据类型必须为 np.float64 或 np.float32
            reset=False                         # 不重置数据的标志位
        )
        if self.whiten:
            X -= self.mean_                     # 如果需要白化，则减去均值

        return np.dot(X, self.components_.T)    # 返回 X 与估计的 unmixing 矩阵的乘积
    # 定义一个方法，用于将源数据反向转换为混合数据（应用混合矩阵）。
    def inverse_transform(self, X, copy=True):
        """Transform the sources back to the mixed data (apply mixing matrix).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
            Sources, where `n_samples` is the number of samples
            and `n_components` is the number of components.
        copy : bool, default=True
            If False, data passed to fit are overwritten. Defaults to True.
        
        Returns
        -------
        X_new : ndarray of shape (n_samples, n_features)
            Reconstructed data obtained with the mixing matrix.
        """
        
        # 检查当前对象是否已拟合（适用于检查对象是否已经被训练过）
        check_is_fitted(self)
        
        # 检查并转换输入的数据 X，如果需要拷贝且数据已经白化，则进行拷贝操作
        X = check_array(X, copy=(copy and self.whiten), dtype=[np.float64, np.float32])
        
        # 使用混合矩阵对输入数据 X 进行反向转换
        X = np.dot(X, self.mixing_.T)
        
        # 如果数据已经被白化处理，则还原数据的均值
        if self.whiten:
            X += self.mean_
        
        # 返回转换后的数据 X
        return X
    
    @property
    def _n_features_out(self):
        """Number of transformed output features."""
        # 返回组件矩阵的行数，即输出特征的数量
        return self.components_.shape[0]

    # 定义一个方法，返回额外的标签信息，表明该方法能够保留的数据类型
    def _more_tags(self):
        return {"preserves_dtype": [np.float32, np.float64]}
```