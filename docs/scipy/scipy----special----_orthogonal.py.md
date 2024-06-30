# `D:\src\scipysrc\scipy\scipy\special\_orthogonal.py`

```
"""
A collection of functions to find the weights and abscissas for
Gaussian Quadrature.

These calculations are done by finding the eigenvalues of a
tridiagonal matrix whose entries are dependent on the coefficients
in the recursion formula for the orthogonal polynomials with the
corresponding weighting function over the interval.

Many recursion relations for orthogonal polynomials are given:

.. math::

    a1n f_{n+1} (x) = (a2n + a3n x ) f_n (x) - a4n f_{n-1} (x)

The recursion relation of interest is

.. math::

    P_{n+1} (x) = (x - A_n) P_n (x) - B_n P_{n-1} (x)

where :math:`P` has a different normalization than :math:`f`.

The coefficients can be found as:

.. math::

    A_n = -a2n / a3n
    \\qquad
    B_n = ( a4n / a3n \\sqrt{h_n-1 / h_n})^2

where

.. math::

    h_n = \\int_a^b w(x) f_n(x)^2

assume:

.. math::

    P_0 (x) = 1
    \\qquad
    P_{-1} (x) == 0

For the mathematical background, see [golub.welsch-1969-mathcomp]_ and
[abramowitz.stegun-1965]_.

References
----------
.. [golub.welsch-1969-mathcomp]
   Golub, Gene H, and John H Welsch. 1969. Calculation of Gauss
   Quadrature Rules. *Mathematics of Computation* 23, 221-230+s1--s10.

.. [abramowitz.stegun-1965]
   Abramowitz, Milton, and Irene A Stegun. (1965) *Handbook of
   Mathematical Functions: with Formulas, Graphs, and Mathematical
   Tables*. Gaithersburg, MD: National Bureau of Standards.
   http://www.math.sfu.ca/~cbm/aands/

.. [townsend.trogdon.olver-2014]
   Townsend, A. and Trogdon, T. and Olver, S. (2014)
   *Fast computation of Gauss quadrature nodes and
   weights on the whole real line*. :arXiv:`1410.5286`.

.. [townsend.trogdon.olver-2015]
   Townsend, A. and Trogdon, T. and Olver, S. (2015)
   *Fast computation of Gauss quadrature nodes and
   weights on the whole real line*.
   IMA Journal of Numerical Analysis
   :doi:`10.1093/imanum/drv002`.
"""
#
# Author:  Travis Oliphant 2000
# Updated Sep. 2003 (fixed bugs --- tested to be accurate)

# SciPy imports.
# 导入 NumPy 库并使用 np 作为别名
import numpy as np
# 从 NumPy 中导入 exp, inf, pi, sqrt, floor, sin, cos, around,
# hstack, arccos, arange 这些函数和常量
from numpy import (exp, inf, pi, sqrt, floor, sin, cos, around,
                   hstack, arccos, arange)
# 导入 SciPy 中的线性代数模块
from scipy import linalg
# 导入 SciPy 中的特殊函数模块 airy
from scipy.special import airy

# Local imports.
# 导入本地模块 _specfun 和 _ufuncs
# 没有 .pyi 文件指定 _specfun 的类型，所以标记为忽略类型检查
from . import _specfun  # type: ignore
from . import _ufuncs
# 从 _ufuncs 模块中导入 gamma 函数，并用 _gam 作为别名
_gam = _ufuncs.gamma

# Correspondence between new and old names of root functions
# 多项式函数的名称映射表
_polyfuns = ['legendre', 'chebyt', 'chebyu', 'chebyc', 'chebys',
             'jacobi', 'laguerre', 'genlaguerre', 'hermite',
             'hermitenorm', 'gegenbauer', 'sh_legendre', 'sh_chebyt',
             'sh_chebyu', 'sh_jacobi']
# 将一些常见的正交多项式函数名映射到相应的根函数名
_rootfuns_map = {'roots_legendre': 'p_roots',
                 'roots_chebyt': 't_roots',
                 'roots_chebyu': 'u_roots',
                 'roots_chebyc': 'c_roots',
                 'roots_chebys': 's_roots',
                 'roots_jacobi': 'j_roots',
                 'roots_laguerre': 'l_roots',
                 'roots_genlaguerre': 'la_roots',
                 'roots_hermite': 'h_roots',
                 'roots_hermitenorm': 'he_roots',
                 'roots_gegenbauer': 'cg_roots',
                 'roots_sh_legendre': 'ps_roots',
                 'roots_sh_chebyt': 'ts_roots',
                 'roots_sh_chebyu': 'us_roots',
                 'roots_sh_jacobi': 'js_roots'}

# 将所有的多项式函数名和根函数名列入模块的导出列表
__all__ = _polyfuns + list(_rootfuns_map.keys())


class orthopoly1d(np.poly1d):
    """一维正交多项式类，继承自 numpy 的 poly1d 类"""

    def __init__(self, roots, weights=None, hn=1.0, kn=1.0, wfunc=None,
                 limits=None, monic=False, eval_func=None):
        """初始化方法，根据给定参数构造正交多项式对象

        Parameters:
        roots : array_like
            多项式的根
        weights : array_like, optional
            每个根对应的权重
        hn : float, optional
            未使用
        kn : float, optional
            系数缩放因子
        wfunc : function, optional
            权重函数
        limits : tuple, optional
            未使用
        monic : bool, optional
            是否设定为首一多项式
        eval_func : function, optional
            评估函数

        Notes:
        如果 monic=True，则将多项式设定为首一多项式。
        如果 eval_func 存在，则对 eval_func 进行处理，但最终会被丢弃。
        """
        # 计算等效权重
        equiv_weights = [weights[k] / wfunc(roots[k]) for
                         k in range(len(roots))]
        mu = sqrt(hn)
        if monic:
            evf = eval_func
            if evf:
                knn = kn
                def eval_func(x):
                    return evf(x) / knn
            mu = mu / abs(kn)
            kn = 1.0

        # 根据根计算多项式系数，并进行缩放
        poly = np.poly1d(roots, r=True)
        np.poly1d.__init__(self, poly.coeffs * float(kn))

        self.weights = np.array(list(zip(roots, weights, equiv_weights)))
        self.weight_func = wfunc
        self.limits = limits
        self.normcoef = mu

        # 注意: eval_func 在算术运算中将被丢弃
        self._eval_func = eval_func

    def __call__(self, v):
        """重载调用运算符，用于对正交多项式进行函数调用

        Parameters:
        v : array_like or float
            输入值

        Returns:
        array_like or float
            多项式在输入值处的值
        """
        if self._eval_func and not isinstance(v, np.poly1d):
            return self._eval_func(v)
        else:
            return np.poly1d.__call__(self, v)

    def _scale(self, p):
        """缩放多项式对象的系数和评估函数

        Parameters:
        p : float
            缩放因子
        """
        if p == 1.0:
            return
        self._coeffs *= p

        evf = self._eval_func
        if evf:
            self._eval_func = lambda x: evf(x) * p
        self.normcoef *= p


def _gen_roots_and_weights(n, mu0, an_func, bn_func, f, df, symmetrize, mu):
    """生成正交多项式的根和权重

    Returns:
    x : array_like
        正交多项式的根
    w : array_like
        使用相应高斯积分的权重

    Parameters:
    n : int
        多项式的阶数
    an_func : function
        返回 A_n 的函数
    bn_func : function
        返回 sqrt(B_n) 的函数
    f : function
        计算根的函数
    df : function
        计算导数的函数
    symmetrize : bool
        是否对根进行对称处理
    mu : float
        在正交区间上的权重的积分值（即 h_0）
    """
    k = np.arange(n, dtype='d')
    c = np.zeros((2, n))
    c[0,1:] = bn_func(k[1:])
    c[1,:] = an_func(k)
    x = linalg.eigvals_banded(c, overwrite_a_band=True)

    # 使用牛顿法改进根的精度
    y = f(n, x)
    dy = df(n, x)
    x -= y/dy
    # fm and dy may contain very large/small values, so we
    # log-normalize them to maintain precision in the product fm*dy
    
    # 计算 f(n-1, x) 的值，可能包含非常大或非常小的值
    fm = f(n-1, x)
    
    # 对 fm 的绝对值取对数，进行对数归一化
    log_fm = np.log(np.abs(fm))
    
    # 对 dy 的绝对值取对数，进行对数归一化
    log_dy = np.log(np.abs(dy))
    
    # 对 fm 和 dy 进行归一化处理，以维持乘积 fm*dy 的精度
    fm /= np.exp((log_fm.max() + log_fm.min()) / 2.)
    dy /= np.exp((log_dy.max() + log_dy.min()) / 2.)
    
    # 计算权重 w，作为 1 / (fm * dy)
    w = 1.0 / (fm * dy)

    # 如果需要对称化处理
    if symmetrize:
        # 对权重 w 进行对称化处理
        w = (w + w[::-1]) / 2
        
        # 对变量 x 进行对称化处理
        x = (x - x[::-1]) / 2

    # 对权重 w 进行缩放，使其总和为 mu0
    w *= mu0 / w.sum()

    # 如果 mu 存在，则返回 x, w, mu0
    if mu:
        return x, w, mu0
    else:
        # 否则，只返回 x, w
        return x, w
# Jacobi Polynomials 1               P^(alpha,beta)_n(x)

def roots_jacobi(n, alpha, beta, mu=False):
    r"""Gauss-Jacobi quadrature.

    Compute the sample points and weights for Gauss-Jacobi
    quadrature. The sample points are the roots of the nth degree
    Jacobi polynomial, :math:`P^{\alpha, \beta}_n(x)`. These sample
    points and weights correctly integrate polynomials of degree
    :math:`2n - 1` or less over the interval :math:`[-1, 1]` with
    weight function :math:`w(x) = (1 - x)^{\alpha} (1 +
    x)^{\beta}`. See 22.2.1 in [AS]_ for details.

    Parameters
    ----------
    n : int
        quadrature order
    alpha : float
        alpha must be > -1
    beta : float
        beta must be > -1
    mu : bool, optional
        If True, return the sum of the weights, optional.

    Returns
    -------
    x : ndarray
        Sample points
    w : ndarray
        Weights
    mu : float
        Sum of the weights

    See Also
    --------
    scipy.integrate.fixed_quad

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """
    # Convert n to an integer
    m = int(n)
    # Check if n is a positive integer
    if n < 1 or n != m:
        raise ValueError("n must be a positive integer.")
    # Check if alpha and beta are greater than -1
    if alpha <= -1 or beta <= -1:
        raise ValueError("alpha and beta must be greater than -1.")

    # Handle specific cases for alpha and beta
    if alpha == 0.0 and beta == 0.0:
        # Return Legendre roots if alpha and beta are both 0
        return roots_legendre(m, mu)
    if alpha == beta:
        # Return Gegenbauer roots if alpha equals beta
        return roots_gegenbauer(m, alpha+0.5, mu)

    # Compute the initial value of mu0 based on alpha and beta
    if (alpha + beta) <= 1000:
        mu0 = 2.0**(alpha+beta+1) * _ufuncs.beta(alpha+1, beta+1)
    else:
        # Avoid overflows in exponential and beta functions
        mu0 = np.exp((alpha + beta + 1) * np.log(2.0)
                     + _ufuncs.betaln(alpha+1, beta+1))
    
    # Assign alpha and beta to local variables a and b
    a = alpha
    b = beta
    # Define an_func(k) based on alpha and beta
    if a + b == 0.0:
        def an_func(k):
            return np.where(k == 0, (b - a) / (2 + a + b), 0.0)
    else:
        def an_func(k):
            return np.where(
                k == 0,
                (b - a) / (2 + a + b),
                (b * b - a * a) / ((2.0 * k + a + b) * (2.0 * k + a + b + 2))
            )
    
    # Define bn_func(k) based on alpha and beta
    def bn_func(k):
        return (
            2.0 / (2.0 * k + a + b)
            * np.sqrt((k + a) * (k + b) / (2 * k + a + b + 1))
            * np.where(k == 1, 1.0, np.sqrt(k * (k + a + b) / (2.0 * k + a + b - 1)))
        )

    # Define f(n, x) and df(n, x) based on alpha and beta
    def f(n, x):
        return _ufuncs.eval_jacobi(n, a, b, x)
    
    def df(n, x):
        return 0.5 * (n + a + b + 1) * _ufuncs.eval_jacobi(n - 1, a + 1, b + 1, x)
    
    # Return the roots and weights using the general function _gen_roots_and_weights
    return _gen_roots_and_weights(m, mu0, an_func, bn_func, f, df, False, mu)


def jacobi(n, alpha, beta, monic=False):
    r"""Jacobi polynomial.

    Defined to be the solution of
    ```
    """
    Compute the Jacobi polynomial :math:`P_n^{(\alpha, \beta)}` using the recurrence relation.
    
    Parameters
    ----------
    n : int
        Degree of the Jacobi polynomial.
    alpha : float
        Parameter :math:`\alpha` of the Jacobi polynomial, must be greater than -1.
    beta : float
        Parameter :math:`\beta` of the Jacobi polynomial, must be greater than -1.
    monic : bool, optional
        If `True`, scale the leading coefficient to be 1. Default is `False`.
    
    Returns
    -------
    P : orthopoly1d
        Jacobi polynomial :math:`P_n^{(\alpha, \beta)}`.
    
    Notes
    -----
    For fixed :math:`\alpha, \beta`, the Jacobi polynomials :math:`P_n^{(\alpha, \beta)}`
    are orthogonal over the interval [-1, 1] with weight function :math:`(1 - x)^\alpha (1 + x)^\beta`.
    
    References
    ----------
    [AS] Milton Abramowitz and Irene A. Stegun, eds.
    Handbook of Mathematical Functions with Formulas,
    Graphs, and Mathematical Tables. New York: Dover, 1972.
    
    Examples
    --------
    Example usage and verification of the Jacobi polynomial recurrence relation.
    Plot of the Jacobi polynomial :math:`P_5^{(\alpha, -0.5)}` for different values of :math:`\alpha`.
    
    """
    if n < 0:
        raise ValueError("n must be nonnegative.")
    
    # Weight function for Jacobi polynomial
    def wfunc(x):
        return (1 - x) ** alpha * (1 + x) ** beta
    
    if n == 0:
        # Special case for n = 0, returning a constant polynomial
        return orthopoly1d([], [], 1.0, 1.0, wfunc, (-1, 1), monic,
                           eval_func=np.ones_like)
    
    # Compute roots, weights, and other coefficients for Jacobi polynomial
    x, w, mu = roots_jacobi(n, alpha, beta, mu=True)
    ab1 = alpha + beta + 1.0
    hn = 2**ab1 / (2 * n + ab1) * _gam(n + alpha + 1)
    hn *= _gam(n + beta + 1.0) / _gam(n + 1) / _gam(n + ab1)
    kn = _gam(2 * n + ab1) / 2.0**n / _gam(n + 1) / _gam(n + ab1)
    # kn represents the coefficient of the x^n term in the Jacobi polynomial
    
    # Create Jacobi polynomial object
    p = orthopoly1d(x, w, hn, kn, wfunc, (-1, 1), monic,
                    lambda x: _ufuncs.eval_jacobi(n, alpha, beta, x))
    return p
# Jacobi Polynomials shifted         G_n(p,q,x)

def roots_sh_jacobi(n, p1, q1, mu=False):
    """Gauss-Jacobi (shifted) quadrature.

    Compute the sample points and weights for Gauss-Jacobi (shifted)
    quadrature. The sample points are the roots of the nth degree
    shifted Jacobi polynomial, :math:`G^{p,q}_n(x)`. These sample
    points and weights correctly integrate polynomials of degree
    :math:`2n - 1` or less over the interval :math:`[0, 1]` with
    weight function :math:`w(x) = (1 - x)^{p-q} x^{q-1}`. See 22.2.2
    in [AS]_ for details.

    Parameters
    ----------
    n : int
        quadrature order
    p1 : float
        (p1 - q1) must be > -1
    q1 : float
        q1 must be > 0
    mu : bool, optional
        If True, return the sum of the weights, optional.

    Returns
    -------
    x : ndarray
        Sample points
    w : ndarray
        Weights
    mu : float
        Sum of the weights

    See Also
    --------
    scipy.integrate.fixed_quad

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """
    # Check if the parameters are valid
    if (p1-q1) <= -1 or q1 <= 0:
        message = "(p - q) must be greater than -1, and q must be greater than 0."
        raise ValueError(message)
    
    # Compute roots and weights using the non-shifted Jacobi polynomials
    x, w, m = roots_jacobi(n, p1-q1, q1-1, True)
    
    # Shift roots to the interval [0, 1]
    x = (x + 1) / 2
    
    # Scale weights and sum of weights
    scale = 2.0**p1
    w /= scale
    m /= scale
    
    # Return results based on the mu flag
    if mu:
        return x, w, m
    else:
        return x, w


def sh_jacobi(n, p, q, monic=False):
    r"""Shifted Jacobi polynomial.

    Defined by

    .. math::

        G_n^{(p, q)}(x)
          = \binom{2n + p - 1}{n}^{-1}P_n^{(p - q, q - 1)}(2x - 1),

    where :math:`P_n^{(\cdot, \cdot)}` is the nth Jacobi polynomial.

    Parameters
    ----------
    n : int
        Degree of the polynomial.
    p : float
        Parameter, must have :math:`p > q - 1`.
    q : float
        Parameter, must be greater than 0.
    monic : bool, optional
        If `True`, scale the leading coefficient to be 1. Default is
        `False`.

    Returns
    -------
    G : orthopoly1d
        Shifted Jacobi polynomial.

    Notes
    -----
    For fixed :math:`p, q`, the polynomials :math:`G_n^{(p, q)}` are
    orthogonal over :math:`[0, 1]` with weight function :math:`(1 -
    x)^{p - q}x^{q - 1}`.

    """
    # Ensure the degree n is nonnegative
    if n < 0:
        raise ValueError("n must be nonnegative.")
    
    # Define the weight function for the shifted Jacobi polynomials
    def wfunc(x):
        return (1.0 - x) ** (p - q) * x ** (q - 1.0)
    
    # Handle the special case for n == 0
    if n == 0:
        return orthopoly1d([], [], 1.0, 1.0, wfunc, (-1, 1), monic,
                           eval_func=np.ones_like)
    
    # Compute roots and weights for shifted Jacobi polynomials
    n1 = n
    x, w = roots_sh_jacobi(n1, p, q)
    
    # Compute normalization constant hn
    hn = _gam(n + 1) * _gam(n + q) * _gam(n + p) * _gam(n + p - q + 1)
    hn /= (2 * n + p) * (_gam(2 * n + p)**2)
    
    # kn is 1.0 in standard form, keeping monic for compatibility
    kn = 1.0
    # 使用给定的参数创建一个正交多项式对象 pp，使用的参数包括 x (节点), w (权重), hn (n 的值), kn (k 的值), wfunc (权重函数), limits (积分范围), monic (是否是首一多项式),
    # 以及 eval_func (评估函数), 它用于计算雅各比多项式的值。
    pp = orthopoly1d(x, w, hn, kn, wfunc=wfunc, limits=(0, 1), monic=monic,
                     eval_func=lambda x: _ufuncs.eval_sh_jacobi(n, p, q, x))
    # 返回创建的正交多项式对象 pp
    return pp
# Generalized Laguerre               L^(alpha)_n(x)

def roots_genlaguerre(n, alpha, mu=False):
    r"""Gauss-generalized Laguerre quadrature.

    Compute the sample points and weights for Gauss-generalized
    Laguerre quadrature. The sample points are the roots of the nth
    degree generalized Laguerre polynomial, :math:`L^{\alpha}_n(x)`.
    These sample points and weights correctly integrate polynomials of
    degree :math:`2n - 1` or less over the interval :math:`[0,
    \infty]` with weight function :math:`w(x) = x^{\alpha}
    e^{-x}`. See 22.3.9 in [AS]_ for details.

    Parameters
    ----------
    n : int
        quadrature order
    alpha : float
        alpha must be > -1
    mu : bool, optional
        If True, return the sum of the weights, optional.

    Returns
    -------
    x : ndarray
        Sample points
    w : ndarray
        Weights
    mu : float
        Sum of the weights

    See Also
    --------
    scipy.integrate.fixed_quad

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """
    m = int(n)  # Convert n to an integer
    if n < 1 or n != m:  # Check if n is not a positive integer
        raise ValueError("n must be a positive integer.")
    if alpha < -1:  # Check if alpha is less than -1
        raise ValueError("alpha must be greater than -1.")

    mu0 = _ufuncs.gamma(alpha + 1)  # Compute gamma(alpha + 1)

    if m == 1:  # If m (converted n) is 1
        x = np.array([alpha+1.0], 'd')  # Sample point array with alpha + 1.0
        w = np.array([mu0], 'd')  # Weight array with mu0
        if mu:  # If mu is True
            return x, w, mu0  # Return sample points, weights, and mu0
        else:
            return x, w  # Return just sample points and weights

    def an_func(k):  # Define function to compute an coefficients
        return 2 * k + alpha + 1
    def bn_func(k):  # Define function to compute bn coefficients
        return -np.sqrt(k * (k + alpha))
    def f(n, x):  # Define function to evaluate generalized Laguerre polynomial
        return _ufuncs.eval_genlaguerre(n, alpha, x)
    def df(n, x):  # Define function to evaluate derivative of generalized Laguerre polynomial
        return (n * _ufuncs.eval_genlaguerre(n, alpha, x)
                - (n + alpha) * _ufuncs.eval_genlaguerre(n - 1, alpha, x)) / x
    return _gen_roots_and_weights(m, mu0, an_func, bn_func, f, df, False, mu)
    """
    如果 alpha 小于等于 -1，则抛出值错误异常
    如果 n 小于 0，则抛出值错误异常
    如果 n 等于 0，则将 n1 设置为 n + 1，否则设置为 n
    计算根和权重，用于广义拉盖尔多项式
    定义广义拉盖尔多项式的权重函数 wfunc(x)
    如果 n 等于 0，则清空 x 和 w 的值
    计算 hn 和 kn，用于正交多项式对象的创建
    创建并返回正交多项式对象 p，包含以下参数：
        - x: 定义域 (0, inf)
        - w: 权重数组
        - hn: 公式 hn
        - kn: 公式 kn
        - wfunc: 权重函数
        - monic: 布尔值，指示是否是首一多项式
        - lambda x: _ufuncs.eval_genlaguerre(n, alpha, x): 用于计算广义拉盖尔函数的 lambda 函数
    """
    if alpha <= -1:
        raise ValueError("alpha must be > -1")
    if n < 0:
        raise ValueError("n must be nonnegative.")

    if n == 0:
        n1 = n + 1
    else:
        n1 = n
    x, w = roots_genlaguerre(n1, alpha)
    def wfunc(x):
        return exp(-x) * x ** alpha
    if n == 0:
        x, w = [], []
    hn = _gam(n + alpha + 1) / _gam(n + 1)
    kn = (-1)**n / _gam(n + 1)
    p = orthopoly1d(x, w, hn, kn, wfunc, (0, inf), monic,
                    lambda x: _ufuncs.eval_genlaguerre(n, alpha, x))
    return p
# Laguerre                      L_n(x)


def roots_laguerre(n, mu=False):
    r"""Gauss-Laguerre quadrature.

    Compute the sample points and weights for Gauss-Laguerre
    quadrature. The sample points are the roots of the nth degree
    Laguerre polynomial, :math:`L_n(x)`. These sample points and
    weights correctly integrate polynomials of degree :math:`2n - 1`
    or less over the interval :math:`[0, \infty]` with weight function
    :math:`w(x) = e^{-x}`. See 22.2.13 in [AS]_ for details.

    Parameters
    ----------
    n : int
        quadrature order
    mu : bool, optional
        If True, return the sum of the weights, optional.

    Returns
    -------
    x : ndarray
        Sample points
    w : ndarray
        Weights
    mu : float
        Sum of the weights

    See Also
    --------
    scipy.integrate.fixed_quad
    numpy.polynomial.laguerre.laggauss

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """
    return roots_genlaguerre(n, 0.0, mu=mu)


def laguerre(n, monic=False):
    r"""Laguerre polynomial.

    Defined to be the solution of

    .. math::
        x\frac{d^2}{dx^2}L_n + (1 - x)\frac{d}{dx}L_n + nL_n = 0;

    :math:`L_n` is a polynomial of degree :math:`n`.

    Parameters
    ----------
    n : int
        Degree of the polynomial.
    monic : bool, optional
        If `True`, scale the leading coefficient to be 1. Default is
        `False`.

    Returns
    -------
    L : orthopoly1d
        Laguerre Polynomial.

    See Also
    --------
    genlaguerre : Generalized (associated) Laguerre polynomial.

    Notes
    -----
    The polynomials :math:`L_n` are orthogonal over :math:`[0,
    \infty)` with weight function :math:`e^{-x}`.

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    Examples
    --------
    The Laguerre polynomials :math:`L_n` are the special case
    :math:`\alpha = 0` of the generalized Laguerre polynomials
    :math:`L_n^{(\alpha)}`.
    Let's verify it on the interval :math:`[-1, 1]`:

    >>> import numpy as np
    >>> from scipy.special import genlaguerre
    >>> from scipy.special import laguerre
    >>> x = np.arange(-1.0, 1.0, 0.01)
    >>> np.allclose(genlaguerre(3, 0)(x), laguerre(3)(x))
    True

    The polynomials :math:`L_n` also satisfy the recurrence relation:

    .. math::
        (n + 1)L_{n+1}(x) = (2n +1 -x)L_n(x) - nL_{n-1}(x)

    This can be easily checked on :math:`[0, 1]` for :math:`n = 3`:

    >>> x = np.arange(0.0, 1.0, 0.01)
    >>> np.allclose(4 * laguerre(4)(x),
    ...             (7 - x) * laguerre(3)(x) - 3 * laguerre(2)(x))
    True

    This is the plot of the first few Laguerre polynomials :math:`L_n`:
    >>> import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块，用于绘图
    >>> x = np.arange(-1.0, 5.0, 0.01)  # 创建一个从 -1.0 到 5.0 的数组 x，步长为 0.01
    >>> fig, ax = plt.subplots()  # 创建一个新的图形和一个包含单个子图的坐标轴对象
    >>> ax.set_ylim(-5.0, 5.0)  # 设置子图的 y 轴范围为 -5.0 到 5.0
    >>> ax.set_title(r'Laguerre polynomials $L_n$')  # 设置子图的标题为拉盖尔多项式 $L_n$
    >>> for n in np.arange(0, 5):  # 循环迭代 n 从 0 到 4
    ...     ax.plot(x, laguerre(n)(x), label=rf'$L_{n}$')  # 在子图上绘制拉盖尔多项式 $L_n$ 的图像
    >>> plt.legend(loc='best')  # 在图形中添加图例，位置为最佳位置
    >>> plt.show()  # 显示绘制的图形

    """
    if n < 0:
        raise ValueError("n must be nonnegative.")  # 如果 n 小于 0，则抛出值错误异常，要求 n 非负

    if n == 0:
        n1 = n + 1  # 如果 n 等于 0，则将 n1 设为 1
    else:
        n1 = n  # 否则，将 n1 设为 n
    x, w = roots_laguerre(n1)  # 调用 roots_laguerre 函数计算拉盖尔多项式的根 x 和权重 w
    if n == 0:
        x, w = [], []  # 如果 n 等于 0，则将 x 和 w 设置为空列表
    hn = 1.0  # 设置 hn 为 1.0
    kn = (-1)**n / _gam(n + 1)  # 计算 kn，即 (-1)^n / n!，用于拉盖尔多项式的系数
    p = orthopoly1d(x, w, hn, kn, lambda x: exp(-x), (0, inf), monic,
                    lambda x: _ufuncs.eval_laguerre(n, x))  # 调用 orthopoly1d 函数创建一个多项式对象 p
    return p  # 返回创建的多项式对象 p
# Hermite  1                         H_n(x)

# 定义 Gauss-Hermite（物理学家版本）积分方法
def roots_hermite(n, mu=False):
    r"""Gauss-Hermite (physicist's) quadrature.

    Compute the sample points and weights for Gauss-Hermite
    quadrature. The sample points are the roots of the nth degree
    Hermite polynomial, :math:`H_n(x)`. These sample points and
    weights correctly integrate polynomials of degree :math:`2n - 1`
    or less over the interval :math:`[-\infty, \infty]` with weight
    function :math:`w(x) = e^{-x^2}`. See 22.2.14 in [AS]_ for
    details.

    Parameters
    ----------
    n : int
        quadrature order
    mu : bool, optional
        If True, return the sum of the weights, optional.

    Returns
    -------
    x : ndarray
        Sample points
    w : ndarray
        Weights
    mu : float
        Sum of the weights

    See Also
    --------
    scipy.integrate.fixed_quad
    numpy.polynomial.hermite.hermgauss
    roots_hermitenorm

    Notes
    -----
    For small n up to 150 a modified version of the Golub-Welsch
    algorithm is used. Nodes are computed from the eigenvalue
    problem and improved by one step of a Newton iteration.
    The weights are computed from the well-known analytical formula.

    For n larger than 150 an optimal asymptotic algorithm is applied
    which computes nodes and weights in a numerically stable manner.
    The algorithm has linear runtime making computation for very
    large n (several thousand or more) feasible.

    References
    ----------
    .. [townsend.trogdon.olver-2014]
        Townsend, A. and Trogdon, T. and Olver, S. (2014)
        *Fast computation of Gauss quadrature nodes and
        weights on the whole real line*. :arXiv:`1410.5286`.
    .. [townsend.trogdon.olver-2015]
        Townsend, A. and Trogdon, T. and Olver, S. (2015)
        *Fast computation of Gauss quadrature nodes and
        weights on the whole real line*.
        IMA Journal of Numerical Analysis
        :doi:`10.1093/imanum/drv002`.
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """
    # 将 n 转换为整数
    m = int(n)
    # 检查 n 是否为正整数
    if n < 1 or n != m:
        raise ValueError("n must be a positive integer.")

    # 计算 mu0 的初始值
    mu0 = np.sqrt(np.pi)
    
    # 对于小于等于 150 的情况，使用改进的 Golub-Welsch 算法
    if n <= 150:
        # 定义计算 an 的函数
        def an_func(k):
            return 0.0 * k
        # 定义计算 bn 的函数
        def bn_func(k):
            return np.sqrt(k / 2.0)
        # 使用 Hermite 多项式进行计算
        f = _ufuncs.eval_hermite
        # 定义 Hermite 多项式的导数函数
        def df(n, x):
            return 2.0 * n * _ufuncs.eval_hermite(n - 1, x)
        # 调用通用的节点和权重生成函数
        return _gen_roots_and_weights(m, mu0, an_func, bn_func, f, df, True, mu)
    else:
        # 对于大于 150 的情况，使用渐近优化算法计算节点和权重
        nodes, weights = _roots_hermite_asy(m)
        if mu:
            return nodes, weights, mu0
        else:
            return nodes, weights


# 定义 Tricomi 初步猜测辅助函数
def _compute_tauk(n, k, maxit=5):
    """Helper function for Tricomi initial guesses

    For details, see formula 3.1 in lemma 3.1 in the
    original paper.

    Parameters
    ----------
    n : int
        Degree of the Hermite polynomial
    k : int
        Index for computing tau_k
    maxit : int, optional
        Maximum number of iterations (default is 5)

    """
    # 定义函数的参数和返回值
    n : int
        Quadrature order  # 积分阶数
    k : ndarray of type int
        Index of roots :math:`\tau_k` to compute  # 要计算的根的索引
    maxit : int
        Number of Newton maxit performed, the default
        value of 5 is sufficient.  # 进行牛顿迭代的最大次数，默认为5次足够

    Returns
    -------
    tauk : ndarray
        Roots of equation 3.1  # 方程3.1的根

    See Also
    --------
    initial_nodes_a
    roots_hermite_asy
    """
    # 计算参数 a 和 c
    a = n % 2 - 0.5  # 计算参数 a
    c = (4.0*floor(n/2.0) - 4.0*k + 3.0)*pi / (4.0*floor(n/2.0) + 2.0*a + 2.0)  # 计算参数 c
    
    # 定义牛顿迭代所需的函数 f(x) 和 f'(x)
    def f(x):
        return x - sin(x) - c  # 定义方程 f(x)
    
    def df(x):
        return 1.0 - cos(x)  # 定义 f'(x)

    xi = 0.5*pi  # 初始值 xi
    # 进行牛顿迭代
    for i in range(maxit):
        xi = xi - f(xi)/df(xi)  # 更新 xi
    
    return xi  # 返回计算得到的根
    r"""Tricomi initial guesses

    Computes an initial approximation to the square of the `k`-th
    (positive) root :math:`x_k` of the Hermite polynomial :math:`H_n`
    of order :math:`n`. The formula is the one from lemma 3.1 in the
    original paper. The guesses are accurate except in the region
    near :math:`\sqrt{2n + 1}`.

    Parameters
    ----------
    n : int
        Quadrature order
    k : ndarray of type int
        Index of roots to compute

    Returns
    -------
    xksq : ndarray
        Square of the approximate roots

    See Also
    --------
    initial_nodes
    roots_hermite_asy
    """
    tauk = _compute_tauk(n, k)
    sigk = cos(0.5*tauk)**2
    a = n % 2 - 0.5
    nu = 4.0*floor(n/2.0) + 2.0*a + 2.0
    # Initial approximation of Hermite roots (square)
    xksq = nu*sigk - 1.0/(3.0*nu) * (5.0/(4.0*(1.0-sigk)**2) - 1.0/(1.0-sigk) - 0.25)
    return xksq


def _initial_nodes_b(n, k):
    r"""Gatteschi initial guesses

    Computes an initial approximation to the square of the kth
    (positive) root :math:`x_k` of the Hermite polynomial :math:`H_n`
    of order :math:`n`. The formula is the one from lemma 3.2 in the
    original paper. The guesses are accurate in the region just
    below :math:`\sqrt{2n + 1}`.

    Parameters
    ----------
    n : int
        Quadrature order
    k : ndarray of type int
        Index of roots to compute

    Returns
    -------
    xksq : ndarray
        Square of the approximate root

    See Also
    --------
    initial_nodes
    roots_hermite_asy
    """
    a = n % 2 - 0.5
    nu = 4.0*floor(n/2.0) + 2.0*a + 2.0
    # Airy roots by approximation
    ak = _specfun.airyzo(k.max(), 1)[0][::-1]
    # Initial approximation of Hermite roots (square)
    xksq = (nu
            + 2.0**(2.0/3.0) * ak * nu**(1.0/3.0)
            + 1.0/5.0 * 2.0**(4.0/3.0) * ak**2 * nu**(-1.0/3.0)
            + (9.0/140.0 - 12.0/175.0 * ak**3) * nu**(-1.0)
            + (16.0/1575.0 * ak + 92.0/7875.0 * ak**4) * 2.0**(2.0/3.0) * nu**(-5.0/3.0)
            - (15152.0/3031875.0 * ak**5 + 1088.0/121275.0 * ak**2)
              * 2.0**(1.0/3.0) * nu**(-7.0/3.0))
    return xksq


def _initial_nodes(n):
    """Initial guesses for the Hermite roots

    Computes an initial approximation to the non-negative
    roots :math:`x_k` of the Hermite polynomial :math:`H_n`
    of order :math:`n`. The Tricomi and Gatteschi initial
    guesses are used in the region where they are accurate.

    Parameters
    ----------
    n : int
        Quadrature order

    Returns
    -------
    xk : ndarray
        Approximate roots

    See Also
    --------
    roots_hermite_asy
    """
    # Turnover point
    # linear polynomial fit to error of 10, 25, 40, ..., 1000 point rules
    fit = 0.49082003*n - 4.37859653
    turnover = around(fit).astype(int)
    # Compute all approximations
    ia = arange(1, int(floor(n*0.5)+1))
    ib = ia[::-1]
    xasq = _initial_nodes_a(n, ia[:turnover+1])
    # 使用 _initial_nodes_b 函数计算从索引 turnover+1 开始的节点列表，并将结果赋给 xbsq
    xbsq = _initial_nodes_b(n, ib[turnover+1:])
    
    # 合并数组 xasq 和 xbsq，并对其元素进行平方根计算，结果赋给 iv
    iv = sqrt(hstack([xasq, xbsq]))
    
    # 如果 n 为奇数，将 0.0 插入到数组 iv 的开头
    if n % 2 == 1:
        iv = hstack([0.0, iv])
    
    # 返回计算结果 iv
    return iv
    # 计算变量 theta 的正弦值
    st = sin(theta)
    # 计算变量 theta 的余弦值
    ct = cos(theta)
    # 计算变量 mu，用于后续计算
    mu = 2.0*n + 1.0
    # 计算变量 eta，基于 theta 和其三角函数的组合
    eta = 0.5*theta - 0.5*st*ct
    # 计算变量 zeta，基于 eta 的计算结果
    zeta = -(3.0*eta/2.0) ** (2.0/3.0)
    # 计算变量 phi，基于 zeta 和 theta 的正弦值的组合
    phi = (-zeta / st**2) ** (0.25)
    # 定义系数变量 a0 到 a5 和 b0 到 b5，用于后续的系数计算
    a0 = 1.0
    a1 = 0.10416666666666666667
    a2 = 0.08355034722222222222
    a3 = 0.12822657455632716049
    a4 = 0.29184902646414046425
    a5 = 0.88162726744375765242
    b0 = 1.0
    b1 = -0.14583333333333333333
    b2 = -0.09874131944444444444
    b3 = -0.14331205391589506173
    b4 = -0.31722720267841354810
    b5 = -0.94242914795712024914
    # 计算多项式变量 ctp，基于 ct 的幂次
    ctp = ct ** arange(16).reshape((-1,1))
    # 定义多项式系数变量 u0 到 u5 和 v0 到 v4，用于多项式的计算
    u0 = 1.0
    u1 = (1.0*ctp[3,:] - 6.0*ct) / 24.0
    u2 = (-9.0*ctp[4,:] + 249.0*ctp[2,:] + 145.0) / 1152.0
    u3 = (-4042.0*ctp[9,:] + 18189.0*ctp[7,:] - 28287.0*ctp[5,:]
          - 151995.0*ctp[3,:] - 259290.0*ct) / 414720.0
    u4 = (72756.0*ctp[10,:] - 321339.0*ctp[8,:] - 154982.0*ctp[6,:]
          + 50938215.0*ctp[4,:] + 122602962.0*ctp[2,:] + 12773113.0) / 39813120.0
    u5 = (82393456.0*ctp[15,:] - 617950920.0*ctp[13,:] + 1994971575.0*ctp[11,:]
          - 3630137104.0*ctp[9,:] + 4433574213.0*ctp[7,:] - 37370295816.0*ctp[5,:]
          - 119582875013.0*ctp[3,:] - 34009066266.0*ct) / 6688604160.0
    v0 = 1.0
    v1 = (1.0*ctp[3,:] + 6.0*ct) / 24.0
    v2 = (15.0*ctp[4,:] - 327.0*ctp[2,:] - 143.0) / 1152.0
    v3 = (-4042.0*ctp[9,:] + 18189.0*ctp[7,:] - 36387.0*ctp[5,:]
          + 238425.0*ctp[3,:] + 259290.0*ct) / 414720.0
    v4 = (-121260.0*ctp[10,:] + 551733.0*ctp[8,:] - 151958.0*ctp[6,:]
          - 57484425.0*ctp[4,:] - 132752238.0*ctp[2,:] - 12118727) / 39813120.0
    # 计算 v5 的值，这是一个复杂的表达式，用于某种空气力学评估
    v5 = (82393456.0*ctp[15,:] - 617950920.0*ctp[13,:] + 2025529095.0*ctp[11,:]
          - 3750839308.0*ctp[9,:] + 3832454253.0*ctp[7,:] + 35213253348.0*ctp[5,:]
          + 130919230435.0*ctp[3,:] + 34009066266*ct) / 6688604160.0
    
    # 调用 airy 函数，计算 Airy 函数及其导数，但 Bi 和 Bip 未被使用
    Ai, Aip, Bi, Bip = airy(mu**(4.0/6.0) * zeta)
    
    # 计算 U 的前因子
    P = 2.0*sqrt(pi) * mu**(1.0/6.0) * phi
    
    # 计算 U 的各项
    # 参考：https://dlmf.nist.gov/12.10#E42
    phip = phi ** arange(6, 31, 6).reshape((-1,1))
    A0 = b0*u0
    A1 = (b2*u0 + phip[0,:]*b1*u1 + phip[1,:]*b0*u2) / zeta**3
    A2 = (b4*u0 + phip[0,:]*b3*u1 + phip[1,:]*b2*u2 + phip[2,:]*b1*u3
          + phip[3,:]*b0*u4) / zeta**6
    B0 = -(a1*u0 + phip[0,:]*a0*u1) / zeta**2
    B1 = -(a3*u0 + phip[0,:]*a2*u1 + phip[1,:]*a1*u2 + phip[2,:]*a0*u3) / zeta**5
    B2 = -(a5*u0 + phip[0,:]*a4*u1 + phip[1,:]*a3*u2 + phip[2,:]*a2*u3
           + phip[3,:]*a1*u4 + phip[4,:]*a0*u5) / zeta**8
    
    # 计算 U
    # 参考：https://dlmf.nist.gov/12.10#E35
    U = P * (Ai * (A0 + A1/mu**2.0 + A2/mu**4.0) +
             Aip * (B0 + B1/mu**2.0 + B2/mu**4.0) / mu**(8.0/6.0))
    
    # 计算 U 的导数的前因子
    Pd = sqrt(2.0*pi) * mu**(2.0/6.0) / phi
    
    # 计算 U 导数的各项
    # 参考：https://dlmf.nist.gov/12.10#E46
    C0 = -(b1*v0 + phip[0,:]*b0*v1) / zeta
    C1 = -(b3*v0 + phip[0,:]*b2*v1 + phip[1,:]*b1*v2 + phip[2,:]*b0*v3) / zeta**4
    C2 = -(b5*v0 + phip[0,:]*b4*v1 + phip[1,:]*b3*v2 + phip[2,:]*b2*v3
           + phip[3,:]*b1*v4 + phip[4,:]*b0*v5) / zeta**7
    D0 = a0*v0
    D1 = (a2*v0 + phip[0,:]*a1*v1 + phip[1,:]*a0*v2) / zeta**3
    D2 = (a4*v0 + phip[0,:]*a3*v1 + phip[1,:]*a2*v2 + phip[2,:]*a1*v3
          + phip[3,:]*a0*v4) / zeta**6
    
    # 计算 U 的导数
    # 参考：https://dlmf.nist.gov/12.10#E36
    Ud = Pd * (Ai * (C0 + C1/mu**2.0 + C2/mu**4.0) / mu**(4.0/6.0) +
               Aip * (D0 + D1/mu**2.0 + D2/mu**4.0))
    
    # 返回 U 和其导数 Ud
    return U, Ud
def _newton(n, x_initial, maxit=5):
    """Newton iteration for polishing the asymptotic approximation
    to the zeros of the Hermite polynomials.

    Parameters
    ----------
    n : int
        Quadrature order
    x_initial : ndarray
        Initial guesses for the roots
    maxit : int
        Maximal number of Newton iterations.
        The default 5 is sufficient, usually
        only one or two steps are needed.

    Returns
    -------
    nodes : ndarray
        Quadrature nodes
    weights : ndarray
        Quadrature weights

    See Also
    --------
    roots_hermite_asy
    """
    # Variable transformation
    mu = sqrt(2.0*n + 1.0)  # 计算变量变换中的 mu 值
    t = x_initial / mu  # 计算变换后的变量 t
    theta = arccos(t)  # 计算初始角度 theta
    # Newton iteration
    for i in range(maxit):
        u, ud = _pbcf(n, theta)  # 调用 _pbcf 函数计算 u 和 ud
        dtheta = u / (sqrt(2.0) * mu * sin(theta) * ud)  # 计算 Newton 迭代中的角度增量 dtheta
        theta = theta + dtheta  # 更新角度 theta
        if max(abs(dtheta)) < 1e-14:  # 检查角度增量是否足够小，如果是则退出迭代
            break
    # Undo variable transformation
    x = mu * cos(theta)  # 恢复原始变量 x
    # Central node is always zero
    if n % 2 == 1:
        x[0] = 0.0  # 如果 n 是奇数，则中心节点应为零
    # Compute weights
    w = exp(-x**2) / (2.0*ud**2)  # 计算权重 w
    return x, w  # 返回节点 x 和权重 w


def _roots_hermite_asy(n):
    r"""Gauss-Hermite (physicist's) quadrature for large n.

    Computes the sample points and weights for Gauss-Hermite quadrature.
    The sample points are the roots of the nth degree Hermite polynomial,
    :math:`H_n(x)`. These sample points and weights correctly integrate
    polynomials of degree :math:`2n - 1` or less over the interval
    :math:`[-\infty, \infty]` with weight function :math:`f(x) = e^{-x^2}`.

    This method relies on asymptotic expansions which work best for n > 150.
    The algorithm has linear runtime making computation for very large n
    feasible.

    Parameters
    ----------
    n : int
        quadrature order

    Returns
    -------
    nodes : ndarray
        Quadrature nodes
    weights : ndarray
        Quadrature weights

    See Also
    --------
    roots_hermite

    References
    ----------
    .. [townsend.trogdon.olver-2014]
       Townsend, A. and Trogdon, T. and Olver, S. (2014)
       *Fast computation of Gauss quadrature nodes and
       weights on the whole real line*. :arXiv:`1410.5286`.

    .. [townsend.trogdon.olver-2015]
       Townsend, A. and Trogdon, T. and Olver, S. (2015)
       *Fast computation of Gauss quadrature nodes and
       weights on the whole real line*.
       IMA Journal of Numerical Analysis
       :doi:`10.1093/imanum/drv002`.
    """
    iv = _initial_nodes(n)  # 获取初始节点
    nodes, weights = _newton(n, iv)  # 使用牛顿迭代法求解节点和权重
    # Combine with negative parts
    if n % 2 == 0:
        nodes = hstack([-nodes[::-1], nodes])  # 若 n 为偶数，合并节点
        weights = hstack([weights[::-1], weights])  # 若 n 为偶数，合并权重
    else:
        nodes = hstack([-nodes[-1:0:-1], nodes])  # 若 n 为奇数，合并节点
        weights = hstack([weights[-1:0:-1], weights])  # 若 n 为奇数，合并权重
    # Scale weights
    weights *= sqrt(pi) / sum(weights)  # 缩放权重
    return nodes, weights  # 返回合并后的节点和权重
    """
    Compute the Hermite polynomial of degree `n` using the roots_hermite function.
    
    Parameters
    ----------
    n : int
        Degree of the Hermite polynomial. Must be nonnegative.
    monic : bool, optional
        If True, scale the leading coefficient to 1. Default is False.
    
    Returns
    -------
    p : orthopoly1d
        Hermite polynomial of degree `n`.
    
    Raises
    ------
    ValueError
        If `n` is negative.
    
    Notes
    -----
    The Hermite polynomials H_n(x) are orthogonal over the entire real line
    with the weight function e^{-x^2}.
    
    Examples
    --------
    >>> from scipy import special
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    
    >>> p_monic = special.hermite(3, monic=True)
    >>> p_monic
    poly1d([ 1. ,  0. , -1.5,  0. ])
    >>> p_monic(1)
    -0.49999999999999983
    >>> x = np.linspace(-3, 3, 400)
    >>> y = p_monic(x)
    >>> plt.plot(x, y)
    >>> plt.title("Monic Hermite polynomial of degree 3")
    >>> plt.xlabel("x")
    >>> plt.ylabel("H_3(x)")
    >>> plt.show()
    """
    # 如果 n 小于 0，则抛出值错误异常
    if n < 0:
        raise ValueError("n must be nonnegative.")
    
    # 根据 n 的值确定要计算的根数 n1
    if n == 0:
        n1 = n + 1
    else:
        n1 = n
    
    # 调用 roots_hermite 函数计算 Hermite 多项式的根 x 和权重 w
    x, w = roots_hermite(n1)
    
    # 定义权重函数 wfunc(x) = e^{-x^2}
    def wfunc(x):
        return exp(-x * x)
    
    # 如果 n 为 0，则将 x 和 w 初始化为空列表
    if n == 0:
        x, w = [], []
    
    # 计算 Hermite 多项式的前导系数 hn 和 kn
    hn = 2**n * _gam(n + 1) * sqrt(pi)
    kn = 2**n
    
    # 创建 orthopoly1d 对象 p，表示 Hermite 多项式
    p = orthopoly1d(x, w, hn, kn, wfunc, (-inf, inf), monic,
                    lambda x: _ufuncs.eval_hermite(n, x))
    
    # 返回 Hermite 多项式 p
    return p
# Hermite  2                         He_n(x)

# 定义了计算Hermite多项式He_n(x)的函数roots_hermitenorm
def roots_hermitenorm(n, mu=False):
    r"""Gauss-Hermite (statistician's) quadrature.

    Compute the sample points and weights for Gauss-Hermite
    quadrature. The sample points are the roots of the nth degree
    Hermite polynomial, :math:`He_n(x)`. These sample points and
    weights correctly integrate polynomials of degree :math:`2n - 1`
    or less over the interval :math:`[-\infty, \infty]` with weight
    function :math:`w(x) = e^{-x^2/2}`. See 22.2.15 in [AS]_ for more
    details.

    Parameters
    ----------
    n : int
        quadrature order
    mu : bool, optional
        If True, return the sum of the weights, optional.

    Returns
    -------
    x : ndarray
        Sample points
    w : ndarray
        Weights
    mu : float
        Sum of the weights

    See Also
    --------
    scipy.integrate.fixed_quad
    numpy.polynomial.hermite_e.hermegauss

    Notes
    -----
    For small n up to 150 a modified version of the Golub-Welsch
    algorithm is used. Nodes are computed from the eigenvalue
    problem and improved by one step of a Newton iteration.
    The weights are computed from the well-known analytical formula.

    For n larger than 150 an optimal asymptotic algorithm is used
    which computes nodes and weights in a numerical stable manner.
    The algorithm has linear runtime making computation for very
    large n (several thousand or more) feasible.

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """
    m = int(n)
    if n < 1 or n != m:
        raise ValueError("n must be a positive integer.")

    mu0 = np.sqrt(2.0*np.pi)
    # 根据n的大小选择合适的算法进行Hermite多项式根和权重的计算
    if n <= 150:
        # 对于小于等于150的n，使用修改过的Golub-Welsch算法
        def an_func(k):
            return 0.0 * k
        def bn_func(k):
            return np.sqrt(k)
        f = _ufuncs.eval_hermitenorm
        def df(n, x):
            return n * _ufuncs.eval_hermitenorm(n - 1, x)
        return _gen_roots_and_weights(m, mu0, an_func, bn_func, f, df, True, mu)
    else:
        # 对于大于150的n，使用渐近最优算法进行计算
        nodes, weights = _roots_hermite_asy(m)
        # 对节点和权重进行变换
        nodes *= sqrt(2)
        weights *= sqrt(2)
        if mu:
            return nodes, weights, mu0
        else:
            return nodes, weights


# 定义了计算标准化Hermite多项式He_n(x)的函数hermitenorm
def hermitenorm(n, monic=False):
    r"""Normalized (probabilist's) Hermite polynomial.

    Defined by

    .. math::

        He_n(x) = (-1)^ne^{x^2/2}\frac{d^n}{dx^n}e^{-x^2/2};

    :math:`He_n` is a polynomial of degree :math:`n`.

    Parameters
    ----------
    n : int
        Degree of the polynomial.
    monic : bool, optional
        If `True`, scale the leading coefficient to be 1. Default is
        `False`.

    Returns
    -------
    He : orthopoly1d
        Hermite polynomial.

    Notes
    -----

    The polynomials :math:`He_n` are orthogonal over :math:`(-\infty,
    ``
    """
    Calculate the nth physicist's Hermite polynomial.

    """
    # 如果 n 小于 0，则抛出值错误异常
    if n < 0:
        raise ValueError("n must be nonnegative.")

    # 如果 n 等于 0，则设置 n1 为 n + 1，否则设置 n1 为 n
    if n == 0:
        n1 = n + 1
    else:
        n1 = n

    # 调用 roots_hermitenorm 函数计算 Hermite 多项式的根 x 和权重 w
    x, w = roots_hermitenorm(n1)

    # 定义权重函数 wfunc(x) = exp(-x * x / 2.0)
    def wfunc(x):
        return exp(-x * x / 2.0)

    # 如果 n 等于 0，则将 x 和 w 设置为空列表
    if n == 0:
        x, w = [], []

    # 计算 Hermite 多项式的归一化系数 hn = sqrt(2 * pi) * Gamma(n + 1)
    hn = sqrt(2 * pi) * _gam(n + 1)
    
    # 设置 kn = 1.0
    kn = 1.0

    # 调用 orthopoly1d 函数创建 Hermite 多项式对象 p，传入参数：
    # - 根 x 和权重 w
    # - 归一化系数 hn
    # - kn
    # - 权重函数 wfunc
    # - 积分范围为 (-inf, inf)
    # - monic=True 或 False（取决于变量 monic 的值）
    # - eval_func 函数，使用 lambda 表达式调用 eval_hermitenorm 函数计算 Hermite 多项式的值
    p = orthopoly1d(x, w, hn, kn, wfunc=wfunc, limits=(-inf, inf), monic=monic,
                    eval_func=lambda x: _ufuncs.eval_hermitenorm(n, x))

    # 返回 Hermite 多项式对象 p
    return p
# Ultraspherical (Gegenbauer) polynomial C^(alpha)_n(x)的根和权重计算函数
def roots_gegenbauer(n, alpha, mu=False):
    r"""Gauss-Gegenbauer quadrature.

    Compute the sample points and weights for Gauss-Gegenbauer
    quadrature. The sample points are the roots of the nth degree
    Gegenbauer polynomial, C^{\alpha}_n(x). These sample
    points and weights correctly integrate polynomials of degree
    2n - 1 or less over the interval [-1, 1] with
    weight function w(x) = (1 - x^2)^{\alpha - 1/2}. See
    22.2.3 in [AS]_ for more details.

    Parameters
    ----------
    n : int
        quadrature order
    alpha : float
        alpha must be > -0.5
    mu : bool, optional
        If True, return the sum of the weights, optional.

    Returns
    -------
    x : ndarray
        Sample points
    w : ndarray
        Weights
    mu : float
        Sum of the weights

    See Also
    --------
    scipy.integrate.fixed_quad

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """
    m = int(n)  # 将n转换为整数m
    if n < 1 or n != m:
        raise ValueError("n must be a positive integer.")  # 如果n不是正整数，抛出数值错误异常
    if alpha < -0.5:
        raise ValueError("alpha must be greater than -0.5.")  # 如果alpha小于-0.5，抛出数值错误异常
    elif alpha == 0.0:
        # 当alpha等于0时，C(n,0,x)在整个区间上等于0，根据alpha->0时的特性，返回Chebyshev多项式的根
        return roots_chebyt(n, mu)

    if alpha <= 170:
        # 计算mu0，当alpha较小时采用的公式
        mu0 = (np.sqrt(np.pi) * _ufuncs.gamma(alpha + 0.5)) \
              / _ufuncs.gamma(alpha + 1)
    else:
        # 当alpha较大时，使用alpha的逆数进行Taylor级数展开，利用Horner方法最小化计算并提高精度
        inv_alpha = 1. / alpha
        coeffs = np.array([0.000207186, -0.00152206, -0.000640869,
                           0.00488281, 0.0078125, -0.125, 1.])
        mu0 = coeffs[0]
        for term in range(1, len(coeffs)):
            mu0 = mu0 * inv_alpha + coeffs[term]
        mu0 = mu0 * np.sqrt(np.pi / alpha)

    # 定义an_func和bn_func函数
    def an_func(k):
        return 0.0 * k

    def bn_func(k):
        return np.sqrt(k * (k + 2 * alpha - 1) / (4 * (k + alpha) * (k + alpha - 1)))

    # 定义f和df函数，用于计算Gegenbauer多项式及其导数
    def f(n, x):
        return _ufuncs.eval_gegenbauer(n, alpha, x)

    def df(n, x):
        return (
            -n * x * _ufuncs.eval_gegenbauer(n, alpha, x)
            + (n + 2 * alpha - 1) * _ufuncs.eval_gegenbauer(n - 1, alpha, x)
        ) / (1 - x ** 2)

    # 调用_gen_roots_and_weights函数计算Gegenbauer多项式的根和权重
    return _gen_roots_and_weights(m, mu0, an_func, bn_func, f, df, True, mu)
    Defined to be the solution of

    .. math::
        (1 - x^2)\frac{d^2}{dx^2}C_n^{(\alpha)}
          - (2\alpha + 1)x\frac{d}{dx}C_n^{(\alpha)}
          + n(n + 2\alpha)C_n^{(\alpha)} = 0

    for :math:`\alpha > -1/2`; :math:`C_n^{(\alpha)}` is a polynomial
    of degree :math:`n`.

    Parameters
    ----------
    n : int
        Degree of the polynomial.
    alpha : float
        Parameter, must be greater than -0.5.
    monic : bool, optional
        If `True`, scale the leading coefficient to be 1. Default is
        `False`.

    Returns
    -------
    C : orthopoly1d
        Gegenbauer polynomial.

    Notes
    -----
    The polynomials :math:`C_n^{(\alpha)}` are orthogonal over
    :math:`[-1,1]` with weight function :math:`(1 - x^2)^{(\alpha -
    1/2)}`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import special
    >>> import matplotlib.pyplot as plt

    We can initialize a variable ``p`` as a Gegenbauer polynomial using the
    `gegenbauer` function and evaluate at a point ``x = 1``.

    >>> p = special.gegenbauer(3, 0.5, monic=False)
    >>> p
    poly1d([ 2.5,  0. , -1.5,  0. ])
    >>> p(1)
    1.0

    To evaluate ``p`` at various points ``x`` in the interval ``(-3, 3)``,
    simply pass an array ``x`` to ``p`` as follows:

    >>> x = np.linspace(-3, 3, 400)
    >>> y = p(x)

    We can then visualize ``x, y`` using `matplotlib.pyplot`.

    >>> fig, ax = plt.subplots()
    >>> ax.plot(x, y)
    >>> ax.set_title("Gegenbauer (ultraspherical) polynomial of degree 3")
    >>> ax.set_xlabel("x")
    >>> ax.set_ylabel("G_3(x)")
    >>> plt.show()

    """
    # 使用 Jacobi 多项式来生成基础的 Gegenbauer 多项式
    base = jacobi(n, alpha - 0.5, alpha - 0.5, monic=monic)
    
    if monic:
        # 如果 monic=True，则直接返回基础的 Gegenbauer 多项式
        return base
    
    # 否则，按照 Abrahmowitz 和 Stegan 22.5.20 的公式计算缩放因子
    factor = (_gam(2*alpha + n) * _gam(alpha + 0.5) /
              _gam(2*alpha) / _gam(alpha + 0.5 + n))
    
    # 缩放基础的 Gegenbauer 多项式
    base._scale(factor)
    
    # 设置 _eval_func 属性，使得可以使用特定函数来评估 Gegenbauer 多项式
    base.__dict__['_eval_func'] = lambda x: _ufuncs.eval_gegenbauer(float(n),
                                                                    alpha, x)
    
    return base
# Chebyshev of the first kind: T_n(x) =
#     n! sqrt(pi) / _gam(n+1./2)* P^(-1/2,-1/2)_n(x)
# Computed anew.

def roots_chebyt(n, mu=False):
    r"""Gauss-Chebyshev (first kind) quadrature.

    Computes the sample points and weights for Gauss-Chebyshev
    quadrature. The sample points are the roots of the nth degree
    Chebyshev polynomial of the first kind, T_n(x). These
    sample points and weights correctly integrate polynomials of
    degree 2n - 1 or less over the interval [-1, 1]
    with weight function w(x) = 1/sqrt(1 - x^2). See 22.2.4
    in [AS]_ for more details.

    Parameters
    ----------
    n : int
        quadrature order
    mu : bool, optional
        If True, return the sum of the weights, optional.

    Returns
    -------
    x : ndarray
        Sample points
    w : ndarray
        Weights
    mu : float
        Sum of the weights

    See Also
    --------
    scipy.integrate.fixed_quad
    numpy.polynomial.chebyshev.chebgauss

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """
    m = int(n)
    # Check if n is a positive integer
    if n < 1 or n != m:
        raise ValueError('n must be a positive integer.')
    # Compute the sample points using a predefined function _ufuncs._sinpi
    x = _ufuncs._sinpi(np.arange(-m + 1, m, 2) / (2*m))
    # Initialize weights with equal values
    w = np.full_like(x, pi/m)
    # Optionally return the sum of weights if mu is True
    if mu:
        return x, w, pi
    else:
        return x, w


def chebyt(n, monic=False):
    r"""Chebyshev polynomial of the first kind.

    Defined to be the solution of

    .. math::
        (1 - x^2)\frac{d^2}{dx^2}T_n - x\frac{d}{dx}T_n + n^2T_n = 0;

    T_n is a polynomial of degree n.

    Parameters
    ----------
    n : int
        Degree of the polynomial.
    monic : bool, optional
        If True, scale the leading coefficient to be 1. Default is
        False.

    Returns
    -------
    T : orthopoly1d
        Chebyshev polynomial of the first kind.

    See Also
    --------
    chebyu : Chebyshev polynomial of the second kind.

    Notes
    -----
    The polynomials T_n are orthogonal over [-1, 1]
    with weight function (1 - x^2)^(-1/2).

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    Examples
    --------
    Chebyshev polynomials of the first kind of order n can
    be obtained as the determinant of specific n x n
    matrices. As an example we can check how the points obtained from
    the determinant of the following 3 x 3 matrix
    lay exactly on T_3:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.linalg import det
    >>> from scipy.special import chebyt
    >>> x = np.arange(-1.0, 1.0, 0.01)
    >>> fig, ax = plt.subplots()
    if n < 0:
        raise ValueError("n must be nonnegative.")


    # 检查参数 n 是否小于 0，如果是则抛出值错误异常
    if n < 0:
        raise ValueError("n must be nonnegative.")



    def wfunc(x):
        return 1.0 / sqrt(1 - x * x)


    # 定义权重函数 wfunc，用于计算正交多项式的权重
    def wfunc(x):
        return 1.0 / sqrt(1 - x * x)



    if n == 0:
        return orthopoly1d([], [], pi, 1.0, wfunc, (-1, 1), monic,
                           lambda x: _ufuncs.eval_chebyt(n, x))


    # 如果 n 等于 0，则返回特殊处理的正交多项式对象
    if n == 0:
        return orthopoly1d([], [], pi, 1.0, wfunc, (-1, 1), monic,
                           lambda x: _ufuncs.eval_chebyt(n, x))



    n1 = n
    x, w, mu = roots_chebyt(n1, mu=True)
    hn = pi / 2
    kn = 2**(n - 1)
    p = orthopoly1d(x, w, hn, kn, wfunc, (-1, 1), monic,
                    lambda x: _ufuncs.eval_chebyt(n, x))


    # 计算 Chebyshev 多项式的正交多项式对象 p
    n1 = n
    x, w, mu = roots_chebyt(n1, mu=True)  # 计算 Chebyshev 多项式的根和权重
    hn = pi / 2  # 设定正交区间的半宽度
    kn = 2**(n - 1)  # 计算常数系数
    p = orthopoly1d(x, w, hn, kn, wfunc, (-1, 1), monic,
                    lambda x: _ufuncs.eval_chebyt(n, x))  # 创建正交多项式对象



    return p


    # 返回计算得到的 Chebyshev 多项式对象 p
    return p
# Chebyshev of the second kind
#    U_n(x) = (n+1)! sqrt(pi) / (2*_gam(n+3./2)) * P^(1/2,1/2)_n(x)

def roots_chebyu(n, mu=False):
    r"""Gauss-Chebyshev (second kind) quadrature.

    Computes the sample points and weights for Gauss-Chebyshev
    quadrature. The sample points are the roots of the nth degree
    Chebyshev polynomial of the second kind, U_n(x). These
    sample points and weights correctly integrate polynomials of
    degree 2n - 1 or less over the interval [-1, 1]
    with weight function w(x) = sqrt(1 - x^2). See 22.2.5 in
    [AS]_ for details.

    Parameters
    ----------
    n : int
        quadrature order
    mu : bool, optional
        If True, return the sum of the weights, optional.

    Returns
    -------
    x : ndarray
        Sample points
    w : ndarray
        Weights
    mu : float
        Sum of the weights

    See Also
    --------
    scipy.integrate.fixed_quad

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """
    m = int(n)
    if n < 1 or n != m:
        raise ValueError('n must be a positive integer.')
    t = np.arange(m, 0, -1) * pi / (m + 1)  # Compute the points t_j = j*pi/(m+1)
    x = np.cos(t)  # Compute the sample points x_j = cos(t_j)
    w = pi * np.sin(t)**2 / (m + 1)  # Compute the weights w_j = pi*sin(t_j)^2/(m+1)
    if mu:
        return x, w, pi / 2  # Return sample points, weights, and the sum of weights
    else:
        return x, w  # Return sample points and weights


def chebyu(n, monic=False):
    r"""Chebyshev polynomial of the second kind.

    Defined to be the solution of

    .. math::
        (1 - x^2)\frac{d^2}{dx^2}U_n - 3x\frac{d}{dx}U_n
          + n(n + 2)U_n = 0;

    U_n is a polynomial of degree n.

    Parameters
    ----------
    n : int
        Degree of the polynomial.
    monic : bool, optional
        If `True`, scale the leading coefficient to be 1. Default is
        `False`.

    Returns
    -------
    U : orthopoly1d
        Chebyshev polynomial of the second kind.

    See Also
    --------
    chebyt : Chebyshev polynomial of the first kind.

    Notes
    -----
    The polynomials U_n are orthogonal over [-1, 1]
    with weight function (1 - x^2)^{1/2}.

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    Examples
    --------
    Chebyshev polynomials of the second kind of order n can
    be obtained as the determinant of specific n x n
    matrices. As an example we can check how the points obtained from
    the determinant of the following 3 x 3 matrix
    lay exactly on U_3:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.linalg import det
    >>> from scipy.special import chebyu
    >>> x = np.arange(-1.0, 1.0, 0.01)
    >>> fig, ax = plt.subplots()
    >>> ax.set_ylim(-2.0, 2.0)
    # 计算使用雅可比多项式的基本多项式
    base = jacobi(n, 0.5, 0.5, monic=monic)
    # 如果 monic 参数为真，直接返回基本多项式
    if monic:
        return base
    # 计算缩放因子，用于非monic多项式的标准化
    factor = sqrt(pi) / 2.0 * _gam(n + 2) / _gam(n + 1.5)
    # 对基本多项式进行缩放操作
    base._scale(factor)
    # 返回缩放后的多项式
    return base
# Chebyshev of the first kind        C_n(x)

def roots_chebyc(n, mu=False):
    r"""Gauss-Chebyshev (first kind) quadrature.

    Compute the sample points and weights for Gauss-Chebyshev
    quadrature. The sample points are the roots of the nth degree
    Chebyshev polynomial of the first kind, :math:`C_n(x)`. These
    sample points and weights correctly integrate polynomials of
    degree :math:`2n - 1` or less over the interval :math:`[-2, 2]`
    with weight function :math:`w(x) = 1 / \sqrt{1 - (x/2)^2}`. See
    22.2.6 in [AS]_ for more details.

    Parameters
    ----------
    n : int
        quadrature order
    mu : bool, optional
        If True, return the sum of the weights, optional.

    Returns
    -------
    x : ndarray
        Sample points
    w : ndarray
        Weights
    mu : float
        Sum of the weights

    See Also
    --------
    scipy.integrate.fixed_quad

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """
    # Compute roots and weights for the first kind of Chebyshev quadrature
    x, w, m = roots_chebyt(n, True)
    # Scale the sample points and weights to the interval [-2, 2]
    x *= 2
    w *= 2
    m *= 2
    if mu:
        # Return the sample points, weights, and sum of weights if requested
        return x, w, m
    else:
        # Return only the sample points and weights
        return x, w


def chebyc(n, monic=False):
    r"""Chebyshev polynomial of the first kind on :math:`[-2, 2]`.

    Defined as :math:`C_n(x) = 2T_n(x/2)`, where :math:`T_n` is the
    nth Chebychev polynomial of the first kind.

    Parameters
    ----------
    n : int
        Degree of the polynomial.
    monic : bool, optional
        If `True`, scale the leading coefficient to be 1. Default is
        `False`.

    Returns
    -------
    C : orthopoly1d
        Chebyshev polynomial of the first kind on :math:`[-2, 2]`.

    See Also
    --------
    chebyt : Chebyshev polynomial of the first kind.

    Notes
    -----
    The polynomials :math:`C_n(x)` are orthogonal over :math:`[-2, 2]`
    with weight function :math:`1/\sqrt{1 - (x/2)^2}`.

    References
    ----------
    .. [1] Abramowitz and Stegun, "Handbook of Mathematical Functions"
           Section 22. National Bureau of Standards, 1972.

    """
    # Check if n is nonnegative
    if n < 0:
        raise ValueError("n must be nonnegative.")

    # Adjust degree for polynomial computation
    if n == 0:
        n1 = n + 1
    else:
        n1 = n
    # Compute roots and weights for the first kind of Chebyshev polynomial
    x, w = roots_chebyc(n1)
    # If n is 0, set x and w to empty lists
    if n == 0:
        x, w = [], []
    # Compute additional constants hn and kn
    hn = 4 * pi * ((n == 0) + 1)
    kn = 1.0
    # Create an orthogonal polynomial instance
    p = orthopoly1d(x, w, hn, kn,
                    wfunc=lambda x: 1.0 / sqrt(1 - x * x / 4.0),
                    limits=(-2, 2), monic=monic)
    # If monic is False, scale the polynomial by 2 and adjust its evaluation function
    if not monic:
        p._scale(2.0 / p(2))
        p.__dict__['_eval_func'] = lambda x: _ufuncs.eval_chebyc(n, x)
    # Return the Chebyshev polynomial instance
    return p

# Chebyshev of the second kind       S_n(x)

def roots_chebys(n, mu=False):
    r"""Gauss-Chebyshev (second kind) quadrature.

    Compute the sample points and weights for Gauss-Chebyshev
    quadrature. The sample points are the roots of the nth degree
    # 使用切比雪夫多项式的第二类根和权重计算积分点和权重。
    # 这些样本点和权重可以正确地在区间 [-2, 2] 上积分度小于等于 2n - 1 的多项式，
    # 权重函数为 w(x) = sqrt(1 - (x/2)^2)。详见 [AS]_ 中的 22.2.7 节。
    #
    # Parameters 参数说明：
    # n : int
    #     积分的阶数
    # mu : bool, optional
    #     如果为 True，则返回权重的总和，可选的。
    #
    # Returns 返回值：
    # x : ndarray
    #     样本点
    # w : ndarray
    #     权重
    # mu : float
    #     权重的总和
    #
    # See Also 相关链接：
    # --------
    # scipy.integrate.fixed_quad
    #
    # References 参考文献：
    # ----------
    # .. [AS] Milton Abramowitz 和 Irene A. Stegun 编辑
    #    Handbook of Mathematical Functions with Formulas,
    #    Graphs, and Mathematical Tables. New York: Dover, 1972.

    # 调用 roots_chebyu 函数计算切比雪夫多项式的第二类的根和权重，同时返回权重的总和。
    x, w, m = roots_chebyu(n, True)
    # 将样本点乘以 2，因为积分区间变成了 [-2, 2]
    x *= 2
    # 将权重乘以 2，因为权重函数中包含了一个因子 1/2
    w *= 2
    # 将权重的总和乘以 2，同样是因为权重函数中的因子 1/2
    m *= 2
    # 如果需要返回权重的总和，则返回 x, w, m；否则，仅返回 x, w。
    if mu:
        return x, w, m
    else:
        return x, w
# 第一个函数定义了Chebyshev多项式的第二类，定义在区间[-2, 2]上。
# 其中，S_n(x) = U_n(x/2)，其中U_n是第二类Chebyshev多项式。
def chebys(n, monic=False):
    r"""Chebyshev polynomial of the second kind on :math:`[-2, 2]`.

    Defined as :math:`S_n(x) = U_n(x/2)` where :math:`U_n` is the
    nth Chebychev polynomial of the second kind.

    Parameters
    ----------
    n : int
        Degree of the polynomial.
    monic : bool, optional
        If `True`, scale the leading coefficient to be 1. Default is
        `False`.

    Returns
    -------
    S : orthopoly1d
        Chebyshev polynomial of the second kind on :math:`[-2, 2]`.

    See Also
    --------
    chebyu : Chebyshev polynomial of the second kind

    Notes
    -----
    The polynomials :math:`S_n(x)` are orthogonal over :math:`[-2, 2]`
    with weight function :math:`\sqrt{1 - (x/2)}^2`.

    References
    ----------
    .. [1] Abramowitz and Stegun, "Handbook of Mathematical Functions"
           Section 22. National Bureau of Standards, 1972.

    """
    # 如果n小于0，则抛出ValueError异常
    if n < 0:
        raise ValueError("n must be nonnegative.")

    # 如果n为0，将n1设置为1；否则设置为n
    if n == 0:
        n1 = n + 1
    else:
        n1 = n

    # 调用roots_chebys函数获取Chebyshev多项式的根和权重
    x, w = roots_chebys(n1)

    # 如果n为0，清空x和w
    if n == 0:
        x, w = [], []

    # 设置hn为π，kn为1.0
    hn = pi
    kn = 1.0

    # 创建orthopoly1d对象p，使用给定的根、权重、hn、kn、权重函数、限制和monic参数
    p = orthopoly1d(x, w, hn, kn,
                    wfunc=lambda x: sqrt(1 - x * x / 4.0),
                    limits=(-2, 2), monic=monic)

    # 如果monic为False，调整p的尺度使得其系数的尺度正确，并修改_eval_func属性为lambda表达式
    if not monic:
        factor = (n + 1.0) / p(2)
        p._scale(factor)
        p.__dict__['_eval_func'] = lambda x: _ufuncs.eval_chebys(n, x)

    # 返回Chebyshev多项式对象p
    return p


# Shifted Chebyshev of the first kind     T^*_n(x)


# 第二个函数定义了偏移的第一类Chebyshev多项式，即T^*_n(x)
def roots_sh_chebyt(n, mu=False):
    r"""Gauss-Chebyshev (first kind, shifted) quadrature.

    Compute the sample points and weights for Gauss-Chebyshev
    quadrature. The sample points are the roots of the nth degree
    shifted Chebyshev polynomial of the first kind, :math:`T_n(x)`.
    These sample points and weights correctly integrate polynomials of
    degree :math:`2n - 1` or less over the interval :math:`[0, 1]`
    with weight function :math:`w(x) = 1/\sqrt{x - x^2}`. See 22.2.8
    in [AS]_ for more details.

    Parameters
    ----------
    n : int
        quadrature order
    mu : bool, optional
        If True, return the sum of the weights, optional.

    Returns
    -------
    x : ndarray
        Sample points
    w : ndarray
        Weights
    mu : float
        Sum of the weights

    See Also
    --------
    scipy.integrate.fixed_quad

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """
    # 调用roots_chebyt函数获取偏移的第一类Chebyshev多项式的根和权重
    xw = roots_chebyt(n, mu)

    # 返回结果，将第一个元素(xw[0])做了处理以映射到[0, 1]区间
    return ((xw[0] + 1) / 2,) + xw[1:]


# 第三个函数定义了偏移的第一类Chebyshev多项式，即T^*_n(x)
def sh_chebyt(n, monic=False):
    r"""Shifted Chebyshev polynomial of the first kind.

    Defined as :math:`T^*_n(x) = T_n(2x - 1)` for :math:`T_n` the nth
    Chebyshev polynomial of the first kind.

    Parameters
    ----------
    n : int
        Degree of the polynomial.

    """
    # 根据参数 `monic` 决定是否将多项式的主导系数标准化为1
    base = sh_jacobi(n, 0.0, 0.5, monic=monic)
    
    # 如果 `monic` 为 True，则直接返回基础多项式
    if monic:
        return base
    
    # 如果 `monic` 不为 True，且 `n` 大于0，则计算比例因子
    if n > 0:
        factor = 4**n / 2.0
    # 如果 `n` 不大于0，则使用默认的比例因子1.0
    else:
        factor = 1.0
    
    # 对基础多项式进行缩放，使其系数乘以计算得到的比例因子
    base._scale(factor)
    
    # 返回缩放后的多项式对象
    return base
# Shifted Chebyshev of the second kind    U^*_n(x)
def roots_sh_chebyu(n, mu=False):
    r"""Gauss-Chebyshev (second kind, shifted) quadrature.

    Computes the sample points and weights for Gauss-Chebyshev
    quadrature. The sample points are the roots of the nth degree
    shifted Chebyshev polynomial of the second kind, :math:`U_n(x)`.
    These sample points and weights correctly integrate polynomials of
    degree :math:`2n - 1` or less over the interval :math:`[0, 1]`
    with weight function :math:`w(x) = \sqrt{x - x^2}`. See 22.2.9 in
    [AS]_ for more details.

    Parameters
    ----------
    n : int
        quadrature order
    mu : bool, optional
        If True, return the sum of the weights, optional.

    Returns
    -------
    x : ndarray
        Sample points
    w : ndarray
        Weights
    mu : float
        Sum of the weights

    See Also
    --------
    scipy.integrate.fixed_quad

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """
    # 调用 roots_chebyu 函数获取 Chebyshev 多项式（第二类）的零点和权重
    x, w, m = roots_chebyu(n, True)
    # 对零点进行变换，将其映射到 [0, 1] 区间
    x = (x + 1) / 2
    # 计算修正系数，使得权重符合新的权重函数要求
    m_us = _ufuncs.beta(1.5, 1.5)
    w *= m_us / m
    # 如果 mu 为 True，则返回额外的 m_us
    if mu:
        return x, w, m_us
    else:
        return x, w


def sh_chebyu(n, monic=False):
    r"""Shifted Chebyshev polynomial of the second kind.

    Defined as :math:`U^*_n(x) = U_n(2x - 1)` for :math:`U_n` the nth
    Chebyshev polynomial of the second kind.

    Parameters
    ----------
    n : int
        Degree of the polynomial.
    monic : bool, optional
        If `True`, scale the leading coefficient to be 1. Default is
        `False`.

    Returns
    -------
    U : orthopoly1d
        Shifted Chebyshev polynomial of the second kind.

    Notes
    -----
    The polynomials :math:`U^*_n` are orthogonal over :math:`[0, 1]`
    with weight function :math:`(x - x^2)^{1/2}`.

    """
    # 调用 sh_jacobi 函数获取 Jacobi 多项式的第二类变种（monic 参数为 True 时）
    base = sh_jacobi(n, 2.0, 1.5, monic=monic)
    # 如果 monic 为 True，则直接返回基础多项式
    if monic:
        return base
    # 否则，对多项式进行缩放，以适应 Shifted Chebyshev 多项式的要求
    factor = 4**n
    base._scale(factor)
    return base

# Legendre


def roots_legendre(n, mu=False):
    r"""Gauss-Legendre quadrature.

    Compute the sample points and weights for Gauss-Legendre
    quadrature [GL]_. The sample points are the roots of the nth degree
    Legendre polynomial :math:`P_n(x)`. These sample points and
    weights correctly integrate polynomials of degree :math:`2n - 1`
    or less over the interval :math:`[-1, 1]` with weight function
    :math:`w(x) = 1`. See 2.2.10 in [AS]_ for more details.

    Parameters
    ----------
    n : int
        quadrature order
    mu : bool, optional
        If True, return the sum of the weights, optional.

    Returns
    -------
    x : ndarray
        Sample points
    w : ndarray
        Weights
    mu : float
        Sum of the weights

    See Also
    --------
    scipy.integrate.fixed_quad
    numpy.polynomial.legendre.leggauss

    References
    ----------
    # 将输入参数 n 转换为整数 m
    m = int(n)
    # 如果 n 不是正整数或者 n 不等于 m，抛出数值错误异常
    if n < 1 or n != m:
        raise ValueError("n must be a positive integer.")
    
    # 初始化 mu0 为 2.0
    mu0 = 2.0
    
    # 定义返回 0 的函数 an_func(k)
    def an_func(k):
        return 0.0 * k
    
    # 定义返回 k * sqrt(1 / (4 * k * k - 1)) 的函数 bn_func(k)
    def bn_func(k):
        return k * np.sqrt(1.0 / (4 * k * k - 1))
    
    # 将 _ufuncs.eval_legendre 赋值给 f
    f = _ufuncs.eval_legendre
    
    # 定义 df(n, x) 函数，计算 Legendre 多项式的导数
    def df(n, x):
        return (-n * x * _ufuncs.eval_legendre(n, x)
                + n * _ufuncs.eval_legendre(n - 1, x)) / (1 - x ** 2)
    
    # 调用 _gen_roots_and_weights 函数，并返回其结果
    return _gen_roots_and_weights(m, mu0, an_func, bn_func, f, df, True, mu)
# Shifted Legendre              P^*_n(x)

# 定义一个函数用来计算 n 阶的 Legendre 多项式
def roots_sh_legendre(n, mu=False):
    r"""Gauss-Legendre (shifted) quadrature.

    计算 Gauss-Legendre （平移）积分的采样点和权重。
    采样点是 n 阶平移 Legendre 多项式 :math:`P^*_n(x)` 的根。
    这些采样点和权重可以正确地积分度小于等于 2n-1 的多项式在区间 [0, 1] 上，权函数为 :math:`w(x) = 1.0`。
    参见 [AS]_ 中的 2.2.11 节了解详情。

    Parameters
    ----------
    n : int
        积分阶数
    mu : bool, optional
        如果为 True，返回权重的和，可选。

    Returns
    -------
    x : ndarray
        采样点
    w : ndarray
        权重
    mu : float
        权重的和

    See Also
    --------
    scipy.integrate.fixed_quad

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """
    # 计算 n 阶 Legendre 多项式的根和权重
    x, w = roots_legendre(n)
    # 将采样点变换到 [0, 1] 区间
    x = (x + 1) / 2
    # 权重缩放
    w /= 2
    # 如果需要计算权重的和，返回 x, w, 1.0
    if mu:
        return x, w, 1.0
    else:
        return x, w


# 定义一个函数计算 n 阶的平移 Legendre 多项式
def sh_legendre(n, monic=False):
    r"""Shifted Legendre polynomial.

    定义为 :math:`P^*_n(x) = P_n(2x - 1)`，其中 :math:`P_n` 是 n 阶 Legendre 多项式。

    Parameters
    ----------
    n : int
        多项式的阶数
    monic : bool, optional
        如果为 `True`，将首项系数缩放为 1。默认为 `False`。

    Returns
    -------
    P : orthopoly1d
        平移 Legendre 多项式。

    Notes
    -----
    # 函数定义：生成勒让德正交多项式 :math:`P^*_n`，在区间 [0, 1] 上具有权重函数 1
    if n < 0:
        # 如果 n 小于 0，抛出数值错误
        raise ValueError("n must be nonnegative.")

    def wfunc(x):
        # 定义权重函数 wfunc(x)，在该情况下恒为 1
        return 0.0 * x + 1.0

    if n == 0:
        # 如果 n 等于 0，返回一个特定的正交多项式对象 orthopoly1d
        return orthopoly1d([], [], 1.0, 1.0, wfunc, (0, 1), monic,
                           lambda x: _ufuncs.eval_sh_legendre(n, x))
    
    # 计算勒让德多项式的节点 x 和权重 w
    x, w = roots_sh_legendre(n)
    # 计算系数 hn 和 kn
    hn = 1.0 / (2 * n + 1.0)
    kn = _gam(2 * n + 1) / _gam(n + 1)**2
    # 创建正交多项式对象 p，使用给定的节点 x、权重 w、系数 hn 和 kn，以及权重函数 wfunc 和区间限制 (0, 1)
    p = orthopoly1d(x, w, hn, kn, wfunc, limits=(0, 1), monic=monic,
                    eval_func=lambda x: _ufuncs.eval_sh_legendre(n, x))
    # 返回正交多项式对象 p
    return p
# 将旧的根函数名设置为新函数名的别名
_modattrs = globals()
# 遍历 `_rootfuns_map` 字典中的每一对新旧函数名
for newfun, oldfun in _rootfuns_map.items():
    # 将全局变量中旧函数名的引用指向对应的新函数
    _modattrs[oldfun] = _modattrs[newfun]
    # 将新的函数名添加到模块的 `__all__` 列表中，用于导出
    __all__.append(oldfun)
```