# `.\numpy\numpy\lib\_polynomial_impl.py`

```
"""
Functions to operate on polynomials.

"""
__all__ = ['poly', 'roots', 'polyint', 'polyder', 'polyadd',
           'polysub', 'polymul', 'polydiv', 'polyval', 'poly1d',
           'polyfit']

# 导入 functools 模块，用于函数装饰器和高阶函数操作
import functools
# 导入 re 模块，用于正则表达式操作
import re
# 导入 warnings 模块，用于警告处理
import warnings

# 导入 set_module 函数，用于设置模块名称
from .._utils import set_module
# 导入 numpy 中的核心模块和函数
import numpy._core.numeric as NX

# 导入 numpy 中的核心模块中的函数和类
from numpy._core import (isscalar, abs, finfo, atleast_1d, hstack, dot, array,
                        ones)
# 导入 numpy 中的覆盖方法装饰器
from numpy._core import overrides
# 导入 numpy 中的异常类
from numpy.exceptions import RankWarning
# 导入 numpy 中的二维基础实现函数
from numpy.lib._twodim_base_impl import diag, vander
# 导入 numpy 中的基础函数实现
from numpy.lib._function_base_impl import trim_zeros
# 导入 numpy 中的类型检查函数
from numpy.lib._type_check_impl import iscomplex, real, imag, mintypecode
# 导入 numpy 中的线性代数函数
from numpy.linalg import eigvals, lstsq, inv

# 创建一个偏函数，用于设置数组函数分发时的模块名称
array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')


# 定义一个函数装饰器，用于分发 _poly_dispatcher 函数
def _poly_dispatcher(seq_of_zeros):
    return seq_of_zeros


# 使用函数装饰器装饰 poly 函数，为其添加数组函数分发功能
@array_function_dispatch(_poly_dispatcher)
def poly(seq_of_zeros):
    """
    Find the coefficients of a polynomial with the given sequence of roots.

    .. note::
       This forms part of the old polynomial API. Since version 1.4, the
       new polynomial API defined in `numpy.polynomial` is preferred.
       A summary of the differences can be found in the
       :doc:`transition guide </reference/routines.polynomials>`.

    Returns the coefficients of the polynomial whose leading coefficient
    is one for the given sequence of zeros (multiple roots must be included
    in the sequence as many times as their multiplicity; see Examples).
    A square matrix (or array, which will be treated as a matrix) can also
    be given, in which case the coefficients of the characteristic polynomial
    of the matrix are returned.

    Parameters
    ----------
    seq_of_zeros : array_like, shape (N,) or (N, N)
        A sequence of polynomial roots, or a square array or matrix object.

    Returns
    -------
    c : ndarray
        1D array of polynomial coefficients from highest to lowest degree:

        ``c[0] * x**(N) + c[1] * x**(N-1) + ... + c[N-1] * x + c[N]``
        where c[0] always equals 1.

    Raises
    ------
    ValueError
        If input is the wrong shape (the input must be a 1-D or square
        2-D array).

    See Also
    --------
    polyval : Compute polynomial values.
    roots : Return the roots of a polynomial.
    polyfit : Least squares polynomial fit.
    poly1d : A one-dimensional polynomial class.

    Notes
    -----
    Specifying the roots of a polynomial still leaves one degree of
    freedom, typically represented by an undetermined leading
    coefficient. [1]_ In the case of this function, that coefficient -
    the first one in the returned array - is always taken as one. (If
    for some reason you have one other point, the only automatic way
    presently to leverage that information is to use ``polyfit``.)

    The characteristic polynomial, :math:`p_a(t)`, of an `n`-by-`n`
    matrix **A** is given by
    """
    seq_of_zeros = atleast_1d(seq_of_zeros)
    sh = seq_of_zeros.shape

    if len(sh) == 2 and sh[0] == sh[1] and sh[0] != 0:
        # 如果输入是二维数组且为方阵且非空，则计算其特征值
        seq_of_zeros = eigvals(seq_of_zeros)
    elif len(sh) == 1:
        dt = seq_of_zeros.dtype
        # 让对象数组通过，例如用于任意精度
        if dt != object:
            seq_of_zeros = seq_of_zeros.astype(mintypecode(dt.char))
    else:
        # 若输入不是一维数组或者不是非空方阵，则引发值错误
        raise ValueError("input must be 1d or non-empty square 2d array.")

    if len(seq_of_zeros) == 0:
        return 1.0
    dt = seq_of_zeros.dtype
    a = ones((1,), dtype=dt)
    for zero in seq_of_zeros:
        # 利用 numpy 的卷积函数构造多项式的系数数组
        a = NX.convolve(a, array([1, -zero], dtype=dt), mode='full')

    if issubclass(a.dtype.type, NX.complexfloating):
        # 如果复数根都是共轭复数，则多项式的根为实数
        roots = NX.asarray(seq_of_zeros, complex)
        if NX.all(NX.sort(roots) == NX.sort(roots.conjugate())):
            a = a.real.copy()

    return a
# 定义一个函数 _roots_dispatcher，返回输入参数 p，用于分派多项式的根
def _roots_dispatcher(p):
    return p


# 使用装饰器 array_function_dispatch 将函数 roots 与 _roots_dispatcher 关联起来，
# 用于处理多项式的根，是 numpy 的多项式 API 的一部分
@array_function_dispatch(_roots_dispatcher)
def roots(p):
    """
    Return the roots of a polynomial with coefficients given in p.

    .. note::
       This forms part of the old polynomial API. Since version 1.4, the
       new polynomial API defined in `numpy.polynomial` is preferred.
       A summary of the differences can be found in the
       :doc:`transition guide </reference/routines.polynomials>`.

    The values in the rank-1 array `p` are coefficients of a polynomial.
    If the length of `p` is n+1 then the polynomial is described by::

      p[0] * x**n + p[1] * x**(n-1) + ... + p[n-1]*x + p[n]

    Parameters
    ----------
    p : array_like
        Rank-1 array of polynomial coefficients.

    Returns
    -------
    out : ndarray
        An array containing the roots of the polynomial.

    Raises
    ------
    ValueError
        When `p` cannot be converted to a rank-1 array.

    See also
    --------
    poly : Find the coefficients of a polynomial with a given sequence
           of roots.
    polyval : Compute polynomial values.
    polyfit : Least squares polynomial fit.
    poly1d : A one-dimensional polynomial class.

    Notes
    -----
    The algorithm relies on computing the eigenvalues of the
    companion matrix [1]_.

    References
    ----------
    .. [1] R. A. Horn & C. R. Johnson, *Matrix Analysis*.  Cambridge, UK:
        Cambridge University Press, 1999, pp. 146-7.

    Examples
    --------
    >>> coeff = [3.2, 2, 1]
    >>> np.roots(coeff)
    array([-0.3125+0.46351241j, -0.3125-0.46351241j])

    """
    # 如果输入是标量，则将其转换为数组
    p = atleast_1d(p)
    if p.ndim != 1:
        raise ValueError("Input must be a rank-1 array.")

    # 找到非零数组条目的索引
    non_zero = NX.nonzero(NX.ravel(p))[0]

    # 如果多项式全为零，则返回一个空数组
    if len(non_zero) == 0:
        return NX.array([])

    # 找到末尾的零的数量，这是多项式在 0 处的根数
    trailing_zeros = len(p) - non_zero[-1] - 1

    # 去除首尾的零系数
    p = p[int(non_zero[0]):int(non_zero[-1])+1]

    # 类型转换：如果输入数组不是浮点数类型，则转换为浮点数
    if not issubclass(p.dtype.type, (NX.floating, NX.complexfloating)):
        p = p.astype(float)

    N = len(p)
    if N > 1:
        # 构建伴随矩阵并找到其特征值（即多项式的根）
        A = diag(NX.ones((N-2,), p.dtype), -1)
        A[0,:] = -p[1:] / p[0]
        roots = eigvals(A)
    else:
        roots = NX.array([])

    # 将任何零根附加到数组的末尾
    roots = hstack((roots, NX.zeros(trailing_zeros, roots.dtype)))
    return roots


# 定义一个函数 _polyint_dispatcher，返回元组 (p,)
def _polyint_dispatcher(p, m=None, k=None):
    return (p,)


# 使用装饰器 array_function_dispatch 将函数 polyint 与 _polyint_dispatcher 关联起来，
# 用于计算多项式的不定积分，是 numpy 的多项式 API 的一部分
@array_function_dispatch(_polyint_dispatcher)
def polyint(p, m=1, k=None):
    """
    Return an antiderivative (indefinite integral) of a polynomial.
    
    """
    # 将 m 转换为整数，确保是非负整数
    m = int(m)
    # 如果 m 小于 0，则抛出值错误异常，要求积分的阶数必须是正数（参见 polyder）
    if m < 0:
        raise ValueError("Order of integral must be positive (see polyder)")
    
    # 如果 k 为 None，则设定 k 为 m 个零值的数组
    if k is None:
        k = NX.zeros(m, float)
    
    # 确保 k 至少是一维数组
    k = atleast_1d(k)
    
    # 如果 k 的长度为 1 且 m 大于 1，则将 k 扩展为包含 m 个相同元素的数组
    if len(k) == 1 and m > 1:
        k = k[0]*NX.ones(m, float)
    
    # 如果 k 的长度小于 m，则抛出值错误异常，要求 k 是一个标量或长度大于 m 的一维数组
    if len(k) < m:
        raise ValueError(
              "k must be a scalar or a rank-1 array of length 1 or >m.")
    
    # 判断 p 是否为 poly1d 类型的对象
    truepoly = isinstance(p, poly1d)
    
    # 将 p 转换为 NumPy 数组
    p = NX.asarray(p)
    
    # 如果 m 为 0，则根据 truepoly 的类型返回相应结果
    if m == 0:
        if truepoly:
            return poly1d(p)
        return p
    else:
        # 合并 p 的元素与 k[0]，作为新的多项式 y，以便继续积分
        y = NX.concatenate((p.__truediv__(NX.arange(len(p), 0, -1)), [k[0]]))
        # 递归调用 polyint 函数，积分阶数减少 1，传入 k 的其余部分作为参数
        val = polyint(y, m - 1, k=k[1:])
        # 根据 truepoly 的类型返回相应结果
        if truepoly:
            return poly1d(val)
        return val
# 定义一个函数 _polyder_dispatcher，用于多项式求导函数 polyder 的分派器
def _polyder_dispatcher(p, m=None):
    # 返回一个包含参数 p 的元组
    return (p,)


# 使用 array_function_dispatch 装饰器将 _polyder_dispatcher 注册为 polyder 函数的分派器
@array_function_dispatch(_polyder_dispatcher)
# 定义多项式求导函数 polyder
def polyder(p, m=1):
    """
    Return the derivative of the specified order of a polynomial.

    .. note::
       This forms part of the old polynomial API. Since version 1.4, the
       new polynomial API defined in `numpy.polynomial` is preferred.
       A summary of the differences can be found in the
       :doc:`transition guide </reference/routines.polynomials>`.

    Parameters
    ----------
    p : poly1d or sequence
        Polynomial to differentiate.
        A sequence is interpreted as polynomial coefficients, see `poly1d`.
    m : int, optional
        Order of differentiation (default: 1)

    Returns
    -------
    der : poly1d
        A new polynomial representing the derivative.

    See Also
    --------
    polyint : Anti-derivative of a polynomial.
    poly1d : Class for one-dimensional polynomials.

    Examples
    --------
    The derivative of the polynomial :math:`x^3 + x^2 + x^1 + 1` is:

    >>> p = np.poly1d([1,1,1,1])
    >>> p2 = np.polyder(p)
    >>> p2
    poly1d([3, 2, 1])

    which evaluates to:

    >>> p2(2.)
    17.0

    We can verify this, approximating the derivative with
    ``(f(x + h) - f(x))/h``:

    >>> (p(2. + 0.001) - p(2.)) / 0.001
    17.007000999997857

    The fourth-order derivative of a 3rd-order polynomial is zero:

    >>> np.polyder(p, 2)
    poly1d([6, 2])
    >>> np.polyder(p, 3)
    poly1d([6])
    >>> np.polyder(p, 4)
    poly1d([0])

    """
    # 将 m 转换为整数
    m = int(m)
    # 如果 m 小于 0，则抛出 ValueError 异常
    if m < 0:
        raise ValueError("Order of derivative must be positive (see polyint)")

    # 检查 p 是否为 poly1d 对象
    truepoly = isinstance(p, poly1d)
    # 将 p 转换为 ndarray 对象
    p = NX.asarray(p)
    # 获取多项式 p 的阶数
    n = len(p) - 1
    # 计算导数多项式的系数
    y = p[:-1] * NX.arange(n, 0, -1)
    # 如果 m 为 0，则直接返回原多项式 p
    if m == 0:
        val = p
    else:
        # 递归调用 polyder 函数，计算更高阶导数
        val = polyder(y, m - 1)
    # 如果原始输入是 poly1d 对象，则将结果转换为 poly1d 对象
    if truepoly:
        val = poly1d(val)
    # 返回导数多项式
    return val


# 定义一个函数 _polyfit_dispatcher，用于多项式拟合函数 polyfit 的分派器
def _polyfit_dispatcher(x, y, deg, rcond=None, full=None, w=None, cov=None):
    # 返回一个包含参数 x, y, w 的元组
    return (x, y, w)


# 使用 array_function_dispatch 装饰器将 _polyfit_dispatcher 注册为 polyfit 函数的分派器
@array_function_dispatch(_polyfit_dispatcher)
# 定义多项式拟合函数 polyfit
def polyfit(x, y, deg, rcond=None, full=False, w=None, cov=False):
    """
    Least squares polynomial fit.

    .. note::
       This forms part of the old polynomial API. Since version 1.4, the
       new polynomial API defined in `numpy.polynomial` is preferred.
       A summary of the differences can be found in the
       :doc:`transition guide </reference/routines.polynomials>`.

    Parameters
    ----------
    x : array_like, shape (M,)
        x-coordinates of the M sample points ``(x[i], y[i])``.
    # y 是样本点的 y 坐标，可以是形状为 (M,) 或 (M, K) 的数组
    # 其中 M 是样本点的数量，K 是数据集的数量
    y : array_like, shape (M,) or (M, K)

    # deg 是拟合多项式的阶数
    deg : int

    # rcond 是拟合条件数的相对值，小于这个值的奇异值将被忽略
    # 默认值为 len(x)*eps，其中 eps 是浮点数的相对精度，大约为 2e-16
    rcond : float, optional

    # full 控制返回值的性质，当为 False 时（默认值），仅返回系数
    # 当为 True 时，还返回奇异值分解的诊断信息
    full : bool, optional

    # w 是样本点的权重，形状为 (M,) 的数组，可选参数
    # 如果不为 None，则权重 w[i] 适用于 x[i] 处的未平方残差 y[i] - y_hat[i]
    # 默认值为 None
    w : array_like, shape (M,), optional

    # cov 控制是否返回估计值及其协方差矩阵
    # 如果给出且不是 False，则返回估计值及其协方差矩阵
    # 默认情况下，协方差按 chi2/dof 缩放，其中 dof = M - (deg + 1)
    # 如果 cov='unscaled'，则省略此缩放，适用于权重 w = 1/sigma 的情况
    cov : bool or str, optional

    # 返回结果
    # p 是形状为 (deg + 1,) 或 (deg + 1, K) 的数组
    # 包含多项式系数，最高阶次排在最前面
    # 如果 y 是二维数组，则第 k 个数据集的系数在 p[:,k] 中
    Returns
    -------
    p : ndarray, shape (deg + 1,) or (deg + 1, K)

    # 仅当 full == True 时返回以下值
    # residuals -- 最小二乘拟合的残差平方和
    # rank -- 缩放后的 Vandermonde 系数矩阵的有效秩
    # singular_values -- 缩放后的 Vandermonde 系数矩阵的奇异值
    # rcond -- rcond 的值
    residuals, rank, singular_values, rcond

    # 仅当 full == False 且 cov == True 时才存在
    # V 是形状为 (deg + 1, deg + 1) 或 (deg + 1, deg + 1, K) 的数组
    # 包含多项式系数估计的协方差矩阵
    # 对角线上的元素是每个系数的方差估计
    # 如果 y 是二维数组，则第 k 个数据集的协方差矩阵在 V[:,:,k] 中
    V : ndarray, shape (deg + 1, deg + 1) or (deg + 1, deg + 1, K)

    # 警告
    Warns
    -----
    RankWarning
        The rank of the coefficient matrix in the least-squares fit is
        deficient. The warning is only raised if ``full == False``.

        The warnings can be turned off by

        >>> import warnings
        >>> warnings.simplefilter('ignore', np.exceptions.RankWarning)

    See Also
    --------
    polyval : Compute polynomial values.
    linalg.lstsq : Computes a least-squares fit.
    scipy.interpolate.UnivariateSpline : Computes spline fits.

    Notes
    -----
    The solution minimizes the squared error

    .. math::
        E = \\sum_{j=0}^k |p(x_j) - y_j|^2

    in the equations::

        x[0]**n * p[0] + ... + x[0] * p[n-1] + p[n] = y[0]
        x[1]**n * p[0] + ... + x[1] * p[n-1] + p[n] = y[1]
        ...
        x[k]**n * p[0] + ... + x[k] * p[n-1] + p[n] = y[k]

    The coefficient matrix of the coefficients `p` is a Vandermonde matrix.

    `polyfit` issues a `~exceptions.RankWarning` when the least-squares fit is
    badly conditioned. This implies that the best fit is not well-defined due
    to numerical error. The results may be improved by lowering the polynomial
    degree or by replacing `x` by `x` - `x`.mean(). The `rcond` parameter
    can also be set to a value smaller than its default, but the resulting
    fit may be spurious: including contributions from the small singular
    values can add numerical noise to the result.

    Note that fitting polynomial coefficients is inherently badly conditioned
    when the degree of the polynomial is large or the interval of sample points
    is badly centered. The quality of the fit should always be checked in these
    cases. When polynomial fits are not satisfactory, splines may be a good
    alternative.

    References
    ----------
    .. [1] Wikipedia, "Curve fitting",
           https://en.wikipedia.org/wiki/Curve_fitting
    .. [2] Wikipedia, "Polynomial interpolation",
           https://en.wikipedia.org/wiki/Polynomial_interpolation

    Examples
    --------
    >>> import warnings
    >>> x = np.array([0.0, 1.0, 2.0, 3.0,  4.0,  5.0])
    >>> y = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0])
    >>> z = np.polyfit(x, y, 3)
    >>> z
    array([ 0.08703704, -0.81349206,  1.69312169, -0.03968254]) # may vary

    It is convenient to use `poly1d` objects for dealing with polynomials:

    >>> p = np.poly1d(z)
    >>> p(0.5)
    0.6143849206349179 # may vary
    >>> p(3.5)
    -0.34732142857143039 # may vary
    >>> p(10)
    22.579365079365115 # may vary

    High-order polynomials may oscillate wildly:

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter('ignore', np.exceptions.RankWarning)
    ...     p30 = np.poly1d(np.polyfit(x, y, 30))
    ...
    >>> p30(4)
    -0.80000000000000204 # may vary
    >>> p30(5)
    -0.99999999999999445 # may vary
    >>> p30(4.5)
    -0.10547061179440398 # may vary

    Illustration:

    >>> import matplotlib.pyplot as plt
    >>> xp = np.linspace(-2, 6, 100)
    >>> _ = plt.plot(x, y, '.', xp, p(xp), '-', xp, p30(xp), '--')
    # 绘制散点图、一次多项式拟合曲线和三十次多项式拟合曲线
    >>> plt.ylim(-2,2)
    # 设置 y 轴的显示范围为 -2 到 2
    (-2, 2)
    >>> plt.show()
    # 显示绘制的图形

    """
    order = int(deg) + 1
    # 计算多项式的阶数
    x = NX.asarray(x) + 0.0
    y = NX.asarray(y) + 0.0

    # 检查参数
    if deg < 0:
        raise ValueError("expected deg >= 0")
    if x.ndim != 1:
        raise TypeError("expected 1D vector for x")
    if x.size == 0:
        raise TypeError("expected non-empty vector for x")
    if y.ndim < 1 or y.ndim > 2:
        raise TypeError("expected 1D or 2D array for y")
    if x.shape[0] != y.shape[0]:
        raise TypeError("expected x and y to have same length")

    # 设置 rcond
    if rcond is None:
        rcond = len(x)*finfo(x.dtype).eps

    # 设置最小二乘法方程的左右两侧
    lhs = vander(x, order)
    rhs = y

    # 应用加权
    if w is not None:
        w = NX.asarray(w) + 0.0
        if w.ndim != 1:
            raise TypeError("expected a 1-d array for weights")
        if w.shape[0] != y.shape[0]:
            raise TypeError("expected w and y to have the same length")
        lhs *= w[:, NX.newaxis]
        if rhs.ndim == 2:
            rhs *= w[:, NX.newaxis]
        else:
            rhs *= w

    # 缩放 lhs 以改善条件数并求解
    scale = NX.sqrt((lhs*lhs).sum(axis=0))
    lhs /= scale
    c, resids, rank, s = lstsq(lhs, rhs, rcond)
    c = (c.T/scale).T  # 广播缩放系数

    # 警告阶数降低，表明矩阵条件不佳
    if rank != order and not full:
        msg = "Polyfit may be poorly conditioned"
        warnings.warn(msg, RankWarning, stacklevel=2)

    if full:
        return c, resids, rank, s, rcond
    elif cov:
        Vbase = inv(dot(lhs.T, lhs))
        Vbase /= NX.outer(scale, scale)
        if cov == "unscaled":
            fac = 1
        else:
            if len(x) <= order:
                raise ValueError("the number of data points must exceed order "
                                 "to scale the covariance matrix")
            # 注意，这里曾经是: fac = resids / (len(x) - order - 2.0)
            # 决定不使用 "- 2"（最初通过“贝叶斯不确定性分析”来证明），因为用户不期望这样
            # （见 gh-11196 和 gh-11197）
            fac = resids / (len(x) - order)
        if y.ndim == 1:
            return c, Vbase * fac
        else:
            return c, Vbase[:,:, NX.newaxis] * fac
    else:
        return c
# 定义一个分派函数，用于多项式求值函数 polyval
def _polyval_dispatcher(p, x):
    return (p, x)


# 使用 array_function_dispatch 装饰器将 _polyval_dispatcher 函数注册为 polyval 函数的分派函数
@array_function_dispatch(_polyval_dispatcher)
# 定义多项式求值函数 polyval，用于在特定值处评估多项式
def polyval(p, x):
    """
    Evaluate a polynomial at specific values.

    .. note::
       This forms part of the old polynomial API. Since version 1.4, the
       new polynomial API defined in `numpy.polynomial` is preferred.
       A summary of the differences can be found in the
       :doc:`transition guide </reference/routines.polynomials>`.

    If `p` is of length N, this function returns the value::

        p[0]*x**(N-1) + p[1]*x**(N-2) + ... + p[N-2]*x + p[N-1]

    If `x` is a sequence, then ``p(x)`` is returned for each element of ``x``.
    If `x` is another polynomial then the composite polynomial ``p(x(t))``
    is returned.

    Parameters
    ----------
    p : array_like or poly1d object
       1D array of polynomial coefficients (including coefficients equal
       to zero) from highest degree to the constant term, or an
       instance of poly1d.
    x : array_like or poly1d object
       A number, an array of numbers, or an instance of poly1d, at
       which to evaluate `p`.

    Returns
    -------
    values : ndarray or poly1d
       If `x` is a poly1d instance, the result is the composition of the two
       polynomials, i.e., `x` is "substituted" in `p` and the simplified
       result is returned. In addition, the type of `x` - array_like or
       poly1d - governs the type of the output: `x` array_like => `values`
       array_like, `x` a poly1d object => `values` is also.

    See Also
    --------
    poly1d: A polynomial class.

    Notes
    -----
    Horner's scheme [1]_ is used to evaluate the polynomial. Even so,
    for polynomials of high degree the values may be inaccurate due to
    rounding errors. Use carefully.

    If `x` is a subtype of `ndarray` the return value will be of the same type.

    References
    ----------
    .. [1] I. N. Bronshtein, K. A. Semendyayev, and K. A. Hirsch (Eng.
       trans. Ed.), *Handbook of Mathematics*, New York, Van Nostrand
       Reinhold Co., 1985, pg. 720.

    Examples
    --------
    >>> np.polyval([3,0,1], 5)  # 3 * 5**2 + 0 * 5**1 + 1
    76
    >>> np.polyval([3,0,1], np.poly1d(5))
    poly1d([76])
    >>> np.polyval(np.poly1d([3,0,1]), 5)
    76
    >>> np.polyval(np.poly1d([3,0,1]), np.poly1d(5))
    poly1d([76])

    """
    # 将 p 转换为 ndarray 类型
    p = NX.asarray(p)
    # 如果 x 是 poly1d 类型，则初始化 y 为 0
    if isinstance(x, poly1d):
        y = 0
    else:
        # 否则将 x 转换为任意数组类型
        x = NX.asanyarray(x)
        # 初始化 y 为和 x 形状相同的零数组
        y = NX.zeros_like(x)
    # 使用 Horner's scheme 计算多项式的值
    for pv in p:
        y = y * x + pv
    # 返回计算结果
    return y


# 定义一个分派函数，用于二元操作函数 polyadd
def _binary_op_dispatcher(a1, a2):
    return (a1, a2)


# 使用 array_function_dispatch 装饰器将 _binary_op_dispatcher 函数注册为 polyadd 函数的分派函数
@array_function_dispatch(_binary_op_dispatcher)
# 定义多项式相加函数 polyadd，用于计算两个多项式的和
def polyadd(a1, a2):
    """
    Find the sum of two polynomials.

    .. note::
       This forms part of the old polynomial API. Since version 1.4, the
       new polynomial API defined in `numpy.polynomial` is preferred.
       A summary of the differences can be found in the
       :doc:`transition guide </reference/routines.polynomials>`.

    """
    # 返回两个多项式相加后的结果。每个输入可以是 poly1d 对象或者从高次到低次排列的多项式系数的一维序列。
    
    # 参数：
    # a1, a2 : array_like 或者 poly1d 对象
    #     输入的多项式。
    
    # 返回：
    # out : ndarray 或者 poly1d 对象
    #     输入的两个多项式的和。如果任何一个输入是 poly1d 对象，那么输出也是 poly1d 对象。否则，输出是一个从高次到低次排列的多项式系数的一维数组。
    
    # 参见：
    # poly1d : 一维多项式类。
    # poly, polyadd, polyder, polydiv, polyfit, polyint, polysub, polyval
    
    # 示例：
    # >>> np.polyadd([1, 2], [9, 5, 4])
    # array([9, 6, 6])
    
    # 使用 poly1d 对象：
    
    # >>> p1 = np.poly1d([1, 2])
    # >>> p2 = np.poly1d([9, 5, 4])
    # >>> print(p1)
    # 1 x + 2
    # >>> print(p2)
    #    2
    # 9 x + 5 x + 4
    # >>> print(np.polyadd(p1, p2))
    #    2
    # 9 x + 6 x + 6
    
    truepoly = (isinstance(a1, poly1d) or isinstance(a2, poly1d))  # 检查是否存在至少一个输入是 poly1d 对象
    a1 = atleast_1d(a1)  # 确保 a1 至少是一维数组
    a2 = atleast_1d(a2)  # 确保 a2 至少是一维数组
    diff = len(a2) - len(a1)  # 计算两个输入数组的长度差异
    if diff == 0:
        val = a1 + a2  # 如果长度相同，直接相加
    elif diff > 0:
        zr = NX.zeros(diff, a1.dtype)  # 创建一个长度为 diff 的零数组，类型与 a1 相同
        val = NX.concatenate((zr, a1)) + a2  # 将零数组和 a1 拼接起来后再相加
    else:
        zr = NX.zeros(abs(diff), a2.dtype)  # 创建一个长度为 |diff| 的零数组，类型与 a2 相同
        val = a1 + NX.concatenate((zr, a2))  # 将零数组和 a2 拼接起来后再相加
    if truepoly:
        val = poly1d(val)  # 如果输入中至少有一个是 poly1d 对象，则将结果转换为 poly1d 对象
    return val  # 返回相加后的结果
# 定义 polysub 函数，用于计算两个多项式的差（减法）
@array_function_dispatch(_binary_op_dispatcher)
def polysub(a1, a2):
    """
    Difference (subtraction) of two polynomials.

    .. note::
       This forms part of the old polynomial API. Since version 1.4, the
       new polynomial API defined in `numpy.polynomial` is preferred.
       A summary of the differences can be found in the
       :doc:`transition guide </reference/routines.polynomials>`.

    Given two polynomials `a1` and `a2`, returns ``a1 - a2``.
    `a1` and `a2` can be either array_like sequences of the polynomials'
    coefficients (including coefficients equal to zero), or `poly1d` objects.

    Parameters
    ----------
    a1, a2 : array_like or poly1d
        Minuend and subtrahend polynomials, respectively.

    Returns
    -------
    out : ndarray or poly1d
        Array or `poly1d` object of the difference polynomial's coefficients.

    See Also
    --------
    polyval, polydiv, polymul, polyadd

    Examples
    --------
    .. math:: (2 x^2 + 10 x - 2) - (3 x^2 + 10 x -4) = (-x^2 + 2)

    >>> np.polysub([2, 10, -2], [3, 10, -4])
    array([-1,  0,  2])

    """
    # 判断输入是否为 poly1d 类型
    truepoly = (isinstance(a1, poly1d) or isinstance(a2, poly1d))
    # 将 a1 和 a2 转换为至少为一维的数组
    a1 = atleast_1d(a1)
    a2 = atleast_1d(a2)
    # 计算 a2 和 a1 的长度差
    diff = len(a2) - len(a1)
    # 根据长度差进行多项式的减法操作
    if diff == 0:
        val = a1 - a2
    elif diff > 0:
        # 如果 a2 长度大于 a1，则在 a1 前面补零再进行减法操作
        zr = NX.zeros(diff, a1.dtype)
        val = NX.concatenate((zr, a1)) - a2
    else:
        # 如果 a1 长度大于 a2，则在 a2 前面补零再进行减法操作
        zr = NX.zeros(abs(diff), a2.dtype)
        val = a1 - NX.concatenate((zr, a2))
    # 如果输入是 poly1d 对象，则将结果转换为 poly1d 对象
    if truepoly:
        val = poly1d(val)
    # 返回结果
    return val


@array_function_dispatch(_binary_op_dispatcher)
def polymul(a1, a2):
    """
    Find the product of two polynomials.

    .. note::
       This forms part of the old polynomial API. Since version 1.4, the
       new polynomial API defined in `numpy.polynomial` is preferred.
       A summary of the differences can be found in the
       :doc:`transition guide </reference/routines.polynomials>`.

    Finds the polynomial resulting from the multiplication of the two input
    polynomials. Each input must be either a poly1d object or a 1D sequence
    of polynomial coefficients, from highest to lowest degree.

    Parameters
    ----------
    a1, a2 : array_like or poly1d object
        Input polynomials.

    Returns
    -------
    out : ndarray or poly1d object
        The polynomial resulting from the multiplication of the inputs. If
        either inputs is a poly1d object, then the output is also a poly1d
        object. Otherwise, it is a 1D array of polynomial coefficients from
        highest to lowest degree.

    See Also
    --------
    poly1d : A one-dimensional polynomial class.
    poly, polyadd, polyder, polydiv, polyfit, polyint, polysub, polyval
    convolve : Array convolution. Same output as polymul, but has parameter
               for overlap mode.

    Examples
    --------
    >>> np.polymul([1, 2, 3], [9, 5, 1])
    array([ 9, 23, 38, 17,  3])

    Using poly1d objects:
    """
    # 计算两个多项式的乘积
    # 检查输入是否为 poly1d 对象
    truepoly = (isinstance(a1, poly1d) or isinstance(a2, poly1d))
    # 将输入转换为至少为一维的数组
    a1 = atleast_1d(a1)
    a2 = atleast_1d(a2)
    # 使用 numpy 的多项式乘法函数，得到乘积多项式的系数
    val = NX.polymul(a1, a2)
    # 如果输入是 poly1d 对象，则将结果转换为 poly1d 对象
    if truepoly:
        val = poly1d(val)
    # 返回结果
    return val
    # 创建一个一元多项式 p1，系数为 [1, 2, 3]
    p1 = np.poly1d([1, 2, 3])
    # 创建一个一元多项式 p2，系数为 [9, 5, 1]
    p2 = np.poly1d([9, 5, 1])
    
    # 打印 p1 的字符串表示，显示为 1*x^2 + 2*x + 3
    print(p1)
    # 打印 p2 的字符串表示，显示为 9*x^2 + 5*x + 1
    print(p2)
    
    # 计算 p1 和 p2 的乘积
    # 打印结果显示为 9*x^4 + 23*x^3 + 38*x^2 + 17*x + 3
    print(np.polymul(p1, p2))
    
    """
    判断输入参数 a1 和 a2 是否为 poly1d 类型的对象
    如果其中一个是 poly1d 类型，则 truepoly 为 True，否则为 False
    """
    truepoly = (isinstance(a1, poly1d) or isinstance(a2, poly1d))
    
    # 将输入参数 a1 和 a2 都转换为 poly1d 类型的对象
    a1, a2 = poly1d(a1), poly1d(a2)
    
    # 使用 NumPy 中的 convolve 函数计算多项式的卷积
    val = NX.convolve(a1, a2)
    
    # 如果 truepoly 为 True，则将 val 转换为 poly1d 类型的对象
    if truepoly:
        val = poly1d(val)
    
    # 返回计算结果 val
    return val
# 定义一个分发函数，用于多项式除法，返回输入参数本身
def _polydiv_dispatcher(u, v):
    return (u, v)

# 使用装饰器实现多项式除法函数的分派，这里的函数本身并未实现多项式除法逻辑
@array_function_dispatch(_polydiv_dispatcher)
def polydiv(u, v):
    """
    Returns the quotient and remainder of polynomial division.

    .. note::
       This forms part of the old polynomial API. Since version 1.4, the
       new polynomial API defined in `numpy.polynomial` is preferred.
       A summary of the differences can be found in the
       :doc:`transition guide </reference/routines.polynomials>`.

    The input arrays are the coefficients (including any coefficients
    equal to zero) of the "numerator" (dividend) and "denominator"
    (divisor) polynomials, respectively.

    Parameters
    ----------
    u : array_like or poly1d
        Dividend polynomial's coefficients.

    v : array_like or poly1d
        Divisor polynomial's coefficients.

    Returns
    -------
    q : ndarray
        Coefficients, including those equal to zero, of the quotient.
    r : ndarray
        Coefficients, including those equal to zero, of the remainder.

    See Also
    --------
    poly, polyadd, polyder, polydiv, polyfit, polyint, polymul, polysub
    polyval

    Notes
    -----
    Both `u` and `v` must be 0-d or 1-d (ndim = 0 or 1), but `u.ndim` need
    not equal `v.ndim`. In other words, all four possible combinations -
    ``u.ndim = v.ndim = 0``, ``u.ndim = v.ndim = 1``,
    ``u.ndim = 1, v.ndim = 0``, and ``u.ndim = 0, v.ndim = 1`` - work.

    Examples
    --------
    .. math:: \\frac{3x^2 + 5x + 2}{2x + 1} = 1.5x + 1.75, remainder 0.25

    >>> x = np.array([3.0, 5.0, 2.0])
    >>> y = np.array([2.0, 1.0])
    >>> np.polydiv(x, y)
    (array([1.5 , 1.75]), array([0.25]))

    """
    # 检查输入参数是否为多项式对象
    truepoly = (isinstance(u, poly1d) or isinstance(v, poly1d))
    # 将输入参数转换为至少是一维的 ndarray，并添加浮点数类型
    u = atleast_1d(u) + 0.0
    v = atleast_1d(v) + 0.0
    # 计算 u 和 v 的第一个元素之和，并赋值给变量 w
    w = u[0] + v[0]
    # 计算 u 和 v 的长度，减去 1 后分别赋值给 m 和 n
    m = len(u) - 1
    n = len(v) - 1
    # 计算 v[0] 的倒数，并赋值给 scale
    scale = 1. / v[0]
    # 创建一个全零数组 q，形状为 (m-n+1,)，数据类型与 w 的数据类型一致
    q = NX.zeros((max(m - n + 1, 1),), w.dtype)
    # 将 u 转换为 w 的数据类型，并赋值给数组 r
    r = u.astype(w.dtype)
    # 执行多项式长除法算法
    for k in range(0, m-n+1):
        d = scale * r[k]
        q[k] = d
        r[k:k+n+1] -= d*v
    # 移除 r 数组开头为零的元素，直到不满足 NX.allclose(r[0], 0, rtol=1e-14) 或 r 的长度为 1
    while NX.allclose(r[0], 0, rtol=1e-14) and (r.shape[-1] > 1):
        r = r[1:]
    # 如果输入参数包含多项式对象，则将 q 和 r 转换为 poly1d 对象返回，否则返回原始数组 q 和 r
    if truepoly:
        return poly1d(q), poly1d(r)
    return q, r

# 编译正则表达式，用于查找字符串中的多项式幂运算符
_poly_mat = re.compile(r"\*\*([0-9]*)")

# 定义一个函数，将多项式字符串中的幂运算符进行处理，使输出的字符串在指定宽度内包含的字符数不超过 wrap
def _raise_power(astr, wrap=70):
    n = 0
    line1 = ''
    line2 = ''
    output = ' '
    while True:
        mat = _poly_mat.search(astr, n)
        if mat is None:
            break
        span = mat.span()
        power = mat.groups()[0]
        partstr = astr[n:span[0]]
        n = span[1]
        toadd2 = partstr + ' '*(len(power)-1)
        toadd1 = ' '*(len(partstr)-1) + power
        if ((len(line2) + len(toadd2) > wrap) or
                (len(line1) + len(toadd1) > wrap)):
            output += line1 + "\n" + line2 + "\n "
            line1 = toadd1
            line2 = toadd2
        else:
            line2 += partstr + ' '*(len(power)-1)
            line1 += ' '*(len(partstr)-1) + power
    # 将 line1 和 line2 拼接到 output 变量中，并在它们之间添加换行符
    output += line1 + "\n" + line2
    # 返回 output 变量与从索引 n 开始到末尾的 astr 字符串拼接的结果
    return output + astr[n:]
# 设置模块名为 'numpy'，用于类的装饰
@set_module('numpy')
# 定义一维多项式类 poly1d
class poly1d:
    """
    一维多项式类。

    .. note::
       这是旧的多项式 API 的一部分。从版本 1.4 开始，推荐使用 `numpy.polynomial` 中定义的新多项式 API。
       有关差异的摘要，请参阅 :doc:`过渡指南 </reference/routines.polynomials>`。

    一个便捷的类，用于封装多项式的自然操作，使这些操作可以在代码中按照习惯的方式执行（参见示例）。

    Parameters
    ----------
    c_or_r : array_like
        多项式的系数，按降幂排列；或者如果第二个参数的值为 True，则是多项式的根（使得多项式为 0 的值）。
        例如，`poly1d([1, 2, 3])` 返回表示 :math:`x^2 + 2x + 3` 的对象，而 `poly1d([1, 2, 3], True)` 返回
        表示 :math:`(x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6` 的对象。
    r : bool, optional
        如果为 True，`c_or_r` 指定多项式的根；默认为 False。
    variable : str, optional
        更改打印多项式 `p` 时使用的变量名称，从 `x` 改为 `variable`（参见示例）。

    Examples
    --------
    构造多项式 :math:`x^2 + 2x + 3`：

    >>> p = np.poly1d([1, 2, 3])
    >>> print(np.poly1d(p))
       2
    1 x + 2 x + 3

    在 :math:`x = 0.5` 处求多项式的值：

    >>> p(0.5)
    4.25

    查找多项式的根：

    >>> p.r
    array([-1.+1.41421356j, -1.-1.41421356j])
    >>> p(p.r)
    array([ -4.44089210e-16+0.j,  -4.44089210e-16+0.j]) # 可能会有所不同

    上一行中的数字表示（0, 0）的机器精度

    显示多项式的系数：

    >>> p.c
    array([1, 2, 3])

    显示多项式的阶数（移除了前导零系数）：

    >>> p.order
    2

    显示多项式中第 k 次幂的系数
    （等同于 `p.c[-(i+1)]`）：

    >>> p[1]
    2

    多项式可以进行加法、减法、乘法和除法
    （返回商和余数）：

    >>> p * p
    poly1d([ 1,  4, 10, 12,  9])

    >>> (p**3 + 4) / p
    (poly1d([ 1.,  4., 10., 12.,  9.]), poly1d([4.]))

    `asarray(p)` 返回系数数组，因此多项式可以在接受数组的所有函数中使用：

    >>> p**2 # 多项式的平方
    poly1d([ 1,  4, 10, 12,  9])

    >>> np.square(p) # 各系数的平方
    array([1, 4, 9])

    在多项式的字符串表示中可以修改使用的变量，
    使用 `variable` 参数：

    >>> p = np.poly1d([1,2,3], variable='z')
    >>> print(p)
       2
    1 z + 2 z + 3

    从根构造多项式：

    >>> np.poly1d([1, 2], True)
    poly1d([ 1., -3.,  2.])

    这与以下方式获得的多项式相同：

    >>> np.poly1d([1, -1]) * np.poly1d([1, -2])
    poly1d([ 1, -3,  2])

    """
    # 禁用对象的哈希功能
    __hash__ = None

    # 设置属性
    @property
    # 返回多项式的系数
    def coeffs(self):
        """ The polynomial coefficients """
        return self._coeffs

    # 设置多项式的系数，禁止直接设置属性，只读
    @coeffs.setter
    def coeffs(self, value):
        # 允许这样做可以使 p.coeffs *= 2 成为合法操作
        if value is not self._coeffs:
            raise AttributeError("Cannot set attribute")

    # 返回多项式的变量名
    @property
    def variable(self):
        """ The name of the polynomial variable """
        return self._variable

    # 返回多项式的阶数或次数
    @property
    def order(self):
        """ The order or degree of the polynomial """
        return len(self._coeffs) - 1

    # 返回多项式的根
    @property
    def roots(self):
        """ The roots of the polynomial, where self(x) == 0 """
        return roots(self._coeffs)

    # 内部属性 _coeffs 需要在 __dict__['coeffs'] 中备份以便 scipy 正常工作
    @property
    def _coeffs(self):
        return self.__dict__['coeffs']
    @_coeffs.setter
    def _coeffs(self, coeffs):
        self.__dict__['coeffs'] = coeffs

    # 别名属性设置
    r = roots
    c = coef = coefficients = coeffs
    o = order

    # 初始化方法，根据输入的 c_or_r（系数或者另一个 poly1d 对象），r（是否是根），以及可选的变量名 variable 来初始化多项式对象
    def __init__(self, c_or_r, r=False, variable=None):
        if isinstance(c_or_r, poly1d):
            # 如果输入的是另一个 poly1d 对象，则复制其变量名和系数
            self._variable = c_or_r._variable
            self._coeffs = c_or_r._coeffs

            # 如果源对象有额外的属性，在未来的版本中将不会被复制
            if set(c_or_r.__dict__) - set(self.__dict__):
                msg = ("In the future extra properties will not be copied "
                       "across when constructing one poly1d from another")
                warnings.warn(msg, FutureWarning, stacklevel=2)
                self.__dict__.update(c_or_r.__dict__)

            # 如果指定了变量名，则使用指定的变量名
            if variable is not None:
                self._variable = variable
            return
        
        # 如果指定了 r=True，则将 c_or_r 视为根来构造 poly 对象
        if r:
            c_or_r = poly(c_or_r)
        
        # 将 c_or_r 至少转换为一维数组
        c_or_r = atleast_1d(c_or_r)
        
        # 如果多项式不是一维的，则抛出异常
        if c_or_r.ndim > 1:
            raise ValueError("Polynomial must be 1d only.")
        
        # 去除系数数组末尾的零
        c_or_r = trim_zeros(c_or_r, trim='f')
        
        # 如果系数数组长度为零，则将其重置为只包含一个元素 0 的数组
        if len(c_or_r) == 0:
            c_or_r = NX.array([0], dtype=c_or_r.dtype)
        
        # 将处理后的系数数组设置为多项式的系数
        self._coeffs = c_or_r
        
        # 如果未指定变量名，默认使用 'x'
        if variable is None:
            variable = 'x'
        
        # 设置多项式的变量名
        self._variable = variable

    # 将对象转换为数组
    def __array__(self, t=None, copy=None):
        if t:
            return NX.asarray(self.coeffs, t, copy=copy)
        else:
            return NX.asarray(self.coeffs, copy=copy)

    # 返回多项式的字符串表示形式
    def __repr__(self):
        vals = repr(self.coeffs)
        vals = vals[6:-1]
        return "poly1d(%s)" % vals

    # 返回多项式的阶数或次数
    def __len__(self):
        return self.order
    # 返回多项式的字符串表示形式
    def __str__(self):
        # 初始化字符串表示为 "0"
        thestr = "0"
        # 获取多项式的变量名称
        var = self.variable

        # 去除系数数组中前导的零元素
        coeffs = self.coeffs[NX.logical_or.accumulate(self.coeffs != 0)]
        # 系数数组的长度减一，即多项式的最高次数
        N = len(coeffs) - 1

        # 定义格式化浮点数的函数
        def fmt_float(q):
            s = '%.4g' % q
            # 如果以 '.0000' 结尾，则去掉
            if s.endswith('.0000'):
                s = s[:-5]
            return s

        # 遍历系数数组
        for k, coeff in enumerate(coeffs):
            # 判断系数是否为复数
            if not iscomplex(coeff):
                # 如果系数为实数，则格式化实部
                coefstr = fmt_float(real(coeff))
            elif real(coeff) == 0:
                # 如果实部为零，则格式化虚部，并加上 'j' 后缀
                coefstr = '%sj' % fmt_float(imag(coeff))
            else:
                # 否则，格式化为复数形式
                coefstr = '(%s + %sj)' % (fmt_float(real(coeff)),
                                          fmt_float(imag(coeff)))

            # 计算当前项的幂次
            power = (N - k)
            if power == 0:
                # 如果幂次为零
                if coefstr != '0':
                    newstr = '%s' % (coefstr,)
                else:
                    # 如果系数为零且不是首项，则为空字符串
                    if k == 0:
                        newstr = '0'
                    else:
                        newstr = ''
            elif power == 1:
                # 如果幂次为一
                if coefstr == '0':
                    newstr = ''
                elif coefstr == 'b':
                    newstr = var
                else:
                    newstr = '%s %s' % (coefstr, var)
            else:
                # 对于其他幂次
                if coefstr == '0':
                    newstr = ''
                elif coefstr == 'b':
                    newstr = '%s**%d' % (var, power,)
                else:
                    newstr = '%s %s**%d' % (coefstr, var, power)

            # 拼接当前项到结果字符串
            if k > 0:
                if newstr != '':
                    if newstr.startswith('-'):
                        thestr = "%s - %s" % (thestr, newstr[1:])
                    else:
                        thestr = "%s + %s" % (thestr, newstr)
            else:
                thestr = newstr
        
        # 返回最终结果字符串，调用 _raise_power 函数处理可能存在的次方符号
        return _raise_power(thestr)

    # 对象可调用时的行为，计算多项式在给定值处的值
    def __call__(self, val):
        return polyval(self.coeffs, val)

    # 多项式取负运算
    def __neg__(self):
        return poly1d(-self.coeffs)

    # 多项式取正运算
    def __pos__(self):
        return self

    # 多项式乘法运算（左乘）
    def __mul__(self, other):
        # 如果 other 是标量，则与系数数组相乘
        if isscalar(other):
            return poly1d(self.coeffs * other)
        else:
            # 否则将 other 转换为 poly1d 对象，再进行多项式乘法
            other = poly1d(other)
            return poly1d(polymul(self.coeffs, other.coeffs))

    # 多项式乘法运算（右乘）
    def __rmul__(self, other):
        # 如果 other 是标量，则与系数数组相乘
        if isscalar(other):
            return poly1d(other * self.coeffs)
        else:
            # 否则将 other 转换为 poly1d 对象，再进行多项式乘法
            other = poly1d(other)
            return poly1d(polymul(self.coeffs, other.coeffs))

    # 多项式加法运算（左加）
    def __add__(self, other):
        # 将 other 转换为 poly1d 对象，然后进行多项式加法
        other = poly1d(other)
        return poly1d(polyadd(self.coeffs, other.coeffs))

    # 多项式加法运算（右加）
    def __radd__(self, other):
        # 将 other 转换为 poly1d 对象，然后进行多项式加法
        other = poly1d(other)
        return poly1d(polyadd(self.coeffs, other.coeffs))

    # 多项式乘方运算
    def __pow__(self, val):
        # 如果 val 不是标量或不是非负整数或小于零，则抛出 ValueError 异常
        if not isscalar(val) or int(val) != val or val < 0:
            raise ValueError("Power to non-negative integers only.")
        # 初始化结果为 [1]
        res = [1]
        # 执行 val 次多项式乘法
        for _ in range(val):
            res = polymul(self.coeffs, res)
        return poly1d(res)
    # 定义另一个多项式对象与当前多项式对象的减法操作
    def __sub__(self, other):
        # 将输入的参数转换为 poly1d 对象
        other = poly1d(other)
        # 返回一个新的 poly1d 对象，其系数为两个多项式系数相减的结果
        return poly1d(polysub(self.coeffs, other.coeffs))

    # 定义另一个多项式对象与当前多项式对象的反向减法操作
    def __rsub__(self, other):
        # 将输入的参数转换为 poly1d 对象
        other = poly1d(other)
        # 返回一个新的 poly1d 对象，其系数为输入多项式系数与当前多项式系数相减的结果
        return poly1d(polysub(other.coeffs, self.coeffs))

    # 定义当前多项式对象除以另一个多项式对象或标量的除法操作
    def __div__(self, other):
        # 如果输入参数是标量，则返回一个新的 poly1d 对象，其系数为当前多项式系数除以标量的结果
        if isscalar(other):
            return poly1d(self.coeffs / other)
        else:
            # 将输入参数转换为 poly1d 对象，然后返回一个新的 poly1d 对象，其系数为当前多项式对象与输入多项式对象相除的结果
            other = poly1d(other)
            return polydiv(self, other)

    # 定义当前多项式对象与另一个多项式对象的真除法操作
    __truediv__ = __div__

    # 定义另一个多项式对象除以当前多项式对象或标量的反向除法操作
    def __rdiv__(self, other):
        # 如果输入参数是标量，则返回一个新的 poly1d 对象，其系数为标量除以当前多项式系数的结果
        if isscalar(other):
            return poly1d(other / self.coeffs)
        else:
            # 将输入参数转换为 poly1d 对象，然后返回一个新的 poly1d 对象，其系数为输入多项式对象与当前多项式对象相除的结果
            other = poly1d(other)
            return polydiv(other, self)

    # 定义另一个多项式对象与当前多项式对象的真反向除法操作
    __rtruediv__ = __rdiv__

    # 定义当前多项式对象与另一个多项式对象的相等性比较操作
    def __eq__(self, other):
        # 如果输入参数不是 poly1d 对象，则返回 Not Implemented
        if not isinstance(other, poly1d):
            return NotImplemented
        # 如果当前多项式系数形状与输入多项式系数形状不相等，则返回 False
        if self.coeffs.shape != other.coeffs.shape:
            return False
        # 返回当前多项式系数与输入多项式系数逐元素比较的结果
        return (self.coeffs == other.coeffs).all()

    # 定义当前多项式对象与另一个多项式对象的不等性比较操作
    def __ne__(self, other):
        # 如果输入参数不是 poly1d 对象，则返回 Not Implemented
        if not isinstance(other, poly1d):
            return NotImplemented
        # 返回当前多项式对象与输入多项式对象相等性比较的否定结果
        return not self.__eq__(other)


    # 定义通过索引访问多项式对象的操作
    def __getitem__(self, val):
        # 计算索引对应的系数在多项式系数数组中的位置
        ind = self.order - val
        # 如果索引超出多项式次数范围，则返回 0
        if val > self.order:
            return self.coeffs.dtype.type(0)
        # 如果索引为负数，则返回 0
        if val < 0:
            return self.coeffs.dtype.type(0)
        # 返回索引对应的系数值
        return self.coeffs[ind]

    # 定义通过索引设置多项式对象的操作
    def __setitem__(self, key, val):
        # 计算索引对应的系数在多项式系数数组中的位置
        ind = self.order - key
        # 如果索引为负数，则引发异常
        if key < 0:
            raise ValueError("Does not support negative powers.")
        # 如果索引超出当前多项式次数范围，则在多项式系数数组前补零，使其能容纳索引位置对应的系数
        if key > self.order:
            zr = NX.zeros(key - self.order, self.coeffs.dtype)
            self._coeffs = NX.concatenate((zr, self.coeffs))
            ind = 0
        # 设置索引位置对应的系数值
        self._coeffs[ind] = val
        return

    # 定义迭代多项式对象的操作
    def __iter__(self):
        # 返回多项式系数数组的迭代器
        return iter(self.coeffs)

    # 定义多项式对象的积分操作
    def integ(self, m=1, k=0):
        """
        Return an antiderivative (indefinite integral) of this polynomial.

        Refer to `polyint` for full documentation.

        See Also
        --------
        polyint : equivalent function

        """
        # 返回当前多项式对象的积分多项式对象
        return poly1d(polyint(self.coeffs, m=m, k=k))

    # 定义多项式对象的求导操作
    def deriv(self, m=1):
        """
        Return a derivative of this polynomial.

        Refer to `polyder` for full documentation.

        See Also
        --------
        polyder : equivalent function

        """
        # 返回当前多项式对象的导数多项式对象
        return poly1d(polyder(self.coeffs, m=m))
# 在模块导入时执行的操作

# 设置警告过滤器，始终显示 RankWarning 警告
warnings.simplefilter('always', RankWarning)
```