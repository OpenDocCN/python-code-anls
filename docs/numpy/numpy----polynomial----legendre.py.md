# `.\numpy\numpy\polynomial\legendre.py`

```py
"""
==================================================
Legendre Series (:mod:`numpy.polynomial.legendre`)
==================================================

This module provides a number of objects (mostly functions) useful for
dealing with Legendre series, including a `Legendre` class that
encapsulates the usual arithmetic operations.  (General information
on how this module represents and works with such polynomials is in the
docstring for its "parent" sub-package, `numpy.polynomial`).

Classes
-------
.. autosummary::
   :toctree: generated/

    Legendre

Constants
---------

.. autosummary::
   :toctree: generated/

   legdomain
   legzero
   legone
   legx

Arithmetic
----------

.. autosummary::
   :toctree: generated/

   legadd
   legsub
   legmulx
   legmul
   legdiv
   legpow
   legval
   legval2d
   legval3d
   leggrid2d
   leggrid3d

Calculus
--------

.. autosummary::
   :toctree: generated/

   legder
   legint

Misc Functions
--------------

.. autosummary::
   :toctree: generated/

   legfromroots
   legroots
   legvander
   legvander2d
   legvander3d
   leggauss
   legweight
   legcompanion
   legfit
   legtrim
   legline
   leg2poly
   poly2leg

See also
--------
numpy.polynomial

"""
import numpy as np
import numpy.linalg as la
from numpy.lib.array_utils import normalize_axis_index

from . import polyutils as pu
from ._polybase import ABCPolyBase

__all__ = [
    'legzero', 'legone', 'legx', 'legdomain', 'legline', 'legadd',
    'legsub', 'legmulx', 'legmul', 'legdiv', 'legpow', 'legval', 'legder',
    'legint', 'leg2poly', 'poly2leg', 'legfromroots', 'legvander',
    'legfit', 'legtrim', 'legroots', 'Legendre', 'legval2d', 'legval3d',
    'leggrid2d', 'leggrid3d', 'legvander2d', 'legvander3d', 'legcompanion',
    'leggauss', 'legweight']

# 使用pu模块中的trimcoef函数，将其赋值给全局变量legtrim
legtrim = pu.trimcoef


def poly2leg(pol):
    """
    Convert a polynomial to a Legendre series.

    Convert an array representing the coefficients of a polynomial (relative
    to the "standard" basis) ordered from lowest degree to highest, to an
    array of the coefficients of the equivalent Legendre series, ordered
    from lowest to highest degree.

    Parameters
    ----------
    pol : array_like
        1-D array containing the polynomial coefficients

    Returns
    -------
    c : ndarray
        1-D array containing the coefficients of the equivalent Legendre
        series.

    See Also
    --------
    leg2poly

    Notes
    -----
    The easy way to do conversions between polynomial basis sets
    is to use the convert method of a class instance.

    Examples
    --------
    >>> from numpy import polynomial as P
    >>> p = P.Polynomial(np.arange(4))
    >>> p
    Polynomial([0.,  1.,  2.,  3.], domain=[-1.,  1.], window=[-1.,  1.], ...
    >>> c = P.Legendre(P.legendre.poly2leg(p.coef))
    >>> c
    Legendre([ 1.  ,  3.25,  1.  ,  0.75], domain=[-1,  1], window=[-1,  1]) # may vary

    """
    # 将输入的多项式系数转换为系列（series）
    [pol] = pu.as_series([pol])
    # 计算多项式的最高次数
    deg = len(pol) - 1
    # 初始化结果为0
    res = 0
    # 从最高次数 deg 到常数项 0 逐步计算多项式的值
    for i in range(deg, -1, -1):
        # 使用 legmulx 函数将当前结果 res 乘以 x，然后使用 legadd 函数加上 pol[i] 多项式
        res = legadd(legmulx(res), pol[i])
    # 返回计算后的多项式结果 res
    return res
def leg2poly(c):
    """
    Convert a Legendre series to a polynomial.

    Convert an array representing the coefficients of a Legendre series,
    ordered from lowest degree to highest, to an array of the coefficients
    of the equivalent polynomial (relative to the "standard" basis) ordered
    from lowest to highest degree.

    Parameters
    ----------
    c : array_like
        1-D array containing the Legendre series coefficients, ordered
        from lowest order term to highest.

    Returns
    -------
    pol : ndarray
        1-D array containing the coefficients of the equivalent polynomial
        (relative to the "standard" basis) ordered from lowest order term
        to highest.

    See Also
    --------
    poly2leg

    Notes
    -----
    The easy way to do conversions between polynomial basis sets
    is to use the convert method of a class instance.

    Examples
    --------
    >>> from numpy import polynomial as P
    >>> c = P.Legendre(range(4))
    >>> c
    Legendre([0., 1., 2., 3.], domain=[-1.,  1.], window=[-1.,  1.], symbol='x')
    >>> p = c.convert(kind=P.Polynomial)
    >>> p
    Polynomial([-1. , -3.5,  3. ,  7.5], domain=[-1.,  1.], window=[-1., ...
    >>> P.legendre.leg2poly(range(4))
    array([-1. , -3.5,  3. ,  7.5])


    """
    from .polynomial import polyadd, polysub, polymulx  # 导入多项式运算函数

    [c] = pu.as_series([c])  # 确保c是一个系列对象
    n = len(c)  # 获取系列的长度
    if n < 3:  # 如果系列长度小于3
        return c  # 直接返回系列c
    else:
        c0 = c[-2]  # 设置初始值c0为系列倒数第二项
        c1 = c[-1]  # 设置初始值c1为系列最后一项
        # i 是当前 c1 的阶数
        for i in range(n - 1, 1, -1):  # 从n-1到1逆序遍历
            tmp = c0  # 临时变量tmp为c0
            c0 = polysub(c[i - 2], (c1*(i - 1))/i)  # 更新c0为c[i-2]减去(c1*(i-1))/i
            c1 = polyadd(tmp, (polymulx(c1)*(2*i - 1))/i)  # 更新c1为tmp加上(polymulx(c1)*(2*i-1))/i
        return polyadd(c0, polymulx(c1))  # 返回c0与polymulx(c1)的和


#
# These are constant arrays are of integer type so as to be compatible
# with the widest range of other types, such as Decimal.
#

# Legendre
legdomain = np.array([-1., 1.])  # 定义Legendre多项式的定义域

# Legendre coefficients representing zero.
legzero = np.array([0])  # 定义表示零的Legendre系数数组

# Legendre coefficients representing one.
legone = np.array([1])  # 定义表示一的Legendre系数数组

# Legendre coefficients representing the identity x.
legx = np.array([0, 1])  # 定义表示x的Legendre系数数组


def legline(off, scl):
    """
    Legendre series whose graph is a straight line.

    Parameters
    ----------
    off, scl : scalars
        The specified line is given by ``off + scl*x``.

    Returns
    -------
    y : ndarray
        This module's representation of the Legendre series for
        ``off + scl*x``.

    See Also
    --------
    numpy.polynomial.polynomial.polyline
    numpy.polynomial.chebyshev.chebline
    numpy.polynomial.laguerre.lagline
    numpy.polynomial.hermite.hermline
    numpy.polynomial.hermite_e.hermeline

    Examples
    --------
    >>> import numpy.polynomial.legendre as L
    >>> L.legline(3,2)
    array([3, 2])
    >>> L.legval(-3, L.legline(3,2)) # should be -3
    -3.0

    """
    if scl != 0:
        return np.array([off, scl])  # 如果scl不等于0，返回数组[off, scl]
    else:
        return np.array([off])  # 如果scl等于0，返回数组[off]


def legfromroots(roots):
    """
    """
    Generate a Legendre series with given roots.

    The function returns the coefficients of the polynomial

    .. math:: p(x) = (x - r_0) * (x - r_1) * ... * (x - r_n),

    in Legendre form, where the :math:`r_n` are the roots specified in `roots`.
    If a zero has multiplicity n, then it must appear in `roots` n times.
    For instance, if 2 is a root of multiplicity three and 3 is a root of
    multiplicity 2, then `roots` looks something like [2, 2, 2, 3, 3]. The
    roots can appear in any order.

    If the returned coefficients are `c`, then

    .. math:: p(x) = c_0 + c_1 * L_1(x) + ... +  c_n * L_n(x)

    The coefficient of the last term is not generally 1 for monic
    polynomials in Legendre form.

    Parameters
    ----------
    roots : array_like
        Sequence containing the roots.

    Returns
    -------
    out : ndarray
        1-D array of coefficients.  If all roots are real then `out` is a
        real array, if some of the roots are complex, then `out` is complex
        even if all the coefficients in the result are real (see Examples
        below).

    See Also
    --------
    numpy.polynomial.polynomial.polyfromroots
    numpy.polynomial.chebyshev.chebfromroots
    numpy.polynomial.laguerre.lagfromroots
    numpy.polynomial.hermite.hermfromroots
    numpy.polynomial.hermite_e.hermefromroots

    Examples
    --------
    >>> import numpy.polynomial.legendre as L
    >>> L.legfromroots((-1,0,1)) # x^3 - x relative to the standard basis
    array([ 0. , -0.4,  0. ,  0.4])
    >>> j = complex(0,1)
    >>> L.legfromroots((-j,j)) # x^2 + 1 relative to the standard basis
    array([ 1.33333333+0.j,  0.00000000+0.j,  0.66666667+0.j]) # may vary

    """
    # 使用传入的 roots 参数调用内部函数 _fromroots，生成 Legendre 多项式的系数
    return pu._fromroots(legline, legmul, roots)
# 将两个 Legendre 级数相加。

# 返回两个 Legendre 级数 `c1` + `c2` 的和。参数 `c1` 和 `c2` 是按从低到高排序的系数序列，
# 比如，[1,2,3] 表示级数 ``P_0 + 2*P_1 + 3*P_2``。

# 导入必要的包 numpy.polynomial.legendre as L
# c1 和 c2：Legendre 级数系数的 1-D 数组，按从低到高排序。

# 返回
# -------
# out : ndarray
#     表示它们的和的 Legendre 级数的数组。

# See Also
# --------
# legsub, legmulx, legmul, legdiv, legpow

# Notes
# -----
# 与乘法、除法等不同，两个 Legendre 级数的和是一个 Legendre 级数（不需要将结果“重新投影”到基函数集中），
# 因此加法，就像“标准”多项式一样，只是“分量方式”。

# Examples
# --------
# >>> from numpy.polynomial import legendre as L
# >>> c1 = (1,2,3)
# >>> c2 = (3,2,1)
# >>> L.legadd(c1,c2)
# array([4.,  4.,  4.])



# 将一个 Legendre 级数从另一个中减去。

# 返回两个 Legendre 级数 `c1` - `c2` 的差。系数序列是按从低到高排序的，
# 比如，[1,2,3] 表示级数 ``P_0 + 2*P_1 + 3*P_2``。

# Parameters
# ----------
# c1, c2 : array_like
#     1-D 数组的 Legendre 级数系数，按从低到高排序。

# Returns
# -------
# out : ndarray
#     表示它们的差的 Legendre 级数系数。

# See Also
# --------
# legadd, legmulx, legmul, legdiv, legpow

# Notes
# -----
# 与乘法、除法等不同，两个 Legendre 级数的差是一个 Legendre 级数（不需要将结果“重新投影”到基函数集中），
# 因此减法，就像“标准”多项式一样，只是“分量方式”。

# Examples
# --------
# >>> from numpy.polynomial import legendre as L
# >>> c1 = (1,2,3)
# >>> c2 = (3,2,1)
# >>> L.legsub(c1,c2)
# array([-2.,  0.,  2.])
# >>> L.legsub(c2,c1) # -C.legsub(c1,c2)
# array([ 2.,  0., -2.])



# 将 Legendre 级数乘以 x。

# 将 Legendre 级数 `c` 乘以 x，其中 x 是自变量。

# Parameters
# ----------
# c : array_like
#     按从低到高排序的 1-D 数组的 Legendre 级数系数。

# Returns
# -------
# out : ndarray
#     表示乘法结果的数组。

# See Also
# --------
# legadd, legsub, legmul, legdiv, legpow

# Notes
# -----
# 乘法使用 Legendre 多项式的递归关系形式

# .. math::

#   xP_i(x) = ((i + 1)*P_{i + 1}(x) + i*P_{i - 1}(x))/(2i + 1)

# Examples
# --------
# >>> from numpy.polynomial import legendre as L
# >>> L.legmulx([1,2,3])
    array([ 0.66666667, 2.2, 1.33333333, 1.8]) # may vary

    """
    # 创建一个包含浮点数的数组，实际数值可能会有所不同
    # 这里的数组表示可能是某个算法或计算过程中的输出结果
    # 数组的具体含义和用途需要结合上下文来理解
    # 如果这是函数的返回值，则这个数组将被返回给调用方
    # 可能需要根据上下文进一步推断数组的具体作用和含义
    # 如果这个数组作为某个更大计算过程的一部分，可能需要查看上下文以获取更多信息
    [c] = pu.as_series([c])
    # 创建一个包含单个元素的列表，元素是经过处理的输入参数 c 的序列
    # 如果输入参数 c 是一个标量，将其转换为一个序列以便进行后续处理
    # 如果 c 是一个序列，则保持不变
    # 本行代码确保了后续操作始终能够处理 c 作为序列的情况

    # 需要特殊处理的情况是零序列
    if len(c) == 1 and c[0] == 0:
        # 如果 c 的长度为 1，并且唯一的元素是 0，则直接返回 c
        return c

    # 创建一个与 c 长度加一相同的空数组，数组的数据类型与 c 相同
    prd = np.empty(len(c) + 1, dtype=c.dtype)
    # 将第一个元素设置为 c 的第一个元素乘以 0
    prd[0] = c[0]*0
    # 将第二个元素设置为 c 的第一个元素
    prd[1] = c[0]
    # 遍历 c 中的每个元素，计算 prd 数组中对应的值
    for i in range(1, len(c)):
        j = i + 1
        k = i - 1
        s = i + j
        # 计算 prd 数组的第 j 个元素值
        prd[j] = (c[i]*j)/s
        # 将 c 的第 i 个元素乘以 i/s 加到 prd 数组的第 k 个元素上
        prd[k] += (c[i]*i)/s
    # 返回计算结果数组 prd
    return prd
    """
def legdiv(c1, c2):
    """
    Divide one Legendre series by another.

    Returns the quotient-with-remainder of two Legendre series
    `c1` / `c2`.  The arguments are sequences of coefficients from lowest
    order "term" to highest, e.g., [1,2,3] represents the series
    ``P_0 + 2*P_1 + 3*P_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of Legendre series coefficients ordered from low to
        high.

    Returns
    -------
    quo, rem : ndarrays
        Of Legendre series coefficients representing the quotient and
        remainder.

    See Also
    --------
    legadd, legsub, legmulx, legmul, legpow

    Notes
    -----
    In general, the (polynomial) division of one Legendre series by another
    results in quotient and remainder terms that are not in the Legendre
    polynomial basis set.  Thus, to express these results as a Legendre
    series, it is necessary to "reproject" the results onto the Legendre
    basis set, which may produce "unintuitive" (but correct) results; see
    Examples section below.

    Examples
    --------
    >>> from numpy.polynomial import legendre as L
    """
    # s1, s2 are trimmed copies
    [c1, c2] = pu.as_series([c1, c2])

    # Determine the smaller and larger series
    if len(c1) > len(c2):
        c = c2
        xs = c1
    else:
        c = c1
        xs = c2

    # Handle division for different lengths of c
    if len(c) == 1:
        c0 = c[0] / xs
        c1 = 0
    elif len(c) == 2:
        c0 = c[0] / xs
        c1 = c[1] / xs
    else:
        nd = len(c)
        c0 = c[-2] / xs
        c1 = c[-1] / xs
        for i in range(3, len(c) + 1):
            tmp = c0
            nd = nd - 1
            c0 = legsub(c[-i] / xs, (c1 * (nd - 1)) / nd)
            c1 = legadd(tmp, (legmulx(c1) * (2 * nd - 1)) / nd)

    # Return the result of the division as a Legendre series
    return legadd(c0, legmulx(c1))
    # 定义两个元组 c1 和 c2，表示为 (1,2,3) 和 (3,2,1)
    >>> c1 = (1,2,3)
    >>> c2 = (3,2,1)
    # 调用 L 对象的 legdiv 方法，对 c1 和 c2 进行 Legendre 多项式的除法运算，返回商和余数
    >>> L.legdiv(c1,c2) # quotient "intuitive," remainder not
    # 返回结果为 (array([3.]), array([-8., -4.]))，第一个数组是商，第二个数组是余数
    (array([3.]), array([-8., -4.]))
    # 修改 c2 的值为 (0,1,2,3)，重新调用 legdiv 方法进行除法运算
    >>> c2 = (0,1,2,3)
    # 返回结果为 (array([-0.07407407,  1.66666667]), array([-1.03703704, -2.51851852]))，可能会有变化
    >>> L.legdiv(c2,c1) # neither "intuitive"
    # 返回结果分别是商和余数，注意这里的结果可能会有所不同
    (array([-0.07407407,  1.66666667]), array([-1.03703704, -2.51851852])) # may vary

    """
    # 调用 pu 对象的 _div 方法，进行 Legendre 多项式的除法运算，传入的参数是 legmul, c1, c2
    return pu._div(legmul, c1, c2)
def legder(c, m=1, scl=1, axis=0):
    """
    Differentiate a Legendre series.

    Returns the Legendre series coefficients `c` differentiated `m` times
    along `axis`.  At each iteration the result is multiplied by `scl` (the
    scaling factor is for use in a linear change of variable). The argument
    `c` is an array of coefficients from low to high degree along each
    axis, e.g., [1,2,3] represents the series ``1*L_0 + 2*L_1 + 3*L_2``
    while [[1,2],[1,2]] represents ``1*L_0(x)*L_0(y) + 1*L_1(x)*L_0(y) +
    2*L_0(x)*L_1(y) + 2*L_1(x)*L_1(y)`` if axis=0 is ``x`` and axis=1 is
    ``y``.

    Parameters
    ----------
    c : array_like
        Array of Legendre series coefficients. If c is multidimensional the
        different axis correspond to different variables with the degree in
        each axis given by the corresponding index.
    m : int, optional
        Number of derivatives taken, must be non-negative. (Default: 1)
    scl : scalar, optional
        Each differentiation is multiplied by `scl`.  The end result is
        multiplication by ``scl**m``.  This is for use in a linear change of
        variable. (Default: 1)
    axis : int, optional
        Axis over which the derivative is taken. (Default: 0).

        .. versionadded:: 1.7.0

    Returns
    -------
    der : ndarray
        Legendre series of the derivative.

    See Also
    --------
    legint

    Notes
    -----
    In general, the result of differentiating a Legendre series does not
    resemble the same operation on a power series. Thus the result of this
    function may be "unintuitive," albeit correct; see Examples section
    below.

    Examples
    --------
    >>> from numpy.polynomial import legendre as L
    >>> c = (1,2,3,4)
    >>> L.legder(c)
    array([  6.,   9.,  20.])
    >>> L.legder(c, 3)
    array([60.])
    >>> L.legder(c, scl=-1)
    array([ -6.,  -9., -20.])
    >>> L.legder(c, 2,-1)
    array([  9.,  60.])

    """
    # Ensure `c` is at least 1-dimensional and make a copy
    c = np.array(c, ndmin=1, copy=True)
    # Convert `c` to double precision if it is of a different type
    if c.dtype.char in '?bBhHiIlLqQpP':
        c = c.astype(np.double)
    # Convert `m` to integer; raise an error if not possible
    cnt = pu._as_int(m, "the order of derivation")
    # 将 axis 转换为整数索引，如果无效则引发异常
    iaxis = pu._as_int(axis, "the axis")
    
    # 如果 cnt 小于 0，则抛出数值错误异常
    if cnt < 0:
        raise ValueError("The order of derivation must be non-negative")
    
    # 根据数组 c 的维度，规范化 iaxis 的索引
    iaxis = normalize_axis_index(iaxis, c.ndim)
    
    # 如果 cnt 等于 0，直接返回数组 c
    if cnt == 0:
        return c
    
    # 将 c 数组的轴 iaxis 移动到索引 0 的位置
    c = np.moveaxis(c, iaxis, 0)
    
    # 获取数组 c 的长度
    n = len(c)
    
    # 如果 cnt 大于等于 n，则将 c 的第一个元素置零并返回
    if cnt >= n:
        c = c[:1]*0
    else:
        # 对于 cnt 次导数计算
        for i in range(cnt):
            n = n - 1  # 更新导数的阶数
            c *= scl  # 对 c 应用缩放因子
            # 创建导数数组 der
            der = np.empty((n,) + c.shape[1:], dtype=c.dtype)
            for j in range(n, 2, -1):
                der[j - 1] = (2*j - 1)*c[j]  # 计算高阶导数
                c[j - 2] += c[j]  # 更新 c 数组
            if n > 1:
                der[1] = 3*c[2]  # 计算第二阶导数
            der[0] = c[1]  # 计算一阶导数
            c = der  # 更新 c 数组为新的导数数组
    
    # 将 c 数组的索引 0 移回到 iaxis 轴的位置
    c = np.moveaxis(c, 0, iaxis)
    
    # 返回计算结果数组 c
    return c
# 定义函数 `legint`，用于对勒让德级数进行积分操作
def legint(c, m=1, k=[], lbnd=0, scl=1, axis=0):
    """
    Integrate a Legendre series.

    Returns the Legendre series coefficients `c` integrated `m` times from
    `lbnd` along `axis`. At each iteration the resulting series is
    **multiplied** by `scl` and an integration constant, `k`, is added.
    The scaling factor is for use in a linear change of variable.  ("Buyer
    beware": note that, depending on what one is doing, one may want `scl`
    to be the reciprocal of what one might expect; for more information,
    see the Notes section below.)  The argument `c` is an array of
    coefficients from low to high degree along each axis, e.g., [1,2,3]
    represents the series ``L_0 + 2*L_1 + 3*L_2`` while [[1,2],[1,2]]
    represents ``1*L_0(x)*L_0(y) + 1*L_1(x)*L_0(y) + 2*L_0(x)*L_1(y) +
    2*L_1(x)*L_1(y)`` if axis=0 is ``x`` and axis=1 is ``y``.

    Parameters
    ----------
    c : array_like
        Array of Legendre series coefficients. If c is multidimensional the
        different axis correspond to different variables with the degree in
        each axis given by the corresponding index.
    m : int, optional
        Order of integration, must be positive. (Default: 1)
    k : {[], list, scalar}, optional
        Integration constant(s).  The value of the first integral at
        ``lbnd`` is the first value in the list, the value of the second
        integral at ``lbnd`` is the second value, etc.  If ``k == []`` (the
        default), all constants are set to zero.  If ``m == 1``, a single
        scalar can be given instead of a list.
    lbnd : scalar, optional
        The lower bound of the integral. (Default: 0)
    scl : scalar, optional
        Following each integration the result is *multiplied* by `scl`
        before the integration constant is added. (Default: 1)
    axis : int, optional
        Axis over which the integral is taken. (Default: 0).

        .. versionadded:: 1.7.0

    Returns
    -------
    S : ndarray
        Legendre series coefficient array of the integral.

    Raises
    ------
    ValueError
        If ``m < 0``, ``len(k) > m``, ``np.ndim(lbnd) != 0``, or
        ``np.ndim(scl) != 0``.

    See Also
    --------
    legder

    Notes
    -----
    Note that the result of each integration is *multiplied* by `scl`.
    Why is this important to note?  Say one is making a linear change of
    variable :math:`u = ax + b` in an integral relative to `x`.  Then
    :math:`dx = du/a`, so one will need to set `scl` equal to
    :math:`1/a` - perhaps not what one would have first thought.

    Also note that, in general, the result of integrating a C-series needs
    to be "reprojected" onto the C-series basis set.  Thus, typically,
    the result of this function is "unintuitive," albeit correct; see
    Examples section below.

    Examples
    --------
    >>> from numpy.polynomial import legendre as L
    >>> c = (1,2,3)
    >>> L.legint(c)

    """
    # 数组包含四个浮点数，可能会有变化
    array([ 0.33333333,  0.4       ,  0.66666667,  0.6       ]) # may vary
    # 使用 L 对象的 legint 方法，传入参数 c 和 3
    >>> L.legint(c, 3)
    # 返回一个数组，可能会有变化
    array([  1.66666667e-02,  -1.78571429e-02,   4.76190476e-02, # may vary
             -1.73472348e-18,   1.90476190e-02,   9.52380952e-03])
    # 使用 L 对象的 legint 方法，传入参数 c 和 k=3
    >>> L.legint(c, k=3)
     # 返回一个数组，可能会有变化
     array([ 3.33333333,  0.4       ,  0.66666667,  0.6       ]) # may vary
    # 使用 L 对象的 legint 方法，传入参数 c 和 lbnd=-2
    >>> L.legint(c, lbnd=-2)
    # 返回一个数组，可能会有变化
    array([ 7.33333333,  0.4       ,  0.66666667,  0.6       ]) # may vary
    # 使用 L 对象的 legint 方法，传入参数 c 和 scl=2
    >>> L.legint(c, scl=2)
    # 返回一个数组，可能会有变化
    array([ 0.66666667,  0.8       ,  1.33333333,  1.2       ]) # may vary

    """
    # 将 c 转换成至少一维的数组，确保是副本
    c = np.array(c, ndmin=1, copy=True)
    # 如果 c 是布尔类型或者整数类型，将其转换成双精度浮点型
    if c.dtype.char in '?bBhHiIlLqQpP':
        c = c.astype(np.double)
    # 如果 k 不可迭代，将其转换成列表
    if not np.iterable(k):
        k = [k]
    # 将 m 转换成整数，表示积分的阶数
    cnt = pu._as_int(m, "the order of integration")
    # 将 axis 转换成整数，表示坐标轴
    iaxis = pu._as_int(axis, "the axis")
    # 如果积分阶数为负数，抛出异常
    if cnt < 0:
        raise ValueError("The order of integration must be non-negative")
    # 如果 k 的长度大于积分阶数，抛出异常
    if len(k) > cnt:
        raise ValueError("Too many integration constants")
    # 如果 lbnd 不是标量，抛出异常
    if np.ndim(lbnd) != 0:
        raise ValueError("lbnd must be a scalar.")
    # 如果 scl 不是标量，抛出异常
    if np.ndim(scl) != 0:
        raise ValueError("scl must be a scalar.")
    # 根据 c 的维度和 axis 规范化 iaxis
    iaxis = normalize_axis_index(iaxis, c.ndim)

    # 如果阶数为 0，直接返回 c
    if cnt == 0:
        return c

    # 将 c 沿着轴移动到第一个位置
    c = np.moveaxis(c, iaxis, 0)
    # 将 k 转换成列表，并在末尾补充 0 至 cnt 长度
    k = list(k) + [0]*(cnt - len(k))
    for i in range(cnt):
        n = len(c)
        c *= scl
        if n == 1 and np.all(c[0] == 0):
            c[0] += k[i]
        else:
            tmp = np.empty((n + 1,) + c.shape[1:], dtype=c.dtype)
            tmp[0] = c[0]*0
            tmp[1] = c[0]
            if n > 1:
                tmp[2] = c[1]/3
            for j in range(2, n):
                t = c[j]/(2*j + 1)
                tmp[j + 1] = t
                tmp[j - 1] -= t
            tmp[0] += k[i] - legval(lbnd, tmp)
            c = tmp
    # 将 c 沿着第一个轴移动到 iaxis 轴上
    c = np.moveaxis(c, 0, iaxis)
    # 返回 c
    return c
# 定义函数，用于在给定点 x 处评估 Legendre 级数
def legval(x, c, tensor=True):
    """
    Evaluate a Legendre series at points x.

    If `c` is of length ``n + 1``, this function returns the value:

    .. math:: p(x) = c_0 * L_0(x) + c_1 * L_1(x) + ... + c_n * L_n(x)

    The parameter `x` is converted to an array only if it is a tuple or a
    list, otherwise it is treated as a scalar. In either case, either `x`
    or its elements must support multiplication and addition both with
    themselves and with the elements of `c`.

    If `c` is a 1-D array, then ``p(x)`` will have the same shape as `x`.  If
    `c` is multidimensional, then the shape of the result depends on the
    value of `tensor`. If `tensor` is true the shape will be c.shape[1:] +
    x.shape. If `tensor` is false the shape will be c.shape[1:]. Note that
    scalars have shape (,).

    Trailing zeros in the coefficients will be used in the evaluation, so
    they should be avoided if efficiency is a concern.

    Parameters
    ----------
    x : array_like, compatible object
        If `x` is a list or tuple, it is converted to an ndarray, otherwise
        it is left unchanged and treated as a scalar. In either case, `x`
        or its elements must support addition and multiplication with
        themselves and with the elements of `c`.
    c : array_like
        Array of coefficients ordered so that the coefficients for terms of
        degree n are contained in c[n]. If `c` is multidimensional the
        remaining indices enumerate multiple polynomials. In the two
        dimensional case the coefficients may be thought of as stored in
        the columns of `c`.
    tensor : boolean, optional
        If True, the shape of the coefficient array is extended with ones
        on the right, one for each dimension of `x`. Scalars have dimension 0
        for this action. The result is that every column of coefficients in
        `c` is evaluated for every element of `x`. If False, `x` is broadcast
        over the columns of `c` for the evaluation.  This keyword is useful
        when `c` is multidimensional. The default value is True.

        .. versionadded:: 1.7.0

    Returns
    -------
    values : ndarray, algebra_like
        The shape of the return value is described above.

    See Also
    --------
    legval2d, leggrid2d, legval3d, leggrid3d

    Notes
    -----
    The evaluation uses Clenshaw recursion, aka synthetic division.

    """
    # 将 c 转换为至少是一维的 numpy 数组，确保数据类型为浮点型
    c = np.array(c, ndmin=1, copy=None)
    # 如果 c 的数据类型为布尔型、字节型、短整型、整型或长整型，转换为双精度浮点型
    if c.dtype.char in '?bBhHiIlLqQpP':
        c = c.astype(np.double)
    # 如果 x 是元组或列表，则将其转换为 ndarray
    if isinstance(x, (tuple, list)):
        x = np.asarray(x)
    # 如果 x 是 ndarray 且 tensor=True，则将 c 的形状重塑为 c.shape + (1,)*x.ndim
    if isinstance(x, np.ndarray) and tensor:
        c = c.reshape(c.shape + (1,)*x.ndim)

    # 根据 c 的长度选择性地设置 c0 和 c1
    if len(c) == 1:
        c0 = c[0]
        c1 = 0
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        # 计算输入列表 c 的长度
        nd = len(c)
        # 获取 c 的倒数第二个元素
        c0 = c[-2]
        # 获取 c 的最后一个元素
        c1 = c[-1]
        # 循环遍历 c 的子集，从索引 3 开始到末尾
        for i in range(3, len(c) + 1):
            # 临时保存 c0 的值
            tmp = c0
            # 更新 nd 的值
            nd = nd - 1
            # 更新 c0 的值，根据公式计算
            c0 = c[-i] - (c1*(nd - 1))/nd
            # 更新 c1 的值，根据公式计算
            c1 = tmp + (c1*x*(2*nd - 1))/nd
    # 返回计算结果 c0 + c1*x
    return c0 + c1*x
# 计算二维Legendre级数在点(x, y)处的值。

def legval2d(x, y, c):
    """
    Evaluate a 2-D Legendre series at points (x, y).

    This function returns the values:

    .. math:: p(x,y) = \\sum_{i,j} c_{i,j} * L_i(x) * L_j(y)

    The parameters `x` and `y` are converted to arrays only if they are
    tuples or a lists, otherwise they are treated as scalars and they
    must have the same shape after conversion. In either case, either `x`
    and `y` or their elements must support multiplication and addition both
    with themselves and with the elements of `c`.

    If `c` is a 1-D array a one is implicitly appended to its shape to make
    it 2-D. The shape of the result will be c.shape[2:] + x.shape.

    Parameters
    ----------
    x, y : array_like, compatible objects
        The two-dimensional series is evaluated at the points ``(x, y)``,
        where `x` and `y` must have the same shape. If `x` or `y` is a list
        or tuple, it is first converted to an ndarray, otherwise it is left
        unchanged and if it isn't an ndarray it is treated as a scalar.
    c : array_like
        Array of coefficients ordered so that the coefficient of the term
        of multi-degree i,j is contained in ``c[i,j]``. If `c` has
        dimension greater than two the remaining indices enumerate multiple
        sets of coefficients.

    Returns
    -------
    values : ndarray, compatible object
        The values of the two-dimensional Legendre series at points formed
        from pairs of corresponding values from `x` and `y`.

    See Also
    --------
    legval, leggrid2d, legval3d, leggrid3d

    Notes
    -----

    .. versionadded:: 1.7.0

    """
    # 调用私有函数 `_valnd` 计算Legendre级数在给定点 `(x, y)` 处的值，使用系数 `c`
    return pu._valnd(legval, c, x, y)


def leggrid2d(x, y, c):
    """
    Evaluate a 2-D Legendre series on the Cartesian product of x and y.

    This function returns the values:

    .. math:: p(a,b) = \\sum_{i,j} c_{i,j} * L_i(a) * L_j(b)

    where the points ``(a, b)`` consist of all pairs formed by taking
    `a` from `x` and `b` from `y`. The resulting points form a grid with
    `x` in the first dimension and `y` in the second.

    The parameters `x` and `y` are converted to arrays only if they are
    tuples or a lists, otherwise they are treated as scalars. In either
    case, either `x` and `y` or their elements must support multiplication
    and addition both with themselves and with the elements of `c`.

    If `c` has fewer than two dimensions, ones are implicitly appended to
    its shape to make it 2-D. The shape of the result will be c.shape[2:] +
    x.shape + y.shape.

    Parameters
    ----------
    x, y : array_like, compatible objects
        The two-dimensional series is evaluated at the points in the
        Cartesian product of `x` and `y`.  If `x` or `y` is a list or
        tuple, it is first converted to an ndarray, otherwise it is left
        unchanged and, if it isn't an ndarray, it is treated as a scalar.
    c : array_like
        Array of coefficients ordered so that the coefficient of the term
        of multi-degree i,j is contained in ``c[i,j]``. If `c` has
        dimension greater than two the remaining indices enumerate multiple
        sets of coefficients.

    Returns
    -------
    values : ndarray, compatible object
        The values of the two-dimensional Legendre series at points formed
        from pairs of corresponding values from `x` and `y`.

    See Also
    --------
    legval, leggrid2d, legval3d, leggrid3d
    """
    c : array_like
        # 输入参数 c 是一个类数组对象，用于存储系数，按照多重度 i,j 的项的系数存储在 c[i,j] 中。
        # 如果 c 的维度大于两维，则剩余的索引用于枚举多组系数。

    Returns
    -------
    values : ndarray, compatible object
        # 返回值是一个 ndarray 数组或兼容对象，表示在笛卡尔积空间中点 (x, y) 处的二维切比雪夫级数的值。

    See Also
    --------
    legval, legval2d, legval3d, leggrid3d
        # 参见相关函数 legval, legval2d, legval3d, leggrid3d。

    Notes
    -----
    # 版本备注：自版本 1.7.0 起添加。

    """
    # 调用 pu 模块中的 _gridnd 函数，计算二维切比雪夫级数在点 (x, y) 处的值，并返回结果。
    return pu._gridnd(legval, c, x, y)
# 计算三维Legendre系列在点(x, y, z)处的值。

def legval3d(x, y, z, c):
    """
    Evaluate a 3-D Legendre series at points (x, y, z).

    This function returns the values:

    .. math:: p(x,y,z) = \\sum_{i,j,k} c_{i,j,k} * L_i(x) * L_j(y) * L_k(z)

    The parameters `x`, `y`, and `z` are converted to arrays only if
    they are tuples or a lists, otherwise they are treated as scalars and
    must have the same shape after conversion. In either case, either
    `x`, `y`, and `z` or their elements must support multiplication and
    addition both with themselves and with the elements of `c`.

    If `c` has fewer than 3 dimensions, ones are implicitly appended to its
    shape to make it 3-D. The shape of the result will be c.shape[3:] +
    x.shape.

    Parameters
    ----------
    x, y, z : array_like, compatible object
        The three-dimensional series is evaluated at the points
        ``(x, y, z)``, where `x`, `y`, and `z` must have the same shape. If
        any of `x`, `y`, or `z` is a list or tuple, it is first converted
        to an ndarray; otherwise, it is treated as a scalar.
    c : array_like
        Array of coefficients ordered so that the coefficient of the term of
        multi-degree i,j,k is contained in ``c[i,j,k]``. If `c` has dimension
        greater than 3, the remaining indices enumerate multiple sets of
        coefficients.

    Returns
    -------
    values : ndarray, compatible object
        The values of the multidimensional polynomial on points formed with
        triples of corresponding values from `x`, `y`, and `z`.

    See Also
    --------
    legval, legval2d, leggrid2d, leggrid3d

    Notes
    -----

    .. versionadded:: 1.7.0

    """
    # 调用内部函数 _valnd，传递参数 legval, c, x, y, z，并返回结果
    return pu._valnd(legval, c, x, y, z)


def leggrid3d(x, y, z, c):
    """
    Evaluate a 3-D Legendre series on the Cartesian product of x, y, and z.

    This function returns the values:

    .. math:: p(a,b,c) = \\sum_{i,j,k} c_{i,j,k} * L_i(a) * L_j(b) * L_k(c)

    where the points ``(a, b, c)`` consist of all triples formed by taking
    `a` from `x`, `b` from `y`, and `c` from `z`. The resulting points form
    a grid with `x` in the first dimension, `y` in the second, and `z` in
    the third.

    The parameters `x`, `y`, and `z` are converted to arrays only if they
    are tuples or lists; otherwise, they are treated as scalars. In
    either case, either `x`, `y`, and `z` or their elements must support
    multiplication and addition both with themselves and with the elements
    of `c`.

    If `c` has fewer than three dimensions, ones are implicitly appended to
    its shape to make it 3-D. The shape of the result will be c.shape[3:] +
    x.shape + y.shape + z.shape.

    Parameters
    ----------
    x, y, z : array_like, compatible object
        Arrays defining the grid points where the Legendre series is evaluated.
        These arrays should have compatible shapes.
    c : array_like
        Array of coefficients ordered such that the coefficient of the term of
        multi-degree i,j,k is contained in ``c[i,j,k]``. If `c` has dimension
        greater than 3, the remaining indices enumerate multiple sets of
        coefficients.

    Returns
    -------
    values : ndarray, compatible object
        The values of the multidimensional polynomial on the Cartesian product
        of `x`, `y`, and `z`.

    See Also
    --------
    legval, legval2d, leggrid2d, leggrid3d

    Notes
    -----

    .. versionadded:: 1.7.0

    """
    x, y, z : array_like, compatible objects
        三维序列在 `x`、`y` 和 `z` 的笛卡尔积点处进行评估。如果 `x`、`y` 或 `z` 是列表或元组，则首先转换为 ndarray；否则保持不变，并且如果它不是 ndarray，则视为标量。
    c : array_like
        按照次数 i,j 排列的系数数组，系数 `c[i,j]` 对应于二维多项式的系数。如果 `c` 的维度大于二，则其余的索引用于枚举多组系数。

    Returns
    -------
    values : ndarray, compatible object
        二维多项式在 `x` 和 `y` 的笛卡尔积点处的值。

    See Also
    --------
    legval, legval2d, leggrid2d, legval3d

    Notes
    -----
    在版本 1.7.0 中添加

    """
    返回 pu._gridnd(legval, c, x, y, z)


注释：
- `x, y, z : array_like, compatible objects`: 描述了函数参数 `x`, `y`, `z` 应为类似数组的兼容对象，表示三维序列在这些坐标的笛卡尔积点处进行评估。
- `c : array_like`: 描述了参数 `c` 应为类似数组的对象，按照二维多项式的次数排列，用于存储系数。
- `Returns`: 描述了函数返回一个 ndarray 或兼容对象，表示二维多项式在给定点处的值。
- `See Also`: 提供了相关函数的链接，供进一步参考。
- `Notes`: 指出该功能是在版本 1.7.0 中添加的。
def legvander(x, deg):
    """Pseudo-Vandermonde matrix of given degree.

    Returns the pseudo-Vandermonde matrix of degree `deg` and sample points
    `x`. The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., i] = L_i(x)

    where ``0 <= i <= deg``. The leading indices of `V` index the elements of
    `x` and the last index is the degree of the Legendre polynomial.

    If `c` is a 1-D array of coefficients of length ``n + 1`` and `V` is the
    array ``V = legvander(x, n)``, then ``np.dot(V, c)`` and
    ``legval(x, c)`` are the same up to roundoff. This equivalence is
    useful both for least squares fitting and for the evaluation of a large
    number of Legendre series of the same degree and sample points.

    Parameters
    ----------
    x : array_like
        Array of points. The dtype is converted to float64 or complex128
        depending on whether any of the elements are complex. If `x` is
        scalar it is converted to a 1-D array.
    deg : int
        Degree of the resulting matrix.

    Returns
    -------
    vander : ndarray
        The pseudo-Vandermonde matrix. The shape of the returned matrix is
        ``x.shape + (deg + 1,)``, where The last index is the degree of the
        corresponding Legendre polynomial.  The dtype will be the same as
        the converted `x`.

    """
    # Convert deg to an integer if possible, raises ValueError if deg < 0
    ideg = pu._as_int(deg, "deg")
    if ideg < 0:
        raise ValueError("deg must be non-negative")

    # Ensure x is converted to a 1-D array of float64 or complex128
    x = np.array(x, copy=None, ndmin=1) + 0.0

    # Calculate dimensions for the output array
    dims = (ideg + 1,) + x.shape
    dtyp = x.dtype

    # Create an uninitialized array v with dimensions and dtype calculated above
    v = np.empty(dims, dtype=dtyp)

    # Initialize the first column of v with ones
    v[0] = x*0 + 1

    # Fill the remaining columns using forward recursion
    if ideg > 0:
        v[1] = x
        for i in range(2, ideg + 1):
            v[i] = (v[i-1]*x*(2*i - 1) - v[i-2]*(i - 1))/i

    # Move the first axis to the last axis of v and return
    return np.moveaxis(v, 0, -1)
    x, y : array_like
        Arrays of point coordinates, all of the same shape. The dtypes
        will be converted to either float64 or complex128 depending on
        whether any of the elements are complex. Scalars are converted to
        1-D arrays.
    deg : list of ints
        List of maximum degrees of the form [x_deg, y_deg].

    Returns
    -------
    vander2d : ndarray
        The shape of the returned matrix is ``x.shape + (order,)``, where
        :math:`order = (deg[0]+1)*(deg[1]+1)`.  The dtype will be the same
        as the converted `x` and `y`.

    See Also
    --------
    legvander, legvander3d, legval2d, legval3d

    Notes
    -----

    .. versionadded:: 1.7.0

    """
    # 调用 _vander_nd_flat 函数来生成 2D Vandermonde 矩阵
    return pu._vander_nd_flat((legvander, legvander), (x, y), deg)
# 返回给定三维样本点 (x, y, z) 和阶数 deg 的伪范德蒙矩阵
def legvander3d(x, y, z, deg):
    """Pseudo-Vandermonde matrix of given degrees.

    Returns the pseudo-Vandermonde matrix of degrees `deg` and sample
    points ``(x, y, z)``. If `l`, `m`, `n` are the given degrees in `x`, `y`, `z`,
    then The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., (m+1)(n+1)i + (n+1)j + k] = L_i(x)*L_j(y)*L_k(z),

    where ``0 <= i <= l``, ``0 <= j <= m``, and ``0 <= j <= n``.  The leading
    indices of `V` index the points ``(x, y, z)`` and the last index encodes
    the degrees of the Legendre polynomials.

    If ``V = legvander3d(x, y, z, [xdeg, ydeg, zdeg])``, then the columns
    of `V` correspond to the elements of a 3-D coefficient array `c` of
    shape (xdeg + 1, ydeg + 1, zdeg + 1) in the order

    .. math:: c_{000}, c_{001}, c_{002},... , c_{010}, c_{011}, c_{012},...

    and ``np.dot(V, c.flat)`` and ``legval3d(x, y, z, c)`` will be the
    same up to roundoff. This equivalence is useful both for least squares
    fitting and for the evaluation of a large number of 3-D Legendre
    series of the same degrees and sample points.

    Parameters
    ----------
    x, y, z : array_like
        Arrays of point coordinates, all of the same shape. The dtypes will
        be converted to either float64 or complex128 depending on whether
        any of the elements are complex. Scalars are converted to 1-D
        arrays.
    deg : list of ints
        List of maximum degrees of the form [x_deg, y_deg, z_deg].

    Returns
    -------
    vander3d : ndarray
        The shape of the returned matrix is ``x.shape + (order,)``, where
        :math:`order = (deg[0]+1)*(deg[1]+1)*(deg[2]+1)`.  The dtype will
        be the same as the converted `x`, `y`, and `z`.

    See Also
    --------
    legvander, legval2d, legval3d

    Notes
    -----

    .. versionadded:: 1.7.0

    """
    # 调用 _vander_nd_flat 函数计算伪范德蒙矩阵
    return pu._vander_nd_flat((legvander, legvander, legvander), (x, y, z), deg)


# 最小二乘拟合 Legendre 级数到数据
def legfit(x, y, deg, rcond=None, full=False, w=None):
    """
    Least squares fit of Legendre series to data.

    Return the coefficients of a Legendre series of degree `deg` that is the
    least squares fit to the data values `y` given at points `x`. If `y` is
    1-D the returned coefficients will also be 1-D. If `y` is 2-D multiple
    fits are done, one for each column of `y`, and the resulting
    coefficients are stored in the corresponding columns of a 2-D return.
    The fitted polynomial(s) are in the form

    .. math::  p(x) = c_0 + c_1 * L_1(x) + ... + c_n * L_n(x),

    where `n` is `deg`.

    Parameters
    ----------
    x : array_like, shape (M,)
        x-coordinates of the M sample points ``(x[i], y[i])``.
    y : array_like, shape (M,) or (M, K)
        y-coordinates of the sample points. Several data sets of sample
        points sharing the same x-coordinates can be fitted at once by
        passing in a 2D-array that contains one dataset per column.
    deg : int
        Degree of the Legendre series to fit.

    Returns
    -------
    c : ndarray, shape (deg + 1,) or (deg + 1, K)
        Coefficients of the Legendre series that minimizes the squared
        error in the least-squares sense.

    See Also
    --------
    legvander, legvander2d, legvander3d, legval, legval2d, legval3d
    """
    # 返回拟合数据的 Legendre 级数的系数
    return pu._fit((legvander, legvander, legvander), x, y, deg, rcond, full, w)
    deg : int or 1-D array_like
        # 多项式拟合的次数或次数列表。如果 `deg` 是一个整数，那么包括直到第 `deg` 项的所有项在内。
        # 对于 NumPy 版本 >= 1.11.0，可以使用整数列表指定要包括的项的次数。
    rcond : float, optional
        # 拟合过程中使用的相对条件数。忽略比最大奇异值小的奇异值的相对值。
        # 默认值为 len(x)*eps，其中 eps 是浮点数类型的相对精度，在大多数情况下约为 2e-16。
    full : bool, optional
        # 返回值的性质开关。当为 False（默认值）时，只返回系数；当为 True 时，同时返回奇异值分解的诊断信息。
    w : array_like, shape (`M`,), optional
        # 权重数组。如果不为 None，则权重 `w[i]` 适用于 `x[i]` 处的未平方残差 `y[i] - y_hat[i]`。
        # 理想情况下，应选择使得乘积 `w[i]*y[i]` 的误差具有相同的方差的权重。
        # 当使用逆方差加权时，使用 `w[i] = 1/sigma(y[i])`。默认值为 None。

        .. versionadded:: 1.5.0

    Returns
    -------
    coef : ndarray, shape (M,) or (M, K)
        # 按从低到高顺序排列的 Legendre 系数。如果 `y` 是二维的，那么 `y` 中第 k 列的数据的系数位于返回的 `coef` 的第 k 列中。
        # 如果 `deg` 被指定为一个列表，则在拟合中未包含的项的系数在返回的 `coef` 中被设为零。

    [residuals, rank, singular_values, rcond] : list
        # 只有在 ``full == True`` 时才返回这些值

        - residuals -- 最小二乘拟合的残差平方和
        - rank -- 缩放后的 Vandermonde 矩阵的数值秩
        - singular_values -- 缩放后的 Vandermonde 矩阵的奇异值
        - rcond -- `rcond` 的值

        更多细节请参阅 `numpy.linalg.lstsq`。

    Warns
    -----
    RankWarning
        # 在最小二乘拟合中，系数矩阵的秩是不足的。只有在 ``full == False`` 时才会发出警告。
        # 可以通过以下方式关闭警告：

        >>> import warnings
        >>> warnings.simplefilter('ignore', np.exceptions.RankWarning)

    See Also
    --------
    numpy.polynomial.polynomial.polyfit
    numpy.polynomial.chebyshev.chebfit
    numpy.polynomial.laguerre.lagfit
    numpy.polynomial.hermite.hermfit
    numpy.polynomial.hermite_e.hermefit
    legval : 对 Legendre 级数进行求值。
    legvander : Legendre 级数的 Vandermonde 矩阵。
    legweight : Legendre 权重函数（= 1）。
    numpy.linalg.lstsq : 计算矩阵的最小二乘拟合。
    scipy.interpolate.UnivariateSpline : 计算样条拟合。

    Notes
    -----
    # 解决方案是 Legendre 级数 `p` 的系数，满足以下条件：
    minimizes the sum of the weighted squared errors

    .. math:: E = \\sum_j w_j^2 * |y_j - p(x_j)|^2,

    where :math:`w_j` are the weights. This problem is solved by setting up
    as the (typically) overdetermined matrix equation

    .. math:: V(x) * c = w * y,

    where `V` is the weighted pseudo Vandermonde matrix of `x`, `c` are the
    coefficients to be solved for, `w` are the weights, and `y` are the
    observed values.  This equation is then solved using the singular value
    decomposition of `V`.

    If some of the singular values of `V` are so small that they are
    neglected, then a `~exceptions.RankWarning` will be issued. This means that
    the coefficient values may be poorly determined. Using a lower order fit
    will usually get rid of the warning.  The `rcond` parameter can also be
    set to a value smaller than its default, but the resulting fit may be
    spurious and have large contributions from roundoff error.

    Fits using Legendre series are usually better conditioned than fits
    using power series, but much can depend on the distribution of the
    sample points and the smoothness of the data. If the quality of the fit
    is inadequate splines may be a good alternative.

    References
    ----------
    .. [1] Wikipedia, "Curve fitting",
           https://en.wikipedia.org/wiki/Curve_fitting

    Examples
    --------

    """
    使用Legendre多项式进行最小二乘拟合，并返回拟合结果。

    Parameters
    ----------
    legvander : callable
        函数用于计算Legendre多项式的伪Vandermonde矩阵。
    x : array_like, shape (M,)
        输入数据点的数组。
    y : array_like, shape (M,)
        观测到的数据值。
    deg : int
        拟合多项式的次数。
    rcond : float, optional
        用于判断奇异值的截断值，默认为None。
    full : bool, optional
        控制输出内容，若为True则返回完整的分解，否则返回拟合系数，默认为False。
    w : array_like, shape (M,), optional
        每个观测值的权重。

    Returns
    -------
    ndarray
        拟合系数。

    Notes
    -----
    使用Legendre多项式进行拟合通常比使用幂级数更好，但很大程度上取决于样本点的分布和数据的平滑程度。
    如果拟合质量不足，样条曲线可能是一个更好的选择。

    """
    return pu._fit(legvander, x, y, deg, rcond, full, w)
def legcompanion(c):
    """
    Return the scaled companion matrix of c.

    The basis polynomials are scaled so that the companion matrix is
    symmetric when `c` is an Legendre basis polynomial. This provides
    better eigenvalue estimates than the unscaled case and for basis
    polynomials the eigenvalues are guaranteed to be real if
    `numpy.linalg.eigvalsh` is used to obtain them.

    Parameters
    ----------
    c : array_like
        1-D array of Legendre series coefficients ordered from low to high
        degree.

    Returns
    -------
    mat : ndarray
        Scaled companion matrix of dimensions (deg, deg).

    Notes
    -----

    .. versionadded:: 1.7.0

    """
    # c is a trimmed copy
    [c] = pu.as_series([c])  # 将输入的系数数组 c 转换为一个修剪过的系列
    if len(c) < 2:
        raise ValueError('Series must have maximum degree of at least 1.')  # 如果系数数组长度小于2，抛出异常
    if len(c) == 2:
        return np.array([[-c[0]/c[1]]])  # 如果系数数组长度为2，返回一个1x1的数组

    n = len(c) - 1
    mat = np.zeros((n, n), dtype=c.dtype)  # 创建一个形状为 (n, n) 的零矩阵，数据类型与 c 的数据类型相同
    scl = 1./np.sqrt(2*np.arange(n) + 1)  # 计算一个尺度系数，用于缩放伴随矩阵的每一行
    top = mat.reshape(-1)[1::n+1]  # 获取矩阵的上三角元素
    bot = mat.reshape(-1)[n::n+1]  # 获取矩阵的下三角元素
    top[...] = np.arange(1, n)*scl[:n-1]*scl[1:n]  # 对上三角元素赋值，这些值是尺度系数的函数
    bot[...] = top  # 下三角元素与上三角元素相同
    mat[:, -1] -= (c[:-1]/c[-1])*(scl/scl[-1])*(n/(2*n - 1))  # 调整矩阵的最后一列，以便估算伴随矩阵的特征值
    return mat


def legroots(c):
    """
    Compute the roots of a Legendre series.

    Return the roots (a.k.a. "zeros") of the polynomial

    .. math:: p(x) = \\sum_i c[i] * L_i(x).

    Parameters
    ----------
    c : 1-D array_like
        1-D array of coefficients.

    Returns
    -------
    out : ndarray
        Array of the roots of the series. If all the roots are real,
        then `out` is also real, otherwise it is complex.

    See Also
    --------
    numpy.polynomial.polynomial.polyroots
    numpy.polynomial.chebyshev.chebroots
    numpy.polynomial.laguerre.lagroots
    numpy.polynomial.hermite.hermroots
    numpy.polynomial.hermite_e.hermeroots

    Notes
    -----
    The root estimates are obtained as the eigenvalues of the companion
    matrix, Roots far from the origin of the complex plane may have large
    errors due to the numerical instability of the series for such values.
    Roots with multiplicity greater than 1 will also show larger errors as
    the value of the series near such points is relatively insensitive to
    errors in the roots. Isolated roots near the origin can be improved by
    a few iterations of Newton's method.

    The Legendre series basis polynomials aren't powers of ``x`` so the
    results of this function may seem unintuitive.

    Examples
    --------
    >>> import numpy.polynomial.legendre as leg
    >>> leg.legroots((1, 2, 3, 4)) # 4L_3 + 3L_2 + 2L_1 + 1L_0, all real roots
    array([-0.85099543, -0.11407192,  0.51506735]) # may vary

    """
    # c is a trimmed copy
    [c] = pu.as_series([c])  # 将输入的系数数组 c 转换为一个修剪过的系列
    if len(c) < 2:
        return np.array([], dtype=c.dtype)  # 如果系数数组长度小于2，返回一个空数组
    if len(c) == 2:
        return np.array([-c[0]/c[1]])  # 如果系数数组长度为2，返回一个包含一个根的数组

    # rotated companion matrix reduces error
    m = legcompanion(c)[::-1,::-1]  # 计算伴随矩阵并进行旋转，以减小数值误差
    # 计算矩阵 m 的特征值
    r = la.eigvals(m)
    # 对特征值进行排序（默认按照实部进行排序）
    r.sort()
    # 返回排序后的特征值数组作为结果
    return r
# Gauss-Legendre quadrature function to compute sample points and weights for integration
def leggauss(deg):
    # Ensure deg is a positive integer; raise error if not
    ideg = pu._as_int(deg, "deg")
    if ideg <= 0:
        raise ValueError("deg must be a positive integer")

    # Initial approximation of roots using companion matrix symmetry
    c = np.array([0]*deg + [1])
    m = legcompanion(c)  # Compute companion matrix
    x = la.eigvalsh(m)   # Compute eigenvalues of the companion matrix

    # Improve roots using one application of Newton's method
    dy = legval(x, c)     # Compute polynomial value at roots
    df = legval(x, legder(c))  # Compute derivative of the polynomial at roots
    x -= dy / df          # Apply Newton's method to improve roots

    # Compute the weights, scaling to avoid numerical overflow
    fm = legval(x, c[1:])  # Compute polynomial value at improved roots
    fm /= np.abs(fm).max()  # Scale to avoid overflow
    df /= np.abs(df).max()
    w = 1 / (fm * df)     # Compute weights using the formula

    # Symmetrize weights and roots for Legendre polynomial
    w = (w + w[::-1]) / 2
    x = (x - x[::-1]) / 2

    # Scale weights to ensure correct integration value
    w *= 2. / w.sum()

    return x, w


# Function defining weight function of Legendre polynomials
def legweight(x):
    # Weight function is 1 over the interval [-1, 1]
    w = x * 0.0 + 1.0
    return w


# Legendre series class implementing various numerical methods and attributes
class Legendre(ABCPolyBase):
    """A Legendre series class.

    The Legendre class provides the standard Python numerical methods
    '+', '-', '*', '//', '%', 'divmod', '**', and '()' as well as the
    attributes and methods listed below.

    Parameters
    ----------
    coef : array_like
        Legendre coefficients in order of increasing degree, i.e.,
        ``(1, 2, 3)`` gives ``1*P_0(x) + 2*P_1(x) + 3*P_2(x)``.
    """
    # domain : (2,) array_like, optional
    #     Domain to use. The interval ``[domain[0], domain[1]]`` is mapped
    #     to the interval ``[window[0], window[1]]`` by shifting and scaling.
    #     The default value is [-1, 1].

    # window : (2,) array_like, optional
    #     Window, see `domain` for its use. The default value is [-1, 1].
    #
    #     .. versionadded:: 1.6.0

    # symbol : str, optional
    #     Symbol used to represent the independent variable in string
    #     representations of the polynomial expression, e.g. for printing.
    #     The symbol must be a valid Python identifier. Default value is 'x'.
    #
    #     .. versionadded:: 1.24

    """
    # Virtual Functions
    # 定义静态方法来实现多项式的不同操作
    _add = staticmethod(legadd)
    _sub = staticmethod(legsub)
    _mul = staticmethod(legmul)
    _div = staticmethod(legdiv)
    _pow = staticmethod(legpow)
    _val = staticmethod(legval)
    _int = staticmethod(legint)
    _der = staticmethod(legder)
    _fit = staticmethod(legfit)
    _line = staticmethod(legline)
    _roots = staticmethod(legroots)
    _fromroots = staticmethod(legfromroots)

    # Virtual properties
    # 将定义的区间数组转换为NumPy数组并赋值给domain和window属性
    domain = np.array(legdomain)
    window = np.array(legdomain)
    # 设置多项式的基函数名称为'P'
    basis_name = 'P'
```