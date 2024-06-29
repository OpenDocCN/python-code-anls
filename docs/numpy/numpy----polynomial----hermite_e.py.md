# `.\numpy\numpy\polynomial\hermite_e.py`

```
"""
===================================================================
HermiteE Series, "Probabilists" (:mod:`numpy.polynomial.hermite_e`)
===================================================================

This module provides a number of objects (mostly functions) useful for
dealing with Hermite_e series, including a `HermiteE` class that
encapsulates the usual arithmetic operations.  (General information
on how this module represents and works with such polynomials is in the
docstring for its "parent" sub-package, `numpy.polynomial`).

Classes
-------
.. autosummary::
   :toctree: generated/

   HermiteE

Constants
---------
.. autosummary::
   :toctree: generated/

   hermedomain
   hermezero
   hermeone
   hermex

Arithmetic
----------
.. autosummary::
   :toctree: generated/

   hermeadd
   hermesub
   hermemulx
   hermemul
   hermediv
   hermepow
   hermeval
   hermeval2d
   hermeval3d
   hermegrid2d
   hermegrid3d

Calculus
--------
.. autosummary::
   :toctree: generated/

   hermeder
   hermeint

Misc Functions
--------------
.. autosummary::
   :toctree: generated/

   hermefromroots
   hermeroots
   hermevander
   hermevander2d
   hermevander3d
   hermegauss
   hermeweight
   hermecompanion
   hermefit
   hermetrim
   hermeline
   herme2poly
   poly2herme

See also
--------
`numpy.polynomial`

"""
import numpy as np
import numpy.linalg as la
from numpy.lib.array_utils import normalize_axis_index

from . import polyutils as pu
from ._polybase import ABCPolyBase

__all__ = [
    'hermezero', 'hermeone', 'hermex', 'hermedomain', 'hermeline',
    'hermeadd', 'hermesub', 'hermemulx', 'hermemul', 'hermediv',
    'hermepow', 'hermeval', 'hermeder', 'hermeint', 'herme2poly',
    'poly2herme', 'hermefromroots', 'hermevander', 'hermefit', 'hermetrim',
    'hermeroots', 'HermiteE', 'hermeval2d', 'hermeval3d', 'hermegrid2d',
    'hermegrid3d', 'hermevander2d', 'hermevander3d', 'hermecompanion',
    'hermegauss', 'hermeweight']

# Alias for the trimcoef function from polyutils
hermetrim = pu.trimcoef


def poly2herme(pol):
    """
    poly2herme(pol)

    Convert a polynomial to a Hermite series.

    Convert an array representing the coefficients of a polynomial (relative
    to the "standard" basis) ordered from lowest degree to highest, to an
    array of the coefficients of the equivalent Hermite series, ordered
    from lowest to highest degree.

    Parameters
    ----------
    pol : array_like
        1-D array containing the polynomial coefficients

    Returns
    -------
    c : ndarray
        1-D array containing the coefficients of the equivalent Hermite
        series.

    See Also
    --------
    herme2poly

    Notes
    -----
    The easy way to do conversions between polynomial basis sets
    is to use the convert method of a class instance.

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import poly2herme
    >>> poly2herme(np.arange(4))
    array([  2.,  10.,   2.,   3.])

    """
    [pol] = pu.as_series([pol])  # Ensure pol is a 1-D polynomial array
    deg = len(pol) - 1  # Degree of the polynomial
    res = 0  # Initialize result variable as 0
    # 从最高次数 deg 开始到常数项 0 的范围，逐次进行多项式运算
    for i in range(deg, -1, -1):
        # 调用 hermemulx 函数，将 res 与 x 的 Hermite 多项式相乘，并将结果赋给 res
        res = hermeadd(hermemulx(res), pol[i])
    # 返回最终的多项式计算结果 res
    return res
def herme2poly(c):
    """
    Convert a Hermite series to a polynomial.

    Convert an array representing the coefficients of a Hermite series,
    ordered from lowest degree to highest, to an array of the coefficients
    of the equivalent polynomial (relative to the "standard" basis) ordered
    from lowest to highest degree.

    Parameters
    ----------
    c : array_like
        1-D array containing the Hermite series coefficients, ordered
        from lowest order term to highest.

    Returns
    -------
    pol : ndarray
        1-D array containing the coefficients of the equivalent polynomial
        (relative to the "standard" basis) ordered from lowest order term
        to highest.

    See Also
    --------
    poly2herme

    Notes
    -----
    The easy way to do conversions between polynomial basis sets
    is to use the convert method of a class instance.

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import herme2poly
    >>> herme2poly([  2.,  10.,   2.,   3.])
    array([0.,  1.,  2.,  3.])

    """
    # 导入多项式操作函数
    from .polynomial import polyadd, polysub, polymulx
    # 将输入的系数数组转换为多项式系数表示
    [c] = pu.as_series([c])
    # 获取系数数组的长度
    n = len(c)
    # 如果长度为1或2，直接返回系数数组
    if n == 1:
        return c
    if n == 2:
        return c
    else:
        # 初始化多项式的高阶系数和次高阶系数
        c0 = c[-2]
        c1 = c[-1]
        # 从最高阶次逐步计算多项式系数
        # i 是当前 c1 的阶数
        for i in range(n - 1, 1, -1):
            tmp = c0
            # 更新 c0 和 c1 的值
            c0 = polysub(c[i - 2], c1*(i - 1))
            c1 = polyadd(tmp, polymulx(c1))
        # 返回最终计算得到的多项式系数数组
        return polyadd(c0, polymulx(c1))


#
# These are constant arrays are of integer type so as to be compatible
# with the widest range of other types, such as Decimal.
#

# Hermite
hermedomain = np.array([-1., 1.])

# Hermite coefficients representing zero.
hermezero = np.array([0])

# Hermite coefficients representing one.
hermeone = np.array([1])

# Hermite coefficients representing the identity x.
hermex = np.array([0, 1])


def hermeline(off, scl):
    """
    Hermite series whose graph is a straight line.

    Parameters
    ----------
    off, scl : scalars
        The specified line is given by ``off + scl*x``.

    Returns
    -------
    y : ndarray
        This module's representation of the Hermite series for
        ``off + scl*x``.

    See Also
    --------
    numpy.polynomial.polynomial.polyline
    numpy.polynomial.chebyshev.chebline
    numpy.polynomial.legendre.legline
    numpy.polynomial.laguerre.lagline
    numpy.polynomial.hermite.hermline

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import hermeline
    >>> from numpy.polynomial.hermite_e import hermeline, hermeval
    >>> hermeval(0,hermeline(3, 2))
    3.0
    >>> hermeval(1,hermeline(3, 2))
    5.0

    """
    # 如果 scl 不为零，返回表示直线的 Hermite 系列
    if scl != 0:
        return np.array([off, scl])
    else:
        # 如果 scl 为零，返回只包含常数 off 的 Hermite 系列
        return np.array([off])


def hermefromroots(roots):
    """
    Generate a HermiteE series with given roots.

    The function returns the coefficients of the polynomial

    .. math:: p(x) = (x - r_0) * (x - r_1) * ... * (x - r_n),

    where `roots` is a list of roots.

    Parameters
    ----------
    roots : array_like
        1-D array containing the roots of the HermiteE series polynomial.

    Returns
    -------
    c : ndarray
        1-D array containing the coefficients of the HermiteE series polynomial.

    """
    # 实现从给定根生成 HermiteE 系列的系数
    return np.poly(roots)
    # 返回 HermiteE 形式的多项式的系数，根据给定的根 `roots`。
    # `roots` 参数是一个数组，包含多项式的根。
    # 如果某个零点的重数是 n，则它在 `roots` 中出现 n 次。
    # 例如，如果数字 2 的重数是 3，数字 3 的重数是 2，则 `roots` 可能是 [2, 2, 2, 3, 3]。
    # 根可以以任何顺序出现。
    
    # 如果返回的系数是 `c`，则多项式 p(x) 表示为：
    # p(x) = c_0 + c_1 * He_1(x) + ... + c_n * He_n(x)
    # 其中 He_i(x) 是 HermiteE 多项式的基函数。
    
    # 最后一项的系数通常不为 1，因为 HermiteE 形式的多项式不一定是首一的。
    
    # Parameters
    # ----------
    # roots : array_like
    #     包含多项式根的序列。
    
    # Returns
    # -------
    # out : ndarray
    #     系数的一维数组。如果所有的根都是实数，则 `out` 是实数组；如果有一些根是复数，则 `out` 是复数数组，即使结果中所有的系数都是实数（参见下面的示例）。
    
    # See Also
    # --------
    # numpy.polynomial.polynomial.polyfromroots
    # numpy.polynomial.legendre.legfromroots
    # numpy.polynomial.laguerre.lagfromroots
    # numpy.polynomial.hermite.hermfromroots
    # numpy.polynomial.chebyshev.chebfromroots
    
    # Examples
    # --------
    # >>> from numpy.polynomial.hermite_e import hermefromroots, hermeval
    # >>> coef = hermefromroots((-1, 0, 1))
    # >>> hermeval((-1, 0, 1), coef)
    # array([0., 0., 0.])
    # >>> coef = hermefromroots((-1j, 1j))
    # >>> hermeval((-1j, 1j), coef)
    # array([0.+0.j, 0.+0.j])
def hermeadd(c1, c2):
    """
    Add one Hermite series to another.

    Returns the sum of two Hermite series `c1` + `c2`.  The arguments
    are sequences of coefficients ordered from lowest order term to
    highest, i.e., [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of Hermite series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Array representing the Hermite series of their sum.

    See Also
    --------
    hermesub, hermemulx, hermemul, hermediv, hermepow

    Notes
    -----
    Unlike multiplication, division, etc., the sum of two Hermite series
    is a Hermite series (without having to "reproject" the result onto
    the basis set) so addition, just like that of "standard" polynomials,
    is simply "component-wise."

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import hermeadd
    >>> hermeadd([1, 2, 3], [1, 2, 3, 4])
    array([2.,  4.,  6.,  4.])

    """
    # 使用私有模块 pu 中的 _add 函数来执行 Hermite 级数的加法
    return pu._add(c1, c2)


def hermesub(c1, c2):
    """
    Subtract one Hermite series from another.

    Returns the difference of two Hermite series `c1` - `c2`.  The
    sequences of coefficients are from lowest order term to highest, i.e.,
    [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of Hermite series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Of Hermite series coefficients representing their difference.

    See Also
    --------
    hermeadd, hermemulx, hermemul, hermediv, hermepow

    Notes
    -----
    Unlike multiplication, division, etc., the difference of two Hermite
    series is a Hermite series (without having to "reproject" the result
    onto the basis set) so subtraction, just like that of "standard"
    polynomials, is simply "component-wise."

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import hermesub
    >>> hermesub([1, 2, 3, 4], [1, 2, 3])
    array([0., 0., 0., 4.])

    """
    # 使用私有模块 pu 中的 _sub 函数来执行 Hermite 级数的减法
    return pu._sub(c1, c2)


def hermemulx(c):
    """Multiply a Hermite series by x.

    Multiply the Hermite series `c` by x, where x is the independent
    variable.


    Parameters
    ----------
    c : array_like
        1-D array of Hermite series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Array representing the result of the multiplication.

    See Also
    --------
    hermeadd, hermesub, hermemul, hermediv, hermepow

    Notes
    -----
    The multiplication uses the recursion relationship for Hermite
    polynomials in the form

    .. math::

        xP_i(x) = (P_{i + 1}(x) + iP_{i - 1}(x)))

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import hermemulx
    >>> hermemulx([1, 2, 3])
    array([2.,  7.,  2.,  3.])

    """
    # c 是 Hermite 级数系数的一维数组，函数返回乘以 x 后的结果
    # c is a trimmed copy
    # 将输入列表转换为 Pandas Series，并且确保只取其中的第一个元素
    [c] = pu.as_series([c])
    # 处理特殊情况：如果输入列表只包含一个元素且该元素为0，则直接返回该元素作为结果
    if len(c) == 1 and c[0] == 0:
        return c

    # 创建一个长度比输入列表 c 长度多一的空数组，数据类型与 c 相同
    prd = np.empty(len(c) + 1, dtype=c.dtype)
    # 设置 prd 数组的第一个元素为 c 的第一个元素乘以 0
    prd[0] = c[0]*0
    # 设置 prd 数组的第二个元素为 c 的第一个元素
    prd[1] = c[0]
    # 遍历输入列表 c 中的每个元素（除了第一个和最后一个元素），计算 prd 数组中每个位置的值
    for i in range(1, len(c)):
        prd[i + 1] = c[i]   # 当前位置的值为 c 中对应位置的值
        prd[i - 1] += c[i]*i  # 前一个位置的值加上 c 中对应位置的值乘以当前位置的索引值 i
    # 返回计算后的 prd 数组作为结果
    return prd
def hermemul(c1, c2):
    """
    Multiply one Hermite series by another.

    Returns the product of two Hermite series `c1` * `c2`.  The arguments
    are sequences of coefficients, from lowest order "term" to highest,
    e.g., [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of Hermite series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Of Hermite series coefficients representing their product.

    See Also
    --------
    hermeadd, hermesub, hermemulx, hermediv, hermepow

    Notes
    -----
    In general, the (polynomial) product of two C-series results in terms
    that are not in the Hermite polynomial basis set.  Thus, to express
    the product as a Hermite series, it is necessary to "reproject" the
    product onto said basis set, which may produce "unintuitive" (but
    correct) results; see Examples section below.

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import hermemul
    >>> hermemul([1, 2, 3], [0, 1, 2])
    array([14.,  15.,  28.,   7.,   6.])

    """

    # s1, s2 are trimmed copies
    # Trim c1 and c2 to remove leading coefficients all of whose terms are zero
    [c1, c2] = pu.as_series([c1, c2])

    # Compare the lengths of the two arrays and swap them accordingly
    if len(c1) > len(c2):
        c = c2
        xs = c1
    else:
        c = c1
        xs = c2

    # Initialize c0 and c1
    if len(c) == 1:
        c0 = c[0]*xs
        c1 = 0
    elif len(c) == 2:
        c0 = c[0]*xs
        c1 = c[1]*xs
    else:
        nd = len(c)
        c0 = c[-2]*xs
        c1 = c[-1]*xs

        # Loop through the remaining coefficients to calculate c0 and c1
        for i in range(3, len(c) + 1):
            tmp = c0
            nd = nd - 1
            c0 = hermesub(c[-i]*xs, c1*(nd - 1))
            c1 = hermeadd(tmp, hermemulx(c1))

    # Return the sum of c0 and the product of c1 and the highest order sequence 
    return hermeadd(c0, hermemulx(c1))


def hermediv(c1, c2):
    """
    Divide one Hermite series by another.

    Returns the quotient-with-remainder of two Hermite series
    `c1` / `c2`.  The arguments are sequences of coefficients from lowest
    order "term" to highest, e.g., [1,2,3] represents the series
    ``P_0 + 2*P_1 + 3*P_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of Hermite series coefficients ordered from low to
        high.

    Returns
    -------
    [quo, rem] : ndarrays
        Of Hermite series coefficients representing the quotient and
        remainder.

    See Also
    --------
    hermeadd, hermesub, hermemulx, hermemul, hermepow

    Notes
    -----
    In general, the (polynomial) division of one Hermite series by another
    results in quotient and remainder terms that are not in the Hermite
    polynomial basis set.  Thus, to express these results as a Hermite
    series, it is necessary to "reproject" the results onto the Hermite
    basis set, which may produce "unintuitive" (but correct) results; see
    Examples section below.

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import hermediv
    >>> hermediv([ 14.,  15.,  28.,   7.,   6.], [0, 1, 2])
    (array([1., 2., 3.]), array([0.]))

    """
    # 调用 hermediv 函数，计算两个数组的除法操作，并返回结果
    >>> hermediv([ 15.,  17.,  28.,   7.,   6.], [0, 1, 2])
    # 返回一个元组，包含两个数组：第一个数组是除法运算后的结果，第二个数组是部分结果
    (array([1., 2., 3.]), array([1., 2.]))

    """
    # 返回一个调用 pu 模块中 _div 函数的结果，传递 hermemul, c1, c2 作为参数
    return pu._div(hermemul, c1, c2)
# 定义函数 hermepow，用于计算 Hermite 级数的幂次方
def hermepow(c, pow, maxpower=16):
    """Raise a Hermite series to a power.

    Returns the Hermite series `c` raised to the power `pow`. The
    argument `c` is a sequence of coefficients ordered from low to high.
    i.e., [1,2,3] is the series  ``P_0 + 2*P_1 + 3*P_2.``

    Parameters
    ----------
    c : array_like
        1-D array of Hermite series coefficients ordered from low to
        high.
    pow : integer
        Power to which the series will be raised
    maxpower : integer, optional
        Maximum power allowed. This is mainly to limit growth of the series
        to unmanageable size. Default is 16

    Returns
    -------
    coef : ndarray
        Hermite series of power.

    See Also
    --------
    hermeadd, hermesub, hermemulx, hermemul, hermediv

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import hermepow
    >>> hermepow([1, 2, 3], 2)
    array([23.,  28.,  46.,  12.,   9.])

    """
    # 调用内部函数 _pow 进行 Hermite 级数的乘幂操作
    return pu._pow(hermemul, c, pow, maxpower)


# 定义函数 hermeder，用于计算 Hermite_e 级数的微分
def hermeder(c, m=1, scl=1, axis=0):
    """
    Differentiate a Hermite_e series.

    Returns the series coefficients `c` differentiated `m` times along
    `axis`.  At each iteration the result is multiplied by `scl` (the
    scaling factor is for use in a linear change of variable). The argument
    `c` is an array of coefficients from low to high degree along each
    axis, e.g., [1,2,3] represents the series ``1*He_0 + 2*He_1 + 3*He_2``
    while [[1,2],[1,2]] represents ``1*He_0(x)*He_0(y) + 1*He_1(x)*He_0(y)
    + 2*He_0(x)*He_1(y) + 2*He_1(x)*He_1(y)`` if axis=0 is ``x`` and axis=1
    is ``y``.

    Parameters
    ----------
    c : array_like
        Array of Hermite_e series coefficients. If `c` is multidimensional
        the different axis correspond to different variables with the
        degree in each axis given by the corresponding index.
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
        Hermite series of the derivative.

    See Also
    --------
    hermeint

    Notes
    -----
    In general, the result of differentiating a Hermite series does not
    resemble the same operation on a power series. Thus the result of this
    function may be "unintuitive," albeit correct; see Examples section
    below.

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import hermeder
    >>> hermeder([ 1.,  1.,  1.,  1.])
    array([1.,  2.,  3.])
    >>> hermeder([-0.25,  1.,  1./2.,  1./3.,  1./4 ], m=2)
    array([1.,  2.,  3.])

    """
    # 将输入的 Hermite 系数数组 c 转换为至少是 1 维的 ndarray 对象
    c = np.array(c, ndmin=1, copy=True)
    # 如果数组 c 的数据类型字符在 '?bBhHiIlLqQpP' 中的任意一个，将其转换为 np.double 类型
    if c.dtype.char in '?bBhHiIlLqQpP':
        c = c.astype(np.double)
    
    # 将 m 转换为整数，并赋给 cnt，用作求导的阶数
    cnt = pu._as_int(m, "the order of derivation")
    
    # 将 axis 转换为整数，并赋给 iaxis，表示操作的轴
    iaxis = pu._as_int(axis, "the axis")
    
    # 如果 cnt 小于 0，抛出值错误异常，要求求导的阶数必须是非负数
    if cnt < 0:
        raise ValueError("The order of derivation must be non-negative")
    
    # 根据数组 c 的维度调整 iaxis，确保其值在有效范围内
    iaxis = normalize_axis_index(iaxis, c.ndim)
    
    # 如果求导的阶数 cnt 为 0，直接返回原始数组 c
    if cnt == 0:
        return c
    
    # 将操作轴 iaxis 移动到数组 c 的最前面
    c = np.moveaxis(c, iaxis, 0)
    
    # 计算数组 c 的长度 n
    n = len(c)
    
    # 如果求导的阶数 cnt 大于等于数组长度 n，返回数组 c 的第一个元素乘以 0
    if cnt >= n:
        return c[:1] * 0
    else:
        # 否则，进行 cnt 次求导操作
        for i in range(cnt):
            n = n - 1  # 更新数组长度
            c *= scl  # 数组 c 每个元素乘以 scl
            # 创建一个空数组 der，用来存储求导后的结果
            der = np.empty((n,) + c.shape[1:], dtype=c.dtype)
            # 按照求导的顺序计算每一阶导数
            for j in range(n, 0, -1):
                der[j - 1] = j * c[j]
            c = der  # 更新 c 为当前求导阶数的结果数组
    
    # 将操作轴 iaxis 移回到数组 c 的原始位置
    c = np.moveaxis(c, 0, iaxis)
    
    # 返回求导后的结果数组 c
    return c
# 定义函数 hermeint，用于对 Hermite_e 级数进行积分
def hermeint(c, m=1, k=[], lbnd=0, scl=1, axis=0):
    """
    Integrate a Hermite_e series.

    Returns the Hermite_e series coefficients `c` integrated `m` times from
    `lbnd` along `axis`. At each iteration the resulting series is
    **multiplied** by `scl` and an integration constant, `k`, is added.
    The scaling factor is for use in a linear change of variable.  ("Buyer
    beware": note that, depending on what one is doing, one may want `scl`
    to be the reciprocal of what one might expect; for more information,
    see the Notes section below.)  The argument `c` is an array of
    coefficients from low to high degree along each axis, e.g., [1,2,3]
    represents the series ``H_0 + 2*H_1 + 3*H_2`` while [[1,2],[1,2]]
    represents ``1*H_0(x)*H_0(y) + 1*H_1(x)*H_0(y) + 2*H_0(x)*H_1(y) +
    2*H_1(x)*H_1(y)`` if axis=0 is ``x`` and axis=1 is ``y``.

    Parameters
    ----------
    c : array_like
        Array of Hermite_e series coefficients. If c is multidimensional
        the different axis correspond to different variables with the
        degree in each axis given by the corresponding index.
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
        Hermite_e series coefficients of the integral.

    Raises
    ------
    ValueError
        If ``m < 0``, ``len(k) > m``, ``np.ndim(lbnd) != 0``, or
        ``np.ndim(scl) != 0``.

    See Also
    --------
    hermeder

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
    >>> from numpy.polynomial.hermite_e import hermeint
    >>> hermeint([1, 2, 3]) # integrate once, value 0 at 0.
    array([1., 1., 1., 1.])
    """
    c = np.array(c, ndmin=1, copy=True)
    # 将输入的参数 c 转换为 numpy 数组，确保至少是一维的，并且是拷贝副本
    if c.dtype.char in '?bBhHiIlLqQpP':
        # 如果数组的数据类型是布尔型或整数型（有符号或无符号），则转换为双精度浮点型
        c = c.astype(np.double)
    if not np.iterable(k):
        # 如果 k 不可迭代，则将其转换为列表
        k = [k]
    cnt = pu._as_int(m, "the order of integration")
    # 将 m 转换为整数，用作积分的阶数
    iaxis = pu._as_int(axis, "the axis")
    # 将 axis 转换为整数，表示操作的轴向
    if cnt < 0:
        raise ValueError("The order of integration must be non-negative")
        # 如果积分的阶数小于 0，则抛出值错误异常
    if len(k) > cnt:
        raise ValueError("Too many integration constants")
        # 如果积分常数 k 的数量超过了积分的阶数 cnt，则抛出值错误异常
    if np.ndim(lbnd) != 0:
        raise ValueError("lbnd must be a scalar.")
        # 如果 lbnd 的维数不为 0（即不是标量），则抛出值错误异常
    if np.ndim(scl) != 0:
        raise ValueError("scl must be a scalar.")
        # 如果 scl 的维数不为 0（即不是标量），则抛出值错误异常
    iaxis = normalize_axis_index(iaxis, c.ndim)
    # 根据输入的轴索引 iaxis 和数组 c 的维数，规范化轴索引

    if cnt == 0:
        return c
        # 如果积分的阶数为 0，则直接返回输入的数组 c

    c = np.moveaxis(c, iaxis, 0)
    # 将数组 c 的轴 iaxis 移动到索引 0 的位置

    k = list(k) + [0]*(cnt - len(k))
    # 将积分常数 k 转换为列表形式，并在末尾补充零，使其长度达到积分阶数 cnt

    for i in range(cnt):
        n = len(c)
        # 获取数组 c 的长度
        c *= scl
        # 将数组 c 中的每个元素乘以标量 scl
        if n == 1 and np.all(c[0] == 0):
            c[0] += k[i]
            # 如果数组 c 只有一个元素且该元素全部为 0，则将其加上积分常数 k[i]
        else:
            tmp = np.empty((n + 1,) + c.shape[1:], dtype=c.dtype)
            # 创建一个形状为 (n+1, ...) 的空数组 tmp，数据类型与 c 相同
            tmp[0] = c[0]*0
            tmp[1] = c[0]
            # 设置 tmp 的前两个元素，用于存储积分后的系数
            for j in range(1, n):
                tmp[j + 1] = c[j]/(j + 1)
                # 计算每个系数的积分值
            tmp[0] += k[i] - hermeval(lbnd, tmp)
            # 计算整体的积分常数并减去边界值的 Hermite 插值
            c = tmp
            # 将 tmp 赋值给 c，继续进行下一阶的积分计算
    c = np.moveaxis(c, 0, iaxis)
    # 将数组 c 的轴 0 移回到 iaxis 的位置
    return c
    # 返回最终的积分结果数组 c
def hermeval(x, c, tensor=True):
    """
    Evaluate an HermiteE series at points x.

    If `c` is of length ``n + 1``, this function returns the value:

    .. math:: p(x) = c_0 * He_0(x) + c_1 * He_1(x) + ... + c_n * He_n(x)

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
        with themselves and with the elements of `c`.
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
    hermeval2d, hermegrid2d, hermeval3d, hermegrid3d

    Notes
    -----
    The evaluation uses Clenshaw recursion, aka synthetic division.

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import hermeval
    >>> coef = [1,2,3]
    >>> hermeval(1, coef)
    3.0
    >>> hermeval([[1,2],[3,4]], coef)
    array([[ 3., 14.],
           [31., 54.]])

    """
    # 将 c 转换为至少是一维的 NumPy 数组，并确保其数据类型为双精度浮点数
    c = np.array(c, ndmin=1, copy=None)
    # 如果 c 的数据类型为布尔类型或整数类型，将其转换为双精度浮点数类型
    if c.dtype.char in '?bBhHiIlLqQpP':
        c = c.astype(np.double)
    # 如果 x 是 tuple 或 list 类型，则将其转换为 NumPy 数组
    if isinstance(x, (tuple, list)):
        x = np.asarray(x)
    # 如果 x 是 NumPy 数组并且 tensor 参数为 True，则将 c 的形状重塑为 c.shape + (1,)*x.ndim
    if isinstance(x, np.ndarray) and tensor:
        c = c.reshape(c.shape + (1,)*x.ndim)

    # 如果 c 的长度为 1，设置 c0 为 c[0]，c1 为 0
    if len(c) == 1:
        c0 = c[0]
        c1 = 0
    elif len(c) == 2:
        # 如果列表 c 的长度为 2，则执行以下操作
        c0 = c[0]
        c1 = c[1]
    else:
        # 如果列表 c 的长度不为 2，则执行以下操作
        nd = len(c)
        # 取倒数第二个元素作为 c0
        c0 = c[-2]
        # 取最后一个元素作为 c1
        c1 = c[-1]
        # 对于列表 c 中索引从 3 到最后一个元素的范围，依次进行以下操作
        for i in range(3, len(c) + 1):
            # 临时保存 c0 的值
            tmp = c0
            # 更新 nd 的值
            nd = nd - 1
            # 计算新的 c0 值
            c0 = c[-i] - c1*(nd - 1)
            # 更新 c1 的值
            c1 = tmp + c1*x
    # 返回 c0 + c1*x 的计算结果
    return c0 + c1*x
# 定义一个函数，用于在二维 HermiteE 系列上评估点 (x, y)
def hermeval2d(x, y, c):
    # 调用 _valnd 函数，返回 HermiteE 系列在给定点 (x, y) 处的值
    return pu._valnd(hermeval, c, x, y)


# 定义一个函数，用于在 x 和 y 的笛卡尔积上评估二维 HermiteE 系列
def hermegrid2d(x, y, c):
    """
    Evaluate a 2-D HermiteE series on the Cartesian product of x and y.

    This function returns the values:

    .. math:: p(a,b) = \\sum_{i,j} c_{i,j} * H_i(a) * H_j(b)

    where the points ``(a, b)`` consist of all pairs formed by taking
    `a` from `x` and `b` from `y`. The resulting points form a grid with
    `x` in the first dimension and `y` in the second.

    The parameters `x` and `y` are converted to arrays only if they are
    tuples or a lists, otherwise they are treated as scalars. In either
    case, either `x` and `y` or their elements must support multiplication
    and addition both with themselves and with the elements of `c`.

    If `c` has fewer than two dimensions, ones are implicitly appended to
    its shape to make it 2-D. The shape of the result will be c.shape[2:] +
    x.shape.

    Parameters
    ----------
    x, y : array_like, compatible objects
        The two dimensional series is evaluated at the points in the
        Cartesian product of `x` and `y`.  If `x` or `y` is a list or
        tuple, it is first converted to an ndarray, otherwise it is left
        unchanged and, if it isn't an ndarray, it is treated as a scalar.
    """
    c : array_like
        Array of coefficients ordered so that the coefficients for terms of
        degree i,j are contained in ``c[i,j]``. If `c` has dimension
        greater than two the remaining indices enumerate multiple sets of
        coefficients.
        # 参数 c：系数数组，按照度为 i,j 的项的顺序排列。如果 `c` 的维度大于两，
        # 则其余的索引用来枚举多组系数。

    Returns
    -------
    values : ndarray, compatible object
        The values of the two dimensional polynomial at points in the Cartesian
        product of `x` and `y`.
        # 返回值：ndarray 或兼容对象，二维多项式在 `x` 和 `y` 的笛卡尔积点上的值。

    See Also
    --------
    hermeval, hermeval2d, hermeval3d, hermegrid3d
        # 参见：hermeval, hermeval2d, hermeval3d, hermegrid3d

    Notes
    -----

    .. versionadded:: 1.7.0
        # 注意：添加于版本 1.7.0

    """
    return pu._gridnd(hermeval, c, x, y)
        # 调用 `pu._gridnd` 函数计算 Hermite 插值的结果，使用 `hermeval` 函数，
        # 系数数组 `c`，以及给定的 `x` 和 `y` 值作为参数。
# 返回一个 3-D Hermite_e 系列在给定点 (x, y, z) 处的值。

def hermeval3d(x, y, z, c):
    # 导入 `pu` 模块，并调用其 `_valnd` 函数来计算 HermiteE 系列的值
    return pu._valnd(hermeval, c, x, y, z)


# 在 x、y、z 的笛卡尔乘积上评估 3-D HermiteE 系列。

def hermegrid3d(x, y, z, c):
    # 返回形如 p(a,b,c) = \sum_{i,j,k} c_{i,j,k} * He_i(a) * He_j(b) * He_k(c) 的值，
    # 其中 (a, b, c) 由 x、y、z 中的所有三元组组成。
    # 结果形成一个网格，x 是第一维，y 是第二维，z 是第三维。
    # 导入 `pu` 模块，并调用其 `_valnd` 函数来计算 HermiteE 系列的值
    x, y, z : array_like, compatible objects
        The three dimensional series is evaluated at the points in the
        Cartesian product of `x`, `y`, and `z`.  If `x`, `y`, or `z` is a
        list or tuple, it is first converted to an ndarray, otherwise it is
        left unchanged and, if it isn't an ndarray, it is treated as a
        scalar.
    c : array_like
        Array of coefficients ordered so that the coefficients for terms of
        degree i,j are contained in ``c[i,j]``. If `c` has dimension
        greater than two the remaining indices enumerate multiple sets of
        coefficients.

这部分代码定义了函数的参数和说明，指定了输入参数的类型和用途。


    Returns
    -------
    values : ndarray, compatible object
        The values of the two dimensional polynomial at points in the Cartesian
        product of `x` and `y`.

这里指明了函数的返回值类型和说明，描述了返回值是在 `x` 和 `y` 的笛卡尔积中计算的二维多项式的值。


    See Also
    --------
    hermeval, hermeval2d, hermegrid2d, hermeval3d

列出了与本函数相关的其他函数，供用户参考。


    Notes
    -----

    .. versionadded:: 1.7.0

    """

这里是函数的额外说明部分，标明了函数被加入的版本信息。


    return pu._gridnd(hermeval, c, x, y, z)

函数的实际实现部分，调用了 `pu._gridnd` 函数来计算 Hermite 多项式在给定点 `(x, y, z)` 处的值，使用了传入的系数 `c`。
def hermevander(x, deg):
    """Pseudo-Vandermonde matrix of given degree.

    Returns the pseudo-Vandermonde matrix of degree `deg` and sample points
    `x`. The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., i] = He_i(x),

    where ``0 <= i <= deg``. The leading indices of `V` index the elements of
    `x` and the last index is the degree of the HermiteE polynomial.

    If `c` is a 1-D array of coefficients of length ``n + 1`` and `V` is the
    array ``V = hermevander(x, n)``, then ``np.dot(V, c)`` and
    ``hermeval(x, c)`` are the same up to roundoff. This equivalence is
    useful both for least squares fitting and for the evaluation of a large
    number of HermiteE series of the same degree and sample points.

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
        corresponding HermiteE polynomial.  The dtype will be the same as
        the converted `x`.

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import hermevander
    >>> x = np.array([-1, 0, 1])
    >>> hermevander(x, 3)
    array([[ 1., -1.,  0.,  2.],
           [ 1.,  0., -1., -0.],
           [ 1.,  1.,  0., -2.]])

    """
    # Convert `deg` to an integer, raising a ValueError if it's negative
    ideg = pu._as_int(deg, "deg")
    if ideg < 0:
        raise ValueError("deg must be non-negative")

    # Ensure `x` is converted to a 1-D array of type float64 or complex128
    x = np.array(x, copy=None, ndmin=1) + 0.0

    # Create dimensions for the output array `v`
    dims = (ideg + 1,) + x.shape
    dtyp = x.dtype

    # Create an empty array `v` with the computed dimensions and type
    v = np.empty(dims, dtype=dtyp)

    # Initialize the first row of `v` to represent the constant term of the polynomial
    v[0] = x*0 + 1

    # Fill in the rest of `v` based on Hermite polynomial recursion
    if ideg > 0:
        v[1] = x
        for i in range(2, ideg + 1):
            v[i] = (v[i-1]*x - v[i-2]*(i - 1))

    # Move the leading index of `v` to the last index for final output
    return np.moveaxis(v, 0, -1)


def hermevander2d(x, y, deg):
    """Pseudo-Vandermonde matrix of given degrees.

    Returns the pseudo-Vandermonde matrix of degrees `deg` and sample
    points ``(x, y)``. The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., (deg[1] + 1)*i + j] = He_i(x) * He_j(y),

    where ``0 <= i <= deg[0]`` and ``0 <= j <= deg[1]``. The leading indices of
    `V` index the points ``(x, y)`` and the last index encodes the degrees of
    the HermiteE polynomials.

    If ``V = hermevander2d(x, y, [xdeg, ydeg])``, then the columns of `V`
    correspond to the elements of a 2-D coefficient array `c` of shape
    (xdeg + 1, ydeg + 1) in the order

    .. math:: c_{00}, c_{01}, c_{02} ... , c_{10}, c_{11}, c_{12} ...

    and ``np.dot(V, c.flat)`` and ``hermeval2d(x, y, c)`` will be the same
    up to roundoff. This equivalence is useful both for least squares
    fitting and for the evaluation of a large number of 2-D HermiteE
    series of the same degrees and sample points.
    """
    # This function is intentionally left without implementation for now.
    pass
    Parameters
    ----------
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
    hermevander, hermevander3d, hermeval2d, hermeval3d

    Notes
    -----

    .. versionadded:: 1.7.0

    """
    # 调用 `_vander_nd_flat` 函数，计算 Hermite 多项式的 Vandermonde 矩阵
    return pu._vander_nd_flat((hermevander, hermevander), (x, y), deg)
def hermevander3d(x, y, z, deg):
    """Generate a pseudo-Vandermonde matrix for 3D Hermite polynomials.

    Returns a pseudo-Vandermonde matrix `V` for given degrees `deg` and sample
    points `(x, y, z)`. The matrix `V` is structured such that the elements are
    products of Hermite polynomials in `x`, `y`, and `z` coordinates.

    Parameters
    ----------
    x, y, z : array_like
        Arrays of point coordinates for the sample points.
    deg : list of ints
        List of maximum degrees `[x_deg, y_deg, z_deg]` for Hermite polynomials.

    Returns
    -------
    vander3d : ndarray
        Pseudo-Vandermonde matrix `V` of shape `(x.shape + (order,))`, where
        `order = (deg[0]+1) * (deg[1]+1) * (deg[2]+1)`.

    See Also
    --------
    hermevander, hermeval2d, hermeval3d

    Notes
    -----
    This function calculates the pseudo-Vandermonde matrix using the product of
    Hermite polynomials, which is useful for fitting data with 3D Hermite series.

    .. versionadded:: 1.7.0

    """
    return pu._vander_nd_flat((hermevander, hermevander, hermevander), (x, y, z), deg)
    deg : int or 1-D array_like
        # 多项式拟合的阶数或阶数组成的一维数组。如果 `deg` 是一个整数，
        # 则拟合包括从0到第 `deg` 阶的所有项。对于 NumPy 版本 >= 1.11.0，
        # 可以使用指定要包含的项阶数的整数列表。
    rcond : float, optional
        # 拟合的相对条件数。比最大奇异值小的奇异值相对于最大奇异值的倍数将被忽略。
        # 默认值是 len(x)*eps，其中 eps 是浮点类型的相对精度，在大多数情况下约为 2e-16。
    full : bool, optional
        # 控制返回值的性质。当为 False（默认）时，只返回系数；当为 True 时，
        # 还返回奇异值分解的诊断信息。
    w : array_like, shape (`M`,), optional
        # 权重数组。如果不为 None，则权重 `w[i]` 应用于 `x[i]` 处的未平方残差 `y[i] - y_hat[i]`。
        # 理想情况下，权重应选择使得所有乘积 `w[i]*y[i]` 的误差具有相同的方差。
        # 当使用逆方差加权时，使用 `w[i] = 1/sigma(y[i])`。默认值为 None。

    Returns
    -------
    coef : ndarray, shape (M,) or (M, K)
        # 按升序排列的 HermiteE 系数。如果 `y` 是二维的，则数据的第 k 列的系数在列 `k` 中。

    [residuals, rank, singular_values, rcond] : list
        # 仅当 ``full == True`` 时返回这些值

        - residuals -- 最小二乘拟合的平方残差和
        - rank -- 缩放后的 Vandermonde 矩阵的数值秩
        - singular_values -- 缩放后的 Vandermonde 矩阵的奇异值
        - rcond -- `rcond` 的值。

        更多细节请参阅 `numpy.linalg.lstsq`。

    Warns
    -----
    RankWarning
        # 最小二乘拟合中系数矩阵的秩不足。仅当 ``full = False`` 时引发警告。
        # 可以通过以下方式关闭警告：

        >>> import warnings
        >>> warnings.simplefilter('ignore', np.exceptions.RankWarning)

    See Also
    --------
    numpy.polynomial.chebyshev.chebfit
    numpy.polynomial.legendre.legfit
    numpy.polynomial.polynomial.polyfit
    numpy.polynomial.hermite.hermfit
    numpy.polynomial.laguerre.lagfit
    hermeval : 评估 HermiteE 系列。
    hermevander : HermiteE 系列的伪 Vandermonde 矩阵。
    hermeweight : HermiteE 权重函数。
    numpy.linalg.lstsq : 从矩阵计算最小二乘拟合。
    scipy.interpolate.UnivariateSpline : 计算样条拟合。

    Notes
    -----
    # 解决方案是最小化加权平方误差的 HermiteE 系列 `p` 的系数，其中

    .. math:: E = \\sum_j w_j^2 * |y_j - p(x_j)|^2,
    # 返回 HermiteE 系列的拟合结果，通过解决奇异值分解的矩阵方程来实现
    return pu._fit(hermevander, x, y, deg, rcond, full, w)
# c is a trimmed copy
# 使用`pu.as_series`将`c`转换为系列（可能是多项式系列），并返回第一个元素的引用
[c] = pu.as_series([c])

# 如果系列`c`的长度小于2，抛出数值错误异常，要求系列至少包含1阶多项式
if len(c) < 2:
    raise ValueError('Series must have maximum degree of at least 1.')

# 如果系列`c`的长度为2，直接返回一个包含根的数组，根据HermiteE系列的定义
if len(c) == 2:
    return np.array([[-c[0]/c[1]]])
    # 生成厄米特伴随矩阵，反转其行列顺序
    m = hermecompanion(c)[::-1,::-1]
    # 计算反转后矩阵的特征值
    r = la.eigvals(m)
    # 对特征值进行排序
    r.sort()
    # 返回排序后的特征值数组
    return r
# 定义一个函数用于计算标准化 HermiteE 多项式
def _normed_hermite_e_n(x, n):
    """
    Evaluate a normalized HermiteE polynomial.

    Compute the value of the normalized HermiteE polynomial of degree ``n``
    at the points ``x``.

    Parameters
    ----------
    x : ndarray of double.
        Points at which to evaluate the function
    n : int
        Degree of the normalized HermiteE function to be evaluated.

    Returns
    -------
    values : ndarray
        The shape of the return value is described above.

    Notes
    -----
    .. versionadded:: 1.10.0

    This function is needed for finding the Gauss points and integration
    weights for high degrees. The values of the standard HermiteE functions
    overflow when n >= 207.

    """
    # 如果 n 等于 0，则返回一个形状与 x 相同的数组，其值为标准化常数
    if n == 0:
        return np.full(x.shape, 1/np.sqrt(np.sqrt(2*np.pi)))

    # 初始化 HermiteE 多项式的前两项
    c0 = 0.
    c1 = 1./np.sqrt(np.sqrt(2*np.pi))
    nd = float(n)
    # 循环计算 HermiteE 多项式的后续项直到第 n 项
    for i in range(n - 1):
        tmp = c0
        c0 = -c1*np.sqrt((nd - 1.)/nd)
        c1 = tmp + c1*x*np.sqrt(1./nd)
        nd = nd - 1.0
    # 返回计算结果
    return c0 + c1*x


def hermegauss(deg):
    """
    Gauss-HermiteE quadrature.

    Computes the sample points and weights for Gauss-HermiteE quadrature.
    These sample points and weights will correctly integrate polynomials of
    degree :math:`2*deg - 1` or less over the interval :math:`[-\\inf, \\inf]`
    with the weight function :math:`f(x) = \\exp(-x^2/2)`.

    Parameters
    ----------
    deg : int
        Number of sample points and weights. It must be >= 1.

    Returns
    -------
    x : ndarray
        1-D ndarray containing the sample points.
    y : ndarray
        1-D ndarray containing the weights.

    Notes
    -----

    .. versionadded:: 1.7.0

    The results have only been tested up to degree 100, higher degrees may
    be problematic. The weights are determined by using the fact that

    .. math:: w_k = c / (He'_n(x_k) * He_{n-1}(x_k))

    where :math:`c` is a constant independent of :math:`k` and :math:`x_k`
    is the k'th root of :math:`He_n`, and then scaling the results to get
    the right value when integrating 1.

    """
    # 将 deg 转换为整数，如果小于等于 0 则抛出 ValueError
    ideg = pu._as_int(deg, "deg")
    if ideg <= 0:
        raise ValueError("deg must be a positive integer")

    # 初始化 HermiteE 多项式的伴随矩阵的第一个近似根
    c = np.array([0]*deg + [1])
    m = hermecompanion(c)
    x = la.eigvalsh(m)

    # 通过一次牛顿法提升根的精度
    dy = _normed_hermite_e_n(x, ideg)
    df = _normed_hermite_e_n(x, ideg - 1) * np.sqrt(ideg)
    x -= dy/df

    # 计算权重，通过缩放因子避免可能的数值溢出
    fm = _normed_hermite_e_n(x, ideg - 1)
    fm /= np.abs(fm).max()
    w = 1/(fm * fm)

    # 对 Hermite_e 进行对称化处理
    w = (w + w[::-1])/2
    x = (x - x[::-1])/2

    # 缩放权重以获得正确的值
    w *= np.sqrt(2*np.pi) / w.sum()

    # 返回样本点和权重数组
    return x, w


def hermeweight(x):
    """
    Hermite_e 多项式的权重函数。

    权重函数为 :math:`\\exp(-x^2/2)`，积分区间为 :math:`[-\\inf, \\inf]`。HermiteE 多项式
    相对于这个权重函数是正交的，但不是归一化的。

    Parameters
    ----------
    x : array_like
       要计算权重函数的数值。

    Returns
    -------
    w : ndarray
       在 `x` 处的权重函数值。

    Notes
    -----
    Hermite_e 多项式权重函数的版本新增于 1.7.0。

    """
    # 计算 Hermite_e 多项式权重函数的值，即 exp(-0.5 * x^2)
    w = np.exp(-.5*x**2)
    # 返回计算得到的权重函数值
    return w
# HermiteE series class
#

class HermiteE(ABCPolyBase):
    """An HermiteE series class.

    The HermiteE class provides the standard Python numerical methods
    '+', '-', '*', '//', '%', 'divmod', '**', and '()' as well as the
    attributes and methods listed below.

    Parameters
    ----------
    coef : array_like
        HermiteE coefficients in order of increasing degree, i.e,
        ``(1, 2, 3)`` gives ``1*He_0(x) + 2*He_1(X) + 3*He_2(x)``.
    domain : (2,) array_like, optional
        Domain to use. The interval ``[domain[0], domain[1]]`` is mapped
        to the interval ``[window[0], window[1]]`` by shifting and scaling.
        The default value is [-1, 1].
    window : (2,) array_like, optional
        Window, see `domain` for its use. The default value is [-1, 1].

        .. versionadded:: 1.6.0
    symbol : str, optional
        Symbol used to represent the independent variable in string
        representations of the polynomial expression, e.g. for printing.
        The symbol must be a valid Python identifier. Default value is 'x'.

        .. versionadded:: 1.24

    """
    # Virtual Functions
    # 静态方法，用于加法操作
    _add = staticmethod(hermeadd)
    # 静态方法，用于减法操作
    _sub = staticmethod(hermesub)
    # 静态方法，用于乘法操作
    _mul = staticmethod(hermemul)
    # 静态方法，用于整数除法操作
    _div = staticmethod(hermediv)
    # 静态方法，用于幂运算操作
    _pow = staticmethod(hermepow)
    # 静态方法，用于计算多项式在给定点的值
    _val = staticmethod(hermeval)
    # 静态方法，用于多项式的积分操作
    _int = staticmethod(hermeint)
    # 静态方法，用于多项式的微分操作
    _der = staticmethod(hermeder)
    # 静态方法，用于拟合数据得到多项式系数
    _fit = staticmethod(hermefit)
    # 静态方法，用于生成通过指定点的直线
    _line = staticmethod(hermeline)
    # 静态方法，用于计算多项式的根
    _roots = staticmethod(hermeroots)
    # 静态方法，用于根据给定的根生成多项式
    _fromroots = staticmethod(hermefromroots)

    # Virtual properties
    # 定义多项式的定义域
    domain = np.array(hermedomain)
    # 定义多项式的窗口
    window = np.array(hermedomain)
    # 基函数名称
    basis_name = 'He'
```