# `.\numpy\numpy\polynomial\laguerre.py`

```py
# 引入必要的库和模块
import numpy as np  # 引入 NumPy 库，用于数值计算
import numpy.linalg as la  # 引入 NumPy 的线性代数模块
from numpy.lib.array_utils import normalize_axis_index  # 从 NumPy 库的数组工具模块中引入数组轴索引归一化函数

# 从当前包中引入其他模块和函数
from . import polyutils as pu  # 从当前包中引入 polyutils 模块并重命名为 pu
from ._polybase import ABCPolyBase  # 从当前包的 _polybase 模块中引入 ABCPolyBase 类

# 声明本模块中公开的符号列表
__all__ = [
    'lagzero', 'lagone', 'lagx', 'lagdomain', 'lagline', 'lagadd',
    'lagsub', 'lagmulx', 'lagmul', 'lagdiv', 'lagpow', 'lagval', 'lagder',
    'lagint', 'lag2poly', 'poly2lag', 'lagfromroots', 'lagvander',
    'lagfit', 'lagtrim', 'lagroots', 'Laguerre', 'lagval2d', 'lagval3d',
    'laggrid2d', 'laggrid3d', 'lagvander2d', 'lagvander3d', 'lagcompanion',
    'laggauss', 'lagweight'
]

# 将 pu.trimcoef 函数重命名为 lagtrim，便于使用
lagtrim = pu.trimcoef


def poly2lag(pol):
    """
    poly2lag(pol)

    Convert a polynomial to a Laguerre series.

    Convert an array representing the coefficients of a polynomial (relative
    to the "standard" basis) ordered from lowest degree to highest, to an
    array of the coefficients of the equivalent Laguerre series, ordered
    from lowest to highest degree.

    Parameters
    ----------
    pol : array_like
        1-D array containing the polynomial coefficients

    Returns
    -------
    c : ndarray
        1-D array containing the coefficients of the equivalent Laguerre
        series.

    See Also
    --------
    lag2poly

    Notes
    -----
    The easy way to do conversions between polynomial basis sets
    is to use the convert method of a class instance.

    Examples
    --------
    >>> from numpy.polynomial.laguerre import poly2lag
    >>> poly2lag(np.arange(4))
    array([ 23., -63.,  58., -18.])

    """
    [pol] = pu.as_series([pol])  # 将输入的 pol 转换为系列（数组）
    res = 0  # 初始化结果变量 res
    # 从高次到低次处理多项式的系数
    for p in pol[::-1]:
        # 通过 lagadd 和 lagmulx 函数实现 Laguerre 系列转换过程
        res = lagadd(lagmulx(res), p)
    return res  # 返回转换后的 Laguerre 系列


def lag2poly(c):
    """
    Convert a Laguerre series to a polynomial.

    ```
    
    Convert an array representing the coefficients of a Laguerre series,
    ordered from lowest degree to highest, to an array of the coefficients
    of the equivalent polynomial (relative to the "standard" basis) ordered
    from lowest to highest degree.

    Parameters
    ----------
    c : array_like
        1-D array containing the Laguerre series coefficients, ordered
        from lowest order term to highest.

    Returns
    -------
    pol : ndarray
        1-D array containing the coefficients of the equivalent polynomial
        (relative to the "standard" basis) ordered from lowest order term
        to highest.

    See Also
    --------
    poly2lag

    Notes
    -----
    The easy way to do conversions between polynomial basis sets
    is to use the convert method of a class instance.

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lag2poly
    >>> lag2poly([ 23., -63.,  58., -18.])
    array([0., 1., 2., 3.])

    """
    # 导入必要的多项式操作函数：加法、减法、乘法
    from .polynomial import polyadd, polysub, polymulx

    # 将输入的系数数组 c 规范化为多项式系数对象
    [c] = pu.as_series([c])
    # 获取系数数组的长度
    n = len(c)
    # 如果数组长度为1，则直接返回该系数数组作为多项式的系数
    if n == 1:
        return c
    else:
        # 获取系数数组的倒数第二个和最后一个系数
        c0 = c[-2]
        c1 = c[-1]
        # i 表示当前处理的 c1 的次数
        for i in range(n - 1, 1, -1):
            # 临时保存 c0
            tmp = c0
            # 更新 c0 为当前处理的 c[i-2] 减去 c1 * (i-1) / i
            c0 = polysub(c[i - 2], (c1*(i - 1))/i)
            # 更新 c1 为 tmp 加上 (2*i - 1)*c1 减去 c1 * x 的多项式乘法结果除以 i
            c1 = polyadd(tmp, polysub((2*i - 1)*c1, polymulx(c1))/i)
        # 返回最终的多项式系数数组，即 c0 加上 c1 减去 c1 * x 的多项式乘法结果
        return polyadd(c0, polysub(c1, polymulx(c1)))
# 这些是整数类型的常数数组，以便与最广泛的其他类型兼容，如Decimal。

# Laguerre
# Laguerre函数的定义域
lagdomain = np.array([0., 1.])

# Laguerre系数，表示零
lagzero = np.array([0])

# Laguerre系数，表示一
lagone = np.array([1])

# Laguerre系数，表示标识函数x
lagx = np.array([1, -1])


def lagline(off, scl):
    """
    Laguerre级数，其图形是一条直线。

    Parameters
    ----------
    off, scl : scalars
        指定的直线由``off + scl*x``给出。

    Returns
    -------
    y : ndarray
        表示``off + scl*x``的Laguerre系列的数组表示形式。

    See Also
    --------
    numpy.polynomial.polynomial.polyline
    numpy.polynomial.chebyshev.chebline
    numpy.polynomial.legendre.legline
    numpy.polynomial.hermite.hermline
    numpy.polynomial.hermite_e.hermeline

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagline, lagval
    >>> lagval(0,lagline(3, 2))
    3.0
    >>> lagval(1,lagline(3, 2))
    5.0

    """
    if scl != 0:
        return np.array([off + scl, -scl])
    else:
        return np.array([off])


def lagfromroots(roots):
    """
    根据给定的根生成Laguerre级数。

    函数返回多项式的系数

    .. math:: p(x) = (x - r_0) * (x - r_1) * ... * (x - r_n),

    的Laguerre形式，其中 :math:`r_n` 是`roots`中指定的根。如果一个零有多重性为n，
    那么它必须在`roots`中出现n次。例如，如果2是三重根，3是双重根，
    那么`roots`看起来像 [2, 2, 2, 3, 3]。根可以以任何顺序出现。

    如果返回的系数是 `c`，那么

    .. math:: p(x) = c_0 + c_1 * L_1(x) + ... +  c_n * L_n(x)

    最后一项的系数通常不是Laguerre形式中的1。

    Parameters
    ----------
    roots : array_like
        包含根的序列。

    Returns
    -------
    out : ndarray
        系数的1-D数组。如果所有根都是实数，则 `out` 是实数组；如果一些根是复数，
        则 `out` 是复数数组，即使结果中的所有系数都是实数（参见下面的示例）。

    See Also
    --------
    numpy.polynomial.polynomial.polyfromroots
    numpy.polynomial.legendre.legfromroots
    numpy.polynomial.chebyshev.chebfromroots
    numpy.polynomial.hermite.hermfromroots
    numpy.polynomial.hermite_e.hermefromroots

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagfromroots, lagval
    >>> coef = lagfromroots((-1, 0, 1))
    >>> lagval((-1, 0, 1), coef)
    array([0.,  0.,  0.])
    >>> coef = lagfromroots((-1j, 1j))
    >>> lagval((-1j, 1j), coef)
    array([0.+0.j, 0.+0.j])

    """
    return pu._fromroots(lagline, lagmul, roots)


def lagadd(c1, c2):
    """
    ```py
    Add one Laguerre series to another.

    Returns the sum of two Laguerre series `c1` + `c2`.  The arguments
    are sequences of coefficients ordered from lowest order term to
    highest, i.e., [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of Laguerre series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Array representing the Laguerre series of their sum.

    See Also
    --------
    lagsub, lagmulx, lagmul, lagdiv, lagpow

    Notes
    -----
    Unlike multiplication, division, etc., the sum of two Laguerre series
    is a Laguerre series (without having to "reproject" the result onto
    the basis set) so addition, just like that of "standard" polynomials,
    is simply "component-wise."

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagadd
    >>> lagadd([1, 2, 3], [1, 2, 3, 4])
    array([2.,  4.,  6.,  4.])

    """
    # 使用私有函数 `_add` 计算两个 Laguerre 级数的和，并返回结果
    return pu._add(c1, c2)
def lagsub(c1, c2):
    """
    Subtract one Laguerre series from another.

    Returns the difference of two Laguerre series `c1` - `c2`.  The
    sequences of coefficients are from lowest order term to highest, i.e.,
    [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of Laguerre series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Of Laguerre series coefficients representing their difference.

    See Also
    --------
    lagadd, lagmulx, lagmul, lagdiv, lagpow

    Notes
    -----
    Unlike multiplication, division, etc., the difference of two Laguerre
    series is a Laguerre series (without having to "reproject" the result
    onto the basis set) so subtraction, just like that of "standard"
    polynomials, is simply "component-wise."

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagsub
    >>> lagsub([1, 2, 3, 4], [1, 2, 3])
    array([0.,  0.,  0.,  4.])

    """
    # 使用 pu._sub 函数计算 Laguerre 系列 c1 和 c2 的差
    return pu._sub(c1, c2)


def lagmulx(c):
    """Multiply a Laguerre series by x.

    Multiply the Laguerre series `c` by x, where x is the independent
    variable.


    Parameters
    ----------
    c : array_like
        1-D array of Laguerre series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Array representing the result of the multiplication.

    See Also
    --------
    lagadd, lagsub, lagmul, lagdiv, lagpow

    Notes
    -----
    The multiplication uses the recursion relationship for Laguerre
    polynomials in the form

    .. math::

        xP_i(x) = (-(i + 1)*P_{i + 1}(x) + (2i + 1)P_{i}(x) - iP_{i - 1}(x))

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagmulx
    >>> lagmulx([1, 2, 3])
    array([-1.,  -1.,  11.,  -9.])

    """
    # 将输入的系数数组 c 转换为 Laguerre 系列
    [c] = pu.as_series([c])
    # 处理零系列的特殊情况
    if len(c) == 1 and c[0] == 0:
        return c

    # 初始化乘积数组 prd，长度比输入数组 c 长 1
    prd = np.empty(len(c) + 1, dtype=c.dtype)
    prd[0] = c[0]  # 设置第一个值为 c[0]
    prd[1] = -c[0]  # 设置第二个值为 -c[0]
    # 迭代计算 Laguerre 系列的乘积
    for i in range(1, len(c)):
        prd[i + 1] = -c[i] * (i + 1)
        prd[i] += c[i] * (2 * i + 1)
        prd[i - 1] -= c[i] * i
    return prd


def lagmul(c1, c2):
    """
    Multiply one Laguerre series by another.

    Returns the product of two Laguerre series `c1` * `c2`.  The arguments
    are sequences of coefficients, from lowest order "term" to highest,
    e.g., [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of Laguerre series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Of Laguerre series coefficients representing their product.

    See Also
    --------
    lagadd, lagsub, lagmulx, lagdiv, lagpow

    Notes
    -----
    In general, the (polynomial) product of two C-series results in terms
    that are summed over the range of the series.

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagmul
    >>> lagmul([1, 2, 3], [4, 5])
    array([ 4., 13., 22., 15.])

    """
    # 使用 pu._mulx 函数计算 Laguerre 系列 c1 和 c2 的乘积
    return pu._mulx(c1, c2)
    # s1, s2 are trimmed copies
    [c1, c2] = pu.as_series([c1, c2])

    # Determine the shorter series between c1 and c2
    if len(c1) > len(c2):
        c = c2
        xs = c1
    else:
        c = c1
        xs = c2

    # Compute coefficients based on the length of c
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
        # Iterate to compute higher order coefficients using Laguerre polynomial operations
        for i in range(3, len(c) + 1):
            tmp = c0
            nd = nd - 1
            c0 = lagsub(c[-i]*xs, (c1*(nd - 1))/nd)
            c1 = lagadd(tmp, lagsub((2*nd - 1)*c1, lagmulx(c1))/nd)

    # Return the sum of c0 and modified c1 coefficients
    return lagadd(c0, lagsub(c1, lagmulx(c1)))
# 定义函数 lagdiv，用于计算两个拉盖尔级数的除法
def lagdiv(c1, c2):
    """
    Divide one Laguerre series by another.

    Returns the quotient-with-remainder of two Laguerre series
    `c1` / `c2`.  The arguments are sequences of coefficients from lowest
    order "term" to highest, e.g., [1,2,3] represents the series
    ``P_0 + 2*P_1 + 3*P_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of Laguerre series coefficients ordered from low to
        high.

    Returns
    -------
    [quo, rem] : ndarrays
        Of Laguerre series coefficients representing the quotient and
        remainder.

    See Also
    --------
    lagadd, lagsub, lagmulx, lagmul, lagpow

    Notes
    -----
    In general, the (polynomial) division of one Laguerre series by another
    results in quotient and remainder terms that are not in the Laguerre
    polynomial basis set.  Thus, to express these results as a Laguerre
    series, it is necessary to "reproject" the results onto the Laguerre
    basis set, which may produce "unintuitive" (but correct) results; see
    Examples section below.

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagdiv
    >>> lagdiv([  8., -13.,  38., -51.,  36.], [0, 1, 2])
    (array([1., 2., 3.]), array([0.]))
    >>> lagdiv([  9., -12.,  38., -51.,  36.], [0, 1, 2])
    (array([1., 2., 3.]), array([1., 1.]))

    """
    # 使用私有函数 _div 来实现拉盖尔级数的除法
    return pu._div(lagmul, c1, c2)


# 定义函数 lagpow，用于计算拉盖尔级数的幂
def lagpow(c, pow, maxpower=16):
    """Raise a Laguerre series to a power.

    Returns the Laguerre series `c` raised to the power `pow`. The
    argument `c` is a sequence of coefficients ordered from low to high.
    i.e., [1,2,3] is the series  ``P_0 + 2*P_1 + 3*P_2.``

    Parameters
    ----------
    c : array_like
        1-D array of Laguerre series coefficients ordered from low to
        high.
    pow : integer
        Power to which the series will be raised
    maxpower : integer, optional
        Maximum power allowed. This is mainly to limit growth of the series
        to unmanageable size. Default is 16

    Returns
    -------
    coef : ndarray
        Laguerre series of power.

    See Also
    --------
    lagadd, lagsub, lagmulx, lagmul, lagdiv

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagpow
    >>> lagpow([1, 2, 3], 2)
    array([ 14., -16.,  56., -72.,  54.])

    """
    # 使用私有函数 _pow 来实现拉盖尔级数的幂运算
    return pu._pow(lagmul, c, pow, maxpower)


# 定义函数 lagder，用于计算拉盖尔级数的微分
def lagder(c, m=1, scl=1, axis=0):
    """
    Differentiate a Laguerre series.

    Returns the Laguerre series coefficients `c` differentiated `m` times
    along `axis`.  At each iteration the result is multiplied by `scl` (the
    scaling factor is for use in a linear change of variable). The argument
    `c` is an array of coefficients from low to high degree along each
    axis, e.g., [1,2,3] represents the series ``1*L_0 + 2*L_1 + 3*L_2``
    while [[1,2],[1,2]] represents ``1*L_0(x)*L_0(y) + 1*L_1(x)*L_0(y) +
    2*L_0(x)*L_1(y) + 2*L_1(x)*L_1(y)`` if axis=0 is ``x`` and axis=1 is
    ``y``.

    """
    # 使用私有函数 _der 来实现拉盖尔级数的微分
    return pu._der(lagmul, c, m, scl, axis)
    # 将输入的系列系数转换为 numpy 数组，确保至少为一维，并复制数据
    c = np.array(c, ndmin=1, copy=True)
    
    # 如果系列系数的数据类型是布尔型、整型或长整型，则转换为双精度浮点型
    if c.dtype.char in '?bBhHiIlLqQpP':
        c = c.astype(np.double)
    
    # 将导数阶数转换为整数，用于计数，并验证其非负性
    cnt = pu._as_int(m, "the order of derivation")
    
    # 将轴的索引转换为整数
    iaxis = pu._as_int(axis, "the axis")
    
    # 如果导数阶数小于 0，则抛出数值错误
    if cnt < 0:
        raise ValueError("The order of derivation must be non-negative")
    
    # 根据轴的索引规范化轴的索引值，确保其在合法范围内
    iaxis = normalize_axis_index(iaxis, c.ndim)
    
    # 如果导数阶数为 0，则直接返回输入的系列系数
    if cnt == 0:
        return c
    
    # 将指定轴移动到数组的第一个位置
    c = np.moveaxis(c, iaxis, 0)
    
    # 获取系列系数的长度
    n = len(c)
    
    # 如果导数阶数大于等于系列长度，则返回长度为 1 的数组乘以 0
    if cnt >= n:
        c = c[:1]*0
    else:
        # 循环进行导数操作，每次操作将系数乘以 scl，并更新系数数组
        for i in range(cnt):
            n = n - 1
            c *= scl
            # 创建一个空的数组来存储导数结果
            der = np.empty((n,) + c.shape[1:], dtype=c.dtype)
            # 计算导数值并更新系数数组
            for j in range(n, 1, -1):
                der[j - 1] = -c[j]
                c[j - 1] += c[j]
            der[0] = -c[1]
            c = der
    
    # 将第一个位置的轴移回原来的位置
    c = np.moveaxis(c, 0, iaxis)
    
    # 返回最终的 Laguerre 系列导数结果数组
    return c
# 定义一个函数 lagint，用于对拉盖尔级数进行积分
def lagint(c, m=1, k=[], lbnd=0, scl=1, axis=0):
    """
    Integrate a Laguerre series.

    Returns the Laguerre series coefficients `c` integrated `m` times from
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
        Array of Laguerre series coefficients. If `c` is multidimensional
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
        Laguerre series coefficients of the integral.

    Raises
    ------
    ValueError
        If ``m < 0``, ``len(k) > m``, ``np.ndim(lbnd) != 0``, or
        ``np.ndim(scl) != 0``.

    See Also
    --------
    lagder

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
    >>> from numpy.polynomial.laguerre import lagint
    >>> lagint([1,2,3])
    array([ 1.,  1.,  1., -3.])
    >>> lagint([1,2,3], m=2)
    """
    """
    Calculate Laguerre integration coefficients for a given set of parameters.

    Parameters:
    c : array_like
        Coefficients to integrate.
    k : int or list of int, optional
        Integration constants. Default is 0.
    m : int, optional
        Order of integration. Default is 1.
    lbnd : scalar, optional
        Lower bound of integration. Default is 0.
    scl : scalar, optional
        Scaling factor. Default is 1.
    axis : int, optional
        Integration axis. Default is 0.

    Returns:
    array
        Integrated coefficients.

    Raises:
    ValueError
        If input parameters are invalid or out of bounds.

    """

    # Ensure `c` is at least 1-dimensional and cast to double if necessary
    c = np.array(c, ndmin=1, copy=True)
    if c.dtype.char in '?bBhHiIlLqQpP':
        c = c.astype(np.double)

    # Ensure `k` is iterable
    if not np.iterable(k):
        k = [k]

    # Ensure `m` is a non-negative integer
    cnt = pu._as_int(m, "the order of integration")

    # Ensure `axis` is a valid axis index
    iaxis = pu._as_int(axis, "the axis")
    iaxis = normalize_axis_index(iaxis, c.ndim)

    # Validate order of integration `cnt`
    if cnt < 0:
        raise ValueError("The order of integration must be non-negative")

    # Validate number of integration constants `k`
    if len(k) > cnt:
        raise ValueError("Too many integration constants")

    # Ensure `lbnd` and `scl` are scalars
    if np.ndim(lbnd) != 0:
        raise ValueError("lbnd must be a scalar.")
    if np.ndim(scl) != 0:
        raise ValueError("scl must be a scalar.")

    # Perform Laguerre integration
    if cnt == 0:
        return c

    # Move integration axis to the front
    c = np.moveaxis(c, iaxis, 0)

    # Pad `k` with zeros if necessary
    k = list(k) + [0] * (cnt - len(k))

    # Iteratively compute Laguerre integration
    for i in range(cnt):
        n = len(c)
        c *= scl
        if n == 1 and np.all(c[0] == 0):
            c[0] += k[i]
        else:
            tmp = np.empty((n + 1,) + c.shape[1:], dtype=c.dtype)
            tmp[0] = c[0]
            tmp[1] = -c[0]
            for j in range(1, n):
                tmp[j] += c[j]
                tmp[j + 1] = -c[j]
            tmp[0] += k[i] - lagval(lbnd, tmp)
            c = tmp

    # Move integration axis back to its original position
    c = np.moveaxis(c, 0, iaxis)

    return c
def lagval(x, c, tensor=True):
    """
    Evaluate a Laguerre series at points x.

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
    lagval2d, laggrid2d, lagval3d, laggrid3d

    Notes
    -----
    The evaluation uses Clenshaw recursion, aka synthetic division.

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagval
    >>> coef = [1, 2, 3]
    >>> lagval(1, coef)
    -0.5
    >>> lagval([[1, 2],[3, 4]], coef)
    array([[-0.5, -4. ],
           [-4.5, -2. ]])

    """
    # 将输入的系数转换为至少为一维的 numpy 数组
    c = np.array(c, ndmin=1, copy=None)
    # 如果系数数组的数据类型为布尔类型，将其转换为双精度浮点型
    if c.dtype.char in '?bBhHiIlLqQpP':
        c = c.astype(np.double)
    # 如果 x 是元组或列表，则转换为 numpy 数组
    if isinstance(x, (tuple, list)):
        x = np.asarray(x)
    # 如果 x 是 numpy 数组且 tensor 为 True，则将系数数组 c 扩展为 x 的维度
    if isinstance(x, np.ndarray) and tensor:
        c = c.reshape(c.shape + (1,)*x.ndim)

    # 如果系数数组 c 的长度为 1，设置 c0 为 c[0]，c1 为 0
    if len(c) == 1:
        c0 = c[0]
        c1 = 0
    # 如果列表 c 的长度为 2，执行以下操作
    elif len(c) == 2:
        # 将 c[0] 赋值给 c0
        c0 = c[0]
        # 将 c[1] 赋值给 c1
        c1 = c[1]
    # 如果列表 c 的长度不为 2，执行以下操作
    else:
        # 获取列表 c 的长度，并将其赋值给 nd
        nd = len(c)
        # 将倒数第二个元素 c[-2] 赋值给 c0
        c0 = c[-2]
        # 将最后一个元素 c[-1] 赋值给 c1
        c1 = c[-1]
        # 对于 i 从 3 到 len(c) + 1 的范围进行循环
        for i in range(3, len(c) + 1):
            # 将 c0 的值暂存到 tmp 中
            tmp = c0
            # 更新 nd 的值，减去 1
            nd = nd - 1
            # 根据公式计算新的 c0 值
            c0 = c[-i] - (c1*(nd - 1))/nd
            # 根据公式计算新的 c1 值
            c1 = tmp + (c1*((2*nd - 1) - x))/nd
    # 返回计算结果 c0 + c1*(1 - x)
    return c0 + c1*(1 - x)
# 定义一个函数，用于在二维空间中评估拉盖尔级数在点 (x, y) 处的值
def lagval2d(x, y, c):
    # 调用 pu 模块中的 _valnd 函数来计算二维拉盖尔级数的值，并返回结果
    return pu._valnd(lagval, c, x, y)


# 定义一个函数，用于在笛卡尔积 x 和 y 上评估二维拉盖尔级数的值
def laggrid2d(x, y, c):
    # 返回值是二维拉盖尔级数在所有点 (a, b) 上的求和结果，其中 a 取自 x，b 取自 y
    """
    Evaluate a 2-D Laguerre series on the Cartesian product of x and y.

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
        The two dimensional series is evaluated at the points ``(x, y)``,
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
        The values of the two dimensional polynomial at points formed with
        pairs of corresponding values from `x` and `y`.

    See Also
    --------
    lagval, laggrid2d, lagval3d, laggrid3d

    Notes
    -----

    .. versionadded:: 1.7.0

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagval2d
    >>> c = [[1, 2],[3, 4]]
    >>> lagval2d(1, 1, c)
    1.0
    """
    x, y : array_like, compatible objects
        两个一维序列或兼容对象，表示在其笛卡尔乘积中求解二维系列的值。如果 `x` 或 `y` 是列表或元组，则首先转换为 ndarray 数组；否则，保持不变，若不是 ndarray，则视为标量。

    c : array_like
        系数数组，按照多重度 i,j 的顺序排列，其中多重度 i,j 的系数包含在 `c[i,j]` 中。如果 `c` 的维度大于两，剩余的索引用于枚举多组系数。

    Returns
    -------
    values : ndarray, compatible object
        在 `x` 和 `y` 的笛卡尔乘积中，二维切比雪夫级数在各点的值。

    See Also
    --------
    lagval, lagval2d, lagval3d, laggrid3d
        相关函数以及更高维度的 Laguerre 多项式计算函数。

    Notes
    -----

    .. versionadded:: 1.7.0
        引入版本说明：此函数在 NumPy 1.7.0 版本中首次引入。

    Examples
    --------
    >>> from numpy.polynomial.laguerre import laggrid2d
    >>> c = [[1, 2], [3, 4]]
    >>> laggrid2d([0, 1], [0, 1], c)
    array([[10.,  4.],
           [ 3.,  1.]])
        示例说明：使用 laggrid2d 函数计算给定系数数组 `c` 的二维 Laguerre 多项式在指定点网格上的值。
    
    """
    return pu._gridnd(lagval, c, x, y)
# 定义一个函数，用于在给定点(x, y, z)处评估三维拉盖尔级数
def lagval3d(x, y, z, c):
    # 返回由私有函数 _valnd 处理的结果，传入拉盖尔函数 lagval 和参数 x, y, z, c
    return pu._valnd(lagval, c, x, y, z)


# 定义一个函数，用于在 x, y, z 的笛卡尔乘积上评估三维拉盖尔级数
def laggrid3d(x, y, z, c):
    """
    Evaluate a 3-D Laguerre series on the Cartesian product of x, y, and z.

    This function returns the values:

    .. math:: p(a,b,c) = \\sum_{i,j,k} c_{i,j,k} * L_i(a) * L_j(b) * L_k(c)

    where the points ``(a, b, c)`` consist of all triples formed by taking
    `a` from `x`, `b` from `y`, and `c` from `z`. The resulting points form
    a grid with `x` in the first dimension, `y` in the second, and `z` in
    the third.

    The parameters `x`, `y`, and `z` are converted to arrays only if they
    are tuples or a lists, otherwise they are treated as scalars. In
    either case, either `x`, `y`, and `z` or their elements must support
    multiplication and addition both with themselves and with the elements
    of `c`.

    If `c` has fewer than three dimensions, ones are implicitly appended to
    its shape to make it 3-D. The shape of the result will be c.shape[3:] +
    x.shape + y.shape + z.shape.

    Parameters
    ----------
    x, y, z : array_like, compatible object
        Points at which the three-dimensional series is evaluated, forming a grid
        with `x` in the first dimension, `y` in the second, and `z` in the third.
        If `x`, `y`, or `z` is a list or tuple, it is first converted to an ndarray,
        otherwise it is treated as a scalar.
    c : array_like
        Array of coefficients where `c[i,j,k]` contains the coefficient of the term
        of multi-degree `i,j,k`. If `c` has more than 3 dimensions, the additional
        dimensions enumerate multiple sets of coefficients.

    Returns
    -------
    values : ndarray, compatible object
        The values of the multidimensional polynomial evaluated at the Cartesian
        product of `x`, `y`, and `z`.

    See Also
    --------
    lagval, lagval2d, laggrid2d, lagval3d

    Notes
    -----
    This function was added in version 1.7.0 of the library.

    Examples
    --------
    >>> from numpy.polynomial.laguerre import laggrid3d
    >>> c = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    >>> laggrid3d(1, 1, 2, c)
    array([[[[ -1.,  -2.],
             [ -3.,  -4.]],
            [[ -5.,  -6.],
             [ -7.,  -8.]]]])
    """
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

    Returns
    -------
    values : ndarray, compatible object
        The values of the two dimensional polynomial at points in the Cartesian
        product of `x` and `y`.

    See Also
    --------
    lagval, lagval2d, laggrid2d, lagval3d

    Notes
    -----

    .. versionadded:: 1.7.0

    Examples
    --------
    >>> from numpy.polynomial.laguerre import laggrid3d
    >>> c = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    >>> laggrid3d([0, 1], [0, 1], [2, 4], c)
    array([[[ -4., -44.],
            [ -2., -18.]],
           [[ -2., -14.],
            [ -1.,  -5.]]])
    
    """
    # 调用 _gridnd 函数，计算三维拉盖尔多项式的值
    return pu._gridnd(lagval, c, x, y, z)
def lagvander(x, deg):
    """Pseudo-Vandermonde matrix of given degree.

    Returns the pseudo-Vandermonde matrix of degree `deg` and sample points
    `x`. The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., i] = L_i(x)

    where ``0 <= i <= deg``. The leading indices of `V` index the elements of
    `x` and the last index is the degree of the Laguerre polynomial.

    If `c` is a 1-D array of coefficients of length ``n + 1`` and `V` is the
    array ``V = lagvander(x, n)``, then ``np.dot(V, c)`` and
    ``lagval(x, c)`` are the same up to roundoff. This equivalence is
    useful both for least squares fitting and for the evaluation of a large
    number of Laguerre series of the same degree and sample points.

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
        corresponding Laguerre polynomial.  The dtype will be the same as
        the converted `x`.

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagvander
    >>> x = np.array([0, 1, 2])
    >>> lagvander(x, 3)
    array([[ 1.        ,  1.        ,  1.        ,  1.        ],
           [ 1.        ,  0.        , -0.5       , -0.66666667],
           [ 1.        , -1.        , -1.        , -0.33333333]])

    """
    # Ensure deg is converted to an integer
    ideg = pu._as_int(deg, "deg")
    # Check if the degree is non-negative
    if ideg < 0:
        raise ValueError("deg must be non-negative")

    # Convert `x` to a numpy array, ensuring it's at least 1-D
    x = np.array(x, copy=None, ndmin=1) + 0.0
    # Define dimensions of the result matrix `v`
    dims = (ideg + 1,) + x.shape
    # Determine the dtype for `v` based on `x`
    dtyp = x.dtype
    # Create an uninitialized array `v` with specified dimensions and dtype
    v = np.empty(dims, dtype=dtyp)
    # Initialize the first row of `v` as 1's
    v[0] = x * 0 + 1
    # If degree is greater than 0, compute subsequent rows of `v`
    if ideg > 0:
        v[1] = 1 - x
        for i in range(2, ideg + 1):
            v[i] = (v[i-1] * (2*i - 1 - x) - v[i-2] * (i - 1)) / i
    # Move the first axis of `v` to the last axis
    return np.moveaxis(v, 0, -1)


def lagvander2d(x, y, deg):
    """Pseudo-Vandermonde matrix of given degrees.

    Returns the pseudo-Vandermonde matrix of degrees `deg` and sample
    points ``(x, y)``. The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., (deg[1] + 1)*i + j] = L_i(x) * L_j(y),

    where ``0 <= i <= deg[0]`` and ``0 <= j <= deg[1]``. The leading indices of
    `V` index the points ``(x, y)`` and the last index encodes the degrees of
    the Laguerre polynomials.

    If ``V = lagvander2d(x, y, [xdeg, ydeg])``, then the columns of `V`
    correspond to the elements of a 2-D coefficient array `c` of shape
    (xdeg + 1, ydeg + 1) in the order

    .. math:: c_{00}, c_{01}, c_{02} ... , c_{10}, c_{11}, c_{12} ...

    and ``np.dot(V, c.flat)`` and ``lagval2d(x, y, c)`` will be the same
    up to roundoff. This equivalence is useful both for least squares

    """
    # Function definition and docstring are complete as they are.
    fitting and for the evaluation of a large number of 2-D Laguerre
    series of the same degrees and sample points.

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
    lagvander, lagvander3d, lagval2d, lagval3d

    Notes
    -----

    .. versionadded:: 1.7.0

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagvander2d
    >>> x = np.array([0])
    >>> y = np.array([2])
    >>> lagvander2d(x, y, [2, 1])
    array([[ 1., -1.,  1., -1.,  1., -1.]])
    
"""
return pu._vander_nd_flat((lagvander, lagvander), (x, y), deg)
def lagfit(x, y, deg, rcond=None, full=False, w=None):
    """
    Least squares fit of Laguerre series to data.

    Return the coefficients of a Laguerre series of degree `deg` that is the
    least squares fit to the data values `y` given at points `x`. If `y` is
    1-D the returned coefficients will also be 1-D. If `y` is 2-D multiple
    fits are done, one for each column of `y`, and the resulting
    coefficients are stored in the corresponding columns of a 2-D return.
    The fitted polynomial(s) are in the form

    .. math::  p(x) = c_0 + c_1 * L_1(x) + ... + c_n * L_n(x),

    where ``n`` is `deg`.

    Parameters
    ----------
    x : array_like
        Array of independent variable values.
    y : array_like
        Array of dependent variable values. If `y` is 2-D, each column represents
        a separate set of data.
    deg : int
        Degree(s) of the Laguerre series to fit.
    rcond : float, optional
        Cut-off ratio for small singular values of the coefficient matrix.
    full : bool, optional
        If True, return additional information.
    w : array_like, optional
        Weights to apply to the residuals.

    Returns
    -------
    coef : ndarray
        Coefficients of the Laguerre series, or array of coefficients if `y` is 2-D.
    residues, rank, singular_values, rcond : ndarray, int, ndarray, float
        Returned only if `full` is True; details of the fit.

    See Also
    --------
    lagvander, lagvander3d, lagfit, lagval2d, lagval3d

    Notes
    -----
    The least squares fit minimizes the sum of the squares of the residuals,
    weighted by the optional parameter `w`.

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.polynomial.laguerre import lagfit
    >>> x = np.array([1, 2, 3])
    >>> y = np.array([2, 3, 4])
    >>> lagfit(x, y, 2)
    array([ 2.,  1.,  0.])
    """
    return pu._fit(legvander, x, y, deg, rcond, full, w)
    x : array_like, shape (M,)
        # M个样本点的x坐标 ``(x[i], y[i])`` 的x坐标
    y : array_like, shape (M,) or (M, K)
        # 样本点的y坐标。可以同时拟合多个共享相同x坐标的数据集，通过传入包含每列数据集的二维数组。
    deg : int or 1-D array_like
        # 拟合多项式的阶数。如果`deg`是一个整数，则包括直到第`deg`项的所有项。对于NumPy版本 >= 1.11.0，可以使用指定要包括的项的阶数的整数列表。
    rcond : float, optional
        # 拟合条件数的相对值。相对于最大奇异值，小于此值的奇异值将被忽略。默认值是len(x)*eps，其中eps是浮点类型的相对精度，在大多数情况下约为2e-16。
    full : bool, optional
        # 返回值的性质开关。当为False时（默认值），仅返回系数；当为True时，还返回奇异值分解的诊断信息。
    w : array_like, shape (`M`,), optional
        # 权重。如果不为None，则权重``w[i]``应用于``x[i]``处未平方的残差``y[i] - y_hat[i]``。理想情况下，选择权重使得所有产品``w[i]*y[i]``的误差具有相同的方差。使用逆方差加权时，使用``w[i] = 1/sigma(y[i])``。默认值为None。

    Returns
    -------
    coef : ndarray, shape (M,) or (M, K)
        # 从低到高排序的Laguerre系数。如果`y`是2-D，则数据列*k*的系数在列*k*中。

    [residuals, rank, singular_values, rcond] : list
        # 仅在``full == True``时返回这些值

        - residuals -- 最小二乘拟合的残差平方和
        - rank -- 缩放Vandermonde矩阵的数值秩
        - singular_values -- 缩放Vandermonde矩阵的奇异值
        - rcond -- `rcond`的值。

        更多细节，请参见`numpy.linalg.lstsq`。

    Warns
    -----
    RankWarning
        # 最小二乘拟合中系数矩阵的秩不足。仅当``full == False``时才会引发警告。可以通过以下方式关闭警告

        >>> import warnings
        >>> warnings.simplefilter('ignore', np.exceptions.RankWarning)

    See Also
    --------
    numpy.polynomial.polynomial.polyfit
    numpy.polynomial.legendre.legfit
    numpy.polynomial.chebyshev.chebfit
    numpy.polynomial.hermite.hermfit
    numpy.polynomial.hermite_e.hermefit
    lagval : 评估Laguerre级数。
    lagvander : Laguerre级数的伪Vandermonde矩阵。
    lagweight : Laguerre权重函数。
    # 使用 Laguerre 系列进行曲线拟合，计算最小二乘拟合系数
    numpy.linalg.lstsq : Computes a least-squares fit from the matrix.
    scipy.interpolate.UnivariateSpline : Computes spline fits.

    Notes
    -----
    The solution is the coefficients of the Laguerre series ``p`` that
    minimizes the sum of the weighted squared errors

    .. math:: E = \\sum_j w_j^2 * |y_j - p(x_j)|^2,

    where the :math:`w_j` are the weights. This problem is solved by
    setting up as the (typically) overdetermined matrix equation

    .. math:: V(x) * c = w * y,

    where ``V`` is the weighted pseudo Vandermonde matrix of `x`, ``c`` are the
    coefficients to be solved for, `w` are the weights, and `y` are the
    observed values.  This equation is then solved using the singular value
    decomposition of ``V``.

    If some of the singular values of `V` are so small that they are
    neglected, then a `~exceptions.RankWarning` will be issued. This means that
    the coefficient values may be poorly determined. Using a lower order fit
    will usually get rid of the warning.  The `rcond` parameter can also be
    set to a value smaller than its default, but the resulting fit may be
    spurious and have large contributions from roundoff error.

    Fits using Laguerre series are probably most useful when the data can
    be approximated by ``sqrt(w(x)) * p(x)``, where ``w(x)`` is the Laguerre
    weight. In that case the weight ``sqrt(w(x[i]))`` should be used
    together with data values ``y[i]/sqrt(w(x[i]))``. The weight function is
    available as `lagweight`.

    References
    ----------
    .. [1] Wikipedia, "Curve fitting",
           https://en.wikipedia.org/wiki/Curve_fitting

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagfit, lagval
    >>> x = np.linspace(0, 10)
    >>> rng = np.random.default_rng()
    >>> err = rng.normal(scale=1./10, size=len(x))
    >>> y = lagval(x, [1, 2, 3]) + err
    >>> lagfit(x, y, 2)
    array([1.00578369, 1.99417356, 2.99827656]) # may vary

    """
    # 使用 lagvander 函数调用 _fit 函数，进行 Laguerre 系列拟合
    return pu._fit(lagvander, x, y, deg, rcond, full, w)
# 返回 Laguerre 多项式 c 的伴随矩阵。

# 当 c 是 Laguerre 多项式的基础时，通常的伴随矩阵已经是对称的，因此不需要进行缩放。

# Parameters 参数
# ----------
# c : array_like
#     按从低到高次序排列的 Laguerre 系数的一维数组。

# Returns 返回
# -------
# mat : ndarray
#     维度为 (deg, deg) 的伴随矩阵。

# Notes 注意
# -----
# 这个函数在版本 1.7.0 中被添加。

def lagcompanion(c):
    # c 是 c 的一个修剪副本
    [c] = pu.as_series([c])
    # 如果 c 的长度小于 2，则抛出值错误
    if len(c) < 2:
        raise ValueError('Series must have maximum degree of at least 1.')
    # 如果 c 的长度为 2，则返回一个数组
    if len(c) == 2:
        return np.array([[1 + c[0]/c[1]]])

    # n 是 c 长度减去 1
    n = len(c) - 1
    # 创建一个 dtype 为 c.dtype 的 n x n 的零矩阵
    mat = np.zeros((n, n), dtype=c.dtype)
    # 将矩阵展开后按列设置元素的引用
    top = mat.reshape(-1)[1::n+1]
    mid = mat.reshape(-1)[0::n+1]
    bot = mat.reshape(-1)[n::n+1]
    # 设置 top 为 -1 到 -n 的数组
    top[...] = -np.arange(1, n)
    # 设置 mid 为 2 到 2n+1 的数组
    mid[...] = 2.*np.arange(n) + 1.
    # 设置 bot 为 top 的引用
    bot[...] = top
    # 最后一列加上 (c[:-1]/c[-1])*n
    mat[:, -1] += (c[:-1]/c[-1])*n
    # 返回矩阵
    return mat


# 计算 Laguerre 级数的根。

# 返回多项式 p(x) = sum_i c[i] * L_i(x) 的根（也称为“零点”）。

# Parameters 参数
# ----------
# c : 1-D array_like
#     系数的一维数组。

# Returns 返回
# -------
# out : ndarray
#     级数的根数组。如果所有的根都是实数，则 out 也是实数，否则是复数。

# See Also 参见
# --------
# numpy.polynomial.polynomial.polyroots
# numpy.polynomial.legendre.legroots
# numpy.polynomial.chebyshev.chebroots
# numpy.polynomial.hermite.hermroots
# numpy.polynomial.hermite_e.hermeroots

# Notes 注意
# -----
# 根的估计是通过伴随矩阵的特征值获得的。远离复平面原点的根可能由于这些值的数值不稳定性而具有较大误差。
# 具有大于 1 的重复度的根也会显示较大的误差，因为在这些点附近，系列的值对根的误差相对不敏感。
# 靠近原点的孤立根可以通过牛顿法的几次迭代来改进。

# Laguerre 级数基础多项式不是 x 的幂，因此这个函数的结果可能看起来不直观。

def lagroots(c):
    # c 是 c 的一个修剪副本
    [c] = pu.as_series([c])
    # 如果 c 的长度小于等于 1，则返回一个空数组
    if len(c) <= 1:
        return np.array([], dtype=c.dtype)
    # 如果 c 的长度为 2，则返回一个数组
    if len(c) == 2:
        return np.array([1 + c[0]/c[1]])
    # 创建一个旋转后的伴随矩阵以减小误差
    m = lagcompanion(c)[::-1,::-1]  # 使用lagcompanion函数生成伴随矩阵，并进行行列逆序操作
    r = la.eigvals(m)  # 计算矩阵m的特征值
    r.sort()  # 对特征值进行排序
    return r  # 返回排序后的特征值数组
# 定义 Gauss-Laguerre 积分函数
def laggauss(deg):
    """
    Gauss-Laguerre quadrature.

    Computes the sample points and weights for Gauss-Laguerre quadrature.
    These sample points and weights will correctly integrate polynomials of
    degree :math:`2*deg - 1` or less over the interval :math:`[0, \\inf]`
    with the weight function :math:`f(x) = \\exp(-x)`.

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

    The results have only been tested up to degree 100; higher degrees may
    be problematic. The weights are determined by using the fact that

    .. math:: w_k = c / (L'_n(x_k) * L_{n-1}(x_k))

    where :math:`c` is a constant independent of :math:`k` and :math:`x_k`
    is the k'th root of :math:`L_n`, and then scaling the results to get
    the right value when integrating 1.

    Examples
    --------
    >>> from numpy.polynomial.laguerre import laggauss
    >>> laggauss(2)
    (array([0.58578644, 3.41421356]), array([0.85355339, 0.14644661]))

    """
    # 将 deg 转换为整数，如果无法转换，将引发异常
    ideg = pu._as_int(deg, "deg")
    if ideg <= 0:
        raise ValueError("deg must be a positive integer")

    # 创建伴随矩阵，并通过其特征值计算样本点
    c = np.array([0]*deg + [1])
    m = lagcompanion(c)
    x = la.eigvalsh(m)

    # 通过牛顿法进一步优化样本点的精度
    dy = lagval(x, c)
    df = lagval(x, lagder(c))
    x -= dy / df

    # 计算权重，并进行缩放以避免可能的数值溢出
    fm = lagval(x, c[1:])
    fm /= np.abs(fm).max()
    df /= np.abs(df).max()
    w = 1 / (fm * df)

    # 缩放权重以确保积分 1 的准确性
    w /= w.sum()

    return x, w


# 定义 Laguerre 多项式的权重函数
def lagweight(x):
    """Weight function of the Laguerre polynomials.

    The weight function is :math:`exp(-x)` and the interval of integration
    is :math:`[0, \\inf]`. The Laguerre polynomials are orthogonal, but not
    normalized, with respect to this weight function.

    Parameters
    ----------
    x : array_like
       Values at which the weight function will be computed.

    Returns
    -------
    w : ndarray
       The weight function at `x`.

    Notes
    -----

    .. versionadded:: 1.7.0

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagweight
    >>> x = np.array([0, 1, 2])
    >>> lagweight(x)
    array([1.        , 0.36787944, 0.13533528])

    """
    # 计算 Laguerre 多项式的权重，权重函数为 exp(-x)
    w = np.exp(-x)
    return w

#
# Laguerre series class
#

class Laguerre(ABCPolyBase):
    """A Laguerre series class.

    The Laguerre class provides the standard Python numerical methods
    '+', '-', '*', '//', '%', 'divmod', '**', and '()' as well as the
    attributes and methods listed below.

    Parameters
    ----------
    # Laguerre多项式的系数，按照递增次数排序，例如 (1, 2, 3) 给出的是 1*L_0(x) + 2*L_1(X) + 3*L_2(x)
    coef : array_like
    # （可选参数）要使用的区间。将区间 [domain[0], domain[1]] 通过移位和缩放映射到区间 [window[0], window[1]]。
    # 默认值为 [0, 1]。
    domain : (2,) array_like, optional
    # （可选参数）窗口，参见“domain”来使用它。默认值为 [0, 1]。
    # .. versionadded:: 1.6.0
    window : (2,) array_like, optional
    # （可选参数）用于表示多项式表达式中自变量的符号，例如用于打印。
    # 符号必须是有效的Python标识符。默认值为 'x'。
    # .. versionadded:: 1.24
    symbol : str, optional

    # 虚拟函数
    _add = staticmethod(lagadd)
    _sub = staticmethod(lagsub)
    _mul = staticmethod(lagmul)
    _div = staticmethod(lagdiv)
    _pow = staticmethod(lagpow)
    _val = staticmethod(lagval)
    _int = staticmethod(lagint)
    _der = staticmethod(lagder)
    _fit = staticmethod(lagfit)
    _line = staticmethod(lagline)
    _roots = staticmethod(lagroots)
    _fromroots = staticmethod(lagfromroots)

    # 虚拟属性
    domain = np.array(lagdomain)
    window = np.array(lagdomain)
    basis_name = 'L'
```