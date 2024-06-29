# `.\numpy\numpy\polynomial\hermite.py`

```
# 导入NumPy库并指定别名np，导入NumPy的线性代数模块别名为la
import numpy as np
import numpy.linalg as la
# 从NumPy库中的数组工具模块导入normalize_axis_index函数
from numpy.lib.array_utils import normalize_axis_index

# 从当前目录下的polyutils模块导入trimcoef函数并赋给hermtrim变量
from . import polyutils as pu

# 从当前目录下的_polybase模块导入ABCPolyBase类
from ._polybase import ABCPolyBase

# 定义__all__列表，列出模块中公开的函数和类的名称
__all__ = [
    'hermzero', 'hermone', 'hermx', 'hermdomain', 'hermline', 'hermadd',
    'hermsub', 'hermmulx', 'hermmul', 'hermdiv', 'hermpow', 'hermval',
    'hermder', 'hermint', 'herm2poly', 'poly2herm', 'hermfromroots',
    'hermvander', 'hermfit', 'hermtrim', 'hermroots', 'Hermite',
    'hermval2d', 'hermval3d', 'hermgrid2d', 'hermgrid3d', 'hermvander2d',
    'hermvander3d', 'hermcompanion', 'hermgauss', 'hermweight']

# 将pu.trimcoef函数赋给hermtrim变量，用于多项式系数的裁剪
hermtrim = pu.trimcoef


def poly2herm(pol):
    """
    poly2herm(pol)

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
    herm2poly

    Notes
    -----
    The easy way to do conversions between polynomial basis sets
    is to use the convert method of a class instance.

    Examples
    --------
    >>> from numpy.polynomial.hermite import poly2herm
    >>> poly2herm(np.arange(4))
    array([1.   ,  2.75 ,  0.5  ,  0.375])

    """
    # 将输入的多项式系数数组转换为NumPy的多项式系数表示形式
    [pol] = pu.as_series([pol])
    # 获取多项式的最高次数
    deg = len(pol) - 1
    # 初始化结果为0
    res = 0
    # 从最高次数到最低次数迭代
    for i in range(deg, -1, -1):
        # 逐步构建Hermite系数，hermadd为Hermite系数的加法操作，hermmulx为乘以x的操作
        res = hermadd(hermmulx(res), pol[i])
    # 返回函数中定义的变量 res
    return res
def herm2poly(c):
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
    poly2herm

    Notes
    -----
    The easy way to do conversions between polynomial basis sets
    is to use the convert method of a class instance.

    Examples
    --------
    >>> from numpy.polynomial.hermite import herm2poly
    >>> herm2poly([ 1.   ,  2.75 ,  0.5  ,  0.375])
    array([0., 1., 2., 3.])

    """
    from .polynomial import polyadd, polysub, polymulx  # 导入所需的多项式运算函数

    [c] = pu.as_series([c])  # 将输入的系数转换为数组表示
    n = len(c)  # 系数数组的长度
    if n == 1:
        return c  # 如果数组长度为1，直接返回该数组作为多项式系数
    if n == 2:
        c[1] *= 2  # 如果数组长度为2，将第二项乘以2
        return c  # 返回修改后的系数数组作为多项式系数
    else:
        c0 = c[-2]  # 初始化 c0 为倒数第二项系数
        c1 = c[-1]  # 初始化 c1 为最后一项系数
        # i 是当前 c1 的次数
        for i in range(n - 1, 1, -1):
            tmp = c0
            c0 = polysub(c[i - 2], c1*(2*(i - 1)))  # 更新 c0 的值
            c1 = polyadd(tmp, polymulx(c1)*2)  # 更新 c1 的值
        return polyadd(c0, polymulx(c1)*2)  # 返回多项式的系数数组


"""
These are constant arrays are of integer type so as to be compatible
with the widest range of other types, such as Decimal.
"""

# Hermite
hermdomain = np.array([-1., 1.])  # Hermite 多项式的定义域

# Hermite coefficients representing zero.
hermzero = np.array([0])  # 表示零的 Hermite 系数

# Hermite coefficients representing one.
hermone = np.array([1])  # 表示一的 Hermite 系数

# Hermite coefficients representing the identity x.
hermx = np.array([0, 1/2])  # 表示恒等函数 x 的 Hermite 系数


def hermline(off, scl):
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
    numpy.polynomial.hermite_e.hermeline

    Examples
    --------
    >>> from numpy.polynomial.hermite import hermline, hermval
    >>> hermval(0,hermline(3, 2))
    3.0
    >>> hermval(1,hermline(3, 2))
    5.0

    """
    if scl != 0:
        return np.array([off, scl/2])  # 如果斜率不为零，返回 Hermite 系数数组
    else:
        return np.array([off])  # 如果斜率为零，返回 Hermite 系数数组


def hermfromroots(roots):
    """
    Generate a Hermite series with given roots.

    The function returns the coefficients of the polynomial

    .. math:: p(x) = (x - r_0) * (x - r_1) * ... * (x - r_n),

    ```
    # 返回 Hermite 多项式的系数，这些系数定义了一个多项式，其根由参数 `roots` 指定
    # 如果一个零点具有多重度 n，则它在 `roots` 中必须出现 n 次
    # 例如，如果 2 是一个三重根，3 是一个双重根，则 `roots` 可能是 [2, 2, 2, 3, 3]
    # 根可以以任何顺序出现

    # 如果返回的系数是 `c`，则 Hermite 多项式 `p(x)` 定义为：
    # p(x) = c_0 + c_1 * H_1(x) + ... + c_n * H_n(x)
    # 其中 `H_i(x)` 是 Hermite 多项式的基函数

    # 最后一项的系数通常不为 1，即使 Hermite 多项式是首一的（monic）

    # 参数：
    # roots : array_like
    #     包含根的序列

    # 返回：
    # out : ndarray
    #     1-D 系数数组。如果所有的根都是实数，则 `out` 是实数组；如果其中一些根是复数，则 `out` 是复数数组，
    #     即使结果中的所有系数都是实数（参见下面的示例）

    # 参见：
    # numpy.polynomial.polynomial.polyfromroots
    # numpy.polynomial.legendre.legfromroots
    # numpy.polynomial.laguerre.lagfromroots
    # numpy.polynomial.chebyshev.chebfromroots
    # numpy.polynomial.hermite_e.hermefromroots

    # 示例：
    # >>> from numpy.polynomial.hermite import hermfromroots, hermval
    # >>> coef = hermfromroots((-1, 0, 1))
    # >>> hermval((-1, 0, 1), coef)
    # array([0.,  0.,  0.])
    # >>> coef = hermfromroots((-1j, 1j))
    # >>> hermval((-1j, 1j), coef)
    # array([0.+0.j, 0.+0.j])

    # 返回通过 `_fromroots` 函数计算得到的 Hermite 多项式的系数
    return pu._fromroots(hermline, hermmul, roots)
# 定义函数 hermadd，用于将两个 Hermite 级数相加
def hermadd(c1, c2):
    # 调用私有函数 _add，返回两个 Hermite 级数的和
    return pu._add(c1, c2)


# 定义函数 hermsub，用于从一个 Hermite 级数中减去另一个 Hermite 级数
def hermsub(c1, c2):
    # 调用私有函数 _sub，返回两个 Hermite 级数的差
    return pu._sub(c1, c2)


# 定义函数 hermmulx，用于将 Hermite 级数乘以自变量 x
def hermmulx(c):
    """
    Multiply a Hermite series by x.

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
    hermadd, hermsub, hermmul, hermdiv, hermpow

    Notes
    -----
    The multiplication uses the recursion relationship for Hermite
    polynomials in the form

    .. math::

        xP_i(x) = (P_{i + 1}(x)/2 + i*P_{i - 1}(x))

    Examples
    --------
    >>> from numpy.polynomial.hermite import hermmulx
    >>> hermmulx([1, 2, 3])
    array([2. , 6.5, 1. , 1.5])

    """
    # 使用 pu.as_series([c]) 将 c 转换为 Hermite 级数，并返回其修剪版本
    [c] = pu.as_series([c])
    # 如果列表 c 的长度为 1 并且唯一的元素是 0，则直接返回 c，因为零系列需要特殊处理
    if len(c) == 1 and c[0] == 0:
        return c

    # 创建一个新的 NumPy 数组 prd，长度比 c 多 1，数据类型与 c 相同
    prd = np.empty(len(c) + 1, dtype=c.dtype)
    
    # 计算 prd 的第一个元素，设为 c 的第一个元素乘以 0
    prd[0] = c[0]*0
    
    # 计算 prd 的第二个元素，设为 c 的第一个元素除以 2
    prd[1] = c[0]/2
    
    # 遍历列表 c 的每个元素（从第二个到最后一个）
    for i in range(1, len(c)):
        # 计算 prd 的第 i+1 个元素，设为 c 的第 i 个元素除以 2
        prd[i + 1] = c[i]/2
        
        # 计算 prd 的第 i-1 个元素，加上 c 的第 i 个元素乘以 i
        prd[i - 1] += c[i]*i
    
    # 返回计算后的数组 prd
    return prd
def hermmul(c1, c2):
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
    hermadd, hermsub, hermmulx, hermdiv, hermpow

    Notes
    -----
    In general, the (polynomial) product of two C-series results in terms
    that are not in the Hermite polynomial basis set.  Thus, to express
    the product as a Hermite series, it is necessary to "reproject" the
    product onto said basis set, which may produce "unintuitive" (but
    correct) results; see Examples section below.

    Examples
    --------
    >>> from numpy.polynomial.hermite import hermmul
    >>> hermmul([1, 2, 3], [0, 1, 2])
    array([52.,  29.,  52.,   7.,   6.])

    """
    # 将输入的系数序列转换为标准的 Hermite 系列格式
    [c1, c2] = pu.as_series([c1, c2])

    # 选择较短的系数序列进行计算
    if len(c1) > len(c2):
        c = c2
        xs = c1
    else:
        c = c1
        xs = c2

    # 根据系数序列的长度选择不同的计算方式
    if len(c) == 1:
        # 如果系数序列长度为1，直接计算乘积
        c0 = c[0]*xs
        c1 = 0
    elif len(c) == 2:
        # 如果系数序列长度为2，按照二阶 Hermite 系列的计算方式进行乘积计算
        c0 = c[0]*xs
        c1 = c[1]*xs
    else:
        # 对于长度大于2的系数序列，采用递推计算 Hermite 系列的乘积
        nd = len(c)
        c0 = c[-2]*xs
        c1 = c[-1]*xs
        for i in range(3, len(c) + 1):
            tmp = c0
            nd = nd - 1
            c0 = hermsub(c[-i]*xs, c1*(2*(nd - 1)))
            c1 = hermadd(tmp, hermmulx(c1)*2)

    # 返回计算得到的 Hermite 系列乘积
    return hermadd(c0, hermmulx(c1)*2)


def hermdiv(c1, c2):
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
    hermadd, hermsub, hermmulx, hermmul, hermpow

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
    >>> from numpy.polynomial.hermite import hermdiv
    >>> hermdiv([ 52.,  29.,  52.,   7.,   6.], [0, 1, 2])
    (array([1., 2., 3.]), array([0.]))

    """
    # 将输入的系数序列转换为标准的 Hermite 系列格式
    [c1, c2] = pu.as_series([c1, c2])

    # 计算商和余数
    quo, rem = np.polydiv(c1, c2)

    # 返回计算得到的 Hermite 系列商和余数
    return quo, rem
    # 调用 hermdiv 函数，传入参数 [54., 31., 52., 7., 6.] 和 [0, 1, 2]
    # 返回结果为 (array([1., 2., 3.]), array([2., 2.]))
    >>> hermdiv([ 54.,  31.,  52.,   7.,   6.], [0, 1, 2])
    (array([1., 2., 3.]), array([2., 2.]))

    # 调用 hermdiv 函数，传入参数 [53., 30., 52., 7., 6.] 和 [0, 1, 2]
    # 返回结果为 (array([1., 2., 3.]), array([1., 1.]))
    >>> hermdiv([ 53.,  30.,  52.,   7.,   6.], [0, 1, 2])
    (array([1., 2., 3.]), array([1., 1.]))

    """
    # 返回 pu._div(hermmul, c1, c2) 的结果
    return pu._div(hermmul, c1, c2)
# 定义函数 hermder，用于对 Hermite 级数进行求导操作
def hermder(c, m=1, scl=1, axis=0):
    """
    Differentiate a Hermite series.

    Returns the Hermite series coefficients `c` differentiated `m` times
    along `axis`.  At each iteration the result is multiplied by `scl` (the
    scaling factor is for use in a linear change of variable). The argument
    `c` is an array of coefficients from low to high degree along each
    axis, e.g., [1,2,3] represents the series ``1*H_0 + 2*H_1 + 3*H_2``
    while [[1,2],[1,2]] represents ``1*H_0(x)*H_0(y) + 1*H_1(x)*H_0(y) +
    2*H_0(x)*H_1(y) + 2*H_1(x)*H_1(y)`` if axis=0 is ``x`` and axis=1 is
    ``y``.

    Parameters
    ----------
    c : array_like
        Array of Hermite series coefficients. If `c` is multidimensional the
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
        Hermite series of the derivative.

    See Also
    --------
    hermint

    Notes
    -----
    In general, the result of differentiating a Hermite series does not
    resemble the same operation on a power series. Thus the result of this
    function may be "unintuitive," albeit correct; see Examples section
    below.

    Examples
    --------
    >>> from numpy.polynomial.hermite import hermder
    >>> hermder([ 1. ,  0.5,  0.5,  0.5])
    array([1., 2., 3.])
    >>> hermder([-0.5,  1./2.,  1./8.,  1./12.,  1./16.], m=2)
    array([1., 2., 3.])

    """
    # 将输入的 Hermite 系数数组转换为至少为一维的 numpy 数组，并进行深拷贝
    c = np.array(c, ndmin=1, copy=True)
    # 如果输入数组的数据类型是布尔型或整数型，则转换为双精度浮点型
    if c.dtype.char in '?bBhHiIlLqQpP':
        c = c.astype(np.double)
    # 将 m 转换为整数，表示导数的阶数 cnt
    cnt = pu._as_int(m, "the order of derivation")
    # 将 axis 转换为整数，表示操作的轴向 iaxis
    iaxis = pu._as_int(axis, "the axis")
    # 如果导数阶数 cnt 小于 0，则抛出数值错误异常
    if cnt < 0:
        raise ValueError("The order of derivation must be non-negative")
    # 根据数组 c 的维度调整轴向 iaxis，确保其在有效范围内
    iaxis = normalize_axis_index(iaxis, c.ndim)

    # 如果导数阶数 cnt 为 0，则直接返回数组 c
    if cnt == 0:
        return c

    # 将轴向 iaxis 移动到数组 c 的第一个维度
    c = np.moveaxis(c, iaxis, 0)
    # 获取数组 c 的长度
    n = len(c)
    # 如果导数阶数 cnt 大于等于数组长度 n，则将数组 c 的前一项乘以 0
    if cnt >= n:
        c = c[:1]*0
    else:
        # 否则，进行 cnt 次导数运算
        for i in range(cnt):
            n = n - 1
            # 将数组 c 每一项乘以缩放因子 scl
            c *= scl
            # 创建一个空的数组 der，用于存储导数结果
            der = np.empty((n,) + c.shape[1:], dtype=c.dtype)
            # 计算每一阶导数的值并存储在 der 中
            for j in range(n, 0, -1):
                der[j - 1] = (2*j)*c[j]
            # 将计算得到的导数结果赋值给数组 c
            c = der
    # 将第一个维度移动回原来的轴向 iaxis
    c = np.moveaxis(c, 0, iaxis)
    # 返回最终的导数结果数组 c
    return c
# 定义函数 hermint，用于对 Hermite 级数进行积分处理
def hermint(c, m=1, k=[], lbnd=0, scl=1, axis=0):
    """
    Integrate a Hermite series.

    Returns the Hermite series coefficients `c` integrated `m` times from
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
        Array of Hermite series coefficients. If c is multidimensional the
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
        Hermite series coefficients of the integral.

    Raises
    ------
    ValueError
        If ``m < 0``, ``len(k) > m``, ``np.ndim(lbnd) != 0``, or
        ``np.ndim(scl) != 0``.

    See Also
    --------
    hermder

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
    >>> from numpy.polynomial.hermite import hermint
    >>> hermint([1,2,3]) # integrate once, value 0 at 0.
    array([1. , 0.5, 0.5, 0.5])
    """
    c = np.array(c, ndmin=1, copy=True)
    # 将输入的数组 c 转换为 NumPy 数组，确保至少是一维的，并且进行复制以防止原始数据被修改
    if c.dtype.char in '?bBhHiIlLqQpP':
        # 检查数组 c 的数据类型是否属于布尔型、整型或指针类型之一，如果是，则将其转换为双精度浮点数
        c = c.astype(np.double)
    if not np.iterable(k):
        # 如果 k 不可迭代（即不是列表、元组等），则将其转换为包含 k 的列表
        k = [k]
    cnt = pu._as_int(m, "the order of integration")
    # 将 m 转换为整数，如果无法转换则会引发错误，用于表示积分的阶数
    iaxis = pu._as_int(axis, "the axis")
    # 将 axis 转换为整数，用于表示数组的轴
    if cnt < 0:
        # 如果积分的阶数小于 0，则引发 ValueError
        raise ValueError("The order of integration must be non-negative")
    if len(k) > cnt:
        # 如果 k 中的积分常数个数多于阶数 cnt，则引发 ValueError
        raise ValueError("Too many integration constants")
    if np.ndim(lbnd) != 0:
        # 如果 lbnd 的维度不为 0（即不是标量），则引发 ValueError
        raise ValueError("lbnd must be a scalar.")
    if np.ndim(scl) != 0:
        # 如果 scl 的维度不为 0（即不是标量），则引发 ValueError
        raise ValueError("scl must be a scalar.")
    iaxis = normalize_axis_index(iaxis, c.ndim)
    # 根据数组 c 的维度和指定的轴值对 iaxis 进行标准化处理

    if cnt == 0:
        # 如果积分的阶数为 0，则直接返回数组 c，不进行积分计算
        return c

    c = np.moveaxis(c, iaxis, 0)
    # 将数组 c 的轴移动到指定位置，这里是将第 iaxis 轴移动到第 0 位置
    k = list(k) + [0]*(cnt - len(k))
    # 将 k 扩展到长度为 cnt，不足部分用 0 填充
    for i in range(cnt):
        # 循环进行 cnt 次积分操作
        n = len(c)
        c *= scl
        # 将数组 c 的每个元素乘以 scl
        if n == 1 and np.all(c[0] == 0):
            # 如果数组 c 只有一个元素且该元素全为 0，则在第一个元素上加上 k[i]
            c[0] += k[i]
        else:
            # 否则，创建临时数组 tmp，并进行 Hermite 插值
            tmp = np.empty((n + 1,) + c.shape[1:], dtype=c.dtype)
            tmp[0] = c[0]*0
            tmp[1] = c[0]/2
            for j in range(1, n):
                tmp[j + 1] = c[j]/(2*(j + 1))
            tmp[0] += k[i] - hermval(lbnd, tmp)
            c = tmp
    c = np.moveaxis(c, 0, iaxis)
    # 将数组 c 的轴移回原始位置
    return c
    # 返回积分结果的数组 c
def hermval(x, c, tensor=True):
    """
    Evaluate an Hermite series at points x.

    If `c` is of length ``n + 1``, this function returns the value:

    .. math:: p(x) = c_0 * H_0(x) + c_1 * H_1(x) + ... + c_n * H_n(x)

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
    hermval2d, hermgrid2d, hermval3d, hermgrid3d

    Notes
    -----
    The evaluation uses Clenshaw recursion, aka synthetic division.

    Examples
    --------
    >>> from numpy.polynomial.hermite import hermval
    >>> coef = [1,2,3]
    >>> hermval(1, coef)
    11.0
    >>> hermval([[1,2],[3,4]], coef)
    array([[ 11.,   51.],
           [115.,  203.]])

    """
    # 将系数数组 `c` 转换为至少为一维的 numpy 数组
    c = np.array(c, ndmin=1, copy=None)
    # 如果 `c` 的数据类型是布尔类型或整数类型，则转换为双精度浮点数类型
    if c.dtype.char in '?bBhHiIlLqQpP':
        c = c.astype(np.double)
    # 如果 `x` 是 tuple 或 list 类型，则将其转换为 numpy 数组
    if isinstance(x, (tuple, list)):
        x = np.asarray(x)
    # 如果 `x` 是 numpy 数组且 `tensor` 参数为 True，则调整 `c` 的形状
    if isinstance(x, np.ndarray) and tensor:
        c = c.reshape(c.shape + (1,)*x.ndim)

    # 计算 `x` 的两倍
    x2 = x * 2
    # 如果系数数组 `c` 的长度为 1
    if len(c) == 1:
        # 取出第一个系数和设置第二个系数为 0
        c0 = c[0]
        c1 = 0
    elif len(c) == 2:
        # 如果列表 c 的长度为 2，则执行以下操作
        c0 = c[0]
        # 取列表 c 的第一个元素赋值给 c0
        c1 = c[1]
        # 取列表 c 的第二个元素赋值给 c1
    else:
        # 如果列表 c 的长度不为 2，则执行以下操作
        nd = len(c)
        # 获取列表 c 的长度并赋值给 nd
        c0 = c[-2]
        # 取列表 c 的倒数第二个元素赋值给 c0
        c1 = c[-1]
        # 取列表 c 的最后一个元素赋值给 c1
        for i in range(3, len(c) + 1):
            # 循环迭代，从 3 到列表 c 的长度（加 1）
            tmp = c0
            # 将 c0 的值赋给临时变量 tmp
            nd = nd - 1
            # nd 减 1
            c0 = c[-i] - c1*(2*(nd - 1))
            # 更新 c0 的值，使用列表 c 的倒数第 i 个元素减去 c1 乘以表达式结果
            c1 = tmp + c1*x2
            # 更新 c1 的值，将 tmp 加上 c1 乘以 x2
    return c0 + c1*x2
    # 返回 c0 加上 c1 乘以 x2 的结果
# 根据给定的 Hermite 系数和输入的 x, y 值，计算二维 Hermite 级数的值
def hermval2d(x, y, c):
    # 使用内部函数 _valnd 对 hermval 进行求值，并返回结果
    return pu._valnd(hermval, c, x, y)


def hermgrid2d(x, y, c):
    """
    Evaluate a 2-D Hermite series on the Cartesian product of x and y.

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
    hermval, hermval2d, hermval3d, hermgrid3d

    Notes
    -----

    .. versionadded:: 1.7.0

    Examples
    --------
    >>> from numpy.polynomial.hermite import hermval2d
    >>> x = [1, 2]
    >>> y = [4, 5]
    >>> c = [[1, 2, 3], [4, 5, 6]]
    >>> hermval2d(x, y, c)
    array ([1035., 2883.])

    """
    # 定义函数 hermgrid2d，用于计算二维埃尔米特多项式在给定点上的值
    # x, y : array_like, 兼容的对象，表示二维系列在笛卡尔积上的点的评估
    # 如果 x 或 y 是列表或元组，则首先转换为 ndarray；否则保持不变，如果不是 ndarray，则视为标量处理
    # c : array_like，系数数组，按照度 i,j 的顺序排列，系数为 ``c[i,j]``。如果 `c` 的维度大于二，则其余索引用于枚举多组系数

    # 返回值
    # ------
    # values : ndarray，兼容的对象
    # 在 `x` 和 `y` 的笛卡尔积上的二维多项式的值

    # 参见
    # --------
    # hermval, hermval2d, hermval3d, hermgrid3d

    # 注意
    # -----
    # .. versionadded:: 1.7.0

    # 示例
    # --------
    # >>> from numpy.polynomial.hermite import hermgrid2d
    # >>> x = [1, 2, 3]
    # >>> y = [4, 5]
    # >>> c = [[1, 2, 3], [4, 5, 6]]
    # >>> hermgrid2d(x, y, c)
    # array([[1035., 1599.],
    #        [1867., 2883.],
    #        [2699., 4167.]])
    """
    调用 _gridnd 函数来计算二维埃尔米特多项式在给定点 (x, y) 上的值，使用了 hermval 函数和系数 c
    """
    return pu._gridnd(hermval, c, x, y)
# 导入 pu 模块的 _valnd 函数，用于计算多维 Hermite 多项式的值
# 返回 _valnd 函数的结果，该函数计算 hermval 函数的值
def hermval3d(x, y, z, c):
    return pu._valnd(hermval, c, x, y, z)


# 在三维笛卡尔积上评估三维 Hermite 级数
# 返回结果是三维 Hermite 级数在点对 (x, y, z) 组成的笛卡尔积上的值
# x, y, z 参数被转换为数组，如果它们是元组或列表；否则被视为标量
# 如果 c 的维数少于三维，会隐式添加维度使其成为三维
# 结果的形状将是 c.shape[3:] + x.shape
def hermgrid3d(x, y, z, c):
    # 返回三维多项式的值在给定点的笛卡尔乘积中
    Parameters
    ----------
    x, y, z : array_like, compatible objects
        在`x`、`y`、`z`的笛卡尔乘积中评估三维系列。如果`x`、`y`或`z`是列表或元组，则首先转换为ndarray；否则保持不变，并且如果它不是ndarray，则视为标量。
    c : array_like
        系数数组，按照i、j度项的系数包含在`c[i,j]`中。如果`c`的维度大于两，则剩余索引列举多组系数。

    Returns
    -------
    values : ndarray, compatible object
        三维多项式在`x`和`y`的笛卡尔乘积中的值。

    See Also
    --------
    hermval, hermval2d, hermgrid2d, hermval3d

    Notes
    -----

    .. versionadded:: 1.7.0

    Examples
    --------
    >>> from numpy.polynomial.hermite import hermgrid3d
    >>> x = [1, 2]
    >>> y = [4, 5]
    >>> z = [6, 7]
    >>> c = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
    >>> hermgrid3d(x, y, z, c)
    返回三维Hermite多项式在给定点处的值。
    array([[[ 40077.,  54117.],
            [ 49293.,  66561.]],
           [[ 72375.,  97719.],
            [ 88975., 120131.]]])

    """
    return pu._gridnd(hermval, c, x, y, z)
def hermvander(x, deg):
    """Pseudo-Vandermonde matrix of given degree.

    Returns the pseudo-Vandermonde matrix of degree `deg` and sample points
    `x`. The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., i] = H_i(x),

    where ``0 <= i <= deg``. The leading indices of `V` index the elements of
    `x` and the last index is the degree of the Hermite polynomial.

    If `c` is a 1-D array of coefficients of length ``n + 1`` and `V` is the
    array ``V = hermvander(x, n)``, then ``np.dot(V, c)`` and
    ``hermval(x, c)`` are the same up to roundoff. This equivalence is
    useful both for least squares fitting and for the evaluation of a large
    number of Hermite series of the same degree and sample points.

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
        corresponding Hermite polynomial.  The dtype will be the same as
        the converted `x`.

    Examples
    --------
    >>> from numpy.polynomial.hermite import hermvander
    >>> x = np.array([-1, 0, 1])
    >>> hermvander(x, 3)
    array([[ 1., -2.,  2.,  4.],
           [ 1.,  0., -2., -0.],
           [ 1.,  2.,  2., -4.]])

    """
    # Convert deg to an integer if possible
    ideg = pu._as_int(deg, "deg")
    # Raise ValueError if deg is negative
    if ideg < 0:
        raise ValueError("deg must be non-negative")

    # Ensure x is converted to a 1-D array of type float64 or complex128
    x = np.array(x, copy=None, ndmin=1) + 0.0
    # Determine the dimensions of the output matrix
    dims = (ideg + 1,) + x.shape
    # Determine the dtype for the output matrix
    dtyp = x.dtype
    # Create an empty array with the determined dimensions and dtype
    v = np.empty(dims, dtype=dtyp)
    # Initialize the first row of v to be 1s
    v[0] = x*0 + 1
    # Compute subsequent rows using Hermite polynomial recurrence relations
    if ideg > 0:
        x2 = x*2
        v[1] = x2
        for i in range(2, ideg + 1):
            v[i] = (v[i-1]*x2 - v[i-2]*(2*(i - 1)))
    # Return the pseudo-Vandermonde matrix with the first axis moved to the last
    return np.moveaxis(v, 0, -1)


def hermvander2d(x, y, deg):
    """Pseudo-Vandermonde matrix of given degrees.

    Returns the pseudo-Vandermonde matrix of degrees `deg` and sample
    points ``(x, y)``. The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., (deg[1] + 1)*i + j] = H_i(x) * H_j(y),

    where ``0 <= i <= deg[0]`` and ``0 <= j <= deg[1]``. The leading indices of
    `V` index the points ``(x, y)`` and the last index encodes the degrees of
    the Hermite polynomials.

    If ``V = hermvander2d(x, y, [xdeg, ydeg])``, then the columns of `V`
    correspond to the elements of a 2-D coefficient array `c` of shape
    (xdeg + 1, ydeg + 1) in the order

    .. math:: c_{00}, c_{01}, c_{02} ... , c_{10}, c_{11}, c_{12} ...

    and ``np.dot(V, c.flat)`` and ``hermval2d(x, y, c)`` will be the same
    up to roundoff. This equivalence is useful both for least squares
    fitting and for the evaluation of a large number of 2-D Hermite

    """
    # Function definition for computing a pseudo-Vandermonde matrix for 2D Hermite polynomials
    # The detailed mathematical description is provided in the docstring
    # 返回一个二维埃尔米特（Hermite）Vandermonde 矩阵，用于给定的一系列点和最大度数。

    Parameters
    ----------
    x, y : array_like
        表示点的坐标数组，形状相同。元素的数据类型将转换为float64或complex128，取决于是否有复数元素。
        标量将转换为1维数组。
    deg : list of ints
        最大度数的列表，形式为 [x_deg, y_deg]。

    Returns
    -------
    vander2d : ndarray
        返回的矩阵形状为 ``x.shape + (order,)``，其中 :math:`order = (deg[0]+1)*(deg[1]+1)`。
        数据类型与转换后的 `x` 和 `y` 相同。

    See Also
    --------
    hermvander, hermvander3d, hermval2d, hermval3d

    Notes
    -----

    .. versionadded:: 1.7.0

    Examples
    --------
    >>> from numpy.polynomial.hermite import hermvander2d
    >>> x = np.array([-1, 0, 1])
    >>> y = np.array([-1, 0, 1])
    >>> hermvander2d(x, y, [2, 2])
    array([[ 1., -2.,  2., -2.,  4., -4.,  2., -4.,  4.],
           [ 1.,  0., -2.,  0.,  0., -0., -2., -0.,  4.],
           [ 1.,  2.,  2.,  2.,  4.,  4.,  2.,  4.,  4.]])

    """
    return pu._vander_nd_flat((hermvander, hermvander), (x, y), deg)
def hermvander3d(x, y, z, deg):
    """
    Pseudo-Vandermonde matrix of given degrees.

    Returns the pseudo-Vandermonde matrix of degrees `deg` and sample
    points ``(x, y, z)``. If `l`, `m`, `n` are the given degrees in `x`, `y`, `z`,
    then The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., (m+1)(n+1)i + (n+1)j + k] = H_i(x)*H_j(y)*H_k(z),

    where ``0 <= i <= l``, ``0 <= j <= m``, and ``0 <= j <= n``.  The leading
    indices of `V` index the points ``(x, y, z)`` and the last index encodes
    the degrees of the Hermite polynomials.

    If ``V = hermvander3d(x, y, z, [xdeg, ydeg, zdeg])``, then the columns
    of `V` correspond to the elements of a 3-D coefficient array `c` of
    shape (xdeg + 1, ydeg + 1, zdeg + 1) in the order

    .. math:: c_{000}, c_{001}, c_{002},... , c_{010}, c_{011}, c_{012},...

    and  ``np.dot(V, c.flat)`` and ``hermval3d(x, y, z, c)`` will be the
    same up to roundoff. This equivalence is useful both for least squares
    fitting and for the evaluation of a large number of 3-D Hermite
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
    hermvander, hermvander3d, hermval2d, hermval3d

    Notes
    -----

    .. versionadded:: 1.7.0

    Examples
    --------
    >>> from numpy.polynomial.hermite import hermvander3d
    >>> x = np.array([-1, 0, 1])
    >>> y = np.array([-1, 0, 1])
    >>> z = np.array([-1, 0, 1])
    >>> hermvander3d(x, y, z, [0, 1, 2])
    array([[ 1., -2.,  2., -2.,  4., -4.],
           [ 1.,  0., -2.,  0.,  0., -0.],
           [ 1.,  2.,  2.,  2.,  4.,  4.]])

    """
    return pu._vander_nd_flat((hermvander, hermvander, hermvander), (x, y, z), deg)
    x : array_like, shape (M,)
        # M个样本点的x坐标，形状为(M,)，即(x[i], y[i])中的x[i]。
    y : array_like, shape (M,) or (M, K)
        # 样本点的y坐标。可以是形状为(M,)或(M, K)的数组，多组共享相同x坐标的样本点可以一次性拟合，
        # 通过传入一个包含每列数据集的2D数组来实现。
    deg : int or 1-D array_like
        # 拟合多项式的次数。如果deg是一个整数，则包括到第deg阶的所有项。对于NumPy版本 >= 1.11.0，
        # 可以使用包含要包括的项的次数的整数列表。
    rcond : float, optional
        # 拟合条件数的相对值。比最大奇异值小于这个值的奇异值将被忽略。默认值为len(x)*eps，
        # 其中eps是浮点类型的相对精度，大多数情况下约为2e-16。
    full : bool, optional
        # 决定返回值类型的开关。当为False（默认）时，只返回系数；当为True时，还返回奇异值分解的诊断信息。
    w : array_like, shape (`M`,), optional
        # 权重。如果不为None，则权重``w[i]``适用于在``x[i]``处的未平方残差``y[i] - y_hat[i]``。
        # 理想情况下，选择权重使得所有产品``w[i]*y[i]``的误差具有相同的方差。当使用逆方差加权时，
        # 使用``w[i] = 1/sigma(y[i])``。默认值为None。

    Returns
    -------
    coef : ndarray, shape (M,) or (M, K)
        # 按从低到高排序的Hermite系数。如果`y`是2D的，则列k中的数据的系数位于列`k`中。

    [residuals, rank, singular_values, rcond] : list
        # 仅在`full == True`时返回这些值

        - residuals -- 最小二乘拟合的残差平方和
        - rank -- 缩放的Vandermonde矩阵的数值秩
        - singular_values -- 缩放的Vandermonde矩阵的奇异值
        - rcond -- `rcond`的值

        有关更多详细信息，请参阅`numpy.linalg.lstsq`。

    Warns
    -----
    RankWarning
        # 最小二乘拟合中的系数矩阵的秩不足。仅当`full == False`时才会引发警告。
        # 可以通过以下方式关闭警告：

        >>> import warnings
        >>> warnings.simplefilter('ignore', np.exceptions.RankWarning)

    See Also
    --------
    numpy.polynomial.chebyshev.chebfit
    numpy.polynomial.legendre.legfit
    numpy.polynomial.laguerre.lagfit
    numpy.polynomial.polynomial.polyfit
    numpy.polynomial.hermite_e.hermefit
    hermval : 评估Hermite级数。
    hermvander : Hermite级数的Vandermonde矩阵。
    hermweight : Hermite权重函数
    numpy.linalg.lstsq : Computes a least-squares fit from the matrix.
    scipy.interpolate.UnivariateSpline : Computes spline fits.

    Notes
    -----
    The solution is the coefficients of the Hermite series `p` that
    minimizes the sum of the weighted squared errors

    .. math:: E = \\sum_j w_j^2 * |y_j - p(x_j)|^2,

    where the :math:`w_j` are the weights. This problem is solved by
    setting up the (typically) overdetermined matrix equation

    .. math:: V(x) * c = w * y,

    where `V` is the weighted pseudo Vandermonde matrix of `x`, `c` are the
    coefficients to be solved for, `w` are the weights, `y` are the
    observed values.  This equation is then solved using the singular value
    decomposition of `V`.

    If some of the singular values of `V` are so small that they are
    neglected, then a `~exceptions.RankWarning` will be issued. This means that
    the coefficient values may be poorly determined. Using a lower order fit
    will usually get rid of the warning.  The `rcond` parameter can also be
    set to a value smaller than its default, but the resulting fit may be
    spurious and have large contributions from roundoff error.

    Fits using Hermite series are probably most useful when the data can be
    approximated by ``sqrt(w(x)) * p(x)``, where ``w(x)`` is the Hermite
    weight. In that case the weight ``sqrt(w(x[i]))`` should be used
    together with data values ``y[i]/sqrt(w(x[i]))``. The weight function is
    available as `hermweight`.

    References
    ----------
    .. [1] Wikipedia, "Curve fitting",
           https://en.wikipedia.org/wiki/Curve_fitting

    Examples
    --------
    >>> from numpy.polynomial.hermite import hermfit, hermval
    >>> x = np.linspace(-10, 10)
    >>> rng = np.random.default_rng()
    >>> err = rng.normal(scale=1./10, size=len(x))
    >>> y = hermval(x, [1, 2, 3]) + err
    >>> hermfit(x, y, 2)
    array([1.02294967, 2.00016403, 2.99994614]) # may vary

    """
    return pu._fit(hermvander, x, y, deg, rcond, full, w)


注释：


    # 调用 `_fit` 函数进行曲线拟合，使用 Hermite 系列作为基础
    return pu._fit(hermvander, x, y, deg, rcond, full, w)


这段代码是一个函数的返回语句，返回了 `_fit` 函数的调用结果，该函数用于进行曲线拟合，其中使用了 Hermite 系列作为基础。
def hermcompanion(c):
    """Return the scaled companion matrix of c.

    The basis polynomials are scaled so that the companion matrix is
    symmetric when `c` is an Hermite basis polynomial. This provides
    better eigenvalue estimates than the unscaled case and for basis
    polynomials the eigenvalues are guaranteed to be real if
    `numpy.linalg.eigvalsh` is used to obtain them.

    Parameters
    ----------
    c : array_like
        1-D array of Hermite series coefficients ordered from low to high
        degree.

    Returns
    -------
    mat : ndarray
        Scaled companion matrix of dimensions (deg, deg).

    Notes
    -----

    .. versionadded:: 1.7.0

    Examples
    --------
    >>> from numpy.polynomial.hermite import hermcompanion
    >>> hermcompanion([1, 0, 1])
    array([[0.        , 0.35355339],
           [0.70710678, 0.        ]])

    """
    # c is a trimmed copy
    [c] = pu.as_series([c])  # 将输入的系数数组c转换为多项式对象
    if len(c) < 2:
        raise ValueError('Series must have maximum degree of at least 1.')
    if len(c) == 2:
        return np.array([[-.5*c[0]/c[1]]])  # 对于一阶多项式，返回其伴随矩阵的特殊形式

    n = len(c) - 1
    mat = np.zeros((n, n), dtype=c.dtype)  # 创建一个全零的方阵，数据类型与输入系数数组c相同
    scl = np.hstack((1., 1./np.sqrt(2.*np.arange(n - 1, 0, -1))))  # 创建一个缩放系数向量
    scl = np.multiply.accumulate(scl)[::-1]  # 对缩放系数向量进行累积乘积并反向排序
    top = mat.reshape(-1)[1::n+1]  # 提取伴随矩阵中的上三角元素
    bot = mat.reshape(-1)[n::n+1]  # 提取伴随矩阵中的下三角元素
    top[...] = np.sqrt(.5*np.arange(1, n))  # 设置上三角元素的值
    bot[...] = top  # 设置下三角元素的值
    mat[:, -1] -= scl*c[:-1]/(2.0*c[-1])  # 修改伴随矩阵的最后一列
    return mat  # 返回计算得到的伴随矩阵


def hermroots(c):
    """
    Compute the roots of a Hermite series.

    Return the roots (a.k.a. "zeros") of the polynomial

    .. math:: p(x) = \\sum_i c[i] * H_i(x).

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
    numpy.polynomial.legendre.legroots
    numpy.polynomial.laguerre.lagroots
    numpy.polynomial.chebyshev.chebroots
    numpy.polynomial.hermite_e.hermeroots

    Notes
    -----
    The root estimates are obtained as the eigenvalues of the companion
    matrix, Roots far from the origin of the complex plane may have large
    errors due to the numerical instability of the series for such
    values. Roots with multiplicity greater than 1 will also show larger
    errors as the value of the series near such points is relatively
    insensitive to errors in the roots. Isolated roots near the origin can
    be improved by a few iterations of Newton's method.

    The Hermite series basis polynomials aren't powers of `x` so the
    results of this function may seem unintuitive.

    Examples
    --------
    >>> from numpy.polynomial.hermite import hermroots, hermfromroots
    >>> coef = hermfromroots([-1, 0, 1])
    >>> coef
    array([0.   ,  0.25 ,  0.   ,  0.125])
    >>> hermroots(coef)
    """
    # Compute roots of the Hermite series using the companion matrix
    return np.linalg.eigvals(hermcompanion(c))  # 使用伴随矩阵计算 Hermite 系列的根，并返回根的数组
    array([-1.00000000e+00, -1.38777878e-17,  1.00000000e+00])

    """
    # 将输入的数组 c 复制一份，并转换为一维数组
    [c] = pu.as_series([c])
    # 如果 c 的长度小于等于 1，则返回一个空的 NumPy 数组，其数据类型与 c 相同
    if len(c) <= 1:
        return np.array([], dtype=c.dtype)
    # 如果 c 的长度为 2，则计算并返回一个包含单个元素的 NumPy 数组，值为 -.5*c[0]/c[1]
    if len(c) == 2:
        return np.array([-.5*c[0]/c[1]])

    # 创建旋转后的伴随矩阵以减少误差
    # 使用 hermcompanion 函数生成 c 的伴随矩阵，并对其进行逆序排列
    m = hermcompanion(c)[::-1,::-1]
    # 计算矩阵 m 的特征值
    r = la.eigvals(m)
    # 对特征值数组 r 进行排序
    r.sort()
    # 返回排序后的特征值数组
    return r
    ```
    # 将给定的 x 和 n 作为参数，计算规范化 Hermite 多项式的值
    def _normed_hermite_n(x, n):
        """
        Evaluate a normalized Hermite polynomial.

        Compute the value of the normalized Hermite polynomial of degree ``n``
        at the points ``x``.

        Parameters
        ----------
        x : ndarray of double.
            Points at which to evaluate the function
        n : int
            Degree of the normalized Hermite function to be evaluated.

        Returns
        -------
        values : ndarray
            The shape of the return value is described above.

        Notes
        -----
        .. versionadded:: 1.10.0

        This function is needed for finding the Gauss points and integration
        weights for high degrees. The values of the standard Hermite functions
        overflow when n >= 207.

        """
        if n == 0:
            # Return a constant value if n is 0
            return np.full(x.shape, 1/np.sqrt(np.sqrt(np.pi)))

        c0 = 0.
        c1 = 1./np.sqrt(np.sqrt(np.pi))
        nd = float(n)
        for i in range(n - 1):
            # Recurrence relation to compute the Hermite polynomial coefficients
            tmp = c0
            c0 = -c1*np.sqrt((nd - 1.)/nd)
            c1 = tmp + c1*x*np.sqrt(2./nd)
            nd = nd - 1.0
        # Return the computed Hermite polynomial value
        return c0 + c1*x*np.sqrt(2)


    # 计算 Gauss-Hermite 积分的采样点和权重
    def hermgauss(deg):
        """
        Gauss-Hermite quadrature.

        Computes the sample points and weights for Gauss-Hermite quadrature.
        These sample points and weights will correctly integrate polynomials of
        degree :math:`2*deg - 1` or less over the interval :math:`[-\\inf, \\inf]`
        with the weight function :math:`f(x) = \\exp(-x^2)`.

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

        .. math:: w_k = c / (H'_n(x_k) * H_{n-1}(x_k))

        where :math:`c` is a constant independent of :math:`k` and :math:`x_k`
        is the k'th root of :math:`H_n`, and then scaling the results to get
        the right value when integrating 1.

        Examples
        --------
        >>> from numpy.polynomial.hermite import hermgauss
        >>> hermgauss(2)
        (array([-0.70710678,  0.70710678]), array([0.88622693, 0.88622693]))

        """
        ideg = pu._as_int(deg, "deg")  # Convert deg to integer
        if ideg <= 0:
            raise ValueError("deg must be a positive integer")

        # first approximation of roots. We use the fact that the companion
        # matrix is symmetric in this case in order to obtain better zeros.
        c = np.array([0]*deg + [1], dtype=np.float64)
        m = hermcompanion(c)
        x = la.eigvalsh(m)

        # improve roots by one application of Newton
        dy = _normed_hermite_n(x, ideg)
        df = _normed_hermite_n(x, ideg - 1) * np.sqrt(2*ideg)
        x -= dy/df

        # compute the weights. We scale the factor to avoid possible numerical
        # overflow.
        fm = _normed_hermite_n(x, ideg - 1)
        fm /= np.abs(fm).max()
        w = 1/(fm * fm)

        # for Hermite we can also symmetrize
    # 将 w 扩展为 w 与其反转的平均值
    w = (w + w[::-1]) / 2
    # 将 x 缩放以获得正确的值
    x = (x - x[::-1]) / 2

    # 将 w 缩放以使其具有正确的值
    w *= np.sqrt(np.pi) / w.sum()

    # 返回处理后的 x 和 w
    return x, w
class Hermite(ABCPolyBase):
    """An Hermite series class.

    The Hermite class provides the standard Python numerical methods
    '+', '-', '*', '//', '%', 'divmod', '**', and '()' as well as the
    attributes and methods listed below.

    Parameters
    ----------
    coef : array_like
        Hermite coefficients in order of increasing degree, i.e,
        ``(1, 2, 3)`` gives ``1*H_0(x) + 2*H_1(x) + 3*H_2(x)``.
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
    # 设置静态方法为对应的Hermite多项式操作
    _add = staticmethod(hermadd)
    _sub = staticmethod(hermsub)
    _mul = staticmethod(hermmul)
    _div = staticmethod(hermdiv)
    _pow = staticmethod(hermpow)
    _val = staticmethod(hermval)
    _int = staticmethod(hermint)
    _der = staticmethod(hermder)
    _fit = staticmethod(hermfit)
    _line = staticmethod(hermline)
    _roots = staticmethod(hermroots)
    _fromroots = staticmethod(hermfromroots)

    # 定义虚拟属性
    domain = np.array(hermdomain)
    window = np.array(hermdomain)
    basis_name = 'H'
```