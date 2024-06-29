# `.\numpy\numpy\polynomial\polynomial.py`

```
# 定义一个模块级别变量，包含此模块公开的所有函数和常量的列表
__all__ = [
    'polyzero', 'polyone', 'polyx', 'polydomain', 'polyline', 'polyadd',
    'polysub', 'polymulx', 'polymul', 'polydiv', 'polypow', 'polyval',
    'polyvalfromroots', 'polyder', 'polyint', 'polyfromroots', 'polyvander',
    'polyfit', 'polytrim', 'polyroots', 'Polynomial', 'polyval2d', 'polyval3d',
    'polygrid2d', 'polygrid3d', 'polyvander2d', 'polyvander3d',
    'polycompanion']

# 导入 NumPy 库，并将其命名为 np
import numpy as np
# 导入 NumPy 的线性代数模块，并将其命名为 la
import numpy.linalg as la
# 从 NumPy 的数组工具模块中导入 normalize_axis_index 函数
from numpy.lib.array_utils import normalize_axis_index

# 从当前包中导入 polyutils 模块，并将其命名为 pu
from . import polyutils as pu
# 从当前包中导入 _polybase 模块中的 ABCPolyBase 类
from ._polybase import ABCPolyBase

# 将 polytrim 函数指向 polyutils 模块中的 trimcoef 函数
polytrim = pu.trimcoef

# 定义一个常数数组，表示多项式的默认定义域，值为 [-1., 1.]
polydomain = np.array([-1., 1.])

# 定义一个常数数组，表示多项式系数为零，值为 [0]
polyzero = np.array([0])

# 定义一个常数数组，表示多项式系数为一，值为 [1]
polyone = np.array([1])

# 定义一个常数数组，表示多项式的标识符 x，系数为 [0, 1]
polyx = np.array([0, 1])

# 定义一个函数 polyline，返回表示线性多项式的数组
def polyline(off, scl):
    """
    返回表示线性多项式的数组。

    Parameters
    ----------
    off, scl : scalars
        线性多项式的截距和斜率。

    Returns
    -------
    y : ndarray
        表示线性多项式 `off + scl*x` 的数组。

    See Also
    --------
    numpy.polynomial.chebyshev.chebline
    numpy.polynomial.legendre.legline
    numpy.polynomial.laguerre.lagline
    numpy.polynomial.hermite.hermline
    numpy.polynomial.hermite_e.hermeline

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> P.polyline(1, -1)
    array([ 1, -1])
    >>> P.polyval(1, P.polyline(1, -1))  # 应该为 0
    """
    # 返回线性多项式的系数数组
    return np.array([off, scl])
    # 检查缩放因子是否为零，根据情况返回包含偏移量和缩放因子的 NumPy 数组或仅包含偏移量的 NumPy 数组
    if scl != 0:
        # 如果缩放因子不为零，返回包含偏移量和缩放因子的 NumPy 数组
        return np.array([off, scl])
    else:
        # 如果缩放因子为零，返回仅包含偏移量的 NumPy 数组
        return np.array([off])
# 生成一个以给定根数为根的首一多项式

def polyfromroots(roots):
    # 返回的多项式系数是通过传入的根数生成的
    return pu._fromroots(polyline, polymul, roots)


# 将两个多项式相加

def polyadd(c1, c2):
    # 返回两个多项式的和，其中参数 c1 和 c2 是按照从低到高阶排列的多项式系数序列
    return pu._add(c1, c2)


# 将一个多项式减去另一个多项式

def polysub(c1, c2):
    # 返回两个多项式的差，其中参数 c1 和 c2 是按照从低到高阶排列的多项式系数序列
    # 定义一个函数，用于计算两个多项式系数之间的差
    def polysub(c1, c2):
        # 被减数和减数是一维数组，表示多项式的系数，从低到高排序
        # c1 表示第一个多项式的系数
        # c2 表示第二个多项式的系数
        # 返回一个包含系数的数组，表示两个多项式的差
        return pu._sub(c1, c2)
# 定义函数：将多项式乘以 x
def polymulx(c):
    """Multiply a polynomial by x.

    Multiply the polynomial `c` by x, where x is the independent
    variable.

    Parameters
    ----------
    c : array_like
        1-D array of polynomial coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Array representing the result of the multiplication.

    See Also
    --------
    polyadd, polysub, polymul, polydiv, polypow

    Notes
    -----

    .. versionadded:: 1.5.0

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> c = (1, 2, 3)
    >>> P.polymulx(c)
    array([0., 1., 2., 3.])

    """
    # 使用 pu.as_series([c]) 将 c 转换成修剪后的副本
    [c] = pu.as_series([c])
    # 如果 c 只包含一个元素且为 0，则直接返回 c
    if len(c) == 1 and c[0] == 0:
        return c

    # 创建一个长度比 c 长度大一的空数组，其数据类型与 c 相同
    prd = np.empty(len(c) + 1, dtype=c.dtype)
    # 将 prd 的第一个元素设置为 c[0] * 0
    prd[0] = c[0]*0
    # 将 prd 的剩余元素设置为 c 的内容
    prd[1:] = c
    return prd


# 定义函数：将一个多项式乘以另一个多项式
def polymul(c1, c2):
    """
    Multiply one polynomial by another.

    Returns the product of two polynomials `c1` * `c2`.  The arguments are
    sequences of coefficients, from lowest order term to highest, e.g.,
    [1,2,3] represents the polynomial ``1 + 2*x + 3*x**2.``

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of coefficients representing a polynomial, relative to the
        "standard" basis, and ordered from lowest order term to highest.

    Returns
    -------
    out : ndarray
        Of the coefficients of their product.

    See Also
    --------
    polyadd, polysub, polymulx, polydiv, polypow

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> c1 = (1, 2, 3)
    >>> c2 = (3, 2, 1)
    >>> P.polymul(c1, c2)
    array([  3.,   8.,  14.,   8.,   3.])

    """
    # 使用 pu.as_series([c1, c2]) 将 c1 和 c2 转换成修剪后的副本
    [c1, c2] = pu.as_series([c1, c2])
    # 计算 c1 和 c2 的卷积（convolution），即它们的乘积
    ret = np.convolve(c1, c2)
    return pu.trimseq(ret)


# 定义函数：将一个多项式除以另一个多项式
def polydiv(c1, c2):
    """
    Divide one polynomial by another.

    Returns the quotient-with-remainder of two polynomials `c1` / `c2`.
    The arguments are sequences of coefficients, from lowest order term
    to highest, e.g., [1,2,3] represents ``1 + 2*x + 3*x**2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of polynomial coefficients ordered from low to high.

    Returns
    -------
    [quo, rem] : ndarrays
        Of coefficient series representing the quotient and remainder.

    See Also
    --------
    polyadd, polysub, polymulx, polymul, polypow

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> c1 = (1, 2, 3)
    >>> c2 = (3, 2, 1)
    >>> P.polydiv(c1, c2)
    (array([3.]), array([-8., -4.]))
    >>> P.polydiv(c2, c1)
    (array([ 0.33333333]), array([ 2.66666667,  1.33333333]))  # may vary

    """
    # 使用 pu.as_series([c1, c2]) 将 c1 和 c2 转换成修剪后的副本
    [c1, c2] = pu.as_series([c1, c2])
    # 如果 c2 的最高阶系数为 0，则抛出 ZeroDivisionError
    if c2[-1] == 0:
        raise ZeroDivisionError()

    # 计算 c1 和 c2 的商和余数，使用 np.convolve 实现
    lc1 = len(c1)
    # 计算列表 c1 和 c2 的长度
    lc2 = len(c2)
    # 如果 c1 的长度小于 c2 的长度，则返回 c1 的第一个元素重复零次的列表和 c1 本身
    if lc1 < lc2:
        return c1[:1]*0, c1
    # 如果 c2 的长度为 1，则返回 c1 除以 c2 的最后一个元素的结果和 c1 的第一个元素重复零次的列表
    elif lc2 == 1:
        return c1/c2[-1], c1[:1]*0
    # 否则，计算 c1 和 c2 的长度差，初始化除数 scl 为 c2 的最后一个元素
    else:
        dlen = lc1 - lc2
        scl = c2[-1]
        # 对 c2 进行除法运算，将结果重新赋值给 c2
        c2 = c2[:-1]/scl
        # 初始化循环变量 i 和 j
        i = dlen
        j = lc1 - 1
        # 循环执行减法操作，修改 c1 的部分值
        while i >= 0:
            c1[i:j] -= c2*c1[j]
            i -= 1
            j -= 1
        # 返回计算后的 c1 的剩余部分除以 scl 的结果和经过 pu.trimseq 函数处理后的 c1 的前部分
        return c1[j+1:]/scl, pu.trimseq(c1[:j+1])
def polypow(c, pow, maxpower=None):
    """Raise a polynomial to a power.

    Returns the polynomial `c` raised to the power `pow`. The argument
    `c` is a sequence of coefficients ordered from low to high. i.e.,
    [1,2,3] is the series  ``1 + 2*x + 3*x**2.``

    Parameters
    ----------
    c : array_like
        1-D array of array of series coefficients ordered from low to
        high degree.
    pow : integer
        Power to which the series will be raised
    maxpower : integer, optional
        Maximum power allowed. This is mainly to limit growth of the series
        to unmanageable size. Default is 16

    Returns
    -------
    coef : ndarray
        Power series of power.

    See Also
    --------
    polyadd, polysub, polymulx, polymul, polydiv

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> P.polypow([1, 2, 3], 2)
    array([ 1., 4., 10., 12., 9.])

    """
    # note: this is more efficient than `pu._pow(polymul, c1, c2)`, as it
    # avoids calling `as_series` repeatedly
    return pu._pow(np.convolve, c, pow, maxpower)


def polyder(c, m=1, scl=1, axis=0):
    """
    Differentiate a polynomial.

    Returns the polynomial coefficients `c` differentiated `m` times along
    `axis`.  At each iteration the result is multiplied by `scl` (the
    scaling factor is for use in a linear change of variable).  The
    argument `c` is an array of coefficients from low to high degree along
    each axis, e.g., [1,2,3] represents the polynomial ``1 + 2*x + 3*x**2``
    while [[1,2],[1,2]] represents ``1 + 1*x + 2*y + 2*x*y`` if axis=0 is
    ``x`` and axis=1 is ``y``.

    Parameters
    ----------
    c : array_like
        Array of polynomial coefficients. If c is multidimensional the
        different axis correspond to different variables with the degree
        in each axis given by the corresponding index.
    m : int, optional
        Number of derivatives taken, must be non-negative. (Default: 1)
    scl : scalar, optional
        Each differentiation is multiplied by `scl`.  The end result is
        multiplication by ``scl**m``.  This is for use in a linear change
        of variable. (Default: 1)
    axis : int, optional
        Axis over which the derivative is taken. (Default: 0).

        .. versionadded:: 1.7.0

    Returns
    -------
    der : ndarray
        Polynomial coefficients of the derivative.

    See Also
    --------
    polyint

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> c = (1, 2, 3, 4)
    >>> P.polyder(c)  # (d/dx)(c)
    array([  2.,   6.,  12.])
    >>> P.polyder(c, 3)  # (d**3/dx**3)(c)
    array([24.])
    >>> P.polyder(c, scl=-1)  # (d/d(-x))(c)
    array([ -2.,  -6., -12.])
    >>> P.polyder(c, 2, -1)  # (d**2/d(-x)**2)(c)
    array([  6.,  24.])

    """
    c = np.array(c, ndmin=1, copy=True)  # 将输入的多项式系数转换为数组
    if c.dtype.char in '?bBhHiIlLqQpP':
        # astype fails with NA
        c = c + 0.0  # 将数组中的元素转换为浮点数
    cdt = c.dtype  # 获取数组的数据类型
    # 获取导数的阶数，使用 pu._as_int 函数将 m 转换为整数，"the order of derivation" 是错误消息的一部分
    cnt = pu._as_int(m, "the order of derivation")
    
    # 获取轴的索引，使用 pu._as_int 函数将 axis 转换为整数，"the axis" 是错误消息的一部分
    iaxis = pu._as_int(axis, "the axis")
    
    # 如果导数的阶数小于 0，则抛出 ValueError 异常
    if cnt < 0:
        raise ValueError("The order of derivation must be non-negative")
    
    # 根据轴的索引，将其规范化为 c 的维度中的有效索引
    iaxis = normalize_axis_index(iaxis, c.ndim)
    
    # 如果导数的阶数为 0，则直接返回 c，不需要进行导数计算
    if cnt == 0:
        return c
    
    # 将 c 数组的轴 iaxis 移动到第一个位置，以便进行导数计算
    c = np.moveaxis(c, iaxis, 0)
    
    # 获取 c 数组的长度
    n = len(c)
    
    # 如果导数的阶数大于等于 c 的长度 n，则将 c 的第一个元素置为零数组
    if cnt >= n:
        c = c[:1]*0
    else:
        # 否则，进行 cnt 次导数计算
        for i in range(cnt):
            n = n - 1  # 更新导数次数
            c *= scl  # 将 c 数组每个元素乘以 scl
            # 创建一个空数组 der，用于存储导数结果，形状为 (n,) + c.shape[1:]
            der = np.empty((n,) + c.shape[1:], dtype=cdt)
            # 计算导数，j 从 n 递减到 1
            for j in range(n, 0, -1):
                der[j - 1] = j * c[j]
            # 将导数结果赋给 c
            c = der
    
    # 将 c 数组的第一个轴移动回 iaxis 轴的位置
    c = np.moveaxis(c, 0, iaxis)
    
    # 返回最终的导数计算结果 c
    return c
# 定义一个函数用于对多项式进行积分操作
def polyint(c, m=1, k=[], lbnd=0, scl=1, axis=0):
    """
    Integrate a polynomial.

    Returns the polynomial coefficients `c` integrated `m` times from
    `lbnd` along `axis`.  At each iteration the resulting series is
    **multiplied** by `scl` and an integration constant, `k`, is added.
    The scaling factor is for use in a linear change of variable.  ("Buyer
    beware": note that, depending on what one is doing, one may want `scl`
    to be the reciprocal of what one might expect; for more information,
    see the Notes section below.) The argument `c` is an array of
    coefficients, from low to high degree along each axis, e.g., [1,2,3]
    represents the polynomial ``1 + 2*x + 3*x**2`` while [[1,2],[1,2]]
    represents ``1 + 1*x + 2*y + 2*x*y`` if axis=0 is ``x`` and axis=1 is
    ``y``.

    Parameters
    ----------
    c : array_like
        1-D array of polynomial coefficients, ordered from low to high.
    m : int, optional
        Order of integration, must be positive. (Default: 1)
    k : {[], list, scalar}, optional
        Integration constant(s).  The value of the first integral at zero
        is the first value in the list, the value of the second integral
        at zero is the second value, etc.  If ``k == []`` (the default),
        all constants are set to zero.  If ``m == 1``, a single scalar can
        be given instead of a list.
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
        Coefficient array of the integral.

    Raises
    ------
    ValueError
        If ``m < 1``, ``len(k) > m``, ``np.ndim(lbnd) != 0``, or
        ``np.ndim(scl) != 0``.

    See Also
    --------
    polyder

    Notes
    -----
    Note that the result of each integration is *multiplied* by `scl`.  Why
    is this important to note?  Say one is making a linear change of
    variable :math:`u = ax + b` in an integral relative to `x`. Then
    :math:`dx = du/a`, so one will need to set `scl` equal to
    :math:`1/a` - perhaps not what one would have first thought.

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> c = (1, 2, 3)
    >>> P.polyint(c)  # should return array([0, 1, 1, 1])
    array([0.,  1.,  1.,  1.])
    >>> P.polyint(c, 3)  # should return array([0, 0, 0, 1/6, 1/12, 1/20])
     array([ 0.        ,  0.        ,  0.        ,  0.16666667,  0.08333333, # may vary
             0.05      ])
    >>> P.polyint(c, k=3)  # should return array([3, 1, 1, 1])
    array([3.,  1.,  1.,  1.])
    >>> P.polyint(c,lbnd=-2)  # should return array([6, 1, 1, 1])
    array([6.,  1.,  1.,  1.])
    """

    # 确认积分次数大于等于1，且积分常数列表长度不超过积分次数
    if m < 1 or len(k) > m:
        raise ValueError("Invalid integration parameters.")

    # 确认积分下限为标量
    if np.ndim(lbnd) != 0:
        raise ValueError("Lower bound lbnd must be scalar.")

    # 确认缩放因子为标量
    if np.ndim(scl) != 0:
        raise ValueError("Scaling factor scl must be scalar.")

    # 遍历积分次数，执行多次积分操作
    for _ in range(m):
        # 执行积分操作，乘以缩放因子后加上积分常数
        c = np.polyint(c, lbnd=lbnd, axis=axis) * scl + k.pop(0)

    # 返回积分后的多项式系数数组
    return c
    # 将输入参数 c 转换为至少包含一个维度的 NumPy 数组，并确保是副本
    c = np.array(c, ndmin=1, copy=True)
    
    # 如果 c 的数据类型是布尔型或者整型，将其强制转换为浮点型，以便后续操作
    if c.dtype.char in '?bBhHiIlLqQpP':
        c = c + 0.0
    
    # 保存 c 的原始数据类型
    cdt = c.dtype
    
    # 如果 k 不是可迭代对象，则将其转换为包含 k 的单元素列表
    if not np.iterable(k):
        k = [k]
    
    # 将 m 强制转换为整数，表示积分的阶数
    cnt = pu._as_int(m, "the order of integration")
    
    # 将 axis 强制转换为整数，表示操作的轴
    iaxis = pu._as_int(axis, "the axis")
    
    # 如果 cnt 小于 0，则抛出 ValueError
    if cnt < 0:
        raise ValueError("The order of integration must be non-negative")
    
    # 如果 k 的长度大于 cnt，则抛出 ValueError
    if len(k) > cnt:
        raise ValueError("Too many integration constants")
    
    # 如果 lbnd 不是标量，则抛出 ValueError
    if np.ndim(lbnd) != 0:
        raise ValueError("lbnd must be a scalar.")
    
    # 如果 scl 不是标量，则抛出 ValueError
    if np.ndim(scl) != 0:
        raise ValueError("scl must be a scalar.")
    
    # 根据 iaxis 的值标准化轴的索引，确保在 c 的维度范围内
    iaxis = normalize_axis_index(iaxis, c.ndim)

    # 如果 cnt 为 0，则直接返回 c
    if cnt == 0:
        return c

    # 将 k 扩展为长度为 cnt 的列表，不足部分用 0 填充
    k = list(k) + [0]*(cnt - len(k))
    
    # 将 c 在维度 iaxis 上进行移动，以便后续操作
    c = np.moveaxis(c, iaxis, 0)
    
    # 执行 cnt 次积分操作
    for i in range(cnt):
        n = len(c)
        c *= scl
        if n == 1 and np.all(c[0] == 0):
            c[0] += k[i]
        else:
            tmp = np.empty((n + 1,) + c.shape[1:], dtype=cdt)
            tmp[0] = c[0]*0
            tmp[1] = c[0]
            for j in range(1, n):
                tmp[j + 1] = c[j]/(j + 1)
            tmp[0] += k[i] - polyval(lbnd, tmp)
            c = tmp
    
    # 将 c 恢复原来的轴顺序
    c = np.moveaxis(c, 0, iaxis)
    
    # 返回积分后的结果 c
    return c
# 定义一个函数用于在指定点 x 处评估多项式的值
def polyval(x, c, tensor=True):
    """
    Evaluate a polynomial at points x.

    If `c` is of length ``n + 1``, this function returns the value

    .. math:: p(x) = c_0 + c_1 * x + ... + c_n * x^n

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
    values : ndarray, compatible object
        The shape of the returned array is described above.

    See Also
    --------
    polyval2d, polygrid2d, polyval3d, polygrid3d

    Notes
    -----
    The evaluation uses Horner's method.

    Examples
    --------
    >>> from numpy.polynomial.polynomial import polyval
    >>> polyval(1, [1,2,3])
    6.0
    >>> a = np.arange(4).reshape(2,2)
    >>> a
    array([[0, 1],
           [2, 3]])
    >>> polyval(a, [1, 2, 3])
    array([[ 1.,   6.],
           [17.,  34.]])
    >>> coef = np.arange(4).reshape(2, 2)  # multidimensional coefficients
    >>> coef
    array([[0, 1],
           [2, 3]])
    >>> polyval([1, 2], coef, tensor=True)
    array([[2.,  4.],
           [4.,  7.]])
    >>> polyval([1, 2], coef, tensor=False)
    array([2.,  7.])

    """
    # 将输入的系数 c 转换为至少一维的 numpy 数组，确保可以进行后续的数学运算
    c = np.array(c, ndmin=1, copy=None)
    # 检查 c 的数据类型的字符是否在 '?bBhHiIlLqQpP' 中
    if c.dtype.char in '?bBhHiIlLqQpP':
        # 如果是，将 c 转换为浮点数类型，避免 astype 在 NA 值时出错
        c = c + 0.0
    
    # 检查 x 是否为元组或列表类型，如果是则转换为 NumPy 数组
    if isinstance(x, (tuple, list)):
        x = np.asarray(x)
    
    # 如果 x 是 NumPy 数组并且 tensor 为真，则将 c 调整为与 x 维度相匹配的形状
    if isinstance(x, np.ndarray) and tensor:
        c = c.reshape(c.shape + (1,)*x.ndim)
    
    # 初始化 c0 为 c 的倒数第一项加上 x 的零乘积
    c0 = c[-1] + x*0
    
    # 从倒数第二项开始迭代计算 c0，每次乘以 x 并加上 c 中的前一个项
    for i in range(2, len(c) + 1):
        c0 = c[-i] + c0*x
    
    # 返回计算得到的 c0
    return c0
# 定义函数 polyvalfromroots，用于基于给定的根在点 x 处评估多项式的值
def polyvalfromroots(x, r, tensor=True):
    """
    Evaluate a polynomial specified by its roots at points x.

    If `r` is of length ``N``, this function returns the value

    .. math:: p(x) = \\prod_{n=1}^{N} (x - r_n)

    The parameter `x` is converted to an array only if it is a tuple or a
    list, otherwise it is treated as a scalar. In either case, either `x`
    or its elements must support multiplication and addition both with
    themselves and with the elements of `r`.

    If `r` is a 1-D array, then ``p(x)`` will have the same shape as `x`.  If `r`
    is multidimensional, then the shape of the result depends on the value of
    `tensor`. If `tensor` is ``True`` the shape will be r.shape[1:] + x.shape;
    that is, each polynomial is evaluated at every value of `x`. If `tensor` is
    ``False``, the shape will be r.shape[1:]; that is, each polynomial is
    evaluated only for the corresponding broadcast value of `x`. Note that
    scalars have shape (,).

    .. versionadded:: 1.12

    Parameters
    ----------
    x : array_like, compatible object
        If `x` is a list or tuple, it is converted to an ndarray, otherwise
        it is left unchanged and treated as a scalar. In either case, `x`
        or its elements must support addition and multiplication with
        with themselves and with the elements of `r`.
    r : array_like
        Array of roots. If `r` is multidimensional the first index is the
        root index, while the remaining indices enumerate multiple
        polynomials. For instance, in the two dimensional case the roots
        of each polynomial may be thought of as stored in the columns of `r`.
    tensor : boolean, optional
        If True, the shape of the roots array is extended with ones on the
        right, one for each dimension of `x`. Scalars have dimension 0 for this
        action. The result is that every column of coefficients in `r` is
        evaluated for every element of `x`. If False, `x` is broadcast over the
        columns of `r` for the evaluation.  This keyword is useful when `r` is
        multidimensional. The default value is True.

    Returns
    -------
    values : ndarray, compatible object
        The shape of the returned array is described above.

    See Also
    --------
    polyroots, polyfromroots, polyval
    """

    # 判断 x 是否为列表或元组，如果是，则转换为 ndarray
    if isinstance(x, (list, tuple)):
        x = np.array(x)

    # 计算多项式在给定 x 处的值，根据 tensor 参数决定是否进行广播
    if tensor:
        # 广播计算，r 的每列系数对应每个 x 的值进行多项式计算
        values = np.prod(x.reshape(-1, 1) - r, axis=-1)
    else:
        # 非广播计算，每个 x 对应一个多项式的计算结果
        values = np.prod(x - r, axis=0)

    # 返回计算结果
    return values
    """
    将输入的数组 r 转换为至少有一维的 numpy 数组，不进行复制
    r = np.array(r, ndmin=1, copy=None)
    
    如果 r 的数据类型在 '?bBhHiIlLqQpP' 中的一种，将其转换为 np.double 类型
    if r.dtype.char in '?bBhHiIlLqQpP':
        r = r.astype(np.double)
    
    如果 x 是 tuple 或者 list 类型，则将其转换为 numpy 数组
    if isinstance(x, (tuple, list)):
        x = np.asarray(x)
    
    如果 x 是 numpy 数组：
        如果 tensor 为 True，则将 r 重塑为其形状加上 x 维度的 1 的形状
        r = r.reshape(r.shape + (1,)*x.ndim)
        否则，如果 x 的维度数 x.ndim 大于等于 r 的维度数 r.ndim，则抛出 ValueError
        elif x.ndim >= r.ndim:
            raise ValueError("x.ndim must be < r.ndim when tensor == False")
    
    计算并返回 x 与 r 之间的元素级乘积，沿着第 0 轴
    return np.prod(x - r, axis=0)
    """
# 定义一个函数，用于在二维空间中评估多项式的值，即 p(x,y) = ∑ c_{i,j} * x^i * y^j
def polyval2d(x, y, c):
    # 调用内部函数 _valnd，用于计算多项式的值
    return pu._valnd(polyval, c, x, y)


# 定义一个函数，用于在 x 和 y 的笛卡尔积上评估二维多项式的值，即 p(a,b) = ∑ c_{i,j} * a^i * b^j
def polygrid2d(x, y, c):
    """
    Evaluate a 2-D polynomial on the Cartesian product of x and y.

    This function returns the values:

    .. math:: p(a,b) = \\sum_{i,j} c_{i,j} * a^i * b^j

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
        unchanged and, if it isn't an ndarray, it is treated as a scalar.
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
    polyval, polygrid2d, polyval3d, polygrid3d

    Notes
    -----

    .. versionadded:: 1.7.0

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> c = ((1, 2, 3), (4, 5, 6))
    >>> P.polyval2d(1, 1, c) 
    21.0

    """
    x, y : array_like, compatible objects
        两个二维序列，将在`x`和`y`的笛卡尔积点处进行评估。如果`x`或`y`是列表或元组，首先将其转换为ndarray；否则保持不变，如果它不是ndarray，则将其视为标量。
    c : array_like
        按照顺序排列的系数数组，其中i,j次项的系数包含在``c[i,j]``中。如果`c`的维度大于二，则剩余的索引用于列举多组系数。

    Returns
    -------
    values : ndarray, compatible object
        在`x`和`y`的笛卡尔积点处的二维多项式的值。

    See Also
    --------
    polyval, polyval2d, polyval3d, polygrid3d

    Notes
    -----

    .. versionadded:: 1.7.0

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> c = ((1, 2, 3), (4, 5, 6))
    >>> P.polygrid2d([0, 1], [0, 1], c)
    array([[ 1.,  6.],
           [ 5., 21.]])
    
    """
    return pu._gridnd(polyval, c, x, y)
# 计算三维多项式在点 (x, y, z) 处的值。

# 导入模块，使用 _valnd 函数计算多维多项式的值
def polyval3d(x, y, z, c):
    return pu._valnd(polyval, c, x, y, z)


# 计算三维多项式在 x, y, z 的笛卡尔积上的值。

# 导入模块，使用 _valnd 函数计算三维网格上的多项式值
def polygrid3d(x, y, z, c):
    x, y, z : array_like, compatible objects
        三维系列在笛卡尔积 `x`, `y`, 和 `z` 点处进行评估。如果 `x`, `y`, 或 `z` 是列表或元组，它们首先被转换为 ndarray 数组；否则保持不变，并且如果它不是 ndarray 数组，则被视为标量。
    c : array_like
        系数数组，按照这样的顺序排列：i, j 次项的系数包含在 ``c[i,j]`` 中。如果 `c` 的维度大于两，剩余的索引用来枚举多组系数。

    Returns
    -------
    values : ndarray, compatible object
        二维多项式在 `x` 和 `y` 的笛卡尔积点处的值。

    See Also
    --------
    polyval, polyval2d, polygrid2d, polyval3d

    Notes
    -----

    .. versionadded:: 1.7.0
        添加于版本 1.7.0

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> c = ((1, 2, 3), (4, 5, 6), (7, 8, 9))
    >>> P.polygrid3d([0, 1], [0, 1], [0, 1], c)
    array([[ 1., 13.],
           [ 6., 51.]])

    """
    return pu._gridnd(polyval, c, x, y, z)
def polyvander(x, deg):
    """Vandermonde matrix of given degree.

    Returns the Vandermonde matrix of degree `deg` and sample points
    `x`. The Vandermonde matrix is defined by

    .. math:: V[..., i] = x^i,

    where ``0 <= i <= deg``. The leading indices of `V` index the elements of
    `x` and the last index is the power of `x`.

    If `c` is a 1-D array of coefficients of length ``n + 1`` and `V` is the
    matrix ``V = polyvander(x, n)``, then ``np.dot(V, c)`` and
    ``polyval(x, c)`` are the same up to roundoff. This equivalence is
    useful both for least squares fitting and for the evaluation of a large
    number of polynomials of the same degree and sample points.

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
    vander : ndarray.
        The Vandermonde matrix. The shape of the returned matrix is
        ``x.shape + (deg + 1,)``, where the last index is the power of `x`.
        The dtype will be the same as the converted `x`.

    See Also
    --------
    polyvander2d, polyvander3d

    Examples
    --------
    The Vandermonde matrix of degree ``deg = 5`` and sample points
    ``x = [-1, 2, 3]`` contains the element-wise powers of `x` 
    from 0 to 5 as its columns.

    >>> from numpy.polynomial import polynomial as P
    >>> x, deg = [-1, 2, 3], 5
    >>> P.polyvander(x=x, deg=deg)
    array([[  1.,  -1.,   1.,  -1.,   1.,  -1.],
           [  1.,   2.,   4.,   8.,  16.,  32.],
           [  1.,   3.,   9.,  27.,  81., 243.]])

    """
    # Convert deg to integer if possible, raising ValueError if deg < 0
    ideg = pu._as_int(deg, "deg")
    if ideg < 0:
        raise ValueError("deg must be non-negative")

    # Ensure x is a 1-D array of float64 or complex128 type
    x = np.array(x, copy=None, ndmin=1) + 0.0

    # Prepare dimensions for the Vandermonde matrix
    dims = (ideg + 1,) + x.shape
    dtyp = x.dtype

    # Initialize an empty array v with specified dimensions and dtype
    v = np.empty(dims, dtype=dtyp)

    # Fill the first row of v with 1's
    v[0] = x*0 + 1

    # Fill subsequent rows of v with powers of x up to degree ideg
    if ideg > 0:
        v[1] = x
        for i in range(2, ideg + 1):
            v[i] = v[i-1]*x

    # Rearrange axes of v to have the last axis as the power of x
    return np.moveaxis(v, 0, -1)
    # 返回两个多项式 Vandermonde 矩阵的张量积结果，对应给定的点和最大次数
    # 用于生成多维情况下的伪 Vandermonde 矩阵，用于多项式拟合和评估
    # Parameters 参数
    # ----------
    # x, y : array_like
    #     点的坐标数组，形状相同。元素的数据类型会转换为 float64 或 complex128，取决于是否有复数元素。标量将转换为 1-D 数组。
    # deg : list of ints
    #     最大次数的列表，格式为 [x_deg, y_deg]。

    # Returns 返回值
    # -------
    # vander2d : ndarray
    #     返回矩阵的形状为 ``x.shape + (order,)``，其中 :math:`order = (deg[0]+1)*(deg[1]+1)`。数据类型与转换后的 `x` 和 `y` 相同。

    # See Also 参见
    # --------
    # polyvander, polyvander3d, polyval2d, polyval3d

    # Examples 示例
    # --------
    # 给定样本点 ``x = [-1, 2]`` 和 ``y = [1, 3]``，生成度为 ``[1, 2]`` 的二维伪 Vandermonde 矩阵如下：

    # >>> from numpy.polynomial import polynomial as P
    # >>> x = np.array([-1, 2])
    # >>> y = np.array([1, 3])
    # >>> m, n = 1, 2
    # >>> deg = np.array([m, n])
    # >>> V = P.polyvander2d(x=x, y=y, deg=deg)
    # >>> V
    # array([[ 1.,  1.,  1., -1., -1., -1.],
    #        [ 1.,  3.,  9.,  2.,  6., 18.]])

    # 可以验证任意 ``0 <= i <= m`` 和 ``0 <= j <= n`` 的列：

    # >>> i, j = 0, 1
    # >>> V[:, (deg[1]+1)*i + j] == x**i * y**j
    # array([ True,  True])

    # 当 ``y`` 点全部为零且度为 ``[m, 0]`` 时，样本点 ``x`` 的一维 Vandermonde 矩阵是二维伪 Vandermonde 矩阵的特例。

    # >>> P.polyvander2d(x=x, y=0*x, deg=(m, 0)) == P.polyvander(x=x, deg=m)
    # array([[ True,  True],
    #        [ True,  True]])

    """
    return pu._vander_nd_flat((polyvander, polyvander), (x, y), deg)
# 多项式Vandermonde矩阵生成函数，用于三维情况

def polyvander3d(x, y, z, deg):
    """Pseudo-Vandermonde matrix of given degrees.

    Returns the pseudo-Vandermonde matrix of degrees `deg` and sample
    points ``(x, y, z)``. If `l`, `m`, `n` are the given degrees in `x`, `y`, `z`,
    then The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., (m+1)(n+1)i + (n+1)j + k] = x^i * y^j * z^k,

    where ``0 <= i <= l``, ``0 <= j <= m``, and ``0 <= k <= n``.  The leading
    indices of `V` index the points ``(x, y, z)`` and the last index encodes
    the powers of `x`, `y`, and `z`.

    If ``V = polyvander3d(x, y, z, [xdeg, ydeg, zdeg])``, then the columns
    of `V` correspond to the elements of a 3-D coefficient array `c` of
    shape (xdeg + 1, ydeg + 1, zdeg + 1) in the order

    .. math:: c_{000}, c_{001}, c_{002},... , c_{010}, c_{011}, c_{012},...

    and  ``np.dot(V, c.flat)`` and ``polyval3d(x, y, z, c)`` will be the
    same up to roundoff. This equivalence is useful both for least squares
    fitting and for the evaluation of a large number of 3-D polynomials
    of the same degrees and sample points.

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
        :math:`order = (deg[0]+1)*(deg([1]+1)*(deg[2]+1)`.  The dtype will
        be the same as the converted `x`, `y`, and `z`.

    See Also
    --------
    polyvander, polyvander3d, polyval2d, polyval3d

    Notes
    -----

    .. versionadded:: 1.7.0

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> x = np.asarray([-1, 2, 1])
    >>> y = np.asarray([1, -2, -3])
    >>> z = np.asarray([2, 2, 5])
    >>> l, m, n = [2, 2, 1]
    >>> deg = [l, m, n]
    >>> V = P.polyvander3d(x=x, y=y, z=z, deg=deg)
    >>> V
    array([[  1.,   2.,   1.,   2.,   1.,   2.,  -1.,  -2.,  -1.,
             -2.,  -1.,  -2.,   1.,   2.,   1.,   2.,   1.,   2.],
           [  1.,   2.,  -2.,  -4.,   4.,   8.,   2.,   4.,  -4.,
             -8.,   8.,  16.,   4.,   8.,  -8., -16.,  16.,  32.],
           [  1.,   5.,  -3., -15.,   9.,  45.,   1.,   5.,  -3.,
            -15.,   9.,  45.,   1.,   5.,  -3., -15.,   9.,  45.]])

    We can verify the columns for any ``0 <= i <= l``, ``0 <= j <= m``,
    and ``0 <= k <= n``

    >>> i, j, k = 2, 1, 0
    >>> V[:, (m+1)*(n+1)*i + (n+1)*j + k] == x**i * y**j * z**k
    array([ True,  True,  True])

    """
    # 调用私有函数 `_vander_nd_flat` 生成高维Vandermonde矩阵
    return pu._vander_nd_flat((polyvander, polyvander, polyvander), (x, y, z), deg)
    # 返回多项式拟合的系数，该多项式的阶数为 `deg`，通过最小二乘法拟合给定点 `x` 上的数据值 `y`。
    # 如果 `y` 是一维的，则返回的系数也是一维的。如果 `y` 是二维的，则对每一列 `y` 进行多次拟合，
    # 并将结果系数存储在返回的二维数组的相应列中。
    Return the coefficients of a polynomial of degree `deg` that is the
    least squares fit to the data values `y` given at points `x`. If `y` is
    1-D the returned coefficients will also be 1-D. If `y` is 2-D multiple
    fits are done, one for each column of `y`, and the resulting
    coefficients are stored in the corresponding columns of a 2-D return.
    The fitted polynomial(s) are in the form
    
    # 多项式的形式为
    
    .. math::  p(x) = c_0 + c_1 * x + ... + c_n * x^n,
    
    where `n` is `deg`.
    
    Parameters
    ----------
    x : array_like, shape (`M`,)
        x-coordinates of the `M` sample (data) points ``(x[i], y[i])``.
    y : array_like, shape (`M`,) or (`M`, `K`)
        y-coordinates of the sample points.  Several sets of sample points
        sharing the same x-coordinates can be (independently) fit with one
        call to `polyfit` by passing in for `y` a 2-D array that contains
        one data set per column.
    deg : int or 1-D array_like
        Degree(s) of the fitting polynomials. If `deg` is a single integer
        all terms up to and including the `deg`'th term are included in the
        fit. For NumPy versions >= 1.11.0 a list of integers specifying the
        degrees of the terms to include may be used instead.
    rcond : float, optional
        Relative condition number of the fit.  Singular values smaller
        than `rcond`, relative to the largest singular value, will be
        ignored.  The default value is ``len(x)*eps``, where `eps` is the
        relative precision of the platform's float type, about 2e-16 in
        most cases.
    full : bool, optional
        Switch determining the nature of the return value.  When ``False``
        (the default) just the coefficients are returned; when ``True``,
        diagnostic information from the singular value decomposition (used
        to solve the fit's matrix equation) is also returned.
    w : array_like, shape (`M`,), optional
        Weights. If not None, the weight ``w[i]`` applies to the unsquared
        residual ``y[i] - y_hat[i]`` at ``x[i]``. Ideally the weights are
        chosen so that the errors of the products ``w[i]*y[i]`` all have the
        same variance.  When using inverse-variance weighting, use
        ``w[i] = 1/sigma(y[i])``.  The default value is None.
    
        .. versionadded:: 1.5.0
    
    Returns
    -------
    coef : ndarray, shape (`deg` + 1,) or (`deg` + 1, `K`)
        Polynomial coefficients ordered from low to high.  If `y` was 2-D,
        the coefficients in column `k` of `coef` represent the polynomial
        fit to the data in `y`'s `k`-th column.
    [residuals, rank, singular_values, rcond] : list
        # 返回值列表，仅当 `full == True` 时返回以下值：

        - residuals -- 最小二乘拟合的残差平方和
        - rank -- 缩放后的范德蒙德矩阵的数值秩
        - singular_values -- 缩放后的范德蒙德矩阵的奇异值
        - rcond -- `rcond` 的值

        更多细节，请参阅 `numpy.linalg.lstsq`。

    Raises
    ------
    RankWarning
        # 如果最小二乘拟合中的矩阵秩不足，将引发此警告。
        仅当 `full == False` 时才会引发警告。可以通过以下方式关闭警告：

        >>> import warnings
        >>> warnings.simplefilter('ignore', np.exceptions.RankWarning)

    See Also
    --------
    numpy.polynomial.chebyshev.chebfit
    numpy.polynomial.legendre.legfit
    numpy.polynomial.laguerre.lagfit
    numpy.polynomial.hermite.hermfit
    numpy.polynomial.hermite_e.hermefit
    polyval : 计算多项式值。
    polyvander : 用于幂的范德蒙德矩阵。
    numpy.linalg.lstsq : 计算矩阵的最小二乘拟合。
    scipy.interpolate.UnivariateSpline : 计算样条拟合。

    Notes
    -----
    解决方案是多项式 `p` 的系数，该多项式最小化加权平方误差的和

    .. math:: E = \\sum_j w_j^2 * |y_j - p(x_j)|^2,

    其中 :math:`w_j` 是权重。该问题通过建立（通常为）过度确定的矩阵方程解决：

    .. math:: V(x) * c = w * y,

    其中 `V` 是 `x` 的加权伪范德蒙德矩阵，`c` 是要解的系数，`w` 是权重，`y` 是观察值。然后使用 `V` 的奇异值分解来解决此方程。

    如果 `V` 的某些奇异值很小以至于被忽略（且 `full` == ``False``），将引发 `~exceptions.RankWarning`。这意味着系数值可能确定不好。通常通过拟合较低阶多项式可以消除警告（但这可能不是您想要的；如果您有选择阶数的独立理由不起作用，您可能需要：a）重新考虑这些理由，和/或 b）重新考虑数据质量）。`rcond` 参数也可以设置为小于其默认值的值，但所得拟合可能是虚假的，并且可能会受到舍入误差的大贡献。

    使用双精度进行多项式拟合通常在多项式阶数约为 20 时“失败”。使用Chebyshev或Legendre系列进行拟合通常条件更好，但很大程度上取决于样本点的分布和数据的平滑性。如果拟合质量不够好，样条可能是一个好的选择。

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> x = np.linspace(-1,1,51)  # 生成包含51个元素的数组x，范围从-1到1，等间隔分布
    >>> rng = np.random.default_rng()
    >>> err = rng.normal(size=len(x))  # 创建长度与x相同的正态分布误差数组err
    >>> y = x**3 - x + err  # 根据公式y = x^3 - x + Gaussian noise计算y数组
    >>> c, stats = P.polyfit(x,y,3,full=True)  # 对x和y进行3次多项式拟合，返回拟合系数c和统计信息stats
    >>> c # c[0], c[1] 大约为 -1，c[2] 应该接近于0，c[3] 大约为 1
    array([ 0.23111996, -1.02785049, -0.2241444 ,  1.08405657])  # 可能有所变化
    >>> stats # 注意到较大的SSR，解释了拟合结果较差的原因
    [array([48.312088]),  # 可能有所变化
     4,
     array([1.38446749, 1.32119158, 0.50443316, 0.28853036]),
     1.1324274851176597e-14]

    Same thing without the added noise  # 在没有额外噪声的情况下进行相同的操作

    >>> y = x**3 - x  # 根据公式y = x^3 - x 计算y数组
    >>> c, stats = P.polyfit(x,y,3,full=True)  # 对x和y进行3次多项式拟合，返回拟合系数c和统计信息stats
    >>> c # c[0], c[1] 约为 -1，c[2] 应该非常接近于0，c[3] 约为 1
    array([-6.73496154e-17, -1.00000000e+00,  0.00000000e+00,  1.00000000e+00])
    >>> stats # 注意到极小的SSR
    [array([8.79579319e-31]),  # 可能有所变化
     4,
     array([1.38446749, 1.32119158, 0.50443316, 0.28853036]),
     1.1324274851176597e-14]

    """
    return pu._fit(polyvander, x, y, deg, rcond, full, w)


注释：
- `x = np.linspace(-1,1,51)`：生成包含51个元素的数组x，范围从-1到1，等间隔分布。
- `rng = np.random.default_rng()`：创建一个默认的随机数生成器对象。
- `err = rng.normal(size=len(x))`：创建一个长度与x相同的数组，其中的元素符合标准正态分布。
- `y = x**3 - x + err`：根据公式y = x^3 - x + Gaussian noise 计算y数组，将x的立方、x本身和误差err相加。
- `c, stats = P.polyfit(x,y,3,full=True)`：使用3次多项式拟合函数`P.polyfit`拟合x和y，返回拟合系数c和拟合统计信息stats。
- `c`：拟合系数数组c，其中c[0]、c[1]等分别对应多项式的各阶系数。
- `stats`：拟合的统计信息数组，包含SSR（残差平方和）、拟合自由度、标准误差估计和拟合使用的条件数等。
- `return pu._fit(polyvander, x, y, deg, rcond, full, w)`：返回调用`pu._fit`函数的结果，该函数用于多项式拟合过程中计算内部使用的Vandermonde矩阵和其他相关信息。
# c 是一个修剪后的副本，确保其是一个多项式系数的数组表示
[c] = pu.as_series([c])
# 如果系数数组的长度小于2，则抛出数值错误，因为多项式至少需要二次项
if len(c) < 2:
    raise ValueError('Series must have maximum degree of at least 1.')
# 如果系数数组的长度为2，则直接返回多项式的根，这时候只有一个根
if len(c) == 2:
    return np.array([-c[0]/c[1]])

# 计算多项式的次数
n = len(c) - 1
# 创建一个以系数数据类型为元素的零矩阵，维度为 (n, n)
mat = np.zeros((n, n), dtype=c.dtype)
# 获取矩阵的对角线下方的元素，并设置为1，这是伴随矩阵的特性
bot = mat.reshape(-1)[n::n+1]
bot[...] = 1
# 将矩阵的最后一列设置为系数数组除去最高阶系数后的数组与最高阶系数的比值的负数
mat[:, -1] -= c[:-1] / c[-1]
# 返回计算得到的伴随矩阵
return mat
    # 返回函数内部变量 r 的值作为函数的返回结果
    return r
# 多项式类，继承自ABCPolyBase
class Polynomial(ABCPolyBase):
    """A power series class.

    The Polynomial class provides the standard Python numerical methods
    '+', '-', '*', '//', '%', 'divmod', '**', and '()' as well as the
    attributes and methods listed below.

    Parameters
    ----------
    coef : array_like
        Polynomial coefficients in order of increasing degree, i.e.,
        ``(1, 2, 3)`` give ``1 + 2*x + 3*x**2``.
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
    
    # 静态方法：多项式加法
    _add = staticmethod(polyadd)
    # 静态方法：多项式减法
    _sub = staticmethod(polysub)
    # 静态方法：多项式乘法
    _mul = staticmethod(polymul)
    # 静态方法：多项式整除
    _div = staticmethod(polydiv)
    # 静态方法：多项式幂运算
    _pow = staticmethod(polypow)
    # 静态方法：计算多项式在给定点的值
    _val = staticmethod(polyval)
    # 静态方法：多项式积分
    _int = staticmethod(polyint)
    # 静态方法：多项式求导
    _der = staticmethod(polyder)
    # 静态方法：多项式拟合
    _fit = staticmethod(polyfit)
    # 静态方法：生成连接两点的线性多项式
    _line = staticmethod(polyline)
    # 静态方法：计算多项式的根
    _roots = staticmethod(polyroots)
    # 静态方法：根据给定的根生成多项式
    _fromroots = staticmethod(polyfromroots)

    # 虚拟属性：多项式的定义域
    domain = np.array(polydomain)
    # 虚拟属性：多项式的窗口
    window = np.array(polydomain)
    # 虚拟属性：基函数名称，默认为None
    basis_name = None

    @classmethod
    def _str_term_unicode(cls, i, arg_str):
        """Generate a unicode representation of a term in the polynomial.

        Parameters
        ----------
        i : int
            The exponent of the term.
        arg_str : str
            The string representation of the term without exponent.

        Returns
        -------
        str
            Unicode representation of the term.

        """
        if i == '1':
            return f"·{arg_str}"
        else:
            return f"·{arg_str}{i.translate(cls._superscript_mapping)}"

    @staticmethod
    def _str_term_ascii(i, arg_str):
        """Generate an ASCII representation of a term in the polynomial.

        Parameters
        ----------
        i : int
            The exponent of the term.
        arg_str : str
            The string representation of the term without exponent.

        Returns
        -------
        str
            ASCII representation of the term.

        """
        if i == '1':
            return f" {arg_str}"
        else:
            return f" {arg_str}**{i}"

    @staticmethod
    def _repr_latex_term(i, arg_str, needs_parens):
        """Generate a LaTeX representation of a term in the polynomial.

        Parameters
        ----------
        i : int
            The exponent of the term.
        arg_str : str
            The LaTeX string representation of the term without exponent.
        needs_parens : bool
            Whether the term needs to be wrapped in parentheses.

        Returns
        -------
        str
            LaTeX representation of the term.

        """
        if needs_parens:
            arg_str = rf"\left({arg_str}\right)"
        if i == 0:
            return '1'
        elif i == 1:
            return arg_str
        else:
            return f"{arg_str}^{{{i}}}"
```