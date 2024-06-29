# `.\numpy\numpy\polynomial\chebyshev.py`

```
# 将切比雪夫级数转换为 z 级数的函数
def _cseries_to_zseries(c):
    """Convert Chebyshev series to z-series.

    Convert a Chebyshev series to the equivalent z-series. The result is
    obtained by applying the algebraic identity involving z, transforming
    the Chebyshev series coefficients to their corresponding z-series
    coefficients.
    """
    # 返回转换后的 z 级数系数
    return (c + np.flip(c)) / 2
    # 计算输入 Chebyshev 系数数组的大小
    n = c.size
    # 创建一个全零数组，长度为 2*n-1，数据类型与输入数组 c 相同
    zs = np.zeros(2*n-1, dtype=c.dtype)
    # 将 Chebyshev 系数数组 c 的一半赋值给 zs 数组的后半部分
    zs[n-1:] = c/2
    # 返回 zs 数组与其反转数组的元素对应相加的结果
    return zs + zs[::-1]
def _zseries_to_cseries(zs):
    """Convert z-series to a Chebyshev series.

    Convert a z series to the equivalent Chebyshev series. The result is
    never an empty array. The dtype of the return is the same as that of
    the input. No checks are run on the arguments as this routine is for
    internal use.

    Parameters
    ----------
    zs : 1-D ndarray
        Odd length symmetric z-series, ordered from  low to high.

    Returns
    -------
    c : 1-D ndarray
        Chebyshev coefficients, ordered from  low to high.

    """
    # Calculate the number of terms in the resulting Chebyshev series
    n = (zs.size + 1)//2
    # Copy the central part of zs to c, which becomes Chebyshev coefficients
    c = zs[n-1:].copy()
    # Double the coefficients from the second to the middle term
    c[1:n] *= 2
    return c


def _zseries_mul(z1, z2):
    """Multiply two z-series.

    Multiply two z-series to produce a z-series.

    Parameters
    ----------
    z1, z2 : 1-D ndarray
        The arrays must be 1-D but this is not checked.

    Returns
    -------
    product : 1-D ndarray
        The product z-series.

    Notes
    -----
    This is simply convolution. If symmetric/anti-symmetric z-series are
    denoted by S/A then the following rules apply:

    S*S, A*A -> S
    S*A, A*S -> A

    """
    # Compute the convolution of z1 and z2
    return np.convolve(z1, z2)


def _zseries_div(z1, z2):
    """Divide the first z-series by the second.

    Divide `z1` by `z2` and return the quotient and remainder as z-series.
    Warning: this implementation only applies when both z1 and z2 have the
    same symmetry, which is sufficient for present purposes.

    Parameters
    ----------
    z1, z2 : 1-D ndarray
        The arrays must be 1-D and have the same symmetry, but this is not
        checked.

    Returns
    -------

    (quotient, remainder) : 1-D ndarrays
        Quotient and remainder as z-series.

    Notes
    -----
    This is not the same as polynomial division on account of the desired form
    of the remainder. If symmetric/anti-symmetric z-series are denoted by S/A
    then the following rules apply:

    S/S -> S,S
    A/A -> S,A

    The restriction to types of the same symmetry could be fixed but seems like
    unneeded generality. There is no natural form for the remainder in the case
    where there is no symmetry.

    """
    # Make copies of z1 and z2
    z1 = z1.copy()
    z2 = z2.copy()
    # Get the lengths of z1 and z2
    lc1 = len(z1)
    lc2 = len(z2)
    # Case when z2 is a scalar
    if lc2 == 1:
        z1 /= z2
        return z1, z1[:1]*0
    # Case when z1 has fewer elements than z2
    elif lc1 < lc2:
        return z1[:1]*0, z1
    else:
        # Case when z1 has more elements than z2
        dlen = lc1 - lc2
        scl = z2[0]
        z2 /= scl
        quo = np.empty(dlen + 1, dtype=z1.dtype)
        i = 0
        j = dlen
        # Perform long division
        while i < j:
            r = z1[i]
            quo[i] = z1[i]
            quo[dlen - i] = r
            tmp = r*z2
            z1[i:i+lc2] -= tmp
            z1[j:j+lc2] -= tmp
            i += 1
            j -= 1
        r = z1[i]
        quo[i] = r
        tmp = r*z2
        z1[i:i+lc2] -= tmp
        quo /= scl
        rem = z1[i+1:i-1+lc2].copy()
        return quo, rem


def _zseries_der(zs):
    """Differentiate a z-series.

    The derivative is with respect to x, not z. This is achieved using the
    """
    # No code is provided for this function in the prompt
    pass
    # 计算输入 z-series 的导数，基于链式法则和模块注释中给出的 dx/dz 的值。

    Parameters
    ----------
    zs : z-series
        待求导的 z-series。

    Returns
    -------
    derivative : z-series
        导数结果。

    Notes
    -----
    对于 x 的 z-series (ns)，已乘以二以避免使用与 Decimal 和其他特定标量类型不兼容的浮点数。
    为了补偿这种缩放效果，zs 的值也乘以二，这样两者在除法中相互抵消。

    """
    # 计算 ns 的长度并取整除以 2
    n = len(zs) // 2
    # 创建一个 numpy 数组 ns，包含 [-1, 0, 1]，数据类型与 zs 相同
    ns = np.array([-1, 0, 1], dtype=zs.dtype)
    # 将 zs 的每个元素乘以对应位置的 (-n 到 n 的范围) * 2，以补偿之前的乘法操作
    zs *= np.arange(-n, n+1) * 2
    # 使用 _zseries_div 函数计算 zs 和 ns 的商，结果包括商 d 和余数 r
    d, r = _zseries_div(zs, ns)
    # 返回导数结果 d
    return d
# 将 z-series 进行积分处理，积分是相对于 x 而非 z。这是通过变量变换 dx/dz 来实现的，变换细节见模块注释。
def _zseries_int(zs):
    # 计算 z-series 的长度，并计算对应的 x-series 的长度
    n = 1 + len(zs)//2
    # 创建一个包含 [-1, 0, 1] 的 NumPy 数组，数据类型与 zs 相同
    ns = np.array([-1, 0, 1], dtype=zs.dtype)
    # 将 zs 乘以 x-series 的系数，以进行变量变换
    zs = _zseries_mul(zs, ns)
    # 创建一个数组，其值为 [-n*2, ..., -2, 0, 2, ..., n*2]，用于后续的除法
    div = np.arange(-n, n+1)*2
    # 对 zs 进行分段除法，以完成变量变换后的调整
    zs[:n] /= div[:n]
    zs[n+1:] /= div[n+1:]
    # 将中间值置为 0，符合积分常数的定义
    zs[n] = 0
    # 返回处理后的 z-series，即 x-series 的积分结果
    return zs

#
# Chebyshev series functions
#


def poly2cheb(pol):
    """
    Convert a polynomial to a Chebyshev series.

    Convert an array representing the coefficients of a polynomial (relative
    to the "standard" basis) ordered from lowest degree to highest, to an
    array of the coefficients of the equivalent Chebyshev series, ordered
    from lowest to highest degree.

    Parameters
    ----------
    pol : array_like
        1-D array containing the polynomial coefficients

    Returns
    -------
    c : ndarray
        1-D array containing the coefficients of the equivalent Chebyshev
        series.

    See Also
    --------
    cheb2poly

    Notes
    -----
    The easy way to do conversions between polynomial basis sets
    is to use the convert method of a class instance.

    Examples
    --------
    >>> from numpy import polynomial as P
    >>> p = P.Polynomial(range(4))
    >>> p
    Polynomial([0., 1., 2., 3.], domain=[-1.,  1.], window=[-1.,  1.], symbol='x')
    >>> c = p.convert(kind=P.Chebyshev)
    >>> c
    Chebyshev([1.  , 3.25, 1.  , 0.75], domain=[-1.,  1.], window=[-1., ...
    >>> P.chebyshev.poly2cheb(range(4))
    array([1.  , 3.25, 1.  , 0.75])

    """
    # 使用 pu.as_series 将输入 pol 转换为 series
    [pol] = pu.as_series([pol])
    # 计算多项式的最高阶数
    deg = len(pol) - 1
    # 初始化结果为空的 Chebyshev 系数数组
    res = 0
    # 从最高阶开始逐步构建 Chebyshev 系数
    for i in range(deg, -1, -1):
        # 将当前多项式系数转换为 Chebyshev 系数并累加到 res 中
        res = chebadd(chebmulx(res), pol[i])
    # 返回转换后的 Chebyshev 系数数组
    return res


def cheb2poly(c):
    """
    Convert a Chebyshev series to a polynomial.

    Convert an array representing the coefficients of a Chebyshev series,
    ordered from lowest degree to highest, to an array of the coefficients
    of the equivalent polynomial (relative to the "standard" basis) ordered
    from lowest to highest degree.

    Parameters
    ----------
    c : array_like
        1-D array containing the Chebyshev series coefficients, ordered
        from lowest order term to highest.

    Returns
    -------
    pol : ndarray
        1-D array containing the coefficients of the equivalent polynomial
        (relative to the "standard" basis) ordered from lowest order term
        to highest.

    See Also
    --------
    poly2cheb

    Notes
    -----

    """
    # 该函数未实现详细注释，仅提供函数的基本结构和说明
    pass
    # 导入必要的函数和类来执行多项式操作
    from .polynomial import polyadd, polysub, polymulx
    
    # 将输入的多项式系数转换为系列（series），即确保 c 是一个多项式系数列表
    [c] = pu.as_series([c])
    
    # 计算多项式的阶数
    n = len(c)
    
    # 如果多项式阶数小于3，则直接返回多项式系数 c，不需要进行转换操作
    if n < 3:
        return c
    else:
        # 从 c 中提取倒数第二个系数和最后一个系数作为初始值
        c0 = c[-2]
        c1 = c[-1]
        
        # 从高阶到低阶（从 n-1 到 2），依次进行转换操作
        for i in range(n - 1, 1, -1):
            tmp = c0
            # 计算当前阶数 c1 的多项式转换结果
            c0 = polysub(c[i - 2], c1)
            c1 = polyadd(tmp, polymulx(c1)*2)
        
        # 返回转换后的多项式系数列表
        return polyadd(c0, polymulx(c1))
# 这些是整数类型的常量数组，以便与广泛的其他类型兼容，比如 Decimal。

# Chebyshev 默认定义域。
chebdomain = np.array([-1., 1.])

# 表示零的 Chebyshev 系数。
chebzero = np.array([0])

# 表示一的 Chebyshev 系数。
chebone = np.array([1])

# 表示恒等函数 x 的 Chebyshev 系数。
chebx = np.array([0, 1])


def chebline(off, scl):
    """
    表示一条直线的 Chebyshev 级数。

    Parameters
    ----------
    off, scl : scalar
        指定的直线由 ``off + scl*x`` 给出。

    Returns
    -------
    y : ndarray
        该模块对于 ``off + scl*x`` 的 Chebyshev 级数表示。

    See Also
    --------
    numpy.polynomial.polynomial.polyline
    numpy.polynomial.legendre.legline
    numpy.polynomial.laguerre.lagline
    numpy.polynomial.hermite.hermline
    numpy.polynomial.hermite_e.hermeline

    Examples
    --------
    >>> import numpy.polynomial.chebyshev as C
    >>> C.chebline(3,2)
    array([3, 2])
    >>> C.chebval(-3, C.chebline(3,2)) # 应该得到 -3
    -3.0

    """
    if scl != 0:
        return np.array([off, scl])
    else:
        return np.array([off])


def chebfromroots(roots):
    """
    生成具有给定根的 Chebyshev 级数。

    函数返回多项式

    .. math:: p(x) = (x - r_0) * (x - r_1) * ... * (x - r_n)

    的 Chebyshev 形式的系数，其中 :math:`r_n` 是 `roots` 中指定的根。如果一个零有多重性 n，
    那么它必须在 `roots` 中出现 n 次。例如，如果 2 是多重性为三的根，3 是多重性为两的根，
    那么 `roots` 看起来像是 [2, 2, 2, 3, 3]。根可以以任何顺序出现。

    如果返回的系数是 `c`，则

    .. math:: p(x) = c_0 + c_1 * T_1(x) + ... +  c_n * T_n(x)

    最后一项的系数通常不是 Chebyshev 形式的单项多项式的 1。

    Parameters
    ----------
    roots : array_like
        包含根的序列。

    Returns
    -------
    out : ndarray
        系数的一维数组。如果所有根都是实数，则 `out` 是实数组；如果某些根是复数，则 `out` 是复数，
        即使结果中的所有系数都是实数（见下面的示例）。

    See Also
    --------
    numpy.polynomial.polynomial.polyfromroots
    numpy.polynomial.legendre.legfromroots
    numpy.polynomial.laguerre.lagfromroots
    numpy.polynomial.hermite.hermfromroots
    numpy.polynomial.hermite_e.hermefromroots

    Examples
    --------
    >>> import numpy.polynomial.chebyshev as C
    >>> C.chebfromroots((-1,0,1)) # 相对于标准基础的 x^3 - x
    array([ 0.  , -0.25,  0.  ,  0.25])
    >>> j = complex(0,1)
    >>> C.chebfromroots((-j,j)) # 相对于标准基础的 x^2 + 1
    array([1.5+0.j, 0. +0.j, 0.5+0.j])

    """
    # 调用 pu 模块的 _fromroots 函数，传入 chebline、chebmul、roots 作为参数，并返回其结果
    return pu._fromroots(chebline, chebmul, roots)
# 定义函数 chebadd，用于将两个切比雪夫级数相加
def chebadd(c1, c2):
    # 调用私有函数 _add，执行切比雪夫级数的加法操作
    return pu._add(c1, c2)


# 定义函数 chebsub，用于从一个切比雪夫级数中减去另一个切比雪夫级数
def chebsub(c1, c2):
    # 调用私有函数 _sub，执行切比雪夫级数的减法操作
    return pu._sub(c1, c2)


# 定义函数 chebmulx，用于将一个切比雪夫级数乘以自变量 x
def chebmulx(c):
    """
    Multiply a Chebyshev series by x.

    Multiply the polynomial `c` by x, where x is the independent
    variable.

    Parameters
    ----------
    c : array_like
        1-D array of Chebyshev series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Array representing the result of the multiplication.

    See Also
    --------
    chebadd, chebsub, chebmul, chebdiv, chebpow

    Notes
    -----

    .. versionadded:: 1.5.0

    Examples
    --------
    >>> from numpy.polynomial import chebyshev as C
    >>> C.chebmulx([1,2,3])
    array([1. , 2.5, 1. , 1.5])

    """
    # 调用 pu.as_series([c]) 方法返回的结果，取其第一个元素作为 c
    [c] = pu.as_series([c])
    # 如果输入列表 c 的长度为1且唯一元素为0，则直接返回该列表，不进行后续计算
    if len(c) == 1 and c[0] == 0:
        return c

    # 创建一个长度为 len(c)+1 的空数组 prd，其数据类型与 c 相同
    prd = np.empty(len(c) + 1, dtype=c.dtype)
    
    # 将 prd 数组的第一个元素设置为 c[0] 的零乘积
    prd[0] = c[0]*0
    
    # 将 prd 数组的第二个元素设置为 c[0]
    prd[1] = c[0]
    
    # 如果 c 的长度大于1，则进行以下操作
    if len(c) > 1:
        # 将 c 的第二个元素及其后的元素都除以2，存储到 tmp 变量中
        tmp = c[1:]/2
        # 将 tmp 中的值赋给 prd 数组的第三个元素及其后的元素
        prd[2:] = tmp
        # 将 tmp 中的值累加到 prd 数组的倒数第二个元素及其前的元素上
        prd[0:-2] += tmp
    
    # 返回计算结果数组 prd
    return prd
# Multiply one Chebyshev series by another.
# 返回两个切比雪夫级数的乘积。
def chebmul(c1, c2):
    """
    Multiply one Chebyshev series by another.

    Returns the product of two Chebyshev series `c1` * `c2`.  The arguments
    are sequences of coefficients, from lowest order "term" to highest,
    e.g., [1,2,3] represents the series ``T_0 + 2*T_1 + 3*T_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of Chebyshev series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Of Chebyshev series coefficients representing their product.

    See Also
    --------
    chebadd, chebsub, chebmulx, chebdiv, chebpow

    Notes
    -----
    In general, the (polynomial) product of two C-series results in terms
    that are not in the Chebyshev polynomial basis set.  Thus, to express
    the product as a C-series, it is typically necessary to "reproject"
    the product onto said basis set, which typically produces
    "unintuitive live" (but correct) results; see Examples section below.

    Examples
    --------
    >>> from numpy.polynomial import chebyshev as C
    >>> c1 = (1,2,3)
    >>> c2 = (3,2,1)
    >>> C.chebmul(c1,c2) # multiplication requires "reprojection"
    array([  6.5,  12. ,  12. ,   4. ,   1.5])

    """
    # Trim c1, c2 to ensure they are valid series
    [c1, c2] = pu.as_series([c1, c2])
    # Convert c1 to a z-series (series of complex numbers)
    z1 = _cseries_to_zseries(c1)
    # Convert c2 to a z-series
    z2 = _cseries_to_zseries(c2)
    # Multiply the z-series z1 and z2
    prd = _zseries_mul(z1, z2)
    # Convert the resulting z-series back to a Chebyshev series
    ret = _zseries_to_cseries(prd)
    # Trim any unnecessary zeros from the resulting series and return
    return pu.trimseq(ret)


# Divide one Chebyshev series by another.
# 返回两个切比雪夫级数的商和余数。
def chebdiv(c1, c2):
    """
    Divide one Chebyshev series by another.

    Returns the quotient-with-remainder of two Chebyshev series
    `c1` / `c2`.  The arguments are sequences of coefficients from lowest
    order "term" to highest, e.g., [1,2,3] represents the series
    ``T_0 + 2*T_1 + 3*T_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of Chebyshev series coefficients ordered from low to
        high.

    Returns
    -------
    [quo, rem] : ndarrays
        Of Chebyshev series coefficients representing the quotient and
        remainder.

    See Also
    --------
    chebadd, chebsub, chebmulx, chebmul, chebpow

    Notes
    -----
    In general, the (polynomial) division of one C-series by another
    results in quotient and remainder terms that are not in the Chebyshev
    polynomial basis set.  Thus, to express these results as C-series, it
    is typically necessary to "reproject" the results onto said basis
    set, which typically produces "unintuitive" (but correct) results;
    see Examples section below.

    Examples
    --------
    >>> from numpy.polynomial import chebyshev as C
    >>> c1 = (1,2,3)
    >>> c2 = (3,2,1)
    >>> C.chebdiv(c1,c2) # quotient "intuitive," remainder not
    (array([3.]), array([-8., -4.]))
    >>> c2 = (0,1,2,3)
    >>> C.chebdiv(c2,c1) # neither "intuitive"
    (array([0., 2.]), array([-2., -4.]))

    """
    # Trim c1, c2 to ensure they are valid series
    [c1, c2] = pu.as_series([c1, c2])
    # 如果 c2 的最后一个元素为 0，则抛出 ZeroDivisionError 异常
    if c2[-1] == 0:
        raise ZeroDivisionError()

    # 注意：这种写法比 `pu._div(chebmul, c1, c2)` 更有效率

    # 计算 c1 和 c2 的长度
    lc1 = len(c1)
    lc2 = len(c2)

    # 根据 c1 和 c2 的长度关系进行不同的处理
    if lc1 < lc2:
        # 如果 c1 的长度小于 c2，则返回 (c1[0]*0, c1)
        return c1[:1]*0, c1
    elif lc2 == 1:
        # 如果 c2 的长度为 1，则返回 (c1/c2[-1], c1[0]*0)
        return c1/c2[-1], c1[:1]*0
    else:
        # 将 c1 和 c2 转换为复数系数表示
        z1 = _cseries_to_zseries(c1)
        z2 = _cseries_to_zseries(c2)
        
        # 对复数系数进行除法运算，得到商和余数
        quo, rem = _zseries_div(z1, z2)
        
        # 将得到的复数系数表示的商和余数转换为普通系数表示，并去除多余的零系数
        quo = pu.trimseq(_zseries_to_cseries(quo))
        rem = pu.trimseq(_zseries_to_cseries(rem))
        
        # 返回计算结果：商和余数
        return quo, rem
def chebpow(c, pow, maxpower=16):
    """Raise a Chebyshev series to a power.

    Returns the Chebyshev series `c` raised to the power `pow`. The
    argument `c` is a sequence of coefficients ordered from low to high.
    i.e., [1,2,3] is the series  ``T_0 + 2*T_1 + 3*T_2.``

    Parameters
    ----------
    c : array_like
        1-D array of Chebyshev series coefficients ordered from low to
        high.
    pow : integer
        Power to which the series will be raised
    maxpower : integer, optional
        Maximum power allowed. This is mainly to limit growth of the series
        to unmanageable size. Default is 16

    Returns
    -------
    coef : ndarray
        Chebyshev series of power.

    See Also
    --------
    chebadd, chebsub, chebmulx, chebmul, chebdiv

    Examples
    --------
    >>> from numpy.polynomial import chebyshev as C
    >>> C.chebpow([1, 2, 3, 4], 2)
    array([15.5, 22. , 16. , ..., 12.5, 12. ,  8. ])

    """
    # note: this is more efficient than `pu._pow(chebmul, c1, c2)`, as it
    # avoids converting between z and c series repeatedly

    # c is a trimmed copy
    [c] = pu.as_series([c])  # Convert `c` to a series if it's not already
    power = int(pow)  # Ensure `pow` is an integer
    if power != pow or power < 0:  # Check if `pow` is a non-negative integer
        raise ValueError("Power must be a non-negative integer.")
    elif maxpower is not None and power > maxpower:  # Check if `pow` exceeds `maxpower`
        raise ValueError("Power is too large")
    elif power == 0:  # Return 1 if power is 0
        return np.array([1], dtype=c.dtype)
    elif power == 1:  # Return `c` itself if power is 1
        return c
    else:
        # This can be made more efficient by using powers of two
        # in the usual way.
        zs = _cseries_to_zseries(c)  # Convert Chebyshev series `c` to z-series
        prd = zs
        for i in range(2, power + 1):
            prd = np.convolve(prd, zs)  # Perform convolution to compute higher powers
        return _zseries_to_cseries(prd)  # Convert back to Chebyshev series and return


def chebder(c, m=1, scl=1, axis=0):
    """
    Differentiate a Chebyshev series.

    Returns the Chebyshev series coefficients `c` differentiated `m` times
    along `axis`.  At each iteration the result is multiplied by `scl` (the
    scaling factor is for use in a linear change of variable). The argument
    `c` is an array of coefficients from low to high degree along each
    axis, e.g., [1,2,3] represents the series ``1*T_0 + 2*T_1 + 3*T_2``
    while [[1,2],[1,2]] represents ``1*T_0(x)*T_0(y) + 1*T_1(x)*T_0(y) +
    2*T_0(x)*T_1(y) + 2*T_1(x)*T_1(y)`` if axis=0 is ``x`` and axis=1 is
    ``y``.

    Parameters
    ----------
    c : array_like
        Array of Chebyshev series coefficients. If c is multidimensional
        the different axis correspond to different variables with the
        degree in each axis given by the corresponding index.
    m : int, optional
        Number of derivatives taken, must be non-negative. (Default: 1)
    scl : scalar, optional
        Each differentiation is multiplied by `scl`.  The end result is
        multiplication by ``scl**m``.  This is for use in a linear change of
        variable. (Default: 1)
    axis : int, optional
        Axis along which differentiation is performed. (Default: 0)

    """
    # 将输入参数 `c` 转换为至少一维的 numpy 数组，并进行复制
    c = np.array(c, ndmin=1, copy=True)
    
    # 如果数组 `c` 的数据类型是布尔型、字节型、短整型、整型或长整型，则转换为双精度浮点型
    if c.dtype.char in '?bBhHiIlLqQpP':
        c = c.astype(np.double)
    
    # 将参数 `m` 转换为整数，表示导数的阶数
    cnt = pu._as_int(m, "the order of derivation")
    
    # 将参数 `axis` 转换为整数，表示进行导数计算的轴
    iaxis = pu._as_int(axis, "the axis")
    
    # 如果导数的阶数 `cnt` 小于 0，则抛出 ValueError 异常
    if cnt < 0:
        raise ValueError("The order of derivation must be non-negative")
    
    # 根据参数 `axis` 规范化轴的索引，确保在数组 `c` 的维度范围内
    iaxis = normalize_axis_index(iaxis, c.ndim)
    
    # 如果导数的阶数 `cnt` 等于 0，则直接返回输入数组 `c`
    if cnt == 0:
        return c
    
    # 将数组 `c` 的轴 `iaxis` 移动到最前面，便于后续计算
    c = np.moveaxis(c, iaxis, 0)
    
    # 获取数组 `c` 的长度 `n`
    n = len(c)
    
    # 如果导数的阶数 `cnt` 大于等于 `n`，则将数组 `c` 的前部分清零
    if cnt >= n:
        c = c[:1] * 0
    else:
        # 否则，进行导数的计算
        for i in range(cnt):
            n = n - 1
            c *= scl  # 将数组 `c` 各元素乘以缩放因子 `scl`
            # 创建空的导数数组 `der`，其形状与数组 `c` 的后续维度一致
            der = np.empty((n,) + c.shape[1:], dtype=c.dtype)
            for j in range(n, 2, -1):
                # 计算 Chebyshev 多项式的导数
                der[j - 1] = (2 * j) * c[j]
                c[j - 2] += (j * c[j]) / (j - 2)
            if n > 1:
                der[1] = 4 * c[2]
            der[0] = c[1]
            c = der
    
    # 将导数计算后的数组 `c` 恢复原来的轴顺序，与输入数组 `c` 一致
    c = np.moveaxis(c, 0, iaxis)
    
    # 返回计算得到的 Chebyshev 多项式的导数数组 `c`
    return c
# 定义函数 chebint，用于对 Chebyshev 级数进行积分
def chebint(c, m=1, k=[], lbnd=0, scl=1, axis=0):
    """
    Integrate a Chebyshev series.

    Returns the Chebyshev series coefficients `c` integrated `m` times from
    `lbnd` along `axis`. At each iteration the resulting series is
    **multiplied** by `scl` and an integration constant, `k`, is added.
    The scaling factor is for use in a linear change of variable.  ("Buyer
    beware": note that, depending on what one is doing, one may want `scl`
    to be the reciprocal of what one might expect; for more information,
    see the Notes section below.)  The argument `c` is an array of
    coefficients from low to high degree along each axis, e.g., [1,2,3]
    represents the series ``T_0 + 2*T_1 + 3*T_2`` while [[1,2],[1,2]]
    represents ``1*T_0(x)*T_0(y) + 1*T_1(x)*T_0(y) + 2*T_0(x)*T_1(y) +
    2*T_1(x)*T_1(y)`` if axis=0 is ``x`` and axis=1 is ``y``.

    Parameters
    ----------
    c : array_like
        Array of Chebyshev series coefficients. If c is multidimensional
        the different axis correspond to different variables with the
        degree in each axis given by the corresponding index.
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
        C-series coefficients of the integral.

    Raises
    ------
    ValueError
        If ``m < 1``, ``len(k) > m``, ``np.ndim(lbnd) != 0``, or
        ``np.ndim(scl) != 0``.

    See Also
    --------
    chebder

    Notes
    -----
    Note that the result of each integration is *multiplied* by `scl`.
    Why is this important to note?  Say one is making a linear change of
    variable :math:`u = ax + b` in an integral relative to `x`.  Then
    :math:`dx = du/a`, so one will need to set `scl` equal to
    :math:`1/a`- perhaps not what one would have first thought.

    Also note that, in general, the result of integrating a C-series needs
    to be "reprojected" onto the C-series basis set.  Thus, typically,
    the result of this function is "unintuitive," albeit correct; see
    Examples section below.

    Examples
    --------
    >>> from numpy.polynomial import chebyshev as C
    >>> c = (1,2,3)
    >>> C.chebint(c)
    array([ 0.5, -0.5,  0.5,  0.5])
    >>> C.chebint(c,3)

    """
    # 将输入的参数 c 转换为至少有一维的 numpy 数组，并保留其副本
    c = np.array(c, ndmin=1, copy=True)
    
    # 如果数组 c 的数据类型为布尔型或整型，将其转换为双精度浮点型
    if c.dtype.char in '?bBhHiIlLqQpP':
        c = c.astype(np.double)
    
    # 如果参数 k 不可迭代（即非列表或数组），将其转换为列表
    if not np.iterable(k):
        k = [k]
    
    # 将参数 m 规范化为整数，表示积分的阶数
    cnt = pu._as_int(m, "the order of integration")
    
    # 将参数 axis 规范化为整数，表示操作的轴
    iaxis = pu._as_int(axis, "the axis")
    
    # 如果积分的阶数 cnt 小于 0，抛出 ValueError 异常
    if cnt < 0:
        raise ValueError("The order of integration must be non-negative")
    
    # 如果 k 的长度大于 cnt，抛出 ValueError 异常
    if len(k) > cnt:
        raise ValueError("Too many integration constants")
    
    # 如果 lbnd 不是标量，抛出 ValueError 异常
    if np.ndim(lbnd) != 0:
        raise ValueError("lbnd must be a scalar.")
    
    # 如果 scl 不是标量，抛出 ValueError 异常
    if np.ndim(scl) != 0:
        raise ValueError("scl must be a scalar.")
    
    # 将操作的轴 iaxis 规范化，确保其在 c 的维度范围内
    iaxis = normalize_axis_index(iaxis, c.ndim)
    
    # 如果积分的阶数 cnt 等于 0，直接返回参数 c，无需积分计算
    if cnt == 0:
        return c
    
    # 将 c 数组中的轴移动到索引位置 0，方便后续的积分操作
    c = np.moveaxis(c, iaxis, 0)
    
    # 将列表 k 补充到长度为 cnt，不足部分用 0 填充
    k = list(k) + [0]*(cnt - len(k))
    
    # 开始进行积分操作，循环 cnt 次
    for i in range(cnt):
        n = len(c)  # 获取当前 c 数组的长度
        c *= scl  # 将 c 数组中的所有元素乘以标量 scl
    
        # 如果 c 数组长度为 1，且其唯一元素为 0，则在该元素上加上 k[i]
        if n == 1 and np.all(c[0] == 0):
            c[0] += k[i]
        else:
            # 创建一个临时数组 tmp，用于存储积分过程中的中间结果
            tmp = np.empty((n + 1,) + c.shape[1:], dtype=c.dtype)
            tmp[0] = c[0]*0  # 第一个元素为 0
            tmp[1] = c[0]  # 第二个元素为 c[0]
    
            # 如果 n 大于 1，则计算后续元素的值
            if n > 1:
                tmp[2] = c[1]/4
            for j in range(2, n):
                tmp[j + 1] = c[j]/(2*(j + 1))
                tmp[j - 1] -= c[j]/(2*(j - 1))
    
            # 计算 tmp[0]，加上 k[i] 减去 chebval(lbnd, tmp) 的值
            tmp[0] += k[i] - chebval(lbnd, tmp)
            c = tmp
    
    # 完成积分操作后，将 c 数组中的轴移回到原始位置 iaxis
    c = np.moveaxis(c, 0, iaxis)
    
    # 返回最终的积分结果数组 c
    return c
def chebval(x, c, tensor=True):
    """
    Evaluate a Chebyshev series at points x.

    If `c` is of length `n + 1`, this function returns the value:

    .. math:: p(x) = c_0 * T_0(x) + c_1 * T_1(x) + ... + c_n * T_n(x)

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
    chebval2d, chebgrid2d, chebval3d, chebgrid3d

    Notes
    -----
    The evaluation uses Clenshaw recursion, aka synthetic division.

    """
    # 将输入的系数数组转换成至少为1维的 numpy 数组，确保数据可用性和一致性
    c = np.array(c, ndmin=1, copy=True)
    # 如果系数数组的数据类型是布尔型或整型，则转换为双精度浮点型
    if c.dtype.char in '?bBhHiIlLqQpP':
        c = c.astype(np.double)
    # 如果输入的 x 是列表或元组，则转换为 ndarray 数组
    if isinstance(x, (tuple, list)):
        x = np.asarray(x)
    # 如果 x 是 ndarray 类型且 tensor 为 True，则调整系数数组的形状以便广播计算
    if isinstance(x, np.ndarray) and tensor:
        c = c.reshape(c.shape + (1,)*x.ndim)

    # 根据系数数组的长度选择不同的计算方法
    if len(c) == 1:
        c0 = c[0]
        c1 = 0
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        x2 = 2*x
        c0 = c[-2]
        c1 = c[-1]
        # 使用 Clenshaw 递归方法计算 Chebyshev 多项式的值
        for i in range(3, len(c) + 1):
            tmp = c0
            c0 = c[-i] - c1
            c1 = tmp + c1*x2
    # 返回表达式 c0 + c1*x 的计算结果
    return c0 + c1*x
# 在给定点(x, y)评估二维切比雪夫级数。

def chebval2d(x, y, c):
    # 调用 pu 模块的 _valnd 函数，用于评估二维切比雪夫级数的值。
    return pu._valnd(chebval, c, x, y)


# 在 x 和 y 的笛卡尔积上评估二维切比雪夫级数。

def chebgrid2d(x, y, c):
    # 返回值为在网格点 (a, b) 上的二维切比雪夫级数值。

    # 如果 c 的维度少于两个，隐式添加维度使其成为二维数组。
    # 结果的形状将为 c.shape[2:] + x.shape + y.shape。
    c : array_like
        # 输入参数 c 是一个类数组结构，包含系数，按照多重度量 i,j 的系数存储在 `c[i,j]` 中。
        # 如果 `c` 的维度高于两个，其余的索引用来枚举多组系数。

    Returns
    -------
    values : ndarray, compatible object
        # 返回值 `values` 是一个 ndarray 或兼容对象，包含二维切比雪夫级数在笛卡尔积 `x` 和 `y` 点上的值。

    See Also
    --------
    chebval, chebval2d, chebval3d, chebgrid3d
        # 参见其他相关函数：chebval, chebval2d, chebval3d, chebgrid3d。

    Notes
    -----
    .. versionadded:: 1.7.0
        # 添加于版本 1.7.0。

    """
    # 调用内部函数 `_gridnd`，计算二维切比雪夫系列在给定参数下的值
    return pu._gridnd(chebval, c, x, y)
# 计算三维切比雪夫级数在点 (x, y, z) 处的值。
def chebval3d(x, y, z, c):
    # 调用 _valnd 函数来计算三维切比雪夫级数的值，传入的参数是 chebval, c, x, y, z。
    return pu._valnd(chebval, c, x, y, z)


# 在 x、y 和 z 的笛卡尔积上评估三维切比雪夫级数。
def chebgrid3d(x, y, z, c):
    # 返回值是在点 (a, b, c) 处的值，其中 a 取自 x，b 取自 y，c 取自 z，使用三维切比雪夫多项式系数 c。
    # 结果点形成一个网格，x 在第一维，y 在第二维，z 在第三维。
    # 参数 x, y 和 z 如果是元组或列表，则转换为数组；否则视为标量。
    # 如果 c 的维度少于三维，会隐式添加维度使其变为三维。
    # 返回值的形状将是 c.shape[3:] + x.shape + y.shape + z.shape。
    return pu._valnd(chebval, c, x, y, z)
    x, y, z : array_like, compatible objects
        三维系列在`x`、`y`和`z`的笛卡尔积中的点上进行评估。如果`x`、`y`或`z`是列表或元组，则首先转换为ndarray，否则保持不变，如果它不是ndarray，则视为标量。
    c : array_like
        有序系数数组，使得度为i、j的项的系数包含在`c[i,j]`中。如果`c`的维度大于两个，则其余的索引列举多组系数。

    Returns
    -------
    values : ndarray, compatible object
        在`x`和`y`的笛卡尔积中点上的二维多项式的值。

    See Also
    --------
    chebval, chebval2d, chebgrid2d, chebval3d

    Notes
    -----

    .. versionadded:: 1.7.0

    """
    使用`chebval`函数计算在给定参数下的多项式值，并通过`pu._gridnd`进行网格化处理。
def chebvander(x, deg):
    """Pseudo-Vandermonde matrix of given degree.

    Returns the pseudo-Vandermonde matrix of degree `deg` and sample points
    `x`. The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., i] = T_i(x),

    where ``0 <= i <= deg``. The leading indices of `V` index the elements of
    `x` and the last index is the degree of the Chebyshev polynomial.

    If `c` is a 1-D array of coefficients of length ``n + 1`` and `V` is the
    matrix ``V = chebvander(x, n)``, then ``np.dot(V, c)`` and
    ``chebval(x, c)`` are the same up to roundoff.  This equivalence is
    useful both for least squares fitting and for the evaluation of a large
    number of Chebyshev series of the same degree and sample points.

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
        The pseudo Vandermonde matrix. The shape of the returned matrix is
        ``x.shape + (deg + 1,)``, where The last index is the degree of the
        corresponding Chebyshev polynomial.  The dtype will be the same as
        the converted `x`.

    """
    # Convert deg to an integer if possible, raising a ValueError if deg < 0
    ideg = pu._as_int(deg, "deg")
    if ideg < 0:
        raise ValueError("deg must be non-negative")

    # Ensure x is converted to a 1-D array of type float64 or complex128
    x = np.array(x, copy=None, ndmin=1) + 0.0
    # Create dimensions for the result matrix v
    dims = (ideg + 1,) + x.shape
    dtyp = x.dtype
    # Create an uninitialized array v with specified dimensions and type
    v = np.empty(dims, dtype=dtyp)

    # Use forward recursion to generate the entries of v
    v[0] = x*0 + 1  # Initialize the first row of v
    if ideg > 0:
        x2 = 2*x
        v[1] = x  # Initialize the second row of v
        # Use recursion to fill in the remaining rows of v
        for i in range(2, ideg + 1):
            v[i] = v[i-1]*x2 - v[i-2]

    # Move the first axis to the last axis to match the expected output shape
    return np.moveaxis(v, 0, -1)


def chebvander2d(x, y, deg):
    """Pseudo-Vandermonde matrix of given degrees.

    Returns the pseudo-Vandermonde matrix of degrees `deg` and sample
    points ``(x, y)``. The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., (deg[1] + 1)*i + j] = T_i(x) * T_j(y),

    where ``0 <= i <= deg[0]`` and ``0 <= j <= deg[1]``. The leading indices of
    `V` index the points ``(x, y)`` and the last index encodes the degrees of
    the Chebyshev polynomials.

    If ``V = chebvander2d(x, y, [xdeg, ydeg])``, then the columns of `V`
    correspond to the elements of a 2-D coefficient array `c` of shape
    (xdeg + 1, ydeg + 1) in the order

    .. math:: c_{00}, c_{01}, c_{02} ... , c_{10}, c_{11}, c_{12} ...

    and ``np.dot(V, c.flat)`` and ``chebval2d(x, y, c)`` will be the same
    up to roundoff. This equivalence is useful both for least squares
    fitting and for the evaluation of a large number of 2-D Chebyshev
    series of the same degrees and sample points.

    Parameters
    ----------
    x : array_like
        Array of points for the first dimension.
    y : array_like
        Array of points for the second dimension.
    deg : list or tuple
        List or tuple specifying the degrees of the resulting matrix along each dimension.

    Returns
    -------
    vander : ndarray
        The pseudo Vandermonde matrix. The shape of the returned matrix is
        ``x.shape + y.shape + (np.prod(deg) + 1,)``. The last index is the degree of the
        corresponding 2-D Chebyshev polynomial. The dtype will be the same as
        the dtype of the concatenated `x` and `y`.

    """
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
    chebvander, chebvander3d, chebval2d, chebval3d

    Notes
    -----

    .. versionadded:: 1.7.0

    """
    # 调用 _vander_nd_flat 函数计算 2D Vandermonde 矩阵
    return pu._vander_nd_flat((chebvander, chebvander), (x, y), deg)
# 导入 pu 模块，该模块包含了 _vander_nd_flat 函数
import pu

# 计算 3D Chebyshev 矩阵的伪范德蒙德矩阵
def chebvander3d(x, y, z, deg):
    """Pseudo-Vandermonde matrix of given degrees.

    Returns the pseudo-Vandermonde matrix of degrees `deg` and sample
    points ``(x, y, z)``. If `l`, `m`, `n` are the given degrees in `x`, `y`, `z`,
    then The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., (m+1)(n+1)i + (n+1)j + k] = T_i(x)*T_j(y)*T_k(z),

    where ``0 <= i <= l``, ``0 <= j <= m``, and ``0 <= j <= n``.  The leading
    indices of `V` index the points ``(x, y, z)`` and the last index encodes
    the degrees of the Chebyshev polynomials.

    If ``V = chebvander3d(x, y, z, [xdeg, ydeg, zdeg])``, then the columns
    of `V` correspond to the elements of a 3-D coefficient array `c` of
    shape (xdeg + 1, ydeg + 1, zdeg + 1) in the order

    .. math:: c_{000}, c_{001}, c_{002},... , c_{010}, c_{011}, c_{012},...

    and ``np.dot(V, c.flat)`` and ``chebval3d(x, y, z, c)`` will be the
    same up to roundoff. This equivalence is useful both for least squares
    fitting and for the evaluation of a large number of 3-D Chebyshev
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
    chebvander, chebvander3d, chebval2d, chebval3d

    Notes
    -----

    .. versionadded:: 1.7.0

    """
    # 调用 _vander_nd_flat 函数来计算伪范德蒙德矩阵
    return pu._vander_nd_flat((chebvander, chebvander, chebvander), (x, y, z), deg)


# Chebyshev 多项式最小二乘拟合
def chebfit(x, y, deg, rcond=None, full=False, w=None):
    """
    Least squares fit of Chebyshev series to data.

    Return the coefficients of a Chebyshev series of degree `deg` that is the
    least squares fit to the data values `y` given at points `x`. If `y` is
    1-D the returned coefficients will also be 1-D. If `y` is 2-D multiple
    fits are done, one for each column of `y`, and the resulting
    coefficients are stored in the corresponding columns of a 2-D return.
    The fitted polynomial(s) are in the form

    .. math::  p(x) = c_0 + c_1 * T_1(x) + ... + c_n * T_n(x),

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
        Degree of the Chebyshev series to be used for the fit.
    rcond : float, optional
        Cutoff for small singular values.
    full : bool, optional
        If True, return additional outputs.
    w : array_like, optional
        Weights applied to the y-coordinates.

    Returns
    -------
    coef : ndarray
        Coefficients of the Chebyshev series, with shape (deg + 1,) or
        (deg + 1, K) if `y` is 2-D.
    residuals, rank, s : ndarray, int, ndarray
        Optional outputs as determined by `full`.

    See Also
    --------
    chebvander, chebvander2d, chebvander3d, chebval, chebval2d, chebval3d

    Notes
    -----
    This function is generally preferred over polyfit when the fitting
    is done with respect to Chebyshev series due to its better numerical
    properties.

    """
    pass  # 这个函数仅作为占位符，没有实际代码实现
    deg : int or 1-D array_like
        # 多项式拟合的阶数或阶数组成的一维数组。如果 `deg` 是一个整数，
        # 则包括从0阶到`deg`阶的所有项。对于 NumPy 版本 >= 1.11.0，可以使用
        # 整数列表指定要包含的项的阶数。

    rcond : float, optional
        # 拟合中使用的相对条件数。比最大奇异值小于此值的奇异值将被忽略。
        # 默认值是 `len(x)*eps`，其中 eps 是浮点类型的相对精度，
        # 在大多数情况下约为2e-16。

    full : bool, optional
        # 决定返回值的性质的开关。当为 False（默认值）时，仅返回系数；
        # 当为 True 时，还返回奇异值分解的诊断信息。

    w : array_like, shape (`M`,), optional
        # 权重数组。如果不为 None，则权重 `w[i]` 应用于 `x[i]` 处的非平方残差 `y[i] - y_hat[i]`。
        # 理想情况下，权重应选择使得所有产品 `w[i]*y[i]` 的误差具有相同的方差。
        # 当使用逆方差加权时，使用 `w[i] = 1/sigma(y[i])`。
        # 默认值为 None。

        .. versionadded:: 1.5.0
            # 添加于 NumPy 1.5.0 版本。

    Returns
    -------
    coef : ndarray, shape (M,) or (M, K)
        # 从低到高排序的切比雪夫系数。如果 `y` 是二维的，则第 k 列数据的系数位于第 k 列中。

    [residuals, rank, singular_values, rcond] : list
        # 仅当 `full == True` 时返回这些值。

        - residuals -- 最小二乘拟合的残差平方和
        - rank -- 缩放后的范德蒙矩阵的数值秩
        - singular_values -- 缩放后的范德蒙矩阵的奇异值
        - rcond -- `rcond` 的值

        详细信息请参见 `numpy.linalg.lstsq`。

    Warns
    -----
    RankWarning
        # 在最小二乘拟合的系数矩阵的秩不足时发出警告。仅当 `full == False` 时才会引发警告。
        # 可以通过以下方式关闭警告：

        >>> import warnings
        >>> warnings.simplefilter('ignore', np.exceptions.RankWarning)

    See Also
    --------
    numpy.polynomial.polynomial.polyfit
    numpy.polynomial.legendre.legfit
    numpy.polynomial.laguerre.lagfit
    numpy.polynomial.hermite.hermfit
    numpy.polynomial.hermite_e.hermefit
    chebval : 计算切比雪夫级数的值。
    chebvander : 切比雪夫级数的范德蒙矩阵。
    chebweight : 切比雪夫权重函数。
    numpy.linalg.lstsq : 从矩阵计算最小二乘拟合。
    scipy.interpolate.UnivariateSpline : 计算样条拟合。

    Notes
    -----
    # 解决方案是切比雪夫级数 `p` 的系数，该级数最小化加权平方误差的和。

    .. math:: E = \\sum_j w_j^2 * |y_j - p(x_j)|^2,
    # 返回使用 `_fit` 函数进行拟合后的结果
    return pu._fit(chebvander, x, y, deg, rcond, full, w)
    # c is a trimmed copy
    [c] = pu.as_series([c])
    # Ensure the input coefficient array `c` is treated as a 1-D series
    # using the `pu.as_series` function from the `pu` module.

    if len(c) < 2:
        # Raise an error if the series length is less than 2,
        # as a Chebyshev series must have at least degree 1.
        raise ValueError('Series must have maximum degree of at least 1.')

    if len(c) == 2:
        # Return a 1x1 array with the root of the linear polynomial if its length is 2.
        return np.array([[-c[0]/c[1]]])

    # Calculate the degree of the polynomial represented by `c`.
    n = len(c) - 1
    # Initialize a square matrix `mat` of zeros with dimensions (n, n),
    # using the data type of `c`.
    mat = np.zeros((n, n), dtype=c.dtype)
    
    # Create an array `scl` containing scaling factors for the companion matrix,
    # where the first element is 1 and subsequent elements are square root of 0.5.
    scl = np.array([1.] + [np.sqrt(.5)]*(n-1))
    
    # Extract the top and bottom diagonals of the reshaped matrix `mat`.
    top = mat.reshape(-1)[1::n+1]
    bot = mat.reshape(-1)[n::n+1]
    
    # Set specific values in `top` and `bot` arrays to improve eigenvalue estimates.
    top[0] = np.sqrt(.5)
    top[1:] = 1/2
    bot[...] = top
    
    # Adjust the last column of `mat` using coefficients from `c` to form the scaled
    # companion matrix, enhancing stability and eigenvalue accuracy.
    mat[:, -1] -= (c[:-1]/c[-1])*(scl/scl[-1])*.5

    # Return the scaled companion matrix `mat`.
    return mat
    # 计算矩阵 m 的特征值
    r = la.eigvals(m)
    # 对特征值列表进行排序（默认是按照实部的大小进行排序）
    r.sort()
    # 返回排序后的特征值列表
    return r
# 使用切比雪夫点的第一类插值法插值函数 `func`。
def chebinterpolate(func, deg, args=()):
    """Interpolate a function at the Chebyshev points of the first kind.

    Returns the Chebyshev series that interpolates `func` at the Chebyshev
    points of the first kind in the interval [-1, 1]. The interpolating
    series tends to a minmax approximation to `func` with increasing `deg`
    if the function is continuous in the interval.

    .. versionadded:: 1.14.0

    Parameters
    ----------
    func : function
        The function to be approximated. It must be a function of a single
        variable of the form ``f(x, a, b, c...)``, where ``a, b, c...`` are
        extra arguments passed in the `args` parameter.
    deg : int
        Degree of the interpolating polynomial
    args : tuple, optional
        Extra arguments to be used in the function call. Default is no extra
        arguments.

    Returns
    -------
    coef : ndarray, shape (deg + 1,)
        Chebyshev coefficients of the interpolating series ordered from low to
        high.

    Examples
    --------
    >>> import numpy.polynomial.chebyshev as C
    >>> C.chebinterpolate(lambda x: np.tanh(x) + 0.5, 8)
    array([  5.00000000e-01,   8.11675684e-01,  -9.86864911e-17,
            -5.42457905e-02,  -2.71387850e-16,   4.51658839e-03,
             2.46716228e-17,  -3.79694221e-04,  -3.26899002e-16])

    Notes
    -----

    The Chebyshev polynomials used in the interpolation are orthogonal when
    sampled at the Chebyshev points of the first kind. If it is desired to
    constrain some of the coefficients they can simply be set to the desired
    value after the interpolation, no new interpolation or fit is needed. This
    is especially useful if it is known apriori that some of coefficients are
    zero. For instance, if the function is even then the coefficients of the
    terms of odd degree in the result can be set to zero.

    """
    # 将 deg 转换为 NumPy 数组，确保其为一维整数类型
    deg = np.asarray(deg)

    # 检查参数
    if deg.ndim > 0 or deg.dtype.kind not in 'iu' or deg.size == 0:
        raise TypeError("deg must be an int")
    if deg < 0:
        raise ValueError("expected deg >= 0")

    # 计算插值多项式的阶数
    order = deg + 1

    # 获取切比雪夫点的第一类
    xcheb = chebpts1(order)

    # 计算函数在切比雪夫点的第一类上的取值
    yfunc = func(xcheb, *args)

    # 计算切比雪夫多项式的伴随矩阵
    m = chebvander(xcheb, deg)

    # 计算插值多项式的系数
    c = np.dot(m.T, yfunc)

    # 将首个系数除以阶数
    c[0] /= order

    # 将其余系数除以 0.5 倍的阶数
    c[1:] /= 0.5 * order

    return c


# Gauss-Chebyshev 积分
def chebgauss(deg):
    """
    Gauss-Chebyshev quadrature.

    Computes the sample points and weights for Gauss-Chebyshev quadrature.
    These sample points and weights will correctly integrate polynomials of
    degree :math:`2*deg - 1` or less over the interval :math:`[-1, 1]` with
    the weight function :math:`f(x) = 1/\\sqrt{1 - x^2}`.

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


    """
    """
        The results have only been tested up to degree 100, higher degrees may
        be problematic. For Gauss-Chebyshev there are closed form solutions for
        the sample points and weights. If n = `deg`, then
    
        .. math:: x_i = \\cos(\\pi (2 i - 1) / (2 n))
    
        .. math:: w_i = \\pi / n
    
        """
    
        # 将 deg 转换为整数
        ideg = pu._as_int(deg, "deg")
    
        # 如果 ideg 小于等于 0，则抛出数值错误
        if ideg <= 0:
            raise ValueError("deg must be a positive integer")
    
        # 根据 Gauss-Chebyshev 公式计算采样点 x_i
        x = np.cos(np.pi * np.arange(1, 2*ideg, 2) / (2.0*ideg))
        
        # 设置权重 w_i，每个权重为 π/n
        w = np.ones(ideg)*(np.pi/ideg)
    
        # 返回计算得到的采样点 x 和权重 w
        return x, w
# 定义了一个计算第一类切比雪夫点的函数
def chebpts1(npts):
    # 将 npts 转换为整数
    _npts = int(npts)
    # 检查 npts 是否与 _npts 不相等，如果不相等则抛出值错误异常
    if _npts != npts:
        raise ValueError("npts must be integer")
    # 检查 _npts 是否小于 1，如果是则抛出值错误异常
    if _npts < 1:
        raise ValueError("npts must be >= 1")

    # 计算切比雪夫点的角度值，x = [pi*(k + .5)/npts for k in range(npts)]
    x = 0.5 * np.pi / _npts * np.arange(-_npts+1, _npts+1, 2)
    # 返回计算得到的切比雪夫点的第一类函数值的正弦值
    return np.sin(x)


# 定义了一个计算第二类切比雪夫点的函数
def chebpts2(npts):
    # 将 npts 转换为整数
    _npts = int(npts)
    # 检查 npts 是否与 _npts 不相等，如果不相等则抛出值错误异常
    if _npts != npts:
        raise ValueError("npts must be integer")
    # 检查 _npts 是否小于 2，如果是则抛出值错误异常
    if _npts < 2:
        raise ValueError("npts must be >= 2")

    # 计算切比雪夫点的角度值，x = [pi*k/(npts - 1) for k in range(npts)]，并按升序排序
    x = np.linspace(-np.pi, 0, _npts)
    # 返回计算得到的切比雪夫点的第二类函数值的余弦值
    return np.cos(x)


# Chebyshev 类，继承自 ABCPolyBase 类
class Chebyshev(ABCPolyBase):
    """A Chebyshev series class.

    The Chebyshev class provides the standard Python numerical methods
    '+', '-', '*', '//', '%', 'divmod', '**', and '()' as well as the
    attributes and methods listed below.

    Parameters
    ----------
    coef : array_like
        Chebyshev coefficients in order of increasing degree, i.e.,
        ``(1, 2, 3)`` gives ``1*T_0(x) + 2*T_1(x) + 3*T_2(x)``.
    domain : (2,) array_like, optional
        Domain to use. The interval ``[domain[0], domain[1]]`` is mapped
        to the interval ``[window[0], window[1]]`` by shifting and scaling.
        The default value is [-1, 1].
    window : (2,) array_like, optional
        Window, see `domain` for its use. The default value is [-1, 1].

        .. versionadded:: 1.6.0

    """
    pass
    symbol : str, optional
        Symbol used to represent the independent variable in string
        representations of the polynomial expression, e.g. for printing.
        The symbol must be a valid Python identifier. Default value is 'x'.

        .. versionadded:: 1.24

    """
    # 定义一些虚拟函数，这些函数用于处理 Chebyshev 多项式的各种运算
    _add = staticmethod(chebadd)
    _sub = staticmethod(chebsub)
    _mul = staticmethod(chebmul)
    _div = staticmethod(chebdiv)
    _pow = staticmethod(chebpow)
    _val = staticmethod(chebval)
    _int = staticmethod(chebint)
    _der = staticmethod(chebder)
    _fit = staticmethod(chebfit)
    _line = staticmethod(chebline)
    _roots = staticmethod(chebroots)
    _fromroots = staticmethod(chebfromroots)

    @classmethod
    def interpolate(cls, func, deg, domain=None, args=()):
        """Interpolate a function at the Chebyshev points of the first kind.

        Returns the series that interpolates `func` at the Chebyshev points of
        the first kind scaled and shifted to the `domain`. The resulting series
        tends to a minmax approximation of `func` when the function is
        continuous in the domain.

        .. versionadded:: 1.14.0

        Parameters
        ----------
        func : function
            The function to be interpolated. It must be a function of a single
            variable of the form ``f(x, a, b, c...)``, where ``a, b, c...`` are
            extra arguments passed in the `args` parameter.
        deg : int
            Degree of the interpolating polynomial.
        domain : {None, [beg, end]}, optional
            Domain over which `func` is interpolated. The default is None, in
            which case the domain is [-1, 1].
        args : tuple, optional
            Extra arguments to be used in the function call. Default is no
            extra arguments.

        Returns
        -------
        polynomial : Chebyshev instance
            Interpolating Chebyshev instance.

        Notes
        -----
        See `numpy.polynomial.chebinterpolate` for more details.

        """
        if domain is None:
            domain = cls.domain
        # 创建一个函数 xfunc，将 func 在指定 domain 上的映射和参数 args 一起应用
        xfunc = lambda x: func(pu.mapdomain(x, cls.window, domain), *args)
        # 调用 chebinterpolate 函数生成系数 coef
        coef = chebinterpolate(xfunc, deg)
        # 返回一个 Chebyshev 实例，以 coef 作为系数，domain 作为定义域
        return cls(coef, domain=domain)

    # 定义一些虚拟属性
    domain = np.array(chebdomain)
    window = np.array(chebdomain)
    basis_name = 'T'
```