# `.\numpy\numpy\polynomial\polyutils.py`

```
# 导入操作符和函数工具
import operator
# 导入函数工具包装，用于函数操作
import functools
# 导入警告模块
import warnings

# 导入 NumPy 库并重命名为 np
import numpy as np

# 从 NumPy 库中导入 dragon4_positional 和 dragon4_scientific 函数
from numpy._core.multiarray import dragon4_positional, dragon4_scientific
# 从 NumPy 库中导入 RankWarning 异常类
from numpy.exceptions import RankWarning

# 模块中公开的函数和类列表
__all__ = [
    'as_series', 'trimseq', 'trimcoef', 'getdomain', 'mapdomain', 'mapparms',
    'format_float']

# 
# 辅助函数用于将输入转换为 1-D 数组
#

def trimseq(seq):
    """Remove small Poly series coefficients.

    Parameters
    ----------
    seq : sequence
        Sequence of Poly series coefficients.

    Returns
    -------
    series : sequence
        Subsequence with trailing zeros removed. If the resulting sequence
        would be empty, return the first element. The returned sequence may
        or may not be a view.

    Notes
    -----
    Do not lose the type info if the sequence contains unknown objects.

    """
    # 如果序列长度为 0 或者最后一个元素不为 0，则直接返回序列
    if len(seq) == 0 or seq[-1] != 0:
        return seq
    else:
        # 从后向前遍历序列，找到第一个不为 0 的元素的索引
        for i in range(len(seq) - 1, -1, -1):
            if seq[i] != 0:
                break
        # 返回从开头到第一个不为 0 的元素的子序列
        return seq[:i+1]


def as_series(alist, trim=True):
    """
    Return argument as a list of 1-d arrays.

    The returned list contains array(s) of dtype double, complex double, or
    object.  A 1-d argument of shape ``(N,)`` is parsed into ``N`` arrays of
    size one; a 2-d argument of shape ``(M,N)`` is parsed into ``M`` arrays
    of size ``N`` (i.e., is "parsed by row"); and a higher dimensional array
    raises a Value Error if it is not first reshaped into either a 1-d or 2-d
    array.

    Parameters
    ----------
    alist : array_like
        A 1- or 2-d array_like
    trim : boolean, optional
        When True, trailing zeros are removed from the inputs.
        When False, the inputs are passed through intact.

    Returns
    -------
    [a1, a2,...] : list of 1-D arrays
        A copy of the input data as a list of 1-d arrays.

    Raises
    ------
    ValueError
        Raised when `as_series` cannot convert its input to 1-d arrays, or at
        least one of the resulting arrays is empty.

    Examples
    --------
    >>> from numpy.polynomial import polyutils as pu
    >>> a = np.arange(4)
    >>> pu.as_series(a)
    [array([0.]), array([1.]), array([2.]), array([3.])]
    >>> b = np.arange(6).reshape((2,3))
    >>> pu.as_series(b)
    [array([0., 1., 2.]), array([3., 4., 5.])]


    """
    # 如果 trim 参数为 True，则调用 trimseq 函数去除输入序列中的尾部零
    if trim:
        return [trimseq(a) for a in np.atleast_1d(alist)]
    else:
        # 否则直接返回输入的至少为 1-D 的数组列表
        return [np.asarray(a).flatten() for a in np.atleast_1d(alist)]
    """
    将输入的列表转换为 NumPy 数组的列表，每个数组至少有一维，并进行必要的验证和转换操作。

    Parameters:
    - alist: 输入的列表，每个元素将被转换为一个 NumPy 数组
    - trim: 可选参数，默认为 True。如果为 True，则对每个数组进行修剪操作。

    Returns:
    - ret: 转换后的 NumPy 数组的列表

    Raises:
    - ValueError: 如果任何一个数组为空，或者不是一维数组。
    - ValueError: 如果无法找到数组的公共数据类型。
    """
    # 将输入的每个元素转换为 NumPy 数组，确保每个数组至少有一维
    arrays = [np.array(a, ndmin=1, copy=None) for a in alist]
    
    # 检查每个数组是否为空，如果是，则引发异常
    for a in arrays:
        if a.size == 0:
            raise ValueError("Coefficient array is empty")
    
    # 检查每个数组是否为一维数组，如果不是，则引发异常
    if any(a.ndim != 1 for a in arrays):
        raise ValueError("Coefficient array is not 1-d")
    
    # 如果 trim 参数为 True，则对每个数组进行修剪操作
    if trim:
        arrays = [trimseq(a) for a in arrays]

    # 检查是否存在包含对象类型的数组，如果有，则进行特殊处理
    if any(a.dtype == np.dtype(object) for a in arrays):
        ret = []
        for a in arrays:
            if a.dtype != np.dtype(object):
                tmp = np.empty(len(a), dtype=np.dtype(object))
                tmp[:] = a[:]
                ret.append(tmp)
            else:
                ret.append(a.copy())
    else:
        # 尝试找到所有数组的公共数据类型
        try:
            dtype = np.common_type(*arrays)
        except Exception as e:
            raise ValueError("Coefficient arrays have no common type") from e
        
        # 使用公共数据类型创建新的 NumPy 数组列表
        ret = [np.array(a, copy=True, dtype=dtype) for a in arrays]
    
    # 返回转换后的 NumPy 数组列表
    return ret
# 定义函数 trimcoef，用于从多项式中移除末尾的小系数
def trimcoef(c, tol=0):
    """
    Remove "small" "trailing" coefficients from a polynomial.

    "Small" means "small in absolute value" and is controlled by the
    parameter `tol`; "trailing" means highest order coefficient(s), e.g., in
    ``[0, 1, 1, 0, 0]`` (which represents ``0 + x + x**2 + 0*x**3 + 0*x**4``)
    both the 3-rd and 4-th order coefficients would be "trimmed."

    Parameters
    ----------
    c : array_like
        1-d array of coefficients, ordered from lowest order to highest.
    tol : number, optional
        Trailing (i.e., highest order) elements with absolute value less
        than or equal to `tol` (default value is zero) are removed.

    Returns
    -------
    trimmed : ndarray
        1-d array with trailing zeros removed.  If the resulting series
        would be empty, a series containing a single zero is returned.

    Raises
    ------
    ValueError
        If `tol` < 0

    Examples
    --------
    >>> from numpy.polynomial import polyutils as pu
    >>> pu.trimcoef((0,0,3,0,5,0,0))
    array([0.,  0.,  3.,  0.,  5.])
    >>> pu.trimcoef((0,0,1e-3,0,1e-5,0,0),1e-3) # item == tol is trimmed
    array([0.])
    >>> i = complex(0,1) # works for complex
    >>> pu.trimcoef((3e-4,1e-3*(1-i),5e-4,2e-5*(1+i)), 1e-3)
    array([0.0003+0.j   , 0.001 -0.001j])

    """
    # 检查 tol 是否小于 0，若是则抛出 ValueError 异常
    if tol < 0:
        raise ValueError("tol must be non-negative")

    # 将输入系数 c 转换为序列
    [c] = as_series([c])
    # 找到绝对值大于 tol 的系数的索引
    [ind] = np.nonzero(np.abs(c) > tol)
    # 如果没有找到满足条件的索引，则返回一个包含单个零的数组
    if len(ind) == 0:
        return c[:1]*0
    else:
        # 否则，返回从开始到最后一个满足条件的系数的子数组的副本
        return c[:ind[-1] + 1].copy()

# 定义函数 getdomain，用于确定给定横坐标适合的定义域
def getdomain(x):
    """
    Return a domain suitable for given abscissae.

    Find a domain suitable for a polynomial or Chebyshev series
    defined at the values supplied.

    Parameters
    ----------
    x : array_like
        1-d array of abscissae whose domain will be determined.

    Returns
    -------
    domain : ndarray
        1-d array containing two values.  If the inputs are complex, then
        the two returned points are the lower left and upper right corners
        of the smallest rectangle (aligned with the axes) in the complex
        plane containing the points `x`. If the inputs are real, then the
        two points are the ends of the smallest interval containing the
        points `x`.

    See Also
    --------
    mapparms, mapdomain

    Examples
    --------
    >>> from numpy.polynomial import polyutils as pu
    >>> points = np.arange(4)**2 - 5; points
    array([-5, -4, -1,  4])
    >>> pu.getdomain(points)
    array([-5.,  4.])
    >>> c = np.exp(complex(0,1)*np.pi*np.arange(12)/6) # unit circle
    >>> pu.getdomain(c)
    array([-1.-1.j,  1.+1.j])

    """
    # 将输入的 x 转换为序列
    [x] = as_series([x], trim=False)
    # 如果 x 的数据类型为复数，则返回包含 x 所有点的最小矩形的两个角点
    if x.dtype.char in np.typecodes['Complex']:
        rmin, rmax = x.real.min(), x.real.max()
        imin, imax = x.imag.min(), x.imag.max()
        return np.array((complex(rmin, imin), complex(rmax, imax)))
    else:
        # 否则，返回包含 x 所有点的最小区间的两个端点
        return np.array((x.min(), x.max()))

# 定义函数 mapparms，未完待续...
    """
    Linear map parameters between domains.

    Return the parameters of the linear map ``offset + scale*x`` that maps
    `old` to `new` such that ``old[i] -> new[i]``, ``i = 0, 1``.

    Parameters
    ----------
    old, new : array_like
        Domains. Each domain must (successfully) convert to a 1-d array
        containing precisely two values.

    Returns
    -------
    offset, scale : scalars
        The map ``L(x) = offset + scale*x`` maps the first domain to the
        second.

    See Also
    --------
    getdomain, mapdomain

    Notes
    -----
    Also works for complex numbers, and thus can be used to calculate the
    parameters required to map any line in the complex plane to any other
    line therein.

    Examples
    --------
    >>> from numpy.polynomial import polyutils as pu
    >>> pu.mapparms((-1,1),(-1,1))
    (0.0, 1.0)
    >>> pu.mapparms((1,-1),(-1,1))
    (-0.0, -1.0)
    >>> i = complex(0,1)
    >>> pu.mapparms((-i,-1),(1,i))
    ((1+1j), (1-0j))

    """
    # Calculate the length of the old and new domains
    oldlen = old[1] - old[0]
    newlen = new[1] - new[0]
    # Calculate the offset using the formula for linear map parameters
    off = (old[1]*new[0] - old[0]*new[1]) / oldlen
    # Calculate the scale using the ratio of lengths between old and new domains
    scl = newlen / oldlen
    # Return the calculated offset and scale
    return off, scl
def mapdomain(x, old, new):
    """
    Apply linear map to input points.

    The linear map ``offset + scale*x`` that maps the domain `old` to
    the domain `new` is applied to the points `x`.

    Parameters
    ----------
    x : array_like
        Points to be mapped. If `x` is a subtype of ndarray the subtype
        will be preserved.
    old, new : array_like
        The two domains that determine the map.  Each must (successfully)
        convert to 1-d arrays containing precisely two values.

    Returns
    -------
    x_out : ndarray
        Array of points of the same shape as `x`, after application of the
        linear map between the two domains.

    See Also
    --------
    getdomain, mapparms

    Notes
    -----
    Effectively, this implements:

    .. math::
        x\\_out = new[0] + m(x - old[0])

    where

    .. math::
        m = \\frac{new[1]-new[0]}{old[1]-old[0]}

    Examples
    --------
    >>> from numpy.polynomial import polyutils as pu
    >>> old_domain = (-1,1)
    >>> new_domain = (0,2*np.pi)
    >>> x = np.linspace(-1,1,6); x
    array([-1. , -0.6, -0.2,  0.2,  0.6,  1. ])
    >>> x_out = pu.mapdomain(x, old_domain, new_domain); x_out
    array([ 0.        ,  1.25663706,  2.51327412,  3.76991118,  5.02654825, # may vary
            6.28318531])
    >>> x - pu.mapdomain(x_out, new_domain, old_domain)
    array([0., 0., 0., 0., 0., 0.])

    Also works for complex numbers (and thus can be used to map any line in
    the complex plane to any other line therein).

    >>> i = complex(0,1)
    >>> old = (-1 - i, 1 + i)
    >>> new = (-1 + i, 1 - i)
    >>> z = np.linspace(old[0], old[1], 6); z
    array([-1. -1.j , -0.6-0.6j, -0.2-0.2j,  0.2+0.2j,  0.6+0.6j,  1. +1.j ])
    >>> new_z = pu.mapdomain(z, old, new); new_z
    array([-1.0+1.j , -0.6+0.6j, -0.2+0.2j,  0.2-0.2j,  0.6-0.6j,  1.0-1.j ]) # may vary

    """
    x = np.asanyarray(x)  # Convert input `x` to an ndarray if it's not already
    off, scl = mapparms(old, new)  # Calculate offset and scale using mapparms function
    return off + scl*x


def _nth_slice(i, ndim):
    """
    Generate a slice object that selects the i-th index in an array.

    Parameters
    ----------
    i : int
        Index to be selected.
    ndim : int
        Number of dimensions in the array.

    Returns
    -------
    tuple
        Tuple of slice objects where the i-th position is sliced with `slice(None)`.

    """
    sl = [np.newaxis] * ndim  # Create a list of `None` slice objects with length `ndim`
    sl[i] = slice(None)  # Replace the i-th `None` with a full slice (`:`)
    return tuple(sl)  # Return as a tuple of slice objects


def _vander_nd(vander_fs, points, degrees):
    r"""
    A generalization of the Vandermonde matrix for N dimensions

    The result is built by combining the results of 1d Vandermonde matrices,

    .. math::
        W[i_0, \ldots, i_M, j_0, \ldots, j_N] = \prod_{k=0}^N{V_k(x_k)[i_0, \ldots, i_M, j_k]}

    where

    .. math::
        N &= \texttt{len(points)} = \texttt{len(degrees)} = \texttt{len(vander\_fs)} \\
        M &= \texttt{points[k].ndim} \\
        V_k &= \texttt{vander\_fs[k]} \\
        x_k &= \texttt{points[k]} \\
        0 \le j_k &\le \texttt{degrees[k]}

    Expanding the one-dimensional :math:`V_k` functions gives:

    .. math::
        W[i_0, \ldots, i_M, j_0, \ldots, j_N] = \prod_{k=0}^N{B_{k, j_k}(x_k[i_0, \ldots, i_M])}

    where :math:`B_{k,m}` is the m'th basis of the polynomial construction used along
    dimension :math:`k`. For a regular polynomial, :math:`B_{k, m}(x) = P_m(x) = x^m`.

    """
    pass  # Placeholder function, does not perform any operations
    Parameters
    ----------
    vander_fs : Sequence[function(array_like, int) -> ndarray]
        每个轴上要使用的一维范德蒙德函数，例如 `polyvander`
    points : Sequence[array_like]
        点的坐标数组，所有数组的形状必须相同。其元素的数据类型将转换为 float64 或 complex128，具体取决于是否包含复数。标量将转换为一维数组。
        这个参数的长度必须与 `vander_fs` 相同。
    degrees : Sequence[int]
        每个轴要使用的最大阶数（包括在内）。这个参数的长度必须与 `vander_fs` 相同。

    Returns
    -------
    vander_nd : ndarray
        形状为 ``points[0].shape + tuple(d + 1 for d in degrees)`` 的数组。
    """
    # 确定维度数量
    n_dims = len(vander_fs)
    # 检查样本点的维度数量是否与给定的函数数量相同
    if n_dims != len(points):
        raise ValueError(
            f"Expected {n_dims} dimensions of sample points, got {len(points)}")
    # 检查阶数的维度数量是否与给定的函数数量相同
    if n_dims != len(degrees):
        raise ValueError(
            f"Expected {n_dims} dimensions of degrees, got {len(degrees)}")
    # 如果没有提供样本点，则无法猜测 dtype 或形状
    if n_dims == 0:
        raise ValueError("Unable to guess a dtype or shape when no points are given")

    # 将所有点转换为相同的形状和类型
    points = tuple(np.asarray(tuple(points)) + 0.0)

    # 为每个维度生成范德蒙德矩阵，将每个维度的最后一个轴放置在输出的独立尾随轴中
    vander_arrays = (
        vander_fs[i](points[i], degrees[i])[(...,) + _nth_slice(i, n_dims)]
        for i in range(n_dims)
    )

    # 我们已经检查过这不是空的，因此不需要 `initial`
    return functools.reduce(operator.mul, vander_arrays)
def _vander_nd_flat(vander_fs, points, degrees):
    """
    Like `_vander_nd`, but flattens the last ``len(degrees)`` axes into a single axis

    Used to implement the public ``<type>vander<n>d`` functions.
    """
    # 调用 `_vander_nd` 函数，获取多维 Vandermonde 矩阵
    v = _vander_nd(vander_fs, points, degrees)
    # 将多维矩阵展平为最后 ``len(degrees)`` 个轴合并到一个轴
    return v.reshape(v.shape[:-len(degrees)] + (-1,))


def _fromroots(line_f, mul_f, roots):
    """
    Helper function used to implement the ``<type>fromroots`` functions.

    Parameters
    ----------
    line_f : function(float, float) -> ndarray
        The ``<type>line`` function, such as ``polyline``
    mul_f : function(array_like, array_like) -> ndarray
        The ``<type>mul`` function, such as ``polymul``
    roots
        See the ``<type>fromroots`` functions for more detail
    """
    # 如果 roots 为空，则返回一个包含一个元素的数组
    if len(roots) == 0:
        return np.ones(1)
    else:
        # 将 roots 转换为系列，并进行排序
        [roots] = as_series([roots], trim=False)
        roots.sort()
        # 生成关于每个根的线性函数的列表
        p = [line_f(-r, 1) for r in roots]
        n = len(p)
        while n > 1:
            m, r = divmod(n, 2)
            # 对每对线性函数进行乘法运算
            tmp = [mul_f(p[i], p[i+m]) for i in range(m)]
            if r:
                tmp[0] = mul_f(tmp[0], p[-1])
            p = tmp
            n = m
        # 返回最终的多项式系数数组
        return p[0]


def _valnd(val_f, c, *args):
    """
    Helper function used to implement the ``<type>val<n>d`` functions.

    Parameters
    ----------
    val_f : function(array_like, array_like, tensor: bool) -> array_like
        The ``<type>val`` function, such as ``polyval``
    c, args
        See the ``<type>val<n>d`` functions for more detail
    """
    # 将所有参数转换为 NumPy 数组
    args = [np.asanyarray(a) for a in args]
    shape0 = args[0].shape
    # 检查所有参数的形状是否与第一个参数相同
    if not all((a.shape == shape0 for a in args[1:])):
        if len(args) == 3:
            raise ValueError('x, y, z are incompatible')
        elif len(args) == 2:
            raise ValueError('x, y are incompatible')
        else:
            raise ValueError('ordinates are incompatible')
    it = iter(args)
    x0 = next(it)

    # 对第一个参数使用 val_f 函数
    c = val_f(x0, c)
    # 对剩余的参数依次使用 val_f 函数
    for xi in it:
        c = val_f(xi, c, tensor=False)
    # 返回计算结果
    return c


def _gridnd(val_f, c, *args):
    """
    Helper function used to implement the ``<type>grid<n>d`` functions.

    Parameters
    ----------
    val_f : function(array_like, array_like, tensor: bool) -> array_like
        The ``<type>val`` function, such as ``polyval``
    c, args
        See the ``<type>grid<n>d`` functions for more detail
    """
    # 对所有参数依次使用 val_f 函数
    for xi in args:
        c = val_f(xi, c)
    # 返回计算结果
    return c


def _div(mul_f, c1, c2):
    """
    Helper function used to implement the ``<type>div`` functions.

    Implementation uses repeated subtraction of c2 multiplied by the nth basis.
    For some polynomial types, a more efficient approach may be possible.

    Parameters
    ----------
    mul_f : function(array_like, array_like) -> array_like
        The ``<type>mul`` function, such as ``polymul``
    c1, c2
        See the ``<type>div`` functions for more detail
    """
    # c1, c2 are trimmed copies
    # 此处是对 ``<type>div`` 函数的简要描述和提示
    # 将输入的两个参数 c1 和 c2 转换为序列（数组或类似结构），分别存储到 c1 和 c2 中
    [c1, c2] = as_series([c1, c2])
    # 检查 c2 数组的最后一个元素是否为 0，如果是则抛出 ZeroDivisionError 异常
    if c2[-1] == 0:
        raise ZeroDivisionError()

    # 计算 c1 和 c2 的长度，分别存储到 lc1 和 lc2 中
    lc1 = len(c1)
    lc2 = len(c2)
    
    # 如果 c1 的长度小于 c2 的长度，返回一个以 c1[0] 为元素、长度为 0 的数组，并返回 c1
    if lc1 < lc2:
        return c1[:1]*0, c1
    # 如果 c2 的长度为 1，返回 c1 除以 c2 的最后一个元素，以及一个以 c1[0] 为元素、长度为 0 的数组
    elif lc2 == 1:
        return c1/c2[-1], c1[:1]*0
    else:
        # 创建一个类型与 c1 相同的空数组 quo，长度为 lc1 - lc2 + 1
        quo = np.empty(lc1 - lc2 + 1, dtype=c1.dtype)
        # 将 rem 初始化为 c1
        rem = c1
        # 从 lc1 - lc2 循环到 0（包括 0）
        for i in range(lc1 - lc2, - 1, -1):
            # 创建一个多项式 p，其中 i 个 0 后接 1，然后与 c2 相乘
            p = mul_f([0]*i + [1], c2)
            # 计算余数 rem 的最后一个元素与 p 的最后一个元素的除法商 q
            q = rem[-1]/p[-1]
            # 更新 rem 为 rem 去除最后一个元素减去 q 乘以 p 去除最后一个元素
            rem = rem[:-1] - q*p[:-1]
            # 将 q 存储到 quo 的第 i 个位置
            quo[i] = q
        # 返回商 quo 和修剪后的余数 trimseq(rem)
        return quo, trimseq(rem)
def _add(c1, c2):
    """ Helper function used to implement the ``<type>add`` functions. """
    # c1, c2 are trimmed copies
    [c1, c2] = as_series([c1, c2])  # 调用as_series函数，将c1和c2转换为Series类型，并进行必要的修剪
    if len(c1) > len(c2):
        c1[:c2.size] += c2  # 如果c1长度大于c2，则将c2加到c1的前c2.size部分
        ret = c1
    else:
        c2[:c1.size] += c1  # 否则，将c1加到c2的前c1.size部分
        ret = c2
    return trimseq(ret)  # 返回修剪过的结果序列


def _sub(c1, c2):
    """ Helper function used to implement the ``<type>sub`` functions. """
    # c1, c2 are trimmed copies
    [c1, c2] = as_series([c1, c2])  # 调用as_series函数，将c1和c2转换为Series类型，并进行必要的修剪
    if len(c1) > len(c2):
        c1[:c2.size] -= c2  # 如果c1长度大于c2，则从c1的前c2.size部分减去c2
        ret = c1
    else:
        c2 = -c2  # 否则，将c2取反
        c2[:c1.size] += c1  # 然后将c1加到c2的前c1.size部分
        ret = c2
    return trimseq(ret)  # 返回修剪过的结果序列


def _fit(vander_f, x, y, deg, rcond=None, full=False, w=None):
    """
    Helper function used to implement the ``<type>fit`` functions.

    Parameters
    ----------
    vander_f : function(array_like, int) -> ndarray
        The 1d vander function, such as ``polyvander``
    x, y, deg
        See the ``<type>fit`` functions for more detail
    """
    x = np.asarray(x) + 0.0  # 将x转换为NumPy数组，并确保是浮点型
    y = np.asarray(y) + 0.0  # 将y转换为NumPy数组，并确保是浮点型
    deg = np.asarray(deg)  # 将deg转换为NumPy数组

    # check arguments.
    if deg.ndim > 1 or deg.dtype.kind not in 'iu' or deg.size == 0:
        raise TypeError("deg must be an int or non-empty 1-D array of int")  # 检查deg的类型和形状是否符合要求
    if deg.min() < 0:
        raise ValueError("expected deg >= 0")  # 检查deg是否都大于等于0
    if x.ndim != 1:
        raise TypeError("expected 1D vector for x")  # 检查x是否是一维数组
    if x.size == 0:
        raise TypeError("expected non-empty vector for x")  # 检查x是否为空
    if y.ndim < 1 or y.ndim > 2:
        raise TypeError("expected 1D or 2D array for y")  # 检查y是否是一维或二维数组
    if len(x) != len(y):
        raise TypeError("expected x and y to have same length")  # 检查x和y的长度是否相同

    if deg.ndim == 0:
        lmax = deg
        order = lmax + 1
        van = vander_f(x, lmax)  # 调用vander_f函数生成Vandermonde矩阵
    else:
        deg = np.sort(deg)
        lmax = deg[-1]
        order = len(deg)
        van = vander_f(x, lmax)[:, deg]  # 对于多项式阶数按deg给出的顺序进行排序，并生成对应的Vandermonde矩阵

    # set up the least squares matrices in transposed form
    lhs = van.T  # 左侧矩阵为Vandermonde矩阵的转置
    rhs = y.T  # 右侧矩阵为y的转置
    if w is not None:
        w = np.asarray(w) + 0.0  # 将权重w转换为NumPy数组，并确保是浮点型
        if w.ndim != 1:
            raise TypeError("expected 1D vector for w")  # 检查w是否是一维数组
        if len(x) != len(w):
            raise TypeError("expected x and w to have same length")  # 检查x和w的长度是否相同
        # apply weights. Don't use inplace operations as they
        # can cause problems with NA.
        lhs = lhs * w  # 对左侧矩阵应用权重w
        rhs = rhs * w  # 对右侧矩阵应用权重w

    # set rcond
    if rcond is None:
        rcond = len(x) * np.finfo(x.dtype).eps  # 如果rcond未指定，则设为x数组中元素类型的机器精度的倍数

    # Determine the norms of the design matrix columns.
    if issubclass(lhs.dtype.type, np.complexfloating):
        scl = np.sqrt((np.square(lhs.real) + np.square(lhs.imag)).sum(1))  # 计算复数类型左侧矩阵各列的范数
    else:
        scl = np.sqrt(np.square(lhs).sum(1))  # 计算实数类型左侧矩阵各列的范数
    scl[scl == 0] = 1  # 将范数为0的元素设为1，避免除以0

    # Solve the least squares problem.
    c, resids, rank, s = np.linalg.lstsq(lhs.T / scl, rhs.T, rcond)  # 解最小二乘问题，得到系数c
    c = (c.T / scl).T  # 将系数c按照左侧矩阵列的范数进行归一化

    # Expand c to include non-fitted coefficients which are set to zero
    # 检查 deg 的维度是否大于 0
    if deg.ndim > 0:
        # 如果 c 的维度为 2，则创建一个零矩阵 cc，形状为 (lmax+1, c.shape[1])，数据类型与 c 相同
        if c.ndim == 2:
            cc = np.zeros((lmax+1, c.shape[1]), dtype=c.dtype)
        else:
            # 否则，创建一个零数组 cc，形状为 (lmax+1)，数据类型与 c 相同
            cc = np.zeros(lmax+1, dtype=c.dtype)
        # 将 cc 中索引为 deg 的位置赋值为 c
        cc[deg] = c
        # 将 c 更新为 cc
        c = cc

    # 在秩降低时发出警告
    if rank != order and not full:
        # 创建警告信息字符串
        msg = "The fit may be poorly conditioned"
        # 发出警告，使用 RankWarning 类，设置堆栈深度为 2
        warnings.warn(msg, RankWarning, stacklevel=2)

    # 如果设置为返回完整结果
    if full:
        # 返回 c 和包含剩余项、秩、奇异值、rcond 值的列表
        return c, [resids, rank, s, rcond]
    else:
        # 否则，只返回 c
        return c
# 实现 ``<type>pow`` 函数的辅助函数，用于计算给定系列的幂次方
def _pow(mul_f, c, pow, maxpower):
    """
    Helper function used to implement the ``<type>pow`` functions.

    Parameters
    ----------
    mul_f : function(array_like, array_like) -> ndarray
        用于乘法操作的函数，如 ``polymul``
    c : array_like
        系列系数的一维数组
    pow, maxpower
        ``<type>pow`` 函数的参数，详见其具体说明
    """
    # c 是裁剪后的副本
    [c] = as_series([c])
    power = int(pow)
    # 检查幂次必须为非负整数
    if power != pow or power < 0:
        raise ValueError("Power must be a non-negative integer.")
    # 如果 maxpower 不为 None，则检查幂次不能超过 maxpower
    elif maxpower is not None and power > maxpower:
        raise ValueError("Power is too large")
    elif power == 0:
        # 幂次为 0 时，返回系数为 1 的数组
        return np.array([1], dtype=c.dtype)
    elif power == 1:
        # 幂次为 1 时，直接返回系列 c
        return c
    else:
        # 否则，通过循环计算幂次的结果
        # 可以通过使用二进制的幂次计算来提高效率
        prd = c
        for i in range(2, power + 1):
            prd = mul_f(prd, c)
        return prd


# 类似于 `operator.index`，但当传入不正确类型时会抛出自定义异常的函数
def _as_int(x, desc):
    """
    Like `operator.index`, but emits a custom exception when passed an 
    incorrect type

    Parameters
    ----------
    x : int-like
        要解释为整数的值
    desc : str
        错误消息中包含的描述信息

    Raises
    ------
    TypeError : 若 x 是浮点数或非数值类型
    """
    try:
        return operator.index(x)
    except TypeError as e:
        raise TypeError(f"{desc} must be an integer, received {x}") from e


# 格式化浮点数 x，可选择是否使用括号包裹
def format_float(x, parens=False):
    if not np.issubdtype(type(x), np.floating):
        # 若 x 不是浮点数类型，则直接返回其字符串表示
        return str(x)

    opts = np.get_printoptions()

    if np.isnan(x):
        # 若 x 是 NaN，则返回指定的 NaN 字符串
        return opts['nanstr']
    elif np.isinf(x):
        # 若 x 是无穷大，则返回指定的无穷大字符串
        return opts['infstr']

    exp_format = False
    if x != 0:
        a = np.abs(x)
        # 根据浮点数的大小和精度设置，决定是否使用科学计数法
        if a >= 1.e8 or a < 10**min(0, -(opts['precision']-1)//2):
            exp_format = True

    trim, unique = '0', True
    if opts['floatmode'] == 'fixed':
        trim, unique = 'k', False

    if exp_format:
        # 若使用科学计数法，则调用 dragon4_scientific 函数进行格式化
        s = dragon4_scientific(x, precision=opts['precision'],
                               unique=unique, trim=trim, 
                               sign=opts['sign'] == '+')
        if parens:
            # 若需要用括号包裹，则添加括号
            s = '(' + s + ')'
    else:
        # 若使用定点表示法，则调用 dragon4_positional 函数进行格式化
        s = dragon4_positional(x, precision=opts['precision'],
                               fractional=True,
                               unique=unique, trim=trim,
                               sign=opts['sign'] == '+')
    return s
```