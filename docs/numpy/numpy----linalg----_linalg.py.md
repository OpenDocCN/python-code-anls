# `.\numpy\numpy\linalg\_linalg.py`

```py
"""Lite version of scipy.linalg.

Notes
-----
This module is a lite version of the linalg.py module in SciPy which
contains high-level Python interface to the LAPACK library.  The lite
version only accesses the following LAPACK functions: dgesv, zgesv,
dgeev, zgeev, dgesdd, zgesdd, dgelsd, zgelsd, dsyevd, zheevd, dgetrf,
zgetrf, dpotrf, zpotrf, dgeqrf, zgeqrf, zungqr, dorgqr.
"""

__all__ = ['matrix_power', 'solve', 'tensorsolve', 'tensorinv', 'inv',
           'cholesky', 'eigvals', 'eigvalsh', 'pinv', 'slogdet', 'det',
           'svd', 'svdvals', 'eig', 'eigh', 'lstsq', 'norm', 'qr', 'cond',
           'matrix_rank', 'LinAlgError', 'multi_dot', 'trace', 'diagonal',
           'cross', 'outer', 'tensordot', 'matmul', 'matrix_transpose',
           'matrix_norm', 'vector_norm', 'vecdot']

# 导入必要的模块和函数
import functools  # 导入 functools 模块，用于创建偏函数
import operator  # 导入 operator 模块，用于操作符函数
import warnings  # 导入 warnings 模块，用于警告处理
from typing import NamedTuple, Any  # 导入 NamedTuple 和 Any 类型提示

# 导入 numpy 相关模块和函数
from numpy._utils import set_module  # 导入 set_module 函数，设置模块
from numpy._core import (
    array, asarray, zeros, empty, empty_like, intc, single, double,
    csingle, cdouble, inexact, complexfloating, newaxis, all, inf, dot,
    add, multiply, sqrt, sum, isfinite, finfo, errstate, moveaxis, amin,
    amax, prod, abs, atleast_2d, intp, asanyarray, object_, matmul,
    swapaxes, divide, count_nonzero, isnan, sign, argsort, sort,
    reciprocal, overrides, diagonal as _core_diagonal, trace as _core_trace,
    cross as _core_cross, outer as _core_outer, tensordot as _core_tensordot,
    matmul as _core_matmul, matrix_transpose as _core_matrix_transpose,
    transpose as _core_transpose, vecdot as _core_vecdot,
)
from numpy._globals import _NoValue  # 导入 _NoValue 对象
from numpy.lib._twodim_base_impl import triu, eye  # 导入 triu 和 eye 函数
from numpy.lib.array_utils import normalize_axis_index, normalize_axis_tuple  # 导入索引和轴规范化函数
from numpy.linalg import _umath_linalg  # 导入 _umath_linalg 函数

# 导入类型提示
from numpy._typing import NDArray

# 定义几个命名元组用于返回结果
class EigResult(NamedTuple):
    eigenvalues: NDArray[Any]
    eigenvectors: NDArray[Any]

class EighResult(NamedTuple):
    eigenvalues: NDArray[Any]
    eigenvectors: NDArray[Any]

class QRResult(NamedTuple):
    Q: NDArray[Any]
    R: NDArray[Any]

class SlogdetResult(NamedTuple):
    sign: NDArray[Any]
    logabsdet: NDArray[Any]

class SVDResult(NamedTuple):
    U: NDArray[Any]
    S: NDArray[Any]
    Vh: NDArray[Any]

# 设置数组函数分发
array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy.linalg'
)

# 设置 Fortran 整数类型
fortran_int = intc

# 设置 numpy.linalg 的模块
@set_module('numpy.linalg')
class LinAlgError(ValueError):
    """
    Generic Python-exception-derived object raised by linalg functions.

    General purpose exception class, derived from Python's ValueError
    class, programmatically raised in linalg functions when a Linear
    Algebra-related condition would prevent further correct execution of the
    function.

    Parameters
    ----------
    None

    Examples
    --------
    >>> from numpy import linalg as LA
    >>> LA.inv(np.zeros((2,2)))

    """
    # 这部分代码是一个例外追踪（Traceback），显示了程序执行过程中发生的异常情况
    # 最后一行显示了具体的异常类型：LinAlgError，表示线性代数操作中出现了异常
    # 异常发生的原因是矩阵是奇异矩阵，无法进行逆矩阵操作
    # 这部分代码没有实际的可注释内容，只是异常的追踪记录
# 抛出一个线性代数错误，指示矩阵是奇异的（不可逆的）
def _raise_linalgerror_singular(err, flag):
    raise LinAlgError("Singular matrix")

# 抛出一个线性代数错误，指示矩阵不是正定的
def _raise_linalgerror_nonposdef(err, flag):
    raise LinAlgError("Matrix is not positive definite")

# 抛出一个线性代数错误，指示特征值没有收敛
def _raise_linalgerror_eigenvalues_nonconvergence(err, flag):
    raise LinAlgError("Eigenvalues did not converge")

# 抛出一个线性代数错误，指示奇异值分解没有收敛
def _raise_linalgerror_svd_nonconvergence(err, flag):
    raise LinAlgError("SVD did not converge")

# 抛出一个线性代数错误，指示在线性最小二乘中奇异值分解没有收敛
def _raise_linalgerror_lstsq(err, flag):
    raise LinAlgError("SVD did not converge in Linear Least Squares")

# 抛出一个线性代数错误，指示在QR分解时发现了不正确的参数
def _raise_linalgerror_qr(err, flag):
    raise LinAlgError("Incorrect argument found while performing "
                      "QR factorization")


# 将输入数组转换为数组对象，保持数组的数组包装
def _makearray(a):
    new = asarray(a)
    wrap = getattr(a, "__array_wrap__", new.__array_wrap__)
    return new, wrap

# 检查类型是否为复数类型
def isComplexType(t):
    return issubclass(t, complexfloating)


# 实数类型映射表，将特定类型映射到对应的实数类型
_real_types_map = {single: single,
                   double: double,
                   csingle: single,
                   cdouble: double}

# 复数类型映射表，将特定类型映射到对应的复数类型
_complex_types_map = {single: csingle,
                      double: cdouble,
                      csingle: csingle,
                      cdouble: cdouble}

# 根据输入类型返回对应的实数类型，默认为双精度
def _realType(t, default=double):
    return _real_types_map.get(t, default)

# 根据输入类型返回对应的复数类型，默认为双精度复数
def _complexType(t, default=cdouble):
    return _complex_types_map.get(t, default)

# 确定多个数组中的公共类型，如果涉及复数则使用较高精度（总是双精度或双精度复数）
def _commonType(*arrays):
    # 在精简版本中，总是使用较高精度（双精度或双精度复数）
    result_type = single
    is_complex = False
    for a in arrays:
        type_ = a.dtype.type
        if issubclass(type_, inexact):
            if isComplexType(type_):
                is_complex = True
            rt = _realType(type_, default=None)
            if rt is double:
                result_type = double
            elif rt is None:
                # 不支持的浮点类型标量
                raise TypeError("array type %s is unsupported in linalg" %
                        (a.dtype.name,))
        else:
            result_type = double
    if is_complex:
        result_type = _complex_types_map[result_type]
        return cdouble, result_type
    else:
        return double, result_type


# 将多个数组转换为本机字节顺序
def _to_native_byte_order(*arrays):
    ret = []
    for arr in arrays:
        if arr.dtype.byteorder not in ('=', '|'):
            ret.append(asarray(arr, dtype=arr.dtype.newbyteorder('=')))
        else:
            ret.append(arr)
    if len(ret) == 1:
        return ret[0]
    else:
        return ret


# 断言多个数组都是二维的
def _assert_2d(*arrays):
    for a in arrays:
        if a.ndim != 2:
            raise LinAlgError('%d-dimensional array given. Array must be '
                    'two-dimensional' % a.ndim)

# 断言多个数组都至少是二维的
def _assert_stacked_2d(*arrays):
    for a in arrays:
        if a.ndim < 2:
            raise LinAlgError('%d-dimensional array given. Array must be '
                    'at least two-dimensional' % a.ndim)

# 断言多个数组都是方阵
def _assert_stacked_square(*arrays):
    # 省略部分，暂未提供完整代码
    # 遍历数组列表 `arrays`
    for a in arrays:
        # 获取数组 `a` 的最后两个维度的形状
        m, n = a.shape[-2:]
        # 检查最后两个维度是否相等，如果不相等则引发线性代数错误异常
        if m != n:
            raise LinAlgError('Last 2 dimensions of the array must be square')
# 确保所有输入数组不包含无穷大或NaN值
def _assert_finite(*arrays):
    for a in arrays:
        if not isfinite(a).all():
            raise LinAlgError("Array must not contain infs or NaNs")

# 检查二维数组是否为空
def _is_empty_2d(arr):
    # 首先检查数组的大小以提高效率
    return arr.size == 0 and prod(arr.shape[-2:]) == 0

# 对输入数组进行转置操作，仅交换最后两个轴
def transpose(a):
    """
    Transpose each matrix in a stack of matrices.

    Unlike np.transpose, this only swaps the last two axes, rather than all of
    them

    Parameters
    ----------
    a : (...,M,N) array_like

    Returns
    -------
    aT : (...,N,M) ndarray
    """
    return swapaxes(a, -1, -2)

# 解张量方程 a x = b 中的 x
def tensorsolve(a, b, axes=None):
    """
    Solve the tensor equation ``a x = b`` for x.

    It is assumed that all indices of `x` are summed over in the product,
    together with the rightmost indices of `a`, as is done in, for example,
    ``tensordot(a, x, axes=x.ndim)``.

    Parameters
    ----------
    a : array_like
        Coefficient tensor, of shape ``b.shape + Q``. `Q`, a tuple, equals
        the shape of that sub-tensor of `a` consisting of the appropriate
        number of its rightmost indices, and must be such that
        ``prod(Q) == prod(b.shape)`` (in which sense `a` is said to be
        'square').
    b : array_like
        Right-hand tensor, which can be of any shape.
    axes : tuple of ints, optional
        Axes in `a` to reorder to the right, before inversion.
        If None (default), no reordering is done.

    Returns
    -------
    x : ndarray, shape Q

    Raises
    ------
    LinAlgError
        If `a` is singular or not 'square' (in the above sense).

    See Also
    --------
    numpy.tensordot, tensorinv, numpy.einsum

    Examples
    --------
    >>> a = np.eye(2*3*4)
    >>> a.shape = (2*3, 4, 2, 3, 4)
    >>> rng = np.random.default_rng()
    >>> b = rng.normal(size=(2*3, 4))
    >>> x = np.linalg.tensorsolve(a, b)
    >>> x.shape
    (2, 3, 4)
    >>> np.allclose(np.tensordot(a, x, axes=3), b)
    True

    """
    a, wrap = _makearray(a)
    b = asarray(b)
    an = a.ndim

    if axes is not None:
        allaxes = list(range(0, an))
        for k in axes:
            allaxes.remove(k)
            allaxes.insert(an, k)
        a = a.transpose(allaxes)

    oldshape = a.shape[-(an-b.ndim):]
    prod = 1
    for k in oldshape:
        prod *= k

    if a.size != prod ** 2:
        raise LinAlgError(
            "Input arrays must satisfy the requirement \
            prod(a.shape[b.ndim:]) == prod(a.shape[:b.ndim])"
        )

    a = a.reshape(prod, prod)
    b = b.ravel()
    res = wrap(solve(a, b))
    res.shape = oldshape
    return res

# 解线性矩阵方程或线性标量方程组
def solve(a, b):
    """
    Solve a linear matrix equation, or system of linear scalar equations.

    """
    a, wrap = _makearray(a)
    b = asarray(b)
    an = a.ndim

    if b.ndim == 0:
        b = b[newaxis]

    _assertRankAtLeast2(a)

    b, wrap = _makearray(b)

    if len(a.shape) != 2 or a.shape[0] != a.shape[1]:
        raise LinAlgError('a must be square')

    if len(a.shape) != len(b.shape):
        raise LinAlgError('Incompatible dimensions')

    extobj = get_linalg_error_extobj(
        _raise_linalgerror_singular)
    retval = _umath_linalg.solve(a, b, signature=signature,
                                 extobj=extobj)
    return wrap(retval)
    # 计算线性矩阵方程 ax = b 的精确解 x。
    # 参数：
    # a : (..., M, M) array_like
    #     系数矩阵。
    # b : {(M,), (..., M, K)}, array_like
    #     纵坐标或“因变量”值。
    # 返回：
    # x : {(..., M,), (..., M, K)} ndarray
    #     系统 a x = b 的解。如果 b 的形状是 (M,)，返回的形状是 (..., M)；如果 b 的形状是 (..., M, K)，返回的形状是 (..., M, K)，
    #     其中 "..." 部分在 a 和 b 之间广播。
    # 抛出异常：
    # LinAlgError
    #     如果 `a` 是奇异的或不是方阵。
    # 参见：
    # scipy.linalg.solve : SciPy 中类似的函数。
    # 注意：
    # 广播规则适用，请参阅 `numpy.linalg` 文档获取详细信息。
    # 解是使用 LAPACK 例程 ``_gesv`` 计算的。
    # `a` 必须是方阵且全秩的，即所有行（或等效地，所有列）必须线性独立；如果不是这样，请使用 `lstsq` 获取系统/方程的最小二乘最佳“解决方案”。
    # 引用：
    # [1] G. Strang, *Linear Algebra and Its Applications*, 2nd Ed., Orlando,
    #        FL, Academic Press, Inc., 1980, pg. 22.
    # 示例：
    # 解方程组：
    # ``x0 + 2 * x1 = 1`` 和
    # ``3 * x0 + 5 * x1 = 2``：
    # 
    # >>> a = np.array([[1, 2], [3, 5]])
    # >>> b = np.array([1, 2])
    # >>> x = np.linalg.solve(a, b)
    # >>> x
    # array([-1.,  1.])
    # 
    # 检查解是否正确：
    # 
    # >>> np.allclose(np.dot(a, x), b)
    # True
    def solve(a, b):
        # 将 a 转换为数组，并返回第一维度的尺寸和 wrap 函数
        a, _ = _makearray(a)
        # 断言 a 是二维堆叠数组
        _assert_stacked_2d(a)
        # 断言 a 是方阵
        _assert_stacked_square(a)
        # 将 b 转换为数组，并返回第一维度的尺寸和 wrap 函数
        b, wrap = _makearray(b)
        # 获取 a 和 b 的公共类型 t，以及结果类型 result_t
        t, result_t = _commonType(a, b)
        
        # 如果 b 的维度是 1，选择 _umath_linalg.solve1；否则选择 _umath_linalg.solve
        if b.ndim == 1:
            gufunc = _umath_linalg.solve1
        else:
            gufunc = _umath_linalg.solve
        
        # 签名是 'DD->D' 如果 t 是复数类型，否则是 'dd->d'
        signature = 'DD->D' if isComplexType(t) else 'dd->d'
        
        # 使用 errstate 设置错误处理方式，调用 _raise_linalgerror_singular 抛出 LinAlgError
        with errstate(call=_raise_linalgerror_singular, invalid='call',
                      over='ignore', divide='ignore', under='ignore'):
            # 调用 gufunc 计算解
            r = gufunc(a, b, signature=signature)
        
        # 返回结果，转换为 result_t 类型，不复制
        return wrap(r.astype(result_t, copy=False))
# 定义函数 _tensorinv_dispatcher，返回元组 (a,)
def _tensorinv_dispatcher(a, ind=None):
    return (a,)


# 使用 array_function_dispatch 装饰器，将 _tensorinv_dispatcher 作为调度函数
@array_function_dispatch(_tensorinv_dispatcher)
def tensorinv(a, ind=2):
    """
    Compute the 'inverse' of an N-dimensional array.

    The result is an inverse for `a` relative to the tensordot operation
    ``tensordot(a, b, ind)``, i. e., up to floating-point accuracy,
    ``tensordot(tensorinv(a), a, ind)`` is the "identity" tensor for the
    tensordot operation.

    Parameters
    ----------
    a : array_like
        Tensor to 'invert'. Its shape must be 'square', i. e.,
        ``prod(a.shape[:ind]) == prod(a.shape[ind:])``.
    ind : int, optional
        Number of first indices that are involved in the inverse sum.
        Must be a positive integer, default is 2.

    Returns
    -------
    b : ndarray
        `a`'s tensordot inverse, shape ``a.shape[ind:] + a.shape[:ind]``.

    Raises
    ------
    LinAlgError
        If `a` is singular or not 'square' (in the above sense).

    See Also
    --------
    numpy.tensordot, tensorsolve

    Examples
    --------
    >>> a = np.eye(4*6)
    >>> a.shape = (4, 6, 8, 3)
    >>> ainv = np.linalg.tensorinv(a, ind=2)
    >>> ainv.shape
    (8, 3, 4, 6)
    >>> rng = np.random.default_rng()
    >>> b = rng.normal(size=(4, 6))
    >>> np.allclose(np.tensordot(ainv, b), np.linalg.tensorsolve(a, b))
    True

    >>> a = np.eye(4*6)
    >>> a.shape = (24, 8, 3)
    >>> ainv = np.linalg.tensorinv(a, ind=1)
    >>> ainv.shape
    (8, 3, 24)
    >>> rng = np.random.default_rng()
    >>> b = rng.normal(size=24)
    >>> np.allclose(np.tensordot(ainv, b, 1), np.linalg.tensorsolve(a, b))
    True

    """
    # 将输入数组 a 转换为 ndarray
    a = asarray(a)
    # 记录原始形状
    oldshape = a.shape
    # 初始化乘积变量为 1
    prod = 1
    # 如果 ind 大于 0，则计算逆变换后的形状
    if ind > 0:
        invshape = oldshape[ind:] + oldshape[:ind]
        # 计算乘积，乘以 oldshape[ind:] 中的每个值
        for k in oldshape[ind:]:
            prod *= k
    else:
        # 如果 ind 不合法，抛出 ValueError
        raise ValueError("Invalid ind argument.")
    # 将 a 重塑为 (prod, -1) 的形状
    a = a.reshape(prod, -1)
    # 计算 a 的逆矩阵
    ia = inv(a)
    # 将 ia 重塑为 invshape 形状并返回
    return ia.reshape(*invshape)


# 矩阵求逆

# 定义函数 _unary_dispatcher，返回元组 (a,)
def _unary_dispatcher(a):
    return (a,)


# 使用 array_function_dispatch 装饰器，将 _unary_dispatcher 作为调度函数
@array_function_dispatch(_unary_dispatcher)
def inv(a):
    """
    Compute the inverse of a matrix.

    Given a square matrix `a`, return the matrix `ainv` satisfying
    ``a @ ainv = ainv @ a = eye(a.shape[0])``.

    Parameters
    ----------
    a : (..., M, M) array_like
        Matrix to be inverted.

    Returns
    -------
    ainv : (..., M, M) ndarray or matrix
        Inverse of the matrix `a`.

    Raises
    ------
    LinAlgError
        If `a` is not square or inversion fails.

    See Also
    --------
    scipy.linalg.inv : Similar function in SciPy.
    numpy.linalg.cond : Compute the condition number of a matrix.
    numpy.linalg.svd : Compute the singular value decomposition of a matrix.

    Notes
    -----

    .. versionadded:: 1.8.0

    Broadcasting rules apply, see the `numpy.linalg` documentation for
    details.

    If `a` is detected to be singular, a `LinAlgError` is raised. If `a` is
    """
    # 将输入数组 a 转换为 ndarray
    a = asarray(a)
    # 返回 a 的逆矩阵
    return _umath_linalg.inv(a)
    """
    Compute the inverse of a matrix `a` using NumPy's linear algebra functions.
    
    This function computes the inverse of the input matrix `a`. If `a` is ill-conditioned,
    it may not raise a `LinAlgError`, but the computed results may be inaccurate due to
    floating-point errors.
    
    Parameters
    ----------
    a : array_like
        Input matrix to be inverted.
    
    Returns
    -------
    array_like
        Inverse of the input matrix `a`.
    
    Raises
    ------
    LinAlgError
        If `a` is singular or not square.
    
    Notes
    -----
    This function wraps around NumPy's `inv` function from `numpy.linalg`, which computes
    the matrix inverse. It handles complex type matrices (`D->D` signature) differently
    from real type matrices (`d->d` signature).
    
    References
    ----------
    .. [1] Wikipedia, "Condition number",
           https://en.wikipedia.org/wiki/Condition_number
    
    Examples
    --------
    >>> from numpy.linalg import inv
    >>> a = np.array([[1., 2.], [3., 4.]])
    >>> ainv = inv(a)
    >>> np.allclose(a @ ainv, np.eye(2))
    True
    >>> np.allclose(ainv @ a, np.eye(2))
    True
    
    If `a` is a matrix object, then the return value is a matrix as well:
    
    >>> ainv = inv(np.matrix(a))
    >>> ainv
    matrix([[-2. ,  1. ],
            [ 1.5, -0.5]])
    
    Inverses of several matrices can be computed at once:
    
    >>> a = np.array([[[1., 2.], [3., 4.]], [[1, 3], [3, 5]]])
    >>> inv(a)
    array([[[-2.  ,  1.  ],
            [ 1.5 , -0.5 ]],
           [[-1.25,  0.75],
            [ 0.75, -0.25]]])
    
    If a matrix is close to singular, the computed inverse may not satisfy
    ``a @ ainv = ainv @ a = eye(a.shape[0])`` even if a `LinAlgError`
    is not raised:
    
    >>> a = np.array([[2,4,6],[2,0,2],[6,8,14]])
    >>> inv(a)  # No errors raised
    array([[-1.12589991e+15, -5.62949953e+14,  5.62949953e+14],
           [-1.12589991e+15, -5.62949953e+14,  5.62949953e+14],
           [ 1.12589991e+15,  5.62949953e+14, -5.62949953e+14]])
    >>> a @ inv(a)
    array([[ 0.   , -0.5  ,  0.   ],  # may vary
           [-0.5  ,  0.625,  0.25 ],
           [ 0.   ,  0.   ,  1.   ]])
    
    To detect ill-conditioned matrices, you can use `numpy.linalg.cond` to
    compute its *condition number* [1]_. The larger the condition number, the
    more ill-conditioned the matrix is. As a rule of thumb, if the condition
    number ``cond(a) = 10**k``, then you may lose up to ``k`` digits of
    accuracy on top of what would be lost to the numerical method due to loss
    of precision from arithmetic methods.
    
    >>> from numpy.linalg import cond
    >>> cond(a)
    np.float64(8.659885634118668e+17)  # may vary
    
    It is also possible to detect ill-conditioning by inspecting the matrix's
    singular values directly. The ratio between the largest and the smallest
    singular value is the condition number:
    
    >>> from numpy.linalg import svd
    >>> sigma = svd(a, compute_uv=False)  # Do not compute singular vectors
    >>> sigma.max()/sigma.min()
    8.659885634118668e+17  # may vary
    """
    a, wrap = _makearray(a)  # Convert input `a` to an array and assign to `a`, `wrap`
    _assert_stacked_2d(a)  # Check if `a` is 2-dimensional stacked array
    _assert_stacked_square(a)  # Check if `a` is square
    t, result_t = _commonType(a)  # Determine common data type `t` and result type `result_t`
    
    signature = 'D->D' if isComplexType(t) else 'd->d'  # Define signature for complex or real types
    with errstate(call=_raise_linalgerror_singular, invalid='call',
                  over='ignore', divide='ignore', under='ignore'):
        ainv = _umath_linalg.inv(a, signature=signature)  # Compute matrix inverse with error handling
    return wrap(ainv.astype(result_t, copy=False))  # Return inverse matrix converted to `result_t`
# 定义一个用于矩阵幂运算分发的函数，始终返回元组 (a,)
def _matrix_power_dispatcher(a, n):
    return (a,)

# 使用装饰器将 _matrix_power_dispatcher 注册为 matrix_power 函数的分发函数
@array_function_dispatch(_matrix_power_dispatcher)
def matrix_power(a, n):
    """
    Raise a square matrix to the (integer) power `n`.

    For positive integers `n`, the power is computed by repeated matrix
    squarings and matrix multiplications. If ``n == 0``, the identity matrix
    of the same shape as M is returned. If ``n < 0``, the inverse
    is computed and then raised to the ``abs(n)``.

    .. note:: Stacks of object matrices are not currently supported.

    Parameters
    ----------
    a : (..., M, M) array_like
        Matrix to be "powered".
    n : int
        The exponent can be any integer or long integer, positive,
        negative, or zero.

    Returns
    -------
    a**n : (..., M, M) ndarray or matrix object
        The return value is the same shape and type as `M`;
        if the exponent is positive or zero then the type of the
        elements is the same as those of `M`. If the exponent is
        negative the elements are floating-point.

    Raises
    ------
    LinAlgError
        For matrices that are not square or that (for negative powers) cannot
        be inverted numerically.

    Examples
    --------
    >>> from numpy.linalg import matrix_power
    >>> i = np.array([[0, 1], [-1, 0]]) # matrix equiv. of the imaginary unit
    >>> matrix_power(i, 3) # should = -i
    array([[ 0, -1],
           [ 1,  0]])
    >>> matrix_power(i, 0)
    array([[1, 0],
           [0, 1]])
    >>> matrix_power(i, -3) # should = 1/(-i) = i, but w/ f.p. elements
    array([[ 0.,  1.],
           [-1.,  0.]])

    Somewhat more sophisticated example

    >>> q = np.zeros((4, 4))
    >>> q[0:2, 0:2] = -i
    >>> q[2:4, 2:4] = i
    >>> q # one of the three quaternion units not equal to 1
    array([[ 0., -1.,  0.,  0.],
           [ 1.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  1.],
           [ 0.,  0., -1.,  0.]])
    >>> matrix_power(q, 2) # = -np.eye(4)
    array([[-1.,  0.,  0.,  0.],
           [ 0., -1.,  0.,  0.],
           [ 0.,  0., -1.,  0.],
           [ 0.,  0.,  0., -1.]])

    """
    # 将输入矩阵转换为 ndarray
    a = asanyarray(a)
    # 检查输入矩阵是否为堆叠的二维数组
    _assert_stacked_2d(a)
    # 检查输入矩阵是否为方阵
    _assert_stacked_square(a)

    try:
        # 尝试将 n 转换为整数
        n = operator.index(n)
    except TypeError as e:
        # 抛出类型错误异常，说明指数必须是整数
        raise TypeError("exponent must be an integer") from e

    # 对于非对象数组，使用 matmul 函数；对象数组不支持当前实现的 matmul 使用 einsum
    if a.dtype != object:
        fmatmul = matmul
    elif a.ndim == 2:
        fmatmul = dot
    else:
        # 抛出未实现错误，说明暂不支持对象数组的矩阵幂运算
        raise NotImplementedError(
            "matrix_power not supported for stacks of object arrays")

    # 若 n 为 0，则返回与 a 相同形状的空数组，并赋予单位矩阵值
    if n == 0:
        a = empty_like(a)
        a[...] = eye(a.shape[-2], dtype=a.dtype)
        return a

    # 若 n 为负数，则先计算 a 的逆矩阵，并将 n 取绝对值
    elif n < 0:
        a = inv(a)
        n = abs(n)

    # 以下是 n 不同取值时的快捷计算
    if n == 1:
        return a

    elif n == 2:
        return fmatmul(a, a)

    elif n == 3:
        return fmatmul(fmatmul(a, a), a)
    # 使用二进制分解来减少矩阵乘法的次数。
    # 在这里，我们从最低有效位到最高有效位迭代 n 的位，
    # 将 `a` 的幂逐步乘到结果中。
    z = result = None  # 初始化变量 z 和 result 为 None
    while n > 0:  # 当 n 大于 0 时循环执行以下操作
        z = a if z is None else fmatmul(z, z)  # 如果 z 是 None，将 z 设置为 a；否则 z 为 z 自身与自身的矩阵乘积
        n, bit = divmod(n, 2)  # 将 n 除以 2，得到商和余数，分别赋给 n 和 bit
        if bit:  # 如果余数 bit 为真（即为 1）
            result = z if result is None else fmatmul(result, z)  # 如果 result 是 None，将 result 设置为 z；否则 result 为 result 自身与 z 的矩阵乘积

    return result  # 返回计算结果 result
# Cholesky decomposition

# 定义了一个函数 _cholesky_dispatcher，用于分派 Cholesky 分解操作
def _cholesky_dispatcher(a, /, *, upper=None):
    return (a,)

# 使用 array_function_dispatch 装饰器，将 _cholesky_dispatcher 函数注册为 cholesky 函数的分派器
@array_function_dispatch(_cholesky_dispatcher)
# 定义了 Cholesky 分解函数 cholesky
def cholesky(a, /, *, upper=False):
    """
    Cholesky decomposition.

    Return the lower or upper Cholesky decomposition, ``L * L.H`` or
    ``U.H * U``, of the square matrix ``a``, where ``L`` is lower-triangular,
    ``U`` is upper-triangular, and ``.H`` is the conjugate transpose operator
    (which is the ordinary transpose if ``a`` is real-valued). ``a`` must be
    Hermitian (symmetric if real-valued) and positive-definite. No checking is
    performed to verify whether ``a`` is Hermitian or not. In addition, only
    the lower or upper-triangular and diagonal elements of ``a`` are used.
    Only ``L`` or ``U`` is actually returned.

    Parameters
    ----------
    a : (..., M, M) array_like
        Hermitian (symmetric if all elements are real), positive-definite
        input matrix.
    upper : bool
        If ``True``, the result must be the upper-triangular Cholesky factor.
        If ``False``, the result must be the lower-triangular Cholesky factor.
        Default: ``False``.

    Returns
    -------
    L : (..., M, M) array_like
        Lower or upper-triangular Cholesky factor of `a`. Returns a matrix
        object if `a` is a matrix object.

    Raises
    ------
    LinAlgError
       If the decomposition fails, for example, if `a` is not
       positive-definite.

    See Also
    --------
    scipy.linalg.cholesky : Similar function in SciPy.
    scipy.linalg.cholesky_banded : Cholesky decompose a banded Hermitian
                                   positive-definite matrix.
    scipy.linalg.cho_factor : Cholesky decomposition of a matrix, to use in
                              `scipy.linalg.cho_solve`.

    Notes
    -----

    .. versionadded:: 1.8.0

    Broadcasting rules apply, see the `numpy.linalg` documentation for
    details.

    The Cholesky decomposition is often used as a fast way of solving

    .. math:: A \\mathbf{x} = \\mathbf{b}

    (when `A` is both Hermitian/symmetric and positive-definite).

    First, we solve for :math:`\\mathbf{y}` in

    .. math:: L \\mathbf{y} = \\mathbf{b},

    and then for :math:`\\mathbf{x}` in

    .. math:: L^{H} \\mathbf{x} = \\mathbf{y}.

    Examples
    --------
    >>> A = np.array([[1,-2j],[2j,5]])
    >>> A
    array([[ 1.+0.j, -0.-2.j],
           [ 0.+2.j,  5.+0.j]])
    >>> L = np.linalg.cholesky(A)
    >>> L
    array([[1.+0.j, 0.+0.j],
           [0.+2.j, 1.+0.j]])
    >>> np.dot(L, L.T.conj()) # verify that L * L.H = A
    array([[1.+0.j, 0.-2.j],
           [0.+2.j, 5.+0.j]])
    >>> A = [[1,-2j],[2j,5]] # what happens if A is only array_like?
    >>> np.linalg.cholesky(A) # an ndarray object is returned
    array([[1.+0.j, 0.+0.j],
           [0.+2.j, 1.+0.j]])
    >>> # But a matrix object is returned if A is a matrix object
    >>> np.linalg.cholesky(np.matrix(A))
    """

    # 实现 Cholesky 分解的具体功能
    # 此处未给出具体实现代码，因为该函数是通过 array_function_dispatch 分派器调用 _cholesky_dispatcher 处理不同输入类型的函数
    # 创建一个复数矩阵对象，这里是一个示例矩阵
    matrix([[ 1.+0.j,  0.+0.j],
            [ 0.+2.j,  1.+0.j]])
    # 可以通过设置参数 `upper=True` 获得上三角形式的 Cholesky 分解结果
    # 使用 NumPy 提供的线性代数模块进行 Cholesky 分解，并指定为上三角形式
    >>> np.linalg.cholesky(A, upper=True)
    # 返回一个上三角 Cholesky 因子的数组
    array([[1.-0.j, 0.-2.j],
           [0.-0.j, 1.-0.j]])

    """
    # 根据参数 `upper` 选择相应的 Cholesky 分解函数
    gufunc = _umath_linalg.cholesky_up if upper else _umath_linalg.cholesky_lo
    # 将输入矩阵 `a` 转换成数组，并进行必要的包装
    a, wrap = _makearray(a)
    # 确保输入数组是二维的
    _assert_stacked_2d(a)
    # 确保输入数组是方阵
    _assert_stacked_square(a)
    # 确定输入数组的数据类型和结果数组的类型
    t, result_t = _commonType(a)
    # 根据输入数组的数据类型确定函数签名
    signature = 'D->D' if isComplexType(t) else 'd->d'
    # 在特定的数学错误状态下执行 Cholesky 分解
    with errstate(call=_raise_linalgerror_nonposdef, invalid='call',
                  over='ignore', divide='ignore', under='ignore'):
        # 调用选择的 Cholesky 分解函数进行计算
        r = gufunc(a, signature=signature)
    # 将结果数组重新转换为指定的数据类型，并确保不进行复制
    return wrap(r.astype(result_t, copy=False))
# outer product

# 定义一个调度函数，用于决定如何分派参数 x1 和 x2
def _outer_dispatcher(x1, x2):
    return (x1, x2)

# 使用装饰器将 outer 函数注册到数组函数分发机制中
@array_function_dispatch(_outer_dispatcher)
def outer(x1, x2, /):
    """
    计算两个向量的外积。

    该函数兼容数组 API。与 `np.outer` 不同的是，它仅接受一维输入。

    Parameters
    ----------
    x1 : (M,) array_like
        大小为 ``N`` 的一维输入数组。
        必须具有数值数据类型。
    x2 : (N,) array_like
        大小为 ``M`` 的一维输入数组。
        必须具有数值数据类型。

    Returns
    -------
    out : (M, N) ndarray
        ``out[i, j] = a[i] * b[j]``

    See also
    --------
    outer

    """
    # 将输入数组转换为数组对象
    x1 = asanyarray(x1)
    x2 = asanyarray(x2)
    # 如果输入数组不是一维的，则引发 ValueError 异常
    if x1.ndim != 1 or x2.ndim != 1:
        raise ValueError(
            "Input arrays must be one-dimensional, but they are "
            f"{x1.ndim=} and {x2.ndim=}."
        )
    # 调用核心的外积计算函数，并返回结果
    return _core_outer(x1, x2, out=None)


# QR decomposition

# 定义一个调度函数，用于决定如何分派参数 a 和 mode
def _qr_dispatcher(a, mode=None):
    return (a,)

# 使用装饰器将 qr 函数注册到数组函数分发机制中
@array_function_dispatch(_qr_dispatcher)
def qr(a, mode='reduced'):
    """
    计算矩阵的 QR 分解。

    将矩阵 `a` 分解为 *QR*，其中 `q` 是正交的（单位长度）且 `r` 是上三角形。

    Parameters
    ----------
    a : array_like, shape (..., M, N)
        至少为 2 维的类数组对象。
    mode : {'reduced', 'complete', 'r', 'raw'}, optional, default: 'reduced'
        如果 K = min(M, N)，则

        * 'reduced'  : 返回维度为 (..., M, K), (..., K, N) 的 Q, R
        * 'complete' : 返回维度为 (..., M, M), (..., M, N) 的 Q, R
        * 'r'        : 仅返回维度为 (..., K, N) 的 R
        * 'raw'      : 返回维度为 (..., N, M), (..., K,) 的 h, tau

        选项 'reduced', 'complete' 和 'raw' 在 numpy 1.8 中引入，
        更多信息请参见备注。默认为 'reduced'，为了向后兼容，
        还支持旧的默认值 'full'。注意，在 'raw' 模式下返回的数组 h
        是为调用 Fortran 而转置的。'economic' 模式已弃用。可以使用
        'full' 和 'economic' 作为第一个字母进行传递，以保持向后兼容性，
        但其他模式必须完整拼写。详见备注以获取更多解释。

    Returns
    -------
    当 mode 是 'reduced' 或 'complete' 时，结果将是一个命名元组，
    具有属性 `Q` 和 `R`。

    Q : ndarray of float or complex, optional
        具有正交列的矩阵。当 mode = 'complete' 时，结果是正交/酉矩阵，
        取决于 a 是否实数/复数。在输入数组的维度大于 2 时，返回上述属性的堆栈矩阵。
    R : ndarray of float or complex, optional
        The upper-triangular matrix or a stack of upper-triangular
        matrices if the number of dimensions in the input array is greater
        than 2.
    (h, tau) : ndarrays of np.double or np.cdouble, optional
        The array h contains the Householder reflectors that generate q
        along with r. The tau array contains scaling factors for the
        reflectors. In the deprecated 'economic' mode only h is returned.

    Raises
    ------
    LinAlgError
        If factoring fails.

    See Also
    --------
    scipy.linalg.qr : Similar function in SciPy.
    scipy.linalg.rq : Compute RQ decomposition of a matrix.

    Notes
    -----
    This is an interface to the LAPACK routines ``dgeqrf``, ``zgeqrf``,
    ``dorgqr``, and ``zungqr``.

    For more information on the qr factorization, see for example:
    https://en.wikipedia.org/wiki/QR_factorization

    Subclasses of `ndarray` are preserved except for the 'raw' mode. So if
    `a` is of type `matrix`, all the return values will be matrices too.

    New 'reduced', 'complete', and 'raw' options for mode were added in
    NumPy 1.8.0 and the old option 'full' was made an alias of 'reduced'.  In
    addition the options 'full' and 'economic' were deprecated.  Because
    'full' was the previous default and 'reduced' is the new default,
    backward compatibility can be maintained by letting `mode` default.
    The 'raw' option was added so that LAPACK routines that can multiply
    arrays by q using the Householder reflectors can be used. Note that in
    this case the returned arrays are of type np.double or np.cdouble and
    the h array is transposed to be FORTRAN compatible.  No routines using
    the 'raw' return are currently exposed by numpy, but some are available
    in lapack_lite and just await the necessary work.

    Examples
    --------
    >>> rng = np.random.default_rng()
    >>> a = rng.normal(size=(9, 6))
    >>> Q, R = np.linalg.qr(a)
    >>> np.allclose(a, np.dot(Q, R))  # a does equal QR
    True
    >>> R2 = np.linalg.qr(a, mode='r')
    >>> np.allclose(R, R2)  # mode='r' returns the same R as mode='full'
    True
    >>> a = np.random.normal(size=(3, 2, 2)) # Stack of 2 x 2 matrices as input
    >>> Q, R = np.linalg.qr(a)
    >>> Q.shape
    (3, 2, 2)
    >>> R.shape
    (3, 2, 2)
    >>> np.allclose(a, np.matmul(Q, R))
    True

    Example illustrating a common use of `qr`: solving of least squares
    problems

    What are the least-squares-best `m` and `y0` in ``y = y0 + mx`` for
    the following data: {(0,1), (1,0), (1,2), (2,1)}. (Graph the points
    and you'll see that it should be y0 = 0, m = 1.)  The answer is provided
    by solving the over-determined matrix equation ``Ax = b``, where::

      A = array([[0, 1], [1, 1], [1, 1], [2, 1]])
      x = array([[y0], [m]])
      b = array([[1], [0], [2], [1]])

    If A = QR such that Q is orthonormal (which is always possible via


注释：
    """
    Handle QR decomposition of a matrix `a` with various modes.
    
    Parameters:
    - a : array_like
        Input matrix to decompose.
    - mode : {'reduced', 'complete', 'r', 'raw'}, optional
        Mode of decomposition. Defaults to 'reduced'.
        - 'reduced': Return the reduced QR decomposition.
        - 'complete': Return the complete QR decomposition.
        - 'r': Return only the upper triangular matrix R.
        - 'raw': Return the transposed Q and the tau factors.
    
    Returns:
    - QRResult
        Object containing the decomposition result.
    
    Raises:
    - ValueError
        If an unrecognized mode is provided.
    
    Notes:
    - The function utilizes optimized QR decomposition routines from the NumPy library.
    
    Examples:
        >>> A = np.array([[0, 1], [1, 1], [1, 1], [2, 1]])
        >>> A
        array([[0, 1],
               [1, 1],
               [1, 1],
               [2, 1]])
        >>> b = np.array([1, 2, 2, 3])
        >>> Q, R = np.linalg.qr(A)
        >>> p = np.dot(Q.T, b)
        >>> np.dot(np.linalg.inv(R), p)
        array([  1.,   1.])
    
    """
    if mode not in ('reduced', 'complete', 'r', 'raw'):
        if mode in ('f', 'full'):
            # Deprecation warning for 'full' mode
            msg = "".join((
                    "The 'full' option is deprecated in favor of 'reduced'.\n",
                    "For backward compatibility let mode default."))
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            mode = 'reduced'
        elif mode in ('e', 'economic'):
            # Deprecation warning for 'economic' mode
            msg = "The 'economic' option is deprecated."
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            mode = 'economic'
        else:
            # Raise an error for unrecognized mode
            raise ValueError(f"Unrecognized mode '{mode}'")
    
    # Convert input matrix 'a' to array and handle its dimensions
    a, wrap = _makearray(a)
    _assert_stacked_2d(a)
    m, n = a.shape[-2:]
    t, result_t = _commonType(a)
    a = a.astype(t, copy=True)
    a = _to_native_byte_order(a)
    mn = min(m, n)
    
    # Determine which QR decomposition function to use based on matrix dimensions
    if m <= n:
        gufunc = _umath_linalg.qr_r_raw_m
    else:
        gufunc = _umath_linalg.qr_r_raw_n
    
    # Define the signature for the gufunc based on the data type of 'a'
    signature = 'D->D' if isComplexType(t) else 'd->d'
    
    # Perform QR decomposition using the selected gufunc
    with errstate(call=_raise_linalgerror_qr, invalid='call',
                  over='ignore', divide='ignore', under='ignore'):
        tau = gufunc(a, signature=signature)
    
    # Handle different modes of decomposition
    if mode == 'r':
        # Return only the upper triangular matrix R
        r = triu(a[..., :mn, :])
        r = r.astype(result_t, copy=False)
        return wrap(r)
    
    if mode == 'raw':
        # Return transposed Q and tau factors
        q = transpose(a)
        q = q.astype(result_t, copy=False)
        tau = tau.astype(result_t, copy=False)
        return wrap(q), tau
    
    if mode == 'economic':
        # Return economic decomposition
        a = a.astype(result_t, copy=False)
        return wrap(a)
    
    # Determine the number of columns in the resulting Q matrix ('mc')
    # 'mc' depends on the mode:
    # - For 'complete' mode, it is equal to the number of rows 'm'
    # - For 'reduced' mode, it is the minimum of number of rows and columns 'mn'
    if mode == 'complete' and m > n:
        mc = m
        gufunc = _umath_linalg.qr_complete
    else:
        mc = mn
        gufunc = _umath_linalg.qr_reduced
    
    # Define the signature for the gufunc in the second QR decomposition
    signature = 'DD->D' if isComplexType(t) else 'dd->d'
    
    # Perform the second QR decomposition
    with errstate(call=_raise_linalgerror_qr, invalid='call',
                  over='ignore', divide='ignore', under='ignore'):
        q = gufunc(a, tau, signature=signature)
    
    # Extract the upper triangular matrix R from 'a'
    r = triu(a[..., :mc, :])
    
    # Convert Q and R to the desired result type
    q = q.astype(result_t, copy=False)
    r = r.astype(result_t, copy=False)
    
    # Return the QRResult object containing Q and R
    return QRResult(wrap(q), wrap(r))
# Eigenvalues

@array_function_dispatch(_unary_dispatcher)
def eigvals(a):
    """
    Compute the eigenvalues of a general matrix.

    Main difference between `eigvals` and `eig`: the eigenvectors aren't
    returned.

    Parameters
    ----------
    a : (..., M, M) array_like
        A complex- or real-valued matrix whose eigenvalues will be computed.

    Returns
    -------
    w : (..., M,) ndarray
        The eigenvalues, each repeated according to its multiplicity.
        They are not necessarily ordered, nor are they necessarily
        real for real matrices.

    Raises
    ------
    LinAlgError
        If the eigenvalue computation does not converge.

    See Also
    --------
    eig : eigenvalues and right eigenvectors of general arrays
    eigvalsh : eigenvalues of real symmetric or complex Hermitian
               (conjugate symmetric) arrays.
    eigh : eigenvalues and eigenvectors of real symmetric or complex
           Hermitian (conjugate symmetric) arrays.
    scipy.linalg.eigvals : Similar function in SciPy.

    Notes
    -----

    .. versionadded:: 1.8.0

    Broadcasting rules apply, see the `numpy.linalg` documentation for
    details.

    This is implemented using the ``_geev`` LAPACK routines which compute
    the eigenvalues and eigenvectors of general square arrays.

    Examples
    --------
    Illustration, using the fact that the eigenvalues of a diagonal matrix
    are its diagonal elements, that multiplying a matrix on the left
    by an orthogonal matrix, `Q`, and on the right by `Q.T` (the transpose
    of `Q`), preserves the eigenvalues of the "middle" matrix. In other words,
    if `Q` is orthogonal, then ``Q * A * Q.T`` has the same eigenvalues as
    ``A``:

    >>> from numpy import linalg as LA
    >>> x = np.random.random()
    >>> Q = np.array([[np.cos(x), -np.sin(x)], [np.sin(x), np.cos(x)]])
    >>> LA.norm(Q[0, :]), LA.norm(Q[1, :]), np.dot(Q[0, :],Q[1, :])
    (1.0, 1.0, 0.0)

    Now multiply a diagonal matrix by ``Q`` on one side and
    by ``Q.T`` on the other:

    >>> D = np.diag((-1,1))
    >>> LA.eigvals(D)
    array([-1.,  1.])
    >>> A = np.dot(Q, D)
    >>> A = np.dot(A, Q.T)
    >>> LA.eigvals(A)
    array([ 1., -1.]) # random

    """
    # 将输入矩阵转换为数组并进行封装
    a, wrap = _makearray(a)
    # 检查输入矩阵的维度是否正确
    _assert_stacked_2d(a)
    # 检查输入矩阵是否为方阵
    _assert_stacked_square(a)
    # 检查输入矩阵是否包含有限数值
    _assert_finite(a)
    # 确定输入矩阵的数据类型，并获取适当的返回结果类型
    t, result_t = _commonType(a)

    # 根据输入矩阵的数据类型选择适当的 LAPACK 签名
    signature = 'D->D' if isComplexType(t) else 'd->D'
    # 调用 LAPACK 函数计算矩阵的特征值
    with errstate(call=_raise_linalgerror_eigenvalues_nonconvergence,
                  invalid='call', over='ignore', divide='ignore',
                  under='ignore'):
        w = _umath_linalg.eigvals(a, signature=signature)

    # 如果输入矩阵为实数类型且所有特征值的虚部均为零，则返回实数类型的特征值
    if not isComplexType(t):
        if all(w.imag == 0):
            w = w.real
            result_t = _realType(result_t)
        else:
            result_t = _complexType(result_t)

    # 将结果转换为指定类型并返回，不复制数据
    return w.astype(result_t, copy=False)


def _eigvalsh_dispatcher(a, UPLO=None):
    return (a,)
# 使用 array_function_dispatch 装饰器，将函数 eigvalsh 注册到特定的分发机制中，用于根据输入的类型选择适当的实现。
@array_function_dispatch(_eigvalsh_dispatcher)
# 定义函数 eigvalsh，用于计算复共轭对称或实对称矩阵的特征值。
def eigvalsh(a, UPLO='L'):
    """
    计算复共轭对称或实对称矩阵的特征值。

    与 eigh 的主要区别在于：不计算特征向量。

    Parameters
    ----------
    a : (..., M, M) array_like
        待计算特征值的复数或实数矩阵。
    UPLO : {'L', 'U'}, optional
        指定计算是在矩阵的下三角部分 ('L'，默认) 还是上三角部分 ('U') 进行。
        不论该值如何设置，计算仅考虑对角线的实部以保持复共轭对称矩阵的概念。
        因此，对角线的虚部始终被视为零。

    Returns
    -------
    w : (..., M,) ndarray
        按升序排列的特征值，每个根据其重复次数重复。

    Raises
    ------
    LinAlgError
        如果特征值计算未收敛。

    See Also
    --------
    eigh : 实对称或复共轭对称矩阵的特征值和特征向量。
    eigvals : 一般实数或复数数组的特征值。
    eig : 一般实数或复数数组的特征值和右特征向量。
    scipy.linalg.eigvalsh : SciPy 中类似的函数。

    Notes
    -----
    .. versionadded:: 1.8.0

    Broadcasting rules apply, see the `numpy.linalg` documentation for
    details.

    特征值是使用 LAPACK 程序 ``_syevd``、``_heevd`` 计算的。

    Examples
    --------
    >>> from numpy import linalg as LA
    >>> a = np.array([[1, -2j], [2j, 5]])
    >>> LA.eigvalsh(a)
    array([ 0.17157288,  5.82842712]) # 可能有所不同

    >>> # 展示对角线虚部的处理
    >>> a = np.array([[5+2j, 9-2j], [0+2j, 2-1j]])
    >>> a
    array([[5.+2.j, 9.-2.j],
           [0.+2.j, 2.-1.j]])
    >>> # 使用 UPLO='L'，这在数值上等同于使用 LA.eigvals()
    >>> # 使用：
    >>> b = np.array([[5.+0.j, 0.-2.j], [0.+2.j, 2.-0.j]])
    >>> b
    array([[5.+0.j, 0.-2.j],
           [0.+2.j, 2.+0.j]])
    >>> wa = LA.eigvalsh(a)
    >>> wb = LA.eigvals(b)
    >>> wa; wb
    array([1., 6.])
    array([6.+0.j, 1.+0.j])

    """
    # 将 UPLO 转换为大写形式
    UPLO = UPLO.upper()
    # 如果 UPLO 不是 'L' 或 'U'，则引发 ValueError 异常
    if UPLO not in ('L', 'U'):
        raise ValueError("UPLO argument must be 'L' or 'U'")

    # 根据 UPLO 的值选择合适的 gufunc 函数
    if UPLO == 'L':
        gufunc = _umath_linalg.eigvalsh_lo
    else:
        gufunc = _umath_linalg.eigvalsh_up

    # 将输入数组 a 转换为 ndarray，并返回它及是否需要包装的信息
    a, wrap = _makearray(a)
    # 检查 a 是否是堆叠的二维数组
    _assert_stacked_2d(a)
    # 检查 a 是否是方阵
    _assert_stacked_square(a)
    # 获取数组 a 的常见数据类型和结果类型
    t, result_t = _commonType(a)
    # 根据数据类型 t 确定函数签名
    signature = 'D->d' if isComplexType(t) else 'd->d'
    # 使用 errstate 设定特定的错误状态处理方式，并计算特征值
    with errstate(call=_raise_linalgerror_eigenvalues_nonconvergence,
                  invalid='call', over='ignore', divide='ignore',
                  under='ignore'):
        w = gufunc(a, signature=signature)
    # 将数组 w 转换为指定类型 result_t，并返回结果
    return w.astype(_realType(result_t), copy=False)
# 定义一个函数 `_convertarray`，用于将输入数组转换成特定类型的数组，并返回转换后的数组、类型以及结果类型
def _convertarray(a):
    # 调用 `_commonType` 函数，获取数组 `a` 的通用类型 `t` 和结果类型 `result_t`
    t, result_t = _commonType(a)
    # 将数组 `a` 转换成类型 `t`，并进行转置操作，复制结果到新的数组 `a`
    a = a.astype(t).T.copy()
    # 返回转换后的数组 `a`、类型 `t` 和结果类型 `result_t`
    return a, t, result_t


# Eigenvectors

# 使用装饰器 `@array_function_dispatch(_unary_dispatcher)`，将函数 `eig` 分派给合适的数组函数
@array_function_dispatch(_unary_dispatcher)
# 定义函数 `eig`，计算方阵的特征值和右特征向量
def eig(a):
    """
    计算方阵的特征值和右特征向量。

    Parameters
    ----------
    a : (..., M, M) 数组
        要计算特征值和右特征向量的方阵

    Returns
    -------
    一个具有以下属性的命名元组：

    eigenvalues : (..., M) 数组
        特征值，每个特征值根据其重数重复。特征值不一定是有序的。如果虚部为零，则结果数组将为复数类型。当 `a`
        是实数时，得到的特征值将为实数（虚部为0）或以共轭对出现。

    eigenvectors : (..., M, M) 数组
        标准化（单位“长度”）的右特征向量，使得列 ``eigenvectors[:,i]`` 是对应于特征值 ``eigenvalues[i]`` 的特征向量。

    Raises
    ------
    LinAlgError
        如果特征值计算未收敛。

    See Also
    --------
    eigvals : 非对称数组的特征值。
    eigh : 实对称或复共轭对称（共轭对称）数组的特征值和特征向量。
    eigvalsh : 实对称或复共轭对称（共轭对称）数组的特征值。
    scipy.linalg.eig : SciPy 中的类似函数，也解决广义特征值问题。
    scipy.linalg.schur : 单位ary 和其他非共轭对称正规矩阵的最佳选择。

    Notes
    -----

    .. versionadded:: 1.8.0

    应用广播规则，详细信息请参阅 `numpy.linalg` 文档。

    此函数使用 `_geev` LAPACK 例程来计算一般方阵的特征值和特征向量。

    如果存在向量 `v`，使得 ``a @ v = w * v``，则数字 `w` 是 `a` 的特征值。因此，数组 `a`、`eigenvalues` 和
    `eigenvectors` 满足等式 ``a @ eigenvectors[:,i] = eigenvalues[i] * eigenvectors[:,i]``，对于 :math:`i \\in \\{0,...,M-1\\}`。

    数组 `eigenvectors` 可能不具有最大秩，也就是说，某些列可能是线性相关的，尽管舍入误差可能会掩盖这一事实。如果特征值都不同，则理论上特征向量是线性独立的，
    并且可以通过使用 `eigenvectors` 进行相似变换将 `a` 对角化，即 ``inv(eigenvectors) @ a @ eigenvectors`` 是对角的。

    对于非共轭对称正规矩阵，推荐使用 SciPy 函数 `scipy.linalg.schur`，因为矩阵 `eigenvectors` 被保证是单位ary，而在使用 `eig` 时不是这样。Schur 分解
    是

    """
    pass  # 函数体留空，实际计算逻辑未提供在注释中
    """
    Compute eigenvalues and right eigenvectors of a square matrix `a`.

    Parameters
    ----------
    a : array_like
        Input matrix.
    
    Returns
    -------
    EigResult
        Eigenvalues and eigenvectors. Eigenvalues are stored in `w`, and
        eigenvectors in `vt`.

    Notes
    -----
    This function computes the eigenvalues and right eigenvectors of a square
    matrix `a`. For a real matrix `a`, the eigenvalues `w` may be real or
    complex. The eigenvectors `vt` are normalized so that the column `vt[:,i]`
    corresponds to the eigenvector associated with `w[i]`.

    If the matrix `a` is real and all eigenvalues are real, `w` and `vt` are
    real. If the matrix has complex eigenvalues, `w` is complex, and `vt` is
    complex with complex-valued eigenvectors.

    References
    ----------
    G. Strang, *Linear Algebra and Its Applications*, 2nd Ed., Orlando, FL,
    Academic Press, Inc., 1980, Various pp.

    Examples
    --------
    >>> from numpy import linalg as LA

    (Almost) trivial example with real eigenvalues and eigenvectors.

    >>> eigenvalues, eigenvectors = LA.eig(np.diag((1, 2, 3)))
    >>> eigenvalues
    array([1., 2., 3.])
    >>> eigenvectors
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])

    Real matrix possessing complex eigenvalues and eigenvectors;
    note that the eigenvalues are complex conjugates of each other.

    >>> eigenvalues, eigenvectors = LA.eig(np.array([[1, -1], [1, 1]]))
    >>> eigenvalues
    array([1.+1.j, 1.-1.j])
    >>> eigenvectors
    array([[0.70710678+0.j        , 0.70710678-0.j        ],
           [0.        -0.70710678j, 0.        +0.70710678j]])

    Complex-valued matrix with real eigenvalues (but complex-valued
    eigenvectors); note that ``a.conj().T == a``, i.e., `a` is Hermitian.

    >>> a = np.array([[1, 1j], [-1j, 1]])
    >>> eigenvalues, eigenvectors = LA.eig(a)
    >>> eigenvalues
    array([2.+0.j, 0.+0.j])
    >>> eigenvectors
    array([[ 0.        +0.70710678j,  0.70710678+0.j        ], # may vary
           [ 0.70710678+0.j        , -0.        +0.70710678j]])

    Be careful about round-off error!

    >>> a = np.array([[1 + 1e-9, 0], [0, 1 - 1e-9]])
    >>> # Theor. eigenvalues are 1 +/- 1e-9
    >>> eigenvalues, eigenvectors = LA.eig(a)
    >>> eigenvalues
    array([1., 1.])
    >>> eigenvectors
    array([[1., 0.],
           [0., 1.]])
    """

    # 将输入数组 `a` 转换为数组并处理边界情况
    a, wrap = _makearray(a)
    # 确保输入数组 `a` 是二维的
    _assert_stacked_2d(a)
    # 确保输入数组 `a` 是方阵
    _assert_stacked_square(a)
    # 确保输入数组 `a` 中所有元素都是有限的数值
    _assert_finite(a)
    # 确定输入数组 `a` 的数据类型 `t`，以及输出结果的数据类型 `result_t`
    t, result_t = _commonType(a)

    # 确定调用 `umath_linalg.eig` 的函数签名，根据输入数组 `a` 是否为复数类型决定
    signature = 'D->DD' if isComplexType(t) else 'd->DD'
    # 调用 `umath_linalg.eig` 计算特征值 `w` 和右特征向量 `vt`
    with errstate(call=_raise_linalgerror_eigenvalues_nonconvergence,
                  invalid='call', over='ignore', divide='ignore',
                  under='ignore'):
        w, vt = _umath_linalg.eig(a, signature=signature)

    # 如果输入数组 `a` 不是复数类型且所有特征值的虚部都为零，则将 `w` 和 `vt` 转换为实数类型
    if not isComplexType(t) and all(w.imag == 0.0):
        w = w.real
        vt = vt.real
        result_t = _realType(result_t)
    else:
        # 否则，将输出结果的数据类型 `result_t` 设置为复数类型
        result_t = _complexType(result_t)

    # 将右特征向量 `vt` 转换为指定的数据类型 `result_t`
    vt = vt.astype(result_t, copy=False)
    # 返回特征值 `w` 和右特征向量 `vt` 的封装结果
    return EigResult(w.astype(result_t, copy=False), wrap(vt))
@array_function_dispatch(_eigvalsh_dispatcher)
# 使用 _eigvalsh_dispatcher 来分派到对应的数组函数
def eigh(a, UPLO='L'):
    """
    Return the eigenvalues and eigenvectors of a complex Hermitian
    (conjugate symmetric) or a real symmetric matrix.

    Returns two objects, a 1-D array containing the eigenvalues of `a`, and
    a 2-D square array or matrix (depending on the input type) of the
    corresponding eigenvectors (in columns).

    Parameters
    ----------
    a : (..., M, M) array
        Hermitian or real symmetric matrices whose eigenvalues and
        eigenvectors are to be computed.
    UPLO : {'L', 'U'}, optional
        Specifies whether the calculation is done with the lower triangular
        part of `a` ('L', default) or the upper triangular part ('U').
        Irrespective of this value only the real parts of the diagonal will
        be considered in the computation to preserve the notion of a Hermitian
        matrix. It therefore follows that the imaginary part of the diagonal
        will always be treated as zero.

    Returns
    -------
    A namedtuple with the following attributes:

    eigenvalues : (..., M) ndarray
        The eigenvalues in ascending order, each repeated according to
        its multiplicity.
    eigenvectors : {(..., M, M) ndarray, (..., M, M) matrix}
        The column ``eigenvectors[:, i]`` is the normalized eigenvector
        corresponding to the eigenvalue ``eigenvalues[i]``.  Will return a
        matrix object if `a` is a matrix object.

    Raises
    ------
    LinAlgError
        If the eigenvalue computation does not converge.

    See Also
    --------
    eigvalsh : eigenvalues of real symmetric or complex Hermitian
               (conjugate symmetric) arrays.
    eig : eigenvalues and right eigenvectors for non-symmetric arrays.
    eigvals : eigenvalues of non-symmetric arrays.
    scipy.linalg.eigh : Similar function in SciPy (but also solves the
                        generalized eigenvalue problem).

    Notes
    -----

    .. versionadded:: 1.8.0

    Broadcasting rules apply, see the `numpy.linalg` documentation for
    details.

    The eigenvalues/eigenvectors are computed using LAPACK routines ``_syevd``,
    ``_heevd``.

    The eigenvalues of real symmetric or complex Hermitian matrices are always
    real. [1]_ The array `eigenvalues` of (column) eigenvectors is unitary and
    `a`, `eigenvalues`, and `eigenvectors` satisfy the equations ``dot(a,
    eigenvectors[:, i]) = eigenvalues[i] * eigenvectors[:, i]``.

    References
    ----------
    .. [1] G. Strang, *Linear Algebra and Its Applications*, 2nd Ed., Orlando,
           FL, Academic Press, Inc., 1980, pg. 222.

    Examples
    --------
    >>> from numpy import linalg as LA
    >>> a = np.array([[1, -2j], [2j, 5]])
    >>> a
    array([[ 1.+0.j, -0.-2.j],
           [ 0.+2.j,  5.+0.j]])
    >>> eigenvalues, eigenvectors = LA.eigh(a)
    >>> eigenvalues
    array([0.17157288, 5.82842712])
    >>> eigenvectors
    """
    # 实现复数共轭对称或实数对称矩阵的特征值和特征向量计算
    # 调用 LAPACK 中的 _syevd 或 _heevd 进行计算
    pass
    """
    UPLO = UPLO.upper()
    if UPLO not in ('L', 'U'):
        raise ValueError("UPLO argument must be 'L' or 'U'")
    
    a, wrap = _makearray(a)
    _assert_stacked_2d(a)
    _assert_stacked_square(a)
    t, result_t = _commonType(a)
    
    if UPLO == 'L':
        gufunc = _umath_linalg.eigh_lo
    else:
        gufunc = _umath_linalg.eigh_up
    
    signature = 'D->dD' if isComplexType(t) else 'd->dd'
    with errstate(call=_raise_linalgerror_eigenvalues_nonconvergence,
                  invalid='call', over='ignore', divide='ignore',
                  under='ignore'):
        w, vt = gufunc(a, signature=signature)
    w = w.astype(_realType(result_t), copy=False)
    vt = vt.astype(result_t, copy=False)
    return EighResult(w, wrap(vt))
    """
    
    # 将 UPLO 参数转换为大写
    UPLO = UPLO.upper()
    # 如果 UPLO 不是 'L' 或 'U'，则抛出数值错误
    if UPLO not in ('L', 'U'):
        raise ValueError("UPLO argument must be 'L' or 'U'")
    
    # 使用 _makearray 将输入参数 a 包装为数组 a 和 wrap 对象
    a, wrap = _makearray(a)
    # 检查数组 a 是否为二维堆叠数组
    _assert_stacked_2d(a)
    # 检查数组 a 是否为方阵
    _assert_stacked_square(a)
    # 获取数组 a 的数据类型 t 和结果数据类型 result_t
    t, result_t = _commonType(a)
    
    # 根据 UPLO 参数选择对应的 LAPACK 函数
    if UPLO == 'L':
        gufunc = _umath_linalg.eigh_lo
    else:
        gufunc = _umath_linalg.eigh_up
    
    # 根据数组元素的复杂性确定函数签名
    signature = 'D->dD' if isComplexType(t) else 'd->dd'
    # 设置错误状态处理器，处理特定的线性代数错误
    with errstate(call=_raise_linalgerror_eigenvalues_nonconvergence,
                  invalid='call', over='ignore', divide='ignore',
                  under='ignore'):
        # 调用 LAPACK 函数计算数组 a 的特征值 w 和特征向量 vt
        w, vt = gufunc(a, signature=signature)
    
    # 将特征值 w 转换为与结果数据类型对应的实数类型
    w = w.astype(_realType(result_t), copy=False)
    # 将特征向量 vt 转换为结果数据类型，不进行复制
    vt = vt.astype(result_t, copy=False)
    
    # 返回特征值和特征向量的结果对象 EighResult
    return EighResult(w, wrap(vt))
# 奇异值分解

# 定义一个分发函数，用于分发参数 `a` 到奇异值分解的函数 `_svd_dispatcher`
def _svd_dispatcher(a, full_matrices=None, compute_uv=None, hermitian=None):
    # 返回参数 `a` 的元组形式，保持不变
    return (a,)

# 使用装饰器 `array_function_dispatch` 注册 `_svd_dispatcher` 函数作为 `svd` 函数的分发器
@array_function_dispatch(_svd_dispatcher)
# 主奇异值分解函数 `svd`
def svd(a, full_matrices=True, compute_uv=True, hermitian=False):
    """
    奇异值分解（Singular Value Decomposition，SVD）。

    当 `a` 是一个二维数组且 `full_matrices=False` 时，分解为 `u @ np.diag(s) @ vh = (u * s) @ vh`，
    其中 `u` 和 `vh` 的 Hermitian 转置是具有正交列的二维数组，`s` 是 `a` 的奇异值的一维数组。
    当 `a` 是高维的时，SVD 将按照下面解释的堆叠模式应用。

    参数
    ----------
    a : (..., M, N) array_like
        一个实数或复数数组，`a.ndim >= 2`。
    full_matrices : bool, optional
        如果为 True（默认值），`u` 和 `vh` 的形状分别为 `(..., M, M)` 和 `(..., N, N)`。
        否则，形状分别为 `(..., M, K)` 和 `(..., K, N)`，其中 `K = min(M, N)`。
    compute_uv : bool, optional
        是否计算 `u` 和 `vh`。默认为 True。
    hermitian : bool, optional
        如果为 True，假设 `a` 是 Hermitian（如果是实数则是对称的），可以启用一种更有效的方法来查找奇异值。
        默认为 False。

        .. versionadded:: 1.17.0

    返回
    -------
    当 `compute_uv` 为 True 时，结果是一个具名元组，具有以下属性名：

    U : { (..., M, M), (..., M, K) } array
        单位数组。前 `a.ndim - 2` 维度与输入 `a` 的大小相同。最后两个维度的大小取决于 `full_matrices` 的值。
        仅在 `compute_uv` 为 True 时返回。
    S : (..., K) array
        奇异值的向量，每个向量按降序排列。前 `a.ndim - 2` 维度与输入 `a` 的大小相同。
    Vh : { (..., N, N), (..., K, N) } array
        单位数组。前 `a.ndim - 2` 维度与输入 `a` 的大小相同。最后两个维度的大小取决于 `full_matrices` 的值。
        仅在 `compute_uv` 为 True 时返回。

    异常
    ------
    LinAlgError
        如果奇异值分解无法收敛。

    参见
    --------
    scipy.linalg.svd : SciPy 中类似的函数。
    scipy.linalg.svdvals : 计算矩阵的奇异值。

    注意
    -----

    .. versionchanged:: 1.8.0
       广播规则适用，请参阅 `numpy.linalg` 文档以获取详细信息。

    使用 LAPACK 例程 `_gesdd` 执行分解。

    SVD 通常用于描述二维矩阵 :math:`A` 的因子分解。
    高维情况将在下面讨论。在二维情况下，SVD 被写为 :math:`A = U S V^H`，其中 :math:`A = a`，:math:`U = u`，
    # 导入 NumPy 库，将其重命名为 _nx
    import numpy as _nx
    # 调用 _makearray 函数，将 a 作为参数传入，并返回结果给变量 a 和 wrap
    a, wrap = _makearray(a)
    # 如果 hermitian 为 True，则执行以下操作
    if hermitian:
        # Lapack SVD 返回的特征值 s ** 2 是按降序排列的，
        # 而 eig 返回的特征值 s 是按升序排列的，因此我们重新排序特征值
        # 和相关数组，以确保它们的顺序是正确的

        # 如果 compute_uv 也为 True，则执行以下操作
        if compute_uv:
            # 对矩阵 a 进行 Hermitian 特征分解，返回特征值 s 和特征向量 u
            s, u = eigh(a)
            # 计算特征值 s 的符号
            sgn = sign(s)
            # 取特征值 s 的绝对值
            s = abs(s)
            # 对特征值 s 进行降序排序，并记录排序后的索引
            sidx = argsort(s)[..., ::-1]
            # 根据排序后的索引重新排列符号 sgn
            sgn = _nx.take_along_axis(sgn, sidx, axis=-1)
            # 根据排序后的索引重新排列特征值 s
            s = _nx.take_along_axis(s, sidx, axis=-1)
            # 根据排序后的索引重新排列特征向量 u
            u = _nx.take_along_axis(u, sidx[..., None, :], axis=-1)
            # 将奇异值的符号移动到 v 中，生成 vt
            vt = transpose(u * sgn[..., None, :]).conjugate()
            # 返回奇异值分解的结果，包括 u、s 和 vt
            return SVDResult(wrap(u), s, wrap(vt))
        else:
            # 对 Hermitian 矩阵 a 进行特征值分解，返回特征值 s
            s = eigvalsh(a)
            # 取特征值 s 的绝对值
            s = abs(s)
            # 对特征值 s 进行降序排序
            return sort(s)[..., ::-1]

    # 如果 hermitian 不为 True，则执行以下操作
    _assert_stacked_2d(a)
    # 根据矩阵 a 确定通用类型 t，并返回类型和处理结果
    t, result_t = _commonType(a)

    # 获取矩阵 a 的形状 m 和 n
    m, n = a.shape[-2:]

    # 如果 compute_uv 为 True，则执行以下操作
    if compute_uv:
        # 如果需要完整的 U 矩阵
        if full_matrices:
            # 根据 m 和 n 的大小选择适当的 SVD 函数
            if m < n:
                gufunc = _umath_linalg.svd_m_f
            else:
                gufunc = _umath_linalg.svd_n_f
        else:
            # 根据 m 和 n 的大小选择适当的 SVD 函数
            if m < n:
                gufunc = _umath_linalg.svd_m_s
            else:
                gufunc = _umath_linalg.svd_n_s

        # 签名定义，复杂类型和实数类型的签名略有不同
        signature = 'D->DdD' if isComplexType(t) else 'd->ddd'

        # 在特定的错误状态下调用 SVD 函数 gufunc
        with errstate(call=_raise_linalgerror_svd_nonconvergence,
                      invalid='call', over='ignore', divide='ignore',
                      under='ignore'):
            # 执行 SVD 计算，返回 u、s 和 vh
            u, s, vh = gufunc(a, signature=signature)

        # 将 u、s、vh 转换为指定类型 result_t，并避免复制
        u = u.astype(result_t, copy=False)
        s = s.astype(_realType(result_t), copy=False)
        vh = vh.astype(result_t, copy=False)

        # 返回奇异值分解的结果，包括 u、s 和 vh
        return SVDResult(wrap(u), s, wrap(vh))
    else:
        # 如果不需要计算 u，则执行以下操作
        # 根据 m 和 n 的大小选择适当的 SVD 函数
        if m < n:
            gufunc = _umath_linalg.svd_m
        else:
            gufunc = _umath_linalg.svd_n

        # 签名定义，复杂类型和实数类型的签名略有不同
        signature = 'D->d' if isComplexType(t) else 'd->d'

        # 在特定的错误状态下调用 SVD 函数 gufunc
        with errstate(call=_raise_linalgerror_svd_nonconvergence,
                      invalid='call', over='ignore', divide='ignore',
                      under='ignore'):
            # 执行 SVD 计算，返回 s
            s = gufunc(a, signature=signature)

        # 将 s 转换为指定类型 result_t 的实数类型，并避免复制
        s = s.astype(_realType(result_t), copy=False)

        # 返回奇异值分解的结果，即 s
        return s
# 定义函数 _svdvals_dispatcher，接受一个参数 x，并返回一个包含 x 的元组
def _svdvals_dispatcher(x):
    return (x,)


# 使用装饰器 array_function_dispatch 将函数 svdvals 和 _svdvals_dispatcher 关联起来，
# 以便处理不同类型的输入 x
@array_function_dispatch(_svdvals_dispatcher)
# 定义函数 svdvals，接受一个参数 x，使用 Array API
def svdvals(x, /):
    """
    Returns the singular values of a matrix (or a stack of matrices) ``x``.
    When x is a stack of matrices, the function will compute the singular
    values for each matrix in the stack.

    This function is Array API compatible.

    Calling ``np.svdvals(x)`` to get singular values is the same as
    ``np.svd(x, compute_uv=False, hermitian=False)``.

    Parameters
    ----------
    x : (..., M, N) array_like
        Input array having shape (..., M, N) and whose last two
        dimensions form matrices on which to perform singular value
        decomposition. Should have a floating-point data type.

    Returns
    -------
    out : ndarray
        An array with shape (..., K) that contains the vector(s)
        of singular values of length K, where K = min(M, N).

    See Also
    --------
    scipy.linalg.svdvals : Compute singular values of a matrix.

    """
    # 调用 svd 函数计算 x 的奇异值，不计算左右奇异向量，并非 Hermitian
    return svd(x, compute_uv=False, hermitian=False)


# 定义函数 _cond_dispatcher，接受两个参数 x 和可选参数 p，并返回一个包含 x 的元组
def _cond_dispatcher(x, p=None):
    return (x,)


# 使用装饰器 array_function_dispatch 将函数 cond 和 _cond_dispatcher 关联起来，
# 以便处理不同类型的输入 x 和可选参数 p
@array_function_dispatch(_cond_dispatcher)
# 定义函数 cond，用于计算矩阵的条件数
def cond(x, p=None):
    """
    Compute the condition number of a matrix.

    This function is capable of returning the condition number using
    one of seven different norms, depending on the value of `p` (see
    Parameters below).

    Parameters
    ----------
    x : (..., M, N) array_like
        The matrix whose condition number is sought.
    p : {None, 1, -1, 2, -2, inf, -inf, 'fro'}, optional
        Order of the norm used in the condition number computation:

        =====  ============================
        p      norm for matrices
        =====  ============================
        None   2-norm, computed directly using the ``SVD``
        'fro'  Frobenius norm
        inf    max(sum(abs(x), axis=1))
        -inf   min(sum(abs(x), axis=1))
        1      max(sum(abs(x), axis=0))
        -1     min(sum(abs(x), axis=0))
        2      2-norm (largest sing. value)
        -2     smallest singular value
        =====  ============================

        inf means the `numpy.inf` object, and the Frobenius norm is
        the root-of-sum-of-squares norm.

    Returns
    -------
    c : {float, inf}
        The condition number of the matrix. May be infinite.

    See Also
    --------
    numpy.linalg.norm

    Notes
    -----
    The condition number of `x` is defined as the norm of `x` times the
    norm of the inverse of `x` [1]_; the norm can be the usual L2-norm
    (root-of-sum-of-squares) or one of a number of other matrix norms.

    References
    ----------
    .. [1] G. Strang, *Linear Algebra and Its Applications*, Orlando, FL,
           Academic Press, Inc., 1980, pg. 285.

    Examples
    --------
    >>> from numpy import linalg as LA
    >>> a = np.array([[1, 0, -1], [0, 1, 0], [1, 0, 1]])
    >>> a
    array([[ 1,  0, -1],
           [ 0,  1,  0],
           [ 1,  0,  1]])

    """
    # 返回矩阵 x 的条件数，根据参数 p 选择不同的范数计算条件数
    # 若 p 为 None，则使用 SVD 直接计算 2-范数的条件数
    # 其他参数 p 分别对应不同的矩阵范数计算方法
    return cond(x, p)
    # 计算数组 x 的条件数，可以指定不同的范数 p
    x = asarray(x)  # 将输入 x 转换为数组（考虑到可能输入为矩阵）
    if _is_empty_2d(x):
        # 如果 x 是空数组（二维），抛出线性代数错误
        raise LinAlgError("cond is not defined on empty arrays")
    if p is None or p == 2 or p == -2:
        # 如果未指定范数 p 或者 p 为 2 或 -2
        s = svd(x, compute_uv=False)  # 对 x 进行奇异值分解，仅计算奇异值
        with errstate(all='ignore'):
            if p == -2:
                # 如果 p 为 -2，计算条件数的反比
                r = s[..., -1] / s[..., 0]
            else:
                # 否则计算条件数的比值
                r = s[..., 0] / s[..., -1]
    else:
        # 否则，使用给定范数 p 计算条件数
        _assert_stacked_2d(x)  # 断言 x 是叠加的二维数组
        _assert_stacked_square(x)  # 断言 x 是叠加的方阵
        t, result_t = _commonType(x)  # 确定 x 的常见数据类型和结果类型
        signature = 'D->D' if isComplexType(t) else 'd->d'
        with errstate(all='ignore'):
            # 计算 x 的逆矩阵，忽略错误（结果数组中可能包含 NaN）
            invx = _umath_linalg.inv(x, signature=signature)
            # 计算以给定范数 p 的 x 的范数和逆矩阵 invx 的范数的乘积
            r = norm(x, p, axis=(-2, -1)) * norm(invx, p, axis=(-2, -1))
        r = r.astype(result_t, copy=False)  # 将结果 r 转换为指定类型，不复制数据

    # 将 NaN 转换为无穷大，除非原始数组 x 中有 NaN 条目
    r = asarray(r)  # 将 r 转换为数组
    nan_mask = isnan(r)  # 创建 NaN 掩码
    if nan_mask.any():
        nan_mask &= ~isnan(x).any(axis=(-2, -1))  # 除非 x 的任何轴上都有 NaN 条目
        if r.ndim > 0:
            r[nan_mask] = inf  # 如果 r 的维度大于 0，将 NaN 替换为无穷大
        elif nan_mask:
            r[()] = inf  # 否则，如果存在 NaN 掩码，则将整个 r 替换为无穷大

    # 按照惯例，返回标量而不是 0 维数组
    if r.ndim == 0:
        r = r[()]

    return r
# 定义一个私有函数 _matrix_rank_dispatcher，接受参数 A, tol, hermitian 和关键字参数 rtol
def _matrix_rank_dispatcher(A, tol=None, hermitian=None, *, rtol=None):
    # 直接返回参数 A，用于 array_function_dispatch 调度
    return (A,)


# 使用 array_function_dispatch 装饰器，将 _matrix_rank_dispatcher 作为调度器
@array_function_dispatch(_matrix_rank_dispatcher)
# 定义 matrix_rank 函数，计算输入数组或矩阵的秩，使用奇异值分解（SVD）方法
def matrix_rank(A, tol=None, hermitian=False, *, rtol=None):
    """
    使用奇异值分解（SVD）方法计算数组或矩阵的秩

    数组的秩是数组的奇异值中大于 `tol` 的数量。

    .. versionchanged:: 1.14
       现在可以操作矩阵的堆叠

    Parameters
    ----------
    A : {(M,), (..., M, N)} array_like
        输入的向量或堆叠的矩阵。
    tol : (...) array_like, float, optional
        当奇异值小于 `tol` 时被视为零的阈值。如果 `tol` 是 None，并且 ``S`` 是包含 `M` 的奇异值数组，``eps`` 是 ``S`` 的数据类型的 epsilon 值，
        则 `tol` 被设置为 ``S.max() * max(M, N) * eps``。

        .. versionchanged:: 1.14
           与堆叠的矩阵广播

    hermitian : bool, optional
        如果为 True，假定 `A` 是 Hermitian（如果是实数则为对称），这样可以更有效地找到奇异值。
        默认为 False。

        .. versionadded:: 1.14
    rtol : (...) array_like, float, optional
        相对容差分量的参数。只能设置 ``tol`` 或 ``rtol`` 中的一个。默认为 ``max(M, N) * eps``。

        .. versionadded:: 2.0.0

    Returns
    -------
    rank : (...) array_like
        A 的秩。

    Notes
    -----
    检测秩不足的默认阈值是对 `A` 的奇异值的大小进行测试。默认情况下，我们将小于 ``S.max() * max(M, N) * eps`` 的奇异值视为指示秩不足
    的标志（其中定义的符号）。这是 MATLAB 使用的算法 [1]。它还出现在 *Numerical recipes* 中关于线性最小二乘 SVD 解的讨论中 [2]。

    此默认阈值设计用于检测秩不足，考虑了 SVD 计算的数值误差。想象一下，在 `A` 中有一列是其他列的精确（在浮点数中）线性组合。在 `A` 上计算
    SVD 通常不会产生一个确切等于 0 的奇异值：最小奇异值与 0 的任何差异将由于计算 SVD 时的数值不精确而引起。我们对小的奇异值设定了阈值，
    这考虑了这种数值误差，并且默认阈值将检测到这种数值秩不足。即使 `A` 的某些列的线性组合不完全等于 `A` 的另一列，而只是数值上非常接近
    另一列，该阈值也可能声明矩阵 `A` 秩不足。

    我们选择默认阈值是因为它被广泛使用。其他阈值也是可能的。例如，在 *Numerical recipes* 2007 版本的其他地方，有一个替代阈值，
    即 ``S.max() * max(M, N) * eps``。
    """
    # 返回 A 的秩，使用默认参数或者根据参数计算相应的阈值
    return (A,)
    """
    Calculate the effective rank of a matrix `A` based on singular value decomposition (SVD).

    Parameters
    ----------
    A : array_like
        Input matrix.
    rtol : float or None, optional
        Relative tolerance. If None, it defaults to `max(A.shape[-2:]) * finfo(S.dtype).eps`.
    tol : float or None, optional
        Absolute tolerance. If None, it defaults to `S.max(axis=-1, keepdims=True) * rtol`.
    hermitian : bool, optional
        Whether `A` is Hermitian (symmetric if real-valued).

    Returns
    -------
    int
        The effective rank of `A` based on the provided tolerance.

    Raises
    ------
    ValueError
        If both `rtol` and `tol` are provided.

    Notes
    -----
    The effective rank is determined by counting non-zero singular values greater than the tolerance `tol`.

    References
    ----------
    [1] MATLAB reference documentation, "Rank"
           https://www.mathworks.com/help/techdoc/ref/rank.html
    [2] W. H. Press, S. A. Teukolsky, W. T. Vetterling and B. P. Flannery,
           "Numerical Recipes (3rd edition)", Cambridge University Press, 2007,
           page 795.

    Examples
    --------
    >>> from numpy.linalg import matrix_rank
    >>> matrix_rank(np.eye(4)) # Full rank matrix
    4
    >>> I=np.eye(4); I[-1,-1] = 0. # rank deficient matrix
    >>> matrix_rank(I)
    3
    >>> matrix_rank(np.ones((4,))) # 1 dimension - rank 1 unless all 0
    1
    >>> matrix_rank(np.zeros((4,)))
    0
    """
    if rtol is not None and tol is not None:
        raise ValueError("`tol` and `rtol` can't be both set.")

    A = asarray(A)  # Convert input `A` to a numpy array
    if A.ndim < 2:  # Check if `A` is less than 2-dimensional
        return int(not all(A == 0))  # Return 1 if `A` is not all zeros, otherwise 0
    S = svd(A, compute_uv=False, hermitian=hermitian)  # Perform SVD on `A` to get singular values `S`

    if tol is None:  # If `tol` is not provided
        if rtol is None:  # If `rtol` is also not provided
            rtol = max(A.shape[-2:]) * finfo(S.dtype).eps  # Set `rtol` based on machine epsilon and shape of `A`
        else:
            rtol = asarray(rtol)[..., newaxis]  # Convert `rtol` to a numpy array with an additional axis
        tol = S.max(axis=-1, keepdims=True) * rtol  # Calculate tolerance based on maximum singular value and `rtol`
    else:
        tol = asarray(tol)[..., newaxis]  # Convert `tol` to a numpy array with an additional axis

    return count_nonzero(S > tol, axis=-1)  # Count non-zero singular values greater than tolerance `tol`
# Generalized inverse

# 定义一个分发函数 _pinv_dispatcher，用于派发参数 a
def _pinv_dispatcher(a, rcond=None, hermitian=None, *, rtol=None):
    # 返回参数 a 的元组形式，用于后续处理
    return (a,)

# 使用装饰器 array_function_dispatch 将 _pinv_dispatcher 注册为 pinv 函数的分发器
@array_function_dispatch(_pinv_dispatcher)
# 定义函数 pinv，计算矩阵的广义逆
def pinv(a, rcond=None, hermitian=False, *, rtol=_NoValue):
    """
    Compute the (Moore-Penrose) pseudo-inverse of a matrix.

    Calculate the generalized inverse of a matrix using its
    singular-value decomposition (SVD) and including all
    *large* singular values.

    .. versionchanged:: 1.14
       Can now operate on stacks of matrices

    Parameters
    ----------
    a : (..., M, N) array_like
        Matrix or stack of matrices to be pseudo-inverted.
    rcond : (...) array_like of float, optional
        Cutoff for small singular values.
        Singular values less than or equal to
        ``rcond * largest_singular_value`` are set to zero.
        Broadcasts against the stack of matrices. Default: ``1e-15``.
    hermitian : bool, optional
        If True, `a` is assumed to be Hermitian (symmetric if real-valued),
        enabling a more efficient method for finding singular values.
        Defaults to False.

        .. versionadded:: 1.17.0
    rtol : (...) array_like of float, optional
        Same as `rcond`, but it's an Array API compatible parameter name.
        Only `rcond` or `rtol` can be set at a time. If none of them are
        provided then NumPy's ``1e-15`` default is used. If ``rtol=None``
        is passed then the API standard default is used.

        .. versionadded:: 2.0.0

    Returns
    -------
    B : (..., N, M) ndarray
        The pseudo-inverse of `a`. If `a` is a `matrix` instance, then so
        is `B`.

    Raises
    ------
    LinAlgError
        If the SVD computation does not converge.

    See Also
    --------
    scipy.linalg.pinv : Similar function in SciPy.
    scipy.linalg.pinvh : Compute the (Moore-Penrose) pseudo-inverse of a
                         Hermitian matrix.

    Notes
    -----
    The pseudo-inverse of a matrix A, denoted :math:`A^+`, is
    defined as: "the matrix that 'solves' [the least-squares problem]
    :math:`Ax = b`," i.e., if :math:`\\bar{x}` is said solution, then
    :math:`A^+` is that matrix such that :math:`\\bar{x} = A^+b`.

    It can be shown that if :math:`Q_1 \\Sigma Q_2^T = A` is the singular
    value decomposition of A, then
    :math:`A^+ = Q_2 \\Sigma^+ Q_1^T`, where :math:`Q_{1,2}` are
    orthogonal matrices, :math:`\\Sigma` is a diagonal matrix consisting
    of A's so-called singular values, (followed, typically, by
    zeros), and then :math:`\\Sigma^+` is simply the diagonal matrix
    consisting of the reciprocals of A's singular values
    (again, followed by zeros). [1]_

    References
    ----------
    .. [1] G. Strang, *Linear Algebra and Its Applications*, 2nd Ed., Orlando,
           FL, Academic Press, Inc., 1980, pp. 139-142.

    Examples
    --------
    The following example checks that ``a * a+ * a == a`` and
    ``a+ * a * a+ == a+:

    >>> rng = np.random.default_rng()

"""
    >>> a = rng.normal(size=(9, 6))
    >>> B = np.linalg.pinv(a)
    >>> np.allclose(a, np.dot(a, np.dot(B, a)))
    True
    >>> np.allclose(B, np.dot(B, np.dot(a, B)))
    True

    """
    # 将输入数组 `a` 规范化为数组对象，并根据需要进行包装
    a, wrap = _makearray(a)
    # 如果未指定 rcond 参数，则根据 rtol 或默认值设置 rcond
    if rcond is None:
        if rtol is _NoValue:
            rcond = 1e-15
        elif rtol is None:
            rcond = max(a.shape[-2:]) * finfo(a.dtype).eps
        else:
            rcond = rtol
    elif rtol is not _NoValue:
        # 如果同时设置了 rtol 和 rcond，抛出 ValueError 异常
        raise ValueError("`rtol` and `rcond` can't be both set.")
    else:
        # 注意：在未来的几个版本中将会弃用 `rcond`
        pass

    # 将 rcond 转换为数组形式
    rcond = asarray(rcond)
    # 如果数组 `a` 是一个空的二维数组
    if _is_empty_2d(a):
        # 获取数组 `a` 的形状信息
        m, n = a.shape[-2:]
        # 创建一个空的结果数组，保持与输入数组相同的形状
        res = empty(a.shape[:-2] + (n, m), dtype=a.dtype)
        return wrap(res)
    # 对数组 `a` 进行共轭处理
    a = a.conjugate()
    # 进行奇异值分解（SVD），获取左奇异向量 (`u`), 奇异值 (`s`), 右奇异向量的转置 (`vt`)
    u, s, vt = svd(a, full_matrices=False, hermitian=hermitian)

    # 丢弃较小的奇异值
    # 计算一个截断值，通过 rcond 乘以奇异值中的最大值
    cutoff = rcond[..., newaxis] * amax(s, axis=-1, keepdims=True)
    # 确定哪些奇异值大于截断值
    large = s > cutoff
    # 将奇异值矩阵进行修剪，对于较小的奇异值置为零
    s = divide(1, s, where=large, out=s)
    s[~large] = 0

    # 计算修正后的矩阵乘积结果
    res = matmul(transpose(vt), multiply(s[..., newaxis], transpose(u)))
    return wrap(res)
# Determinant

# 导入必要的模块和函数
@array_function_dispatch(_unary_dispatcher)
def slogdet(a):
    """
    计算数组的行列式的符号和自然对数。

    如果数组的行列式非常小或非常大，那么调用 `det` 可能会溢出或下溢。
    这个函数比较稳健，因为它计算的是行列式的自然对数，而不是行列式本身。

    Parameters
    ----------
    a : (..., M, M) array_like
        输入数组，必须是一个方形的二维数组。

    Returns
    -------
    一个命名元组，包含以下属性:

    sign : (...) array_like
        行列式的符号。对于实数矩阵，可能是 1、0 或 -1。
        对于复数矩阵，可能是一个绝对值为1的复数（即在单位圆上），或者是0。
    logabsdet : (...) array_like
        行列式绝对值的自然对数。

    如果行列式为零，那么 `sign` 将为 0，`logabsdet` 将为 -inf。
    在所有情况下，行列式等于 ``sign * np.exp(logabsdet)``。

    See Also
    --------
    det

    Notes
    -----
    Broadcasting rules apply, see the `numpy.linalg` documentation for
    details.

    .. versionadded:: 1.8.0

    行列式通过使用 LAPACK 的 LU 分解计算，具体使用了 ``z/dgetrf`` 的例程。

    Examples
    --------
    2-D 数组 ``[[a, b], [c, d]]`` 的行列式为 ``ad - bc``:

    >>> a = np.array([[1, 2], [3, 4]])
    >>> (sign, logabsdet) = np.linalg.slogdet(a)
    >>> (sign, logabsdet)
    (-1, 0.69314718055994529) # 可能有所不同
    >>> sign * np.exp(logabsdet)
    -2.0

    计算一组矩阵的对数行列式：

    >>> a = np.array([ [[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]] ])
    >>> a.shape
    (3, 2, 2)
    >>> sign, logabsdet = np.linalg.slogdet(a)
    >>> (sign, logabsdet)
    (array([-1., -1., -1.]), array([ 0.69314718,  1.09861229,  2.07944154]))
    >>> sign * np.exp(logabsdet)
    array([-2., -3., -8.])

    这个函数在普通 `det` 函数失败的情况下依然能成功：

    >>> np.linalg.det(np.eye(500) * 0.1)
    0.0
    >>> np.linalg.slogdet(np.eye(500) * 0.1)
    (1, -1151.2925464970228)

    """
    # 将输入转换为数组
    a = asarray(a)
    # 确保输入是堆叠的二维数组
    _assert_stacked_2d(a)
    # 确保输入是方形的堆叠数组
    _assert_stacked_square(a)
    # 确定数组的公共类型
    t, result_t = _commonType(a)
    # 确定结果的实数类型
    real_t = _realType(result_t)
    # 确定函数签名，根据输入数组的类型（实数或复数）
    signature = 'D->Dd' if isComplexType(t) else 'd->dd'
    # 调用底层的 umath_linalg.slogdet 函数计算符号和对数行列式
    sign, logdet = _umath_linalg.slogdet(a, signature=signature)
    # 将符号和对数行列式转换为指定的结果类型
    sign = sign.astype(result_t, copy=False)
    logdet = logdet.astype(real_t, copy=False)
    # 返回命名元组 SlogdetResult，包含计算结果
    return SlogdetResult(sign, logdet)


@array_function_dispatch(_unary_dispatcher)
def det(a):
    """
    计算数组的行列式。

    Parameters
    ----------
    a : (..., M, M) array_like
        输入数组，计算其行列式。

    Returns
    -------
    det : (...) array_like
        输入数组的行列式。

    See Also
    --------
    np.linalg.slogdet

    """
    """
    计算输入数组或矩阵的行列式。

    Parameters
    ----------
    a : array_like
        输入的数组或矩阵，可以是复数类型。
    
    Returns
    -------
    r : ndarray
        返回计算得到的行列式值，类型与输入数组或矩阵的类型相同。

    Raises
    ------
    LinAlgError
        如果输入数组不是二维的或不是方阵，则会引发此异常。

    See Also
    --------
    numpy.linalg.slogdet : 另一种表示行列式的方式，适用于可能出现溢出/下溢的大矩阵。
    scipy.linalg.det : SciPy 中类似的行列式计算函数。

    Notes
    -----
    添加于 NumPy 1.8.0 版本。

    使用广播规则，具体细节参见 `numpy.linalg` 文档。

    行列式通过 LU 分解（LAPACK 的 ``z/dgetrf``）计算。

    Examples
    --------
    计算二维数组 [[a, b], [c, d]] 的行列式，结果为 ad - bc：

    >>> a = np.array([[1, 2], [3, 4]])
    >>> np.linalg.det(a)
    -2.0 # 可能会有所不同

    对一组矩阵计算行列式：

    >>> a = np.array([ [[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]] ])
    >>> a.shape
    (3, 2, 2)
    >>> np.linalg.det(a)
    array([-2., -3., -8.])

    """
    a = asarray(a)
    _assert_stacked_2d(a)  # 确保输入是二维的堆叠数组
    _assert_stacked_square(a)  # 确保输入是方阵
    t, result_t = _commonType(a)  # 确定输入数组的通用类型和结果类型
    signature = 'D->D' if isComplexType(t) else 'd->d'  # 根据输入数组类型确定签名
    r = _umath_linalg.det(a, signature=signature)  # 调用底层的行列式计算函数
    r = r.astype(result_t, copy=False)  # 将计算得到的行列式值转换为结果类型
    return r  # 返回计算得到的行列式值
# Linear Least Squares

# 定义一个函数 _lstsq_dispatcher，用于分派参数 a, b 和可选参数 rcond
def _lstsq_dispatcher(a, b, rcond=None):
    # 返回元组 (a, b)
    return (a, b)

# 使用装饰器 array_function_dispatch 将 _lstsq_dispatcher 函数注册为 lstsq 的分派器
@array_function_dispatch(_lstsq_dispatcher)
# 定义函数 lstsq，用于求解线性矩阵方程的最小二乘解
def lstsq(a, b, rcond=None):
    r"""
    Return the least-squares solution to a linear matrix equation.

    Computes the vector `x` that approximately solves the equation
    ``a @ x = b``. The equation may be under-, well-, or over-determined
    (i.e., the number of linearly independent rows of `a` can be less than,
    equal to, or greater than its number of linearly independent columns).
    If `a` is square and of full rank, then `x` (but for round-off error)
    is the "exact" solution of the equation. Else, `x` minimizes the
    Euclidean 2-norm :math:`||b - ax||`. If there are multiple minimizing
    solutions, the one with the smallest 2-norm :math:`||x||` is returned.

    Parameters
    ----------
    a : (M, N) array_like
        "Coefficient" matrix.
    b : {(M,), (M, K)} array_like
        Ordinate or "dependent variable" values. If `b` is two-dimensional,
        the least-squares solution is calculated for each of the `K` columns
        of `b`.
    rcond : float, optional
        Cut-off ratio for small singular values of `a`.
        For the purposes of rank determination, singular values are treated
        as zero if they are smaller than `rcond` times the largest singular
        value of `a`.
        The default uses the machine precision times ``max(M, N)``.  Passing
        ``-1`` will use machine precision.

        .. versionchanged:: 2.0
            Previously, the default was ``-1``, but a warning was given that
            this would change.

    Returns
    -------
    x : {(N,), (N, K)} ndarray
        Least-squares solution. If `b` is two-dimensional,
        the solutions are in the `K` columns of `x`.
    residuals : {(1,), (K,), (0,)} ndarray
        Sums of squared residuals: Squared Euclidean 2-norm for each column in
        ``b - a @ x``.
        If the rank of `a` is < N or M <= N, this is an empty array.
        If `b` is 1-dimensional, this is a (1,) shape array.
        Otherwise the shape is (K,).
    rank : int
        Rank of matrix `a`.
    s : (min(M, N),) ndarray
        Singular values of `a`.

    Raises
    ------
    LinAlgError
        If computation does not converge.

    See Also
    --------
    scipy.linalg.lstsq : Similar function in SciPy.

    Notes
    -----
    If `b` is a matrix, then all array results are returned as matrices.

    Examples
    --------
    Fit a line, ``y = mx + c``, through some noisy data-points:

    >>> x = np.array([0, 1, 2, 3])
    >>> y = np.array([-1, 0.2, 0.9, 2.1])

    By examining the coefficients, we see that the line should have a
    gradient of roughly 1 and cut the y-axis at, more or less, -1.

    We can rewrite the line equation as ``y = Ap``, where ``A = [[x 1]]``
    and ``p = [[m], [c]]``.  Now use `lstsq` to solve for `p`:

    >>> A = np.vstack([x, np.ones(len(x))]).T
    >>> A
    # 定义一个二维数组
    array([[ 0.,  1.],
           [ 1.,  1.],
           [ 2.,  1.],
           [ 3.,  1.]])
    
    # 使用 np.linalg.lstsq 函数进行最小二乘法拟合
    >>> m, c = np.linalg.lstsq(A, y)[0]
    >>> m, c
    (1.0 -0.95) # 可能会有所不同

    # 绘制原始数据和拟合直线的图像
    >>> import matplotlib.pyplot as plt
    >>> _ = plt.plot(x, y, 'o', label='Original data', markersize=10)
    >>> _ = plt.plot(x, m*x + c, 'r', label='Fitted line')
    >>> _ = plt.legend()
    >>> plt.show()

    """
    # 将输入数组 a 和 b 转换为数组
    a, _ = _makearray(a)
    b, wrap = _makearray(b)
    # 检查 b 是否是一维数组，如果是则将其转换为二维数组
    is_1d = b.ndim == 1
    if is_1d:
        b = b[:, newaxis]
    # 检查 a 和 b 的维度是否满足要求
    _assert_2d(a, b)
    # 获取矩阵 a 的行数 m 和列数 n
    m, n = a.shape[-2:]
    # 获取矩阵 b 的行数 m2 和列数 n_rhs
    m2, n_rhs = b.shape[-2:]
    # 如果矩阵 a 和 b 的行数不匹配，则引发 LinAlgError 异常
    if m != m2:
        raise LinAlgError('Incompatible dimensions')

    # 获取输入数组 a 和 b 的公共数据类型，并获取其实数类型
    t, result_t = _commonType(a, b)
    result_real_t = _realType(result_t)

    # 如果 rcond 为 None，则设置其值为 t 类型的机器 epsilon 乘以 n 和 m 中的较大者
    if rcond is None:
        rcond = finfo(t).eps * max(n, m)

    # 根据 m 和 n 的大小选择适当的 gufunc 函数
    if m <= n:
        gufunc = _umath_linalg.lstsq_m
    else:
        gufunc = _umath_linalg.lstsq_n

    # 确定 gufunc 函数的签名
    signature = 'DDd->Ddid' if isComplexType(t) else 'ddd->ddid'
    if n_rhs == 0:
        # 当 n_rhs 为 0 时，lapack 无法处理，因此在该轴上分配数组的更大尺寸
        b = zeros(b.shape[:-2] + (m, n_rhs + 1), dtype=b.dtype)

    # 使用 errstate 设置异常处理上下文，调用 _raise_linalgerror_lstsq 函数处理错误
    with errstate(call=_raise_linalgerror_lstsq, invalid='call',
                  over='ignore', divide='ignore', under='ignore'):
        # 调用 gufunc 函数进行最小二乘解，返回解 x、残差 resids、秩 rank 和奇异值 s
        x, resids, rank, s = gufunc(a, b, rcond, signature=signature)
    if m == 0:
        # 如果 m 为 0，则将 x 的所有元素置为 0
        x[...] = 0
    if n_rhs == 0:
        # 如果 n_rhs 为 0，则移除添加的项
        x = x[..., :n_rhs]
        resids = resids[..., :n_rhs]

    # 移除添加的轴
    if is_1d:
        # 如果输入数组 b 是一维数组，则挤压 x 的轴
        x = x.squeeze(axis=-1)
        # 注意：不能挤压 resids 轴，以保持兼容性

    # 根据文档要求，如果秩不等于 n 或者 m 小于等于 n，则将 resids 设置为空数组
    if rank != n or m <= n:
        resids = array([], result_real_t)

    # 强制转换输出数组的类型
    s = s.astype(result_real_t, copy=False)
    resids = resids.astype(result_real_t, copy=False)
    # 复制 x，以释放 r_parts 中的内存
    x = x.astype(result_t, copy=True)
    return wrap(x), wrap(resids), rank, s
def _multi_svd_norm(x, row_axis, col_axis, op):
    """Compute a function of the singular values of the 2-D matrices in `x`.

    This is a private utility function used by `numpy.linalg.norm()`.

    Parameters
    ----------
    x : ndarray
        Input array containing 2-D matrices whose singular values will be computed.
    row_axis, col_axis : int
        The axes of `x` that hold the 2-D matrices.
    op : callable
        This should be either numpy.amin or `numpy.amax` or `numpy.sum`.
        Function to apply to the singular values.

    Returns
    -------
    result : float or ndarray
        If `x` is 2-D, the return value is a float representing the computed norm.
        If `x` is higher-dimensional, it returns an array with ``x.ndim - 2`` dimensions.
        The return values are either the minimum or maximum or sum of the
        singular values of the matrices, depending on whether `op`
        is `numpy.amin` or `numpy.amax` or `numpy.sum`.
    """
    # Move the specified axes to the end to prepare for singular value decomposition
    y = moveaxis(x, (row_axis, col_axis), (-2, -1))
    # Compute the singular values without computing U and V matrices
    result = op(svd(y, compute_uv=False), axis=-1)
    return result


def _norm_dispatcher(x, ord=None, axis=None, keepdims=None):
    # Simply returns the input `x` as a tuple
    return (x,)


@array_function_dispatch(_norm_dispatcher)
def norm(x, ord=None, axis=None, keepdims=False):
    """
    Matrix or vector norm.

    This function is able to return one of eight different matrix norms,
    or one of an infinite number of vector norms (described below), depending
    on the value of the ``ord`` parameter.

    Parameters
    ----------
    x : array_like
        Input array.  If `axis` is None, `x` must be 1-D or 2-D, unless `ord`
        is None. If both `axis` and `ord` are None, the 2-norm of
        ``x.ravel`` will be returned.
    ord : {non-zero int, inf, -inf, 'fro', 'nuc'}, optional
        Order of the norm (see table under ``Notes``). inf means numpy's
        `inf` object. The default is None.
    axis : {None, int, 2-tuple of ints}, optional.
        If `axis` is an integer, it specifies the axis of `x` along which to
        compute the vector norms.  If `axis` is a 2-tuple, it specifies the
        axes that hold 2-D matrices, and the matrix norms of these matrices
        are computed.  If `axis` is None then either a vector norm (when `x`
        is 1-D) or a matrix norm (when `x` is 2-D) is returned. The default
        is None.

        .. versionadded:: 1.8.0

    keepdims : bool, optional
        If this is set to True, the axes which are normed over are left in the
        result as dimensions with size one.  With this option the result will
        broadcast correctly against the original `x`.

        .. versionadded:: 1.10.0

    Returns
    -------
    n : float or ndarray
        Norm of the matrix or vector(s).

    See Also
    --------
    scipy.linalg.norm : Similar function in SciPy.

    Notes
    -----
    For values of ``ord < 1``, the result is, strictly speaking, not a
    mathematical 'norm', but it may still be useful for various numerical
    purposes.

    The following norms can be calculated:

    =====  ============================  ==========================
    """
    ord    norm for matrices             norm for vectors
    =====  ============================  ==========================
    None   Frobenius norm                2-norm
    'fro'  Frobenius norm                --
    'nuc'  nuclear norm                  --
    inf    max(sum(abs(x), axis=1))      max(abs(x))
    -inf   min(sum(abs(x), axis=1))      min(abs(x))
    0      --                            sum(x != 0)
    1      max(sum(abs(x), axis=0))      as below
    -1     min(sum(abs(x), axis=0))      as below
    2      2-norm (largest sing. value)  as below
    -2     smallest singular value       as below
    other  --                            sum(abs(x)**ord)**(1./ord)
    =====  ============================  ==========================

    The Frobenius norm is given by [1]_:

    :math:`||A||_F = [\\sum_{i,j} abs(a_{i,j})^2]^{1/2}`

    The nuclear norm is the sum of the singular values.

    Both the Frobenius and nuclear norm orders are only defined for
    matrices and raise a ValueError when ``x.ndim != 2``.

    References
    ----------
    .. [1] G. H. Golub and C. F. Van Loan, *Matrix Computations*,
           Baltimore, MD, Johns Hopkins University Press, 1985, pg. 15

    Examples
    --------
    >>> from numpy import linalg as LA
    >>> a = np.arange(9) - 4
    >>> a
    array([-4, -3, -2, ...,  2,  3,  4])
    >>> b = a.reshape((3, 3))
    >>> b
    array([[-4, -3, -2],
           [-1,  0,  1],
           [ 2,  3,  4]])

    >>> LA.norm(a)
    7.745966692414834
    >>> LA.norm(b)
    7.745966692414834
    >>> LA.norm(b, 'fro')
    7.745966692414834
    >>> LA.norm(a, np.inf)
    4.0
    >>> LA.norm(b, np.inf)
    9.0
    >>> LA.norm(a, -np.inf)
    0.0
    >>> LA.norm(b, -np.inf)
    2.0

    >>> LA.norm(a, 1)
    20.0
    >>> LA.norm(b, 1)
    7.0
    >>> LA.norm(a, -1)
    -4.6566128774142013e-010
    >>> LA.norm(b, -1)
    6.0
    >>> LA.norm(a, 2)
    7.745966692414834
    >>> LA.norm(b, 2)
    7.3484692283495345

    >>> LA.norm(a, -2)
    0.0
    >>> LA.norm(b, -2)
    1.8570331885190563e-016 # may vary
    >>> LA.norm(a, 3)
    5.8480354764257312 # may vary
    >>> LA.norm(a, -3)
    0.0

    Using the `axis` argument to compute vector norms:

    >>> c = np.array([[ 1, 2, 3],
    ...               [-1, 1, 4]])
    >>> LA.norm(c, axis=0)
    array([ 1.41421356,  2.23606798,  5.        ])
    >>> LA.norm(c, axis=1)
    array([ 3.74165739,  4.24264069])
    >>> LA.norm(c, ord=1, axis=1)
    array([ 6.,  6.])

    Using the `axis` argument to compute matrix norms:

    >>> m = np.arange(8).reshape(2,2,2)
    >>> LA.norm(m, axis=(1,2))
    array([  3.74165739,  11.22497216])
    >>> LA.norm(m[0, :, :]), LA.norm(m[1, :, :])
    (3.7416573867739413, 11.224972160321824)

    """
    x = asarray(x)  # 将输入 x 转换为 numpy 数组

    if not issubclass(x.dtype.type, (inexact, object_)):
        x = x.astype(float)  # 如果 x 不是浮点数或对象类型，则转换为浮点数类型

    # 立即处理一些默认、简单、快速和常见的情况。
    # 如果 axis 参数为 None，则需要根据 x 的维度确定操作
    if axis is None:
        # 获取 x 的维度
        ndim = x.ndim
        # 检查 ord 参数是否为 None 或者符合特定条件，以确定是否需要进行向量展平操作
        if (
            (ord is None) or
            (ord in ('f', 'fro') and ndim == 2) or
            (ord == 2 and ndim == 1)
        ):
            # 将 x 按 'K'（元素在内存中的存储顺序）展平
            x = x.ravel(order='K')
            # 如果 x 的数据类型为复数类型，则分别获取实部和虚部
            if isComplexType(x.dtype.type):
                x_real = x.real
                x_imag = x.imag
                # 计算复数的模的平方
                sqnorm = x_real.dot(x_real) + x_imag.dot(x_imag)
            else:
                # 计算向量的内积
                sqnorm = x.dot(x)
            # 计算向量的二范数
            ret = sqrt(sqnorm)
            # 如果 keepdims 为 True，则将结果调整为原始维度形状
            if keepdims:
                ret = ret.reshape(ndim*[1])
            return ret

    # 将 axis 参数标准化为一个元组
    nd = x.ndim
    if axis is None:
        axis = tuple(range(nd))
    elif not isinstance(axis, tuple):
        try:
            axis = int(axis)
        except Exception as e:
            # 如果无法转换为整数，则抛出类型错误
            raise TypeError(
                "'axis' must be None, an integer or a tuple of integers"
            ) from e
        # 将整数转换为包含该整数的元组
        axis = (axis,)

    # 如果 axis 的长度为 1
    if len(axis) == 1:
        # 根据 ord 的不同取值进行不同的向量范数计算
        if ord == inf:
            # 计算向量的无穷范数（绝对值最大值）
            return abs(x).max(axis=axis, keepdims=keepdims)
        elif ord == -inf:
            # 计算向量的负无穷范数（绝对值最小值）
            return abs(x).min(axis=axis, keepdims=keepdims)
        elif ord == 0:
            # 计算向量的零范数
            return (
                (x != 0)
                .astype(x.real.dtype)
                .sum(axis=axis, keepdims=keepdims)
            )
        elif ord == 1:
            # 计算向量的一范数
            return add.reduce(abs(x), axis=axis, keepdims=keepdims)
        elif ord is None or ord == 2:
            # 计算向量的二范数
            s = (x.conj() * x).real
            return sqrt(add.reduce(s, axis=axis, keepdims=keepdims))
        # 如果 ord 是字符串类型且不是有效的向量范数关键词
        elif isinstance(ord, str):
            raise ValueError(f"Invalid norm order '{ord}' for vectors")
        else:
            # 计算向量的自定义范数
            absx = abs(x)
            absx **= ord
            ret = add.reduce(absx, axis=axis, keepdims=keepdims)
            ret **= reciprocal(ord, dtype=ret.dtype)
            return ret
    # 如果轴数为2，则进入这个分支处理
    elif len(axis) == 2:
        # 解包轴数
        row_axis, col_axis = axis
        # 规范化行轴和列轴的索引
        row_axis = normalize_axis_index(row_axis, nd)
        col_axis = normalize_axis_index(col_axis, nd)
        # 如果行轴和列轴相同，则抛出数值错误异常
        if row_axis == col_axis:
            raise ValueError('Duplicate axes given.')
        # 根据不同的ord值执行不同的操作
        if ord == 2:
            # 计算二范数
            ret = _multi_svd_norm(x, row_axis, col_axis, amax)
        elif ord == -2:
            # 计算最小的奇异值
            ret = _multi_svd_norm(x, row_axis, col_axis, amin)
        elif ord == 1:
            # 计算1-范数
            if col_axis > row_axis:
                col_axis -= 1
            # 沿着行轴求和后，再沿着列轴找到最大值
            ret = add.reduce(abs(x), axis=row_axis).max(axis=col_axis)
        elif ord == inf:
            # 计算无穷范数
            if row_axis > col_axis:
                row_axis -= 1
            # 沿着列轴求和后，再沿着行轴找到最大值
            ret = add.reduce(abs(x), axis=col_axis).max(axis=row_axis)
        elif ord == -1:
            # 计算负1-范数
            if col_axis > row_axis:
                col_axis -= 1
            # 沿着行轴求和后，再沿着列轴找到最小值
            ret = add.reduce(abs(x), axis=row_axis).min(axis=col_axis)
        elif ord == -inf:
            # 计算负无穷范数
            if row_axis > col_axis:
                row_axis -= 1
            # 沿着列轴求和后，再沿着行轴找到最小值
            ret = add.reduce(abs(x), axis=col_axis).min(axis=row_axis)
        elif ord in [None, 'fro', 'f']:
            # 计算Frobenius范数（矩阵的F范数）
            ret = sqrt(add.reduce((x.conj() * x).real, axis=axis))
        elif ord == 'nuc':
            # 计算核范数
            ret = _multi_svd_norm(x, row_axis, col_axis, sum)
        else:
            # 如果ord不在预期范围内，抛出数值错误异常
            raise ValueError("Invalid norm order for matrices.")
        # 如果keepdims为True，则保持结果的维度
        if keepdims:
            ret_shape = list(x.shape)
            ret_shape[axis[0]] = 1
            ret_shape[axis[1]] = 1
            ret = ret.reshape(ret_shape)
        # 返回计算得到的结果
        return ret
    else:
        # 如果轴数不为1或2，则抛出数值错误异常
        raise ValueError("Improper number of dimensions to norm.")
# 定义函数 _multidot_dispatcher，用于生成器分发输入数组及输出参数
def _multidot_dispatcher(arrays, *, out=None):
    # 生成器返回输入的数组和输出参数（如果有）
    yield from arrays
    yield out

# 使用 array_function_dispatch 装饰器，将 _multidot_dispatcher 分发到 multi_dot 函数
@array_function_dispatch(_multidot_dispatcher)
def multi_dot(arrays, *, out=None):
    """
    一次性计算两个或多个数组的点乘积，并自动选择最快的计算顺序。

    `multi_dot` 链接 `numpy.dot` 并使用最佳的矩阵括号化方法 [1]_ [2]_。
    根据矩阵的形状，这种方法可以大大加速乘法运算。

    如果第一个参数是 1-D 数组，则被视为行向量。
    如果最后一个参数是 1-D 数组，则被视为列向量。
    其他参数必须是 2-D 数组。

    可以把 `multi_dot` 理解为::

        def multi_dot(arrays): return functools.reduce(np.dot, arrays)

    Parameters
    ----------
    arrays : sequence of array_like
        如果第一个参数是 1-D 数组，则被视为行向量。
        如果最后一个参数是 1-D 数组，则被视为列向量。
        其他参数必须是 2-D 数组。
    out : ndarray, optional
        输出参数。必须与返回的数组类型完全相同，必须是 C 连续的，其数据类型必须与 `dot(a, b)` 返回的数据类型一致。
        这是一个性能特性。因此，如果不满足这些条件，会引发异常，而不是尝试灵活处理。

        .. versionadded:: 1.19.0

    Returns
    -------
    output : ndarray
        返回输入数组的点乘积。

    See Also
    --------
    numpy.dot : 两个数组的点乘积。

    References
    ----------

    .. [1] Cormen, "Introduction to Algorithms", Chapter 15.2, p. 370-378
    .. [2] https://en.wikipedia.org/wiki/Matrix_chain_multiplication

    Examples
    --------
    `multi_dot` 允许您这样写::

    >>> from numpy.linalg import multi_dot
    >>> # 准备一些数据
    >>> A = np.random.random((10000, 100))
    >>> B = np.random.random((100, 1000))
    >>> C = np.random.random((1000, 5))
    >>> D = np.random.random((5, 333))
    >>> # 实际的点乘积
    >>> _ = multi_dot([A, B, C, D])

    而不是::

    >>> _ = np.dot(np.dot(np.dot(A, B), C), D)
    >>> # 或者
    >>> _ = A.dot(B).dot(C).dot(D)

    Notes
    -----
    可以使用以下函数计算矩阵乘法的成本::

        def cost(A, B):
            return A.shape[0] * A.shape[1] * B.shape[1]

    假设我们有三个矩阵
    :math:`A_{10x100}, B_{100x5}, C_{5x50}`。

    两种不同括号化的成本如下所示::

        cost((AB)C) = 10*100*5 + 10*5*50   = 5000 + 2500   = 7500
        cost(A(BC)) = 10*100*50 + 100*5*50 = 50000 + 25000 = 75000

    """
    n = len(arrays)
    # 优化只有在数组长度大于 2 时才有意义
    if n < 2:
        raise ValueError("Expecting at least two arrays.")
    # 如果 n 等于 2，则调用 dot 函数计算两个数组的点积，结果存入 out 中并返回
    elif n == 2:
        return dot(arrays[0], arrays[1], out=out)

    # 将数组列表中的每个数组转换为 asanyarray 类型
    arrays = [asanyarray(a) for a in arrays]

    # 保存原始的 ndim 属性，以便稍后将结果数组重塑为正确的形式
    ndim_first, ndim_last = arrays[0].ndim, arrays[-1].ndim

    # 如果第一个数组是一维的，则将其至少转换为二维数组，以简化内部 _multi_dot_* 函数的逻辑
    if arrays[0].ndim == 1:
        arrays[0] = atleast_2d(arrays[0])

    # 如果最后一个数组是一维的，则将其至少转换为转置后的二维数组
    if arrays[-1].ndim == 1:
        arrays[-1] = atleast_2d(arrays[-1]).T

    # 确保所有数组都是二维的
    _assert_2d(*arrays)

    # 如果 n 等于 3，则调用 _multi_dot_three 函数计算三个数组的多重点积
    # 否则调用 _multi_dot 函数计算多个数组的多重点积，使用矩阵链的顺序
    if n == 3:
        result = _multi_dot_three(arrays[0], arrays[1], arrays[2], out=out)
    else:
        order = _multi_dot_matrix_chain_order(arrays)
        result = _multi_dot(arrays, order, 0, n - 1, out=out)

    # 返回结果数组的正确形状
    if ndim_first == 1 and ndim_last == 1:
        return result[0, 0]  # 返回标量（0,0）位置的值
    elif ndim_first == 1 or ndim_last == 1:
        return result.ravel()  # 返回扁平化的结果数组（一维数组）
    else:
        return result
# 定义一个函数 `_multi_dot_three`，用于计算三个矩阵的最佳乘法顺序并执行乘法操作。
# 参数 A, B, C 是三个矩阵，out 是可选的输出矩阵。
def _multi_dot_three(A, B, C, out=None):
    # 获取矩阵 A 的形状信息
    a0, a1b0 = A.shape
    # 获取矩阵 C 的形状信息
    b1c0, c1 = C.shape
    # 计算第一种乘法顺序 (AB)C 的代价
    cost1 = a0 * b1c0 * (a1b0 + c1)
    # 计算第二种乘法顺序 A(BC) 的代价
    cost2 = a1b0 * c1 * (a0 + b1c0)

    # 比较两种乘法顺序的代价，选择代价较小的顺序进行矩阵乘法操作
    if cost1 < cost2:
        return dot(dot(A, B), C, out=out)  # 返回 (AB)C 的乘法结果
    else:
        return dot(A, dot(B, C), out=out)  # 返回 A(BC) 的乘法结果


# 定义一个函数 `_multi_dot_matrix_chain_order`，用于计算多个矩阵乘法的最佳顺序。
# 参数 arrays 是包含多个矩阵的列表，return_costs 指定是否返回代价矩阵。
def _multi_dot_matrix_chain_order(arrays, return_costs=False):
    # 获取矩阵数量
    n = len(arrays)
    # p 存储每个矩阵的维度信息，最后一个元素是最后一个矩阵的列数
    p = [a.shape[0] for a in arrays] + [arrays[-1].shape[1]]
    # m 是子问题的代价矩阵，初始化为零矩阵
    m = zeros((n, n), dtype=double)
    # s 是实际的乘法顺序矩阵，初始化为空矩阵
    s = empty((n, n), dtype=intp)

    # 根据动态规划算法计算最优乘法顺序和代价
    for l in range(1, n):
        for i in range(n - l):
            j = i + l
            m[i, j] = inf
            for k in range(i, j):
                # 计算以 k 为分割点的乘法代价
                q = m[i, k] + m[k+1, j] + p[i] * p[k+1] * p[j+1]
                if q < m[i, j]:
                    m[i, j] = q
                    s[i, j] = k  # 注意 Cormen 使用的是 1-based 索引

    # 如果指定返回代价矩阵，则同时返回代价矩阵和乘法顺序矩阵
    return (s, m) if return_costs else s


# 定义一个函数 `_multi_dot`，根据给定的乘法顺序进行矩阵乘法操作。
# 参数 arrays 是矩阵列表，order 是乘法顺序矩阵，i, j 指定当前乘法的起始和结束索引，out 是可选的输出矩阵。
def _multi_dot(arrays, order, i, j, out=None):
    # 如果起始索引等于结束索引，说明已经到达最小子问题，直接返回对应的矩阵
    if i == j:
        # 初始调用时，out 不应为非 None，这里使用断言确保其为 None
        assert out is None
        return arrays[i]
    else:
        # 否则，根据乘法顺序递归进行矩阵乘法操作
        return dot(_multi_dot(arrays, order, i, order[i, j]),
                   _multi_dot(arrays, order, order[i, j] + 1, j),
                   out=out)


# 定义一个函数 `_diagonal_dispatcher`，用于将参数 x 传递给 diagonal 函数。
# 参数 x 是输入的矩阵或者矩阵栈，offset 是可选的对角线偏移量。
def _diagonal_dispatcher(x, /, *, offset=None):
    return (x,)


# 使用 array_function_dispatch 装饰器，将 _diagonal_dispatcher 函数注册为 diagonal 函数的分派函数。
@array_function_dispatch(_diagonal_dispatcher)
# 定义 diagonal 函数，用于返回矩阵（或者矩阵栈）的指定对角线。
# 这个函数兼容 Array API，与 numpy.diagonal 不同，此处矩阵假定由最后两个维度定义。
def diagonal(x, /, *, offset=0):
    """
    Returns specified diagonals of a matrix (or a stack of matrices) ``x``.

    This function is Array API compatible, contrary to
    :py:func:`numpy.diagonal`, the matrix is assumed
    to be defined by the last two dimensions.

    Parameters
    ----------
    x : array_like
        Input array or stack of matrices.
    offset : int, optional
        Diagonal offset. Default is 0.

    """
    x : (...,M,N) array_like
        Input array having shape (..., M, N) and whose innermost two
        dimensions form MxN matrices.
    offset : int, optional
        Offset specifying the off-diagonal relative to the main diagonal,
        where::

            * offset = 0: the main diagonal.
            * offset > 0: off-diagonal above the main diagonal.
            * offset < 0: off-diagonal below the main diagonal.

    Returns
    -------
    out : (...,min(N,M)) ndarray
        An array containing the diagonals and whose shape is determined by
        removing the last two dimensions and appending a dimension equal to
        the size of the resulting diagonals. The returned array must have
        the same data type as ``x``.

    See Also
    --------
    numpy.diagonal

    """
    # 调用核心函数 _core_diagonal 来提取指定偏移的对角线元素
    return _core_diagonal(x, offset, axis1=-2, axis2=-1)
# trace

# 定义了一个私有函数 _trace_dispatcher，用于调度 trace 函数的参数
def _trace_dispatcher(x, /, *, offset=None, dtype=None):
    return (x,)

# 使用装饰器 array_function_dispatch 将 _trace_dispatcher 与 trace 函数关联，
# 用于 Array API 的兼容性
@array_function_dispatch(_trace_dispatcher)
def trace(x, /, *, offset=0, dtype=None):
    """
    返回矩阵（或矩阵堆栈）``x`` 沿指定对角线的和。

    与 numpy.trace 不同，此函数与 Array API 兼容。

    Parameters
    ----------
    x : (...,M,N) array_like
        输入数组，形状为 (..., M, N)，其最内层两个维度形成 MxN 的矩阵。
    offset : int, optional
        相对于主对角线的偏移，指定非主对角线的位置，
        其中:
            * offset = 0: 主对角线。
            * offset > 0: 主对角线之上的非主对角线。
            * offset < 0: 主对角线之下的非主对角线。
    dtype : dtype, optional
        返回数组的数据类型。

    Returns
    -------
    out : ndarray
        包含跟踪值的数组，其形状由去除最后两个维度确定，并在最后一个数组维度中存储跟踪值。
        例如，如果 x 的秩为 k 并且形状为 (I, J, K, ..., L, M, N)，则输出数组的秩为 k-2，
        形状为 (I, J, K, ..., L)，其中:
            out[i, j, k, ..., l] = trace(a[i, j, k, ..., l, :, :])
        
        返回的数组必须具有上述 dtype 参数描述的数据类型。

    See Also
    --------
    numpy.trace

    """
    # 调用核心函数 _core_trace，对 x 进行跟踪操作，指定 axis1 和 axis2 为 -2 和 -1
    return _core_trace(x, offset, axis1=-2, axis2=-1, dtype=dtype)


# cross

# 定义了一个私有函数 _cross_dispatcher，用于调度 cross 函数的参数
def _cross_dispatcher(x1, x2, /, *, axis=None):
    return (x1, x2,)

# 使用装饰器 array_function_dispatch 将 _cross_dispatcher 与 cross 函数关联，
# 用于 Array API 的兼容性
@array_function_dispatch(_cross_dispatcher)
def cross(x1, x2, /, *, axis=-1):
    """
    返回三维向量的叉积。

    如果 ``x1`` 和/或 ``x2`` 是多维数组，则独立计算每对对应的三维向量的叉积。

    与 numpy.cross 不同，此函数与 Array API 兼容。

    Parameters
    ----------
    x1 : array_like
        第一个输入数组。
    x2 : array_like
        第二个输入数组。必须与 ``x1`` 在所有非计算轴上兼容。
        在计算叉积的轴上，大小必须与 ``x1`` 中的相应轴大小相同。
    axis : int, optional
        包含要计算叉积的向量的轴（维度）。默认为 ``-1``。

    Returns
    -------
    out : ndarray
        包含叉积的数组。

    See Also
    --------
    numpy.cross

    """
    # 将 x1 和 x2 转换为 asanyarray，确保是数组对象
    x1 = asanyarray(x1)
    x2 = asanyarray(x2)

    # 如果 x1 和 x2 在指定的轴上不是三维向量，则引发 ValueError
    if x1.shape[axis] != 3 or x2.shape[axis] != 3:
        raise ValueError(
            "Both input arrays must be (arrays of) 3-dimensional vectors, "
            f"but they are {x1.shape[axis]} and {x2.shape[axis]} "
            "dimensional instead."
        )
    # 调用名为 _core_cross 的函数，将 x1 和 x2 作为参数传入，并指定 axis 参数为当前作用域中定义的 axis 变量的值。
    return _core_cross(x1, x2, axis=axis)
# matmul

# 创建一个分派器函数，用于确定正确的实现来处理矩阵乘法
def _matmul_dispatcher(x1, x2, /):
    return (x1, x2)


# 使用分派器装饰器，将矩阵乘法函数与Array API兼容起来
@array_function_dispatch(_matmul_dispatcher)
def matmul(x1, x2, /):
    """
    Computes the matrix product.

    This function is Array API compatible, contrary to
    :func:`numpy.matmul`.

    Parameters
    ----------
    x1 : array_like
        The first input array.
    x2 : array_like
        The second input array.

    Returns
    -------
    out : ndarray
        The matrix product of the inputs.
        This is a scalar only when both ``x1``, ``x2`` are 1-d vectors.

    Raises
    ------
    ValueError
        If the last dimension of ``x1`` is not the same size as
        the second-to-last dimension of ``x2``.

        If a scalar value is passed in.

    See Also
    --------
    numpy.matmul

    """
    # 调用核心矩阵乘法函数来执行实际计算
    return _core_matmul(x1, x2)


# tensordot

# 创建一个分派器函数，用于确定正确的实现来处理张量点积
def _tensordot_dispatcher(x1, x2, /, *, axes=None):
    return (x1, x2)


# 使用分派器装饰器，将张量点积函数与Array API兼容起来
@array_function_dispatch(_tensordot_dispatcher)
def tensordot(x1, x2, /, *, axes=2):
    # 调用核心张量点积函数来执行实际计算
    return _core_tensordot(x1, x2, axes=axes)


# 将张量点积函数的文档字符串设置为核心张量点积函数的文档字符串
tensordot.__doc__ = _core_tensordot.__doc__


# matrix_transpose

# 创建一个分派器函数，用于确定正确的实现来处理矩阵转置
def _matrix_transpose_dispatcher(x):
    return (x,)


# 使用分派器装饰器，将矩阵转置函数与Array API兼容起来
@array_function_dispatch(_matrix_transpose_dispatcher)
def matrix_transpose(x, /):
    # 调用核心矩阵转置函数来执行实际计算
    return _core_matrix_transpose(x)


# 将矩阵转置函数的文档字符串设置为核心矩阵转置函数的文档字符串
matrix_transpose.__doc__ = _core_matrix_transpose.__doc__


# matrix_norm

# 创建一个分派器函数，用于确定正确的实现来处理矩阵范数
def _matrix_norm_dispatcher(x, /, *, keepdims=None, ord=None):
    return (x,)


# 使用分派器装饰器，将矩阵范数函数与Array API兼容起来
@array_function_dispatch(_matrix_norm_dispatcher)
def matrix_norm(x, /, *, keepdims=False, ord="fro"):
    """
    Computes the matrix norm of a matrix (or a stack of matrices) ``x``.

    This function is Array API compatible.

    Parameters
    ----------
    x : array_like
        Input array having shape (..., M, N) and whose two innermost
        dimensions form ``MxN`` matrices.
    keepdims : bool, optional
        If this is set to True, the axes which are normed over are left in
        the result as dimensions with size one. Default: False.
    ord : {1, -1, 2, -2, inf, -inf, 'fro', 'nuc'}, optional
        The order of the norm. For details see the table under ``Notes``
        in `numpy.linalg.norm`.

    See Also
    --------
    numpy.linalg.norm : Generic norm function

    """
    # 将输入转换为数组
    x = asanyarray(x)
    # 调用通用范数函数来计算矩阵范数
    return norm(x, axis=(-2, -1), keepdims=keepdims, ord=ord)


# vector_norm

# 创建一个分派器函数，用于确定正确的实现来处理向量范数
def _vector_norm_dispatcher(x, /, *, axis=None, keepdims=None, ord=None):
    return (x,)


# 使用分派器装饰器，将向量范数函数与Array API兼容起来
@array_function_dispatch(_vector_norm_dispatcher)
def vector_norm(x, /, *, axis=None, keepdims=False, ord=2):
    """
    Computes the vector norm of a vector (or batch of vectors) ``x``.

    This function is Array API compatible.

    Parameters
    ----------
    x : array_like
        Input array.

    """
    # 实际计算向量范数的部分由底层的norm函数处理
    pass  # Placeholder, as the actual computation is handled by the underlying `norm` function
    # 将输入数组 x 转换为 ndarray 类型
    x = asanyarray(x)
    # 获取输入数组 x 的形状，并转换为列表
    shape = list(x.shape)
    # 如果 axis 参数为 None
    if axis is None:
        # 注意：np.linalg.norm() 不处理 0 维数组
        # 将 x 展平为一维数组
        x = x.ravel()
        # 设置 _axis 为 0
        _axis = 0
    # 如果 axis 参数是一个元组
    elif isinstance(axis, tuple):
        # 注意：axis 参数支持任意数量的轴，而 np.linalg.norm() 只支持单个轴用于向量范数
        # 标准化 axis 参数，确保在 x 的维度范围内
        normalized_axis = normalize_axis_tuple(axis, x.ndim)
        # 计算除了标准化后的轴之外的其余轴
        rest = tuple(i for i in range(x.ndim) if i not in normalized_axis)
        # 创建新的形状，将标准化后的轴放在前面，其余轴放在后面
        newshape = axis + rest
        # 对 x 进行转置操作，使得标准化后的轴在前面
        x = _core_transpose(x, newshape).reshape(
            (
                prod([x.shape[i] for i in axis], dtype=int),
                *[x.shape[i] for i in rest]
            )
        )
        # 设置 _axis 为 0
        _axis = 0
    else:
        # 否则，直接使用输入的 axis 参数
        _axis = axis

    # 调用 numpy.linalg.norm 计算 x 的范数，使用指定的轴和顺序 ord
    res = norm(x, axis=_axis, ord=ord)

    # 如果 keepdims 参数为 True
    if keepdims:
        # 由于上面的 reshape 操作避免了矩阵范数逻辑，无法重用 np.linalg.norm(keepdims)
        # 标准化 _axis 参数，确保在 shape 的长度范围内
        _axis = normalize_axis_tuple(
            range(len(shape)) if axis is None else axis, len(shape)
        )
        # 将被标准化的轴设置为大小为 1
        for i in _axis:
            shape[i] = 1
        # 将 res 调整为新的形状
        res = res.reshape(tuple(shape))

    # 返回计算得到的范数结果
    return res
# vecdot

# 定义 _vecdot_dispatcher 函数，用于分发参数 x1, x2，以及可选参数 axis
def _vecdot_dispatcher(x1, x2, /, *, axis=None):
    return (x1, x2)

# 使用 array_function_dispatch 装饰器将 _vecdot_dispatcher 函数应用到 vecdot 函数上
@array_function_dispatch(_vecdot_dispatcher)
# 定义 vecdot 函数，计算向量的点积
def vecdot(x1, x2, /, *, axis=-1):
    """
    计算向量的点积。

    与 numpy.vecdot 不同，此函数限制参数与数组API兼容。

    设 :math:`\\mathbf{a}` 为 ``x1`` 中的向量，:math:`\\mathbf{b}` 为 ``x2`` 中的对应向量。
    点积定义为：

    .. math::
       \\mathbf{a} \\cdot \\mathbf{b} = \\sum_{i=0}^{n-1} \\overline{a_i}b_i

    其中 :math:`\\overline{a_i}` 表示如果 :math:`a_i` 是复数则为其复共轭，否则为其本身。

    参数
    ----------
    x1 : array_like
        第一个输入数组。
    x2 : array_like
        第二个输入数组。
    axis : int, optional
        计算点积的轴。默认为 ``-1``。

    返回
    -------
    output : ndarray
        输入的向量点积。

    参见
    --------
    numpy.vecdot

    """
    # 调用内部函数 _core_vecdot，计算向量点积，传递轴参数
    return _core_vecdot(x1, x2, axis=axis)
```