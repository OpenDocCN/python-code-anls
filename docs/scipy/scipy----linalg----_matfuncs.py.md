# `D:\src\scipysrc\scipy\scipy\linalg\_matfuncs.py`

```
# 作者信息，Travis Oliphant，编写于2002年3月
from itertools import product

import numpy as np
from numpy import (dot, diag, prod, logical_not, ravel, transpose,
                   conjugate, absolute, amax, sign, isfinite, triu)

# 本地导入
from scipy.linalg import LinAlgError, bandwidth
from ._misc import norm
from ._basic import solve, inv
from ._decomp_svd import svd
from ._decomp_schur import schur, rsf2csf
from ._expm_frechet import expm_frechet, expm_cond
from ._matfuncs_sqrtm import sqrtm
from ._matfuncs_expm import pick_pade_structure, pade_UV_calc

# 所有公开的函数和类的列表
__all__ = ['expm', 'cosm', 'sinm', 'tanm', 'coshm', 'sinhm', 'tanhm', 'logm',
           'funm', 'signm', 'sqrtm', 'fractional_matrix_power', 'expm_frechet',
           'expm_cond', 'khatri_rao']

# 浮点数的机器精度
eps = np.finfo('d').eps
feps = np.finfo('f').eps

# 数组精度字典
_array_precision = {'i': 1, 'l': 1, 'f': 0, 'd': 1, 'F': 0, 'D': 1}


###############################################################################
# 实用函数


def _asarray_square(A):
    """
    用额外的要求包装asarray，要求输入是方阵。

    动机是matfuncs模块包含了被提升为方阵函数的实函数。

    参数
    ----------
    A : array_like
        方阵。

    返回
    -------
    out : ndarray
        A的ndarray副本、视图或其他表示形式。

    """
    # 将输入转换为ndarray
    A = np.asarray(A)
    # 如果A不是二维数组或者不是方阵，抛出值错误异常
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected square array_like input')
    return A


def _maybe_real(A, B, tol=None):
    """
    根据A和B的属性返回B或B的实部。

    动机是B作为A的复杂函数计算的结果，
    B可能被微不足道的虚部扰动。
    如果A是实数而B是带有小虚部分量的复数，
    则返回B的实数副本。
    在这种情况下的假设是B的虚部是数值上的人为因素。

    参数
    ----------
    A : ndarray
        要检查类型的输入数组，是实数还是复数。
    B : ndarray
        要返回的数组，可能没有其虚部。
    tol : float
        绝对容差。

    返回
    -------
    out : 实数或复数数组
        输入数组B或其实数部分。

    """
    # 注意布尔值和整数被视为实数。
    if np.isrealobj(A) and np.iscomplexobj(B):
        # 如果未指定容差，则使用特定精度的默认容差
        if tol is None:
            tol = {0: feps*1e3, 1: eps*1e6}[_array_precision[B.dtype.char]]
        # 如果B的虚部在给定容差下全为0，则将B设为实部
        if np.allclose(B.imag, 0.0, atol=tol):
            B = B.real
    return B


###############################################################################
# 矩阵函数


def fractional_matrix_power(A, t):
    """
    计算矩阵的分数幂。

    根据文献[1]_第6节的讨论进行计算。

    参数
    ----------
    A : ndarray
        输入矩阵。
    t : float
        幂指数。

    返回
    -------
    out : ndarray
        A的分数幂矩阵。

    """
    # A 是一个形状为 (N, N) 的数组或类似数组，表示待计算的矩阵
    A : (N, N) array_like
        Matrix whose fractional power to evaluate.
    
    # t 是一个浮点数，表示所需计算的矩阵的分数幂
    t : float
        Fractional power.
    
    # 返回一个形状为 (N, N) 的数组，表示输入矩阵的分数幂
    Returns
    -------
    X : (N, N) array_like
        The fractional power of the matrix.
    
    # 引用文献说明了算法来源和详细信息
    References
    ----------
    .. [1] Nicholas J. Higham and Lijing lin (2011)
           "A Schur-Pade Algorithm for Fractional Powers of a Matrix."
           SIAM Journal on Matrix Analysis and Applications,
           32 (3). pp. 1056-1078. ISSN 0895-4798
    
    # 示例展示了如何使用 fractional_matrix_power 函数计算矩阵的分数幂
    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import fractional_matrix_power
    >>> a = np.array([[1.0, 3.0], [1.0, 4.0]])
    >>> b = fractional_matrix_power(a, 0.5)
    >>> b
    array([[ 0.75592895,  1.13389342],
           [ 0.37796447,  1.88982237]])
    >>> np.dot(b, b)      # Verify square root
    array([[ 1.,  3.],
           [ 1.,  4.]])
    
    """
    # 修复一些导入的问题；
    # 这个函数调用了 onenormest，它位于 scipy.sparse 中。
    A = _asarray_square(A)
    import scipy.linalg._matfuncs_inv_ssq
    # 调用 scipy.linalg._matfuncs_inv_ssq._fractional_matrix_power 计算输入矩阵的分数幂并返回结果
    return scipy.linalg._matfuncs_inv_ssq._fractional_matrix_power(A, t)
def logm(A, disp=True):
    """
    Compute matrix logarithm.

    The matrix logarithm is the inverse of
    expm: expm(logm(`A`)) == `A`

    Parameters
    ----------
    A : (N, N) array_like
        Matrix whose logarithm to evaluate
    disp : bool, optional
        Print warning if error in the result is estimated large
        instead of returning estimated error. (Default: True)

    Returns
    -------
    logm : (N, N) ndarray
        Matrix logarithm of `A`
    errest : float
        (if disp == False)

        1-norm of the estimated error, ||err||_1 / ||A||_1

    References
    ----------
    .. [1] Awad H. Al-Mohy and Nicholas J. Higham (2012)
           "Improved Inverse Scaling and Squaring Algorithms
           for the Matrix Logarithm."
           SIAM Journal on Scientific Computing, 34 (4). C152-C169.
           ISSN 1095-7197

    .. [2] Nicholas J. Higham (2008)
           "Functions of Matrices: Theory and Computation"
           ISBN 978-0-898716-46-7

    .. [3] Nicholas J. Higham and Lijing lin (2011)
           "A Schur-Pade Algorithm for Fractional Powers of a Matrix."
           SIAM Journal on Matrix Analysis and Applications,
           32 (3). pp. 1056-1078. ISSN 0895-4798

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import logm, expm
    >>> a = np.array([[1.0, 3.0], [1.0, 4.0]])
    >>> b = logm(a)
    >>> b
    array([[-1.02571087,  2.05142174],
           [ 0.68380725,  1.02571087]])
    >>> expm(b)         # Verify expm(logm(a)) returns a
    array([[ 1.,  3.],
           [ 1.,  4.]])

    """
    # Ensure A is a square matrix
    A = _asarray_square(A)
    
    # Import the internal function for computing matrix logarithm
    # Avoid circular import ... this is OK, right?
    import scipy.linalg._matfuncs_inv_ssq
    F = scipy.linalg._matfuncs_inv_ssq._logm(A)
    
    # Ensure F is real if A is real
    F = _maybe_real(A, F)
    
    # Set the error tolerance for the estimated error
    errtol = 1000 * eps
    
    # TODO: Use a better error approximation
    # Calculate the estimated error ||expm(F) - A||_1 / ||A||_1
    errest = norm(expm(F) - A, 1) / norm(A, 1)
    
    # Display a warning if disp is True and the error is large
    if disp:
        if not isfinite(errest) or errest >= errtol:
            print("logm result may be inaccurate, approximate err =", errest)
        return F
    else:
        # Return both F and the estimated error if disp is False
        return F, errest
    """
    a = np.asarray(A)
    # 将输入 A 转换为 NumPy 数组 a
    if a.size == 1 and a.ndim < 2:
        # 如果 a 只有一个元素且维度小于2，返回包含 exp(a.item()) 的 2D 数组
        return np.array([[np.exp(a.item())]])

    if a.ndim < 2:
        # 如果 a 的维度小于2，抛出线性代数错误：输入数组必须至少是二维的
        raise LinAlgError('The input array must be at least two-dimensional')
    if a.shape[-1] != a.shape[-2]:
        # 如果数组的最后两个维度不是方阵，抛出线性代数错误：数组的最后两个维度必须是方阵
        raise LinAlgError('Last 2 dimensions of the array must be square')
    n = a.shape[-1]
    # 如果数组的任何一个维度为0，返回与 a 相同形状的空数组
    if min(*a.shape) == 0:
        return np.empty_like(a)

    # 如果数组是标量，即形状为 (1, 1)，返回包含 exp(a) 的 2x2 数组
    if a.shape[-2:] == (1, 1):
        return np.exp(a)

    if not np.issubdtype(a.dtype, np.inexact):
        # 如果数组的数据类型不是浮点数类型，转换为 np.float64 类型
        a = a.astype(np.float64)
    elif a.dtype == np.float16:
        # 如果数组的数据类型是 np.float16，转换为 np.float32 类型
        a = a.astype(np.float32)

    # 2x2 情况的显式公式存在（参考文献 [1] 中的公式 (2.2)）。然而，没有使用 Kahan 方法，可能会导致数值不稳定性（见 gh-19584）。
    # 因此，在我们有一个更稳定的实现之前，这里先移除。

    n = a.shape[-1]
    eA = np.empty(a.shape, dtype=a.dtype)
    # 创建空数组 eA，与 a 具有相同的形状和数据类型，用于存储结果

    Am = np.empty((5, n, n), dtype=a.dtype)
    # 创建形状为 (5, n, n) 的空数组 Am，用于存储中间数组

    # 主循环，遍历 ndarray 的切片并传递给 expm
    # 遍历数组 a 的前两个维度之外的所有索引组合
    for ind in product(*[range(x) for x in a.shape[:-2]]):
        # 根据当前索引组合获取数组 a 中的切片 aw
        aw = a[ind]

        # 计算 aw 的带宽 lu
        lu = bandwidth(aw)
        
        # 如果 lu 全为 False，说明 aw 是对角矩阵
        if not any(lu):  # a is diagonal?
            # 对角矩阵情况下，直接对对角线元素进行指数化处理并返回
            eA[ind] = np.diag(np.exp(np.diag(aw)))
            continue

        # 一般/三角形情况；将切片复制到临时数组 Am 中并进行处理
        # Am 将在 pick_pade_structure 中被修改
        Am[0, :, :] = aw
        m, s = pick_pade_structure(Am)

        # 如果 s 不等于 0，表示需要进行缩放操作
        if s != 0:  # scaling needed
            # 根据 s 的值对 Am 进行缩放处理
            Am[:4] *= [[[2**(-s)]], [[4**(-s)]], [[16**(-s)]], [[64**(-s)]]]

        # 使用 Pade 近似计算 eAw
        pade_UV_calc(Am, n, m)
        eAw = Am[0]

        # 如果 s 不等于 0，表示还需要进行平方操作
        if s != 0:  # squaring needed

            # 如果 lu[1] == 0 或者 lu[0] == 0，表示 aw 是下三角形或者上三角形
            if (lu[1] == 0) or (lu[0] == 0):  # lower/upper triangular
                # 实现论文 [1] 的代码片段 2.1

                # 提取 aw 的对角线元素
                diag_aw = np.diag(aw)
                # 使用 einsum 返回可写的视图
                np.einsum('ii->i', eAw)[:] = np.exp(diag_aw * 2**(-s))
                # 超级对角线/次对角线
                sd = np.diag(aw, k=-1 if lu[1] == 0 else 1)

                # 对每个 s 的值进行迭代
                for i in range(s-1, -1, -1):
                    eAw = eAw @ eAw

                    # 对角线元素
                    np.einsum('ii->i', eAw)[:] = np.exp(diag_aw * 2.**(-i))
                    exp_sd = _exp_sinch(diag_aw * (2.**(-i))) * (sd * 2**(-i))
                    if lu[1] == 0:  # lower
                        np.einsum('ii->i', eAw[1:, :-1])[:] = exp_sd
                    else:  # upper
                        np.einsum('ii->i', eAw[:-1, 1:])[:] = exp_sd

            else:  # generic
                # 对于一般情况，对 eAw 进行 s 次平方操作
                for _ in range(s):
                    eAw = eAw @ eAw

        # 如果 lu[0] == 0 或者 lu[1] == 0，则将 np.empty 中的条目置零，以防三角形输入
        if (lu[0] == 0) or (lu[1] == 0):
            eA[ind] = np.triu(eAw) if lu[0] == 0 else np.tril(eAw)
        else:
            eA[ind] = eAw

    # 返回结果 eA
    return eA
def _exp_sinch(x):
    # 使用 Higham 的公式 (10.42) 计算 np.exp(x) 的差分
    lexp_diff = np.diff(np.exp(x))
    # 计算 x 的差分
    l_diff = np.diff(x)
    # 找到差分为零的位置
    mask_z = l_diff == 0.
    # 对非零差分进行除法运算
    lexp_diff[~mask_z] /= l_diff[~mask_z]
    # 对差分为零的位置进行处理
    lexp_diff[mask_z] = np.exp(x[:-1][mask_z])
    # 返回 lexp_diff 结果
    return lexp_diff


def cosm(A):
    """
    计算矩阵的余弦函数。

    此函数使用 expm 来计算矩阵的指数函数。

    Parameters
    ----------
    A : (N, N) array_like
        输入的数组

    Returns
    -------
    cosm : (N, N) ndarray
        A 的矩阵余弦函数

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import expm, sinm, cosm

    将欧拉恒等式 (exp(i*theta) = cos(theta) + i*sin(theta)) 应用于矩阵：

    >>> a = np.array([[1.0, 2.0], [-1.0, 3.0]])
    >>> expm(1j*a)
    array([[ 0.42645930+1.89217551j, -2.13721484-0.97811252j],
           [ 1.06860742+0.48905626j, -1.71075555+0.91406299j]])
    >>> cosm(a) + 1j*sinm(a)
    array([[ 0.42645930+1.89217551j, -2.13721484-0.97811252j],
           [ 1.06860742+0.48905626j, -1.71075555+0.91406299j]])

    """
    A = _asarray_square(A)
    # 如果 A 是复数类型的数组，返回 0.5*(expm(1j*A) + expm(-1j*A))
    if np.iscomplexobj(A):
        return 0.5*(expm(1j*A) + expm(-1j*A))
    # 如果 A 是实数类型的数组，返回 expm(1j*A) 的实部
    else:
        return expm(1j*A).real


def sinm(A):
    """
    计算矩阵的正弦函数。

    此函数使用 expm 来计算矩阵的指数函数。

    Parameters
    ----------
    A : (N, N) array_like
        输入的数组

    Returns
    -------
    sinm : (N, N) ndarray
        A 的矩阵正弦函数

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import expm, sinm, cosm

    将欧拉恒等式 (exp(i*theta) = cos(theta) + i*sin(theta)) 应用于矩阵：

    >>> a = np.array([[1.0, 2.0], [-1.0, 3.0]])
    >>> expm(1j*a)
    array([[ 0.42645930+1.89217551j, -2.13721484-0.97811252j],
           [ 1.06860742+0.48905626j, -1.71075555+0.91406299j]])
    >>> cosm(a) + 1j*sinm(a)
    array([[ 0.42645930+1.89217551j, -2.13721484-0.97811252j],
           [ 1.06860742+0.48905626j, -1.71075555+0.91406299j]])

    """
    A = _asarray_square(A)
    # 如果 A 是复数类型的数组，返回 -0.5j*(expm(1j*A) - expm(-1j*A))
    if np.iscomplexobj(A):
        return -0.5j*(expm(1j*A) - expm(-1j*A))
    # 如果 A 是实数类型的数组，返回 expm(1j*A) 的虚部
    else:
        return expm(1j*A).imag


def tanm(A):
    """
    计算矩阵的正切函数。

    此函数使用 expm 来计算矩阵的指数函数。

    Parameters
    ----------
    A : (N, N) array_like
        输入的数组

    Returns
    -------
    tanm : (N, N) ndarray
        A 的矩阵正切函数

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import tanm, sinm, cosm
    >>> a = np.array([[1.0, 3.0], [1.0, 4.0]])
    >>> t = tanm(a)
    >>> t
    array([[ -2.00876993,  -8.41880636],
           [ -2.80626879, -10.42757629]])

    验证 tanm(a) = sinm(a).dot(inv(cosm(a)))

    >>> s = sinm(a)
    >>> c = cosm(a)
    >>> s.dot(np.linalg.inv(c))
    array([[ -2.00876993,  -8.41880636],
           [ -2.80626879, -10.42757629]])

    """
    A = _asarray_square(A)
    # 调用 _maybe_real 函数，传入 A 作为参数，并将结果作为返回值
    return _maybe_real(A, solve(cosm(A), sinm(A)))
# 计算矩阵的双曲余弦函数
def coshm(A):
    # 将输入矩阵强制转换为方阵
    A = _asarray_square(A)
    # 计算矩阵指数的和与差的一半，即双曲余弦函数的矩阵值
    return _maybe_real(A, 0.5 * (expm(A) + expm(-A)))


# 计算矩阵的双曲正弦函数
def sinhm(A):
    # 将输入矩阵强制转换为方阵
    A = _asarray_square(A)
    # 计算矩阵指数的和与差的一半，即双曲正弦函数的矩阵值
    return _maybe_real(A, 0.5 * (expm(A) - expm(-A)))


# 计算矩阵的双曲正切函数
def tanhm(A):
    # 将输入矩阵强制转换为方阵
    A = _asarray_square(A)
    # 解线性方程组求解双曲正切函数的矩阵值
    return _maybe_real(A, solve(coshm(A), sinhm(A)))


# 评估由可调用对象指定的矩阵函数
def funm(A, func, disp=True):
    """
    Evaluate a matrix function specified by a callable.

    Returns the value of matrix-valued function ``f`` at `A`. The
    function ``f`` is an extension of the scalar-valued function `func`
    to matrices.

    Parameters
    ----------
    A = _asarray_square(A)
    # 将输入的矩阵 A 转换为方阵（如果不是方阵则引发异常）

    # 执行 Schur 分解 (lapack ?gees)
    T, Z = schur(A)
    # T 是 Schur 分解的上三角矩阵，Z 是相似变换矩阵使得 A = Z*T*Z^H

    T, Z = rsf2csf(T, Z)
    # 将实 Schur 形式转换为复 Schur 形式

    n, n = T.shape
    # 获取矩阵 T 的维度大小

    F = diag(func(diag(T)))  # apply function to diagonal elements
    # 对 T 的对角线元素应用函数 func，并构造对角矩阵 F

    F = F.astype(T.dtype.char)  # e.g., when F is real but T is complex
    # 将 F 的数据类型转换为 T 的数据类型的字符表示（例如，当 F 是实数而 T 是复数时）

    minden = abs(T[0, 0])
    # 初始化 minden 为 T 的第一个元素的绝对值

    # 实现 Golub 和 Van Loan 的算法 11.1.1
    # "matrix Computations."
    for p in range(1, n):
        for i in range(1, n-p+1):
            j = i + p
            s = T[i-1, j-1] * (F[j-1, j-1] - F[i-1, i-1])
            ksl = slice(i, j-1)
            val = dot(T[i-1, ksl], F[ksl, j-1]) - dot(F[i-1, ksl], T[ksl, j-1])
            s = s + val
            den = T[j-1, j-1] - T[i-1, i-1]
            if den != 0.0:
                s = s / den
            F[i-1, j-1] = s
            minden = min(minden, abs(den))

    F = dot(dot(Z, F), transpose(conjugate(Z)))
    # 恢复 F 到原始的实数或复数类型

    F = _maybe_real(A, F)
    # 如果输入矩阵 A 是实数，则确保 F 也是实数

    tol = {0: feps, 1: eps}[_array_precision[F.dtype.char]]
    # 根据 F 的数据类型字符确定容差 tol

    if minden == 0.0:
        minden = tol
    # 如果 minden 为零，将其设置为容差 tol

    err = min(1, max(tol, (tol/minden)*norm(triu(T, 1), 1)))
    # 计算估计误差的 1-范数 ||err||_1 / ||A||_1

    if prod(ravel(logical_not(isfinite(F))), axis=0):
        err = np.inf
    # 如果 F 包含无穷大或 NaN 值，则将误差设置为无穷大
    # 如果 disp 参数为 True，则进入条件判断
    if disp:
        # 如果误差 err 超过设定的容忍值 tol 的一千倍，打印警告信息
        if err > 1000 * tol:
            print("funm result may be inaccurate, approximate err =", err)
        # 返回计算结果 F
        return F
    else:
        # 如果 disp 参数为 False，则直接返回计算结果 F 和误差 err
        return F, err
def signm(A, disp=True):
    """
    Matrix sign function.

    Extension of the scalar sign(x) to matrices.

    Parameters
    ----------
    A : (N, N) array_like
        Matrix at which to evaluate the sign function
    disp : bool, optional
        Print warning if error in the result is estimated large
        instead of returning estimated error. (Default: True)

    Returns
    -------
    signm : (N, N) ndarray
        Value of the sign function at `A`
    errest : float
        (if disp == False)

        1-norm of the estimated error, ||err||_1 / ||A||_1

    Examples
    --------
    >>> from scipy.linalg import signm, eigvals
    >>> a = [[1,2,3], [1,2,1], [1,1,1]]
    >>> eigvals(a)
    array([ 4.12488542+0.j, -0.76155718+0.j,  0.63667176+0.j])
    >>> eigvals(signm(a))
    array([-1.+0.j,  1.+0.j,  1.+0.j])

    """
    # Ensure A is a square array
    A = _asarray_square(A)

    # Define a function to compute the rounded sign
    def rounded_sign(x):
        rx = np.real(x)
        # Determine precision and compute threshold
        if rx.dtype.char == 'f':
            c = 1e3*feps*amax(x)
        else:
            c = 1e3*eps*amax(x)
        # Compute the sign with threshold
        return sign((absolute(rx) > c) * rx)

    # Compute the matrix sign function using funm
    result, errest = funm(A, rounded_sign, disp=0)

    # Determine error tolerance based on precision of result
    errtol = {0: 1e3*feps, 1: 1e3*eps}[_array_precision[result.dtype.char]]

    # Return result if estimated error is within tolerance
    if errest < errtol:
        return result

    # Handle signm of defective matrices:

    # Compute singular values of A
    vals = svd(A, compute_uv=False)
    max_sv = np.amax(vals)

    # Determine a shifting constant to avoid zero eigenvalues
    c = 0.5 / max_sv

    # Initial shifted matrix S0
    S0 = A + c*np.identity(A.shape[0])
    prev_errest = errest

    # Iteratively refine S0 to reduce error
    for i in range(100):
        iS0 = inv(S0)
        S0 = 0.5*(S0 + iS0)
        Pp = 0.5*(dot(S0, S0) + S0)
        errest = norm(dot(Pp, Pp) - Pp, 1)
        if errest < errtol or prev_errest == errest:
            break
        prev_errest = errest

    # Print warning if error is large or not finite
    if disp:
        if not isfinite(errest) or errest >= errtol:
            print("signm result may be inaccurate, approximate err =", errest)
        return S0
    else:
        return S0, errest
    a = np.asarray(a)
    b = np.asarray(b)

将输入的 `a` 和 `b` 转换为 NumPy 数组，确保它们可以被正确处理和操作。


    if not (a.ndim == 2 and b.ndim == 2):
        raise ValueError("The both arrays should be 2-dimensional.")

检查 `a` 和 `b` 是否都是二维数组，如果不是，则抛出 ValueError 异常。


    if not a.shape[1] == b.shape[1]:
        raise ValueError("The number of columns for both arrays "
                         "should be equal.")

检查 `a` 和 `b` 的列数是否相等，如果不相等，则抛出 ValueError 异常。


    # accommodate empty arrays
    if a.size == 0 or b.size == 0:
        m = a.shape[0] * b.shape[0]
        n = a.shape[1]
        return np.empty_like(a, shape=(m, n))

处理输入数组为空的情况，返回一个与 `a` 结构相同的空数组，其形状是 `m x n`，其中 `m` 是 `a` 的行数乘以 `b` 的行数，`n` 是 `a` 的列数。


    # c = np.vstack([np.kron(a[:, k], b[:, k]) for k in range(b.shape[1])]).T
    c = a[..., :, np.newaxis, :] * b[..., np.newaxis, :, :]

计算 Khatri-Rao 乘积 `c`，这里使用了 NumPy 的广播功能来避免显式循环。这一行代码效果与注释中的代码等效，但更高效。


    return c.reshape((-1,) + c.shape[2:])

将计算得到的 `c` 重塑为一维数组，以便符合 Khatri-Rao 乘积的标准输出形式，并返回结果。
```