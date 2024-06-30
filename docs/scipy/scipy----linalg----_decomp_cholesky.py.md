# `D:\src\scipysrc\scipy\scipy\linalg\_decomp_cholesky.py`

```
# 导入必要的库和模块
import numpy as np
from numpy import asarray_chkfinite, asarray, atleast_2d, empty_like

# 导入本地模块
from ._misc import LinAlgError, _datacopied
from .lapack import get_lapack_funcs

# 定义公共函数 _cholesky，供 cholesky() 和 cho_factor() 使用
def _cholesky(a, lower=False, overwrite_a=False, clean=True,
              check_finite=True):
    """Common code for cholesky() and cho_factor()."""
    
    # 检查并转换输入数组为确保包含有限数值的数组
    a1 = asarray_chkfinite(a) if check_finite else asarray(a)
    a1 = atleast_2d(a1)
    
    # 检查输入数组的维度是否为2
    if a1.ndim != 2:
        raise ValueError(f'Input array needs to be 2D but received a {a1.ndim}d-array.')
    
    # 检查输入数组是否为方阵
    if a1.shape[0] != a1.shape[1]:
        raise ValueError('Input array is expected to be square but has '
                         f'the shape: {a1.shape}.')
    
    # 如果输入数组是空的方阵，返回空数组和指定的三角形状（上三角或下三角）
    if a1.size == 0:
        dt = cholesky(np.eye(1, dtype=a1.dtype)).dtype
        return empty_like(a1, dtype=dt), lower
    
    # 检查是否需要覆盖输入数组
    overwrite_a = overwrite_a or _datacopied(a1, a)
    
    # 获取 LAPACK 中的 Cholesky 分解函数
    potrf, = get_lapack_funcs(('potrf',), (a1,))
    
    # 调用 LAPACK 中的 Cholesky 分解函数进行分解
    c, info = potrf(a1, lower=lower, overwrite_a=overwrite_a, clean=clean)
    
    # 处理 LAPACK 返回的信息
    if info > 0:
        raise LinAlgError("%d-th leading minor of the array is not positive "
                          "definite" % info)
    if info < 0:
        raise ValueError(f'LAPACK reported an illegal value in {-info}-th argument'
                         'on entry to "POTRF".')
    
    # 返回 Cholesky 分解后的结果和三角形状
    return c, lower


def cholesky(a, lower=False, overwrite_a=False, check_finite=True):
    """
    Compute the Cholesky decomposition of a matrix.

    Returns the Cholesky decomposition, A = L L^* or
    A = U^* U of a Hermitian positive-definite matrix A.

    Parameters
    ----------
    a : (M, M) array_like
        Matrix to be decomposed
    lower : bool, optional
        Whether to compute the upper- or lower-triangular Cholesky
        factorization.  Default is upper-triangular.
    overwrite_a : bool, optional
        Whether to overwrite data in `a` (may improve performance).
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    c : (M, M) ndarray
        Upper- or lower-triangular Cholesky factor of `a`.

    Raises
    ------
    LinAlgError : if decomposition fails.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import cholesky
    >>> a = np.array([[1,-2j],[2j,5]])
    >>> L = cholesky(a, lower=True)
    >>> L
    array([[ 1.+0.j,  0.+0.j],
           [ 0.+2.j,  1.+0.j]])
    >>> L @ L.T.conj()
    array([[ 1.+0.j,  0.-2.j],
           [ 0.+2.j,  5.+0.j]])

    """
    # 调用 _cholesky 函数，进行 Cholesky 分解，并返回结果 c 和 lower
    c, lower = _cholesky(a, lower=lower, overwrite_a=overwrite_a, clean=True,
                         check_finite=check_finite)
    # 返回 Cholesky 分解后的结果 c
    return c
# 计算矩阵的 Cholesky 分解，用于 cho_solve 函数
def cho_factor(a, lower=False, overwrite_a=False, check_finite=True):
    """
    Compute the Cholesky decomposition of a matrix, to use in cho_solve

    Returns a matrix containing the Cholesky decomposition,
    ``A = L L*`` or ``A = U* U`` of a Hermitian positive-definite matrix `a`.
    The return value can be directly used as the first parameter to cho_solve.

    .. warning::
        The returned matrix also contains random data in the entries not
        used by the Cholesky decomposition. If you need to zero these
        entries, use the function `cholesky` instead.

    Parameters
    ----------
    a : (M, M) array_like
        Matrix to be decomposed
    lower : bool, optional
        Whether to compute the upper or lower triangular Cholesky factorization
        (Default: upper-triangular)
    overwrite_a : bool, optional
        Whether to overwrite data in a (may improve performance)
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    c : (M, M) ndarray
        Matrix whose upper or lower triangle contains the Cholesky factor
        of `a`. Other parts of the matrix contain random data.
    lower : bool
        Flag indicating whether the factor is in the lower or upper triangle

    Raises
    ------
    LinAlgError
        Raised if decomposition fails.

    See Also
    --------
    cho_solve : Solve a linear set equations using the Cholesky factorization
                of a matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import cho_factor
    >>> A = np.array([[9, 3, 1, 5], [3, 7, 5, 1], [1, 5, 9, 2], [5, 1, 2, 6]])
    >>> c, low = cho_factor(A)
    >>> c
    array([[3.        , 1.        , 0.33333333, 1.66666667],
           [3.        , 2.44948974, 1.90515869, -0.27216553],
           [1.        , 5.        , 2.29330749, 0.8559528 ],
           [5.        , 1.        , 2.        , 1.55418563]])
    >>> np.allclose(np.triu(c).T @ np. triu(c) - A, np.zeros((4, 4)))
    True

    """
    # 调用 _cholesky 函数进行 Cholesky 分解
    c, lower = _cholesky(a, lower=lower, overwrite_a=overwrite_a, clean=False,
                         check_finite=check_finite)
    # 返回 Cholesky 分解后的矩阵和 lower 标志
    return c, lower


# 解线性方程组 A x = b，给定矩阵 A 的 Cholesky 分解
def cho_solve(c_and_lower, b, overwrite_b=False, check_finite=True):
    """Solve the linear equations A x = b, given the Cholesky factorization of A.

    Parameters
    ----------
    (c, lower) : tuple, (array, bool)
        Cholesky factorization of a, as given by cho_factor
    b : array
        Right-hand side
    overwrite_b : bool, optional
        Whether to overwrite data in b (may improve performance)
    """
    # 实现解线性方程组的功能，基于给定的 Cholesky 分解和右侧向量 b
    pass  # Placeholder for actual implementation
    # 参数 check_finite: bool, 可选项
    # 是否检查输入矩阵是否只包含有限数值。
    # 禁用此选项可能提高性能，但如果输入包含无穷大或NaN，可能导致问题（崩溃、非终止）。

    # 返回值 x: array
    # 方程 A x = b 的解。

    # 参见
    # -------
    # cho_factor: 矩阵的 Cholesky 分解

    # 示例
    # -------
    # >>> import numpy as np
    # >>> from scipy.linalg import cho_factor, cho_solve
    # >>> A = np.array([[9, 3, 1, 5], [3, 7, 5, 1], [1, 5, 9, 2], [5, 1, 2, 6]])
    # >>> c, low = cho_factor(A)
    # >>> x = cho_solve((c, low), [1, 1, 1, 1])
    # >>> np.allclose(A @ x - [1, 1, 1, 1], np.zeros(4))
    # True

    (c, lower) = c_and_lower
    # 解包元组 c_and_lower，获得 c 和 lower

    if check_finite:
        # 如果 check_finite 为 True，则使用 asarray_chkfinite 来验证 b 和 c 是否包含有限数值
        b1 = asarray_chkfinite(b)
        c = asarray_chkfinite(c)
    else:
        # 如果 check_finite 为 False，则直接转换 b 和 c 为 ndarray
        b1 = asarray(b)
        c = asarray(c)

    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        # 如果 c 不是二维数组，或者 c 的行列数不相等，抛出 ValueError
        raise ValueError("The factored matrix c is not square.")
    if c.shape[1] != b1.shape[0]:
        # 如果 c 的列数不等于 b1 的行数，抛出 ValueError，因为维度不兼容
        raise ValueError(f"incompatible dimensions ({c.shape} and {b1.shape})")

    # 处理空数组的情况
    if b1.size == 0:
        # 如果 b1 是空数组，则根据 c 的 dtype 创建一个空数组返回
        dt = cho_solve((np.eye(2, dtype=b1.dtype), True),
                        np.ones(2, dtype=c.dtype)).dtype
        return empty_like(b1, dtype=dt)

    # 检查是否覆盖了数组 b
    overwrite_b = overwrite_b or _datacopied(b1, b)

    # 调用 LAPACK 中的 potrs 函数求解线性方程组
    potrs, = get_lapack_funcs(('potrs',), (c, b1))
    x, info = potrs(c, b1, lower=lower, overwrite_b=overwrite_b)
    if info != 0:
        # 如果 info 不为 0，表示 potrs 函数调用中出现错误，抛出 ValueError
        raise ValueError('illegal value in %dth argument of internal potrs'
                         % -info)
    return x
# Cholesky分解带状Hermitian正定矩阵的函数
def cholesky_banded(ab, overwrite_ab=False, lower=False, check_finite=True):
    """
    Cholesky decompose a banded Hermitian positive-definite matrix

    The matrix a is stored in ab either in lower-diagonal or upper-
    diagonal ordered form::

        ab[u + i - j, j] == a[i,j]        (if upper form; i <= j)
        ab[    i - j, j] == a[i,j]        (if lower form; i >= j)

    Example of ab (shape of a is (6,6), u=2)::

        upper form:
        *   *   a02 a13 a24 a35
        *   a01 a12 a23 a34 a45
        a00 a11 a22 a33 a44 a55

        lower form:
        a00 a11 a22 a33 a44 a55
        a10 a21 a32 a43 a54 *
        a20 a31 a42 a53 *   *

    Parameters
    ----------
    ab : (u + 1, M) array_like
        Banded matrix
    overwrite_ab : bool, optional
        Discard data in ab (may enhance performance)
    lower : bool, optional
        Is the matrix in the lower form. (Default is upper form)
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    c : (u + 1, M) ndarray
        Cholesky factorization of a, in the same banded format as ab

    See Also
    --------
    cho_solve_banded :
        Solve a linear set equations, given the Cholesky factorization
        of a banded Hermitian.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import cholesky_banded
    >>> from numpy import allclose, zeros, diag
    >>> Ab = np.array([[0, 0, 1j, 2, 3j], [0, -1, -2, 3, 4], [9, 8, 7, 6, 9]])
    >>> A = np.diag(Ab[0,2:], k=2) + np.diag(Ab[1,1:], k=1)
    >>> A = A + A.conj().T + np.diag(Ab[2, :])
    >>> c = cholesky_banded(Ab)
    >>> C = np.diag(c[0, 2:], k=2) + np.diag(c[1, 1:], k=1) + np.diag(c[2, :])
    >>> np.allclose(C.conj().T @ C - A, np.zeros((5, 5)))
    True

    """
    # 如果需要检查有限性，则使用asarray_chkfinite进行检查，否则直接转换为数组
    if check_finite:
        ab = asarray_chkfinite(ab)
    else:
        ab = asarray(ab)

    # 如果输入矩阵为空，返回与其dtype匹配的空数组
    if ab.size == 0:
        dt = cholesky_banded(np.array([[0, 0], [1, 1]], dtype=ab.dtype)).dtype
        return empty_like(ab, dtype=dt)

    # 获取LAPACK的pbtrf函数并调用，进行Cholesky分解
    pbtrf, = get_lapack_funcs(('pbtrf',), (ab,))
    c, info = pbtrf(ab, lower=lower, overwrite_ab=overwrite_ab)

    # 如果info > 0，抛出异常，指示第info个主要子式不是正定的
    if info > 0:
        raise LinAlgError("%d-th leading minor not positive definite" % info)
    
    # 如果info < 0，抛出异常，指示内部pbtrf函数的第-info个参数值非法
    if info < 0:
        raise ValueError('illegal value in %d-th argument of internal pbtrf'
                         % -info)
    
    # 返回Cholesky分解后的矩阵c
    return c
    (cb, lower) = cb_and_lower
    # 解包输入参数 cb_and_lower，cb 是由 cholesky_banded 给出的 A 的 Cholesky 分解
    # lower 必须与传递给 cholesky_banded 的相同值

    if check_finite:
        # 如果 check_finite 为 True，则检查输入矩阵是否包含有限数字
        cb = asarray_chkfinite(cb)
        b = asarray_chkfinite(b)
    else:
        # 如果 check_finite 为 False，则不进行有限性检查
        cb = asarray(cb)
        b = asarray(b)

    # Validate shapes.
    # 验证形状是否兼容
    if cb.shape[-1] != b.shape[0]:
        raise ValueError("shapes of cb and b are not compatible.")
    
    # accommodate empty arrays
    # 处理空数组情况
    if b.size == 0:
        # 创建一个与 b 相同形状和数据类型的空数组
        m = cholesky_banded(np.array([[0, 0], [1, 1]], dtype=cb.dtype))
        # 获取解 cho_solve_banded 的数据类型
        dt = cho_solve_banded((m, True), np.ones(2, dtype=b.dtype)).dtype
        return empty_like(b, dtype=dt)

    # 获取 LAPACK 函数 pbtrs
    pbtrs, = get_lapack_funcs(('pbtrs',), (cb, b))
    # 调用 pbtrs 函数解方程
    x, info = pbtrs(cb, b, lower=lower, overwrite_b=overwrite_b)
    if info > 0:
        # 如果 info > 0，说明第 info 个主子矩阵不是正定的
        raise LinAlgError("%dth leading minor not positive definite" % info)
    if info < 0:
        # 如果 info < 0，说明 pbtrs 内部第 -info 个参数值非法
        raise ValueError('illegal value in %dth argument of internal pbtrs'
                         % -info)
    # 返回解 x
    return x
```