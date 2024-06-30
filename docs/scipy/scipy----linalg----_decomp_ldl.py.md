# `D:\src\scipysrc\scipy\scipy\linalg\_decomp_ldl.py`

```
# 从警告模块中导入警告函数
from warnings import warn

# 导入 NumPy 库，并且从中导入一些特定函数和对象
import numpy as np
from numpy import (atleast_2d, arange, zeros_like, imag, diag,
                   iscomplexobj, tril, triu, argsort, empty_like)

# 从 SciPy 库的子模块中导入复数警告
from scipy._lib._util import ComplexWarning

# 从当前包的 _decomp 模块导入 _asarray_validated 函数
from ._decomp import _asarray_validated

# 从 lapack 模块导入 get_lapack_funcs 和 _compute_lwork 函数
from .lapack import get_lapack_funcs, _compute_lwork

# 定义模块中公开的接口，只有 ldl 函数是公开的
__all__ = ['ldl']


def ldl(A, lower=True, hermitian=True, overwrite_a=False, check_finite=True):
    """ Computes the LDLt or Bunch-Kaufman factorization of a symmetric/
    hermitian matrix.

    This function returns a block diagonal matrix D consisting blocks of size
    at most 2x2 and also a possibly permuted unit lower triangular matrix
    ``L`` such that the factorization ``A = L D L^H`` or ``A = L D L^T``
    holds. If `lower` is False then (again possibly permuted) upper
    triangular matrices are returned as outer factors.

    The permutation array can be used to triangularize the outer factors
    simply by a row shuffle, i.e., ``lu[perm, :]`` is an upper/lower
    triangular matrix. This is also equivalent to multiplication with a
    permutation matrix ``P.dot(lu)``, where ``P`` is a column-permuted
    identity matrix ``I[:, perm]``.

    Depending on the value of the boolean `lower`, only upper or lower
    triangular part of the input array is referenced. Hence, a triangular
    matrix on entry would give the same result as if the full matrix is
    supplied.

    Parameters
    ----------
    A : array_like
        Square input array
    lower : bool, optional
        This switches between the lower and upper triangular outer factors of
        the factorization. Lower triangular (``lower=True``) is the default.
    hermitian : bool, optional
        For complex-valued arrays, this defines whether ``A = A.conj().T`` or
        ``A = A.T`` is assumed. For real-valued arrays, this switch has no
        effect.
    overwrite_a : bool, optional
        Allow overwriting data in `A` (may enhance performance). The default
        is False.
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    lu : ndarray
        The (possibly) permuted upper/lower triangular outer factor of the
        factorization.
    d : ndarray
        The block diagonal multiplier of the factorization.
    perm : ndarray
        The row-permutation index array that brings lu into triangular form.

    Raises
    ------
    ValueError
        If input array is not square.
    ComplexWarning
        If a complex-valued array with nonzero imaginary parts on the
        diagonal is given and hermitian is set to True.

    See Also
    --------
    cholesky, lu

    Notes
    -----
    This function uses ``?SYTRF`` routines for symmetric matrices and
    ``?HETRF`` routines for hermitian matrices in LAPACK.

    """
    # 这里是函数体的开始，代码逻辑将从这里开始执行
    """
    ?HETRF routines for Hermitian matrices from LAPACK. See [1]_ for
    the algorithm details.

    Depending on the `lower` keyword value, only lower or upper triangular
    part of the input array is referenced. Moreover, this keyword also defines
    the structure of the outer factors of the factorization.

    .. versionadded:: 1.1.0

    References
    ----------
    .. [1] J.R. Bunch, L. Kaufman, Some stable methods for calculating
       inertia and solving symmetric linear systems, Math. Comput. Vol.31,
       1977. :doi:`10.2307/2005787`

    Examples
    --------
    Given an upper triangular array ``a`` that represents the full symmetric
    array with its entries, obtain ``l``, 'd' and the permutation vector `perm`:

    >>> import numpy as np
    >>> from scipy.linalg import ldl
    >>> a = np.array([[2, -1, 3], [0, 2, 0], [0, 0, 1]])
    >>> lu, d, perm = ldl(a, lower=0) # Use the upper part
    >>> lu
    array([[ 0. ,  0. ,  1. ],
           [ 0. ,  1. , -0.5],
           [ 1. ,  1. ,  1.5]])
    >>> d
    array([[-5. ,  0. ,  0. ],
           [ 0. ,  1.5,  0. ],
           [ 0. ,  0. ,  2. ]])
    >>> perm
    array([2, 1, 0])
    >>> lu[perm, :]
    array([[ 1. ,  1. ,  1.5],
           [ 0. ,  1. , -0.5],
           [ 0. ,  0. ,  1. ]])
    >>> lu.dot(d).dot(lu.T)
    array([[ 2., -1.,  3.],
           [-1.,  2.,  0.],
           [ 3.,  0.,  1.]])

    """
    # 将输入数组 A 转换为至少是二维的数组 a，并确保其为有效的数组
    a = atleast_2d(_asarray_validated(A, check_finite=check_finite))
    # 检查 a 是否为方阵，若不是则抛出 ValueError 异常
    if a.shape[0] != a.shape[1]:
        raise ValueError('The input array "a" should be square.')
    # 若输入的方阵 a 是空数组，则返回空数组的相同形状的 l、d 和空的置换向量 perm
    if a.size == 0:
        return empty_like(a), empty_like(a), np.array([], dtype=int)

    # 获取输入方阵 a 的维度 n
    n = a.shape[0]
    # 根据 a 的复数性质选择相应的数据类型
    r_or_c = complex if iscomplexobj(a) else float

    # 根据 hermitian 和数据类型选择 LAPACK 的相关例程
    if r_or_c is complex and hermitian:
        s, sl = 'hetrf', 'hetrf_lwork'
        # 如果对角线上存在虚部，则发出警告
        if np.any(imag(diag(a))):
            warn('scipy.linalg.ldl():\nThe imaginary parts of the diagonal'
                 'are ignored. Use "hermitian=False" for factorization of'
                 'complex symmetric arrays.', ComplexWarning, stacklevel=2)
    else:
        s, sl = 'sytrf', 'sytrf_lwork'

    # 获取相应 LAPACK 例程及其工作空间的计算函数
    solver, solver_lwork = get_lapack_funcs((s, sl), (a,))
    # 计算所需的工作空间大小
    lwork = _compute_lwork(solver_lwork, n, lower=lower)
    # 调用 LAPACK 例程进行因式分解，获取 LDL 分解后的结果
    ldu, piv, info = solver(a, lwork=lwork, lower=lower,
                            overwrite_a=overwrite_a)
    # 如果返回值 info 小于 0，则抛出相应的异常
    if info < 0:
        raise ValueError(f'{s.upper()} exited with the internal error "illegal value '
                         f'in argument number {-info}". See LAPACK documentation '
                         'for the error codes.')

    # 对置换向量 piv 进行处理，确保其符合规范
    swap_arr, pivot_arr = _ldl_sanitize_ipiv(piv, lower=lower)
    # 根据 LDL 分解结果获取对角阵 d 和下三角矩阵 lu
    d, lu = _ldl_get_d_and_l(ldu, pivot_arr, lower=lower, hermitian=hermitian)
    # 根据 lu 的交换数组 swap_arr、置换数组 pivot_arr 和 lower 参数构造出三角因子 lu 和置换向量 perm
    lu, perm = _ldl_construct_tri_factor(lu, swap_arr, pivot_arr, lower=lower)

    # 返回 LDL 分解的结果 lu、d 和 perm
    return lu, d, perm
# 定义一个函数，用于处理由 LAPACK 程序返回的奇怪编码的置换数组，将其转换为规范的置换和对角线主元大小格式。
def _ldl_sanitize_ipiv(a, lower=True):
    """
    This helper function takes the rather strangely encoded permutation array
    returned by the LAPACK routines ?(HE/SY)TRF and converts it into
    regularized permutation and diagonal pivot size format.

    Since FORTRAN uses 1-indexing and LAPACK uses different start points for
    upper and lower formats there are certain offsets in the indices used
    below.

    Let's assume a result where the matrix is 6x6 and there are two 2x2
    and two 1x1 blocks reported by the routine. To ease the coding efforts,
    we still populate a 6-sized array and fill zeros as the following ::
    
        pivots = [2, 0, 2, 0, 1, 1]
    
    This denotes a diagonal matrix of the form ::
    
        [x x        ]
        [x x        ]
        [    x x    ]
        [    x x    ]
        [        x  ]
        [          x]
    
    In other words, we write 2 when the 2x2 block is first encountered and
    automatically write 0 to the next entry and skip the next spin of the
    loop. Thus, a separate counter or array appends to keep track of block
    sizes are avoided. If needed, zeros can be filtered out later without
    losing the block structure.
    
    Parameters
    ----------
    a : ndarray
        The permutation array ipiv returned by LAPACK
    lower : bool, optional
        The switch to select whether upper or lower triangle is chosen in
        the LAPACK call.
    
    Returns
    -------
    swap_ : ndarray
        The array that defines the row/column swap operations. For example,
        if row two is swapped with row four, the result is [0, 3, 2, 3].
    pivots : ndarray
        The array that defines the block diagonal structure as given above.
    
    """
    # 获取数组 a 的大小
    n = a.size
    # 创建一个长度为 n 的索引置换数组 swap_
    swap_ = arange(n)
    # 创建一个与 swap_ 相同大小的全零数组 pivots，用于记录块对角结构
    pivots = zeros_like(swap_, dtype=int)
    # 初始化一个标志，用于跳过下一个 2x2 块
    skip_2x2 = False

    # 根据 lower 参数确定上三角或下三角的起始、结束和增量值
    x, y, rs, re, ri = (1, 0, 0, n, 1) if lower else (-1, -1, n-1, -1, -1)

    # 循环处理数组 a
    for ind in range(rs, re, ri):
        # 如果上一个循环已经处理了一个 2x2 块，则跳过当前循环
        if skip_2x2:
            skip_2x2 = False
            continue
        
        # 获取当前索引处的值
        cur_val = a[ind]
        # 检查当前值是否大于 0，如果是，表示有一个 1x1 块
        if cur_val > 0:
            # 如果当前值不等于索引加一，则需要置换操作
            if cur_val != ind + 1:
                swap_[ind] = swap_[cur_val - 1]
            # 将 pivots 对应位置置为 1，表示 1x1 块
            pivots[ind] = 1
        # 如果当前值小于 0，并且与下一个值构成 2x2 块
        elif cur_val < 0 and cur_val == a[ind + x]:
            # 如果当前值的相反数不等于索引加二，则需要置换操作
            if -cur_val != ind + 2:
                swap_[ind + x] = swap_[-cur_val - 1]
            # 将 pivots 对应位置置为 2，表示 2x2 块
            pivots[ind + y] = 2
            # 标记下一个循环需要跳过，因为已处理了一个 2x2 块
            skip_2x2 = True
        else:
            # 如果出现不合理的值，抛出 ValueError 异常
            raise ValueError('While parsing the permutation array '
                             'in "scipy.linalg.ldl", invalid entries '
                             'found. The array syntax is invalid.')
    # 返回排序后的数组和所有轴点的索引
    return swap_, pivots
# 辅助函数用于提取LDL.T分解的对角线和三角形矩阵。

Parameters
----------
ldu : ndarray
    LAPACK路由返回的紧凑输出。
pivs : ndarray
    经过处理的数组，其中的元素是{0, 1, 2}，表示枢轴的大小。每个2后面跟着一个0。
lower : bool, optional
    如果设置为False，则考虑上三角部分。
hermitian : bool, optional
    如果设置为False，则假定为对称复数数组。

Returns
-------
d : ndarray
    块对角矩阵。
lu : ndarray
    上/下三角矩阵。
"""
def _ldl_get_d_and_l(ldu, pivs, lower=True, hermitian=True):
    """
    Helper function to extract the diagonal and triangular matrices for
    LDL.T factorization.

    Parameters
    ----------
    ldu : ndarray
        The compact output returned by the LAPACK routing
    pivs : ndarray
        The sanitized array of {0, 1, 2} denoting the sizes of the pivots. For
        every 2 there is a succeeding 0.
    lower : bool, optional
        If set to False, upper triangular part is considered.
    hermitian : bool, optional
        If set to False a symmetric complex array is assumed.

    Returns
    -------
    d : ndarray
        The block diagonal matrix.
    lu : ndarray
        The upper/lower triangular matrix
    """
    is_c = iscomplexobj(ldu)
    d = diag(diag(ldu))
    n = d.shape[0]
    blk_i = 0  # block index

    # row/column offsets for selecting sub-, super-diagonal
    x, y = (1, 0) if lower else (0, 1)

    lu = tril(ldu, -1) if lower else triu(ldu, 1)
    diag_inds = arange(n)
    lu[diag_inds, diag_inds] = 1

    for blk in pivs[pivs != 0]:
        # increment the block index and check for 2s
        # if 2 then copy the off diagonals depending on uplo
        inc = blk_i + blk

        if blk == 2:
            d[blk_i+x, blk_i+y] = ldu[blk_i+x, blk_i+y]
            # If Hermitian matrix is factorized, the cross-offdiagonal element
            # should be conjugated.
            if is_c and hermitian:
                d[blk_i+y, blk_i+x] = ldu[blk_i+x, blk_i+y].conj()
            else:
                d[blk_i+y, blk_i+x] = ldu[blk_i+x, blk_i+y]

            lu[blk_i+x, blk_i+y] = 0.
        blk_i = inc

    return d, lu


def _ldl_construct_tri_factor(lu, swap_vec, pivs, lower=True):
    """
    Helper function to construct explicit outer factors of LDL factorization.

    If lower is True the permuted factors are multiplied as L(1)*L(2)*...*L(k).
    Otherwise, the permuted factors are multiplied as L(k)*...*L(2)*L(1). See
    LAPACK documentation for more details.

    Parameters
    ----------
    lu : ndarray
        The triangular array that is extracted from LAPACK routine call with
        ones on the diagonals.
    swap_vec : ndarray
        The array that defines the row swapping indices. If the kth entry is m
        then rows k,m are swapped. Notice that the mth entry is not necessarily
        k to avoid undoing the swapping.
    pivs : ndarray
        The array that defines the block diagonal structure returned by
        _ldl_sanitize_ipiv().
    lower : bool, optional
        The boolean to switch between lower and upper triangular structure.

    Returns
    -------
    lu : ndarray
        The square outer factor which satisfies the L * D * L.T = A
    perm : ndarray
        The permutation vector that brings the lu to the triangular form

    Notes
    -----
    Note that the original argument "lu" is overwritten.
    """
    n = lu.shape[0]
    perm = arange(n)
    # 根据 lower 参数设置置换矩阵的读取顺序，确定行索引范围和步长
    rs, re, ri = (n-1, -1, -1) if lower else (0, n, 1)

    # 遍历指定范围内的行索引
    for ind in range(rs, re, ri):
        # 获取当前行需要交换的目标索引
        s_ind = swap_vec[ind]
        # 如果当前行需要进行交换
        if s_ind != ind:
            # 确定列的起始和结束位置
            col_s = ind if lower else 0
            col_e = n if lower else ind+1

            # 如果遇到一个 2x2 的块，则在置换中包含这两列
            if pivs[ind] == (0 if lower else 2):
                col_s += -1 if lower else 0
                col_e += 0 if lower else 1

            # 执行行交换操作，更新 LU 分解矩阵
            lu[[s_ind, ind], col_s:col_e] = lu[[ind, s_ind], col_s:col_e]
            # 更新置换向量，记录行交换顺序
            perm[[s_ind, ind]] = perm[[ind, s_ind]]

    # 返回更新后的 LU 分解矩阵和置换向量的排序
    return lu, argsort(perm)
```