# `D:\src\scipysrc\scipy\scipy\linalg\_decomp.py`

```
# Author: Pearu Peterson, March 2002
# additions by Travis Oliphant, March 2002
# additions by Eric Jones,      June 2002
# additions by Johannes Loehnert, June 2006
# additions by Bart Vandereycken, June 2006
# additions by Andrew D Straw, May 2007
# additions by Tiziano Zito, November 2008
#
# April 2010: Functions for LU, QR, SVD, Schur, and Cholesky decompositions
# were moved to their own files. Still in this file are functions for
# eigenstuff and for the Hessenberg form.

__all__ = ['eig', 'eigvals', 'eigh', 'eigvalsh',
           'eig_banded', 'eigvals_banded',
           'eigh_tridiagonal', 'eigvalsh_tridiagonal', 'hessenberg', 'cdf2rdf']

import numpy as np
# 导入numpy模块并指定别名np

from numpy import (array, isfinite, inexact, nonzero, iscomplexobj,
                   flatnonzero, conj, asarray, argsort, empty,
                   iscomplex, zeros, einsum, eye, inf)
# 从numpy模块导入多个函数和对象

# Local imports
from scipy._lib._util import _asarray_validated
# 从scipy的内部模块_scipy._lib._util中导入_asarray_validated函数

from ._misc import LinAlgError, _datacopied, norm
# 从当前包的_misc模块中导入LinAlgError异常类、_datacopied和norm函数

from .lapack import get_lapack_funcs, _compute_lwork
# 从当前包的lapack模块中导入get_lapack_funcs函数和_compute_lwork函数

_I = np.array(1j, dtype='F')
# 创建一个复数单位虚数部分为1的numpy数组_I，数据类型为单精度浮点型

def _make_complex_eigvecs(w, vin, dtype):
    """
    Produce complex-valued eigenvectors from LAPACK DGGEV real-valued output
    """
    # 从LAPACK的DGGEV实值输出产生复数值的特征向量

    v = np.array(vin, dtype=dtype)
    # 使用指定的数据类型dtype创建vin的numpy数组副本v

    m = (w.imag > 0)
    # 创建一个布尔数组m，表示特征值w的虚部大于0的位置

    m[:-1] |= (w.imag[1:] < 0)  # workaround for LAPACK bug, cf. ticket #709
    # 解决LAPACK的bug的一种方法，参见票号#709

    for i in flatnonzero(m):
        # 对于m中为True的每个索引i

        v.imag[:, i] = vin[:, i+1]
        # 将vin的下一个列作为v的虚部

        conj(v[:, i], v[:, i+1])
        # 求第i列和第i+1列的共轭

    return v
    # 返回复数值的特征向量v

def _make_eigvals(alpha, beta, homogeneous_eigvals):
    # 生成特征值数组

    if homogeneous_eigvals:
        # 如果是齐次特征值问题

        if beta is None:
            return np.vstack((alpha, np.ones_like(alpha)))
            # 返回垂直堆叠的alpha和与alpha形状相同的全1数组

        else:
            return np.vstack((alpha, beta))
            # 返回垂直堆叠的alpha和beta数组

    else:
        # 如果不是齐次特征值问题

        if beta is None:
            return alpha
            # 返回特征值数组alpha

        else:
            w = np.empty_like(alpha)
            # 创建一个与alpha形状相同的空数组w

            alpha_zero = (alpha == 0)
            # 创建一个布尔数组alpha_zero，表示alpha是否为0

            beta_zero = (beta == 0)
            # 创建一个布尔数组beta_zero，表示beta是否为0

            beta_nonzero = ~beta_zero
            # 创建一个布尔数组beta_nonzero，表示beta是否不为0

            w[beta_nonzero] = alpha[beta_nonzero]/beta[beta_nonzero]
            # 计算非零beta的特征值

            # Use np.inf for complex values too since
            # 1/np.inf = 0, i.e., it correctly behaves as projective
            # infinity.
            w[~alpha_zero & beta_zero] = np.inf
            # 对于实部为零而虚部不为零的情况，将特征值设为无穷大

            if np.all(alpha.imag == 0):
                w[alpha_zero & beta_zero] = np.nan
                # 如果所有特征值的虚部为零，则将特征值设为NaN

            else:
                w[alpha_zero & beta_zero] = complex(np.nan, np.nan)
                # 否则，将特征值设为复数NaN

            return w
            # 返回特征值数组w

def _geneig(a1, b1, left, right, overwrite_a, overwrite_b,
            homogeneous_eigvals):
    # 通用广义特征值问题求解函数

    ggev, = get_lapack_funcs(('ggev',), (a1, b1))
    # 获取LAPACK函数ggev，用于解决广义特征值问题

    cvl, cvr = left, right
    # 将左右特征向量设置为输入参数left和right

    res = ggev(a1, b1, lwork=-1)
    # 调用ggev函数计算最优工作空间大小

    lwork = res[-2][0].real.astype(np.int_)
    # 从ggev函数结果中获取最优工作空间大小并转换为整数类型

    if ggev.typecode in 'cz':
        # 如果ggev函数处理复数问题

        alpha, beta, vl, vr, work, info = ggev(a1, b1, cvl, cvr, lwork,
                                               overwrite_a, overwrite_b)
        # 调用ggev函数求解复数广义特征值问题

        w = _make_eigvals(alpha, beta, homogeneous_eigvals)
        # 调用_make_eigvals函数生成特征值数组w
    # 使用 ggev 函数计算广义特征值问题的解
    alphar, alphai, beta, vl, vr, work, info = ggev(a1, b1, cvl, cvr,
                                                    lwork, overwrite_a,
                                                    overwrite_b)
    # 将复数特征值和实数特征值组合成复数值
    alpha = alphar + _I * alphai
    # 根据复数特征值和实数特征值生成特征值数组
    w = _make_eigvals(alpha, beta, homogeneous_eigvals)
    # 检查 ggev 函数运行状态，确保算法正确执行
    _check_info(info, 'generalized eig algorithm (ggev)')

    # 检查特征值是否全部为实数
    only_real = np.all(w.imag == 0.0)
    # 如果 ggev 返回的特征值类型为 'c'（复数）或 'z'（复数），或者特征值不全为实数，则进行以下处理
    if not (ggev.typecode in 'cz' or only_real):
        # 获取特征值的数据类型字符表示
        t = w.dtype.char
        # 如果左特征向量需要转换成复数型，则进行转换
        if left:
            vl = _make_complex_eigvecs(w, vl, t)
        # 如果右特征向量需要转换成复数型，则进行转换
        if right:
            vr = _make_complex_eigvecs(w, vr, t)

    # LAPACK 函数返回的特征向量未经过归一化处理
    # 对右特征向量进行归一化处理
    for i in range(vr.shape[0]):
        if right:
            vr[:, i] /= norm(vr[:, i])
        # 对左特征向量进行归一化处理
        if left:
            vl[:, i] /= norm(vl[:, i])

    # 如果既不需要左特征向量也不需要右特征向量，则直接返回特征值数组 w
    if not (left or right):
        return w
    # 如果需要左特征向量
    if left:
        # 如果同时需要右特征向量，则返回特征值 w、左特征向量 vl、右特征向量 vr
        if right:
            return w, vl, vr
        # 如果只需要左特征向量，则返回特征值 w、左特征向量 vl
        return w, vl
    # 如果只需要右特征向量，则返回特征值 w、右特征向量 vr
    return w, vr
# 定义函数 eig，用于解决普通或广义特征值问题的函数
def eig(a, b=None, left=False, right=True, overwrite_a=False,
        overwrite_b=False, check_finite=True, homogeneous_eigvals=False):
    """
    Solve an ordinary or generalized eigenvalue problem of a square matrix.

    Find eigenvalues w and right or left eigenvectors of a general matrix::

        a   vr[:,i] = w[i]        b   vr[:,i]
        a.H vl[:,i] = w[i].conj() b.H vl[:,i]

    where ``.H`` is the Hermitian conjugation.

    Parameters
    ----------
    a : (M, M) array_like
        A complex or real matrix whose eigenvalues and eigenvectors
        will be computed.
    b : (M, M) array_like, optional
        Right-hand side matrix in a generalized eigenvalue problem.
        Default is None, identity matrix is assumed.
    left : bool, optional
        Whether to calculate and return left eigenvectors.  Default is False.
    right : bool, optional
        Whether to calculate and return right eigenvectors.  Default is True.
    overwrite_a : bool, optional
        Whether to overwrite `a`; may improve performance.  Default is False.
    overwrite_b : bool, optional
        Whether to overwrite `b`; may improve performance.  Default is False.
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
    homogeneous_eigvals : bool, optional
        If True, return the eigenvalues in homogeneous coordinates.
        In this case ``w`` is a (2, M) array so that::

            w[1,i] a vr[:,i] = w[0,i] b vr[:,i]

        Default is False.

    Returns
    -------
    w : (M,) or (2, M) double or complex ndarray
        The eigenvalues, each repeated according to its
        multiplicity. The shape is (M,) unless
        ``homogeneous_eigvals=True``.
    vl : (M, M) double or complex ndarray
        The left eigenvector corresponding to the eigenvalue
        ``w[i]`` is the column ``vl[:,i]``. Only returned if ``left=True``.
        The left eigenvector is not normalized.
    vr : (M, M) double or complex ndarray
        The normalized right eigenvector corresponding to the eigenvalue
        ``w[i]`` is the column ``vr[:,i]``.  Only returned if ``right=True``.

    Raises
    ------
    LinAlgError
        If eigenvalue computation does not converge.

    See Also
    --------
    eigvals : eigenvalues of general arrays
    eigh : Eigenvalues and right eigenvectors for symmetric/Hermitian arrays.
    eig_banded : eigenvalues and right eigenvectors for symmetric/Hermitian
        band matrices
    eigh_tridiagonal : eigenvalues and right eiegenvectors for
        symmetric/Hermitian tridiagonal matrices

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import linalg
    >>> a = np.array([[0., -1.], [1., 0.]])
    >>> linalg.eigvals(a)
    array([0.+1.j, 0.-1.j])

    >>> b = np.array([[0., 1.], [1., 1.]])
    """
    # 计算给定矩阵的特征值
    >>> linalg.eigvals(a, b)
    array([ 1.+0.j, -1.+0.j])

    # 创建一个3x3的对角矩阵，并计算其特征值，使用 homogeneous_eigvals=True 返回齐次特征值
    >>> a = np.array([[3., 0., 0.], [0., 8., 0.], [0., 0., 7.]])
    >>> linalg.eigvals(a, homogeneous_eigvals=True)
    array([[3.+0.j, 8.+0.j, 7.+0.j],
           [1.+0.j, 1.+0.j, 1.+0.j]])

    # 创建一个2x2的矩阵，并验证 eigvals 方法是否与 eig 方法返回的特征值相等
    >>> a = np.array([[0., -1.], [1., 0.]])
    >>> linalg.eigvals(a) == linalg.eig(a)[0]
    array([ True,  True])

    # 计算左特征向量，left=True，right=False
    >>> linalg.eig(a, left=True, right=False)[1] # normalized left eigenvector
    array([[-0.70710678+0.j        , -0.70710678-0.j        ],
           [-0.        +0.70710678j, -0.        -0.70710678j]])

    # 计算右特征向量，left=False，right=True
    >>> linalg.eig(a, left=False, right=True)[1] # normalized right eigenvector
    array([[0.70710678+0.j        , 0.70710678-0.j        ],
           [0.        -0.70710678j, 0.        +0.70710678j]])


    """
    # 对输入的矩阵进行有效性验证和处理
    a1 = _asarray_validated(a, check_finite=check_finite)
    if len(a1.shape) != 2 or a1.shape[0] != a1.shape[1]:
        raise ValueError('expected square matrix')

    # 处理空方阵的情况
    if a1.size == 0:
        # 对单位矩阵计算特征值和左右特征向量
        w_n, vr_n = eig(np.eye(2, dtype=a1.dtype))
        w = np.empty_like(a1, shape=(0,), dtype=w_n.dtype)
        w = _make_eigvals(w, None, homogeneous_eigvals)
        vl = np.empty_like(a1, shape=(0, 0), dtype=vr_n.dtype)
        vr = np.empty_like(a1, shape=(0, 0), dtype=vr_n.dtype)
        if not (left or right):
            return w
        if left:
            if right:
                return w, vl, vr
            return w, vl
        return w, vr

    # 检查是否需要覆盖原始矩阵
    overwrite_a = overwrite_a or (_datacopied(a1, a))
    if b is not None:
        # 对矩阵 b 进行有效性验证和处理
        b1 = _asarray_validated(b, check_finite=check_finite)
        overwrite_b = overwrite_b or _datacopied(b1, b)
        if len(b1.shape) != 2 or b1.shape[0] != b1.shape[1]:
            raise ValueError('expected square matrix')
        if b1.shape != a1.shape:
            raise ValueError('a and b must have the same shape')
        # 调用 _geneig 函数计算广义特征值和特征向量
        return _geneig(a1, b1, left, right, overwrite_a, overwrite_b,
                       homogeneous_eigvals)

    # 获取 LAPACK 函数 'geev' 和 'geev_lwork'
    geev, geev_lwork = get_lapack_funcs(('geev', 'geev_lwork'), (a1,))
    compute_vl, compute_vr = left, right

    # 计算所需的工作空间大小
    lwork = _compute_lwork(geev_lwork, a1.shape[0],
                           compute_vl=compute_vl,
                           compute_vr=compute_vr)

    # 调用 LAPACK 函数计算特征值和特征向量
    if geev.typecode in 'cz':
        w, vl, vr, info = geev(a1, lwork=lwork,
                               compute_vl=compute_vl,
                               compute_vr=compute_vr,
                               overwrite_a=overwrite_a)
        w = _make_eigvals(w, None, homogeneous_eigvals)
    else:
        wr, wi, vl, vr, info = geev(a1, lwork=lwork,
                                    compute_vl=compute_vl,
                                    compute_vr=compute_vr,
                                    overwrite_a=overwrite_a)
        w = wr + _I * wi
        w = _make_eigvals(w, None, homogeneous_eigvals)
    # 调用 _check_info 函数，检查 info 中的信息是否符合要求，特别是 'eig algorithm (geev)' 这一项
    _check_info(info, 'eig algorithm (geev)',
                positive='did not converge (only eigenvalues '
                         'with order >= %d have converged)')
    
    # 检查特征值 w 是否全为实数，即其虚部是否全为零
    only_real = np.all(w.imag == 0.0)
    
    # 如果不满足 geev.typecode 在 'cz' 中或者所有特征值都是实数，则进入条件判断
    if not (geev.typecode in 'cz' or only_real):
        # 获取特征值数组 w 的数据类型字符表示
        t = w.dtype.char
        # 如果 left 为真，则将特征向量 vl 转换为复数形式
        if left:
            vl = _make_complex_eigvecs(w, vl, t)
        # 如果 right 为真，则将特征向量 vr 转换为复数形式
        if right:
            vr = _make_complex_eigvecs(w, vr, t)
    
    # 如果 left 和 right 都为假，则直接返回特征值 w
    if not (left or right):
        return w
    
    # 如果 left 为真
    if left:
        # 如果 right 也为真，则返回特征值 w、左特征向量 vl、右特征向量 vr
        if right:
            return w, vl, vr
        # 如果 right 为假，则返回特征值 w、左特征向量 vl
        return w, vl
    
    # 如果 left 为假且 right 为真，则返回特征值 w、右特征向量 vr
    return w, vr
# 解决复杂埃尔米特或实对称矩阵的标准或广义特征值问题
def eigh(a, b=None, *, lower=True, eigvals_only=False, overwrite_a=False,
         overwrite_b=False, type=1, check_finite=True, subset_by_index=None,
         subset_by_value=None, driver=None):
    """
    Solve a standard or generalized eigenvalue problem for a complex
    Hermitian or real symmetric matrix.

    Find eigenvalues array ``w`` and optionally eigenvectors array ``v`` of
    array ``a``, where ``b`` is positive definite such that for every
    eigenvalue λ (i-th entry of w) and its eigenvector ``vi`` (i-th column of
    ``v``) satisfies::

                      a @ vi = λ * b @ vi
        vi.conj().T @ a @ vi = λ
        vi.conj().T @ b @ vi = 1

    In the standard problem, ``b`` is assumed to be the identity matrix.

    Parameters
    ----------
    a : (M, M) array_like
        A complex Hermitian or real symmetric matrix whose eigenvalues and
        eigenvectors will be computed.
    b : (M, M) array_like, optional
        A complex Hermitian or real symmetric definite positive matrix in.
        If omitted, identity matrix is assumed.
    lower : bool, optional
        Whether the pertinent array data is taken from the lower or upper
        triangle of ``a`` and, if applicable, ``b``. (Default: lower)
    eigvals_only : bool, optional
        Whether to calculate only eigenvalues and no eigenvectors.
        (Default: both are calculated)
    subset_by_index : iterable, optional
        If provided, this two-element iterable defines the start and the end
        indices of the desired eigenvalues (ascending order and 0-indexed).
        To return only the second smallest to fifth smallest eigenvalues,
        ``[1, 4]`` is used. ``[n-3, n-1]`` returns the largest three. Only
        available with "evr", "evx", and "gvx" drivers. The entries are
        directly converted to integers via ``int()``.
    subset_by_value : iterable, optional
        If provided, this two-element iterable defines the half-open interval
        ``(a, b]`` that, if any, only the eigenvalues between these values
        are returned. Only available with "evr", "evx", and "gvx" drivers. Use
        ``np.inf`` for the unconstrained ends.
    driver : str, optional
        Defines which LAPACK driver should be used. Valid options are "ev",
        "evd", "evr", "evx" for standard problems and "gv", "gvd", "gvx" for
        generalized (where b is not None) problems. See the Notes section.
        The default for standard problems is "evr". For generalized problems,
        "gvd" is used for full set, and "gvx" for subset requested cases.
    type : int, optional
        For the generalized problems, this keyword specifies the problem type
        to be solved for ``w`` and ``v`` (only takes 1, 2, 3 as possible
        inputs)::

            1 =>     a @ v = w @ b @ v
            2 => a @ b @ v = w @ v
            3 => b @ a @ v = w @ v

        This keyword is ignored for standard problems.

    """
    # 是否覆盖矩阵 a 中的数据以提升性能，默认为 False
    overwrite_a : bool, optional
    # 是否覆盖矩阵 b 中的数据以提升性能，默认为 False
    overwrite_b : bool, optional
    # 是否检查输入矩阵只包含有限数值，默认为 True；禁用此检查可能提升性能，
    # 但如果输入包含无穷大或 NaN，可能导致问题（崩溃或非终止状态）。

    Returns
    -------
    w : (N,) ndarray
        选择的 N（N<=M）个特征值，按升序排列，每个按其重数重复。
    v : (M, N) ndarray
        对应于特征值 ``w[i]`` 的归一化特征向量是列 ``v[:,i]``。
        仅当 ``eigvals_only=False`` 时返回。

    Raises
    ------
    LinAlgError
        如果特征值计算未收敛，发生错误，或者矩阵 b 不是正定的。注意，如果输入矩阵
        不对称或不埃尔米特，将不会报告错误，但结果将是错误的。

    See Also
    --------
    eigvalsh : 对称或埃尔米特数组的特征值
    eig : 非对称数组的特征值和右特征向量
    eigh_tridiagonal : 对称/埃尔米特三对角矩阵的特征值和右特征向量

    Notes
    -----
    此函数不检查输入数组是否埃尔米特/对称，以便表示仅使用其上/下三角部分的数组。
    此外，注意即使不考虑，有限性检查适用于整个数组，并且不受 "lower" 关键字影响。

    此函数使用 LAPACK 驱动程序进行所有可能的关键字组合的计算，如果数组是实数则以
    "sy" 开头，如果复数则以 "he" 开头，例如，带有 "evr" 驱动程序的浮点数组通过
    "syevr" 解决，复数数组带有 "gvx" 驱动程序的问题通过 "hegvx" 等解决。

    简要总结，最慢且最稳健的驱动程序是经典的 ``<sy/he>ev``，使用对称 QR。``<sy/he>evr``
    被视为最一般情况的最佳选择。然而，在某些情况下，``<sy/he>evd`` 在更多内存使用的
    代价下计算更快。``<sy/he>evx``，虽然仍比 ``<sy/he>ev`` 快，但通常比其余表现差，
    除非对大数组请求非常少的特征值，尽管仍然没有性能保证。

    请注意，根据 ``eigvals_only`` 是 True 还是 False，底层的 LAPACK 算法是不同的
    --- 因此特征值可能因是否请求特征向量而异。这种差异通常是机器 epsilon 乘以最大
    特征值的量级，因此仅在零或几乎为零的特征值时才可能看得见。
    For the generalized problem, normalization with respect to the given
    type argument::

            type 1 and 3 :      v.conj().T @ a @ v = w
            type 2       : inv(v).conj().T @ a @ inv(v) = w

            type 1 or 2  :      v.conj().T @ b @ v  = I
            type 3       : v.conj().T @ inv(b) @ v  = I

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import eigh
    >>> A = np.array([[6, 3, 1, 5], [3, 0, 5, 1], [1, 5, 6, 2], [5, 1, 2, 2]])
    >>> w, v = eigh(A)
    >>> np.allclose(A @ v - v @ np.diag(w), np.zeros((4, 4)))
    True

    Request only the eigenvalues

    >>> w = eigh(A, eigvals_only=True)

    Request eigenvalues that are less than 10.

    >>> A = np.array([[34, -4, -10, -7, 2],
    ...               [-4, 7, 2, 12, 0],
    ...               [-10, 2, 44, 2, -19],
    ...               [-7, 12, 2, 79, -34],
    ...               [2, 0, -19, -34, 29]])
    >>> eigh(A, eigvals_only=True, subset_by_value=[-np.inf, 10])
    array([6.69199443e-07, 9.11938152e+00])

    Request the second smallest eigenvalue and its eigenvector

    >>> w, v = eigh(A, subset_by_index=[1, 1])
    >>> w
    array([9.11938152])
    >>> v.shape  # only a single column is returned
    (5, 1)

    """
    # Determine whether the matrix is upper or lower triangular
    uplo = 'L' if lower else 'U'
    # Specify the job type for the LAPACK routine
    _job = 'N' if eigvals_only else 'V'

    # List of valid driver strings for LAPACK
    drv_str = [None, "ev", "evd", "evr", "evx", "gv", "gvd", "gvx"]
    # Validate the specified driver string
    if driver not in drv_str:
        raise ValueError('"{}" is unknown. Possible values are "None", "{}".'
                         ''.format(driver, '", "'.join(drv_str[1:])))

    # Validate and convert matrix `a` to a numpy array
    a1 = _asarray_validated(a, check_finite=check_finite)
    # Check if `a` is a square matrix
    if len(a1.shape) != 2 or a1.shape[0] != a1.shape[1]:
        raise ValueError('expected square "a" matrix')

    # Handle special case for empty square matrices
    if a1.size == 0:
        # Compute eigenvalues and eigenvectors for a 2x2 identity matrix
        w_n, v_n = eigh(np.eye(2, dtype=a1.dtype))
        # Return empty arrays for eigenvalues and eigenvectors
        w = np.empty_like(a1, shape=(0,), dtype=w_n.dtype)
        v = np.empty_like(a1, shape=(0, 0), dtype=v_n.dtype)
        if eigvals_only:
            return w
        else:
            return w, v

    # Determine if the matrix `a` should be overwritten
    overwrite_a = overwrite_a or (_datacopied(a1, a))
    # Check if `a` contains complex elements
    cplx = True if iscomplexobj(a1) else False
    # Dimension of matrix `a`
    n = a1.shape[0]
    # Arguments for the LAPACK driver routine
    drv_args = {'overwrite_a': overwrite_a}

    # Process matrix `b` if it is provided
    if b is not None:
        # Validate and convert matrix `b` to a numpy array
        b1 = _asarray_validated(b, check_finite=check_finite)
        # Determine if `b` should be overwritten
        overwrite_b = overwrite_b or _datacopied(b1, b)
        # Check if `b` is a square matrix
        if len(b1.shape) != 2 or b1.shape[0] != b1.shape[1]:
            raise ValueError('expected square "b" matrix')

        # Check if dimensions of `a` and `b` match
        if b1.shape != a1.shape:
            raise ValueError(f"wrong b dimensions {b1.shape}, should be {a1.shape}")

        # Validate the `type` parameter
        if type not in [1, 2, 3]:
            raise ValueError('"type" keyword only accepts 1, 2, and 3.')

        # Check if `b` contains complex elements
        cplx = True if iscomplexobj(b1) else (cplx or False)
        # Update driver arguments with `b` related parameters
        drv_args.update({'overwrite_b': overwrite_b, 'itype': type})

    # Determine if subset selection is used
    subset = (subset_by_index is not None) or (subset_by_value is not None)
    # 如果同时指定了索引和数值子集，则抛出数值错误
    if subset_by_index and subset_by_value:
        raise ValueError('Either index or value subset can be requested.')

    # 如果指定了索引子集，则检查索引的有效性
    if subset_by_index:
        # 将索引转换为整数
        lo, hi = (int(x) for x in subset_by_index)
        # 检查索引范围是否有效
        if not (0 <= lo <= hi < n):
            raise ValueError('Requested eigenvalue indices are not valid. '
                             f'Valid range is [0, {n-1}] and start <= end, but '
                             f'start={lo}, end={hi} is given')
        # 对于 Fortran 风格的 1 索引，更新驱动参数
        drv_args.update({'range': 'I', 'il': lo + 1, 'iu': hi + 1})

    # 如果指定了数值子集，则检查数值范围的有效性
    if subset_by_value:
        lo, hi = subset_by_value
        # 检查数值范围是否有效
        if not (-inf <= lo < hi <= inf):
            raise ValueError('Requested eigenvalue bounds are not valid. '
                             'Valid range is (-inf, inf) and low < high, but '
                             f'low={lo}, high={hi} is given')
        # 更新驱动参数
        drv_args.update({'range': 'V', 'vl': lo, 'vu': hi})

    # 根据复数性质确定 LAPACK 程序的前缀
    pfx = 'he' if cplx else 'sy'

    # 如果未指定驱动程序，则选择默认驱动程序
    # 首先检查不兼容的选择情况
    if driver:
        # 对于广义特征值问题，要求输入 b 数组存在
        if b is None and (driver in ["gv", "gvd", "gvx"]):
            raise ValueError(f'{driver} requires input b array to be supplied '
                             'for generalized eigenvalue problems.')
        # 对于标准特征值问题，不接受输入 b 数组
        if (b is not None) and (driver in ['ev', 'evd', 'evr', 'evx']):
            raise ValueError(f'"{driver}" does not accept input b array '
                             'for standard eigenvalue problems.')
        # 对于特征值子集，某些驱动程序无法计算
        if subset and (driver in ["ev", "evd", "gv", "gvd"]):
            raise ValueError(f'"{driver}" cannot compute subsets of eigenvalues')

    # 默认情况下选择驱动程序为 evr 或 gvd
    else:
        driver = "evr" if b is None else ("gvx" if subset else "gvd")

    # 指定每个 LAPACK 程序所需的工作空间参数
    lwork_spec = {
                  'syevd': ['lwork', 'liwork'],
                  'syevr': ['lwork', 'liwork'],
                  'heevd': ['lwork', 'liwork', 'lrwork'],
                  'heevr': ['lwork', 'lrwork', 'liwork'],
                  }

    # 如果是标准问题（无输入 b 数组）
    if b is None:
        # 获取 LAPACK 函数及其对应的工作空间计算函数
        drv, drvlw = get_lapack_funcs((pfx + driver, pfx+driver+'_lwork'),
                                      [a1])
        # 设置调用 LAPACK 函数所需的参数
        clw_args = {'n': n, 'lower': lower}
        if driver == 'evd':
            clw_args.update({'compute_v': 0 if _job == "N" else 1})

        # 计算所需的工作空间大小
        lw = _compute_lwork(drvlw, **clw_args)
        # 如果需要多个工作空间变量
        if isinstance(lw, tuple):
            lwork_args = dict(zip(lwork_spec[pfx+driver], lw))
        else:
            lwork_args = {'lwork': lw}

        # 更新驱动参数
        drv_args.update({'lower': lower, 'compute_v': 0 if _job == "N" else 1})
        # 调用 LAPACK 函数并返回结果
        w, v, *other_args, info = drv(a=a1, **drv_args, **lwork_args)
    else:  # Generalized problem
        # 处理一般化问题
        if driver == "gvd":
            # 对于 'gvd'，没有 lwork 查询
            drv = get_lapack_funcs(pfx + "gvd", [a1, b1])
            lwork_args = {}
        else:
            # 获取 LAPACK 函数，包括需要查询 lwork 的情况
            drv, drvlw = get_lapack_funcs((pfx + driver, pfx+driver+'_lwork'),
                                          [a1, b1])
            # 对于广义驱动程序，使用 uplo 替代 lower
            lw = _compute_lwork(drvlw, n, uplo=uplo)
            lwork_args = {'lwork': lw}

        # 更新驱动参数字典
        drv_args.update({'uplo': uplo, 'jobz': _job})

        # 调用 LAPACK 驱动函数，获取结果 w, v 和其他参数，同时返回信息码 info
        w, v, *other_args, info = drv(a=a1, b=b1, **drv_args, **lwork_args)

    # m 总是第一个额外参数
    w = w[:other_args[0]] if subset else w
    v = v[:, :other_args[0]] if (subset and not eigvals_only) else v

    # 检查是否成功完成
    if info == 0:
        if eigvals_only:
            return w
        else:
            return w, v
    else:
        if info < -1:
            # 抛出线性代数错误，显示内部 drv.typecode + pfx + driver 的非法值参数
            raise LinAlgError(f'Illegal value in argument {-info} of internal '
                              f'{drv.typecode + pfx + driver}')
        elif info > n:
            # 抛出线性代数错误，显示 B 的阶数不是正定的主子阵
            raise LinAlgError(f'The leading minor of order {info-n} of B is not '
                              'positive definite. The factorization of B '
                              'could not be completed and no eigenvalues '
                              'or eigenvectors were computed.')
        else:
            # 出现算法失败的情况，根据驱动程序类型抛出相应的线性代数错误信息
            drv_err = {'ev': 'The algorithm failed to converge; {} '
                             'off-diagonal elements of an intermediate '
                             'tridiagonal form did not converge to zero.',
                       'evx': '{} eigenvectors failed to converge.',
                       'evd': 'The algorithm failed to compute an eigenvalue '
                              'while working on the submatrix lying in rows '
                              'and columns {0}/{1} through mod({0},{1}).',
                       'evr': 'Internal Error.'
                       }
            if driver in ['ev', 'gv']:
                msg = drv_err['ev'].format(info)
            elif driver in ['evx', 'gvx']:
                msg = drv_err['evx'].format(info)
            elif driver in ['evd', 'gvd']:
                if eigvals_only:
                    msg = drv_err['ev'].format(info)
                else:
                    msg = drv_err['evd'].format(info, n+1)
            else:
                msg = drv_err['evr']

            raise LinAlgError(msg)
# 创建一个转换字典，将不同的选择标识符映射到其对应的数值
_conv_dict = {0: 0, 1: 1, 2: 2,
              'all': 0, 'value': 1, 'index': 2,
              'a': 0, 'v': 1, 'i': 2}

# 检查并确保选择标识符有效，并将其转换为Fortran风格
def _check_select(select, select_range, max_ev, max_len):
    """Check that select is valid, convert to Fortran style."""
    # 如果选择标识符是字符串，则转换为小写
    if isinstance(select, str):
        select = select.lower()
    try:
        # 尝试使用转换字典将选择标识符转换为相应的数值
        select = _conv_dict[select]
    except KeyError as e:
        # 如果转换失败，则抛出异常
        raise ValueError('invalid argument for select') from e
    
    # 默认值设置
    vl, vu = 0., 1.  # 默认值范围
    il = iu = 1     # 默认索引
    
    # 如果选择不是全选（select != 0）
    if select != 0:
        # 将选择范围转换为数组并检查合法性
        sr = asarray(select_range)
        if sr.ndim != 1 or sr.size != 2 or sr[1] < sr[0]:
            raise ValueError('select_range must be a 2-element array-like '
                             'in nondecreasing order')
        
        # 根据选择的不同类型进行处理
        if select == 1:  # (value)
            vl, vu = sr  # 设置值的范围
            if max_ev == 0:
                max_ev = max_len  # 如果最大特征值为0，则使用最大长度
        else:  # 2 (index)
            # 检查选择范围的数据类型，必须是整数类型
            if sr.dtype.char.lower() not in 'hilqp':
                raise ValueError(
                    f'when using select="i", select_range must '
                    f'contain integers, got dtype {sr.dtype} ({sr.dtype.char})'
                )
            # 将Python风格的索引（0到N-1）转换为Fortran风格的索引（1到N）
            il, iu = sr + 1
            # 检查索引范围是否在合理范围内
            if min(il, iu) < 1 or max(il, iu) > max_len:
                raise ValueError('select_range out of bounds')
            max_ev = iu - il + 1  # 计算最大特征值数量

    # 返回处理后的选择标识符及相关参数
    return select, vl, vu, il, iu, max_ev
    """
    if eigvals_only or overwrite_a_band:
        如果 eigvals_only 或 overwrite_a_band 为 True，则执行以下操作：
        将 a_band 转换为验证过的数组 a1，确保其中的数值都是有限的。
        如果 overwrite_a_band 为 True 或者在转换过程中数据被复制了，则将 overwrite_a_band 设为 True。
    else:
        否则，执行以下操作：
        将 a_band 转换为普通的数组 a1。
        如果 a1 的数据类型是浮点数类型，并且包含无穷大或 NaN，则抛出 ValueError 异常。
        将 overwrite_a_band 设为 1。

    if len(a1.shape) != 2:
        如果数组 a1 不是二维数组，则抛出 ValueError 异常，提示期望一个二维数组。

    # accommodate square empty matrices
    # 处理空的方阵情况
    """
    # 检查数组 a1 是否为空
    if a1.size == 0:
        # 对空数组进行特殊处理：计算带状矩阵的特征值和特征向量
        w_n, v_n = eig_banded(np.array([[0, 0], [1, 1]], dtype=a1.dtype))

        # 创建空数组 w 和 v，保持与 a1 相同的类型和形状
        w = np.empty_like(a1, shape=(0,), dtype=w_n.dtype)
        v = np.empty_like(a1, shape=(0, 0), dtype=v_n.dtype)

        # 如果仅需要特征值，则返回 w
        if eigvals_only:
            return w
        else:
            return w, v

    # 检查选择参数并确定计算的范围和最大特征值
    select, vl, vu, il, iu, max_ev = _check_select(
        select, select_range, max_ev, a1.shape[1])

    # 清除不再需要的变量 select_range
    del select_range

    # 根据选择条件执行相应的特征值计算操作
    if select == 0:
        # 根据数据类型选择对称或非对称矩阵特征值计算方法
        if a1.dtype.char in 'GFD':
            # TODO: 实现此部分功能，目前使用内置值
            # TODO: 通过调用 ?hbevd(lwork=-1) 或使用 calc_lwork.f 计算最优的 lwork
            internal_name = 'hbevd'
        else:  # a1.dtype.char in 'fd':
            # TODO: 实现此部分功能，目前使用内置值
            # TODO: 参考上述方法
            internal_name = 'sbevd'

        # 获取 LAPACK 函数 bevd
        bevd, = get_lapack_funcs((internal_name,), (a1,))

        # 调用特征值计算函数 bevd，返回特征值 w、特征向量 v 和状态信息 info
        w, v, info = bevd(a1, compute_v=not eigvals_only,
                          lower=lower, overwrite_ab=overwrite_a_band)

    else:  # select in [1, 2]
        # 如果仅需要特征值，则最大特征值设为 1
        if eigvals_only:
            max_ev = 1

        # 计算双精度浮点数或单精度浮点数的最优 abstol 值
        if a1.dtype.char in 'fF':  # 单精度
            lamch, = get_lapack_funcs(('lamch',), (np.array(0, dtype='f'),))
        else:  # 双精度
            lamch, = get_lapack_funcs(('lamch',), (np.array(0, dtype='d'),))
        abstol = 2 * lamch('s')

        # 根据数据类型选择广义特征值计算方法
        if a1.dtype.char in 'GFD':
            internal_name = 'hbevx'
        else:  # a1.dtype.char in 'gfd'
            internal_name = 'sbevx'

        # 获取 LAPACK 函数 bevx
        bevx, = get_lapack_funcs((internal_name,), (a1,))

        # 调用广义特征值计算函数 bevx，返回特征值 w、特征向量 v、计算的个数 m、状态信息 ifail 和 info
        w, v, m, ifail, info = bevx(
            a1, vl, vu, il, iu, compute_v=not eigvals_only, mmax=max_ev,
            range=select, lower=lower, overwrite_ab=overwrite_a_band,
            abstol=abstol)

        # 截取前 m 个特征值和特征向量
        w = w[:m]
        if not eigvals_only:
            v = v[:, :m]

    # 检查状态信息是否正常
    _check_info(info, internal_name)

    # 如果仅需要特征值，则返回 w
    if eigvals_only:
        return w
    # 否则返回特征值 w 和特征向量 v
    return w, v
# 计算普通或广义特征值问题的特征值
def eigvals(a, b=None, overwrite_a=False, check_finite=True,
            homogeneous_eigvals=False):
    """
    Compute eigenvalues from an ordinary or generalized eigenvalue problem.

    Find eigenvalues of a general matrix::

        a   vr[:,i] = w[i]        b   vr[:,i]

    Parameters
    ----------
    a : (M, M) array_like
        A complex or real matrix whose eigenvalues and eigenvectors
        will be computed.
    b : (M, M) array_like, optional
        Right-hand side matrix in a generalized eigenvalue problem.
        If omitted, identity matrix is assumed.
    overwrite_a : bool, optional
        Whether to overwrite data in a (may improve performance)
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities
        or NaNs.
    homogeneous_eigvals : bool, optional
        If True, return the eigenvalues in homogeneous coordinates.
        In this case ``w`` is a (2, M) array so that::

            w[1,i] a vr[:,i] = w[0,i] b vr[:,i]

        Default is False.

    Returns
    -------
    w : (M,) or (2, M) double or complex ndarray
        The eigenvalues, each repeated according to its multiplicity
        but not in any specific order. The shape is (M,) unless
        ``homogeneous_eigvals=True``.

    Raises
    ------
    LinAlgError
        If eigenvalue computation does not converge

    See Also
    --------
    eig : eigenvalues and right eigenvectors of general arrays.
    eigvalsh : eigenvalues of symmetric or Hermitian arrays
    eigvals_banded : eigenvalues for symmetric/Hermitian band matrices
    eigvalsh_tridiagonal : eigenvalues of symmetric/Hermitian tridiagonal
        matrices

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import linalg
    >>> a = np.array([[0., -1.], [1., 0.]])
    >>> linalg.eigvals(a)
    array([0.+1.j, 0.-1.j])

    >>> b = np.array([[0., 1.], [1., 1.]])
    >>> linalg.eigvals(a, b)
    array([ 1.+0.j, -1.+0.j])

    >>> a = np.array([[3., 0., 0.], [0., 8., 0.], [0., 0., 7.]])
    >>> linalg.eigvals(a, homogeneous_eigvals=True)
    array([[3.+0.j, 8.+0.j, 7.+0.j],
           [1.+0.j, 1.+0.j, 1.+0.j]])

    """
    # 调用 scipy 的 eig 函数计算特征值
    return eig(a, b=b, left=0, right=0, overwrite_a=overwrite_a,
               check_finite=check_finite,
               homogeneous_eigvals=homogeneous_eigvals)


# 解决复共轭 Hermitian 或实对称矩阵的标准或广义特征值问题
def eigvalsh(a, b=None, *, lower=True, overwrite_a=False,
             overwrite_b=False, type=1, check_finite=True, subset_by_index=None,
             subset_by_value=None, driver=None):
    """
    Solves a standard or generalized eigenvalue problem for a complex
    Hermitian or real symmetric matrix.

    Find eigenvalues array ``w`` of array ``a``, where ``b`` is positive
    definite such that for every eigenvalue λ (i-th entry of w) and its

    """
    eigenvector vi (i-th column of v) satisfies::

                      a @ vi = λ * b @ vi
        vi.conj().T @ a @ vi = λ
        vi.conj().T @ b @ vi = 1


    # 定义特征向量 vi（v 的第 i 列），满足以下关系：
    #
    #                   a @ vi = λ * b @ vi
    #     vi.conj().T @ a @ vi = λ
    #     vi.conj().T @ b @ vi = 1
    #
    # 其中，a 和 b 是输入的复共轭转置或实对称矩阵，λ 是特征值。


    In the standard problem, b is assumed to be the identity matrix.


    # 在标准问题中，假设 b 是单位矩阵。


    Parameters
    ----------
    a : (M, M) array_like
        A complex Hermitian or real symmetric matrix whose eigenvalues will
        be computed.


    # 参数：
    # a : (M, M) 类数组
    #     复共轭转置或实对称矩阵，计算其特征值。


    b : (M, M) array_like, optional
        A complex Hermitian or real symmetric definite positive matrix in.
        If omitted, identity matrix is assumed.


    # b : (M, M) 类数组，可选
    #     复共轭转置或实对称的正定矩阵。如果未提供，则假定为单位矩阵。


    lower : bool, optional
        Whether the pertinent array data is taken from the lower or upper
        triangle of ``a`` and, if applicable, ``b``. (Default: lower)


    # lower : bool，可选
    #     指定从数组的下三角还是上三角获取数据，如果适用，也适用于 b。（默认为下三角）


    overwrite_a : bool, optional
        Whether to overwrite data in ``a`` (may improve performance). Default
        is False.


    # overwrite_a : bool，可选
    #     是否覆盖数据在 a 中（可能提高性能）。默认为 False。


    overwrite_b : bool, optional
        Whether to overwrite data in ``b`` (may improve performance). Default
        is False.


    # overwrite_b : bool，可选
    #     是否覆盖数据在 b 中（可能提高性能）。默认为 False。


    type : int, optional
        For the generalized problems, this keyword specifies the problem type
        to be solved for ``w`` and ``v`` (only takes 1, 2, 3 as possible
        inputs)::


    # type : int，可选
    #     对于广义问题，此关键字指定解决问题类型为 ``w`` 和 ``v`` （仅接受 1、2、3 作为可能的输入）::


            1 =>     a @ v = w @ b @ v
            2 => a @ b @ v = w @ v
            3 => b @ a @ v = w @ v


    #         1 =>     a @ v = w @ b @ v
    #         2 => a @ b @ v = w @ v
    #         3 => b @ a @ v = w @ v
    #
    # 此关键字仅用于广义问题。


    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.


    # check_finite : bool，可选
    #     是否检查输入矩阵是否只包含有限数。禁用可能会提高性能，但如果输入包含无穷大或 NaN，
    #     可能会导致问题（崩溃、不终止）。


    subset_by_index : iterable, optional
        If provided, this two-element iterable defines the start and the end
        indices of the desired eigenvalues (ascending order and 0-indexed).
        To return only the second smallest to fifth smallest eigenvalues,
        ``[1, 4]`` is used. ``[n-3, n-1]`` returns the largest three. Only
        available with "evr", "evx", and "gvx" drivers. The entries are
        directly converted to integers via ``int()``.


    # subset_by_index : iterable，可选
    #     如果提供，此两元素迭代器定义所需特征值的开始和结束索引（升序排列和从 0 开始索引）。
    #     例如，返回第二小到第五小的特征值，使用 ``[1, 4]``。``[n-3, n-1]`` 返回最大的三个特征值。
    #     仅适用于 "evr"、"evx" 和 "gvx" 驱动程序。条目通过 ``int()`` 直接转换为整数。


    subset_by_value : iterable, optional
        If provided, this two-element iterable defines the half-open interval
        ``(a, b]`` that, if any, only the eigenvalues between these values
        are returned. Only available with "evr", "evx", and "gvx" drivers. Use
        ``np.inf`` for the unconstrained ends.


    # subset_by_value : iterable，可选
    #     如果提供，此两元素迭代器定义半开区间 ``(a, b]`` ，如果存在，则仅返回这些值之间的特征值。
    #     仅适用于 "evr"、"evx" 和 "gvx" 驱动程序。使用 ``np.inf`` 表示无约束的端点。


    driver : str, optional
        Defines which LAPACK driver should be used. Valid options are "ev",
        "evd", "evr", "evx" for standard problems and "gv", "gvd", "gvx" for
        generalized (where b is not None) problems. See the Notes section of
        `scipy.linalg.eigh`.


    # driver : str，可选
    #     定义应使用哪个 LAPACK 驱动程序。标准问题的有效选项为 "ev"、"evd"、"evr"、"evx"，
    #     广义问题（b 不为 None）的有效选项为 "gv"、"gvd"、"gvx"。请参阅 `scipy.linalg.eigh` 的说明部分。


    Returns
    -------
    w : (N,) ndarray
        The N (N<=M) selected eigenvalues, in ascending order, each
        repeated according to its multiplicity.


    # 返回
    # -------
    # w : (N,) ndarray
    #     选择的特征值 w，按升序排列，每个特征值的重复次数相应增加。


    Raises
    ------


    # Raises
    # ------
    # （未完整列出异常）
    # 返回对称/Hermitian矩阵的特征值，只计算特征值而不计算特征向量。
    # 这个函数相当于使用 `scipy.linalg.eigh` 函数，并设置 `eigvals_only=True`。
    # 参见 `scipy.linalg.eigh` 获取更多对称/Hermitian数组的特征值和右特征向量。
    # 如果输入矩阵不是对称或共轭对称的，不会报错，但结果将是错误的。

    return eigh(a,                # 主矩阵
                b=b,               # 可选的矩阵 b，通常用于广义特征值问题
                lower=lower,       # 是否使用下三角矩阵（对称矩阵求解时）
                eigvals_only=True, # 仅计算特征值，不计算特征向量
                overwrite_a=overwrite_a,   # 是否允许重写主矩阵 a
                overwrite_b=overwrite_b,   # 是否允许重写矩阵 b
                type=type,         # 特征值问题的类型，一般默认即可
                check_finite=check_finite, # 是否检查输入矩阵的有限性
                subset_by_index=subset_by_index,   # 通过索引子集选择特征值
                subset_by_value=subset_by_value,   # 通过值子集选择特征值
                driver=driver      # LAPACK 驱动程序的选择
                )
# 解决实数对称或复数共轭厄米特带状矩阵的特征值问题。
#
# 找到矩阵 a 的特征值 w，并满足以下方程：
#     a v[:,i] = w[i] v[:,i]
#     v.H v    = identity
#
# 矩阵 a 存储在 a_band 中，可以是以下三角或上三角形式存储：
#     - 上三角形式：a_band[u + i - j, j] == a[i,j] （如果 i <= j）
#     - 下三角形式：a_band[i - j, j] == a[i,j] （如果 i >= j）
# 其中 u 是对角线上方的带数目。
#
# Parameters 参数:
# ----------
# a_band : (u+1, M) array_like
#     MxM 矩阵 a 的带状表示。
# lower : bool, optional
#     矩阵是否以下三角形式存储（默认为上三角形式）。
# overwrite_a_band : bool, optional
#     是否覆盖 a_band 中的数据（可能提升性能）。
# select : {'a', 'v', 'i'}, optional
#     计算哪些特征值：
#         'a' - 所有特征值
#         'v' - 在区间 (min, max] 内的特征值
#         'i' - 在索引范围 min <= i <= max 内的特征值
# select_range : (min, max), optional
#     选定特征值的范围。
# check_finite : bool, optional
#     是否检查输入矩阵只包含有限数值。
#     禁用此选项可能提升性能，但如果输入包含无穷大或 NaN，可能会导致问题（崩溃或非终止）。
#
# Returns 返回:
# -------
# w : (M,) ndarray
#     特征值，按升序排列，每个特征值按其重数重复。
#
# Raises 抛出:
# ------
# LinAlgError
#     如果特征值计算不收敛。
#
# See Also 参见:
# --------
# eig_banded : 解对称/厄米特带状矩阵的特征值和右特征向量
# eigvalsh_tridiagonal : 解对称/厄米特三对角矩阵的特征值
# eigvals : 解一般数组的特征值
# eigh : 解对称/厄米特数组的特征值和右特征向量
# eig : 解非对称数组的特征值和右特征向量
#
# Examples 示例:
# --------
# >>> import numpy as np
# >>> from scipy.linalg import eigvals_banded
# >>> A = np.array([[1, 5, 2, 0], [5, 2, 5, 2], [2, 5, 3, 5], [0, 2, 5, 4]])
# >>> Ab = np.array([[1, 2, 3, 4], [5, 5, 5, 0], [2, 2, 0, 0]])
# >>> w = eigvals_banded(Ab, lower=True)
# >>> w
# array([-4.26200532, -2.22987175,  3.95222349, 12.53965359])
    # 返回带宽矩阵 `a_band` 的带状矩阵的特征值，仅返回特征值
    # 参数解释：
    # - `a_band`: 带状矩阵输入
    # - `lower`: 控制是否返回特征值的位置，默认为下三角形式
    # - `eigvals_only=1`: 仅返回特征值，而不返回特征向量
    # - `overwrite_a_band`: 控制是否覆盖输入矩阵
    # - `select`: 控制如何选择要计算的特征值
    # - `select_range`: 控制特征值的计算范围
    # - `check_finite`: 控制输入是否需要有限性检查
    return eig_banded(a_band, lower=lower, eigvals_only=1,
                      overwrite_a_band=overwrite_a_band, select=select,
                      select_range=select_range, check_finite=check_finite)
# 解决实对称三对角矩阵的特征值问题。
# 计算矩阵 ``a`` 的特征值 `w`，满足以下条件：
# 
#     a v[:,i] = w[i] v[:,i]
#     v.H v    = identity
# 
# 其中矩阵 ``a`` 是一个实对称矩阵，其对角元素为 `d`，非对角元素为 `e`。
def eigvalsh_tridiagonal(d, e, select='a', select_range=None,
                         check_finite=True, tol=0., lapack_driver='auto'):
    """
    Solve eigenvalue problem for a real symmetric tridiagonal matrix.

    Find eigenvalues `w` of ``a``::

        a v[:,i] = w[i] v[:,i]
        v.H v    = identity

    For a real symmetric matrix ``a`` with diagonal elements `d` and
    off-diagonal elements `e`.

    Parameters
    ----------
    d : ndarray, shape (ndim,)
        The diagonal elements of the array.
    e : ndarray, shape (ndim-1,)
        The off-diagonal elements of the array.
    select : {'a', 'v', 'i'}, optional
        Which eigenvalues to calculate

        ======  ========================================
        select  calculated
        ======  ========================================
        'a'     All eigenvalues
        'v'     Eigenvalues in the interval (min, max]
        'i'     Eigenvalues with indices min <= i <= max
        ======  ========================================
    select_range : (min, max), optional
        Range of selected eigenvalues
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
    tol : float
        The absolute tolerance to which each eigenvalue is required
        (only used when ``lapack_driver='stebz'``).
        An eigenvalue (or cluster) is considered to have converged if it
        lies in an interval of this width. If <= 0. (default),
        the value ``eps*|a|`` is used where eps is the machine precision,
        and ``|a|`` is the 1-norm of the matrix ``a``.
    lapack_driver : str
        LAPACK function to use, can be 'auto', 'stemr', 'stebz',  'sterf',
        or 'stev'. When 'auto' (default), it will use 'stemr' if ``select='a'``
        and 'stebz' otherwise. 'sterf' and 'stev' can only be used when
        ``select='a'``.

    Returns
    -------
    w : (M,) ndarray
        The eigenvalues, in ascending order, each repeated according to its
        multiplicity.

    Raises
    ------
    LinAlgError
        If eigenvalue computation does not converge.

    See Also
    --------
    eigh_tridiagonal : eigenvalues and right eiegenvectors for
        symmetric/Hermitian tridiagonal matrices

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import eigvalsh_tridiagonal, eigvalsh
    >>> d = 3*np.ones(4)
    >>> e = -1*np.ones(3)
    >>> w = eigvalsh_tridiagonal(d, e)
    >>> A = np.diag(d) + np.diag(e, k=1) + np.diag(e, k=-1)
    >>> w2 = eigvalsh(A)  # Verify with other eigenvalue routines
    >>> np.allclose(w - w2, np.zeros(4))
    True
    """
    # 调用 `eigh_tridiagonal` 函数解决特征值问题，只返回特征值而不返回特征向量
    return eigh_tridiagonal(
        d, e, eigvals_only=True, select=select, select_range=select_range,
        check_finite=check_finite, tol=tol, lapack_driver=lapack_driver)
# 定义函数，解决实对称三对角矩阵的特征值问题
def eigh_tridiagonal(d, e, eigvals_only=False, select='a', select_range=None,
                     check_finite=True, tol=0., lapack_driver='auto'):
    """
    Solve eigenvalue problem for a real symmetric tridiagonal matrix.

    Find eigenvalues `w` and optionally right eigenvectors `v` of ``a``::

        a v[:,i] = w[i] v[:,i]
        v.H v    = identity

    For a real symmetric matrix ``a`` with diagonal elements `d` and
    off-diagonal elements `e`.

    Parameters
    ----------
    d : ndarray, shape (ndim,)
        The diagonal elements of the array.
    e : ndarray, shape (ndim-1,)
        The off-diagonal elements of the array.
    eigvals_only : bool, optional
        Compute only the eigenvalues and no eigenvectors.
        (Default: calculate also eigenvectors)
    select : {'a', 'v', 'i'}, optional
        Which eigenvalues to calculate

        ======  ========================================
        select  calculated
        ======  ========================================
        'a'     All eigenvalues
        'v'     Eigenvalues in the interval (min, max]
        'i'     Eigenvalues with indices min <= i <= max
        ======  ========================================
    select_range : (min, max), optional
        Range of selected eigenvalues
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
    tol : float
        The absolute tolerance to which each eigenvalue is required
        (only used when 'stebz' is the `lapack_driver`).
        An eigenvalue (or cluster) is considered to have converged if it
        lies in an interval of this width. If <= 0. (default),
        the value ``eps*|a|`` is used where eps is the machine precision,
        and ``|a|`` is the 1-norm of the matrix ``a``.
    lapack_driver : str
        LAPACK function to use, can be 'auto', 'stemr', 'stebz', 'sterf',
        or 'stev'. When 'auto' (default), it will use 'stemr' if ``select='a'``
        and 'stebz' otherwise. When 'stebz' is used to find the eigenvalues and
        ``eigvals_only=False``, then a second LAPACK call (to ``?STEIN``) is
        used to find the corresponding eigenvectors. 'sterf' can only be
        used when ``eigvals_only=True`` and ``select='a'``. 'stev' can only
        be used when ``select='a'``.

    Returns
    -------
    w : (M,) ndarray
        The eigenvalues, in ascending order, each repeated according to its
        multiplicity.
    v : (M, M) ndarray
        The normalized eigenvector corresponding to the eigenvalue ``w[i]`` is
        the column ``v[:,i]``. Only returned if ``eigvals_only=False``.

    Raises
    ------
    LinAlgError
        If eigenvalue computation does not converge.

    See Also
    --------
    """
    eigvalsh_tridiagonal : eigenvalues of symmetric/Hermitian tridiagonal
        matrices
    eig : eigenvalues and right eigenvectors for non-symmetric arrays
    eigh : eigenvalues and right eigenvectors for symmetric/Hermitian arrays
    eig_banded : eigenvalues and right eigenvectors for symmetric/Hermitian
        band matrices

    Notes
    -----
    This function makes use of LAPACK ``S/DSTEMR`` routines.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import eigh_tridiagonal
    >>> d = 3*np.ones(4)
    >>> e = -1*np.ones(3)
    >>> w, v = eigh_tridiagonal(d, e)
    >>> A = np.diag(d) + np.diag(e, k=1) + np.diag(e, k=-1)
    >>> np.allclose(A @ v - v @ np.diag(w), np.zeros((4, 4)))
    True
    """
    # Validate input arrays `d` and `e` to ensure they are numpy arrays and finite
    d = _asarray_validated(d, check_finite=check_finite)
    e = _asarray_validated(e, check_finite=check_finite)
    
    # Check dimensions and types of `d` and `e`
    for check in (d, e):
        if check.ndim != 1:
            raise ValueError('expected a 1-D array')
        if check.dtype.char in 'GFD':  # complex
            raise TypeError('Only real arrays currently supported')
    
    # Ensure the length relationship between `d` and `e`
    if d.size != e.size + 1:
        raise ValueError(f'd ({d.size}) must have one more element than e ({e.size})')
    
    # Validate and parse optional parameters related to eigenvalue selection
    select, vl, vu, il, iu, _ = _check_select(
        select, select_range, 0, d.size)
    
    # Validate LAPACK driver type
    if not isinstance(lapack_driver, str):
        raise TypeError('lapack_driver must be str')
    drivers = ('auto', 'stemr', 'sterf', 'stebz', 'stev')
    if lapack_driver not in drivers:
        raise ValueError(f'lapack_driver must be one of {drivers}, '
                         f'got {lapack_driver}')
    
    # Automatically determine LAPACK driver if set to 'auto'
    if lapack_driver == 'auto':
        lapack_driver = 'stemr' if select == 0 else 'stebz'

    # Quick exit for the trivial 1x1 case
    if len(d) == 1:
        if select == 1 and (not (vl < d[0] <= vu)):  # request by value
            w = array([])
            v = empty([1, 0], dtype=d.dtype)
        else:  # all and request by index
            w = array([d[0]], dtype=d.dtype)
            v = array([[1.]], dtype=d.dtype)

        # Return only eigenvalues if requested
        if eigvals_only:
            return w
        else:
            return w, v

    # Obtain the LAPACK function corresponding to the chosen driver
    func, = get_lapack_funcs((lapack_driver,), (d, e))
    
    # Determine whether eigenvectors should be computed
    compute_v = not eigvals_only
    
    # Handle different LAPACK drivers: sterf and stev
    if lapack_driver == 'sterf':
        if select != 0:
            raise ValueError('sterf can only be used when select == "a"')
        if not eigvals_only:
            raise ValueError('sterf can only be used when eigvals_only is '
                             'True')
        w, info = func(d, e)
        m = len(w)
    elif lapack_driver == 'stev':
        if select != 0:
            raise ValueError('stev can only be used when select == "a"')
        w, v, info = func(d, e, compute_v=compute_v)
        m = len(w)
    elif lapack_driver == 'stebz':
        # 将 tol 转换为浮点数
        tol = float(tol)
        # 内部名称设为 'stebz'
        internal_name = 'stebz'
        # 获取 LAPACK 函数 stebz
        stebz, = get_lapack_funcs((internal_name,), (d, e))
        # 如果需要计算特征向量，则使用块排序 ('B')，否则使用矩阵排序 ('E')，稍后将重新排序
        order = 'E' if eigvals_only else 'B'
        # 调用 LAPACK 函数 stebz 进行计算
        m, w, iblock, isplit, info = stebz(d, e, select, vl, vu, il, iu, tol,
                                           order)
    else:   # 'stemr'
        # ?STEMR 需要的尺寸是 N 而不是 N-1
        # 创建一个 e_ 数组，比 e 数组多出一个元素
        e_ = empty(e.size+1, e.dtype)
        # 将 e 数组的值复制到 e_ 数组中（除最后一个元素外）
        e_[:-1] = e
        # 获取 LAPACK 函数 stemr_lwork
        stemr_lwork, = get_lapack_funcs(('stemr_lwork',), (d, e))
        # 调用 LAPACK 函数 stemr_lwork 获取所需工作空间和信息
        lwork, liwork, info = stemr_lwork(d, e_, select, vl, vu, il, iu,
                                          compute_v=compute_v)
        # 检查 LAPACK 函数的返回信息
        _check_info(info, 'stemr_lwork')
        # 调用 func 函数计算特征值和（可选）特征向量
        m, w, v, info = func(d, e_, select, vl, vu, il, iu,
                             compute_v=compute_v, lwork=lwork, liwork=liwork)
    # 检查 LAPACK 函数的返回信息
    _check_info(info, lapack_driver + ' (eigh_tridiagonal)')
    # 只保留前 m 个特征值
    w = w[:m]
    if eigvals_only:
        # 如果只需要特征值，则返回特征值 w
        return w
    else:
        # 否则需要计算特征向量
        if lapack_driver == 'stebz':
            # 获取 LAPACK 函数 stein
            func, = get_lapack_funcs(('stein',), (d, e))
            # 调用 LAPACK 函数 stein 计算特征向量 v
            v, info = func(d, e, w, iblock, isplit)
            # 检查 LAPACK 函数的返回信息
            _check_info(info, 'stein (eigh_tridiagonal)',
                        positive='%d eigenvectors failed to converge')
            # 将块排序的结果转换为矩阵排序
            order = argsort(w)
            w, v = w[order], v[:, order]
        else:
            # 对于 'stemr'，只保留前 m 列的特征向量
            v = v[:, :m]
        # 返回特征值 w 和特征向量 v
        return w, v
def _check_info(info, driver, positive='did not converge (LAPACK info=%d)'):
    """Check info return value."""
    # 检查 info 返回值，如果小于 0 抛出异常
    if info < 0:
        raise ValueError('illegal value in argument %d of internal %s'
                         % (-info, driver))
    # 如果 info 大于 0 且有正面信息，抛出线性代数错误
    if info > 0 and positive:
        raise LinAlgError(("%s " + positive) % (driver, info,))


def hessenberg(a, calc_q=False, overwrite_a=False, check_finite=True):
    """
    Compute Hessenberg form of a matrix.

    The Hessenberg decomposition is::

        A = Q H Q^H

    where `Q` is unitary/orthogonal and `H` has only zero elements below
    the first sub-diagonal.

    Parameters
    ----------
    a : (M, M) array_like
        Matrix to bring into Hessenberg form.
    calc_q : bool, optional
        Whether to compute the transformation matrix.  Default is False.
    overwrite_a : bool, optional
        Whether to overwrite `a`; may improve performance.
        Default is False.
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    H : (M, M) ndarray
        Hessenberg form of `a`.
    Q : (M, M) ndarray
        Unitary/orthogonal similarity transformation matrix ``A = Q H Q^H``.
        Only returned if ``calc_q=True``.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import hessenberg
    >>> A = np.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]])
    >>> H, Q = hessenberg(A, calc_q=True)
    >>> H
    array([[  2.        , -11.65843866,   1.42005301,   0.25349066],
           [ -9.94987437,  14.53535354,  -5.31022304,   2.43081618],
           [  0.        ,  -1.83299243,   0.38969961,  -0.51527034],
           [  0.        ,   0.        ,  -3.83189513,   1.07494686]])
    >>> np.allclose(Q @ H @ Q.conj().T - A, np.zeros((4, 4)))
    True
    """
    # 验证并转换输入矩阵 a 为验证过的 ndarray
    a1 = _asarray_validated(a, check_finite=check_finite)
    # 如果不是方阵或者空矩阵，抛出异常
    if len(a1.shape) != 2 or (a1.shape[0] != a1.shape[1]):
        raise ValueError('expected square matrix')
    # 如果要求覆盖 a 或者数据已经被复制过，则覆盖标志为真
    overwrite_a = overwrite_a or (_datacopied(a1, a))

    # 如果 a1 是空矩阵，返回单位矩阵或空矩阵，取决于 calc_q 的设置
    if a1.size == 0:
        h3 = hessenberg(np.eye(3, dtype=a1.dtype))
        h = np.empty(a1.shape, dtype=h3.dtype)
        if not calc_q:
            return h
        else:
            h3, q3 = hessenberg(np.eye(3, dtype=a1.dtype), calc_q=True)
            q = np.empty(a1.shape, dtype=q3.dtype)
            h = np.empty(a1.shape, dtype=h3.dtype)
            return h, q

    # 如果矩阵大小为 2x2 或更小，已经是 Hessenberg 形式，直接返回
    if a1.shape[0] <= 2:
        if calc_q:
            return a1, eye(a1.shape[0])
        return a1

    # 获取 LAPACK 函数 gehrd, gebal, gehrd_lwork
    gehrd, gebal, gehrd_lwork = get_lapack_funcs(('gehrd', 'gebal',
                                                  'gehrd_lwork'), (a1,))
    # 调用 gebal 函数对矩阵进行平衡化
    ba, lo, hi, pivscale, info = gebal(a1, permute=0, overwrite_a=overwrite_a)
    # 检查 info 对象，确保 'gebal (hessenberg)' 的信息验证正确性
    _check_info(info, 'gebal (hessenberg)', positive=False)
    
    # 计算数组 a1 的长度，并将结果赋给变量 n
    n = len(a1)
    
    # 根据 gehrd_lwork 函数计算所需的工作空间大小 lwork，ba 的行数作为参数传入，同时指定 lo 和 hi 的范围
    lwork = _compute_lwork(gehrd_lwork, ba.shape[0], lo=lo, hi=hi)
    
    # 调用 gehrd 函数，进行 Hessenberg 分解，返回的结果分别赋给 hq, tau, info
    hq, tau, info = gehrd(ba, lo=lo, hi=hi, lwork=lwork, overwrite_a=1)
    
    # 再次检查 info 对象，确保 'gehrd (hessenberg)' 的信息验证正确性
    _check_info(info, 'gehrd (hessenberg)', positive=False)
    
    # 从 hq 中提取出上三角矩阵 h
    h = np.triu(hq, -1)
    
    # 如果不需要计算 Q 矩阵，则直接返回 h
    if not calc_q:
        return h
    
    # 获取 lapack 函数 orghr 和 orghr_lwork
    orghr, orghr_lwork = get_lapack_funcs(('orghr', 'orghr_lwork'), (a1,))
    
    # 根据 orghr_lwork 函数计算所需的工作空间大小 lwork，传入参数 n，同时指定 lo 和 hi 的范围
    lwork = _compute_lwork(orghr_lwork, n, lo=lo, hi=hi)
    
    # 调用 orghr 函数，计算正交矩阵 Q，返回结果赋给 q 和 info
    q, info = orghr(a=hq, tau=tau, lo=lo, hi=hi, lwork=lwork, overwrite_a=1)
    
    # 再次检查 info 对象，确保 'orghr (hessenberg)' 的信息验证正确性
    _check_info(info, 'orghr (hessenberg)', positive=False)
    
    # 返回上三角矩阵 h 和正交矩阵 q（如果需要计算 Q）
    return h, q
    """
    Converts complex eigenvalues ``w`` and eigenvectors ``v`` to real
    eigenvalues in a block diagonal form ``wr`` and the associated real
    eigenvectors ``vr``, such that::

        vr @ wr = X @ vr

    continues to hold, where ``X`` is the original array for which ``w`` and
    ``v`` are the eigenvalues and eigenvectors.

    .. versionadded:: 1.1.0

    Parameters
    ----------
    w : (..., M) array_like
        Complex or real eigenvalues, an array or stack of arrays

        Conjugate pairs must not be interleaved, else the wrong result
        will be produced. So ``[1+1j, 1, 1-1j]`` will give a correct result,
        but ``[1+1j, 2+1j, 1-1j, 2-1j]`` will not.

    v : (..., M, M) array_like
        Complex or real eigenvectors, a square array or stack of square arrays.

    Returns
    -------
    wr : (..., M, M) ndarray
        Real diagonal block form of eigenvalues
    vr : (..., M, M) ndarray
        Real eigenvectors associated with ``wr``

    See Also
    --------
    eig : Eigenvalues and right eigenvectors for non-symmetric arrays
    rsf2csf : Convert real Schur form to complex Schur form

    Notes
    -----
    ``w``, ``v`` must be the eigenstructure for some *real* matrix ``X``.
    For example, obtained by ``w, v = scipy.linalg.eig(X)`` or
    ``w, v = numpy.linalg.eig(X)`` in which case ``X`` can also represent
    stacked arrays.

    .. versionadded:: 1.1.0

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 2, 3], [0, 4, 5], [0, -5, 4]])
    >>> X
    array([[ 1,  2,  3],
           [ 0,  4,  5],
           [ 0, -5,  4]])

    >>> from scipy import linalg
    >>> w, v = linalg.eig(X)
    >>> w
    array([ 1.+0.j,  4.+5.j,  4.-5.j])
    >>> v
    array([[ 1.00000+0.j     , -0.01906-0.40016j, -0.01906+0.40016j],
           [ 0.00000+0.j     ,  0.00000-0.64788j,  0.00000+0.64788j],
           [ 0.00000+0.j     ,  0.64788+0.j     ,  0.64788-0.j     ]])

    >>> wr, vr = linalg.cdf2rdf(w, v)
    >>> wr
    array([[ 1.,  0.,  0.],
           [ 0.,  4.,  5.],
           [ 0., -5.,  4.]])
    >>> vr
    array([[ 1.     ,  0.40016, -0.01906],
           [ 0.     ,  0.64788,  0.     ],
           [ 0.     ,  0.     ,  0.64788]])

    >>> vr @ wr
    array([[ 1.     ,  1.69593,  1.9246 ],
           [ 0.     ,  2.59153,  3.23942],
           [ 0.     , -3.23942,  2.59153]])
    >>> X @ vr
    array([[ 1.     ,  1.69593,  1.9246 ],
           [ 0.     ,  2.59153,  3.23942],
           [ 0.     , -3.23942,  2.59153]])
    """

    # Validate input arrays w and v to ensure they are suitable for computation
    w, v = _asarray_validated(w), _asarray_validated(v)

    # Check dimensions of w and v
    if w.ndim < 1:
        raise ValueError('expected w to be at least 1D')
    if v.ndim < 2:
        raise ValueError('expected v to be at least 2D')
    if v.ndim != w.ndim + 1:
        raise ValueError('expected eigenvectors array to have exactly one '
                         'dimension more than eigenvalues array')

    # Determine the size of the last dimension of w, which corresponds to the size of the square matrix
    n = w.shape[-1]
    # 获取矩阵 w 的形状的前面所有维度
    M = w.shape[:-1]
    # 检查 v 的倒数第二维和倒数第一维是否相等，若不相等则引发 ValueError
    if v.shape[-2] != v.shape[-1]:
        raise ValueError('expected v to be a square matrix or stacked square '
                         'matrices: v.shape[-2] = v.shape[-1]')
    # 检查 v 的最后一维是否与 n 相等，若不相等则引发 ValueError
    if v.shape[-1] != n:
        raise ValueError('expected the same number of eigenvalues as '
                         'eigenvectors')

    # 获取每个复数特征值对应的索引
    complex_mask = iscomplex(w)
    # 统计每行中复数特征值的数量
    n_complex = complex_mask.sum(axis=-1)

    # 检查所有复数特征值是否有共轭对
    if not (n_complex % 2 == 0).all():
        raise ValueError('expected complex-conjugate pairs of eigenvalues')

    # 找到复数特征值的索引
    idx = nonzero(complex_mask)
    idx_stack = idx[:-1]
    idx_elem = idx[-1]

    # 过滤出共轭索引，假设对不交错的数组
    j = idx_elem[0::2]
    k = idx_elem[1::2]
    stack_ind = ()
    for i in idx_stack:
        # 断言，确保共轭对不跨越不同的数组
        assert (i[0::2] == i[1::2]).all(), \
                "Conjugate pair spanned different arrays!"
        stack_ind += (i[0::2],)

    # 所有特征值转换为对角形式
    wr = zeros(M + (n, n), dtype=w.real.dtype)
    di = range(n)
    wr[..., di, di] = w.real

    # 将复数特征值转换为实数块对角形式
    wr[stack_ind + (j, k)] = w[stack_ind + (j,)].imag
    wr[stack_ind + (k, j)] = w[stack_ind + (k,)].imag

    # 计算与实数块对角特征值相关联的实数特征向量
    u = zeros(M + (n, n), dtype=np.cdouble)
    u[..., di, di] = 1.0
    u[stack_ind + (j, j)] = 0.5j
    u[stack_ind + (j, k)] = 0.5
    u[stack_ind + (k, j)] = -0.5j
    u[stack_ind + (k, k)] = 0.5

    # 计算矩阵 v 和 u 的乘积（等同于 v @ u）
    vr = einsum('...ij,...jk->...ik', v, u).real

    # 返回计算得到的实数块对角特征值矩阵和对应的实数特征向量矩阵
    return wr, vr
```