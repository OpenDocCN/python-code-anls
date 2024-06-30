# `D:\src\scipysrc\scipy\scipy\linalg\_decomp_svd.py`

```
# 导入必要的库和模块
import numpy as np
# 从 numpy 中导入需要的函数和类
from numpy import zeros, r_, diag, dot, arccos, arcsin, where, clip

# 从本地模块中导入异常类和函数
from ._misc import LinAlgError, _datacopied
# 从 LAPACK 模块中导入函数和计算工作空间的函数
from .lapack import get_lapack_funcs, _compute_lwork
# 从分解模块中导入数组验证函数
from ._decomp import _asarray_validated

# 指定可以通过 from package import * 导入的公共接口
__all__ = ['svd', 'svdvals', 'diagsvd', 'orth', 'subspace_angles', 'null_space']


def svd(a, full_matrices=True, compute_uv=True, overwrite_a=False,
        check_finite=True, lapack_driver='gesdd'):
    """
    Singular Value Decomposition.

    Factorizes the matrix `a` into two unitary matrices ``U`` and ``Vh``, and
    a 1-D array ``s`` of singular values (real, non-negative) such that
    ``a == U @ S @ Vh``, where ``S`` is a suitably shaped matrix of zeros with
    main diagonal ``s``.

    Parameters
    ----------
    a : (M, N) array_like
        Matrix to decompose.
    full_matrices : bool, optional
        If True (default), `U` and `Vh` are of shape ``(M, M)``, ``(N, N)``.
        If False, the shapes are ``(M, K)`` and ``(K, N)``, where
        ``K = min(M, N)``.
    compute_uv : bool, optional
        Whether to compute also ``U`` and ``Vh`` in addition to ``s``.
        Default is True.
    overwrite_a : bool, optional
        Whether to overwrite `a`; may improve performance.
        Default is False.
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
    lapack_driver : {'gesdd', 'gesvd'}, optional
        Whether to use the more efficient divide-and-conquer approach
        (``'gesdd'``) or general rectangular approach (``'gesvd'``)
        to compute the SVD. MATLAB and Octave use the ``'gesvd'`` approach.
        Default is ``'gesdd'``.

        .. versionadded:: 0.18

    Returns
    -------
    U : ndarray
        Unitary matrix having left singular vectors as columns.
        Of shape ``(M, M)`` or ``(M, K)``, depending on `full_matrices`.
    s : ndarray
        The singular values, sorted in non-increasing order.
        Of shape (K,), with ``K = min(M, N)``.
    Vh : ndarray
        Unitary matrix having right singular vectors as rows.
        Of shape ``(N, N)`` or ``(K, N)`` depending on `full_matrices`.

    For ``compute_uv=False``, only ``s`` is returned.

    Raises
    ------
    LinAlgError
        If SVD computation does not converge.

    See Also
    --------
    svdvals : Compute singular values of a matrix.
    diagsvd : Construct the Sigma matrix, given the vector s.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import linalg
    >>> rng = np.random.default_rng()
    >>> m, n = 9, 6
    >>> a = rng.standard_normal((m, n)) + 1.j*rng.standard_normal((m, n))
    >>> U, s, Vh = linalg.svd(a)
    >>> U.shape,  s.shape, Vh.shape
    ((9, 9), (6,), (6, 6))
    """
    Reconstruct the original matrix from the decomposition:

    >>> sigma = np.zeros((m, n))
    >>> for i in range(min(m, n)):
    ...     sigma[i, i] = s[i]
    >>> a1 = np.dot(U, np.dot(sigma, Vh))
    >>> np.allclose(a, a1)
    True

    Alternatively, use ``full_matrices=False`` (notice that the shape of
    ``U`` is then ``(m, n)`` instead of ``(m, m)``):

    >>> U, s, Vh = linalg.svd(a, full_matrices=False)
    >>> U.shape, s.shape, Vh.shape
    ((9, 6), (6,), (6, 6))
    >>> S = np.diag(s)
    >>> np.allclose(a, np.dot(U, np.dot(S, Vh)))
    True

    >>> s2 = linalg.svd(a, compute_uv=False)
    >>> np.allclose(s, s2)
    True

    """
    # 将输入矩阵转换为有效数组，并检查是否为二维矩阵
    a1 = _asarray_validated(a, check_finite=check_finite)
    if len(a1.shape) != 2:
        raise ValueError('expected matrix')
    m, n = a1.shape

    # 处理空矩阵的情况
    if a1.size == 0:
        # 对于空矩阵，执行特殊处理
        u0, s0, v0 = svd(np.eye(2, dtype=a1.dtype))

        # 创建形状为空的结果数组
        s = np.empty_like(a1, shape=(0,), dtype=s0.dtype)
        if full_matrices:
            # 如果需要完整的 U 和 V 矩阵，创建对应形状的单位矩阵
            u = np.empty_like(a1, shape=(m, m), dtype=u0.dtype)
            u[...] = np.identity(m)
            v = np.empty_like(a1, shape=(n, n), dtype=v0.dtype)
            v[...] = np.identity(n)
        else:
            # 否则，创建形状为空的 U 和 V 矩阵
            u = np.empty_like(a1, shape=(m, 0), dtype=u0.dtype)
            v = np.empty_like(a1, shape=(0, n), dtype=v0.dtype)
        if compute_uv:
            return u, s, v
        else:
            return s

    # 检查是否需要复制输入数组
    overwrite_a = overwrite_a or (_datacopied(a1, a))

    # 检查 lapack_driver 类型是否为字符串
    if not isinstance(lapack_driver, str):
        raise TypeError('lapack_driver must be a string')
    # 检查 lapack_driver 是否为有效的字符串值
    if lapack_driver not in ('gesdd', 'gesvd'):
        message = f'lapack_driver must be "gesdd" or "gesvd", not "{lapack_driver}"'
        raise ValueError(message)

    # 当使用 gesdd 并且需要计算 U 和 V 时，执行以下操作
    if lapack_driver == 'gesdd' and compute_uv:
        # 根据矩阵的大小确定合适的索引类型和大小限制
        max_mn, min_mn = (m, n) if m > n else (n, m)
        if full_matrices:
            # 如果需要完整的 U 和 V 矩阵，检查索引是否超出 int32 范围
            if max_mn * max_mn > np.iinfo(np.int32).max:
                raise ValueError(f"Indexing a matrix size {max_mn} x {max_mn} "
                                  " would incur integer overflow in LAPACK.")
        else:
            # 否则，计算元素数目并检查是否超出 int32 范围
            sz = max(m * min_mn, n * min_mn)
            if sz > np.iinfo(np.int32).max:
                raise ValueError(f"Indexing a matrix of {sz} elements would "
                                  "incur an in integer overflow in LAPACK.")

    # 获取 LAPACK 函数及其工作长度函数
    funcs = (lapack_driver, lapack_driver + '_lwork')
    gesXd, gesXd_lwork = get_lapack_funcs(funcs, (a1,), ilp64='preferred')

    # 计算最优的工作空间长度
    lwork = _compute_lwork(gesXd_lwork, a1.shape[0], a1.shape[1],
                           compute_uv=compute_uv, full_matrices=full_matrices)

    # 执行矩阵分解
    u, s, v, info = gesXd(a1, compute_uv=compute_uv, lwork=lwork,
                          full_matrices=full_matrices, overwrite_a=overwrite_a)

    # 检查分解是否成功收敛
    if info > 0:
        raise LinAlgError("SVD did not converge")
    # 检查返回的信息是否小于零，如果是则抛出值错误异常，指示在 gesdd 内部的第 info 个参数中存在非法值。
    if info < 0:
        raise ValueError('illegal value in %dth argument of internal gesdd'
                         % -info)
    # 如果 compute_uv 为真，则返回计算的 u、s、v。
    if compute_uv:
        return u, s, v
    # 如果 compute_uv 不为真，则仅返回 s。
    else:
        return s
# 计算矩阵的奇异值。

def svdvals(a, overwrite_a=False, check_finite=True):
    """
    Compute singular values of a matrix.

    Parameters
    ----------
    a : (M, N) array_like
        Matrix to decompose.
    overwrite_a : bool, optional
        Whether to overwrite `a`; may improve performance.
        Default is False.
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    s : (min(M, N),) ndarray
        The singular values, sorted in decreasing order.

    Raises
    ------
    LinAlgError
        If SVD computation does not converge.

    See Also
    --------
    svd : Compute the full singular value decomposition of a matrix.
    diagsvd : Construct the Sigma matrix, given the vector s.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import svdvals
    >>> m = np.array([[1.0, 0.0],
    ...               [2.0, 3.0],
    ...               [1.0, 1.0],
    ...               [0.0, 2.0],
    ...               [1.0, 0.0]])
    >>> svdvals(m)
    array([ 4.28091555,  1.63516424])

    We can verify the maximum singular value of `m` by computing the maximum
    length of `m.dot(u)` over all the unit vectors `u` in the (x,y) plane.
    We approximate "all" the unit vectors with a large sample. Because
    of linearity, we only need the unit vectors with angles in [0, pi].

    >>> t = np.linspace(0, np.pi, 2000)
    >>> u = np.array([np.cos(t), np.sin(t)])
    >>> np.linalg.norm(m.dot(u), axis=0).max()
    4.2809152422538475

    `p` is a projection matrix with rank 1. With exact arithmetic,
    its singular values would be [1, 0, 0, 0].

    >>> v = np.array([0.1, 0.3, 0.9, 0.3])
    >>> p = np.outer(v, v)
    >>> svdvals(p)
    array([  1.00000000e+00,   2.02021698e-17,   1.56692500e-17,
             8.15115104e-34])

    The singular values of an orthogonal matrix are all 1. Here, we
    create a random orthogonal matrix by using the `rvs()` method of
    `scipy.stats.ortho_group`.

    >>> from scipy.stats import ortho_group
    >>> orth = ortho_group.rvs(4)
    >>> svdvals(orth)
    array([ 1.,  1.,  1.,  1.])

    """
    # 调用 `svd` 函数计算矩阵 `a` 的奇异值，返回不计算 U 和 V 矩阵的结果
    return svd(a, compute_uv=0, overwrite_a=overwrite_a,
               check_finite=check_finite)
    # 导入需要的库
    import numpy as np
    from scipy.linalg import diagsvd
    
    # 给定的数组表示计算得到的奇异值分解 (SVD)
    vals = np.array([1, 2, 3])
    
    # 调用 diagsvd 函数进行奇异值对角矩阵的创建
    # 第一个示例：创建一个 3x4 的对角矩阵
    diagsvd(vals, 3, 4)
    # 返回一个 3x4 的数组，对角线上的元素为 [1, 2, 3]，其余元素为 0
    
    # 第二个示例：创建一个 4x3 的对角矩阵
    diagsvd(vals, 4, 3)
    # 返回一个 4x3 的数组，对角线上的元素为 [1, 2, 3]，其余元素为 0
    
    """
    根据奇异值数组 s 的长度 MorN 和给定的 M 和 N，生成特定形状的对角矩阵或补零。
    """
    part = diag(s)  # 从奇异值数组 s 创建对角矩阵
    typ = part.dtype.char  # 获取对角矩阵的元素类型字符码
    MorN = len(s)  # 获取奇异值数组 s 的长度
    
    # 根据 MorN 与 M 的比较，决定如何补零以达到目标形状
    if MorN == M:
        return np.hstack((part, zeros((M, N - M), dtype=typ)))  # 如果 MorN 等于 M，则横向补零
    elif MorN == N:
        return r_[part, zeros((M - N, N), dtype=typ)]  # 如果 MorN 等于 N，则纵向补零
    else:
        raise ValueError("Length of s must be M or N.")  # 如果 MorN 既不等于 M 也不等于 N，则抛出异常
# Orthonormal decomposition

def orth(A, rcond=None):
    """
    Construct an orthonormal basis for the range of A using SVD

    Parameters
    ----------
    A : (M, N) array_like
        Input array
    rcond : float, optional
        Relative condition number. Singular values ``s`` smaller than
        ``rcond * max(s)`` are considered zero.
        Default: floating point eps * max(M,N).

    Returns
    -------
    Q : (M, K) ndarray
        Orthonormal basis for the range of A.
        K = effective rank of A, as determined by rcond

    See Also
    --------
    svd : Singular value decomposition of a matrix
    null_space : Matrix null space

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import orth
    >>> A = np.array([[2, 0, 0], [0, 5, 0]])  # rank 2 array
    >>> orth(A)
    array([[0., 1.],
           [1., 0.]])
    >>> orth(A.T)
    array([[0., 1.],
           [1., 0.],
           [0., 0.]])

    """
    # Perform Singular Value Decomposition (SVD) on A
    u, s, vh = svd(A, full_matrices=False)
    # Dimensions of matrices u and vh
    M, N = u.shape[0], vh.shape[1]
    # Calculate default rcond if not provided
    if rcond is None:
        rcond = np.finfo(s.dtype).eps * max(M, N)
    # Compute tolerance for singular values
    tol = np.amax(s, initial=0.) * rcond
    # Number of non-zero singular values determines the size of Q
    num = np.sum(s > tol, dtype=int)
    # Construct the orthonormal basis Q for the range of A
    Q = u[:, :num]
    return Q


def null_space(A, rcond=None):
    """
    Construct an orthonormal basis for the null space of A using SVD

    Parameters
    ----------
    A : (M, N) array_like
        Input array
    rcond : float, optional
        Relative condition number. Singular values ``s`` smaller than
        ``rcond * max(s)`` are considered zero.
        Default: floating point eps * max(M,N).

    Returns
    -------
    Z : (N, K) ndarray
        Orthonormal basis for the null space of A.
        K = dimension of effective null space, as determined by rcond

    See Also
    --------
    svd : Singular value decomposition of a matrix
    orth : Matrix range

    Examples
    --------
    1-D null space:

    >>> import numpy as np
    >>> from scipy.linalg import null_space
    >>> A = np.array([[1, 1], [1, 1]])
    >>> ns = null_space(A)
    >>> ns * np.copysign(1, ns[0,0])  # Remove the sign ambiguity of the vector
    array([[ 0.70710678],
           [-0.70710678]])

    2-D null space:

    >>> from numpy.random import default_rng
    >>> rng = default_rng()
    >>> B = rng.random((3, 5))
    >>> Z = null_space(B)
    >>> Z.shape
    (5, 2)
    >>> np.allclose(B.dot(Z), 0)
    True

    The basis vectors are orthonormal (up to rounding error):

    >>> Z.T.dot(Z)
    array([[  1.00000000e+00,   6.92087741e-17],
           [  6.92087741e-17,   1.00000000e+00]])

    """
    # Perform Singular Value Decomposition (SVD) on A
    u, s, vh = svd(A, full_matrices=True)
    # Dimensions of matrices u and vh
    M, N = u.shape[0], vh.shape[1]
    # Calculate default rcond if not provided
    if rcond is None:
        rcond = np.finfo(s.dtype).eps * max(M, N)
    # Compute tolerance for singular values
    tol = np.amax(s, initial=0.) * rcond
    # Number of zero singular values determines the size of Q
    num = np.sum(s > tol, dtype=int)
    # Construct the orthonormal basis Q for the null space of A
    Q = vh[num:,:].T.conj()
    return Q


def subspace_angles(A, B):
    r"""
    Compute the subspace angles between two matrices.

    Parameters
    ----------
    A, B : array_like
        Input arrays

    """
    """
    A : (M, N) array_like
        第一个输入数组，形状为 (M, N)。
    B : (M, K) array_like
        第二个输入数组，形状为 (M, K)。

    Returns
    -------
    angles : ndarray, shape (min(N, K),)
        `A` 和 `B` 列空间之间的子空间角度，按降序排列。

    See Also
    --------
    orth
    svd

    Notes
    -----
    根据文献 [1]_ 中的公式计算子空间角度。
    为了与 MATLAB 和 Octave 的行为保持一致，使用 `angles[0]`。

    .. versionadded:: 1.0

    References
    ----------
    .. [1] Knyazev A, Argentati M (2002) Principal Angles between Subspaces
           in an A-Based Scalar Product: Algorithms and Perturbation
           Estimates. SIAM J. Sci. Comput. 23:2008-2040.

    Examples
    --------
    哈达玛矩阵具有正交列，因此我们期望子空间角度为 :math:`\frac{\pi}{2}`：

    >>> import numpy as np
    >>> from scipy.linalg import hadamard, subspace_angles
    >>> rng = np.random.default_rng()
    >>> H = hadamard(4)
    >>> print(H)
    [[ 1  1  1  1]
     [ 1 -1  1 -1]
     [ 1  1 -1 -1]
     [ 1 -1 -1  1]]
    >>> np.rad2deg(subspace_angles(H[:, :2], H[:, 2:]))
    array([ 90.,  90.])

    矩阵与自身的子空间角度应为零：

    >>> subspace_angles(H[:, :2], H[:, :2]) <= 2 * np.finfo(float).eps
    array([ True,  True], dtype=bool)

    非正交子空间之间的角度在这些极端之间：

    >>> x = rng.standard_normal((4, 3))
    >>> np.rad2deg(subspace_angles(x[:, :2], x[:, [2]]))
    array([ 55.832])  # 随机生成的值
    """
    # 步骤中省略了从文献中计算 U 和 V 的步骤

    # 1. 计算列空间的正交基
    A = _asarray_validated(A, check_finite=True)
    if len(A.shape) != 2:
        raise ValueError(f'expected 2D array, got shape {A.shape}')
    QA = orth(A)
    del A

    B = _asarray_validated(B, check_finite=True)
    if len(B.shape) != 2:
        raise ValueError(f'expected 2D array, got shape {B.shape}')
    if len(B) != len(QA):
        raise ValueError('A and B must have the same number of rows, got '
                         f'{QA.shape[0]} and {B.shape[0]}')
    QB = orth(B)
    del B

    # 2. 计算余弦的奇异值分解（SVD）
    QA_H_QB = dot(QA.T.conj(), QB)
    sigma = svdvals(QA_H_QB)

    # 3. 计算矩阵 B
    if QA.shape[1] >= QB.shape[1]:
        B = QB - dot(QA, QA_H_QB)
    else:
        B = QA - dot(QB, QA_H_QB.T.conj())
    del QA, QB, QA_H_QB

    # 4. 计算正弦的奇异值分解（SVD）
    mask = sigma ** 2 >= 0.5
    if mask.any():
        mu_arcsin = arcsin(clip(svdvals(B, overwrite_a=True), -1., 1.))
    else:
        mu_arcsin = 0.

    # 5. 计算主角度
    # 将 sigma 的顺序反转，因为最小的 sigma 对应最大的角度 theta
    theta = where(mask, mu_arcsin, arccos(clip(sigma[::-1], -1., 1.)))
    return theta
```