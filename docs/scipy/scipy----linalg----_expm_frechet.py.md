# `D:\src\scipysrc\scipy\scipy\linalg\_expm_frechet.py`

```
"""Frechet derivative of the matrix exponential."""
import numpy as np
import scipy.linalg

__all__ = ['expm_frechet', 'expm_cond']


def expm_frechet(A, E, method=None, compute_expm=True, check_finite=True):
    """
    Frechet derivative of the matrix exponential of A in the direction E.

    Parameters
    ----------
    A : (N, N) array_like
        Matrix of which to take the matrix exponential.
    E : (N, N) array_like
        Matrix direction in which to take the Frechet derivative.
    method : str, optional
        Choice of algorithm. Should be one of

        - `SPS` (default)
        - `blockEnlarge`

    compute_expm : bool, optional
        Whether to compute also `expm_A` in addition to `expm_frechet_AE`.
        Default is True.
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    expm_A : ndarray
        Matrix exponential of A.
    expm_frechet_AE : ndarray
        Frechet derivative of the matrix exponential of A in the direction E.
    For ``compute_expm = False``, only `expm_frechet_AE` is returned.

    See Also
    --------
    expm : Compute the exponential of a matrix.

    Notes
    -----
    This section describes the available implementations that can be selected
    by the `method` parameter. The default method is *SPS*.

    Method *blockEnlarge* is a naive algorithm.

    Method *SPS* is Scaling-Pade-Squaring [1]_.
    It is a sophisticated implementation which should take
    only about 3/8 as much time as the naive implementation.
    The asymptotics are the same.

    .. versionadded:: 0.13.0

    References
    ----------
    .. [1] Awad H. Al-Mohy and Nicholas J. Higham (2009)
           Computing the Frechet Derivative of the Matrix Exponential,
           with an application to Condition Number Estimation.
           SIAM Journal On Matrix Analysis and Applications.,
           30 (4). pp. 1639-1657. ISSN 1095-7162

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import linalg
    >>> rng = np.random.default_rng()

    >>> A = rng.standard_normal((3, 3))
    >>> E = rng.standard_normal((3, 3))
    >>> expm_A, expm_frechet_AE = linalg.expm_frechet(A, E)
    >>> expm_A.shape, expm_frechet_AE.shape
    ((3, 3), (3, 3))

    Create a 6x6 matrix containing [[A, E], [0, A]]:

    >>> M = np.zeros((6, 6))
    >>> M[:3, :3] = A
    >>> M[:3, 3:] = E
    >>> M[3:, 3:] = A

    >>> expm_M = linalg.expm(M)
    >>> np.allclose(expm_A, expm_M[:3, :3])
    True
    >>> np.allclose(expm_frechet_AE, expm_M[:3, 3:])
    True

    """
    # 根据 check_finite 参数选择是否进行有限性检查，转换输入矩阵为 ndarray 类型
    if check_finite:
        A = np.asarray_chkfinite(A)
        E = np.asarray_chkfinite(E)
    else:
        A = np.asarray(A)
        E = np.asarray(E)
    # 检查矩阵 A 是否为二维且为方阵（行数等于列数），否则引发值错误异常
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected A to be a square matrix')
    
    # 检查矩阵 E 是否为二维且为方阵（行数等于列数），否则引发值错误异常
    if E.ndim != 2 or E.shape[0] != E.shape[1]:
        raise ValueError('expected E to be a square matrix')
    
    # 检查矩阵 A 和 E 的形状是否相同，否则引发值错误异常
    if A.shape != E.shape:
        raise ValueError('expected A and E to be the same shape')
    
    # 如果方法参数 method 为 None，则设置默认值为 'SPS'
    if method is None:
        method = 'SPS'
    
    # 根据选择的方法调用相应的函数进行计算
    if method == 'SPS':
        # 使用 SPS 方法计算矩阵 A 的指数和 Frechet 导数 AE
        expm_A, expm_frechet_AE = expm_frechet_algo_64(A, E)
    elif method == 'blockEnlarge':
        # 使用 blockEnlarge 方法计算矩阵 A 的指数和 Frechet 导数 AE
        expm_A, expm_frechet_AE = expm_frechet_block_enlarge(A, E)
    else:
        # 如果 method 不是预期的值，则引发值错误异常
        raise ValueError('Unknown implementation %s' % method)
    
    # 如果 compute_expm 参数为真，则返回 expm_A 和 expm_frechet_AE
    if compute_expm:
        return expm_A, expm_frechet_AE
    else:
        # 否则，只返回 expm_frechet_AE
        return expm_frechet_AE
# 定义一个辅助函数，用于测试和性能分析，计算 expm(A) 和 frechet(A, E)
def expm_frechet_block_enlarge(A, E):
    n = A.shape[0]  # 获取矩阵 A 的维度
    # 构造一个扩展的矩阵 M，按照给定规则组合 A 和 E
    M = np.vstack([
        np.hstack([A, E]),
        np.hstack([np.zeros_like(A), A])])
    # 计算矩阵 M 的指数函数 expm(M)
    expm_M = scipy.linalg.expm(M)
    # 返回 expm(M) 的前半部分和后半部分，分别对应于 expm(A) 和 frechet(A, E)
    return expm_M[:n, :n], expm_M[:n, n:]


"""
Maximal values ell_m of ||2**-s A|| such that the backward error bound
does not exceed 2**-53.
"""
ell_table_61 = (
        None,  # 第一个元素为 None
        # 以下是一系列的浮点数常量，用于计算特定的误差边界
        2.11e-8,
        3.56e-4,
        1.08e-2,
        6.49e-2,
        2.00e-1,
        4.37e-1,
        7.83e-1,
        1.23e0,
        1.78e0,
        2.42e0,
        3.13e0,
        3.90e0,
        4.74e0,
        5.63e0,
        6.56e0,
        7.52e0,
        8.53e0,
        9.56e0,
        1.06e1,
        1.17e1,
        )


# 下面是从 scipy.sparse.linalg.matfuncs.py 中复制粘贴得到的向量和矩阵 U、V 的定义
# M, Lu, Lv 遵循 (6.11), (6.12), (6.13), (3.3) 中的公式

def _diff_pade3(A, E, ident):
    b = (120., 60., 12., 1.)  # 定义一个元组 b
    A2 = A.dot(A)  # 计算 A 的平方
    M2 = np.dot(A, E) + np.dot(E, A)  # 计算 M2
    # 计算 U, V, Lu, Lv 按照 Pade(3) 公式的定义
    U = A.dot(b[3]*A2 + b[1]*ident)
    V = b[2]*A2 + b[0]*ident
    Lu = A.dot(b[3]*M2) + E.dot(b[3]*A2 + b[1]*ident)
    Lv = b[2]*M2
    return U, V, Lu, Lv


def _diff_pade5(A, E, ident):
    b = (30240., 15120., 3360., 420., 30., 1.)  # 定义一个元组 b
    A2 = A.dot(A)  # 计算 A 的平方
    M2 = np.dot(A, E) + np.dot(E, A)  # 计算 M2
    A4 = np.dot(A2, A2)  # 计算 A 的四次方
    M4 = np.dot(A2, M2) + np.dot(M2, A2)  # 计算 M4
    # 计算 U, V, Lu, Lv 按照 Pade(5) 公式的定义
    U = A.dot(b[5]*A4 + b[3]*A2 + b[1]*ident)
    V = b[4]*A4 + b[2]*A2 + b[0]*ident
    Lu = (A.dot(b[5]*M4 + b[3]*M2) +
            E.dot(b[5]*A4 + b[3]*A2 + b[1]*ident))
    Lv = b[4]*M4 + b[2]*M2
    return U, V, Lu, Lv


def _diff_pade7(A, E, ident):
    b = (17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.)  # 定义一个元组 b
    A2 = A.dot(A)  # 计算 A 的平方
    M2 = np.dot(A, E) + np.dot(E, A)  # 计算 M2
    A4 = np.dot(A2, A2)  # 计算 A 的四次方
    M4 = np.dot(A2, M2) + np.dot(M2, A2)  # 计算 M4
    A6 = np.dot(A2, A4)  # 计算 A 的六次方
    M6 = np.dot(A4, M2) + np.dot(M4, A2)  # 计算 M6
    # 计算 U, V, Lu, Lv 按照 Pade(7) 公式的定义
    U = A.dot(b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident)
    V = b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*ident
    Lu = (A.dot(b[7]*M6 + b[5]*M4 + b[3]*M2) +
            E.dot(b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident))
    Lv = b[6]*M6 + b[4]*M4 + b[2]*M2
    return U, V, Lu, Lv


def _diff_pade9(A, E, ident):
    b = (17643225600., 8821612800., 2075673600., 302702400., 30270240.,
            2162160., 110880., 3960., 90., 1.)  # 定义一个元组 b
    A2 = A.dot(A)  # 计算 A 的平方
    M2 = np.dot(A, E) + np.dot(E, A)  # 计算 M2
    A4 = np.dot(A2, A2)  # 计算 A 的四次方
    M4 = np.dot(A2, M2) + np.dot(M2, A2)  # 计算 M4
    A6 = np.dot(A2, A4)  # 计算 A 的六次方
    M6 = np.dot(A4, M2) + np.dot(M4, A2)  # 计算 M6
    A8 = np.dot(A4, A4)  # 计算 A 的八次方
    M8 = np.dot(A4, M4) + np.dot(M4, A4)  # 计算 M8
    # 计算 U, V, Lu, Lv 按照 Pade(9) 公式的定义
    U = A.dot(b[9]*A8 + b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident)
    V = b[8]*A8 + b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*ident
    Lu = (A.dot(b[9]*M8 + b[7]*M6 + b[5]*M4 + b[3]*M2) +
            E.dot(b[9]*A8 + b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident))
    Lv = b[8]*M8 + b[6]*M6 + b[4]*M4 + b[2]*M2
    return U, V, Lu, Lv


# 定义一个函数 expm_frechet_algo_64，但是没有给出具体实现
def expm_frechet_algo_64(A, E):
    # 获取矩阵 A 的行数
    n = A.shape[0]
    # 初始化变量 s 为 None
    s = None
    # 创建 n 阶单位矩阵 ident
    ident = np.identity(n)
    # 计算 A 的 1-范数
    A_norm_1 = scipy.linalg.norm(A, 1)
    # 预定义不同 Pade逼近阶数 的组合
    m_pade_pairs = (
            (3, _diff_pade3),
            (5, _diff_pade5),
            (7, _diff_pade7),
            (9, _diff_pade9))
    # 遍历 Pade逼近阶数 和 对应的函数
    for m, pade in m_pade_pairs:
        # 如果 A 的 1-范数小于等于预设的 ell_table_61[m]
        if A_norm_1 <= ell_table_61[m]:
            # 计算 Pade逼近 的结果 U, V, Lu, Lv
            U, V, Lu, Lv = pade(A, E, ident)
            # 设置 s 为 0 并结束循环
            s = 0
            break
    # 如果 s 仍为 None
    if s is None:
        # 根据 A 的 1-范数计算 scaling 值 s
        s = max(0, int(np.ceil(np.log2(A_norm_1 / ell_table_61[13]))))
        # 对 A 和 E 进行 scaling
        A = A * 2.0**-s
        E = E * 2.0**-s
        # 计算 Pade阶数为 13 的相关矩阵
        A2 = np.dot(A, A)
        M2 = np.dot(A, E) + np.dot(E, A)
        A4 = np.dot(A2, A2)
        M4 = np.dot(A2, M2) + np.dot(M2, A2)
        A6 = np.dot(A2, A4)
        M6 = np.dot(A4, M2) + np.dot(M4, A2)
        # 定义系数向量 b
        b = (64764752532480000., 32382376266240000., 7771770303897600.,
                1187353796428800., 129060195264000., 10559470521600.,
                670442572800., 33522128640., 1323241920., 40840800., 960960.,
                16380., 182., 1.)
        # 计算 W1, W2, Z1, Z2
        W1 = b[13]*A6 + b[11]*A4 + b[9]*A2
        W2 = b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident
        Z1 = b[12]*A6 + b[10]*A4 + b[8]*A2
        Z2 = b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*ident
        # 计算 W, U, V, Lw1, Lw2, Lz1, Lz2, Lw, Lu, Lv
        W = np.dot(A6, W1) + W2
        U = np.dot(A, W)
        V = np.dot(A6, Z1) + Z2
        Lw1 = b[13]*M6 + b[11]*M4 + b[9]*M2
        Lw2 = b[7]*M6 + b[5]*M4 + b[3]*M2
        Lz1 = b[12]*M6 + b[10]*M4 + b[8]*M2
        Lz2 = b[6]*M6 + b[4]*M4 + b[2]*M2
        Lw = np.dot(A6, Lw1) + np.dot(M6, W1) + Lw2
        Lu = np.dot(A, Lw) + np.dot(E, W)
        Lv = np.dot(A6, Lz1) + np.dot(M6, Z1) + Lz2
    # 使用 LU 分解求解线性方程组 -U + V * R = 0
    lu_piv = scipy.linalg.lu_factor(-U + V)
    # 求解线性方程组 L = L * R + (Lu - Lv) * R
    R = scipy.linalg.lu_solve(lu_piv, U + V)
    L = scipy.linalg.lu_solve(lu_piv, Lu + Lv + np.dot((Lu - Lv), R))
    # 对 L 和 R 进行 s 次平方操作
    for k in range(s):
        L = np.dot(R, L) + np.dot(L, R)
        R = np.dot(R, R)
    # 返回计算结果 R, L
    return R, L
# 将矩阵 M 的列堆叠成一个单一的向量。
#
# 这是线性代数中的标准符号表示。
def vec(M):
    return M.T.ravel()


# 构造 expm 的 Frechet 导数的 Kronecker 形式。
#
# Parameters
# ----------
# A : array_like，形状为 (N, N)
#     要计算 expm 的矩阵。
# method : str, optional
#     传递给 expm_frechet 的额外关键字。
# check_finite : bool, optional
#     是否检查输入矩阵是否只包含有限的数字。
#     禁用此选项可能会提高性能，但如果输入包含无穷大或 NaN，可能会导致问题（崩溃、无法终止）。
#
# Returns
# -------
# K : 2-D ndarray，形状为 (N*N, N*N)
#     expm 的 Frechet 导数的 Kronecker 形式。
#
# Notes
# -----
# 此函数用于帮助计算矩阵指数的条件数。
#
# See Also
# --------
# expm : 计算矩阵指数。
# expm_frechet : 计算矩阵指数的 Frechet 导数。
# expm_cond : 计算矩阵指数在 Frobenius 范数中的相对条件数。
def expm_frechet_kronform(A, method=None, check_finite=True):
    if check_finite:
        A = np.asarray_chkfinite(A)
    else:
        A = np.asarray(A)
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected a square matrix')

    n = A.shape[0]
    ident = np.identity(n)
    cols = []
    for i in range(n):
        for j in range(n):
            E = np.outer(ident[i], ident[j])
            F = expm_frechet(A, E,
                    method=method, compute_expm=False, check_finite=False)
            cols.append(vec(F))
    return np.vstack(cols).T


# 计算矩阵指数在 Frobenius 范数中的相对条件数。
#
# Parameters
# ----------
# A : 2-D array_like
#     形状为 (N, N) 的方阵输入。
# check_finite : bool, optional
#     是否检查输入矩阵是否只包含有限的数字。
#     禁用此选项可能会提高性能，但如果输入包含无穷大或 NaN，可能会导致问题（崩溃、无法终止）。
#
# Returns
# -------
# kappa : float
#     矩阵指数在 Frobenius 范数中的相对条件数。
#
# See Also
# --------
# expm : 计算矩阵的指数。
# expm_frechet : 计算矩阵指数的 Frechet 导数。
#
# Notes
# -----
# 用于 1-范数中条件数的更快估计已经发布，但尚未在 SciPy 中实现。
#
# .. versionadded:: 0.14.0
#
# Examples
# --------
# >>> import numpy as np
# >>> from scipy.linalg import expm_cond
def expm_cond(A, check_finite=True):
    pass
    >>> A = np.array([[-0.3, 0.2, 0.6], [0.6, 0.3, -0.1], [-0.7, 1.2, 0.9]])
    >>> k = expm_cond(A)
    >>> k
    1.7787805864469866

    """
    # 如果 check_finite 参数为 True，确保 A 是一个有限的 NumPy 数组
    if check_finite:
        A = np.asarray_chkfinite(A)
    else:
        # 否则，将 A 转换为 NumPy 数组
        A = np.asarray(A)
    # 检查 A 是否为二维方阵，如果不是则抛出 ValueError 异常
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected a square matrix')

    # 计算矩阵的指数映射
    X = scipy.linalg.expm(A)
    # 计算 Frechet derivative K 的 Kronecker 形式，check_finite 参数默认为 False
    K = expm_frechet_kronform(A, check_finite=False)

    # 以下选择的范数是有意义的：
    # A 和 X 的范数为 Frobenius 范数，
    # K 的范数为 2-范数（induced 2-norm）。
    A_norm = scipy.linalg.norm(A, 'fro')
    X_norm = scipy.linalg.norm(X, 'fro')
    K_norm = scipy.linalg.norm(K, 2)

    # 计算条件数 kappa
    kappa = (K_norm * A_norm) / X_norm
    return kappa
```