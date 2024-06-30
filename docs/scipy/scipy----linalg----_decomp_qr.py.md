# `D:\src\scipysrc\scipy\scipy\linalg\_decomp_qr.py`

```
# 导入 NumPy 库，用于数值计算
import numpy as np

# 从本地模块导入 LAPACK 相关函数
from .lapack import get_lapack_funcs
# 从本地模块导入 _datacopied 函数
from ._misc import _datacopied

# 定义可以公开访问的函数和变量列表
__all__ = ['qr', 'qr_multiply', 'rq']

# 定义一个安全调用 LAPACK 子程序的函数
def safecall(f, name, *args, **kwargs):
    """Call a LAPACK routine, determining lwork automatically and handling
    error return values"""
    # 获取 kwargs 中的 lwork 参数，若为 None 或 -1，则进行预处理
    lwork = kwargs.get("lwork", None)
    if lwork in (None, -1):
        kwargs['lwork'] = -1
        # 调用 LAPACK 子程序，并获取返回结果的 lwork 部分
        ret = f(*args, **kwargs)
        # 将 lwork 转换为整数类型
        kwargs['lwork'] = ret[-2][0].real.astype(np.int_)
    # 再次调用 LAPACK 子程序，获取最终结果
    ret = f(*args, **kwargs)
    # 检查返回结果中是否有负数，若有则抛出 ValueError 异常
    if ret[-1] < 0:
        raise ValueError("illegal value in %dth argument of internal %s"
                         % (-ret[-1], name))
    # 返回处理后的结果，去除掉最后两个无关的元素
    return ret[:-2]

# 对输入的矩阵进行 QR 分解
def qr(a, overwrite_a=False, lwork=None, mode='full', pivoting=False,
       check_finite=True):
    """
    Compute QR decomposition of a matrix.

    Calculate the decomposition ``A = Q R`` where Q is unitary/orthogonal
    and R upper triangular.

    Parameters
    ----------
    a : (M, N) array_like
        Matrix to be decomposed
    overwrite_a : bool, optional
        Whether data in `a` is overwritten (may improve performance if
        `overwrite_a` is set to True by reusing the existing input data
        structure rather than creating a new one.)
    lwork : int, optional
        Work array size, lwork >= a.shape[1]. If None or -1, an optimal size
        is computed.
    mode : {'full', 'r', 'economic', 'raw'}, optional
        Determines what information is to be returned: either both Q and R
        ('full', default), only R ('r') or both Q and R but computed in
        economy-size ('economic', see Notes). The final option 'raw'
        (added in SciPy 0.11) makes the function return two matrices
        (Q, TAU) in the internal format used by LAPACK.
    pivoting : bool, optional
        Whether or not factorization should include pivoting for rank-revealing
        qr decomposition. If pivoting, compute the decomposition
        ``A[:, P] = Q @ R`` as above, but where P is chosen such that the 
        diagonal of R is non-increasing.
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    Q : float or complex ndarray
        Of shape (M, M), or (M, K) for ``mode='economic'``. Not returned
        if ``mode='r'``. Replaced by tuple ``(Q, TAU)`` if ``mode='raw'``.
    R : float or complex ndarray
        Of shape (M, N), or (K, N) for ``mode in ['economic', 'raw']``.
        ``K = min(M, N)``.
    P : int ndarray
        Of shape (N,) for ``pivoting=True``. Not returned if
        ``pivoting=False``.

    Raises
    ------
    LinAlgError
        Raised if decomposition fails

    Notes
    -----
    This is an interface to the LAPACK routines dgeqrf, zgeqrf,
    dorgqr, zungqr, dgeqp3, and zgeqp3.
    """
    # 此处实现 QR 分解的详细说明和处理
    # 如果 `mode=economic`，则 Q 和 R 的形状为 (M, K) 和 (K, N)，而不是标准的 (M,M) 和 (M,N)，这里 `K=min(M,N)`。
    
    Examples
    --------
    >>> import numpy as np
    >>> from scipy import linalg
    >>> rng = np.random.default_rng()
    >>> a = rng.standard_normal((9, 6))
    
    >>> q, r = linalg.qr(a)
    >>> np.allclose(a, np.dot(q, r))
    True
    >>> q.shape, r.shape
    ((9, 9), (9, 6))
    
    >>> r2 = linalg.qr(a, mode='r')
    >>> np.allclose(r, r2)
    True
    
    >>> q3, r3 = linalg.qr(a, mode='economic')
    >>> q3.shape, r3.shape
    ((9, 6), (6, 6))
    
    >>> q4, r4, p4 = linalg.qr(a, pivoting=True)
    >>> d = np.abs(np.diag(r4))
    >>> np.all(d[1:] <= d[:-1])
    True
    >>> np.allclose(a[:, p4], np.dot(q4, r4))
    True
    >>> q4.shape, r4.shape, p4.shape
    ((9, 9), (9, 6), (6,))
    
    >>> q5, r5, p5 = linalg.qr(a, mode='economic', pivoting=True)
    >>> q5.shape, r5.shape, p5.shape
    ((9, 6), (6, 6), (6,))
    
    """
    # 'qr' 曾经是默认选项，相当于 'full'。以下未使用 'full' 或 'qr'。
    # 'raw' 在 qr_multiply 内部使用。
    if mode not in ['full', 'qr', 'r', 'economic', 'raw']:
        raise ValueError("Mode argument should be one of ['full', 'r', "
                         "'economic', 'raw']")
    
    if check_finite:
        # 将数组转换为 numpy 数组，并检查其所有元素是否有限
        a1 = np.asarray_chkfinite(a)
    else:
        # 将数组转换为 numpy 数组，不进行有限性检查
        a1 = np.asarray(a)
    if len(a1.shape) != 2:
        raise ValueError("expected a 2-D array")
    
    M, N = a1.shape
    
    # 处理空数组的情况
    if a1.size == 0:
        K = min(M, N)
    
        if mode not in ['economic', 'raw']:
            # 如果不是 'economic' 或 'raw' 模式，创建 MxM 的空数组 Q 并初始化为单位矩阵
            Q = np.empty_like(a1, shape=(M, M))
            Q[...] = np.identity(M)
            # 创建空数组 R
            R = np.empty_like(a1)
        else:
            # 如果是 'economic' 或 'raw' 模式，创建 MxK 和 KxN 的空数组 Q 和 R
            Q = np.empty_like(a1, shape=(M, K))
            R = np.empty_like(a1, shape=(K, N))
    
        if pivoting:
            # 如果进行了枢轴选取，返回 R 和对应的枢轴数组 jpvt
            Rj = R, np.arange(N, dtype=np.int32)
        else:
            # 如果没有进行枢轴选取，只返回 R
            Rj = R,
    
        if mode == 'r':
            return Rj
        elif mode == 'raw':
            # 如果是 'raw' 模式，返回 qr 和 tau 组成的元组，加上 Rj
            qr = np.empty_like(a1, shape=(M, N))
            tau = np.zeros_like(a1, shape=(K,))
            return ((qr, tau),) + Rj
        return (Q,) + Rj
    
    overwrite_a = overwrite_a or (_datacopied(a1, a))
    
    if pivoting:
        # 如果进行了枢轴选取，使用 LAPACK 的 geqp3 函数求解 QR 分解
        geqp3, = get_lapack_funcs(('geqp3',), (a1,))
        qr, jpvt, tau = safecall(geqp3, "geqp3", a1, overwrite_a=overwrite_a)
        jpvt -= 1  # geqp3 返回基于 1 的索引数组，需要减去 1
    else:
        # 如果没有进行枢轴选取，使用 LAPACK 的 geqrf 函数求解 QR 分解
        geqrf, = get_lapack_funcs(('geqrf',), (a1,))
        qr, tau = safecall(geqrf, "geqrf", a1, lwork=lwork,
                           overwrite_a=overwrite_a)
    
    if mode not in ['economic', 'raw'] or M < N:
        # 如果不是 'economic' 或 'raw' 模式，或者 M < N，则取 qr 的上三角部分作为 R
        R = np.triu(qr)
    else:
        # 如果是 'economic' 或 'raw' 模式，并且 M >= N，则取 qr 的前 N 行和所有列作为 R
        R = np.triu(qr[:N, :])
    
    if pivoting:
        # 如果进行了枢轴选取，返回 R 和对应的枢轴数组 jpvt
        Rj = R, jpvt
    else:
        # 如果没有进行枢轴选取，只返回 R
        Rj = R,
    
    if mode == 'r':
        return Rj
    elif mode == 'raw':
        # 如果是 'raw' 模式，返回 qr 和 tau 组成的元组，加上 Rj
        return ((qr, tau),) + Rj
    
    # 获取 LAPACK 的 orgqr 函数用于计算 Q
    gor_un_gqr, = get_lapack_funcs(('orgqr',), (qr,))
    # 如果 M 小于 N，则进行 QR 分解并返回 Q 矩阵
    if M < N:
        Q, = safecall(gor_un_gqr, "gorgqr/gungqr", qr[:, :M], tau,
                      lwork=lwork, overwrite_a=1)
    # 如果 mode 参数为 'economic'，则进行经济型 QR 分解并返回 Q 矩阵
    elif mode == 'economic':
        Q, = safecall(gor_un_gqr, "gorgqr/gungqr", qr, tau, lwork=lwork,
                      overwrite_a=1)
    else:
        # 否则，创建一个 M × M 的空矩阵 qqr，并将 qr 的前 N 列赋值给 qqr
        t = qr.dtype.char
        qqr = np.empty((M, M), dtype=t)
        qqr[:, :N] = qr
        # 对 qqr 进行 QR 分解并返回 Q 矩阵
        Q, = safecall(gor_un_gqr, "gorgqr/gungqr", qqr, tau, lwork=lwork,
                      overwrite_a=1)

    # 返回 Q 矩阵和 Rj，以元组形式
    return (Q,) + Rj
# 定义一个函数，用于进行 QR 分解并将 Q 与一个矩阵相乘
def qr_multiply(a, c, mode='right', pivoting=False, conjugate=False,
                overwrite_a=False, overwrite_c=False):
    """
    Calculate the QR decomposition and multiply Q with a matrix.

    Calculate the decomposition ``A = Q R`` where Q is unitary/orthogonal
    and R upper triangular. Multiply Q with a vector or a matrix c.

    Parameters
    ----------
    a : (M, N), array_like
        Input array
    c : array_like
        Input array to be multiplied by ``q``.
    mode : {'left', 'right'}, optional
        ``Q @ c`` is returned if mode is 'left', ``c @ Q`` is returned if
        mode is 'right'.
        The shape of c must be appropriate for the matrix multiplications,
        if mode is 'left', ``min(a.shape) == c.shape[0]``,
        if mode is 'right', ``a.shape[0] == c.shape[1]``.
    pivoting : bool, optional
        Whether or not factorization should include pivoting for rank-revealing
        qr decomposition, see the documentation of qr.
    conjugate : bool, optional
        Whether Q should be complex-conjugated. This might be faster
        than explicit conjugation.
    overwrite_a : bool, optional
        Whether data in a is overwritten (may improve performance)
    overwrite_c : bool, optional
        Whether data in c is overwritten (may improve performance).
        If this is used, c must be big enough to keep the result,
        i.e. ``c.shape[0]`` = ``a.shape[0]`` if mode is 'left'.

    Returns
    -------
    CQ : ndarray
        The product of ``Q`` and ``c``.
    R : (K, N), ndarray
        R array of the resulting QR factorization where ``K = min(M, N)``.
    P : (N,) ndarray
        Integer pivot array. Only returned when ``pivoting=True``.

    Raises
    ------
    LinAlgError
        Raised if QR decomposition fails.

    Notes
    -----
    This is an interface to the LAPACK routines ``?GEQRF``, ``?ORMQR``,
    ``?UNMQR``, and ``?GEQP3``.

    .. versionadded:: 0.11.0

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import qr_multiply, qr
    >>> A = np.array([[1, 3, 3], [2, 3, 2], [2, 3, 3], [1, 3, 2]])
    >>> qc, r1, piv1 = qr_multiply(A, 2*np.eye(4), pivoting=1)
    >>> qc
    array([[-1.,  1., -1.],
           [-1., -1.,  1.],
           [-1., -1., -1.],
           [-1.,  1.,  1.]])
    >>> r1
    array([[-6., -3., -5.            ],
           [ 0., -1., -1.11022302e-16],
           [ 0.,  0., -1.            ]])
    >>> piv1
    array([1, 0, 2], dtype=int32)
    >>> q2, r2, piv2 = qr(A, mode='economic', pivoting=1)
    >>> np.allclose(2*q2 - qc, np.zeros((4, 3)))
    True

    """
    # 检查 mode 参数是否有效，必须为 'left' 或 'right'
    if mode not in ['left', 'right']:
        raise ValueError("Mode argument can only be 'left' or 'right' but "
                         f"not '{mode}'")
    # 将 c 转换为数组，并确保数组中没有无穷大或 NaN 值
    c = np.asarray_chkfinite(c)
    # 如果 c 是一维数组，则转换为至少二维数组
    if c.ndim < 2:
        onedim = True
        c = np.atleast_2d(c)
        # 如果 mode 是 'left'，则将 c 转置
        if mode == "left":
            c = c.T
    else:
        onedim = False
    a = np.atleast_2d(np.asarray(a))  # 将数组 a 至少视为二维数组，并确保其为 NumPy 数组
    M, N = a.shape  # 获取数组 a 的形状 M 和 N

    if mode == 'left':
        # 如果 mode 是 'left'
        if c.shape[0] != min(M, N + overwrite_c*(M-N)):
            # 检查 c 的第一维长度是否与 M 和 N + overwrite_c*(M-N) 的最小值相等
            raise ValueError('Array shapes are not compatible for Q @ c'
                             f' operation: {a.shape} vs {c.shape}')
    else:
        # 如果 mode 不是 'left'
        if M != c.shape[1]:
            # 检查 M 是否等于 c 的第二维长度
            raise ValueError('Array shapes are not compatible for c @ Q'
                             f' operation: {c.shape} vs {a.shape}')

    raw = qr(a, overwrite_a, None, "raw", pivoting)  # 对数组 a 进行 QR 分解，返回原始结果 raw
    Q, tau = raw[0]  # 从原始结果中获取 Q 矩阵和 tau 向量

    # 处理空数组的情况
    if c.size == 0:
        return (np.empty_like(c),) + raw[1:]

    gor_un_mqr, = get_lapack_funcs(('ormqr',), (Q,))
    if gor_un_mqr.typecode in ('s', 'd'):
        trans = "T"
    else:
        trans = "C"

    Q = Q[:, :min(M, N)]  # 截取 Q 矩阵的部分列，确保其列数不超过 min(M, N)

    if M > N and mode == "left" and not overwrite_c:
        # 如果 M > N 且 mode 是 "left" 并且不覆盖 c
        if conjugate:
            cc = np.zeros((c.shape[1], M), dtype=c.dtype, order="F")
            cc[:, :N] = c.T  # 将 c 转置后的前 N 列复制到 cc 中
        else:
            cc = np.zeros((M, c.shape[1]), dtype=c.dtype, order="F")
            cc[:N, :] = c  # 将 c 的前 N 行复制到 cc 中
            trans = "N"
        if conjugate:
            lr = "R"
        else:
            lr = "L"
        overwrite_c = True
    elif c.flags["C_CONTIGUOUS"] and trans == "T" or conjugate:
        # 如果 c 是 C 连续的并且 trans 是 "T" 或者是共轭的情况
        cc = c.T  # 将 c 转置后赋值给 cc
        if mode == "left":
            lr = "R"
        else:
            lr = "L"
    else:
        # 否则
        trans = "N"
        cc = c
        if mode == "left":
            lr = "L"
        else:
            lr = "R"
    cQ, = safecall(gor_un_mqr, "gormqr/gunmqr", lr, trans, Q, tau, cc,
                   overwrite_c=overwrite_c)  # 调用 gor_un_mqr 函数执行 gormqr/gunmqr 操作
    if trans != "N":
        cQ = cQ.T  # 如果 trans 不是 "N"，则对 cQ 进行转置
    if mode == "right":
        cQ = cQ[:, :min(M, N)]  # 如果 mode 是 "right"，则截取 cQ 的部分列
    if onedim:
        cQ = cQ.ravel()  # 如果 onedim 为 True，则将 cQ 展平成一维数组

    return (cQ,) + raw[1:]  # 返回结果元组，包含 cQ 和 raw[1:] 的内容
def rq(a, overwrite_a=False, lwork=None, mode='full', check_finite=True):
    """
    Compute RQ decomposition of a matrix.

    Calculate the decomposition ``A = R Q`` where Q is unitary/orthogonal
    and R upper triangular.

    Parameters
    ----------
    a : (M, N) array_like
        Matrix to be decomposed
    overwrite_a : bool, optional
        Whether data in a is overwritten (may improve performance)
    lwork : int, optional
        Work array size, lwork >= a.shape[1]. If None or -1, an optimal size
        is computed.
    mode : {'full', 'r', 'economic'}, optional
        Determines what information is to be returned: either both Q and R
        ('full', default), only R ('r') or both Q and R but computed in
        economy-size ('economic', see Notes).
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    R : float or complex ndarray
        Of shape (M, N) or (M, K) for ``mode='economic'``. ``K = min(M, N)``.
    Q : float or complex ndarray
        Of shape (N, N) or (K, N) for ``mode='economic'``. Not returned
        if ``mode='r'``.

    Raises
    ------
    LinAlgError
        If decomposition fails.

    Notes
    -----
    This is an interface to the LAPACK routines sgerqf, dgerqf, cgerqf, zgerqf,
    sorgrq, dorgrq, cungrq and zungrq.

    If ``mode=economic``, the shapes of Q and R are (K, N) and (M, K) instead
    of (N,N) and (M,N), with ``K=min(M,N)``.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import linalg
    >>> rng = np.random.default_rng()
    >>> a = rng.standard_normal((6, 9))
    >>> r, q = linalg.rq(a)
    >>> np.allclose(a, r @ q)
    True
    >>> r.shape, q.shape
    ((6, 9), (9, 9))
    >>> r2 = linalg.rq(a, mode='r')
    >>> np.allclose(r, r2)
    True
    >>> r3, q3 = linalg.rq(a, mode='economic')
    >>> r3.shape, q3.shape
    ((6, 6), (6, 9))

    """

    # 检查 mode 参数是否有效，必须是 'full', 'r', 'economic' 之一
    if mode not in ['full', 'r', 'economic']:
        raise ValueError(
                 "Mode argument should be one of ['full', 'r', 'economic']")

    # 检查是否需要检查输入矩阵中是否包含非有限数值
    if check_finite:
        a1 = np.asarray_chkfinite(a)
    else:
        a1 = np.asarray(a)
    
    # 确保输入是二维矩阵
    if len(a1.shape) != 2:
        raise ValueError('expected matrix')

    M, N = a1.shape

    # 处理空数组的情况
    if a1.size == 0:
        K = min(M, N)

        # 根据 mode 的不同选择初始化 R 和 Q 的形状
        if not mode == 'economic':
            R = np.empty_like(a1)
            Q = np.empty_like(a1, shape=(N, N))
            Q[...] = np.identity(N)
        else:
            R = np.empty_like(a1, shape=(M, K))
            Q = np.empty_like(a1, shape=(K, N))

        # 如果 mode 是 'r'，则只返回 R
        if mode == 'r':
            return R
        return R, Q

    # 检查是否需要复制数据
    overwrite_a = overwrite_a or (_datacopied(a1, a))

    # 获取 LAPACK 函数接口，此处为 RQ 分解相关函数
    gerqf, = get_lapack_funcs(('gerqf',), (a1,))
    # 调用 safecall 函数执行 GERQF 分解，返回结果 rq 和 tau
    rq, tau = safecall(gerqf, 'gerqf', a1, lwork=lwork,
                       overwrite_a=overwrite_a)
    
    # 如果不是经济模式或者 N < M，则计算 R 矩阵为 rq 的右上三角部分
    if not mode == 'economic' or N < M:
        R = np.triu(rq, N-M)
    else:
        # 否则，计算 R 矩阵为 rq 的后 M 行和后 M 列的右上三角部分
        R = np.triu(rq[-M:, -M:])

    # 如果模式为 'r'，直接返回 R 矩阵
    if mode == 'r':
        return R

    # 获取 LAPACK 中 ORGRQ 函数的接口函数
    gor_un_grq, = get_lapack_funcs(('orgrq',), (rq,))

    # 根据 N 和 M 的大小关系选择不同的计算方式得到 Q 矩阵
    if N < M:
        # 当 N < M 时，使用 GERQF 结果的最后 N 行计算 Q 矩阵
        Q, = safecall(gor_un_grq, "gorgrq/gungrq", rq[-N:], tau, lwork=lwork,
                      overwrite_a=1)
    elif mode == 'economic':
        # 当模式为经济模式时，使用全部 GERQF 结果计算 Q 矩阵
        Q, = safecall(gor_un_grq, "gorgrq/gungrq", rq, tau, lwork=lwork,
                      overwrite_a=1)
    else:
        # 否则，构造一个空的 N×N 矩阵 rq1，将 rq 的后 M 行放入 rq1 中，再计算 Q 矩阵
        rq1 = np.empty((N, N), dtype=rq.dtype)
        rq1[-M:] = rq
        Q, = safecall(gor_un_grq, "gorgrq/gungrq", rq1, tau, lwork=lwork,
                      overwrite_a=1)

    # 返回计算得到的 R 和 Q 矩阵
    return R, Q
```