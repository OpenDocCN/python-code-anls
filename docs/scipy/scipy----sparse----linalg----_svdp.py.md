# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_svdp.py`

```
def _svdp(A, k, which='LM', irl_mode=True, kmax=None,
          compute_u=True, compute_v=True, v0=None, full_output=False, tol=0,
          delta=None, eta=None, anorm=0, cgs=False, elr=True,
          min_relgap=0.002, shifts=None, maxiter=None, random_state=None):
    """
    Compute the singular value decomposition of a linear operator using PROPACK

    Parameters
    ----------
    A : array_like, sparse matrix, or LinearOperator
        Operator for which SVD will be computed.  If `A` is a LinearOperator
        object, it must define both ``matvec`` and ``rmatvec`` methods.
    k : int
        Number of singular values/vectors to compute
    which : {"LM", "SM"}
        Which singular triplets to compute:
        - 'LM': compute triplets corresponding to the `k` largest singular
                values
        - 'SM': compute triplets corresponding to the `k` smallest singular
                values
        `which='SM'` requires `irl_mode=True`.  Computes largest singular
        values by default.
    以上是函数的参数说明和文档字符串，定义了函数_svdp的输入参数及其作用。

    """
    pass



class _AProd:
    """
    Wrapper class for linear operator

    The call signature of the __call__ method matches the callback of
    the PROPACK routines.
    """
    def __init__(self, A):
        try:
            self.A = aslinearoperator(A)
        except TypeError:
            self.A = aslinearoperator(np.asarray(A))
    """
    用于包装线性算子的类。

    """
    
    def __call__(self, transa, m, n, x, y, sparm, iparm):
        if transa == 'n':
            y[:] = self.A.matvec(x)
        else:
            y[:] = numpy 的  y  y .
 Modify
    irl_mode : bool, optional
        如果为 `True`，则使用 IRL（implicitly restarted Lanczos）模式计算 SVD。
        默认为 `True`。

    kmax : int, optional
        最大迭代次数或者 Krylov 子空间的最大维度。
        默认为 ``10 * k``。

    compute_u : bool, optional
        如果为 `True`（默认），则计算左奇异向量 `u`。

    compute_v : bool, optional
        如果为 `True`（默认），则计算右奇异向量 `v`。

    tol : float, optional
        计算奇异值时所需的相对精度。
        如果未指定，将根据机器精度设置。

    v0 : array_like, optional
        迭代的起始向量：必须是长度为 ``A.shape[0]`` 的数组。
        如果未指定，PROPACK 将生成一个起始向量。

    full_output : bool, optional
        如果为 `True`，则返回 `sigma_bound`。
        默认为 `False`。

    delta : float, optional
        Lanczos 向量之间要保持的正交性水平。
        默认根据机器精度设置。

    eta : float, optional
        正交性截断值。在重新正交化期间，沿着 Lanczos 向量的分量大于 `eta` 的向量将被清除。
        默认根据机器精度设置。

    anorm : float, optional
        对 ``||A||`` 的估计。默认为 ``0``。

    cgs : bool, optional
        如果为 `True`，则使用经典的 Gram-Schmidt 方法进行重新正交化。
        如果为 `False`（默认），则使用改进的 Gram-Schmidt 方法。

    elr : bool, optional
        如果为 `True`（默认），则在获取奇异向量时执行扩展的局部正交性。

    min_relgap : float, optional
        在 IRL 模式中允许的最小相对间隙。
        默认为 ``0.001``。仅在 ``irl_mode=True`` 时访问。

    shifts : int, optional
        IRL 模式中每次重启时的转移数量。
        默认值根据 ``k <= min(kmax-shifts, m, n)`` 来确定。
        必须 >= 0，但选择 0 可能会导致性能下降。
        仅在 ``irl_mode=True`` 时访问。

    maxiter : int, optional
        IRL 模式中的最大重启次数。
        默认为 ``1000``。
        仅在 ``irl_mode=True`` 时访问。

    random_state : {None, int, `numpy.random.Generator`,
                    `numpy.random.RandomState`}, optional
        用于生成重新采样的伪随机数发生器状态。

        如果 `random_state` 是 ``None``（或 `np.random`），则使用 `numpy.random.RandomState` 单例。
        如果 `random_state` 是一个整数，则使用一个新的 ``RandomState`` 实例，种子为 `random_state`。
        如果 `random_state` 已经是 ``Generator`` 或 ``RandomState`` 实例，则使用该实例。
    """
    u : ndarray
        The `k` largest (``which="LM"``) or smallest (``which="SM"``) left
        singular vectors, ``shape == (A.shape[0], 3)``, returned only if
        ``compute_u=True``.
    sigma : ndarray
        The top `k` singular values, ``shape == (k,)``
    vt : ndarray
        The `k` largest (``which="LM"``) or smallest (``which="SM"``) right
        singular vectors, ``shape == (3, A.shape[1])``, returned only if
        ``compute_v=True``.
    sigma_bound : ndarray
        the error bounds on the singular values sigma, returned only if
        ``full_output=True``.
    """

    # 使用指定的随机状态对象，确保每次运行的结果可复现
    random_state = check_random_state(random_state)

    # 将 which 参数转换为大写
    which = which.upper()
    # 如果 which 不是 'LM' 或 'SM'，则抛出 ValueError 异常
    if which not in {'LM', 'SM'}:
        raise ValueError("`which` must be either 'LM' or 'SM'")
    # 如果不是 IRL 模式并且 which 是 'SM'，则抛出 ValueError 异常
    if not irl_mode and which == 'SM':
        raise ValueError("`which`='SM' requires irl_mode=True")

    # 创建 _AProd 类的实例，用于计算 A 的乘积
    aprod = _AProd(A)
    # 获取 A 的数据类型字符
    typ = aprod.dtype.char

    try:
        # 根据数据类型获取不同算法的实现
        lansvd_irl = _lansvd_irl_dict[typ]
        lansvd = _lansvd_dict[typ]
    except KeyError:
        # 处理不支持的数据类型，默认使用系统原生精度
        if np.iscomplexobj(np.empty(0, dtype=typ)):
            typ = np.dtype(complex).char
        else:
            typ = np.dtype(float).char
        lansvd_irl = _lansvd_irl_dict[typ]
        lansvd = _lansvd_dict[typ]

    # 获取矩阵 A 的行数和列数
    m, n = aprod.shape
    # 检查 k 是否在有效范围内
    if (k < 1) or (k > min(m, n)):
        raise ValueError("k must be positive and not greater than m or n")

    # 如果 kmax 未指定，则设为 10*k
    if kmax is None:
        kmax = 10*k
    # 如果 maxiter 未指定，则设为 1000
    if maxiter is None:
        maxiter = 1000

    # 防止 kmax 设定过大
    kmax = min(m + 1, n + 1, kmax)
    # 如果 kmax 小于 k，则抛出 ValueError 异常
    if kmax < k:
        raise ValueError(
            "kmax must be greater than or equal to k, "
            f"but kmax ({kmax}) < k ({k})")

    # 将 Python 的参数转换为 Fortran 的参数
    jobu = 'y' if compute_u else 'n'
    jobv = 'y' if compute_v else 'n'

    # 准备输出的数组
    u = np.zeros((m, kmax + 1), order='F', dtype=typ)
    v = np.zeros((n, kmax), order='F', dtype=typ)

    # 指定起始向量。如果 v0 是 None，则使用随机向量作为起始向量
    if v0 is None:
        u[:, 0] = random_state.uniform(size=m)
        # 如果是复数类型，生成虚部为随机数的复数随机向量
        if np.iscomplexobj(np.empty(0, dtype=typ)):
            u[:, 0] += 1j * random_state.uniform(size=m)
    else:
        try:
            # 如果 v0 不为 None，则将其作为起始向量
            u[:, 0] = v0
        except ValueError:
            raise ValueError(f"v0 must be of length {m}")

    # 处理拟合选项的参数
    if delta is None:
        delta = np.sqrt(np.finfo(typ).eps)
    if eta is None:
        eta = np.finfo(typ).eps ** 0.75
    if irl_mode:
        # 根据不同的irl_mode设置doption数组，包含四个参数delta, eta, anorm, min_relgap
        doption = np.array((delta, eta, anorm, min_relgap), dtype=typ.lower())

        # 验证或找到默认的shifts值
        if shifts is None:
            shifts = kmax - k
        # 检查k是否超出有效范围
        if k > min(kmax - shifts, m, n):
            raise ValueError('shifts must satisfy '
                             'k <= min(kmax-shifts, m, n)!')
        elif shifts < 0:
            raise ValueError('shifts must be >= 0!')

    else:
        # 根据不同的irl_mode设置doption数组，包含三个参数delta, eta, anorm
        doption = np.array((delta, eta, anorm), dtype=typ.lower())

    # 根据compute_u或compute_v确定blocksize的大小
    # blocksize控制BLAS level 3操作中处理的工作量大小，影响计算速度和内存消耗
    blocksize = 16

    # 确定lwork和liwork的长度
    if compute_u or compute_v:
        # 计算u和v时需要的工作空间长度
        lwork = m + n + 9*kmax + 5*kmax*kmax + 4 + max(
            3*kmax*kmax + 4*kmax + 4,
            blocksize*max(m, n))
        liwork = 8*kmax
    else:
        # 计算sigma和bnd时需要的工作空间长度
        lwork = m + n + 9*kmax + 2*kmax*kmax + 4 + max(m + n, 4*kmax + 4)
        liwork = 2*kmax + 1

    # 创建工作空间数组
    work = np.empty(lwork, dtype=typ.lower())
    iwork = np.empty(liwork, dtype=np.int32)

    # 创建虚拟参数数组，传递给aprod函数，但在当前包装器中未使用
    dparm = np.empty(1, dtype=typ.lower())
    iparm = np.empty(1, dtype=np.int32)

    if typ.isupper():
        # 如果数据类型为大写，则创建zwork数组，根据PROPACK文档和Julia的包装器长度推测
        zwork = np.empty(m + n + 32*m, dtype=typ)
        works = work, zwork, iwork
    else:
        # 如果数据类型为小写，则不包含zwork数组
        works = work, iwork

    if irl_mode:
        # 调用lansvd_irl函数进行奇异值分解计算，返回u, sigma, bnd, v, info
        u, sigma, bnd, v, info = lansvd_irl(_which_converter[which], jobu,
                                            jobv, m, n, shifts, k, maxiter,
                                            aprod, u, v, tol, *works, doption,
                                            ioption, dparm, iparm)
    else:
        # 调用lansvd函数进行奇异值分解计算，返回u, sigma, bnd, v, info
        u, sigma, bnd, v, info = lansvd(jobu, jobv, m, n, k, aprod, u, v, tol,
                                        *works, doption, ioption, dparm, iparm)

    # 根据info的值，抛出相应的异常或继续处理
    if info > 0:
        raise LinAlgError(
            f"An invariant subspace of dimension {info} was found.")
    elif info < 0:
        raise LinAlgError(
            f"k={k} singular triplets did not converge within "
            f"kmax={kmax} iterations")

    # 成功计算出前k个奇异三元组，返回结果
    return u[:, :k], sigma, v[:, :k].conj().T, bnd
```