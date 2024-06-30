# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_isolve\_gcrotmk.py`

```
# 导入必要的库和模块
import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import (get_blas_funcs, qr, solve, svd, qr_insert, lstsq)
from .iterative import _get_atol_rtol
from scipy.sparse.linalg._isolve.utils import make_system

# 定义公开的函数和类
__all__ = ['gcrotmk']

# 定义私有函数，执行FGMRES Arnoldi过程，可选投影或增广
def _fgmres(matvec, v0, m, atol, lpsolve=None, rpsolve=None, cs=(), outer_v=(),
            prepend_outer_v=False):
    """
    FGMRES Arnoldi process, with optional projection or augmentation

    Parameters
    ----------
    matvec : callable
        Operation A*x
    v0 : ndarray
        Initial vector, normalized to nrm2(v0) == 1
    m : int
        Number of GMRES rounds
    atol : float
        Absolute tolerance for early exit
    lpsolve : callable, optional
        Left preconditioner L
    rpsolve : callable, optional
        Right preconditioner R
    cs : list of (ndarray, ndarray), optional
        Columns of matrices C and U in GCROT
    outer_v : list of ndarrays, optional
        Augmentation vectors in LGMRES
    prepend_outer_v : bool, optional
        Whether augmentation vectors come before or after
        Krylov iterates

    Raises
    ------
    LinAlgError
        If nans encountered

    Returns
    -------
    Q, R : ndarray
        QR decomposition of the upper Hessenberg H=QR
    B : ndarray
        Projections corresponding to matrix C
    vs : list of ndarray
        Columns of matrix V
    zs : list of ndarray
        Columns of matrix Z
    y : ndarray
        Solution to ||H y - e_1||_2 = min!
    res : float
        The final (preconditioned) residual norm

    """

    # 若未提供左预条件解，则默认为恒等函数
    if lpsolve is None:
        def lpsolve(x):
            return x
    
    # 若未提供右预条件解，则默认为恒等函数
    if rpsolve is None:
        def rpsolve(x):
            return x

    # 获取BLAS函数：向量操作和范数计算
    axpy, dot, scal, nrm2 = get_blas_funcs(['axpy', 'dot', 'scal', 'nrm2'], (v0,))

    # 初始化向量列表：V和Z
    vs = [v0]
    zs = []
    
    # 初始化变量y和残差res
    y = None
    res = np.nan

    # 计算总迭代次数m，包括外部增广向量数量
    m = m + len(outer_v)

    # 初始化正交投影系数矩阵B
    B = np.zeros((len(cs), m), dtype=v0.dtype)

    # 初始化H的QR分解形式：Q为单位矩阵，R为空矩阵
    Q = np.ones((1, 1), dtype=v0.dtype)
    R = np.zeros((1, 0), dtype=v0.dtype)

    # 机器精度
    eps = np.finfo(v0.dtype).eps

    # 是否发生方法失效
    breakdown = False

    # FGMRES Arnoldi过程开始
    for j in range(m):
        # 迭代变量 j 控制循环次数，从 0 到 m-1
        
        # 根据条件选择外部向量 outer_v 或者解析函数 rpsolve 进行计算
        if prepend_outer_v and j < len(outer_v):
            z, w = outer_v[j]
        elif prepend_outer_v and j == len(outer_v):
            z = rpsolve(v0)
            w = None
        elif not prepend_outer_v and j >= m - len(outer_v):
            z, w = outer_v[j - (m - len(outer_v))]
        else:
            z = rpsolve(vs[-1])
            w = None

        if w is None:
            # 如果 w 为空，则使用 matvec(z) 计算 w
            w = lpsolve(matvec(z))
        else:
            # 复制 w，避免原始数据的改变
            w = w.copy()

        # 计算向量 w 的二范数
        w_norm = nrm2(w)

        # GCROT 投影操作：L A -> (1 - C C^H) L A
        # 即针对 C 正交化操作
        for i, c in enumerate(cs):
            alpha = dot(c, w)
            B[i,j] = alpha
            w = axpy(c, w, c.shape[0], -alpha)  # w -= alpha*c

        # 针对 V 进行正交化操作
        hcur = np.zeros(j+2, dtype=Q.dtype)
        for i, v in enumerate(vs):
            alpha = dot(v, w)
            hcur[i] = alpha
            w = axpy(v, w, v.shape[0], -alpha)  # w -= alpha*v
        hcur[i+1] = nrm2(w)

        with np.errstate(over='ignore', divide='ignore'):
            # 处理 denormals 的情况
            alpha = 1/hcur[-1]

        if np.isfinite(alpha):
            # 如果 alpha 是有限的，则对 w 进行缩放
            w = scal(alpha, w)

        if not (hcur[-1] > eps * w_norm):
            # 如果 hcur[-1] 接近于 0，说明 w 几乎在之前向量的线性空间内，或者存在 NaN 值，结束 QR 分解
            breakdown = True

        # 将计算得到的向量 w 和 z 添加到 vs 和 zs 列表中
        vs.append(w)
        zs.append(z)

        # Arnoldi LSQ 问题

        # 将新列添加到 H=Q@R，其余列用零填充
        Q2 = np.zeros((j+2, j+2), dtype=Q.dtype, order='F')
        Q2[:j+1,:j+1] = Q
        Q2[j+1,j+1] = 1

        R2 = np.zeros((j+2, j), dtype=R.dtype, order='F')
        R2[:j+1,:] = R

        # 插入新列到 QR 分解中
        Q, R = qr_insert(Q2, R2, hcur, j, which='col',
                         overwrite_qru=True, check_finite=False)

        # 变换的最小二乘问题
        # || Q R y - inner_res_0 * e_1 ||_2 = min!
        # 由于 R = [R'; 0]，解是 y = inner_res_0 (R')^{-1} (Q^H)[:j,0]

        # 计算残差
        res = abs(Q[0,-1])

        # 检查是否满足终止条件
        if res < atol or breakdown:
            break

    if not np.isfinite(R[j,j]):
        # 遇到 NaN 值，结束运算
        raise LinAlgError()

    # 获取最小二乘问题的解
    # 问题是上三角形的，但条件数可能很差（或者在 breakdown 情况下，最后对角线条目可能为零），因此使用 lstsq 而不是 trtrs
    y, _, _, _, = lstsq(R[:j+1,:j+1], Q[0,:j+1].conj())

    # 截取 B 的列数
    B = B[:,:j+1]

    return Q, R, B, vs, zs, y, res
# 使用 GCROT(m,k) 算法解决矩阵方程的函数定义
def gcrotmk(A, b, x0=None, *, rtol=1e-5, atol=0., maxiter=1000, M=None, callback=None,
            m=20, k=None, CU=None, discard_C=False, truncate='oldest'):
    """
    Solve a matrix equation using flexible GCROT(m,k) algorithm.

    Parameters
    ----------
    A : {sparse matrix, ndarray, LinearOperator}
        The real or complex N-by-N matrix of the linear system.
        Alternatively, ``A`` can be a linear operator which can
        produce ``Ax`` using, e.g.,
        ``scipy.sparse.linalg.LinearOperator``.
    b : ndarray
        Right hand side of the linear system. Has shape (N,) or (N,1).
    x0 : ndarray
        Starting guess for the solution.
    rtol, atol : float, optional
        Parameters for the convergence test. For convergence,
        ``norm(b - A @ x) <= max(rtol*norm(b), atol)`` should be satisfied.
        The default is ``rtol=1e-5``, the default for ``atol`` is ``0.0``.
    maxiter : int, optional
        Maximum number of iterations.  Iteration will stop after maxiter
        steps even if the specified tolerance has not been achieved.
    M : {sparse matrix, ndarray, LinearOperator}, optional
        Preconditioner for A.  The preconditioner should approximate the
        inverse of A. gcrotmk is a 'flexible' algorithm and the preconditioner
        can vary from iteration to iteration. Effective preconditioning
        dramatically improves the rate of convergence, which implies that
        fewer iterations are needed to reach a given error tolerance.
    callback : function, optional
        User-supplied function to call after each iteration.  It is called
        as callback(xk), where xk is the current solution vector.
    m : int, optional
        Number of inner FGMRES iterations per each outer iteration.
        Default: 20
    k : int, optional
        Number of vectors to carry between inner FGMRES iterations.
        According to [2]_, good values are around m.
        Default: m
    CU : list of tuples, optional
        List of tuples ``(c, u)`` which contain the columns of the matrices
        C and U in the GCROT(m,k) algorithm. For details, see [2]_.
        The list given and vectors contained in it are modified in-place.
        If not given, start from empty matrices. The ``c`` elements in the
        tuples can be ``None``, in which case the vectors are recomputed
        via ``c = A u`` on start and orthogonalized as described in [3]_.
    discard_C : bool, optional
        Discard the C-vectors at the end. Useful if recycling Krylov subspaces
        for different linear systems.
    truncate : {'oldest', 'smallest'}, optional
        Truncation scheme to use. Drop: oldest vectors, or vectors with
        smallest singular values using the scheme discussed in [1,2].
        See [2]_ for detailed comparison.
        Default: 'oldest'

    Returns
    -------
    x : ndarray
        The solution found.
    """
    info : int
        提供收敛信息:

        * 0  : 成功退出
        * >0 : 未达到收敛容差，迭代次数

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csc_matrix
    >>> from scipy.sparse.linalg import gcrotmk
    >>> R = np.random.randn(5, 5)
    >>> A = csc_matrix(R)
    >>> b = np.random.randn(5)
    >>> x, exit_code = gcrotmk(A, b, atol=1e-5)
    >>> print(exit_code)
    0
    >>> np.allclose(A.dot(x), b)
    True

    References
    ----------
    .. [1] E. de Sturler, ''Truncation strategies for optimal Krylov subspace
           methods'', SIAM J. Numer. Anal. 36, 864 (1999).
    .. [2] J.E. Hicken and D.W. Zingg, ''A simplified and flexible variant
           of GCROT for solving nonsymmetric linear systems'',
           SIAM J. Sci. Comput. 32, 172 (2010).
    .. [3] M.L. Parks, E. de Sturler, G. Mackey, D.D. Johnson, S. Maiti,
           ''Recycling Krylov subspaces for sequences of linear systems'',
           SIAM J. Sci. Comput. 28, 1651 (2006).

    """
    A,M,x,b,postprocess = make_system(A,M,x0,b)  # 创建线性系统和后处理步骤

    if not np.isfinite(b).all():  # 检查右端向量是否全部为有限数值
        raise ValueError("RHS must contain only finite numbers")  # 抛出异常：右端向量必须仅包含有限数值

    if truncate not in ('oldest', 'smallest'):  # 检查截断选项是否合法
        raise ValueError(f"Invalid value for 'truncate': {truncate!r}")  # 抛出异常：截断选项无效

    matvec = A.matvec  # 获取矩阵向量乘法函数
    psolve = M.matvec  # 获取预处理矩阵向量乘法函数

    if CU is None:
        CU = []  # 如果CU为空，则初始化为空列表

    if k is None:
        k = m  # 如果k为空，则设置为m

    axpy, dot, scal = None, None, None  # 初始化axpy, dot, scal为None

    if x0 is None:
        r = b.copy()  # 如果初始解x0为空，则r为b的副本
    else:
        r = b - matvec(x)  # 否则，计算初始残差向量r = b - A*x

    axpy, dot, scal, nrm2 = get_blas_funcs(['axpy', 'dot', 'scal', 'nrm2'], (x, r))  # 获取BLAS函数

    b_norm = nrm2(b)  # 计算向量b的2范数

    # 调用此函数以获取正确的atol/rtol，并根据需要引发错误
    atol, rtol = _get_atol_rtol('gcrotmk', b_norm, atol, rtol)

    if b_norm == 0:
        x = b
        return (postprocess(x), 0)  # 如果b的范数为0，则直接返回结果和退出码0

    if discard_C:
        CU[:] = [(None, u) for c, u in CU]  # 如果discard_C为True，则清空CU列表中的内容

    # Reorthogonalize old vectors
    if CU:
        # 如果CU不为空，则执行以下操作

        # 将已经存在的向量移到前面进行排序
        CU.sort(key=lambda cu: cu[0] is not None)

        # 创建一个空的数组C，用来存储数据，以及一个空列表us
        C = np.empty((A.shape[0], len(CU)), dtype=r.dtype, order='F')
        us = []
        j = 0
        while CU:
            # 更加节省内存：随着处理过程中丢弃旧的向量
            c, u = CU.pop(0)
            if c is None:
                c = matvec(u)  # 如果c为空，则通过u计算c
            C[:,j] = c  # 将c存储在C的第j列
            j += 1
            us.append(u)  # 将u添加到us列表中

        # 对C进行正交化处理
        Q, R, P = qr(C, overwrite_a=True, mode='economic', pivoting=True)
        del C  # 删除C，释放内存

        # 将Q赋值给cs
        cs = list(Q.T)

        # U := U P R^-1，进行回代
        new_us = []
        for j in range(len(cs)):
            u = us[P[j]]
            for i in range(j):
                u = axpy(us[P[i]], u, u.shape[0], -R[i,j])
            if abs(R[j,j]) < 1e-12 * abs(R[0,0]):
                # 如果R的对角元素小于阈值，丢弃剩余的向量
                break
            u = scal(1.0/R[j,j], u)
            new_us.append(u)

        # 形成新的CU列表
        CU[:] = list(zip(cs, new_us))[::-1]

    if CU:
        axpy, dot = get_blas_funcs(['axpy', 'dot'], (r,))

        # 针对CU中的向量进行投影操作的解算
        # 对初始猜测进行修改，使得
        #
        #     x' = x + U y
        #     y = argmin_y || b - A (x + U y) ||^2
        #
        # 解为 y = C^H (b - A x)
        for c, u in CU:
            yc = dot(c, r)
            x = axpy(u, x, x.shape[0], yc)
            r = axpy(c, r, r.shape[0], -yc)

    # GCROT主迭代
    # 将解向量包含到span中
    CU.append((None, x.copy()))
    if discard_C:
        # 如果需要丢弃C，则将CU中的向量cz替换成uz
        CU[:] = [(None, uz) for cz, uz in CU]

    # 返回后处理的结果x和迭代次数j_outer+1
    return postprocess(x), j_outer + 1
```