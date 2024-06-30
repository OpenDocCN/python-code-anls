# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_isolve\minres.py`

```
# 从 numpy 库中导入 inner, zeros, inf, finfo 函数
from numpy import inner, zeros, inf, finfo
# 从 numpy.linalg 中导入 norm 函数
from numpy.linalg import norm
# 从 math 库中导入 sqrt 函数

# 从 .utils 模块中导入 make_system 函数
from .utils import make_system

# 将 minres 函数添加到 __all__ 列表中，表示它是该模块的公开接口
__all__ = ['minres']

# 定义 minres 函数，用于使用最小残差迭代法求解线性方程 Ax = b
def minres(A, b, x0=None, *, rtol=1e-5, shift=0.0, maxiter=None,
           M=None, callback=None, show=False, check=False):
    """
    Use MINimum RESidual iteration to solve Ax=b

    MINRES minimizes norm(Ax - b) for a real symmetric matrix A.  Unlike
    the Conjugate Gradient method, A can be indefinite or singular.

    If shift != 0 then the method solves (A - shift*I)x = b

    Parameters
    ----------
    A : {sparse matrix, ndarray, LinearOperator}
        The real symmetric N-by-N matrix of the linear system
        Alternatively, ``A`` can be a linear operator which can
        produce ``Ax`` using, e.g.,
        ``scipy.sparse.linalg.LinearOperator``.
    b : ndarray
        Right hand side of the linear system. Has shape (N,) or (N,1).

    Returns
    -------
    x : ndarray
        The converged solution.
    info : integer
        Provides convergence information:
            0  : successful exit
            >0 : convergence to tolerance not achieved, number of iterations
            <0 : illegal input or breakdown

    Other Parameters
    ----------------
    x0 : ndarray
        Starting guess for the solution.
    shift : float
        Value to apply to the system ``(A - shift * I)x = b``. Default is 0.
    rtol : float
        Tolerance to achieve. The algorithm terminates when the relative
        residual is below ``rtol``.
    maxiter : integer
        Maximum number of iterations.  Iteration will stop after maxiter
        steps even if the specified tolerance has not been achieved.
    M : {sparse matrix, ndarray, LinearOperator}
        Preconditioner for A.  The preconditioner should approximate the
        inverse of A.  Effective preconditioning dramatically improves the
        rate of convergence, which implies that fewer iterations are needed
        to reach a given error tolerance.
    callback : function
        User-supplied function to call after each iteration.  It is called
        as callback(xk), where xk is the current solution vector.
    show : bool
        If ``True``, print out a summary and metrics related to the solution
        during iterations. Default is ``False``.
    check : bool
        If ``True``, run additional input validation to check that `A` and
        `M` (if specified) are symmetric. Default is ``False``.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csc_matrix
    >>> from scipy.sparse.linalg import minres
    >>> A = csc_matrix([[3, 2, 0], [1, -1, 0], [0, 5, 1]], dtype=float)
    >>> A = A + A.T
    >>> b = np.array([2, 4, -1], dtype=float)
    >>> x, exitCode = minres(A, b)
    >>> print(exitCode)            # 0 indicates successful convergence
    0
    >>> np.allclose(A.dot(x), b)
    True

    References
    ----------

    """

# 以下是函数主体部分，用于实现最小残差迭代法求解线性方程 Ax = b 的过程
    """
    Solution of sparse indefinite systems of linear equations,
        C. C. Paige and M. A. Saunders (1975),
        SIAM J. Numer. Anal. 12(4), pp. 617-629.
        https://web.stanford.edu/group/SOL/software/minres/

    This file is a translation of the following MATLAB implementation:
        https://web.stanford.edu/group/SOL/software/minres/minres-matlab.zip
    """

    # 调用 make_system 函数，生成求解稀疏不定方程组所需的系统参数和预处理器
    A, M, x, b, postprocess = make_system(A, M, x0, b)

    # 获取矩阵向量乘法函数和预处理函数
    matvec = A.matvec
    psolve = M.matvec

    # 定义程序运行起止信息
    first = 'Enter minres.   '
    last = 'Exit  minres.   '

    # 获取矩阵 A 的行数（即方程组的维度）
    n = A.shape[0]

    # 如果最大迭代次数未指定，则设为默认值 5*n
    if maxiter is None:
        maxiter = 5 * n

    # 定义迭代过程中的状态信息列表
    msg = [' beta2 = 0.  If M = I, b and x are eigenvectors    ',   # -1
            ' beta1 = 0.  The exact solution is x0          ',   # 0
            ' A solution to Ax = b was found, given rtol        ',   # 1
            ' A least-squares solution was found, given rtol    ',   # 2
            ' Reasonable accuracy achieved, given eps           ',   # 3
            ' x has converged to an eigenvector                 ',   # 4
            ' acond has exceeded 0.1/eps                        ',   # 5
            ' The iteration limit was reached                   ',   # 6
            ' A  does not define a symmetric matrix             ',   # 7
            ' M  does not define a symmetric matrix             ',   # 8
            ' M  does not define a pos-def preconditioner       ']   # 9

    # 如果 show 标志为真，则打印初始状态信息
    if show:
        print(first + 'Solution of symmetric Ax = b')
        print(first + f'n      =  {n:3g}     shift  =  {shift:23.14e}')
        print(first + f'itnlim =  {maxiter:3g}     rtol   =  {rtol:11.2e}')
        print()

    # 初始化迭代中的各种量
    istop = 0
    itn = 0
    Anorm = 0
    Acond = 0
    rnorm = 0
    ynorm = 0

    # 获取解向量 x 的数据类型
    xtype = x.dtype

    # 获取 x 数据类型的机器精度
    eps = finfo(xtype).eps

    # 设置初始 Lanczos 向量 v1 对应的 y 和 v
    # y  =  beta1 P' v1,  where  P = C**(-1).
    # v 实际上是 P' v1.
    if x0 is None:
        r1 = b.copy()
    else:
        r1 = b - A @ x
    y = psolve(r1)

    # 计算 beta1 = r1 和 y 的内积
    beta1 = inner(r1, y)

    # 如果 beta1 小于 0，则抛出错误，表明预处理器非正定
    if beta1 < 0:
        raise ValueError('indefinite preconditioner')
    elif beta1 == 0:
        # 如果 beta1 等于 0，则返回初始解 x0
        return (postprocess(x), 0)

    # 计算向量 b 的范数
    bnorm = norm(b)
    if bnorm == 0:
        # 如果 b 的范数为 0，则直接返回 b 作为解
        x = b
        return (postprocess(x), 0)

    # 计算 beta1 的平方根
    beta1 = sqrt(beta1)

    # 如果 check 标志为真，则检查矩阵 A 和预处理器 M 是否对称
    if check:
        # 是否过于严格？

        # 检查矩阵 A 是否对称
        w = matvec(y)
        r2 = matvec(w)
        s = inner(w, w)
        t = inner(y, r2)
        z = abs(s - t)
        epsa = (s + eps) * eps**(1.0 / 3.0)
        if z > epsa:
            raise ValueError('non-symmetric matrix')

        # 检查预处理器 M 是否对称
        r2 = psolve(y)
        s = inner(y, y)
        t = inner(r1, r2)
        z = abs(s - t)
        epsa = (s + eps) * eps**(1.0 / 3.0)
        if z > epsa:
            raise ValueError('non-symmetric preconditioner')

    # 初始化其他量
    oldb = 0
    beta = beta1
    dbar = 0
    epsln = 0
    qrnorm = beta1
    phibar = beta1
    rhs1 = beta1
    rhs2 = 0
    tnorm2 = 0
    # 初始化全局变量 gmax 和 gmin
    gmax = 0
    # 使用给定类型的 finfo 函数获取最大值，并赋值给 gmin
    gmin = finfo(xtype).max
    # 初始化 cs 为 -1
    cs = -1
    # 初始化 sn 为 0
    sn = 0
    # 创建一个长度为 n 的零数组，数据类型为 xtype，并赋给 w
    w = zeros(n, dtype=xtype)
    # 创建一个长度为 n 的零数组，数据类型为 xtype，并赋给 w2
    w2 = zeros(n, dtype=xtype)
    # 将 r1 的值赋给 r2
    r2 = r1

    # 如果 show 为 True，则打印以下内容
    if show:
        print()
        print()
        print('   Itn     x(1)     Compatible    LS       norm(A)  cond(A) gbar/|A|')

    # 如果 show 为 True，则打印以下内容，包括 last、istop、itn、Anorm、Acond、rnorm、ynorm、Arnorm 和 msg[istop+1] 的值
    if show:
        print()
        print(last + f' istop   =  {istop:3g}               itn   ={itn:5g}')
        print(last + f' Anorm   =  {Anorm:12.4e}      Acond =  {Acond:12.4e}')
        print(last + f' rnorm   =  {rnorm:12.4e}      ynorm =  {ynorm:12.4e}')
        print(last + f' Arnorm  =  {Arnorm:12.4e}')
        print(last + msg[istop+1])

    # 如果 istop 等于 6，则将 maxiter 赋给 info，否则将 0 赋给 info
    if istop == 6:
        info = maxiter
    else:
        info = 0

    # 返回 postprocess(x) 的结果和 info
    return (postprocess(x),info)
```