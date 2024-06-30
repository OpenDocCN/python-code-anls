# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_isolve\tfqmr.py`

```
# 导入NumPy库，并将其命名为np，用于数值计算和数组操作
import numpy as np
# 从当前包中导入_iterative模块中的_get_atol_rtol函数
from .iterative import _get_atol_rtol
# 从当前包中导入utils模块中的make_system函数
from .utils import make_system

# 将tfqmr函数加入到当前模块的公开接口列表中，使其可以被外部访问
__all__ = ['tfqmr']

# 定义tfqmr函数，使用Transpose-Free Quasi-Minimal Residual迭代方法解Ax=b线性方程
def tfqmr(A, b, x0=None, *, rtol=1e-5, atol=0., maxiter=None, M=None,
          callback=None, show=False):
    """
    Use Transpose-Free Quasi-Minimal Residual iteration to solve ``Ax = b``.

    Parameters
    ----------
    A : {sparse matrix, ndarray, LinearOperator}
        The real or complex N-by-N matrix of the linear system.
        Alternatively, `A` can be a linear operator which can
        produce ``Ax`` using, e.g.,
        `scipy.sparse.linalg.LinearOperator`.
    b : {ndarray}
        Right hand side of the linear system. Has shape (N,) or (N,1).
    x0 : {ndarray}
        Starting guess for the solution.
    rtol, atol : float, optional
        Parameters for the convergence test. For convergence,
        ``norm(b - A @ x) <= max(rtol*norm(b), atol)`` should be satisfied.
        The default is ``rtol=1e-5``, the default for ``atol`` is ``0.0``.
    maxiter : int, optional
        Maximum number of iterations.  Iteration will stop after maxiter
        steps even if the specified tolerance has not been achieved.
        Default is ``min(10000, ndofs * 10)``, where ``ndofs = A.shape[0]``.
    M : {sparse matrix, ndarray, LinearOperator}
        Inverse of the preconditioner of A.  M should approximate the
        inverse of A and be easy to solve for (see Notes).  Effective
        preconditioning dramatically improves the rate of convergence,
        which implies that fewer iterations are needed to reach a given
        error tolerance.  By default, no preconditioner is used.
    callback : function, optional
        User-supplied function to call after each iteration.  It is called
        as ``callback(xk)``, where ``xk`` is the current solution vector.
    show : bool, optional
        Specify ``show = True`` to show the convergence, ``show = False`` is
        to close the output of the convergence.
        Default is `False`.

    Returns
    -------
    x : ndarray
        The converged solution.
    info : int
        Provides convergence information:

            - 0  : successful exit
            - >0 : convergence to tolerance not achieved, number of iterations
            - <0 : illegal input or breakdown

    Notes
    -----
    The Transpose-Free QMR algorithm is derived from the CGS algorithm.
    However, unlike CGS, the convergence curves for the TFQMR method is
    smoothed by computing a quasi minimization of the residual norm. The
    implementation supports left preconditioner, and the "residual norm"
    to compute in convergence criterion is actually an upper bound on the
    actual residual norm ``||b - Axk||``.

    References
    ----------
    .. [1] R. W. Freund, A Transpose-Free Quasi-Minimal Residual Algorithm for
           Non-Hermitian Linear Systems, SIAM J. Sci. Comput., 14(2), 470-482,
           1993.
    """
    # 函数体实现省略，用于解释如何使用TFQMR算法解线性方程Ax=b，详细参数和返回信息见上述文档字符串
    # 检查矩阵 A 的数据类型
    dtype = A.dtype
    # 如果 A 的数据类型是 np.int64，则转换为 float 类型
    if np.issubdtype(dtype, np.int64):
        dtype = float
        A = A.astype(dtype)
    # 如果向量 b 的数据类型是 np.int64，则也转换为与 A 相同的数据类型
    if np.issubdtype(b.dtype, np.int64):
        b = b.astype(dtype)

    # 调用 make_system 函数，处理输入的 A, M, x0, b，返回处理后的变量
    A, M, x, b, postprocess = make_system(A, M, x0, b)

    # 检查右侧向量 b 是否为零向量
    if np.linalg.norm(b) == 0.:
        # 如果是，则直接返回后处理过的零向量和成功收敛的标志 0
        x = b.copy()
        return (postprocess(x), 0)

    # 计算自由度数目
    ndofs = A.shape[0]
    # 如果未指定最大迭代次数 maxiter，则设定为默认值，不超过 10000 或者自由度数目的 10 倍
    if maxiter is None:
        maxiter = min(10000, ndofs * 10)

    # 初始化 r 向量
    if x0 is None:
        r = b.copy()
    else:
        r = b - A.matvec(x)
    u = r
    w = r.copy()

    # 将 rstar 视为数学上的 b - Ax0，即 rstar := r = b - Ax0
    rstar = r
    v = M.matvec(A.matvec(r))
    uhat = v
    d = theta = eta = 0.

    # 在此时我们知道 rstar == r，因此 rho 始终是实数
    rho = np.inner(rstar.conjugate(), r).real
    rhoLast = rho
    r0norm = np.sqrt(rho)
    tau = r0norm

    # 如果 r0norm 为零，则直接返回后处理过的解 x 和成功收敛的标志 0
    if r0norm == 0:
        return (postprocess(x), 0)

    # 调用 _get_atol_rtol 函数以获取正确的公差 atol，并在必要时引发错误
    atol, _ = _get_atol_rtol('tfqmr', r0norm, atol, rtol)
    for iter in range(maxiter):
        # 检查当前迭代次数是奇数还是偶数
        even = iter % 2 == 0
        if (even):
            vtrstar = np.inner(rstar.conjugate(), v)
            # 检查是否出现计算中断（分母为零）
            if vtrstar == 0.:
                return (postprocess(x), -1)
            # 计算 alpha，用于更新 uNext
            alpha = rho / vtrstar
            uNext = u - alpha * v  # [1]-(5.6)
        # 更新 w 的值，用于下一步迭代
        w -= alpha * uhat  # [1]-(5.8)
        # 计算 d，用于下一步迭代
        d = u + (theta**2 / alpha) * eta * d  # [1]-(5.5)
        # 更新 theta 和 c 的值，用于下一步迭代
        theta = np.linalg.norm(w) / tau
        c = np.sqrt(1. / (1 + theta**2))
        tau *= theta * c
        # 计算 eta 的值，用于下一步迭代
        eta = (c**2) * alpha
        # 计算 z，用于更新 x 的值
        z = M.matvec(d)
        x += eta * z

        if callback is not None:
            callback(x)

        # 判断是否满足收敛条件
        if tau * np.sqrt(iter+1) < atol:
            if (show):
                print("TFQMR: Linear solve converged due to reach TOL "
                      f"iterations {iter+1}")
            return (postprocess(x), 0)

        if (not even):
            # 更新 rho，用于下一步迭代
            rho = np.inner(rstar.conjugate(), w)
            beta = rho / rhoLast
            # 更新 u 和 v，用于下一步迭代
            u = w + beta * u
            v = beta * uhat + (beta**2) * v
            uhat = M.matvec(A.matvec(u))
            v += uhat
        else:
            # 更新 uhat 和 u，用于下一步迭代
            uhat = M.matvec(A.matvec(uNext))
            u = uNext
            rhoLast = rho

    if (show):
        print("TFQMR: Linear solve not converged due to reach MAXIT "
              f"iterations {iter+1}")
    return (postprocess(x), maxiter)


这段代码实现了一个 TFQMR（Transpose-Free Quasi-Minimal Residual）算法的迭代过程，用于解线性方程组。
```