# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_isolve\iterative.py`

```
import warnings  # 导入警告模块，用于处理警告信息
import numpy as np  # 导入 NumPy 库，用于数值计算
from scipy.sparse.linalg._interface import LinearOperator  # 导入线性算子接口
from .utils import make_system  # 从当前目录下的 utils 模块导入 make_system 函数
from scipy.linalg import get_lapack_funcs  # 从 SciPy 的 linalg 子模块导入 get_lapack_funcs 函数

__all__ = ['bicg', 'bicgstab', 'cg', 'cgs', 'gmres', 'qmr']  # 定义导出的模块列表

def _get_atol_rtol(name, b_norm, atol=0., rtol=1e-5):
    """
    A helper function to handle tolerance normalization
    """
    if atol == 'legacy' or atol is None or atol < 0:  # 如果 atol 参数不合法，则抛出异常
        msg = (f"'scipy.sparse.linalg.{name}' called with invalid `atol`={atol}; "
               "if set, `atol` must be a real, non-negative number.")
        raise ValueError(msg)

    atol = max(float(atol), float(rtol) * float(b_norm))  # 根据 rtol 和 b_norm 计算有效的 atol

    return atol, rtol  # 返回处理后的 atol 和 rtol

def bicg(A, b, x0=None, *, rtol=1e-5, atol=0., maxiter=None, M=None, callback=None):
    """
    Use BIConjugate Gradient iteration to solve ``Ax = b``.

    Parameters
    ----------
    A : {sparse matrix, ndarray, LinearOperator}
        The real or complex N-by-N matrix of the linear system.
        Alternatively, ``A`` can be a linear operator which can
        produce ``Ax`` and ``A^T x`` using, e.g.,
        ``scipy.sparse.linalg.LinearOperator``.
    b : ndarray
        Right hand side of the linear system. Has shape (N,) or (N,1).
    x0 : ndarray
        Starting guess for the solution.
    rtol, atol : float, optional
        Parameters for the convergence test. For convergence,
        ``norm(b - A @ x) <= max(rtol*norm(b), atol)`` should be satisfied.
        The default is ``atol=0.`` and ``rtol=1e-5``.
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

    Returns
    -------
    x : ndarray
        The converged solution.
    info : integer
        Provides convergence information:
            0  : successful exit
            >0 : convergence to tolerance not achieved, number of iterations
            <0 : parameter breakdown

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csc_matrix
    >>> from scipy.sparse.linalg import bicg
    >>> A = csc_matrix([[3, 2, 0], [1, -1, 0], [0, 5, 1.]])
    >>> b = np.array([2., 4., -1.])
    >>> x, exitCode = bicg(A, b, atol=1e-5)
    >>> print(exitCode)  # 0 indicates successful convergence
    0
    >>> np.allclose(A.dot(x), b)
    True

    """
    A, M, x, b, postprocess = make_system(A, M, x0, b)  # 调用 make_system 函数，准备解线性系统所需的参数
    bnrm2 = np.linalg.norm(b)  # 计算右侧向量 b 的二范数

    atol, _ = _get_atol_rtol('bicg', bnrm2, atol, rtol)  # 调用 _get_atol_rtol 函数，获取有效的 atol
    # 如果向量 b 的二范数为零，直接返回处理后的结果和迭代次数 0
    if bnrm2 == 0:
        return postprocess(b), 0

    # 获取向量 b 的长度
    n = len(b)
    # 根据向量 x 是否为复数类型选择不同的向量点积函数
    dotprod = np.vdot if np.iscomplexobj(x) else np.dot

    # 如果未指定最大迭代次数，设定默认值为 n*10
    if maxiter is None:
        maxiter = n * 10

    # 从 A 中获取 matvec 和 rmatvec 函数
    matvec, rmatvec = A.matvec, A.rmatvec
    # 从 M 中获取 psolve 和 rpsolve 函数
    psolve, rpsolve = M.matvec, M.rmatvec

    # 计算精度阈值 rhotol，使用 x 的数据类型精度
    rhotol = np.finfo(x.dtype.char).eps ** 2

    # 初始化变量，避免 linter 警告
    rho_prev, p, ptilde = None, None, None

    # 计算初始残差 r 和 rtilde
    r = b - matvec(x) if x.any() else b.copy()
    rtilde = r.copy()

    # 开始迭代求解
    for iteration in range(maxiter):
        # 如果残差 r 的二范数小于设定的容差 atol，认为收敛，返回处理后的结果和迭代次数 0
        if np.linalg.norm(r) < atol:  # Are we done?
            return postprocess(x), 0

        # 解方程 Psolve(z) = r 和 rpsolve(ztilde) = rtilde，得到 z 和 ztilde
        z = psolve(r)
        ztilde = rpsolve(rtilde)

        # 计算当前 rho_cur = rtilde · z
        rho_cur = dotprod(rtilde, z)

        # 如果 rho_cur 的绝对值小于 rhotol，表示算法发生异常情况，返回处理函数和错误码 -10
        if np.abs(rho_cur) < rhotol:  # Breakdown case
            return postprocess, -10

        # 根据迭代次数进行不同处理
        if iteration > 0:
            # 计算 beta 和更新 p, ptilde
            beta = rho_cur / rho_prev
            p *= beta
            p += z
            ptilde *= beta.conj()
            ptilde += ztilde
        else:  # First spin
            # 第一次迭代时直接复制 z 和 ztilde 到 p 和 ptilde
            p = z.copy()
            ptilde = ztilde.copy()

        # 计算 q 和 qtilde
        q = matvec(p)
        qtilde = rmatvec(ptilde)

        # 计算 rv = ptilde · q
        rv = dotprod(ptilde, q)

        # 如果 rv 等于零，表示算法异常，返回处理函数和错误码 -11
        if rv == 0:
            return postprocess(x), -11

        # 计算 alpha 和更新 x, r, rtilde
        alpha = rho_cur / rv
        x += alpha * p
        r -= alpha * q
        rtilde -= alpha.conj() * qtilde
        rho_prev = rho_cur

        # 如果存在回调函数，调用回调函数传入当前解 x
        if callback:
            callback(x)

    else:  # for loop exhausted
        # 如果迭代次数达到最大值仍未收敛，返回当前解和最大迭代次数
        return postprocess(x), maxiter
    # 使用 BIConjugate Gradient STABilized 方法解决线性方程组 Ax = b
    """Use BIConjugate Gradient STABilized iteration to solve ``Ax = b``.
    
    Parameters
    ----------
    A : {sparse matrix, ndarray, LinearOperator}
        线性系统的实数或复数 N×N 矩阵。
        或者可以是一个线性操作符，可以使用例如
        ``scipy.sparse.linalg.LinearOperator`` 产生 ``Ax`` 和 ``A^T x``。
    b : ndarray
        线性系统的右侧向量。具有形状 (N,) 或 (N,1)。
    x0 : ndarray
        解的初始猜测。
    rtol, atol : float, optional
        收敛测试的参数。为了收敛，
        应满足 ``norm(b - A @ x) <= max(rtol*norm(b), atol)``。
        默认值为 ``atol=0.`` 和 ``rtol=1e-5``。
    maxiter : integer
        最大迭代次数。即使未达到指定的容差，迭代也将在 maxiter 步之后停止。
    M : {sparse matrix, ndarray, LinearOperator}
        A 的预处理器。预处理器应该近似于 A 的逆。
        有效的预处理显著提高了收敛速度，这意味着需要更少的迭代来达到给定的误差容限。
    callback : function
        每次迭代后调用的用户提供的函数。调用形式为 callback(xk)，其中 xk 是当前解向量。

    Returns
    -------
    x : ndarray
        收敛的解。
    info : integer
        提供收敛信息：
            0  : 成功退出
            >0 : 未达到容差的收敛，迭代次数
            <0 : 参数中断

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csc_matrix
    >>> from scipy.sparse.linalg import bicgstab
    >>> R = np.array([[4, 2, 0, 1],
    ...               [3, 0, 0, 2],
    ...               [0, 1, 1, 1],
    ...               [0, 2, 1, 0]])
    >>> A = csc_matrix(R)
    >>> b = np.array([-1, -0.5, -1, 2])
    >>> x, exit_code = bicgstab(A, b, atol=1e-5)
    >>> print(exit_code)  # 0 表示成功收敛
    0
    >>> np.allclose(A.dot(x), b)
    True

    """
    # 使用 make_system 函数准备线性系统的输入参数和后处理器
    A, M, x, b, postprocess = make_system(A, M, x0, b)
    # 计算右侧向量 b 的二范数
    bnrm2 = np.linalg.norm(b)

    # 获取精度控制参数 atol 和 rtol
    atol, _ = _get_atol_rtol('bicgstab', bnrm2, atol, rtol)

    # 如果 bnrm2 为零，返回后处理过的 b 和 0
    if bnrm2 == 0:
        return postprocess(b), 0

    # 确定问题的大小 N
    n = len(b)

    # 根据 x 的类型选择点积函数
    dotprod = np.vdot if np.iscomplexobj(x) else np.dot

    # 如果未指定 maxiter，则设置为 n*10
    if maxiter is None:
        maxiter = n*10

    # 定义矩阵向量乘法函数和预处理函数
    matvec = A.matvec
    psolve = M.matvec

    # 下面的值没有意义，但来自于原始的 Fortran 代码
    # 可能原意是 sqrt，但这里使用了平方，这是历史原因
    rhotol = np.finfo(x.dtype.char).eps**2
    omegatol = rhotol

    # 初始化一些变量以避免静态分析器警告，这些变量在后续代码中会被重新赋值和使用
    rho_prev, omega, alpha, p, v = None, None, None, None, None
    # 计算残差 r，如果 x 的任何元素为真则使用 matvec(x)，否则复制 b
    r = b - matvec(x) if x.any() else b.copy()
    # 复制 r 到 rtilde
    rtilde = r.copy()

    # 迭代最大次数 maxiter 次
    for iteration in range(maxiter):
        # 如果残差 r 的二范数小于 atol，则任务完成
        if np.linalg.norm(r) < atol:  # Are we done?
            return postprocess(x), 0

        # 计算 rho = rtilde 和 r 的点积
        rho = dotprod(rtilde, r)
        # 如果 rho 的绝对值小于 rhotol，则认为 rho 发生了异常
        if np.abs(rho) < rhotol:  # rho breakdown
            return postprocess(x), -10

        # 在迭代大于 0 时执行以下操作
        if iteration > 0:
            # 如果 omega 的绝对值小于 omegatol，则认为 omega 发生了异常
            if np.abs(omega) < omegatol:  # omega breakdown
                return postprocess(x), -11

            # 计算 beta 值用于更新搜索方向 p
            beta = (rho / rho_prev) * (alpha / omega)
            # 更新 p 向量
            p -= omega*v
            p *= beta
            p += r
        else:  # 第一次迭代
            # 创建与 r 相同类型的空数组 s，并将 p 初始化为 r 的副本
            s = np.empty_like(r)
            p = r.copy()

        # 解 p 的线性系统得到 phat
        phat = psolve(p)
        # 计算 matvec(phat)，得到 v
        v = matvec(phat)
        # 计算 rtilde 和 v 的点积得到 rv
        rv = dotprod(rtilde, v)
        # 如果 rv 为零，则认为发生了异常
        if rv == 0:
            return postprocess(x), -11
        # 计算 alpha 作为 rho 与 rv 的比值
        alpha = rho / rv
        # 更新 r 向量
        r -= alpha*v
        # 将 r 复制给 s
        s[:] = r[:]

        # 如果 s 的二范数小于 atol，则任务完成
        if np.linalg.norm(s) < atol:
            x += alpha*phat
            return postprocess(x), 0

        # 解 s 的线性系统得到 shat
        shat = psolve(s)
        # 计算 matvec(shat)，得到 t
        t = matvec(shat)
        # 计算 omega 作为 t 和 s 的点积比 t 和 t 的点积
        omega = dotprod(t, s) / dotprod(t, t)
        # 更新 x 向量
        x += alpha*phat
        x += omega*shat
        # 更新 r 向量
        r -= omega*t
        # 更新 rho_prev
        rho_prev = rho

        # 如果有回调函数，则调用回调函数并传递当前的解向量 x
        if callback:
            callback(x)

    else:  # for 循环耗尽
        # 返回不完整的进度信息
        return postprocess(x), maxiter
    # 使用共轭梯度法解决线性方程组 Ax = b。

    A, M, x, b, postprocess = make_system(A, M, x0, b)
    # 调用 make_system 函数，对输入进行处理并返回必要的变量

    bnrm2 = np.linalg.norm(b)
    # 计算向量 b 的二范数，用于后续的收敛性测试

    atol, _ = _get_atol_rtol('cg', bnrm2, atol, rtol)
    # 获取实际使用的容差 atol，基于用户提供的 rtol 和 bnrm2

    if bnrm2 == 0:
        return postprocess(b), 0
    # 如果 b 的范数为零，直接返回后处理过的 b 和收敛信息 0，表示成功

    n = len(b)
    # 获取向量 b 的长度

    if maxiter is None:
        maxiter = n*10
    # 如果未指定最大迭代次数，设定默认值为向量长度的十倍

    dotprod = np.vdot if np.iscomplexobj(x) else np.dot
    # 根据解向量 x 是否复数确定使用 np.vdot 还是 np.dot 进行向量内积运算

    matvec = A.matvec
    # 获取 A 的 matvec 方法，用于执行矩阵向量乘法

    psolve = M.matvec if M is not None else None
    # 如果存在预处理器 M，则获取其 matvec 方法；否则设为 None

    r = b - matvec(x) if x.any() else b.copy()
    # 计算初始残差向量 r，如果初始解 x 不为空则计算 b - A*x，否则直接复制 b

    # Dummy value to initialize var, silences warnings
    rho_prev, p = None, None
    # 初始化 rho_prev 和 p，用于共轭梯度法中的迭代更新
    for iteration in range(maxiter):
        # 检查残差向量的二范数是否小于指定的绝对容差 atol，判断是否完成迭代
        if np.linalg.norm(r) < atol:  # Are we done?
            # 如果满足条件，则返回处理后的解 x 和迭代次数 0（表示成功收敛）
            return postprocess(x), 0

        # 使用预处理函数 psolve 处理残差向量 r，得到 z
        z = psolve(r)
        # 计算当前残差与 z 的内积，作为当前迭代的 rho_cur
        rho_cur = dotprod(r, z)

        if iteration > 0:
            # 如果不是第一次迭代，则计算 beta 系数
            beta = rho_cur / rho_prev
            # 更新搜索方向 p
            p *= beta
            p += z
        else:  # First spin
            # 如果是第一次迭代，则初始化搜索方向 p 为 z 的副本
            p = np.empty_like(r)
            p[:] = z[:]

        # 计算搜索方向 p 对应的矩阵向量乘积 q
        q = matvec(p)
        # 计算步长 alpha
        alpha = rho_cur / dotprod(p, q)
        # 更新解向量 x
        x += alpha * p
        # 更新残差向量 r
        r -= alpha * q
        # 更新前一次迭代的 rho_prev
        rho_prev = rho_cur

        # 如果提供了回调函数 callback，则在每次迭代后调用回调函数
        if callback:
            callback(x)

    else:  # for loop exhausted
        # 如果迭代次数达到最大值但仍未满足收敛条件，则返回当前解 x 和迭代次数 maxiter
        return postprocess(x), maxiter
# 定义 CGS 方法，用于解决线性方程组 Ax = b
def cgs(A, b, x0=None, *, rtol=1e-5, atol=0., maxiter=None, M=None, callback=None):
    """Use Conjugate Gradient Squared iteration to solve ``Ax = b``.

    Parameters
    ----------
    A : {sparse matrix, ndarray, LinearOperator}
        The real-valued N-by-N matrix of the linear system.
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
        The default is ``atol=0.`` and ``rtol=1e-5``.
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

    Returns
    -------
    x : ndarray
        The converged solution.
    info : integer
        Provides convergence information:
            0  : successful exit
            >0 : convergence to tolerance not achieved, number of iterations
            <0 : parameter breakdown

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csc_matrix
    >>> from scipy.sparse.linalg import cgs
    >>> R = np.array([[4, 2, 0, 1],
    ...               [3, 0, 0, 2],
    ...               [0, 1, 1, 1],
    ...               [0, 2, 1, 0]])
    >>> A = csc_matrix(R)
    >>> b = np.array([-1, -0.5, -1, 2])
    >>> x, exit_code = cgs(A, b)
    >>> print(exit_code)  # 0 indicates successful convergence
    0
    >>> np.allclose(A.dot(x), b)
    True

    """
    # 调用 make_system 函数进行初始化和预处理
    A, M, x, b, postprocess = make_system(A, M, x0, b)
    # 计算右侧向量 b 的二范数
    bnrm2 = np.linalg.norm(b)

    # 获取实际的 atol 和 rtol 值，用于收敛测试
    atol, _ = _get_atol_rtol('cgs', bnrm2, atol, rtol)

    # 如果 b 的二范数为 0，则直接返回处理后的 b 和成功标志 0
    if bnrm2 == 0:
        return postprocess(b), 0

    # 确定线性方程组的维度
    n = len(b)

    # 根据 x 的类型选择 dot 乘法运算函数
    dotprod = np.vdot if np.iscomplexobj(x) else np.dot

    # 如果未指定最大迭代次数，则设定为 n*10
    if maxiter is None:
        maxiter = n * 10

    # 获取 A 的矩阵向量乘法函数和 M 的预处理函数
    matvec = A.matvec
    psolve = M.matvec

    # 计算 rhotol 作为比较小的浮点数值
    rhotol = np.finfo(x.dtype.char).eps ** 2

    # 初始化残差 r
    r = b - matvec(x) if x.any() else b.copy()

    # 复制 r 以备后用
    rtilde = r.copy()

    # 计算向量 b 的二范数
    bnorm = np.linalg.norm(b)
    if bnorm == 0:
        bnorm = 1

    # Dummy values to initialize vars, silence linter warnings
    rho_prev, p, u, q = None, None, None, None
    for iteration in range(maxiter):
        rnorm = np.linalg.norm(r)
        if rnorm < atol:  # 判断是否达到指定的收敛精度
            return postprocess(x), 0

        rho_cur = dotprod(rtilde, r)
        if np.abs(rho_cur) < rhotol:  # 判断是否发生了迭代过程中的矩阵分解情况
            return postprocess, -10

        if iteration > 0:
            beta = rho_cur / rho_prev

            # 更新向量 u 和 p
            u[:] = r[:]
            u += beta*q

            p *= beta
            p += q
            p *= beta
            p += u

        else:  # 第一次迭代的特殊处理
            p = r.copy()
            u = r.copy()
            q = np.empty_like(r)

        phat = psolve(p)  # 解决 p 的线性系统
        vhat = matvec(phat)  # 使用 phat 计算向量乘法
        rv = dotprod(rtilde, vhat)

        if rv == 0:  # 判断点积是否出现分解情况
            return postprocess(x), -11

        alpha = rho_cur / rv
        q[:] = u[:]
        q -= alpha*vhat
        uhat = psolve(u + q)  # 解决 u + q 的线性系统
        x += alpha*uhat  # 更新解向量 x

        # 由于数值误差累积，计算实际的残差而不是下面的两行
        # 原始 FORTRAN 模板中的计算方式，仍然使用单个 matvec。

        r = b - matvec(x)  # 计算新的残差

        rho_prev = rho_cur

        if callback:
            callback(x)

    else:  # for 循环耗尽
        # 返回未完成的进展
        return postprocess(x), maxiter
# 使用 GMRES 方法解决线性方程组 Ax = b
def gmres(A, b, x0=None, *, rtol=1e-5, atol=0., restart=None, maxiter=None, M=None,
          callback=None, callback_type=None):
    """
    Use Generalized Minimal RESidual iteration to solve ``Ax = b``.

    Parameters
    ----------
    A : {sparse matrix, ndarray, LinearOperator}
        线性系统的实数或复数 N×N 矩阵。也可以是一个线性操作符，可以使用例如
        ``scipy.sparse.linalg.LinearOperator`` 来产生 ``Ax``。
    b : ndarray
        线性系统的右手边。具有形状 (N,) 或 (N,1)。
    x0 : ndarray
        解的初始猜测（默认为零向量）。
    atol, rtol : float
        收敛测试的参数。为了达到收敛性，应满足 ``norm(b - A @ x) <= max(rtol*norm(b), atol)``。
        默认值为 ``atol=0.`` 和 ``rtol=1e-5``。
    restart : int, optional
        重启之间的迭代次数。较大的值增加迭代成本，但可能对收敛性有所帮助。
        如果省略，将使用 ``min(20, n)``。
    maxiter : int, optional
        最大迭代次数（重启周期数）。即使在达到指定的容差之前，迭代也将在 maxiter 步之后停止。
        参见 `callback_type`。
    M : {sparse matrix, ndarray, LinearOperator}
        A 的预条件器的逆。M 应近似于 A 的逆并且易于求解（见 Notes）。有效的预条件显著提高了
        收敛速度，这意味着需要更少的迭代来达到给定的误差容限。
        默认情况下，不使用预条件器。在这个实现中，使用左预条件，最小化预处理残差。
        然而，最终收敛性是根据 ``b - A @ x`` 残差进行测试。
    callback : function
        用户提供的函数，在每次迭代后调用。调用形式为 ``callback(args)``, 其中 `args` 根据 `callback_type` 选择。
    callback_type : {'x', 'pr_norm', 'legacy'}, optional
        请求的回调函数参数:
          - ``x``: 当前迭代值（ndarray），在每次重启时调用
          - ``pr_norm``: 相对（预处理）残差范数（float），在每次内部迭代时调用
          - ``legacy``（默认）: 与 ``pr_norm`` 相同，但也改变了 `maxiter` 的含义，计算内部迭代次数而不是重启周期数。

        如果未设置 `callback`，此关键字不起作用。

    Returns
    -------
    x : ndarray
        收敛的解。
    info : int
        提供收敛信息:
            0  : 成功退出
            >0 : 未达到容差的收敛，迭代次数

    See Also
    --------
    LinearOperator

    Notes
    -----
    ```
    """
    A preconditioner, P, is chosen such that P is close to A but easy to solve
    for. The preconditioner parameter required by this routine is
    ``M = P^-1``. The inverse should preferably not be calculated
    explicitly. Rather, use the following template to produce M::

      # Construct a linear operator that computes P^-1 @ x.
      import scipy.sparse.linalg as spla
      M_x = lambda x: spla.spsolve(P, x)
      M = spla.LinearOperator((n, n), M_x)

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csc_matrix
    >>> from scipy.sparse.linalg import gmres
    >>> A = csc_matrix([[3, 2, 0], [1, -1, 0], [0, 5, 1]], dtype=float)
    >>> b = np.array([2, 4, -1], dtype=float)
    >>> x, exitCode = gmres(A, b, atol=1e-5)
    >>> print(exitCode)            # 0 indicates successful convergence
    0
    >>> np.allclose(A.dot(x), b)
    True
    """
    if callback is not None and callback_type is None:
        # Warn about 'callback_type' semantic changes.
        # Probably should be removed only in far future, Scipy 2.0 or so.
        msg = ("scipy.sparse.linalg.gmres called without specifying "
               "`callback_type`. The default value will be changed in"
               " a future release. For compatibility, specify a value "
               "for `callback_type` explicitly, e.g., "
               "``gmres(..., callback_type='pr_norm')``, or to retain the "
               "old behavior ``gmres(..., callback_type='legacy')``"
               )
        warnings.warn(msg, category=DeprecationWarning, stacklevel=3)

    if callback_type is None:
        callback_type = 'legacy'

    if callback_type not in ('x', 'pr_norm', 'legacy'):
        raise ValueError(f"Unknown callback_type: {callback_type!r}")

    if callback is None:
        callback_type = None

    A, M, x, b, postprocess = make_system(A, M, x0, b)
    matvec = A.matvec
    psolve = M.matvec
    n = len(b)
    bnrm2 = np.linalg.norm(b)

    atol, _ = _get_atol_rtol('gmres', bnrm2, atol, rtol)

    if bnrm2 == 0:
        return postprocess(b), 0

    eps = np.finfo(x.dtype.char).eps

    dotprod = np.vdot if np.iscomplexobj(x) else np.dot

    if maxiter is None:
        maxiter = n*10

    if restart is None:
        restart = 20
    restart = min(restart, n)

    Mb_nrm2 = np.linalg.norm(psolve(b))

    # ====================================================
    # =========== Tolerance control from gh-8400 =========
    # ====================================================
    # Tolerance passed to GMRESREVCOM applies to the inner
    # iteration and deals with the left-preconditioned
    # residual.
    ptol_max_factor = 1.
    ptol = Mb_nrm2 * min(ptol_max_factor, atol / bnrm2)
    presid = 0.
    # ====================================================
    lartg = get_lapack_funcs('lartg', dtype=x.dtype)

    # allocate internal variables
    v = np.empty([restart+1, n], dtype=x.dtype)
    h = np.zeros([restart, restart+1], dtype=x.dtype)
    # 创建一个大小为 [restart, 2] 的全零数组，数据类型与 x 相同
    givens = np.zeros([restart, 2], dtype=x.dtype)
    
    # 设置 legacy iteration count 的初始值为 0
    inner_iter = 0
    
    # 根据残差范数 rnorm 和容差 atol 的比较结果，确定 info 的值：
    # 若 rnorm <= atol，则 info 为 0；否则为 maxiter
    info = 0 if (rnorm <= atol) else maxiter
    
    # 返回经过后处理函数 postprocess 处理后的 x，以及计算得到的 info
    return postprocess(x), info
    # 保存原始的 A 矩阵，可能在 make_system 函数中被修改
    A_ = A
    # 调用 make_system 函数，处理 A 矩阵、M1、M2、x0、b，返回处理后的 A 矩阵、M、x、b 和后处理函数 postprocess
    A, M, x, b, postprocess = make_system(A, None, x0, b)
    # 计算向量 b 的二范数
    bnrm2 = np.linalg.norm(b)

    # 根据 bnrm2 和用户指定的 atol、rtol 计算实际的 atol
    atol, _ = _get_atol_rtol('qmr', bnrm2, atol, rtol)

    # 如果 bnrm2 为零，直接返回后处理过的 b 和迭代次数 0
    if bnrm2 == 0:
        return postprocess(b), 0
    # 如果 M1 和 M2 都为 None，则根据 A_ 的属性动态定义四个函数
    if M1 is None and M2 is None:
        # 检查 A_ 是否有 'psolve' 属性
        if hasattr(A_, 'psolve'):
            # 定义左侧预解函数 left_psolve
            def left_psolve(b):
                return A_.psolve(b, 'left')

            # 定义右侧预解函数 right_psolve
            def right_psolve(b):
                return A_.psolve(b, 'right')

            # 定义左侧反预解函数 left_rpsolve
            def left_rpsolve(b):
                return A_.rpsolve(b, 'left')

            # 定义右侧反预解函数 right_rpsolve
            def right_rpsolve(b):
                return A_.rpsolve(b, 'right')

            # 根据定义的函数创建线性操作对象 M1 和 M2
            M1 = LinearOperator(A.shape,
                                matvec=left_psolve,
                                rmatvec=left_rpsolve)
            M2 = LinearOperator(A.shape,
                                matvec=right_psolve,
                                rmatvec=right_rpsolve)
        else:
            # 如果 A_ 没有 'psolve' 属性，则定义一个恒等函数 id
            def id(b):
                return b

            # 根据恒等函数创建线性操作对象 M1 和 M2
            M1 = LinearOperator(A.shape, matvec=id, rmatvec=id)
            M2 = LinearOperator(A.shape, matvec=id, rmatvec=id)

    # 计算向量 b 的长度
    n = len(b)

    # 如果 maxiter 未指定，则设置为 n 的 10 倍
    if maxiter is None:
        maxiter = n * 10

    # 根据 x 的复杂性选择点乘函数，复数使用 np.vdot，实数使用 np.dot
    dotprod = np.vdot if np.iscomplexobj(x) else np.dot

    # 计算浮点数精度，设置一系列容差值
    rhotol = np.finfo(x.dtype.char).eps
    betatol = rhotol
    gammatol = rhotol
    deltatol = rhotol
    epsilontol = rhotol
    xitol = rhotol

    # 计算初始残差 r
    r = b - A.matvec(x) if x.any() else b.copy()

    # 初始化 vtilde 和 wtilde 为 r 的副本
    vtilde = r.copy()

    # 计算 M1 对 vtilde 的作用结果 y，并计算其二范数 rho
    y = M1.matvec(vtilde)
    rho = np.linalg.norm(y)

    # 初始化 wtilde 为 r 的副本
    wtilde = r.copy()

    # 计算 M2 对 wtilde 的作用结果 z，并计算其二范数 xi
    z = M2.rmatvec(wtilde)
    xi = np.linalg.norm(z)

    # 初始化 gamma, eta, theta 和 v, w 的数组
    gamma, eta, theta = 1, -1, 0
    v = np.empty_like(vtilde)
    w = np.empty_like(wtilde)

    # 初始化 epsilon, q, d, p, s 为 None，用于抑制 Linter 警告
    epsilon, q, d, p, s = None, None, None, None, None
    for iteration in range(maxiter):
        # 检查残差 r 的二范数是否小于给定的收敛容差 atol，判断是否收敛
        if np.linalg.norm(r) < atol:  # Are we done?
            return postprocess(x), 0
        # 检查 rho 的绝对值是否小于 rho 容差 rhotol，判断 rho 是否趋近于零
        if np.abs(rho) < rhotol:  # rho breakdown
            return postprocess(x), -10
        # 检查 xi 的绝对值是否小于 xi 容差 xitol，判断 xi 是否趋近于零
        if np.abs(xi) < xitol:  # xi breakdown
            return postprocess(x), -15

        # 更新 v 和 y 的值
        v[:] = vtilde[:]
        v *= (1 / rho)
        y *= (1 / rho)

        # 更新 w 和 z 的值
        w[:] = wtilde[:]
        w *= (1 / xi)
        z *= (1 / xi)

        # 计算 delta，并检查其绝对值是否小于 delta 容差 deltatol
        delta = dotprod(z, y)
        if np.abs(delta) < deltatol:  # delta breakdown
            return postprocess(x), -13

        # 计算 ytilde 和 ztilde
        ytilde = M2.matvec(y)
        ztilde = M1.rmatvec(z)

        # 如果迭代次数大于 0，执行以下操作；否则，执行首次迭代步骤
        if iteration > 0:
            ytilde -= (xi * delta / epsilon) * p
            p[:] = ytilde[:]
            ztilde -= (rho * (delta / epsilon).conj()) * q
            q[:] = ztilde[:]
        else:  # First spin
            p = ytilde.copy()
            q = ztilde.copy()

        # 计算 ptilde，并计算 epsilon
        ptilde = A.matvec(p)
        epsilon = dotprod(q, ptilde)

        # 检查 epsilon 的绝对值是否小于 epsilon 容差 epsilontol
        if np.abs(epsilon) < epsilontol:  # epsilon breakdown
            return postprocess(x), -14

        # 计算 beta，并检查其绝对值是否小于 beta 容差 betatol
        beta = epsilon / delta
        if np.abs(beta) < betatol:  # beta breakdown
            return postprocess(x), -11

        # 更新 vtilde 和 y 的值
        vtilde[:] = ptilde[:]
        vtilde -= beta*v
        y = M1.matvec(vtilde)

        # 更新 rho 和 xi 的前一次值，并计算 gamma 和 theta
        rho_prev = rho
        rho = np.linalg.norm(y)
        wtilde[:] = w[:]
        wtilde *= - beta.conj()
        wtilde += A.rmatvec(q)
        z = M2.rmatvec(wtilde)
        xi = np.linalg.norm(z)
        gamma_prev = gamma
        theta_prev = theta
        theta = rho / (gamma_prev * np.abs(beta))
        gamma = 1 / np.sqrt(1 + theta**2)

        # 检查 gamma 的绝对值是否小于 gamma 容差 gammatol
        if np.abs(gamma) < gammatol:  # gamma breakdown
            return postprocess(x), -12

        # 更新 eta
        eta *= -(rho_prev / beta) * (gamma / gamma_prev)**2

        # 如果迭代次数大于 0，更新 d 和 s 的值；否则，执行首次迭代步骤
        if iteration > 0:
            d *= (theta_prev * gamma) ** 2
            d += eta*p
            s *= (theta_prev * gamma) ** 2
            s += eta*ptilde
        else:
            d = p.copy()
            d *= eta
            s = ptilde.copy()
            s *= eta

        # 更新解向量 x 和残差 r
        x += d
        r -= s

        # 如果有回调函数，则执行回调函数
        if callback:
            callback(x)

    else:  # for loop exhausted
        # 如果迭代次数达到最大值但仍未收敛，则返回当前解的后处理结果和最大迭代次数
        return postprocess(x), maxiter
```