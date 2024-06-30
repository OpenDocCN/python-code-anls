# `D:\src\scipysrc\scipy\scipy\optimize\_linprog_ip.py`

```
# 内点法解线性规划问题
"""
The *interior-point* method uses the primal-dual path following algorithm
outlined in [1]_. This algorithm supports sparse constraint matrices and
is typically faster than the simplex methods, especially for large, sparse
problems. Note, however, that the solution returned may be slightly less
accurate than those of the simplex methods and will not, in general,
correspond with a vertex of the polytope defined by the constraints.

    .. versionadded:: 1.0.0

References
----------
.. [1] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
       optimizer for linear programming: an implementation of the
       homogeneous algorithm." High performance optimization. Springer US,
       2000. 197-232.
"""

# 作者: Matt Haberland

import numpy as np
import scipy as sp
import scipy.sparse as sps
from warnings import warn
from scipy.linalg import LinAlgError
from ._optimize import OptimizeWarning, OptimizeResult, _check_unknown_options
from ._linprog_util import _postsolve
has_umfpack = True
has_cholmod = True

# 尝试导入必要的稀疏矩阵解析器和优化库
try:
    import sksparse  # noqa: F401
    from sksparse.cholmod import cholesky as cholmod  # noqa: F401
    from sksparse.cholmod import analyze as cholmod_analyze
except ImportError:
    has_cholmod = False

# 尝试导入UMFPACK线性代数库
try:
    import scikits.umfpack  # test whether to use factorized  # noqa: F401
except ImportError:
    has_umfpack = False


def _get_solver(M, sparse=False, lstsq=False, sym_pos=True,
                cholesky=True, permc_spec='MMD_AT_PLUS_A'):
    """
    根据求解器选项返回适当的线性系统求解器。

    Parameters
    ----------
    M : 2-D array
        在参考文献[4]中定义的方程式8.31
    sparse : bool (default = False)
        如果要求解的系统是稀疏的，则为True。当原始的``A_ub``和``A_eq``数组是稀疏时通常设置为True。
    lstsq : bool (default = False)
        如果系统病态和/或（几乎）奇异，因此需要更稳健的最小二乘求解器，则为True。有时在接近解决方案时需要这样做。
    sym_pos : bool (default = True)
        如果系统矩阵是对称正定的，则为True。
        有时由于数值困难，即使系统应该是对称正定的，在接近解决方案时也需要将其设置为false。
    cholesky : bool (default = True)
        如果要通过Cholesky分解而不是LU分解来解决系统，则为True。通常情况下，除非问题非常小或容易发生数值困难，否则这比较快速。
    """
    permc_spec : str (default = 'MMD_AT_PLUS_A')
        # 指定给 SuperLU 使用的稀疏性保持策略。可接受的取值有：

        # - ``NATURAL``: 自然顺序。
        # - ``MMD_ATA``: A^T A 结构的最小度排序。
        # - ``MMD_AT_PLUS_A``: A^T+A 结构的最小度排序。
        # - ``COLAMD``: 近似最小度列排序。

        # 参见 SuperLU 文档。

    Returns
    -------
    solve : function
        # 返回适当求解器函数的句柄

    """
    try:
        if sparse:
            if lstsq:
                # 如果稀疏并且需要最小二乘解决方案
                def solve(r, sym_pos=False):
                    return sps.linalg.lsqr(M, r)[0]
            elif cholesky:
                try:
                    # 尝试进行 Cholesky 分解，若出错将会抛出异常
                    _get_solver.cholmod_factor.cholesky_inplace(M)
                except Exception:
                    # 分析并进行 Cholesky 分解
                    _get_solver.cholmod_factor = cholmod_analyze(M)
                    _get_solver.cholmod_factor.cholesky_inplace(M)
                solve = _get_solver.cholmod_factor
            else:
                if has_umfpack and sym_pos:
                    # 使用预因子化的解决方法
                    solve = sps.linalg.factorized(M)
                else:  # factorized 不支持 permc_spec
                    # 使用 SPLU 分解来解决线性方程组
                    solve = sps.linalg.splu(M, permc_spec=permc_spec).solve

        else:
            if lstsq:  # 有时需要作为解决方案的一部分
                # 如果不是稀疏并且需要最小二乘解决方案
                def solve(r):
                    return sp.linalg.lstsq(M, r)[0]
            elif cholesky:
                # 进行 Cholesky 分解
                L = sp.linalg.cho_factor(M)

                def solve(r):
                    return sp.linalg.cho_solve(L, r)
            else:
                # 似乎缓存了矩阵的因子化，因此解决多个右手边会更快
                def solve(r, sym_pos=sym_pos):
                    if sym_pos:
                        return sp.linalg.solve(M, r, assume_a="pos")
                    else:
                        return sp.linalg.solve(M, r)
    # 这里可能会出现许多问题，很难说清楚所有可能的问题。这并不重要：
    # 如果矩阵无法进行因子化，返回 None。get_solver 将使用不同的输入再次调用，
    # 新的过程将尝试对矩阵进行因子化。
    except KeyboardInterrupt:
        raise
    except Exception:
        return None
    return solve
# 定义函数_get_delta，用于求解线性规划问题的搜索方向增量
def _get_delta(A, b, c, x, y, z, tau, kappa, gamma, eta, sparse=False,
               lstsq=False, sym_pos=True, cholesky=True, pc=True, ip=False,
               permc_spec='MMD_AT_PLUS_A'):
    """
    Given standard form problem defined by ``A``, ``b``, and ``c``;
    current variable estimates ``x``, ``y``, ``z``, ``tau``, and ``kappa``;
    algorithmic parameters ``gamma`` and ``eta``;
    and options ``sparse``, ``lstsq``, ``sym_pos``, ``cholesky``, ``pc``
    (predictor-corrector), and ``ip`` (initial point improvement),
    get the search direction for increments to the variable estimates.

    Parameters
    ----------
    As defined in [4], except:
    sparse : bool
        True if the system to be solved is sparse. This is typically set
        True when the original ``A_ub`` and ``A_eq`` arrays are sparse.
    lstsq : bool
        True if the system is ill-conditioned and/or (nearly) singular and
        thus a more robust least-squares solver is desired. This is sometimes
        needed as the solution is approached.
    sym_pos : bool
        True if the system matrix is symmetric positive definite
        Sometimes this needs to be set false as the solution is approached,
        even when the system should be symmetric positive definite, due to
        numerical difficulties.
    cholesky : bool
        True if the system is to be solved by Cholesky, rather than LU,
        decomposition. This is typically faster unless the problem is very
        small or prone to numerical difficulties.
    pc : bool
        True if the predictor-corrector method of Mehrota is to be used. This
        is almost always (if not always) beneficial. Even though it requires
        the solution of an additional linear system, the factorization
        is typically (implicitly) reused so solution is efficient, and the
        number of algorithm iterations is typically reduced.
    ip : bool
        True if the improved initial point suggestion due to [4] section 4.3
        is desired. It's unclear whether this is beneficial.
    permc_spec : str (default = 'MMD_AT_PLUS_A')
        (Has effect only with ``sparse = True``, ``lstsq = False``, ``sym_pos =
        True``.) A matrix is factorized in each iteration of the algorithm.
        This option specifies how to permute the columns of the matrix for
        sparsity preservation. Acceptable values are:

        - ``NATURAL``: natural ordering.
        - ``MMD_ATA``: minimum degree ordering on the structure of A^T A.
        - ``MMD_AT_PLUS_A``: minimum degree ordering on the structure of A^T+A.
        - ``COLAMD``: approximate minimum degree column ordering.

        This option can impact the convergence of the
        interior point algorithm; test different values to determine which
        performs best for your problem. For more information, refer to
        ``scipy.sparse.linalg.splu``.

    Returns
    -------
    Search directions as defined in [4]

    References
    ----------
    [4] Placeholder for reference details.
    """
    """
    if A.shape[0] == 0:
        # 如果没有约束条件，某些求解器可能会失败，而不是返回空解。这里确保处理这种情况。
        sparse, lstsq, sym_pos, cholesky = False, False, True, False
    n_x = len(x)

    # [4] Equation 8.8
    # 计算 r_P, r_D, r_G 和 mu，这些是优化问题中的残差和对偶间隙
    r_P = b * tau - A.dot(x)
    r_D = c * tau - A.T.dot(y) - z
    r_G = c.dot(x) - b.transpose().dot(y) + kappa
    mu = (x.dot(z) + tau * kappa) / (n_x + 1)

    #  根据 [4] Equation 8.31 组装矩阵 M
    Dinv = x / z

    if sparse:
        # 如果稀疏矩阵优化被启用，则构建 M
        M = A.dot(sps.diags(Dinv, 0, format="csc").dot(A.T))
    else:
        # 否则，构建 M
        M = A.dot(Dinv.reshape(-1, 1) * A.T)
    solve = _get_solver(M, sparse, lstsq, sym_pos, cholesky, permc_spec)

    # pc: "predictor-corrector" [4] Section 4.1
    # 在开发中，此选项可以关闭，但通常显著提高性能
    n_corrections = 1 if pc else 0

    i = 0
    alpha, d_x, d_z, d_tau, d_kappa = 0, 0, 0, 0, 0
    # 返回优化变量的增量，包括 d_x, d_y, d_z, d_tau, d_kappa
    return d_x, d_y, d_z, d_tau, d_kappa
# 根据文献 [4] 中的方程 8.31 和 8.32 实现的功能
def _sym_solve(Dinv, A, r1, r2, solve):
    # 计算 r 向量，按照 [4] 中的方程 8.31
    r = r2 + A.dot(Dinv * r1)
    # 使用给定的求解器 solve 求解 v 向量
    v = solve(r)
    # 计算 u 向量，按照 [4] 中的方程 8.32
    u = Dinv * (A.T.dot(v) - r1)
    return u, v


# 根据文献 [4] 中的方程 8.21 实现的功能
def _get_step(x, d_x, z, d_z, tau, d_tau, kappa, d_kappa, alpha0):
    # [4] 中的 8.21 方程，忽略了对 8.20 的要求
    # 在原始空间和对偶空间中采取相同的步骤
    # alpha0 基本上是 [4] 表 8.1 中的 beta3，但在 Mehrota 修正器和初始点修正中，使用了值 1
    # 计算 x 和 z 向量的步长
    i_x = d_x < 0
    i_z = d_z < 0
    alpha_x = alpha0 * np.min(x[i_x] / -d_x[i_x]) if np.any(i_x) else 1
    alpha_tau = alpha0 * tau / -d_tau if d_tau < 0 else 1
    alpha_z = alpha0 * np.min(z[i_z] / -d_z[i_z]) if np.any(i_z) else 1
    alpha_kappa = alpha0 * kappa / -d_kappa if d_kappa < 0 else 1
    # 综合计算出最终的步长 alpha
    alpha = np.min([1, alpha_x, alpha_tau, alpha_z, alpha_kappa])
    return alpha


# 给定问题状态码，返回更详细的信息消息
def _get_message(status):
    # 根据给定的 status 码返回相应的优化过程状态描述
    messages = (
        ["优化成功终止。",
         "算法在收敛之前达到了迭代限制。",
         "算法成功终止，并确定问题是不可行的。",
         "算法成功终止，并确定问题是无界的。",
         "在问题收敛之前遇到了严重的数值困难。请检查问题的表达方式是否有误，"
         "线性等式约束的独立性，以及合理的缩放和矩阵条件数。如果继续遇到此错误，请提交 Bug 报告。"
         ])
    return messages[status]


# 执行一步优化步骤
def _do_step(x, y, z, tau, kappa, d_x, d_y, d_z, d_tau, d_kappa, alpha):
    # 实现了 [4] 中的 Equation 8.9，即根据给定的方程式更新多个变量的值
    
    """
    x = x + alpha * d_x
    tau = tau + alpha * d_tau
    z = z + alpha * d_z
    kappa = kappa + alpha * d_kappa
    y = y + alpha * d_y
    return x, y, z, tau, kappa
    """
def _get_blind_start(shape):
    """
    Return the starting point from [4] 4.4

    References
    ----------
    .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
           optimizer for linear programming: an implementation of the
           homogeneous algorithm." High performance optimization. Springer US,
           2000. 197-232.

    """
    # 解构形状元组
    m, n = shape
    # 初始化 x0 为全1向量，y0 为全0向量，z0 为全1向量
    x0 = np.ones(n)
    y0 = np.zeros(m)
    z0 = np.ones(n)
    # 初始化 tau0 和 kappa0 为1
    tau0 = 1
    kappa0 = 1
    return x0, y0, z0, tau0, kappa0


def _indicators(A, b, c, c0, x, y, z, tau, kappa):
    """
    Implementation of several equations from [4] used as indicators of
    the status of optimization.

    References
    ----------
    .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
           optimizer for linear programming: an implementation of the
           homogeneous algorithm." High performance optimization. Springer US,
           2000. 197-232.

    """

    # residuals for termination are relative to initial values
    # 获取初始值 x0, y0, z0, tau0, kappa0
    x0, y0, z0, tau0, kappa0 = _get_blind_start(A.shape)

    # See [4], Section 4 - The Homogeneous Algorithm, Equation 8.8
    # 定义对应于公式 8.8 的残差函数
    def r_p(x, tau):
        return b * tau - A.dot(x)

    def r_d(y, z, tau):
        return c * tau - A.T.dot(y) - z

    def r_g(x, y, kappa):
        return kappa + c.dot(x) - b.dot(y)

    # np.dot unpacks if they are arrays of size one
    # 定义计算 mu 的函数
    def mu(x, tau, z, kappa):
        return (x.dot(z) + np.dot(tau, kappa)) / (len(x) + 1)

    # 计算当前迭代的目标函数值
    obj = c.dot(x / tau) + c0

    # 定义计算向量范数的函数
    def norm(a):
        return np.linalg.norm(a)

    # See [4], Section 4.5 - The Stopping Criteria
    # 计算各种残差的相对值
    r_p0 = r_p(x0, tau0)
    r_d0 = r_d(y0, z0, tau0)
    r_g0 = r_g(x0, y0, kappa0)
    mu_0 = mu(x0, tau0, z0, kappa0)
    rho_A = norm(c.T.dot(x) - b.T.dot(y)) / (tau + norm(b.T.dot(y)))
    rho_p = norm(r_p(x, tau)) / max(1, norm(r_p0))
    rho_d = norm(r_d(y, z, tau)) / max(1, norm(r_d0))
    rho_g = norm(r_g(x, y, kappa)) / max(1, norm(r_g0))
    rho_mu = mu(x, tau, z, kappa) / mu_0
    # 返回各项指标以及目标函数值
    return rho_p, rho_d, rho_A, rho_g, rho_mu, obj


def _display_iter(rho_p, rho_d, rho_g, alpha, rho_mu, obj, header=False):
    """
    Print indicators of optimization status to the console.

    Parameters
    ----------
    rho_p : float
        The (normalized) primal feasibility, see [4] 4.5
    rho_d : float
        The (normalized) dual feasibility, see [4] 4.5
    rho_g : float
        The (normalized) duality gap, see [4] 4.5
    alpha : float
        The step size, see [4] 4.3
    rho_mu : float
        The (normalized) path parameter, see [4] 4.5
    obj : float
        The objective function value of the current iterate
    header : bool
        True if a header is to be printed

    References
    ----------

    """
    # 打印优化状态指标到控制台
    if header:
        print("Primal Feasibility\tDual Feasibility\tDuality Gap\tStep Size\tPath Parameter\tObjective")
    print(f"{rho_p}\t{rho_d}\t{rho_g}\t{alpha}\t{rho_mu}\t{obj}")
    """
    如果设置了 header 参数，则打印下面的表头信息：
    'Primal Feasibility ', 'Dual Feasibility   ', 'Duality Gap        ',
    'Step            ', 'Path Parameter     ', 'Objective          '
    
    这里使用的格式化字符串定义了输出的格式，每个字段都是固定宽度，左对齐，并且保留13位小数。
    """
    if header:
        # 打印表头信息，每列之间使用空格分隔
        print("Primal Feasibility ",
              "Dual Feasibility   ",
              "Duality Gap        ",
              "Step            ",
              "Path Parameter     ",
              "Objective          ")
    
    # no clue why this works
    # fmt 是一个格式化字符串，定义了输出的格式，确保各个数据项的对齐和精度
    fmt = '{0:<20.13}{1:<20.13}{2:<20.13}{3:<17.13}{4:<20.13}{5:<20.13}'
    # 使用 fmt 格式化输出各个变量的值，确保输出的精度和对齐性
    print(fmt.format(
        float(rho_p),               # 输出 rho_p 的值，保留13位小数，左对齐到20个字符宽度
        float(rho_d),               # 输出 rho_d 的值，保留13位小数，左对齐到20个字符宽度
        float(rho_g),               # 输出 rho_g 的值，保留13位小数，左对齐到20个字符宽度
        alpha if isinstance(alpha, str) else float(alpha),  # 根据 alpha 的类型输出不同的格式，保留13位小数，左对齐到17个字符宽度
        float(rho_mu),              # 输出 rho_mu 的值，保留13位小数，左对齐到20个字符宽度
        float(obj)))                # 输出 obj 的值，保留13位小数，左对齐到20个字符宽度
# 定义一个函数，用于解决标准形式的线性规划问题，使用内点方法 [4]
def _ip_hsd(A, b, c, c0, alpha0, beta, maxiter, disp, tol, sparse, lstsq,
            sym_pos, cholesky, pc, ip, permc_spec, callback, postsolve_args):
    r"""
    Solve a linear programming problem in standard form:

    Minimize::

        c @ x

    Subject to::

        A @ x == b
            x >= 0

    using the interior point method of [4].

    Parameters
    ----------
    A : 2-D array
        2-D array such that ``A @ x``, gives the values of the equality
        constraints at ``x``.
    b : 1-D array
        1-D array of values representing the RHS of each equality constraint
        (row) in ``A`` (for standard form problem).
    c : 1-D array
        Coefficients of the linear objective function to be minimized (for
        standard form problem).
    c0 : float
        Constant term in objective function due to fixed (and eliminated)
        variables. (Purely for display.)
    alpha0 : float
        The maximal step size for Mehrota's predictor-corrector search
        direction; see :math:`\beta_3`of [4] Table 8.1
    beta : float
        The desired reduction of the path parameter :math:`\mu` (see  [6]_)
    maxiter : int
        The maximum number of iterations of the algorithm.
    disp : bool
        Set to ``True`` if indicators of optimization status are to be printed
        to the console each iteration.
    tol : float
        Termination tolerance; see [4]_ Section 4.5.
    sparse : bool
        Set to ``True`` if the problem is to be treated as sparse. However,
        the inputs ``A_eq`` and ``A_ub`` should nonetheless be provided as
        (dense) arrays rather than sparse matrices.
    lstsq : bool
        Set to ``True`` if the problem is expected to be very poorly
        conditioned. This should always be left as ``False`` unless severe
        numerical difficulties are frequently encountered, and a better option
        would be to improve the formulation of the problem.
    sym_pos : bool
        Leave ``True`` if the problem is expected to yield a well conditioned
        symmetric positive definite normal equation matrix (almost always).
    cholesky : bool
        Set to ``True`` if the normal equations are to be solved by explicit
        Cholesky decomposition followed by explicit forward/backward
        substitution. This is typically faster for moderate, dense problems
        that are numerically well-behaved.
    pc : bool
        Leave ``True`` if the predictor-corrector method of Mehrota is to be
        used. This is almost always (if not always) beneficial.
    ip : bool
        Set to ``True`` if the improved initial point suggestion due to [4]_
        Section 4.3 is desired. It's unclear whether this is beneficial.
    permc_spec : str or None
        Permutation used to decompose the constraint matrix; see [4]_ Section 4.8
    callback : callable or None
        If a callback function is given, it will be called after each
        iteration. See [4]_ Section 4.7 for more detail.
    postsolve_args : tuple
        Extra arguments passed to the postsolve routine; see [4]_ Section 4.9
    """
    # 实现线性规划问题求解，具体细节在 [4] 中描述
    permc_spec : str (default = 'MMD_AT_PLUS_A')
        (Has effect only with ``sparse = True``, ``lstsq = False``, ``sym_pos =
        True``.) A matrix is factorized in each iteration of the algorithm.
        This option specifies how to permute the columns of the matrix for
        sparsity preservation. Acceptable values are:

        - ``NATURAL``: natural ordering.
        - ``MMD_ATA``: minimum degree ordering on the structure of A^T A.
        - ``MMD_AT_PLUS_A``: minimum degree ordering on the structure of A^T+A.
        - ``COLAMD``: approximate minimum degree column ordering.

        This option can impact the convergence of the
        interior point algorithm; test different values to determine which
        performs best for your problem. For more information, refer to
        ``scipy.sparse.linalg.splu``.
    callback : callable, optional
        If a callback function is provided, it will be called within each
        iteration of the algorithm. The callback function must accept a single
        `scipy.optimize.OptimizeResult` consisting of the following fields:

            x : 1-D array
                Current solution vector
            fun : float
                Current value of the objective function
            success : bool
                True only when an algorithm has completed successfully,
                so this is always False as the callback function is called
                only while the algorithm is still iterating.
            slack : 1-D array
                The values of the slack variables. Each slack variable
                corresponds to an inequality constraint. If the slack is zero,
                the corresponding constraint is active.
            con : 1-D array
                The (nominally zero) residuals of the equality constraints,
                that is, ``b - A_eq @ x``
            phase : int
                The phase of the algorithm being executed. This is always
                1 for the interior-point method because it has only one phase.
            status : int
                For revised simplex, this is always 0 because if a different
                status is detected, the algorithm terminates.
            nit : int
                The number of iterations performed.
            message : str
                A string descriptor of the exit status of the optimization.
    postsolve_args : tuple
        Data needed by _postsolve to convert the solution to the standard-form
        problem into the solution to the original problem.

    Returns
    -------
    x_hat : float
        Solution vector (for standard form problem).
    status : int
        An integer representing the exit status of the optimization::

         0 : Optimization terminated successfully
         1 : Iteration limit reached
         2 : Problem appears to be infeasible
         3 : Problem appears to be unbounded
         4 : Serious numerical difficulties encountered
    """
    message : str
        优化过程的退出状态描述字符串。
    iteration : int
        解决问题所需的迭代次数。

    References
    ----------
    .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
           optimizer for linear programming: an implementation of the
           homogeneous algorithm." High performance optimization. Springer US,
           2000. 197-232.
    .. [6] Freund, Robert M. "Primal-Dual Interior-Point Methods for Linear
           Programming based on Newton's Method." Unpublished Course Notes,
           March 2004. Available 2/25/2017 at:
           https://ocw.mit.edu/courses/sloan-school-of-management/15-084j-nonlinear-programming-spring-2004/lecture-notes/lec14_int_pt_mthd.pdf

    """

    iteration = 0  # 初始化迭代次数为0

    # 默认的初始点
    x, y, z, tau, kappa = _get_blind_start(A.shape)

    # 第一次迭代是对初始点的特殊改进
    ip = ip if pc else False  # 如果 pc 为真则 ip 为真，否则为假

    # [4] 4.5
    # 计算指示量
    rho_p, rho_d, rho_A, rho_g, rho_mu, obj = _indicators(
        A, b, c, c0, x, y, z, tau, kappa)
    go = rho_p > tol or rho_d > tol or rho_A > tol  # 我们可能会幸运地得到优化结果 :)

    if disp:
        _display_iter(rho_p, rho_d, rho_g, "-", rho_mu, obj, header=True)  # 显示迭代信息
    if callback is not None:
        x_o, fun, slack, con = _postsolve(x/tau, postsolve_args)
        # 调用回调函数，生成优化结果对象
        res = OptimizeResult({'x': x_o, 'fun': fun, 'slack': slack,
                              'con': con, 'nit': iteration, 'phase': 1,
                              'complete': False, 'status': 0,
                              'message': "", 'success': False})
        callback(res)  # 调用回调函数传递结果对象

    status = 0  # 设置优化状态为0
    message = "Optimization terminated successfully."  # 默认优化成功的信息提示

    if sparse:
        A = sps.csc_matrix(A)  # 如果稀疏标志位为真，则将 A 转换为 CSC 矩阵格式

    x_hat = x / tau
    # [4] Theorem 8.2 后的声明
    return x_hat, status, message, iteration  # 返回优化结果及相关信息
# 定义一个函数，使用内点法最小化一个线性目标函数，同时满足线性相等约束和非负约束。
def _linprog_ip(c, c0, A, b, callback, postsolve_args, maxiter=1000, tol=1e-8,
                disp=False, alpha0=.99995, beta=0.1, sparse=False, lstsq=False,
                sym_pos=True, cholesky=None, pc=True, ip=False,
                permc_spec='MMD_AT_PLUS_A', **unknown_options):
    r"""
    Minimize a linear objective function subject to linear
    equality and non-negativity constraints using the interior point method
    of [4]_. Linear programming is intended to solve problems
    of the following form:

    Minimize::

        c @ x

    Subject to::

        A @ x == b
            x >= 0

    User-facing documentation is in _linprog_doc.py.

    Parameters
    ----------
    c : 1-D array
        Coefficients of the linear objective function to be minimized.
    c0 : float
        Constant term in objective function due to fixed (and eliminated)
        variables. (Purely for display.)
    A : 2-D array
        2-D array such that ``A @ x``, gives the values of the equality
        constraints at ``x``.
    b : 1-D array
        1-D array of values representing the right hand side of each equality
        constraint (row) in ``A``.
    callback : callable, optional
        Callback function to be executed once per iteration.
    postsolve_args : tuple
        Data needed by _postsolve to convert the solution to the standard-form
        problem into the solution to the original problem.

    Options
    -------
    maxiter : int (default = 1000)
        The maximum number of iterations of the algorithm.
    tol : float (default = 1e-8)
        Termination tolerance to be used for all termination criteria;
        see [4]_ Section 4.5.
    disp : bool (default = False)
        Set to ``True`` if indicators of optimization status are to be printed
        to the console each iteration.
    alpha0 : float (default = 0.99995)
        The maximal step size for Mehrota's predictor-corrector search
        direction; see :math:`\beta_{3}` of [4]_ Table 8.1.
    beta : float (default = 0.1)
        The desired reduction of the path parameter :math:`\mu` (see [6]_)
        when Mehrota's predictor-corrector is not in use (uncommon).
    sparse : bool (default = False)
        Set to ``True`` if the problem is to be treated as sparse after
        presolve. If either ``A_eq`` or ``A_ub`` is a sparse matrix,
        this option will automatically be set ``True``, and the problem
        will be treated as sparse even during presolve. If your constraint
        matrices contain mostly zeros and the problem is not very small (less
        than about 100 constraints or variables), consider setting ``True``
        or providing ``A_eq`` and ``A_ub`` as sparse matrices.
    lstsq : bool (default = False)
        是否将问题设为非常糟糕条件下的情况。除非遇到严重的数值困难，否则应始终保持为``False``。除非收到建议更改的警告消息，否则应保持默认设置。

    sym_pos : bool (default = True)
        如果预计问题将产生良好条件的对称正定法方程矩阵（几乎总是如此），则保持为``True``。除非收到建议更改的警告消息，否则应保持默认设置。

    cholesky : bool (default = True)
        如果要通过显式的Cholesky分解，然后是显式的前向/后向替换来解决法方程，则设为``True``。对于数值行为良好的问题，这通常更快。

    pc : bool (default = True)
        如果要使用Mehrota的预测校正方法，则保持为``True``。这几乎总是有益的（如果不是总是有益的）。

    ip : bool (default = False)
        如果希望使用第[4]_节4.3中提到的改进的初始点建议，则设为``True``。这是否有益取决于问题的性质。

    permc_spec : str (default = 'MMD_AT_PLUS_A')
        （仅当``sparse = True``，``lstsq = False``，``sym_pos = True``，且没有SuiteSparse时有效。）
        算法的每次迭代中对矩阵进行因子分解。此选项指定如何对矩阵的列进行排列以保持稀疏性。可接受的值有：

        - ``NATURAL``：自然顺序。
        - ``MMD_ATA``：A^T A结构的最小度量序列。
        - ``MMD_AT_PLUS_A``：A^T+A结构的最小度量序列。
        - ``COLAMD``：近似最小度量列序列。

        此选项可能会影响内点算法的收敛性；测试不同的值以确定哪个对您的问题性能最佳。更多信息请参考``scipy.sparse.linalg.splu``。

    unknown_options : dict
        此解算器未使用的可选参数。如果`unknown_options`非空，则会发出警告，列出所有未使用的选项。

    Returns
    -------
    x : 1-D array
        解向量。

    status : int
        表示优化结束状态的整数：

         0 : 优化成功终止
         1 : 达到迭代限制
         2 : 问题似乎是不可行的
         3 : 问题似乎是无界的
         4 : 遇到严重的数值困难

    message : str
        优化结束状态的字符串描述。

    iteration : int
        解决问题所需的迭代次数。

    Notes
    -----
    此方法实现了[4]_中概述的算法，并结合了[8]_的思想。
    and a structure inspired by the simpler methods of [6]_.

# 将本段文本省略，不涉及具体代码功能。


    The primal-dual path following method begins with initial 'guesses' of
    the primal and dual variables of the standard form problem and iteratively
    attempts to solve the (nonlinear) Karush-Kuhn-Tucker conditions for the
    problem with a gradually reduced logarithmic barrier term added to the
    objective. This particular implementation uses a homogeneous self-dual
    formulation, which provides certificates of infeasibility or unboundedness
    where applicable.

# 实现基于原始-对偶路径跟随方法，从标准形式问题的原始和对偶变量的初始“猜测”开始，并迭代地尝试解决问题的（非线性）Karush-Kuhn-Tucker条件。逐渐减小添加到目标函数的对数障碍项。这个特定的实现使用均匀自对偶形式，可以提供不可行性或无界性的证明（在适用时）。


    The default initial point for the primal and dual variables is that
    defined in [4]_ Section 4.4 Equation 8.22. Optionally (by setting initial
    point option ``ip=True``), an alternate (potentially improved) starting
    point can be calculated according to the additional recommendations of
    [4]_ Section 4.4.

# 原始和对偶变量的默认初始点是在[4]_节4.4中定义的方程式8.22。可选地（通过设置初始点选项`ip=True`），可以根据第4节4.4的额外建议计算备用（潜在改进的）起始点。


    A search direction is calculated using the predictor-corrector method
    (single correction) proposed by Mehrota and detailed in [4]_ Section 4.1.
    (A potential improvement would be to implement the method of multiple
    corrections described in [4]_ Section 4.2.) In practice, this is
    accomplished by solving the normal equations, [4]_ Section 5.1 Equations
    8.31 and 8.32, derived from the Newton equations [4]_ Section 5 Equations
    8.25 (compare to [4]_ Section 4 Equations 8.6-8.8). The advantage of
    solving the normal equations rather than 8.25 directly is that the
    matrices involved are symmetric positive definite, so Cholesky
    decomposition can be used rather than the more expensive LU factorization.

# 使用Mehrota提出的预测-校正方法（单一校正）计算搜索方向，详细描述在[4]_节4.1中。（一个潜在的改进是实现第4节4.2描述的多个校正方法。）在实践中，这通过解决正规方程来实现，[4]_节5.1方程8.31和8.32，这些方程来自Newton方程[4]_节5方程8.25（与[4]_节4方程8.6-8.8相比）。与直接解8.25相比，解正规方程的优势在于所涉及的矩阵是对称正定的，因此可以使用Cholesky分解而不是更昂贵的LU分解。


    With default options, the solver used to perform the factorization depends
    on third-party software availability and the conditioning of the problem.

# 默认情况下，用于执行因式分解的求解器取决于第三方软件的可用性和问题的条件。


    For dense problems, solvers are tried in the following order:

    1. ``scipy.linalg.cho_factor``

    2. ``scipy.linalg.solve`` with option ``sym_pos=True``

    3. ``scipy.linalg.solve`` with option ``sym_pos=False``

    4. ``scipy.linalg.lstsq``

# 对于密集问题，求解器按以下顺序尝试：

# 1. 使用 ``scipy.linalg.cho_factor``
# 2. 使用选项 ``sym_pos=True`` 的 ``scipy.linalg.solve``
# 3. 使用选项 ``sym_pos=False`` 的 ``scipy.linalg.solve``
# 4. 使用 ``scipy.linalg.lstsq``


    For sparse problems:

    1. ``sksparse.cholmod.cholesky`` (if scikit-sparse and SuiteSparse are installed)

    2. ``scipy.sparse.linalg.factorized``
        (if scikit-umfpack and SuiteSparse are installed)

    3. ``scipy.sparse.linalg.splu`` (which uses SuperLU distributed with SciPy)

    4. ``scipy.sparse.linalg.lsqr``

# 对于稀疏问题：

# 1. 如果安装了scikit-sparse和SuiteSparse，则使用 ``sksparse.cholmod.cholesky``
# 2. 如果安装了scikit-umfpack和SuiteSparse，则使用 ``scipy.sparse.linalg.factorized``
# 3. 使用SciPy附带的SuperLU，即 ``scipy.sparse.linalg.splu``
# 4. 使用 ``scipy.sparse.linalg.lsqr``


    If the solver fails for any reason, successively more robust (but slower)
    solvers are attempted in the order indicated. Attempting, failing, and
    re-starting factorization can be time consuming, so if the problem is
    numerically challenging, options can be set to  bypass solvers that are
    failing. Setting ``cholesky=False`` skips to solver 2,
    ``sym_pos=False`` skips to solver 3, and ``lstsq=True`` skips
    to solver 4 for both sparse and dense problems.

# 如果由于任何原因求解器失败，将按照指定的顺序尝试更加健壮（但更慢）的求解器。尝试、失败和重新启动因式分解可能耗时较长，因此如果问题在数值上具有挑战性，则可以设置选项以跳过失败的求解器。设置 ``cholesky=False`` 将跳转到求解器2，``sym_pos=False`` 将跳转到求解器3，``lstsq=True`` 将跳转到求解器4，适用于稀疏和密集问题。


    Potential improvements for combatting issues associated with dense
    columns in otherwise sparse problems are outlined in [4]_ Section 5.3 and

# 关于如何解决在其他情况下稀疏问题中密集列相关问题的潜在改进方法在[4]_节5.3中概述。


    the required mathematics and methods.

.. [6] H. H. Bauschke, J. M. Borwein, and P. L. Combettes, ``Essential
    concepts of variational analysis,'' American Mathematical Society,
    2017, under contract.

# 引用[6]：H. H. Bauschke, J. M. Borwein, and P. L. Combettes，《变分分析的基本概念》，美国数学会，2017年，按合同约定。
    # Section 4.1-4.2; the latter discusses alleviating accuracy issues related to free variables.
    
    # 计算搜索方向后，计算不激活非负约束条件的最大步长，并应用其中较小的步长（如[4]_第4.1节所述）。
    # [4]_ 第4.3节建议改进选择步长的方法。
    
    # 根据[4]_ 第4.5节的终止条件测试新点。所有检查都使用相同的容差，可以通过“tol”选项设置。
    # （一个潜在的改进是将不同的容差独立设置。）如果检测到最优性、无界性或不可行性，则求解过程终止；否则重复。
    
    # 顶层“linprog”模块和特定方法的求解器之间期望的问题形式不同。
    # 特定方法的求解器期望标准形式问题：
    #
    # 最小化::
    #
    #     c @ x
    #
    # Subject to::
    #
    #     A @ x == b
    #         x >= 0
    #
    # 而顶层“linprog”模块期望的问题形式为：
    #
    # 最小化::
    #
    #     c @ x
    #
    # Subject to::
    #
    #     A_ub @ x <= b_ub
    #     A_eq @ x == b_eq
    #      lb <= x <= ub
    #
    # 其中“lb = 0”，“ub = None”，除非在“bounds”中设置。
    #
    # 原始问题包含等式、上界和变量约束条件，而特定方法的求解器要求等式约束和变量非负性。
    #
    # “linprog”模块通过将简单界限转换为上界约束，为不等式约束引入非负松弛变量，并将无界变量表达为两个非负变量之差来将原始问题转换为标准形式。
    
    # 参考文献
    # ----------
    # .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
    #        optimizer for linear programming: an implementation of the
    #        homogeneous algorithm." High performance optimization. Springer US,
    #        2000. 197-232.
    # .. [6] Freund, Robert M. "Primal-Dual Interior-Point Methods for Linear
    #        Programming based on Newton's Method." Unpublished Course Notes,
    #        March 2004. Available 2/25/2017 at
    #        https://ocw.mit.edu/courses/sloan-school-of-management/15-084j-nonlinear-programming-spring-2004/lecture-notes/lec14_int_pt_mthd.pdf
    # .. [8] Andersen, Erling D., and Knud D. Andersen. "Presolving in linear
    #        programming." Mathematical Programming 71.2 (1995): 221-245.
    # .. [9] Bertsimas, Dimitris, and J. Tsitsiklis. "Introduction to linear
    #        programming." Athena Scientific 1 (1997): 997.
    # .. [10] Andersen, Erling D., et al. Implementation of interior point methods
    #         for large scale linear programming. HEC/Universite de Geneve, 1996.
    # 检查未知选项，确保没有未定义的选项传递进来
    _check_unknown_options(unknown_options)

    # 下面的条件应该是警告而不是错误
    if (cholesky or cholesky is None) and sparse and not has_cholmod:
        if cholesky:
            # 如果 cholesky 为 True，但没有 scikit-sparse 支持，发出警告并设置 cholesky 为 False
            warn("Sparse cholesky is only available with scikit-sparse. "
                 "Setting `cholesky = False`",
                 OptimizeWarning, stacklevel=3)
        cholesky = False

    # 如果 sparse 为 True 并且 lstsq 为 True，发出不推荐的选项组合警告
    if sparse and lstsq:
        warn("Option combination 'sparse':True and 'lstsq':True "
             "is not recommended.",
             OptimizeWarning, stacklevel=3)

    # 如果 lstsq 为 True 并且 cholesky 为 True，发出无效选项组合警告
    if lstsq and cholesky:
        warn("Invalid option combination 'lstsq':True "
             "and 'cholesky':True; option 'cholesky' has no effect when "
             "'lstsq' is set True.",
             OptimizeWarning, stacklevel=3)

    # 定义有效的 permc_spec 规范
    valid_permc_spec = ('NATURAL', 'MMD_ATA', 'MMD_AT_PLUS_A', 'COLAMD')
    # 如果 permc_spec 不在有效规范列表中，发出警告并将其设置为默认值 'MMD_AT_PLUS_A'
    if permc_spec.upper() not in valid_permc_spec:
        warn("Invalid permc_spec option: '" + str(permc_spec) + "'. "
             "Acceptable values are 'NATURAL', 'MMD_ATA', 'MMD_AT_PLUS_A', "
             "and 'COLAMD'. Reverting to default.",
             OptimizeWarning, stacklevel=3)
        permc_spec = 'MMD_AT_PLUS_A'

    # 如果 sym_pos 为 False 并且 cholesky 为 True，引发值错误，因为 cholesky 分解仅适用于对称正定矩阵
    if not sym_pos and cholesky:
        raise ValueError(
            "Invalid option combination 'sym_pos':False "
            "and 'cholesky':True: Cholesky decomposition is only possible "
            "for symmetric positive definite matrices.")

    # 根据条件设置 cholesky 的最终值
    cholesky = cholesky or (cholesky is None and sym_pos and not lstsq)

    # 调用 _ip_hsd 函数执行实际的优化问题求解
    x, status, message, iteration = _ip_hsd(A, b, c, c0, alpha0, beta,
                                            maxiter, disp, tol, sparse,
                                            lstsq, sym_pos, cholesky,
                                            pc, ip, permc_spec, callback,
                                            postsolve_args)

    # 返回求解结果和相关信息
    return x, status, message, iteration
```