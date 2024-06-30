# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_isolve\lsqr.py`

```
# 导入必要的库和模块
import numpy as np
from math import sqrt
from scipy.sparse.linalg._interface import aslinearoperator
from scipy.sparse._sputils import convert_pydata_sparse_to_scipy

# 设置机器精度的全局变量
eps = np.finfo(np.float64).eps

# 定义稳定的双正交（Givens rotation）实现函数
def _sym_ortho(a, b):
    """
    稳定的 Givens 旋转实现。

    Notes
    -----
    这个函数 'SymOrtho' 是为了数值稳定性而添加的。建议来自 S.-C. Choi [1]_。
    它消除了一些重要位置上可能出现的 ``1/eps`` 的问题（例如 minres.py 中
    "Compute the next plane rotation Qk" 之后的文本）。

    References
    ----------

    [1] 参考文献等等
    """
    """
    根据 S.-C. Choi 的论文 [1] 中描述的方法，计算给定的两个数 a 和 b 的 SVD（奇异值分解）中的旋转参数。
    
    参数：
    a: 第一个输入数
    b: 第二个输入数
    
    返回值：
    c: SVD 中的旋转角度的余弦值
    s: SVD 中的旋转角度的正弦值
    r: SVD 中的缩放因子
    
    参考文献：
    [1] S.-C. Choi, "Iterative Methods for Singular Linear Equations
           and Least-Squares Problems", Dissertation,
           http://www.stanford.edu/group/SOL/dissertations/sou-cheng-choi-thesis.pdf
    """
    if b == 0:
        # 当 b 为零时，返回 a 的符号作为 c，0 作为 s，a 的绝对值作为 r
        return np.sign(a), 0, abs(a)
    elif a == 0:
        # 当 a 为零时，返回 0 作为 c，b 的符号作为 s，b 的绝对值作为 r
        return 0, np.sign(b), abs(b)
    elif abs(b) > abs(a):
        # 当 |b| > |a| 时，计算 tau = a / b，然后计算旋转角度的正弦和余弦值以及缩放因子
        tau = a / b
        s = np.sign(b) / sqrt(1 + tau * tau)
        c = s * tau
        r = b / s
    else:
        # 当 |a| >= |b| 时，计算 tau = b / a，然后计算旋转角度的正弦和余弦值以及缩放因子
        tau = b / a
        c = np.sign(a) / sqrt(1 + tau * tau)
        s = c * tau
        r = a / c
    return c, s, r
# 定义函数 lsqr，用于解决大型稀疏线性方程组的最小二乘解
def lsqr(A, b, damp=0.0, atol=1e-6, btol=1e-6, conlim=1e8,
         iter_lim=None, show=False, calc_var=False, x0=None):
    """Find the least-squares solution to a large, sparse, linear system
    of equations.

    The function solves ``Ax = b``  or  ``min ||Ax - b||^2`` or
    ``min ||Ax - b||^2 + d^2 ||x - x0||^2``.

    The matrix A may be square or rectangular (over-determined or
    under-determined), and may have any rank.

    ::

      1. Unsymmetric equations --    solve  Ax = b

      2. Linear least squares  --    solve  Ax = b
                                     in the least-squares sense

      3. Damped least squares  --    solve  (   A    )*x = (    b    )
                                            ( damp*I )     ( damp*x0 )
                                     in the least-squares sense

    Parameters
    ----------
    A : {sparse matrix, ndarray, LinearOperator}
        Representation of an m-by-n matrix.
        Alternatively, ``A`` can be a linear operator which can
        produce ``Ax`` and ``A^T x`` using, e.g.,
        ``scipy.sparse.linalg.LinearOperator``.
    b : array_like, shape (m,)
        Right-hand side vector ``b``.
    damp : float
        Damping coefficient. Default is 0.
    atol, btol : float, optional
        Stopping tolerances. `lsqr` continues iterations until a
        certain backward error estimate is smaller than some quantity
        depending on atol and btol.  Let ``r = b - Ax`` be the
        residual vector for the current approximate solution ``x``.
        If ``Ax = b`` seems to be consistent, `lsqr` terminates
        when ``norm(r) <= atol * norm(A) * norm(x) + btol * norm(b)``.
        Otherwise, `lsqr` terminates when ``norm(A^H r) <=
        atol * norm(A) * norm(r)``.  If both tolerances are 1.0e-6 (default),
        the final ``norm(r)`` should be accurate to about 6
        digits. (The final ``x`` will usually have fewer correct digits,
        depending on ``cond(A)`` and the size of LAMBDA.)  If `atol`
        or `btol` is None, a default value of 1.0e-6 will be used.
        Ideally, they should be estimates of the relative error in the
        entries of ``A`` and ``b`` respectively.  For example, if the entries
        of ``A`` have 7 correct digits, set ``atol = 1e-7``. This prevents
        the algorithm from doing unnecessary work beyond the
        uncertainty of the input data.
    conlim : float, optional
        Another stopping tolerance.  lsqr terminates if an estimate of
        ``cond(A)`` exceeds `conlim`.  For compatible systems ``Ax =
        b``, `conlim` could be as large as 1.0e+12 (say).  For
        least-squares problems, conlim should be less than 1.0e+8.
        Maximum precision can be obtained by setting ``atol = btol =
        conlim = zero``, but the number of iterations may then be
        excessive. Default is 1e8.
    iter_lim : int, optional
        Explicit limitation on number of iterations (for safety).
    show : bool, optional
        If True, display iteration status.
    calc_var : bool, optional
        If True, calculate variance.
    x0 : array_like, optional
        Initial guess.

    """
    # show : bool, optional
    # 是否显示迭代日志。默认为 False。
    calc_var : bool, optional
    # 是否计算 ``(A'A + damp^2*I)^{-1}`` 的对角线估计值。
    x0 : array_like, shape (n,), optional
    # 初始猜测的 x 值。如果为 None，则使用零向量。默认为 None。

        .. versionadded:: 1.0.0
    # 版本新增功能说明：1.0.0 版本新增了 x0 参数。

    Returns
    -------
    x : ndarray of float
    # 最终的解决方案。
    istop : int
    # 终止原因。
    # 1 表示 x 是方程 Ax = b 的近似解。
    # 2 表示 x 近似解决了最小二乘问题。
    itn : int
    # 迭代次数。
    r1norm : float
    # ``norm(r)``, 其中 ``r = b - Ax``。
    r2norm : float
    # ``sqrt( norm(r)^2  +  damp^2 * norm(x - x0)^2 )``.
    # 当 ``damp == 0`` 时等于 `r1norm`。
    anorm : float
    # ``Abar = [[A]; [damp*I]]`` 的 Frobenius 范数估计。
    acond : float
    # ``cond(Abar)`` 的估计值。
    arnorm : float
    # ``norm(A'@r - damp^2*(x - x0))`` 的估计值。
    xnorm : float
    # ``norm(x)``
    var : ndarray of float
    # 如果 ``calc_var`` 为 True，则估计所有 ``(A'A)^{-1}`` 的对角线（如果 ``damp == 0``）或更一般的 ``(A'A + damp^2*I)^{-1}``。
    # 如果 A 的列满秩或 ``damp > 0``，则这是定义良好的。（不确定在 ``rank(A) < n`` 和 ``damp = 0.`` 时 var 的含义。）

    Notes
    -----
    # LSQR 使用迭代方法来近似解决方案。
    # 达到特定精度所需的迭代次数强烈依赖于问题的缩放。
    # 因此应尽量避免 A 的行或列的不良缩放。

    # 例如，在问题 1 中，通过行缩放不会改变解决方案。
    # 如果 A 的某行与其他行相比非常小或大，那么应相应地调整 ( A  b ) 的相应行。

    # 在问题 1 和 2 中，通过列缩放很容易恢复解 x。
    # 除非有更好的信息，否则 A 的非零列应缩放，使它们的欧几里德范数相同（例如，1.0）。

    # 在问题 3 中，如果 damp 不为零，则无法重新缩放。
    # 但是，应在注意到 A 的缩放后才分配 damp 的值。

    # 参数 damp 旨在通过防止真解变得非常大来帮助正则化病态系统。
    # 另一个正则化的辅助参数是 acond，它可以用于在计算的解变得非常大之前终止迭代。

    # 如果知道某些初始估计 ``x0`` 并且 ``damp == 0``，则可以按以下步骤进行：

    #   1. 计算残差向量 ``r0 = b - A@x0``。
    #   2. 使用 LSQR 解系统 ``A@dx = r0``。
    #   3. 添加修正项 dx 以获得最终解 ``x = x0 + dx``。

    # 这要求在调用前后都可以使用 ``x0``。
    to LSQR.  To judge the benefits, suppose LSQR takes k1 iterations
    to solve A@x = b and k2 iterations to solve A@dx = r0.
    If x0 is "good", norm(r0) will be smaller than norm(b).
    If the same stopping tolerances atol and btol are used for each
    system, k1 and k2 will be similar, but the final solution x0 + dx
    should be more accurate.  The only way to reduce the total work
    is to use a larger stopping tolerance for the second system.
    If some value btol is suitable for A@x = b, the larger value
    btol*norm(b)/norm(r0)  should be suitable for A@dx = r0.

    Preconditioning is another way to reduce the number of iterations.
    If it is possible to solve a related system ``M@x = b``
    efficiently, where M approximates A in some helpful way (e.g. M -
    A has low rank or its elements are small relative to those of A),
    LSQR may converge more rapidly on the system ``A@M(inverse)@z =
    b``, after which x can be recovered by solving M@x = z.

    If A is symmetric, LSQR should not be used!

    Alternatives are the symmetric conjugate-gradient method (cg)
    and/or SYMMLQ.  SYMMLQ is an implementation of symmetric cg that
    applies to any symmetric A and will converge more rapidly than
    LSQR.  If A is positive definite, there are other implementations
    of symmetric cg that require slightly less work per iteration than
    SYMMLQ (but will take the same number of iterations).

    References
    ----------
    .. [1] C. C. Paige and M. A. Saunders (1982a).
           "LSQR: An algorithm for sparse linear equations and
           sparse least squares", ACM TOMS 8(1), 43-71.
    .. [2] C. C. Paige and M. A. Saunders (1982b).
           "Algorithm 583.  LSQR: Sparse linear equations and least
           squares problems", ACM TOMS 8(2), 195-209.
    .. [3] M. A. Saunders (1995).  "Solution of sparse rectangular
           systems using LSQR and CRAIG", BIT 35, 588-604.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csc_matrix
    >>> from scipy.sparse.linalg import lsqr
    >>> A = csc_matrix([[1., 0.], [1., 1.], [0., 1.]], dtype=float)

    The first example has the trivial solution ``[0, 0]``

    >>> b = np.array([0., 0., 0.], dtype=float)
    >>> x, istop, itn, normr = lsqr(A, b)[:4]
    >>> istop
    0
    >>> x
    array([ 0.,  0.])

    The stopping code ``istop=0`` returned indicates that a vector of zeros was
    found as a solution. The returned solution `x` indeed contains
    ``[0., 0.]``. The next example has a non-trivial solution:

    >>> b = np.array([1., 0., -1.], dtype=float)
    >>> x, istop, itn, r1norm = lsqr(A, b)[:4]
    >>> istop
    1
    >>> x
    array([ 1., -1.])
    >>> itn
    1
    >>> r1norm
    4.440892098500627e-16

    As indicated by ``istop=1``, `lsqr` found a solution obeying the tolerance
    limits. The given solution ``[1., -1.]`` obviously solves the equation. The
    remaining return values include information about the number of iterations
    (`itn=1`) and the remaining difference of left and right side of the solved
    equation.
    The final example demonstrates the behavior in the case where there is no
    solution for the equation:

    >>> b = np.array([1., 0.01, -1.], dtype=float)
    >>> x, istop, itn, r1norm = lsqr(A, b)[:4]
    >>> istop
    2
    >>> x
    array([ 1.00333333, -0.99666667])
    >>> A.dot(x)-b
    array([ 0.00333333, -0.00333333,  0.00333333])
    >>> r1norm
    0.005773502691896255

    `istop` indicates that the system is inconsistent and thus `x` is rather an
    approximate solution to the corresponding least-squares problem. `r1norm`
    contains the norm of the minimal residual that was found.
    """
    A = convert_pydata_sparse_to_scipy(A)
    A = aslinearoperator(A)
    b = np.atleast_1d(b)
    if b.ndim > 1:
        b = b.squeeze()

    m, n = A.shape
    if iter_lim is None:
        iter_lim = 2 * n
    var = np.zeros(n)

    msg = ('The exact solution is  x = 0                              ',
           'Ax - b is small enough, given atol, btol                  ',
           'The least-squares solution is good enough, given atol     ',
           'The estimate of cond(Abar) has exceeded conlim            ',
           'Ax - b is small enough for this machine                   ',
           'The least-squares solution is good enough for this machine',
           'Cond(Abar) seems to be too large for this machine         ',
           'The iteration limit has been reached                      ')

    if show:
        print(' ')
        print('LSQR            Least-squares solution of  Ax = b')
        str1 = f'The matrix A has {m} rows and {n} columns'
        str2 = f'damp = {damp:20.14e}   calc_var = {calc_var:8g}'
        str3 = f'atol = {atol:8.2e}                 conlim = {conlim:8.2e}'
        str4 = f'btol = {btol:8.2e}               iter_lim = {iter_lim:8g}'
        print(str1)
        print(str2)
        print(str3)
        print(str4)

    itn = 0
    istop = 0
    ctol = 0
    if conlim > 0:
        ctol = 1/conlim
    anorm = 0
    acond = 0
    dampsq = damp**2
    ddnorm = 0
    res2 = 0
    xnorm = 0
    xxnorm = 0
    z = 0
    cs2 = -1
    sn2 = 0

    # Set up the first vectors u and v for the bidiagonalization.
    # These satisfy  beta*u = b - A@x,  alfa*v = A'@u.
    # 初始化第一对向量 u 和 v 用于双对角化过程
    # 它们满足关系 beta*u = b - A@x,  alfa*v = A'@u.
    u = b
    bnorm = np.linalg.norm(b)

    if x0 is None:
        x = np.zeros(n)
        beta = bnorm.copy()
    else:
        x = np.asarray(x0)
        u = u - A.matvec(x)
        beta = np.linalg.norm(u)

    if beta > 0:
        u = (1/beta) * u
        v = A.rmatvec(u)
        alfa = np.linalg.norm(v)
    else:
        v = x.copy()
        alfa = 0

    if alfa > 0:
        v = (1/alfa) * v
    w = v.copy()

    rhobar = alfa
    phibar = beta
    rnorm = beta
    r1norm = rnorm
    r2norm = rnorm

    # Reverse the order here from the original matlab code because


这段代码的注释包括对变量初始化和向量设置的解释，确保描述了代码中每一步的操作和目的。
    # 计算 arnorm，即 alfa 和 beta 的乘积
    arnorm = alfa * beta
    # 如果 arnorm 为 0，则处理返回值，并显示错误信息（如果 show 为 True）
    if arnorm == 0:
        if show:
            print(msg[0])
        return x, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var

    # 定义表头信息用于显示
    head1 = '   Itn      x[0]       r1norm     r2norm '
    head2 = ' Compatible    LS      Norm A   Cond A'

    # 如果 show 为 True，则打印表头信息
    if show:
        print(' ')
        print(head1, head2)
        # 设置测试值
        test1 = 1
        test2 = alfa / beta
        # 格式化输出迭代次数、x[0]、r1norm、r2norm 和测试值
        str1 = f'{itn:6g} {x[0]:12.5e}'
        str2 = f' {r1norm:10.3e} {r2norm:10.3e}'
        str3 = f'  {test1:8.1e} {test2:8.1e}'
        print(str1, str2, str3)

    # 主迭代循环结束
    # 打印停止条件信息
    if show:
        print(' ')
        print('LSQR finished')
        print(msg[istop])
        print(' ')
        # 格式化输出停止条件相关信息
        str1 = f'istop ={istop:8g}   r1norm ={r1norm:8.1e}'
        str2 = f'anorm ={anorm:8.1e}   arnorm ={arnorm:8.1e}'
        str3 = f'itn   ={itn:8g}   r2norm ={r2norm:8.1e}'
        str4 = f'acond ={acond:8.1e}   xnorm  ={xnorm:8.1e}'
        print(str1 + '   ' + str2)
        print(str3 + '   ' + str4)
        print(' ')

    # 返回最终结果
    return x, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var
```