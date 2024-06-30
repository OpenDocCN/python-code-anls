# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_isolve\lsmr.py`

```
# 定义模块的公开接口列表，只包含 lsmr 函数
__all__ = ['lsmr']

# 导入所需的库和模块
from numpy import zeros, inf, atleast_1d, result_type  # 导入 numpy 库中的函数和类型
from numpy.linalg import norm  # 导入 numpy.linalg 模块中的 norm 函数
from math import sqrt  # 导入 math 模块中的 sqrt 函数
from scipy.sparse.linalg._interface import aslinearoperator  # 导入 scipy.sparse.linalg._interface 模块中的 aslinearoperator 函数

# 导入 lsqr 模块中的 _sym_ortho 函数
from scipy.sparse.linalg._isolve.lsqr import _sym_ortho


def lsmr(A, b, damp=0.0, atol=1e-6, btol=1e-6, conlim=1e8,
         maxiter=None, show=False, x0=None):
    """Iterative solver for least-squares problems.

    lsmr solves the system of linear equations ``Ax = b``. If the system
    is inconsistent, it solves the least-squares problem ``min ||b - Ax||_2``.
    ``A`` is a rectangular matrix of dimension m-by-n, where all cases are
    allowed: m = n, m > n, or m < n. ``b`` is a vector of length m.
    The matrix A may be dense or sparse (usually sparse).

    Parameters
    ----------
    A : {sparse matrix, ndarray, LinearOperator}
        Matrix A in the linear system.
        Alternatively, ``A`` can be a linear operator which can
        produce ``Ax`` and ``A^H x`` using, e.g.,
        ``scipy.sparse.linalg.LinearOperator``.
    b : array_like, shape (m,)
        Vector ``b`` in the linear system.
    damp : float
        Damping factor for regularized least-squares. `lsmr` solves
        the regularized least-squares problem::

         min ||(b) - (  A   )x||
             ||(0)   (damp*I) ||_2

        where damp is a scalar.  If damp is None or 0, the system
        is solved without regularization. Default is 0.
    atol : float
        Absolute tolerance for the stopping criterion. Default is 1e-6.
    btol : float
        Relative tolerance for the stopping criterion. Default is 1e-6.
    conlim : float
        Relative limit for the condition number. Default is 1e8.
    maxiter : int or None
        Maximum number of iterations. If None, no limit is applied. Default is None.
    show : bool
        If True, display iteration number, norm of the residual, and condition number
        at each iteration. Default is False.
    x0 : array_like, shape (n,)
        Initial guess for the solution vector ``x``. Default is None.

    Returns
    -------
    x : ndarray of shape (n,)
        Solution to the linear system.

    Raises
    ------
    ValueError
        If an invalid parameter is passed.
    """
    # lsmr 函数实现的具体内容未在此处展示
    pass  # 占位符，表示函数体未实现，仅定义了函数签名和文档字符串
    # atol, btol 参数：浮点数，可选项
    # 停止容差。`lsmr` 在继续迭代，直到某个后向误差估计小于某个取决于 atol 和 btol 的量时停止。
    # 设 ``r = b - Ax`` 为当前近似解 ``x`` 的残差向量。
    # 如果 ``Ax = b`` 看起来是一致的，`lsmr` 将在 ``norm(r) <= atol * norm(A) * norm(x) + btol * norm(b)`` 时终止。
    # 否则，`lsmr` 将在 ``norm(A^H r) <= atol * norm(A) * norm(r)`` 时终止。
    # 如果两个容差都是 1.0e-6（默认值），最终的 ``norm(r)`` 应该准确到大约 6 位小数。
    # （最终的 ``x`` 通常会有更少的正确位数，这取决于 ``cond(A)`` 和 LAMBDA 的大小。）
    # 如果 `atol` 或 `btol` 是 None，则使用默认值 1.0e-6。
    # 理想情况下，它们应该是 ``A`` 和 ``b`` 条目中相对误差的估计值。
    # 例如，如果 ``A`` 的条目有 7 位正确数字，则设置 ``atol = 1e-7``。这可以防止算法在超出输入数据不确定性之外进行不必要的工作。
    conlim : float, optional
        # `lsmr` 将在估计的 ``cond(A)`` 超过 `conlim` 时终止。
        # 对于兼容系统 ``Ax = b``，conlim 可能会大到 1.0e+12（例如）。
        # 对于最小二乘问题，`conlim` 应该小于 1.0e+8。如果 `conlim` 是 None，则默认值为 1e+8。
        # 通过设置 ``atol = btol = conlim = 0`` 可以获得最大精度，但迭代次数可能会过多。默认值为 1e8。
    maxiter : int, optional
        # `lsmr` 将在达到 `maxiter` 次迭代时终止。
        # 默认为 ``maxiter = min(m, n)``。对于病态系统，可能需要更大的 `maxiter` 值。默认值为 False。
    show : bool, optional
        # 如果 ``show=True``，打印迭代日志。默认为 False。
    x0 : array_like, shape (n,), optional
        # ``x`` 的初始猜测值，如果为 None，则使用零向量。默认为 None。
        # .. versionadded:: 1.0.0
    # 返回
    # -------
    x : ndarray of float
        # 返回最小二乘解。
    # istop : int
    #   定义停止迭代的原因：
    #   istop = 0 表示 x=0 是一个解。如果提供了 x0，则 x=x0 是一个解。
    #   istop = 1 表示 x 是 A@x = B 的近似解，符合给定的 atol 和 btol。
    #   istop = 2 表示 x 大致解决了最小二乘问题，符合给定的 atol。
    #   istop = 3 表示 COND(A) 似乎大于 CONLIM。
    #   istop = 4 与 istop = 1 相同，但 atol = btol = eps（机器精度）。
    #   istop = 5 与 istop = 2 相同，但 atol = eps。
    #   istop = 6 与 istop = 3 相同，但 CONLIM = 1/eps。
    #   istop = 7 表示在其他停止条件满足之前，达到了最大迭代次数 maxiter。

    # itn : int
    #   使用的迭代次数。

    # normr : float
    #   norm(b - Ax)，即残差向量的范数。

    # normar : float
    #   norm(A^H (b - Ax))，A 的共轭转置乘以残差向量的范数。

    # norma : float
    #   A 的范数。

    # conda : float
    #   A 的条件数。

    # normx : float
    #   norm(x)，即解向量 x 的范数。

    Notes
    -----
    .. versionadded:: 0.11.0

    References
    ----------
    .. [1] D. C.-L. Fong and M. A. Saunders,
           "LSMR: An iterative algorithm for sparse least-squares problems",
           SIAM J. Sci. Comput., vol. 33, pp. 2950-2971, 2011.
           :arxiv:`1006.0758`
    .. [2] LSMR Software, https://web.stanford.edu/group/SOL/software/lsmr/

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csc_matrix
    >>> from scipy.sparse.linalg import lsmr
    >>> A = csc_matrix([[1., 0.], [1., 1.], [0., 1.]], dtype=float)

    第一个例子展示了平凡解 `[0, 0]`：

    >>> b = np.array([0., 0., 0.], dtype=float)
    >>> x, istop, itn, normr = lsmr(A, b)[:4]
    >>> istop
    0
    >>> x
    array([0., 0.])

    返回的停止代码 `istop=0` 表示找到了一个全零向量作为解。返回的解 `x` 确实包含 `[0., 0.]`。
    下一个例子展示了一个非平凡解：

    >>> b = np.array([1., 0., -1.], dtype=float)
    >>> x, istop, itn, normr = lsmr(A, b)[:4]
    >>> istop
    1
    >>> x
    array([ 1., -1.])
    >>> itn
    1
    >>> normr
    4.440892098500627e-16

    如 `istop=1` 所示，`lsmr` 找到了一个满足容差限制的解。返回的解 `[1., -1.]` 显然满足方程。
    其余返回值包含关于迭代次数 `itn=1` 和已解方程左右两侧残差的信息。
    最后一个例子演示了在没有解的情况下的行为：

    >>> b = np.array([1., 0.01, -1.], dtype=float)
    >>> x, istop, itn, normr = lsmr(A, b)[:4]
    >>> istop
    2
    >>> x
    array([ 1.00333333, -0.99666667])
    >>> A.dot(x)-b
    array([ 0.00333333, -0.00333333,  0.00333333])
    >>> normr
    """
    `istop` indicates the reason for termination:
        0 - Normal termination. `x` is a solution to the least-squares problem.
        1 - `x` is an approximate solution; system is inconsistent. `normr` shows minimal distance found.
    """

    # Convert A to a linear operator if it is not already
    A = aslinearoperator(A)
    # Ensure b is at least 1-dimensional
    b = atleast_1d(b)
    # If b has more than 1 dimension, squeeze it to a 1-dimensional array
    if b.ndim > 1:
        b = b.squeeze()

    # Messages for different termination conditions
    msg = ('The exact solution is x = 0, or x = x0, if x0 was given  ',
           'Ax - b is small enough, given atol, btol                  ',
           'The least-squares solution is good enough, given atol     ',
           'The estimate of cond(Abar) has exceeded conlim            ',
           'Ax - b is small enough for this machine                   ',
           'The least-squares solution is good enough for this machine',
           'Cond(Abar) seems to be too large for this machine         ',
           'The iteration limit has been reached                      ')

    # Headings for printed output
    hdg1 = '   itn      x(1)       norm r    norm Ar'
    hdg2 = ' compatible   LS      norm A   cond A'
    pfreq = 20   # Print frequency (for repeating the heading)
    pcount = 0   # Print counter

    # Dimensions of matrix A
    m, n = A.shape

    # Minimum dimension between m and n
    minDim = min([m, n])

    # If maxiter is not provided, set it to the minimum dimension
    if maxiter is None:
        maxiter = minDim

    # Determine dtype based on A, b, and optionally x0
    if x0 is None:
        dtype = result_type(A, b, float)
    else:
        dtype = result_type(A, b, x0, float)

    # Print information about the LSMR algorithm if show is True
    if show:
        print(' ')
        print('LSMR            Least-squares solution of  Ax = b\n')
        print(f'The matrix A has {m} rows and {n} columns')
        print('damp = %20.14e\n' % (damp))
        print(f'atol = {atol:8.2e}                 conlim = {conlim:8.2e}\n')
        print(f'btol = {btol:8.2e}             maxiter = {maxiter:8g}\n')

    # Initialize variables u, normb, x, beta, and possibly adjust u and beta if x0 is provided
    u = b
    normb = norm(b)
    if x0 is None:
        x = zeros(n, dtype)
        beta = normb.copy()
    else:
        x = atleast_1d(x0.copy())
        u = u - A.matvec(x)
        beta = norm(u)

    # Initialize variables v, alpha, and possibly adjust v if beta > 0
    if beta > 0:
        u = (1 / beta) * u
        v = A.rmatvec(u)
        alpha = norm(v)
    else:
        v = zeros(n, dtype)
        alpha = 0

    # Adjust v if alpha > 0
    if alpha > 0:
        v = (1 / alpha) * v

    # Initialize variables for the main iteration loop
    itn = 0
    zetabar = alpha * beta
    alphabar = alpha
    rho = 1
    rhobar = 1
    cbar = 1
    sbar = 0

    h = v.copy()
    hbar = zeros(n, dtype)

    # Initialize variables for estimating ||r||
    betadd = beta
    betad = 0
    rhodold = 1
    tautildeold = 0
    thetatilde = 0
    zeta = 0
    d = 0

    # Initialize variables for estimating ||A|| and cond(A)
    normA2 = alpha * alpha
    maxrbar = 0
    minrbar = 1e+100
    normA = sqrt(normA2)
    condA = 1
    normx = 0

    # Items for use in stopping rules, normb set earlier
    istop = 0
    ctol = 0
    if conlim > 0:
        ctol = 1 / conlim
    normr = beta

    # Compute norm(A)*norm(x) for use in stopping criteria
    normar = alpha * beta
    # 如果 normar 等于 0，则判断为零，执行以下操作
    if normar == 0:
        # 如果 show 为真，则打印消息列表中的第一个消息
        if show:
            print(msg[0])
        # 返回当前迭代的结果及统计数据
        return x, istop, itn, normr, normar, normA, condA, normx

    # 如果 normb 等于 0，则将 x 的内容设置为零向量
    if normb == 0:
        x[()] = 0
        # 返回当前迭代的结果及统计数据
        return x, istop, itn, normr, normar, normA, condA, normx

    # 如果 show 为真，则打印迭代过程中的相关信息
    if show:
        print(' ')
        print(hdg1, hdg2)
        # 设定测试值 test1 和 test2，用于迭代过程中的输出
        test1 = 1
        test2 = alpha / beta
        # 格式化字符串，显示迭代次数、x 的第一个元素、残差的规范、以及 test1 和 test2 的值
        str1 = f'{itn:6g} {x[0]:12.5e}'
        str2 = f' {normr:10.3e} {normar:10.3e}'
        str3 = f'  {test1:8.1e} {test2:8.1e}'
        print(''.join([str1, str2, str3]))

    # 主迭代循环。
    # 打印停止条件。
    if show:
        print(' ')
        print('LSMR finished')
        # 打印迭代结束后的消息、停止原因及相应的统计数据
        print(msg[istop])
        print(f'istop ={istop:8g}    normr ={normr:8.1e}')
        print(f'    normA ={normA:8.1e}    normAr ={normar:8.1e}')
        print(f'itn   ={itn:8g}    condA ={condA:8.1e}')
        print('    normx =%8.1e' % (normx))
        print(str1, str2)
        print(str3, str4)

    # 返回当前迭代的结果及统计数据
    return x, istop, itn, normr, normar, normA, condA, normx
```