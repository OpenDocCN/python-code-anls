# `D:\src\scipysrc\scipy\scipy\optimize\_nnls.py`

```
import numpy as np  # 导入NumPy库，用于数值计算
from scipy.linalg import solve, LinAlgWarning  # 导入SciPy的线性代数模块中的solve函数和LinAlgWarning警告类
import warnings  # 导入警告模块，用于处理警告信息

__all__ = ['nnls']  # 定义模块的公开接口，仅包括nnls函数

def nnls(A, b, maxiter=None, *, atol=None):
    """
    Solve ``argmin_x || Ax - b ||_2`` for ``x>=0``.

    This problem, often called as NonNegative Least Squares, is a convex
    optimization problem with convex constraints. It typically arises when
    the ``x`` models quantities for which only nonnegative values are
    attainable; weight of ingredients, component costs and so on.

    Parameters
    ----------
    A : (m, n) ndarray
        Coefficient array
    b : (m,) ndarray, float
        Right-hand side vector.
    maxiter: int, optional
        Maximum number of iterations, optional. Default value is ``3 * n``.
    atol: float
        Tolerance value used in the algorithm to assess closeness to zero in
        the projected residual ``(A.T @ (A x - b)`` entries. Increasing this
        value relaxes the solution constraints. A typical relaxation value can
        be selected as ``max(m, n) * np.linalg.norm(a, 1) * np.spacing(1.)``.
        This value is not set as default since the norm operation becomes
        expensive for large problems hence can be used only when necessary.

    Returns
    -------
    x : ndarray
        Solution vector.
    rnorm : float
        The 2-norm of the residual, ``|| Ax-b ||_2``.

    See Also
    --------
    lsq_linear : Linear least squares with bounds on the variables

    Notes
    -----
    The code is based on [2]_ which is an improved version of the classical
    algorithm of [1]_. It utilizes an active set method and solves the KKT
    (Karush-Kuhn-Tucker) conditions for the non-negative least squares problem.

    References
    ----------
    .. [1] : Lawson C., Hanson R.J., "Solving Least Squares Problems", SIAM,
       1995, :doi:`10.1137/1.9781611971217`
    .. [2] : Bro, Rasmus and de Jong, Sijmen, "A Fast Non-Negativity-
       Constrained Least Squares Algorithm", Journal Of Chemometrics, 1997,
       :doi:`10.1002/(SICI)1099-128X(199709/10)11:5<393::AID-CEM483>3.0.CO;2-L`

     Examples
    --------
    >>> import numpy as np
    >>> from scipy.optimize import nnls
    ...
    >>> A = np.array([[1, 0], [1, 0], [0, 1]])
    >>> b = np.array([2, 1, 1])
    >>> nnls(A, b)
    (array([1.5, 1. ]), 0.7071067811865475)

    >>> b = np.array([-1, -1, -1])
    >>> nnls(A, b)
    (array([0., 0.]), 1.7320508075688772)

    """

    A = np.asarray_chkfinite(A)  # 将A转换为NumPy数组，并检查其中是否包含无穷大或NaN值
    b = np.asarray_chkfinite(b)  # 将b转换为NumPy数组，并检查其中是否包含无穷大或NaN值

    if len(A.shape) != 2:
        raise ValueError("Expected a two-dimensional array (matrix)" +
                         f", but the shape of A is {A.shape}")  # 如果A不是二维数组，抛出异常
    if len(b.shape) != 1:
        raise ValueError("Expected a one-dimensional array (vector)" +
                         f", but the shape of b is {b.shape}")  # 如果b不是一维数组，抛出异常

    m, n = A.shape  # 获取A的行数和列数
    # 如果矩阵 A 的行数 m 不等于向量 b 的长度，抛出数值错误异常
    if m != b.shape[0]:
        raise ValueError(
                "Incompatible dimensions. The first dimension of " +
                f"A is {m}, while the shape of b is {(b.shape[0], )}")

    # 调用 _nnls 函数进行非负最小二乘法求解
    # 返回值 x 是解向量，rnorm 是残差的范数，mode 是求解模式
    x, rnorm, mode = _nnls(A, b, maxiter, tol=atol)
    
    # 如果求解模式 mode 不等于 1，抛出运行时错误异常
    if mode != 1:
        raise RuntimeError("Maximum number of iterations reached.")

    # 返回解向量 x 和残差范数 rnorm
    return x, rnorm
# 定义一个非负最小二乘（NNLS）求解函数，用于解决稀疏线性方程组
def _nnls(A, b, maxiter=None, tol=None):
    """
    This is a single RHS algorithm from ref [2] above. For multiple RHS
    support, the algorithm is given in  :doi:`10.1002/cem.889`
    """

    # 获取矩阵 A 的行数和列数
    m, n = A.shape

    # 计算 A 的转置乘以 A 的结果
    AtA = A.T @ A
    # 计算 b 乘以 A 的结果，结果是一维数组，由 NumPy 自动确定类型
    Atb = b @ A  # Result is 1D - let NumPy figure it out

    # 如果未提供 maxiter 参数，则设置为 3*n
    if not maxiter:
        maxiter = 3*n
    # 如果未提供 tol 参数，则设置为 10 倍 m 和 n 中较大者乘以 np.spacing(1.)
    if tol is None:
        tol = 10 * max(m, n) * np.spacing(1.)

    # 初始化变量 x 和 s，均为浮点数类型的零数组
    x = np.zeros(n, dtype=np.float64)
    s = np.zeros(n, dtype=np.float64)

    # Inactive constraint switches，创建一个布尔类型的零数组 P
    P = np.zeros(n, dtype=bool)

    # 初始化投影残差 w，复制并转换为浮点数类型的 Atb
    w = Atb.copy().astype(np.float64)  # x=0. Skip (-AtA @ x) term

    # 初始化总迭代计数器 iter
    iter = 0

    # 主循环，直到所有约束都激活或者 w 中非 P 元素中的任一元素小于 tol
    while (not P.all()) and (w[~P] > tol).any():  # B
        # 找到 w * (~P) 中最大的索引 k，并将 P[k] 设置为 True
        k = np.argmax(w * (~P))  # B.2
        P[k] = True  # B.3

        # 迭代解
        s[:] = 0.
        # 使用 solve 函数解决 AtA[np.ix_(P, P)] @ x = Atb[P] 的问题
        # 忽略警告信息 'Ill-conditioned matrix'，此处 assume_a='sym'，check_finite=False
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Ill-conditioned matrix',
                                    category=LinAlgWarning)
            s[P] = solve(AtA[np.ix_(P, P)], Atb[P], assume_a='sym', check_finite=False)

        # 内部循环，直到 iter 达到 maxiter 或者 s[P] 中的最小值小于 0
        while (iter < maxiter) and (s[P].min() < 0):  # C.1
            iter += 1
            inds = P * (s < 0)
            # 计算 alpha，确保 s[inds] 中的值非零，防止除以零错误
            alpha = (x[inds] / (x[inds] - s[inds])).min()  # C.2
            x *= (1 - alpha)
            x += alpha*s
            # 如果 x 中的元素小于等于 tol，则将 P 中对应位置的元素设置为 False
            P[x <= tol] = False
            # 继续使用 solve 函数解决 AtA[np.ix_(P, P)] @ x = Atb[P] 的问题
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Ill-conditioned matrix',
                                        category=LinAlgWarning)
                s[P] = solve(AtA[np.ix_(P, P)], Atb[P], assume_a='sym',
                             check_finite=False)
            # 将 s 中非 P 元素置为零
            s[~P] = 0  # C.6

        # 更新 x 和 w
        x[:] = s[:]
        w[:] = Atb - AtA @ x

        # 如果 iter 达到 maxiter，则返回结果（此处为虚拟数值 0）
        if iter == maxiter:
            # 通常情况下应该返回 return x, np.linalg.norm(A@x - b), -1
            # 但是顶级上下文中，-1 可能会引发异常，因此返回虚拟数值 0
            return x, 0., -1

    # 如果未达到 maxiter 或者 w 中的所有元素都小于 tol，则返回结果
    return x, np.linalg.norm(A@x - b), 1
```