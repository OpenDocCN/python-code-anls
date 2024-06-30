# `D:\src\scipysrc\scipy\scipy\integrate\_bvp.py`

```
"""Boundary value problem solver."""
# 导入警告模块中的警告函数
from warnings import warn

# 导入 NumPy 库及其组件
import numpy as np
from numpy.linalg import pinv

# 导入 SciPy 中的稀疏矩阵及线性代数求解模块
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import splu

# 导入 SciPy 中的优化结果模块
from scipy.optimize import OptimizeResult

# 定义机器 epsilon
EPS = np.finfo(float).eps


def estimate_fun_jac(fun, x, y, p, f0=None):
    """Estimate derivatives of an ODE system rhs with forward differences.

    Returns
    -------
    df_dy : ndarray, shape (n, n, m)
        Derivatives with respect to y. An element (i, j, q) corresponds to
        d f_i(x_q, y_q) / d (y_q)_j.
    df_dp : ndarray with shape (n, k, m) or None
        Derivatives with respect to p. An element (i, j, q) corresponds to
        d f_i(x_q, y_q, p) / d p_j. If `p` is empty, None is returned.
    """
    # 获取 y 的维度 n 和 m
    n, m = y.shape
    # 如果没有提供初始函数值 f0，则计算初始值
    if f0 is None:
        f0 = fun(x, y, p)

    # 确定数据类型
    dtype = y.dtype

    # 初始化 df_dy 数组
    df_dy = np.empty((n, n, m), dtype=dtype)
    # 设置步长 h
    h = EPS**0.5 * (1 + np.abs(y))
    # 循环计算每个 df_dy[i]，即对 y[i] 的偏导数
    for i in range(n):
        y_new = y.copy()
        y_new[i] += h[i]
        hi = y_new[i] - y[i]
        f_new = fun(x, y_new, p)
        df_dy[:, i, :] = (f_new - f0) / hi

    # 计算 p 的维度 k
    k = p.shape[0]
    # 如果 k 为 0，表示没有参数 p，则 df_dp 为 None
    if k == 0:
        df_dp = None
    else:
        # 初始化 df_dp 数组
        df_dp = np.empty((n, k, m), dtype=dtype)
        # 设置步长 h
        h = EPS**0.5 * (1 + np.abs(p))
        # 循环计算每个 df_dp[i]，即对 p[i] 的偏导数
        for i in range(k):
            p_new = p.copy()
            p_new[i] += h[i]
            hi = p_new[i] - p[i]
            f_new = fun(x, y, p_new)
            df_dp[:, i, :] = (f_new - f0) / hi

    # 返回结果 df_dy 和 df_dp
    return df_dy, df_dp


def estimate_bc_jac(bc, ya, yb, p, bc0=None):
    """Estimate derivatives of boundary conditions with forward differences.

    Returns
    -------
    dbc_dya : ndarray, shape (n + k, n)
        Derivatives with respect to ya. An element (i, j) corresponds to
        d bc_i / d ya_j.
    dbc_dyb : ndarray, shape (n + k, n)
        Derivatives with respect to yb. An element (i, j) corresponds to
        d bc_i / d ya_j.
    dbc_dp : ndarray with shape (n + k, k) or None
        Derivatives with respect to p. An element (i, j) corresponds to
        d bc_i / d p_j. If `p` is empty, None is returned.
    """
    # 获取 ya 和 p 的维度 n 和 k
    n = ya.shape[0]
    k = p.shape[0]

    # 如果没有提供初始边界条件值 bc0，则计算初始值
    if bc0 is None:
        bc0 = bc(ya, yb, p)

    # 确定数据类型
    dtype = ya.dtype

    # 初始化 dbc_dya 数组
    dbc_dya = np.empty((n, n + k), dtype=dtype)
    # 设置步长 h
    h = EPS**0.5 * (1 + np.abs(ya))
    # 循环计算每个 dbc_dya[i]，即对 ya[i] 的偏导数
    for i in range(n):
        ya_new = ya.copy()
        ya_new[i] += h[i]
        hi = ya_new[i] - ya[i]
        bc_new = bc(ya_new, yb, p)
        dbc_dya[i] = (bc_new - bc0) / hi
    dbc_dya = dbc_dya.T

    # 设置步长 h
    h = EPS**0.5 * (1 + np.abs(yb))
    # 初始化 dbc_dyb 数组
    dbc_dyb = np.empty((n, n + k), dtype=dtype)
    # 循环计算每个 dbc_dyb[i]，即对 yb[i] 的偏导数
    for i in range(n):
        yb_new = yb.copy()
        yb_new[i] += h[i]
        hi = yb_new[i] - yb[i]
        bc_new = bc(ya, yb_new, p)
        dbc_dyb[i] = (bc_new - bc0) / hi
    dbc_dyb = dbc_dyb.T

    # 如果 k 为 0，表示没有参数 p，则 dbc_dp 为 None
    if k == 0:
        dbc_dp = None
    ```
    # 如果不是特殊情况，执行以下操作
    else:
        # 计算 EPS 的平方根乘以 (1 + p 的绝对值)
        h = EPS**0.5 * (1 + np.abs(p))
        # 创建一个空的二维数组，用于存储 dbc_dp 的结果，形状为 (k, n + k)
        dbc_dp = np.empty((k, n + k), dtype=dtype)
        # 对于每一个索引 i 在范围内进行迭代
        for i in range(k):
            # 复制 p 到 p_new
            p_new = p.copy()
            # 增加 h[i] 到 p_new[i]
            p_new[i] += h[i]
            # 计算 hi 作为 p_new[i] 和 p[i] 的差值
            hi = p_new[i] - p[i]
            # 使用 bc 函数计算新的边界条件 bc_new
            bc_new = bc(ya, yb, p_new)
            # 计算 dbc_dp[i] 作为 (bc_new - bc0) 除以 hi 的结果
            dbc_dp[i] = (bc_new - bc0) / hi
        # 转置 dbc_dp，使其形状变为 (n + k, k)
        dbc_dp = dbc_dp.T

    # 返回 dbc_dya, dbc_dyb, dbc_dp 作为函数结果
    return dbc_dya, dbc_dyb, dbc_dp
def compute_jac_indices(n, m, k):
    """Compute indices for the collocation system Jacobian construction.

    See `construct_global_jac` for the explanation.
    """
    # Compute indices for the Jacobian matrix of the collocation system
    # Columns indices for collocation residuals
    i_col = np.repeat(np.arange((m - 1) * n), n)
    # Rows indices for collocation residuals
    j_col = (np.tile(np.arange(n), n * (m - 1)) +
             np.repeat(np.arange(m - 1) * n, n**2))

    # Rows indices for boundary condition residuals
    i_bc = np.repeat(np.arange((m - 1) * n, m * n + k), n)
    # Columns indices for boundary condition residuals
    j_bc = np.tile(np.arange(n), n + k)

    # Rows indices for derivative of collocation residuals with respect to y
    i_p_col = np.repeat(np.arange((m - 1) * n), k)
    # Columns indices for derivative of collocation residuals with respect to p
    j_p_col = np.tile(np.arange(m * n, m * n + k), (m - 1) * n)

    # Rows indices for derivative of boundary conditions with respect to y
    i_p_bc = np.repeat(np.arange((m - 1) * n, m * n + k), k)
    # Columns indices for derivative of boundary conditions with respect to p
    j_p_bc = np.tile(np.arange(m * n, m * n + k), n + k)

    # Concatenate all row indices
    i = np.hstack((i_col, i_col, i_bc, i_bc, i_p_col, i_p_bc))
    # Concatenate all column indices
    j = np.hstack((j_col, j_col + n,
                   j_bc, j_bc + (m - 1) * n,
                   j_p_col, j_p_bc))

    return i, j



def stacked_matmul(a, b):
    """Stacked matrix multiply: out[i,:,:] = np.dot(a[i,:,:], b[i,:,:]).

    Empirical optimization. Use outer Python loop and BLAS for large
    matrices, otherwise use a single einsum call.
    """
    # If matrix a is large, use a loop-based multiplication for optimization
    if a.shape[1] > 50:
        out = np.empty((a.shape[0], a.shape[1], b.shape[2]))
        for i in range(a.shape[0]):
            out[i] = np.dot(a[i], b[i])
        return out
    else:
        # Otherwise, use einsum for matrix multiplication
        return np.einsum('...ij,...jk->...ik', a, b)



def construct_global_jac(n, m, k, i_jac, j_jac, h, df_dy, df_dy_middle, df_dp,
                         df_dp_middle, dbc_dya, dbc_dyb, dbc_dp):
    """Construct the Jacobian of the collocation system.

    There are n * m + k functions: m - 1 collocations residuals, each
    containing n components, followed by n + k boundary condition residuals.

    There are n * m + k variables: m vectors of y, each containing n
    components, followed by k values of vector p.

    For example, let m = 4, n = 2 and k = 1, then the Jacobian will have
    the following sparsity structure:

        1 1 2 2 0 0 0 0  5
        1 1 2 2 0 0 0 0  5
        0 0 1 1 2 2 0 0  5
        0 0 1 1 2 2 0 0  5
        0 0 0 0 1 1 2 2  5
        0 0 0 0 1 1 2 2  5

        3 3 0 0 0 0 4 4  6
        3 3 0 0 0 0 4 4  6
        3 3 0 0 0 0 4 4  6

    Zeros denote identically zero values, other values denote different kinds
    of blocks in the matrix (see below). The blank row indicates the separation
    of collocation residuals from boundary conditions. And the blank column
    indicates the separation of y values from p values.

    Refer to [1]_  (p. 306) for the formula of n x n blocks for derivatives
    of collocation residuals with respect to y.

    Parameters
    ----------
    n : int
        Number of equations in the ODE system.
    m : int
        Number of nodes in the mesh.
    k : int
        Number of the unknown parameters.
    i_jac : ndarray
        Row indices for the Jacobian matrix.
    j_jac : ndarray
        Column indices for the Jacobian matrix.
    h : float
        Step size for numerical differentiation.
    df_dy : callable
        Function to compute derivative of collocation residuals with respect to y.
    df_dy_middle : callable
        Function to compute derivative of boundary conditions with respect to y.
    df_dp : callable
        Function to compute derivative of collocation residuals with respect to p.
    df_dp_middle : callable
        Function to compute derivative of boundary conditions with respect to p.
    dbc_dya : callable
        Function to compute derivative of boundary conditions with respect to y at point a.
    dbc_dyb : callable
        Function to compute derivative of boundary conditions with respect to y at point b.
    dbc_dp : callable
        Function to compute derivative of boundary conditions with respect to p.
    """
    # Implementation of Jacobian construction based on the specified functions and indices
    pass
    # 将 df_dy 按照指定的轴顺序重新排列，使得维度顺序变为 (m, n, n)
    df_dy = np.transpose(df_dy, (2, 0, 1))
    # 将 df_dy_middle 按照指定的轴顺序重新排列，使得维度顺序变为 (m-1, n, n)
    df_dy_middle = np.transpose(df_dy_middle, (2, 0, 1))

    # 将 h 变为列向量和二维数组，并保留其他维度不变
    h = h[:, np.newaxis, np.newaxis]

    # 获取 df_dy 的数据类型
    dtype = df_dy.dtype

    # 计算对角线 n x n 块
    dPhi_dy_0 = np.empty((m - 1, n, n), dtype=dtype)
    dPhi_dy_0[:] = -np.identity(n)
    dPhi_dy_0 -= h / 6 * (df_dy[:-1] + 2 * df_dy_middle)
    T = stacked_matmul(df_dy_middle, df_dy[:-1])
    dPhi_dy_0 -= h**2 / 12 * T

    # 计算非对角线 n x n 块
    dPhi_dy_1 = np.empty((m - 1, n, n), dtype=dtype)
    dPhi_dy_1[:] = np.identity(n)
    dPhi_dy_1 -= h / 6 * (df_dy[1:] + 2 * df_dy_middle)
    T = stacked_matmul(df_dy_middle, df_dy[1:])
    dPhi_dy_1 += h**2 / 12 * T

    # 将所有计算得到的值按照一定的顺序堆叠成一个一维数组
    values = np.hstack((dPhi_dy_0.ravel(), dPhi_dy_1.ravel(), dbc_dya.ravel(),
                        dbc_dyb.ravel()))
    # 如果 k 大于 0，则执行以下操作
    if k > 0:
        # 将 df_dp 的维度重新排列为 (2, 0, 1)，主要是为了后续的计算方便
        df_dp = np.transpose(df_dp, (2, 0, 1))
        # 将 df_dp_middle 的维度重新排列为 (2, 0, 1)，同样是为了后续的计算方便
        df_dp_middle = np.transpose(df_dp_middle, (2, 0, 1))
        # 计算 T 矩阵，T = stacked_matmul(df_dy_middle, df_dp[:-1] - df_dp[1:])
        T = stacked_matmul(df_dy_middle, df_dp[:-1] - df_dp[1:])
        # 更新 df_dp_middle，加上 0.125 * h * T
        df_dp_middle += 0.125 * h * T
        # 计算 dPhi_dp，根据公式 dPhi_dp = -h/6 * (df_dp[:-1] + df_dp[1:] + 4 * df_dp_middle)
        dPhi_dp = -h/6 * (df_dp[:-1] + df_dp[1:] + 4 * df_dp_middle)
        # 将 dPhi_dp 的扁平化版本与 values 和 dbc_dp 的扁平化版本水平堆叠起来
        values = np.hstack((values, dPhi_dp.ravel(), dbc_dp.ravel()))

    # 创建一个稀疏矩阵 J，使用 values、i_jac 和 j_jac 来初始化
    J = coo_matrix((values, (i_jac, j_jac)))
    # 返回 J 的压缩稀疏列 (CSC) 表示
    return csc_matrix(J)
# 定义一个函数，用于求解某个节点处的共轭残差
def collocation_fun(fun, y, p, x, h):
    """Evaluate collocation residuals.

    This function lies in the core of the method. The solution is sought
    as a cubic C1 continuous spline with derivatives matching the ODE rhs
    at given nodes `x`. Collocation conditions are formed from the equality
    of the spline derivatives and rhs of the ODE system in the middle points
    between nodes.

    Such method is classified to Lobbato IIIA family in ODE literature.
    Refer to [1]_ for the formula and some discussion.

    Returns
    -------
    col_res : ndarray, shape (n, m - 1)
        Collocation residuals at the middle points of the mesh intervals.
    y_middle : ndarray, shape (n, m - 1)
        Values of the cubic spline evaluated at the middle points of the mesh
        intervals.
    f : ndarray, shape (n, m)
        RHS of the ODE system evaluated at the mesh nodes.
    f_middle : ndarray, shape (n, m - 1)
        RHS of the ODE system evaluated at the middle points of the mesh
        intervals (and using `y_middle`).

    References
    ----------
    .. [1] J. Kierzenka, L. F. Shampine, "A BVP Solver Based on Residual
           Control and the Maltab PSE", ACM Trans. Math. Softw., Vol. 27,
           Number 3, pp. 299-316, 2001.
    """
    # 计算 ODE 系统在节点 x 处的右手边函数值
    f = fun(x, y, p)
    # 计算中间点的 y 值，采用三次样条插值方法，同时考虑导数匹配和 ODE 右手边函数的影响
    y_middle = (0.5 * (y[:, 1:] + y[:, :-1]) -
                0.125 * h * (f[:, 1:] - f[:, :-1]))
    # 计算中间点处 ODE 系统的右手边函数值，使用 y_middle 进行计算
    f_middle = fun(x[:-1] + 0.5 * h, y_middle, p)
    # 计算共轭残差，即在网格间隔的中间点处的残差
    col_res = y[:, 1:] - y[:, :-1] - h / 6 * (f[:, :-1] + f[:, 1:] +
                                              4 * f_middle)

    return col_res, y_middle, f, f_middle


# 创建用于求解共轭系统的函数和雅可比矩阵
def prepare_sys(n, m, k, fun, bc, fun_jac, bc_jac, x, h):
    """Create the function and the Jacobian for the collocation system."""
    # 计算中间点的 x 坐标
    x_middle = x[:-1] + 0.5 * h
    # 计算雅可比矩阵的索引
    i_jac, j_jac = compute_jac_indices(n, m, k)

    # 定义共轭系统的函数 col_fun
    def col_fun(y, p):
        return collocation_fun(fun, y, p, x, h)

    # 定义系统雅可比矩阵的函数 sys_jac
    def sys_jac(y, p, y_middle, f, f_middle, bc0):
        # 估算函数 fun_jac 在当前 y 和 p 下的雅可比矩阵
        if fun_jac is None:
            df_dy, df_dp = estimate_fun_jac(fun, x, y, p, f)
            df_dy_middle, df_dp_middle = estimate_fun_jac(
                fun, x_middle, y_middle, p, f_middle)
        else:
            df_dy, df_dp = fun_jac(x, y, p)
            df_dy_middle, df_dp_middle = fun_jac(x_middle, y_middle, p)

        # 估算边界条件雅可比矩阵 bc_jac 在当前 y[:, 0]、y[:, -1] 和 p 下的值
        if bc_jac is None:
            dbc_dya, dbc_dyb, dbc_dp = estimate_bc_jac(bc, y[:, 0], y[:, -1],
                                                       p, bc0)
        else:
            dbc_dya, dbc_dyb, dbc_dp = bc_jac(y[:, 0], y[:, -1], p)

        # 构建全局雅可比矩阵
        return construct_global_jac(n, m, k, i_jac, j_jac, h, df_dy,
                                    df_dy_middle, df_dp, df_dp_middle, dbc_dya,
                                    dbc_dyb, dbc_dp)

    return col_fun, sys_jac


# 通过牛顿方法求解非线性共轭系统
def solve_newton(n, m, h, col_fun, bc, jac, y, p, B, bvp_tol, bc_tol):
    """Solve the nonlinear collocation system by a Newton method.

    This is a simple Newton method with a backtracking line search. As
    """
    # 简单的牛顿方法求解非线性共轭系统
    # 我们知道在网格中间点处的解残差与配点残差相关，即 r_middle = 1.5 * col_res / h。
    # 由于我们的边界值问题（BVP）求解器试图将相对残差降低到某个特定容差以下，因此通过比较
    # r_middle / (1 + np.abs(f_middle)) 和某个阈值来终止牛顿迭代是合理的。我们选择的阈值
    # 比 BVP 容差低 1.5 个数量级。我们重写条件为 col_res < tol_r * (1 + np.abs(f_middle))，
    # 那么 tol_r 应该如下计算：
    tol_r = 2/3 * h * 5e-2 * bvp_tol

    # 最大允许的雅可比矩阵评估和因式分解次数，换句话说，完整牛顿迭代的最大次数。文献中推荐使用
    # 较小的值。
    max_njev = 4
    # 最大迭代次数，考虑到一些迭代可以通过固定雅可比矩阵来执行。理论上，这些迭代是廉价的，
    # 但在 Python 中情况并非如此简单。
    max_iter = 8

    # 目标函数相对改善的最小值，用于接受步长（Armijo 常数）。
    sigma = 0.2

    # 用于回溯的步长减小因子。
    tau = 0.5

    # 最大回溯步数，最小步长为 tau ** n_trial。
    n_trial = 4

    # 计算列方程组的结果以及中间变量，同时计算边界条件的结果。
    col_res, y_middle, f, f_middle = col_fun(y, p)

    # 计算边界条件的结果。
    bc_res = bc(y[:, 0], y[:, -1], p)

    # 组合列方程组和边界条件的结果。
    res = np.hstack((col_res.ravel(order='F'), bc_res))

    # 雅可比矩阵计算的计数器。
    njev = 0

    # 是否出现奇异矩阵的标志。
    singular = False

    # 是否重新计算雅可比矩阵的标志。
    recompute_jac = True

    # 主迭代循环，迭代次数受 max_iter 控制。
    for iteration in range(max_iter):
        # 如果需要重新计算雅可比矩阵，则重新计算并尝试进行 LU 分解。
        if recompute_jac:
            J = jac(y, p, y_middle, f, f_middle, bc_res)
            njev += 1
            try:
                LU = splu(J)
            except RuntimeError:
                singular = True
                break

            # 解方程 LU * step = res。
            step = LU.solve(res)
            cost = np.dot(step, step)

        # 分离出更新的 y 和 p 的步长。
        y_step = step[:m * n].reshape((n, m), order='F')
        p_step = step[m * n:]

        # 初始化步长大小为 1。
        alpha = 1

        # 回溯迭代，尝试不同的步长。
        for trial in range(n_trial + 1):
            # 计算新的 y 和 p。
            y_new = y - alpha * y_step
            if B is not None:
                y_new[:, 0] = np.dot(B, y_new[:, 0])
            p_new = p - alpha * p_step

            # 计算列方程组的结果以及中间变量，同时计算边界条件的结果。
            col_res, y_middle, f, f_middle = col_fun(y_new, p_new)
            bc_res = bc(y_new[:, 0], y_new[:, -1], p_new)
            res = np.hstack((col_res.ravel(order='F'), bc_res))

            # 解方程 LU * step_new = res。
            step_new = LU.solve(res)
            cost_new = np.dot(step_new, step_new)

            # 根据 Armijo 准则接受步长。
            if cost_new < (1 - 2 * alpha * sigma) * cost:
                break

            # 如果未达到最大回溯步数，则减小步长。
            if trial < n_trial:
                alpha *= tau

        # 更新 y 和 p。
        y = y_new
        p = p_new

        # 如果达到最大允许的雅可比矩阵计算次数，则终止迭代。
        if njev == max_njev:
            break

        # 如果列方程组的残差和边界条件的残差满足收敛条件，则终止迭代。
        if (np.all(np.abs(col_res) < tol_r * (1 + np.abs(f_middle))) and
                np.all(np.abs(bc_res) < bc_tol)):
            break

        # 如果采用了完整步长，则继续使用相同的雅可比矩阵。
        if alpha == 1:
            step = step_new
            cost = cost_new
            recompute_jac = False
        else:
            recompute_jac = True

    # 返回计算结果 y, p，以及奇异矩阵标志。
    return y, p, singular
# 打印迭代头部信息，包括迭代次数、最大残差、最大边界条件残差、总节点数和添加的节点数
def print_iteration_header():
    print("{:^15}{:^15}{:^15}{:^15}{:^15}".format(
        "Iteration", "Max residual", "Max BC residual", "Total nodes",
        "Nodes added"))


# 打印迭代进展信息，包括迭代次数、残差、边界条件残差、总节点数和添加的节点数
def print_iteration_progress(iteration, residual, bc_residual, total_nodes,
                             nodes_added):
    print("{:^15}{:^15.2e}{:^15.2e}{:^15}{:^15}".format(
        iteration, residual, bc_residual, total_nodes, nodes_added))


# BVPResult 类，继承自 OptimizeResult
class BVPResult(OptimizeResult):
    pass


# 迭代终止消息字典，对应不同的终止状态码返回相应的消息
TERMINATION_MESSAGES = {
    0: "The algorithm converged to the desired accuracy.",
    1: "The maximum number of mesh nodes is exceeded.",
    2: "A singular Jacobian encountered when solving the collocation system.",
    3: "The solver was unable to satisfy boundary conditions tolerance on iteration 10."
}


def estimate_rms_residuals(fun, sol, x, h, p, r_middle, f_middle):
    """使用 Lobatto 定积分估计共轭梯度的均方根值。

    根据 Lobatto 四点法计算归一化的相对残差的均方根值。残差是指解的导数与ODE系统的右手边之间的差异。
    我们使用相对残差进行归一化，即除以1 + np.abs(f)。均方根值通过对每个区间上归一化的平方相对残差的积分的平方根来计算。
    积分使用5点 Lobatto 定积分 [1]_ 进行估计，这里假设在网格节点上残差为零。

    返回
    -------
    rms_res : ndarray, shape (m - 1,)
        每个区间上相对残差的均方根估计值。

    参考文献
    ----------
    .. [1] http://mathworld.wolfram.com/LobattoQuadrature.html
    .. [2] J. Kierzenka, L. F. Shampine, "A BVP Solver Based on Residual
       Control and the Maltab PSE", ACM Trans. Math. Softw., Vol. 27,
       Number 3, pp. 299-316, 2001.
    """
    x_middle = x[:-1] + 0.5 * h
    s = 0.5 * h * (3/7)**0.5
    x1 = x_middle + s
    x2 = x_middle - s
    y1 = sol(x1)
    y2 = sol(x2)
    y1_prime = sol(x1, 1)
    y2_prime = sol(x2, 1)
    f1 = fun(x1, y1, p)
    f2 = fun(x2, y2, p)
    r1 = y1_prime - f1
    r2 = y2_prime - f2

    r_middle /= 1 + np.abs(f_middle)
    r1 /= 1 + np.abs(f1)
    r2 /= 1 + np.abs(f2)

    r1 = np.sum(np.real(r1 * np.conj(r1)), axis=0)
    r2 = np.sum(np.real(r2 * np.conj(r2)), axis=0)
    r_middle = np.sum(np.real(r_middle * np.conj(r_middle)), axis=0)

    return (0.5 * (32 / 45 * r_middle + 49 / 90 * (r1 + r2))) ** 0.5


def create_spline(y, yp, x, h):
    """根据给定的值和导数创建三次样条插值。

    使用 interpolate.CubicSpline 中的公式计算系数。

    返回
    -------
    sol : PPoly
        作为 PPoly 实例的构造的样条插值。
    """
    from scipy.interpolate import PPoly

    n, m = y.shape
    # 创建一个形状为 (4, n, m-1) 的空数组 c，使用与 y 相同的数据类型
    c = np.empty((4, n, m - 1), dtype=y.dtype)
    # 计算斜率，即相邻元素之差除以步长 h
    slope = (y[:, 1:] - y[:, :-1]) / h
    # 计算二阶导数 t，使用 y' 的前后元素和斜率计算
    t = (yp[:, :-1] + yp[:, 1:] - 2 * slope) / h
    # 计算 c 的各个分量
    c[0] = t / h
    c[1] = (slope - yp[:, :-1]) / h - t
    c[2] = yp[:, :-1]
    c[3] = y[:, :-1]
    # 将 c 数组的第二个轴移动到第一个轴的位置
    c = np.moveaxis(c, 1, 0)

    # 使用 c 数组创建 PPoly 对象，指定 x 作为定义域，开启外推功能，以第二个轴作为主轴
    return PPoly(c, x, extrapolate=True, axis=1)
# 将节点插入到网格中的函数。

# Nodes removal logic is not established, its impact on the solver is
# presumably negligible. So, only insertion is done in this function.
# 网格中尚未建立节点移除逻辑，其对求解器的影响可能可以忽略不计。因此，此函数仅进行插入操作。

# Parameters
# ----------
# x : ndarray, shape (m,)
#     网格节点。
# insert_1 : ndarray
#     每个插入点1的间隔，用于在中间插入一个新节点。
# insert_2 : ndarray
#     每个插入点2的间隔，用于将一个间隔分为3等分，插入2个新节点。

# Returns
# -------
# x_new : ndarray
#     新的网格节点。

# Notes
# -----
# `insert_1` 和 `insert_2` 不应具有共同的值。
def modify_mesh(x, insert_1, insert_2):
    # 因为 np.insert 的实现显然随 NumPy 版本而异，我们使用了一个简单可靠的排序方法。
    return np.sort(np.hstack((
        x,
        0.5 * (x[insert_1] + x[insert_1 + 1]),
        (2 * x[insert_2] + x[insert_2 + 1]) / 3,
        (x[insert_2] + 2 * x[insert_2 + 1]) / 3
    )))


# 为求解器中的统一使用封装函数。
def wrap_functions(fun, bc, fun_jac, bc_jac, k, a, S, D, dtype):
    if fun_jac is None:
        fun_jac_wrapped = None

    if bc_jac is None:
        bc_jac_wrapped = None

    if k == 0:
        def fun_p(x, y, _):
            return np.asarray(fun(x, y), dtype)

        def bc_wrapped(ya, yb, _):
            return np.asarray(bc(ya, yb), dtype)

        if fun_jac is not None:
            def fun_jac_p(x, y, _):
                return np.asarray(fun_jac(x, y), dtype), None

        if bc_jac is not None:
            def bc_jac_wrapped(ya, yb, _):
                dbc_dya, dbc_dyb = bc_jac(ya, yb)
                return (np.asarray(dbc_dya, dtype),
                        np.asarray(dbc_dyb, dtype), None)
    else:
        def fun_p(x, y, p):
            return np.asarray(fun(x, y, p), dtype)

        def bc_wrapped(x, y, p):
            return np.asarray(bc(x, y, p), dtype)

        if fun_jac is not None:
            def fun_jac_p(x, y, p):
                df_dy, df_dp = fun_jac(x, y, p)
                return np.asarray(df_dy, dtype), np.asarray(df_dp, dtype)

        if bc_jac is not None:
            def bc_jac_wrapped(ya, yb, p):
                dbc_dya, dbc_dyb, dbc_dp = bc_jac(ya, yb, p)
                return (np.asarray(dbc_dya, dtype), np.asarray(dbc_dyb, dtype),
                        np.asarray(dbc_dp, dtype))

    if S is None:
        fun_wrapped = fun_p
    else:
        def fun_wrapped(x, y, p):
            f = fun_p(x, y, p)
            if x[0] == a:
                f[:, 0] = np.dot(D, f[:, 0])
                f[:, 1:] += np.dot(S, y[:, 1:]) / (x[1:] - a)
            else:
                f += np.dot(S, y) / (x - a)
            return f
    # 如果给定的 fun_jac 不是 None，则执行以下操作
    if fun_jac is not None:
        # 如果 S 是 None，则使用预定义的 fun_jac_p 作为包装后的函数
        if S is None:
            fun_jac_wrapped = fun_jac_p
        else:
            # 否则，将 S 在第三个维度扩展，以便与其他数组兼容
            Sr = S[:, :, np.newaxis]

            # 定义一个新的函数 fun_jac_wrapped，接受参数 x, y, p
            def fun_jac_wrapped(x, y, p):
                # 调用原始的 fun_jac_p 函数，获取 df_dy 和 df_dp
                df_dy, df_dp = fun_jac_p(x, y, p)
                # 如果 x 的第一个元素等于 a，则对 df_dy 的处理
                if x[0] == a:
                    # 将 D 与 df_dy 的第一个维度进行矩阵乘法运算
                    df_dy[:, :, 0] = np.dot(D, df_dy[:, :, 0])
                    # 对 df_dy 的其余部分添加 Sr 除以 (x[1:] - a) 的值
                    df_dy[:, :, 1:] += Sr / (x[1:] - a)
                else:
                    # 否则，将 Sr 除以 (x - a) 的值添加到 df_dy 中
                    df_dy += Sr / (x - a)

                # 返回处理后的 df_dy 和原始的 df_dp
                return df_dy, df_dp

    # 返回四个函数：fun_wrapped、bc_wrapped、fun_jac_wrapped、bc_jac_wrapped
    return fun_wrapped, bc_wrapped, fun_jac_wrapped, bc_jac_wrapped
# 解决一个边界值问题的函数，求解一个一阶ODE系统。
def solve_bvp(fun, bc, x, y, p=None, S=None, fun_jac=None, bc_jac=None,
              tol=1e-3, max_nodes=1000, verbose=0, bc_tol=None):
    """Solve a boundary value problem for a system of ODEs.

    This function numerically solves a first order system of ODEs subject to
    two-point boundary conditions::

        dy / dx = f(x, y, p) + S * y / (x - a), a <= x <= b
        bc(y(a), y(b), p) = 0

    Here x is a 1-D independent variable, y(x) is an N-D
    vector-valued function and p is a k-D vector of unknown
    parameters which is to be found along with y(x). For the problem to be
    determined, there must be n + k boundary conditions, i.e., bc must be an
    (n + k)-D function.

    The last singular term on the right-hand side of the system is optional.
    It is defined by an n-by-n matrix S, such that the solution must satisfy
    S y(a) = 0. This condition will be forced during iterations, so it must not
    contradict boundary conditions. See [2]_ for the explanation how this term
    is handled when solving BVPs numerically.

    Problems in a complex domain can be solved as well. In this case, y and p
    are considered to be complex, and f and bc are assumed to be complex-valued
    functions, but x stays real. Note that f and bc must be complex
    differentiable (satisfy Cauchy-Riemann equations [4]_), otherwise you
    should rewrite your problem for real and imaginary parts separately. To
    solve a problem in a complex domain, pass an initial guess for y with a
    complex data type (see below).

    Parameters
    ----------
    fun : callable
        Right-hand side of the system. The calling signature is ``fun(x, y)``,
        or ``fun(x, y, p)`` if parameters are present. All arguments are
        ndarray: ``x`` with shape (m,), ``y`` with shape (n, m), meaning that
        ``y[:, i]`` corresponds to ``x[i]``, and ``p`` with shape (k,). The
        return value must be an array with shape (n, m) and with the same
        layout as ``y``.
    bc : callable
        Function evaluating residuals of the boundary conditions. The calling
        signature is ``bc(ya, yb)``, or ``bc(ya, yb, p)`` if parameters are
        present. All arguments are ndarray: ``ya`` and ``yb`` with shape (n,),
        and ``p`` with shape (k,). The return value must be an array with
        shape (n + k,).
    x : array_like, shape (m,)
        Initial mesh. Must be a strictly increasing sequence of real numbers
        with ``x[0]=a`` and ``x[-1]=b``.
    y : array_like, shape (n, m)
        Initial guess for the function values at the mesh nodes, ith column
        corresponds to ``x[i]``. For problems in a complex domain pass `y`
        with a complex data type (even if the initial guess is purely real).
    p : array_like with shape (k,) or None, optional
        Initial guess for the unknown parameters. If None (default), it is
        assumed that the problem doesn't depend on any parameters.
    S : array_like with shape (n, n) or None, optional
        Matrix defining a singular term in the ODE system, forcing the
        solution to satisfy S y(a) = 0. This term is handled during iterations
        and should not contradict boundary conditions.
    fun_jac : callable or None, optional
        Jacobian of `fun` with respect to `y` and `p`. The calling signature
        is ``fun_jac(x, y)``, or ``fun_jac(x, y, p)`` if parameters are present.
        The return value must be an array with shape (n, n, m) if `p` is None,
        or (n, n, m, k) if `p` is not None.
    bc_jac : callable or None, optional
        Jacobian of `bc` with respect to `ya`, `yb`, and `p`. The calling
        signature is ``bc_jac(ya, yb)``, or ``bc_jac(ya, yb, p)`` if parameters
        are present. The return value must be an array with shape (n + k, n).
    tol : float, optional
        Tolerance for termination. Default is 1e-3.
    max_nodes : int, optional
        Maximum number of mesh nodes. Default is 1000.
    verbose : int, optional
        Level of algorithm's verbosity. Default is 0.
    bc_tol : float or None, optional
        Tolerance for boundary conditions. If None, the default is `tol`.

    Returns
    -------
    sol : Bunch object with the following fields:
        x : ndarray, shape (m,)
            The mesh nodes.
        y : ndarray, shape (n, m)
            The computed solution values at the mesh nodes.
        p : ndarray, shape (k,)
            The computed parameters.
        success : bool
            True if the solver succeeded in finding a solution.
        message : str
            Description of the cause of the termination.
        nodes : int
            Number of mesh nodes used in the solution.
    """
    S : array_like with shape (n, n) or None
        矩阵，定义了奇异项。如果为None（默认），则问题在没有奇异项的情况下解决。

    fun_jac : callable or None, optional
        计算函数f对y和p的导数的函数。调用签名为``fun_jac(x, y)``，或者如果存在参数则为``fun_jac(x, y, p)``。
        返回值必须按以下顺序包含1或2个元素：

            * df_dy : array_like with shape (n, n, m)，其中元素 (i, j, q) 等于 d f_i(x_q, y_q, p) / d (y_q)_j。
            * df_dp : array_like with shape (n, k, m)，其中元素 (i, j, q) 等于 d f_i(x_q, y_q, p) / d p_j。

        这里q表示定义x和y的节点数量，而i和j表示向量分量。如果问题在没有未知参数的情况下解决，应该不返回df_dp。

        如果`fun_jac`为None（默认），则将通过前向有限差分来估计导数。

    bc_jac : callable or None, optional
        计算边界条件bc对ya、yb和p的导数的函数。调用签名为``bc_jac(ya, yb)``，或者如果存在参数则为``bc_jac(ya, yb, p)``。
        返回值必须按以下顺序包含2或3个元素：

            * dbc_dya : array_like with shape (n, n)，其中元素 (i, j) 等于 d bc_i(ya, yb, p) / d ya_j。
            * dbc_dyb : array_like with shape (n, n)，其中元素 (i, j) 等于 d bc_i(ya, yb, p) / d yb_j。
            * dbc_dp : array_like with shape (n, k)，其中元素 (i, j) 等于 d bc_i(ya, yb, p) / d p_j。

        如果问题在没有未知参数的情况下解决，应该不返回dbc_dp。

        如果`bc_jac`为None（默认），则将通过前向有限差分来估计导数。

    tol : float, optional
        解的期望容差。如果定义``r = y' - f(x, y)``，其中y是找到的解，则求解器在每个网格间隔上尝试达到以下条件：
        ``norm(r / (1 + abs(f)) < tol``，这里的``norm``是在均方根意义上估算（使用数值积分公式）。默认值为1e-3。

    max_nodes : int, optional
        允许的最大网格节点数。如果超过这个数，算法将终止。默认值为1000。

    verbose : {0, 1, 2}, optional
        算法详细程度：

            * 0（默认）：静默工作。
            * 1：显示终止报告。
            * 2：在迭代过程中显示进度。

    bc_tol : float, optional
        边界条件残差的期望绝对容差：`bc`值应满足``abs(bc) < bc_tol``。逐分量计算。默认值与`tol`相等。允许最多10次迭代以达到此容差。

    Returns
    -------
    Bunch object with the following fields defined:
    # 定义一个包含以下字段的 Bunch 对象:

    sol : PPoly
        # 解的表示为 `scipy.interpolate.PPoly` 实例，即 C1 连续的三次样条插值。
    p : ndarray or None, shape (k,)
        # 找到的参数。如果问题中不存在参数，则为 None。
    x : ndarray, shape (m,)
        # 最终网格的节点。
    y : ndarray, shape (n, m)
        # 网格节点处的解值。
    yp : ndarray, shape (n, m)
        # 网格节点处的解的导数值。
    rms_residuals : ndarray, shape (m - 1,)
        # 每个网格间隔上相对残差的均方根值（参见 `tol` 参数的描述）。
    niter : int
        # 已完成的迭代次数。
    status : int
        # 算法终止的原因:

            * 0: 算法收敛到所需精度。
            * 1: 超过最大网格节点数。
            * 2: 解决协作系统时遇到奇异雅可比矩阵。

    message : string
        # 算法终止原因的文字描述。
    success : bool
        # 如果算法收敛到所需精度（`status=0`），则为 True。

    Notes
    -----
    # 这个函数实现了一个带有类似于 [1]_ 的残差控制和修正的四阶协作算法。
    # 利用一个带有仿射不变准则函数的阻尼牛顿法解决协作系统，如 [3]_ 中描述的那样。

    Note that in [1]_  integral residuals are defined without normalization
    by interval lengths. So, their definition is different by a multiplier of
    h**0.5 (h is an interval length) from the definition used here.
    # 注意，在 [1]_ 中，积分残差的定义不通过区间长度进行标准化。因此，它们的定义与此处使用的定义不同，差一个 h**0.5 的乘数（h 是区间长度）。

    .. versionadded:: 0.18.0
    # .. versionadded:: 0.18.0

    References
    ----------
    .. [1] J. Kierzenka, L. F. Shampine, "A BVP Solver Based on Residual
           Control and the Maltab PSE", ACM Trans. Math. Softw., Vol. 27,
           Number 3, pp. 299-316, 2001.
    # .. [1] J. Kierzenka, L. F. Shampine, "基于残差控制和 Maltab PSE 的 BVP 求解器"，ACM Trans. Math. Softw.，Vol. 27，Number 3，pp. 299-316，2001。

    .. [2] L.F. Shampine, P. H. Muir and H. Xu, "A User-Friendly Fortran BVP
           Solver".
    # .. [2] L.F. Shampine, P. H. Muir 和 H. Xu，"一个用户友好的 Fortran BVP 求解器"。

    .. [3] U. Ascher, R. Mattheij and R. Russell "Numerical Solution of
           Boundary Value Problems for Ordinary Differential Equations".
    # .. [3] U. Ascher, R. Mattheij 和 R. Russell，"常微分方程边界值问题的数值解法"。

    .. [4] `Cauchy-Riemann equations
            <https://en.wikipedia.org/wiki/Cauchy-Riemann_equations>`_ on
            Wikipedia.
    # .. [4] `Cauchy-Riemann 方程 <https://en.wikipedia.org/wiki/Cauchy-Riemann_equations>`_ 在维基百科上的解释。

    Examples
    --------
    # 在第一个示例中，我们解决 Bratu 问题::

        y'' + k * exp(y) = 0
        y(0) = y(1) = 0

    for k = 1.

    # 我们将方程重写为一个一阶系统，并实现其右侧的评估函数::

        y1' = y2
        y2' = -exp(y1)

    >>> import numpy as np
    >>> def fun(x, y):
    ...     return np.vstack((y[1], -np.exp(y[0])))

    Implement evaluation of the boundary condition residuals:

    >>> def bc(ya, yb):
    ...     return np.array([ya[0], yb[0]])

    Define the initial mesh with 5 nodes:

    >>> x = np.linspace(0, 1, 5)

    This problem is known to have two solutions. To obtain both of them, we
    # 将输入的 `x` 转换为浮点数的 NumPy 数组
    x = np.asarray(x, dtype=float)
    # 检查 `x` 是否为一维数组，如果不是则抛出数值错误
    if x.ndim != 1:
        raise ValueError("`x` must be 1 dimensional.")
    # 计算 `x` 中相邻元素的差异
    h = np.diff(x)
    # 如果有任何差异小于或等于零的情况，则抛出数值错误
    if np.any(h <= 0):
        raise ValueError("`x` must be strictly increasing.")
    # 获取数组 `x` 的第一个元素作为起始点 `a`
    a = x[0]

    # 将输入的 `y` 转换为 NumPy 数组
    y = np.asarray(y)
    # 如果 `y` 的数据类型是复数浮点型，则设置数据类型为复数，否则为浮点数
    if np.issubdtype(y.dtype, np.complexfloating):
        dtype = complex
    else:
        dtype = float
    # 强制转换 `y` 的数据类型为上述确定的类型，并且不进行拷贝操作
    y = y.astype(dtype, copy=False)

    # 检查 `y` 是否为二维数组，如果不是则抛出数值错误
    if y.ndim != 2:
        raise ValueError("`y` must be 2 dimensional.")
    # 检查 `y` 的列数是否与 `x` 的行数相同，若不同则引发数值错误异常
    if y.shape[1] != x.shape[0]:
        raise ValueError(f"`y` is expected to have {x.shape[0]} columns, but actually "
                         f"has {y.shape[1]}.")

    # 若 `p` 为 None，则设置为空数组；否则将其转换为指定数据类型的一维数组
    if p is None:
        p = np.array([])
    else:
        p = np.asarray(p, dtype=dtype)

    # 检查 `p` 是否为一维数组，若不是则引发数值错误异常
    if p.ndim != 1:
        raise ValueError("`p` must be 1 dimensional.")

    # 如果 `tol` 小于 100 倍的机器精度 `EPS`，则发出警告并设置 `tol` 为 100 倍的 `EPS`
    if tol < 100 * EPS:
        warn(f"`tol` is too low, setting to {100 * EPS:.2e}", stacklevel=2)
        tol = 100 * EPS

    # 检查 `verbose` 是否在 [0, 1, 2] 中，若不是则引发数值错误异常
    if verbose not in [0, 1, 2]:
        raise ValueError("`verbose` must be in [0, 1, 2].")

    # 获取 `y` 的行数 `n` 和 `p` 的元素个数 `k`
    n = y.shape[0]
    k = p.shape[0]

    # 如果给定了 `S`，则将其转换为指定数据类型的数组，并检查其形状是否为 (n, n)
    if S is not None:
        S = np.asarray(S, dtype=dtype)
        if S.shape != (n, n):
            raise ValueError(f"`S` is expected to have shape {(n, n)}, "
                             f"but actually has {S.shape}")

        # 计算矩阵 B = I - S^+ S，以施加必要的边界条件
        B = np.identity(n) - np.dot(pinv(S), S)

        # 对 y[:, 0] 应用变换 B
        y[:, 0] = np.dot(B, y[:, 0])

        # 计算矩阵 (I - S)^+
        D = pinv(np.identity(n) - S)
    else:
        B = None
        D = None

    # 如果未指定 bc_tol，则将其设为 tol 的值
    if bc_tol is None:
        bc_tol = tol

    # 最大迭代次数设为 10
    max_iteration = 10

    # 将函数和其雅可比矩阵包装起来，以便进行后续计算
    fun_wrapped, bc_wrapped, fun_jac_wrapped, bc_jac_wrapped = wrap_functions(
        fun, bc, fun_jac, bc_jac, k, a, S, D, dtype)

    # 计算函数值 f = fun(x, y, p)，并检查其形状是否与 y 相同
    f = fun_wrapped(x, y, p)
    if f.shape != y.shape:
        raise ValueError(f"`fun` return is expected to have shape {y.shape}, "
                         f"but actually has {f.shape}.")

    # 计算边界条件的残差 bc_res = bc(y[:, 0], y[:, -1], p)，并检查其形状是否为 (n + k,)
    bc_res = bc_wrapped(y[:, 0], y[:, -1], p)
    if bc_res.shape != (n + k,):
        raise ValueError(f"`bc` return is expected to have shape {(n + k,)}, "
                         f"but actually has {bc_res.shape}.")

    # 初始化状态和迭代次数
    status = 0
    iteration = 0

    # 如果 verbose 为 2，则打印迭代的头部信息
    if verbose == 2:
        print_iteration_header()
    # 进入无限循环，执行以下操作直到条件变化
    while True:
        # 获取当前解向量的长度
        m = x.shape[0]

        # 准备求解系统所需的函数和雅可比矩阵
        col_fun, jac_sys = prepare_sys(n, m, k, fun_wrapped, bc_wrapped,
                                       fun_jac_wrapped, bc_jac_wrapped, x, h)
        
        # 使用牛顿法求解非线性系统
        y, p, singular = solve_newton(n, m, h, col_fun, bc_wrapped, jac_sys,
                                      y, p, B, tol, bc_tol)
        
        # 迭代次数加一
        iteration += 1

        # 计算残差、中间变量以及相关函数
        col_res, y_middle, f, f_middle = collocation_fun(fun_wrapped, y,
                                                         p, x, h)
        
        # 计算边界条件的残差
        bc_res = bc_wrapped(y[:, 0], y[:, -1], p)
        max_bc_res = np.max(abs(bc_res))

        # 计算中间变量的残差
        r_middle = 1.5 * col_res / h
        
        # 根据解向量和相关函数创建样条插值
        sol = create_spline(y, f, x, h)
        
        # 评估均方根残差
        rms_res = estimate_rms_residuals(fun_wrapped, sol, x, h, p,
                                         r_middle, f_middle)
        max_rms_res = np.max(rms_res)

        # 如果雅可比矩阵奇异，设置状态并跳出循环
        if singular:
            status = 2
            break

        # 寻找需要添加的节点数
        insert_1, = np.nonzero((rms_res > tol) & (rms_res < 100 * tol))
        insert_2, = np.nonzero(rms_res >= 100 * tol)
        nodes_added = insert_1.shape[0] + 2 * insert_2.shape[0]

        # 如果添加节点后超过最大节点数，设置状态并跳出循环
        if m + nodes_added > max_nodes:
            status = 1
            if verbose == 2:
                nodes_added = f"({nodes_added})"
                print_iteration_progress(iteration, max_rms_res, max_bc_res,
                                         m, nodes_added)
            break

        # 如果需要详细输出信息，打印当前迭代的进展
        if verbose == 2:
            print_iteration_progress(iteration, max_rms_res, max_bc_res, m,
                                     nodes_added)

        # 如果添加了节点，修改网格并重新计算步长和解向量
        if nodes_added > 0:
            x = modify_mesh(x, insert_1, insert_2)
            h = np.diff(x)
            y = sol(x)
        # 如果边界条件残差满足要求，设置状态并跳出循环
        elif max_bc_res <= bc_tol:
            status = 0
            break
        # 如果达到最大迭代次数，设置状态并跳出循环
        elif iteration >= max_iteration:
            status = 3
            break

    # 如果需要输出详细信息，根据状态打印相应的解决方案信息
    if verbose > 0:
        if status == 0:
            print(f"Solved in {iteration} iterations, number of nodes {x.shape[0]}. \n"
                  f"Maximum relative residual: {max_rms_res:.2e} \n"
                  f"Maximum boundary residual: {max_bc_res:.2e}")
        elif status == 1:
            print(f"Number of nodes is exceeded after iteration {iteration}. \n"
                  f"Maximum relative residual: {max_rms_res:.2e} \n"
                  f"Maximum boundary residual: {max_bc_res:.2e}")
        elif status == 2:
            print("Singular Jacobian encountered when solving the collocation "
                  f"system on iteration {iteration}. \n"
                  f"Maximum relative residual: {max_rms_res:.2e} \n"
                  f"Maximum boundary residual: {max_bc_res:.2e}")
        elif status == 3:
            print("The solver was unable to satisfy boundary conditions "
                  f"tolerance on iteration {iteration}. \n"
                  f"Maximum relative residual: {max_rms_res:.2e} \n"
                  f"Maximum boundary residual: {max_bc_res:.2e}")
    # 如果 p 的大小为 0，则将其设置为 None
    if p.size == 0:
        p = None

    # 返回 BVPResult 对象，传递解 sol、参数 p、坐标 x、y，导数 yp，均方根残差 rms_residuals，
    # 迭代次数 niter，求解状态 status，状态信息 message，成功标志 success 是否等于 0
    return BVPResult(sol=sol, p=p, x=x, y=y, yp=f, rms_residuals=rms_res,
                     niter=iteration, status=status,
                     message=TERMINATION_MESSAGES[status], success=status == 0)
```