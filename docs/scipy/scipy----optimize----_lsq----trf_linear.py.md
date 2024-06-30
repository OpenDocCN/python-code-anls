# `D:\src\scipysrc\scipy\scipy\optimize\_lsq\trf_linear.py`

```
# 导入所需的库和模块
"""The adaptation of Trust Region Reflective algorithm for a linear
least-squares problem."""
import numpy as np  # 导入 NumPy 库，用于数值计算
from numpy.linalg import norm  # 导入 norm 函数，用于计算向量的范数
from scipy.linalg import qr, solve_triangular  # 导入 QR 分解和三角求解函数
from scipy.sparse.linalg import lsmr  # 导入最小二乘最小残差算法函数
from scipy.optimize import OptimizeResult  # 导入优化结果类

from .givens_elimination import givens_elimination  # 导入 Givens 消元函数
from .common import (  # 导入一系列辅助函数和常量
    EPS, step_size_to_bound, find_active_constraints, in_bounds,
    make_strictly_feasible, build_quadratic_1d, evaluate_quadratic,
    minimize_quadratic_1d, CL_scaling_vector, reflective_transformation,
    print_header_linear, print_iteration_linear, compute_grad,
    regularized_lsq_operator, right_multiplied_operator)


def regularized_lsq_with_qr(m, n, R, QTb, perm, diag, copy_R=True):
    """Solve regularized least squares using information from QR-decomposition.

    The initial problem is to solve the following system in a least-squares
    sense::

        A x = b
        D x = 0

    where D is diagonal matrix. The method is based on QR decomposition
    of the form A P = Q R, where P is a column permutation matrix, Q is an
    orthogonal matrix and R is an upper triangular matrix.

    Parameters
    ----------
    m, n : int
        Initial shape of A.
    R : ndarray, shape (n, n)
        Upper triangular matrix from QR decomposition of A.
    QTb : ndarray, shape (n,)
        First n components of Q^T b.
    perm : ndarray, shape (n,)
        Array defining column permutation of A, such that ith column of
        P is perm[i]-th column of identity matrix.
    diag : ndarray, shape (n,)
        Array containing diagonal elements of D.

    Returns
    -------
    x : ndarray, shape (n,)
        Found least-squares solution.
    """
    if copy_R:
        R = R.copy()  # 如果需要复制 R 矩阵，则进行复制操作
    v = QTb.copy()  # 复制 QTb 向量

    givens_elimination(R, v, diag[perm])  # 调用 Givens 消元函数进行消元操作

    abs_diag_R = np.abs(np.diag(R))
    threshold = EPS * max(m, n) * np.max(abs_diag_R)
    nns, = np.nonzero(abs_diag_R > threshold)

    R = R[np.ix_(nns, nns)]  # 根据非零元素的位置索引截取 R 矩阵的子矩阵
    v = v[nns]  # 截取 v 向量的子向量

    x = np.zeros(n)  # 初始化长度为 n 的零向量
    x[perm[nns]] = solve_triangular(R, v)  # 使用三角求解得到最终的最小二乘解

    return x  # 返回解向量 x


def backtracking(A, g, x, p, theta, p_dot_g, lb, ub):
    """Find an appropriate step size using backtracking line search."""
    alpha = 1  # 初始化步长参数 alpha 为 1
    while True:
        x_new, _ = reflective_transformation(x + alpha * p, lb, ub)  # 使用反射变换获取新的点 x_new
        step = x_new - x  # 计算步长
        cost_change = -evaluate_quadratic(A, g, step)  # 计算损失变化
        if cost_change > -0.1 * alpha * p_dot_g:  # 判断损失变化是否符合条件
            break
        alpha *= 0.5  # 缩小步长

    active = find_active_constraints(x_new, lb, ub)  # 查找活跃约束
    if np.any(active != 0):
        x_new, _ = reflective_transformation(x + theta * alpha * p, lb, ub)  # 使用调整步长后的反射变换获取新的点 x_new
        x_new = make_strictly_feasible(x_new, lb, ub, rstep=0)  # 使 x_new 成为严格可行点
        step = x_new - x  # 计算步长
        cost_change = -evaluate_quadratic(A, g, step)  # 计算损失变化

    return x, step, cost_change  # 返回原始点 x，步长和损失变化


def select_step(x, A_h, g_h, c_h, p, p_h, d, lb, ub, theta):
    """Select the best step according to Trust Region Reflective algorithm."""
    if in_bounds(x + p, lb, ub):  # 判断步长 p 是否保持在边界内
        return p  # 如果在边界内，则直接返回步长 p
    # 调用函数 `step_size_to_bound`，计算步长和命中情况
    p_stride, hits = step_size_to_bound(x, p, lb, ub)
    # 复制数组 `p_h` 到 `r_h`
    r_h = np.copy(p_h)
    # 将命中的部分取反
    r_h[hits.astype(bool)] *= -1
    # 计算最终步长向量 `r`
    r = d * r_h

    # 限制步长，使其在边界上
    p *= p_stride
    p_h *= p_stride
    # 计算在边界上的新位置 `x_on_bound`
    x_on_bound = x + p

    # 计算沿反射方向的步长
    r_stride_u, _ = step_size_to_bound(x_on_bound, r, lb, ub)

    # 保持在内部
    r_stride_l = (1 - theta) * r_stride_u
    r_stride_u *= theta

    # 如果步长大于零，则进行二次函数最小化
    if r_stride_u > 0:
        # 构建一维二次函数
        a, b, c = build_quadratic_1d(A_h, g_h, r_h, s0=p_h, diag=c_h)
        # 最小化二次函数
        r_stride, r_value = minimize_quadratic_1d(
            a, b, r_stride_l, r_stride_u, c=c)
        # 更新 `r_h` 为最小化后的步长
        r_h = p_h + r_h * r_stride
        # 更新 `r` 为最小化后的向量
        r = d * r_h
    else:
        # 否则，设置 `r_value` 为无穷大
        r_value = np.inf

    # 现在调整 `p_h` 以确保它严格在内部
    p_h *= theta
    p *= theta
    # 计算 `p_h` 的二次函数值
    p_value = evaluate_quadratic(A_h, g_h, p_h, diag=c_h)

    # 计算负梯度的步长向量
    ag_h = -g_h
    ag = d * ag_h
    ag_stride_u, _ = step_size_to_bound(x, ag, lb, ub)
    ag_stride_u *= theta
    # 构建负梯度的二次函数
    a, b = build_quadratic_1d(A_h, g_h, ag_h, diag=c_h)
    # 最小化负梯度的二次函数
    ag_stride, ag_value = minimize_quadratic_1d(a, b, 0, ag_stride_u)
    # 更新负梯度的向量 `ag`
    ag *= ag_stride

    # 根据二次函数的值决定返回哪个向量
    if p_value < r_value and p_value < ag_value:
        return p
    elif r_value < p_value and r_value < ag_value:
        return r
    else:
        return ag
# 计算矩阵 A 的行数和列数
m, n = A.shape
# 对初始解 x_lsq 进行反射变换，使其落在 lb 和 ub 的范围内，并返回变换后的解 x
x, _ = reflective_transformation(x_lsq, lb, ub)
# 对变换后的解 x 进行调整，确保其严格落在 lb 和 ub 的范围内，步长为 0.1
x = make_strictly_feasible(x, lb, ub, rstep=0.1)

# 根据求解器类型 lsq_solver 的选择，进行不同的最小二乘问题求解策略
if lsq_solver == 'exact':
    # 对矩阵 A 进行 QR 分解，经济模式，并启用枢轴选择
    QT, R, perm = qr(A, mode='economic', pivoting=True)
    QT = QT.T

    # 如果 m 小于 n，则在 R 矩阵末尾补充零，使其变为 n × n 维
    if m < n:
        R = np.vstack((R, np.zeros((n - m, n))))

    # 初始化 QTr 为 n 维零向量，k 为 m 和 n 中较小的值
    QTr = np.zeros(n)
    k = min(m, n)
elif lsq_solver == 'lsmr':
    # 初始化扩展的残差向量 r_aug 为 m+n 维零向量
    r_aug = np.zeros(m + n)
    auto_lsmr_tol = False
    # 如果 lsmr_tol 为 None，则设置为 1e-2 倍的 tol
    if lsmr_tol is None:
        lsmr_tol = 1e-2 * tol
    # 如果 lsmr_tol 为 'auto'，则自动计算 lsmr_tol
    elif lsmr_tol == 'auto':
        auto_lsmr_tol = True

# 计算残差 r = A*x - b
r = A.dot(x) - b
# 计算梯度 g = compute_grad(A, r)
g = compute_grad(A, r)
# 计算成本函数值 cost = 0.5 * r^T * r
cost = 0.5 * np.dot(r, r)
# 记录初始成本函数值
initial_cost = cost

# 初始化终止状态、步长范数和成本变化量
termination_status = None
step_norm = None
cost_change = None

# 如果 max_iter 为 None，则设定最大迭代次数为 100
if max_iter is None:
    max_iter = 100

# 如果 verbose 为 2，则打印线性问题的标题信息
if verbose == 2:
    print_header_linear()
    # 迭代优化过程，执行最大迭代次数内的循环
    for iteration in range(max_iter):
        # 使用 CL_scaling_vector 函数对变量 x 和梯度 g 进行缩放，确保在边界 lb 和 ub 内
        v, dv = CL_scaling_vector(x, g, lb, ub)
        # 缩放梯度 g
        g_scaled = g * v
        # 计算缩放后的梯度的无穷范数
        g_norm = norm(g_scaled, ord=np.inf)
        # 如果梯度的无穷范数小于设定的容差 tol，则设置终止状态为 1
        if g_norm < tol:
            termination_status = 1

        # 如果设置为详细输出模式 verbose == 2，则打印迭代信息
        if verbose == 2:
            print_iteration_linear(iteration, cost, cost_change,
                                   step_norm, g_norm)

        # 如果已经设置了终止状态，则跳出迭代循环
        if termination_status is not None:
            break

        # 计算对角线矩阵 h 的对角元素
        diag_h = g * dv
        # 计算 h 对角线元素的平方根
        diag_root_h = diag_h ** 0.5
        # 计算 v 的平方根
        d = v ** 0.5
        # 计算缩放后的梯度 g_h
        g_h = d * g

        # 对 A 应用右乘操作符 d，得到 A_h
        A_h = right_multiplied_operator(A, d)
        
        # 根据 lsq_solver 的选择进行线性最小二乘问题的求解
        if lsq_solver == 'exact':
            # 使用 QR 分解求解正则化最小二乘问题，返回修改后的方程
            QTr[:k] = QT.dot(r)
            p_h = -regularized_lsq_with_qr(m, n, R * d[perm], QTr, perm,
                                           diag_root_h, copy_R=False)
        elif lsq_solver == 'lsmr':
            # 使用 LSMR 方法求解正则化最小二乘问题
            lsmr_op = regularized_lsq_operator(A_h, diag_root_h)
            r_aug[:m] = r
            # 自动设置 LSMR 容差
            if auto_lsmr_tol:
                eta = 1e-2 * min(0.5, g_norm)
                lsmr_tol = max(EPS, min(0.1, eta * g_norm))
            p_h = -lsmr(lsmr_op, r_aug, maxiter=lsmr_maxiter,
                        atol=lsmr_tol, btol=lsmr_tol)[0]

        # 计算最终的搜索方向 p
        p = d * p_h

        # 计算 p 和 g 的点积
        p_dot_g = np.dot(p, g)
        # 如果 p 和 g 的点积大于 0，则设置终止状态为 -1
        if p_dot_g > 0:
            termination_status = -1

        # 计算步长选择的参数 theta
        theta = 1 - min(0.005, g_norm)
        # 使用 select_step 函数选择步长 step
        step = select_step(x, A_h, g_h, diag_h, p, p_h, d, lb, ub, theta)
        # 计算成本变化量，负数表示成功减少成本
        cost_change = -evaluate_quadratic(A, g, step)

        # 如果 cost_change 小于 0，则尝试使用回溯法调整 step
        if cost_change < 0:
            x, step, cost_change = backtracking(
                A, g, x, p, theta, p_dot_g, lb, ub)
        else:
            # 否则，确保新的 x 在边界 lb 和 ub 内
            x = make_strictly_feasible(x + step, lb, ub, rstep=0)

        # 计算步长的范数
        step_norm = norm(step)
        # 计算残差向量 r
        r = A.dot(x) - b
        # 计算新的梯度向量 g
        g = compute_grad(A, r)

        # 如果 cost_change 小于 cost 的容差 tol，设置终止状态为 2
        if cost_change < tol * cost:
            termination_status = 2

        # 更新当前的成本值
        cost = 0.5 * np.dot(r, r)

    # 如果没有设置终止状态，则默认设置为 0
    if termination_status is None:
        termination_status = 0

    # 查找活跃约束条件的掩码
    active_mask = find_active_constraints(x, lb, ub, rtol=tol)

    # 返回优化结果对象
    return OptimizeResult(
        x=x, fun=r, cost=cost, optimality=g_norm, active_mask=active_mask,
        nit=iteration + 1, status=termination_status,
        initial_cost=initial_cost)
```