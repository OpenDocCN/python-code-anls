# `D:\src\scipysrc\scipy\scipy\optimize\_lsq\trf.py`

```
"""Trust Region Reflective algorithm for least-squares optimization.

The algorithm is based on ideas from paper [STIR]_. The main idea is to
account for the presence of the bounds by appropriate scaling of the variables (or,
equivalently, changing a trust-region shape). Let's introduce a vector v:

           | ub[i] - x[i], if g[i] < 0 and ub[i] < np.inf
    v[i] = | x[i] - lb[i], if g[i] > 0 and lb[i] > -np.inf
           | 1,           otherwise

where g is the gradient of a cost function and lb, ub are the bounds. Its
components are distances to the bounds at which the anti-gradient points (if
this distance is finite). Define a scaling matrix D = diag(v**0.5).
First-order optimality conditions can be stated as

    D^2 g(x) = 0.

Meaning that components of the gradient should be zero for strictly interior
variables, and components must point inside the feasible region for variables
on the bound.

Now consider this system of equations as a new optimization problem. If the
point x is strictly interior (not on the bound), then the left-hand side is
differentiable and the Newton step for it satisfies

    (D^2 H + diag(g) Jv) p = -D^2 g

where H is the Hessian matrix (or its J^T J approximation in least squares),
Jv is the Jacobian matrix of v with components -1, 1 or 0, such that all
elements of matrix C = diag(g) Jv are non-negative. Introduce the change
of the variables x = D x_h (_h would be "hat" in LaTeX). In the new variables,
we have a Newton step satisfying

    B_h p_h = -g_h,

where B_h = D H D + C, g_h = D g. In least squares B_h = J_h^T J_h, where
J_h = J D. Note that J_h and g_h are proper Jacobian and gradient with respect
to "hat" variables. To guarantee global convergence we formulate a
trust-region problem based on the Newton step in the new variables:

    0.5 * p_h^T B_h p + g_h^T p_h -> min, ||p_h|| <= Delta

In the original space B = H + D^{-1} C D^{-1}, and the equivalent trust-region
problem is

    0.5 * p^T B p + g^T p -> min, ||D^{-1} p|| <= Delta

Here, the meaning of the matrix D becomes more clear: it alters the shape
of a trust-region, such that large steps towards the bounds are not allowed.
In the implementation, the trust-region problem is solved in "hat" space,
but handling of the bounds is done in the original space (see below and read
the code).

The introduction of the matrix D doesn't allow to ignore bounds, the algorithm
must keep iterates strictly feasible (to satisfy aforementioned
differentiability), the parameter theta controls step back from the boundary
(see the code for details).

The algorithm does another important trick. If the trust-region solution
doesn't fit into the bounds, then a reflected (from a firstly encountered
bound) search direction is considered. For motivation and analysis refer to
[STIR]_ paper (and other papers of the authors). In practice, it doesn't need
a lot of justifications, the algorithm simply chooses the best step among
"""
# 导入所需的库和模块
import numpy as np
from numpy.linalg import norm
from scipy.linalg import svd, qr
from scipy.sparse.linalg import lsmr
from scipy.optimize import OptimizeResult
# 导入自定义的common模块中的函数和类
from .common import (
    step_size_to_bound, find_active_constraints, in_bounds,
    make_strictly_feasible, intersect_trust_region, solve_lsq_trust_region,
    solve_trust_region_2d, minimize_quadratic_1d, build_quadratic_1d,
    evaluate_quadratic, right_multiplied_operator, regularized_lsq_operator,
    CL_scaling_vector, compute_grad, compute_jac_scale, check_termination,
    update_tr_radius, scale_for_robust_loss_function, print_header_nonlinear,
    print_iteration_nonlinear)

# 定义主函数trf，用于执行带边界的或无边界的Trust-Region算法
def trf(fun, jac, x0, f0, J0, lb, ub, ftol, xtol, gtol, max_nfev, x_scale,
        loss_function, tr_solver, tr_options, verbose):
    # 为了提高效率，当不需要施加边界时，运行简化版本的算法
    # 尽管违反了DRY原则，但为了保持函数的可读性，我们决定编写两个单独的函数。
    if np.all(lb == -np.inf) and np.all(ub == np.inf):
        # 调用无边界情况下的Trust-Region算法函数trf_no_bounds
        return trf_no_bounds(
            fun, jac, x0, f0, J0, ftol, xtol, gtol, max_nfev, x_scale,
            loss_function, tr_solver, tr_options, verbose)
    else:
        # 如果不满足任何优化器条件，返回通过 trf_bounds 函数计算的结果
        return trf_bounds(
            fun, jac, x0, f0, J0, lb, ub, ftol, xtol, gtol, max_nfev, x_scale,
            loss_function, tr_solver, tr_options, verbose)
# 处理步骤选择函数，根据信赖域反射算法选择最佳步长
def select_step(x, J_h, diag_h, g_h, p, p_h, d, Delta, lb, ub, theta):
    """Select the best step according to Trust Region Reflective algorithm."""
    
    # 检查如果沿着步长 p 移动仍在边界内，则评估二次函数
    if in_bounds(x + p, lb, ub):
        p_value = evaluate_quadratic(J_h, g_h, p_h, diag=diag_h)
        return p, p_h, -p_value
    
    # 计算到达边界的步长及其命中情况
    p_stride, hits = step_size_to_bound(x, p, lb, ub)
    
    # 计算反射方向
    r_h = np.copy(p_h)
    r_h[hits.astype(bool)] *= -1
    r = d * r_h
    
    # 限制信赖域步长，使其命中边界
    p *= p_stride
    p_h *= p_stride
    x_on_bound = x + p
    
    # 计算反射方向会首先穿过的可行区域或信赖域边界
    _, to_tr = intersect_trust_region(p_h, r_h, Delta)
    to_bound, _ = step_size_to_bound(x_on_bound, r, lb, ub)
    
    # 找到反射方向上步长的下界和上界，考虑严格的可行性要求
    # 对于测试问题，选择的方法似乎最佳
    r_stride = min(to_bound, to_tr)
    if r_stride > 0:
        r_stride_l = (1 - theta) * p_stride / r_stride
        if r_stride == to_bound:
            r_stride_u = theta * to_bound
        else:
            r_stride_u = to_tr
    else:
        r_stride_l = 0
        r_stride_u = -1
    
    # 检查反射步长是否可用
    if r_stride_l <= r_stride_u:
        a, b, c = build_quadratic_1d(J_h, g_h, r_h, s0=p_h, diag=diag_h)
        r_stride, r_value = minimize_quadratic_1d(
            a, b, r_stride_l, r_stride_u, c=c)
        r_h *= r_stride
        r_h += p_h
        r = r_h * d
    else:
        r_value = np.inf
    
    # 现在调整 p_h 以使其严格内部
    p *= theta
    p_h *= theta
    p_value = evaluate_quadratic(J_h, g_h, p_h, diag=diag_h)
    
    # 计算负梯度以判断沿反梯度方向的步长
    ag_h = -g_h
    ag = d * ag_h
    
    to_tr = Delta / norm(ag_h)
    to_bound, _ = step_size_to_bound(x, ag, lb, ub)
    if to_bound < to_tr:
        ag_stride = theta * to_bound
    else:
        ag_stride = to_tr
    
    # 构建一维二次函数并最小化以获得沿负梯度方向的步长
    a, b = build_quadratic_1d(J_h, g_h, ag_h, diag=diag_h)
    ag_stride, ag_value = minimize_quadratic_1d(a, b, 0, ag_stride)
    ag_h *= ag_stride
    ag *= ag_stride
    
    # 选择最优步长及其相应的值
    if p_value < r_value and p_value < ag_value:
        return p, p_h, -p_value
    elif r_value < p_value and r_value < ag_value:
        return r, r_h, -r_value
    else:
        return ag, ag_h, -ag_value


# 处理 Trust Region Reflective 方法中的边界情况
def trf_bounds(fun, jac, x0, f0, J0, lb, ub, ftol, xtol, gtol, max_nfev,
               x_scale, loss_function, tr_solver, tr_options, verbose):
    x = x0.copy()
    
    f = f0
    f_true = f.copy()
    nfev = 1
    
    J = J0
    njev = 1
    m, n = J.shape
    
    # 如果有损失函数，则进行损失函数的初始化和损失函数的缩放
    if loss_function is not None:
        rho = loss_function(f)
        cost = 0.5 * np.sum(rho[0])
        J, f = scale_for_robust_loss_function(J, f, rho)
    else:
        cost = 0.5 * np.dot(f, f)
    
    # 计算梯度并进行缩放
    g = compute_grad(J, f)
    
    # 根据 x_scale 是否为 'jac' 来确定是否进行 Jacobian 缩放
    jac_scale = isinstance(x_scale, str) and x_scale == 'jac'
    if jac_scale:
        # 如果jac_scale为真，则调用compute_jac_scale函数计算缩放系数
        scale, scale_inv = compute_jac_scale(J)
    else:
        # 如果jac_scale为假，则使用预定义的缩放系数x_scale和其倒数
        scale, scale_inv = x_scale, 1 / x_scale

    # 调用CL_scaling_vector函数计算缩放向量v和其非零部分的导数向量dv
    v, dv = CL_scaling_vector(x, g, lb, ub)
    # 对于dv中非零元素，使用对应的scale_inv进行缩放
    v[dv != 0] *= scale_inv[dv != 0]
    # 计算Delta作为初始步长的估计值，基于初始点x0的逆缩放和v的平方根倒数
    Delta = norm(x0 * scale_inv / v**0.5)
    if Delta == 0:
        Delta = 1.0  # 如果Delta计算结果为零，则设置为1.0

    # 计算g在v的缩放下的无穷范数，作为优化过程中的一个指标
    g_norm = norm(g * v, ord=np.inf)

    # 初始化扩展的目标函数数组，长度为m+n
    f_augmented = np.zeros(m + n)
    if tr_solver == 'exact':
        # 如果求解器选择为'exact'，则初始化J_augmented为m+n行n列的空数组
        J_augmented = np.empty((m + n, n))
    elif tr_solver == 'lsmr':
        # 如果求解器选择为'lsmr'，则初始化正则化项为0.0，并检查是否需要正则化
        reg_term = 0.0
        regularize = tr_options.pop('regularize', True)

    # 如果max_nfev为None，则设置其默认值为x0的长度乘以100
    if max_nfev is None:
        max_nfev = x0.size * 100

    alpha = 0.0  # 设置"Levenberg-Marquardt"参数为0.0

    termination_status = None  # 初始化终止状态为None
    iteration = 0  # 初始化迭代次数为0
    step_norm = None  # 初始化步长的范数为None
    actual_reduction = None  # 初始化实际减少量为None

    if verbose == 2:
        print_header_nonlinear()  # 如果verbose为2，则打印非线性优化的标题信息

    if termination_status is None:
        termination_status = 0  # 如果终止状态仍然是None，则将其设为0

    # 根据当前点x、下降函数值cost、真实目标函数值f_true、Jacobian矩阵J、梯度向量g、优化性能指标g_norm、
    # 活跃约束掩码active_mask、评估函数调用次数nfev、Jacobian评估次数njev、终止状态termination_status，
    # 返回一个OptimizeResult对象作为优化结果
    return OptimizeResult(
        x=x, cost=cost, fun=f_true, jac=J, grad=g, optimality=g_norm,
        active_mask=active_mask, nfev=nfev, njev=njev,
        status=termination_status)
def trf_no_bounds(fun, jac, x0, f0, J0, ftol, xtol, gtol, max_nfev,
                  x_scale, loss_function, tr_solver, tr_options, verbose):
    # 复制初始参数，以避免在函数内部修改原始数据
    x = x0.copy()

    # 初始化目标函数值
    f = f0
    # 复制目标函数值，以便记录真实的目标函数值
    f_true = f.copy()
    nfev = 1  # 函数计算次数

    # 初始化雅可比矩阵
    J = J0
    njev = 1  # 雅可比矩阵计算次数
    m, n = J.shape  # 获取雅可比矩阵的行数 m 和列数 n

    # 如果定义了损失函数，计算损失并调整雅可比矩阵和目标函数
    if loss_function is not None:
        rho = loss_function(f)
        cost = 0.5 * np.sum(rho[0])  # 计算损失函数的代价
        J, f = scale_for_robust_loss_function(J, f, rho)  # 调整雅可比矩阵和目标函数
    else:
        cost = 0.5 * np.dot(f, f)  # 计算无损失函数情况下的代价

    # 计算梯度
    g = compute_grad(J, f)

    # 判断是否使用雅可比矩阵缩放
    jac_scale = isinstance(x_scale, str) and x_scale == 'jac'
    if jac_scale:
        scale, scale_inv = compute_jac_scale(J)  # 计算雅可比矩阵的缩放因子
    else:
        scale, scale_inv = x_scale, 1 / x_scale  # 使用给定的缩放因子

    # 计算初始步长
    Delta = norm(x0 * scale_inv)
    if Delta == 0:
        Delta = 1.0

    # 如果选择的是'lsmr'求解器，进行相应的设置
    if tr_solver == 'lsmr':
        reg_term = 0  # 正则项初始化
        damp = tr_options.pop('damp', 0.0)  # 弹性系数的设定
        regularize = tr_options.pop('regularize', True)  # 是否正则化

    # 如果最大函数计算次数未指定，默认设定为参数向量大小的100倍
    if max_nfev is None:
        max_nfev = x0.size * 100

    alpha = 0.0  # "Levenberg-Marquardt" 参数

    termination_status = None  # 终止状态初始化
    iteration = 0  # 迭代次数初始化
    step_norm = None  # 步长范数初始化
    actual_reduction = None  # 实际减少量初始化

    # 如果 verbose 设为 2，打印非线性优化的头部信息
    if verbose == 2:
        print_header_nonlinear()

    # 如果终止状态未指定，设为初始状态
    if termination_status is None:
        termination_status = 0

    # 活跃掩码初始化为与参数向量 x 相同大小的零数组
    active_mask = np.zeros_like(x)

    # 返回优化结果对象
    return OptimizeResult(
        x=x, cost=cost, fun=f_true, jac=J, grad=g, optimality=g_norm,
        active_mask=active_mask, nfev=nfev, njev=njev,
        status=termination_status)
```