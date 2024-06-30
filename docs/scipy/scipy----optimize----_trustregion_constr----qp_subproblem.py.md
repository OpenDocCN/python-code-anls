# `D:\src\scipysrc\scipy\scipy\optimize\_trustregion_constr\qp_subproblem.py`

```
"""Equality-constrained quadratic programming solvers."""

# 导入必要的模块和函数
from scipy.sparse import (linalg, bmat, csc_matrix)
from math import copysign
import numpy as np
from numpy.linalg import norm

# 定义公开接口
__all__ = [
    'eqp_kktfact',
    'sphere_intersections',
    'box_intersections',
    'box_sphere_intersections',
    'inside_box_boundaries',
    'modified_dogleg',
    'projected_cg'
]

# 解决等式约束二次规划问题
def eqp_kktfact(H, c, A, b):
    """Solve equality-constrained quadratic programming (EQP) problem.

    Solve ``min 1/2 x.T H x + x.t c`` subject to ``A x + b = 0``
    using direct factorization of the KKT system.

    Parameters
    ----------
    H : sparse matrix, shape (n, n)
        Hessian matrix of the EQP problem.
    c : array_like, shape (n,)
        Gradient of the quadratic objective function.
    A : sparse matrix
        Jacobian matrix of the EQP problem.
    b : array_like, shape (m,)
        Right-hand side of the constraint equation.

    Returns
    -------
    x : array_like, shape (n,)
        Solution of the KKT problem.
    lagrange_multipliers : ndarray, shape (m,)
        Lagrange multipliers of the KKT problem.
    """
    n, = np.shape(c)  # 参数数量
    m, = np.shape(b)  # 约束数量

    # 构建 KKT 系统的系数矩阵
    kkt_matrix = csc_matrix(bmat([[H, A.T], [A, None]]))
    # 构建 KKT 系统的系数向量
    kkt_vec = np.hstack([-c, -b])

    # 使用对称不定因子分解解决线性系统
    lu = linalg.splu(kkt_matrix)
    kkt_sol = lu.solve(kkt_vec)
    x = kkt_sol[:n]  # 解向量 x
    lagrange_multipliers = -kkt_sol[n:n+m]  # 拉格朗日乘子

    return x, lagrange_multipliers


def sphere_intersections(z, d, trust_radius,
                         entire_line=False):
    """Find the intersection between segment (or line) and spherical constraints.

    Find the intersection between the segment (or line) defined by the
    parametric  equation ``x(t) = z + t*d`` and the ball
    ``||x|| <= trust_radius``.

    Parameters
    ----------
    z : array_like, shape (n,)
        Initial point.
    d : array_like, shape (n,)
        Direction.
    trust_radius : float
        Ball radius.
    entire_line : bool, optional
        When ``True``, the function returns the intersection between the line
        ``x(t) = z + t*d`` (``t`` can assume any value) and the ball
        ``||x|| <= trust_radius``. When ``False``, the function returns the intersection
        between the segment ``x(t) = z + t*d``, ``0 <= t <= 1``, and the ball.

    Returns
    -------
    ta, tb : float
        The line/segment ``x(t) = z + t*d`` is inside the ball for
        for ``ta <= t <= tb``.
    """
    # 实现了参数化直线与球的交点计算
    pass  # 这里的 pass 表示函数当前没有实现任何功能，仅作占位符使用
    intersect : bool
        当 ``True`` 时，表示线段与球体相交；当 ``False`` 时，表示线段与球体不相交。
    """
    # 当 d=0 时的特殊情况处理
    if norm(d) == 0:
        return 0, 0, False
    # 检查是否为无穷的信任半径
    if np.isinf(trust_radius):
        if entire_line:
            ta = -np.inf
            tb = np.inf
        else:
            ta = 0
            tb = 1
        intersect = True  # 线段一定与球体相交
        return ta, tb, intersect

    a = np.dot(d, d)
    b = 2 * np.dot(z, d)
    c = np.dot(z, z) - trust_radius**2
    discriminant = b*b - 4*a*c
    if discriminant < 0:
        intersect = False  # 无实数解，线段不与球体相交
        return 0, 0, intersect
    sqrt_discriminant = np.sqrt(discriminant)

    # 以下计算在数学上等价于:
    # ta = (-b - sqrt_discriminant) / (2*a)
    # tb = (-b + sqrt_discriminant) / (2*a)
    # 但会产生更小的舍入误差。
    # 参考 Matrix Computation p.97 进行更好的证明。
    aux = b + copysign(sqrt_discriminant, b)
    ta = -aux / (2*a)
    tb = -2*c / aux
    ta, tb = sorted([ta, tb])  # 对 ta 和 tb 进行排序

    if entire_line:
        intersect = True  # 线段一定与球体相交
    else:
        # 检查是否在向量长度范围内发生交点
        if tb < 0 or ta > 1:
            intersect = False  # 线段不与球体相交
            ta = 0
            tb = 0
        else:
            intersect = True  # 线段与球体相交
            # 限制交点区间在 0 到 1 之间
            ta = max(0, ta)
            tb = min(1, tb)

    return ta, tb, intersect
def box_intersections(z, d, lb, ub,
                      entire_line=False):
    """Find the intersection between segment (or line) and box constraints.

    Find the intersection between the segment (or line) defined by the
    parametric  equation ``x(t) = z + t*d`` and the rectangular box
    ``lb <= x <= ub``.

    Parameters
    ----------
    z : array_like, shape (n,)
        Initial point.
    d : array_like, shape (n,)
        Direction.
    lb : array_like, shape (n,)
        Lower bounds to each one of the components of ``x``. Used
        to delimit the rectangular box.
    ub : array_like, shape (n, )
        Upper bounds to each one of the components of ``x``. Used
        to delimit the rectangular box.
    entire_line : bool, optional
        When ``True``, the function returns the intersection between the line
        ``x(t) = z + t*d`` (``t`` can assume any value) and the rectangular
        box. When ``False``, the function returns the intersection between the segment
        ``x(t) = z + t*d``, ``0 <= t <= 1``, and the rectangular box.

    Returns
    -------
    ta, tb : float
        The line/segment ``x(t) = z + t*d`` is inside the box for
        for ``ta <= t <= tb``.
    intersect : bool
        When ``True``, there is a intersection between the line (or segment)
        and the rectangular box. On the other hand, when ``False``, there is no
        intersection.
    """
    # Make sure it is a numpy array
    z = np.asarray(z)  # 将初始点转换为 NumPy 数组
    d = np.asarray(d)  # 将方向转换为 NumPy 数组
    lb = np.asarray(lb)  # 将下界转换为 NumPy 数组
    ub = np.asarray(ub)  # 将上界转换为 NumPy 数组
    
    # Special case when d=0
    if norm(d) == 0:  # 如果方向向量 d 的范数为零，即 d 是零向量
        return 0, 0, False  # 返回默认的无交点情况

    # Get values for which d==0
    zero_d = (d == 0)  # 找到方向向量 d 中值为零的索引
    # If the boundaries are not satisfied for some coordinate
    # for which "d" is zero, there is no box-line intersection.
    if (z[zero_d] < lb[zero_d]).any() or (z[zero_d] > ub[zero_d]).any():
        intersect = False  # 如果 z 的某些坐标在对应的 lb 或 ub 范围之外，则无交点
        return 0, 0, intersect  # 返回无交点的情况
    
    # Remove values for which d is zero
    not_zero_d = np.logical_not(zero_d)  # 找到方向向量 d 中非零值的索引
    z = z[not_zero_d]  # 更新 z，去除零元素
    d = d[not_zero_d]  # 更新 d，去除零元素
    lb = lb[not_zero_d]  # 更新 lb，去除零元素
    ub = ub[not_zero_d]  # 更新 ub，去除零元素

    # Find a series of intervals (t_lb[i], t_ub[i]).
    t_lb = (lb-z) / d  # 计算每个维度上的下界交点
    t_ub = (ub-z) / d  # 计算每个维度上的上界交点
    # Get the intersection of all those intervals.
    ta = max(np.minimum(t_lb, t_ub))  # 计算最大的下界交点
    tb = min(np.maximum(t_lb, t_ub))  # 计算最小的上界交点

    # Check if intersection is feasible
    if ta <= tb:
        intersect = True  # 如果存在交点，则设置为 True
    else:
        intersect = False  # 如果不存在交点，则设置为 False
    
    # Restrict intersection to segment [0, 1] if not entire line
    if not entire_line:
        if tb < 0 or ta > 1:
            intersect = False  # 如果交点在 [0, 1] 之外，则无交点
            ta = 0  # 重置 ta
            tb = 0  # 重置 tb
        else:
            ta = max(0, ta)  # 确保 ta 在 [0, 1] 范围内
            tb = min(1, tb)  # 确保 tb 在 [0, 1] 范围内

    return ta, tb, intersect  # 返回计算得到的交点信息
    """Find the intersection between segment (or line) and box/sphere constraints.

    Find the intersection between the segment (or line) defined by the
    parametric  equation ``x(t) = z + t*d``, the rectangular box
    ``lb <= x <= ub`` and the ball ``||x|| <= trust_radius``.

    Parameters
    ----------
    z : array_like, shape (n,)
        Initial point.
    d : array_like, shape (n,)
        Direction.
    lb : array_like, shape (n,)
        Lower bounds to each one of the components of ``x``. Used
        to delimit the rectangular box.
    ub : array_like, shape (n, )
        Upper bounds to each one of the components of ``x``. Used
        to delimit the rectangular box.
    trust_radius : float
        Ball radius.
    entire_line : bool, optional
        When ``True``, the function returns the intersection between the line
        ``x(t) = z + t*d`` (``t`` can assume any value) and the constraints.
        When ``False``, the function returns the intersection between the segment
        ``x(t) = z + t*d``, ``0 <= t <= 1`` and the constraints.
    extra_info : bool, optional
        When ``True``, the function returns ``intersect_sphere`` and ``intersect_box``.

    Returns
    -------
    ta, tb : float
        The line/segment ``x(t) = z + t*d`` is inside the rectangular box and
        inside the ball for ``ta <= t <= tb``.
    intersect : bool
        When ``True``, there is a intersection between the line (or segment)
        and both constraints. On the other hand, when ``False``, there is no
        intersection.
    sphere_info : dict, optional
        Dictionary ``{ta, tb, intersect}`` containing the interval ``[ta, tb]``
        for which the line intercepts the ball. And a boolean value indicating
        whether the sphere is intersected by the line.
    box_info : dict, optional
        Dictionary ``{ta, tb, intersect}`` containing the interval ``[ta, tb]``
        for which the line intercepts the box. And a boolean value indicating
        whether the box is intersected by the line.
    """
    # Find intersections with the box constraints
    ta_b, tb_b, intersect_b = box_intersections(z, d, lb, ub,
                                                entire_line)
    
    # Find intersections with the sphere constraint
    ta_s, tb_s, intersect_s = sphere_intersections(z, d,
                                                   trust_radius,
                                                   entire_line)
    
    # Calculate the intersection intervals
    ta = np.maximum(ta_b, ta_s)
    tb = np.minimum(tb_b, tb_s)
    
    # Determine overall intersection based on both constraints
    if intersect_b and intersect_s and ta <= tb:
        intersect = True
    else:
        intersect = False

    # Optionally return detailed information about sphere and box intersections
    if extra_info:
        sphere_info = {'ta': ta_s, 'tb': tb_s, 'intersect': intersect_s}
        box_info = {'ta': ta_b, 'tb': tb_b, 'intersect': intersect_b}
        return ta, tb, intersect, sphere_info, box_info
    else:
        return ta, tb, intersect
# 计算 1/2*|| A x + b ||^2 的最小范数解。
newton_point = -Y.dot(b)

# 检查是否在指定箱体边界内的函数调用。
if inside_box_boundaries(newton_point, lb, ub)  \
   and norm(newton_point) <= trust_radius:
    # 如果在边界内且在信赖域半径内，则返回 newton_point 作为解。
    x = newton_point
    return x

# 计算梯度向量 ``g = A.T b``。
g = A.T.dot(b)

# 计算 Cauchy 点。
# `cauchy_point = g.T g / (g.T A.T A g)``。
A_g = A.dot(g)
cauchy_point = -np.dot(g, g) / np.dot(A_g, A_g) * g

# 原点
origin_point = np.zeros_like(cauchy_point)

# 检查 Cauchy 点和 Newton 点之间的线段是否有可能有解。
z = cauchy_point
p = newton_point - cauchy_point
_, alpha, intersect = box_sphere_intersections(z, p, lb, ub,
                                               trust_radius)
    # 如果存在交点（intersect为True），则使用 z + alpha*p 作为下一个解 x1
    if intersect:
        x1 = z + alpha*p
    else:
        # 如果不存在交点，则检查从原点到 cauchy_point 的线段
        z = origin_point
        p = cauchy_point
        # 调用 box_sphere_intersections 函数，返回相交点的信息
        _, alpha, _ = box_sphere_intersections(z, p, lb, ub,
                                               trust_radius)
        # 计算新的解 x1 = z + alpha*p
        x1 = z + alpha*p

    # 检查从原点到 newton_point 的线段，寻找可能的解 x2
    z = origin_point
    p = newton_point
    # 调用 box_sphere_intersections 函数，返回相交点的信息
    _, alpha, _ = box_sphere_intersections(z, p, lb, ub,
                                           trust_radius)
    # 计算新的解 x2 = z + alpha*p
    x2 = z + alpha*p

    # 返回在 x1 和 x2 中较优的解，基于 A.dot(x) + b 的范数
    if norm(A.dot(x1) + b) < norm(A.dot(x2) + b):
        return x1
    else:
        return x2
# 定义函数，使用投影共轭梯度法解决带有投影的等式约束二次规划问题
def projected_cg(H, c, Z, Y, b, trust_radius=np.inf,
                 lb=None, ub=None, tol=None,
                 max_iter=None, max_infeasible_iter=None,
                 return_all=False):
    """Solve EQP problem with projected CG method.

    Solve equality-constrained quadratic programming problem
    ``min 1/2 x.T H x + x.t c``  subject to ``A x + b = 0`` and,
    possibly, to trust region constraints ``||x|| < trust_radius``
    and box constraints ``lb <= x <= ub``.

    Parameters
    ----------
    H : LinearOperator (or sparse matrix or ndarray), shape (n, n)
        Operator for computing ``H v``.
    c : array_like, shape (n,)
        Gradient of the quadratic objective function.
    Z : LinearOperator (or sparse matrix or ndarray), shape (n, n)
        Operator for projecting ``x`` into the null space of A.
    Y : LinearOperator,  sparse matrix, ndarray, shape (n, m)
        Operator that, for a given a vector ``b``, compute smallest
        norm solution of ``A x + b = 0``.
    b : array_like, shape (m,)
        Right-hand side of the constraint equation.
    trust_radius : float, optional
        Trust radius to be considered. By default, uses ``trust_radius=inf``,
        which means no trust radius at all.
    lb : array_like, shape (n,), optional
        Lower bounds to each one of the components of ``x``.
        If ``lb[i] = -Inf`` the lower bound for the i-th
        component is just ignored (default).
    ub : array_like, shape (n, ), optional
        Upper bounds to each one of the components of ``x``.
        If ``ub[i] = Inf`` the upper bound for the i-th component
        is just ignored (default).
    tol : float, optional
        Tolerance used to interrupt the algorithm.
    max_iter : int, optional
        Maximum algorithm iterations. Where ``max_inter <= n-m``.
        By default, uses ``max_iter = n-m``.
    max_infeasible_iter : int, optional
        Maximum infeasible (regarding box constraints) iterations the
        algorithm is allowed to take.
        By default, uses ``max_infeasible_iter = n-m``.
    return_all : bool, optional
        When ``true``, return the list of all vectors through the iterations.

    Returns
    -------
    x : array_like, shape (n,)
        Solution of the EQP problem.
    info : Dict
        Dictionary containing the following:

            - niter : Number of iterations.
            - stop_cond : Reason for algorithm termination:
                1. Iteration limit was reached;
                2. Reached the trust-region boundary;
                3. Negative curvature detected;
                4. Tolerance was satisfied.
            - allvecs : List containing all intermediary vectors (optional).
            - hits_boundary : True if the proposed step is on the boundary
              of the trust region.

    Notes
    -----
    Implementation of Algorithm 6.2 on [1]_.

    In the absence of spherical and box constraints, for sufficient
    """
    # 极小值的数值近似阈值
    CLOSE_TO_ZERO = 1e-25

    # 获取矩阵 c 的形状，并得到参数个数 n
    n, = np.shape(c)  # 参数个数

    # 获取向量 b 的形状，并得到约束个数 m
    m, = np.shape(b)  # 约束个数

    # 初始化变量 x
    x = Y.dot(-b)
    
    # 计算 r、g 和 p 的初始值
    r = Z.dot(H.dot(x) + c)
    g = Z.dot(r)
    p = -g

    # 如果需要返回所有迭代结果，则初始化存储列表 allvecs
    if return_all:
        allvecs = [x]

    # 计算 H_p 和 rt_g 的值作为第一次迭代的参考
    H_p = H.dot(p)
    rt_g = norm(g)**2  # g.T g = r.T Z g = r.T g (ref [1]_ p.1389)

    # 如果 x 超出了信赖域半径，则抛出 ValueError
    tr_distance = trust_radius - norm(x)
    if tr_distance < 0:
        raise ValueError("Trust region problem does not have a solution.")
    # 如果 x 接近信赖域半径，则返回 x 作为优化问题的解
    elif tr_distance < CLOSE_TO_ZERO:
        info = {'niter': 0, 'stop_cond': 2, 'hits_boundary': True}
        if return_all:
            allvecs.append(x)
            info['allvecs'] = allvecs
        return x, info

    # 设置默认的容差 tol
    if tol is None:
        tol = max(min(0.01 * np.sqrt(rt_g), 0.1 * rt_g), CLOSE_TO_ZERO)

    # 设置默认的下界 lb 和上界 ub
    if lb is None:
        lb = np.full(n, -np.inf)
    if ub is None:
        ub = np.full(n, np.inf)

    # 设置最大迭代次数 max_iter
    if max_iter is None:
        max_iter = n - m
    max_iter = min(max_iter, n - m)

    # 设置最大非可行迭代次数 max_infeasible_iter
    if max_infeasible_iter is None:
        max_infeasible_iter = n - m

    # 初始化一些标志和计数器
    hits_boundary = False
    stop_cond = 1
    counter = 0
    last_feasible_x = np.zeros_like(x)
    k = 0

    # 如果当前 x 不在指定的箱式边界内，则将 x 重置为上次可行的 x
    if not inside_box_boundaries(x, lb, ub):
        x = last_feasible_x
        hits_boundary = True

    # 设置迭代信息字典 info
    info = {'niter': k, 'stop_cond': stop_cond,
            'hits_boundary': hits_boundary}
    if return_all:
        info['allvecs'] = allvecs

    # 返回最终优化结果 x 和迭代信息 info
    return x, info
```