# `D:\src\scipysrc\scipy\scipy\optimize\_lsq\bvls.py`

```
"""Bounded-variable least-squares algorithm."""
import numpy as np
from numpy.linalg import norm, lstsq
from scipy.optimize import OptimizeResult

from .common import print_header_linear, print_iteration_linear


def compute_kkt_optimality(g, on_bound):
    """Compute the maximum violation of KKT conditions."""
    g_kkt = g * on_bound
    free_set = on_bound == 0
    g_kkt[free_set] = np.abs(g[free_set])
    return np.max(g_kkt)


def bvls(A, b, x_lsq, lb, ub, tol, max_iter, verbose, rcond=None):
    """Bounded-variable least-squares solver using BVLS algorithm."""
    m, n = A.shape

    x = x_lsq.copy()
    on_bound = np.zeros(n)

    # Initialize x and on_bound based on lower and upper bounds
    mask = x <= lb
    x[mask] = lb[mask]
    on_bound[mask] = -1

    mask = x >= ub
    x[mask] = ub[mask]
    on_bound[mask] = 1

    free_set = on_bound == 0
    active_set = ~free_set
    free_set, = np.nonzero(free_set)

    r = A.dot(x) - b
    cost = 0.5 * np.dot(r, r)
    initial_cost = cost
    g = A.T.dot(r)

    cost_change = None
    step_norm = None
    iteration = 0

    if verbose == 2:
        print_header_linear()

    # Initialization loop to ensure feasible least-squares solution on free variables
    while free_set.size > 0:
        if verbose == 2:
            optimality = compute_kkt_optimality(g, on_bound)
            print_iteration_linear(iteration, cost, cost_change, step_norm,
                                   optimality)

        iteration += 1
        x_free_old = x[free_set].copy()

        A_free = A[:, free_set]
        b_free = b - A.dot(x * active_set)
        z = lstsq(A_free, b_free, rcond=rcond)[0]

        lbv = z < lb[free_set]
        ubv = z > ub[free_set]
        v = lbv | ubv

        # Adjust x, active_set, and on_bound based on bounds violations
        if np.any(lbv):
            ind = free_set[lbv]
            x[ind] = lb[ind]
            active_set[ind] = True
            on_bound[ind] = -1

        if np.any(ubv):
            ind = free_set[ubv]
            x[ind] = ub[ind]
            active_set[ind] = True
            on_bound[ind] = 1

        ind = free_set[~v]
        x[ind] = z[~v]

        r = A.dot(x) - b
        cost_new = 0.5 * np.dot(r, r)
        cost_change = cost - cost_new
        cost = cost_new
        g = A.T.dot(r)
        step_norm = norm(x[free_set] - x_free_old)

        # Update free_set based on the violations
        if np.any(v):
            free_set = free_set[~v]
        else:
            break

    if max_iter is None:
        max_iter = n
    max_iter += iteration

    termination_status = None

    # Main BVLS loop.
    optimality = compute_kkt_optimality(g, on_bound)
    for iteration in range(iteration, max_iter):  # BVLS Loop A
        # BVLS 算法的主循环，迭代执行直到达到最大迭代次数 max_iter
        if verbose == 2:
            # 如果 verbose 等于 2，则打印当前迭代信息
            print_iteration_linear(iteration, cost, cost_change,
                                   step_norm, optimality)

        if optimality < tol:
            # 如果当前优化性小于给定的容许误差 tol，则设置终止状态为 1
            termination_status = 1

        if termination_status is not None:
            # 如果终止状态不为 None，则跳出循环
            break

        move_to_free = np.argmax(g * on_bound)
        # 找出 g * on_bound 中最大值的索引，用于将对应位置的 on_bound 置为 0
        on_bound[move_to_free] = 0
        
        while True:   # BVLS Loop B
            # BVLS 算法的内部循环，持续寻找最优解直到满足约束条件

            free_set = on_bound == 0
            active_set = ~free_set
            free_set, = np.nonzero(free_set)
    
            x_free = x[free_set]
            x_free_old = x_free.copy()
            lb_free = lb[free_set]
            ub_free = ub[free_set]

            A_free = A[:, free_set]
            b_free = b - A.dot(x * active_set)
            z = lstsq(A_free, b_free, rcond=rcond)[0]

            lbv, = np.nonzero(z < lb_free)
            ubv, = np.nonzero(z > ub_free)
            v = np.hstack((lbv, ubv))

            if v.size > 0:
                alphas = np.hstack((
                    lb_free[lbv] - x_free[lbv],
                    ub_free[ubv] - x_free[ubv])) / (z[v] - x_free[v])

                i = np.argmin(alphas)
                i_free = v[i]
                alpha = alphas[i]

                x_free *= 1 - alpha
                x_free += alpha * z
                x[free_set] = x_free

                if i < lbv.size:
                    on_bound[free_set[i_free]] = -1
                else:
                    on_bound[free_set[i_free]] = 1
            else:
                x_free = z
                x[free_set] = x_free
                break

        step_norm = norm(x_free - x_free_old)

        r = A.dot(x) - b
        cost_new = 0.5 * np.dot(r, r)
        cost_change = cost - cost_new

        if cost_change < tol * cost:
            # 如果成本变化小于给定的容许误差乘以当前成本，则设置终止状态为 2
            termination_status = 2
        cost = cost_new

        g = A.T.dot(r)
        optimality = compute_kkt_optimality(g, on_bound)

    if termination_status is None:
        # 如果终止状态仍为 None，则将其设置为 0，表示正常结束
        termination_status = 0

    return OptimizeResult(
        x=x, fun=r, cost=cost, optimality=optimality, active_mask=on_bound,
        nit=iteration + 1, status=termination_status,
        initial_cost=initial_cost)
```