# `D:\src\scipysrc\scipy\scipy\optimize\_trustregion_constr\equality_constrained_sqp.py`

```
    """Byrd-Omojokun Trust-Region SQP method."""

    # 导入必要的库和模块
    from scipy.sparse import eye as speye
    from .projections import projections
    from .qp_subproblem import modified_dogleg, projected_cg, box_intersections
    import numpy as np
    from numpy.linalg import norm

    __all__ = ['equality_constrained_sqp']

    # 默认的变量缩放函数，返回单位矩阵
    def default_scaling(x):
        n, = np.shape(x)
        return speye(n)

    # 使用Byrd-Omojokun信任区域SQP方法求解非线性等式约束问题
    def equality_constrained_sqp(fun_and_constr, grad_and_jac, lagr_hess,
                                 x0, fun0, grad0, constr0,
                                 jac0, stop_criteria,
                                 state,
                                 initial_penalty,
                                 initial_trust_radius,
                                 factorization_method,
                                 trust_lb=None,
                                 trust_ub=None,
                                 scaling=default_scaling):
        """Solve nonlinear equality-constrained problem using trust-region SQP.

        Solve optimization problem:

            minimize fun(x)
            subject to: constr(x) = 0

        using Byrd-Omojokun Trust-Region SQP method described in [1]_. Several
        implementation details are based on [2]_ and [3]_, p. 549.

        References
        ----------
        .. [1] Lalee, Marucha, Jorge Nocedal, and Todd Plantenga. "On the
               implementation of an algorithm for large-scale equality
               constrained optimization." SIAM Journal on
               Optimization 8.3 (1998): 682-706.
        .. [2] Byrd, Richard H., Mary E. Hribar, and Jorge Nocedal.
               "An interior point algorithm for large-scale nonlinear
               programming." SIAM Journal on Optimization 9.4 (1999): 877-900.
        .. [3] Nocedal, Jorge, and Stephen J. Wright. "Numerical optimization"
               Second Edition (2006).
        """
        PENALTY_FACTOR = 0.3  # Rho from formula (3.51), reference [2]_, p.891.
        LARGE_REDUCTION_RATIO = 0.9
        INTERMEDIARY_REDUCTION_RATIO = 0.3
        SUFFICIENT_REDUCTION_RATIO = 1e-8  # Eta from reference [2]_, p.892.
        TRUST_ENLARGEMENT_FACTOR_L = 7.0
        TRUST_ENLARGEMENT_FACTOR_S = 2.0
        MAX_TRUST_REDUCTION = 0.5
        MIN_TRUST_REDUCTION = 0.1
        SOC_THRESHOLD = 0.1
        TR_FACTOR = 0.8  # Zeta from formula (3.21), reference [2]_, p.885.
        BOX_FACTOR = 0.5

        n, = np.shape(x0)  # Number of parameters

        # Set default lower and upper bounds.
        if trust_lb is None:
            trust_lb = np.full(n, -np.inf)
        if trust_ub is None:
            trust_ub = np.full(n, np.inf)

        # Initial values
        x = np.copy(x0)  # 复制初始点
        trust_radius = initial_trust_radius  # 设置初始信任域半径
        penalty = initial_penalty  # 设置初始惩罚参数

        # Compute Values
        f = fun0  # 计算目标函数值
        c = grad0  # 计算目标函数梯度
        b = constr0  # 计算约束函数值
        A = jac0  # 计算雅可比矩阵
        S = scaling(x)  # 计算变量缩放矩阵

        # Get projections
        try:
            Z, LS, Y = projections(A, factorization_method)
    except ValueError as e:
        if str(e) == "expected square matrix":
            # 如果出现值错误，且错误消息为“expected square matrix”
            # 可能是由于等式约束多于独立变量引起的
            raise ValueError(
                "The 'expected square matrix' error can occur if there are"
                " more equality constraints than independent variables."
                " Consider how your constraints are set up, or use"
                " factorization_method='SVDFactorization'."
            ) from e
        else:
            # 如果出现其他值错误，则直接重新抛出该错误
            raise e

    # 计算最小二乘拉格朗日乘子
    v = -LS.dot(c)
    # 计算黑塞矩阵
    H = lagr_hess(x, v)

    # 更新状态参数
    optimality = norm(c + A.T.dot(v), np.inf)
    constr_violation = norm(b, np.inf) if len(b) > 0 else 0
    cg_info = {'niter': 0, 'stop_cond': 0,
               'hits_boundary': False}

    last_iteration_failed = False
    # 返回计算结果 x 和状态参数 state
    return x, state
```