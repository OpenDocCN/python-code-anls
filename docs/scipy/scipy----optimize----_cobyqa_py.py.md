# `D:\src\scipysrc\scipy\scipy\optimize\_cobyqa_py.py`

```
def _minimize_cobyqa(fun, x0, args=(), bounds=None, constraints=(),
                     callback=None, disp=False, maxfev=None, maxiter=None,
                     f_target=-np.inf, feasibility_tol=1e-8,
                     initial_tr_radius=1.0, final_tr_radius=1e-6, scale=False,
                     **unknown_options):
    """
    Minimize a scalar function of one or more variables using the
    Constrained Optimization BY Quadratic Approximations (COBYQA) algorithm [1]_.

    .. versionadded:: 1.14.0

    Options
    -------
    disp : bool
        Set to True to print information about the optimization procedure.
    maxfev : int
        Maximum number of function evaluations.
    maxiter : int
        Maximum number of iterations.
    f_target : float
        Target value for the objective function. The optimization procedure is
        terminated when the objective function value of a feasible point (see
        `feasibility_tol` below) is less than or equal to this target.
    feasibility_tol : float
        Absolute tolerance for the constraint violation.
    initial_tr_radius : float
        Initial trust-region radius. Typically, this value should be in the
        order of one tenth of the greatest expected change to the variables.
    final_tr_radius : float
        Final trust-region radius. It should indicate the accuracy required in
        the final values of the variables. If provided, this option overrides
        the value of `tol` in the `minimize` function.
    scale : bool
        Set to True to scale the variables according to the bounds. If True and
        if all the lower and upper bounds are finite, the variables are scaled
        to be within the range :math:`[-1, 1]`. If any of the lower or upper
        bounds is infinite, the variables are not scaled.

    References
    ----------
    .. [1] COBYQA
           https://www.cobyqa.com/stable/
    """
    # 导入 minimize 函数，避免循环导入问题
    from .._lib.cobyqa import minimize

    # 检查未知选项是否有效
    _check_unknown_options(unknown_options)
    
    # 设置优化选项字典
    options = {
        'disp': bool(disp),  # 是否显示优化过程信息
        'maxfev': int(maxfev) if maxfev is not None else 500 * len(x0),  # 最大函数评估次数
        'maxiter': int(maxiter) if maxiter is not None else 1000 * len(x0),  # 最大迭代次数
        'target': float(f_target),  # 目标函数值，达到此值优化停止
        'feasibility_tol': float(feasibility_tol),  # 约束违反的绝对容差
        'radius_init': float(initial_tr_radius),  # 初始信赖域半径
        'radius_final': float(final_tr_radius),  # 最终信赖域半径
        'scale': bool(scale),  # 是否按照边界缩放变量
    }
    
    # 调用 minimize 函数进行优化，并返回其结果
    return minimize(fun, x0, args, bounds, constraints, callback, options)
```