# `D:\src\scipysrc\scipy\scipy\optimize\_lsq\lsq_linear.py`

```
    """Linear least squares with bound constraints on independent variables."""
    # 导入必要的库和模块
    import numpy as np  # 导入 NumPy 库，用于数值计算
    from numpy.linalg import norm  # 导入 norm 函数，用于计算向量或矩阵的范数
    from scipy.sparse import issparse, csr_matrix  # 导入 issparse 和 csr_matrix，用于稀疏矩阵处理
    from scipy.sparse.linalg import LinearOperator, lsmr  # 导入 LinearOperator 和 lsmr，用于处理稀疏线性运算
    from scipy.optimize import OptimizeResult  # 导入 OptimizeResult 类，用于优化结果的封装
    from scipy.optimize._minimize import Bounds  # 导入 Bounds 类，用于处理变量的边界约束

    from .common import in_bounds, compute_grad  # 从自定义模块导入必要函数
    from .trf_linear import trf_linear  # 从自定义模块导入 trf_linear 函数
    from .bvls import bvls  # 从自定义模块导入 bvls 函数


    def prepare_bounds(bounds, n):
        # 检查 bounds 是否包含 2 个元素
        if len(bounds) != 2:
            raise ValueError("`bounds` must contain 2 elements.")
        # 将 bounds 转换为 NumPy 数组，分别为下界 lb 和上界 ub
        lb, ub = (np.asarray(b, dtype=float) for b in bounds)

        # 如果 lb 是标量，则将其重复 n 次，确保与参数数量匹配
        if lb.ndim == 0:
            lb = np.resize(lb, n)

        # 如果 ub 是标量，则将其重复 n 次，确保与参数数量匹配
        if ub.ndim == 0:
            ub = np.resize(ub, n)

        # 返回处理后的下界 lb 和上界 ub
        return lb, ub


    # 定义算法终止时的消息字典，对应不同的终止条件
    TERMINATION_MESSAGES = {
        -1: "The algorithm was not able to make progress on the last iteration.",
        0: "The maximum number of iterations is exceeded.",
        1: "The first-order optimality measure is less than `tol`.",
        2: "The relative change of the cost function is less than `tol`.",
        3: "The unconstrained solution is optimal."
    }


    def lsq_linear(A, b, bounds=(-np.inf, np.inf), method='trf', tol=1e-10,
                   lsq_solver=None, lsmr_tol=None, max_iter=None,
                   verbose=0, *, lsmr_maxiter=None):
        r"""Solve a linear least-squares problem with bounds on the variables.

        Given a m-by-n design matrix A and a target vector b with m elements,
        `lsq_linear` solves the following optimization problem::

            minimize 0.5 * ||A x - b||**2
            subject to lb <= x <= ub

        This optimization problem is convex, hence a found minimum (if iterations
        have converged) is guaranteed to be global.

        Parameters
        ----------
        A : array_like, sparse matrix of LinearOperator, shape (m, n)
            Design matrix. Can be `scipy.sparse.linalg.LinearOperator`.
        b : array_like, shape (m,)
            Target vector.
        bounds : 2-tuple of array_like or `Bounds`, optional
            Lower and upper bounds on parameters. Defaults to no bounds.
            There are two ways to specify the bounds:

                - Instance of `Bounds` class.

                - 2-tuple of array_like: Each element of the tuple must be either
                  an array with the length equal to the number of parameters, or a
                  scalar (in which case the bound is taken to be the same for all
                  parameters). Use ``np.inf`` with an appropriate sign to disable
                  bounds on all or some parameters.
    method : 'trf' or 'bvls', optional
        # 选择执行最小化的方法。
        # - 'trf'：用于线性最小二乘问题的信赖域反射算法。这是一种类似内点法的方法，所需的迭代次数与变量数量弱相关。
        # - 'bvls'：有界变量最小二乘算法。这是一种活动集方法，所需的迭代次数与变量数量相当。当 `A` 是稀疏矩阵或线性操作符时无法使用。
        默认值为 'trf'。

    tol : float, optional
        # 容差参数。如果在最后一次迭代中，成本函数的相对变化小于 `tol`，则算法终止。
        此外，还考虑了一阶最优性测度：
        # - ``method='trf'``：如果梯度的均匀范数，经过边界的影响缩放后，小于 `tol`，则终止。
        # - ``method='bvls'``：如果卡鲁什-库恩-塔克条件在 `tol` 容差内满足，则终止。

    lsq_solver : {None, 'exact', 'lsmr'}, optional
        # 解决无界最小二乘问题的方法，用于迭代过程中：
        # - 'exact'：使用密集 QR 或 SVD 分解方法。当 `A` 是稀疏矩阵或线性操作符时无法使用。
        # - 'lsmr'：使用 `scipy.sparse.linalg.lsmr` 迭代过程，仅需要矩阵-向量乘积的评估。无法与 ``method='bvls'`` 同时使用。
        如果为 None（默认），则根据 `A` 的类型选择解算器。

    lsmr_tol : None, float or 'auto', optional
        # `scipy.sparse.linalg.lsmr` 的容差参数 'atol' 和 'btol'
        如果为 None（默认），则设置为 ``1e-2 * tol``。如果为 'auto'，则基于当前迭代的最优性调整容差，可以加速优化过程，但不总是可靠的。

    max_iter : None or int, optional
        # 终止之前的最大迭代次数。如果为 None（默认），对于 ``method='trf'``，设置为 100；对于 ``method='bvls'``，设置为变量数量（不计算 'bvls' 初始化的迭代次数）。

    verbose : {0, 1, 2}, optional
        # 算法的详细程度：
        # - 0：静默工作（默认）。
        # - 1：显示终止报告。
        # - 2：显示迭代过程中的进展。

    lsmr_maxiter : None or int, optional
        # lsmr 最小二乘求解器的最大迭代次数，如果使用（通过设置 ``lsq_solver='lsmr'``）。如果为 None（默认），则使用 lsmr 的默认值 ``min(m, n)``，其中 ``m`` 和 ``n`` 分别为 `A` 的行数和列数。如果 ``lsq_solver='exact'``，则无效。
    # OptimizeResult 结构包含以下字段：
    # x : ndarray, shape (n,)
    #     找到的解。
    # cost : float
    #     解处的成本函数值。
    # fun : ndarray, shape (m,)
    #     解处的残差向量。
    # optimality : float
    #     一阶最优性度量。确切含义取决于 `method`，参考 `tol` 参数的描述。
    # active_mask : ndarray of int, shape (n,)
    #     每个分量指示相应约束是否活跃（即变量是否在边界上）：
    #
    #         *  0 : 约束未活跃。
    #         * -1 : 下界活跃。
    #         *  1 : 上界活跃。
    #
    #     对于 `trf` 方法可能有些随意，因为它生成严格可行的迭代点，并且 `active_mask` 在容差阈值内确定。
    # unbounded_sol : tuple
    #     由最小二乘求解器返回的无界最小二乘解元组（使用 `lsq_solver` 选项设置）。如果未设置 `lsq_solver` 或设置为 ``'exact'``，元组包含形状为 (n,) 的 ndarray 与无界解、残差平方和的 ndarray、A 的秩和 A 的奇异值的 ndarray（详见 NumPy 的 ``linalg.lstsq``）。如果 `lsq_solver` 设置为 ``'lsmr'``，元组包含形状为 (n,) 的 ndarray 与无界解、退出代码的 int、迭代次数的 int 和五个浮点数（各种范数和 A 的条件数，详见 SciPy 的 ``sparse.linalg.lsmr``）。此输出可用于确定最小二乘求解器的收敛性，特别是迭代 ``'lsmr'`` 求解器。
    #     无界最小二乘问题是最小化 ``0.5 * ||A x - b||**2``。
    # nit : int
    #     迭代次数。如果无约束解是最优的，则为零。
    # status : int
    #     算法终止的原因：
    #
    #         * -1 : 算法在最后一次迭代中无法取得进展。
    #         *  0 : 超过最大迭代次数。
    #         *  1 : 一阶最优性度量小于 `tol`。
    #         *  2 : 成本函数的相对变化小于 `tol`。
    #         *  3 : 无约束解是最优的。
    #
    # message : str
    #     终止原因的口头描述。
    # success : bool
    #     如果满足收敛标准之一，则为 True（`status` > 0）。
    # 检查方法参数是否合法，必须为 'trf' 或 'bvls'
    if method not in ['trf', 'bvls']:
        raise ValueError("`method` must be 'trf' or 'bvls'")

    # 检查 lsq_solver 参数是否合法，必须为 None, 'exact' 或 'lsmr'
    if lsq_solver not in [None, 'exact', 'lsmr']:
        raise ValueError("`solver` must be None, 'exact' or 'lsmr'.")

    # 检查 verbose 参数是否合法，必须为 0, 1, 2 中的一个
    if verbose not in [0, 1, 2]:
        raise ValueError("`verbose` must be in [0, 1, 2].")

    # 如果输入的 A 是稀疏矩阵，则转换为 CSR 格式的稀疏矩阵
    if issparse(A):
        A = csr_matrix(A)
    # 如果 A 不是 LinearOperator 类型的实例，则将其转换为至少二维的 numpy 数组
    elif not isinstance(A, LinearOperator):
        A = np.atleast_2d(np.asarray(A))

    # 如果使用 'bvls' 方法
    if method == 'bvls':
        # 如果 lsq_solver 选择了 'lsmr'，则抛出异常，因为 'bvls' 方法不能和 'lsmr' lsq_solver一起使用
        if lsq_solver == 'lsmr':
            raise ValueError("method='bvls' can't be used with "
                             "lsq_solver='lsmr'")

        # 如果 A 不是 numpy 数组，则抛出异常，因为 'bvls' 方法要求 A 是密集矩阵
        if not isinstance(A, np.ndarray):
            raise ValueError("method='bvls' can't be used with `A` being "
                             "sparse or LinearOperator.")

    # 如果 lsq_solver 没有指定
    if lsq_solver is None:
        # 如果 A 是 numpy 数组，则默认选择 'exact' 解法
        if isinstance(A, np.ndarray):
            lsq_solver = 'exact'
        # 否则，默认选择 'lsmr' 解法
        else:
            lsq_solver = 'lsmr'
    # 如果 lsq_solver 选择了 'exact' 但 A 不是 numpy 数组，则抛出异常，因为 'exact' 方法要求 A 是密集矩阵
    elif lsq_solver == 'exact' and not isinstance(A, np.ndarray):
        raise ValueError("`exact` solver can't be used when `A` is "
                         "sparse or LinearOperator.")

    # 如果 A 的维度不是 2，则抛出异常，因为该算法要求 A 是二维的
    if len(A.shape) != 2:  # No ndim for LinearOperator.
        raise ValueError("`A` must have at most 2 dimensions.")

    # 如果 max_iter 被指定且小于等于 0，则抛出异常，max_iter 必须是正整数或 None
    if max_iter is not None and max_iter <= 0:
        raise ValueError("`max_iter` must be None or positive integer.")

    # 获取矩阵 A 的行数 m 和列数 n
    m, n = A.shape

    # 将向量 b 至少转换为一维数组
    b = np.atleast_1d(b)
    # 如果 b 的维度不是 1，则抛出异常，b 必须是一维的向量
    if b.ndim != 1:
        raise ValueError("`b` must have at most 1 dimension.")

    # 如果 b 的长度不等于 A 的行数 m，则抛出异常，A 和 b 的形状不一致
    if b.size != m:
        raise ValueError("Inconsistent shapes between `A` and `b`.")

    # 如果 bounds 是 Bounds 类型的实例，则获取上下界 lb 和 ub
    if isinstance(bounds, Bounds):
        lb = bounds.lb
        ub = bounds.ub
    else:
        # 否则，根据 bounds 函数准备上下界 lb 和 ub
        lb, ub = prepare_bounds(bounds, n)

    # 如果 lb 或 ub 的形状不是 (n,)，则抛出异常，上下界的形状不正确
    if lb.shape != (n,) and ub.shape != (n,):
        raise ValueError("Bounds have wrong shape.")

    # 如果存在 lb >= ub 的情况，则抛出异常，每个下界必须严格小于对应的上界
    if np.any(lb >= ub):
        raise ValueError("Each lower bound must be strictly less than each "
                         "upper bound.")

    # 如果 lsmr_maxiter 被指定且小于 1，则抛出异常，lsmr_maxiter 必须是正整数或 None
    if lsmr_maxiter is not None and lsmr_maxiter < 1:
        raise ValueError("`lsmr_maxiter` must be None or positive integer.")

    # 如果 lsmr_tol 不是正浮点数、'auto' 或 None，则抛出异常，lsmr_tol 必须是这些值之一
    if not ((isinstance(lsmr_tol, float) and lsmr_tol > 0) or
            lsmr_tol in ('auto', None)):
        raise ValueError("`lsmr_tol` must be None, 'auto', or positive float.")

    # 根据 lsq_solver 的选择，进行相应的最小二乘求解
    if lsq_solver == 'exact':
        # 使用精确解法求解最小二乘问题
        unbd_lsq = np.linalg.lstsq(A, b, rcond=-1)
    elif lsq_solver == 'lsmr':
        # 使用 lsmr 解法求解最小二乘问题，指定初始的 tolerance 参数
        first_lsmr_tol = lsmr_tol  # tol of first call to lsmr
        if lsmr_tol is None or lsmr_tol == 'auto':
            first_lsmr_tol = 1e-2 * tol  # default if lsmr_tol not defined
        unbd_lsq = lsmr(A, b, maxiter=lsmr_maxiter,
                        atol=first_lsmr_tol, btol=first_lsmr_tol)
    # 从最小二乘求解器的结果中提取解 x_lsq
    x_lsq = unbd_lsq[0]  # extract the solution from the least squares solver
    # 检查 x_lsq 是否在界限 lb 和 ub 内
    if in_bounds(x_lsq, lb, ub):
        # 计算残差 r = A @ x_lsq - b
        r = A @ x_lsq - b
        # 计算损失函数值 cost = 0.5 * r^T @ r
        cost = 0.5 * np.dot(r, r)
        # 设置终止状态为 3
        termination_status = 3
        # 根据终止状态获取终止消息
        termination_message = TERMINATION_MESSAGES[termination_status]
        # 计算梯度 g = compute_grad(A, r)
        g = compute_grad(A, r)
        # 计算梯度的无穷范数 g_norm
        g_norm = norm(g, ord=np.inf)

        # 如果 verbose 大于 0，则打印终止消息、最终损失值和一阶优化性
        if verbose > 0:
            print(termination_message)
            print(f"Final cost {cost:.4e}, first-order optimality {g_norm:.2e}")

        # 返回优化结果对象 OptimizeResult
        return OptimizeResult(
            x=x_lsq, fun=r, cost=cost, optimality=g_norm,
            active_mask=np.zeros(n), unbounded_sol=unbd_lsq,
            nit=0, status=termination_status,
            message=termination_message, success=True)

    # 如果方法为 'trf'，则调用 trf_linear 函数
    if method == 'trf':
        res = trf_linear(A, b, x_lsq, lb, ub, tol, lsq_solver, lsmr_tol,
                         max_iter, verbose, lsmr_maxiter=lsmr_maxiter)
    # 如果方法为 'bvls'，则调用 bvls 函数
    elif method == 'bvls':
        res = bvls(A, b, x_lsq, lb, ub, tol, max_iter, verbose)

    # 设置 res 的无界解为 unbd_lsq
    res.unbounded_sol = unbd_lsq
    # 设置 res 的消息为对应状态的终止消息
    res.message = TERMINATION_MESSAGES[res.status]
    # 设置 res 的成功状态为 res.status 大于 0
    res.success = res.status > 0

    # 如果 verbose 大于 0，则打印 res 的消息和优化迭代次数等信息
    if verbose > 0:
        print(res.message)
        print(
            f"Number of iterations {res.nit}, initial cost {res.initial_cost:.4e}, "
            f"final cost {res.cost:.4e}, first-order optimality {res.optimality:.2e}."
        )

    # 删除 res 的初始损失值
    del res.initial_cost

    # 返回结果 res
    return res
```