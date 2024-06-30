# `D:\src\scipysrc\scipy\scipy\optimize\_trustregion_constr\minimize_trustregion_constr.py`

```
# 导入时间模块，用于处理时间相关操作
import time
# 导入 NumPy 库，用于科学计算和数组操作
import numpy as np
# 导入 LinearOperator 类，用于定义线性算子
from scipy.sparse.linalg import LinearOperator
# 导入 VectorFunction 类，用于处理可微分函数
from .._differentiable_functions import VectorFunction
# 导入各种约束条件类
from .._constraints import (
    NonlinearConstraint, LinearConstraint, PreparedConstraint, Bounds, strict_bounds)
# 导入 BFGS 类，用于拟牛顿法的 Hessian 更新策略
from .._hessian_update_strategy import BFGS
# 导入 OptimizeResult 类，用于优化结果的封装
from .._optimize import OptimizeResult
# 导入 ScalarFunction 类，用于处理标量函数
from .._differentiable_functions import ScalarFunction
# 导入 equality_constrained_sqp 函数，用于处理等式约束的 SQP 优化
from .equality_constrained_sqp import equality_constrained_sqp
# 导入 CanonicalConstraint 和 initial_constraints_as_canonical 函数
from .canonical_constraint import (CanonicalConstraint,
                                   initial_constraints_as_canonical)
# 导入 tr_interior_point 函数，用于处理 TR 内点法
from .tr_interior_point import tr_interior_point
# 导入报告相关的类，用于输出优化过程的报告
from .report import BasicReport, SQPReport, IPReport

# 定义终止消息的字典，用于记录优化终止条件对应的消息
TERMINATION_MESSAGES = {
    0: "The maximum number of function evaluations is exceeded.",
    1: "`gtol` termination condition is satisfied.",
    2: "`xtol` termination condition is satisfied.",
    3: "`callback` function requested termination."
}

class HessianLinearOperator:
    """从 hessp 构建 LinearOperator 对象"""
    def __init__(self, hessp, n):
        self.hessp = hessp
        self.n = n

    def __call__(self, x, *args):
        """调用对象时，返回一个 LinearOperator 对象"""
        def matvec(p):
            return self.hessp(x, p, *args)

        return LinearOperator((self.n, self.n), matvec=matvec)


class LagrangianHessian:
    """Lagrangian 的 Hessian 矩阵作为 LinearOperator 对象。

    Lagrangian 是由目标函数和所有约束函数乘以拉格朗日乘子后得到的。
    """
    def __init__(self, n, objective_hess, constraints_hess):
        self.n = n
        self.objective_hess = objective_hess
        self.constraints_hess = constraints_hess

    def __call__(self, x, v_eq=np.empty(0), v_ineq=np.empty(0)):
        """调用对象时，返回 Lagrangian 的 Hessian 作为 LinearOperator 对象"""
        H_objective = self.objective_hess(x)
        H_constraints = self.constraints_hess(x, v_eq, v_ineq)

        def matvec(p):
            return H_objective.dot(p) + H_constraints.dot(p)

        return LinearOperator((self.n, self.n), matvec)


def update_state_sqp(state, x, last_iteration_failed, objective, prepared_constraints,
                     start_time, tr_radius, constr_penalty, cg_info):
    """更新 SQP 状态对象的信息"""
    # 增加迭代次数计数
    state.nit += 1
    # 更新函数评估次数
    state.nfev = objective.nfev
    # 更新梯度计算次数
    state.njev = objective.ngev
    # 更新 Hessian 矩阵计算次数
    state.nhev = objective.nhev
    # 更新约束函数评估次数列表
    state.constr_nfev = [c.fun.nfev if isinstance(c.fun, VectorFunction) else 0
                         for c in prepared_constraints]
    # 更新约束函数梯度计算次数列表
    state.constr_njev = [c.fun.njev if isinstance(c.fun, VectorFunction) else 0
                         for c in prepared_constraints]
    # 更新约束函数 Hessian 矩阵计算次数列表
    state.constr_nhev = [c.fun.nhev if isinstance(c.fun, VectorFunction) else 0
                         for c in prepared_constraints]
    # 如果上一次迭代未失败，则更新状态对象的各项属性
    if not last_iteration_failed:
        # 更新状态对象的 x 属性为当前 x
        state.x = x
        # 更新状态对象的 fun 属性为目标函数 objective.f
        state.fun = objective.f
        # 更新状态对象的 grad 属性为目标函数的梯度 objective.g
        state.grad = objective.g
        # 更新状态对象的 v 属性为准备好的约束条件列表中每个约束的函数值 c.fun.v 的集合
        state.v = [c.fun.v for c in prepared_constraints]
        # 更新状态对象的 constr 属性为准备好的约束条件列表中每个约束的函数 f 的集合
        state.constr = [c.fun.f for c in prepared_constraints]
        # 更新状态对象的 jac 属性为准备好的约束条件列表中每个约束的雅可比矩阵 J 的集合
        state.jac = [c.fun.J for c in prepared_constraints]
        
        # 计算拉格朗日函数的梯度
        state.lagrangian_grad = np.copy(state.grad)
        for c in prepared_constraints:
            state.lagrangian_grad += c.fun.J.T.dot(c.fun.v)
        
        # 计算最优性条件，使用无穷范数求梯度的范数
        state.optimality = np.linalg.norm(state.lagrangian_grad, np.inf)
        
        # 计算最大约束违反
        state.constr_violation = 0
        for i in range(len(prepared_constraints)):
            lb, ub = prepared_constraints[i].bounds
            c = state.constr[i]
            state.constr_violation = np.max([state.constr_violation,
                                             np.max(lb - c),
                                             np.max(c - ub)])

    # 计算执行时间并更新状态对象的 execution_time 属性
    state.execution_time = time.time() - start_time
    # 更新状态对象的 tr_radius 属性为给定的 tr_radius
    state.tr_radius = tr_radius
    # 更新状态对象的 constr_penalty 属性为给定的 constr_penalty
    state.constr_penalty = constr_penalty
    # 将共轭梯度法的迭代次数加到状态对象的 cg_niter 属性上
    state.cg_niter += cg_info["niter"]
    # 更新状态对象的 cg_stop_cond 属性为共轭梯度法的停止条件
    state.cg_stop_cond = cg_info["stop_cond"]

    # 返回更新后的状态对象
    return state
# 更新状态变量 `state`，通过调用 `update_state_sqp` 函数更新 SQP 算法的状态
def update_state_ip(state, x, last_iteration_failed, objective,
                    prepared_constraints, start_time,
                    tr_radius, constr_penalty, cg_info,
                    barrier_parameter, barrier_tolerance):
    state = update_state_sqp(state, x, last_iteration_failed, objective,
                             prepared_constraints, start_time, tr_radius,
                             constr_penalty, cg_info)
    # 设置状态变量 `state` 的障碍参数为 `barrier_parameter`
    state.barrier_parameter = barrier_parameter
    # 设置状态变量 `state` 的障碍容忍度为 `barrier_tolerance`
    state.barrier_tolerance = barrier_tolerance
    # 返回更新后的状态变量 `state`
    return state


# 最小化带约束条件的标量函数
def _minimize_trustregion_constr(fun, x0, args, grad,
                                 hess, hessp, bounds, constraints,
                                 xtol=1e-8, gtol=1e-8,
                                 barrier_tol=1e-8,
                                 sparse_jacobian=None,
                                 callback=None, maxiter=1000,
                                 verbose=0, finite_diff_rel_step=None,
                                 initial_constr_penalty=1.0, initial_tr_radius=1.0,
                                 initial_barrier_parameter=0.1,
                                 initial_barrier_tolerance=0.1,
                                 factorization_method=None,
                                 disp=False):
    """Minimize a scalar function subject to constraints.

    Parameters
    ----------
    gtol : float, optional
        终止算法的拉格朗日梯度的范数容忍度。当拉格朗日梯度的无穷范数（即最大绝对值）和约束违反度
        均小于 `gtol` 时，算法终止。默认值为 1e-8。
    xtol : float, optional
        终止算法的自变量变化容忍度。当 `tr_radius < xtol` 时，其中 `tr_radius` 是算法中
        使用的信赖域的半径。默认值为 1e-8。
    barrier_tol : float, optional
        算法终止的障碍参数阈值。当存在不等式约束时，算法只有在障碍参数小于 `barrier_tol` 时才
        终止。默认值为 1e-8。
    sparse_jacobian : {bool, None}, optional
        决定如何表示约束的雅可比矩阵。如果是布尔值，则所有约束的雅可比矩阵将被转换为相应的格式。
        如果是 None（默认值），则雅可比矩阵不会被转换，但算法只有在它们都有相同格式时才能继续进行。
    # 初始信任半径，用于定义在优化过程中解之间的最大距离
    # 信任半径反映了算法对优化问题局部近似的信任程度。对于准确的局部近似，信任半径应该很大；
    # 对于仅在当前点附近有效的近似，信任半径应该很小。
    # 信任半径在优化过程中会自动更新，初始值为 initial_tr_radius。
    initial_tr_radius: float, optional
    
    # 初始约束惩罚参数，用于平衡减少目标函数和满足约束的要求
    # 惩罚参数用于定义优化过程中的优势函数：
    # merit_function(x) = fun(x) + constr_penalty * constr_norm_l2(x)
    # 其中 constr_norm_l2(x) 是一个包含所有约束的向量的L2范数。
    # 优势函数用于接受或拒绝试探点，constr_penalty 权衡减少目标函数和约束两个冲突的目标。
    # 惩罚参数在优化过程中会自动更新，初始值为 initial_constr_penalty。
    initial_constr_penalty: float, optional
    
    # 初始障碍参数和初始障碍子问题容差，仅在存在不等式约束时使用
    # 当存在不等式约束时，障碍参数和容差用于定义障碍子问题的解决方式。
    # 对于优化问题 min_x f(x) 满足不等式约束 c(x) <= 0，
    # 算法引入松弛变量，解决问题 min_(x,s) f(x) + barrier_parameter*sum(ln(s))
    # 满足约束条件 c(x) + s = 0 的问题，而不是原始问题。
    # 障碍参数和容差会逐步减小用于终止子问题的解决过程，
    # 初始障碍参数为 initial_barrier_parameter，初始容差为 initial_barrier_tolerance。
    # 默认值分别为 0.1（参见 [1]，第19页）。
    # 注意，障碍参数和障碍容差会以相同的比例因子更新。
    initial_barrier_parameter, initial_barrier_tolerance: float, optional
    factorization_method : string or None, optional
        # 选择用于因式分解约束雅可比矩阵的方法。使用 None 表示自动选择，或者以下之一：

            - 'NormalEquation'（需要 scikit-sparse）
            - 'AugmentedSystem'
            - 'QRFactorization'
            - 'SVDFactorization'

        # 方法 'NormalEquation' 和 'AugmentedSystem' 只能用于稀疏约束。算法需要的投影将分别通过正规方程和增广系统方法进行计算，具体请参见 [1]。'NormalEquation' 计算 `A A.T` 的 Cholesky 因式分解，而 'AugmentedSystem' 执行增广系统的 LU 因式分解。它们通常提供类似的结果。对于稀疏矩阵，默认使用 'AugmentedSystem'。

        # 方法 'QRFactorization' 和 'SVDFactorization' 只能用于稠密约束。它们分别通过 QR 和 SVD 因式分解计算所需的投影。'SVDFactorization' 方法可以处理行秩不足的雅可比矩阵，并且会在其他因式分解方法失败时使用（可能需要将稀疏矩阵转换为稠密格式）。默认情况下，对于稠密矩阵使用 'QRFactorization'。

    finite_diff_rel_step : None or array_like, optional
        # 有限差分逼近的相对步长。

    maxiter : int, optional
        # 算法的最大迭代次数。默认为 1000。

    verbose : {0, 1, 2}, optional
        # 算法的详细程度：

            * 0（默认）：静默工作。
            * 1：显示终止报告。
            * 2：显示迭代过程中的进展。
            * 3：显示迭代过程中的进展（更详细的报告）。

    disp : bool, optional
        # 如果为 True（默认），则如果 verbose 为 0，则将 verbose 设置为 1。

    Returns
    -------
    `OptimizeResult` with the fields documented below. Note the following:

        1. All values corresponding to the constraints are ordered as they
           were passed to the solver. And values corresponding to `bounds`
           constraints are put *after* other constraints.
        2. All numbers of function, Jacobian or Hessian evaluations correspond
           to numbers of actual Python function calls. It means, for example,
           that if a Jacobian is estimated by finite differences, then the
           number of Jacobian evaluations will be zero and the number of
           function evaluations will be incremented by all calls during the
           finite difference estimation.

    x : ndarray, shape (n,)
        # 找到的解。
    optimality : float
        # 解处的拉格朗日梯度的无穷范数。
    constr_violation : float
        # 解处的最大约束违反量。
    # fun : float
    #     Objective function at the solution.

    # grad : ndarray, shape (n,)
    #     Gradient of the objective function at the solution.

    # lagrangian_grad : ndarray, shape (n,)
    #     Gradient of the Lagrangian function at the solution.

    # nit : int
    #     Total number of iterations.

    # nfev : integer
    #     Number of the objective function evaluations.

    # njev : integer
    #     Number of the objective function gradient evaluations.

    # nhev : integer
    #     Number of the objective function Hessian evaluations.

    # cg_niter : int
    #     Total number of the conjugate gradient method iterations.

    # method : {'equality_constrained_sqp', 'tr_interior_point'}
    #     Optimization method used.

    # constr : list of ndarray
    #     List of constraint values at the solution.

    # jac : list of {ndarray, sparse matrix}
    #     List of the Jacobian matrices of the constraints at the solution.

    # v : list of ndarray
    #     List of the Lagrange multipliers for the constraints at the solution.
    #     For an inequality constraint a positive multiplier means that the upper
    #     bound is active, a negative multiplier means that the lower bound is
    #     active and if a multiplier is zero it means the constraint is not
    #     active.

    # constr_nfev : list of int
    #     Number of constraint evaluations for each of the constraints.

    # constr_njev : list of int
    #     Number of Jacobian matrix evaluations for each of the constraints.

    # constr_nhev : list of int
    #     Number of Hessian evaluations for each of the constraints.

    # tr_radius : float
    #     Radius of the trust region at the last iteration.

    # constr_penalty : float
    #     Penalty parameter at the last iteration, see `initial_constr_penalty`.

    # barrier_tolerance : float
    #     Tolerance for the barrier subproblem at the last iteration.
    #     Only for problems with inequality constraints.

    # barrier_parameter : float
    #     Barrier parameter at the last iteration. Only for problems
    #     with inequality constraints.

    # execution_time : float
    #     Total execution time.

    # message : str
    #     Termination message.

    # status : {0, 1, 2, 3}
    #     Termination status:
    #         * 0 : The maximum number of function evaluations is exceeded.
    #         * 1 : `gtol` termination condition is satisfied.
    #         * 2 : `xtol` termination condition is satisfied.
    #         * 3 : `callback` function requested termination.

    # cg_stop_cond : int
    #     Reason for CG subproblem termination at the last iteration:
    #         * 0 : CG subproblem not evaluated.
    #         * 1 : Iteration limit was reached.
    #         * 2 : Reached the trust-region boundary.
    #         * 3 : Negative curvature detected.
    #         * 4 : Tolerance was satisfied.

    # References
    # ----------
    # .. [1] Conn, A. R., Gould, N. I., & Toint, P. L.
    #        Trust region methods. 2000. Siam. pp. 19.
    x0 = np.atleast_1d(x0).astype(float)
    n_vars = np.size(x0)
    # 如果没有提供 Hessian 矩阵，则根据 hessp 是否可调用来选择默认的 Hessian 近似器
    if hess is None:
        if callable(hessp):
            hess = HessianLinearOperator(hessp, n_vars)
        else:
            hess = BFGS()
    
    # 如果开启了显示选项且 verbose 等于 0，则将 verbose 设置为 1
    if disp and verbose == 0:
        verbose = 1

    # 如果给定了约束条件 bounds，则进行以下修改使其保持可行性
    if bounds is not None:
        # 调整 bounds.lb 和 bounds.ub，使其无限接近但不等于负无穷和正无穷
        modified_lb = np.nextafter(bounds.lb, -np.inf, where=bounds.lb > -np.inf)
        modified_ub = np.nextafter(bounds.ub, np.inf, where=bounds.ub < np.inf)
        # 将无限接近但不等于负无穷和正无穷的值应用到有限的 lb 和 ub 上
        modified_lb = np.where(np.isfinite(bounds.lb), modified_lb, bounds.lb)
        modified_ub = np.where(np.isfinite(bounds.ub), modified_ub, bounds.ub)
        # 创建 Bounds 对象，保持可行性
        bounds = Bounds(modified_lb, modified_ub, keep_feasible=bounds.keep_feasible)
        # 根据 bounds 创建有严格约束的有限差分边界
        finite_diff_bounds = strict_bounds(bounds.lb, bounds.ub,
                                           bounds.keep_feasible, n_vars)
    else:
        # 如果未提供 bounds，则将有严格约束的有限差分边界设置为 (-无穷, +无穷)
        finite_diff_bounds = (-np.inf, np.inf)

    # 定义目标函数对象
    objective = ScalarFunction(fun, x0, args, grad, hess,
                               finite_diff_rel_step, finite_diff_bounds)

    # 当 constraints 是 NonlinearConstraint 或 LinearConstraint 类型时，将其转换成列表格式
    if isinstance(constraints, (NonlinearConstraint, LinearConstraint)):
        constraints = [constraints]

    # 准备约束条件
    prepared_constraints = [
        PreparedConstraint(c, x0, sparse_jacobian, finite_diff_bounds)
        for c in constraints]

    # 检查所有约束条件是否都是稀疏或密集类型
    n_sparse = sum(c.fun.sparse_jacobian for c in prepared_constraints)
    if 0 < n_sparse < len(prepared_constraints):
        raise ValueError("All constraints must have the same kind of the "
                         "Jacobian --- either all sparse or all dense. "
                         "You can set the sparsity globally by setting "
                         "`sparse_jacobian` to either True of False.")
    # 如果有约束条件存在，则确定是否使用稀疏雅可比矩阵
    if prepared_constraints:
        sparse_jacobian = n_sparse > 0

    # 如果提供了 bounds，并且 sparse_jacobian 为 None，则将其设为 True
    if bounds is not None:
        if sparse_jacobian is None:
            sparse_jacobian = True
        # 添加 bounds 作为一个准备好的约束条件
        prepared_constraints.append(PreparedConstraint(bounds, x0,
                                                       sparse_jacobian))

    # 将初始约束条件转换为规范形式
    c_eq0, c_ineq0, J_eq0, J_ineq0 = initial_constraints_as_canonical(
        n_vars, prepared_constraints, sparse_jacobian)

    # 准备所有规范约束条件并将其连接成一个
    canonical_all = [CanonicalConstraint.from_PreparedConstraint(c)
                     for c in prepared_constraints]

    # 根据规范约束条件的数量确定 canonical 是空对象、单个对象还是多个对象的连接
    if len(canonical_all) == 0:
        canonical = CanonicalConstraint.empty(n_vars)
    elif len(canonical_all) == 1:
        canonical = canonical_all[0]
    else:
        canonical = CanonicalConstraint.concatenate(canonical_all,
                                                    sparse_jacobian)

    # 生成拉格朗日函数的 Hessian 矩阵
    lagrangian_hess = LagrangianHessian(n_vars, objective.hess, canonical.hess)

    # 选择适当的方法
    # 如果canonical对象的不等式约束数量为0，则选择equality_constrained_sqp方法
    if canonical.n_ineq == 0:
        method = 'equality_constrained_sqp'
    else:
        # 否则选择tr_interior_point方法
        method = 'tr_interior_point'

    # 构建OptimizeResult对象，用于存储优化结果
    state = OptimizeResult(
        nit=0, nfev=0, njev=0, nhev=0,  # 迭代次数、函数评估次数、雅可比矩阵评估次数、黑塞矩阵评估次数
        cg_niter=0, cg_stop_cond=0,  # 共轭梯度法迭代次数、停止条件
        fun=objective.f, grad=objective.g,  # 目标函数和梯度
        lagrangian_grad=np.copy(objective.g),  # 拉格朗日函数的梯度
        constr=[c.fun.f for c in prepared_constraints],  # 约束函数值列表
        jac=[c.fun.J for c in prepared_constraints],  # 约束雅可比矩阵列表
        constr_nfev=[0 for c in prepared_constraints],  # 每个约束函数的函数评估次数列表
        constr_njev=[0 for c in prepared_constraints],  # 每个约束函数的雅可比矩阵评估次数列表
        constr_nhev=[0 for c in prepared_constraints],  # 每个约束函数的黑塞矩阵评估次数列表
        v=[c.fun.v for c in prepared_constraints],  # 约束向量列表
        method=method)  # 优化方法名称

    # 开始计时
    start_time = time.time()

    # 定义停止条件
    # 如果优化方法选择为 'equality_constrained_sqp'，则定义停止标准函数
    if method == 'equality_constrained_sqp':
        # 定义停止标准函数，接受多个参数以评估优化是否应该停止
        def stop_criteria(state, x, last_iteration_failed,
                          optimality, constr_violation,
                          tr_radius, constr_penalty, cg_info):
            # 更新状态信息，包括目标函数值、约束信息等
            state = update_state_sqp(state, x, last_iteration_failed,
                                     objective, prepared_constraints,
                                     start_time, tr_radius, constr_penalty,
                                     cg_info)
            # 如果设置为详细输出模式 2，则使用基本报告打印迭代信息
            if verbose == 2:
                BasicReport.print_iteration(state.nit,
                                            state.nfev,
                                            state.cg_niter,
                                            state.fun,
                                            state.tr_radius,
                                            state.optimality,
                                            state.constr_violation)
            # 如果设置为更详细输出模式，则使用SQP报告打印迭代信息
            elif verbose > 2:
                SQPReport.print_iteration(state.nit,
                                          state.nfev,
                                          state.cg_niter,
                                          state.fun,
                                          state.tr_radius,
                                          state.optimality,
                                          state.constr_violation,
                                          state.constr_penalty,
                                          state.cg_stop_cond)
            # 将状态设置为None，即清除状态信息
            state.status = None
            # 使用别名state.nit为state.niter（向后兼容回调）
            state.niter = state.nit
            # 如果提供了回调函数，则尝试调用它
            if callback is not None:
                callback_stop = False
                try:
                    callback_stop = callback(state)
                except StopIteration:
                    callback_stop = True
                # 如果回调返回True，则设置状态为3，表示停止优化
                if callback_stop:
                    state.status = 3
                    return True
            # 根据优化的停止标准评估状态是否应该为0、1、2或3
            if state.optimality < gtol and state.constr_violation < gtol:
                state.status = 1  # 优化达到全局优化阈值gtol
            elif state.tr_radius < xtol:
                state.status = 2  # 优化达到步长阈值xtol
            elif state.nit >= maxiter:
                state.status = 0  # 达到最大迭代次数maxiter
            return state.status in (0, 1, 2, 3)  # 返回状态是否在停止状态集合中
    `
        # 如果优化方法为 'tr_interior_point'，则定义停止条件函数
        elif method == 'tr_interior_point':
            # 定义停止条件函数，根据当前状态、变量 x、上次迭代失败标志、TR 半径、约束惩罚、CG 信息、障碍参数、障碍容忍度更新状态
            def stop_criteria(state, x, last_iteration_failed, tr_radius,
                              constr_penalty, cg_info, barrier_parameter,
                              barrier_tolerance):
                state = update_state_ip(state, x, last_iteration_failed,
                                        objective, prepared_constraints,
                                        start_time, tr_radius, constr_penalty,
                                        cg_info, barrier_parameter, barrier_tolerance)
                # 如果 verbose 等于 2，打印基础报告的迭代信息
                if verbose == 2:
                    BasicReport.print_iteration(state.nit,
                                                state.nfev,
                                                state.cg_niter,
                                                state.fun,
                                                state.tr_radius,
                                                state.optimality,
                                                state.constr_violation)
                # 如果 verbose 大于 2，打印 IP 算法报告的迭代信息
                elif verbose > 2:
                    IPReport.print_iteration(state.nit,
                                             state.nfev,
                                             state.cg_niter,
                                             state.fun,
                                             state.tr_radius,
                                             state.optimality,
                                             state.constr_violation,
                                             state.constr_penalty,
                                             state.barrier_parameter,
                                             state.cg_stop_cond)
                state.status = None
                state.niter = state.nit  # 为回调函数设置别名（向后兼容性）
                # 如果存在回调函数，则尝试执行回调函数，处理 StopIteration 异常
                if callback is not None:
                    callback_stop = False
                    try:
                        callback_stop = callback(state)
                    except StopIteration:
                        callback_stop = True
                    # 如果回调函数返回 True，设置状态为 3，并返回 True 表示停止优化过程
                    if callback_stop:
                        state.status = 3
                        return True
                # 根据优化精度条件判断优化状态
                if state.optimality < gtol and state.constr_violation < gtol:
                    state.status = 1
                elif (state.tr_radius < xtol
                      and state.barrier_parameter < barrier_tol):
                    state.status = 2
                elif state.nit >= maxiter:
                    state.status = 0
                # 返回状态是否在 (0, 1, 2, 3) 中，表示是否继续优化
                return state.status in (0, 1, 2, 3)
    
        # 如果 verbose 等于 2，打印基础报告的标题
        if verbose == 2:
            BasicReport.print_header()
        # 如果 verbose 大于 2，根据优化方法选择打印 SQP 报告或者 IP 报告的标题
        elif verbose > 2:
            if method == 'equality_constrained_sqp':
                SQPReport.print_header()
            elif method == 'tr_interior_point':
                IPReport.print_header()
    
        # 调用下层函数执行优化过程
    # 如果优化方法为 'equality_constrained_sqp'，则定义一个函数 fun_and_constr(x)
    # 返回目标函数值和等式约束函数值的元组
    def fun_and_constr(x):
        f = objective.fun(x)  # 计算目标函数在 x 处的值
        c_eq, _ = canonical.fun(x)  # 计算等式约束函数在 x 处的值
        return f, c_eq

    # 如果优化方法为 'equality_constrained_sqp'，则定义一个函数 grad_and_jac(x)
    # 返回目标函数梯度和等式约束函数雅可比矩阵的元组
    def grad_and_jac(x):
        g = objective.grad(x)  # 计算目标函数在 x 处的梯度
        J_eq, _ = canonical.jac(x)  # 计算等式约束函数在 x 处的雅可比矩阵
        return g, J_eq

    # 如果方法为 'equality_constrained_sqp'，则调用相应的优化函数 equality_constrained_sqp
    _, result = equality_constrained_sqp(
        fun_and_constr, grad_and_jac, lagrangian_hess,
        x0, objective.f, objective.g,
        c_eq0, J_eq0,
        stop_criteria, state,
        initial_constr_penalty, initial_tr_radius,
        factorization_method)

    # 如果优化方法为 'tr_interior_point'，则调用相应的优化函数 tr_interior_point
    elif method == 'tr_interior_point':
        _, result = tr_interior_point(
            objective.fun, objective.grad, lagrangian_hess,
            n_vars, canonical.n_ineq, canonical.n_eq,
            canonical.fun, canonical.jac,
            x0, objective.f, objective.g,
            c_ineq0, J_ineq0, c_eq0, J_eq0,
            stop_criteria,
            canonical.keep_feasible,
            xtol, state, initial_barrier_parameter,
            initial_barrier_tolerance,
            initial_constr_penalty, initial_tr_radius,
            factorization_method)

    # 如果结果状态为 1 或 2，则将 result.success 设为 True，否则设为 False
    result.success = True if result.status in (1, 2) else False

    # 根据结果状态设置 result.message
    result.message = TERMINATION_MESSAGES[result.status]

    # 将 result.nit 赋值给 result.niter 作为向后兼容性的别名
    result.niter = result.nit

    # 如果 verbose 为 2，则输出基本报告的页脚
    if verbose == 2:
        BasicReport.print_footer()
    # 如果 verbose 大于 2，则根据方法输出相应优化方法的报告页脚
    elif verbose > 2:
        if method == 'equality_constrained_sqp':
            SQPReport.print_footer()
        elif method == 'tr_interior_point':
            IPReport.print_footer()

    # 如果 verbose 大于等于 1，则输出优化结果的消息和详细信息
    if verbose >= 1:
        print(result.message)
        print("Number of iterations: {}, function evaluations: {}, "
              "CG iterations: {}, optimality: {:.2e}, "
              "constraint violation: {:.2e}, execution time: {:4.2} s."
              .format(result.nit, result.nfev, result.cg_niter,
                      result.optimality, result.constr_violation,
                      result.execution_time))

    # 返回优化结果对象 result
    return result
```