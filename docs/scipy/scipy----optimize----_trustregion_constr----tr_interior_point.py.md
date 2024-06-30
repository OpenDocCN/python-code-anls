# `D:\src\scipysrc\scipy\scipy\optimize\_trustregion_constr\tr_interior_point.py`

```
"""Trust-region interior point method.

References
----------
.. [1] Byrd, Richard H., Mary E. Hribar, and Jorge Nocedal.
       "An interior point algorithm for large-scale nonlinear
       programming." SIAM Journal on Optimization 9.4 (1999): 877-900.
.. [2] Byrd, Richard H., Guanghui Liu, and Jorge Nocedal.
       "On the local behavior of an interior point method for
       nonlinear programming." Numerical analysis 1997 (1997): 37-56.
.. [3] Nocedal, Jorge, and Stephen J. Wright. "Numerical optimization"
       Second Edition (2006).
"""

# 导入必要的库
import scipy.sparse as sps
import numpy as np
from .equality_constrained_sqp import equality_constrained_sqp
from scipy.sparse.linalg import LinearOperator

# 定义可导出的符号列表
__all__ = ['tr_interior_point']

# 定义一个类，描述障碍子问题
class BarrierSubproblem:
    """
    Barrier optimization problem:
        minimize fun(x) - barrier_parameter*sum(log(s))
        subject to: constr_eq(x)     = 0
                  constr_ineq(x) + s = 0
    """

    def __init__(self, x0, s0, fun, grad, lagr_hess, n_vars, n_ineq, n_eq,
                 constr, jac, barrier_parameter, tolerance,
                 enforce_feasibility, global_stop_criteria,
                 xtol, fun0, grad0, constr_ineq0, jac_ineq0, constr_eq0,
                 jac_eq0):
        # 存储参数
        self.n_vars = n_vars
        self.x0 = x0
        self.s0 = s0
        self.fun = fun
        self.grad = grad
        self.lagr_hess = lagr_hess
        self.constr = constr
        self.jac = jac
        self.barrier_parameter = barrier_parameter
        self.tolerance = tolerance
        self.n_eq = n_eq
        self.n_ineq = n_ineq
        self.enforce_feasibility = enforce_feasibility
        self.global_stop_criteria = global_stop_criteria
        self.xtol = xtol
        # 计算初始函数值
        self.fun0 = self._compute_function(fun0, constr_ineq0, s0)
        # 计算初始梯度
        self.grad0 = self._compute_gradient(grad0)
        # 计算初始约束
        self.constr0 = self._compute_constr(constr_ineq0, constr_eq0, s0)
        # 计算初始雅可比矩阵
        self.jac0 = self._compute_jacobian(jac_eq0, jac_ineq0, s0)
        # 终止标志
        self.terminate = False

    # 更新障碍参数和容差
    def update(self, barrier_parameter, tolerance):
        self.barrier_parameter = barrier_parameter
        self.tolerance = tolerance

    # 获取松弛变量
    def get_slack(self, z):
        return z[self.n_vars:self.n_vars+self.n_ineq]

    # 获取变量
    def get_variables(self, z):
        return z[:self.n_vars]
    def function_and_constraints(self, z):
        """Returns barrier function and constraints at given point.

        For z = [x, s], returns barrier function:
            function(z) = fun(x) - barrier_parameter*sum(log(s))
        and barrier constraints:
            constraints(z) = [   constr_eq(x)     ]
                             [ constr_ineq(x) + s ]

        """
        # Get variables and slack variables
        x = self.get_variables(z)
        s = self.get_slack(z)
        # Compute function value
        f = self.fun(x)
        # Compute equality and inequality constraints
        c_eq, c_ineq = self.constr(x)
        # Return objective function and constraints
        return (self._compute_function(f, c_ineq, s),
                self._compute_constr(c_ineq, c_eq, s))

    def _compute_function(self, f, c_ineq, s):
        # Adjust slack variables to enforce feasibility
        s[self.enforce_feasibility] = -c_ineq[self.enforce_feasibility]
        # Compute logarithm of slack variables
        log_s = [np.log(s_i) if s_i > 0 else -np.inf for s_i in s]
        # Compute barrier objective function
        return f - self.barrier_parameter*np.sum(log_s)

    def _compute_constr(self, c_ineq, c_eq, s):
        # Compute barrier constraints
        return np.hstack((c_eq,
                          c_ineq + s))

    def scaling(self, z):
        """Returns scaling vector.
        Given by:
            scaling = [ones(n_vars), s]
        """
        s = self.get_slack(z)
        # Formulate diagonal elements for scaling vector
        diag_elements = np.hstack((np.ones(self.n_vars), s))

        # Define a linear operator based on diagonal matrix
        def matvec(vec):
            return diag_elements * vec
        # Return linear operator for scaling
        return LinearOperator((self.n_vars+self.n_ineq,
                               self.n_vars+self.n_ineq),
                              matvec)

    def gradient_and_jacobian(self, z):
        """Returns scaled gradient.

        Return scaled gradient:
            gradient = [             grad(x)             ]
                       [ -barrier_parameter*ones(n_ineq) ]
        and scaled Jacobian matrix:
            jacobian = [  jac_eq(x)  0  ]
                       [ jac_ineq(x) S  ]
        Both of them scaled by the previously defined scaling factor.
        """
        # Get variables and slack variables
        x = self.get_variables(z)
        s = self.get_slack(z)
        # Compute gradients
        g = self.grad(x)
        # Compute Jacobian matrices for equality and inequality constraints
        J_eq, J_ineq = self.jac(x)
        # Return scaled gradient and Jacobian
        return (self._compute_gradient(g),
                self._compute_jacobian(J_eq, J_ineq, s))

    def _compute_gradient(self, g):
        # Compute scaled gradient including slack variables
        return np.hstack((g, -self.barrier_parameter*np.ones(self.n_ineq)))
    def _compute_jacobian(self, J_eq, J_ineq, s):
        # 如果没有不等式约束，直接返回等式约束的雅可比矩阵
        if self.n_ineq == 0:
            return J_eq
        else:
            # 如果输入的雅可比矩阵 J_eq 或 J_ineq 是稀疏矩阵
            if sps.issparse(J_eq) or sps.issparse(J_ineq):
                # 将 J_eq 和 J_ineq 转换为 CSR 格式的稀疏矩阵
                J_eq = sps.csr_matrix(J_eq)
                J_ineq = sps.csr_matrix(J_ineq)
                # 调用 _assemble_sparse_jacobian 方法组装稀疏雅可比矩阵
                return self._assemble_sparse_jacobian(J_eq, J_ineq, s)
            else:
                # 创建对角矩阵 S，其中对角线元素由向量 s 给出
                S = np.diag(s)
                # 创建与 J_eq 和 J_ineq 相匹配的零矩阵
                zeros = np.zeros((self.n_eq, self.n_ineq))
                # 如果 J_ineq 是稀疏矩阵，则转换为稠密矩阵
                if sps.issparse(J_ineq):
                    J_ineq = J_ineq.toarray()
                # 如果 J_eq 是稀疏矩阵，则转换为稠密矩阵
                if sps.issparse(J_eq):
                    J_eq = J_eq.toarray()
                # 拼接矩阵 J_eq, zeros, J_ineq, S 成为一个大矩阵
                return np.block([[J_eq, zeros],
                                 [J_ineq, S]])

    def _assemble_sparse_jacobian(self, J_eq, J_ineq, s):
        """Assemble sparse Jacobian given its components.

        Given ``J_eq``, ``J_ineq`` and ``s`` returns:
            jacobian = [ J_eq,     0     ]
                       [ J_ineq, diag(s) ]

        It is equivalent to:
            sps.bmat([[ J_eq,   None    ],
                      [ J_ineq, diag(s) ]], "csr")
        but significantly more efficient for this
        given structure.
        """
        # 获取变量数、不等式约束数和等式约束数
        n_vars, n_ineq, n_eq = self.n_vars, self.n_ineq, self.n_eq
        # 垂直堆叠 J_eq 和 J_ineq 成为一个稀疏 CSR 格式的矩阵 J_aux
        J_aux = sps.vstack([J_eq, J_ineq], "csr")
        # 提取 J_aux 的指针、索引和数据
        indptr, indices, data = J_aux.indptr, J_aux.indices, J_aux.data
        # 构造新的指针数组 new_indptr
        new_indptr = indptr + np.hstack((np.zeros(n_eq, dtype=int),
                                         np.arange(n_ineq+1, dtype=int)))
        size = indices.size+n_ineq
        # 创建新的索引数组 new_indices 和数据数组 new_data
        new_indices = np.empty(size)
        new_data = np.empty(size)
        mask = np.full(size, False, bool)
        # 在 mask 中标记新的索引位置
        mask[new_indptr[-n_ineq:]-1] = True
        new_indices[mask] = n_vars+np.arange(n_ineq)
        new_indices[~mask] = indices
        new_data[mask] = s
        new_data[~mask] = data
        # 创建稀疏 CSR 格式的矩阵 J
        J = sps.csr_matrix((new_data, new_indices, new_indptr),
                           (n_eq + n_ineq, n_vars + n_ineq))
        return J

    def lagrangian_hessian_x(self, z, v):
        """Returns Lagrangian Hessian (in relation to `x`) -> Hx"""
        # 从 z 中获取变量 x
        x = self.get_variables(z)
        # 获取与非线性等式约束相关的拉格朗日乘子 v_eq
        v_eq = v[:self.n_eq]
        # 获取与非线性不等式约束相关的拉格朗日乘子 v_ineq
        v_ineq = v[self.n_eq:self.n_eq+self.n_ineq]
        # 调用 self.lagr_hess 方法计算拉格朗日黑塞矩阵
        lagr_hess = self.lagr_hess
        return lagr_hess(x, v_eq, v_ineq)
    def lagrangian_hessian_s(self, z, v):
        """Returns scaled Lagrangian Hessian (in relation to `s`) -> S Hs S"""
        # 计算松弛变量 s
        s = self.get_slack(z)
        # 使用原始形式:
        #     S Hs S = diag(s)*diag(barrier_parameter/s**2)*diag(s).
        # 参考文献 [1]_ p. 882, 公式 (3.1)
        primal = self.barrier_parameter
        # 使用原始-对偶形式
        #     S Hs S = diag(s)*diag(v/s)*diag(s)
        # 参考文献 [1]_ p. 883, 公式 (3.11)
        primal_dual = v[-self.n_ineq:] * s
        # 对于 v_ineq 的正值，使用原始-对偶形式；对于其余情况，使用原始形式。
        return np.where(v[-self.n_ineq:] > 0, primal_dual, primal)

    def lagrangian_hessian(self, z, v):
        """Returns scaled Lagrangian Hessian"""
        # 计算相对于 x 和 s 的 Hessian
        Hx = self.lagrangian_hessian_x(z, v)
        if self.n_ineq > 0:
            S_Hs_S = self.lagrangian_hessian_s(z, v)

        # 缩放的拉格朗日 Hessian 矩阵:
        #     [ Hx    0    ]
        #     [ 0   S Hs S ]
        def matvec(vec):
            vec_x = self.get_variables(vec)
            vec_s = self.get_slack(vec)
            if self.n_ineq > 0:
                return np.hstack((Hx.dot(vec_x), S_Hs_S * vec_s))
            else:
                return Hx.dot(vec_x)
        return LinearOperator((self.n_vars + self.n_ineq,
                               self.n_vars + self.n_ineq),
                              matvec)

    def stop_criteria(self, state, z, last_iteration_failed,
                      optimality, constr_violation,
                      trust_radius, penalty, cg_info):
        """Stop criteria to the barrier problem.
        The criteria here proposed is similar to formula (2.3)
        from [1]_, p.879.
        """
        x = self.get_variables(z)
        # 如果全局停止条件满足，则设置终止标志为 True 并返回 True
        if self.global_stop_criteria(state, x,
                                     last_iteration_failed,
                                     trust_radius, penalty,
                                     cg_info,
                                     self.barrier_parameter,
                                     self.tolerance):
            self.terminate = True
            return True
        else:
            # 计算梯度条件和变量条件
            g_cond = (optimality < self.tolerance and
                      constr_violation < self.tolerance)
            x_cond = trust_radius < self.xtol
            return g_cond or x_cond
# 定义内点法的信赖域方法，用于解决带约束的优化问题
def tr_interior_point(fun, grad, lagr_hess, n_vars, n_ineq, n_eq,
                      constr, jac, x0, fun0, grad0,
                      constr_ineq0, jac_ineq0, constr_eq0,
                      jac_eq0, stop_criteria,
                      enforce_feasibility, xtol, state,
                      initial_barrier_parameter,
                      initial_tolerance,
                      initial_penalty,
                      initial_trust_radius,
                      factorization_method):
    """Trust-region interior points method.

    Solve problem:
        minimize fun(x)
        subject to: constr_ineq(x) <= 0
                    constr_eq(x) = 0
    using trust-region interior point method described in [1]_.
    """

    # BOUNDARY_PARAMETER controls the decrease on the slack
    # variables. Represents ``tau`` from [1]_ p.885, formula (3.18).
    BOUNDARY_PARAMETER = 0.995

    # BARRIER_DECAY_RATIO controls the decay of the barrier parameter
    # and of the subproblem tolerance. Represents ``theta`` from [1]_ p.879.
    BARRIER_DECAY_RATIO = 0.2

    # TRUST_ENLARGEMENT controls the enlargement on trust radius
    # after each iteration
    TRUST_ENLARGEMENT = 5

    # Default enforce_feasibility
    if enforce_feasibility is None:
        enforce_feasibility = np.zeros(n_ineq, bool)

    # Initial Values
    barrier_parameter = initial_barrier_parameter
    tolerance = initial_tolerance
    trust_radius = initial_trust_radius

    # Define initial value for the slack variables
    s0 = np.maximum(-1.5 * constr_ineq0, np.ones(n_ineq))

    # Define barrier subproblem
    subprob = BarrierSubproblem(
        x0, s0, fun, grad, lagr_hess, n_vars, n_ineq, n_eq, constr, jac,
        barrier_parameter, tolerance, enforce_feasibility,
        stop_criteria, xtol, fun0, grad0, constr_ineq0, jac_ineq0,
        constr_eq0, jac_eq0)

    # Define initial parameter for the first iteration.
    z = np.hstack((x0, s0))
    fun0_subprob, constr0_subprob = subprob.fun0, subprob.constr0
    grad0_subprob, jac0_subprob = subprob.grad0, subprob.jac0

    # Define trust region bounds
    trust_lb = np.hstack((np.full(subprob.n_vars, -np.inf),
                          np.full(subprob.n_ineq, -BOUNDARY_PARAMETER)))
    trust_ub = np.full(subprob.n_vars + subprob.n_ineq, np.inf)

    # Solves a sequence of barrier problems
    while True:
        # 解决 SQP 子问题
        z, state = equality_constrained_sqp(
            subprob.function_and_constraints,  # 子问题的目标函数和约束条件
            subprob.gradient_and_jacobian,     # 子问题的梯度和雅可比矩阵
            subprob.lagrangian_hessian,        # 子问题的拉格朗日 Hessian 矩阵
            z, fun0_subprob, grad0_subprob,
            constr0_subprob, jac0_subprob, subprob.stop_criteria,
            state, initial_penalty, trust_radius,
            factorization_method, trust_lb, trust_ub, subprob.scaling)
        if subprob.terminate:
            break
        # 更新参数
        trust_radius = max(initial_trust_radius,
                           TRUST_ENLARGEMENT*state.tr_radius)
        # TODO: 使用 [2]_ 中更高级的策略来更新这些参数。
        barrier_parameter *= BARRIER_DECAY_RATIO
        tolerance *= BARRIER_DECAY_RATIO
        # 更新障碍问题
        subprob.update(barrier_parameter, tolerance)
        # 计算下一次迭代的初始值
        fun0_subprob, constr0_subprob = subprob.function_and_constraints(z)
        grad0_subprob, jac0_subprob = subprob.gradient_and_jacobian(z)

    # 获取 x 和 s
    x = subprob.get_variables(z)
    return x, state
```