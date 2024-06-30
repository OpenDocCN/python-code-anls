# `D:\src\scipysrc\scipy\scipy\optimize\_trustregion_constr\canonical_constraint.py`

```
    import numpy as np  # 导入 NumPy 库，用于数值计算
    import scipy.sparse as sps  # 导入 SciPy 库中的稀疏矩阵模块

    class CanonicalConstraint:
        """Canonical constraint to use with trust-constr algorithm.

        It represents the set of constraints of the form::

            f_eq(x) = 0
            f_ineq(x) <= 0

        where ``f_eq`` and ``f_ineq`` are evaluated by a single function, see
        below.

        The class is supposed to be instantiated by factory methods, which
        should prepare the parameters listed below.

        Parameters
        ----------
        n_eq, n_ineq : int
            Number of equality and inequality constraints respectively.
        fun : callable
            Function defining the constraints. The signature is
            ``fun(x) -> c_eq, c_ineq``, where ``c_eq`` is ndarray with `n_eq`
            components and ``c_ineq`` is ndarray with `n_ineq` components.
        jac : callable
            Function to evaluate the Jacobian of the constraint. The signature
            is ``jac(x) -> J_eq, J_ineq``, where ``J_eq`` and ``J_ineq`` are
            either ndarray of csr_matrix of shapes (n_eq, n) and (n_ineq, n),
            respectively.
        hess : callable
            Function to evaluate the Hessian of the constraints multiplied
            by Lagrange multipliers, that is
            ``dot(f_eq, v_eq) + dot(f_ineq, v_ineq)``. The signature is
            ``hess(x, v_eq, v_ineq) -> H``, where ``H`` has an implied
            shape (n, n) and provide a matrix-vector product operation
            ``H.dot(p)``.
        keep_feasible : ndarray, shape (n_ineq,)
            Mask indicating which inequality constraints should be kept feasible.
        """
        def __init__(self, n_eq, n_ineq, fun, jac, hess, keep_feasible):
            self.n_eq = n_eq  # 设置等式约束的数量
            self.n_ineq = n_ineq  # 设置不等式约束的数量
            self.fun = fun  # 设置约束函数
            self.jac = jac  # 设置约束雅可比矩阵函数
            self.hess = hess  # 设置约束 Hessian 矩阵函数
            self.keep_feasible = keep_feasible  # 设置保持不等式约束可行性的掩码数组

        @classmethod
        def from_PreparedConstraint(cls, constraint):
            """Create an instance from `PreparedConstrained` object."""
            lb, ub = constraint.bounds  # 从 constraint 对象中获取上下界
            cfun = constraint.fun  # 从 constraint 对象中获取约束函数
            keep_feasible = constraint.keep_feasible  # 从 constraint 对象中获取保持可行性的掩码数组

            if np.all(lb == -np.inf) and np.all(ub == np.inf):
                return cls.empty(cfun.n)  # 如果上下界均为无穷，则返回一个空的约束对象

            if np.all(lb == ub):
                return cls._equal_to_canonical(cfun, lb)  # 如果上下界相等，则将约束转换为等式约束形式
            elif np.all(lb == -np.inf):
                return cls._less_to_canonical(cfun, ub, keep_feasible)  # 如果下界为无穷，则将约束转换为 <= 形式的约束
            elif np.all(ub == np.inf):
                return cls._greater_to_canonical(cfun, lb, keep_feasible)  # 如果上界为无穷，则将约束转换为 >= 形式的约束
            else:
                return cls._interval_to_canonical(cfun, lb, ub, keep_feasible)  # 否则，将约束转换为区间约束形式

        @classmethod
    def empty(cls, n):
        """Create an "empty" instance.

        This "empty" instance is required to allow working with unconstrained
        problems as if they have some constraints.
        """
        # 创建一个空的函数数组，长度为0
        empty_fun = np.empty(0)
        # 创建一个空的雅可比矩阵，形状为 (0, n)
        empty_jac = np.empty((0, n))
        # 创建一个空的稀疏的 Hessian 矩阵，形状为 (n, n)
        empty_hess = sps.csr_matrix((n, n))

        # 定义一个返回空函数数组的函数 fun
        def fun(x):
            return empty_fun, empty_fun

        # 定义一个返回空雅可比矩阵的函数 jac
        def jac(x):
            return empty_jac, empty_jac

        # 定义一个返回空稀疏 Hessian 矩阵的函数 hess
        def hess(x, v_eq, v_ineq):
            return empty_hess

        # 返回一个类的实例，包含空的函数、雅可比矩阵和 Hessian 矩阵，以及空的布尔数组
        return cls(0, 0, fun, jac, hess, np.empty(0, dtype=np.bool_))

    @classmethod
    def concatenate(cls, canonical_constraints, sparse_jacobian):
        """Concatenate multiple `CanonicalConstraint` into one.

        `sparse_jacobian` (bool) determines the Jacobian format of the
        concatenated constraint. Note that items in `canonical_constraints`
        must have their Jacobians in the same format.
        """
        # 定义一个合并多个规范约束的函数 fun
        def fun(x):
            # 如果有规范约束存在
            if canonical_constraints:
                # 分别计算所有规范约束的等式和不等式部分
                eq_all, ineq_all = zip(
                        *[c.fun(x) for c in canonical_constraints])
            else:
                eq_all, ineq_all = [], []

            return np.hstack(eq_all), np.hstack(ineq_all)

        # 根据 sparse_jacobian 的值选择不同的垂直堆叠函数 vstack
        if sparse_jacobian:
            vstack = sps.vstack
        else:
            vstack = np.vstack

        # 定义一个合并多个规范约束的雅可比矩阵的函数 jac
        def jac(x):
            # 如果有规范约束存在
            if canonical_constraints:
                # 分别计算所有规范约束的雅可比矩阵
                eq_all, ineq_all = zip(
                        *[c.jac(x) for c in canonical_constraints])
            else:
                eq_all, ineq_all = [], []

            return vstack(eq_all), vstack(ineq_all)

        # 定义一个合并多个规范约束的 Hessian 矩阵的函数 hess
        def hess(x, v_eq, v_ineq):
            hess_all = []
            index_eq = 0
            index_ineq = 0
            # 遍历所有规范约束
            for c in canonical_constraints:
                # 提取当前规范约束的等式和不等式乘子
                vc_eq = v_eq[index_eq:index_eq + c.n_eq]
                vc_ineq = v_ineq[index_ineq:index_ineq + c.n_ineq]
                # 计算当前规范约束的 Hessian 矩阵
                hess_all.append(c.hess(x, vc_eq, vc_ineq))
                index_eq += c.n_eq
                index_ineq += c.n_ineq

            # 定义一个矩阵向量乘法的函数 matvec，用于构建稀疏的线性操作对象
            def matvec(p):
                result = np.zeros_like(p)
                for h in hess_all:
                    result += h.dot(p)
                return result

            n = x.shape[0]
            return sps.linalg.LinearOperator((n, n), matvec, dtype=float)

        # 计算所有规范约束的等式和不等式的总数
        n_eq = sum(c.n_eq for c in canonical_constraints)
        n_ineq = sum(c.n_ineq for c in canonical_constraints)
        # 拼接所有规范约束的 keep_feasible 属性
        keep_feasible = np.hstack([c.keep_feasible for c in
                                   canonical_constraints])

        # 返回一个类的实例，包含合并后的等式和不等式数目、函数、雅可比矩阵、Hessian 矩阵和布尔数组
        return cls(n_eq, n_ineq, fun, jac, hess, keep_feasible)
    def _equal_to_canonical(cls, cfun, value):
        # 创建一个空的 numpy 数组
        empty_fun = np.empty(0)
        # 获取 cfun 的维度
        n = cfun.n

        # 计算 value 的行数
        n_eq = value.shape[0]
        # 初始化不等式约束数目为 0
        n_ineq = 0
        # 创建一个空的布尔数组
        keep_feasible = np.empty(0, dtype=bool)

        # 如果 cfun 使用稀疏雅可比矩阵
        if cfun.sparse_jacobian:
            # 创建一个空的稀疏矩阵
            empty_jac = sps.csr_matrix((0, n))
        else:
            # 创建一个空的 numpy 数组
            empty_jac = np.empty((0, n))

        # 定义函数 fun，返回值为 cfun.fun(x) - value，和空的函数数组
        def fun(x):
            return cfun.fun(x) - value, empty_fun

        # 定义函数 jac，返回值为 cfun.jac(x) 和 empty_jac
        def jac(x):
            return cfun.jac(x), empty_jac

        # 定义函数 hess，返回值为 cfun.hess(x, v_eq)
        def hess(x, v_eq, v_ineq):
            return cfun.hess(x, v_eq)

        # 重新初始化 empty_fun 和 n 变量
        empty_fun = np.empty(0)
        n = cfun.n
        # 根据 cfun 的稀疏雅可比矩阵属性，初始化 empty_jac
        if cfun.sparse_jacobian:
            empty_jac = sps.csr_matrix((0, n))
        else:
            empty_jac = np.empty((0, n))

        # 返回一个实例化后的 cls 对象，包含给定的参数
        return cls(n_eq, n_ineq, fun, jac, hess, keep_feasible)

    @classmethod
    def _less_to_canonical(cls, cfun, ub, keep_feasible):
        # 创建一个空的 numpy 数组
        empty_fun = np.empty(0)
        # 获取 cfun 的维度
        n = cfun.n
        # 根据 cfun 的稀疏雅可比矩阵属性，初始化 empty_jac
        if cfun.sparse_jacobian:
            empty_jac = sps.csr_matrix((0, n))
        else:
            empty_jac = np.empty((0, n))

        # 计算有限上界的布尔值
        finite_ub = ub < np.inf
        # 初始化等式约束数目为 0
        n_eq = 0
        # 计算有限上界的不等式约束数目
        n_ineq = np.sum(finite_ub)

        # 如果所有的上界都是有限的
        if np.all(finite_ub):
            # 定义函数 fun，返回值为 empty_fun 和 cfun.fun(x) - ub
            def fun(x):
                return empty_fun, cfun.fun(x) - ub

            # 定义函数 jac，返回值为 empty_jac 和 cfun.jac(x)
            def jac(x):
                return empty_jac, cfun.jac(x)

            # 定义函数 hess，返回值为 cfun.hess(x, v_ineq)
            def hess(x, v_eq, v_ineq):
                return cfun.hess(x, v_ineq)
        else:
            # 获取有限上界的索引
            finite_ub = np.nonzero(finite_ub)[0]
            # 根据索引更新 keep_feasible
            keep_feasible = keep_feasible[finite_ub]
            # 根据索引更新 ub
            ub = ub[finite_ub]

            # 定义函数 fun，返回值为 empty_fun 和 cfun.fun(x)[finite_ub] - ub
            def fun(x):
                return empty_fun, cfun.fun(x)[finite_ub] - ub

            # 定义函数 jac，返回值为 empty_jac 和 cfun.jac(x)[finite_ub]
            def jac(x):
                return empty_jac, cfun.jac(x)[finite_ub]

            # 定义函数 hess，返回值为经过更新的 cfun.hess(x, v)
            def hess(x, v_eq, v_ineq):
                v = np.zeros(cfun.m)
                v[finite_ub] = v_ineq
                return cfun.hess(x, v)

        # 返回一个实例化后的 cls 对象，包含给定的参数
        return cls(n_eq, n_ineq, fun, jac, hess, keep_feasible)
    # 定义一个类方法，用于将不等式约束函数转换为规范形式
    def _greater_to_canonical(cls, cfun, lb, keep_feasible):
        # 创建一个空的 numpy 数组，长度为 0
        empty_fun = np.empty(0)
        # 获取约束函数的维度 n
        n = cfun.n
        # 根据约束函数的稀疏性决定创建空的雅可比矩阵的方式
        if cfun.sparse_jacobian:
            empty_jac = sps.csr_matrix((0, n))  # 稀疏矩阵形式
        else:
            empty_jac = np.empty((0, n))  # 密集矩阵形式

        # 计算有限下界的数量
        finite_lb = lb > -np.inf
        # 初始化等式约束和不等式约束数量
        n_eq = 0
        n_ineq = np.sum(finite_lb)

        # 如果所有下界都是有限的
        if np.all(finite_lb):
            # 定义一个函数 fun(x)，返回空的函数值和转换后的不等式约束
            def fun(x):
                return empty_fun, lb - cfun.fun(x)

            # 定义一个函数 jac(x)，返回空的雅可比矩阵和转换后的不等式约束的雅可比矩阵的相反数
            def jac(x):
                return empty_jac, -cfun.jac(x)

            # 定义一个函数 hess(x, v_eq, v_ineq)，返回不等式约束的黑塞矩阵
            def hess(x, v_eq, v_ineq):
                return cfun.hess(x, -v_ineq)
        else:
            # 确定有限下界的索引
            finite_lb = np.nonzero(finite_lb)[0]
            # 根据有限下界的索引更新 keep_feasible 和 lb
            keep_feasible = keep_feasible[finite_lb]
            lb = lb[finite_lb]

            # 定义一个函数 fun(x)，返回空的函数值和转换后的有限下界约束
            def fun(x):
                return empty_fun, lb - cfun.fun(x)[finite_lb]

            # 定义一个函数 jac(x)，返回空的雅可比矩阵和转换后的有限下界约束的雅可比矩阵的相反数
            def jac(x):
                return empty_jac, -cfun.jac(x)[finite_lb]

            # 定义一个函数 hess(x, v_eq, v_ineq)，构造完整的黑塞矩阵
            def hess(x, v_eq, v_ineq):
                v = np.zeros(cfun.m)
                v[finite_lb] = -v_ineq
                return cfun.hess(x, v)

        # 返回转换后的约束对象
        return cls(n_eq, n_ineq, fun, jac, hess, keep_feasible)

    @classmethod
    # 将给定的区间转换为规范形式的静态方法
    def _interval_to_canonical(cls, cfun, lb, ub, keep_feasible):
        # 检查下界是否为负无穷
        lb_inf = lb == -np.inf
        # 检查上界是否为正无穷
        ub_inf = ub == np.inf
        # 检查下界和上界是否相等
        equal = lb == ub
        # 检查下界为负无穷且上界不为正无穷
        less = lb_inf & ~ub_inf
        # 检查上界为正无穷且下界不为负无穷
        greater = ub_inf & ~lb_inf
        # 检查既不相等也不是无穷的区间
        interval = ~equal & ~lb_inf & ~ub_inf

        # 获取相等约束的索引
        equal = np.nonzero(equal)[0]
        # 获取下界为负无穷的约束的索引
        less = np.nonzero(less)[0]
        # 获取上界为正无穷的约束的索引
        greater = np.nonzero(greater)[0]
        # 获取既不相等也不是无穷的区间的索引
        interval = np.nonzero(interval)[0]

        # 计算不同类型约束的数量
        n_less = less.shape[0]
        n_greater = greater.shape[0]
        n_interval = interval.shape[0]
        # 计算总不等式约束的数量
        n_ineq = n_less + n_greater + 2 * n_interval
        # 计算等式约束的数量
        n_eq = equal.shape[0]

        # 重新构造 keep_feasible，以包含所有类型约束的保持可行性信息
        keep_feasible = np.hstack((keep_feasible[less],
                                   keep_feasible[greater],
                                   keep_feasible[interval],
                                   keep_feasible[interval]))

        # 定义目标函数，返回等式约束和不等式约束的值
        def fun(x):
            # 计算目标函数值
            f = cfun.fun(x)
            # 等式约束的残差
            eq = f[equal] - lb[equal]
            # 不等式约束的残差
            le = f[less] - ub[less]
            ge = lb[greater] - f[greater]
            il = f[interval] - ub[interval]
            ig = lb[interval] - f[interval]
            return eq, np.hstack((le, ge, il, ig))

        # 定义雅可比矩阵函数，返回等式约束和不等式约束的雅可比矩阵
        def jac(x):
            # 计算雅可比矩阵
            J = cfun.jac(x)
            # 提取等式约束的雅可比矩阵
            eq = J[equal]
            # 提取下界为负无穷约束的雅可比矩阵
            le = J[less]
            # 提取上界为正无穷约束的雅可比矩阵，同时转换方向
            ge = -J[greater]
            # 提取区间约束的雅可比矩阵
            il = J[interval]
            ig = -il
            # 如果雅可比矩阵是稀疏的，则使用稀疏矩阵操作
            if sps.issparse(J):
                ineq = sps.vstack((le, ge, il, ig))
            else:
                ineq = np.vstack((le, ge, il, ig))
            return eq, ineq

        # 定义黑塞矩阵函数，返回黑塞矩阵
        def hess(x, v_eq, v_ineq):
            n_start = 0
            # 提取不等式约束的下界为负无穷部分的乘子向量
            v_l = v_ineq[n_start:n_start + n_less]
            n_start += n_less
            # 提取不等式约束的上界为正无穷部分的乘子向量
            v_g = v_ineq[n_start:n_start + n_greater]
            n_start += n_greater
            # 提取不等式约束的区间部分的乘子向量
            v_il = v_ineq[n_start:n_start + n_interval]
            n_start += n_interval
            # 提取不等式约束的区间部分的乘子向量，反向
            v_ig = v_ineq[n_start:n_start + n_interval]

            # 初始化乘子向量
            v = np.zeros_like(lb)
            # 设置等式约束的乘子向量
            v[equal] = v_eq
            # 设置下界为负无穷约束的乘子向量
            v[less] = v_l
            # 设置上界为正无穷约束的乘子向量，反向
            v[greater] = -v_g
            # 设置区间约束的乘子向量
            v[interval] = v_il - v_ig

            # 计算黑塞矩阵
            return cfun.hess(x, v)

        # 返回包含所有信息的 cls 对象
        return cls(n_eq, n_ineq, fun, jac, hess, keep_feasible)
# 将约束条件的初始值转换为规范格式

def initial_constraints_as_canonical(n, prepared_constraints, sparse_jacobian):
    """Convert initial values of the constraints to the canonical format.

    The purpose to avoid one additional call to the constraints at the initial
    point. It takes saved values in `PreparedConstraint`, modififies and
    concatenates them to the canonical constraint format.
    """
    # 初始化空列表，用于存储等式约束条件、不等式约束条件及其对应的雅可比矩阵
    c_eq = []
    c_ineq = []
    J_eq = []
    J_ineq = []

    # 遍历预处理后的约束条件列表
    for c in prepared_constraints:
        # 提取约束函数、雅可比矩阵、上下界
        f = c.fun.f
        J = c.fun.J
        lb, ub = c.bounds
        
        # 根据上下界的情况将约束条件分类存储到对应的列表中
        if np.all(lb == ub):  # 等式约束条件
            c_eq.append(f - lb)
            J_eq.append(J)
        elif np.all(lb == -np.inf):  # 仅有上界的不等式约束条件
            finite_ub = ub < np.inf
            c_ineq.append(f[finite_ub] - ub[finite_ub])
            J_ineq.append(J[finite_ub])
        elif np.all(ub == np.inf):  # 仅有下界的不等式约束条件
            finite_lb = lb > -np.inf
            c_ineq.append(lb[finite_lb] - f[finite_lb])
            J_ineq.append(-J[finite_lb])
        else:  # 同时存在上下界的不等式约束条件
            lb_inf = lb == -np.inf
            ub_inf = ub == np.inf
            equal = lb == ub
            less = lb_inf & ~ub_inf
            greater = ub_inf & ~lb_inf
            interval = ~equal & ~lb_inf & ~ub_inf

            c_eq.append(f[equal] - lb[equal])
            c_ineq.append(f[less] - ub[less])
            c_ineq.append(lb[greater] - f[greater])
            c_ineq.append(f[interval] - ub[interval])
            c_ineq.append(lb[interval] - f[interval])

            J_eq.append(J[equal])
            J_ineq.append(J[less])
            J_ineq.append(-J[greater])
            J_ineq.append(J[interval])
            J_ineq.append(-J[interval])

    # 将列表转换为 numpy 数组或者稀疏矩阵
    c_eq = np.hstack(c_eq) if c_eq else np.empty(0)
    c_ineq = np.hstack(c_ineq) if c_ineq else np.empty(0)

    # 根据 sparse_jacobian 参数选择使用稀疏矩阵或者密集矩阵
    if sparse_jacobian:
        vstack = sps.vstack
        empty = sps.csr_matrix((0, n))
    else:
        vstack = np.vstack
        empty = np.empty((0, n))

    # 垂直堆叠等式约束和不等式约束的雅可比矩阵
    J_eq = vstack(J_eq) if J_eq else empty
    J_ineq = vstack(J_ineq) if J_ineq else empty

    # 返回转换后的等式约束值、不等式约束值及其雅可比矩阵
    return c_eq, c_ineq, J_eq, J_ineq
```