# `D:\src\scipysrc\scipy\scipy\optimize\_constraints.py`

```
"""Constraints definition for minimize."""
# 导入所需的库
import numpy as np
from ._hessian_update_strategy import BFGS
from ._differentiable_functions import (
    VectorFunction, LinearVectorFunction, IdentityVectorFunction)
from ._optimize import OptimizeWarning
from warnings import warn, catch_warnings, simplefilter, filterwarnings
from scipy.sparse import issparse

# 定义一个函数，将 numpy 数组转换为标量
def _arr_to_scalar(x):
    # 如果 x 是 numpy 数组，则返回其唯一的元素值。如果数组有多个元素，则抛出异常。
    return x.item() if isinstance(x, np.ndarray) else x

# 定义一个非线性约束类
class NonlinearConstraint:
    """Nonlinear constraint on the variables.

    The constraint has the general inequality form::

        lb <= fun(x) <= ub

    Here the vector of independent variables x is passed as ndarray of shape
    (n,) and ``fun`` returns a vector with m components.

    It is possible to use equal bounds to represent an equality constraint or
    infinite bounds to represent a one-sided constraint.

    Parameters
    ----------
    fun : callable
        The function defining the constraint.
        The signature is ``fun(x) -> array_like, shape (m,)``.
    lb, ub : array_like
        Lower and upper bounds on the constraint. Each array must have the
        shape (m,) or be a scalar, in the latter case a bound will be the same
        for all components of the constraint. Use ``np.inf`` with an
        appropriate sign to specify a one-sided constraint.
        Set components of `lb` and `ub` equal to represent an equality
        constraint. Note that you can mix constraints of different types:
        interval, one-sided or equality, by setting different components of
        `lb` and `ub` as  necessary.
    jac : {callable,  '2-point', '3-point', 'cs'}, optional
        Method of computing the Jacobian matrix (an m-by-n matrix,
        where element (i, j) is the partial derivative of f[i] with
        respect to x[j]).  The keywords {'2-point', '3-point',
        'cs'} select a finite difference scheme for the numerical estimation.
        A callable must have the following signature:
        ``jac(x) -> {ndarray, sparse matrix}, shape (m, n)``.
        Default is '2-point'.
    hess : {callable, '2-point', '3-point', 'cs', HessianUpdateStrategy, None}, optional
        Method for computing the Hessian matrix. The keywords
        {'2-point', '3-point', 'cs'} select a finite difference scheme for
        numerical  estimation.  Alternatively, objects implementing
        `HessianUpdateStrategy` interface can be used to approximate the
        Hessian. Currently available implementations are:

            - `BFGS` (default option)
            - `SR1`

        A callable must return the Hessian matrix of ``dot(fun, v)`` and
        must have the following signature:
        ``hess(x, v) -> {LinearOperator, sparse matrix, array_like}, shape (n, n)``.
        Here ``v`` is ndarray with shape (m,) containing Lagrange multipliers.
    """
    keep_feasible : array_like of bool, optional
        Whether to keep the constraint components feasible throughout
        iterations. A single value set this property for all components.
        Default is False. Has no effect for equality constraints.
    finite_diff_rel_step: None or array_like, optional
        Relative step size for the finite difference approximation. Default is
        None, which will select a reasonable value automatically depending
        on a finite difference scheme.
    finite_diff_jac_sparsity: {None, array_like, sparse matrix}, optional
        Defines the sparsity structure of the Jacobian matrix for finite
        difference estimation, its shape must be (m, n). If the Jacobian has
        only few non-zero elements in *each* row, providing the sparsity
        structure will greatly speed up the computations. A zero entry means
        that a corresponding element in the Jacobian is identically zero.
        If provided, forces the use of 'lsmr' trust-region solver.
        If None (default) then dense differencing will be used.

    Notes
    -----
    Finite difference schemes {'2-point', '3-point', 'cs'} may be used for
    approximating either the Jacobian or the Hessian. We, however, do not allow
    its use for approximating both simultaneously. Hence whenever the Jacobian
    is estimated via finite-differences, we require the Hessian to be estimated
    using one of the quasi-Newton strategies.

    The scheme 'cs' is potentially the most accurate, but requires the function
    to correctly handles complex inputs and be analytically continuable to the
    complex plane. The scheme '3-point' is more accurate than '2-point' but
    requires twice as many operations.

    Examples
    --------
    Constrain ``x[0] < sin(x[1]) + 1.9``

    >>> from scipy.optimize import NonlinearConstraint
    >>> import numpy as np
    >>> con = lambda x: x[0] - np.sin(x[1])
    >>> nlc = NonlinearConstraint(con, -np.inf, 1.9)

    """
    # 初始化非线性约束对象
    def __init__(self, fun, lb, ub, jac='2-point', hess=BFGS(),
                 keep_feasible=False, finite_diff_rel_step=None,
                 finite_diff_jac_sparsity=None):
        self.fun = fun
        self.lb = lb  # 设置下界
        self.ub = ub  # 设置上界
        self.finite_diff_rel_step = finite_diff_rel_step  # 设置有限差分相对步长
        self.finite_diff_jac_sparsity = finite_diff_jac_sparsity  # 设置有限差分雅可比稀疏性
        self.jac = jac  # 设置雅可比矩阵的估算方式
        self.hess = hess  # 设置黑塞矩阵的估算方式
        self.keep_feasible = keep_feasible  # 设置是否保持约束的可行性
    """Linear constraint on the variables.

    The constraint has the general inequality form::

        lb <= A.dot(x) <= ub

    Here the vector of independent variables x is passed as ndarray of shape
    (n,) and the matrix A has shape (m, n).

    It is possible to use equal bounds to represent an equality constraint or
    infinite bounds to represent a one-sided constraint.

    Parameters
    ----------
    A : {array_like, sparse matrix}, shape (m, n)
        Matrix defining the constraint.
    lb, ub : dense array_like, optional
        Lower and upper limits on the constraint. Each array must have the
        shape (m,) or be a scalar, in the latter case a bound will be the same
        for all components of the constraint. Use ``np.inf`` with an
        appropriate sign to specify a one-sided constraint.
        Set components of `lb` and `ub` equal to represent an equality
        constraint. Note that you can mix constraints of different types:
        interval, one-sided or equality, by setting different components of
        `lb` and `ub` as  necessary. Defaults to ``lb = -np.inf``
        and ``ub = np.inf`` (no limits).
    keep_feasible : dense array_like of bool, optional
        Whether to keep the constraint components feasible throughout
        iterations. A single value set this property for all components.
        Default is False. Has no effect for equality constraints.
    """
    # 定义 LinearConstraint 类，表示线性约束条件
    def _input_validation(self):
        # 检查矩阵 A 的维度是否为二维
        if self.A.ndim != 2:
            message = "`A` must have exactly two dimensions."
            raise ValueError(message)

        # 尝试将 lb、ub 和 keep_feasible 广播到与 A.shape[0:1] 一致的形状
        try:
            shape = self.A.shape[0:1]
            self.lb = np.broadcast_to(self.lb, shape)
            self.ub = np.broadcast_to(self.ub, shape)
            self.keep_feasible = np.broadcast_to(self.keep_feasible, shape)
        except ValueError:
            message = ("`lb`, `ub`, and `keep_feasible` must be broadcastable "
                       "to shape `A.shape[0:1]`")
            raise ValueError(message)
    def __init__(self, A, lb=-np.inf, ub=np.inf, keep_feasible=False):
        # 如果输入的 A 不是稀疏矩阵，则将其至少视为二维数组，并转换为 np.float64 类型
        if not issparse(A):
            # 在某些情况下，如果约束条件无效，会发出关于不规则嵌套序列的 VisibleDeprecationWarning
            # 最终会导致错误。`scipy.optimize.milp` 更希望立即报错以便它能够处理，而不是让用户担心。
            with catch_warnings():
                simplefilter("error")
                # 将 A 至少视为二维数组，并转换为 np.float64 类型
                self.A = np.atleast_2d(A).astype(np.float64)
        else:
            # 如果 A 是稀疏矩阵，则直接赋值给 self.A
            self.A = A
        
        # 检查 lb 和 ub 是否是稀疏数组，如果是则抛出 ValueError 异常
        if issparse(lb) or issparse(ub):
            raise ValueError("Constraint limits must be dense arrays.")
        
        # 将 lb 和 ub 至少视为一维数组，并转换为 np.float64 类型
        self.lb = np.atleast_1d(lb).astype(np.float64)
        self.ub = np.atleast_1d(ub).astype(np.float64)

        # 检查 keep_feasible 是否是稀疏数组，如果是则抛出 ValueError 异常
        if issparse(keep_feasible):
            raise ValueError("`keep_feasible` must be a dense array.")
        
        # 将 keep_feasible 至少视为一维数组，并转换为 bool 类型
        self.keep_feasible = np.atleast_1d(keep_feasible).astype(bool)
        
        # 执行输入验证
        self._input_validation()

    def residual(self, x):
        """
        Calculate the residual between the constraint function and the limits

        For a linear constraint of the form::

            lb <= A@x <= ub

        the lower and upper residuals between ``A@x`` and the limits are values
        ``sl`` and ``sb`` such that::

            lb + sl == A@x == ub - sb

        When all elements of ``sl`` and ``sb`` are positive, all elements of
        the constraint are satisfied; a negative element in ``sl`` or ``sb``
        indicates that the corresponding element of the constraint is not
        satisfied.

        Parameters
        ----------
        x: array_like
            Vector of independent variables

        Returns
        -------
        sl, sb : array-like
            The lower and upper residuals
        """
        # 计算约束函数 A@x 与上下界之间的残差
        return self.A@x - self.lb, self.ub - self.A@x
class Bounds:
    """Bounds constraint on the variables.

    The constraint has the general inequality form::

        lb <= x <= ub

    It is possible to use equal bounds to represent an equality constraint or
    infinite bounds to represent a one-sided constraint.

    Parameters
    ----------
    lb, ub : dense array_like, optional
        Lower and upper bounds on independent variables. `lb`, `ub`, and
        `keep_feasible` must be the same shape or broadcastable.
        Set components of `lb` and `ub` equal
        to fix a variable. Use ``np.inf`` with an appropriate sign to disable
        bounds on all or some variables. Note that you can mix constraints of
        different types: interval, one-sided or equality, by setting different
        components of `lb` and `ub` as necessary. Defaults to ``lb = -np.inf``
        and ``ub = np.inf`` (no bounds).
    keep_feasible : dense array_like of bool, optional
        Whether to keep the constraint components feasible throughout
        iterations. Must be broadcastable with `lb` and `ub`.
        Default is False. Has no effect for equality constraints.
    """

    def _input_validation(self):
        # 尝试将 lb、ub 和 keep_feasible 广播到相同的形状
        try:
            res = np.broadcast_arrays(self.lb, self.ub, self.keep_feasible)
            self.lb, self.ub, self.keep_feasible = res
        except ValueError:
            # 如果广播失败，抛出错误
            message = "`lb`, `ub`, and `keep_feasible` must be broadcastable."
            raise ValueError(message)

    def __init__(self, lb=-np.inf, ub=np.inf, keep_feasible=False):
        # 如果 lb 或 ub 是稀疏矩阵，则抛出值错误
        if issparse(lb) or issparse(ub):
            raise ValueError("Lower and upper bounds must be dense arrays.")
        # 将 lb 和 ub 至少转换为一维数组
        self.lb = np.atleast_1d(lb)
        self.ub = np.atleast_1d(ub)

        # 如果 keep_feasible 是稀疏矩阵，则抛出值错误
        if issparse(keep_feasible):
            raise ValueError("`keep_feasible` must be a dense array.")
        # 将 keep_feasible 至少转换为一维布尔数组并强制类型转换为布尔类型
        self.keep_feasible = np.atleast_1d(keep_feasible).astype(bool)
        # 进行输入验证
        self._input_validation()

    def __repr__(self):
        # 返回表示对象的字符串表示形式
        start = f"{type(self).__name__}({self.lb!r}, {self.ub!r}"
        if np.any(self.keep_feasible):
            end = f", keep_feasible={self.keep_feasible!r})"
        else:
            end = ")"
        return start + end
    # 定义一个方法来计算变量与边界之间的残差（即差值）

    """Calculate the residual (slack) between the input and the bounds

    For a bound constraint of the form::

        lb <= x <= ub

    the lower and upper residuals between `x` and the bounds are values
    ``sl`` and ``sb`` such that::

        lb + sl == x == ub - sb

    When all elements of ``sl`` and ``sb`` are positive, all elements of
    ``x`` lie within the bounds; a negative element in ``sl`` or ``sb``
    indicates that the corresponding element of ``x`` is out of bounds.

    Parameters
    ----------
    x: array_like
        Vector of independent variables

    Returns
    -------
    sl, sb : array-like
        The lower and upper residuals
    """
    
    # 返回一个元组，包含 x 与 lb 的差值和 ub 与 x 的差值
    return x - self.lb, self.ub - x
# 定义一个名为 PreparedConstraint 的类，用于封装从用户定义的约束创建的准备好的约束对象。
# 在创建时会检查约束定义是否有效，并验证初始点是否可行。如果成功创建，该对象将包含以下列出的属性。

class PreparedConstraint:
    """Constraint prepared from a user defined constraint.

    On creation it will check whether a constraint definition is valid and
    the initial point is feasible. If created successfully, it will contain
    the attributes listed below.

    Parameters
    ----------
    constraint : {NonlinearConstraint, LinearConstraint`, Bounds}
        Constraint to check and prepare.
    x0 : array_like
        Initial vector of independent variables.
    sparse_jacobian : bool or None, optional
        If bool, then the Jacobian of the constraint will be converted
        to the corresponded format if necessary. If None (default), such
        conversion is not made.
    finite_diff_bounds : 2-tuple, optional
        Lower and upper bounds on the independent variables for the finite
        difference approximation, if applicable. Defaults to no bounds.

    Attributes
    ----------
    fun : {VectorFunction, LinearVectorFunction, IdentityVectorFunction}
        Function defining the constraint wrapped by one of the convenience
        classes.
    bounds : 2-tuple
        Contains lower and upper bounds for the constraints --- lb and ub.
        These are converted to ndarray and have a size equal to the number of
        the constraints.
    keep_feasible : ndarray
         Array indicating which components must be kept feasible with a size
         equal to the number of the constraints.
    """
    def __init__(self, constraint, x0, sparse_jacobian=None,
                 finite_diff_bounds=(-np.inf, np.inf)):
        # 根据传入的约束类型不同，选择合适的向量函数类来初始化
        if isinstance(constraint, NonlinearConstraint):
            # 对于非线性约束，使用VectorFunction类初始化
            fun = VectorFunction(constraint.fun, x0,
                                 constraint.jac, constraint.hess,
                                 constraint.finite_diff_rel_step,
                                 constraint.finite_diff_jac_sparsity,
                                 finite_diff_bounds, sparse_jacobian)
        elif isinstance(constraint, LinearConstraint):
            # 对于线性约束，使用LinearVectorFunction类初始化
            fun = LinearVectorFunction(constraint.A, x0, sparse_jacobian)
        elif isinstance(constraint, Bounds):
            # 对于Bounds类型的约束，使用IdentityVectorFunction类初始化
            fun = IdentityVectorFunction(x0, sparse_jacobian)
        else:
            # 如果传入的约束类型不被支持，则抛出ValueError异常
            raise ValueError("`constraint` of an unknown type is passed.")

        # 获取向量函数类中的约束数量m
        m = fun.m

        # 将约束的下界、上界、以及是否保持可行性的布尔数组转换为NumPy数组
        lb = np.asarray(constraint.lb, dtype=float)
        ub = np.asarray(constraint.ub, dtype=float)
        keep_feasible = np.asarray(constraint.keep_feasible, dtype=bool)

        # 将lb、ub、keep_feasible广播到长度为m的数组
        lb = np.broadcast_to(lb, m)
        ub = np.broadcast_to(ub, m)
        keep_feasible = np.broadcast_to(keep_feasible, m)

        # 检查keep_feasible数组的形状是否为(m,)，如果不是则抛出异常
        if keep_feasible.shape != (m,):
            raise ValueError("`keep_feasible` has a wrong shape.")

        # 创建一个布尔掩码，用于标识保持可行性且不等式约束不满足的情况
        mask = keep_feasible & (lb != ub)

        # 计算当前x0对应的函数值
        f0 = fun.f

        # 检查是否有任何不满足约束条件的情况，如果有则抛出异常
        if np.any(f0[mask] < lb[mask]) or np.any(f0[mask] > ub[mask]):
            raise ValueError("`x0` is infeasible with respect to some "
                             "inequality constraint with `keep_feasible` "
                             "set to True.")

        # 将初始化得到的向量函数fun、约束的下界上界lb、ub以及keep_feasible存储在对象属性中
        self.fun = fun
        self.bounds = (lb, ub)
        self.keep_feasible = keep_feasible

    def violation(self, x):
        """How much the constraint is exceeded by.

        Parameters
        ----------
        x : array-like
            Vector of independent variables

        Returns
        -------
        excess : array-like
            How much the constraint is exceeded by, for each of the
            constraints specified by `PreparedConstraint.fun`.
        """
        with catch_warnings():
            # 忽略下面的警告，它在计算约束违反程度时并不重要
            # UserWarning: delta_grad == 0.0. Check if the approximated
            # function is linear
            filterwarnings("ignore", "delta_grad", UserWarning)
            # 计算当前变量向量x下的约束函数值
            ev = self.fun.fun(np.asarray(x))

        # 计算超出下界的部分
        excess_lb = np.maximum(self.bounds[0] - ev, 0)
        # 计算超出上界的部分
        excess_ub = np.maximum(ev - self.bounds[1], 0)

        # 返回超出约束的总量
        return excess_lb + excess_ub
# 将新边界表示转换为旧边界表示的函数
def new_bounds_to_old(lb, ub, n):
    """Convert the new bounds representation to the old one.

    The new representation is a tuple (lb, ub) and the old one is a list
    containing n tuples, ith containing lower and upper bound on a ith
    variable.
    If any of the entries in lb/ub are -np.inf/np.inf they are replaced by
    None.
    """
    # 将 lb 和 ub 广播到长度为 n 的数组
    lb = np.broadcast_to(lb, n)
    ub = np.broadcast_to(ub, n)

    # 将 lb 中的 -np.inf 替换为 None，并转换为浮点数列表
    lb = [float(x) if x > -np.inf else None for x in lb]
    # 将 ub 中的 np.inf 替换为 None，并转换为浮点数列表
    ub = [float(x) if x < np.inf else None for x in ub]

    return list(zip(lb, ub))


# 将旧边界表示转换为新边界表示的函数
def old_bound_to_new(bounds):
    """Convert the old bounds representation to the new one.

    The new representation is a tuple (lb, ub) and the old one is a list
    containing n tuples, ith containing lower and upper bound on a ith
    variable.
    If any of the entries in lb/ub are None they are replaced by
    -np.inf/np.inf.
    """
    # 将 bounds 解压缩为 lb 和 ub
    lb, ub = zip(*bounds)

    # 将 lb 中的 None 替换为 -np.inf，并转换为浮点数 numpy 数组
    lb = np.array([float(_arr_to_scalar(x)) if x is not None else -np.inf
                   for x in lb])
    # 将 ub 中的 None 替换为 np.inf，并转换为浮点数 numpy 数组
    ub = np.array([float(_arr_to_scalar(x)) if x is not None else np.inf
                   for x in ub])

    return lb, ub


# 严格化边界，根据 keep_feasible 的需求保留可行的边界
def strict_bounds(lb, ub, keep_feasible, n_vars):
    """Remove bounds which are not asked to be kept feasible."""
    # 将 lb 和 ub 重新调整大小为 n_vars，转换为浮点数数组
    strict_lb = np.resize(lb, n_vars).astype(float)
    strict_ub = np.resize(ub, n_vars).astype(float)
    keep_feasible = np.resize(keep_feasible, n_vars)
    # 根据 keep_feasible 条件，将不需要保持可行的边界设置为 -np.inf 和 np.inf
    strict_lb[~keep_feasible] = -np.inf
    strict_ub[~keep_feasible] = np.inf
    return strict_lb, strict_ub


# 将新风格约束对象转换为旧风格约束字典的函数
def new_constraint_to_old(con, x0):
    """
    Converts new-style constraint objects to old-style constraint dictionaries.
    """
    if isinstance(con, NonlinearConstraint):
        if (con.finite_diff_jac_sparsity is not None or
                con.finite_diff_rel_step is not None or
                not isinstance(con.hess, BFGS) or  # misses user specified BFGS
                con.keep_feasible):
            # 如果存在特定的参数，警告并忽略这些选项
            warn("Constraint options `finite_diff_jac_sparsity`, "
                 "`finite_diff_rel_step`, `keep_feasible`, and `hess`"
                 "are ignored by this method.",
                 OptimizeWarning, stacklevel=3)

        fun = con.fun
        if callable(con.jac):
            jac = con.jac
        else:
            jac = None

    else:  # LinearConstraint
        if np.any(con.keep_feasible):
            # 如果存在 keep_feasible 参数，警告并忽略这些选项
            warn("Constraint option `keep_feasible` is ignored by this method.",
                 OptimizeWarning, stacklevel=3)

        A = con.A
        if issparse(A):
            A = A.toarray()

        # 定义线性约束函数 fun(x)，返回 A 和 x 的点乘结果
        def fun(x):
            return np.dot(A, x)

        # 定义线性约束雅可比矩阵函数 jac(x)，返回 A
        def jac(x):
            return A

    # FIXME: when bugs in VectorFunction/LinearVectorFunction are worked out,
    # use pcon.fun.fun and pcon.fun.jac. Until then, get fun/jac above.
    # 使用给定的约束条件和初始点创建预处理的约束对象
    pcon = PreparedConstraint(con, x0)
    # 获取约束的上下界
    lb, ub = pcon.bounds

    # 检查是否存在完全相等的约束
    i_eq = lb == ub
    # 检查下界和上界是否不相等，并将结果存储在布尔数组中
    i_bound_below = np.logical_xor(lb != -np.inf, i_eq)
    # 检查上界和下界是否不相等，并将结果存储在布尔数组中
    i_bound_above = np.logical_xor(ub != np.inf, i_eq)
    # 检查约束是否完全未限制
    i_unbounded = np.logical_and(lb == -np.inf, ub == np.inf)

    # 如果存在未限制的约束，则发出警告
    if np.any(i_unbounded):
        warn("At least one constraint is unbounded above and below. Such "
             "constraints are ignored.",
             OptimizeWarning, stacklevel=3)

    # 初始化等式约束列表
    ceq = []
    # 如果存在完全相等的约束
    if np.any(i_eq):
        # 定义等式约束的函数
        def f_eq(x):
            # 计算目标函数在当前点的取值，并转换为一维数组
            y = np.array(fun(x)).flatten()
            # 返回等式约束的残差
            return y[i_eq] - lb[i_eq]
        # 将等式约束添加到列表中
        ceq = [{"type": "eq", "fun": f_eq}]

        # 如果提供了雅可比矩阵函数
        if jac is not None:
            # 定义等式约束的雅可比矩阵函数
            def j_eq(x):
                # 计算雅可比矩阵，并将稀疏矩阵转换为稠密数组
                dy = jac(x)
                if issparse(dy):
                    dy = dy.toarray()
                dy = np.atleast_2d(dy)
                return dy[i_eq, :]
            # 将雅可比矩阵函数添加到等式约束中
            ceq[0]["jac"] = j_eq

    # 初始化不等式约束列表
    cineq = []
    # 计算下界和上界不同的约束数量
    n_bound_below = np.sum(i_bound_below)
    n_bound_above = np.sum(i_bound_above)
    # 如果存在下界或上界不同的约束
    if n_bound_below + n_bound_above:
        # 定义不等式约束的函数
        def f_ineq(x):
            # 初始化残差数组
            y = np.zeros(n_bound_below + n_bound_above)
            # 计算目标函数在当前点的取值，并转换为一维数组
            y_all = np.array(fun(x)).flatten()
            # 计算下界不同的部分的残差
            y[:n_bound_below] = y_all[i_bound_below] - lb[i_bound_below]
            # 计算上界不同的部分的残差
            y[n_bound_below:] = -(y_all[i_bound_above] - ub[i_bound_above])
            return y
        # 将不等式约束添加到列表中
        cineq = [{"type": "ineq", "fun": f_ineq}]

        # 如果提供了雅可比矩阵函数
        if jac is not None:
            # 定义不等式约束的雅可比矩阵函数
            def j_ineq(x):
                # 初始化雅可比矩阵
                dy = np.zeros((n_bound_below + n_bound_above, len(x0)))
                # 计算雅可比矩阵，并将稀疏矩阵转换为稠密数组
                dy_all = jac(x)
                if issparse(dy_all):
                    dy_all = dy_all.toarray()
                dy_all = np.atleast_2d(dy_all)
                # 设置下界不同部分的雅可比矩阵
                dy[:n_bound_below, :] = dy_all[i_bound_below]
                # 设置上界不同部分的雅可比矩阵
                dy[n_bound_below:, :] = -dy_all[i_bound_above]
                return dy
            # 将雅可比矩阵函数添加到不等式约束中
            cineq[0]["jac"] = j_ineq

    # 组合所有的约束：等式约束和不等式约束
    old_constraints = ceq + cineq

    # 如果约束的数量大于1，则发出警告
    if len(old_constraints) > 1:
        warn("Equality and inequality constraints are specified in the same "
             "element of the constraint list. For efficient use with this "
             "method, equality and inequality constraints should be specified "
             "in separate elements of the constraint list. ",
             OptimizeWarning, stacklevel=3)
    # 返回所有约束的列表
    return old_constraints
def old_constraint_to_new(ic, con):
    """
    Converts old-style constraint dictionaries to new-style constraint objects.
    """
    # 尝试获取约束类型
    try:
        ctype = con['type'].lower()  # 获取约束类型并转换为小写
    except KeyError as e:
        raise KeyError('Constraint %d has no type defined.' % ic) from e  # 如果缺少'type'键，抛出 KeyError
    except TypeError as e:
        raise TypeError(
            'Constraints must be a sequence of dictionaries.'
        ) from e  # 如果参数 con 不是字典序列，抛出 TypeError
    except AttributeError as e:
        raise TypeError("Constraint's type must be a string.") from e  # 如果约束类型不是字符串，抛出 TypeError
    else:
        if ctype not in ['eq', 'ineq']:
            raise ValueError("Unknown constraint type '%s'." % con['type'])  # 如果约束类型不是 'eq' 或 'ineq'，抛出 ValueError

    # 检查是否定义了约束函数
    if 'fun' not in con:
        raise ValueError('Constraint %d has no function defined.' % ic)  # 如果没有定义约束函数，抛出 ValueError

    lb = 0
    if ctype == 'eq':
        ub = 0
    else:
        ub = np.inf  # 设置上界为无穷大（np.inf）

    jac = '2-point'
    if 'args' in con:
        args = con['args']
        def fun(x):
            return con["fun"](x, *args)  # 定义带参数的约束函数
        if 'jac' in con:
            def jac(x):
                return con["jac"](x, *args)  # 定义带参数的雅可比矩阵函数
    else:
        fun = con['fun']  # 使用原始的约束函数
        if 'jac' in con:
            jac = con['jac']  # 使用指定的雅可比矩阵函数

    return NonlinearConstraint(fun, lb, ub, jac)  # 返回新的约束对象
```