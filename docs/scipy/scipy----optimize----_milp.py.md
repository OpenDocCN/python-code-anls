# `D:\src\scipysrc\scipy\scipy\optimize\_milp.py`

```
# 导入警告模块，用于处理警告信息
import warnings
# 导入 NumPy 库，并使用别名 np
import numpy as np
# 从 scipy.sparse 模块中导入 csc_array、vstack 和 issparse 函数
from scipy.sparse import csc_array, vstack, issparse
# 从 scipy._lib._util 模块中导入 VisibleDeprecationWarning 警告
from scipy._lib._util import VisibleDeprecationWarning
# 导入 _highs_wrapper 模块中的 _highs_wrapper 类，忽略类型未指定和导入未找到的警告
from ._highs._highs_wrapper import _highs_wrapper  # type: ignore[import-not-found,import-untyped]
# 从 ._constraints 模块中导入 LinearConstraint 和 Bounds 类
from ._constraints import LinearConstraint, Bounds
# 从 ._optimize 模块中导入 OptimizeResult 类
from ._optimize import OptimizeResult
# 从 ._linprog_highs 模块中导入 _highs_to_scipy_status_message 函数

def _constraints_to_components(constraints):
    """
    Convert sequence of constraints to a single set of components A, b_l, b_u.

    `constraints` could be

    1. A LinearConstraint
    2. A tuple representing a LinearConstraint
    3. An invalid object
    4. A sequence of composed entirely of objects of type 1/2
    5. A sequence containing at least one object of type 3

    We want to accept 1, 2, and 4 and reject 3 and 5.
    """
    # 提示消息，用于非法输入时的异常处理
    message = ("`constraints` (or each element within `constraints`) must be "
               "convertible into an instance of "
               "`scipy.optimize.LinearConstraint`.")
    # 初始化空列表，用于存储 A、b_l 和 b_u 的组件
    As = []
    b_ls = []
    b_us = []

    # 如果 constraints 是 LinearConstraint 类型，则将其转换为列表形式，标准化为 case 4
    if isinstance(constraints, LinearConstraint):
        constraints = [constraints]
    else:
        # 如果 constraints 不可迭代，则抛出 ValueError 异常
        try:
            iter(constraints)
        except TypeError as exc:
            raise ValueError(message) from exc

        # 如果 constraints 长度为 3，则尝试将其转换为 LinearConstraint 类型的单元素列表
        if len(constraints) == 3:
            try:
                constraints = [LinearConstraint(*constraints)]
            except (TypeError, ValueError, VisibleDeprecationWarning):
                pass

    # 处理 cases 4/5
    for constraint in constraints:
        # 如果 constraint 不是 LinearConstraint 类型或不能表示 LinearConstraint 类型，则视为无效
        if not isinstance(constraint, LinearConstraint):
            try:
                constraint = LinearConstraint(*constraint)
            except TypeError as exc:
                raise ValueError(message) from exc
        # 将 constraint.A 转换为 csc 格式的稀疏矩阵，添加到 As 列表
        As.append(csc_array(constraint.A))
        # 将 constraint.lb 转换为至少为 1 维的 float64 类型数组，添加到 b_ls 列表
        b_ls.append(np.atleast_1d(constraint.lb).astype(np.float64))
        # 将 constraint.ub 转换为至少为 1 维的 float64 类型数组，添加到 b_us 列表
        b_us.append(np.atleast_1d(constraint.ub).astype(np.float64))

    # 如果 As 列表长度大于 1，则使用 vstack 将其堆叠为一个 csc 格式的稀疏矩阵 A
    if len(As) > 1:
        A = vstack(As, format="csc")
        # 连接 b_ls 列表中的所有元素，形成 b_l 数组
        b_l = np.concatenate(b_ls)
        # 连接 b_us 列表中的所有元素，形成 b_u 数组
        b_u = np.concatenate(b_us)
    else:  # 避免不必要的复制
        # 如果 As 列表只有一个元素，则直接赋值给 A
        A = As[0]
        b_l = b_ls[0]
        b_u = b_us[0]

    # 返回 A、b_l 和 b_u 组成的元组
    return A, b_l, b_u


def _milp_iv(c, integrality, bounds, constraints, options):
    # 目标函数 IV
    # 如果 c 是稀疏数组，则抛出 ValueError 异常
    if issparse(c):
        raise ValueError("`c` must be a dense array.")
    # 将 c 转换为至少为 1 维的 float64 类型数组
    c = np.atleast_1d(c).astype(np.float64)
    # 检查输入向量 `c` 是否为一维且包含有限数值，如果不是则引发错误
    if c.ndim != 1 or c.size == 0 or not np.all(np.isfinite(c)):
        message = ("`c` must be a one-dimensional array of finite numbers "
                   "with at least one element.")
        raise ValueError(message)

    # 检查 `integrality` 是否为稠密数组，如果是稀疏数组则引发错误
    if issparse(integrality):
        raise ValueError("`integrality` must be a dense array.")
    message = ("`integrality` must contain integers 0-3 and be broadcastable "
               "to `c.shape`.")
    # 如果 `integrality` 为 None，则设为 0
    if integrality is None:
        integrality = 0
    try:
        # 将 `integrality` 广播到与 `c` 相同的形状，并转换为 uint8 类型
        integrality = np.broadcast_to(integrality, c.shape).astype(np.uint8)
    except ValueError:
        raise ValueError(message)
    # 检查 `integrality` 的最小和最大值是否在 0 到 3 之间，否则引发错误
    if integrality.min() < 0 or integrality.max() > 3:
        raise ValueError(message)

    # 处理 `bounds` 参数
    # 如果 `bounds` 为 None，则设为默认的 Bounds(0, np.inf)
    if bounds is None:
        bounds = Bounds(0, np.inf)
    elif not isinstance(bounds, Bounds):
        # 尝试将 `bounds` 转换为 Bounds 类型，如果失败则引发错误
        message = ("`bounds` must be convertible into an instance of "
                   "`scipy.optimize.Bounds`.")
        try:
            bounds = Bounds(*bounds)
        except TypeError as exc:
            raise ValueError(message) from exc

    try:
        # 将 `bounds.lb` 和 `bounds.ub` 广播到与 `c` 相同的形状，并转换为 float64 类型
        lb = np.broadcast_to(bounds.lb, c.shape).astype(np.float64)
        ub = np.broadcast_to(bounds.ub, c.shape).astype(np.float64)
    except (ValueError, TypeError) as exc:
        message = ("`bounds.lb` and `bounds.ub` must contain reals and "
                   "be broadcastable to `c.shape`.")
        raise ValueError(message) from exc

    # 处理 `constraints` 参数
    # 如果 `constraints` 为空，则设为空的 LinearConstraint
    if not constraints:
        constraints = [LinearConstraint(np.empty((0, c.size)),
                                        np.empty((0,)), np.empty((0,)))]
    try:
        # 将 `constraints` 转换为其组成部分 A, b_l, b_u
        A, b_l, b_u = _constraints_to_components(constraints)
    except ValueError as exc:
        message = ("`constraints` (or each element within `constraints`) must "
                   "be convertible into an instance of "
                   "`scipy.optimize.LinearConstraint`.")
        raise ValueError(message) from exc

    # 检查 A 的形状是否与 b_l 的长度和 c 的长度相符，不符则引发错误
    if A.shape != (b_l.size, c.size):
        message = "The shape of `A` must be (len(b_l), len(c))."
        raise ValueError(message)
    
    # 提取稀疏矩阵 A 的指针、索引和数据，并将数据转换为 float64 类型
    indptr, indices, data = A.indptr, A.indices, A.data.astype(np.float64)

    # 处理 `options` 参数
    # 如果 `options` 为 None，则设为空字典
    options = options or {}
    supported_options = {'disp', 'presolve', 'time_limit', 'node_limit',
                         'mip_rel_gap'}
    unsupported_options = set(options).difference(supported_options)
    # 如果有不支持的选项，则发出警告并传递给 HiGHS
    if unsupported_options:
        message = (f"Unrecognized options detected: {unsupported_options}. "
                   "These will be passed to HiGHS verbatim.")
        warnings.warn(message, RuntimeWarning, stacklevel=3)
    # 提取并设定 options_iv 字典的内容
    options_iv = {'log_to_console': options.pop("disp", False),
                  'mip_max_nodes': options.pop("node_limit", None)}
    options_iv.update(options)

    # 返回最终结果：c, integrality, lb, ub, indptr, indices, data, b_l, b_u, options_iv
    return c, integrality, lb, ub, indptr, indices, data, b_l, b_u, options_iv
# 定义混合整数线性规划函数
def milp(c, *, integrality=None, bounds=None, constraints=None, options=None):
    r"""
    Mixed-integer linear programming

    Solves problems of the following form:

    .. math::

        \min_x \ & c^T x \\
        \mbox{such that} \ & b_l \leq A x \leq b_u,\\
        & l \leq x \leq u, \\
        & x_i \in \mathbb{Z}, i \in X_i

    where :math:`x` is a vector of decision variables;
    :math:`c`, :math:`b_l`, :math:`b_u`, :math:`l`, and :math:`u` are vectors;
    :math:`A` is a matrix, and :math:`X_i` is the set of indices of
    decision variables that must be integral. (In this context, a
    variable that can assume only integer values is said to be "integral";
    it has an "integrality" constraint.)

    Alternatively, that's:

    minimize::

        c @ x

    such that::

        b_l <= A @ x <= b_u
        l <= x <= u
        Specified elements of x must be integers

    By default, ``l = 0`` and ``u = np.inf`` unless specified with
    ``bounds``.

    Parameters
    ----------
    c : 1D dense array_like
        The coefficients of the linear objective function to be minimized.
        `c` is converted to a double precision array before the problem is
        solved.
    integrality : 1D dense array_like, optional
        Indicates the type of integrality constraint on each decision variable.

        ``0`` : Continuous variable; no integrality constraint.

        ``1`` : Integer variable; decision variable must be an integer
        within `bounds`.

        ``2`` : Semi-continuous variable; decision variable must be within
        `bounds` or take value ``0``.

        ``3`` : Semi-integer variable; decision variable must be an integer
        within `bounds` or take value ``0``.

        By default, all variables are continuous. `integrality` is converted
        to an array of integers before the problem is solved.

    bounds : scipy.optimize.Bounds, optional
        Bounds on the decision variables. Lower and upper bounds are converted
        to double precision arrays before the problem is solved. The
        ``keep_feasible`` parameter of the `Bounds` object is ignored. If
        not specified, all decision variables are constrained to be
        non-negative.
    constraints : sequence of scipy.optimize.LinearConstraint, optional
        Linear constraints of the optimization problem. Arguments may be
        one of the following:

        1. A single `LinearConstraint` object
        2. A single tuple that can be converted to a `LinearConstraint` object
           as ``LinearConstraint(*constraints)``
        3. A sequence composed entirely of objects of type 1. and 2.

        Before the problem is solved, all values are converted to double
        precision, and the matrices of constraint coefficients are converted to
        instances of `scipy.sparse.csc_array`. The ``keep_feasible`` parameter
        of `LinearConstraint` objects is ignored.
    options : dict, optional
        A dictionary of solver options. The following keys are recognized.

        disp : bool (default: ``False``)
            Set to ``True`` if indicators of optimization status are to be
            printed to the console during optimization.
        node_limit : int, optional
            The maximum number of nodes (linear program relaxations) to solve
            before stopping. Default is no maximum number of nodes.
        presolve : bool (default: ``True``)
            Presolve attempts to identify trivial infeasibilities,
            identify trivial unboundedness, and simplify the problem before
            sending it to the main solver.
        time_limit : float, optional
            The maximum number of seconds allotted to solve the problem.
            Default is no time limit.
        mip_rel_gap : float, optional
            Termination criterion for MIP solver: solver will terminate when
            the gap between the primal objective value and the dual objective
            bound, scaled by the primal objective value, is <= mip_rel_gap.

    Returns
    -------
    res : OptimizeResult
        An instance of :class:`scipy.optimize.OptimizeResult`. The object
        is guaranteed to have the following attributes.

        status : int
            An integer representing the exit status of the algorithm.

            ``0`` : Optimal solution found.

            ``1`` : Iteration or time limit reached.

            ``2`` : Problem is infeasible.

            ``3`` : Problem is unbounded.

            ``4`` : Other; see message for details.

        success : bool
            ``True`` when an optimal solution is found and ``False`` otherwise.

        message : str
            A string descriptor of the exit status of the algorithm.

        The following attributes will also be present, but the values may be
        ``None``, depending on the solution status.

        x : ndarray
            The values of the decision variables that minimize the
            objective function while satisfying the constraints.
        fun : float
            The optimal value of the objective function ``c @ x``.
        mip_node_count : int
            The number of subproblems or "nodes" solved by the MILP solver.
        mip_dual_bound : float
            The MILP solver's final estimate of the lower bound on the optimal
            solution.
        mip_gap : float
            The difference between the primal objective value and the dual
            objective bound, scaled by the primal objective value.

    Notes
    -----
    `milp` is a wrapper of the HiGHS linear optimization software [1]_. The
    algorithm is deterministic, and it typically finds the global optimum of
    moderately challenging mixed-integer linear programs (when it exists).

    References
    ----------
    """
    根据给定的 MILP 问题参数调用 `_milp_iv` 函数，并返回其结果。
    `args_iv` 包含了 `_milp_iv` 函数返回的各种参数。

    Parameters:
    ----------
    c : array_like
        决策变量的线性目标函数系数。
    integrality : array_like
        决策变量的整数约束。
    bounds : array_like
        决策变量的界限。
    constraints : LinearConstraint
        线性约束条件。
    options : dict
        MILP 求解器的选项设置。

    Returns:
    -------
    tuple
        返回 `_milp_iv` 函数的结果，包括调整后的参数和设置。

    Examples
    --------
    以下是 MILP 问题的示例，详细说明了如何调用该函数来解决整数线性规划问题。
    """

    # 调用 `_milp_iv` 函数，获取 MILP 问题的各种参数
    args_iv = _milp_iv(c, integrality, bounds, constraints, options)

    # 将返回的参数解包赋值给对应的变量
    c, integrality, lb, ub, indptr, indices, data, b_l, b_u, options = args_iv

    # 使用 `_highs_wrapper` 函数调用 HiGHS 求解器，解决 MILP 问题
    highs_res = _highs_wrapper(c, indptr, indices, data, b_l, b_u,
                               lb, ub, integrality, options)
    # 初始化空字典 res 用于存储优化结果
    res = {}

    # 从 highs_res 字典中获取 'status' 键对应的值，如果不存在则为 None
    highs_status = highs_res.get('status', None)
    
    # 从 highs_res 字典中获取 'message' 键对应的值，如果不存在则为 None
    highs_message = highs_res.get('message', None)
    
    # 调用 _highs_to_scipy_status_message 函数，将 highs_status 和 highs_message 转换为 scipy 风格的状态和消息
    status, message = _highs_to_scipy_status_message(highs_status,
                                                     highs_message)
    
    # 将转换后的状态和消息分别存储到 res 字典中的 'status' 和 'message' 键
    res['status'] = status
    res['message'] = message
    
    # 将 res 字典中的 'success' 键设为状态是否为 0 的布尔值
    res['success'] = (status == 0)
    
    # 从 highs_res 字典中获取 'x' 键对应的值，如果不存在则为 None
    x = highs_res.get('x', None)
    
    # 将 x 转换为 NumPy 数组（如果 x 不为 None），否则设为 None，并存储到 res 字典中的 'x' 键
    res['x'] = np.array(x) if x is not None else None
    
    # 从 highs_res 字典中获取 'fun' 键对应的值，如果不存在则为 None，并存储到 res 字典中的 'fun' 键
    res['fun'] = highs_res.get('fun', None)
    
    # 从 highs_res 字典中获取 'mip_node_count' 键对应的值，如果不存在则为 None，并存储到 res 字典中的 'mip_node_count' 键
    res['mip_node_count'] = highs_res.get('mip_node_count', None)
    
    # 从 highs_res 字典中获取 'mip_dual_bound' 键对应的值，如果不存在则为 None，并存储到 res 字典中的 'mip_dual_bound' 键
    res['mip_dual_bound'] = highs_res.get('mip_dual_bound', None)
    
    # 从 highs_res 字典中获取 'mip_gap' 键对应的值，如果不存在则为 None，并存储到 res 字典中的 'mip_gap' 键
    res['mip_gap'] = highs_res.get('mip_gap', None)

    # 返回包含所有结果的 OptimizeResult 对象
    return OptimizeResult(res)
```