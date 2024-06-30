# `D:\src\scipysrc\scipy\scipy\optimize\_linprog_highs.py`

```
# 模块文档字符串，描述了该模块提供了对 HiGHS 线性优化软件的接口
"""HiGHS Linear Optimization Methods

Interface to HiGHS linear optimization software.
https://highs.dev/

.. versionadded:: 1.5.0

References
----------
.. [1] Q. Huangfu and J.A.J. Hall. "Parallelizing the dual revised simplex
           method." Mathematical Programming Computation, 10 (1), 119-142,
           2018. DOI: 10.1007/s12532-017-0130-5

"""

# 导入 inspect 模块
import inspect
# 导入 numpy 库并使用 np 别名
import numpy as np
# 从 _optimize 模块导入 OptimizeWarning 和 OptimizeResult 类
from ._optimize import OptimizeWarning, OptimizeResult
# 从 warnings 模块导入 warn 函数
from warnings import warn
# 从 _highs._highs_wrapper 模块导入 _highs_wrapper 类
from ._highs._highs_wrapper import _highs_wrapper
# 从 _highs._highs_constants 模块导入一系列常量
from ._highs._highs_constants import (
    CONST_INF,
    MESSAGE_LEVEL_NONE,
    HIGHS_OBJECTIVE_SENSE_MINIMIZE,

    MODEL_STATUS_NOTSET,
    MODEL_STATUS_LOAD_ERROR,
    MODEL_STATUS_MODEL_ERROR,
    MODEL_STATUS_PRESOLVE_ERROR,
    MODEL_STATUS_SOLVE_ERROR,
    MODEL_STATUS_POSTSOLVE_ERROR,
    MODEL_STATUS_MODEL_EMPTY,
    MODEL_STATUS_OPTIMAL,
    MODEL_STATUS_INFEASIBLE,
    MODEL_STATUS_UNBOUNDED_OR_INFEASIBLE,
    MODEL_STATUS_UNBOUNDED,
    MODEL_STATUS_REACHED_DUAL_OBJECTIVE_VALUE_UPPER_BOUND
    as MODEL_STATUS_RDOVUB,
    MODEL_STATUS_REACHED_OBJECTIVE_TARGET,
    MODEL_STATUS_REACHED_TIME_LIMIT,
    MODEL_STATUS_REACHED_ITERATION_LIMIT,

    HIGHS_SIMPLEX_STRATEGY_DUAL,

    HIGHS_SIMPLEX_CRASH_STRATEGY_OFF,

    HIGHS_SIMPLEX_EDGE_WEIGHT_STRATEGY_CHOOSE,
    HIGHS_SIMPLEX_EDGE_WEIGHT_STRATEGY_DANTZIG,
    HIGHS_SIMPLEX_EDGE_WEIGHT_STRATEGY_DEVEX,
    HIGHS_SIMPLEX_EDGE_WEIGHT_STRATEGY_STEEPEST_EDGE,
)
# 从 scipy.sparse 库导入 csc_matrix, vstack, issparse 函数
from scipy.sparse import csc_matrix, vstack, issparse


def _highs_to_scipy_status_message(highs_status, highs_message):
    """Converts HiGHS status number/message to SciPy status number/message"""
    # 定义一个字典映射 HiGHS 状态码到 SciPy 状态码和消息的转换关系
    scipy_statuses_messages = {
        None: (4, "HiGHS did not provide a status code. "),
        MODEL_STATUS_NOTSET: (4, ""),
        MODEL_STATUS_LOAD_ERROR: (4, ""),
        MODEL_STATUS_MODEL_ERROR: (2, ""),
        MODEL_STATUS_PRESOLVE_ERROR: (4, ""),
        MODEL_STATUS_SOLVE_ERROR: (4, ""),
        MODEL_STATUS_POSTSOLVE_ERROR: (4, ""),
        MODEL_STATUS_MODEL_EMPTY: (4, ""),
        MODEL_STATUS_RDOVUB: (4, ""),
        MODEL_STATUS_REACHED_OBJECTIVE_TARGET: (4, ""),
        MODEL_STATUS_OPTIMAL: (0, "Optimization terminated successfully. "),
        MODEL_STATUS_REACHED_TIME_LIMIT: (1, "Time limit reached. "),
        MODEL_STATUS_REACHED_ITERATION_LIMIT: (1, "Iteration limit reached. "),
        MODEL_STATUS_INFEASIBLE: (2, "The problem is infeasible. "),
        MODEL_STATUS_UNBOUNDED: (3, "The problem is unbounded. "),
        MODEL_STATUS_UNBOUNDED_OR_INFEASIBLE: (4, "The problem is unbounded "
                                               "or infeasible. ")}
    # 若 HiGHS 状态码未被识别，则使用未识别的状态码和消息
    unrecognized = (4, "The HiGHS status code was not recognized. ")
    # 根据 HiGHS 状态码获取对应的 SciPy 状态码和消息，默认为未识别的消息
    scipy_status, scipy_message = (
        scipy_statuses_messages.get(highs_status, unrecognized))
    # 构建最终的 SciPy 消息，包含 HiGHS 的状态码和消息
    scipy_message = (f"{scipy_message}"
                     f"(HiGHS Status {highs_status}: {highs_message})")
    return scipy_status, scipy_message


def _replace_inf(x):
    # 使用常量 CONST_INF 替换数组 x 中的 np.inf
    infs = np.isinf(x)  # 找出数组 x 中的无穷值（infinity）
    with np.errstate(invalid="ignore"):  # 设置错误处理状态，忽略无效值错误
        x[infs] = np.sign(x[infs]) * CONST_INF  # 对数组 x 中的无穷值进行替换为其符号乘以常量 CONST_INF 的结果
    # 返回替换后的数组 x
    return x
# 将选项转换为 HiGHS 求解器可接受的枚举值
def _convert_to_highs_enum(option, option_str, choices):
    # 如果选项在给定的选择列表中，则使用其对应的值；否则使用函数签名中的默认值，并发出警告：
    try:
        return choices[option.lower()]  # 尝试按小写形式查找选项
    except AttributeError:
        return choices[option]  # 如果发生属性错误，则按原始选项查找
    except KeyError:
        sig = inspect.signature(_linprog_highs)
        default_str = sig.parameters[option_str].default
        warn(f"Option {option_str} is {option}, but only values in "
             f"{set(choices.keys())} are allowed. Using default: "
             f"{default_str}.",
             OptimizeWarning, stacklevel=3)
        return choices[default_str]  # 如果发生键错误，则使用默认值，并发出警告


def _linprog_highs(lp, solver, time_limit=None, presolve=True,
                   disp=False, maxiter=None,
                   dual_feasibility_tolerance=None,
                   primal_feasibility_tolerance=None,
                   ipm_optimality_tolerance=None,
                   simplex_dual_edge_weight_strategy=None,
                   mip_rel_gap=None,
                   mip_max_nodes=None,
                   **unknown_options):
    r"""
    使用 HiGHS 求解器解决线性规划问题：

    用户可见的文档位于 _linprog_doc.py。

    Parameters
    ----------
    lp :  _LPProblem
        一个 `scipy.optimize._linprog_util._LPProblem` 的 `namedtuple`。
    solver : "ipm" or "simplex" or None
        指定使用的 HiGHS 求解器。如果为 `None`，将使用 "simplex" 求解器。

    Options
    -------
    maxiter : int
        执行迭代的最大次数。对于 `solver='ipm'`，这不包括交叉迭代次数。默认为平台上最大可能值的 `int`。
    disp : bool
        如果要在每次迭代时向控制台打印优化状态指示器，则设置为 `True`；默认为 `False`。
    time_limit : float
        解决问题的最大允许时间（秒）。默认为平台上最大可能值的 `double`。
    presolve : bool
        在发送到主求解器之前，尝试识别平凡的不可行性和非平凡性，并简化问题。通常建议保持默认设置 `True`；如果要禁用预处理，则设置为 `False`。
    dual_feasibility_tolerance : double
        对偶可行性容限。默认为 1e-07。当 `solver='ipm'` 时，将此值与 `primal_feasibility_tolerance` 的最小值用作容限。
    primal_feasibility_tolerance : double
        原始可行性容限。默认为 1e-07。当 `solver='ipm'` 时，将此值与 `dual_feasibility_tolerance` 的最小值用作容限。
    # 如果存在未知选项，则生成警告消息，指出这些选项将直接传递给 HiGHS 求解器
    if unknown_options:
        message = (f"Unrecognized options detected: {unknown_options}. "
                   "These will be passed to HiGHS verbatim.")
        warn(message, OptimizeWarning, stacklevel=3)

    # 将简单型双重边权重策略映射到 HiGHS 枚举值
    simplex_dual_edge_weight_strategy_enum = _convert_to_highs_enum(
        simplex_dual_edge_weight_strategy,
        'simplex_dual_edge_weight_strategy',
        choices={'dantzig': HIGHS_SIMPLEX_EDGE_WEIGHT_STRATEGY_DANTZIG,
                 'devex': HIGHS_SIMPLEX_EDGE_WEIGHT_STRATEGY_DEVEX,
                 'steepest-devex': HIGHS_SIMPLEX_EDGE_WEIGHT_STRATEGY_CHOOSE,
                 'steepest':
                 HIGHS_SIMPLEX_EDGE_WEIGHT_STRATEGY_STEEPEST_EDGE,
                 None: None})
    c, A_ub, b_ub, A_eq, b_eq, bounds, x0, integrality = lp
    # 从输入的线性规划数据结构中解包出所需的变量

    lb, ub = bounds.T.copy()  # 将边界分离并复制为 C-cntgs
    # 复制并分离边界数据，以备后续使用

    # highs_wrapper 解决的是 LHS <= A*x <= RHS 的问题，不处理等式约束
    with np.errstate(invalid="ignore"):
        lhs_ub = -np.ones_like(b_ub)*np.inf  # UB 约束的左侧是 -inf
    rhs_ub = b_ub  # UB 约束的右侧是 b_ub
    lhs_eq = b_eq  # 等式约束转为不等式约束
    rhs_eq = b_eq  # 等式约束的左右两侧相等
    lhs = np.concatenate((lhs_ub, lhs_eq))  # 构建总体的左侧约束
    rhs = np.concatenate((rhs_ub, rhs_eq))  # 构建总体的右侧约束

    if issparse(A_ub) or issparse(A_eq):
        A = vstack((A_ub, A_eq))
    else:
        A = np.vstack((A_ub, A_eq))
    A = csc_matrix(A)
    # 将 A_ub 和 A_eq 垂直堆叠成稀疏矩阵 A

    options = {
        'presolve': presolve,
        'sense': HIGHS_OBJECTIVE_SENSE_MINIMIZE,
        'solver': solver,
        'time_limit': time_limit,
        'highs_debug_level': MESSAGE_LEVEL_NONE,
        'dual_feasibility_tolerance': dual_feasibility_tolerance,
        'ipm_optimality_tolerance': ipm_optimality_tolerance,
        'log_to_console': disp,
        'mip_max_nodes': mip_max_nodes,
        'output_flag': disp,
        'primal_feasibility_tolerance': primal_feasibility_tolerance,
        'simplex_dual_edge_weight_strategy':
            simplex_dual_edge_weight_strategy_enum,
        'simplex_strategy': HIGHS_SIMPLEX_STRATEGY_DUAL,
        'simplex_crash_strategy': HIGHS_SIMPLEX_CRASH_STRATEGY_OFF,
        'ipm_iteration_limit': maxiter,
        'simplex_iteration_limit': maxiter,
        'mip_rel_gap': mip_rel_gap,
    }
    options.update(unknown_options)
    # 设置求解器的参数选项，并更新未知选项

    # np.inf 不适用；使用一个非常大的常数替代
    rhs = _replace_inf(rhs)  # 替换 RHS 中的无穷大值
    lhs = _replace_inf(lhs)  # 替换 LHS 中的无穷大值
    lb = _replace_inf(lb)    # 替换 lb 中的无穷大值
    ub = _replace_inf(ub)    # 替换 ub 中的无穷大值

    if integrality is None or np.sum(integrality) == 0:
        integrality = np.empty(0)
    else:
        integrality = np.array(integrality)
    # 处理整数约束的情况，将 integrality 转为数组或空数组

    res = _highs_wrapper(c, A.indptr, A.indices, A.data, lhs, rhs,
                         lb, ub, integrality.astype(np.uint8), options)
    # 调用高性能求解器的包装函数进行求解，并获取结果

    # HiGHS 将约束表示为 lhs/rhs 的形式，因此需要拆分出来 A_ub 和 A_eq 对应的 slack
    if 'slack' in res:
        slack = res['slack']
        con = np.array(slack[len(b_ub):])
        slack = np.array(slack[:len(b_ub)])
    else:
        slack, con = None, None
    # 如果结果中包含 slack 变量，则拆分出 slack 和 con 变量

    # 对于等式约束和不等式约束以及上界和下界的 Lagrange 乘数
    if 'lambda' in res:
        lamda = res['lambda']
        marg_ineqlin = np.array(lamda[:len(b_ub)])
        marg_eqlin = np.array(lamda[len(b_ub):])
        marg_upper = np.array(res['marg_bnds'][1, :])
        marg_lower = np.array(res['marg_bnds'][0, :])
    else:
        marg_ineqlin, marg_eqlin = None, None
        marg_upper, marg_lower = None, None
    # 如果结果中包含 lambda 变量，则提取对应的 Lagrange 乘数

    # 如果未来开始智能选择求解器，则需要更新这部分内容

    # 转换为 scipy 风格的状态和消息
    highs_status = res.get('status', None)
    highs_message = res.get('message', None)
    # 获取求解器返回结果中的状态和消息
    # 将高性能线性优化库的状态码和消息转换为 SciPy 库的状态和消息
    status, message = _highs_to_scipy_status_message(highs_status,
                                                     highs_message)

    # 如果结果中存在变量 x，则将其转换为 NumPy 数组；否则设置为 None
    x = np.array(res['x']) if 'x' in res else None
    
    # 构建最优解字典 sol，包括变量 x、松弛变量 slack、线性约束 con 等
    sol = {'x': x,
           'slack': slack,
           'con': con,
           'ineqlin': OptimizeResult({
               'residual': slack,  # 不等式约束的残差
               'marginals': marg_ineqlin,  # 不等式约束的边际值
           }),
           'eqlin': OptimizeResult({
               'residual': con,  # 等式约束的残差
               'marginals': marg_eqlin,  # 等式约束的边际值
           }),
           'lower': OptimizeResult({
               'residual': None if x is None else x - lb,  # 下界的残差
               'marginals': marg_lower,  # 下界的边际值
           }),
           'upper': OptimizeResult({
               'residual': None if x is None else ub - x,  # 上界的残差
               'marginals': marg_upper  # 上界的边际值
            }),
           'fun': res.get('fun'),  # 优化目标函数的值
           'status': status,  # 优化的状态
           'success': res['status'] == MODEL_STATUS_OPTIMAL,  # 是否成功找到最优解
           'message': message,  # 优化过程的消息
           'nit': res.get('simplex_nit', 0) or res.get('ipm_nit', 0),  # 迭代次数
           'crossover_nit': res.get('crossover_nit'),  # 十字交叉迭代次数
           }

    # 如果存在变量 x 并且定义了整数约束条件
    if np.any(x) and integrality is not None:
        # 更新 sol 字典，添加整数线性规划的节点数、对偶界限和间隙
        sol.update({
            'mip_node_count': res.get('mip_node_count', 0),  # 整数线性规划的节点数
            'mip_dual_bound': res.get('mip_dual_bound', 0.0),  # 整数线性规划的对偶界限
            'mip_gap': res.get('mip_gap', 0.0),  # 整数线性规划的间隙
        })

    # 返回最优解字典 sol
    return sol
```