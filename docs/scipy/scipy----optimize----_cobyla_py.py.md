# `D:\src\scipysrc\scipy\scipy\optimize\_cobyla_py.py`

```
"""
Interface to Constrained Optimization By Linear Approximation

Functions
---------
.. autosummary::
   :toctree: generated/

    fmin_cobyla

"""

# 导入 functools 模块，用于创建装饰器
import functools
# 导入 RLock 类，用于创建可重入锁
from threading import RLock

# 导入 numpy 库，并将其命名为 np
import numpy as np
# 导入 scipy.optimize 中的 _cobyla 模块
from scipy.optimize import _cobyla as cobyla
# 导入 _optimize 模块中的相关函数
from ._optimize import (OptimizeResult, _check_unknown_options,
    _prepare_scalar_function)

# 尝试导入 izip 函数，如果失败则定义 izip 为 zip 函数
try:
    from itertools import izip
except ImportError:
    izip = zip

# 定义模块级别的公开接口
__all__ = ['fmin_cobyla']

# 解决方案：由于 _cobyla.minimize 存在未知的 f2py 缺陷导致线程不安全，
# 可能导致段错误，见 gh-9658。
# 创建一个可重入锁对象 _module_lock 作为工作回避措施
_module_lock = RLock()

# 定义装饰器 synchronized，用于保证函数同步执行
def synchronized(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 使用模块级别的可重入锁 _module_lock 进行同步控制
        with _module_lock:
            return func(*args, **kwargs)
    return wrapper

# 使用 synchronized 装饰器修饰 fmin_cobyla 函数
@synchronized
def fmin_cobyla(func, x0, cons, args=(), consargs=None, rhobeg=1.0,
                rhoend=1e-4, maxfun=1000, disp=None, catol=2e-4,
                *, callback=None):
    """
    Minimize a function using the Constrained Optimization By Linear
    Approximation (COBYLA) method. This method wraps a FORTRAN
    implementation of the algorithm.

    Parameters
    ----------
    func : callable
        Function to minimize. In the form func(x, \\*args).
    x0 : ndarray
        Initial guess.
    cons : sequence
        Constraint functions; must all be ``>=0`` (a single function
        if only 1 constraint). Each function takes the parameters `x`
        as its first argument, and it can return either a single number or
        an array or list of numbers.
    args : tuple, optional
        Extra arguments to pass to function.
    consargs : tuple, optional
        Extra arguments to pass to constraint functions (default of None means
        use same extra arguments as those passed to func).
        Use ``()`` for no extra arguments.
    rhobeg : float, optional
        Reasonable initial changes to the variables.
    rhoend : float, optional
        Final accuracy in the optimization (not precisely guaranteed). This
        is a lower bound on the size of the trust region.
    disp : {0, 1, 2, 3}, optional
        Controls the frequency of output; 0 implies no output.
    maxfun : int, optional
        Maximum number of function evaluations.
    catol : float, optional
        Absolute tolerance for constraint violations.
    callback : callable, optional
        Called after each iteration, as ``callback(x)``, where ``x`` is the
        current parameter vector.

    Returns
    -------
    x : ndarray
        The argument that minimises `f`.

    See also
    --------
    minimize: Interface to minimization algorithms for multivariate
        functions. See the 'COBYLA' `method` in particular.

    Notes
    -----
    This algorithm is based on linear approximations to the objective
    function and each constraint. We briefly describe the algorithm.

    Suppose the function is being minimized over k variables. At the
    """
    """
    err = "cons must be a sequence of callable functions or a single"\
          " callable function."
    # 检查 cons 是否为可调用函数序列或单个可调用函数，否则抛出错误
    try:
        len(cons)
    except TypeError as e:
        if callable(cons):
            cons = [cons]
        else:
            raise TypeError(err) from e
    else:
        for thisfunc in cons:
            if not callable(thisfunc):
                raise TypeError(err)

    if consargs is None:
        consargs = args

    # 构建约束条件
    con = tuple({'type': 'ineq', 'fun': c, 'args': consargs} for c in cons)

    # options
    # 定义优化选项字典，包括初始步长、收敛容限、是否显示优化信息、最大迭代次数、约束容限和回调函数
    opts = {'rhobeg': rhobeg,
            'tol': rhoend,
            'disp': disp,
            'maxiter': maxfun,
            'catol': catol,
            'callback': callback}

    # 调用 _minimize_cobyla 函数进行优化，传入目标函数、初始点、参数、约束和上述选项
    sol = _minimize_cobyla(func, x0, args, constraints=con,
                           **opts)

    # 如果 disp 为真且优化失败，则打印优化失败的消息
    if disp and not sol['success']:
        print(f"COBYLA failed to find a solution: {sol.message}")

    # 返回优化结果的最优点
    return sol['x']
# 使用装饰器@synchronized确保该函数的线程安全性
@synchronized
# 使用COBYLA算法最小化给定的目标函数fun
def _minimize_cobyla(fun, x0, args=(), constraints=(),
                     rhobeg=1.0, tol=1e-4, maxiter=1000,
                     disp=False, catol=2e-4, callback=None, bounds=None,
                     **unknown_options):
    """
    Minimize a scalar function of one or more variables using the
    Constrained Optimization BY Linear Approximation (COBYLA) algorithm.

    Options
    -------
    rhobeg : float
        Reasonable initial changes to the variables.
    tol : float
        Final accuracy in the optimization (not precisely guaranteed).
        This is a lower bound on the size of the trust region.
    disp : bool
        Set to True to print convergence messages. If False,
        `verbosity` is ignored as set to 0.
    maxiter : int
        Maximum number of function evaluations.
    catol : float
        Tolerance (absolute) for constraint violations

    """
    # 检查并处理未知的额外选项
    _check_unknown_options(unknown_options)
    # 将maxiter赋值给maxfun，用于COBYLA的最大函数评估次数
    maxfun = maxiter
    # 将tol赋值给rhoend，用于COBYLA算法的最终优化精度
    rhoend = tol
    # 将disp转换为整数，用于设置COBYLA算法的打印信息
    iprint = int(bool(disp))

    # 检查constraints的类型，如果是字典则转换为元组
    if isinstance(constraints, dict):
        constraints = (constraints, )

    # 如果存在bounds，则处理上下界约束
    if bounds:
        # 创建处理下界约束的函数lb_constraint
        i_lb = np.isfinite(bounds.lb)
        if np.any(i_lb):
            def lb_constraint(x, *args, **kwargs):
                return x[i_lb] - bounds.lb[i_lb]

            constraints.append({'type': 'ineq', 'fun': lb_constraint})

        # 创建处理上界约束的函数ub_constraint
        i_ub = np.isfinite(bounds.ub)
        if np.any(i_ub):
            def ub_constraint(x):
                return bounds.ub[i_ub] - x[i_ub]

            constraints.append({'type': 'ineq', 'fun': ub_constraint})

    # 遍历constraints列表，检查每个约束的类型、函数和参数
    for ic, con in enumerate(constraints):
        # 检查约束类型是否为'ineq'，COBYLA只处理不等式约束
        try:
            ctype = con['type'].lower()
        except KeyError as e:
            raise KeyError('Constraint %d has no type defined.' % ic) from e
        except TypeError as e:
            raise TypeError('Constraints must be defined using a '
                            'dictionary.') from e
        except AttributeError as e:
            raise TypeError("Constraint's type must be a string.") from e
        else:
            if ctype != 'ineq':
                raise ValueError("Constraints of type '%s' not handled by "
                                 "COBYLA." % con['type'])

        # 检查约束是否定义了函数'fun'
        if 'fun' not in con:
            raise KeyError('Constraint %d has no function defined.' % ic)

        # 检查约束是否定义了额外的参数'args'
        if 'args' not in con:
            con['args'] = ()

    # 计算所有约束函数在初始点x0处的总数m
    cons_lengths = []
    for c in constraints:
        f = c['fun'](x0, *c['args'])
        try:
            cons_length = len(f)
        except TypeError:
            cons_length = 1
        cons_lengths.append(cons_length)
    m = sum(cons_lengths)

    # 创建一个用于COBYLA算法的ScalarFunction对象，无需提供梯度函数
    def _jac(x, *args):
        return None
    # 使用 _prepare_scalar_function 函数准备标量函数，返回标量函数对象 sf
    sf = _prepare_scalar_function(fun, x0, args=args, jac=_jac)

    # 定义计算目标函数值和约束函数值的函数 calcfc
    def calcfc(x, con):
        # 计算目标函数值
        f = sf.fun(x)
        # 初始化索引 i
        i = 0
        # 遍历约束列表，计算每个约束函数的值，并存入 con 数组
        for size, c in izip(cons_lengths, constraints):
            con[i: i + size] = c['fun'](x, *c['args'])
            i += size
        # 返回目标函数值
        return f

    # 定义包装的回调函数 wrapped_callback
    def wrapped_callback(x):
        # 如果有指定回调函数，则执行回调函数，传入 x 的副本
        if callback is not None:
            callback(np.copy(x))

    # 初始化信息数组 info，包含 4 个 float64 类型的元素
    info = np.zeros(4, np.float64)

    # 使用 COBYLA 算法进行优化，返回最优解 xopt 和优化信息 info
    xopt, info = cobyla.minimize(calcfc, m=m, x=np.copy(x0), rhobeg=rhobeg,
                                  rhoend=rhoend, iprint=iprint, maxfun=maxfun,
                                  dinfo=info, callback=wrapped_callback)

    # 如果约束违反量 info[3] 大于容差 catol，则修改 info[0] 的值为 4
    if info[3] > catol:
        info[0] = 4

    # 返回优化结果 OptimizeResult 对象，包括最优解 xopt、状态码 status、成功标志 success、
    # 消息 message 和其他统计信息 nfev、目标函数值 fun、约束违反量 maxcv
    return OptimizeResult(x=xopt,
                          status=int(info[0]),
                          success=info[0] == 1,
                          message={1: 'Optimization terminated successfully.',
                                   2: 'Maximum number of function evaluations '
                                      'has been exceeded.',
                                   3: 'Rounding errors are becoming damaging '
                                      'in COBYLA subroutine.',
                                   4: 'Did not converge to a solution '
                                      'satisfying the constraints. See '
                                      '`maxcv` for magnitude of violation.',
                                   5: 'NaN result encountered.'
                                   }.get(info[0], 'Unknown exit status.'),
                          nfev=int(info[1]),
                          fun=info[2],
                          maxcv=info[3])
```