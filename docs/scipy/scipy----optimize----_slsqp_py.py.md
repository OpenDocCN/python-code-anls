# `D:\src\scipysrc\scipy\scipy\optimize\_slsqp_py.py`

```
"""
This module implements the Sequential Least Squares Programming optimization
algorithm (SLSQP), originally developed by Dieter Kraft.
See http://www.netlib.org/toms/733

Functions
---------
.. autosummary::
   :toctree: generated/

    approx_jacobian
    fmin_slsqp

"""

__all__ = ['approx_jacobian', 'fmin_slsqp']

import numpy as np
from scipy.optimize._slsqp import slsqp
from numpy import (zeros, array, linalg, append, concatenate, finfo,
                   sqrt, vstack, isfinite, atleast_1d)
from ._optimize import (OptimizeResult, _check_unknown_options,
                        _prepare_scalar_function, _clip_x_for_func,
                        _check_clip_x)
from ._numdiff import approx_derivative
from ._constraints import old_bound_to_new, _arr_to_scalar
from scipy._lib._array_api import atleast_nd, array_namespace


__docformat__ = "restructuredtext en"

_epsilon = sqrt(finfo(float).eps)


def approx_jacobian(x, func, epsilon, *args):
    """
    Approximate the Jacobian matrix of a callable function.

    Parameters
    ----------
    x : array_like
        The state vector at which to compute the Jacobian matrix.
    func : callable f(x,*args)
        The vector-valued function.
    epsilon : float
        The perturbation used to determine the partial derivatives.
    args : sequence
        Additional arguments passed to func.

    Returns
    -------
    An array of dimensions ``(lenf, lenx)`` where ``lenf`` is the length
    of the outputs of `func`, and ``lenx`` is the number of elements in
    `x`.

    Notes
    -----
    The approximation is done using forward differences.

    """
    # Use approx_derivative to compute the Jacobian matrix with specified parameters
    jac = approx_derivative(func, x, method='2-point', abs_step=epsilon,
                            args=args)
    # Ensure jac is at least a 2D array, even if func returns a scalar
    return np.atleast_2d(jac)


def fmin_slsqp(func, x0, eqcons=(), f_eqcons=None, ieqcons=(), f_ieqcons=None,
               bounds=(), fprime=None, fprime_eqcons=None,
               fprime_ieqcons=None, args=(), iter=100, acc=1.0E-6,
               iprint=1, disp=None, full_output=0, epsilon=_epsilon,
               callback=None):
    """
    Minimize a function using Sequential Least Squares Programming

    Python interface function for the SLSQP Optimization subroutine
    originally implemented by Dieter Kraft.

    Parameters
    ----------
    func : callable f(x,*args)
        Objective function.  Must return a scalar.
    x0 : 1-D ndarray of float
        Initial guess for the independent variable(s).
    eqcons : list, optional
        A list of functions of length n such that
        eqcons[j](x,*args) == 0.0 in a successfully optimized
        problem.
    f_eqcons : callable f(x,*args), optional
        Returns a 1-D array in which each element must equal 0.0 in a
        successfully optimized problem. If f_eqcons is specified,
        eqcons is ignored.
    """

    # Calls the underlying SLSQP optimizer with provided parameters and returns its result
    return slsqp(func, x0, eqcons, f_eqcons, ieqcons, f_ieqcons, bounds,
                 fprime, fprime_eqcons, fprime_ieqcons, args, iter, acc,
                 iprint, disp, full_output, epsilon, callback)
    ieqcons : list, optional
        # 不等式约束函数的列表，长度为 n，确保在优化问题中 ieqcons[j](x,*args) >= 0.0
    f_ieqcons : callable f(x,*args), optional
        # 如果成功优化问题，返回一个 1-D ndarray，其中每个元素必须大于或等于 0.0
        # 如果指定了 f_ieqcons，则忽略 ieqcons
    bounds : list, optional
        # 每个独立变量的下限和上限的元组列表 [(xl0, xu0),(xl1, xu1),...]
        # 无穷大值将被解释为大浮点值
    fprime : callable ``f(x,*args)``, optional
        # 一个函数，计算 func 的偏导数
    fprime_eqcons : callable ``f(x,*args)``, optional
        # 一个函数，返回 m 行 n 列的等式约束法向量数组
        # 如果未提供，法向量将会被近似计算，fprime_eqcons 返回的数组大小应为 ( len(eqcons), len(x0) )
    fprime_ieqcons : callable ``f(x,*args)``, optional
        # 一个函数，返回 m 行 n 列的不等式约束法向量数组
        # 如果未提供，法向量将会被近似计算，fprime_ieqcons 返回的数组大小应为 ( len(ieqcons), len(x0) )
    args : sequence, optional
        # 传递给 func 和 fprime 的额外参数
    iter : int, optional
        # 最大迭代次数
    acc : float, optional
        # 请求的精度
    iprint : int, optional
        # fmin_slsqp 的详细程度：

        # * iprint <= 0 : 静默操作
        # * iprint == 1 : 完成时打印摘要（默认）
        # * iprint >= 2 : 打印每次迭代的状态和摘要
    disp : int, optional
        # 覆盖 iprint 接口（优先使用）
    full_output : bool, optional
        # 如果为 False，则只返回 func 的最小化器（默认）
        # 否则，输出最终的目标函数值和摘要信息
    epsilon : float, optional
        # 有限差分导数估计的步长大小
    callback : callable, optional
        # 每次迭代后调用，形式为 callback(x)，其中 x 是当前的参数向量

    Returns
    -------
    out : ndarray of float
        # func 的最终最小化器
    fx : ndarray of float, if full_output is true
        # 如果 full_output 为 true，则为目标函数的最终值
    its : int, if full_output is true
        # 如果 full_output 为 true，则为迭代次数
    imode : int, if full_output is true
        # 如果 full_output 为 true，则为优化器的退出模式（见下文）
    smode : string, if full_output is true
        # 如果 full_output 为 true，则描述优化器的退出模式的消息

    See also
    --------
    minimize: 多变量函数的最小化算法接口，特别是 'SLSQP' 方法。

    Notes
    -----
    # 定义退出模式的意义：
    #
    #     -1 : 需要进行梯度评估 (g & a)
    #      0 : 优化成功终止
    #      1 : 需要进行函数评估 (f & c)
    #      2 : 约束条件多于独立变量
    #      3 : LSQ 子问题超过 3*n 次迭代
    #      4 : 不兼容的不等式约束
    #      5 : LSQ 子问题中的矩阵 E 是奇异的
    #      6 : LSQ 子问题中的矩阵 C 是奇异的
    #      7 : 等式约束子问题 HFTI 的秩不足
    #      8 : 线搜索中的正方向导数为正
    #      9 : 达到迭代限制

    Examples
    --------
    示例在教程中详细说明 :ref:`tutorial-sqlsp`。

    """
    if disp is not None:
        iprint = disp

    opts = {'maxiter': iter,
            'ftol': acc,
            'iprint': iprint,
            'disp': iprint != 0,
            'eps': epsilon,
            'callback': callback}

    # 作为字典元组构建约束条件
    cons = ()

    # 1. 第一类约束 (eqcons, ieqcons); 没有雅可比矩阵; 接受与目标函数相同的额外参数。
    cons += tuple({'type': 'eq', 'fun': c, 'args': args} for c in eqcons)
    cons += tuple({'type': 'ineq', 'fun': c, 'args': args} for c in ieqcons)

    # 2. 第二类约束 (f_eqcons, f_ieqcons) 及其雅可比矩阵
    #    (fprime_eqcons, fprime_ieqcons); 同样接受与目标函数相同的额外参数。
    if f_eqcons:
        cons += ({'type': 'eq', 'fun': f_eqcons, 'jac': fprime_eqcons,
                  'args': args}, )
    if f_ieqcons:
        cons += ({'type': 'ineq', 'fun': f_ieqcons, 'jac': fprime_ieqcons,
                  'args': args}, )

    # 调用内部函数 _minimize_slsqp 进行 SLSQP 最小化优化
    res = _minimize_slsqp(func, x0, args, jac=fprime, bounds=bounds,
                          constraints=cons, **opts)
    
    # 根据需要返回结果的部分内容
    if full_output:
        return res['x'], res['fun'], res['nit'], res['status'], res['message']
    else:
        return res['x']
    """
    Minimize a scalar function of one or more variables using Sequential
    Least Squares Programming (SLSQP).

    Options
    -------
    ftol : float
        Precision goal for the value of f in the stopping criterion.
    eps : float
        Step size used for numerical approximation of the Jacobian.
    disp : bool
        Set to True to print convergence messages. If False,
        `verbosity` is ignored and set to 0.
    maxiter : int
        Maximum number of iterations.
    finite_diff_rel_step : None or array_like, optional
        If ``jac in ['2-point', '3-point', 'cs']`` the relative step size to
        use for numerical approximation of `jac`. The absolute step
        size is computed as ``h = rel_step * sign(x) * max(1, abs(x))``,
        possibly adjusted to fit into the bounds. For ``method='3-point'``
        the sign of `h` is ignored. If None (default) then step is selected
        automatically.
    """
    # 检查并处理未知的选项参数
    _check_unknown_options(unknown_options)
    # 计算迭代次数
    iter = maxiter - 1
    # 设置精度
    acc = ftol
    # 设置步长
    epsilon = eps

    # 根据 disp 参数设置是否输出迭代信息
    if not disp:
        iprint = 0

    # 将 x0 转换为数组形式
    xp = array_namespace(x0)
    x0 = atleast_nd(x0, ndim=1, xp=xp)
    # 确定数据类型为 float64
    dtype = xp.float64
    if xp.isdtype(x0.dtype, "real floating"):
        dtype = x0.dtype
    # 将 x0 重塑为一维数组
    x = xp.reshape(xp.astype(x0, dtype), -1)

    # 将旧版边界转换为新版边界形式，以适应 ScalarFunction 的要求
    if bounds is None or len(bounds) == 0:
        new_bounds = (-np.inf, np.inf)
    else:
        new_bounds = old_bound_to_new(bounds)

    # 将初始猜测值限制在边界内，否则 ScalarFunction 无法正常工作
    x = np.clip(x, new_bounds[0], new_bounds[1])

    # 根据 constraints 类型进行分类处理，转换为字典形式
    if isinstance(constraints, dict):
        constraints = (constraints, )

    # 初始化约束条件为字典形式
    cons = {'eq': (), 'ineq': ()}
    # 遍历约束列表 `constraints`，同时获取索引 `ic` 和约束内容 `con`
    for ic, con in enumerate(constraints):
        # 检查约束类型是否定义，并将其转换为小写
        try:
            ctype = con['type'].lower()
        except KeyError as e:
            # 如果缺少类型定义，引发 KeyError 异常并指明具体的约束索引
            raise KeyError('Constraint %d has no type defined.' % ic) from e
        except TypeError as e:
            # 如果约束不是以字典形式定义，引发 TypeError 异常
            raise TypeError('Constraints must be defined using a '
                            'dictionary.') from e
        except AttributeError as e:
            # 如果约束类型不是字符串，引发 TypeError 异常
            raise TypeError("Constraint's type must be a string.") from e
        else:
            # 检查约束类型是否为 'eq' 或 'ineq'，否则引发 ValueError 异常
            if ctype not in ['eq', 'ineq']:
                raise ValueError("Unknown constraint type '%s'." % con['type'])

        # 检查约束是否定义了函数 `fun`
        if 'fun' not in con:
            raise ValueError('Constraint %d has no function defined.' % ic)

        # 检查是否定义了雅可比矩阵 `jac`，如果未定义，则创建近似的雅可比函数
        cjac = con.get('jac')
        if cjac is None:
            # 创建一个工厂函数 `cjac_factory`，用于保留对 `fun` 的引用
            def cjac_factory(fun):
                def cjac(x, *args):
                    x = _check_clip_x(x, new_bounds)

                    # 根据 `jac` 的不同取值，使用不同的方法近似计算导数
                    if jac in ['2-point', '3-point', 'cs']:
                        return approx_derivative(fun, x, method=jac, args=args,
                                                 rel_step=finite_diff_rel_step,
                                                 bounds=new_bounds)
                    else:
                        return approx_derivative(fun, x, method='2-point',
                                                 abs_step=epsilon, args=args,
                                                 bounds=new_bounds)

                return cjac
            # 使用约束函数 `con['fun']` 创建近似的雅可比函数 `cjac`
            cjac = cjac_factory(con['fun'])

        # 更新约束字典 `cons` 中对应类型 `ctype` 的约束列表
        cons[ctype] += ({'fun': con['fun'],
                         'jac': cjac,
                         'args': con.get('args', ())}, )

    # 定义退出模式的字典 `exit_modes`
    exit_modes = {-1: "Gradient evaluation required (g & a)",
                   0: "Optimization terminated successfully",
                   1: "Function evaluation required (f & c)",
                   2: "More equality constraints than independent variables",
                   3: "More than 3*n iterations in LSQ subproblem",
                   4: "Inequality constraints incompatible",
                   5: "Singular matrix E in LSQ subproblem",
                   6: "Singular matrix C in LSQ subproblem",
                   7: "Rank-deficient equality constraint subproblem HFTI",
                   8: "Positive directional derivative for linesearch",
                   9: "Iteration limit reached"}

    # 设置 SLSQP 需要的参数
    # 计算等式约束数 `meq` 和不等式约束数 `mieq`
    meq = sum(map(len, [atleast_1d(c['fun'](x, *c['args']))
              for c in cons['eq']]))
    mieq = sum(map(len, [atleast_1d(c['fun'](x, *c['args']))
               for c in cons['ineq']]))
    # 计算总约束数 `m`
    m = meq + mieq
    # la = 约束条件的数量，如果没有约束则为1
    la = array([1, m]).max()
    # n = 自变量的数量
    n = len(x)

    # 为 SLSQP 算法定义工作空间
    n1 = n + 1
    mineq = m - meq + n1 + n1
    # 计算所需的工作空间长度 len_w
    len_w = (3*n1+m)*(n1+1)+(n1-meq+1)*(mineq+2) + 2*mineq+(n1+mineq)*(n1-meq) \
            + 2*meq + n1 + ((n+1)*n)//2 + 2*m + 3*n + 3*n1 + 1
    w = zeros(len_w)
    # 初始化整数工作空间长度 len_jw
    len_jw = mineq
    jw = zeros(len_jw)

    # 将边界分解为 xl 和 xu
    if bounds is None or len(bounds) == 0:
        xl = np.empty(n, dtype=float)
        xu = np.empty(n, dtype=float)
        xl.fill(np.nan)
        xu.fill(np.nan)
    else:
        # 将边界转换为数组 bnds，并确保长度与自变量 x 的长度相同
        bnds = array([(_arr_to_scalar(l), _arr_to_scalar(u))
                      for (l, u) in bounds], float)
        if bnds.shape[0] != n:
            raise IndexError('SLSQP Error: the length of bounds is not '
                             'compatible with that of x0.')

        # 检查边界是否合法
        with np.errstate(invalid='ignore'):
            bnderr = bnds[:, 0] > bnds[:, 1]

        if bnderr.any():
            raise ValueError('SLSQP Error: lb > ub in bounds %s.' %
                             ', '.join(str(b) for b in bnderr))
        xl, xu = bnds[:, 0], bnds[:, 1]

        # 将无穷边界标记为 NaN，Fortran 代码可以理解这种情况
        infbnd = ~isfinite(bnds)
        xl[infbnd[:, 0]] = np.nan
        xu[infbnd[:, 1]] = np.nan

    # ScalarFunction 提供函数和梯度的评估
    sf = _prepare_scalar_function(func, x, jac=jac, args=args, epsilon=eps,
                                  finite_diff_rel_step=finite_diff_rel_step,
                                  bounds=new_bounds)
    # gh11403 SLSQP 有时会超出边界 1 或 2 个 ULP，确保这不会影响到函数/梯度的评估
    wrapped_fun = _clip_x_for_func(sf.fun, new_bounds)
    wrapped_grad = _clip_x_for_func(sf.grad, new_bounds)

    # 初始化迭代计数器和模式值
    mode = array(0, int)
    acc = array(acc, float)
    majiter = array(iter, int)
    majiter_prev = 0

    # 初始化内部 SLSQP 状态变量
    alpha = array(0, float)
    f0 = array(0, float)
    gs = array(0, float)
    h1 = array(0, float)
    h2 = array(0, float)
    h3 = array(0, float)
    h4 = array(0, float)
    t = array(0, float)
    t0 = array(0, float)
    tol = array(0, float)
    iexact = array(0, int)
    incons = array(0, int)
    ireset = array(0, int)
    itermx = array(0, int)
    line = array(0, int)
    n1 = array(0, int)
    n2 = array(0, int)
    n3 = array(0, int)

    # 如果 iprint >= 2，则打印头部信息
    if iprint >= 2:
        print("%5s %5s %16s %16s" % ("NIT", "FC", "OBJFUN", "GNORM"))

    # mode 在进入时为零，因此调用目标函数、约束和梯度
    # 这里不应该有函数评估，因为已经通过 ScalarFunction 缓存了
    fx = wrapped_fun(x)
    g = append(wrapped_grad(x), 0.0)
    c = _eval_constraint(x, cons)
    # 调用 _eval_con_normals 函数计算约束法线向量
    a = _eval_con_normals(x, cons, la, n, m, meq, mieq)

    while 1:
        # 调用 SLSQP 算法进行优化
        slsqp(m, meq, x, xl, xu, fx, c, g, a, acc, majiter, mode, w, jw,
              alpha, f0, gs, h1, h2, h3, h4, t, t0, tol,
              iexact, incons, ireset, itermx, line,
              n1, n2, n3)

        if mode == 1:  # 如果 mode 为 1，需要重新评估目标函数和约束条件
            # 重新计算目标函数值
            fx = wrapped_fun(x)
            # 重新计算约束条件
            c = _eval_constraint(x, cons)

        if mode == -1:  # 如果 mode 为 -1，需要重新评估梯度
            # 计算并扩展梯度向量
            g = append(wrapped_grad(x), 0.0)
            # 重新计算约束法线向量
            a = _eval_con_normals(x, cons, la, n, m, meq, mieq)

        if majiter > majiter_prev:
            # 如果主迭代次数增加，调用回调函数
            if callback is not None:
                callback(np.copy(x))

            # 如果 iprint 大于等于 2，则打印当前迭代状态
            if iprint >= 2:
                print("%5i %5i % 16.6E % 16.6E" % (majiter, sf.nfev,
                                                   fx, linalg.norm(g)))

        # 如果退出模式不是 -1 或 1，则 SLSQP 算法已完成
        if abs(mode) != 1:
            break

        majiter_prev = int(majiter)

    # 优化循环完成，如果需要，打印状态信息
    if iprint >= 1:
        # 打印退出模式及其状态
        print(exit_modes[int(mode)] + "    (Exit mode " + str(mode) + ')')
        # 打印当前函数值
        print("            Current function value:", fx)
        # 打印迭代次数
        print("            Iterations:", majiter)
        # 打印函数评估次数
        print("            Function evaluations:", sf.nfev)
        # 打印梯度评估次数
        print("            Gradient evaluations:", sf.ngev)

    # 返回优化结果对象
    return OptimizeResult(x=x, fun=fx, jac=g[:-1], nit=int(majiter),
                          nfev=sf.nfev, njev=sf.ngev, status=int(mode),
                          message=exit_modes[int(mode)], success=(mode == 0))
# 计算约束条件的值
def _eval_constraint(x, cons):
    # 如果存在等式约束，计算所有等式约束的值并连接成一个数组
    if cons['eq']:
        c_eq = concatenate([atleast_1d(con['fun'](x, *con['args']))
                            for con in cons['eq']])
    else:
        c_eq = zeros(0)  # 如果没有等式约束，则为一个空数组

    # 如果存在不等式约束，计算所有不等式约束的值并连接成一个数组
    if cons['ineq']:
        c_ieq = concatenate([atleast_1d(con['fun'](x, *con['args']))
                             for con in cons['ineq']])
    else:
        c_ieq = zeros(0)  # 如果没有不等式约束，则为一个空数组

    # 将等式约束和不等式约束的值连接成一个单一的向量
    c = concatenate((c_eq, c_ieq))
    return c  # 返回合并后的约束值向量


# 计算约束的法向量
def _eval_con_normals(x, cons, la, n, m, meq, mieq):
    # 如果存在等式约束，计算所有等式约束的雅可比矩阵并堆叠成一个大矩阵
    if cons['eq']:
        a_eq = vstack([con['jac'](x, *con['args'])
                       for con in cons['eq']])
    else:  # 如果没有等式约束，则创建一个全零矩阵
        a_eq = zeros((meq, n))

    # 如果存在不等式约束，计算所有不等式约束的雅可比矩阵并堆叠成一个大矩阵
    if cons['ineq']:
        a_ieq = vstack([con['jac'](x, *con['args'])
                        for con in cons['ineq']])
    else:  # 如果没有不等式约束，则创建一个全零矩阵
        a_ieq = zeros((mieq, n))

    # 将等式约束和不等式约束的雅可比矩阵连接成一个大矩阵
    if m == 0:  # 如果没有约束条件，则创建一个全零矩阵
        a = zeros((la, n))
    else:
        a = vstack((a_eq, a_ieq))

    # 在矩阵 a 的右侧添加一列全零，以适应后续计算需要的形状
    a = concatenate((a, zeros([la, 1])), 1)

    return a  # 返回合并后的约束法向量矩阵
```