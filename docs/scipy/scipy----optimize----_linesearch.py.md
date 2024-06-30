# `D:\src\scipysrc\scipy\scipy\optimize\_linesearch.py`

```
"""
Functions
---------
.. autosummary::
   :toctree: generated/

    line_search_armijo
    line_search_wolfe1
    line_search_wolfe2
    scalar_search_wolfe1
    scalar_search_wolfe2

"""
# 从警告模块导入警告类
from warnings import warn

# 导入 DCSCRH 模块中的 DCSRCH 类
from ._dcsrch import DCSRCH
# 导入 NumPy 库，并使用别名 np
import numpy as np

# 定义模块的公开接口
__all__ = ['LineSearchWarning', 'line_search_wolfe1', 'line_search_wolfe2',
           'scalar_search_wolfe1', 'scalar_search_wolfe2',
           'line_search_armijo']

# 定义自定义警告类，继承自 RuntimeWarning
class LineSearchWarning(RuntimeWarning):
    pass


# 函数用于检查参数 c1 和 c2 是否满足 0 < c1 < c2 < 1 的条件
def _check_c1_c2(c1, c2):
    if not (0 < c1 < c2 < 1):
        raise ValueError("'c1' and 'c2' do not satisfy"
                         "'0 < c1 < c2 < 1'.")


#------------------------------------------------------------------------------
# Minpack's Wolfe line and scalar searches
#------------------------------------------------------------------------------

# 函数实现 Wolfe 条件的线搜索
def line_search_wolfe1(f, fprime, xk, pk, gfk=None,
                       old_fval=None, old_old_fval=None,
                       args=(), c1=1e-4, c2=0.9, amax=50, amin=1e-8,
                       xtol=1e-14):
    """
    As `scalar_search_wolfe1` but do a line search to direction `pk`

    Parameters
    ----------
    f : callable
        Function `f(x)`
    fprime : callable
        Gradient of `f`
    xk : array_like
        Current point
    pk : array_like
        Search direction
    gfk : array_like, optional
        Gradient of `f` at point `xk`
    old_fval : float, optional
        Value of `f` at point `xk`
    old_old_fval : float, optional
        Value of `f` at point preceding `xk`

    The rest of the parameters are the same as for `scalar_search_wolfe1`.

    Returns
    -------
    stp, f_count, g_count, fval, old_fval
        As in `line_search_wolfe1`
    gval : array
        Gradient of `f` at the final point

    Notes
    -----
    Parameters `c1` and `c2` must satisfy ``0 < c1 < c2 < 1``.

    """
    if gfk is None:
        gfk = fprime(xk, *args)

    gval = [gfk]  # 存储梯度值
    gc = [0]       # 存储调用 fprime 的次数
    fc = [0]       # 存储调用 f 的次数

    # 定义 phi 函数，计算在 s 处的函数值
    def phi(s):
        fc[0] += 1
        return f(xk + s*pk, *args)

    # 定义 derphi 函数，计算在 s 处的函数导数值
    def derphi(s):
        gval[0] = fprime(xk + s*pk, *args)
        gc[0] += 1
        return np.dot(gval[0], pk)

    # 计算初始点的导数值
    derphi0 = np.dot(gfk, pk)

    # 调用 scalar_search_wolfe1 函数进行标量搜索
    stp, fval, old_fval = scalar_search_wolfe1(
            phi, derphi, old_fval, old_old_fval, derphi0,
            c1=c1, c2=c2, amax=amax, amin=amin, xtol=xtol)

    # 返回搜索结果
    return stp, fc[0], gc[0], fval, old_fval, gval[0]


# 函数实现 Wolfe 条件的标量搜索
def scalar_search_wolfe1(phi, derphi, phi0=None, old_phi0=None, derphi0=None,
                         c1=1e-4, c2=0.9,
                         amax=50, amin=1e-8, xtol=1e-14):
    """
    Scalar function search for alpha that satisfies strong Wolfe conditions

    alpha > 0 is assumed to be a descent direction.

    Parameters
    ----------
    phi : callable phi(alpha)
        Function at point `alpha`
    derphi : callable phi'(alpha)
        Objective function derivative. Returns a scalar.
    phi0 : float, optional
        Value of phi at 0

    """
    # `old_phi0`：float，可选参数
    # 先前点处的 phi 值
    # `derphi0`：float，可选参数
    # 0 处的 derphi 值
    # `c1`：float，可选参数
    # Armijo 条件规则的参数
    # `c2`：float，可选参数
    # 曲率条件规则的参数
    # `amax`, `amin`：float，可选参数
    # 最大和最小步长
    # `xtol`：float，可选参数
    # 可接受步长的相对容差

    Returns
    -------
    alpha : float
        步长，如果找不到合适的步长则返回 None
    phi : float
        新点 `alpha` 处的 `phi` 值
    phi0 : float
        `alpha=0` 处的 `phi` 值

    Notes
    -----
    使用 MINPACK 中的 DCSRCH 程序。
    
    参数 `c1` 和 `c2` 必须满足 ``0 < c1 < c2 < 1``，如 [1]_ 中所述。

    References
    ----------
    
    .. [1] Nocedal, J., & Wright, S. J. (2006). Numerical optimization.
       In Springer Series in Operations Research and Financial Engineering.
       (Springer Series in Operations Research and Financial Engineering).
       Springer Nature.
    """

    # 检查 c1 和 c2 的值是否合法
    _check_c1_c2(c1, c2)

    # 如果 phi0 为 None，则计算 phi(0.)
    if phi0 is None:
        phi0 = phi(0.)
    # 如果 derphi0 为 None，则计算 derphi(0.)
    if derphi0 is None:
        derphi0 = derphi(0.)

    # 如果 old_phi0 不为 None 并且 derphi0 不为零
    if old_phi0 is not None and derphi0 != 0:
        # 计算可能的最大步长 alpha1
        alpha1 = min(1.0, 1.01*2*(phi0 - old_phi0)/derphi0)
        # 如果 alpha1 小于 0，则设为 1.0
        if alpha1 < 0:
            alpha1 = 1.0
    else:
        # 否则，设定 alpha1 为 1.0
        alpha1 = 1.0

    # 最大迭代次数
    maxiter = 100

    # 调用 DCSRCH 函数对象
    dcsrch = DCSRCH(phi, derphi, c1, c2, xtol, amin, amax)
    # 调用 dcsrch 函数获取结果
    stp, phi1, phi0, task = dcsrch(
        alpha1, phi0=phi0, derphi0=derphi0, maxiter=maxiter
    )

    # 返回结果
    return stp, phi1, phi0
# 将 `line_search_wolfe1` 赋值给 `line_search`
line_search = line_search_wolfe1


#------------------------------------------------------------------------------
# Pure-Python Wolfe line and scalar searches
#------------------------------------------------------------------------------

# 注意: `line_search_wolfe2` 是公共的 `scipy.optimize.line_search`

# 定义了一个函数 `line_search_wolfe2`，用于寻找满足强 Wolfe 条件的步长 alpha
def line_search_wolfe2(f, myfprime, xk, pk, gfk=None, old_fval=None,
                       old_old_fval=None, args=(), c1=1e-4, c2=0.9, amax=None,
                       extra_condition=None, maxiter=10):
    """Find alpha that satisfies strong Wolfe conditions.

    Parameters
    ----------
    f : callable f(x,*args)
        Objective function.
    myfprime : callable f'(x,*args)
        Objective function gradient.
    xk : ndarray
        Starting point.
    pk : ndarray
        Search direction. The search direction must be a descent direction
        for the algorithm to converge.
    gfk : ndarray, optional
        Gradient value for x=xk (xk being the current parameter
        estimate). Will be recomputed if omitted.
    old_fval : float, optional
        Function value for x=xk. Will be recomputed if omitted.
    old_old_fval : float, optional
        Function value for the point preceding x=xk.
    args : tuple, optional
        Additional arguments passed to objective function.
    c1 : float, optional
        Parameter for Armijo condition rule.
    c2 : float, optional
        Parameter for curvature condition rule.
    amax : float, optional
        Maximum step size
    extra_condition : callable, optional
        A callable of the form ``extra_condition(alpha, x, f, g)``
        returning a boolean. Arguments are the proposed step ``alpha``
        and the corresponding ``x``, ``f`` and ``g`` values. The line search
        accepts the value of ``alpha`` only if this
        callable returns ``True``. If the callable returns ``False``
        for the step length, the algorithm will continue with
        new iterates. The callable is only called for iterates
        satisfying the strong Wolfe conditions.
    maxiter : int, optional
        Maximum number of iterations to perform.

    Returns
    -------
    alpha : float or None
        Alpha for which ``x_new = x0 + alpha * pk``,
        or None if the line search algorithm did not converge.
    fc : int
        Number of function evaluations made.
    gc : int
        Number of gradient evaluations made.
    new_fval : float or None
        New function value ``f(x_new)=f(x0+alpha*pk)``,
        or None if the line search algorithm did not converge.
    old_fval : float
        Old function value ``f(x0)``.
    new_slope : float or None
        The local slope along the search direction at the
        new value ``<myfprime(x_new), pk>``,
        or None if the line search algorithm did not converge.


    Notes
    -----
    Uses the line search algorithm to enforce strong Wolfe
    conditions. See Wright and Nocedal, 'Numerical Optimization',
    1999, pp. 59-61.
    
    """
    # 初始化计数器和存储变量，用于追踪函数调用次数及相关值
    fc = [0]
    gc = [0]
    gval = [None]
    gval_alpha = [None]

    # 定义函数 phi，用于计算给定步长 alpha 下的函数值
    def phi(alpha):
        # 增加函数计数器
        fc[0] += 1
        # 计算并返回函数在 xk + alpha * pk 处的值
        return f(xk + alpha * pk, *args)

    # 将 myfprime 赋值给 fprime，表示导数函数
    fprime = myfprime

    # 定义函数 derphi，用于计算给定步长 alpha 下的导数值
    def derphi(alpha):
        # 增加导数计数器
        gc[0] += 1
        # 计算并存储函数在 xk + alpha * pk 处的梯度值
        gval[0] = fprime(xk + alpha * pk, *args)  # 存储以备后用
        gval_alpha[0] = alpha
        # 返回导数值 np.dot(gval[0], pk)
        return np.dot(gval[0], pk)

    # 如果 gfk 为 None，则将其设为 xk 处的梯度值
    if gfk is None:
        gfk = fprime(xk, *args)
    # 计算初始导数值 derphi0
    derphi0 = np.dot(gfk, pk)

    # 如果存在额外条件函数 extra_condition
    if extra_condition is not None:
        # 定义函数 extra_condition2，带有当前梯度作为参数，避免不必要的重新评估
        def extra_condition2(alpha, phi):
            # 如果当前 alpha 与存储的 alpha 值不同，则重新计算导数值
            if gval_alpha[0] != alpha:
                derphi(alpha)
            # 计算 x = xk + alpha * pk
            x = xk + alpha * pk
            # 调用额外条件函数并返回其结果
            return extra_condition(alpha, x, phi, gval[0])
    else:
        # 如果没有额外条件函数，则将 extra_condition2 设为 None
        extra_condition2 = None

    # 调用 scalar_search_wolfe2 函数，寻找满足强 Wolfe 条件的最优步长
    alpha_star, phi_star, old_fval, derphi_star = scalar_search_wolfe2(
            phi, derphi, old_fval, old_old_fval, derphi0, c1, c2, amax,
            extra_condition2, maxiter=maxiter)

    # 如果未找到满足条件的步长，则发出警告
    if derphi_star is None:
        warn('The line search algorithm did not converge',
             LineSearchWarning, stacklevel=2)
    else:
        # 设置 derphi_star 为最近计算的导数值 gval[0]
        derphi_star = gval[0]

    # 返回计算得到的结果
    return alpha_star, fc[0], gc[0], phi_star, old_fval, derphi_star
    # 检查 Armijo 和 curvature 条件的合法性，确保它们在有效范围内
    _check_c1_c2(c1, c2)

    # 如果未提供 phi0 的值，则通过调用 phi(0.) 获取其值
    if phi0 is None:
        phi0 = phi(0.)

    # 如果未提供 derphi0 的值，则通过调用 derphi(0.) 获取其值
    if derphi0 is None:
        derphi0 = derphi(0.)

    # 设置初始步长为 0
    alpha0 = 0

    # 计算初始步长 alpha1 的推荐值
    if old_phi0 is not None and derphi0 != 0:
        alpha1 = min(1.0, 1.01*2*(phi0 - old_phi0)/derphi0)
    else:
        alpha1 = 1.0

    # 如果推荐的 alpha1 值小于 0，则将其设为 1.0
    if alpha1 < 0:
        alpha1 = 1.0

    # 如果指定了最大步长 amax，则将 alpha1 限制在其内
    if amax is not None:
        alpha1 = min(alpha1, amax)

    # 计算 phi(alpha1) 的值
    phi_a1 = phi(alpha1)
    # 在后续代码中会计算 derphi(alpha1) 的值

    # 记录 phi(0) 和 derphi(0) 的初始值
    phi_a0 = phi0
    derphi_a0 = derphi0

    # 如果未指定额外条件，则定义一个始终返回 True 的默认额外条件函数
    if extra_condition is None:
        def extra_condition(alpha, phi):
            return True
    for i in range(maxiter):
        # 进行最大迭代次数范围内的循环

        if alpha1 == 0 or (amax is not None and alpha0 > amax):
            # 如果 alpha1 等于 0 或者 alpha0 大于 amax（如果定义了的话）
            # 这种情况通常不应该发生。可能是增量低于机器精度？
            alpha_star = None
            # 设置 alpha_star 为 None
            phi_star = phi0
            # 设置 phi_star 为 phi0
            phi0 = old_phi0
            # 将 phi0 设置回 old_phi0
            derphi_star = None
            # 设置 derphi_star 为 None

            if alpha1 == 0:
                # 如果 alpha1 等于 0
                msg = 'Rounding errors prevent the line search from converging'
            else:
                # 否则
                msg = "The line search algorithm could not find a solution " + \
                      "less than or equal to amax: %s" % amax

            warn(msg, LineSearchWarning, stacklevel=2)
            # 发出警告消息，表示无法收敛到预期的结果
            break
            # 跳出循环

        not_first_iteration = i > 0
        # 检查是否不是第一次迭代
        if (phi_a1 > phi0 + c1 * alpha1 * derphi0) or \
           ((phi_a1 >= phi_a0) and not_first_iteration):
            # 如果 Armijo条件不满足，或者不是第一次迭代且强Wolfe条件不满足
            alpha_star, phi_star, derphi_star = \
                        _zoom(alpha0, alpha1, phi_a0,
                              phi_a1, derphi_a0, phi, derphi,
                              phi0, derphi0, c1, c2, extra_condition)
            # 调用 _zoom 函数进行详细搜索
            break
            # 跳出循环

        derphi_a1 = derphi(alpha1)
        # 计算在 alpha1 处的导数值
        if (abs(derphi_a1) <= -c2*derphi0):
            # 如果满足强Wolfe条件
            if extra_condition(alpha1, phi_a1):
                # 并且额外条件也满足
                alpha_star = alpha1
                # 设置 alpha_star 为 alpha1
                phi_star = phi_a1
                # 设置 phi_star 为 phi_a1
                derphi_star = derphi_a1
                # 设置 derphi_star 为 derphi_a1
                break
                # 跳出循环

        if (derphi_a1 >= 0):
            # 如果导数值大于等于 0
            alpha_star, phi_star, derphi_star = \
                        _zoom(alpha1, alpha0, phi_a1,
                              phi_a0, derphi_a1, phi, derphi,
                              phi0, derphi0, c1, c2, extra_condition)
            # 调用 _zoom 函数进行详细搜索
            break
            # 跳出循环

        alpha2 = 2 * alpha1  # 每次迭代将 alpha1 乘以二增加
        if amax is not None:
            # 如果定义了 amax
            alpha2 = min(alpha2, amax)
            # 将 alpha2 限制在 amax 之内
        alpha0 = alpha1
        # 更新 alpha0 为当前的 alpha1
        alpha1 = alpha2
        # 更新 alpha1 为当前的 alpha2
        phi_a0 = phi_a1
        # 更新 phi_a0 为当前的 phi_a1
        phi_a1 = phi(alpha1)
        # 计算在 alpha1 处的函数值
        derphi_a0 = derphi_a1
        # 更新 derphi_a0 为当前的 derphi_a1

    else:
        # 如果达到最大迭代次数仍未满足终止条件
        alpha_star = alpha1
        # 设置 alpha_star 为最后的 alpha1
        phi_star = phi_a1
        # 设置 phi_star 为最后的 phi_a1
        derphi_star = None
        # 设置 derphi_star 为 None
        warn('The line search algorithm did not converge',
             LineSearchWarning, stacklevel=2)
        # 发出警告消息，表示线搜索算法未收敛
def _zoom(a_lo, a_hi, phi_lo, phi_hi, derphi_lo,
          phi, derphi, phi0, derphi0, c1, c2, extra_condition):
    """Zoom stage of approximate linesearch satisfying strong Wolfe conditions.

    Part of the optimization algorithm in `scalar_search_wolfe2`.

    Notes
    -----
    Implements Algorithm 3.6 (zoom) in Wright and Nocedal,
    'Numerical Optimization', 1999, pp. 61.

    """

    # 最大迭代次数设为10
    maxiter = 10
    # 初始化迭代计数器
    i = 0
    # 用于判断是否进入二次插值的阈值
    delta1 = 0.2  # cubic interpolant check
    # 用于判断是否进入线性插值的阈值
    delta2 = 0.1  # quadratic interpolant check
    # 记录初始的函数值
    phi_rec = phi0
    # 记录初始步长
    a_rec = 0
    while True:
        # 循环直到找到满足条件的步长

        # 计算步长区间长度
        dalpha = a_hi - a_lo
        
        # 根据步长区间的大小确定 a 和 b 的顺序
        if dalpha < 0:
            a, b = a_hi, a_lo
        else:
            a, b = a_lo, a_hi

        # 使用三次插值法找到最小值
        # 使用 phi_lo, derphi_lo, phi_hi 和最新的 phi 值
        #
        # 如果结果靠近端点或者超出区间范围，则使用二次插值法
        if (i > 0):
            cchk = delta1 * dalpha
            a_j = _cubicmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi,
                            a_rec, phi_rec)
        
        # 如果是第一次迭代或者三次插值未找到合适的步长
        # 或者找到的步长靠近区间端点，则使用二次插值法
        # 如果二次插值仍然未找到合适步长，则使用二分法
        if (i == 0) or (a_j is None) or (a_j > b - cchk) or (a_j < a + cchk):
            qchk = delta2 * dalpha
            a_j = _quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi)
            if (a_j is None) or (a_j > b - qchk) or (a_j < a + qchk):
                a_j = a_lo + 0.5 * dalpha

        # 检查新的步长值 a_j

        phi_aj = phi(a_j)
        
        # 如果新的函数值超出预期范围，则更新区间并选择新的上界 a_hi
        if (phi_aj > phi0 + c1 * a_j * derphi0) or (phi_aj >= phi_lo):
            phi_rec = phi_hi
            a_rec = a_hi
            a_hi = a_j
            phi_hi = phi_aj
        else:
            derphi_aj = derphi(a_j)
            
            # 如果导数的绝对值满足额外条件，记录最优步长和其对应的函数值和导数值
            if abs(derphi_aj) <= -c2 * derphi0 and extra_condition(a_j, phi_aj):
                a_star = a_j
                val_star = phi_aj
                valprime_star = derphi_aj
                break
            
            # 否则根据导数符号选择新的下界 a_lo
            if derphi_aj * (a_hi - a_lo) >= 0:
                phi_rec = phi_hi
                a_rec = a_hi
                a_hi = a_lo
                phi_hi = phi_lo
            else:
                phi_rec = phi_lo
                a_rec = a_lo
                a_lo = a_j
                phi_lo = phi_aj
                derphi_lo = derphi_aj
        
        # 迭代计数器加一
        i += 1
        
        # 如果超过最大迭代次数，退出循环并返回空值
        if (i > maxiter):
            # 未找到合适的步长
            a_star = None
            val_star = None
            valprime_star = None
            break

    # 返回最优步长以及对应的函数值和导数值
    return a_star, val_star, valprime_star
#------------------------------------------------------------------------------
# Armijo line and scalar searches
#------------------------------------------------------------------------------

# Armijo线搜索和标量搜索

def line_search_armijo(f, xk, pk, gfk, old_fval, args=(), c1=1e-4, alpha0=1):
    """Minimize over alpha, the function ``f(xk+alpha pk)``.

    Parameters
    ----------
    f : callable
        要最小化的函数。
    xk : array_like
        当前点。
    pk : array_like
        搜索方向。
    gfk : array_like
        在点 `xk` 处的 `f` 的梯度。
    old_fval : float
        点 `xk` 处的 `f` 的值。
    args : tuple, optional
        可选参数。
    c1 : float, optional
        控制停止标准的值。
    alpha0 : scalar, optional
        优化开始时 `alpha` 的值。

    Returns
    -------
    alpha
        最小化函数的 `alpha` 值。
    f_count
        函数评估计数。
    f_val_at_alpha
        在最小化点 `alpha` 处的 `f` 的值。

    Notes
    -----
    使用插值算法（Armijo回溯法），如Wright和Nocedal在《数值优化》（1999年，第56-57页）中建议的那样。

    """
    xk = np.atleast_1d(xk)
    fc = [0]

    def phi(alpha1):
        fc[0] += 1
        return f(xk + alpha1*pk, *args)

    if old_fval is None:
        phi0 = phi(0.)
    else:
        phi0 = old_fval  # compute f(xk) -- done in past loop

    derphi0 = np.dot(gfk, pk)
    alpha, phi1 = scalar_search_armijo(phi, phi0, derphi0, c1=c1,
                                       alpha0=alpha0)
    return alpha, fc[0], phi1


def line_search_BFGS(f, xk, pk, gfk, old_fval, args=(), c1=1e-4, alpha0=1):
    """
    兼容性包装器，用于 `line_search_armijo`
    """
    r = line_search_armijo(f, xk, pk, gfk, old_fval, args=args, c1=c1,
                           alpha0=alpha0)
    return r[0], r[1], 0, r[2]


def scalar_search_armijo(phi, phi0, derphi0, c1=1e-4, alpha0=1, amin=0):
    """Minimize over alpha, the function ``phi(alpha)``.

    Uses the interpolation algorithm (Armijo backtracking) as suggested by
    Wright and Nocedal in 'Numerical Optimization', 1999, pp. 56-57

    alpha > 0 is assumed to be a descent direction.

    Returns
    -------
    alpha
        最小化函数的 `alpha` 值。
    phi1
        在最小化点 `alpha` 处的 `phi` 的值。

    """
    phi_a0 = phi(alpha0)
    if phi_a0 <= phi0 + c1*alpha0*derphi0:
        return alpha0, phi_a0

    # Otherwise, compute the minimizer of a quadratic interpolant:

    alpha1 = -(derphi0) * alpha0**2 / 2.0 / (phi_a0 - phi0 - derphi0 * alpha0)
    phi_a1 = phi(alpha1)

    if (phi_a1 <= phi0 + c1*alpha1*derphi0):
        return alpha1, phi_a1

    # Otherwise, loop with cubic interpolation until we find an alpha which
    # satisfies the first Wolfe condition (since we are backtracking, we will
    # assume that the value of alpha is not too small and satisfies the second
    # condition.

    # 否则，使用三次插值循环，直到找到满足第一Wolfe条件的 `alpha` 值
    # （由于我们在回溯，因此假设 `alpha` 的值不会太小并且满足第二个条件）。
    while alpha1 > amin:       # 当前步长大于最小步长时，继续执行
        factor = alpha0**2 * alpha1**2 * (alpha1-alpha0)
        # 计算公式中的 factor
        a = alpha0**2 * (phi_a1 - phi0 - derphi0*alpha1) - \
            alpha1**2 * (phi_a0 - phi0 - derphi0*alpha0)
        # 计算公式中的 a
        a = a / factor
        # 根据计算的 factor 和 a 计算 b
        b = -alpha0**3 * (phi_a1 - phi0 - derphi0*alpha1) + \
            alpha1**3 * (phi_a0 - phi0 - derphi0*alpha0)
        b = b / factor

        # 使用公式求解 alpha2
        alpha2 = (-b + np.sqrt(abs(b**2 - 3 * a * derphi0))) / (3.0*a)
        phi_a2 = phi(alpha2)

        # 检查 Armijo 条件
        if (phi_a2 <= phi0 + c1*alpha2*derphi0):
            return alpha2, phi_a2

        # 如果没有满足条件的步长，则进行调整
        if (alpha1 - alpha2) > alpha1 / 2.0 or (1 - alpha2/alpha1) < 0.96:
            alpha2 = alpha1 / 2.0

        # 更新 alpha0, alpha1, phi_a0, phi_a1
        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi_a2

    # 未找到合适的步长
    return None, phi_a1
#------------------------------------------------------------------------------
# Non-monotone line search for DF-SANE
#------------------------------------------------------------------------------

# 定义一个非单调回溯线搜索函数，用于 DF-SANE 算法

def _nonmonotone_line_search_cruz(f, x_k, d, prev_fs, eta,
                                  gamma=1e-4, tau_min=0.1, tau_max=0.5):
    """
    Nonmonotone backtracking line search as described in [1]_

    Parameters
    ----------
    f : callable
        返回一个元组 ``(f, F)`` 的函数，其中 ``f`` 是优化函数的值，``F`` 是残差。
    x_k : ndarray
        初始位置。
    d : ndarray
        搜索方向。
    prev_fs : float
        先前优化函数值的列表。应满足 ``len(prev_fs) <= M``，其中 ``M`` 是非单调性窗口参数。
    eta : float
        允许的优化函数增加量，参见 [1]_
    gamma, tau_min, tau_max : float, optional
        搜索参数，参见 [1]_

    Returns
    -------
    alpha : float
        步长
    xp : ndarray
        下一个位置
    fp : float
        下一个位置的优化函数值
    Fp : ndarray
        下一个位置的残差

    References
    ----------
    [1] "Spectral residual method without gradient information for solving
        large-scale nonlinear systems of equations." W. La Cruz,
        J.M. Martinez, M. Raydan. Math. Comp. **75**, 1429 (2006).

    """
    # 获取先前的优化函数值和最大值
    f_k = prev_fs[-1]
    f_bar = max(prev_fs)

    # 初始化正负步长
    alpha_p = 1
    alpha_m = 1
    alpha = 1

    while True:
        # 计算正步长下一个位置和对应的优化函数值
        xp = x_k + alpha_p * d
        fp, Fp = f(xp)

        # 检查正步长是否满足非单调性条件
        if fp <= f_bar + eta - gamma * alpha_p**2 * f_k:
            alpha = alpha_p
            break

        # 计算正步长的调整值
        alpha_tp = alpha_p**2 * f_k / (fp + (2*alpha_p - 1)*f_k)

        # 计算负步长下一个位置和对应的优化函数值
        xp = x_k - alpha_m * d
        fp, Fp = f(xp)

        # 检查负步长是否满足非单调性条件
        if fp <= f_bar + eta - gamma * alpha_m**2 * f_k:
            alpha = -alpha_m
            break

        # 计算负步长的调整值
        alpha_tm = alpha_m**2 * f_k / (fp + (2*alpha_m - 1)*f_k)

        # 对步长进行限制
        alpha_p = np.clip(alpha_tp, tau_min * alpha_p, tau_max * alpha_p)
        alpha_m = np.clip(alpha_tm, tau_min * alpha_m, tau_max * alpha_m)

    return alpha, xp, fp, Fp


def _nonmonotone_line_search_cheng(f, x_k, d, f_k, C, Q, eta,
                                   gamma=1e-4, tau_min=0.1, tau_max=0.5,
                                   nu=0.85):
    """
    Nonmonotone line search from [1]

    Parameters
    ----------
    f : callable
        返回一个元组 ``(f, F)`` 的函数，其中 ``f`` 是优化函数的值，``F`` 是残差。
    x_k : ndarray
        初始位置。
    d : ndarray
        搜索方向。
    f_k : float
        初始优化函数值。
    C, Q : float
        控制参数。在第一次迭代时，给定值为 Q=1.0, C=f_k。
    eta : float
        允许的优化函数增加量，参见 [1]_
    nu, gamma, tau_min, tau_max : float, optional
        搜索参数，参见 [1]_

    Returns
    -------

    """
    # TODO: To be continued
    """
    alpha : float
        步长
    xp : ndarray
        下一个位置
    fp : float
        下一个位置的优度函数值
    Fp : ndarray
        下一个位置的残差
    C : float
        控制参数 C 的新值
    Q : float
        控制参数 Q 的新值

    References
    ----------
    .. [1] W. Cheng & D.-H. Li, ''A derivative-free nonmonotone line
           search and its application to the spectral residual
           method'', IMA J. Numer. Anal. 29, 814 (2009).

    """
    # 初始化步长参数
    alpha_p = 1
    alpha_m = 1
    alpha = 1

    # 开始迭代计算合适的步长 alpha
    while True:
        # 计算正向步长位置 xp
        xp = x_k + alpha_p * d
        # 计算在 xp 处的优度函数值 fp 和残差 Fp
        fp, Fp = f(xp)

        # 检查优度函数是否满足条件
        if fp <= C + eta - gamma * alpha_p**2 * f_k:
            alpha = alpha_p
            break

        # 计算正向步长更新公式中的 alpha_tp
        alpha_tp = alpha_p**2 * f_k / (fp + (2*alpha_p - 1)*f_k)

        # 计算反向步长位置 xp
        xp = x_k - alpha_m * d
        # 计算在 xp 处的优度函数值 fp 和残差 Fp
        fp, Fp = f(xp)

        # 检查优度函数是否满足条件
        if fp <= C + eta - gamma * alpha_m**2 * f_k:
            alpha = -alpha_m
            break

        # 计算反向步长更新公式中的 alpha_tm
        alpha_tm = alpha_m**2 * f_k / (fp + (2*alpha_m - 1)*f_k)

        # 根据更新公式，对 alpha_p 和 alpha_m 进行裁剪和更新
        alpha_p = np.clip(alpha_tp, tau_min * alpha_p, tau_max * alpha_p)
        alpha_m = np.clip(alpha_tm, tau_min * alpha_m, tau_max * alpha_m)

    # 更新控制参数 C 和 Q
    Q_next = nu * Q + 1
    C = (nu * Q * (C + eta) + fp) / Q_next
    Q = Q_next

    # 返回计算得到的 alpha, xp, fp, Fp, C, Q
    return alpha, xp, fp, Fp, C, Q
```