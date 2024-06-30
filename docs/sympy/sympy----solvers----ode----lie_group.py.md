# `D:\src\scipysrc\sympy\sympy\solvers\ode\lie_group.py`

```
"""
This module contains the implementation of the internal helper functions for the lie_group hint for
dsolve. These helper functions apply different heuristics on the given equation
and return the solution. These functions are used by :py:meth:`sympy.solvers.ode.single.LieGroup`

References
=========

- `abaco1_simple`, `function_sum` and `chi`  are referenced from E.S Cheb-Terrab, L.G.S Duarte
and L.A,C.P da Mota, Computer Algebra Solving of First Order ODEs Using
Symmetry Methods, pp. 7 - pp. 8

- `abaco1_product`, `abaco2_similar`, `abaco2_unique_unknown`, `linear`  and `abaco2_unique_general`
are referenced from E.S. Cheb-Terrab, A.D. Roche, Symmetries and First Order
ODE Patterns, pp. 7 - pp. 12

- `bivariate` from Lie Groups and Differential Equations pp. 327 - pp. 329

"""
from itertools import islice

from sympy.core import Add, S, Mul, Pow
from sympy.core.exprtools import factor_terms
from sympy.core.function import Function, AppliedUndef, expand
from sympy.core.relational import Equality, Eq
from sympy.core.symbol import Symbol, Wild, Dummy, symbols
from sympy.functions import exp, log
from sympy.integrals.integrals import integrate
from sympy.polys import Poly
from sympy.polys.polytools import cancel, div
from sympy.simplify import (collect, powsimp,  # type: ignore
    separatevars, simplify)
from sympy.solvers import solve
from sympy.solvers.pde import pdsolve

from sympy.utilities import numbered_symbols
from sympy.solvers.deutils import _preprocess, ode_order
from .ode import checkinfsol


lie_heuristics = (
    "abaco1_simple",
    "abaco1_product",
    "abaco2_similar",
    "abaco2_unique_unknown",
    "abaco2_unique_general",
    "linear",
    "function_sum",
    "bivariate",
    "chi"
    )


def _ode_lie_group_try_heuristic(eq, heuristic, func, match, inf):
    # Define symbolic functions xi and eta
    xi = Function("xi")
    eta = Function("eta")
    # Extract the main function from func and its argument
    f = func.func
    x = func.args[0]
    # Extract variables y and h from the match dictionary
    y = match['y']
    h = match['h']
    # Initialize an empty list for temporary solutions
    tempsol = []
    # If inf is not provided, attempt to calculate it using infinitesimals function
    if not inf:
        try:
            inf = infinitesimals(eq, hint=heuristic, func=func, order=1, match=match)
        except ValueError:
            return None
    # 对于输入的每个不可分割的符号表达式infsim，执行以下操作
    for infsim in inf:
        # 计算 xiinf = infsim[xi(x, func)].subs(func, y)
        xiinf = (infsim[xi(x, func)]).subs(func, y)
        # 计算 etainf = infsim[eta(x, func)].subs(func, y)
        etainf = (infsim[eta(x, func)]).subs(func, y)
        
        # 检查条件 simplify(etainf/xiinf) == h 是否成立，如果成立则跳过当前循环
        if simplify(etainf/xiinf) == h:
            continue
        
        # 计算 rpde = f(x, y).diff(x)*xiinf + f(x, y).diff(y)*etainf
        rpde = f(x, y).diff(x)*xiinf + f(x, y).diff(y)*etainf
        
        # 求解偏微分方程 rpde = 0，返回其右手边的表达式
        r = pdsolve(rpde, func=f(x, y)).rhs
        
        # 求解偏微分方程 rpde - 1 = 0，返回其右手边的表达式
        s = pdsolve(rpde - 1, func=f(x, y)).rhs
        
        # 对 newcoord 列表中的每个坐标应用 _lie_group_remove 函数
        newcoord = [_lie_group_remove(coord) for coord in [r, s]]
        
        # 创建虚拟变量 r 和 s
        r = Dummy("r")
        s = Dummy("s")
        
        # 创建符号变量 C1
        C1 = Symbol("C1")
        
        # 将 newcoord 中的第一个和最后一个坐标分别赋给 rcoord 和 scoord
        rcoord = newcoord[0]
        scoord = newcoord[-1]
        
        # 尝试解方程组 [r - rcoord, s - scoord]，解出 x 和 y 的值的字典
        try:
            sol = solve([r - rcoord, s - scoord], x, y, dict=True)
            # 如果没有解，则跳过当前循环
            if sol == []:
                continue
        except NotImplementedError:
            # 如果 solve 函数抛出 NotImplementedError，则跳过当前循环
            continue
        else:
            # 如果成功解出方程组，则取第一个解
            sol = sol[0]
            # 将解中的 x 和 y 分别赋给 xsub 和 ysub
            xsub = sol[x]
            ysub = sol[y]
            
            # 计算 num = scoord.diff(x) + scoord.diff(y)*h 和 denom = rcoord.diff(x) + rcoord.diff(y)*h
            num = simplify(scoord.diff(x) + scoord.diff(y)*h)
            denom = simplify(rcoord.diff(x) + rcoord.diff(y)*h)
            
            # 如果 num 和 denom 都不为零
            if num and denom:
                # 计算 diffeq = simplify((num/denom).subs([(x, xsub), (y, ysub)]))
                diffeq = simplify((num/denom).subs([(x, xsub), (y, ysub)]))
                
                # 尝试使用 separatevars 函数分离变量 r 和 s
                sep = separatevars(diffeq, symbols=[r, s], dict=True)
                
                # 如果成功分离，则进行积分和符号替换操作
                if sep:
                    # 尝试积分操作 integrate((1/sep[s]), s) + C1 - integrate(sep['coeff']*sep[r], r)
                    deq = integrate((1/sep[s]), s) + C1 - integrate(sep['coeff']*sep[r], r)
                    # 将 r 和 s 替换回原始坐标
                    deq = deq.subs([(r, rcoord), (s, scoord)])
                    
                    # 尝试解方程 deq 关于 y
                    try:
                        sdeq = solve(deq, y)
                    except NotImplementedError:
                        # 如果 solve 函数抛出 NotImplementedError，则将 deq 添加到 tempsol 列表中
                        tempsol.append(deq)
                    else:
                        # 如果成功解出方程，则返回以 f(x) = sol 为形式的列表
                        return [Eq(f(x), sol) for sol in sdeq]
            
            # 如果 denom 为真值，即 (ds/dr) 为零，说明 s 是常数
            elif denom:
                # 返回以 f(x) = solve(scoord - C1, y)[0] 为形式的列表
                return [Eq(f(x), solve(scoord - C1, y)[0])]
            
            # 如果 num 为真值，即 (dr/ds) 为零，说明 r 是常数
            elif num:
                # 返回以 f(x) = solve(rcoord - C1, y)[0] 为形式的列表
                return [Eq(f(x), solve(rcoord - C1, y)[0])]
    
    # 如果 tempsol 不为空，则返回以 f(x) = sol.subs(y, f(x)) = 0 为形式的列表
    if tempsol:
        return [Eq(sol.subs(y, f(x)), 0) for sol in tempsol]
    
    # 如果以上条件都不满足，则返回 None
    return None
# 定义一个函数 `_ode_lie_group`，用于求解 Lie 群的无限小元素，以解决给定的常微分方程
def _ode_lie_group(s, func, order, match):
    # 使用默认的 Lie 群启发法 heuristics
    heuristics = lie_heuristics
    # 初始化一个空字典 inf
    inf = {}
    # 获取函数 func 的表达式 f
    f = func.func
    # 获取函数 func 的自变量 x
    x = func.args[0]
    # 对函数 func 求关于 x 的导数 df
    df = func.diff(x)
    # 创建符号函数 xi 和 eta
    xi = Function("xi")
    eta = Function("eta")
    # 从匹配对象 match 中获取 'xi' 和 'eta' 的值
    xis = match['xi']
    etas = match['eta']
    # 从 match 中弹出 'y'，如果存在则计算 h，否则创建一个虚拟变量 y 并计算 h
    y = match.pop('y', None)
    if y:
        h = -simplify(match[match['d']] / match[match['e']])
    else:
        y = Dummy("y")
        h = s.subs(func, y)

    # 如果 xis 和 etas 都不为 None，则将其作为字典形式存入 inf
    if xis is not None and etas is not None:
        inf = [{xi(x, f(x)): S(xis), eta(x, f(x)): S(etas)}]

        # 如果满足条件 Eq(df, s)，则加入用户定义的启发法到 heuristics 中
        if checkinfsol(Eq(df, s), inf, func=f(x), order=1)[0][0]:
            heuristics = ["user_defined"] + list(heuristics)

    # 将 'h' 和 'y' 存入 match 字典
    match = {'h': h, 'y': y}

    # 这样做是为了如果任何一个启发法引发 ValueError，可以尝试使用另一个启发法
    sol = None
    # 遍历 heuristics 中的每一个启发法
    for heuristic in heuristics:
        # 调用 _ode_lie_group_try_heuristic 函数尝试当前启发法 heuristic
        sol = _ode_lie_group_try_heuristic(Eq(df, s), heuristic, func, match, inf)
        # 如果找到解，则返回该解
        if sol:
            return sol
    # 如果所有启发法都未找到解，则返回空值 sol
    return sol


# 定义一个函数 infinitesimals，计算普通微分方程的 Lie 群的无限小元素 xi 和 eta
def infinitesimals(eq, func=None, order=None, hint='default', match=None):
    """
    The infinitesimal functions of an ordinary differential equation, `\xi(x,y)`
    and `\eta(x,y)`, are the infinitesimals of the Lie group of point transformations
    for which the differential equation is invariant. So, the ODE `y'=f(x,y)`
    would admit a Lie group `x^*=X(x,y;\varepsilon)=x+\varepsilon\xi(x,y)`,
    `y^*=Y(x,y;\varepsilon)=y+\varepsilon\eta(x,y)` such that `(y^*)'=f(x^*, y^*)`.
    A change of coordinates, to `r(x,y)` and `s(x,y)`, can be performed so this Lie group
    becomes the translation group, `r^*=r` and `s^*=s+\varepsilon`.
    They are tangents to the coordinate curves of the new system.

    Consider the transformation `(x, y) \to (X, Y)` such that the
    differential equation remains invariant. `\xi` and `\eta` are the tangents to
    the transformed coordinates `X` and `Y`, at `\varepsilon=0`.

    .. math:: \left(\frac{\partial X(x,y;\varepsilon)}{\partial\varepsilon
                }\right)|_{\varepsilon=0} = \xi,
              \left(\frac{\partial Y(x,y;\varepsilon)}{\partial\varepsilon
                }\right)|_{\varepsilon=0} = \eta,

    The infinitesimals can be found by solving the following PDE:

        >>> from sympy import Function, Eq, pprint
        >>> from sympy.abc import x, y
        >>> xi, eta, h = map(Function, ['xi', 'eta', 'h'])
        >>> h = h(x, y)  # dy/dx = h
        >>> eta = eta(x, y)
        >>> xi = xi(x, y)
        >>> genform = Eq(eta.diff(x) + (eta.diff(y) - xi.diff(x))*h
        ... - (xi.diff(y))*h**2 - xi*(h.diff(x)) - eta*(h.diff(y)), 0)
        >>> pprint(genform)
        /d               d           \                     d              2       d                       d             d
        |--(eta(x, y)) - --(xi(x, y))|*h(x, y) - eta(x, y)*--(h(x, y)) - h (x, y)*--(xi(x, y)) - xi(x, y)*--(h(x, y)) + --(eta(x, y)) = 0
        \dy              dx          /                     dy                     dy                      dx            dx

    """
    Solving the above mentioned PDE is not trivial, and can be solved only by
    making intelligent assumptions for `\xi` and `\eta` (heuristics). Once an
    infinitesimal is found, the attempt to find more heuristics stops. This is done to
    optimise the speed of solving the differential equation. If a list of all the
    infinitesimals is needed, ``hint`` should be flagged as ``all``, which gives
    the complete list of infinitesimals. If the infinitesimals for a particular
    heuristic needs to be found, it can be passed as a flag to ``hint``.

    Examples
    ========

    >>> from sympy import Function
    >>> from sympy.solvers.ode.lie_group import infinitesimals
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> eq = f(x).diff(x) - x**2*f(x)
    >>> infinitesimals(eq)
    [{eta(x, f(x)): exp(x**3/3), xi(x, f(x)): 0}]

    References
    ==========

    - Solving differential equations by Symmetry Groups,
      John Starrett, pp. 1 - pp. 14

    """

    # 如果传入的方程是一个等式，将其转换为左侧减右侧的形式
    if isinstance(eq, Equality):
        eq = eq.lhs - eq.rhs
    # 如果未提供函数参数，则预处理方程并获取函数
    if not func:
        eq, func = _preprocess(eq)
    # 获取函数中的变量列表
    variables = func.args
    # 如果变量数量不等于1，则抛出值错误，因为常微分方程只能有一个自变量
    if len(variables) != 1:
        raise ValueError("ODE's have only one independent variable")
        else:
            # 取第一个变量作为自变量
            x = variables[0]
            # 如果没有指定求解顺序，调用ode_order函数确定
            if not order:
                order = ode_order(eq, func)
            # 如果方程阶数不为1，抛出未实现错误
            if order != 1:
                raise NotImplementedError("Infinitesimals for only "
                    "first order ODE's have been implemented")
            else:
                # 求函数func关于自变量x的导数
                df = func.diff(x)
                # 匹配形如 a*df + b 的微分方程
                a = Wild('a', exclude=[df])
                b = Wild('b', exclude=[df])
                if match:  # 如果使用了match参数（由lie_group提示使用）
                    h = match['h']
                    y = match['y']
                else:
                    # 将方程展开并进行收集，然后尝试匹配微分方程形式
                    match = collect(expand(eq), df).match(a*df + b)
                    if match:
                        h = -simplify(match[b] / match[a])
                    else:
                        try:
                            # 尝试解微分方程eq得到df的解
                            sol = solve(eq, df)
                        except NotImplementedError:
                            raise NotImplementedError("Infinitesimals for the "
                                "first order ODE could not be found")
                        else:
                            h = sol[0]  # 取得一个解的无穷小
                    # 创建一个虚拟符号y，用于替代函数func
                    y = Dummy("y")
                    # 将h中的func替换为y
                    h = h.subs(func, y)

                # 创建一个虚拟符号u
                u = Dummy("u")
                # 计算h关于自变量x和y的偏导数
                hx = h.diff(x)
                hy = h.diff(y)
                # 计算微分方程的逆ODE
                hinv = ((1/h).subs([(x, u), (y, x)])).subs(u, y)
                # 将计算结果存入match字典
                match = {'h': h, 'func': func, 'hx': hx, 'hy': hy, 'y': y, 'hinv': hinv}

                # 根据提示参数进行处理
                if hint == 'all':
                    xieta = []
                    # 遍历lie_heuristics中的启发式方法
                    for heuristic in lie_heuristics:
                        # 获取全局函数lie_heuristic_XXXX，并传入match参数进行处理
                        function = globals()['lie_heuristic_' + heuristic]
                        # 获得启发式方法返回的无穷小列表
                        inflist = function(match, comp=True)
                        if inflist:
                            # 将未包含在xieta中的无穷小添加进去
                            xieta.extend([inf for inf in inflist if inf not in xieta])
                    if xieta:
                        return xieta
                    else:
                        raise NotImplementedError("Infinitesimals could not be found for "
                            "the given ODE")

                elif hint == 'default':
                    # 遍历lie_heuristics中的启发式方法
                    for heuristic in lie_heuristics:
                        # 获取全局函数lie_heuristic_XXXX，并传入match参数进行处理
                        function = globals()['lie_heuristic_' + heuristic]
                        # 获得启发式方法返回的无穷小列表
                        xieta = function(match, comp=False)
                        if xieta:
                            return xieta

                    raise NotImplementedError("Infinitesimals could not be found for"
                        " the given ODE")

                elif hint not in lie_heuristics:
                     # 抛出值错误，指出未识别的启发式方法
                     raise ValueError("Heuristic not recognized: " + hint)

                else:
                     # 获取全局函数lie_heuristic_XXXX，并传入match参数进行处理
                     function = globals()['lie_heuristic_' + hint]
                     # 获得启发式方法返回的无穷小列表
                     xieta = function(match, comp=True)
                     if xieta:
                         return xieta
                     else:
                         raise ValueError("Infinitesimals could not be found using the"
                             " given heuristic")
def lie_heuristic_abaco1_simple(match, comp=False):
    r"""
    The first heuristic uses the following four sets of
    assumptions on `\xi` and `\eta`

    .. math:: \xi = 0, \eta = f(x)

    .. math:: \xi = 0, \eta = f(y)

    .. math:: \xi = f(x), \eta = 0

    .. math:: \xi = f(y), \eta = 0

    The success of this heuristic is determined by algebraic factorisation.
    For the first assumption `\xi = 0` and `\eta` to be a function of `x`, the PDE

    .. math:: \frac{\partial \eta}{\partial x} + \left(\frac{\partial \eta}{\partial y}
                - \frac{\partial \xi}{\partial x}\right)h
                - \frac{\partial \xi}{\partial y}h^{2}
                - \xi\frac{\partial h}{\partial x} - \eta\frac{\partial h}{\partial y} = 0

    reduces to `f'(x) - f\frac{\partial h}{\partial y} = 0`
    If `\frac{\partial h}{\partial y}` is a function of `x`, then this can usually
    be integrated easily. A similar idea is applied to the other 3 assumptions as well.


    References
    ==========

    - E.S Cheb-Terrab, L.G.S Duarte and L.A,C.P da Mota, Computer Algebra
      Solving of First Order ODEs Using Symmetry Methods, pp. 8
    """

    xieta = []  # Initialize an empty list to store dictionaries of xi and eta values

    y = match['y']  # Extract 'y' from the input match dictionary
    h = match['h']  # Extract 'h' from the input match dictionary
    func = match['func']  # Extract 'func' from the input match dictionary
    x = func.args[0]  # Extract the first argument of 'func' as 'x'
    hx = match['hx']  # Extract 'hx' from the input match dictionary
    hy = match['hy']  # Extract 'hy' from the input match dictionary

    xi = Function('xi')(x, func)  # Define xi as a function of 'x' and 'func'
    eta = Function('eta')(x, func)  # Define eta as a function of 'x' and 'func'

    hysym = hy.free_symbols  # Get the free symbols in 'hy'
    if y not in hysym:  # Check if 'y' is not a free symbol in 'hy'
        try:
            fx = exp(integrate(hy, x))  # Attempt to integrate 'hy' with respect to 'x'
        except NotImplementedError:
            pass  # If integration fails, do nothing
        else:
            inf = {xi: S.Zero, eta: fx}  # Create a dictionary with xi=0 and eta=fx
            if not comp:
                return [inf]  # Return a list containing 'inf' if comp is False
            if comp and inf not in xieta:
                xieta.append(inf)  # Append 'inf' to 'xieta' if comp is True and 'inf' not already in 'xieta'

    factor = hy/h  # Calculate hy/h
    facsym = factor.free_symbols  # Get the free symbols in 'factor'
    if x not in facsym:  # Check if 'x' is not a free symbol in 'factor'
        try:
            fy = exp(integrate(factor, y))  # Attempt to integrate 'factor' with respect to 'y'
        except NotImplementedError:
            pass  # If integration fails, do nothing
        else:
            inf = {xi: S.Zero, eta: fy.subs(y, func)}  # Create a dictionary with xi=0 and eta=fy(y=func)
            if not comp:
                return [inf]  # Return a list containing 'inf' if comp is False
            if comp and inf not in xieta:
                xieta.append(inf)  # Append 'inf' to 'xieta' if comp is True and 'inf' not already in 'xieta'

    factor = -hx/h  # Calculate -hx/h
    facsym = factor.free_symbols  # Get the free symbols in 'factor'
    if y not in facsym:  # Check if 'y' is not a free symbol in 'factor'
        try:
            fx = exp(integrate(factor, x))  # Attempt to integrate 'factor' with respect to 'x'
        except NotImplementedError:
            pass  # If integration fails, do nothing
        else:
            inf = {xi: fx, eta: S.Zero}  # Create a dictionary with xi=fx and eta=0
            if not comp:
                return [inf]  # Return a list containing 'inf' if comp is False
            if comp and inf not in xieta:
                xieta.append(inf)  # Append 'inf' to 'xieta' if comp is True and 'inf' not already in 'xieta'

    factor = -hx/(h**2)  # Calculate -hx/(h^2)
    facsym = factor.free_symbols  # Get the free symbols in 'factor'
    if x not in facsym:  # Check if 'x' is not a free symbol in 'factor'
        try:
            fy = exp(integrate(factor, y))  # Attempt to integrate 'factor' with respect to 'y'
        except NotImplementedError:
            pass  # If integration fails, do nothing
        else:
            inf = {xi: fy.subs(y, func), eta: S.Zero}  # Create a dictionary with xi=fy(y=func) and eta=0
            if not comp:
                return [inf]  # Return a list containing 'inf' if comp is False
            if comp and inf not in xieta:
                xieta.append(inf)  # Append 'inf' to 'xieta' if comp is True and 'inf' not already in 'xieta'

    if xieta:  # If xieta is not empty
        return xieta  # Return xieta, which contains the collected dictionaries of xi and eta values

def lie_heuristic_abaco1_product(match, comp=False):
    r"""
    """
    根据特定的启发式方法，从匹配对象中提取 `\xi` 和 `\eta` 的值。

    Parameters
    ----------
    match : dict
        包含以下键的字典:
        - 'y': y 变量的值
        - 'h': 函数 h(x, y)
        - 'hinv': 函数 h(y, x) 的倒数
        - 'func': 函数 func(x)

    Returns
    -------
    list of dict
        包含提取出的 {eta: eta_value, xi: xi_value} 字典的列表

    References
    ==========
    - E.S. Cheb-Terrab, A.D. Roche, Symmetries and First Order
      ODE Patterns, pp. 7 - pp. 8

    """

    xieta = []  # 初始化一个空列表，用于存储提取出的 {eta, xi} 字典

    y = match['y']  # 从匹配对象中获取 y 的值
    h = match['h']  # 从匹配对象中获取 h(x, y) 的函数
    hinv = match['hinv']  # 从匹配对象中获取 h(y, x) 的倒数函数
    func = match['func']  # 从匹配对象中获取 func(x) 的函数
    x = func.args[0]  # 获取 func(x) 的参数 x

    # 计算 xi = f(x)*g(y) 的情况
    xi = Function('xi')(x, func)  # 定义 xi(x, func)
    eta = Function('eta')(x, func)  # 定义 eta(x, func)

    # 处理第一种启发式假设
    inf = separatevars(((log(h).diff(y)).diff(x))/h**2, dict=True, symbols=[x, y])
    if inf and inf['coeff']:
        fx = inf[x]
        gy = simplify(fx*((1/(fx*h)).diff(x)))
        gysyms = gy.free_symbols
        if x not in gysyms:
            gy = exp(integrate(gy, y))
            inf = {eta: S.Zero, xi: (fx*gy).subs(y, func)}
            if not comp:  # 如果不需要比较，直接返回结果
                return [inf]
            if comp and inf not in xieta:  # 如果需要比较且结果不在列表中，则添加到列表
                xieta.append(inf)

    # 处理第二种启发式假设
    u1 = Dummy("u1")
    inf = separatevars(((log(hinv).diff(y)).diff(x))/hinv**2, dict=True, symbols=[x, y])
    if inf and inf['coeff']:
        fx = inf[x]
        gy = simplify(fx*((1/(fx*hinv)).diff(x)))
        gysyms = gy.free_symbols
        if x not in gysyms:
            gy = exp(integrate(gy, y))
            etaval = fx*gy
            etaval = (etaval.subs([(x, u1), (y, x)])).subs(u1, y)
            inf = {eta: etaval.subs(y, func), xi: S.Zero}
            if not comp:  # 如果不需要比较，直接返回结果
                return [inf]
            if comp and inf not in xieta:  # 如果需要比较且结果不在列表中，则添加到列表
                xieta.append(inf)

    if xieta:  # 如果列表中有结果，则返回列表
        return xieta
def lie_heuristic_bivariate(match, comp=False):
    r"""
    The third heuristic assumes the infinitesimals `\xi` and `\eta`
    to be bi-variate polynomials in `x` and `y`. The assumption made here
    for the logic below is that `h` is a rational function in `x` and `y`
    though that may not be necessary for the infinitesimals to be
    bivariate polynomials. The coefficients of the infinitesimals
    are found out by substituting them in the PDE and grouping similar terms
    that are polynomials and since they form a linear system, solve and check
    for non trivial solutions. The degree of the assumed bivariates
    are increased till a certain maximum value.

    References
    ==========
    - Lie Groups and Differential Equations
      pp. 327 - pp. 329

    """

    # 从匹配对象中获取函数及其偏导数信息
    h = match['h']    # 主函数 h
    hx = match['hx']  # h 对 x 的偏导数
    hy = match['hy']  # h 对 y 的偏导数
    func = match['func']  # 匹配的函数表达式
    x = func.args[0]   # 函数中的自变量 x
    y = match['y']     # 函数中的另一个自变量 y

    # 定义二元多项式函数 xi(x, func) 和 eta(x, func)
    xi = Function('xi')(x, func)
    eta = Function('eta')(x, func)
    # 检查函数 h 是否为有理函数
    if h.is_rational_function():
        # 计算无穷小的最大次数
        etax, etay, etad, xix, xiy, xid = symbols("etax etay etad xix xiy xid")
        # 构造偏微分方程
        ipde = etax + (etay - xix)*h - xiy*h**2 - xid*hx - etad*hy
        # 化简偏微分方程并分别得到分子和分母
        num, denom = cancel(ipde).as_numer_denom()
        # 计算多项式的总次数
        deg = Poly(num, x, y).total_degree()
        # 定义 deta 和 dxi 函数
        deta = Function('deta')(x, y)
        dxi = Function('dxi')(x, y)
        # 重新定义偏微分方程
        ipde = (deta.diff(x) + (deta.diff(y) - dxi.diff(x))*h - (dxi.diff(y))*h**2
            - dxi*hx - deta*hy)
        # 定义符号
        xieq = Symbol("xi0")
        etaeq = Symbol("eta0")

        # 循环求解每个次数的解
        for i in range(deg + 1):
            if i:
                # 累加新的符号项
                xieq += Add(*[
                    Symbol("xi_" + str(power) + "_" + str(i - power))*x**power*y**(i - power)
                    for power in range(i + 1)])
                etaeq += Add(*[
                    Symbol("eta_" + str(power) + "_" + str(i - power))*x**power*y**(i - power)
                    for power in range(i + 1)])
            
            # 计算偏微分方程的分子并化简
            pden, denom = (ipde.subs({dxi: xieq, deta: etaeq}).doit()).as_numer_denom()
            pden = expand(pden)

            # 如果分子是多项式且为多项式相加形式，则组成字典
            if pden.is_polynomial(x, y) and pden.is_Add:
                polyy = Poly(pden, x, y).as_dict()
            
            # 如果存在多项式字典
            if polyy:
                # 计算符号集合
                symset = xieq.free_symbols.union(etaeq.free_symbols) - {x, y}
                # 求解多项式的值
                soldict = solve(polyy.values(), *symset)
                if isinstance(soldict, list):
                    soldict = soldict[0]
                # 如果求解结果中有任何值
                if any(soldict.values()):
                    # 替换符号并缩放参数
                    xired = xieq.subs(soldict)
                    etared = etaeq.subs(soldict)
                    # 通过替换参数求解无穷小
                    dict_ = dict.fromkeys(symset, 1)
                    inf = {eta: etared.subs(dict_).subs(y, func),
                        xi: xired.subs(dict_).subs(y, func)}
                    # 返回解
                    return [inf]
# 定义一个函数 lie_heuristic_chi，用于实现第四个启发式算法，用于寻找满足偏微分方程的函数 \chi(x, y)。
# 偏微分方程为 \frac{d\chi}{dx} + h\frac{d\chi}{dx} - \frac{\partial h}{\partial y}\chi = 0。

# 函数接受一个名为 match 的字典作为参数，可选参数 comp 为 False。
def lie_heuristic_chi(match, comp=False):
    r"""
    The aim of the fourth heuristic is to find the function `\chi(x, y)`
    that satisfies the PDE `\frac{d\chi}{dx} + h\frac{d\chi}{dx}
    - \frac{\partial h}{\partial y}\chi = 0`.

    This assumes `\chi` to be a bivariate polynomial in `x` and `y`. By intuition,
    `h` should be a rational function in `x` and `y`. The method used here is
    to substitute a general binomial for `\chi` up to a certain maximum degree
    is reached. The coefficients of the polynomials, are calculated by by collecting
    terms of the same order in `x` and `y`.

    After finding `\chi`, the next step is to use `\eta = \xi*h + \chi`, to
    determine `\xi` and `\eta`. This can be done by dividing `\chi` by `h`
    which would give `-\xi` as the quotient and `\eta` as the remainder.


    References
    ==========
    - E.S Cheb-Terrab, L.G.S Duarte and L.A,C.P da Mota, Computer Algebra
      Solving of First Order ODEs Using Symmetry Methods, pp. 8

    """

    # 从 match 字典中获取键为 'h' 的值，表示函数 h(x, y)
    h = match['h']
    # 从 match 字典中获取键为 'hy' 的值，表示 h(x, y) 对 y 的偏导数
    hy = match['hy']
    # 从 match 字典中获取键为 'func' 的值，表示函数 func(x, y)
    func = match['func']
    # 获取 func 函数的第一个参数 x
    x = func.args[0]
    # 从 match 字典中获取键为 'y' 的值，表示函数的第二个参数 y
    y = match['y']
    # 定义一个关于 x 和 func 的未知函数 xi(x, func)
    xi = Function('xi')(x, func)
    # 定义一个关于 x 和 func 的未知函数 eta(x, func)
    eta = Function('eta')(x, func)
    # 检查函数 h 是否是有理函数
    if h.is_rational_function():
        # 定义符号变量 schi, schix, schiy
        schi, schix, schiy = symbols("schi, schix, schiy")
        # 构造方程 cpde = schix + h*schiy - hy*schi
        cpde = schix + h*schiy - hy*schi
        # 化简 cpde 的分子和分母
        num, denom = cancel(cpde).as_numer_denom()
        # 计算多项式 num 关于变量 x, y 的总次数
        deg = Poly(num, x, y).total_degree()

        # 定义函数 chi(x, y) 和其偏导数 chix, chiy
        chi = Function('chi')(x, y)
        chix = chi.diff(x)
        chiy = chi.diff(y)
        
        # 更新方程 cpde 为 chix + h*chiy - hy*chi
        cpde = chix + h*chiy - hy*chi
        
        # 定义符号变量 chieq 为 chi，并根据 deg 构造多项式
        chieq = Symbol("chi")
        for i in range(1, deg + 1):
            chieq += Add(*[
                Symbol("chi_" + str(power) + "_" + str(i - power))*x**power*y**(i - power)
                for power in range(i + 1)])
        
        # 将 chi 替换为 chieq，并求解 cpde 的数值结果
        cnum, cden = cancel(cpde.subs({chi : chieq}).doit()).as_numer_denom()
        cnum = expand(cnum)
        
        # 如果 cnum 是关于 x, y 的多项式并且是加法表达式
        if cnum.is_polynomial(x, y) and cnum.is_Add:
            # 将 cnum 转化为多项式字典形式
            cpoly = Poly(cnum, x, y).as_dict()
            if cpoly:
                # 获取 chieq 中自由的符号集合 solsyms
                solsyms = chieq.free_symbols - {x, y}
                # 求解 cpoly 中的方程组
                soldict = solve(cpoly.values(), *solsyms)
                # 如果 soldict 是列表，则取第一个字典
                if isinstance(soldict, list):
                    soldict = soldict[0]
                # 如果 soldict 中有任何值
                if any(soldict.values()):
                    # 更新 chieq 为其解的结果
                    chieq = chieq.subs(soldict)
                    dict_ = dict.fromkeys(solsyms, 1)
                    chieq = chieq.subs(dict_)
                    # 使用 div 函数将 chieq 拆分为 xic 和 etac
                    xic, etac = div(chieq, h)
                    # 构造结果字典 inf，包含 eta 和 xi 的值
                    inf = {eta: etac.subs(y, func), xi: -xic.subs(y, func)}
                    # 返回包含 inf 的列表
                    return [inf]
# 定义一个启发式函数，计算 `xi` 和 `eta` 的可能值，基于给定的匹配数据
def lie_heuristic_function_sum(match, comp=False):
    r"""
    This heuristic uses the following two assumptions on `\xi` and `\eta`

    .. math:: \eta = 0, \xi = f(x) + g(y)

    .. math:: \eta = f(x) + g(y), \xi = 0

    The first assumption of this heuristic holds good if

    .. math:: \frac{\partial}{\partial y}[(h\frac{\partial^{2}}{
                \partial x^{2}}(h^{-1}))^{-1}]

    is separable in `x` and `y`,

    1. The separated factors containing `y` is `\frac{\partial g}{\partial y}`.
       From this `g(y)` can be determined.
    2. The separated factors containing `x` is `f''(x)`.
    3. `h\frac{\partial^{2}}{\partial x^{2}}(h^{-1})` equals
       `\frac{f''(x)}{f(x) + g(y)}`. From this `f(x)` can be determined.

    The second assumption holds good if `\frac{dy}{dx} = h(x, y)` is rewritten as
    `\frac{dy}{dx} = \frac{1}{h(y, x)}` and the same properties of the first
    assumption satisfies. After obtaining `f(x)` and `g(y)`, the coordinates
    are again interchanged, to get `\eta` as `f(x) + g(y)`.

    For both assumptions, the constant factors are separated among `g(y)`
    and `f''(x)`, such that `f''(x)` obtained from 3] is the same as that
    obtained from 2]. If not possible, then this heuristic fails.


    References
    ==========
    - E.S. Cheb-Terrab, A.D. Roche, Symmetries and First Order
      ODE Patterns, pp. 7 - pp. 8

    """

    # 提取匹配数据中的相关变量和函数
    xieta = []  # 存储可能的 xi 和 eta 的值的列表
    h = match['h']  # 获取匹配数据中的函数 h
    func = match['func']  # 获取匹配数据中的函数 func
    hinv = match['hinv']  # 获取匹配数据中的函数 hinv
    x = func.args[0]  # 从 func 中获取自变量 x
    y = match['y']  # 获取匹配数据中的变量 y
    xi = Function('xi')(x, func)  # 定义 xi 作为函数 xi(x, func)
    eta = Function('eta')(x, func)  # 定义 eta 作为函数 eta(x, func)
    # 对于每个因子 odefac，分别计算其乘积和其导数相对于 x 的二阶导数
    for odefac in [h, hinv]:
        factor = odefac*((1/odefac).diff(x, 2))
        
        # 使用 separatevars 函数分离 (1/factor) 关于 y 的导数，并以字典形式返回
        sep = separatevars((1/factor).diff(y), dict=True, symbols=[x, y])
        
        # 检查分离结果是否有效，并且包含系数和 x、y
        if sep and sep['coeff'] and sep[x].has(x) and sep[y].has(y):
            # 创建一个虚拟符号 k
            k = Dummy("k")
            
            # 尝试计算积分 sep[y] 关于 y 的积分乘以 k
            try:
                gy = k*integrate(sep[y], y)
            except NotImplementedError:
                # 如果积分失败，则跳过当前循环
                pass
            else:
                # 计算 fdd，即 1 / (k * sep[x] * sep['coeff'])
                fdd = 1/(k*sep[x]*sep['coeff'])
                
                # 计算 fx，即 (fdd / factor - gy)，并简化结果
                fx = simplify(fdd/factor - gy)
                
                # 检查 fx 的二阶导数是否等于 fdd
                check = simplify(fx.diff(x, 2) - fdd)
                
                # 如果 fx 不为空
                if fx:
                    # 如果 check 为零，修正 fx 和 gy 的值
                    if not check:
                        fx = fx.subs(k, 1)
                        gy = (gy/k)
                    else:
                        # 否则，解方程 check = 0，得到 k 的解
                        sol = solve(check, k)
                        if sol:
                            sol = sol[0]
                            fx = fx.subs(k, sol)
                            gy = (gy/k)*sol
                        else:
                            # 如果无解，则跳过当前循环
                            continue
                    
                    # 如果 odefac == hinv，则为反向的常微分方程 (ODE)
                    if odefac == hinv:
                        fx = fx.subs(x, y)
                        gy = gy.subs(y, x)
                    
                    # 计算 etaval = factor_terms(fx + gy)，并简化结果
                    etaval = factor_terms(fx + gy)
                    
                    # 如果 etaval 是乘积形式，则只保留包含 x 或 y 的因子
                    if etaval.is_Mul:
                        etaval = Mul(*[arg for arg in etaval.args if arg.has(x, y)])
                    
                    # 根据 odefac 的值，构建包含 eta 和 xi 的字典 inf
                    if odefac == hinv:
                        inf = {eta: etaval.subs(y, func), xi: S.Zero}
                    else:
                        inf = {xi: etaval.subs(y, func), eta: S.Zero}
                    
                    # 如果 comp 为假，返回包含 inf 的列表
                    if not comp:
                        return [inf]
                    else:
                        # 否则，将 inf 添加到 xieta 列表中
                        xieta.append(inf)
        
        # 如果 xieta 列表不为空，则返回 xieta
        if xieta:
            return xieta
    r"""
    This heuristic uses the following two assumptions on `\xi` and `\eta`

    .. math:: \eta = g(x), \xi = f(x)

    .. math:: \eta = f(y), \xi = g(y)

    For the first assumption,

    1. First `\frac{\frac{\partial h}{\partial y}}{\frac{\partial^{2} h}{
       \partial yy}}` is calculated. Let us say this value is A

    2. If this is constant, then `h` is matched to the form `A(x) + B(x)e^{
       \frac{y}{C}}` then, `\frac{e^{\int \frac{A(x)}{C} \,dx}}{B(x)}` gives `f(x)`
       and `A(x)*f(x)` gives `g(x)`

    3. Otherwise `\frac{\frac{\partial A}{\partial X}}{\frac{\partial A}{
       \partial Y}} = \gamma` is calculated. If

       a] `\gamma` is a function of `x` alone

       b] `\frac{\gamma\frac{\partial h}{\partial y} - \gamma'(x) - \frac{
       \partial h}{\partial x}}{h + \gamma} = G` is a function of `x` alone.
       then, `e^{\int G \,dx}` gives `f(x)` and `-\gamma*f(x)` gives `g(x)`

    The second assumption holds good if `\frac{dy}{dx} = h(x, y)` is rewritten as
    `\frac{dy}{dx} = \frac{1}{h(y, x)}` and the same properties of the first assumption
    satisfies. After obtaining `f(x)` and `g(x)`, the coordinates are again
    interchanged, to get `\xi` as `f(x^*)` and `\eta` as `g(y^*)`

    References
    ==========
    - E.S. Cheb-Terrab, A.D. Roche, Symmetries and First Order
      ODE Patterns, pp. 10 - pp. 12

    """

    h = match['h']  # Extract the function h(x, y) from the match dictionary
    hx = match['hx']  # Extract the partial derivative of h with respect to x
    hy = match['hy']  # Extract the partial derivative of h with respect to y
    func = match['func']  # Extract the function object from the match dictionary
    hinv = match['hinv']  # Extract the inverse function of h, h^{-1}(y, x)
    x = func.args[0]  # Extract the independent variable x from the function arguments
    y = match['y']  # Extract the variable y from the match dictionary
    xi = Function('xi')(x, func)  # Define xi as a function of x and func
    eta = Function('eta')(x, func)  # Define eta as a function of x and func

    factor = cancel(h.diff(y)/h.diff(y, 2))  # Calculate the factor A
    factorx = factor.diff(x)  # Calculate the partial derivative of factor with respect to x
    factory = factor.diff(y)  # Calculate the partial derivative of factor with respect to y

    if not factor.has(x) and not factor.has(y):
        A = Wild('A', exclude=[y])  # Define Wild pattern A excluding y
        B = Wild('B', exclude=[y])  # Define Wild pattern B excluding y
        C = Wild('C', exclude=[x, y])  # Define Wild pattern C excluding x and y
        match = h.match(A + B*exp(y/C))  # Match h to the pattern A + B*exp(y/C)
        try:
            tau = exp(-integrate(match[A]/match[C]), x)/match[B]  # Compute tau
        except NotImplementedError:
            pass  # Handle cases where integration is not implemented
        else:
            gx = match[A]*tau  # Compute gx
            return [{xi: tau, eta: gx}]  # Return xi and eta as tau and gx

    else:
        gamma = cancel(factorx/factory)  # Calculate gamma
        if not gamma.has(y):
            tauint = cancel((gamma*hy - gamma.diff(x) - hx)/(h + gamma))  # Calculate tauint
            if not tauint.has(y):
                try:
                    tau = exp(integrate(tauint, x))  # Compute tau
                except NotImplementedError:
                    pass  # Handle cases where integration is not implemented
                else:
                    gx = -tau*gamma  # Compute gx
                    return [{xi: tau, eta: gx}]  # Return xi and eta as tau and gx

    factor = cancel(hinv.diff(y)/hinv.diff(y, 2))  # Calculate factor for hinv
    factorx = factor.diff(x)  # Calculate the partial derivative of factor with respect to x
    factory = factor.diff(y)  # Calculate the partial derivative of factor with respect to y
    # 如果 x 和 y 都不在 factor 的变量中
    if not factor.has(x) and not factor.has(y):
        # 定义通配符 A，B，C，排除 y 变量
        A = Wild('A', exclude=[y])
        B = Wild('B', exclude=[y])
        C = Wild('C', exclude=[x, y])
        # 尝试匹配 h 函数的表达式 A + B*exp(y/C)
        match = h.match(A + B*exp(y/C))
        try:
            # 计算积分 integrate(match[A]/match[C]) 并取指数，求 tau
            tau = exp(-integrate(match[A]/match[C]), x)/match[B]
        except NotImplementedError:
            # 如果积分不可计算则跳过
            pass
        else:
            # 计算 gx = match[A]*tau，返回结果字典列表
            gx = match[A]*tau
            return [{eta: tau.subs(x, func), xi: gx.subs(x, func)}]

    else:
        # 计算 gamma = factorx / factory，并化简
        gamma = cancel(factorx/factory)
        # 如果 gamma 中不包含 y
        if not gamma.has(y):
            # 计算 tauint 表达式
            tauint = cancel((gamma*hinv.diff(y) - gamma.diff(x) - hinv.diff(x))/(
                hinv + gamma))
            # 如果 tauint 中不包含 y
            if not tauint.has(y):
                try:
                    # 计算 tauint 在 x 上的积分，并取指数，求 tau
                    tau = exp(integrate(tauint, x))
                except NotImplementedError:
                    # 如果积分不可计算则跳过
                    pass
                else:
                    # 计算 gx = -tau*gamma，返回结果字典列表
                    gx = -tau*gamma
                    return [{eta: tau.subs(x, func), xi: gx.subs(x, func)}]
def lie_heuristic_abaco2_unique_unknown(match, comp=False):
    r"""
    This heuristic assumes the presence of unknown functions or known functions
    with non-integer powers.

    1. A list of all functions and non-integer powers containing x and y
    2. Loop over each element `f` in the list, find `\frac{\frac{\partial f}{\partial x}}{
       \frac{\partial f}{\partial x}} = R`

       If it is separable in `x` and `y`, let `X` be the factors containing `x`. Then

       a] Check if `\xi = X` and `\eta = -\frac{X}{R}` satisfy the PDE. If yes, then return
          `\xi` and `\eta`
       b] Check if `\xi = \frac{-R}{X}` and `\eta = -\frac{1}{X}` satisfy the PDE.
           If yes, then return `\xi` and `\eta`

       If not, then check if

       a] :math:`\xi = -R,\eta = 1`

       b] :math:`\xi = 1, \eta = -\frac{1}{R}`

       are solutions.

    References
    ==========
    - E.S. Cheb-Terrab, A.D. Roche, Symmetries and First Order
      ODE Patterns, pp. 10 - pp. 12

    """

    # Extract components from the match dictionary
    h = match['h']       # Extract the 'h' component from the match
    hx = match['hx']     # Extract the 'hx' component from the match
    hy = match['hy']     # Extract the 'hy' component from the match
    func = match['func'] # Extract the 'func' component from the match
    x = func.args[0]     # Extract the first argument of 'func' as 'x'
    y = match['y']       # Extract the 'y' component from the match

    # Define functions xi and eta as functions of x and func
    xi = Function('xi')(x, func)   # Define xi as a function of x and func
    eta = Function('eta')(x, func) # Define eta as a function of x and func

    # Initialize an empty list to collect relevant functions
    funclist = []

    # Iterate over all atomic components of h that are powers
    for atom in h.atoms(Pow):
        base, exp = atom.as_base_exp()
        # Check if the base of the power contains both x and y
        if base.has(x) and base.has(y):
            # Check if the exponent is not an integer
            if not exp.is_Integer:
                # If conditions are met, add the atom to funclist
                funclist.append(atom)

    # Iterate over all atomic components of h that are AppliedUndef
    for function in h.atoms(AppliedUndef):
        syms = function.free_symbols
        # Check if the function involves both x and y
        if x in syms and y in syms:
            # If conditions are met, add the function to funclist
            funclist.append(function)

    # Iterate over each function `f` in funclist
    for f in funclist:
        # Compute the fraction of derivatives of f with respect to y and x
        frac = cancel(f.diff(y)/f.diff(x))
        # Attempt to separate variables x and y in the fraction
        sep = separatevars(frac, dict=True, symbols=[x, y])
        
        # If separation is successful and 'coeff' is present in sep
        if sep and sep['coeff']:
            # Extract potential solutions for xi and eta
            xitry1 = sep[x]
            etatry1 = -1/(sep[y]*sep['coeff'])
            # Formulate the PDE and check if it simplifies to zero
            pde1 = etatry1.diff(y)*h - xitry1.diff(x)*h - xitry1*hx - etatry1*hy
            if not simplify(pde1):
                # If PDE simplifies to zero, return the solution as a list
                return [{xi: xitry1, eta: etatry1.subs(y, func)}]
            # Try alternative solutions for xi and eta
            xitry2 = 1/etatry1
            etatry2 = 1/xitry1
            pde2 = etatry2.diff(x) - (xitry2.diff(y))*h**2 - xitry2*hx - etatry2*hy
            if not simplify(expand(pde2)):
                # If alternative PDE simplifies to zero, return the solution as a list
                return [{xi: xitry2.subs(y, func), eta: etatry2}]
        else:
            # If separation is not successful, compute alternative solutions
            etatry = -1/frac
            pde = etatry.diff(x) + etatry.diff(y)*h - hx - etatry*hy
            if not simplify(pde):
                # If PDE simplifies to zero, return the solution as a list
                return [{xi: S.One, eta: etatry.subs(y, func)}]
            xitry = -frac
            pde = -xitry.diff(x)*h -xitry.diff(y)*h**2 - xitry*hx -hy
            if not simplify(expand(pde)):
                # If PDE simplifies to zero, return the solution as a list
                return [{xi: xitry.subs(y, func), eta: S.One}]


def lie_heuristic_abaco2_unique_general(match, comp=False):
    r"""
    This heuristic finds if infinitesimals of the form `\eta = f(x)`, `\xi = g(y)`
    without making any assumptions on `h`.

    The complete sequence of steps is given in the paper mentioned below.

    References
    ==========
    # 从匹配对象中获取变量和函数
    hx = match['hx']  # 获取匹配对象中的 hx 变量
    hy = match['hy']  # 获取匹配对象中的 hy 变量
    func = match['func']  # 获取匹配对象中的 func 函数

    # 从函数中获取变量 x 和 y
    x = func.args[0]  # 获取函数 func 的第一个参数作为 x 变量
    y = match['y']  # 获取匹配对象中的 y 变量

    # 创建新的符号函数 xi 和 eta
    xi = Function('xi')(x, func)  # 创建一个以 x 和 func 作为参数的新符号函数 xi
    eta = Function('eta')(x, func)  # 创建一个以 x 和 func 作为参数的新符号函数 eta

    # 计算各种导数和表达式
    A = hx.diff(y)  # 计算 hx 对 y 的偏导数
    B = hy.diff(y) + hy**2  # 计算 hy 对 y 的偏导数加上 hy 的平方
    C = hx.diff(x) - hx**2  # 计算 hx 对 x 的偏导数减去 hx 的平方

    # 如果 A、B、C 有任何一个为假值，则返回空值
    if not (A and B and C):
        return

    # 计算更多的导数和表达式
    Ax = A.diff(x)  # 计算 A 对 x 的二阶导数
    Ay = A.diff(y)  # 计算 A 对 y 的一阶导数
    Axy = Ax.diff(y)  # 计算 Ax 对 y 的一阶导数
    Axx = Ax.diff(x)  # 计算 Ax 对 x 的二阶导数
    Ayy = Ay.diff(y)  # 计算 Ay 对 y 的二阶导数

    # 计算复杂的表达式 D 和检查其真假
    D = simplify(2*Axy + hx*Ay - Ax*hy + (hx*hy + 2*A)*A)*A - 3*Ax*Ay
    if not D:
        # 计算 E1 并检查其真假
        E1 = simplify(3*Ax**2 + ((hx**2 + 2*C)*A - 2*Axx)*A)
        if E1:
            # 计算 E2 并检查其真假
            E2 = simplify((2*Ayy + (2*B - hy**2)*A)*A - 3*Ay**2)
            if not E2:
                # 计算 E3 并检查其真假
                E3 = simplify(
                    E1*((28*Ax + 4*hx*A)*A**3 - E1*(hy*A + Ay)) - E1.diff(x)*8*A**4)
                if not E3:
                    # 计算 etaval 并检查其真假，处理积分计算可能的异常
                    etaval = cancel((4*A**3*(Ax - hx*A) + E1*(hy*A - Ay))/(S(2)*A*E1))
                    if x not in etaval:
                        try:
                            etaval = exp(integrate(etaval, y))
                        except NotImplementedError:
                            pass
                        else:
                            # 计算 xival 并检查其真假
                            xival = -4*A**3*etaval/E1
                            if y not in xival:
                                # 返回符号函数的字典列表
                                return [{xi: xival, eta: etaval.subs(y, func)}]
    else:
        # 计算 E1 并检查其真假
        E1 = simplify((2*Ayy + (2*B - hy**2)*A)*A - 3*Ay**2)
        if E1:
            # 计算 E2 并检查其真假
            E2 = simplify(
                4*A**3*D - D**2 + E1*((2*Axx - (hx**2 + 2*C)*A)*A - 3*Ax**2))
            if not E2:
                # 计算 E3 并检查其真假
                E3 = simplify(
                   -(A*D)*E1.diff(y) + ((E1.diff(x) - hy*D)*A + 3*Ay*D +
                    (A*hx - 3*Ax)*E1)*E1)
                if not E3:
                    # 计算 etaval 并检查其真假，处理积分计算可能的异常
                    etaval = cancel(((A*hx - Ax)*E1 - (Ay + A*hy)*D)/(S(2)*A*D))
                    if x not in etaval:
                        try:
                            etaval = exp(integrate(etaval, y))
                        except NotImplementedError:
                            pass
                        else:
                            # 计算 xival 并检查其真假
                            xival = -E1*etaval/D
                            if y not in xival:
                                # 返回符号函数的字典列表
                                return [{xi: xival, eta: etaval.subs(y, func)}]
# 定义函数 `lie_heuristic_linear`，用于线性启发式方法求解偏微分方程（PDE）
def lie_heuristic_linear(match, comp=False):
    """
    This heuristic assumes

    1. `\xi = ax + by + c` and
    2. `\eta = fx + gy + h`

    After substituting the following assumptions in the determining PDE, it
    reduces to

    .. math:: f + (g - a)h - bh^{2} - (ax + by + c)\frac{\partial h}{\partial x}
                 - (fx + gy + c)\frac{\partial h}{\partial y}

    Solving the reduced PDE obtained, using the method of characteristics, becomes
    impractical. The method followed is grouping similar terms and solving the system
    of linear equations obtained. The difference between the bivariate heuristic is that
    `h` need not be a rational function in this case.

    References
    ==========
    - E.S. Cheb-Terrab, A.D. Roche, Symmetries and First Order
      ODE Patterns, pp. 10 - pp. 12

    """
    # 从 match 参数中获取变量和函数
    h = match['h']       # 获取 h 变量
    hx = match['hx']     # 获取 h 对 x 的偏导数
    hy = match['hy']     # 获取 h 对 y 的偏导数
    func = match['func'] # 获取 func 函数
    x = func.args[0]     # 获取 func 的第一个参数 x
    y = match['y']       # 获取 y 变量
    xi = Function('xi')(x, func)   # 定义 xi 函数，依赖于 x 和 func
    eta = Function('eta')(x, func)  # 定义 eta 函数，依赖于 x 和 func

    # 初始化系数字典
    coeffdict = {}
    # 创建用于生成 Dummy 符号的生成器
    symbols = numbered_symbols("c", cls=Dummy)
    # 生成 6 个 Dummy 符号列表
    symlist = [next(symbols) for _ in islice(symbols, 6)]
    C0, C1, C2, C3, C4, C5 = symlist
    # 构建简化后的偏微分方程
    pde = C3 + (C4 - C0)*h - (C0*x + C1*y + C2)*hx - (C3*x + C4*y + C5)*hy - C1*h**2
    # 获取偏微分方程的分子和分母
    pde, denom = pde.as_numer_denom()
    # 对偏微分方程进行幂简化和展开
    pde = powsimp(expand(pde))
    # 如果偏微分方程是一个加法表达式
    if pde.is_Add:
        # 获取加法表达式的每一项
        terms = pde.args
        # 遍历每一项
        for term in terms:
            # 如果是乘法表达式
            if term.is_Mul:
                # 获取不包含 x 和 y 的项
                rem = Mul(*[m for m in term.args if not m.has(x, y)])
                # 获取包含 x 和 y 的部分
                xypart = term/rem
                # 将包含 x 和 y 的部分作为键，不包含 x 和 y 的部分作为值存入字典
                if xypart not in coeffdict:
                    coeffdict[xypart] = rem
                else:
                    coeffdict[xypart] += rem
            else:
                # 如果是单个项，直接存入字典
                if term not in coeffdict:
                    coeffdict[term] = S.One
                else:
                    coeffdict[term] += S.One

    # 解方程得到的系数列表
    sollist = coeffdict.values()
    # 解方程组得到解字典
    soldict = solve(sollist, symlist)
    # 如果有解
    if soldict:
        # 如果解是一个列表，取第一个解
        if isinstance(soldict, list):
            soldict = soldict[0]
        # 获取解的值列表
        subval = soldict.values()
        # 如果解的值列表中有非零值
        if any(t for t in subval):
            # 构造单位字典，用于替换解中的符号
            onedict = dict(zip(symlist, [1]*6))
            # 计算 xi 的值并替换解中的符号
            xival = C0*x + C1*func + C2
            xival = xival.subs(soldict)
            xival = xival.subs(onedict)
            # 计算 eta 的值并替换解中的符号
            etaval = C3*x + C4*func + C5
            etaval = etaval.subs(soldict)
            etaval = etaval.subs(onedict)
            # 返回包含 xi 和 eta 的字典列表
            return [{xi: xival, eta: etaval}]

# 定义函数 `_lie_group_remove`，用于 Lie 群求解 ODE 方法内部使用
def _lie_group_remove(coords):
    """
    This function is strictly meant for internal use by the Lie group ODE solving
    method. It replaces arbitrary functions returned by pdsolve as follows:

    1] If coords is an arbitrary function, then its argument is returned.
    2] An arbitrary function in an Add object is replaced by zero.
    3] An arbitrary function in a Mul object is replaced by one.
    4] If there is no arbitrary function coords is returned unchanged.

    Examples
    ========

    """
    # 导入 _lie_group_remove 函数，该函数来自 sympy.solvers.ode.lie_group 模块
    >>> from sympy.solvers.ode.lie_group import _lie_group_remove
    # 导入 sympy 的 Function 类
    >>> from sympy import Function
    # 导入 sympy 的符号 x 和 y
    >>> from sympy.abc import x, y
    # 创建一个名为 F 的符号函数
    >>> F = Function("F")
    # 创建一个等式表达式 eq
    >>> eq = x**2*y
    # 调用 _lie_group_remove 处理 eq，结果保持不变
    >>> _lie_group_remove(eq)
    x**2*y
    # 修改 eq 为 F(x**2*y)
    >>> eq = F(x**2*y)
    # 再次调用 _lie_group_remove 处理 eq，结果保持不变
    >>> _lie_group_remove(eq)
    x**2*y
    # 修改 eq 为 x*y**2 + F(x**3)
    >>> eq = x*y**2 + F(x**3)
    # 调用 _lie_group_remove 处理 eq，移除 F(x**3) 部分，返回 x*y**2
    >>> _lie_group_remove(eq)
    x*y**2
    # 修改 eq 为 (F(x**3) + y)*x**4
    >>> eq = (F(x**3) + y)*x**4
    # 调用 _lie_group_remove 处理 eq，移除 F(x**3) + y 部分，返回 x**4*y
    >>> _lie_group_remove(eq)
    x**4*y

    """
    # 根据 coords 的类型进行不同的处理
    if isinstance(coords, AppliedUndef):
        # 如果 coords 是 AppliedUndef 类型，则返回其第一个参数
        return coords.args[0]
    elif coords.is_Add:
        # 如果 coords 是 Add 类型，则将其子函数中的 AppliedUndef 替换为 0
        subfunc = coords.atoms(AppliedUndef)
        if subfunc:
            for func in subfunc:
                coords = coords.subs(func, 0)
        return coords
    elif coords.is_Pow:
        # 如果 coords 是 Pow 类型，则递归处理其 base 和 exponent
        base, expr = coords.as_base_exp()
        base = _lie_group_remove(base)
        expr = _lie_group_remove(expr)
        return base**expr
    elif coords.is_Mul:
        # 如果 coords 是 Mul 类型，则递归处理其每个参数
        mulargs = []
        coordargs = coords.args
        for arg in coordargs:
            if not isinstance(coords, AppliedUndef):
                mulargs.append(_lie_group_remove(arg))
        return Mul(*mulargs)
    # 如果 coords 不属于上述任何类型，则直接返回 coords
    return coords
```