# `D:\src\scipysrc\sympy\sympy\solvers\pde.py`

```
    """
    This module contains pdsolve() and different helper functions that it
    uses. It is heavily inspired by the ode module and hence the basic
    infrastructure remains the same.

    **Functions in this module**

        These are the user functions in this module:

        - pdsolve()     - Solves PDE's
        - classify_pde() - Classifies PDEs into possible hints for dsolve().
        - pde_separate() - Separate variables in partial differential equation either by
                           additive or multiplicative separation approach.

        These are the helper functions in this module:

        - pde_separate_add() - Helper function for searching additive separable solutions.
        - pde_separate_mul() - Helper function for searching multiplicative
                               separable solutions.

    **Currently implemented solver methods**

    The following methods are implemented for solving partial differential
    equations.  See the docstrings of the various pde_hint() functions for
    more information on each (run help(pde)):

      - 1st order linear homogeneous partial differential equations
        with constant coefficients.
      - 1st order linear general partial differential equations
        with constant coefficients.
      - 1st order linear partial differential equations with
        variable coefficients.

    """
    from functools import reduce  # 导入 reduce 函数，用于在函数式编程中进行累积计算

    from itertools import combinations_with_replacement  # 导入 combinations_with_replacement 函数，用于生成可重复组合
    from sympy.simplify import simplify  # 导入 simplify 函数，用于简化表达式（符号运算）
    from sympy.core import Add, S  # 导入 Add 和 S 符号，用于处理表达式中的加法
    from sympy.core.function import Function, expand, AppliedUndef, Subs  # 导入 Function、expand、AppliedUndef 和 Subs 函数/类
    from sympy.core.relational import Equality, Eq  # 导入 Equality 和 Eq 类，用于表示等式
    from sympy.core.symbol import Symbol, Wild, symbols  # 导入 Symbol、Wild 和 symbols 符号，用于处理符号变量
    from sympy.functions import exp  # 导入 exp 函数，用于指数运算
    from sympy.integrals.integrals import Integral, integrate  # 导入 Integral 和 integrate 函数，用于积分计算
    from sympy.utilities.iterables import has_dups, is_sequence  # 导入 has_dups 和 is_sequence 函数，用于检查序列性质
    from sympy.utilities.misc import filldedent  # 导入 filldedent 函数，用于缩减文本缩进

    from sympy.solvers.deutils import _preprocess, ode_order, _desolve  # 导入 _preprocess、ode_order 和 _desolve 函数，用于微分方程求解前处理
    from sympy.solvers.solvers import solve  # 导入 solve 函数，用于求解方程
    from sympy.simplify.radsimp import collect  # 导入 collect 函数，用于收集表达式中的同类项

    import operator  # 导入 operator 模块，用于操作符函数

    allhints = (
        "1st_linear_constant_coeff_homogeneous",
        "1st_linear_constant_coeff",
        "1st_linear_constant_coeff_Integral",
        "1st_linear_variable_coeff"
        )

    def pdsolve(eq, func=None, hint='default', dict=False, solvefun=None, **kwargs):
        """
        Solves any (supported) kind of partial differential equation.

        **Usage**

            pdsolve(eq, f(x,y), hint) -> Solve partial differential equation
            eq for function f(x,y), using method hint.

        """
    # Details section explaining parameters and their usage for solving partial differential equations (PDEs).
    # `eq` can be either an Equality object or an expression equated to 0, representing the PDE.
    # `f(x,y)` is a function of two variables whose derivatives constitute the PDE.
    # The function `f(x,y)` is automatically detected in many cases; an error is raised if detection fails.
    # `hint` specifies the solving method for `pdsolve()`. Use `classify_pde(eq, f(x,y))` to get available hints.
    # The default hint, 'default', uses the first hint returned by classify_pde().
    # See Hints below for additional options that can be used with the `hint` parameter.
    # `solvefun` denotes the convention for naming arbitrary functions returned by the PDE solver. Default is F if unset.

    # Hints section describing different meta-hints and solving methods available for pdsolve().
    # "default": Uses the first hint returned by classify_pde() as the solving method.
    # "all": Applies all relevant classification hints and returns a dictionary of hint:solution pairs.
    #        Special keys include 'order' (PDE order) and 'default' (solution from the first hint in classify_pde()).
    #        If a hint raises NotImplementedError, its key in the dictionary will hold the exception object.
    # "all_Integral": Similar to "all", but prefers "_Integral" hints over others, avoiding expensive integrations.
    # See classify_pde() docstring for detailed information on hints, and pde docstring for a list of supported hints.
    # 如果没有指定求解函数 solvefun，则默认使用 Function('F')
    if not solvefun:
        solvefun = Function('F')

    # 调用 _desolve 函数进行偏微分方程求解，获取求解提示信息
    # func 参数是用来表示函数的，hint 是指定的求解提示，simplify=True 表示简化结果
    # type='pde' 表示这是一个偏微分方程
    hints = _desolve(eq, func=func, hint=hint, simplify=True,
                     type='pde', **kwargs)

    # 从提示信息中弹出 'eq' 和 'all' 键的值，如果没有则为 False
    eq = hints.pop('eq', False)
    all_ = hints.pop('all', False)

    # 如果 all_ 为 True，则尝试多个提示来解决偏微分方程
    if all_:
        # 初始化一个空字典和一个空字典来保存无法解析的提示
        pdedict = {}
        failed_hints = {}

        # 调用 classify_pde 函数对偏微分方程进行分类并获取提示信息
        gethints = classify_pde(eq, dict=True)

        # 更新 pdedict 字典，包括 'order' 和 'default' 的信息
        pdedict.update({'order': gethints['order'],
                        'default': gethints['default']})

        # 遍历所有的提示，尝试用每个提示来简化方程并保存结果到 pdedict 中
        for hint in hints:
            try:
                rv = _helper_simplify(eq, hint, hints[hint]['func'],
                                      hints[hint]['order'], hints[hint][hint], solvefun)
            except NotImplementedError as detail:
                failed_hints[hint] = detail
            else:
                pdedict[hint] = rv

        # 更新 pdedict 字典，添加无法解析的提示信息到 failed_hints 中
        pdedict.update(failed_hints)

        # 返回包含所有提示及其解析结果的字典 pdedict
        return pdedict

    # 如果 all_ 不为 True，则直接用单个提示来简化方程并返回结果
    else:
        # 调用 _helper_simplify 函数简化方程，返回简化后的结果
        return _helper_simplify(eq, hints['hint'], hints['func'],
                                hints['order'], hints[hints['hint']], solvefun)
# 定义一个辅助函数，用于简化偏微分方程求解过程，减少多次调用 _desolve 的计算量
def _helper_simplify(eq, hint, func, order, match, solvefun):
    """Helper function of pdsolve that calls the respective
    pde functions to solve for the partial differential
    equations. This minimizes the computation in
    calling _desolve multiple times.
    """

    # 根据提示信息选择相应的偏微分方程求解函数
    if hint.endswith("_Integral"):
        solvefunc = globals()["pde_" + hint[:-len("_Integral")]]
    else:
        solvefunc = globals()["pde_" + hint]

    # 调用求解函数解决方程，并处理其中的积分
    return _handle_Integral(solvefunc(eq, func, order,
                                      match, solvefun), func, order, hint)


# 处理带有积分的解，将其转换为实际的解
def _handle_Integral(expr, func, order, hint):
    r"""
    Converts a solution with integrals in it into an actual solution.

    Simplifies the integral mainly using doit()
    """
    # 如果提示信息以 "_Integral" 结尾，则直接返回表达式
    if hint.endswith("_Integral"):
        return expr
    # 对于特定的线性常系数方程式，简化其积分
    elif hint == "1st_linear_constant_coeff":
        return simplify(expr.doit())
    else:
        return expr


# 对偏微分方程进行分类，返回一个元组，包含可能的求解分类
def classify_pde(eq, func=None, dict=False, *, prep=True, **kwargs):
    """
    Returns a tuple of possible pdsolve() classifications for a PDE.

    The tuple is ordered so that first item is the classification that
    pdsolve() uses to solve the PDE by default.  In general,
    classifications near the beginning of the list will produce
    better solutions faster than those near the end, though there are
    always exceptions.  To make pdsolve use a different classification,
    use pdsolve(PDE, func, hint=<classification>).  See also the pdsolve()
    docstring for different meta-hints you can use.

    If ``dict`` is true, classify_pde() will return a dictionary of
    hint:match expression terms. This is intended for internal use by
    pdsolve().  Note that because dictionaries are ordered arbitrarily,
    this will most likely not be in the same order as the tuple.

    You can get help on different hints by doing help(pde.pde_hintname),
    where hintname is the name of the hint without "_Integral".

    See sympy.pde.allhints or the sympy.pde docstring for a list of all
    supported hints that can be returned from classify_pde.


    Examples
    ========

    >>> from sympy.solvers.pde import classify_pde
    >>> from sympy import Function, Eq
    >>> from sympy.abc import x, y
    >>> f = Function('f')
    >>> u = f(x, y)
    >>> ux = u.diff(x)
    >>> uy = u.diff(y)
    >>> eq = Eq(1 + (2*(ux/u)) + (3*(uy/u)), 0)
    >>> classify_pde(eq)
    ('1st_linear_constant_coeff_homogeneous',)
    """

    # 如果指定了 func 参数且其参数数目不为 2，则抛出 NotImplementedError
    if func and len(func.args) != 2:
        raise NotImplementedError("Right now only partial "
                                  "differential equations of two variables are supported")

    # 如果 prep 为 True 或 func 为 None，则进行预处理
    if prep or func is None:
        prep, func_ = _preprocess(eq, func)
        if func is None:
            func = func_

    # 如果 eq 是 Equality 类型且右侧不为 0，则对等式进行调整使其等于 0
    if isinstance(eq, Equality):
        if eq.rhs != 0:
            return classify_pde(eq.lhs - eq.rhs, func)
        eq = eq.lhs

    # 提取函数及其变量
    f = func.func
    x = func.args[0]
    y = func.args[1]
    # 计算函数 f 的偏导数
    fx = f(x, y).diff(x)
    fy = f(x, y).diff(y)
    # 使用 ode_order 函数计算给定方程 eq 关于函数 f(x,y) 的阶数
    order = ode_order(eq, f(x,y))

    # 初始化 matching_hints 字典，包含 'order': order 项
    matching_hints = {'order': order}

    # 如果阶数 order 为假值（如None、0、空字符串等）
    if not order:
        # 如果字典 dict 为真值（非空）
        if dict:
            # 设置 matching_hints["default"] 为 None，并返回 matching_hints
            matching_hints["default"] = None
            return matching_hints
        # 如果 dict 为假值（空），返回空元组 ()
        return ()

    # 对方程 eq 进行展开操作
    eq = expand(eq)

    # 创建用于模式匹配的 Wild 类对象，排除特定变量和函数
    a = Wild('a', exclude = [f(x,y)])
    b = Wild('b', exclude = [f(x,y), fx, fy, x, y])
    c = Wild('c', exclude = [f(x,y), fx, fy, x, y])
    d = Wild('d', exclude = [f(x,y), fx, fy, x, y])
    e = Wild('e', exclude = [f(x,y), fx, fy])
    n = Wild('n', exclude = [x, y])

    # 如果方程 eq 是加法形式
    reduced_eq = eq
    if eq.is_Add:
        power = None
        # 遍历方程中最高阶导数的所有组合
        for i in set(combinations_with_replacement((x,y), order)):
            # 提取系数
            coeff = eq.coeff(f(x,y).diff(*i))
            if coeff == 1:
                continue
            # 匹配系数为 a*f(x,y)**n 的模式
            match = coeff.match(a*f(x,y)**n)
            if match and match[a]:
                # 更新最小幂次 power
                if power is None or match[n] < power:
                    power = match[n]
        # 如果找到最小幂次 power
        if power:
            # 计算分母 den，并进行化简
            den = f(x,y)**power
            reduced_eq = Add(*[arg/den for arg in eq.args])

    # 如果阶数为 1
    if order == 1:
        # 对化简后的方程 reduced_eq 进行整理，收集关于 f(x,y) 的系数
        reduced_eq = collect(reduced_eq, f(x, y))
        # 匹配方程的模式 b*fx + c*fy + d*f(x,y) + e
        r = reduced_eq.match(b*fx + c*fy + d*f(x,y) + e)
        if r:
            # 如果 e 为假值（0）
            if not r[e]:
                ## 线性一阶齐次偏微分方程，且系数是常数
                r.update({'b': b, 'c': c, 'd': d})
                matching_hints["1st_linear_constant_coeff_homogeneous"] = r
            # 如果 e 不为 0
            elif r[b]**2 + r[c]**2 != 0:
                ## 线性一阶常系数偏微分方程
                r.update({'b': b, 'c': c, 'd': d, 'e': e})
                matching_hints["1st_linear_constant_coeff"] = r
                matching_hints["1st_linear_constant_coeff_Integral"] = r

        else:
            # 重新创建 Wild 类对象，用于匹配模式 b*fx + c*fy + d*f(x,y) + e
            b = Wild('b', exclude=[f(x, y), fx, fy])
            c = Wild('c', exclude=[f(x, y), fx, fy])
            d = Wild('d', exclude=[f(x, y), fx, fy])
            r = reduced_eq.match(b*fx + c*fy + d*f(x,y) + e)
            if r:
                r.update({'b': b, 'c': c, 'd': d, 'e': e})
                matching_hints["1st_linear_variable_coeff"] = r

    # 按照 allhints 中的顺序排列键，形成返回的元组
    rettuple = tuple(i for i in allhints if i in matching_hints)
    # 如果给定的 dict 不为空
    if dict:
        # 字典的顺序是任意的，因此需要注意哪个提示会首先传递给 pdsolve()。
        # 在 Python 3 中可以使用 OrderedDict 来保持顺序。
        
        # 将默认值设置为 None，用于匹配提示
        matching_hints["default"] = None
        
        # 将有序提示存储到 matching_hints 中
        matching_hints["ordered_hints"] = rettuple
        
        # 遍历所有提示
        for i in allhints:
            # 如果提示 i 在 matching_hints 中
            if i in matching_hints:
                # 将默认值设置为当前的提示 i
                matching_hints["default"] = i
                # 找到第一个匹配后跳出循环
                break
        
        # 返回匹配的提示字典
        return matching_hints
    
    # 如果 dict 为空，则返回原始的 rettuple
    return rettuple
# 将给定的偏微分方程转换为等式形式
if not isinstance(pde, Equality):
    pde = Eq(pde, 0)

# 如果未提供函数参数，尝试从 pde.lhs 中预处理函数
if func is None:
    try:
        _, func = _preprocess(pde.lhs)
    except ValueError:
        # 如果解 sol 是一个序列，检查每个元素中的未定义函数
        funcs = [s.atoms(AppliedUndef) for s in (sol if is_sequence(sol, set) else [sol])]
        funcs = set().union(funcs)
        # 如果检测到不止一个函数，抛出 ValueError
        if len(funcs) != 1:
            raise ValueError(
                'must pass func arg to checkpdesol for this case.')
        func = funcs.pop()

# 如果给定的解 sol 是一个集合，返回一个集合，其中每个元素都经过检查
if is_sequence(sol, set):
    return type(sol)([checkpdesol(
        pde, i, func=func,
        solve_for_func=solve_for_func) for i in sol])

# 将解 sol 转换为等式形式，如果已经是等式且 rhs 等于 func，则反转等式
if not isinstance(sol, Equality):
    sol = Eq(func, sol)
elif sol.rhs == func:
    sol = sol.reversed

# 尝试解出函数，并且确保 rhs 中不包含 func
solved = sol.lhs == func and not sol.rhs.has(func)
    # 如果需要求解函数，并且尚未求解过
    if solve_for_func and not solved:
        # 对函数进行求解
        solved = solve(sol, func)
        # 如果成功求解
        if solved:
            # 如果只有一个解
            if len(solved) == 1:
                # 检查偏微分方程的解是否满足条件，不再尝试求解函数
                return checkpdesol(pde, Eq(func, solved[0]),
                    func=func, solve_for_func=False)
            else:
                # 检查偏微分方程的解是否满足条件，不再尝试求解函数
                return checkpdesol(pde, [Eq(func, t) for t in solved],
                    func=func, solve_for_func=False)

    # 尝试将解直接代入偏微分方程并简化
    if sol.lhs == func:
        # 将偏微分方程左侧减去右侧
        pde = pde.lhs - pde.rhs
        # 对解进行直接代入并进行简化
        s = simplify(pde.subs(func, sol.rhs).doit())
        # 返回是否简化结果为零及其简化结果
        return s is S.Zero, s

    # 抛出未实现错误，提示无法测试解是否为偏微分方程的解
    raise NotImplementedError(filldedent('''
        Unable to test if %s is a solution to %s.''' % (sol, pde)))
def pde_1st_linear_constant_coeff_homogeneous(eq, func, order, match, solvefun):
    r"""
    Solves a first order linear homogeneous
    partial differential equation with constant coefficients.

    The general form of this partial differential equation is

    .. math:: a \frac{\partial f(x,y)}{\partial x}
              + b \frac{\partial f(x,y)}{\partial y} + c f(x,y) = 0

    where `a`, `b` and `c` are constants.

    The general solution is of the form:

    .. math::
        f(x, y) = F(- a y + b x ) e^{- \frac{c (a x + b y)}{a^2 + b^2}}

    and can be found in SymPy with ``pdsolve``::

        >>> from sympy.solvers import pdsolve
        >>> from sympy.abc import x, y, a, b, c
        >>> from sympy import Function, pprint
        >>> f = Function('f')
        >>> u = f(x,y)
        >>> ux = u.diff(x)
        >>> uy = u.diff(y)
        >>> genform = a*ux + b*uy + c*u
        >>> pprint(genform)
          d               d
        a*--(f(x, y)) + b*--(f(x, y)) + c*f(x, y)
          dx              dy

        >>> pprint(pdsolve(genform))
                                 -c*(a*x + b*y)
                                 ---------------
                                      2    2
                                     a  + b
        f(x, y) = F(-a*y + b*x)*e

    Examples
    ========

    >>> from sympy import pdsolve
    >>> from sympy import Function, pprint
    >>> from sympy.abc import x,y
    >>> f = Function('f')
    >>> pdsolve(f(x,y) + f(x,y).diff(x) + f(x,y).diff(y))
    Eq(f(x, y), F(x - y)*exp(-x/2 - y/2))
    >>> pprint(pdsolve(f(x,y) + f(x,y).diff(x) + f(x,y).diff(y)))
                          x   y
                        - - - -
                          2   2
    f(x, y) = F(x - y)*e

    References
    ==========

    - Viktor Grigoryan, "Partial Differential Equations"
      Math 124A - Fall 2010, pp.7

    """
    # 解决具有常数系数的一阶线性齐次偏微分方程
    # TODO : For now homogeneous first order linear PDE's having
    # two variables are implemented. Once there is support for
    # solving systems of ODE's, this can be extended to n variables.

    # 获取函数对象和变量
    f = func.func
    x = func.args[0]
    y = func.args[1]
    
    # 从匹配结果中获取常数系数
    b = match[match['b']]
    c = match[match['c']]
    d = match[match['d']]
    
    # 返回解的表达式
    return Eq(f(x,y), exp(-S(d)/(b**2 + c**2)*(b*x + c*y))*solvefun(c*x - b*y))
    .. math::
        f(x, y) = \left. \left[F(\eta) + \frac{1}{a^2 + b^2}
        \int\limits^{a x + b y} G\left(\frac{a \xi + b \eta}{a^2 + b^2},
        \frac{- a \eta + b \xi}{a^2 + b^2} \right)
        e^{\frac{c \xi}{a^2 + b^2}}\, d\xi\right]
        e^{- \frac{c \xi}{a^2 + b^2}}
        \right|_{\substack{\eta=- a y + b x\\ \xi=a x + b y }}\, ,

    where `F(\eta)` is an arbitrary single-valued function. The solution
    can be found in SymPy with ``pdsolve``::

        >>> from sympy.solvers import pdsolve
        >>> from sympy.abc import x, y, a, b, c
        >>> from sympy import Function, pprint
        >>> f = Function('f')  # 定义符号函数 f(x, y)
        >>> G = Function('G')  # 定义符号函数 G(x, y)
        >>> u = f(x, y)  # 定义未知函数 u = f(x, y)
        >>> ux = u.diff(x)  # 求 u 对 x 的偏导数
        >>> uy = u.diff(y)  # 求 u 对 y 的偏导数
        >>> genform = a*ux + b*uy + c*u - G(x,y)  # 构造一般形式的偏微分方程
        >>> pprint(genform)  # 使用美观打印输出一般形式的偏微分方程

        d               d
        a*--(f(x, y)) + b*--(f(x, y)) + c*f(x, y) - G(x, y)
        dx              dy

        >>> pprint(pdsolve(genform, hint='1st_linear_constant_coeff_Integral'))
                  //          a*x + b*y                                             \         \|
                  ||              /                                                 |         ||
                  ||             |                                                  |         ||
                  ||             |                                      c*xi        |         ||
                  ||             |                                     -------      |         ||
                  ||             |                                      2    2      |         ||
                  ||             |      /a*xi + b*eta  -a*eta + b*xi\  a  + b       |         ||
                  ||             |     G|------------, -------------|*e        d(xi)|         ||
                  ||             |      |   2    2         2    2   |               |         ||
                  ||             |      \  a  + b         a  + b    /               |  -c*xi  ||
                  ||             |                                                  |  -------||
                  ||            /                                                   |   2    2||
                  ||                                                                |  a  + b ||
        f(x, y) = ||F(eta) + -------------------------------------------------------|*e       ||
                  ||                                  2    2                        |         ||
                  \\                                 a  + b                         /         /|eta=-a*y + b*x, xi=a*x + b*y

    Examples
    ========

    >>> from sympy.solvers.pde import pdsolve  # 导入偏微分方程求解器
    >>> from sympy import Function, pprint, exp  # 导入符号函数、美观打印和指数函数
    >>> from sympy.abc import x,y  # 导入符号 x 和 y
    >>> f = Function('f')  # 定义符号函数 f(x, y)
    >>> eq = -2*f(x,y).diff(x) + 4*f(x,y).diff(y) + 5*f(x,y) - exp(x + 3*y)  # 定义偏微分方程
    >>> pdsolve(eq)  # 解偏微分方程

    Eq(f(x, y), (F(4*x + 2*y)*exp(x/2) + exp(x + 4*y)/15)*exp(-y))

    References
    ==========
    # 引用的文献信息，包括作者和文章标题
    # Viktor Grigoryan, "Partial Differential Equations"
    # Math 124A - Fall 2010, pp.7
    
    # TODO : 目前仅实现了具有两个变量的齐次一阶线性偏微分方程。一旦有解决ODE系统的支持，可以扩展到n个变量。
    # 定义符号变量 xi 和 eta
    xi, eta = symbols("xi eta")
    
    # 从 func 对象中获取函数和参数
    f = func.func
    x = func.args[0]
    y = func.args[1]
    
    # 从 match 字典中获取参数
    b = match[match['b']]
    c = match[match['c']]
    d = match[match['d']]
    e = -match[match['e']]
    
    # 计算指数项
    expterm = exp(-S(d)/(b**2 + c**2)*xi)
    
    # 调用 solvefun 函数处理 eta
    functerm = solvefun(eta)
    
    # 解方程组 (b*x + c*y - xi, c*x - b*y - eta)，求解 x 和 y
    solvedict = solve((b*x + c*y - xi, c*x - b*y - eta), x, y)
    
    # 构造积分表达式，积分变量是 xi，但不应用 doit()，这应在 _handle_Integral 中完成。
    genterm = (1/S(b**2 + c**2))*Integral(
        (1/expterm*e).subs(solvedict), (xi, b*x + c*y))
    
    # 返回偏微分方程的等式形式
    return Eq(f(x,y), Subs(expterm*(functerm + genterm),
        (eta, xi), (c*x - b*y, b*x + c*y)))
    # 定义函数，用于解决具有可变系数的一阶线性偏微分方程
    r"""
    解决具有可变系数的一阶线性偏微分方程。此偏微分方程的一般形式为：

    .. math:: a(x, y) \frac{\partial f(x, y)}{\partial x}
                + b(x, y) \frac{\partial f(x, y)}{\partial y}
                + c(x, y) f(x, y) = G(x, y)

    其中 `a(x, y)`, `b(x, y)`, `c(x, y)` 和 `G(x, y)` 是 `x` 和 `y` 的任意函数。
    通过以下变换将此偏微分方程转化为常微分方程：

    1. `\xi` 作为 `x`

    2. `\eta` 作为微分方程 `\frac{dy}{dx} = -\frac{b}{a}` 的解的常数部分

    进行上述替换后，得到线性常微分方程：

    .. math:: a(\xi, \eta)\frac{du}{d\xi} + c(\xi, \eta)u - G(\xi, \eta) = 0

    可以使用 `dsolve` 函数解决此方程。

    >>> from sympy.abc import x, y
    >>> from sympy import Function, pprint
    >>> a, b, c, G, f= [Function(i) for i in ['a', 'b', 'c', 'G', 'f']]
    >>> u = f(x,y)
    >>> ux = u.diff(x)
    >>> uy = u.diff(y)
    >>> genform = a(x, y)*u + b(x, y)*ux + c(x, y)*uy - G(x,y)
    >>> pprint(genform)
                                         d                     d
    -G(x, y) + a(x, y)*f(x, y) + b(x, y)*--(f(x, y)) + c(x, y)*--(f(x, y))
                                         dx                    dy

    Examples
    ========

    >>> from sympy.solvers.pde import pdsolve
    >>> from sympy import Function, pprint
    >>> from sympy.abc import x,y
    >>> f = Function('f')
    >>> eq =  x*(u.diff(x)) - y*(u.diff(y)) + y**2*u - y**2
    >>> pdsolve(eq)
    Eq(f(x, y), F(x*y)*exp(y**2/2) + 1)

    References
    ==========

    - Viktor Grigoryan, "Partial Differential Equations"
      Math 124A - Fall 2010, pp.7
    """
    from sympy.solvers.ode import dsolve

    # 定义符号 eta
    eta = symbols("eta")
    # 获取函数的符号和参数
    f = func.func
    x = func.args[0]
    y = func.args[1]
    # 从匹配中提取系数
    b = match[match['b']]
    c = match[match['c']]
    d = match[match['d']]
    e = -match[match['e']]

    # 如果 d 不存在
    if not d:
         # 处理类似 b*ux = e 或者 c*uy = e 的情况
         if not (b and c):
            # 如果 c 存在，尝试积分解方程
            if c:
                try:
                    tsol = integrate(e/c, y)
                except NotImplementedError:
                    raise NotImplementedError("Unable to find a solution"
                        " due to inability of integrate")
                else:
                    return Eq(f(x,y), solvefun(x) + tsol)
            # 如果 b 存在，尝试积分解方程
            if b:
                try:
                    tsol = integrate(e/b, x)
                except NotImplementedError:
                    raise NotImplementedError("Unable to find a solution"
                        " due to inability of integrate")
                else:
                    return Eq(f(x,y), solvefun(y) + tsol)
    # 如果 c 为假（即 c 为 0），处理 c 为 0 的情况，使用简化方法。
    # 偏微分方程简化为 b*(u.diff(x)) + d*u = e，这是关于 x 的线性常微分方程
    plode = f(x).diff(x)*b + d*f(x) - e
    sol = dsolve(plode, f(x))
    # 解中的自由符号减去方程 plode 中的自由符号和 {x, y}
    syms = sol.free_symbols - plode.free_symbols - {x, y}
    # 对右侧表达式进行简化，处理变量系数，使用 solvefun 函数，其中 y 是解析的变量
    rhs = _simplify_variable_coeff(sol.rhs, syms, solvefun, y)
    # 返回方程 f(x, y) = rhs
    return Eq(f(x, y), rhs)

    # 如果 b 为假（即 b 为 0），处理 b 为 0 的情况，使用简化方法。
    # 偏微分方程简化为 c*(u.diff(y)) + d*u = e，这是关于 y 的线性常微分方程
    plode = f(y).diff(y)*c + d*f(y) - e
    sol = dsolve(plode, f(y))
    # 解中的自由符号减去方程 plode 中的自由符号和 {x, y}
    syms = sol.free_symbols - plode.free_symbols - {x, y}
    # 对右侧表达式进行简化，处理变量系数，使用 solvefun 函数，其中 x 是解析的变量
    rhs = _simplify_variable_coeff(sol.rhs, syms, solvefun, x)
    # 返回方程 f(x, y) = rhs
    return Eq(f(x, y), rhs)

    # 创建一个名为 dummy 的函数对象
    dummy = Function('d')
    # 计算 h = c/b，并将 y 替换为 dummy(x)
    h = (c/b).subs(y, dummy(x))
    # 解微分方程 dummy(x).diff(x) - h，得到 sol
    sol = dsolve(dummy(x).diff(x) - h, dummy(x))
    # 如果 sol 是列表，则取第一个元素
    if isinstance(sol, list):
        sol = sol[0]
    # 解中的自由符号减去 h 的自由符号和 {x, y}
    solsym = sol.free_symbols - h.free_symbols - {x, y}
    # 如果解中只有一个自由符号
    if len(solsym) == 1:
        solsym = solsym.pop()
        # 解出 solsym 对应的常数 etat，其中用 dummy(x) 替换 y
        etat = (solve(sol, solsym)[0]).subs(dummy(x), y)
        # 解方程 eta - etat = 0，得到 ysub
        ysub = solve(eta - etat, y)[0]
        # 计算新的偏微分方程 deq = b*(f(x).diff(x)) + d*f(x) - e，并用 ysub 替换 y
        deq = (b*(f(x).diff(x)) + d*f(x) - e).subs(y, ysub)
        # 求解 deq 的线性常微分方程的右侧表达式
        final = (dsolve(deq, f(x), hint='1st_linear')).rhs
        # 如果 final 是列表，则取第一个元素
        if isinstance(final, list):
            final = final[0]
        # 解中的自由符号减去 deq 的自由符号和 {x, y}
        finsyms = final.free_symbols - deq.free_symbols - {x, y}
        # 对最终表达式进行简化，处理变量系数，使用 solvefun 函数，其中 etat 是解析的变量
        rhs = _simplify_variable_coeff(final, finsyms, solvefun, etat)
        # 返回方程 f(x, y) = rhs
        return Eq(f(x, y), rhs)

    else:
        # 抛出未实现的错误，因为无法解析这个偏微分方程
        raise NotImplementedError("Cannot solve the partial differential equation due"
            " to inability of constantsimp")
def _simplify_variable_coeff(sol, syms, func, funcarg):
    r"""
    Helper function to replace constants by functions in 1st_linear_variable_coeff
    """
    # 创建符号 eta
    eta = Symbol("eta")
    # 如果变量数为1，替换解 sol 中的一个符号为 func(funcarg)
    if len(syms) == 1:
        sym = syms.pop()
        final = sol.subs(sym, func(funcarg))
    else:
        # 否则，对于每个符号 sym，替换解 sol 中的符号为 func(funcarg)
        for sym in syms:
            final = sol.subs(sym, func(funcarg))

    # 简化最终的表达式，并将 eta 替换为 funcarg
    return simplify(final.subs(eta, funcarg))


def pde_separate(eq, fun, sep, strategy='mul'):
    """Separate variables in partial differential equation either by additive
    or multiplicative separation approach. It tries to rewrite an equation so
    that one of the specified variables occurs on a different side of the
    equation than the others.

    :param eq: Partial differential equation

    :param fun: Original function F(x, y, z)

    :param sep: List of separated functions [X(x), u(y, z)]

    :param strategy: Separation strategy. You can choose between additive
        separation ('add') and multiplicative separation ('mul') which is
        default.

    Examples
    ========

    >>> from sympy import E, Eq, Function, pde_separate, Derivative as D
    >>> from sympy.abc import x, t
    >>> u, X, T = map(Function, 'uXT')

    >>> eq = Eq(D(u(x, t), x), E**(u(x, t))*D(u(x, t), t))
    >>> pde_separate(eq, u(x, t), [X(x), T(t)], strategy='add')
    [exp(-X(x))*Derivative(X(x), x), exp(T(t))*Derivative(T(t), t)]

    >>> eq = Eq(D(u(x, t), x, 2), D(u(x, t), t, 2))
    >>> pde_separate(eq, u(x, t), [X(x), T(t)], strategy='mul')
    [Derivative(X(x), (x, 2))/X(x), Derivative(T(t), (t, 2))/T(t)]

    See Also
    ========
    pde_separate_add, pde_separate_mul
    """

    # 根据策略确定是否使用加法分离
    do_add = False
    if strategy == 'add':
        do_add = True
    elif strategy == 'mul':
        do_add = False
    else:
        raise ValueError('Unknown strategy: %s' % strategy)

    # 如果方程不是 Equality 类型，则转换为 Equality 类型
    if isinstance(eq, Equality):
        if eq.rhs != 0:
            return pde_separate(Eq(eq.lhs - eq.rhs, 0), fun, sep, strategy)
    else:
        return pde_separate(Eq(eq, 0), fun, sep, strategy)

    # 确保等式右侧为0
    if eq.rhs != 0:
        raise ValueError("Value should be 0")

    # 处理参数
    orig_args = list(fun.args)
    subs_args = [arg for s in sep for arg in s.args]

    # 根据策略组合分离后的函数
    if do_add:
        functions = reduce(operator.add, sep)
    else:
        functions = reduce(operator.mul, sep)

    # 检查变量数是否匹配
    if len(subs_args) != len(orig_args):
        raise ValueError("Variable counts do not match")
    # 检查是否存在重复参数
    if has_dups(subs_args):
        raise ValueError("Duplicate substitution arguments detected")
    # 检查变量是否匹配
    if set(orig_args) != set(subs_args):
        raise ValueError("Arguments do not match")

    # 用分离后的函数替换原始函数，并进行计算
    result = eq.lhs.subs(fun, functions).doit()

    # 在进行乘法分离时，对结果进行除法处理
    # 如果条件 do_add 不为真（即为假），执行以下代码块
    if not do_add:
        # 初始化变量 eq 为 0
        eq = 0
        # 遍历 result.args 中的每个元素 i
        for i in result.args:
            # 将 i 除以 functions 后加到 eq 中
            eq += i/functions
        # 将 result 更新为 eq 的值
        result = eq

    # 将 subs_args 中的第一个元素赋值给 svar
    svar = subs_args[0]
    # 将 subs_args 中除第一个元素外的所有元素赋值给 dvar
    dvar = subs_args[1:]
    # 调用 _separate 函数，并返回其结果
    return _separate(result, svar, dvar)
# 将偏微分方程表达式分解为基于变量依赖性的两部分

# 第一次遍历
# 提取依赖于可分离变量的导数...
terms = set()
for term in eq.args:
    if term.is_Mul:  # 如果是乘积
        for i in term.args:
            if i.is_Derivative and not i.has(*others):  # 如果是导数且不依赖于其他变量
                terms.add(term)
                continue
    elif term.is_Derivative and not term.has(*others):  # 如果是导数且不依赖于其他变量
        terms.add(term)
# 找到需要除以的因子
div = set()
for term in terms:
    ext, sep = term.expand().as_independent(dep)
    # 失败？
    if sep.has(*others):  # 如果分离项依赖于其他变量，则返回空
        return None
    div.add(ext)
# FIXME: 找到所有除数的最小公倍数并进行除法，而不是当前的方法 :(
# https://github.com/sympy/sympy/issues/4597
if len(div) > 0:
    # 需要双重求和，否则某些测试将失败
    eq = Add(*[simplify(Add(*[term/i for i in div])) for term in eq.args])

# 第二次遍历 - 分离导数
div = set()
lhs = rhs = 0
    # 对于方程中的每个项进行遍历
    for term in eq.args:
        # 检查该项是否不包含独立变量...
        if not term.has(*others):
            # 如果不包含，将其加入左侧表达式
            lhs += term
            continue
        # ...否则，尝试将其分离
        temp, sep = term.expand().as_independent(dep)
        # 分离失败？
        if sep.has(*others):
            # 如果分离后的独立部分还包含其他变量，返回空值
            return None
        # 提取分离出来的除数
        div.add(sep)
        # 从右侧表达式中去除已分离的部分
        rhs -= term.expand()
    
    # 进行除法操作
    fulldiv = reduce(operator.add, div)
    # 对左侧表达式和右侧表达式分别进行化简和展开
    lhs = simplify(lhs/fulldiv).expand()
    rhs = simplify(rhs/fulldiv).expand()
    
    # ...并检查是否成功分离了所有变量 :)
    if lhs.has(*others) or rhs.has(dep):
        # 如果左侧或右侧仍然包含其他变量，返回空值
        return None
    # 返回成功分离出来的左右两个表达式作为列表
    return [lhs, rhs]
```