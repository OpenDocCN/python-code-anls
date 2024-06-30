# `D:\src\scipysrc\sympy\sympy\solvers\ode\subscheck.py`

```
# 从 sympy.core 模块中导入 S 和 Pow
from sympy.core import S, Pow
# 从 sympy.core.function 模块中导入 Derivative, AppliedUndef, diff
from sympy.core.function import (Derivative, AppliedUndef, diff)
# 从 sympy.core.relational 模块中导入 Equality, Eq
from sympy.core.relational import Equality, Eq
# 从 sympy.core.symbol 模块中导入 Dummy
from sympy.core.symbol import Dummy
# 从 sympy.core.sympify 模块中导入 sympify
from sympy.core.sympify import sympify

# 从 sympy.logic.boolalg 模块中导入 BooleanAtom
from sympy.logic.boolalg import BooleanAtom
# 从 sympy.functions 模块中导入 exp
from sympy.functions import exp
# 从 sympy.series 模块中导入 Order
from sympy.series import Order
# 从 sympy.simplify.simplify 模块中导入 simplify, posify, besselsimp
from sympy.simplify.simplify import simplify, posify, besselsimp
# 从 sympy.simplify.trigsimp 模块中导入 trigsimp
from sympy.simplify.trigsimp import trigsimp
# 从 sympy.simplify.sqrtdenest 模块中导入 sqrtdenest
from sympy.simplify.sqrtdenest import sqrtdenest
# 从 sympy.solvers 模块中导入 solve
from sympy.solvers import solve
# 从 sympy.solvers.deutils 模块中导入 _preprocess, ode_order
from sympy.solvers.deutils import _preprocess, ode_order
# 从 sympy.utilities.iterables 模块中导入 iterable, is_sequence

# 定义一个名为 sub_func_doit 的函数，用于替换函数并评估其导数
def sub_func_doit(eq, func, new):
    r"""
    When replacing the func with something else, we usually want the
    derivative evaluated, so this function helps in making that happen.

    Examples
    ========

    >>> from sympy import Derivative, symbols, Function
    >>> from sympy.solvers.ode.subscheck import sub_func_doit
    >>> x, z = symbols('x, z')
    >>> y = Function('y')

    >>> sub_func_doit(3*Derivative(y(x), x) - 1, y(x), x)
    2

    >>> sub_func_doit(x*Derivative(y(x), x) - y(x)**2 + y(x), y(x),
    ... 1/(x*(z + 1/x)))
    x*(-1/(x**2*(z + 1/x)) + 1/(x**3*(z + 1/x)**2)) + 1/(x*(z + 1/x))
    ...- 1/(x**2*(z + 1/x)**2)
    """
    # 准备替换字典，用新值替换原始函数
    reps = {func: new}
    # 遍历方程中的 Derivative 对象
    for d in eq.atoms(Derivative):
        # 如果 Derivative 对象的表达式与原始函数相同
        if d.expr == func:
            # 计算其导数并加入替换字典
            reps[d] = new.diff(*d.variable_count)
        else:
            # 否则，将原始函数替换为新值并评估（不进行深度评估）
            reps[d] = d.xreplace({func: new}).doit(deep=False)
    # 使用替换字典替换方程中的符号，并返回结果
    return eq.xreplace(reps)


# 定义一个名为 checkodesol 的函数，用于验证常微分方程的解
def checkodesol(ode, sol, func=None, order='auto', solve_for_func=True):
    r"""
    Substitutes ``sol`` into ``ode`` and checks that the result is ``0``.

    This works when ``func`` is one function, like `f(x)` or a list of
    functions like `[f(x), g(x)]` when `ode` is a system of ODEs.  ``sol`` can
    be a single solution or a list of solutions.  Each solution may be an
    :py:class:`~sympy.core.relational.Equality` that the solution satisfies,
    e.g. ``Eq(f(x), C1), Eq(f(x) + C1, 0)``; or simply an
    :py:class:`~sympy.core.expr.Expr`, e.g. ``f(x) - C1``. In most cases it
    will not be necessary to explicitly identify the function, but if the
    function cannot be inferred from the original equation it can be supplied
    through the ``func`` argument.

    If a sequence of solutions is passed, the same sort of container will be
    used to return the result for each solution.

    It tries the following methods, in order, until it finds zero equivalence:

    1. Substitute the solution for `f` in the original equation.  This only
       works if ``ode`` is solved for `f`.  It will attempt to solve it first
       unless ``solve_for_func == False``.
    2. Take `n` derivatives of the solution, where `n` is the order of
       ``ode``, and check to see if that is equal to the solution.  This only
       works on exact ODEs.
    """
    # 待替换的符号与新值的映射
    reps = {func: sol}
    # 如果 sol 是一个列表，则为每个函数的解创建映射
    if isinstance(sol, (list, tuple)):
        for i, s in enumerate(sol):
            reps[func[i]] = s

    # 替换 sol 到 ode 中，并验证结果是否为零
    result = ode.subs(reps)

    # 如果验证失败，尝试进行导数检查
    if result != 0:
        # 如果自动解算标志位为真，则尝试求解 ode
        if solve_for_func:
            try:
                solved = solve(ode, func, dict=True)
                if len(solved) == 1:
                    result = checkodesol(ode, solved[0][func], func)
                else:
                    result = [checkodesol(ode, s[func], func) for s in solved]
            except NotImplementedError:
                pass

        # 如果仍未成功，尝试进行 n 阶导数检查
        if result != 0:
            try:
                n = ode_order(ode, func)
                for i in range(1, n + 1):
                    result = diff(result, func, i) - diff(sol, func, i)
                    if result != 0:
                        break
            except NotImplementedError:
                pass

    # 返回验证结果
    return result
    """
    Take the 1st, 2nd, ..., `n`\th derivatives of the solution, each time
    solving for the derivative of `f` of that order (this will always be
    possible because `f` is a linear operator). Then back substitute each
    derivative into ``ode`` in reverse order.
    """

    # This function returns a tuple. The first item in the tuple is ``True`` if
    # the substitution results in ``0``, and ``False`` otherwise. The second
    # item in the tuple is what the substitution results in. It should always
    # be ``0`` if the first item is ``True``. Sometimes this function will
    # return ``False`` even when an expression is identically equal to ``0``.
    # This happens when :py:meth:`~sympy.simplify.simplify.simplify` does not
    # reduce the expression to ``0``. If an expression returned by this
    # function vanishes identically, then ``sol`` really is a solution to
    # the ``ode``.

    # If this function seems to hang, it is probably because of a hard
    # simplification.

    # To use this function to test, test the first item of the tuple.

    # Examples
    # ========

    # Import necessary components from sympy library
    >>> from sympy import (Eq, Function, checkodesol, symbols,
    ...     Derivative, exp)

    # Define symbols and functions
    >>> x, C1, C2 = symbols('x,C1,C2')
    >>> f, g = symbols('f g', cls=Function)

    # Check if the differential equation f(x).diff(x) = C1 is satisfied
    >>> checkodesol(f(x).diff(x), Eq(f(x), C1))
    (True, 0)

    # Assert that the first item of the tuple for f(x).diff(x) = C1 is True
    >>> assert checkodesol(f(x).diff(x), C1)[0]

    # Assert that the first item of the tuple for f(x).diff(x) = x is False
    >>> assert not checkodesol(f(x).diff(x), x)[0]

    # Check if the differential equation f(x).diff(x, 2) = x**2 is satisfied
    >>> checkodesol(f(x).diff(x, 2), x**2)
    (False, 2)

    # Define a list of differential equations and their solutions
    >>> eqs = [Eq(Derivative(f(x), x), f(x)), Eq(Derivative(g(x), x), g(x))]
    >>> sol = [Eq(f(x), C1*exp(x)), Eq(g(x), C2*exp(x))]

    # Check if the solutions satisfy the system of differential equations
    >>> checkodesol(eqs, sol)
    (True, [0, 0])
    """

    # Check if the given `ode` is iterable, and if so, call `checksysodesol`
    if iterable(ode):
        return checksysodesol(ode, sol, func=func)

    # Ensure `ode` is an Equality object
    if not isinstance(ode, Equality):
        ode = Eq(ode, 0)

    # If `func` is None, attempt to preprocess `ode.lhs` to determine `func`
    if func is None:
        try:
            _, func = _preprocess(ode.lhs)
        except ValueError:
            # Determine unique functions in `sol` and ensure exactly one
            funcs = [s.atoms(AppliedUndef) for s in (
                sol if is_sequence(sol, set) else [sol])]
            funcs = set().union(*funcs)
            if len(funcs) != 1:
                raise ValueError(
                    'must pass func arg to checkodesol for this case.')
            func = funcs.pop()

    # Ensure `func` is a single-variable function
    if not isinstance(func, AppliedUndef) or len(func.args) != 1:
        raise ValueError(
            "func must be a function of one variable, not %s" % func)

    # If `sol` is a set, apply `checkodesol` recursively to each element
    if is_sequence(sol, set):
        return type(sol)([checkodesol(ode, i, order=order, solve_for_func=solve_for_func) for i in sol])

    # Ensure `sol` is an Equality object or convert it if necessary
    if not isinstance(sol, Equality):
        sol = Eq(func, sol)
    elif sol.rhs == func:
        sol = sol.reversed

    # Determine the order of the differential equation if not specified
    if order == 'auto':
        order = ode_order(ode, func)

    # Check if the solution `sol` satisfies the differential equation `ode`
    solved = sol.lhs == func and not sol.rhs.has(func)
    # 如果需要求解函数，并且尚未求解
    if solve_for_func and not solved:
        # 根据求解器求解方程 func = sol，返回右侧的可能解 rhs
        rhs = solve(sol, func)
        # 如果存在右侧解
        if rhs:
            # 将每个解转换为方程 Eq(func, t)，组成列表 eqs
            eqs = [Eq(func, t) for t in rhs]
            # 如果只有一个解
            if len(rhs) == 1:
                eqs = eqs[0]
            # 调用 checkodesol 函数，检查是否为常微分方程 ode 的解
            return checkodesol(ode, eqs, order=order,
                solve_for_func=False)

    # 获取函数 func 的第一个参数 x
    x = func.args[0]

    # 处理级数解的情况
    if sol.has(Order):
        # 断言 sol 是 Order 类型
        assert sol.lhs == func
        # 获取 Order 项 Oterm
        Oterm = sol.rhs.getO()
        # 移除 Order 项后的解 solrhs
        solrhs = sol.rhs.removeO()

        # 获取 Order 项的表达式 Oexpr，并确保是 Pow 类型
        Oexpr = Oterm.expr
        assert isinstance(Oexpr, Pow)
        # 获取序数 sorder
        sorder = Oexpr.exp
        # 断言 Oterm 是 Order(x**sorder)
        assert Oterm == Order(x**sorder)

        # 对于方程的差分 lhs - rhs，用 solrhs 替换 func 后，展开并求值
        odesubs = (ode.lhs-ode.rhs).subs(func, solrhs).doit().expand()

        # 创建新的 Order 对象，次数为 x 的 sorder - order
        neworder = Order(x**(sorder - order))
        # 更新 odesubs，加上新的 Order 项
        odesubs = odesubs + neworder
        # 断言 odesubs 的 Order 项为 neworder
        assert odesubs.getO() == neworder
        # 获取剩余部分（去除 Order 项后的部分）
        residual = odesubs.removeO()

        # 返回是否为解及其剩余部分
        return (residual == 0, residual)

    # 默认情况下，设置 s 为 True，设置测试编号为 0
    s = True
    testnum = 0
    # 如果 s 为 False
    if not s:
        # 返回 True 和 s
        return (True, s)
    # 否则如果 s 是 True，说明上述代码无法改变 s
    elif s is True:
        # 抛出未实现错误，说明无法测试 sol 是否是 ode 的解
        raise NotImplementedError("Unable to test if " + str(sol) +
            " is a solution to " + str(ode) + ".")
    else:
        # 返回 False 和 s
        return (False, s)
# 定义一个函数 checksysodesol，用于检查给定的方程组和解是否满足对应关系。
r"""
Substitutes corresponding ``sols`` for each functions into each ``eqs`` and
checks that the result of substitutions for each equation is ``0``. The
equations and solutions passed can be any iterable.

This only works when each ``sols`` have one function only, like `x(t)` or `y(t)`.
For each function, ``sols`` can have a single solution or a list of solutions.
In most cases it will not be necessary to explicitly identify the function,
but if the function cannot be inferred from the original equation it
can be supplied through the ``func`` argument.

When a sequence of equations is passed, the same sequence is used to return
the result for each equation with each function substituted with corresponding
solutions.

It tries the following method to find zero equivalence for each equation:

Substitute the solutions for functions, like `x(t)` and `y(t)` into the
original equations containing those functions.
This function returns a tuple.  The first item in the tuple is ``True`` if
the substitution results for each equation is ``0``, and ``False`` otherwise.
The second item in the tuple is what the substitution results in.  Each element
of the ``list`` should always be ``0`` corresponding to each equation if the
first item is ``True``. Note that sometimes this function may return ``False``,
but with an expression that is identically equal to ``0``, instead of returning
``True``.  This is because :py:meth:`~sympy.simplify.simplify.simplify` cannot
reduce the expression to ``0``.  If an expression returned by each function
vanishes identically, then ``sols`` really is a solution to ``eqs``.

If this function seems to hang, it is probably because of a difficult simplification.

Examples
========

>>> from sympy import Eq, diff, symbols, sin, cos, exp, sqrt, S, Function
>>> from sympy.solvers.ode.subscheck import checksysodesol
>>> C1, C2 = symbols('C1:3')
>>> t = symbols('t')
>>> x, y = symbols('x, y', cls=Function)
>>> eq = (Eq(diff(x(t),t), x(t) + y(t) + 17), Eq(diff(y(t),t), -2*x(t) + y(t) + 12))
>>> sol = [Eq(x(t), (C1*sin(sqrt(2)*t) + C2*cos(sqrt(2)*t))*exp(t) - S(5)/3),
... Eq(y(t), (sqrt(2)*C1*cos(sqrt(2)*t) - sqrt(2)*C2*sin(sqrt(2)*t))*exp(t) - S(46)/3)]
>>> checksysodesol(eq, sol)
(True, [0, 0])
>>> eq = (Eq(diff(x(t),t),x(t)*y(t)**4), Eq(diff(y(t),t),y(t)**3))
>>> sol = [Eq(x(t), C1*exp(-1/(4*(C2 + t)))), Eq(y(t), -sqrt(2)*sqrt(-1/(C2 + t))/2),
... Eq(x(t), C1*exp(-1/(4*(C2 + t)))), Eq(y(t), sqrt(2)*sqrt(-1/(C2 + t))/2)]
>>> checksysodesol(eq, sol)
(True, [0, 0])

"""
# 定义一个内部函数 _sympify，将输入的方程或方程列表转换为 sympy 的表达式
def _sympify(eq):
    return list(map(sympify, eq if iterable(eq) else [eq]))

# 将输入的方程列表 eqs 转换为 sympy 的表达式列表
eqs = _sympify(eqs)

# 遍历方程列表 eqs，将等式转化为左侧减去右侧的形式，以便后续计算
for i in range(len(eqs)):
    if isinstance(eqs[i], Equality):
        eqs[i] = eqs[i].lhs - eqs[i].rhs
    # 如果 func 参数为 None，则初始化 funcs 列表为空
    if func is None:
        # 遍历给定的方程列表 eqs
        for eq in eqs:
            # 找到方程中所有的导数项
            derivs = eq.atoms(Derivative)
            # 找到每个导数项中的函数符号
            func = set().union(*[d.atoms(AppliedUndef) for d in derivs])
            # 将找到的函数符号添加到 funcs 列表中
            funcs.extend(func)
        # 去除重复的函数符号，并转换为列表形式
        funcs = list(set(funcs))
    
    # 检查 funcs 中的每个函数是否是只含一个变量的函数符号，并且所有函数参数相同
    if not all(isinstance(func, AppliedUndef) and len(func.args) == 1 for func in funcs)\
    and len({func.args for func in funcs})!=1:
        # 如果不符合条件，抛出 ValueError 异常
        raise ValueError("func must be a function of one variable, not %s" % func)
    
    # 检查每个解 sols 是否只包含一个函数符号
    for sol in sols:
        if len(sol.atoms(AppliedUndef)) != 1:
            # 如果不符合条件，抛出 ValueError 异常
            raise ValueError("solutions should have one function only")
    
    # 检查提供的解的数量是否与方程数量相匹配
    if len(funcs) != len({sol.lhs for sol in sols}):
        # 如果数量不匹配，抛出 ValueError 异常
        raise ValueError("number of solutions provided does not match the number of equations")
    
    # 初始化一个空字典，用于存储每个函数的解
    dictsol = {}
    # 遍历每个解 sols
    for sol in sols:
        # 获取解中的函数符号
        func = list(sol.atoms(AppliedUndef))[0]
        # 如果解的右侧等于函数符号本身，则将解取反
        if sol.rhs == func:
            sol = sol.reversed
        # 检查是否成功求解出函数符号
        solved = sol.lhs == func and not sol.rhs.has(func)
        if not solved:
            # 如果未成功求解，调用 solve 函数求解
            rhs = solve(sol, func)
            if not rhs:
                # 如果未找到解，抛出 NotImplementedError 异常
                raise NotImplementedError
        else:
            # 如果已成功求解，则将右侧结果作为解
            rhs = sol.rhs
        # 将求解结果添加到 dictsol 字典中
        dictsol[func] = rhs
    
    # 初始化一个空列表，用于存储简化后的方程
    checkeq = []
    # 遍历每个方程 eqs
    for eq in eqs:
        # 针对每个方程中的函数符号 funcs，用其解 dictsol[func] 替换方程中的对应部分
        for func in funcs:
            eq = sub_func_doit(eq, func, dictsol[func])
        # 简化方程
        ss = simplify(eq)
        if ss != 0:
            # 如果简化结果不为零，进一步展开和化简
            eq = ss.expand(force=True)
            # 对方程进行平方根的展开和化简处理
            eq = sqrtdenest(eq).simplify()
        else:
            # 如果简化结果为零，则方程直接为零
            eq = 0
        # 将处理后的方程添加到 checkeq 列表中
        checkeq.append(eq)
    
    # 检查所有处理后的方程是否都为零
    if len(set(checkeq)) == 1 and list(set(checkeq))[0] == 0:
        # 如果所有方程都为零，返回 True 和 checkeq 列表
        return (True, checkeq)
    else:
        # 如果有非零方程，返回 False 和 checkeq 列表
        return (False, checkeq)
```