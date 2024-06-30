# `D:\src\scipysrc\sympy\sympy\solvers\bivariate.py`

```
from sympy.core.add import Add
from sympy.core.exprtools import factor_terms
from sympy.core.function import expand_log, _mexpand
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy
from sympy.functions.elementary.exponential import (LambertW, exp, log)
from sympy.functions.elementary.miscellaneous import root
from sympy.polys.polyroots import roots
from sympy.polys.polytools import Poly, factor
from sympy.simplify.simplify import separatevars
from sympy.simplify.radsimp import collect
from sympy.simplify.simplify import powsimp
from sympy.solvers.solvers import solve, _invert
from sympy.utilities.iterables import uniq


def _filtered_gens(poly, symbol):
    """Process the generators of ``poly``, returning the set of generators that
    have ``symbol``. If there are two generators that are inverses of each other,
    prefer the one that has no denominator.

    Examples
    ========

    >>> from sympy.solvers.bivariate import _filtered_gens
    >>> from sympy import Poly, exp
    >>> from sympy.abc import x
    >>> _filtered_gens(Poly(x + 1/x + exp(x)), x)
    {x, exp(x)}

    """
    # 选择 ``poly`` 的生成器中包含给定符号 ``symbol`` 的集合
    gens = {g for g in poly.gens if symbol in g.free_symbols}
    for g in list(gens):
        # 对于存在互为倒数的两个生成器，选择没有分母的那一个
        ag = 1/g
        if g in gens and ag in gens:
            if ag.as_numer_denom()[1] is not S.One:
                g = ag
            gens.remove(g)
    return gens


def _mostfunc(lhs, func, X=None):
    """Returns the term in lhs which contains the most of the
    func-type things e.g. log(log(x)) wins over log(x) if both terms appear.

    ``func`` can be a function (exp, log, etc...) or any other SymPy object,
    like Pow.

    If ``X`` is not ``None``, then the function returns the term composed with the
    most ``func`` having the specified variable.

    Examples
    ========

    >>> from sympy.solvers.bivariate import _mostfunc
    >>> from sympy import exp
    >>> from sympy.abc import x, y
    >>> _mostfunc(exp(x) + exp(exp(x) + 2), exp)
    exp(exp(x) + 2)
    >>> _mostfunc(exp(x) + exp(exp(y) + 2), exp)
    exp(exp(y) + 2)
    >>> _mostfunc(exp(x) + exp(exp(y) + 2), exp, x)
    exp(x)
    >>> _mostfunc(x, exp, x) is None
    True
    >>> _mostfunc(exp(x) + exp(x*y), exp, x)
    exp(x)
    """
    # 找到包含最多 ``func`` 类型项的表达式片段
    fterms = [tmp for tmp in lhs.atoms(func) if (not X or
        X.is_Symbol and X in tmp.free_symbols or
        not X.is_Symbol and tmp.has(X))]
    if len(fterms) == 1:
        return fterms[0]
    elif fterms:
        return max(list(ordered(fterms)), key=lambda x: x.count(func))
    return None


def _linab(arg, symbol):
    """Return ``a, b, X`` assuming ``arg`` can be written as ``a*X + b``
    where ``X`` is a symbol-dependent factor and ``a`` and ``b`` are
    independent of ``symbol``.

    Examples
    ========

    ```
    """
    # 假设 ``arg`` 可以写成 ``a*X + b`` 的形式，返回 ``a, b, X``
    # 这里的 ``X`` 是依赖于符号的因子，``a`` 和 ``b`` 与 ``symbol`` 独立
    # 导入需要的函数和符号变量
    >>> from sympy.solvers.bivariate import _linab
    >>> from sympy.abc import x, y
    >>> from sympy import exp, S
    # 对线性代数表达式进行求解，返回元组(a, b, x)，其中：
    # - a 是主导系数
    # - b 是常数项系数
    # - x 是符号变量
    >>> _linab(S(2), x)
    # 当输入为 S(2) 和 x 时，返回结果为 (2, 0, 1)
    (2, 0, 1)
    >>> _linab(2*x, x)
    # 当输入为 2*x 和 x 时，返回结果为 (2, 0, x)
    (2, 0, x)
    >>> _linab(y + y*x + 2*x, x)
    # 当输入为 y + y*x + 2*x 和 x 时，返回结果为 (y + 2, y, x)
    (y + 2, y, x)
    >>> _linab(3 + 2*exp(x), x)
    # 当输入为 3 + 2*exp(x) 和 x 时，返回结果为 (2, 3, exp(x))
    (2, 3, exp(x))
    """
    # 对输入参数进行因式分解和展开处理
    arg = factor_terms(arg.expand())
    # 将表达式 arg 分解为主导系数 ind 和依赖于符号变量 symbol 的部分 dep
    ind, dep = arg.as_independent(symbol)
    # 如果 dep 是乘法并且包含加法项
    if arg.is_Mul and dep.is_Add:
        # 递归调用 _linab 处理 dep，返回结果为 (a, b, x)
        a, b, x = _linab(dep, symbol)
        # 返回处理后的结果，主导系数乘以 ind，常数项系数乘以 ind，符号变量 x 不变
        return ind*a, ind*b, x
    # 如果 arg 不是加法
    if not arg.is_Add:
        # 常数项系数 b 设为 0，主导系数 a 为 ind，符号变量 x 为 dep
        b = 0
        a, x = ind, dep
    else:
        # 否则，常数项系数 b 设为 ind，将 dep 分解为独立的符号变量和加法项
        b = ind
        a, x = separatevars(dep).as_independent(symbol, as_Add=False)
    # 如果符号变量 x 可以提取负号
    if x.could_extract_minus_sign():
        # 将主导系数 a 取反
        a = -a
        # 将符号变量 x 取反
        x = -x
    # 返回最终处理后的主导系数 a，常数项系数 b，符号变量 x 的元组
    return a, b, x
# 定义一个函数 `_lambert`，用于求解 Lambert 方程
def _lambert(eq, x):
    """
    Given an expression assumed to be in the form
        ``F(X, a..f) = a*log(b*X + c) + d*X + f = 0``
    where X = g(x) and x = g^-1(X), return the Lambert solution,
        ``x = g^-1(-c/b + (a/d)*W(d/(a*b)*exp(c*d/a/b)*exp(-f/a)))``.
    """
    # 对方程进行一系列的代数和简化操作
    eq = _mexpand(expand_log(eq))
    # 找出方程中主要的对数项
    mainlog = _mostfunc(eq, log, x)
    # 如果没有找到主要的对数项，则返回空列表，表示违反了假设
    if not mainlog:
        return []

    # 提取出其他项，并检查是否为负的对数
    other = eq.subs(mainlog, 0)
    if isinstance(-other, log):
        eq = (eq - other).subs(mainlog, mainlog.args[0])
        mainlog = mainlog.args[0]
        if not isinstance(mainlog, log):
            return []  # 违反了假设，返回空列表
        other = -(-other).args[0]
        eq += other

    # 如果变量 x 不在其他项中，则返回空列表，表示违反了假设
    if x not in other.free_symbols:
        return []

    # 对其他项进行线性代数处理，得到系数和表达式
    d, f, X2 = _linab(other, x)
    # 将方程中除了主要对数项之外的部分整理成对数的形式
    logterm = collect(eq - other, mainlog)
    a = logterm.as_coefficient(mainlog)
    # 如果系数 a 为 None 或者 x 是其自由符号，则返回空列表，表示违反了假设
    if a is None or x in a.free_symbols:
        return []

    # 获取对数参数的表达式
    logarg = mainlog.args[0]
    b, c, X1 = _linab(logarg, x)
    # 如果 X1 不等于 X2，则返回空列表，表示违反了假设
    if X1 != X2:
        return []

    # 反转生成器 X1，得到 x(u) 的表达式
    u = Dummy('rhs')
    xusolns = solve(X1 - u, x)

    # LambertW 函数有无限多个分支，但只有 k = -1 和 0 的分支可能是实数。
    # 如果 k = 0 的分支是实数，那么它是实数，而 k = -1 的分支只有在 LambertW 的参数在 [-1/e, 0] 范围内时才是实数。
    # `solve` 不会返回无限多的解，所以我们只在 k = -1 的分支实数时才包含它。
    lambert_real_branches = [-1, 0]
    sol = []

    # 计算 LambertW 函数的参数
    num, den = ((c*d-b*f)/a/b).as_numer_denom()
    p, den = den.as_coeff_Mul()
    e = exp(num/den)
    t = Dummy('t')
    # 计算 LambertW 函数的参数列表
    args = [d/(a*b)*t for t in roots(t**p - e, t).keys()]

    # 根据参数计算解
    for arg in args:
        for k in lambert_real_branches:
            w = LambertW(arg, k)
            # 如果 k 不为 0 且 LambertW 不是实数，则跳过
            if k and not w.is_real:
                continue
            rhs = -c/b + (a/d)*w

            # 将 u 替换为 rhs，并将结果添加到解列表中
            sol.extend(xu.subs(u, rhs) for xu in xusolns)

    # 返回所有计算出的解
    return sol


# 定义一个函数 `_solve_lambert`，用于判断并返回 Lambert 类型表达式的解
def _solve_lambert(f, symbol, gens):
    """Return solution to ``f`` if it is a Lambert-type expression
    else raise NotImplementedError.

    For ``f(X, a..f) = a*log(b*X + c) + d*X - f = 0`` the solution
    for ``X`` is ``X = -c/b + (a/d)*W(d/(a*b)*exp(c*d/a/b)*exp(f/a))``.
    """
    There are a variety of forms for `f(X, a..f)` as enumerated below:

    1a1)
      if B**B = R for R not in [0, 1] (since those cases would already
      be solved before getting here) then log of both sides gives
      log(B) + log(log(B)) = log(log(R)) and
      X = log(B), a = 1, b = 1, c = 0, d = 1, f = log(log(R))

    1a2)
      if B*(b*log(B) + c)**a = R then log of both sides gives
      log(B) + a*log(b*log(B) + c) = log(R) and
      X = log(B), d=1, f=log(R)

    1b)
      if a*log(b*B + c) + d*B = R and
      X = B, f = R

    2a)
      if (b*B + c)*exp(d*B + g) = R then log of both sides gives
      log(b*B + c) + d*B + g = log(R) and
      X = B, a = 1, f = log(R) - g

    2b)
      if g*exp(d*B + h) - b*B = c then the log form is
      log(g) + d*B + h - log(b*B + c) = 0 and
      X = B, a = -1, f = -h - log(g)

    3)
      if d*p**(a*B + g) - b*B = c then the log form is
      log(d) + (a*B + g)*log(p) - log(b*B + c) = 0 and
      X = B, a = -1, d = a*log(p), f = -log(d) - g*log(p)


注释：
    def _solve_even_degree_expr(expr, t, symbol):
        """Return the unique solutions of equations derived from
        ``expr`` by replacing ``t`` with ``+/- symbol``.

        Parameters
        ==========

        expr : Expr
            The expression which includes a dummy variable t to be
            replaced with +symbol and -symbol.

        symbol : Symbol
            The symbol for which a solution is being sought.

        Returns
        =======

        List of unique solution of the two equations generated by
        replacing ``t`` with positive and negative ``symbol``.

        Notes
        =====

        If ``expr = 2*log(t) + x/2` then solutions for
        ``2*log(x) + x/2 = 0`` and ``2*log(-x) + x/2 = 0`` are
        returned by this function. Though this may seem
        counter-intuitive, one must note that the ``expr`` being
        solved here has been derived from a different expression. For
        an expression like ``eq = x**2*g(x) = 1``, if we take the
        log of both sides we obtain ``log(x**2) + log(g(x)) = 0``. If
        x is positive then this simplifies to
        ``2*log(x) + log(g(x)) = 0``; the Lambert-solving routines will
        return solutions for this, but we must also consider the
        solutions for  ``2*log(-x) + log(g(x))`` since those must also
        be a solution of ``eq`` which has the same value when the ``x``
        in ``x**2`` is negated. If `g(x)` does not have even powers of
        symbol then we do not want to replace the ``x`` there with
        ``-x``. So the role of the ``t`` in the expression received by
        this function is to mark where ``+/-x`` should be inserted
        before obtaining the Lambert solutions.

        """
        # Replace t in expr with positive and negative symbol and store results
        nlhs, plhs = [
            expr.xreplace({t: sgn*symbol}) for sgn in (-1, 1)]
        # Solve equations using Lambert W function for both nlhs and plhs
        sols = _solve_lambert(nlhs, symbol, gens)
        # If plhs differs from nlhs, solve for plhs as well
        if plhs != nlhs:
            sols.extend(_solve_lambert(plhs, symbol, gens))
        # Return unique solutions ensuring canonical order
        return list(uniq(sols))

    # Separate the expression into its non-symbolic and symbolic parts
    nrhs, lhs = f.as_independent(symbol, as_Add=True)
    # Compute the right-hand side of the equation
    rhs = -nrhs

    # Check if any of the generators (gens) involve exponential or logarithmic functions,
    # or if the symbol appears as an exponent in any power function
    lamcheck = [tmp for tmp in gens
                if (tmp.func in [exp, log] or
                (tmp.is_Pow and symbol in tmp.exp.free_symbols))]
    # If no generators match the criteria, raise NotImplementedError
    if not lamcheck:
        raise NotImplementedError()
    # 如果 lhs 是 Add 或者 Mul 类型
    if lhs.is_Add or lhs.is_Mul:
        # 将符号的偶次幂用虚拟变量 t 替换
        # 这些需要特殊处理；非 Add/Mul 类型不需要这种处理
        t = Dummy('t', **symbol.assumptions0)
        # 替换 lhs 中所有符号的偶次幂为 t**偶次幂
        lhs = lhs.replace(
            lambda i:  # 找到 symbol**偶次幂
                i.is_Pow and i.base == symbol and i.exp.is_even,
            lambda i:  # 替换为 t**偶次幂
                t**i.exp)

        # 如果 lhs 是 Add 类型并且包含 t
        if lhs.is_Add and lhs.has(t):
            # 提取独立于 t 的部分
            t_indep = lhs.subs(t, 0)
            t_term = lhs - t_indep
            _rhs = rhs - t_indep
            # 如果 t_term 不是 Add 类型且 _rhs 存在且 t_term 不包含 ComplexInfinity 或 NaN
            if not t_term.is_Add and _rhs and not (
                    t_term.has(S.ComplexInfinity, S.NaN)):
                # 构造等式 eq = log(t_term) - log(_rhs)
                eq = expand_log(log(t_term) - log(_rhs))
                # 返回解决偶次方程表达式的结果
                return _solve_even_degree_expr(eq, t, symbol)
        # 如果 lhs 是 Mul 类型并且 rhs 存在
        elif lhs.is_Mul and rhs:
            # 强制将 lhs 展开为对数形式
            lhs = expand_log(log(lhs), force=True)
            rhs = log(rhs)
            # 如果 lhs 包含 t 并且是 Add 类型
            if lhs.has(t) and lhs.is_Add:
                # 将 lhs - rhs 构成等式 eq
                eq = lhs - rhs
                # 返回解决偶次方程表达式的结果
                return _solve_even_degree_expr(eq, t, symbol)

        # 恢复 lhs 中的符号为 symbol
        lhs = lhs.xreplace({t: symbol})

    # 对 lhs 进行指数简化和因式分解
    lhs = powsimp(factor(lhs, deep=True))

    # 确保尽可能完全地反转
    r = Dummy()
    # 将 lhs - r 反转，得到 i 和简化后的 lhs
    i, lhs = _invert(lhs - r, symbol)
    # 使用 r 替换 i 中的 rhs
    rhs = i.xreplace({r: rhs})

    # 对于以下形式：
    #
    # 1a1) B**B = R 将到达这里，作为 B*log(B) = log(R)
    #      lhs 是 Mul，所以对两边取对数：
    #        log(B) + log(log(B)) = log(log(R))
    # 1a2) B*(b*log(B) + c)**a = R 将不变地到达这里，因此
    #      lhs 是 Mul，所以对两边取对数：
    #        log(B) + a*log(b*log(B) + c) = log(R)
    # 1b) d*log(a*B + b) + c*B = R 将不变地到达这里，因此
    #      lhs 是 Add，所以分离 c*B 并对两边取对数：
    #        log(c) + log(B) = log(R - d*log(a*B + b))

    soln = []
    # 如果没有解决方案
    if not soln:
        # 找到 lhs 中主要的 log 函数
        mainlog = _mostfunc(lhs, log, symbol)
        if mainlog:
            # 如果 lhs 是 Mul 并且 rhs 不等于 0
            if lhs.is_Mul and rhs != 0:
                # 解决 log(lhs) - log(rhs) 的 Lambert 函数
                soln = _lambert(log(lhs) - log(rhs), symbol)
            # 如果 lhs 是 Add
            elif lhs.is_Add:
                other = lhs.subs(mainlog, 0)
                # 如果 other 存在且不是 Add，并且包含 symbol 的 Pow
                if other and not other.is_Add and [
                        tmp for tmp in other.atoms(Pow)
                        if symbol in tmp.free_symbols]:
                    # 如果 rhs 为 0，则构造差值 diff = log(other) - log(other - lhs)
                    if not rhs:
                        diff = log(other) - log(other - lhs)
                    else:
                        diff = log(lhs - other) - log(rhs - other)
                    # 解决 expand_log(diff) 的 Lambert 函数
                    soln = _lambert(expand_log(diff), symbol)
                else:
                    # 直接解决 lhs - rhs 的 Lambert 函数
                    soln = _lambert(lhs - rhs, symbol)

    # 对于下一种形式，
    #
    #     收集主指数
    #     2a) (b*B + c)*exp(d*B + g) = R
    #         lhs 是 mul，所以对两边取对数：
    #           log(b*B + c) + d*B = log(R) - g
    #     2b) g*exp(d*B + h) - b*B = R
    #         lhs is add, so add b*B to both sides,
    #         take the log of both sides and rearrange to give
    #           log(R + b*B) - d*B = log(g) + h

    # 如果没有找到解决方案
    if not soln:
        # 找出 lhs 中包含最主要函数的表达式
        mainexp = _mostfunc(lhs, exp, symbol)
        # 如果找到了主要表达式
        if mainexp:
            # 将 lhs 中的项按照主要表达式进行收集
            lhs = collect(lhs, mainexp)
            # 如果 lhs 是乘法并且 rhs 不等于 0
            if lhs.is_Mul and rhs != 0:
                # 对 lhs 取对数后，扩展并使用 Lambert W 函数求解
                soln = _lambert(expand_log(log(lhs) - log(rhs)), symbol)
            elif lhs.is_Add:
                # 将所有不包含主要表达式的项移到 rhs
                other = lhs.subs(mainexp, 0)
                mainterm = lhs - other
                rhs = rhs - other
                # 如果主要项可以提取负号且 rhs 也可以提取负号
                if (mainterm.could_extract_minus_sign() and
                    rhs.could_extract_minus_sign()):
                    mainterm *= -1
                    rhs *= -1
                # 计算 log(mainterm) - log(rhs) 后使用 Lambert W 函数求解
                diff = log(mainterm) - log(rhs)
                soln = _lambert(expand_log(diff), symbol)

    # 对于最后一种形式:
    #
    #  3) d*p**(a*B + g) - b*B = c
    #     对主要幂函数进行收集，两边加上 b*B
    #     对两边取对数并重排得到
    #       a*B*log(p) - log(b*B + c) = -log(d) - g*log(p)
    if not soln:
        # 找出 lhs 中包含最主要幂函数的表达式
        mainpow = _mostfunc(lhs, Pow, symbol)
        # 如果找到了主要幂函数并且符号在其指数中
        if mainpow and symbol in mainpow.exp.free_symbols:
            # 将 lhs 中的项按照主要幂函数进行收集
            lhs = collect(lhs, mainpow)
            # 如果 lhs 是乘法并且 rhs 不等于 0
            if lhs.is_Mul and rhs != 0:
                # 对 lhs 取对数后，扩展并使用 Lambert W 函数求解
                soln = _lambert(expand_log(log(lhs) - log(rhs)), symbol)
            elif lhs.is_Add:
                # 将所有不包含主要幂函数的项移到 rhs
                other = lhs.subs(mainpow, 0)
                mainterm = lhs - other
                rhs = rhs - other
                # 计算 log(mainterm) - log(rhs) 后使用 Lambert W 函数求解
                diff = log(mainterm) - log(rhs)
                soln = _lambert(expand_log(diff), symbol)

    # 如果依然没有找到解决方案，抛出未实现的错误
    if not soln:
        raise NotImplementedError('%s does not appear to have a solution in '
            'terms of LambertW' % f)

    # 返回按顺序排列的解决方案列表
    return list(ordered(soln))
# 定义一个函数用于确定给定表达式 `f` 的双变量类型，考虑以下三种可能性：
# u(x, y) = x*y
# u(x, y) = x + y
# u(x, y) = x*y + x
# u(x, y) = x*y + y
def bivariate_type(f, x, y, *, first=True):
    # 创建一个正值假设的虚拟变量 u
    u = Dummy('u', positive=True)

    # 如果是第一次递归调用
    if first:
        # 将 f 转化为多项式对象 p
        p = Poly(f, x, y)
        # 获取多项式表达式
        f = p.as_expr()
        # 创建两个虚拟变量 _x 和 _y
        _x = Dummy()
        _y = Dummy()
        # 递归调用 bivariate_type 函数，将多项式表达式替换为 _x 和 _y，first 设为 False
        rv = bivariate_type(Poly(f.subs({x: _x, y: _y}), _x, _y), _x, _y, first=False)
        # 如果有返回值，则替换 _x 和 _y 并返回
        if rv:
            reps = {_x: x, _y: y}
            return rv[0].xreplace(reps), rv[1].xreplace(reps), rv[2]
        return

    # 如果不是第一次调用，则多项式 p 就是输入的 f
    p = f
    f = p.as_expr()

    # 判断是否是 f(x*y) 的形式
    args = Add.make_args(p.as_expr())
    new = []
    for a in args:
        # 替换 x 为 u/y 并简化
        a = _mexpand(a.subs(x, u/y))
        # 检查是否包含 x 或 y 的自由变量，如果有则终止循环
        free = a.free_symbols
        if x in free or y in free:
            break
        new.append(a)
    else:
        # 如果循环未中断，则返回 x*y, Add(*new), u
        return x*y, Add(*new), u

    # 定义一个辅助函数，用于检查 f 是否满足给定条件
    def ok(f, v, c):
        new = _mexpand(f.subs(v, c))
        free = new.free_symbols
        return None if (x in free or y in free) else new

    # 判断是否是 f(a*x + b*y) 的形式
    new = []
    d = p.degree(x)
    if p.degree(y) == d:
        a = root(p.coeff_monomial(x**d), d)
        b = root(p.coeff_monomial(y**d), d)
        new = ok(f, x, (u - b*y)/a)
        if new is not None:
            return a*x + b*y, new, u

    # 判断是否是 f(a*x*y + b*y) 的形式
    new = []
    d = p.degree(x)
    if p.degree(y) == d:
        for itry in range(2):
            a = root(p.coeff_monomial(x**d*y**d), d)
            b = root(p.coeff_monomial(y**d), d)
            new = ok(f, x, (u - b*y)/a/y)
            if new is not None:
                return a*x*y + b*y, new, u
            # 交换 x 和 y 的位置重新尝试
            x, y = y, x
```