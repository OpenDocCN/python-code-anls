# `D:\src\scipysrc\sympy\sympy\solvers\inequalities.py`

```
"""Tools for solving inequalities and systems of inequalities. """
import itertools  # 导入 itertools 模块，用于迭代工具函数

from sympy.calculus.util import (continuous_domain, periodicity,
    function_range)  # 从 sympy.calculus.util 导入连续域、周期性和函数值域计算函数
from sympy.core import sympify  # 导入 sympy.core 的 sympify 函数，用于转换表达式类型
from sympy.core.exprtools import factor_terms  # 导入 sympy.core.exprtools 的 factor_terms 函数，用于因式分解
from sympy.core.relational import Relational, Lt, Ge, Eq  # 导入关系运算相关类和操作符
from sympy.core.symbol import Symbol, Dummy  # 导入符号和虚拟符号类
from sympy.sets.sets import Interval, FiniteSet, Union, Intersection  # 导入集合运算相关类
from sympy.core.singleton import S  # 导入 sympy.core.singleton 的 S，表示各种常用集合
from sympy.core.function import expand_mul  # 导入 sympy.core.function 的 expand_mul 函数，用于展开乘法
from sympy.functions.elementary.complexes import Abs  # 导入复数函数的绝对值函数
from sympy.logic import And  # 导入逻辑运算中的与运算
from sympy.polys import Poly, PolynomialError, parallel_poly_from_expr  # 导入多项式操作相关函数和类
from sympy.polys.polyutils import _nsort  # 导入多项式工具函数
from sympy.solvers.solveset import solvify, solveset  # 导入解方程的函数
from sympy.utilities.iterables import sift, iterable  # 导入迭代工具函数
from sympy.utilities.misc import filldedent  # 导入填充文本的函数

def solve_poly_inequality(poly, rel):
    """Solve a polynomial inequality with rational coefficients.

    Examples
    ========

    >>> from sympy import solve_poly_inequality, Poly
    >>> from sympy.abc import x

    >>> solve_poly_inequality(Poly(x, x, domain='ZZ'), '==')
    [{0}]

    >>> solve_poly_inequality(Poly(x**2 - 1, x, domain='ZZ'), '!=')
    [Interval.open(-oo, -1), Interval.open(-1, 1), Interval.open(1, oo)]

    >>> solve_poly_inequality(Poly(x**2 - 1, x, domain='ZZ'), '==')
    [{-1}, {1}]

    See Also
    ========
    solve_poly_inequalities
    """
    if not isinstance(poly, Poly):  # 如果 poly 不是 Poly 类的实例，则抛出 ValueError
        raise ValueError(
            'For efficiency reasons, `poly` should be a Poly instance')
    if poly.as_expr().is_number:  # 如果 poly 转换为表达式后是一个数字
        t = Relational(poly.as_expr(), 0, rel)  # 创建关系表达式
        if t is S.true:  # 如果 t 是真值
            return [S.Reals]  # 返回整个实数集
        elif t is S.false:  # 如果 t 是假值
            return [S.EmptySet]  # 返回空集
        else:
            raise NotImplementedError(  # 否则抛出未实现的错误
                "could not determine truth value of %s" % t)

    reals, intervals = poly.real_roots(multiple=False), []  # 计算多项式的实根

    if rel == '==':  # 如果是等号关系
        for root, _ in reals:
            interval = Interval(root, root)  # 创建闭区间
            intervals.append(interval)  # 添加到区间列表中
    elif rel == '!=':  # 如果是不等号关系
        left = S.NegativeInfinity  # 左端点为负无穷

        for right, _ in reals + [(S.Infinity, 1)]:  # 对于每个实根和正无穷
            interval = Interval(left, right, True, True)  # 创建开区间
            intervals.append(interval)  # 添加到区间列表中
            left = right  # 更新左端点为当前右端点
    # 如果多项式 poly 的首项系数大于 0
    else:
        if poly.LC() > 0:
            # 设置符号为正
            sign = +1
        else:
            # 设置符号为负
            sign = -1

        # 初始化等式符号和等式标志
        eq_sign, equal = None, False

        # 根据关系符号 rel 设置等式符号和等式标志
        if rel == '>':
            eq_sign = +1
        elif rel == '<':
            eq_sign = -1
        elif rel == '>=':
            eq_sign, equal = +1, True
        elif rel == '<=':
            eq_sign, equal = -1, True
        else:
            # 如果关系符号不是有效的，则抛出 ValueError 异常
            raise ValueError("'%s' is not a valid relation" % rel)

        # 初始化右边界和右边界开放性
        right, right_open = S.Infinity, True

        # 遍历反转后的实数列表 reals
        for left, multiplicity in reversed(reals):
            # 如果多重度是奇数
            if multiplicity % 2:
                # 根据符号等式将区间插入 intervals 列表中
                if sign == eq_sign:
                    intervals.insert(
                        0, Interval(left, right, not equal, right_open))

                # 更新符号、右边界和右边界开放性
                sign, right, right_open = -sign, left, not equal
            else:
                # 如果多重度是偶数
                if sign == eq_sign and not equal:
                    # 根据符号等式将区间插入 intervals 列表中
                    intervals.insert(
                        0, Interval(left, right, True, right_open))
                    # 更新右边界和右边界开放性
                    right, right_open = left, True
                elif sign != eq_sign and equal:
                    # 根据符号不等式将区间插入 intervals 列表中
                    intervals.insert(0, Interval(left, left))

        # 如果符号等式仍然成立，则插入最小区间
        if sign == eq_sign:
            intervals.insert(
                0, Interval(S.NegativeInfinity, right, True, right_open))

    # 返回计算得到的 intervals 列表
    return intervals
# 解决多项式不等式系统，其中每个不等式用有理数系数的多项式表示
def solve_poly_inequalities(polys):
    """Solve polynomial inequalities with rational coefficients.

    Examples
    ========

    >>> from sympy import Poly
    >>> from sympy.solvers.inequalities import solve_poly_inequalities
    >>> from sympy.abc import x
    >>> solve_poly_inequalities(((
    ... Poly(x**2 - 3), ">"), (
    ... Poly(-x**2 + 1), ">")))
    Union(Interval.open(-oo, -sqrt(3)), Interval.open(-1, 1), Interval.open(sqrt(3), oo))
    """
    # 初始化结果为空集
    return Union(*[s for p in polys for s in solve_poly_inequality(*p)])


# 解决具有有理系数的有理不等式系统
def solve_rational_inequalities(eqs):
    """Solve a system of rational inequalities with rational coefficients.

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy import solve_rational_inequalities, Poly

    >>> solve_rational_inequalities([[
    ... ((Poly(-x + 1), Poly(1, x)), '>='),
    ... ((Poly(-x + 1), Poly(1, x)), '<=')]])
    {1}

    >>> solve_rational_inequalities([[
    ... ((Poly(x), Poly(1, x)), '!='),
    ... ((Poly(-x + 1), Poly(1, x)), '>=')]])
    Union(Interval.open(-oo, 0), Interval.Lopen(0, 1))

    See Also
    ========
    solve_poly_inequality
    """
    # 初始化结果为空集
    result = S.EmptySet

    # 对每个不等式系统进行处理
    for _eqs in eqs:
        if not _eqs:
            continue

        # 初始全局区间为整个实数轴
        global_intervals = [Interval(S.NegativeInfinity, S.Infinity)]

        # 遍历每个不等式及其关系
        for (numer, denom), rel in _eqs:
            # 解决分子乘以分母的不等式
            numer_intervals = solve_poly_inequality(numer*denom, rel)
            # 解决分母等式的不等式
            denom_intervals = solve_poly_inequality(denom, '==')

            intervals = []

            # 使用笛卡尔积找到有效区间
            for numer_interval, global_interval in itertools.product(
                    numer_intervals, global_intervals):
                interval = numer_interval.intersect(global_interval)

                if interval is not S.EmptySet:
                    intervals.append(interval)

            global_intervals = intervals

            intervals = []

            # 移除分母区间内的区间
            for global_interval in global_intervals:
                for denom_interval in denom_intervals:
                    global_interval -= denom_interval

                if global_interval is not S.EmptySet:
                    intervals.append(global_interval)

            global_intervals = intervals

            if not global_intervals:
                break

        # 将所有区间并集到结果中
        for interval in global_intervals:
            result = result.union(interval)

    return result


# 简化具有有理系数的有理不等式系统
def reduce_rational_inequalities(exprs, gen, relational=True):
    """Reduce a system of rational inequalities with rational coefficients.

    Examples
    ========

    >>> from sympy import Symbol
    >>> from sympy.solvers.inequalities import reduce_rational_inequalities

    >>> x = Symbol('x', real=True)

    >>> reduce_rational_inequalities([[x**2 <= 0]], x)
    Eq(x, 0)

    >>> reduce_rational_inequalities([[x + 2 > 0]], x)
    -2 < x
    >>> reduce_rational_inequalities([[(x + 2, ">")]], x)
    -2 < x
    >>> reduce_rational_inequalities([[x + 2]], x)
    Eq(x, -2)
    """
    # 根据 relational 参数来简化有理不等式，返回简化后的表达式或条件
    if relational:
        return solve_rational_inequalities(exprs).as_relational(gen)
    else:
        return solve_rational_inequalities(exprs)
    This function finds the solution set for a system of rational inequalities involving symbols, accommodating extended real numbers if declared.
    >>> y = Symbol('y', extended_real=True)
    >>> reduce_rational_inequalities([[y + 2 > 0]], y)
    (-2 < y) & (y < oo)
    """
    exact = True  # Flag indicating if the solution is exact
    eqs = []  # List to store groups of equations
    solution = S.EmptySet  # Initialize an empty solution set

    # Iterate over each group of inequalities
    for _exprs in exprs:
        if not _exprs:
            continue
        
        _eqs = []  # List to store individual equations in the group
        _sol = S.Reals  # Start with the set of all real numbers for this group

        # Iterate over each inequality expression
        for expr in _exprs:
            if isinstance(expr, tuple):
                expr, rel = expr
            else:
                if expr.is_Relational:
                    expr, rel = expr.lhs - expr.rhs, expr.rel_op
                else:
                    rel = '=='

            # Simplify the expression and determine the numerator and denominator
            if expr is S.true:
                numer, denom, rel = S.Zero, S.One, '=='
            elif expr is S.false:
                numer, denom, rel = S.One, S.One, '=='
            else:
                numer, denom = expr.together().as_numer_denom()

            # Attempt to convert to polynomial form
            try:
                (numer, denom), opt = parallel_poly_from_expr((numer, denom), gen)
            except PolynomialError:
                raise PolynomialError(filldedent('''
                    only polynomials and rational functions are
                    supported in this context.
                    '''))

            # Check if the domain is exact and convert to exact if not
            if not opt.domain.is_Exact:
                numer, denom, exact = numer.to_exact(), denom.to_exact(), False

            domain = opt.domain.get_exact()

            # If domain is not ZZ or QQ, solve the inequality
            if not (domain.is_ZZ or domain.is_QQ):
                expr = numer / denom
                expr = Relational(expr, 0, rel)
                _sol &= solve_univariate_inequality(expr, gen, relational=False)
            else:
                _eqs.append(((numer, denom), rel))

        # If there are equations in _eqs, solve the rational inequalities
        if _eqs:
            _sol &= solve_rational_inequalities([_eqs])

            # Exclude certain solutions based on the equation's characteristics
            exclude = solve_rational_inequalities([[((d, d.one), '==')
                for i in eqs for ((n, d), _) in i if d.has(gen)]])
            _sol -= exclude

        # Aggregate the solutions from this group
        solution |= _sol

    # If solution is not exact and there is a non-empty solution set, evaluate numerically
    if not exact and solution:
        solution = solution.evalf()

    # Convert solution to relational form if required
    if relational:
        solution = solution.as_relational(gen)

    return solution
def reduce_abs_inequality(expr, rel, gen):
    """
    Reduce an inequality with nested absolute values.

    Examples
    ========

    >>> from sympy import reduce_abs_inequality, Abs, Symbol
    >>> x = Symbol('x', real=True)

    >>> reduce_abs_inequality(Abs(x - 5) - 3, '<', x)
    (2 < x) & (x < 8)

    >>> reduce_abs_inequality(Abs(x + 2)*3 - 13, '<', x)
    (-19/3 < x) & (x < 7/3)

    See Also
    ========

    reduce_abs_inequalities
    """
    if gen.is_extended_real is False:
        raise TypeError("""
            Cannot solve inequalities with absolute values containing
            non-real variables.
            """)

    def _bottom_up_scan(expr):
        exprs = []

        if expr.is_Add or expr.is_Mul:
            op = expr.func

            for arg in expr.args:
                _exprs = _bottom_up_scan(arg)

                if not exprs:
                    exprs = _exprs
                else:
                    exprs = [(op(expr, _expr), conds + _conds) for (expr, conds), (_expr, _conds) in
                            itertools.product(exprs, _exprs)]
        elif expr.is_Pow:
            n = expr.exp
            if not n.is_Integer:
                raise ValueError("Only Integer Powers are allowed on Abs.")

            exprs.extend((expr**n, conds) for expr, conds in _bottom_up_scan(expr.base))
        elif isinstance(expr, Abs):
            _exprs = _bottom_up_scan(expr.args[0])

            for expr, conds in _exprs:
                exprs.append((expr, conds + [Ge(expr, 0)]))
                exprs.append((-expr, conds + [Lt(expr, 0)]))
        else:
            exprs = [(expr, [])]

        return exprs

    mapping = {'<': '>', '<=': '>='}
    inequalities = []

    for expr, conds in _bottom_up_scan(expr):
        if rel not in mapping.keys():
            expr = Relational(expr, 0, rel)
        else:
            expr = Relational(-expr, 0, mapping[rel])

        inequalities.append([expr] + conds)

    return reduce_rational_inequalities(inequalities, gen)


def reduce_abs_inequalities(exprs, gen):
    """
    Reduce a system of inequalities with nested absolute values.

    Examples
    ========

    >>> from sympy import reduce_abs_inequalities, Abs, Symbol
    >>> x = Symbol('x', extended_real=True)

    >>> reduce_abs_inequalities([(Abs(3*x - 5) - 7, '<'),
    ... (Abs(x + 25) - 13, '>')], x)
    (-2/3 < x) & (x < 4) & (((-oo < x) & (x < -38)) | ((-12 < x) & (x < oo)))

    >>> reduce_abs_inequalities([(Abs(x - 4) + Abs(3*x - 5) - 7, '<')], x)
    (1/2 < x) & (x < 4)

    See Also
    ========

    reduce_abs_inequality
    """
    return And(*[reduce_abs_inequality(expr, rel, gen)
                 for expr, rel in exprs])


def solve_univariate_inequality(expr, gen, relational=True, domain=S.Reals, continuous=False):
    """
    Solves a real univariate inequality.

    Parameters
    ==========

    expr : Relational
        The target inequality
    gen : Symbol
        The variable for which the inequality is solved
    """
    relational : bool
        是否期望输出关系类型的解
    domain : Set
        方程解的定义域
    continuous: bool
        如果表达式在给定的定义域上是连续的，则为True
        （因此不需要在其上调用continuous_domain()）

    Raises
    ======

    NotImplementedError
        由于在:func:`sympy.solvers.solveset.solvify`中的限制，无法确定不等式的解。

    Notes
    =====

    目前，由于在:func:`sympy.solvers.solveset.solvify`中的限制，我们无法解决所有不等式。
    此外，对于三角不等式返回的解在其周期区间内受到限制。

    See Also
    ========

    sympy.solvers.solveset.solvify: 使用solve输出API返回solveset解的解算器

    Examples
    ========

    >>> from sympy import solve_univariate_inequality, Symbol, sin, Interval, S
    >>> x = Symbol('x')

    >>> solve_univariate_inequality(x**2 >= 4, x)
    ((2 <= x) & (x < oo)) | ((-oo < x) & (x <= -2))

    >>> solve_univariate_inequality(x**2 >= 4, x, relational=False)
    Union(Interval(-oo, -2), Interval(2, oo))

    >>> domain = Interval(0, S.Infinity)
    >>> solve_univariate_inequality(x**2 >= 4, x, False, domain)
    Interval(2, oo)

    >>> solve_univariate_inequality(sin(x) > 0, x, relational=False)
    Interval.open(0, pi)

    """
    from sympy.solvers.solvers import denoms

    # 检查定义域是否为实数集，如果不是则抛出错误
    if domain.is_subset(S.Reals) is False:
        raise NotImplementedError(filldedent('''
        复数域中的不等式不受支持。请通过设置domain=S.Reals尝试实数域'''))
    # 如果定义域不是实数集，则求解不等式并取其与定义域的交集
    elif domain is not S.Reals:
        rv = solve_univariate_inequality(
            expr, gen, relational=False, continuous=continuous).intersection(domain)
        # 如果需要输出关系类型的解，则转换为关系形式
        if relational:
            rv = rv.as_relational(gen)
        return rv
    else:
        pass  # 在实数域中继续尝试解决

    # 保持函数独立于关于`gen`的假设。
    # `solveset`确保仅在定义域为实数时调用此函数。
    _gen = gen
    _domain = domain
    # 如果`gen`不是扩展实数，则返回空集
    if gen.is_extended_real is False:
        rv = S.EmptySet
        return rv if not relational else rv.as_relational(_gen)
    # 如果`gen`的扩展实数性质未知，则将其视为扩展实数并尝试替换表达式
    elif gen.is_extended_real is None:
        gen = Dummy('gen', extended_real=True)
        try:
            expr = expr.xreplace({_gen: gen})
        except TypeError:
            raise TypeError(filldedent('''
                当`gen`为实数时，关系有复数部分导致类似I < 0的无效比较。
                '''))

    rv = None

    # 如果表达式为真，则解为整个定义域
    if expr is S.true:
        rv = domain

    # 如果表达式为假，则解为空集
    elif expr is S.false:
        rv = S.EmptySet

    return rv if not relational else rv.as_relational(_gen)
# 返回介于起点和终点之间的中间点
def _pt(start, end):
    if not start.is_infinite and not end.is_infinite:
        # 如果起点和终点都不是无穷大，则计算它们的中点
        pt = (start + end)/2
    elif start.is_infinite and end.is_infinite:
        # 如果起点和终点都是无穷大，则返回零
        pt = S.Zero
    else:
        if (start.is_infinite and start.is_extended_positive is None or
                end.is_infinite and end.is_extended_positive is None):
            # 如果起点或终点是无穷大且符号未知，则抛出错误
            raise ValueError('cannot proceed with unsigned infinite values')
        if (end.is_infinite and end.is_extended_negative or
                start.is_infinite and start.is_extended_positive):
            # 如果终点为负无穷大，起点为正无穷大，则交换它们
            start, end = end, start
        # 尝试使用起点或终点的倍数，以获得更好的行为
        # 在检查假设时比添加或减去1得到的表达式更好
        if end.is_infinite:
            if start.is_extended_positive:
                # 如果起点为正无穷大，则中点为起点的两倍
                pt = start*2
            elif start.is_extended_negative:
                # 如果起点为负无穷大，则中点为起点的一半
                pt = start*S.Half
            else:
                # 否则，中点为起点加1
                pt = start + 1
        elif start.is_infinite:
            if end.is_extended_positive:
                # 如果终点为正无穷大，则中点为终点的一半
                pt = end*S.Half
            elif end.is_extended_negative:
                # 如果终点为负无穷大，则中点为终点的两倍
                pt = end*2
            else:
                # 否则，中点为终点减1
                pt = end - 1
    return pt


# 解决不等式，将变量 s 在左侧隔离（如果可能）
def _solve_inequality(ie, s, linear=False):
    """Return the inequality with s isolated on the left, if possible.
    If the relationship is non-linear, a solution involving And or Or
    may be returned. False or True are returned if the relationship
    is never True or always True, respectively.

    If `linear` is True (default is False) an `s`-dependent expression
    will be isolated on the left, if possible
    but it will not be solved for `s` unless the expression is linear
    in `s`. Furthermore, only "safe" operations which do not change the
    sense of the relationship are applied: no division by an unsigned
    value is attempted unless the relationship involves Eq or Ne and
    no division by a value not known to be nonzero is ever attempted.

    Examples
    ========

    >>> from sympy import Eq, Symbol
    >>> from sympy.solvers.inequalities import _solve_inequality as f
    >>> from sympy.abc import x, y

    For linear expressions, the symbol can be isolated:

    >>> f(x - 2 < 0, x)
    x < 2
    >>> f(-x - 6 < x, x)
    x > -3

    Sometimes nonlinear relationships will be False

    >>> f(x**2 + 4 < 0, x)
    False

    Or they may involve more than one region of values:

    >>> f(x**2 - 4 < 0, x)
    (-2 < x) & (x < 2)

    To restrict the solution to a relational, set linear=True
    and only the x-dependent portion will be isolated on the left:

    >>> f(x**2 - 4 < 0, x, linear=True)
    x**2 < 4

    Division of only nonzero quantities is allowed, so x cannot
    be isolated by dividing by y:

    >>> y.is_nonzero is None  # it is unknown whether it is 0 or not
    True
    >>> f(x*y < 1, x)
    x*y < 1

    And while an equality (or inequality) still holds after dividing by a
    """
    # 在不等式中将变量 s 隔离在左侧（如果可能）
    # 如果关系是非线性的，则可能返回涉及 And 或 Or 的解决方案
    # 如果关系永远不为真，则返回 False；如果关系总是为真，则返回 True
    # 如果 linear 参数为 True（默认为 False），将尝试将依赖于 s 的表达式隔离在左侧
    # 但除非表达式在 s 中是线性的，否则不会对 s 进行求解
    # 只应用“安全”的操作，不改变关系的意义：除非关系涉及 Eq 或 Ne，否则不会尝试除以无符号值
    # 除非已知其非零，否则不会尝试除以未知为零或非零的值
    non-zero quantity

    >>> nz = Symbol('nz', nonzero=True)
    >>> f(Eq(x*nz, 1), x)
    Eq(x, 1/nz)

    the sign must be known for other inequalities involving > or <:

    >>> f(x*nz <= 1, x)
    nz*x <= 1
    >>> p = Symbol('p', positive=True)
    >>> f(x*p <= 1, x)
    x <= 1/p

    When there are denominators in the original expression that
    are removed by expansion, conditions for them will be returned
    as part of the result:

    >>> f(x < x*(2/x - 1), x)
    (x < 1) & Ne(x, 0)
    """
    from sympy.solvers.solvers import denoms
    # 如果 s 不在不等式表达式 ie 的自由符号中，则直接返回 ie
    if s not in ie.free_symbols:
        return ie
    # 如果 ie 的右侧等于 s，则将不等式 ie 翻转
    if ie.rhs == s:
        ie = ie.reversed
    # 如果 ie 的左侧等于 s，并且 s 不在 ie 的右侧自由符号中，则返回 ie
    if ie.lhs == s and s not in ie.rhs.free_symbols:
        return ie

    def classify(ie, s, i):
        # 用符号 s 替换为 i 后，评估不等式 ie 的真假性：
        # 如果评估结果是 True 或 False，则返回相应的值；
        # 如果未评估，则返回 None；
        # 如果评估出错，则返回 S.NaN。
        try:
            v = ie.subs(s, i)
            if v is S.NaN:
                return v
            elif v not in (True, False):
                return
            return v
        except TypeError:
            return S.NaN

    rv = None
    oo = S.Infinity
    expr = ie.lhs - ie.rhs
    try:
        p = Poly(expr, s)
        if p.degree() == 0:
            rv = ie.func(p.as_expr(), 0)
        elif not linear and p.degree() > 1:
            # 在 except 子句中处理
            raise NotImplementedError
    except (PolynomialError, NotImplementedError):
        if not linear:
            try:
                rv = reduce_rational_inequalities([[ie]], s)
            except PolynomialError:
                rv = solve_univariate_inequality(ie, s)
            # 移除使用集合简化关系时可能应用的对无穷大的限制
            okoo = classify(ie, s, oo)
            if okoo is S.true and classify(rv, s, oo) is S.false:
                rv = rv.subs(s < oo, True)
            oknoo = classify(ie, s, -oo)
            if (oknoo is S.true and
                    classify(rv, s, -oo) is S.false):
                rv = rv.subs(-oo < s, True)
                rv = rv.subs(s > -oo, True)
            if rv is S.true:
                rv = (s <= oo) if okoo is S.true else (s < oo)
                if oknoo is not S.true:
                    rv = And(-oo < s, rv)
        else:
            p = Poly(expr)
    
    conds = []
    if rv is None:
        # 将 p 表达式转换为展开形式，并赋给 e
        e = p.as_expr()  # this is in expanded form
        
        # 进行安全的反转 e 操作，将非 s 项移到右边，如果关系是 Eq/Ne，则除以非零因子
        # 对于其他关系，符号必须是正或负
        rhs = 0
        b, ax = e.as_independent(s, as_Add=True)
        e -= b
        rhs -= b
        
        # 对 e 进行因式分解
        ef = factor_terms(e)
        a, e = ef.as_independent(s, as_Add=False)
        
        # 如果 a 是非零且符号未知，则 e 不变，a 设置为 1
        if (a.is_zero != False or  # don't divide by potential 0
                a.is_negative ==
                a.is_positive is None and  # if sign is not known then
                ie.rel_op not in ('!=', '==')): # reject if not Eq/Ne
            e = ef
            a = S.One
        
        # 对 rhs 进行除法操作
        rhs /= a
        
        # 如果 a 是正数，则 rv 是 e 和 rhs 的函数
        if a.is_positive:
            rv = ie.func(e, rhs)
        else:
            rv = ie.reversed.func(e, rhs)
        
        # 返回值的有效条件
        beginning_denoms = denoms(ie.lhs) | denoms(ie.rhs)
        current_denoms = denoms(rv)
        
        # 检查初值表达式与当前表达式之间的分母
        for d in beginning_denoms - current_denoms:
            c = _solve_inequality(Eq(d, 0), s, linear=linear)
            if isinstance(c, Eq) and c.lhs == s:
                if classify(rv, s, c.rhs) is S.true:
                    # rv 允许这个值，但不应该允许
                    conds.append(~c)
        
        # 检查 (-oo, oo) 范围内的值
        for i in (-oo, oo):
            if (classify(rv, s, i) is S.true and
                    classify(ie, s, i) is not S.true):
                conds.append(s < i if i is oo else i < s)
    
    # 将 rv 加入 conds 列表中
    conds.append(rv)
    
    # 返回条件的 And 结果
    return And(*conds)
# 为 reduce_inequalities 函数提供辅助功能，用于减少不等式系统
def _reduce_inequalities(inequalities, symbols):
    # 初始化多项式部分和绝对值部分的空字典
    poly_part, abs_part = {}, {}
    # 其他类型不等式的列表
    other = []

    # 遍历输入的不等式列表
    for inequality in inequalities:
        # 获取不等式的左手边表达式和关系操作符
        expr, rel = inequality.lhs, inequality.rel_op  # rhs is 0

        # 检查表达式中的符号变量，使用 atoms 方法比 free_symbols 方法更严格，
        # 防止处理无法被 reduce_rational_inequalities 处理的 EX 域
        gens = expr.atoms(Symbol)

        # 如果符号变量数量为1，则将该符号变量作为 gen
        if len(gens) == 1:
            gen = gens.pop()
        else:
            # 否则，找出表达式中与 symbols 的交集
            common = expr.free_symbols & symbols
            # 如果交集中只有一个符号变量，则将其作为 gen 处理
            if len(common) == 1:
                gen = common.pop()
                # 将求解后的不等式结果添加到其他类型不等式列表
                other.append(_solve_inequality(Relational(expr, 0, rel), gen))
                continue
            else:
                # 如果交集中有多个符号变量，则抛出未实现错误
                raise NotImplementedError(filldedent('''
                    inequality has more than one symbol of interest.
                    '''))

        # 如果表达式是关于 gen 的多项式，则将其添加到多项式部分字典中
        if expr.is_polynomial(gen):
            poly_part.setdefault(gen, []).append((expr, rel))
        else:
            # 否则，查找表达式中包含 gen 的函数或幂函数部分
            components = expr.find(lambda u:
                u.has(gen) and (
                u.is_Function or u.is_Pow and not u.exp.is_Integer))
            # 如果找到且所有部分都是绝对值函数 Abs，则将其添加到绝对值部分字典中
            if components and all(isinstance(i, Abs) for i in components):
                abs_part.setdefault(gen, []).append((expr, rel))
            else:
                # 否则，将求解后的不等式结果添加到其他类型不等式列表
                other.append(_solve_inequality(Relational(expr, 0, rel), gen))

    # 对多项式部分进行有理不等式化简
    poly_reduced = [reduce_rational_inequalities([exprs], gen) for gen, exprs in poly_part.items()]
    # 对绝对值部分进行绝对值不等式化简
    abs_reduced = [reduce_abs_inequalities(exprs, gen) for gen, exprs in abs_part.items()]

    # 返回多项式部分、绝对值部分和其他类型不等式的合取结果
    return And(*(poly_reduced + abs_reduced + other))


# 减少具有有理系数的不等式系统的函数
def reduce_inequalities(inequalities, symbols=[]):
    """Reduce a system of inequalities with rational coefficients.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy import reduce_inequalities

    >>> reduce_inequalities(0 <= x + 3, [])
    (-3 <= x) & (x < oo)

    >>> reduce_inequalities(0 <= x + y*2 - 1, [x])
    (x < oo) & (x >= 1 - 2*y)
    """
    # 如果不等式列表不是可迭代对象，则转换为列表
    if not iterable(inequalities):
        inequalities = [inequalities]
    # 将不等式列表中的每个元素转换为 sympy 的表达式对象
    inequalities = [sympify(i) for i in inequalities]

    # 获取所有不等式中涉及的符号变量集合
    gens = set().union(*[i.free_symbols for i in inequalities])

    # 如果 symbols 参数不是可迭代对象，则转换为列表
    if not iterable(symbols):
        symbols = [symbols]
    # 将 symbols 中的符号变量与 gens 的交集作为最终符号变量集合
    symbols = (set(symbols) or gens) & gens
    # 如果符号变量中存在非扩展实数类型的变量，则抛出类型错误
    if any(i.is_extended_real is False for i in symbols):
        raise TypeError(filldedent('''
            inequalities cannot contain symbols that are not real.
            '''))

    # 将 gens 中不明确类型的符号变量重新定义为扩展实数类型的虚拟符号变量
    recast = {i: Dummy(i.name, extended_real=True)
        for i in gens if i.is_extended_real is None}
    # 替换不等式列表中的符号变量为重新定义的虚拟符号变量
    inequalities = [i.xreplace(recast) for i in inequalities]
    # 将 symbols 中的符号变量也替换为重新定义的虚拟符号变量
    symbols = {i.xreplace(recast) for i in symbols}

    # 预处理步骤，暂时保留
    keep = []

    # 返回多项式部分、绝对值部分和其他类型不等式的合取结果
    return _reduce_inequalities(inequalities, symbols)
    # 遍历不等式列表中的每一个不等式
    for i in inequalities:
        # 检查当前不等式是否为 Relational 类型
        if isinstance(i, Relational):
            # 将不等式转换为形式 i.lhs - i.rhs == 0 的关系表达式
            i = i.func(i.lhs.as_expr() - i.rhs.as_expr(), 0)
        # 如果不是 Relational 类型且不是 True 或 False，则将其转换为等式 i == 0
        elif i not in (True, False):
            i = Eq(i, 0)
        # 如果当前不等式为 True，则跳过当前循环
        if i == True:
            continue
        # 如果当前不等式为 False，则直接返回逻辑假值
        elif i == False:
            return S.false
        # 如果不等式左侧为数值，则抛出未实现的错误
        if i.lhs.is_number:
            raise NotImplementedError(
                "could not determine truth value of %s" % i)
        # 将当前处理后的不等式加入保留列表中
        keep.append(i)
    
    # 更新不等式列表为处理后的保留列表
    inequalities = keep
    # 删除保留列表
    del keep

    # 解决简化后的不等式系统
    rv = _reduce_inequalities(inequalities, symbols)

    # 恢复原始符号映射并返回结果
    return rv.xreplace({v: k for k, v in recast.items()})
```