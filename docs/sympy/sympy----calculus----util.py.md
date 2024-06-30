# `D:\src\scipysrc\sympy\sympy\calculus\util.py`

```
# 导入 AccumBounds 和 AccumulationBounds 类，用于误差累积计算
# noqa: F401 是告诉 linter 忽略未使用的导入警告
from .accumulationbounds import AccumBounds, AccumulationBounds # noqa: F401
# 导入 singularities 模块，用于处理函数的奇点
from .singularities import singularities
# 导入 sympy 库中的各种核心模块和函数
from sympy.core import Pow, S
# 导入 sympy 中核心函数的具体实现，如微分、展开、函数等
from sympy.core.function import diff, expand_mul, Function
# 导入 sympy 中数值类型的定义，如整数、有理数、无理数等
from sympy.core.kind import NumberKind
# 导入 sympy 中模数运算相关的类和函数
from sympy.core.mod import Mod
# 导入 sympy 中数值比较相关的函数
from sympy.core.numbers import equal_valued
# 导入 sympy 中关系比较相关的类和函数
from sympy.core.relational import Relational
# 导入 sympy 中符号和虚拟符号的定义
from sympy.core.symbol import Symbol, Dummy
# 导入 sympy 中将字符串转换为表达式的函数
from sympy.core.sympify import _sympify
# 导入 sympy 中处理复数的函数，如绝对值、实部、虚部
from sympy.functions.elementary.complexes import Abs, im, re
# 导入 sympy 中处理指数函数的函数，如指数、对数
from sympy.functions.elementary.exponential import exp, log
# 导入 sympy 中处理整数函数的函数，如取分数部分
from sympy.functions.elementary.integers import frac
# 导入 sympy 中处理分段函数的函数
from sympy.functions.elementary.piecewise import Piecewise
# 导入 sympy 中处理三角函数的函数
from sympy.functions.elementary.trigonometric import (
    TrigonometricFunction, sin, cos, tan, cot, csc, sec,
    asin, acos, acot, atan, asec, acsc)
# 导入 sympy 中处理双曲函数的函数
from sympy.functions.elementary.hyperbolic import (sinh, cosh, tanh, coth,
    sech, csch, asinh, acosh, atanh, acoth, asech, acsch)
# 导入 sympy 中多项式工具的函数，如求多项式的次数、最小公倍数
from sympy.polys.polytools import degree, lcm_list
# 导入 sympy 中集合操作相关的类，如区间、交集、并集、补集
from sympy.sets.sets import (Interval, Intersection, FiniteSet, Union,
                             Complement)
# 导入 sympy 中处理映射集合的函数
from sympy.sets.fancysets import ImageSet
# 导入 sympy 中条件集合的定义和处理函数
from sympy.sets.conditionset import ConditionSet
# 导入 sympy 中处理字符串格式的工具函数
from sympy.utilities import filldedent
# 导入 sympy 中用于检查对象是否可迭代的函数
from sympy.utilities.iterables import iterable
# 导入 sympy 中处理稠密矩阵的函数
from sympy.matrices.dense import hessian

# 定义一个函数，用于确定函数表达式 f 在符号 symbol 上连续的定义域
def continuous_domain(f, symbol, domain):
    """
    Returns the domain on which the function expression f is continuous.

    This function is limited by the ability to determine the various
    singularities and discontinuities of the given function.
    The result is either given as a union of intervals or constructed using
    other set operations.

    Parameters
    ==========

    f : :py:class:`~.Expr`
        The concerned function.
    symbol : :py:class:`~.Symbol`
        The variable for which the intervals are to be determined.
    domain : :py:class:`~.Interval`
        The domain over which the continuity of the symbol has to be checked.

    Examples
    ========

    >>> from sympy import Interval, Symbol, S, tan, log, pi, sqrt
    >>> from sympy.calculus.util import continuous_domain
    >>> x = Symbol('x')
    >>> continuous_domain(1/x, x, S.Reals)
    Union(Interval.open(-oo, 0), Interval.open(0, oo))
    >>> continuous_domain(tan(x), x, Interval(0, pi))
    Union(Interval.Ropen(0, pi/2), Interval.Lopen(pi/2, pi))
    >>> continuous_domain(sqrt(x - 2), x, Interval(-5, 5))
    Interval(2, 5)
    >>> continuous_domain(log(2*x - 1), x, S.Reals)
    Interval.open(1/2, oo)

    Returns
    =======

    :py:class:`~.Interval`
        Union of all intervals where the function is continuous.

    Raises
    ======

    NotImplementedError
        If the method to determine continuity of such a function
        has not yet been developed.

    """
    from sympy.solvers.inequalities import solve_univariate_inequality
    # 检查给定的域是否是实数集 S.Reals 的子集，如果不是则抛出未实现错误
    if not domain.is_subset(S.Reals):
        raise NotImplementedError(filldedent('''
            Domain must be a subset of S.Reals.
            '''))

    # 实现了的数学函数列表，用于检查函数是否实现
    implemented = [Pow, exp, log, Abs, frac,
                   sin, cos, tan, cot, sec, csc,
                   asin, acos, atan, acot, asec, acsc,
                   sinh, cosh, tanh, coth, sech, csch,
                   asinh, acosh, atanh, acoth, asech, acsch]
    
    # 使用到的数学函数列表，从给定函数 f 中提取
    used = [fct.func for fct in f.atoms(Function) if fct.has(symbol)]
    
    # 如果存在未实现的函数，则抛出未实现错误
    if any(func not in implemented for func in used):
        raise NotImplementedError(filldedent('''
            Unable to determine the domain of the given function.
            '''))

    # 创建符号变量 x
    x = Symbol('x')
    
    # 定义数学函数的定义域约束条件
    constraints = {
        log: (x > 0,),
        asin: (x >= -1, x <= 1),
        acos: (x >= -1, x <= 1),
        acosh: (x >= 1,),
        atanh: (x > -1, x < 1),
        asech: (x > 0, x <= 1)
    }
    
    # 定义部分数学函数的额外定义域约束条件
    constraints_union = {
        asec: (x <= -1, x >= 1),
        acsc: (x <= -1, x >= 1),
        acoth: (x < -1, x > 1)
    }

    # 将 domain 赋值给 cont_domain，用于后续计算
    cont_domain = domain
    
    # 遍历 f 中的幂函数（Pow）
    for atom in f.atoms(Pow):
        # 获取幂函数的分母
        den = atom.exp.as_numer_denom()[1]
        
        # 如果指数是有理数且分母为奇数，则忽略（0**负数由 singularities() 处理）
        if atom.exp.is_rational and den.is_odd:
            pass
        else:
            # 解决幂函数的单变量不等式约束，并将结果取交集更新 cont_domain
            constraint = solve_univariate_inequality(atom.base >= 0,
                                                        symbol).as_set()
            cont_domain = Intersection(constraint, cont_domain)
    # 遍历函数 f 中所有类型为 Function 的原子(atom)
    for atom in f.atoms(Function):
        # 如果原子(atom)的函数在 constraints 中
        if atom.func in constraints:
            # 遍历 constraints[atom.func] 中的约束条件
            for c in constraints[atom.func]:
                # 将约束条件中的变量 x 替换为 atom.args[0]，得到约束关系
                constraint_relational = c.subs(x, atom.args[0])
                # 解决一元不等式约束关系 constraint_relational，转换为集合形式
                constraint_set = solve_univariate_inequality(
                    constraint_relational, symbol).as_set()
                # 将 constraint_set 与当前的 cont_domain 取交集
                cont_domain = Intersection(constraint_set, cont_domain)
        # 如果原子(atom)的函数在 constraints_union 中
        elif atom.func in constraints_union:
            # 初始化一个空的约束集合 constraint_set
            constraint_set = S.EmptySet
            # 遍历 constraints_union[atom.func] 中的约束条件
            for c in constraints_union[atom.func]:
                # 将约束条件中的变量 x 替换为 atom.args[0]，得到约束关系
                constraint_relational = c.subs(x, atom.args[0])
                # 解决一元不等式约束关系 constraint_relational，转换为集合形式
                constraint_set += solve_univariate_inequality(
                    constraint_relational, symbol).as_set()
            # 将 constraint_set 与当前的 cont_domain 取交集
            cont_domain = Intersection(constraint_set, cont_domain)
        # 如果原子(atom)的函数是 acot
        elif atom.func == acot:
            # 导入 solveset_real 函数用于求解实数解集
            from sympy.solvers.solveset import solveset_real
            # Sympy 中的 acot() 在 0 处有一个阶跃不连续点。
            # 尽管它既不是本质奇点也不是极点，但 singularities() 不会报告它。
            # 但是这对于确定函数 f 的连续性仍然很重要。
            # 从 cont_domain 中减去 acot(atom.args[0]) 的实数解集
            cont_domain -= solveset_real(atom.args[0], symbol)
            # 注意，上述操作可能引入虚假的不连续点，比如 abs(acot(x)) 在 0 处。
        # 如果原子(atom)的函数是 frac
        elif atom.func == frac:
            # 导入 solveset_real 函数用于求解实数解集
            from sympy.solvers.solveset import solveset_real
            # 计算函数 atom.args[0] 在给定 domain 下的值域范围 r
            r = function_range(atom.args[0], symbol, domain)
            # 将 r 与整数集合 S.Integers 取交集
            r = Intersection(r, S.Integers)
            # 如果 r 是有限集
            if r.is_finite_set:
                # 初始化一个空的不连续点集合 discont
                discont = S.EmptySet
                # 遍历 r 中的每一个整数 n
                for n in r:
                    # 将 solveset_real(atom.args[0]-n, symbol) 添加到 discont 中
                    discont += solveset_real(atom.args[0]-n, symbol)
            else:
                # 如果 r 是无限集，则定义一个条件集合
                discont = ConditionSet(
                    symbol, S.Integers.contains(atom.args[0]), cont_domain)
            # 从 cont_domain 中减去 discont
            cont_domain -= discont

    # 返回 cont_domain 减去函数 f 在给定 symbol 和 domain 下的奇点集合
    return cont_domain - singularities(f, symbol, domain)
def function_range(f, symbol, domain):
    """
    Finds the range of a function in a given domain.
    This method is limited by the ability to determine the singularities and
    determine limits.

    Parameters
    ==========

    f : :py:class:`~.Expr`
        The concerned function.
    symbol : :py:class:`~.Symbol`
        The variable for which the range of function is to be determined.
    domain : :py:class:`~.Interval`
        The domain under which the range of the function has to be found.

    Examples
    ========

    >>> from sympy import Interval, Symbol, S, exp, log, pi, sqrt, sin, tan
    >>> from sympy.calculus.util import function_range
    >>> x = Symbol('x')
    >>> function_range(sin(x), x, Interval(0, 2*pi))
    Interval(-1, 1)
    >>> function_range(tan(x), x, Interval(-pi/2, pi/2))
    Interval(-oo, oo)
    >>> function_range(1/x, x, S.Reals)
    Union(Interval.open(-oo, 0), Interval.open(0, oo))
    >>> function_range(exp(x), x, S.Reals)
    Interval.open(0, oo)
    >>> function_range(log(x), x, S.Reals)
    Interval(-oo, oo)
    >>> function_range(sqrt(x), x, Interval(-5, 9))
    Interval(0, 3)

    Returns
    =======

    :py:class:`~.Interval`
        Union of all ranges for all intervals under domain where function is
        continuous.

    Raises
    ======

    NotImplementedError
        If any of the intervals, in the given domain, for which function
        is continuous are not finite or real,
        OR if the critical points of the function on the domain cannot be found.
    """

    # If the domain is empty, return an empty set
    if domain is S.EmptySet:
        return S.EmptySet

    # Determine the periodicity of the function
    period = periodicity(f, symbol)

    # If the function is constant with respect to the symbol, return its expanded form
    if period == S.Zero:
        return FiniteSet(f.expand())

    # Import necessary functions for limits and solving equations
    from sympy.series.limits import limit
    from sympy.solvers.solveset import solveset

    # Adjust the domain if the function has a periodicity
    if period is not None:
        if isinstance(domain, Interval):
            if (domain.inf - domain.sup).is_infinite:
                domain = Interval(0, period)
        elif isinstance(domain, Union):
            for sub_dom in domain.args:
                if isinstance(sub_dom, Interval) and \
                ((sub_dom.inf - sub_dom.sup).is_infinite):
                    domain = Interval(0, period)

    # Find intervals where the function is continuous within the domain
    intervals = continuous_domain(f, symbol, domain)
    range_int = S.EmptySet

    # Determine the type of intervals and iterate accordingly
    if isinstance(intervals, (Interval, FiniteSet)):
        interval_iter = (intervals,)
    elif isinstance(intervals, Union):
        interval_iter = intervals.args
    else:
        # Raise an error if unable to find the range for the given domain
        raise NotImplementedError("""
            Unable to find range for the given domain.
            """)
    # 遍历给定的区间迭代器，逐个处理每个区间
    for interval in interval_iter:
        # 检查当前区间是否为有限集合类型
        if isinstance(interval, FiniteSet):
            # 遍历有限集合中的每个单元素
            for singleton in interval:
                # 如果单元素在定义域内，则将其对应的函数值添加到范围集合中
                if singleton in domain:
                    range_int += FiniteSet(f.subs(symbol, singleton))
        
        # 如果当前区间是区间（Interval）类型
        elif isinstance(interval, Interval):
            # 初始化集合和变量用于存储临界值、临界点和临界值
            vals = S.EmptySet
            critical_points = S.EmptySet
            critical_values = S.EmptySet
            # 设定区间边界和其对应的极限点以及方向
            bounds = ((interval.left_open, interval.inf, '+'),
                      (interval.right_open, interval.sup, '-'))

            # 遍历区间的左右边界
            for is_open, limit_point, direction in bounds:
                # 如果边界是开放的，则计算对应方向上的极限值，并加入临界值集合
                if is_open:
                    critical_values += FiniteSet(limit(f, symbol, limit_point))
                    vals += critical_values
                # 如果边界是闭合的，则直接将边界点对应的函数值加入集合
                else:
                    vals += FiniteSet(f.subs(symbol, limit_point))

            # 解方程 f'(symbol) = 0 来获取临界点
            solution = solveset(f.diff(symbol), symbol, interval)

            # 如果解不可迭代，抛出未实现错误
            if not iterable(solution):
                raise NotImplementedError(
                    'Unable to find critical points for {}'.format(f))
            # 如果解是图像集合类型，抛出未实现错误
            if isinstance(solution, ImageSet):
                raise NotImplementedError(
                    'Infinite number of critical points for {}'.format(f))

            # 将解集合中的临界点添加到临界点集合中
            critical_points += solution

            # 将每个临界点对应的函数值添加到集合中
            for critical_point in critical_points:
                vals += FiniteSet(f.subs(symbol, critical_point))

            # 初始化左右边界是否开放的标志
            left_open, right_open = False, False

            # 如果临界值集合不为空，则检查范围集合的最小值和最大值是否与临界值集合的最小值和最大值相等
            if critical_values is not S.EmptySet:
                if critical_values.inf == vals.inf:
                    left_open = True
                if critical_values.sup == vals.sup:
                    right_open = True

            # 将当前区间的计算出的值范围添加到总的范围集合中
            range_int += Interval(vals.inf, vals.sup, left_open, right_open)

        # 如果当前区间类型不是有限集合也不是区间，则抛出未实现错误
        else:
            raise NotImplementedError(filldedent('''
                Unable to find range for the given domain.
                '''))

    # 返回计算得到的总的范围集合
    return range_int
# 定义函数 `not_empty_in`，用于找到在给定 `finset_intersection` 中函数定义域非空的部分
def not_empty_in(finset_intersection, *syms):
    """
    Finds the domain of the functions in ``finset_intersection`` in which the
    ``finite_set`` is not-empty.

    Parameters
    ==========

    finset_intersection : Intersection of FiniteSet
        The unevaluated intersection of FiniteSet containing
        real-valued functions with Union of Sets
    syms : Tuple of symbols
        Symbol for which domain is to be found

    Raises
    ======

    NotImplementedError
        The algorithms to find the non-emptiness of the given FiniteSet are
        not yet implemented.
    ValueError
        The input is not valid.
    RuntimeError
        It is a bug, please report it to the github issue tracker
        (https://github.com/sympy/sympy/issues).

    Examples
    ========

    >>> from sympy import FiniteSet, Interval, not_empty_in, oo
    >>> from sympy.abc import x
    >>> not_empty_in(FiniteSet(x/2).intersect(Interval(0, 1)), x)
    Interval(0, 2)
    >>> not_empty_in(FiniteSet(x, x**2).intersect(Interval(1, 2)), x)
    Union(Interval(1, 2), Interval(-sqrt(2), -1))
    >>> not_empty_in(FiniteSet(x**2/(x + 2)).intersect(Interval(1, oo)), x)
    Union(Interval.Lopen(-2, -1), Interval(2, oo))
    """

    # TODO: handle piecewise defined functions
    # TODO: handle transcendental functions
    # TODO: handle multivariate functions

    # 如果未提供任何符号，则抛出 ValueError 异常
    if len(syms) == 0:
        raise ValueError("One or more symbols must be given in syms.")

    # 如果 finset_intersection 是空集，则返回空集
    if finset_intersection is S.EmptySet:
        return S.EmptySet

    # 如果 finset_intersection 是 Union 类型，则递归处理并返回 Union 结果
    if isinstance(finset_intersection, Union):
        elm_in_sets = finset_intersection.args[0]
        return Union(not_empty_in(finset_intersection.args[1], *syms),
                     elm_in_sets)

    # 如果 finset_intersection 是 FiniteSet 类型，则处理 finite_set 和 _sets
    if isinstance(finset_intersection, FiniteSet):
        finite_set = finset_intersection
        _sets = S.Reals
    else:
        finite_set = finset_intersection.args[1]
        _sets = finset_intersection.args[0]

    # 如果 finite_set 不是 FiniteSet 类型，则抛出 ValueError 异常
    if not isinstance(finite_set, FiniteSet):
        raise ValueError('A FiniteSet must be given, not %s: %s' %
                         (type(finite_set), finite_set))

    # 如果提供的符号数量不是 1，则抛出 NotImplementedError 异常
    if len(syms) == 1:
        symb = syms[0]
    else:
        raise NotImplementedError('more than one variables %s not handled' %
                                  (syms,))
    # 定义函数 elm_domain，用于计算表达式在指定区间内的定义域
    def elm_domain(expr, intrvl):
        """ Finds the domain of an expression in any given interval """
        # 导入 solveset 函数，用于求解方程
        from sympy.solvers.solveset import solveset

        # 获取区间的起始和结束值
        _start = intrvl.start
        _end = intrvl.end
        # 解析表达式的分母，找出可能的奇点（分母为零的点）
        _singularities = solveset(expr.as_numer_denom()[1], symb,
                                  domain=S.Reals)

        # 处理右开区间的情况
        if intrvl.right_open:
            if _end is S.Infinity:
                _domain1 = S.Reals  # 区间末尾为正无穷时，定义域为整个实数集
            else:
                _domain1 = solveset(expr < _end, symb, domain=S.Reals)
        else:
            _domain1 = solveset(expr <= _end, symb, domain=S.Reals)

        # 处理左开区间的情况
        if intrvl.left_open:
            if _start is S.NegativeInfinity:
                _domain2 = S.Reals  # 区间起始为负无穷时，定义域为整个实数集
            else:
                _domain2 = solveset(expr > _start, symb, domain=S.Reals)
        else:
            _domain2 = solveset(expr >= _start, symb, domain=S.Reals)

        # 计算表达式在区间内的定义域，去除奇点的影响
        expr_with_sing = Intersection(_domain1, _domain2)
        expr_domain = Complement(expr_with_sing, _singularities)
        return expr_domain

    # 如果 _sets 是一个 Interval 类型的对象
    if isinstance(_sets, Interval):
        # 对有限集合中的每个元素，计算其在指定区间内的定义域，并返回它们的并集
        return Union(*[elm_domain(element, _sets) for element in finite_set])

    # 如果 _sets 是一个 Union 类型的对象
    if isinstance(_sets, Union):
        # 初始化一个空集
        _domain = S.EmptySet
        # 对于 Union 对象中的每个区间 intrvl，计算有限集合中每个元素在该区间内的定义域，最后返回所有区间的并集
        for intrvl in _sets.args:
            _domain_element = Union(*[elm_domain(element, intrvl)
                                      for element in finite_set])
            _domain = Union(_domain, _domain_element)
        return _domain
# 检测给定函数在指定符号上的周期性。

def periodicity(f, symbol, check=False):
    """
    Tests the given function for periodicity in the given symbol.

    Parameters
    ==========

    f : :py:class:`~.Expr`
        The concerned function.
    symbol : :py:class:`~.Symbol`
        The variable for which the period is to be determined.
    check : bool, optional
        The flag to verify whether the value being returned is a period or not.

    Returns
    =======

    period
        The period of the function is returned.
        ``None`` is returned when the function is aperiodic or has a complex period.
        The value of $0$ is returned as the period of a constant function.

    Raises
    ======

    NotImplementedError
        The value of the period computed cannot be verified.


    Notes
    =====

    Currently, we do not support functions with a complex period.
    The period of functions having complex periodic values such
    as ``exp``, ``sinh`` is evaluated to ``None``.

    The value returned might not be the "fundamental" period of the given
    function i.e. it may not be the smallest periodic value of the function.

    The verification of the period through the ``check`` flag is not reliable
    due to internal simplification of the given expression. Hence, it is set
    to ``False`` by default.

    Examples
    ========
    >>> from sympy import periodicity, Symbol, sin, cos, tan, exp
    >>> x = Symbol('x')
    >>> f = sin(x) + sin(2*x) + sin(3*x)
    >>> periodicity(f, x)
    2*pi
    >>> periodicity(sin(x)*cos(x), x)
    pi
    >>> periodicity(exp(tan(2*x) - 1), x)
    pi/2
    >>> periodicity(sin(4*x)**cos(2*x), x)
    pi
    >>> periodicity(exp(x), x)
    """
    
    # 检查符号类型是否为数字类型，如果不是则抛出未实现错误
    if symbol.kind is not NumberKind:
        raise NotImplementedError("Cannot use symbol of kind %s" % symbol.kind)
    
    # 创建一个实数虚拟符号替代原始函数中的符号
    temp = Dummy('x', real=True)
    f = f.subs(symbol, temp)
    symbol = temp

    def _check(orig_f, period):
        '''Return the checked period or raise an error.'''
        # 替换符号为符号加上周期后的函数并检查是否与原始函数相等
        new_f = orig_f.subs(symbol, symbol + period)
        if new_f.equals(orig_f):
            return period
        else:
            raise NotImplementedError(filldedent('''
                The period of the given function cannot be verified.
                When `%s` was replaced with `%s + %s` in `%s`, the result
                was `%s` which was not recognized as being the same as
                the original function.
                So either the period was wrong or the two forms were
                not recognized as being equal.
                Set check=False to obtain the value.''' %
                (symbol, symbol, period, orig_f, new_f)))

    orig_f = f
    period = None

    # 如果函数是关系表达式，则简化为左边减去右边
    if isinstance(f, Relational):
        f = f.lhs - f.rhs

    # 简化函数表达式
    f = f.simplify()

    # 如果符号不在函数的自由符号中，则返回周期为零
    if symbol not in f.free_symbols:
        return S.Zero

    # 如果函数是三角函数，尝试获取其周期
    if isinstance(f, TrigonometricFunction):
        try:
            period = f.period(symbol)
        except NotImplementedError:
            pass
    # 检查 f 是否属于 Abs 类型
    if isinstance(f, Abs):
        # 获取 Abs 函数的参数
        arg = f.args[0]
        # 如果参数是 sec、csc、cos 中的一种，则重新表示为 sin
        if isinstance(arg, (sec, csc, cos)):
            # 这些函数除了 tan 和 cot 外可能具有一半周期的特性
            arg = sin(arg.args[0])
        # 计算参数的周期性
        period = periodicity(arg, symbol)
        # 如果参数是 sin 类型且具有周期性
        if period is not None and isinstance(arg, sin):
            # 使用 Abs(arg) 代替 orig_f 进行测试
            orig_f = Abs(arg)
            try:
                # 检查周期是否有效
                return _check(orig_f, period/2)
            except NotImplementedError as err:
                # 如果发生 NotImplementedError，根据 check 标志决定是否抛出异常
                if check:
                    raise NotImplementedError(err)
            # 如果不抛出异常，则在下面的检查中继续处理新的 orig_f 和 period

    # 如果 f 是 exp 类型或者是 S.Exp1 的幂
    if isinstance(f, exp) or (f.is_Pow and f.base == S.Exp1):
        # 将 S.Exp1 的幂展开为 Pow(S.Exp1, expand_mul(f.exp))
        f = Pow(S.Exp1, expand_mul(f.exp))
        # 如果虚部不为零
        if im(f) != 0:
            # 分别计算实部和虚部的周期性
            period_real = periodicity(re(f), symbol)
            period_imag = periodicity(im(f), symbol)
            # 如果实部和虚部都具有周期性
            if period_real is not None and period_imag is not None:
                # 计算最小公倍数周期
                period = lcim([period_real, period_imag])

    # 如果 f 是 Pow 类型但 base 不是 S.Exp1
    if f.is_Pow and f.base != S.Exp1:
        base, expo = f.args
        base_has_sym = base.has(symbol)
        expo_has_sym = expo.has(symbol)

        # 如果 base 中包含 symbol 而 expo 不包含
        if base_has_sym and not expo_has_sym:
            # 计算 base 的周期性
            period = periodicity(base, symbol)

        # 如果 expo 中包含 symbol 而 base 不包含
        elif expo_has_sym and not base_has_sym:
            # 计算 expo 的周期性
            period = periodicity(expo, symbol)

        else:
            # 计算整个 f 的周期性
            period = _periodicity(f.args, symbol)

    # 如果 f 是 Mul 类型
    elif f.is_Mul:
        # 将 f 分解为系数和基函数 g
        coeff, g = f.as_independent(symbol, as_Add=False)
        # 如果 g 是三角函数或者 coeff 不等于 1
        if isinstance(g, TrigonometricFunction) or not equal_valued(coeff, 1):
            # 计算 g 的周期性
            period = periodicity(g, symbol)
        else:
            # 计算整个 f 的周期性
            period = _periodicity(g.args, symbol)

    # 如果 f 是 Add 类型
    elif f.is_Add:
        # 将 f 分解为常数 k 和剩余部分 g
        k, g = f.as_independent(symbol)
        # 如果 k 不为零，则返回剩余部分 g 的周期性
        if k is not S.Zero:
            return periodicity(g, symbol)
        # 否则计算整个 f 的周期性
        period = _periodicity(g.args, symbol)

    # 如果 f 是 Mod 类型
    elif isinstance(f, Mod):
        a, n = f.args

        # 如果 a 与 symbol 相等，则周期性为 n
        if a == symbol:
            period = n
        # 如果 a 是三角函数类型，则计算其周期性
        elif isinstance(a, TrigonometricFunction):
            period = periodicity(a, symbol)
        # 检查 f 是否在 symbol 上为线性
        elif (a.is_polynomial(symbol) and degree(a, symbol) == 1 and
            symbol not in n.free_symbols):
                # 计算 Mod 的周期性
                period = Abs(n / a.diff(symbol))

    # 如果 f 是 Piecewise 类型
    elif isinstance(f, Piecewise):
        pass  # 暂不处理 Piecewise，因为其返回类型不合适
    # 如果 period 是 None，则执行以下代码块
    elif period is None:
        # 导入 sympy.solvers.decompogen 模块中的 compogen 和 decompogen 函数
        from sympy.solvers.decompogen import compogen, decompogen
        # 使用 decompogen 函数对 f 进行分解生成，并将结果赋给 g_s
        g_s = decompogen(f, symbol)
        # 计算 g_s 列表的长度，即生成的 g 函数数量
        num_of_gs = len(g_s)
        # 如果生成的函数数量大于 1，则执行以下代码块
        if num_of_gs > 1:
            # 反向遍历 g_s 列表
            for index, g in enumerate(reversed(g_s)):
                # 计算当前 g 的起始索引
                start_index = num_of_gs - 1 - index
                # 使用 compogen 函数对 g_s[start_index:] 进行组合生成新的 g 函数
                g = compogen(g_s[start_index:], symbol)
                # 如果生成的 g 函数不等于原始函数 orig_f 和 f，修复问题 12620
                if g not in (orig_f, f): # Fix for issue 12620
                    # 计算 g 函数的周期性，并将结果赋给 period
                    period = periodicity(g, symbol)
                    # 如果计算出的周期性不为 None，则跳出循环
                    if period is not None:
                        break

    # 如果 period 不是 None，则执行以下代码块
    if period is not None:
        # 如果 check 为真，则返回 _check(orig_f, period) 的结果
        if check:
            return _check(orig_f, period)
        # 否则直接返回 period
        return period

    # 如果 period 仍然是 None，则返回 None
    return None
def _periodicity(args, symbol):
    """
    Helper for `periodicity` to find the period of a list of simpler
    functions.
    It uses the `lcim` method to find the least common period of
    all the functions.

    Parameters
    ==========

    args : Tuple of :py:class:`~.Symbol`
        All the symbols present in a function.

    symbol : :py:class:`~.Symbol`
        The symbol over which the function is to be evaluated.

    Returns
    =======

    period
        The least common period of the function for all the symbols
        of the function.
        ``None`` if for at least one of the symbols the function is aperiodic.

    """
    periods = []  # 初始化一个空列表，用于存放各个函数的周期
    for f in args:
        period = periodicity(f, symbol)  # 计算函数 f 关于符号 symbol 的周期
        if period is None:
            return None  # 如果有任何一个函数是非周期的，则返回 None

        if period is not S.Zero:
            periods.append(period)  # 将周期添加到列表中

    if len(periods) > 1:
        return lcim(periods)  # 如果有多个周期，返回它们的最小公倍数作为整体的周期

    if periods:
        return periods[0]  # 如果只有一个周期，直接返回该周期


def lcim(numbers):
    """Returns the least common integral multiple of a list of numbers.

    The numbers can be rational or irrational or a mixture of both.
    `None` is returned for incommensurable numbers.

    Parameters
    ==========

    numbers : list
        Numbers (rational and/or irrational) for which lcim is to be found.

    Returns
    =======

    number
        lcim if it exists, otherwise ``None`` for incommensurable numbers.

    Examples
    ========

    >>> from sympy.calculus.util import lcim
    >>> from sympy import S, pi
    >>> lcim([S(1)/2, S(3)/4, S(5)/6])
    15/2
    >>> lcim([2*pi, 3*pi, pi, pi/2])
    6*pi
    >>> lcim([S(1), 2*pi])
    """
    result = None  # 初始化结果为 None
    if all(num.is_irrational for num in numbers):
        factorized_nums = [num.factor() for num in numbers]  # 因式分解每个数
        factors_num = [num.as_coeff_Mul() for num in factorized_nums]  # 提取每个数的因子
        term = factors_num[0][1]  # 提取第一个数的共同项
        if all(factor == term for coeff, factor in factors_num):
            common_term = term  # 确定共同项
            coeffs = [coeff for coeff, factor in factors_num]  # 提取系数
            result = lcm_list(coeffs) * common_term  # 计算系数的最小公倍数乘以共同项

    elif all(num.is_rational for num in numbers):
        result = lcm_list(numbers)  # 计算有理数的最小公倍数

    else:
        pass

    return result  # 返回计算结果或 None


def is_convex(f, *syms, domain=S.Reals):
    r"""Determines the  convexity of the function passed in the argument.

    Parameters
    ==========

    f : :py:class:`~.Expr`
        The concerned function.
    syms : Tuple of :py:class:`~.Symbol`
        The variables with respect to which the convexity is to be determined.
    domain : :py:class:`~.Interval`, optional
        The domain over which the convexity of the function has to be checked.
        If unspecified, S.Reals will be the default domain.

    Returns
    =======

    bool
        The method returns ``True`` if the function is convex otherwise it
        returns ``False``.

    Raises
    ======

    NotImplementedError
        The check for the convexity of multivariate functions is not implemented yet.

    Notes
    ======

    """
    # 如果符号变量列表 syms 的长度大于1，则进行多元函数的凸性检查，此时返回 Hessian 矩阵的正半定性
    if len(syms) > 1:
        return hessian(f, syms).is_positive_semidefinite

    # 导入 solve_univariate_inequality 函数用于解决一元不等式
    from sympy.solvers.inequalities import solve_univariate_inequality
    # 将输入的函数 f 转换为 SymPy 的表达式
    f = _sympify(f)
    # 获取变量列表中的第一个变量
    var = syms[0]
    # 检查函数 f 在其奇点上是否包含在指定的域 domain 内，如果有则返回 False
    if any(s in domain for s in singularities(f, var)):
        return False
    # 计算函数 f 对变量 var 的二阶导数，并判断其是否小于0
    condition = f.diff(var, 2) < 0
    # 解决二阶导数小于0的不等式条件，如果在指定的 domain 内存在解，则返回 False
    if solve_univariate_inequality(condition, var, False, domain):
        return False
    # 若以上条件都不满足，则返回 True，表示函数 f 在指定条件下是凸的
    return True
# 计算给定函数在指定域内的静止点（导数为零的点）集合
def stationary_points(f, symbol, domain=S.Reals):
    """
    Returns the stationary points of a function (where derivative of the
    function is 0) in the given domain.

    Parameters
    ==========

    f : :py:class:`~.Expr`
        The concerned function.
        输入的函数表达式。
    symbol : :py:class:`~.Symbol`
        The variable for which the stationary points are to be determined.
        需要确定静止点的变量。
    domain : :py:class:`~.Interval`
        The domain over which the stationary points have to be checked.
        If unspecified, ``S.Reals`` will be the default domain.
        静止点要检查的域。如果未指定，默认为``S.Reals``。

    Returns
    =======

    Set
        A set of stationary points for the function. If there are no
        stationary point, an :py:class:`~.EmptySet` is returned.
        函数的静止点集合。如果没有静止点，返回 :py:class:`~.EmptySet`。

    Examples
    ========

    >>> from sympy import Interval, Symbol, S, sin, pi, pprint, stationary_points
    >>> x = Symbol('x')

    >>> stationary_points(1/x, x, S.Reals)
    EmptySet

    >>> pprint(stationary_points(sin(x), x), use_unicode=False)
              pi                              3*pi
    {2*n*pi + -- | n in Integers} U {2*n*pi + ---- | n in Integers}
              2                                2

    >>> stationary_points(sin(x),x, Interval(0, 4*pi))
    {pi/2, 3*pi/2, 5*pi/2, 7*pi/2}

    """
    from sympy.solvers.solveset import solveset

    # 如果域为空集，则直接返回空集
    if domain is S.EmptySet:
        return S.EmptySet

    # 计算函数在给定域内的连续域
    domain = continuous_domain(f, symbol, domain)
    # 求解导数为零的方程，得到静止点集合
    set = solveset(diff(f, symbol), symbol, domain)

    return set


# 计算给定函数在指定域内的最大值
def maximum(f, symbol, domain=S.Reals):
    """
    Returns the maximum value of a function in the given domain.

    Parameters
    ==========

    f : :py:class:`~.Expr`
        The concerned function.
        输入的函数表达式。
    symbol : :py:class:`~.Symbol`
        The variable for maximum value needs to be determined.
        需要确定最大值的变量。
    domain : :py:class:`~.Interval`
        The domain over which the maximum have to be checked.
        If unspecified, then the global maximum is returned.
        需要检查最大值的域。如果未指定，则返回全局最大值。

    Returns
    =======

    number
        Maximum value of the function in given domain.
        函数在给定域内的最大值。

    Examples
    ========

    >>> from sympy import Interval, Symbol, S, sin, cos, pi, maximum
    >>> x = Symbol('x')

    >>> f = -x**2 + 2*x + 5
    >>> maximum(f, x, S.Reals)
    6

    >>> maximum(sin(x), x, Interval(-pi, pi/4))
    sqrt(2)/2

    >>> maximum(sin(x)*cos(x), x)
    1/2

    """
    if isinstance(symbol, Symbol):
        # 如果符号是有效的符号对象
        if domain is S.EmptySet:
            raise ValueError("Maximum value not defined for empty domain.")

        # 返回函数在给定域内的函数值范围的上确界作为最大值
        return function_range(f, symbol, domain).sup
    else:
        raise ValueError("%s is not a valid symbol." % symbol)


# 计算给定函数在指定域内的最小值
def minimum(f, symbol, domain=S.Reals):
    """
    Returns the minimum value of a function in the given domain.

    Parameters
    ==========

    f : :py:class:`~.Expr`
        The concerned function.
        输入的函数表达式。
    symbol : :py:class:`~.Symbol`
        The variable for minimum value needs to be determined.
        需要确定最小值的变量。
    domain : :py:class:`~.Interval`
        The domain over which the minimum have to be checked.
        If unspecified, then the global minimum is returned.
        需要检查最小值的域。如果未指定，则返回全局最小值。

    """
    # 在函数定义中，代码未提供
    # 如果 symbol 是一个符号对象（Symbol类的实例）
    if isinstance(symbol, Symbol):
        # 如果 domain 是空集（EmptySet常量），则抛出值错误异常
        if domain is S.EmptySet:
            raise ValueError("Minimum value not defined for empty domain.")
        
        # 调用 function_range 函数计算函数在给定符号和域上的值域，并返回其下界（inf）
        return function_range(f, symbol, domain).inf
    else:
        # 如果 symbol 不是有效的符号对象，则抛出值错误异常，并显示错误信息
        raise ValueError("%s is not a valid symbol." % symbol)
```