# `D:\src\scipysrc\sympy\sympy\solvers\polysys.py`

```
"""Solvers of systems of polynomial equations. """
# 导入 itertools 模块，用于迭代工具函数
import itertools

# 从 sympy 库中导入必要的模块和类
from sympy.core import S
from sympy.core.sorting import default_sort_key
from sympy.polys import Poly, groebner, roots
from sympy.polys.polytools import parallel_poly_from_expr
from sympy.polys.polyerrors import (ComputationFailed,
    PolificationFailed, CoercionFailed)
from sympy.simplify import rcollect
from sympy.utilities import postfixes
from sympy.utilities.misc import filldedent


# 定义自定义异常类 SolveFailed，用于指示求解器条件未满足
class SolveFailed(Exception):
    """Raised when solver's conditions were not met. """


# 定义函数 solve_poly_system，用于求解多项式方程组
def solve_poly_system(seq, *gens, strict=False, **args):
    """
    Return a list of solutions for the system of polynomial equations
    or else None.

    Parameters
    ==========

    seq: a list/tuple/set
        Listing all the equations that are needed to be solved
    gens: generators
        generators of the equations in seq for which we want the
        solutions
    strict: a boolean (default is False)
        if strict is True, NotImplementedError will be raised if
        the solution is known to be incomplete (which can occur if
        not all solutions are expressible in radicals)
    args: Keyword arguments
        Special options for solving the equations.


    Returns
    =======

    List[Tuple]
        a list of tuples with elements being solutions for the
        symbols in the order they were passed as gens
    None
        None is returned when the computed basis contains only the ground.

    Examples
    ========

    >>> from sympy import solve_poly_system
    >>> from sympy.abc import x, y

    >>> solve_poly_system([x*y - 2*y, 2*y**2 - x**2], x, y)
    [(0, 0), (2, -sqrt(2)), (2, sqrt(2))]

    >>> solve_poly_system([x**5 - x + y**3, y**2 - 1], x, y, strict=True)
    Traceback (most recent call last):
    ...
    UnsolvableFactorError

    """
    try:
        # 将表达式转化为多项式并返回多项式列表及选项对象
        polys, opt = parallel_poly_from_expr(seq, *gens, **args)
    except PolificationFailed as exc:
        # 如果转化失败，抛出计算失败异常，包括失败原因
        raise ComputationFailed('solve_poly_system', len(seq), exc)

    # 如果方程的数量和生成器的数量都为 2
    if len(polys) == len(opt.gens) == 2:
        f, g = polys

        # 如果所有多项式的次数都不大于 2
        if all(i <= 2 for i in f.degree_list() + g.degree_list()):
            try:
                # 尝试解二次齐次方程组
                return solve_biquadratic(f, g, opt)
            except SolveFailed:
                pass

    # 否则调用通用多项式方程组求解函数
    return solve_generic(polys, opt, strict=strict)


# 定义函数 solve_biquadratic，用于解二次齐次方程组
def solve_biquadratic(f, g, opt):
    """Solve a system of two bivariate quadratic polynomial equations.

    Parameters
    ==========

    f: a single Expr or Poly
        First equation
    g: a single Expr or Poly
        Second Equation
    opt: an Options object
        For specifying keyword arguments and generators

    Returns
    =======

    List[Tuple]
        a list of tuples with elements being solutions for the
        symbols in the order they were passed as gens
    None
        None is returned when the computed basis contains only the ground.

    Examples
    ========

    >>> from sympy import Options, Poly

    """
    # 实现解二次齐次方程组的具体逻辑，此处未完待续
    >>> from sympy.abc import x, y
    导入 sympy 库中的 x 和 y 符号变量

    >>> from sympy.solvers.polysys import solve_biquadratic
    导入解二次多项式系统方程的函数

    >>> NewOption = Options((x, y), {'domain': 'ZZ'})
    创建一个新的选项对象 NewOption，指定变量 x 和 y 的整数域域

    >>> a = Poly(y**2 - 4 + x, y, x, domain='ZZ')
    创建多项式对象 a，表示 y^2 - 4 + x，指定域为整数域

    >>> b = Poly(y*2 + 3*x - 7, y, x, domain='ZZ')
    创建多项式对象 b，表示 2*y + 3*x - 7，指定域为整数域

    >>> solve_biquadratic(a, b, NewOption)
    调用 solve_biquadratic 函数解决多项式系统方程 a 和 b，使用 NewOption 选项对象

    >>> a = Poly(y + x**2 - 3, y, x, domain='ZZ')
    更新多项式对象 a，表示 y + x^2 - 3，指定域为整数域

    >>> b = Poly(-y + x - 4, y, x, domain='ZZ')
    更新多项式对象 b，表示 -y + x - 4，指定域为整数域

    >>> solve_biquadratic(a, b, NewOption)
    调用 solve_biquadratic 函数解决更新后的多项式系统方程 a 和 b，使用 NewOption 选项对象

    """
    G = groebner([f, g])
    计算给定多项式列表 [f, g] 的格劳布纳基基础

    if len(G) == 1 and G[0].is_ground:
        如果 G 的长度为 1 并且 G 的第一个元素是地面元素（常数项）
        返回 None

    if len(G) != 2:
        如果 G 的长度不等于 2
        抛出 SolveFailed 异常

    x, y = opt.gens
    从选项对象 opt 中获取变量 x 和 y

    p, q = G
    将格劳布纳基基础 G 中的两个多项式分配给变量 p 和 q

    if not p.gcd(q).is_ground:
        如果 p 和 q 的最大公因数不是地面元素（非常数）
        抛出 SolveFailed 异常

    p = Poly(p, x, expand=False)
    将多项式 p 转换为多项式对象，并关闭其展开功能

    p_roots = [rcollect(expr, y) for expr in roots(p).keys()]
    提取多项式 p 的根并按 y 进行收集

    q = q.ltrim(-1)
    对多项式 q 进行左修剪，移除最高次的项

    q_roots = list(roots(q).keys())
    提取多项式 q 的根并生成根列表

    solutions = [(p_root.subs(y, q_root), q_root) for q_root, p_root in
                 itertools.product(q_roots, p_roots)]
    使用 itertools 的 product 函数生成 p_roots 和 q_roots 的笛卡尔积，并计算解集

    return sorted(solutions, key=default_sort_key)
    返回按默认排序键排序的解集
# 定义一个函数，用于解决通用的多项式方程组
def solve_generic(polys, opt, strict=False):
    """
    Solve a generic system of polynomial equations.

    Returns all possible solutions over C[x_1, x_2, ..., x_m] of a
    set F = { f_1, f_2, ..., f_n } of polynomial equations, using
    Groebner basis approach. For now only zero-dimensional systems
    are supported, which means F can have at most a finite number
    of solutions. If the basis contains only the ground, None is
    returned.

    The algorithm works by the fact that, supposing G is the basis
    of F with respect to an elimination order (here lexicographic
    order is used), G and F generate the same ideal, they have the
    same set of solutions. By the elimination property, if G is a
    reduced, zero-dimensional Groebner basis, then there exists an
    univariate polynomial in G (in its last variable). This can be
    solved by computing its roots. Substituting all computed roots
    for the last (eliminated) variable in other elements of G, new
    polynomial system is generated. Applying the above procedure
    recursively, a finite number of solutions can be found.

    The ability of finding all solutions by this procedure depends
    on the root finding algorithms. If no solutions were found, it
    means only that roots() failed, but the system is solvable. To
    overcome this difficulty use numerical algorithms instead.

    Parameters
    ==========

    polys: a list/tuple/set
        Listing all the polynomial equations that are needed to be solved
    opt: an Options object
        For specifying keyword arguments and generators
    strict: a boolean
        If strict is True, NotImplementedError will be raised if the solution
        is known to be incomplete

    Returns
    =======

    List[Tuple]
        a list of tuples with elements being solutions for the
        symbols in the order they were passed as gens
    None
        None is returned when the computed basis contains only the ground.

    References
    ==========

    .. [Buchberger01] B. Buchberger, Groebner Bases: A Short
    Introduction for Systems Theorists, In: R. Moreno-Diaz,
    B. Buchberger, J.L. Freire, Proceedings of EUROCAST'01,
    February, 2001

    .. [Cox97] D. Cox, J. Little, D. O'Shea, Ideals, Varieties
    and Algorithms, Springer, Second Edition, 1997, pp. 112

    Raises
    ========

    NotImplementedError
        If the system is not zero-dimensional (does not have a finite
        number of solutions)

    UnsolvableFactorError
        If ``strict`` is True and not all solution components are
        expressible in radicals

    Examples
    ========

    >>> from sympy import Poly, Options
    >>> from sympy.solvers.polysys import solve_generic
    >>> from sympy.abc import x, y
    >>> NewOption = Options((x, y), {'domain': 'ZZ'})

    >>> a = Poly(x - y + 5, x, y, domain='ZZ')
    >>> b = Poly(x + y - 3, x, y, domain='ZZ')
    >>> solve_generic([a, b], NewOption)
    [(-1, 4)]
    """
    >>> a = Poly(x - 2*y + 5, x, y, domain='ZZ')
    创建一个多项式对象a，表达式为x - 2*y + 5，定义在整数环'ZZ'上，变量为x和y

    >>> b = Poly(2*x - y - 3, x, y, domain='ZZ')
    创建一个多项式对象b，表达式为2*x - y - 3，定义在整数环'ZZ'上，变量为x和y

    >>> solve_generic([a, b], NewOption)
    调用solve_generic函数，解决多项式列表[a, b]的方程组，使用NewOption作为选项

    [(11/3, 13/3)]
    解方程组的结果为一个元组列表，表示x和y的解为(11/3, 13/3)

    >>> a = Poly(x**2 + y, x, y, domain='ZZ')
    创建一个多项式对象a，表达式为x**2 + y，定义在整数环'ZZ'上，变量为x和y

    >>> b = Poly(x + y*4, x, y, domain='ZZ')
    创建一个多项式对象b，表达式为x + y*4，定义在整数环'ZZ'上，变量为x和y

    >>> solve_generic([a, b], NewOption)
    调用solve_generic函数，解决多项式列表[a, b]的方程组，使用NewOption作为选项

    [(0, 0), (1/4, -1/16)]
    解方程组的结果为一个元组列表，表示x和y的解为(0, 0)和(1/4, -1/16)

    >>> a = Poly(x**5 - x + y**3, x, y, domain='ZZ')
    创建一个多项式对象a，表达式为x**5 - x + y**3，定义在整数环'ZZ'上，变量为x和y

    >>> b = Poly(y**2 - 1, x, y, domain='ZZ')
    创建一个多项式对象b，表达式为y**2 - 1，定义在整数环'ZZ'上，变量为x和y

    >>> solve_generic([a, b], NewOption, strict=True)
    调用solve_generic函数，解决多项式列表[a, b]的方程组，使用NewOption作为选项，并设置strict=True

    Traceback (most recent call last):
    ...
    UnsolvableFactorError
    抛出UnsolvableFactorError异常，表示方程组无法解决

    """
    def _is_univariate(f):
        """Returns True if 'f' is univariate in its last variable. """
        # 检查多项式f是否是最后一个变量上的一元多项式
        for monom in f.monoms():
            if any(monom[:-1]):
                return False

        return True

    def _subs_root(f, gen, zero):
        """Replace generator with a root so that the result is nice. """
        # 将多项式f中的生成器gen替换为根zero，以使结果更好
        p = f.as_expr({gen: zero})

        if f.degree(gen) >= 2:
            p = p.expand(deep=False)

        return p
    def _solve_reduced_system(system, gens, entry=False):
        """Recursively solves reduced polynomial systems. """
        if len(system) == len(gens) == 1:
            # 如果系统和生成元都只有一个元素，使用根据 `roots` 方法的解列表生成的零列表来返回单元素元组的列表
            zeros = list(roots(system[0], gens[-1], strict=strict).keys())
            return [(zero,) for zero in zeros]

        # 使用 Buchberger 算法计算给定生成元的格罗布纳基基础
        basis = groebner(system, gens, polys=True)

        if len(basis) == 1 and basis[0].is_ground:
            if not entry:
                return []
            else:
                return None

        # 从格罗布纳基中筛选出一元多项式
        univariate = list(filter(_is_univariate, basis))

        if len(basis) < len(gens):
            # 抛出未实现错误，仅支持零维系统（有限个解）
            raise NotImplementedError(filldedent('''
                only zero-dimensional systems supported
                (finite number of solutions)
                '''))

        if len(univariate) == 1:
            f = univariate.pop()
        else:
            # 抛出未实现错误，仅支持零维系统（有限个解）
            raise NotImplementedError(filldedent('''
                only zero-dimensional systems supported
                (finite number of solutions)
                '''))

        gens = f.gens
        gen = gens[-1]

        # 如果严格模式开启，并且 `roots` 方法返回的解不完整，下面的代码行会产生 UnsolvableFactorError
        zeros = list(roots(f.ltrim(gen), strict=strict).keys())

        if not zeros:
            return []

        if len(basis) == 1:
            # 返回每个零的单元素元组的列表作为解
            return [(zero,) for zero in zeros]

        solutions = []

        # 对每个零进行迭代处理
        for zero in zeros:
            new_system = []
            new_gens = gens[:-1]

            # 替换根值并构建新的多项式系统
            for b in basis[:-1]:
                eq = _subs_root(b, gen, zero)

                if eq is not S.Zero:
                    new_system.append(eq)

            # 递归解决新的简化系统
            for solution in _solve_reduced_system(new_system, new_gens):
                solutions.append(solution + (zero,))

        if solutions and len(solutions[0]) != len(gens):
            # 抛出未实现错误，仅支持零维系统（有限个解）
            raise NotImplementedError(filldedent('''
                only zero-dimensional systems supported
                (finite number of solutions)
                '''))

        # 返回按默认排序键排序的解列表
        return solutions

    try:
        result = _solve_reduced_system(polys, opt.gens, entry=True)
    except CoercionFailed:
        # 如果多项式的类型转换失败，则抛出未实现错误
        raise NotImplementedError

    if result is not None:
        # 返回排序后的结果列表
        return sorted(result, key=default_sort_key)
# 使用 Gianni-Kalkbrenner 算法解决多项式系统的问题

def solve_triangulated(polys, *gens, **args):
    """
    Solve a polynomial system using Gianni-Kalkbrenner algorithm.

    The algorithm proceeds by computing one Groebner basis in the ground
    domain and then by iteratively computing polynomial factorizations in
    appropriately constructed algebraic extensions of the ground domain.

    Parameters
    ==========

    polys: a list/tuple/set
        Listing all the equations that are needed to be solved
    gens: generators
        generators of the equations in polys for which we want the
        solutions
    args: Keyword arguments
        Special options for solving the equations

    Returns
    =======

    List[Tuple]
        A List of tuples. Solutions for symbols that satisfy the
        equations listed in polys

    Examples
    ========

    >>> from sympy import solve_triangulated
    >>> from sympy.abc import x, y, z

    >>> F = [x**2 + y + z - 1, x + y**2 + z - 1, x + y + z**2 - 1]

    >>> solve_triangulated(F, x, y, z)
    [(0, 0, 1), (0, 1, 0), (1, 0, 0)]

    References
    ==========

    1. Patrizia Gianni, Teo Mora, Algebraic Solution of System of
    Polynomial Equations using Groebner Bases, AAECC-5 on Applied Algebra,
    Algebraic Algorithms and Error-Correcting Codes, LNCS 356 247--257, 1989

    """

    # 计算多项式集合的 Groebner 基，并将结果反转存储在 G 中
    G = groebner(polys, gens, polys=True)
    G = list(reversed(G))

    # 获取域参数
    domain = args.get('domain')

    # 如果指定了域，则将每个基于 G 的多项式设置为该域
    if domain is not None:
        for i, g in enumerate(G):
            G[i] = g.set_domain(domain)

    # 提取 G 的第一个元素的领导项，将其右截取一位，剩下的存入 f，其域存入 dom
    f, G = G[0].ltrim(-1), G[1:]
    dom = f.get_domain()

    # 计算 f 的根，并存入 zeros
    zeros = f.ground_roots()
    # 初始化解集，每个元素为 (根, dom)
    solutions = {((zero,), dom) for zero in zeros}

    # 倒序遍历 gens[:-1] 的顺序生成器序列，以及 gens[1:] 的后缀序列
    var_seq = reversed(gens[:-1])
    vars_seq = postfixes(gens[1:])

    # 对于每个 var 和对应的 vars，执行以下操作
    for var, vars in zip(var_seq, vars_seq):
        _solutions = set()

        # 对于 solutions 中的每个 (values, dom)
        for values, dom in solutions:
            H, mapping = [], list(zip(vars, values))

            # 遍历 G 中的每个多项式 g
            for g in G:
                _vars = (var,) + vars

                # 如果 g 只包含 _vars，并且 g 关于 var 的次数不为 0
                if g.has_only_gens(*_vars) and g.degree(var) != 0:
                    # 对 g 进行 var 的左截取，并使用 mapping 执行评估
                    h = g.ltrim(var).eval(dict(mapping))

                    # 如果 g 关于 var 的次数等于 h 的次数
                    if g.degree(var) == h.degree():
                        H.append(h)

            # 从 H 中选取次数最小的多项式 p
            p = min(H, key=lambda h: h.degree())
            # 获取 p 的根，并存入 zeros
            zeros = p.ground_roots()

            # 对于 zeros 中的每个 zero
            for zero in zeros:
                # 如果 zero 不是有理数
                if not zero.is_Rational:
                    # 使用 dom 的代数域扩展 zero
                    dom_zero = dom.algebraic_field(zero)
                else:
                    # 否则，保持 dom 不变
                    dom_zero = dom

                # 将 ((zero,) + values, dom_zero) 添加到 _solutions 中
                _solutions.add(((zero,) + values, dom_zero))

        # 将 _solutions 赋值给 solutions
        solutions = _solutions

    # 返回按默认排序键排序的解集的列表
    return sorted((s for s, _ in solutions), key=default_sort_key)
```