# `D:\src\scipysrc\sympy\sympy\simplify\radsimp.py`

```
# 导入 defaultdict 模块
from collections import defaultdict

# 导入 sympify, S, Mul, Derivative, Pow 等子模块
from sympy.core import sympify, S, Mul, Derivative, Pow

# 导入 _unevaluated_Add, Add 子模块
from sympy.core.add import _unevaluated_Add, Add

# 导入 assumptions 子模块
from sympy.core.assumptions import assumptions

# 导入 Factors, gcd_terms 子模块
from sympy.core.exprtools import Factors, gcd_terms

# 导入 _mexpand, expand_mul, expand_power_base 子模块
from sympy.core.function import _mexpand, expand_mul, expand_power_base

# 导入 _keep_coeff, _unevaluated_Mul, _mulsort 子模块
from sympy.core.mul import _keep_coeff, _unevaluated_Mul, _mulsort

# 导入 Rational, zoo, nan 子模块
from sympy.core.numbers import Rational, zoo, nan

# 导入 global_parameters 子模块
from sympy.core.parameters import global_parameters

# 导入 ordered, default_sort_key 子模块
from sympy.core.sorting import ordered, default_sort_key

# 导入 Dummy, Wild, symbols 子模块
from sympy.core.symbol import Dummy, Wild, symbols

# 导入 exp, sqrt, log 子模块
from sympy.functions import exp, sqrt, log

# 导入 Abs 子模块
from sympy.functions.elementary.complexes import Abs

# 导入 gcd 子模块
from sympy.polys import gcd

# 导入 sqrtdenest 子模块
from sympy.simplify.sqrtdenest import sqrtdenest

# 导入 iterable, sift 子模块
from sympy.utilities.iterables import iterable, sift

# 定义 collect 函数，用于收集表达式中的加法项
def collect(expr, syms, func=None, evaluate=None, exact=False, distribute_order_term=True):
    """
    Collect additive terms of an expression.

    Explanation
    ===========

    This function collects additive terms of an expression with respect
    to a list of expression up to powers with rational exponents. By the
    term symbol here are meant arbitrary expressions, which can contain
    powers, products, sums etc. In other words symbol is a pattern which
    will be searched for in the expression's terms.

    The input expression is not expanded by :func:`collect`, so user is
    expected to provide an expression in an appropriate form. This makes
    :func:`collect` more predictable as there is no magic happening behind the
    scenes. However, it is important to note, that powers of products are
    converted to products of powers using the :func:`~.expand_power_base`
    function.

    There are two possible types of output. First, if ``evaluate`` flag is
    set, this function will return an expression with collected terms or
    else it will return a dictionary with expressions up to rational powers
    as keys and collected coefficients as values.

    Examples
    ========

    >>> from sympy import S, collect, expand, factor, Wild
    >>> from sympy.abc import a, b, c, x, y

    This function can collect symbolic coefficients in polynomials or
    rational expressions. It will manage to find all integer or rational
    powers of collection variable::

        >>> collect(a*x**2 + b*x**2 + a*x - b*x + c, x)
        c + x**2*(a + b) + x*(a - b)

    The same result can be achieved in dictionary form::

        >>> d = collect(a*x**2 + b*x**2 + a*x - b*x + c, x, evaluate=False)
        >>> d[x**2]
        a + b
        >>> d[x]
        a - b
        >>> d[S.One]
        c

    You can also work with multivariate polynomials. However, remember that
    this function is greedy so it will care only about a single symbol at time,
    in specification order::

        >>> collect(x**2 + y*x**2 + x*y + y + a*y, [x, y])
        x**2*(y + 1) + x*y + y*(a + 1)
    """
    # 复杂表达式的示例，可以作为模式使用：
    
    >>> from sympy import sin, log
    >>> collect(a*sin(2*x) + b*sin(2*x), sin(2*x))
    (a + b)*sin(2*x)
    
    # 另一个示例，使用 x*log(x) 作为模式：
    
    >>> collect(a*x*log(x) + b*(x*log(x)), x*log(x))
    x*(a + b)*log(x)
    
    # 可以在模式中使用通配符：
    
    >>> w = Wild('w1')
    >>> collect(a*x**y - b*x**y, w**y)
    x**y*(a - b)
    
    # 处理符号幂次的情况，基数和指数的符号部分被视为单个符号：
    
    >>> collect(a*x**c + b*x**c, x)
    a*x**c + b*x**c
    >>> collect(a*x**c + b*x**c, x**c)
    x**c*(a + b)
    
    # 当指数中包含有理数时，表现为已知的行为：
    
    >>> collect(a*x**(2*c) + b*x**(2*c), x**c)
    x**(2*c)*(a + b)
    
    # 注意所有关于 collect 函数的先前陈述事实也适用于指数函数，可以得到：
    
    >>> from sympy import exp
    >>> collect(a*exp(2*x) + b*exp(2*x), exp(x))
    (a + b)*exp(2*x)
    
    # 如果只想收集特定符号的特定幂次，则将 exact 标志设置为 True：
    
    >>> collect(a*x**7 + b*x**7, x, exact=True)
    a*x**7 + b*x**7
    >>> collect(a*x**7 + b*x**7, x**7, exact=True)
    x**7*(a + b)
    
    # 如果希望在包含符号的任何对象上收集，则将 exact 设置为 None：
    
    >>> collect(x*exp(x) + sin(x)*y + sin(x)*2 + 3*x, x, exact=None)
    x*exp(x) + 3*x + (y + 2)*sin(x)
    >>> collect(a*x*y + x*y + b*x + x, [x, y], exact=None)
    x*y*(a + 1) + x*(b + 1)
    
    # 还可以将此函数应用于微分方程，其中可以收集任意阶数的导数。注意，如果按照函数或函数的导数收集，则该函数的所有导数也将被收集。使用 exact=True 可以防止这种情况发生：
    
    >>> from sympy import Derivative as D, collect, Function
    >>> f = Function('f')(x)
    
    >>> collect(a*D(f,x) + b*D(f,x), D(f,x))
    (a + b)*Derivative(f(x), x)
    
    >>> collect(a*D(D(f,x),x) + b*D(D(f,x),x), f)
    (a + b)*Derivative(f(x), (x, 2))
    
    >>> collect(a*D(D(f,x),x) + b*D(D(f,x),x), D(f,x), exact=True)
    a*Derivative(f(x), (x, 2)) + b*Derivative(f(x), (x, 2))
    
    >>> collect(a*D(f,x) + b*D(f,x) + a*f + b*f, f)
    (a + b)*f(x) + (a + b)*Derivative(f(x), x)
    
    # 还可以同时匹配导数阶数和指数：
    
    >>> collect(a*D(D(f,x),x)**2 + b*D(D(f,x),x)**2, D(f,x))
    (a + b)*Derivative(f(x), (x, 2))**2
    
    # 最后，可以对收集到的每个系数应用一个函数。
    For example you can factorize symbolic coefficients of polynomial::

        >>> f = expand((x + a + 1)**3)

        >>> collect(f, x, factor)
        x**3 + 3*x**2*(a + 1) + 3*x*(a + 1)**2 + (a + 1)**3

    .. note:: Arguments are expected to be in expanded form, so you might have
              to call :func:`~.expand` prior to calling this function.

    See Also
    ========

    collect_const, collect_sqrt, rcollect
    """
    # 将表达式转换为符号表达式对象
    expr = sympify(expr)
    # 将syms列表中的每个元素转换为符号对象，如果syms是可迭代对象的话
    syms = [sympify(i) for i in (syms if iterable(syms) else [syms])]

    # 根据条件筛选出不是单纯符号（Symbol）或者不是-x，或者包含通配符的表达式
    cond = lambda x: x.is_Symbol or (-x).is_Symbol or bool(
        x.atoms(Wild))
    _, nonsyms = sift(syms, cond, binary=True)
    # 如果存在非符号的情况，为每个非符号创建一个虚拟符号对象，并替换原有的syms列表中的对应项
    if nonsyms:
        reps = dict(zip(nonsyms, [Dummy(**assumptions(i)) for i in nonsyms]))
        syms = [reps.get(s, s) for s in syms]
        # 使用reps字典替换expr中的非符号表达式，并调用collect函数进行收集操作
        rv = collect(expr.subs(reps), syms,
            func=func, evaluate=evaluate, exact=exact,
            distribute_order_term=distribute_order_term)
        urep = {v: k for k, v in reps.items()}
        # 如果rv不是字典，将其视为表达式，并使用urep字典替换其中的符号
        if not isinstance(rv, dict):
            return rv.xreplace(urep)
        else:
            # 对rv字典中的每对键值对进行符号替换操作，使用urep字典
            return {urep.get(k, k).xreplace(urep): v.xreplace(urep)
                    for k, v in rv.items()}

    # 如果exact为None，则尝试处理其他表达式
    if exact is None:
        _syms = set()
        # 遍历expr的每个子项，确定是否需要考虑它们作为符号
        for i in Add.make_args(expr):
            # 如果子项不含有syms中的自由符号或者子项本身就是syms中的一部分，则继续下一个子项
            if not i.has_free(*syms) or i in syms:
                continue
            # 如果子项既不是乘积也不是syms中的一部分，则将其添加到_syms集合中
            if not i.is_Mul and i not in syms:
                _syms.add(i)
            else:
                # 识别复合生成器
                g = i._new_rawargs(*i.as_coeff_mul(*syms)[1])
                if g not in syms:
                    _syms.add(g)
        # 检查是否所有的_syms项都是syms中符号的幂
        simple = all(i.is_Pow and i.base in syms for i in _syms)
        # 将_syms集合转换为有序列表，并加入到syms列表中
        syms = syms + list(ordered(_syms))
        # 如果不是简单的情况，则调用collect函数进行收集操作，exact设置为False
        if not simple:
            return collect(expr, syms,
                func=func, evaluate=evaluate, exact=False,
                distribute_order_term=distribute_order_term)

    # 如果evaluate为None，则将其设置为全局参数中的evaluate值
    if evaluate is None:
        evaluate = global_parameters.evaluate

    # 定义一个函数，用于生成表达式的乘积
    def make_expression(terms):
        product = []

        # 遍历terms列表中的每个元素，生成相应的乘积项
        for term, rat, sym, deriv in terms:
            # 如果deriv不为None，则进行导数操作，直到达到指定的阶数
            if deriv is not None:
                var, order = deriv
                for _ in range(order):
                    term = Derivative(term, var)

            # 如果sym为None，则根据rat的值生成幂次方或者直接添加term
            if sym is None:
                if rat is S.One:
                    product.append(term)
                else:
                    product.append(Pow(term, rat))
            else:
                # 根据rat和sym的值生成幂次方，并添加到product列表中
                product.append(Pow(term, rat*sym))

        # 返回乘积项的乘积结果
        return Mul(*product)
    def parse_derivative(deriv):
        # 解析输入表达式中的导数链，返回基础函数和最大导数阶数
        expr, sym, order = deriv.expr, deriv.variables[0], 1
    
        for s in deriv.variables[1:]:
            # 检查变量列表中是否有重复变量，如果有则抛出未实现的错误
            if s == sym:
                order += 1
            else:
                raise NotImplementedError(
                    'Improve MV Derivative support in collect')
    
        while isinstance(expr, Derivative):
            s0 = expr.variables[0]
    
            # 检查导数链中是否所有导数都是对同一变量的，如果不是则抛出未实现的错误
            if any(s != s0 for s in expr.variables):
                raise NotImplementedError(
                    'Improve MV Derivative support in collect')
    
            if s0 == sym:
                expr, order = expr.expr, order + len(expr.variables)
            else:
                break
    
        return expr, (sym, Rational(order))
    
    def parse_term(expr):
        """解析表达式 expr 并输出元组 (sexpr, rat_expo, sym_expo, deriv)
        其中：
         - sexpr 是基础表达式
         - rat_expo 是基础表达式的有理指数
         - sym_expo 是基础表达式的符号指数
         - deriv 包含表达式的导数链
    
         例如，x 的输出为 (x, 1, None, None)
         2**x 的输出为 (2, 1, x, None).
        """
        rat_expo, sym_expo = S.One, None
        sexpr, deriv = expr, None
    
        if expr.is_Pow:
            if isinstance(expr.base, Derivative):
                # 如果基础是导数，则解析导数链
                sexpr, deriv = parse_derivative(expr.base)
            else:
                sexpr = expr.base
    
            if expr.base == S.Exp1:
                arg = expr.exp
                if arg.is_Rational:
                    sexpr, rat_expo = S.Exp1, arg
                elif arg.is_Mul:
                    coeff, tail = arg.as_coeff_Mul(rational=True)
                    sexpr, rat_expo = exp(tail), coeff
    
            elif expr.exp.is_Number:
                rat_expo = expr.exp
            else:
                coeff, tail = expr.exp.as_coeff_Mul()
    
                if coeff.is_Number:
                    rat_expo, sym_expo = coeff, tail
                else:
                    sym_expo = expr.exp
        elif isinstance(expr, exp):
            arg = expr.exp
            if arg.is_Rational:
                sexpr, rat_expo = S.Exp1, arg
            elif arg.is_Mul:
                coeff, tail = arg.as_coeff_Mul(rational=True)
                sexpr, rat_expo = exp(tail), coeff
        elif isinstance(expr, Derivative):
            # 如果表达式是导数，则解析导数链
            sexpr, deriv = parse_derivative(expr)
    
        return sexpr, rat_expo, sym_expo, deriv
    # 如果需要评估表达式
    if evaluate:
        # 如果表达式是加法类型
        if expr.is_Add:
            # 获取 o 的值，如果不存在则为 0
            o = expr.getO() or 0
            # 对表达式中每个非零参数进行集合操作，排除 o，然后加上 o
            expr = expr.func(*[
                    collect(a, syms, func, True, exact, distribute_order_term)
                    for a in expr.args if a != o]) + o
        # 如果表达式是乘法类型
        elif expr.is_Mul:
            # 对表达式中每个因子进行集合操作
            return expr.func(*[
                collect(term, syms, func, True, exact, distribute_order_term)
                for term in expr.args])
        # 如果表达式是幂运算类型
        elif expr.is_Pow:
            # 对幂运算中的底数进行集合操作
            b = collect(
                expr.base, syms, func, True, exact, distribute_order_term)
            # 返回新的幂运算对象
            return Pow(b, expr.exp)

    # 对符号列表中的每个符号应用 expand_power_base 函数
    syms = [expand_power_base(i, deep=False) for i in syms]

    order_term = None

    # 如果需要分布阶项
    if distribute_order_term:
        # 获取表达式的阶项
        order_term = expr.getO()

        # 如果阶项不为 None
        if order_term is not None:
            # 如果阶项中包含任何符号则置为 None
            if order_term.has(*syms):
                order_term = None
            else:
                # 移除表达式中的阶项
                expr = expr.removeO()

    # 将表达式转换为加法类型的元素列表，每个元素都应用 expand_power_base 函数
    summa = [expand_power_base(i, deep=False) for i in Add.make_args(expr)]

    # 初始化 collected 和 disliked
    collected, disliked = defaultdict(list), S.Zero

    # 对加法列表中的每个乘积进行处理
    for product in summa:
        # 将乘积分解为有乘法和非乘法项
        c, nc = product.args_cnc(split_1=False)
        # 对乘法项进行排序
        args = list(ordered(c)) + nc
        # 解析每个项为一个术语列表
        terms = [parse_term(i) for i in args]
        # 初始化 small_first
        small_first = True

        # 对符号列表中的每个符号进行处理
        for symbol in syms:
            # 如果符号是导数类型并且 small_first 为真
            if isinstance(symbol, Derivative) and small_first:
                # 反转术语列表
                terms = list(reversed(terms))
                small_first = not small_first
            # 解析术语列表以生成表达式
            result = parse_expression(terms, symbol)

            # 如果结果不为 None
            if result is not None:
                # 如果符号不是可交换的，则抛出属性错误
                if not symbol.is_commutative:
                    raise AttributeError("Can not collect noncommutative symbol")

                # 解析结果中的各项
                terms, elems, common_expo, has_deriv = result

                # 当前模式中存在导数时，需要重新构建其表达式
                if not has_deriv:
                    margs = []
                    for elem in elems:
                        if elem[2] is None:
                            e = elem[1]
                        else:
                            e = elem[1] * elem[2]
                        margs.append(Pow(elem[0], e))
                    index = Mul(*margs)
                else:
                    index = make_expression(elems)

                # 对生成的表达式应用 expand_power_base 函数
                terms = expand_power_base(make_expression(terms), deep=False)
                index = expand_power_base(index, deep=False)

                # 将结果添加到 collected 字典中
                collected[index].append(terms)
                break
        else:
            # 如果没有模式匹配，则将当前乘积加入 disliked
            disliked += product

    # 对每个 key 添加项
    collected = {k: Add(*v) for k, v in collected.items()}

    # 如果 disliked 不为零，则将其添加到 collected 中
    if disliked is not S.Zero:
        collected[S.One] = disliked

    # 如果存在阶项，则将其添加到 collected 的每个值中
    if order_term is not None:
        for key, val in collected.items():
            collected[key] = val + order_term

    # 如果存在 func 函数，则对 collected 中的每个值应用 func 函数
    if func is not None:
        collected = {
            key: func(val) for key, val in collected.items()}
    # 如果 evaluate 参数为真，则执行以下语句块
    if evaluate:
        # 构建一个加法表达式，其中包含收集项字典中每个键值对乘积的和
        return Add(*[key*val for key, val in collected.items()])
    # 如果 evaluate 参数为假，则执行以下语句块
    else:
        # 直接返回收集项字典 collected
        return collected
def rcollect(expr, *vars):
    """
    Recursively collect sums in an expression.

    Examples
    ========

    >>> from sympy.simplify import rcollect
    >>> from sympy.abc import x, y

    >>> expr = (x**2*y + x*y + x + y)/(x + y)

    >>> rcollect(expr, y)
    (x + y*(x**2 + x + 1))/(x + y)

    See Also
    ========

    collect, collect_const, collect_sqrt
    """
    # 如果表达式是原子或者不包含给定变量，则直接返回表达式
    if expr.is_Atom or not expr.has(*vars):
        return expr
    else:
        # 对表达式的每个参数进行递归地调用 rcollect 函数
        expr = expr.__class__(*[rcollect(arg, *vars) for arg in expr.args])

        # 如果表达式是加法表达式，则调用 collect 函数进行收集
        if expr.is_Add:
            return collect(expr, vars)
        else:
            return expr


def collect_sqrt(expr, evaluate=None):
    """Return expr with terms having common square roots collected together.
    If ``evaluate`` is False a count indicating the number of sqrt-containing
    terms will be returned and, if non-zero, the terms of the Add will be
    returned, else the expression itself will be returned as a single term.
    If ``evaluate`` is True, the expression with any collected terms will be
    returned.

    Note: since I = sqrt(-1), it is collected, too.

    Examples
    ========

    >>> from sympy import sqrt
    >>> from sympy.simplify.radsimp import collect_sqrt
    >>> from sympy.abc import a, b

    >>> r2, r3, r5 = [sqrt(i) for i in [2, 3, 5]]
    >>> collect_sqrt(a*r2 + b*r2)
    sqrt(2)*(a + b)
    >>> collect_sqrt(a*r2 + b*r2 + a*r3 + b*r3)
    sqrt(2)*(a + b) + sqrt(3)*(a + b)
    >>> collect_sqrt(a*r2 + b*r2 + a*r3 + b*r5)
    sqrt(3)*a + sqrt(5)*b + sqrt(2)*(a + b)

    If evaluate is False then the arguments will be sorted and
    returned as a list and a count of the number of sqrt-containing
    terms will be returned:

    >>> collect_sqrt(a*r2 + b*r2 + a*r3 + b*r5, evaluate=False)
    ((sqrt(3)*a, sqrt(5)*b, sqrt(2)*(a + b)), 3)
    >>> collect_sqrt(a*sqrt(2) + b, evaluate=False)
    ((b, sqrt(2)*a), 1)
    >>> collect_sqrt(a + b, evaluate=False)
    ((a + b,), 0)

    See Also
    ========

    collect, collect_const, rcollect
    """
    # 如果 evaluate 参数未指定，则使用全局参数中的设置
    if evaluate is None:
        evaluate = global_parameters.evaluate
    
    # 将表达式分解为其最简形式
    coeff, expr = expr.as_content_primitive()
    vars = set()
    
    # 遍历表达式的每个加法成员，确定是否包含平方根
    for a in Add.make_args(expr):
        for m in a.args_cnc()[0]:
            if m.is_number and (
                    m.is_Pow and m.exp.is_Rational and m.exp.q == 2 or
                    m is S.ImaginaryUnit):
                vars.add(m)

    # 收集表达式中包含的平方根项，排除数字处理
    d = collect_const(expr, *vars, Numbers=False)
    hit = expr != d
    # 如果不需要评估（evaluate），则执行以下操作
    if not evaluate:
        # 将 nrad 初始化为 0
        nrad = 0
        # 将参数列表 args 设置为按顺序排列的 Add 类的参数的列表
        args = list(ordered(Add.make_args(d)))
        # 遍历参数列表 args，使用 i 作为索引，m 作为当前元素
        for i, m in enumerate(args):
            # 调用 args_cnc 方法，返回 m 的 canonical 和 noncanonical 部分的元组 c, nc
            c, nc = m.args_cnc()
            # 遍历 canonical 部分 c 中的每个元素 ci
            for ci in c:
                # 判断 ci 是否为幂运算且指数是有理数且分母为 2，或者 ci 是虚数单位 S.ImaginaryUnit
                if ci.is_Pow and ci.exp.is_Rational and ci.exp.q == 2 or \
                        ci is S.ImaginaryUnit:
                    # 如果符合条件，则 nrad 加一
                    nrad += 1
                    break
            # 将 args[i] 乘以 coeff
            args[i] *= coeff
        # 如果 hit 或 nrad 有一个不为真，则将 args 封装成单个 Add 对象的列表
        if not (hit or nrad):
            args = [Add(*args)]
        # 返回元组，包含处理后的 args 列表和 nrad
        return tuple(args), nrad

    # 如果 evaluate 为真，则直接返回 coeff 乘以 d
    return coeff*d
# 定义函数 `collect_const`，用于从表达式 `expr` 中收集具有相似数值系数的项
def collect_const(expr, *vars, Numbers=True):
    """A non-greedy collection of terms with similar number coefficients in
    an Add expr. If ``vars`` is given then only those constants will be
    targeted. Although any Number can also be targeted, if this is not
    desired set ``Numbers=False`` and no Float or Rational will be collected.

    Parameters
    ==========

    expr : SymPy expression
        This parameter defines the expression the expression from which
        terms with similar coefficients are to be collected. A non-Add
        expression is returned as it is.

    vars : variable length collection of Numbers, optional
        Specifies the constants to target for collection. Can be multiple in
        number.

    Numbers : bool
        Specifies to target all instance of
        :class:`sympy.core.numbers.Number` class. If ``Numbers=False``, then
        no Float or Rational will be collected.

    Returns
    =======

    expr : Expr
        Returns an expression with similar coefficient terms collected.

    Examples
    ========

    >>> from sympy import sqrt
    >>> from sympy.abc import s, x, y, z
    >>> from sympy.simplify.radsimp import collect_const
    >>> collect_const(sqrt(3) + sqrt(3)*(1 + sqrt(2)))
    sqrt(3)*(sqrt(2) + 2)
    >>> collect_const(sqrt(3)*s + sqrt(7)*s + sqrt(3) + sqrt(7))
    (sqrt(3) + sqrt(7))*(s + 1)
    >>> s = sqrt(2) + 2
    >>> collect_const(sqrt(3)*s + sqrt(3) + sqrt(7)*s + sqrt(7))
    (sqrt(2) + 3)*(sqrt(3) + sqrt(7))
    >>> collect_const(sqrt(3)*s + sqrt(3) + sqrt(7)*s + sqrt(7), sqrt(3))
    """
    # 定义内部函数 `_abs`，用于收集乘积 `mul` 中的绝对值和幂函数
    def _abs(mul):
        # 将 `mul` 中的参数分成常数和非常数两个列表
        c, nc = mul.args_cnc()
        a = []  # 用于存放绝对值的列表
        o = []  # 用于存放其他项的列表
        for i in c:
            if isinstance(i, Abs):
                a.append(i.args[0])  # 将绝对值函数中的参数加入 `a`
            elif isinstance(i, Pow) and isinstance(i.base, Abs) and i.exp.is_real:
                a.append(i.base.args[0]**i.exp)  # 将绝对值函数的幂函数加入 `a`
            else:
                o.append(i)  # 将非绝对值的项加入 `o`
        # 如果 `a` 中的项少于两个，并且 `a` 中的任何幂函数都不是负数次幂
        if len(a) < 2 and not any(i.exp.is_negative for i in a if isinstance(i, Pow)):
            return mul
        absarg = Mul(*a)  # 将 `a` 中的项相乘得到一个新的乘积
        A = Abs(absarg)  # 创建一个新的绝对值函数 `Abs(absarg)`
        args = [A]  # 将 `A` 加入到参数列表中
        args.extend(o)  # 将 `o` 中的项加入到参数列表中
        if not A.has(Abs):  # 如果 `A` 中不再包含绝对值函数
            args.extend(nc)  # 将 `nc` 中的项加入到参数列表中
            return Mul(*args)  # 返回重新构建的乘积
        if not isinstance(A, Abs):  # 如果 `A` 不是绝对值函数
            # 重新评估并将其设为未评估状态
            A = Abs(absarg, evaluate=False)
        _mulsort(args)  # 对参数列表进行排序
        args.extend(nc)  # 将 `nc` 中的项加入到参数列表中，`nc` 总是放在最后
        return Mul._from_args(args, is_commutative=not nc)  # 从参数列表中创建新的乘积对象

    # 返回 `expr` 中所有乘积项被 `_abs` 函数处理后的结果
    return expr.replace(
        lambda x: isinstance(x, Mul),  # 匹配所有乘积项
        lambda x: _abs(x)).replace(
            lambda x: isinstance(x, Pow),  # 匹配所有幂函数项
            lambda x: _abs(x))
    sqrt(7) + sqrt(3)*(sqrt(2) + 3) + sqrt(7)*(sqrt(2) + 2)


# 计算一个复杂的数学表达式，包含多个平方根的乘法和加法
sqrt(7) + sqrt(3)*(sqrt(2) + 3) + sqrt(7)*(sqrt(2) + 2)



    The collection is sign-sensitive, giving higher precedence to the
    unsigned values:

    >>> collect_const(x - y - z)
    x - (y + z)
    >>> collect_const(-y - z)
    -(y + z)
    >>> collect_const(2*x - 2*y - 2*z, 2)
    2*(x - y - z)
    >>> collect_const(2*x - 2*y - 2*z, -2)
    2*x - 2*(y + z)

    See Also
    ========

    collect, collect_sqrt, rcollect
    """


# 这段文本提供了对函数的描述和示例用法，以及相关函数的参考资料

    if not expr.is_Add:
        return expr


# 如果表达式不是加法类型，则直接返回原始表达式
if not expr.is_Add:
    return expr



    recurse = False

    if not vars:
        recurse = True
        vars = set()
        for a in expr.args:
            for m in Mul.make_args(a):
                if m.is_number:
                    vars.add(m)
    else:
        vars = sympify(vars)
    if not Numbers:
        vars = [v for v in vars if not v.is_Number]


# 初始化递归标志和变量集合，如果没有给定变量，则根据表达式的参数生成变量集合
recurse = False

if not vars:
    recurse = True
    vars = set()
    for a in expr.args:
        for m in Mul.make_args(a):
            if m.is_number:
                vars.add(m)
else:
    vars = sympify(vars)
if not Numbers:
    vars = [v for v in vars if not v.is_Number]



    vars = list(ordered(vars))
    for v in vars:
        terms = defaultdict(list)
        Fv = Factors(v)
        for m in Add.make_args(expr):
            f = Factors(m)
            q, r = f.div(Fv)
            if r.is_one:
                # only accept this as a true factor if
                # it didn't change an exponent from an Integer
                # to a non-Integer, e.g. 2/sqrt(2) -> sqrt(2)
                # -- we aren't looking for this sort of change
                fwas = f.factors.copy()
                fnow = q.factors
                if not any(k in fwas and fwas[k].is_Integer and not
                        fnow[k].is_Integer for k in fnow):
                    terms[v].append(q.as_expr())
                    continue
            terms[S.One].append(m)


# 对每个变量进行循环处理，生成包含表达式因子的字典
vars = list(ordered(vars))
for v in vars:
    terms = defaultdict(list)
    Fv = Factors(v)
    for m in Add.make_args(expr):
        f = Factors(m)
        q, r = f.div(Fv)
        if r.is_one:
            # 只接受这个因子作为真正的因子，如果它没有改变从整数到非整数的指数，例如 2/sqrt(2) -> sqrt(2)
            # 我们不关心这种改变
            fwas = f.factors.copy()
            fnow = q.factors
            if not any(k in fwas and fwas[k].is_Integer and not
                    fnow[k].is_Integer for k in fnow):
                terms[v].append(q.as_expr())
                continue
        terms[S.One].append(m)



        args = []
        hit = False
        uneval = False
        for k in ordered(terms):
            v = terms[k]
            if k is S.One:
                args.extend(v)
                continue

            if len(v) > 1:
                v = Add(*v)
                hit = True
                if recurse and v != expr:
                    vars.append(v)
            else:
                v = v[0]

            # be careful not to let uneval become True unless
            # it must be because it's going to be more expensive
            # to rebuild the expression as an unevaluated one
            if Numbers and k.is_Number and v.is_Add:
                args.append(_keep_coeff(k, v, sign=True))
                uneval = True
            else:
                args.append(k*v)


# 处理生成的项列表，合并同类项并根据需要保留未评估的表达式
args = []
hit = False
uneval = False
for k in ordered(terms):
    v = terms[k]
    if k is S.One:
        args.extend(v)
        continue

    if len(v) > 1:
        v = Add(*v)
        hit = True
        if recurse and v != expr:
            vars.append(v)
    else:
        v = v[0]

    if Numbers and k.is_Number and v.is_Add:
        args.append(_keep_coeff(k, v, sign=True))
        uneval = True
    else:
        args.append(k*v)



        if hit:
            if uneval:
                expr = _unevaluated_Add(*args)
            else:
                expr = Add(*args)
            if not expr.is_Add:
                break

    return expr


# 根据收集到的项重新构建表达式，返回重新构建后的表达式
if hit:
    if uneval:
        expr = _unevaluated_Add(*args)
    else:
        expr = Add(*args)
    if not expr.is_Add:
        break

return expr
# 定义一个函数 radsimp，用于简化表达式的分母，特别是移除其中的平方根项
def radsimp(expr, symbolic=True, max_terms=4):
    """
    Rationalize the denominator by removing square roots.

    Explanation
    ===========

    The expression returned from radsimp must be used with caution
    since if the denominator contains symbols, it will be possible to make
    substitutions that violate the assumptions of the simplification process:
    that for a denominator matching a + b*sqrt(c), a != +/-b*sqrt(c). (If
    there are no symbols, this assumptions is made valid by collecting terms
    of sqrt(c) so the match variable ``a`` does not contain ``sqrt(c)``.) If
    you do not want the simplification to occur for symbolic denominators, set
    ``symbolic`` to False.

    If there are more than ``max_terms`` radical terms then the expression is
    returned unchanged.

    Examples
    ========

    >>> from sympy import radsimp, sqrt, Symbol, pprint
    >>> from sympy import factor_terms, fraction, signsimp
    >>> from sympy.simplify.radsimp import collect_sqrt
    >>> from sympy.abc import a, b, c

    >>> radsimp(1/(2 + sqrt(2)))
    (2 - sqrt(2))/2
    >>> x,y = map(Symbol, 'xy')
    >>> e = ((2 + 2*sqrt(2))*x + (2 + sqrt(8))*y)/(2 + sqrt(2))
    >>> radsimp(e)
    sqrt(2)*(x + y)

    No simplification beyond removal of the gcd is done. One might
    want to polish the result a little, however, by collecting
    square root terms:

    >>> r2 = sqrt(2)
    >>> r5 = sqrt(5)
    >>> ans = radsimp(1/(y*r2 + x*r2 + a*r5 + b*r5)); pprint(ans)
        ___       ___       ___       ___
      \/ 5 *a + \/ 5 *b - \/ 2 *x - \/ 2 *y
    ------------------------------------------
       2               2      2              2
    5*a  + 10*a*b + 5*b  - 2*x  - 4*x*y - 2*y

    >>> n, d = fraction(ans)
    >>> pprint(factor_terms(signsimp(collect_sqrt(n))/d, radical=True))
            ___             ___
          \/ 5 *(a + b) - \/ 2 *(x + y)
    ------------------------------------------
       2               2      2              2
    5*a  + 10*a*b + 5*b  - 2*x  - 4*x*y - 2*y

    If radicals in the denominator cannot be removed or there is no denominator,
    the original expression will be returned.

    >>> radsimp(sqrt(2)*x + sqrt(2))
    sqrt(2)*x + sqrt(2)

    Results with symbols will not always be valid for all substitutions:

    >>> eq = 1/(a + b*sqrt(c))
    >>> eq.subs(a, b*sqrt(c))
    1/(2*b*sqrt(c))
    >>> radsimp(eq).subs(a, b*sqrt(c))
    nan

    If ``symbolic=False``, symbolic denominators will not be transformed (but
    numeric denominators will still be processed):

    >>> radsimp(eq, symbolic=False)
    1/(a + b*sqrt(c))

    """
    from sympy.core.expr import Expr
    from sympy.simplify.simplify import signsimp

    # 定义一个符号列表 syms，用于处理表达式中的符号变量
    syms = symbols("a:d A:D")
    # 定义一个名为 _num 的函数，用于处理一个符号表达式的系数化简
    def _num(rterms):
        # 如果 rterms 中只有两个项
        if len(rterms) == 2:
            # 创建一个代换字典，将符号变量与 rterms 中的值对应起来
            reps = dict(list(zip([A, a, B, b], [j for i in rterms for j in i])))
            # 返回一个表达式，代换后的结果
            return (sqrt(A)*a - sqrt(B)*b).xreplace(reps)
        # 如果 rterms 中有三个项
        if len(rterms) == 3:
            # 创建一个代换字典，将符号变量与 rterms 中的值对应起来
            reps = dict(list(zip([A, a, B, b, C, c], [j for i in rterms for j in i])))
            # 返回一个表达式，代换后的结果
            return ((sqrt(A)*a + sqrt(B)*b - sqrt(C)*c)*(2*sqrt(A)*sqrt(B)*a*b - A*a**2 -
                    B*b**2 + C*c**2)).xreplace(reps)
        # 如果 rterms 中有四个项
        elif len(rterms) == 4:
            # 创建一个代换字典，将符号变量与 rterms 中的值对应起来
            reps = dict(list(zip([A, a, B, b, C, c, D, d], [j for i in rterms for j in i])))
            # 返回一个表达式，代换后的结果
            return ((sqrt(A)*a + sqrt(B)*b - sqrt(C)*c - sqrt(D)*d)*(2*sqrt(A)*sqrt(B)*a*b
                    - A*a**2 - B*b**2 - 2*sqrt(C)*sqrt(D)*c*d + C*c**2 +
                    D*d**2)*(-8*sqrt(A)*sqrt(B)*sqrt(C)*sqrt(D)*a*b*c*d + A**2*a**4 -
                    2*A*B*a**2*b**2 - 2*A*C*a**2*c**2 - 2*A*D*a**2*d**2 + B**2*b**4 -
                    2*B*C*b**2*c**2 - 2*B*D*b**2*d**2 + C**2*c**4 - 2*C*D*c**2*d**2 +
                    D**2*d**4)).xreplace(reps)
        # 如果 rterms 中只有一个项
        elif len(rterms) == 1:
            # 返回 rterms 中唯一项的平方根
            return sqrt(rterms[0][0])
        else:
            # 如果 rterms 中项数超出以上情况，抛出未实现的错误
            raise NotImplementedError

    # 定义一个名为 ispow2 的函数，用于判断一个表达式是否为二的幂
    def ispow2(d, log2=False):
        # 如果 d 不是幂次表达式，则返回 False
        if not d.is_Pow:
            return False
        # 获取幂次
        e = d.exp
        # 如果幂次是有理数且分母为 2，或者是符号的且分母是 2，返回 True
        if e.is_Rational and e.q == 2 or symbolic and denom(e) == 2:
            return True
        # 如果需要考虑 log2，则继续判断
        if log2:
            q = 1
            # 如果幂次是有理数，获取其分母
            if e.is_Rational:
                q = e.q
            # 如果是符号的，获取其分母
            elif symbolic:
                d = denom(e)
                if d.is_Integer:
                    q = d
            # 如果分母不为 1 且 log2(q) 是整数，则返回 True
            if q != 1 and log(q, 2).is_Integer:
                return True
        # 其他情况返回 False
        return False

    # 如果 expr 不是符号表达式的实例，则对其参数逐个应用 radsimp 函数并返回结果
    if not isinstance(expr, Expr):
        return expr.func(*[radsimp(a, symbolic=symbolic, max_terms=max_terms) for a in expr.args])

    # 将 expr 分解为系数和表达式部分
    coeff, expr = expr.as_coeff_Add()
    # 将表达式标准化
    expr = expr.normal()
    # 将原表达式分解为分子和分母
    old = fraction(expr)
    # 处理分子部分
    n, d = fraction(handle(expr))
    # 如果分解前后分子不同
    if old != (n, d):
        # 如果分母不是原子或者是加法表达式
        if not d.is_Atom:
            # 保存旧的分子分母
            was = (n, d)
            # 对分子部分进行符号简化
            n = signsimp(n, evaluate=False)
            # 对分母部分进行符号简化
            d = signsimp(d, evaluate=False)
            # 构建一个因子对象
            u = Factors(_unevaluated_Mul(n, 1/d))
            # 将因子展开成乘积形式
            u = _unevaluated_Mul(*[k**v for k, v in u.factors.items()])
            # 更新分子分母
            n, d = fraction(u)
            # 如果更新后分子分母与旧的相同，则还原
            if old == (n, d):
                n, d = was
        # 对分子部分进行展开
        n = expand_mul(n)
        # 如果分母是数或者加法表达式
        if d.is_Number or d.is_Add:
            # 对分子分母应用 gcd_terms 函数，获取最简分数形式
            n2, d2 = fraction(gcd_terms(_unevaluated_Mul(n, 1/d)))
            # 如果最简分数形式中分母是数或者操作数比原分母少
            if d2.is_Number or (d2.count_ops() <= d.count_ops()):
                # 对新的分子分母进行符号简化
                n, d = [signsimp(i) for i in (n2, d2)]
                # 如果分子是乘法且第一个参数是数，保持乘法结构
                if n.is_Mul and n.args[0].is_Number:
                    n = n.func(*n.args)

    # 返回系数与处理后的表达式乘积
    return coeff + _unevaluated_Mul(n, 1/d)
# 返回有理化表达式 num/den，通过移除分母中的平方根来有理化；num 和 den 是其平方是正有理数的项的和。

def rad_rationalize(num, den):
    """
    Rationalize ``num/den`` by removing square roots in the denominator;
    num and den are sum of terms whose squares are positive rationals.

    Examples
    ========

    >>> from sympy import sqrt
    >>> from sympy.simplify.radsimp import rad_rationalize
    >>> rad_rationalize(sqrt(3), 1 + sqrt(2)/3)
    (-sqrt(3) + sqrt(6)/3, -7/9)
    """
    # 如果 den 不是 Add 对象（即不是可加的表达式），则直接返回 num 和 den
    if not den.is_Add:
        return num, den
    
    # 使用 split_surds 函数将 den 分解为 g、a、b 三部分
    g, a, b = split_surds(den)
    
    # 将 a 扩展为 a*sqrt(g)
    a = a * sqrt(g)
    
    # 计算新的分子 num 和分母 den
    num = _mexpand((a - b) * num)
    den = _mexpand(a**2 - b**2)
    
    # 递归调用 rad_rationalize 函数，直至分母不再是可加的表达式
    return rad_rationalize(num, den)


# 返回表达式的分子和分母的元组对。
# 如果给定的表达式不是分数，则返回 (expr, 1)。

def fraction(expr, exact=False):
    """Returns a pair with expression's numerator and denominator.
       If the given expression is not a fraction then this function
       will return the tuple (expr, 1).

       This function will not make any attempt to simplify nested
       fractions or to do any term rewriting at all.

       If only one of the numerator/denominator pair is needed then
       use numer(expr) or denom(expr) functions respectively.

       >>> from sympy import fraction, Rational, Symbol
       >>> from sympy.abc import x, y

       >>> fraction(x/y)
       (x, y)
       >>> fraction(x)
       (x, 1)

       >>> fraction(1/y**2)
       (1, y**2)

       >>> fraction(x*y/2)
       (x*y, 2)
       >>> fraction(Rational(1, 2))
       (1, 2)

       This function will also work fine with assumptions:

       >>> k = Symbol('k', negative=True)
       >>> fraction(x * y**k)
       (x, y**(-k))

       If we know nothing about sign of some exponent and ``exact``
       flag is unset, then the exponent's structure will
       be analyzed and pretty fraction will be returned:

       >>> from sympy import exp, Mul
       >>> fraction(2*x**(-y))
       (2, x**y)

       >>> fraction(exp(-x))
       (1, exp(x))

       >>> fraction(exp(-x), exact=True)
       (exp(-x), 1)

       The ``exact`` flag will also keep any unevaluated Muls from
       being evaluated:

       >>> u = Mul(2, x + 1, evaluate=False)
       >>> fraction(u)
       (2*x + 2, 1)
       >>> fraction(u, exact=True)
       (2*(x + 1), 1)
    """
    # 将输入的表达式 expr 转换为符号表达式对象
    expr = sympify(expr)
    
    # 初始化分子和分母为空列表
    numer, denom = [], []
    # 对表达式的每个项进行迭代处理
    for term in Mul.make_args(expr):
        # 检查项是否可交换且为指数或指数函数
        if term.is_commutative and (term.is_Pow or isinstance(term, exp)):
            # 获取基数和指数
            b, ex = term.as_base_exp()
            # 若指数为负数
            if ex.is_negative:
                # 若指数为 -1，则作为分母的一部分
                if ex is S.NegativeOne:
                    denom.append(b)
                # 若允许精确处理
                elif exact:
                    # 若指数为常数，则以 b^(-ex) 形式作为分母的一部分
                    if ex.is_constant():
                        denom.append(Pow(b, -ex))
                    # 否则保留整个项作为分子的一部分
                    else:
                        numer.append(term)
                # 若不允许精确处理，则以 b^(-ex) 形式作为分母的一部分
                else:
                    denom.append(Pow(b, -ex))
            # 若指数为正数，则作为分子的一部分
            elif ex.is_positive:
                numer.append(term)
            # 若不允许精确处理且指数为乘积形式，则将项拆分为分子和分母
            elif not exact and ex.is_Mul:
                n, d = term.as_numer_denom()  # 此操作将导致求值
                # 若分子不为 1，则将其作为分子的一部分
                if n != 1:
                    numer.append(n)
                # 将分母部分添加到分母列表
                denom.append(d)
            # 其他情况，将项作为分子的一部分
            else:
                numer.append(term)
        # 若项为有理数且不为整数
        elif term.is_Rational and not term.is_Integer:
            # 若分子不为 1，则将其作为分子的一部分
            if term.p != 1:
                numer.append(term.p)
            # 将分母部分添加到分母列表
            denom.append(term.q)
        # 其他情况，将项作为分子的一部分
        else:
            numer.append(term)
    # 返回分子和分母的乘积对象
    return Mul(*numer, evaluate=not exact), Mul(*denom, evaluate=not exact)
# 返回表达式的分子，默认匹配分数的默认行为
def numer(expr, exact=False):  # default matches fraction's default
    return fraction(expr, exact=exact)[0]

# 返回表达式的分母，默认匹配分数的默认行为
def denom(expr, exact=False):  # default matches fraction's default
    return fraction(expr, exact=exact)[1]

# 展开表达式中的分数部分
def fraction_expand(expr, **hints):
    return expr.expand(frac=True, **hints)

# 展开表达式中的分子部分，默认匹配分数的默认行为
def numer_expand(expr, **hints):
    a, b = fraction(expr, exact=hints.get('exact', False))
    return a.expand(numer=True, **hints) / b

# 展开表达式中的分母部分，默认匹配分数的默认行为
def denom_expand(expr, **hints):
    a, b = fraction(expr, exact=hints.get('exact', False))
    return a / b.expand(denom=True, **hints)

# 设置别名以便于使用
expand_numer = numer_expand
expand_denom = denom_expand
expand_fraction = fraction_expand

# 将表达式分解为具有正有理数平方的项的和以及具有与 g 等于 gcd 的项的和
def split_surds(expr):
    """
    Split an expression with terms whose squares are positive rationals
    into a sum of terms whose surds squared have gcd equal to g
    and a sum of terms with surds squared prime with g.

    Examples
    ========

    >>> from sympy import sqrt
    >>> from sympy.simplify.radsimp import split_surds
    >>> split_surds(3*sqrt(3) + sqrt(5)/7 + sqrt(6) + sqrt(10) + sqrt(15))
    (3, sqrt(2) + sqrt(5) + 3, sqrt(5)/7 + sqrt(10))
    """
    # 按默认排序键对表达式的参数进行排序
    args = sorted(expr.args, key=default_sort_key)
    # 提取每个参数的系数和乘积部分
    coeff_muls = [x.as_coeff_Mul() for x in args]
    # 提取所有带有整数平方的项
    surds = [x[1]**2 for x in coeff_muls if x[1].is_Pow]
    # 按默认排序键对整数平方进行排序
    surds.sort(key=default_sort_key)
    # 调用内部函数 _split_gcd，将整数平方列表分成两部分
    g, b1, b2 = _split_gcd(*surds)
    g2 = g
    if not b2 and len(b1) >= 2:
        b1n = [x/g for x in b1]
        b1n = [x for x in b1n if x != 1]
        # 如果只有一个公因数被因式分解，则再次进行分割
        g1, b1n, b2 = _split_gcd(*b1n)
        g2 = g*g1
    a1v, a2v = [], []
    # 根据系数和乘积的列表构建两个和
    for c, s in coeff_muls:
        if s.is_Pow and s.exp == S.Half:
            s1 = s.base
            if s1 in b1:
                a1v.append(c*sqrt(s1/g2))
            else:
                a2v.append(c*s)
        else:
            a2v.append(c*s)
    a = Add(*a1v)
    b = Add(*a2v)
    return g2, a, b

# 将整数列表 a 分成两个部分：第一个部分具有 gcd，第二个部分不可被 gcd 整除
def _split_gcd(*a):
    """
    Split the list of integers ``a`` into a list of integers, ``a1`` having
    ``g = gcd(a1)``, and a list ``a2`` whose elements are not divisible by
    ``g``.  Returns ``g, a1, a2``.

    Examples
    ========

    >>> from sympy.simplify.radsimp import _split_gcd
    >>> _split_gcd(55, 35, 22, 14, 77, 10)
    (5, [55, 35, 10], [22, 14, 77])
    """
    # 初始 gcd 为列表第一个元素
    g = a[0]
    # 初始化两个空列表
    b1 = [g]
    b2 = []
    # 遍历列表中的每个元素
    for x in a[1:]:
        # 计算当前元素与当前 gcd 的 gcd
        g1 = gcd(g, x)
        # 如果 gcd 为 1，则将当前元素加入到 b2 列表中
        if g1 == 1:
            b2.append(x)
        else:
            # 否则更新当前 gcd，并将当前元素加入到 b1 列表中
            g = g1
            b1.append(x)
    # 返回 gcd 值，b1 列表和 b2 列表
    return g, b1, b2
```