# `D:\src\scipysrc\sympy\sympy\simplify\simplify.py`

```
from collections import defaultdict  # 导入 defaultdict 类，用于创建默认字典

from sympy.concrete.products import Product  # 导入 Product 类，用于处理乘积表达式
from sympy.concrete.summations import Sum  # 导入 Sum 类，用于处理求和表达式
from sympy.core import (Basic, S, Add, Mul, Pow, Symbol, sympify,  # 导入多个核心类和函数
                        expand_func, Function, Dummy, Expr, factor_terms,
                        expand_power_exp, Eq)
from sympy.core.exprtools import factor_nc  # 导入 factor_nc 函数，用于因式分解非交换项
from sympy.core.parameters import global_parameters  # 导入 global_parameters 对象，用于全局参数
from sympy.core.function import (expand_log, count_ops, _mexpand,  # 导入多个函数，包括对数展开、操作数计数等
    nfloat, expand_mul, expand)
from sympy.core.numbers import Float, I, pi, Rational, equal_valued  # 导入多个数值类和常数
from sympy.core.relational import Relational  # 导入 Relational 类，用于处理关系表达式
from sympy.core.rules import Transform  # 导入 Transform 类，用于规则转换
from sympy.core.sorting import ordered  # 导入 ordered 函数，用于排序
from sympy.core.sympify import _sympify  # 导入 _sympify 函数，用于将对象转换为 SymPy 对象
from sympy.core.traversal import bottom_up as _bottom_up, walk as _walk  # 导入遍历函数

from sympy.functions import gamma, exp, sqrt, log, exp_polar, re  # 导入数学函数，如 Gamma 函数、指数函数等
from sympy.functions.combinatorial.factorials import CombinatorialFunction  # 导入组合数学函数
from sympy.functions.elementary.complexes import unpolarify, Abs, sign  # 导入复数函数，如解极坐标、绝对值、符号函数等
from sympy.functions.elementary.exponential import ExpBase  # 导入指数函数基类
from sympy.functions.elementary.hyperbolic import HyperbolicFunction  # 导入双曲函数
from sympy.functions.elementary.integers import ceiling  # 导入 ceiling 函数，用于向上取整
from sympy.functions.elementary.piecewise import (Piecewise, piecewise_fold,  # 导入分段函数相关
                                                  piecewise_simplify)
from sympy.functions.elementary.trigonometric import TrigonometricFunction  # 导入三角函数
from sympy.functions.special.bessel import (BesselBase, besselj, besseli,  # 导入贝塞尔函数相关
                                            besselk, bessely, jn)
from sympy.functions.special.tensor_functions import KroneckerDelta  # 导入克罗内克函数
from sympy.integrals.integrals import Integral  # 导入积分类
from sympy.matrices.expressions import (MatrixExpr, MatAdd, MatMul,  # 导入矩阵表达式相关
                                            MatPow, MatrixSymbol)
from sympy.polys import together, cancel, factor  # 导入多项式处理函数，如合并、取消公因式、因式分解等
from sympy.polys.numberfields.minpoly import _is_sum_surds, _minimal_polynomial_sq  # 导入数字段相关函数
from sympy.simplify.combsimp import combsimp  # 导入 combsimp 函数，用于组合简化
from sympy.simplify.cse_opts import sub_pre, sub_post  # 导入简化选项相关函数
from sympy.simplify.hyperexpand import hyperexpand  # 导入 hyperexpand 函数，用于超函式展开
from sympy.simplify.powsimp import powsimp  # 导入 powsimp 函数，用于幂简化
from sympy.simplify.radsimp import radsimp, fraction, collect_abs  # 导入有理化简相关函数
from sympy.simplify.sqrtdenest import sqrtdenest  # 导入 sqrtdenest 函数，用于根式去嵌套
from sympy.simplify.trigsimp import trigsimp, exptrigsimp  # 导入三角函数简化函数
from sympy.utilities.decorator import deprecated  # 导入 deprecated 装饰器，用于标记过时函数
from sympy.utilities.iterables import has_variety, sift, subsets, iterable  # 导入多个工具函数
from sympy.utilities.misc import as_int  # 导入 as_int 函数，用于转换为整数

import mpmath  # 导入 mpmath 数学库


def separatevars(expr, symbols=[], dict=False, force=False):
    """
    Separates variables in an expression, if possible.  By
    default, it separates with respect to all symbols in an
    expression and collects constant coefficients that are
    independent of symbols.

    Explanation
    ===========

    If ``dict=True`` then the separated terms will be returned
    in a dictionary keyed to their corresponding symbols.
    By default, all symbols in the expression will appear as
    """
    # 实现变量分离，可以指定符号列表 symbols，以字典形式返回分离后的项，force 参数用于强制分离
    # 将输入的表达式转换为 SymPy 的表达式对象
    expr = sympify(expr)
    # 如果指定了 dict=True，则返回将表达式分离后的结果字典
    if dict:
        # 调用 _separatevars 函数对表达式进行变量分离，并将结果用字典形式返回
        return _separatevars_dict(_separatevars(expr, force), symbols)
    else:
        # 否则直接返回经过 _separatevars 函数处理后的表达式
        return _separatevars(expr, force)
# 根据表达式和是否强制分离变量，对表达式进行变量分离处理
def _separatevars(expr, force):
    # 如果表达式是 Abs 类型
    if isinstance(expr, Abs):
        # 获取表达式的第一个参数
        arg = expr.args[0]
        # 如果参数是乘法并且不是数字
        if arg.is_Mul and not arg.is_number:
            # 使用 dict=True 和 force 参数调用 separatevars 函数
            s = separatevars(arg, dict=True, force=force)
            # 如果返回值不为空
            if s is not None:
                # 对返回的字典中的值应用 expr.func，并返回乘积
                return Mul(*map(expr.func, s.values()))
            else:
                # 否则返回原始表达式
                return expr

    # 如果自由符号少于 2 个，直接返回表达式
    if len(expr.free_symbols) < 2:
        return expr

    # 如果表达式是乘法，但不要破坏它，因为大部分工作可能已经完成
    if expr.is_Mul:
        # 复制表达式的参数列表
        args = list(expr.args)
        changed = False
        # 对每个参数调用 separatevars，并检查是否有变化
        for i, a in enumerate(args):
            args[i] = separatevars(a, force)
            changed = changed or args[i] != a
        # 如果有变化，则用新参数构造表达式
        if changed:
            expr = expr.func(*args)
        return expr

    # 如果表达式是幂，并且底数不是 S.Exp1
    if expr.is_Pow and expr.base != S.Exp1:
        # 对底数应用 separatevars，并保持指数不变
        expr = Pow(separatevars(expr.base, force=force), expr.exp)

    # 尝试其他扩展方法
    expr = expr.expand(mul=False, multinomial=False, force=force)

    # 对表达式进行 posify 处理，如果 force=True，则返回结果及映射字典
    _expr, reps = posify(expr) if force else (expr, {})
    # 对 _expr 进行因式分解，并用映射字典替换
    expr = factor(_expr).subs(reps)

    # 如果表达式不是加法，则直接返回
    if not expr.is_Add:
        return expr

    # 查找可以提取的共同系数
    args = list(expr.args)
    commonc = args[0].args_cnc(cset=True, warn=False)[0]
    for i in args[1:]:
        commonc &= i.args_cnc(cset=True, warn=False)[0]
    commonc = Mul(*commonc)
    commonc = commonc.as_coeff_Mul()[1]  # 忽略常数
    commonc_set = commonc.args_cnc(cset=True, warn=False)[0]

    # 移除共同系数
    for i, a in enumerate(args):
        c, nc = a.args_cnc(cset=True, warn=False)
        c = c - commonc_set
        args[i] = Mul(*c)*Mul(*nc)
    nonsepar = Add(*args)

    # 如果非分离变量的自由符号多于 1 个
    if len(nonsepar.free_symbols) > 1:
        _expr = nonsepar
        # 对 _expr 进行 posify 处理，如果 force=True，则返回结果及映射字典
        _expr, reps = posify(_expr) if force else (_expr, {})
        # 对 _expr 进行因式分解，并用映射字典替换
        _expr = factor(_expr).subs(reps)

        # 如果 _expr 不是加法，则将其赋值给 nonsepar
        if not _expr.is_Add:
            nonsepar = _expr

    # 返回共同系数乘以非分离变量
    return commonc * nonsepar


# 根据表达式和符号集合，将表达式分解为符号相关的乘积形式
def _separatevars_dict(expr, symbols):
    # 如果 symbols 不为空
    if symbols:
        # 如果 symbols 中不是全部都是原子符号，则引发 ValueError
        if not all(t.is_Atom for t in symbols):
            raise ValueError("symbols must be Atoms.")
        symbols = list(symbols)
    # 如果 symbols 是 None，则返回包含 'coeff' 键的字典，值为 expr
    elif symbols is None:
        return {'coeff': expr}
    else:
        # 获取表达式的自由符号列表
        symbols = list(expr.free_symbols)
        # 如果自由符号为空，则返回 None
        if not symbols:
            return None

    # 初始化 ret 字典，包含每个符号及 'coeff' 的空列表
    ret = {i: [] for i in symbols + ['coeff']}

    # 遍历 Mul.make_args(expr) 的每个因子
    for i in Mul.make_args(expr):
        # 获取因子的自由符号集合
        expsym = i.free_symbols
        # 获取自由符号集合与 symbols 集合的交集
        intersection = set(symbols).intersection(expsym)
        # 如果交集大于 1，则返回 None
        if len(intersection) > 1:
            return None
        # 如果交集为空，则将因子添加到 'coeff' 对应的列表中
        if len(intersection) == 0:
            ret['coeff'].append(i)
        else:
            # 否则，将因子添加到交集中的符号对应的列表中
            ret[intersection.pop()].append(i)

    # 重新构建 ret 字典，每个键对应的值为该键对应列表的乘积
    for k, v in ret.items():
        ret[k] = Mul(*v)

    # 返回 ret 字典
    return ret


# 返回处理后的表达式和一个字典，其中包含新旧符号的映射关系
def posify(eq):
    """Return ``eq`` (with generic symbols made positive) and a
    dictionary containing the mapping between the old and new
    symbols.
    
    Explanation
    ===========
    """
    # 将表达式转换为符号对象，并根据需要进行符号处理
    eq = sympify(eq)
    # 如果输入的表达式是可迭代的（例如列表或元组）
    if iterable(eq):
        # 记录原始输入的类型
        f = type(eq)
        # 将可迭代对象转换为列表
        eq = list(eq)
        # 创建一个空集合来存储所有符号
        syms = set()
        # 遍历所有表达式，收集其中的符号
        for e in eq:
            syms = syms.union(e.atoms(Symbol))
        # 创建一个空字典来存储替换映射
        reps = {}
        # 对每个符号进行符号处理，并更新替换映射
        for s in syms:
            reps.update({v: k for k, v in posify(s)[1].items()})
        # 对每个表达式进行替换，并返回处理后的结果和恢复字典
        for i, e in enumerate(eq):
            eq[i] = e.subs(reps)
        # 返回处理后的表达式列表和恢复字典
        return f(eq), {r: s for s, r in reps.items()}

    # 对于单个表达式，创建一个替换字典来替换没有明确符号属性的正数符号
    reps = {s: Dummy(s.name, positive=True, **s.assumptions0)
                 for s in eq.free_symbols if s.is_positive is None}
    # 应用符号处理并返回处理后的表达式和恢复字典
    eq = eq.subs(reps)
    return eq, {r: s for s, r in reps.items()}
def hypersimp(f, k):
    """Given combinatorial term f(k) simplify its consecutive term ratio
       i.e. f(k+1)/f(k).  The input term can be composed of functions and
       integer sequences which have equivalent representation in terms
       of gamma special function.

       Explanation
       ===========

       The algorithm performs three basic steps:

       1. Rewrite all functions in terms of gamma, if possible.

       2. Rewrite all occurrences of gamma in terms of products
          of gamma and rising factorial with integer,  absolute
          constant exponent.

       3. Perform simplification of nested fractions, powers
          and if the resulting expression is a quotient of
          polynomials, reduce their total degree.

       If f(k) is hypergeometric then as result we arrive with a
       quotient of polynomials of minimal degree. Otherwise None
       is returned.

       For more information on the implemented algorithm refer to:

       1. W. Koepf, Algorithms for m-fold Hypergeometric Summation,
          Journal of Symbolic Computation (1995) 20, 399-417
    """
    f = sympify(f)  # Convert f to a symbolic expression

    g = f.subs(k, k + 1) / f  # Compute f(k+1) / f(k)

    g = g.rewrite(gamma)  # Rewrite g in terms of the gamma function
    if g.has(Piecewise):  # If g contains Piecewise expressions
        g = piecewise_fold(g)  # Fold the Piecewise expressions
        g = g.args[-1][0]  # Extract the simplified expression from Piecewise
    g = expand_func(g)  # Expand g into a more canonical form
    g = powsimp(g, deep=True, combine='exp')  # Simplify powers in g

    if g.is_rational_function(k):  # Check if g is a rational function in k
        return simplify(g, ratio=S.Infinity)  # Simplify g, allowing large simplifications
    else:
        return None  # Return None if g is not a rational function


def hypersimilar(f, g, k):
    """
    Returns True if ``f`` and ``g`` are hyper-similar.

    Explanation
    ===========

    Similarity in hypergeometric sense means that a quotient of
    f(k) and g(k) is a rational function in ``k``. This procedure
    is useful in solving recurrence relations.

    For more information see hypersimp().

    """
    f, g = list(map(sympify, (f, g)))  # Convert f and g to symbolic expressions

    h = (f/g).rewrite(gamma)  # Compute the quotient f(k)/g(k) and rewrite in terms of gamma
    h = h.expand(func=True, basic=False)  # Expand h into a canonical form

    return h.is_rational_function(k)  # Check if h is a rational function in k


def signsimp(expr, evaluate=None):
    """Make all Add sub-expressions canonical wrt sign.

    Explanation
    ===========

    If an Add subexpression, ``a``, can have a sign extracted,
    as determined by could_extract_minus_sign, it is replaced
    with Mul(-1, a, evaluate=False). This allows signs to be
    extracted from powers and products.

    Examples
    ========

    >>> from sympy import signsimp, exp, symbols
    >>> from sympy.abc import x, y
    >>> i = symbols('i', odd=True)
    >>> n = -1 + 1/x
    >>> n/x/(-n)**2 - 1/n/x
    (-1 + 1/x)/(x*(1 - 1/x)**2) - 1/(x*(-1 + 1/x))
    >>> signsimp(_)
    0
    >>> x*n + x*-n
    x*(-1 + 1/x) + x*(1 - 1/x)
    >>> signsimp(_)
    0

    Since powers automatically handle leading signs

    >>> (-2)**i
    -2**i

    signsimp can be used to put the base of a power with an integer
    exponent into canonical form:

    >>> n**i
    (-1 + 1/x)**i

    By default, signsimp does not leave behind any hollow simplification:
    """
    pass  # Placeholder function; no actual implementation shown
    """
    If making an Add canonical with respect to sign didn't change the expression,
    the original Add is restored. If this behavior is not desired, the keyword
    `evaluate` can be set to False:
    
    >>> e = exp(y - x)
    >>> signsimp(e) == e
    True
    >>> signsimp(e, evaluate=False)
    exp(-(x - y))
    
    """
    
    # 如果未指定 evaluate 参数，则使用全局参数中的 evaluate 值
    if evaluate is None:
        evaluate = global_parameters.evaluate
    
    # 将输入的表达式转换为 SymPy 表达式对象
    expr = sympify(expr)
    
    # 如果表达式不是 SymPy 的表达式或关系表达式，或者是原子表达式，则直接返回原表达式
    if not isinstance(expr, (Expr, Relational)) or expr.is_Atom:
        return expr
    
    # 通过替换操作消除已存在的关于符号的不等式
    e = expr.replace(lambda x: x.is_Mul and -(-x) != x, lambda x: -(-x))
    
    # 对表达式进行前处理和后处理的替换操作
    e = sub_post(sub_pre(e))
    
    # 如果处理后的表达式不是 SymPy 的表达式或关系表达式，或者是原子表达式，则直接返回处理后的表达式
    if not isinstance(e, (Expr, Relational)) or e.is_Atom:
        return e
    
    # 如果处理后的表达式是加法表达式
    if e.is_Add:
        # 对加法表达式中的每个项递归应用符号简化函数
        rv = e.func(*[signsimp(a) for a in e.args])
        # 如果 evaluate 参数为 False，并且结果是一个加法表达式且可以提取负号，则返回相应的乘法表达式
        if not evaluate and isinstance(rv, Add) and rv.could_extract_minus_sign():
            return Mul(S.NegativeOne, -rv, evaluate=False)
        return rv
    
    # 如果 evaluate 参数为 True，则再次对表达式进行符号简化操作
    if evaluate:
        e = e.replace(lambda x: x.is_Mul and -(-x) != x, lambda x: -(-x))
    
    # 返回最终处理后的表达式
    return e
# 定义一个函数用于简化给定的数学表达式
def simplify(expr, ratio=1.7, measure=count_ops, rational=False, inverse=False, doit=True, **kwargs):
    """Simplifies the given expression.

    Explanation
    ===========

    Simplification is not a well defined term and the exact strategies
    this function tries can change in the future versions of SymPy. If
    your algorithm relies on "simplification" (whatever it is), try to
    determine what you need exactly  -  is it powsimp()?, radsimp()?,
    together()?, logcombine()?, or something else? And use this particular
    function directly, because those are well defined and thus your algorithm
    will be robust.

    Nonetheless, especially for interactive use, or when you do not know
    anything about the structure of the expression, simplify() tries to apply
    intelligent heuristics to make the input expression "simpler".  For
    example:

    >>> from sympy import simplify, cos, sin
    >>> from sympy.abc import x, y
    >>> a = (x + x**2)/(x*sin(y)**2 + x*cos(y)**2)
    >>> a
    (x**2 + x)/(x*sin(y)**2 + x*cos(y)**2)
    >>> simplify(a)
    x + 1

    Note that we could have obtained the same result by using specific
    simplification functions:

    >>> from sympy import trigsimp, cancel
    >>> trigsimp(a)
    (x**2 + x)/x
    >>> cancel(_)
    x + 1

    In some cases, applying :func:`simplify` may actually result in some more
    complicated expression. The default ``ratio=1.7`` prevents more extreme
    cases: if (result length)/(input length) > ratio, then input is returned
    unmodified.  The ``measure`` parameter lets you specify the function used
    to determine how complex an expression is.  The function should take a
    single argument as an expression and return a number such that if
    expression ``a`` is more complex than expression ``b``, then
    ``measure(a) > measure(b)``.  The default measure function is
    :func:`~.count_ops`, which returns the total number of operations in the
    expression.

    For example, if ``ratio=1``, ``simplify`` output cannot be longer
    than input.

    ::

        >>> from sympy import sqrt, simplify, count_ops, oo
        >>> root = 1/(sqrt(2)+3)

    Since ``simplify(root)`` would result in a slightly longer expression,
    root is returned unchanged instead::

       >>> simplify(root, ratio=1) == root
       True

    If ``ratio=oo``, simplify will be applied anyway::

        >>> count_ops(simplify(root, ratio=oo)) > count_ops(root)
        True

    Note that the shortest expression is not necessary the simplest, so
    setting ``ratio`` to 1 may not be a good idea.
    Heuristically, the default value ``ratio=1.7`` seems like a reasonable
    choice.

    You can easily define your own measure function based on what you feel
    should represent the "size" or "complexity" of the input expression.  Note
    that some choices, such as ``lambda expr: len(str(expr))`` may appear to be
    good metrics, but have other problems (in this case, the measure function
    is not invariant under different simplifications), which can lead to
    inefficient or wrong results.

    """
    # ``simplify()``函数用于尝试简化给定的数学表达式，其默认的简化度量标准是``count_ops``
    # 如果你不知道一个好的度量标准是什么，那么默认的``count_ops``通常是一个不错的选择。
    
    # 例如：
    >>> from sympy import symbols, log
    >>> a, b = symbols('a b', positive=True)
    >>> g = log(a) + log(b) + log(a)*log(1/b)
    >>> h = simplify(g)
    >>> h
    log(a*b**(1 - log(a)))
    >>> count_ops(g)
    8
    >>> count_ops(h)
    5
    
    # 可以看到，使用``count_ops``度量标准，``h``比``g``更简单。
    # 然而，有时我们可能不喜欢``simplify``（在本例中使用``logcombine``）创建的``b**(log(1/a) + 1)``项。
    # 一种简单的方法是增加对幂运算的权重。我们可以通过使用``visual=True``选项来实现这一点：
    
    >>> print(count_ops(g, visual=True))
    2*ADD + DIV + 4*LOG + MUL
    >>> print(count_ops(h, visual=True))
    2*LOG + MUL + POW + SUB
    
    >>> from sympy import Symbol, S
    >>> def my_measure(expr):
    ...     POW = Symbol('POW')
    ...     # 通过给POW赋予权重10来减少幂运算的权重
    ...     count = count_ops(expr, visual=True).subs(POW, 10)
    ...     # 其它每种操作默认赋予权重1
    ...     count = count.replace(Symbol, type(S.One))
    ...     return count
    >>> my_measure(g)
    8
    >>> my_measure(h)
    14
    >>> 15./8 > 1.7 # 1.7是默认的比率
    True
    >>> simplify(g, measure=my_measure)
    -log(a)*log(b) + log(a) + log(b)
    
    # 注意，因为``simplify()``在内部尝试多种不同的简化策略，并使用度量函数进行比较，所以这样做会得到一个与输入表达式不同但简化后的完全不同结果。
    
    # 如果``rational=True``，在简化之前将浮点数转换为有理数。
    # 如果``rational=None``，将浮点数转换为有理数，但结果将重新转换为浮点数。
    # 如果``rational=False``（默认值），则不会对浮点数进行任何处理。
    
    # 如果``inverse=True``，假设复合的反函数（如sin和asin）可以以任何顺序取消。
    # 例如，``asin(sin(x))``将返回``x``，而不检查x是否属于这种关系为真的集合。默认值为False。
    
    # 注意，``simplify()``会自动调用最终表达式的``doit()``。你可以通过传递``doit=False``来避免这种行为。
    
    # 此外，需要注意，简化布尔表达式是不明确的。如果表达式倾向于自动评估（如:obj:`~.Eq()`或:obj:`~.Or()`），简化将返回``True``或``False``，如果可以确定其真值。
    # 如果表达式不是默认评估的（如:obj:`~.Predicate()`），简化不会减少它，你应该使用:func:`~.refine()`或:func:`~.ask()`
    function. This inconsistency will be resolved in future version.

    See Also
    ========

    sympy.assumptions.refine.refine : Simplification using assumptions.
    sympy.assumptions.ask.ask : Query for boolean expressions using assumptions.
    """

    # 定义一个函数 shorter，从多个选项中返回操作数最少的选项，如果操作数相同则选择第一个
    def shorter(*choices):
        """
        Return the choice that has the fewest ops. In case of a tie,
        the expression listed first is selected.
        """
        # 如果选择中没有变量多样性，直接返回第一个选项
        if not has_variety(choices):
            return choices[0]
        # 返回操作数最少的选项
        return min(choices, key=measure)

    # 定义函数 done，执行表达式的计算（如果 doit 为真），然后返回操作数最少的选项
    def done(e):
        rv = e.doit() if doit else e
        return shorter(rv, collect_abs(rv))

    # 将输入的表达式转换为符号表达式，根据参数设置来决定是否保持有理数形式
    expr = sympify(expr, rational=rational)
    # 设置关键字参数，使用默认值或者给定的值
    kwargs = {
        "ratio": kwargs.get('ratio', ratio),
        "measure": kwargs.get('measure', measure),
        "rational": kwargs.get('rational', rational),
        "inverse": kwargs.get('inverse', inverse),
        "doit": kwargs.get('doit', doit)}
    # 如果表达式是 Expr 类型且为零，则根据情况返回 S.Zero 或者表达式本身
    if isinstance(expr, Expr) and expr.is_zero:
        return S.Zero if not expr.is_Number else expr

    # 获取表达式的 _eval_simplify 方法
    _eval_simplify = getattr(expr, '_eval_simplify', None)
    # 如果存在 _eval_simplify 方法，则使用其简化表达式并返回结果
    if _eval_simplify is not None:
        return _eval_simplify(**kwargs)

    # 对原始表达式进行处理：先进行符号化、然后收集绝对值、接着简化符号
    original_expr = expr = collect_abs(signsimp(expr))

    # 如果表达式不是 Basic 类型或者没有参数，则返回表达式本身
    if not isinstance(expr, Basic) or not expr.args:  # XXX: temporary hack
        return expr

    # 如果开启了逆运算且表达式包含函数，则进行逆组合操作
    if inverse and expr.has(Function):
        expr = inversecombine(expr)
        # 如果逆组合后表达式没有参数，则返回表达式本身
        if not expr.args:  # simplified to atomic
            return expr

    # 进行深度简化操作
    handled = Add, Mul, Pow, ExpBase
    expr = expr.replace(
        # 在这里，检查 x.args 不足以判断，因为 Basic 具有 args，但不一定适用于 replace 方法
        lambda x: isinstance(x, Expr) and x.args and not isinstance(
            x, handled),
        lambda x: x.func(*[simplify(i, **kwargs) for i in x.args]),
        simultaneous=False)
    # 如果表达式不属于 handled 类型，则进行 done 操作后返回结果
    if not isinstance(expr, handled):
        return done(expr)

    # 如果表达式不是可交换的，则进行非交换简化
    if not expr.is_commutative:
        expr = nc_simplify(expr)

    # TODO: Apply different strategies, considering expression pattern:
    # is it a purely rational function? Is there any trigonometric function?...
    # See also https://github.com/sympy/sympy/pull/185.


    # 将表达式中的浮点数有理化
    floats = False
    if rational is not False and expr.has(Float):
        floats = True
        expr = nsimplify(expr, rational=True)

    # 应用底向上的简化函数
    expr = _bottom_up(expr, lambda w: getattr(w, 'normal', lambda: w)())
    # 将表达式中的幂因子化为其内容的原始形式
    expr = Mul(*powsimp(expr).as_content_primitive())
    _e = cancel(expr)
    # 使用 shorter 函数选择简化后的表达式，同时解决 issue 6829
    expr1 = shorter(_e, _mexpand(_e).cancel())  # issue 6829
    # 使用 shorter 函数选择通过 together 操作后的表达式，同时解决深层简化问题
    expr2 = shorter(together(expr, deep=True), together(expr1, deep=True))

    # 如果 ratio 为无穷大，则返回 expr2
    if ratio is S.Infinity:
        expr = expr2
    else:
        expr = shorter(expr2, expr1, expr)
    # 如果表达式不是 Basic 类型，返回表达式本身（临时的处理方式）
    if not isinstance(expr, Basic):  # XXX: temporary hack
        return expr

    # 对表达式因式分解，不改变符号
    expr = factor_terms(expr, sign=False)

    # 如果表达式中包含 sign 符号，则重写为绝对值形式
    if expr.has(sign):
        expr = expr.rewrite(Abs)

    # 如果表达式中包含 Piecewise，将其单独处理以避免表达式递归增长
    if expr.has(Piecewise):
        # 将多个 Piecewise 折叠成单个 Piecewise
        expr = piecewise_fold(expr)
        # 如果需要，应用 doit 函数
        expr = done(expr)
        # 如果仍然存在 Piecewise
        if expr.has(Piecewise):
            # 将多个 Piecewise 折叠成单个 Piecewise，以防 doit 导致部分表达式仍为 Piecewise
            expr = piecewise_fold(expr)
            # 如果表达式中含有 KroneckerDelta
            if expr.has(KroneckerDelta):
                expr = kroneckersimp(expr)
            # 如果仍然存在 Piecewise
            if expr.has(Piecewise):
                # 对各段不应用 doit，因为上面已经执行过，但进行简化
                expr = piecewise_simplify(expr, deep=True, doit=False)
                # 如果仍然存在 Piecewise
                if expr.has(Piecewise):
                    # 尝试因式分解公共项
                    expr = shorter(expr, factor_terms(expr))
                    # 由于上面已经使用完整的简化进行了简化，这里不需要进一步处理
                    return expr

    # 对超几何项自动进行超展开，仅在 Piecewise 部分之后执行以避免递归展开
    expr = hyperexpand(expr)

    # 如果表达式中含有 KroneckerDelta，简化之
    if expr.has(KroneckerDelta):
        expr = kroneckersimp(expr)

    # 如果表达式中含有 BesselBase，简化之
    if expr.has(BesselBase):
        expr = besselsimp(expr)

    # 如果表达式中含有三角函数或双曲函数，进行深度三角化简
    if expr.has(TrigonometricFunction, HyperbolicFunction):
        expr = trigsimp(expr, deep=True)

    # 如果表达式中含有对数函数，首先进行对数展开，然后合并对数
    if expr.has(log):
        expr = shorter(expand_log(expr, deep=True), logcombine(expr))

    # 如果表达式中含有组合函数或 gamma 函数，自动调用 gammasimp 进行简化
    if expr.has(CombinatorialFunction, gamma):
        expr = combsimp(expr)

    # 如果表达式中含有求和符号，调用 sum_simplify 进行简化
    if expr.has(Sum):
        expr = sum_simplify(expr, **kwargs)

    # 如果表达式中含有积分符号，对所有积分项进行因式分解
    if expr.has(Integral):
        expr = expr.xreplace({
            i: factor_terms(i) for i in expr.atoms(Integral)})

    # 如果表达式中含有乘积符号，调用 product_simplify 进行简化
    if expr.has(Product):
        expr = product_simplify(expr, **kwargs)

    # 如果表达式中含有 Quantity，调用 quantity_simplify 进行简化
    from sympy.physics.units import Quantity
    if expr.has(Quantity):
        from sympy.physics.units.util import quantity_simplify
        expr = quantity_simplify(expr)

    # 对表达式进行指数三角化简
    short = shorter(powsimp(expr, combine='exp', deep=True), powsimp(expr), expr)
    short = shorter(short, cancel(short))
    short = shorter(short, factor_terms(short), expand_power_exp(expand_mul(short)))
    # 如果表达式中含有三角函数、双曲函数、指数底或指数，进行指数三角化简
    if short.has(TrigonometricFunction, HyperbolicFunction, ExpBase, exp):
        short = exptrigsimp(short)
    # 创建一个转换对象，用于处理特定类型的乘法表达式
    hollow_mul = Transform(
        # 将表达式中的乘法因子解析为乘法操作数
        lambda x: Mul(*x.args),
        # 判断表达式是否为乘法运算，且满足特定条件：第一个操作数是数字，第二个操作数是加法表达式，并且是可交换的
        lambda x:
        x.is_Mul and
        len(x.args) == 2 and
        x.args[0].is_Number and
        x.args[1].is_Add and
        x.is_commutative)
    # 使用上述转换对象，将 short 中的表达式中的空乘法项替换为正确的乘法项
    expr = short.xreplace(hollow_mul)

    # 将表达式分解为分子和分母
    numer, denom = expr.as_numer_denom()
    # 如果分母是加法表达式
    if denom.is_Add:
        # 对分母进行符号简化和有理数简化，最多保留一项
        n, d = fraction(radsimp(1/denom, symbolic=False, max_terms=1))
        # 如果分子不是 1，重新构建表达式
        if n is not S.One:
            expr = (numer*n).expand()/d

    # 如果表达式可以提取负号
    if expr.could_extract_minus_sign():
        # 对表达式进行符号简化和有理数分离
        n, d = fraction(expr)
        # 如果分母不为零，重新构建表达式
        if d != 0:
            expr = signsimp(-n/(-d))

    # 如果表达式的度量值大于比例乘以原始表达式的度量值
    if measure(expr) > ratio*measure(original_expr):
        # 将表达式恢复为原始表达式
        expr = original_expr

    # 如果需要恢复浮点数
    if floats and rational is None:
        # 将表达式转换为浮点数表示，不包括指数部分
        expr = nfloat(expr, exponent=False)

    # 返回处理后的表达式
    return done(expr)
# 主函数，用于简化和求和
def sum_simplify(s, **kwargs):
    # 如果 s 不是 Add 类型，则尝试替换 s 中所有包含 Sum 的 Add 类型表达式
    if not isinstance(s, Add):
        s = s.xreplace({a: sum_simplify(a, **kwargs)
                        for a in s.atoms(Add) if a.has(Sum)})
    # 对表达式 s 进行展开
    s = expand(s)
    # 如果展开后 s 不是 Add 类型，则直接返回
    if not isinstance(s, Add):
        return s

    # 将 s 的参数拆分为各项
    terms = s.args
    s_t = []  # 存放包含 Sum 的项
    o_t = []  # 存放其它项

    for term in terms:
        # 将每个 term 拆分为包含 Sum 的部分和其它部分
        sum_terms, other = sift(Mul.make_args(term),
                                lambda i: isinstance(i, Sum), binary=True)
        # 如果没有包含 Sum 的部分，则将 term 加入到其它项中并继续下一轮循环
        if not sum_terms:
            o_t.append(term)
            continue
        # 将其它部分作为一个 Mul 对象
        other = [Mul(*other)]
        # 对每个包含 Sum 的部分进行简化，并加入 s_t 中
        s_t.append(Mul(*(other + [s._eval_simplify(**kwargs) for s in sum_terms])))

    # 组合所有包含 Sum 的项，并与其它项相加得到结果
    result = Add(sum_combine(s_t), *o_t)

    return result


# 辅助函数，用于组合包含 Sum 的项
def sum_combine(s_t):
    """Helper function for Sum simplification

       Attempts to simplify a list of sums, by combining limits / sum function's
       returns the simplified sum
    """
    used = [False] * len(s_t)

    # 尝试两两组合和简化包含 Sum 的项
    for method in range(2):
        for i, s_term1 in enumerate(s_t):
            if not used[i]:
                for j, s_term2 in enumerate(s_t):
                    if not used[j] and i != j:
                        # 调用 sum_add 函数进行组合
                        temp = sum_add(s_term1, s_term2, method)
                        # 如果结果是 Sum 或 Mul 类型，则更新 s_t[i] 并标记 s_t[j] 为已使用
                        if isinstance(temp, (Sum, Mul)):
                            s_t[i] = temp
                            s_term1 = s_t[i]
                            used[j] = True

    # 将未使用的项相加得到最终结果
    result = S.Zero
    for i, s_term in enumerate(s_t):
        if not used[i]:
            result = Add(result, s_term)

    return result


# 对象方法，用于将 Sum 表达式中的常数因子提取出来
def factor_sum(self, limits=None, radical=False, clear=False, fraction=False, sign=True):
    """Return Sum with constant factors extracted.

    If ``limits`` is specified then ``self`` is the summand; the other
    keywords are passed to ``factor_terms``.

    Examples
    ========

    >>> from sympy import Sum
    >>> from sympy.abc import x, y
    >>> from sympy.simplify.simplify import factor_sum
    >>> s = Sum(x*y, (x, 1, 3))
    >>> factor_sum(s)
    y*Sum(x, (x, 1, 3))
    >>> factor_sum(s.function, s.limits)
    y*Sum(x, (x, 1, 3))
    """
    # XXX deprecate in favor of direct call to factor_terms
    # 准备调用 factor_terms 函数的关键字参数
    kwargs = {"radical": radical, "clear": clear,
              "fraction": fraction, "sign": sign}
    # 如果 limits 存在，则创建 Sum 对象，否则直接使用 self
    expr = Sum(self, *limits) if limits else self
    # 调用 factor_terms 函数进行因子提取，并返回结果
    return factor_terms(expr, **kwargs)


# 辅助函数，用于对两个 Sum 进行组合
def sum_add(self, other, method=0):
    """Helper function for Sum simplification"""
    # 我们知道这是一个常数乘以 Sum 的表达式
    # 所以我们暂时将常数放入内部进行简化
    # 然后简化结果
    def __refactor(val):
        args = Mul.make_args(val)
        sumv = next(x for x in args if isinstance(x, Sum))
        constant = Mul(*[x for x in args if x != sumv])
        return Sum(constant * sumv.function, *sumv.limits)

    # 如果 self 是 Mul 类型，则调用 __refactor 函数进行重构
    if isinstance(self, Mul):
        rself = __refactor(self)
    else:
        rself = self

    # 如果 other 是 Mul 类型，则调用 __refactor 函数进行重构
    if isinstance(other, Mul):
        rother = __refactor(other)
    # 如果`self`和`other`的类型相同，则执行以下操作
    else:
        rother = other

    # 检查`self`和`rother`的类型是否相同
    if type(rself) is type(rother):
        # 如果`method`为0，则执行以下操作
        if method == 0:
            # 如果`self`和`rother`的限制条件相同，则执行以下操作
            if rself.limits == rother.limits:
                # 返回一个包含`self`和`rother`函数和的和的`Sum`对象，并对其进行因子求和
                return factor_sum(Sum(rself.function + rother.function, *rself.limits))
        # 如果`method`为1，则执行以下操作
        elif method == 1:
            # 如果简化后的`self`函数减去`rother`函数等于0，则执行以下操作
            if simplify(rself.function - rother.function) == 0:
                # 如果`self`和`rother`的限制条件列表长度均为1，则执行以下操作
                if len(rself.limits) == len(rother.limits) == 1:
                    # 获取`self`和`rother`的第一个限制条件的各个部分
                    i = rself.limits[0][0]
                    x1 = rself.limits[0][1]
                    y1 = rself.limits[0][2]
                    j = rother.limits[0][0]
                    x2 = rother.limits[0][1]
                    y2 = rother.limits[0][2]

                    # 如果`self`和`rother`的限制条件的索引相同，则执行以下操作
                    if i == j:
                        # 如果`rother`的上限比`self`的下限多1，则执行以下操作
                        if x2 == y1 + 1:
                            # 返回一个新的`Sum`对象，其函数为`self`的函数，且限制条件的上限为`rother`的上限
                            return factor_sum(Sum(rself.function, (i, x1, y2)))
                        # 如果`self`的上限比`rother`的下限多1，则执行以下操作
                        elif x1 == y2 + 1:
                            # 返回一个新的`Sum`对象，其函数为`self`的函数，且限制条件的上限为`rother`的上限
                            return factor_sum(Sum(rself.function, (i, x2, y1)))

    # 如果以上条件均不满足，则返回`self`和`other`的加法结果
    return Add(self, other)
def product_simplify(s, **kwargs):
    """Product simplification function."""
    # 将表达式分解为乘积项和其他项
    terms = Mul.make_args(s)
    p_t = []  # 存储乘积项
    o_t = []  # 存储其他项

    # 获取是否深度简化的参数，默认为 True
    deep = kwargs.get('deep', True)
    for term in terms:
        if isinstance(term, Product):
            if deep:
                # 如果需要深度简化，则对乘积项进行简化
                p_t.append(Product(term.function.simplify(**kwargs),
                                   *term.limits))
            else:
                p_t.append(term)
        else:
            o_t.append(term)  # 将非乘积项添加到其他项列表中

    used = [False] * len(p_t)

    # 进行两种方法的乘积项合并
    for method in range(2):
        for i, p_term1 in enumerate(p_t):
            if not used[i]:
                for j, p_term2 in enumerate(p_t):
                    if not used[j] and i != j:
                        # 调用 product_mul 函数进行乘积项合并
                        tmp_prod = product_mul(p_term1, p_term2, method)
                        if isinstance(tmp_prod, Product):
                            p_t[i] = tmp_prod
                            used[j] = True

    # 构建最终的乘积表达式
    result = Mul(*o_t)

    for i, p_term in enumerate(p_t):
        if not used[i]:
            result = Mul(result, p_term)

    return result


def product_mul(self, other, method=0):
    """Helper function for multiplying Product terms."""
    if type(self) is type(other):
        if method == 0:
            if self.limits == other.limits:
                # 根据指定方法合并两个相同限制的乘积项
                return Product(self.function * other.function, *self.limits)
        elif method == 1:
            if simplify(self.function - other.function) == 0:
                if len(self.limits) == len(other.limits) == 1:
                    i = self.limits[0][0]
                    x1 = self.limits[0][1]
                    y1 = self.limits[0][2]
                    j = other.limits[0][0]
                    x2 = other.limits[0][1]
                    y2 = other.limits[0][2]

                    if i == j:
                        # 根据条件合并单一限制下的乘积项
                        if x2 == y1 + 1:
                            return Product(self.function, (i, x1, y2))
                        elif x1 == y2 + 1:
                            return Product(self.function, (i, x2, y1))

    return Mul(self, other)


def _nthroot_solve(p, n, prec):
    """
     Helper function for solving nth roots.
     It denests ``p**Rational(1, n)`` using its minimal polynomial.
    """
    from sympy.solvers import solve
    while n % 2 == 0:
        p = sqrtdenest(sqrt(p))
        n = n // 2
    if n == 1:
        return p
    pn = p**Rational(1, n)
    x = Symbol('x')
    f = _minimal_polynomial_sq(p, n, x)
    if f is None:
        return None
    sols = solve(f, x)
    for sol in sols:
        if abs(sol - pn).n() < 1./10**prec:
            sol = sqrtdenest(sol)
            if _mexpand(sol**n) == p:
                return sol


def logcombine(expr, force=False):
    """
    Combine logarithmic expressions according to specified rules.

    - log(x) + log(y) == log(x*y) if both are positive
    - a*log(x) == log(x**a) if x is positive and a is real

    If ``force`` is ``True``, assumptions above will be applied.
    """
    # 返回函数 _bottom_up 的结果，传入参数 expr 和 f
    return _bottom_up(expr, f)
# 简化一个函数及其逆函数的复合表达式
def inversecombine(expr):
    """Simplify the composition of a function and its inverse.

    Explanation
    ===========

    No attention is paid to whether the inverse is a left inverse or a
    right inverse; thus, the result will in general not be equivalent
    to the original expression.

    Examples
    ========

    >>> from sympy.simplify.simplify import inversecombine
    >>> from sympy import asin, sin, log, exp
    >>> from sympy.abc import x
    >>> inversecombine(asin(sin(x)))
    x
    >>> inversecombine(2*log(exp(3*x)))
    6*x
    """

    def f(rv):
        if isinstance(rv, log):
            # 处理 log 函数的情况，如果其参数是 exp 函数或等效形式，则简化为 exp 函数
            if isinstance(rv.args[0], exp) or (rv.args[0].is_Pow and rv.args[0].base == S.Exp1):
                rv = rv.args[0].exp
        elif rv.is_Function and hasattr(rv, "inverse"):
            # 处理其他函数的情况，如果函数有逆函数且符合特定条件，则简化为逆函数的参数
            if (len(rv.args) == 1 and len(rv.args[0].args) == 1 and
               isinstance(rv.args[0], rv.inverse(argindex=1))):
                rv = rv.args[0].args[0]
        if rv.is_Pow and rv.base == S.Exp1:
            # 处理指数函数的情况，如果其指数是 log 函数，则简化为 log 函数的参数
            if isinstance(rv.exp, log):
                rv = rv.exp.args[0]
        return rv

    return _bottom_up(expr, f)


# 简化带有 KroneckerDelta 的表达式
def kroneckersimp(expr):
    """
    Simplify expressions with KroneckerDelta.

    The only simplification currently attempted is to identify multiplicative cancellation:

    Examples
    ========

    >>> from sympy import KroneckerDelta, kroneckersimp
    >>> from sympy.abc import i
    >>> kroneckersimp(1 + KroneckerDelta(0, i) * KroneckerDelta(1, i))
    1
    """
    # 检查两个 KroneckerDelta 函数的参数是否可以相互抵消
    def args_cancel(args1, args2):
        for i1 in range(2):
            for i2 in range(2):
                a1 = args1[i1]
                a2 = args2[i2]
                a3 = args1[(i1 + 1) % 2]
                a4 = args2[(i2 + 1) % 2]
                if Eq(a1, a2) is S.true and Eq(a3, a4) is S.false:
                    return True
        return False

    # 取消乘积形式下的 KroneckerDelta 函数
    def cancel_kronecker_mul(m):
        args = m.args
        deltas = [a for a in args if isinstance(a, KroneckerDelta)]
        for delta1, delta2 in subsets(deltas, 2):
            args1 = delta1.args
            args2 = delta2.args
            if args_cancel(args1, args2):
                return S.Zero * m  # 在某些情况下，返回 0，如 oo 等
        return m

    if not expr.has(KroneckerDelta):
        return expr

    if expr.has(Piecewise):
        expr = expr.rewrite(KroneckerDelta)

    newexpr = expr
    expr = None

    # 用 cancel_kronecker_mul 函数替换表达式中的乘积形式
    while newexpr != expr:
        expr = newexpr
        newexpr = expr.replace(lambda e: isinstance(e, Mul), cancel_kronecker_mul)

    return expr


# 简化贝塞尔类型函数的表达式
def besselsimp(expr):
    """
    Simplify bessel-type functions.

    Explanation
    ===========

    This routine tries to simplify bessel-type functions. Currently it only
    works on the Bessel J and I functions, however. It works by looking at all
    such functions in turn, and eliminating factors of "I" and "-1" (actually
    their polar equivalents) in front of the argument. Then, functions of
    """
    # 尝试简化贝塞尔类型函数的表达式
    # 当前仅适用于 Bessel J 和 I 函数
    # 通过检查所有这些函数，消除参数前的 "I" 和 "-1" 的因子（实际上是它们的极坐标等效形式）
    pass  # 未完成的函数，待补充完整
    """
    # TODO
    # - better algorithm?
    # - simplify (cos(pi*b)*besselj(b,z) - besselj(-b,z))/sin(pi*b) ...
    # - use contiguity relations?
    """

    def replacer(fro, to, factors):
        """
        返回一个替换函数，根据给定的因子集合选择是否替换函数。
        
        Parameters:
        fro : function
            原始函数对象
        to : function
            替换后的函数对象
        factors : set
            包含影响替换决策的因子集合

        Returns:
        function
            根据因子选择替换函数的函数
        """
        factors = set(factors)

        def repl(nu, z):
            """
            替换函数，根据因子决定是否替换函数。

            Parameters:
            nu : object
                函数的第一个参数
            z : object
                函数的第二个参数

            Returns:
            object
                替换后的函数结果
            """
            if factors.intersection(Mul.make_args(z)):
                return to(nu, z)
            return fro(nu, z)
        return repl

    def torewrite(fro, to):
        """
        返回一个重写函数，将给定函数转换为另一种形式。

        Parameters:
        fro : function
            原始函数对象
        to : function
            转换后的函数对象

        Returns:
        function
            重写函数
        """
        def tofunc(nu, z):
            return fro(nu, z).rewrite(to)
        return tofunc

    def tominus(fro):
        """
        返回一个取负函数，对给定函数结果取负。

        Parameters:
        fro : function
            原始函数对象

        Returns:
        function
            取负后的函数
        """
        def tofunc(nu, z):
            return exp(I*pi*nu)*fro(nu, exp_polar(-I*pi)*z)
        return tofunc

    orig_expr = expr

    ifactors = [I, exp_polar(I*pi/2), exp_polar(-I*pi/2)]
    expr = expr.replace(
        besselj, replacer(besselj,
        torewrite(besselj, besseli), ifactors))
    expr = expr.replace(
        besseli, replacer(besseli,
        torewrite(besseli, besselj), ifactors))

    minusfactors = [-1, exp_polar(I*pi)]
    expr = expr.replace(
        besselj, replacer(besselj, tominus(besselj), minusfactors))
    expr = expr.replace(
        besseli, replacer(besseli, tominus(besseli), minusfactors))

    z0 = Dummy('z')

    def expander(fro):
        """
        返回一个扩展函数，根据函数的特定条件扩展给定的函数。

        Parameters:
        fro : function
            原始函数对象

        Returns:
        function
            扩展函数
        """
        def repl(nu, z):
            """
            根据 nu 的值和 z 的类型，选择是否扩展函数。

            Parameters:
            nu : object
                函数的第一个参数
            z : object
                函数的第二个参数

            Returns:
            object
                扩展后的函数结果或原始函数结果
            """
            if (nu % 1) == S.Half:
                return simplify(trigsimp(unpolarify(
                        fro(nu, z0).rewrite(besselj).rewrite(jn).expand(
                            func=True)).subs(z0, z)))
            elif nu.is_Integer and nu > 1:
                return fro(nu, z).expand(func=True)
            return fro(nu, z)
        return repl

    expr = expr.replace(besselj, expander(besselj))
    expr = expr.replace(bessely, expander(bessely))
    expr = expr.replace(besseli, expander(besseli))
    expr = expr.replace(besselk, expander(besselk))
    # 定义内部函数 _bessel_simp_recursion，用于简化表达式中的贝塞尔函数递归处理
    def _bessel_simp_recursion(expr):

        # 定义内部函数 _use_recursion，用于递归处理特定类型的贝塞尔函数
        def _use_recursion(bessel, expr):
            # 使用 lambda 表达式查找表达式中所有贝塞尔函数 bessel 的实例
            while True:
                bessels = expr.find(lambda x: isinstance(x, bessel))
                try:
                    # 对找到的贝塞尔函数实例按照第一个参数的正则排序
                    for ba in sorted(bessels, key=lambda x: re(x.args[0])):
                        a, x = ba.args
                        # 计算 bessel(a+1, x) 和 bessel(a+2, x)
                        bap1 = bessel(a+1, x)
                        bap2 = bessel(a+2, x)
                        # 如果表达式中同时包含 bap1 和 bap2
                        if expr.has(bap1) and expr.has(bap2):
                            # 应用递推关系：expr = 2*(a+1)/x * bap1 - bap2
                            expr = expr.subs(ba, 2*(a+1)/x * bap1 - bap2)
                            break
                    else:
                        return expr  # 如果没有找到可处理的贝塞尔函数实例，返回表达式
                except (ValueError, TypeError):
                    return expr  # 处理异常情况，返回表达式

        # 如果表达式中包含 besselj 函数，则使用 _use_recursion 处理
        if expr.has(besselj):
            expr = _use_recursion(besselj, expr)
        # 如果表达式中包含 bessely 函数，则使用 _use_recursion 处理
        if expr.has(bessely):
            expr = _use_recursion(bessely, expr)
        
        # 返回处理后的表达式
        return expr

    # 对表达式进行贝塞尔函数简化递归处理
    expr = _bessel_simp_recursion(expr)
    # 如果简化后的表达式不等于原始表达式，则对其进行因式分解
    if expr != orig_expr:
        expr = expr.factor()

    # 返回最终处理后的表达式
    return expr
def nthroot(expr, n, max_len=4, prec=15):
    """
    Compute a real nth-root of a sum of surds.

    Parameters
    ==========

    expr : sum of surds
        输入的代数表达式，包含多个根式的和
    n : integer
        整数，表示要计算的根的次数
    max_len : maximum number of surds passed as constants to ``nsimplify``
        作为常量传递给 ``nsimplify`` 的最大根式数量
    prec : integer
        精度，用于计算的精确度

    Algorithm
    =========

    First ``nsimplify`` is used to get a candidate root; if it is not a
    root the minimal polynomial is computed; the answer is one of its
    roots.

    Examples
    ========

    >>> from sympy.simplify.simplify import nthroot
    >>> from sympy import sqrt
    >>> nthroot(90 + 34*sqrt(7), 3)
    sqrt(7) + 3

    """
    expr = sympify(expr)  # 将输入的表达式转换为SymPy表达式
    n = sympify(n)  # 将输入的根的次数转换为SymPy表达式
    p = expr**Rational(1, n)  # 计算 expr 的 n 次根作为候选解
    if not n.is_integer:  # 如果 n 不是整数，则直接返回候选解
        return p
    if not _is_sum_surds(expr):  # 如果表达式不是根式的和，则直接返回候选解
        return p
    surds = []
    coeff_muls = [x.as_coeff_Mul() for x in expr.args]
    for x, y in coeff_muls:
        if not x.is_rational:  # 如果系数不是有理数，则直接返回候选解
            return p
        if y is S.One:  # 如果指数是1，则跳过
            continue
        if not (y.is_Pow and y.exp == S.Half and y.base.is_integer):  # 如果指数不是1/2或底数不是整数，则直接返回候选解
            return p
        surds.append(y)
    surds.sort()  # 对根式进行排序
    surds = surds[:max_len]  # 取最多 max_len 个根式
    if expr < 0 and n % 2 == 1:  # 如果表达式小于0且 n 为奇数
        p = (-expr)**Rational(1, n)  # 计算负数的 n 次根
        a = nsimplify(p, constants=surds)  # 使用 nsimplify 函数计算简化后的结果
        res = a if _mexpand(a**n) == _mexpand(-expr) else p  # 如果简化后的结果的 n 次幂等于原表达式的相反数，则返回 a，否则返回 p
        return -res  # 返回负数根
    a = nsimplify(p, constants=surds)  # 使用 nsimplify 函数计算简化后的结果
    if _mexpand(a) is not _mexpand(p) and _mexpand(a**n) == _mexpand(expr):  # 如果简化后的结果不等于 p 且简化后的结果的 n 次幂等于原表达式
        return _mexpand(a)  # 返回简化后的结果
    expr = _nthroot_solve(expr, n, prec)  # 使用 _nthroot_solve 函数进行根的求解
    if expr is None:  # 如果求解失败，则返回候选解
        return p
    return expr  # 返回求解得到的根


def nsimplify(expr, constants=(), tolerance=None, full=False, rational=None,
    rational_conversion='base10'):
    """
    Find a simple representation for a number or, if there are free symbols or
    if ``rational=True``, then replace Floats with their Rational equivalents. If
    no change is made and rational is not False then Floats will at least be
    converted to Rationals.

    Explanation
    ===========

    For numerical expressions, a simple formula that numerically matches the
    given numerical expression is sought (and the input should be possible
    to evalf to a precision of at least 30 digits).

    Optionally, a list of (rationally independent) constants to
    include in the formula may be given.

    A lower tolerance may be set to find less exact matches. If no tolerance
    is given then the least precise value will set the tolerance (e.g. Floats
    default to 15 digits of precision, so would be tolerance=10**-15).

    With ``full=True``, a more extensive search is performed
    (this is useful to find simpler numbers when the tolerance
    is set low).

    When converting to rational, if rational_conversion='base10' (the default), then
    convert floats to rationals using their base-10 (string) representation.
    When rational_conversion='exact' it uses the exact, base-2 representation.

    Examples
    ========

    >>> from sympy import nsimplify, sqrt, GoldenRatio, exp, I, pi

    """
    # To be continued as per the next section...
    # 使用 nsimplify 函数对表达式进行符号化简，可以传入 GoldenRatio 常数用于简化
    >>> nsimplify(4/(1+sqrt(5)), [GoldenRatio])
    # 返回简化后的表达式
    -2 + 2*GoldenRatio

    # 对复杂表达式进行符号化简，不指定特定常数
    >>> nsimplify((1/(exp(3*pi*I/5)+1)))
    # 返回简化后的表达式
    1/2 - I*sqrt(sqrt(5)/10 + 1/4)

    # 对幂次复数进行符号化简，可以传入 pi 用于简化
    >>> nsimplify(I**I, [pi])
    # 返回简化后的表达式
    exp(-pi/2)

    # 使用指定的容差对 pi 进行符号化简
    >>> nsimplify(pi, tolerance=0.01)
    # 返回简化后的表达式
    22/7

    # 将浮点数转换为分数形式，指定精确转换
    >>> nsimplify(0.333333333333333, rational=True, rational_conversion='exact')
    # 返回转换后的分数形式
    6004799503160655/18014398509481984

    # 将浮点数转换为分数形式，不指定精确转换
    >>> nsimplify(0.333333333333333, rational=True)
    # 返回转换后的分数形式
    1/3

    # 查看相关函数文档链接
    See Also
    ========
    # 指向 sympy.core.function.nfloat 函数

    """
    # 尝试将表达式转换为整数，并进行符号化简
    try:
        return sympify(as_int(expr))
    except (TypeError, ValueError):
        pass
    # 将表达式转换为符号表达，并替换浮点数常量
    expr = sympify(expr).xreplace({
        Float('inf'): S.Infinity,
        Float('-inf'): S.NegativeInfinity,
        })
    # 如果表达式为无穷大或负无穷大，则直接返回
    if expr is S.Infinity or expr is S.NegativeInfinity:
        return expr
    # 如果设置了 rational 或表达式包含自由符号，则转换为有理数
    if rational or expr.free_symbols:
        return _real_to_rational(expr, tolerance, rational_conversion)

    # SymPy 默认对有理数的容差为 15；其他数字可能有较低的设置，因此选择最大的容差
    if tolerance is None:
        # 计算表达式中浮点数的精度
        tolerance = 10**-min([15] +
             [mpmath.libmp.libmpf.prec_to_dps(n._prec)
             for n in expr.atoms(Float)])
    # 设置精度值
    prec = 30
    bprec = int(prec*3.33)

    constants_dict = {}
    # 遍历给定的常数列表，计算其数值并存储在字典中
    for constant in constants:
        constant = sympify(constant)
        v = constant.evalf(prec)
        if not v.is_Float:
            raise ValueError("constants must be real-valued")
        constants_dict[str(constant)] = v._to_mpmath(bprec)

    # 对表达式进行指定精度的求值
    exprval = expr.evalf(prec, chop=True)
    re, im = exprval.as_real_imag()

    # 安全检查确保表达式求值为数值
    if not (re.is_Number and im.is_Number):
        return expr

    # 定义实数化简函数
    def nsimplify_real(x):
        orig = mpmath.mp.dps
        xv = x._to_mpmath(bprec)
        try:
            # 对于简单的分数，如果没有指定容差或完整模式，低精度也可以接受
            if not (tolerance or full):
                mpmath.mp.dps = 15
                rat = mpmath.pslq([xv, 1])
                if rat is not None:
                    return Rational(-int(rat[1]), int(rat[0]))
            # 设置精度
            mpmath.mp.dps = prec
            # 根据给定常数字典，识别数值并简化
            newexpr = mpmath.identify(xv, constants=constants_dict,
                tol=tolerance, full=full)
            if not newexpr:
                raise ValueError
            if full:
                newexpr = newexpr[0]
            expr = sympify(newexpr)
            if x and not expr:  # 不让 x 变为 0
                raise ValueError
            if expr.is_finite is False and xv not in [mpmath.inf, mpmath.ninf]:
                raise ValueError
            return expr
        finally:
            # 返回前恢复原始精度
            mpmath.mp.dps = orig
    # 尝试对实部和虚部进行简化处理
    try:
        # 如果实部不为空，则调用 nsimplify_real 函数进行简化
        if re:
            re = nsimplify_real(re)
        # 如果虚部不为空，则调用 nsimplify_real 函数进行简化
        if im:
            im = nsimplify_real(im)
    # 如果出现 ValueError 异常
    except ValueError:
        # 如果 rational 参数为 None，则调用 _real_to_rational 函数将表达式转换为有理数表示
        if rational is None:
            return _real_to_rational(expr, rational_conversion=rational_conversion)
        # 否则直接返回原始表达式
        return expr

    # 计算复数表达式 rv，实部加上虚部乘以虚数单位
    rv = re + im*S.ImaginaryUnit

    # 如果 rv 与原始表达式 expr 不同，或者 rational 显式设置为 False
    # 返回 rv，表示进行了变化或者明确不要有理数表示
    if rv != expr or rational is False:
        return rv
    # 否则调用 _real_to_rational 函数将表达式转换为有理数表示并返回
    return _real_to_rational(expr, rational_conversion=rational_conversion)
# 将表达式中的实数替换为有理数的函数
def _real_to_rational(expr, tolerance=None, rational_conversion='base10'):
    # 将输入的表达式转换为符号表达式（Symbolic expression）
    expr = _sympify(expr)
    # 定义正无穷大
    inf = Float('inf')
    # p表示处理后的表达式，默认为输入表达式
    p = expr
    # 用于存储替换规则的字典
    reps = {}
    # 用于减少有理数分母的数值
    reduce_num = None
    # 如果指定了容差且容差小于1，则计算减少分母的数值
    if tolerance is not None and tolerance < 1:
        reduce_num = ceiling(1/tolerance)
    # 遍历表达式中的所有浮点数
    for fl in p.atoms(Float):
        key = fl
        # 如果指定了减少分母的数值，使用有理数来近似浮点数
        if reduce_num is not None:
            r = Rational(fl).limit_denominator(reduce_num)
        # 如果指定了容差且容差大于等于1且浮点数不是整数，则根据容差近似浮点数
        elif (tolerance is not None and tolerance >= 1 and
              fl.is_Integer is False):
            r = Rational(tolerance*round(fl/tolerance)).limit_denominator(int(tolerance))
        else:
            # 如果是精确模式，则直接转换为有理数
            if rational_conversion == 'exact':
                r = Rational(fl)
                reps[key] = r
                continue
            # 如果不是基于十进制的转换方式，则抛出错误
            elif rational_conversion != 'base10':
                raise ValueError("rational_conversion must be 'base10' or 'exact'")
            
            # 使用精确模式近似浮点数，避免使用有理数
            r = nsimplify(fl, rational=False)
            # 例如，log(3).n() -> log(3)而不是有理数
            if fl and not r:
                r = Rational(fl)
            elif not r.is_Rational:
                # 如果不是有理数，则根据特定情况进行处理
                if fl in (inf, -inf):
                    r = S.ComplexInfinity
                elif fl < 0:
                    fl = -fl
                    d = Pow(10, int(mpmath.log(fl)/mpmath.log(10)))
                    r = -Rational(str(fl/d))*d
                elif fl > 0:
                    d = Pow(10, int(mpmath.log(fl)/mpmath.log(10)))
                    r = Rational(str(fl/d))*d
                else:
                    r = S.Zero
        # 将替换规则存入字典
        reps[key] = r
    # 使用替换规则对表达式进行替换，并启用同时替换模式
    return p.subs(reps, simultaneous=True)


# 清除表达式中有理数加法和乘法系数的函数
def clear_coefficients(expr, rhs=S.Zero):
    """
    返回 `p, r`，其中 `p` 是通过简单剥离 `expr` 中有理数加法和乘法系数得到的表达式（不进行化简）。
    需要应用于 `rhs` 的操作将作用于 `rhs` 并返回为 `r`。

    Examples
    ========

    >>> from sympy.simplify.simplify import clear_coefficients
    >>> from sympy.abc import x, y
    >>> from sympy import Dummy
    >>> expr = 4*y*(6*x + 3)
    >>> clear_coefficients(expr - 2)
    (y*(2*x + 1), 1/6)

    When solving 2 or more expressions like `expr = a`,
    `expr = b`, etc..., it is advantageous to provide a Dummy symbol
    for `rhs` and  simply replace it with `a`, `b`, etc... in `r`.

    >>> rhs = Dummy('rhs')
    """
    # 初始化变量 `was` 为 None
    was = None
    # 获取表达式中的自由符号集合
    free = expr.free_symbols
    # 如果表达式是有理数，则返回 (0, rhs - expr)
    if expr.is_Rational:
        return (S.Zero, rhs - expr)
    # 当表达式非空且上次迭代结果不等于当前表达式时，执行循环
    while expr and was != expr:
        # 记录当前表达式
        was = expr
        # 如果有自由符号，则将表达式分解为内容和基本部分
        m, expr = (
            expr.as_content_primitive()
            if free else
            factor_terms(expr).as_coeff_Mul(rational=True))
        # 更新 rhs
        rhs /= m
        # 将表达式分解为常数和余项
        c, expr = expr.as_coeff_Add(rational=True)
        # 更新 rhs
        rhs -= c
    # 简化表达式中的符号
    expr = signsimp(expr, evaluate=False)
    # 如果表达式可以提取负号，则取反表达式及其 rhs
    if expr.could_extract_minus_sign():
        expr = -expr
        rhs = -rhs
    # 返回更新后的表达式和 rhs
    return expr, rhs
def nc_simplify(expr, deep=True):
    '''
    Simplify a non-commutative expression composed of multiplication
    and raising to a power by grouping repeated subterms into one power.
    Priority is given to simplifications that give the fewest number
    of arguments in the end (for example, in a*b*a*b*c*a*b*c simplifying
    to (a*b)**2*c*a*b*c gives 5 arguments while a*b*(a*b*c)**2 has 3).
    If ``expr`` is a sum of such terms, the sum of the simplified terms
    is returned.

    Keyword argument ``deep`` controls whether or not subexpressions
    nested deeper inside the main expression are simplified. See examples
    below. Setting `deep` to `False` can save time on nested expressions
    that do not need simplifying on all levels.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.simplify.simplify import nc_simplify
    >>> a, b, c = symbols("a b c", commutative=False)
    >>> nc_simplify(a*b*a*b*c*a*b*c)
    a*b*(a*b*c)**2
    >>> expr = a**2*b*a**4*b*a**4
    >>> nc_simplify(expr)
    a**2*(b*a**4)**2
    >>> nc_simplify(a*b*a*b*c**2*(a*b)**2*c**2)
    ((a*b)**2*c**2)**2
    >>> nc_simplify(a*b*a*b + 2*a*c*a**2*c*a**2*c*a)
    (a*b)**2 + 2*(a*c*a)**3
    >>> nc_simplify(b**-1*a**-1*(a*b)**2)
    a*b
    >>> nc_simplify(a**-1*b**-1*c*a)
    (b*a)**(-1)*c*a
    >>> expr = (a*b*a*b)**2*a*c*a*c
    >>> nc_simplify(expr)
    (a*b)**4*(a*c)**2
    >>> nc_simplify(expr, deep=False)
    (a*b*a*b)**2*(a*c)**2

    '''
    if isinstance(expr, MatrixExpr):
        # 如果表达式是矩阵表达式，展开并禁止扩展
        expr = expr.doit(inv_expand=False)
        _Add, _Mul, _Pow, _Symbol = MatAdd, MatMul, MatPow, MatrixSymbol
    else:
        _Add, _Mul, _Pow, _Symbol = Add, Mul, Pow, Symbol

    # =========== Auxiliary functions ========================
    def _overlaps(args):
        # 计算一个列表的列表 m，使得 m[i][j] 包含 args[:i+1] 和 args[i+1+j:] 之间所有可能的重叠长度
        # 重叠是指前缀的后缀与后缀的前缀匹配
        # 例如，对于表达式 c*a*b*a*b*a*b*a*b，m[3][0] 包含 c*a*b*a*b 与 a*b*a*b 的重叠长度
        # 重叠包括 a*b*a*b、a*b 和空字符串，因此 m[3][0]=[4,2,0]
        # 记录所有重叠而不仅仅是最长的重叠，因为这些信息有助于计算其他重叠的长度
        m = [[([1, 0] if a == args[0] else [0]) for a in args[1:]]]
        for i in range(1, len(args)):
            overlaps = []
            j = 0
            for j in range(len(args) - i - 1):
                overlap = []
                for v in m[i-1][j+1]:
                    if j + i + 1 + v < len(args) and args[i] == args[j+i+1+v]:
                        overlap.append(v + 1)
                overlap += [0]
                overlaps.append(overlap)
            m.append(overlaps)
        return m
    def _reduce_inverses(_args):
        # 替换连续的负幂为正幂的乘积的倒数，例如 a**-1*b**-1*c
        # 简化为 (a*b)**-1*c；
        # 返回新的参数列表和其中负幂的数量（inv_tot）
        inv_tot = 0  # 总倒数数目
        inverses = []  # 倒数列表
        args = []  # 新的参数列表
        for arg in _args:
            if isinstance(arg, _Pow) and arg.args[1].is_extended_negative:
                inverses = [arg**-1] + inverses  # 将负幂添加到倒数列表
                inv_tot += 1  # 倒数数目加一
            else:
                if len(inverses) == 1:
                    args.append(inverses[0]**-1)  # 添加单个倒数的倒数
                elif len(inverses) > 1:
                    args.append(_Pow(_Mul(*inverses), -1))  # 添加多个倒数的倒数
                    inv_tot -= len(inverses) - 1
                inverses = []  # 清空倒数列表
                args.append(arg)  # 添加非负幂参数
        if inverses:
            args.append(_Pow(_Mul(*inverses), -1))  # 添加最后的倒数
            inv_tot -= len(inverses) - 1
        return inv_tot, tuple(args)  # 返回倒数总数和新的参数元组

    def get_score(s):
        # 计算表达式 s 的参数数量
        # （包括嵌套表达式），但忽略指数
        if isinstance(s, _Pow):
            return get_score(s.args[0])
        elif isinstance(s, (_Add, _Mul)):
            return sum(get_score(a) for a in s.args)
        return 1

    def compare(s, alt_s):
        # 比较两个可能的简化结果，并返回“更好”的一个
        if s != alt_s and get_score(alt_s) < get_score(s):
            return alt_s
        return s

    # ========================================================

    if not isinstance(expr, (_Add, _Mul, _Pow)) or expr.is_commutative:
        return expr  # 如果表达式是可交换的或不是加、乘、幂运算，直接返回表达式
    args = expr.args[:]  # 复制表达式的参数列表

    if isinstance(expr, _Pow):
        if deep:
            return _Pow(nc_simplify(args[0]), args[1]).doit()  # 深度简化幂表达式
        else:
            return expr  # 否则直接返回幂表达式
    elif isinstance(expr, _Add):
        return _Add(*[nc_simplify(a, deep=deep) for a in args]).doit()  # 简化加法表达式
    else:
        # 获取非交换部分
        c_args, args = expr.args_cnc()
        com_coeff = Mul(*c_args)  # 计算非交换部分的乘积系数
        if not equal_valued(com_coeff, 1):
            return com_coeff * nc_simplify(expr / com_coeff, deep=deep)  # 如果乘积系数不为1，则简化后返回

    inv_tot, args = _reduce_inverses(args)  # 减少参数中的倒数

    # 如果大部分参数是负的，使用表达式的倒数，
    # 例如 a**-1*b*a**-1*c**-1 最终将变为 (c*a*b**-1*a)**-1，可以使用 c*a*b**-1*a
    invert = False
    if inv_tot > len(args) / 2:
        invert = True
        args = [a**-1 for a in args[::-1]]  # 反转参数并取倒数

    if deep:
        args = tuple(nc_simplify(a) for a in args)  # 深度简化参数

    m = _overlaps(args)  # 计算参数的重叠部分

    # simps 将是 {subterm: end}，其中 `end` 是重复子项的结束索引；
    # 这是为了避免处理已经考虑过的较长子项
    simps = {}

    post = 1  # 后缀
    pre = 1  # 前缀

    # 简化系数是参数中的数字
    # 定义最大简化系数，初始为0
    max_simp_coeff = 0
    # 初始化简化信息变量为None，用于存储未来的简化信息
    simp = None

    # 如果存在简化信息
    if simp:
        # 对 simp[1] 进行深度简化，然后构建一个 _Pow 对象
        subterm = _Pow(nc_simplify(simp[1], deep=deep), simp[2])
        # 对前 simp[0] 个参数进行简化后相乘，再乘以 pre
        pre = nc_simplify(_Mul(*args[:simp[0]]) * pre, deep=deep)
        # 对后 simp[3] 个参数进行简化后相乘，再乘以 post
        post = post * nc_simplify(_Mul(*args[simp[3]:]), deep=deep)
        # 更新 simp 为 pre * subterm * post
        simp = pre * subterm * post
        # 如果 pre 或 post 不等于1，则可能存在新的简化机会，但无需递归处理参数
        if pre != 1 or post != 1:
            simp = nc_simplify(simp, deep=False)
    else:
        # 如果没有简化信息，则直接将 args 求乘积作为简化结果
        simp = _Mul(*args)

    # 如果需要反转（invert为True），则对 simp 取倒数
    if invert:
        simp = _Pow(simp, -1)

    # 如果 expr 不是 MatrixExpr 类型，则尝试对其进行非交换因子化
    if not isinstance(expr, MatrixExpr):
        # 对 expr 进行非交换因子化
        f_expr = factor_nc(expr)
        # 如果因子化后的表达式与原表达式不同，则尝试深度简化
        if f_expr != expr:
            alt_simp = nc_simplify(f_expr, deep=deep)
            # 比较简化后的结果，选择更优的
            simp = compare(simp, alt_simp)
    else:
        # 如果 expr 是 MatrixExpr 类型，则尝试对其进行执行操作
        simp = simp.doit(inv_expand=False)

    # 返回简化后的表达式结果
    return simp
# dotprodsimp 函数：对乘积求和的简化，旨在减少在矩阵乘法或类似操作期间出现的表达式膨胀。
# 仅适用于代数表达式，并不递归到非。
def dotprodsimp(expr, withsimp=False):
    """Simplification for a sum of products targeted at the kind of blowup that
    occurs during summation of products. Intended to reduce expression blowup
    during matrix multiplication or other similar operations. Only works with
    algebraic expressions and does not recurse into non.

    Parameters
    ==========

    withsimp : bool, optional
        Specifies whether a flag should be returned along with the expression
        to indicate roughly whether simplification was successful. It is used
        in ``MatrixArithmetic._eval_pow_by_recursion`` to avoid attempting to
        simplify an expression repetitively which does not simplify.
    """
    # 定义一个函数，用于计算代数操作的数量，并优化以避免递归进入非代数参数，与 core.function.count_ops 的功能不同。
    def count_ops_alg(expr):
        """Optimized count algebraic operations with no recursion into
        non-algebraic args that ``core.function.count_ops`` does. Also returns
        whether rational functions may be present according to negative
        exponents of powers or non-number fractions.

        Returns
        =======

        ops, ratfunc : int, bool
            ``ops`` is the number of algebraic operations starting at the top
            level expression (not recursing into non-alg children). ``ratfunc``
            specifies whether the expression MAY contain rational functions
            which ``cancel`` MIGHT optimize.
        """
        
        # 初始化操作数和表达式列表
        ops = 0
        args = [expr]
        ratfunc = False
        
        # 循环处理表达式列表
        while args:
            a = args.pop()
            
            # 如果当前元素不是 SymPy 的基本对象，则跳过
            if not isinstance(a, Basic):
                continue
            
            # 处理有理数情况
            if a.is_Rational:
                # 判断是否为非整数有理数
                if a is not S.One:  # -1/3 = NEG + DIV
                    ops += bool(a.p < 0) + bool(a.q != 1)
            
            # 处理乘法表达式
            elif a.is_Mul:
                # 判断是否可能提取负号
                if a.could_extract_minus_sign():
                    ops += 1
                    if a.args[0] is S.NegativeOne:
                        a = a.as_two_terms()[1]
                    else:
                        a = -a
                
                # 分离出分子和分母
                n, d = fraction(a)
                
                # 处理分子为整数的情况
                if n.is_Integer:
                    ops += 1 + bool(n < 0)
                    args.append(d)  # 分母可能不是乘法表达式但可以是加法表达式
                
                # 处理分母不为1的情况
                elif d is not S.One:
                    if not d.is_Integer:
                        args.append(d)
                        ratfunc = True
                    
                    ops += 1
                    args.append(n)  # 可能是负乘法表达式
                
                else:
                    ops += len(a.args) - 1
                    args.extend(a.args)
            
            # 处理加法表达式
            elif a.is_Add:
                laargs = len(a.args)
                negs = 0
                
                # 遍历加法表达式的每一项
                for ai in a.args:
                    if ai.could_extract_minus_sign():
                        negs += 1
                        ai = -ai
                    args.append(ai)
                
                ops += laargs - (negs != laargs)  # -x - y = NEG + SUB
            
            # 处理幂次方表达式
            elif a.is_Pow:
                ops += 1
                args.append(a.base)
                
                # 判断是否可能包含有理函数
                if not ratfunc:
                    ratfunc = a.exp.is_negative is not False
        
        # 返回计算得到的操作数和是否可能包含有理函数的布尔值
        return ops, ratfunc
    def nonalg_subs_dummies(expr, dummies):
        """Substitute dummy variables for non-algebraic expressions to avoid
        evaluation of non-algebraic terms that ``polys.polytools.cancel`` does.
        """

        # 如果表达式没有参数（即已经是最基本的表达式），直接返回表达式本身
        if not expr.args:
            return expr

        # 如果表达式是加法、乘法或幂运算
        if expr.is_Add or expr.is_Mul or expr.is_Pow:
            args = None

            # 遍历表达式的每个参数
            for i, a in enumerate(expr.args):
                # 递归调用自身，替换表达式中的非代数项为虚拟变量
                c = nonalg_subs_dummies(a, dummies)

                # 如果替换后的结果与原参数相同，继续下一个参数的处理
                if c is a:
                    continue

                # 如果第一次发现需要修改参数列表，创建参数列表的副本
                if args is None:
                    args = list(expr.args)

                # 更新修改后的参数值
                args[i] = c

            # 如果参数列表没有被修改过，返回原始表达式
            if args is None:
                return expr

            # 使用修改后的参数列表构建新的表达式
            return expr.func(*args)

        # 对于其他情况，将表达式关联到对应的虚拟变量，并返回
        return dummies.setdefault(expr, Dummy())

    simplified = False  # 表达式是否被简化过，实际上表示可以再次进行简化

    # 检查表达式是否属于基本类型并且是加法、乘法或幂运算
    if isinstance(expr, Basic) and (expr.is_Add or expr.is_Mul or expr.is_Pow):
        # 对表达式进行展开，深度展开并且不应用模数、不展开指数、不展开基数、不应用对数、不应用多项式展开、不应用基本展开
        expr2 = expr.expand(deep=True, modulus=None, power_base=False,
                            power_exp=False, mul=True, log=False, multinomial=True, basic=False)

        # 如果展开后的表达式与原表达式不相等
        if expr2 != expr:
            expr = expr2  # 更新表达式为展开后的表达式
            simplified = True  # 标记为已简化过

        # 计算代数操作的数量和有理函数
        exprops, ratfunc = count_ops_alg(expr)

        # 如果代数操作的数量大于等于6（经验测试的昂贵简化的截止值）
        if exprops >= 6:
            if ratfunc:
                dummies = {}  # 初始化虚拟变量的字典
                expr2 = nonalg_subs_dummies(expr, dummies)  # 对表达式进行非代数项替换

                # 如果替换后的表达式与原表达式相同，或者替换后的表达式的代数操作数量仍大于等于6
                if expr2 is expr or count_ops_alg(expr2)[0] >= 6:
                    expr3 = cancel(expr2)  # 对替换后的表达式进行取消操作

                    # 如果取消操作后的表达式与替换前不同
                    if expr3 != expr2:
                        # 用虚拟变量字典中的值替换表达式中的虚拟变量，并更新表达式
                        expr = expr3.subs([(d, e) for e, d in dummies.items()])
                        simplified = True  # 标记为已简化过

        # 非常特殊的情况：x/(x-1) - 1/(x-1) -> 1
        elif (exprops == 5 and expr.is_Add and expr.args[0].is_Mul and
              expr.args[1].is_Mul and expr.args[0].args[-1].is_Pow and
              expr.args[1].args[-1].is_Pow and
              expr.args[0].args[-1].exp is S.NegativeOne and
              expr.args[1].args[-1].exp is S.NegativeOne):

            expr2 = together(expr)  # 对表达式进行合并操作
            expr2ops = count_ops_alg(expr2)[0]  # 计算合并后的表达式的代数操作数量

            # 如果合并后的表达式的操作数量小于原表达式
            if expr2ops < exprops:
                expr = expr2  # 更新表达式为合并后的表达式
                simplified = True  # 标记为已简化过

        else:
            simplified = True  # 其他情况下标记为已简化过

    # 根据参数决定返回结果，是否返回简化标记
    return (expr, simplified) if withsimp else expr
# 使用deprecated装饰器标记函数_bottom_up，指出其已废弃
bottom_up = deprecated(
    """
    使用 sympy.simplify.simplify 子模块中的 bottom_up 已被弃用。

    相反，应该使用顶层 sympy 命名空间中的 bottom_up，例如：

        sympy.bottom_up
    """,
    deprecated_since_version="1.10",  # 自版本1.10起废弃
    active_deprecations_target="deprecated-traversal-functions-moved",  # 活跃的废弃迁移目标
)(_bottom_up)  # 应用deprecated装饰器到_bottom_up函数

# XXX: 此函数实际上应该是私有 API，或者应该在顶层 sympy/__init__.py 中导出
# 使用deprecated装饰器标记函数_walk，指出其已废弃
walk = deprecated(
    """
    使用 sympy.simplify.simplify 子模块中的 walk 已被弃用。

    相反，应该使用 sympy.core.traversal.walk 中的 walk。
    """,
    deprecated_since_version="1.10",  # 自版本1.10起废弃
    active_deprecations_target="deprecated-traversal-functions-moved",  # 活跃的废弃迁移目标
)(_walk)  # 应用deprecated装饰器到_walk函数
```