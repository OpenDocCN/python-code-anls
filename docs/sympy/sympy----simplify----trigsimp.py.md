# `D:\src\scipysrc\sympy\sympy\simplify\trigsimp.py`

```
from collections import defaultdict
from functools import reduce

from sympy.core import (sympify, Basic, S, Expr, factor_terms,
                        Mul, Add, bottom_up)
# 导入缓存装饰器函数
from sympy.core.cache import cacheit
# 导入函数计算操作数数量和展开函数等
from sympy.core.function import (count_ops, _mexpand, FunctionClass, expand,
                                 expand_mul, _coeff_isneg, Derivative)
# 导入复数和整数类型
from sympy.core.numbers import I, Integer
# 导入整数最大公约数函数
from sympy.core.intfunc import igcd
# 导入节点排序函数
from sympy.core.sorting import _nodes
# 导入符号和通配符创建函数
from sympy.core.symbol import Dummy, symbols, Wild
# 导入支持符号整数的外部库
from sympy.external.gmpy import SYMPY_INTS
# 导入常见数学函数
from sympy.functions import sin, cos, exp, cosh, tanh, sinh, tan, cot, coth
# 导入反正切二元函数
from sympy.functions import atan2
# 导入双曲函数类和三角函数类
from sympy.functions.elementary.hyperbolic import HyperbolicFunction
from sympy.functions.elementary.trigonometric import TrigonometricFunction
# 导入多项式处理函数
from sympy.polys import Poly, factor, cancel, parallel_poly_from_expr
# 导入整数环
from sympy.polys.domains import ZZ
# 导入多项式异常
from sympy.polys.polyerrors import PolificationFailed
# 导入格罗布纳基础算法函数
from sympy.polys.polytools import groebner
# 导入公共子表达式消除函数
from sympy.simplify.cse_main import cse
# 导入恒等式策略和贪婪策略
from sympy.strategies.core import identity
from sympy.strategies.tree import greedy
# 导入迭代工具函数
from sympy.utilities.iterables import iterable
# 导入调试工具函数
from sympy.utilities.misc import debug

def trigsimp_groebner(expr, hints=[], quick=False, order="grlex",
                      polynomial=False):
    """
    Simplify trigonometric expressions using a groebner basis algorithm.

    Explanation
    ===========

    This routine takes a fraction involving trigonometric or hyperbolic
    expressions, and tries to simplify it. The primary metric is the
    total degree. Some attempts are made to choose the simplest possible
    expression of the minimal degree, but this is non-rigorous, and also
    very slow (see the ``quick=True`` option).

    If ``polynomial`` is set to True, instead of simplifying numerator and
    denominator together, this function just brings numerator and denominator
    into a canonical form. This is much faster, but has potentially worse
    results. However, if the input is a polynomial, then the result is
    guaranteed to be an equivalent polynomial of minimal degree.

    The most important option is hints. Its entries can be any of the
    following:

    - a natural number
    - a function
    - an iterable of the form (func, var1, var2, ...)
    - anything else, interpreted as a generator

    A number is used to indicate that the search space should be increased.
    A function is used to indicate that said function is likely to occur in a
    simplified expression.
    An iterable is used indicate that func(var1 + var2 + ...) is likely to
    occur in a simplified .
    An additional generator also indicates that it is likely to occur.
    (See examples below).

    This routine carries out various computationally intensive algorithms.
    The option ``quick=True`` can be used to suppress one particularly slow
    step (at the expense of potentially more complicated results, but never at
    """
    # 函数主体开始
    pass
    # 函数主体结束
    # TODO
    # 需要做的工作：
    #  - 通过将所有函数替换为我们可以处理的函数来预处理
    #  - 可选地使用 cot 替代 tan
    #  - 更智能的提示。
    #     例如，如果理想很小，而我们有 sin(x)，sin(y)，自动添加 sin(x + y) ...？
    #  - 代数数字 ...
    #  - 最低次数的表达式没有得到正确区分
    #    例如 1 - sin(x)**2
    #  - 我们可以尝试聪明地排序生成器，以影响出现在商基础中的单项式

    # THEORY
    # ------
    # Ratsimpmodprime 可以用于在素理想模下“简化”有理函数。
    # "简化" 主要意味着找到一个等价的、总次数较低的表达式。
    #
    # 我们打算用这个方法来简化三角函数。为此，我们需要决定：
    # (a) 使用哪个环，并且
    # (b) 在哪个理想模下简化。实际上，
    # (a) 意味着确定一组“生成器” a, b, c, ...，使得我们想要简化的分式是关于 a, b, c, ... 的有理函数，系数为 ZZ（整数）。
    # (b) 意味着我们必须决定对生成器施加什么关系。存在两个实际问题：
    #   (1) 理想必须是*素理想*（一个技术术语）。
    #   (2) The relations have to be polynomials in the generators.
    #
    # We typically have two kinds of generators:
    # - trigonometric expressions, like sin(x), cos(5*x), etc
    # - "everything else", like gamma(x), pi, etc.
    #
    # Since this function is trigsimp, we will concentrate on what to do with
    # trigonometric expressions. We can also simplify hyperbolic expressions,
    # but the extensions should be clear.
    #
    # One crucial point is that all *other* generators really should behave
    # like indeterminates. In particular if (say) "I" is one of them, then
    # in fact I**2 + 1 = 0 and we may and will compute non-sensical
    # expressions. However, we can work with a dummy and add the relation
    # I**2 + 1 = 0 to our ideal, then substitute back in the end.
    #
    # Now regarding trigonometric generators. We split them into groups,
    # according to the argument of the trigonometric functions. We want to
    # organise this in such a way that most trigonometric identities apply in
    # the same group. For example, given sin(x), cos(2*x) and cos(y), we would
    # group as [sin(x), cos(2*x)] and [cos(y)].
    #
    # Our prime ideal will be built in three steps:
    # (1) For each group, compute a "geometrically prime" ideal of relations.
    #     Geometrically prime means that it generates a prime ideal in
    #     CC[gens], not just ZZ[gens].
    # (2) Take the union of all the generators of the ideals for all groups.
    #     By the geometric primality condition, this is still prime.
    # (3) Add further inter-group relations which preserve primality.
    #
    # Step (1) works as follows. We will isolate common factors in the
    # argument, so that all our generators are of the form sin(n*x), cos(n*x)
    # or tan(n*x), with n an integer. Suppose first there are no tan terms.
    # The ideal [sin(x)**2 + cos(x)**2 - 1] is geometrically prime, since
    # X**2 + Y**2 - 1 is irreducible over CC.
    # Now, if we have a generator sin(n*x), than we can, using trig identities,
    # express sin(n*x) as a polynomial in sin(x) and cos(x). We can add this
    # relation to the ideal, preserving geometric primality, since the quotient
    # ring is unchanged.
    # Thus we have treated all sin and cos terms.
    # For tan(n*x), we add a relation tan(n*x)*cos(n*x) - sin(n*x) = 0.
    # (This requires of course that we already have relations for cos(n*x) and
    # sin(n*x).) It is not obvious, but it seems that this preserves geometric
    # primality.
    # XXX A real proof would be nice. HELP!
    #     Sketch that <S**2 + C**2 - 1, C*T - S> is a prime ideal of
    #     CC[S, C, T]:
    #     - it suffices to show that the projective closure in CP**3 is
    #       irreducible
    #     - using the half-angle substitutions, we can express sin(x), tan(x),
    #       cos(x) as rational functions in tan(x/2)
    #     - from this, we get a rational map from CP**1 to our curve
    # 定义函数 parse_hints，将提示分解为 (n, funcs, iterables, gens) 四个部分
    def parse_hints(hints):
        """Split hints into (n, funcs, iterables, gens)."""
        # 默认 n 为 1，funcs、iterables、gens 初始化为空列表
        n = 1
        funcs, iterables, gens = [], [], []
        # 遍历提示列表 hints
        for e in hints:
            # 如果元素 e 是整数或者符号类型的整数 (SYMPY_INTS, Integer)，
            # 将 n 更新为 e
            if isinstance(e, (SYMPY_INTS, Integer)):
                n = e
            # 如果 e 是函数类的实例 (FunctionClass)，将 e 添加到 funcs 列表中
            elif isinstance(e, FunctionClass):
                funcs.append(e)
            # 如果 e 是可迭代对象，将其形式为 (e[0], e[1:]) 的元组添加到 iterables 列表中
            elif iterable(e):
                # 备注：我们通过多项式处理，例如 sin(-x) -> -sin(x) -> sin(x)
                # 通过 parallel_poly_from_expr 处理 e[0](x) 和 e[0](Add(*e[1:])) 的表达式
                gens.extend(parallel_poly_from_expr(
                    [e[0](x) for x in e[1:]] + [e[0](Add(*e[1:]))])[1].gens)
            # 否则，将 e 添加到 gens 列表中
            else:
                gens.append(e)
        # 返回解析得到的 n、funcs、iterables、gens 四个部分
        return n, funcs, iterables, gens

    # 定义函数 build_ideal，构建理想的生成器
    def build_ideal(x, terms):
        """
        Build generators for our ideal. ``Terms`` is an iterable with elements of
        the form (fn, coeff), indicating that we have a generator fn(coeff*x).

        If any of the terms is trigonometric, sin(x) and cos(x) are guaranteed
        to appear in terms. Similarly for hyperbolic functions. For tan(n*x),
        sin(n*x) and cos(n*x) are guaranteed.
        """
        # 初始化空列表 I 和虚拟变量 y
        I = []
        y = Dummy('y')
        # 遍历 terms 中的每个元素 (fn, coeff)
        for fn, coeff in terms:
            # 对于每种三角函数和相关关系，根据 fn 和 coeff 构建生成器
            for c, s, t, rel in (
                    [cos, sin, tan, cos(x)**2 + sin(x)**2 - 1],
                    [cosh, sinh, tanh, cosh(x)**2 - sinh(x)**2 - 1]):
                # 如果 coeff 为 1 并且 fn 是 c 或者 s，则将关系 rel 添加到 I 中
                if coeff == 1 and fn in [c, s]:
                    I.append(rel)
                # 如果 fn 是 t，则构建 tan(coeff*x)*cos(coeff*x) - sin(coeff*x) 的关系添加到 I 中
                elif fn == t:
                    I.append(t(coeff*x)*c(coeff*x) - s(coeff*x))
                # 如果 fn 是 c 或者 s，则计算 fn(coeff*y)，然后化简并将结果添加到 I 中
                elif fn in [c, s]:
                    cn = fn(coeff*y).expand(trig=True).subs(y, x)
                    I.append(fn(coeff*x) - cn)
        # 返回去重后的生成器列表 I
        return list(set(I))

    # 创建虚拟变量 myI 并将表达式 expr 中的 S.ImaginaryUnit 替换为 myI
    myI = Dummy('I')
    expr = expr.subs(S.ImaginaryUnit, myI)
    # 将 (myI, S.ImaginaryUnit) 添加到 subs 列表中
    subs = [(myI, S.ImaginaryUnit)]

    # 将 expr 取消化简为分子 num 和分母 denom
    num, denom = cancel(expr).as_numer_denom()
    try:
        # 使用 parallel_poly_from_expr 处理 [num, denom] 并获取结果 pnum, pdenom 和 opt
        (pnum, pdenom), opt = parallel_poly_from_expr([num, denom])
    except PolificationFailed:
        # 如果无法将表达式转化为多项式，直接返回原始表达式 expr
        return expr
    # 输出调试信息，显示初始生成器 opt.gens
    debug('initial gens:', opt.gens)
    # 分析生成器 opt.gens，得到理想、自由生成器和生成器 gens
    ideal, freegens, gens = analyse_gens(opt.gens, hints)
    # 输出调试信息，显示理想和新生成器 gens 的详细信息及长度
    debug('ideal:', ideal)
    debug('new gens:', gens, " -- len", len(gens))
    debug('free gens:', freegens, " -- len", len(gens))
    # 注意：我们强制领域为 ZZ，以阻止多项式注入生成器
    # （这通常表明我们构建理想的方式存在 bug）
    # 如果 gens 列表为空，则直接返回原始表达式 expr
    if not gens:
        return expr
    # 使用 groebner 函数计算理想 ideal 的格罗布纳基 G，指定排序方式 order 和生成器 gens
    G = groebner(ideal, order=order, gens=gens, domain=ZZ)
    # 输出调试信息，显示格罗布纳基 G 的列表及长度
    debug('groebner basis:', list(G), " -- len", len(G))

    # 如果我们的分数在自由生成器的多项式中，单独简化所有系数：
    # 导入 sympy.simplify.ratsimp 模块中的 ratsimpmodprime 函数
    from sympy.simplify.ratsimp import ratsimpmodprime

    # 如果 freegens 不为空且 pdenom 只包含交集中的生成器
    if freegens and pdenom.has_only_gens(*set(gens).intersection(pdenom.gens)):
        # 将 num 转换为多项式，并排除 gens
        num = Poly(num, gens=gens+freegens).eject(*gens)
        # 初始化结果列表
        res = []
        # 遍历 num 的每一项
        for monom, coeff in num.terms():
            # 从 coeff 和 denom 构建并行多项式，获取其生成器集合
            ourgens = set(parallel_poly_from_expr([coeff, denom])[1].gens)
            
            # 计算理想中从我们的生成器出发可以到达的所有生成器的传递闭包
            changed = True
            while changed:
                changed = False
                for p in ideal:
                    p = Poly(p)
                    # 如果 ourgens 不是 p 的超集且 p 不仅包含 ourgens 中没有的生成器
                    if not ourgens.issuperset(p.gens) and \
                       not p.has_only_gens(*set(p.gens).difference(ourgens)):
                        changed = True
                        ourgens.update(p.exclude().gens)
            
            # 保留生成器的顺序
            realgens = [x for x in gens if x in ourgens]
            
            # 理想的生成器现在被（隐式地）分为两组：涉及 ourgens 和不涉及 ourgens 的生成器
            # 由于我们上面取了传递闭包，这两组在由不相交变量集生成的子环中存在
            # 任何合理的 Groebner 基础算法将保持这种不相交结构
            # 这些生成元的两个子集然后可以单独形成 Groebner 基础（对于较小的生成集合来说）
            
            # 从 G.polys 中选择只包含 ourgens 交集的生成器的表达式
            ourG = [g.as_expr() for g in G.polys if
                    g.has_only_gens(*ourgens.intersection(g.gens))]
            
            # 将当前项的计算结果添加到结果列表中
            res.append(Mul(*[a**b for a, b in zip(freegens, monom)]) * \
                       ratsimpmodprime(coeff/denom, ourG, order=order,
                                       gens=realgens, quick=quick, domain=ZZ,
                                       polynomial=polynomial).subs(subs))
        
        # 返回结果项的和
        return Add(*res)
        
        # 如果上述方法出现问题，可以使用以下更简单的方法，对 Groebner 基础算法的假设较少
        # 使用此方法需要将以下代码取消注释并将上面的 return 语句注释掉
        # return Add(*[Mul(*[a**b for a, b in zip(freegens, monom)]) * \
        #              ratsimpmodprime(coeff/denom, list(G), order=order,
        #                              gens=gens, quick=quick, domain=ZZ)
        #              for monom, coeff in num.terms()])
    
    # 如果条件不满足，则直接对 expr 应用 ratsimpmodprime 函数
    else:
        return ratsimpmodprime(
            expr, list(G), order=order, gens=freegens+gens,
            quick=quick, domain=ZZ, polynomial=polynomial).subs(subs)
# 定义一个元组 `_trigs`，包含 `TrigonometricFunction` 和 `HyperbolicFunction` 类型
_trigs = (TrigonometricFunction, HyperbolicFunction)


# 定义一个函数 `_trigsimp_inverse`，用于简化表达式中的反三角函数
def _trigsimp_inverse(rv):

    # 定义一个内部函数 `check_args`，检查两个表达式是否具有相同的第一个参数
    def check_args(x, y):
        try:
            return x.args[0] == y.args[0]
        except IndexError:
            return False

    # 定义函数 `f(rv)`，用于简化表达式 `rv`
    def f(rv):
        # 对于简单函数的情况
        g = getattr(rv, 'inverse', None)
        if (g is not None and isinstance(rv.args[0], g()) and
                isinstance(g()(1), TrigonometricFunction)):
            return rv.args[0].args[0]

        # 对于 atan2 的简化，较复杂，因为 atan2 有两个参数
        if isinstance(rv, atan2):
            y, x = rv.args
            if _coeff_isneg(y):
                return -f(atan2(-y, x))
            elif _coeff_isneg(x):
                return S.Pi - f(atan2(y, -x))

            # 检查参数是否相同
            if check_args(x, y):
                if isinstance(y, sin) and isinstance(x, cos):
                    return x.args[0]
                if isinstance(y, cos) and isinstance(x, sin):
                    return S.Pi / 2 - x.args[0]

        return rv

    # 使用 `bottom_up` 函数将 `f` 应用到 `rv` 表达式上
    return bottom_up(rv, f)


# 定义函数 `trigsimp`，通过已知的三角函数恒等式返回简化后的表达式
def trigsimp(expr, inverse=False, **opts):
    """Returns a reduced expression by using known trig identities.

    Parameters
    ==========

    inverse : bool, optional
        If ``inverse=True``, it will be assumed that a composition of inverse
        functions, such as sin and asin, can be cancelled in any order.
        For example, ``asin(sin(x))`` will yield ``x`` without checking whether
        x belongs to the set where this relation is true. The default is False.
        Default : True

    method : string, optional
        Specifies the method to use. Valid choices are:

        - ``'matching'``, default
        - ``'groebner'``
        - ``'combined'``
        - ``'fu'``
        - ``'old'``

        If ``'matching'``, simplify the expression recursively by targeting
        common patterns. If ``'groebner'``, apply an experimental groebner basis
        algorithm. In this case further options are forwarded to
        ``trigsimp_groebner``, please refer to
        its docstring. If ``'combined'``, it first runs the groebner basis
        algorithm with small default parameters, then runs the ``'matching'``
        algorithm. If ``'fu'``, run the collection of trigonometric
        transformations described by Fu, et al. (see the
        :py:func:`~sympy.simplify.fu.fu` docstring). If ``'old'``, the original
        SymPy trig simplification function is run.
    opts :
        Optional keyword arguments passed to the method. See each method's
        function docstring for details.

    Examples
    ========

    >>> from sympy import trigsimp, sin, cos, log
    >>> from sympy.abc import x
    >>> e = 2*sin(x)**2 + 2*cos(x)**2
    >>> trigsimp(e)
    2

    Simplification occurs wherever trigonometric functions are located.

    >>> trigsimp(log(e))
    log(2)

    Using ``method='groebner'`` (or ``method='combined'``) might lead to
    greater simplification.
    """
    """
    The old trigsimp routine can be accessed as with method ``method='old'``.
    
    >>> from sympy import coth, tanh
    >>> t = 3*tanh(x)**7 - 2/coth(x)**7
    >>> trigsimp(t, method='old') == t
    True
    >>> trigsimp(t)
    tanh(x)**7
    """
    
    # 导入fu函数，该函数在简化中被引用
    from sympy.simplify.fu import fu
    
    # 将输入的表达式转换为Sympy的表达式对象
    expr = sympify(expr)
    
    # 尝试从表达式中获取_eval_trigsimp方法，以实现特定优化
    _eval_trigsimp = getattr(expr, '_eval_trigsimp', None)
    if _eval_trigsimp is not None:
        # 如果_eval_trigsimp方法存在，则直接调用并返回其结果
        return _eval_trigsimp(**opts)
    
    # 弹出参数字典中的'old'键，用以判断是否使用旧的简化方法
    old = opts.pop('old', False)
    if not old:
        # 如果不使用旧方法，则清除深度和递归选项，同时设置简化方法为'matching'
        opts.pop('deep', None)
        opts.pop('recursive', None)
        method = opts.pop('method', 'matching')
    else:
        # 如果使用旧方法，则强制设定方法为'old'
        method = 'old'
    
    # 定义groebnersimp函数，用于使用Groebner基进行简化
    def groebnersimp(ex, **opts):
        def traverse(e):
            # 递归遍历表达式e，应用简化函数trigsimp_groebner
            if e.is_Atom:
                return e
            args = [traverse(x) for x in e.args]
            if e.is_Function or e.is_Pow:
                args = [trigsimp_groebner(x, **opts) for x in args]
            return e.func(*args)
        new = traverse(ex)
        if not isinstance(new, Expr):
            return new
        return trigsimp_groebner(new, **opts)
    
    # 根据方法选择相应的简化函数
    trigsimpfunc = {
        'fu': (lambda x: fu(x, **opts)),
        'matching': (lambda x: futrig(x)),
        'groebner': (lambda x: groebnersimp(x, **opts)),
        'combined': (lambda x: futrig(groebnersimp(x,
                               polynomial=True, hints=[2, tan]))),
        'old': lambda x: trigsimp_old(x, **opts),
    }[method]
    
    # 应用选定的简化函数来简化输入的表达式
    expr_simplified = trigsimpfunc(expr)
    
    # 如果需要反向操作，则应用_trigsimp_inverse函数
    if inverse:
        expr_simplified = _trigsimp_inverse(expr_simplified)
    
    # 返回简化后的表达式
    return expr_simplified
def exptrigsimp(expr):
    """
    Simplifies exponential / trigonometric / hyperbolic functions.

    Examples
    ========

    >>> from sympy import exptrigsimp, exp, cosh, sinh
    >>> from sympy.abc import z

    >>> exptrigsimp(exp(z) + exp(-z))
    2*cosh(z)
    >>> exptrigsimp(cosh(z) - sinh(z))
    exp(-z)
    """
    # 导入必要的函数和常量
    from sympy.simplify.fu import hyper_as_trig, TR2i

    def exp_trig(e):
        # 选择 e 或者将 e 重写为指数或三角函数形式中更好的一个
        choices = [e]
        if e.has(*_trigs):
            choices.append(e.rewrite(exp))
        choices.append(e.rewrite(cos))
        return min(*choices, key=count_ops)

    # 应用 exp_trig 函数来逐层简化表达式
    newexpr = bottom_up(expr, exp_trig)

    def f(rv):
        if not rv.is_Mul:
            return rv
        commutative_part, noncommutative_part = rv.args_cnc()
        # 由于 as_powers_dict 会丢失顺序信息，如果非交换部分超过一个，则只应用于简化交换部分
        if len(noncommutative_part) > 1:
            return f(Mul(*commutative_part)) * Mul(*noncommutative_part)
        rvd = rv.as_powers_dict()
        newd = rvd.copy()

        def signlog(expr, sign=S.One):
            if expr is S.Exp1:
                return sign, S.One
            elif isinstance(expr, exp) or (expr.is_Pow and expr.base == S.Exp1):
                return sign, expr.exp
            elif sign is S.One:
                return signlog(-expr, sign=-S.One)
            else:
                return None, None

        ee = rvd[S.Exp1]
        for k in rvd:
            if k.is_Add and len(k.args) == 2:
                # k == c*(1 + sign*E**x)
                c = k.args[0]
                sign, x = signlog(k.args[1] / c)
                if not x:
                    continue
                m = rvd[k]
                newd[k] -= m
                if ee == -x * m / 2:
                    # 处理 sinh 和 cosh 的情况
                    newd[S.Exp1] -= ee
                    ee = 0
                    if sign == 1:
                        newd[2 * c * cosh(x / 2)] += m
                    else:
                        newd[-2 * c * sinh(x / 2)] += m
                elif newd[1 - sign * S.Exp1 ** x] == -m:
                    # 处理 tanh 的情况
                    del newd[1 - sign * S.Exp1 ** x]
                    if sign == 1:
                        newd[-c / tanh(x / 2)] += m
                    else:
                        newd[-c * tanh(x / 2)] += m
                else:
                    newd[1 + sign * S.Exp1 ** x] += m
                    newd[c] += m

        return Mul(*[k ** newd[k] for k in newd])

    # 再次逐层简化表达式
    newexpr = bottom_up(newexpr, f)

    # 将 sinh/cosh 转换为 tanh
    if newexpr.has(HyperbolicFunction):
        e, f = hyper_as_trig(newexpr)
        newexpr = f(TR2i(e))

    # 将 sin/cos 转换为 tan
    if newexpr.has(TrigonometricFunction):
        newexpr = TR2i(newexpr)

    # 是否可能生成之前不存在的虚数单位 I？
    # 如果条件不满足，则执行以下语句块
    if not (newexpr.has(I) and not expr.has(I)):
        # 将 newexpr 赋值给 expr
        expr = newexpr
    # 返回表达式 expr
    return expr
#-------------------- 旧的 trigsimp 程序 ---------------------

# 定义 trigsimp_old 函数，用于通过已知的三角恒等式来简化表达式
def trigsimp_old(expr, *, first=True, **opts):
    """
    Reduces expression by using known trig identities.

    Notes
    =====

    deep:
    - Apply trigsimp inside all objects with arguments

    recursive:
    - Use common subexpression elimination (cse()) and apply
    trigsimp recursively (this is quite expensive if the
    expression is large)

    method:
    - Determine the method to use. Valid choices are 'matching' (default),
    'groebner', 'combined', 'fu' and 'futrig'. If 'matching', simplify the
    expression recursively by pattern matching. If 'groebner', apply an
    experimental groebner basis algorithm. In this case further options
    are forwarded to ``trigsimp_groebner``, please refer to its docstring.
    If 'combined', first run the groebner basis algorithm with small
    default parameters, then run the 'matching' algorithm. 'fu' runs the
    collection of trigonometric transformations described by Fu, et al.
    (see the `fu` docstring) while `futrig` runs a subset of Fu-transforms
    that mimic the behavior of `trigsimp`.

    compare:
    - show input and output from `trigsimp` and `futrig` when different,
    but returns the `trigsimp` value.

    Examples
    ========

    >>> from sympy import trigsimp, sin, cos, log, cot
    >>> from sympy.abc import x
    >>> e = 2*sin(x)**2 + 2*cos(x)**2
    >>> trigsimp(e, old=True)
    2
    >>> trigsimp(log(e), old=True)
    log(2*sin(x)**2 + 2*cos(x)**2)
    >>> trigsimp(log(e), deep=True, old=True)
    log(2)

    Using `method="groebner"` (or `"combined"`) can sometimes lead to a lot
    more simplification:

    >>> e = (-sin(x) + 1)/cos(x) + cos(x)/(-sin(x) + 1)
    >>> trigsimp(e, old=True)
    (1 - sin(x))/cos(x) + cos(x)/(1 - sin(x))
    >>> trigsimp(e, method="groebner", old=True)
    2/cos(x)

    >>> trigsimp(1/cot(x)**2, compare=True, old=True)
          futrig: tan(x)**2
    cot(x)**(-2)

    """
    old = expr
    # 如果 `first` 为真，则执行以下代码块
    if first:
        # 检查表达式 `expr` 是否包含 `_trigs` 中的任何一个项，若不包含则直接返回 `expr`
        if not expr.has(*_trigs):
            return expr

        # 收集所有 `_trigs` 中各项表达式的自由符号，并存储在 `trigsyms` 中
        trigsyms = set().union(*[t.free_symbols for t in expr.atoms(*_trigs)])

        # 如果 `trigsyms` 的数量大于1，则尝试对 `expr` 进行变量分离（separatevars）
        if len(trigsyms) > 1:
            # 导入 sympy 中的 separatevars 函数
            from sympy.simplify.simplify import separatevars

            # 对表达式 `expr` 进行变量分离，并将结果存储在 `d` 中
            d = separatevars(expr)

            # 如果 `d` 是乘法形式，则再次应用 separatevars 并存储在 `d` 中
            if d.is_Mul:
                d = separatevars(d, dict=True) or d

            # 如果 `d` 是字典形式，则按照其值更新 `expr`
            if isinstance(d, dict):
                expr = 1
                for v in d.values():
                    # 将每个值 `v` 展开，并应用 `trigsimp` 简化操作，将结果累积到 `expr`
                    was = v
                    v = expand_mul(v)
                    opts['first'] = False
                    vnew = trigsimp(v, **opts)
                    if vnew == v:
                        vnew = was
                    expr *= vnew
                old = expr
            else:
                # 如果 `d` 是加法形式，则对其中的每个符号 `s` 进行独立化处理
                if d.is_Add:
                    for s in trigsyms:
                        r, e = expr.as_independent(s)
                        if r:
                            opts['first'] = False
                            expr = r + trigsimp(e, **opts)
                            if not expr.is_Add:
                                break
                    old = expr

    # 从 `opts` 中弹出 'recursive', 'deep', 'method' 参数，并将它们分别存储在变量中
    recursive = opts.pop('recursive', False)
    deep = opts.pop('deep', False)
    method = opts.pop('method', 'matching')

    # 定义 `groebnersimp` 函数，根据参数 `deep` 执行对 `ex` 的简化操作
    def groebnersimp(ex, deep, **opts):
        def traverse(e):
            if e.is_Atom:
                return e
            args = [traverse(x) for x in e.args]
            if e.is_Function or e.is_Pow:
                args = [trigsimp_groebner(x, **opts) for x in args]
            return e.func(*args)
        if deep:
            ex = traverse(ex)
        return trigsimp_groebner(ex, **opts)

    # 根据 `method` 参数选择不同的简化函数 `trigsimpfunc`
    trigsimpfunc = {
        'matching': (lambda x, d: _trigsimp(x, d)),
        'groebner': (lambda x, d: groebnersimp(x, d, **opts)),
        'combined': (lambda x, d: _trigsimp(groebnersimp(x,
                                       d, polynomial=True, hints=[2, tan]),
                                   d))
                   }[method]

    # 如果 `recursive` 为真，则对 `expr` 进行 `cse`（公共子表达式消除）处理，并应用 `trigsimpfunc` 进行简化
    if recursive:
        w, g = cse(expr)
        g = trigsimpfunc(g[0], deep)

        # 逆序应用之前提取的公共子表达式替换，并再次应用 `trigsimpfunc` 进行简化
        for sub in reversed(w):
            g = g.subs(sub[0], sub[1])
            g = trigsimpfunc(g, deep)
        result = g
    else:
        # 如果 `recursive` 为假，则直接对 `expr` 应用 `trigsimpfunc` 进行简化
        result = trigsimpfunc(expr, deep)

    # 如果 `opts` 中存在 'compare' 参数且其值为真，则比较简化前后的结果，并打印相关信息
    if opts.get('compare', False):
        f = futrig(old)
        if f != result:
            print('\tfutrig:', f)

    # 返回最终的简化结果 `result`
    return result
def _dotrig(a, b):
    """Helper to tell whether ``a`` and ``b`` have the same sorts
    of symbols in them -- no need to test hyperbolic patterns against
    expressions that have no hyperbolics in them."""
    # 返回判断条件：a 和 b 的函数类型相同，且包含三角函数或双曲函数
    return a.func == b.func and (
        a.has(TrigonometricFunction) and b.has(TrigonometricFunction) or
        a.has(HyperbolicFunction) and b.has(HyperbolicFunction))


_trigpat = None
def _trigpats():
    global _trigpat
    # 定义通配符 a, b, c 和不可交换的通配符 d
    a, b, c = symbols('a b c', cls=Wild)
    d = Wild('d', commutative=False)

    # 简化模式匹配的公式，例如 sinh/cosh -> tanh:
    # 不要重新排序前 14 个，因为这些假设在 _match_div_rewrite 中是按照这个顺序的。
    matchers_division = (
        (a*sin(b)**c/cos(b)**c, a*tan(b)**c, sin(b), cos(b)),
        (a*tan(b)**c*cos(b)**c, a*sin(b)**c, sin(b), cos(b)),
        (a*cot(b)**c*sin(b)**c, a*cos(b)**c, sin(b), cos(b)),
        (a*tan(b)**c/sin(b)**c, a/cos(b)**c, sin(b), cos(b)),
        (a*cot(b)**c/cos(b)**c, a/sin(b)**c, sin(b), cos(b)),
        (a*cot(b)**c*tan(b)**c, a, sin(b), cos(b)),
        (a*(cos(b) + 1)**c*(cos(b) - 1)**c,
            a*(-sin(b)**2)**c, cos(b) + 1, cos(b) - 1),
        (a*(sin(b) + 1)**c*(sin(b) - 1)**c,
            a*(-cos(b)**2)**c, sin(b) + 1, sin(b) - 1),

        (a*sinh(b)**c/cosh(b)**c, a*tanh(b)**c, S.One, S.One),
        (a*tanh(b)**c*cosh(b)**c, a*sinh(b)**c, S.One, S.One),
        (a*coth(b)**c*sinh(b)**c, a*cosh(b)**c, S.One, S.One),
        (a*tanh(b)**c/sinh(b)**c, a/cosh(b)**c, S.One, S.One),
        (a*coth(b)**c/cosh(b)**c, a/sinh(b)**c, S.One, S.One),
        (a*coth(b)**c*tanh(b)**c, a, S.One, S.One),

        (c*(tanh(a) + tanh(b))/(1 + tanh(a)*tanh(b)),
            tanh(a + b)*c, S.One, S.One),
    )

    matchers_add = (
        (c*sin(a)*cos(b) + c*cos(a)*sin(b) + d, sin(a + b)*c + d),
        (c*cos(a)*cos(b) - c*sin(a)*sin(b) + d, cos(a + b)*c + d),
        (c*sin(a)*cos(b) - c*cos(a)*sin(b) + d, sin(a - b)*c + d),
        (c*cos(a)*cos(b) + c*sin(a)*sin(b) + d, cos(a - b)*c + d),
        (c*sinh(a)*cosh(b) + c*sinh(b)*cosh(a) + d, sinh(a + b)*c + d),
        (c*cosh(a)*cosh(b) + c*sinh(a)*sinh(b) + d, cosh(a + b)*c + d),
    )

    # cos(x)**2 + sin(x)**2 -> 1 的匹配公式
    matchers_identity = (
        (a*sin(b)**2, a - a*cos(b)**2),
        (a*tan(b)**2, a*(1/cos(b))**2 - a),
        (a*cot(b)**2, a*(1/sin(b))**2 - a),
        (a*sin(b + c), a*(sin(b)*cos(c) + sin(c)*cos(b))),
        (a*cos(b + c), a*(cos(b)*cos(c) - sin(b)*sin(c))),
        (a*tan(b + c), a*((tan(b) + tan(c))/(1 - tan(b)*tan(c)))),

        (a*sinh(b)**2, a*cosh(b)**2 - a),
        (a*tanh(b)**2, a - a*(1/cosh(b))**2),
        (a*coth(b)**2, a + a*(1/sinh(b))**2),
        (a*sinh(b + c), a*(sinh(b)*cosh(c) + sinh(c)*cosh(b))),
        (a*cosh(b + c), a*(cosh(b)*cosh(c) + sinh(b)*sinh(c))),
        (a*tanh(b + c), a*((tanh(b) + tanh(c))/(1 + tanh(b)*tanh(c)))),

    )

    # 减少任何残留的痕迹，例如 sin(x)**2 的变化
    # 定义了一个包含多个元组的元组 artifacts，每个元组包含三个元素：表达式1，表达式2，函数
    artifacts = (
        (a - a*cos(b)**2 + c, a*sin(b)**2 + c, cos),
        (a - a*(1/cos(b))**2 + c, -a*tan(b)**2 + c, cos),
        (a - a*(1/sin(b))**2 + c, -a*cot(b)**2 + c, sin),

        (a - a*cosh(b)**2 + c, -a*sinh(b)**2 + c, cosh),
        (a - a*(1/cosh(b))**2 + c, a*tanh(b)**2 + c, cosh),
        (a + a*(1/sinh(b))**2 + c, a*coth(b)**2 + c, sinh),

        # 与上面相同但具有非交换的前因子
        (a*d - a*d*cos(b)**2 + c, a*d*sin(b)**2 + c, cos),
        (a*d - a*d*(1/cos(b))**2 + c, -a*d*tan(b)**2 + c, cos),
        (a*d - a*d*(1/sin(b))**2 + c, -a*d*cot(b)**2 + c, sin),

        (a*d - a*d*cosh(b)**2 + c, -a*d*sinh(b)**2 + c, cosh),
        (a*d - a*d*(1/cosh(b))**2 + c, a*d*tanh(b)**2 + c, cosh),
        (a*d + a*d*(1/sinh(b))**2 + c, a*d*coth(b)**2 + c, sinh),
    )

    # 定义了一个包含多个对象的元组 _trigpat，这些对象包括变量 a, b, c, d 以及四个函数对象 matchers_division, matchers_add, matchers_identity 和 artifacts
    _trigpat = (a, b, c, d, matchers_division, matchers_add,
        matchers_identity, artifacts)
    # 返回 _trigpat 元组作为函数的结果
    return _trigpat
# 缓存修饰器，用于函数 __trigsimp，缓存函数返回结果以提高性能
@cacheit
# 对表达式进行三角函数简化操作
def __trigsimp(expr, deep=False):
    # 如果表达式包含三角函数相关模式，则调用 __trigsimp 函数进行进一步处理
    if expr.has(*_trigs):
        return __trigsimp(expr, deep)
    # 如果表达式不包含三角函数相关模式，则直接返回原表达式
    return expr
    """递归辅助函数用于 trigsimp"""
    # 从 sympy.simplify.fu 模块导入 TR10i 函数
    from sympy.simplify.fu import TR10i

    # 如果 _trigpat 为空，则调用 _trigpats() 函数来初始化
    if _trigpat is None:
        _trigpats()
    
    # 从 _trigpat 中解包得到变量 a, b, c, d, matchers_division, matchers_add,
    # matchers_identity, artifacts
    a, b, c, d, matchers_division, matchers_add, \
    matchers_identity, artifacts = _trigpat

    # 如果表达式 expr 是乘法类型
    if expr.is_Mul:
        # 进行一些简化，例如 sin/cos -> tan:
        # 如果表达式不是交换类型
        if not expr.is_commutative:
            # 将表达式分解为可交换和不可交换的部分
            com, nc = expr.args_cnc()
            # 对可交换部分进行 _trigsimp 简化，深度优先，然后重新组合
            expr = _trigsimp(Mul._from_args(com), deep) * Mul._from_args(nc)
        else:
            # 对于每个分割匹配器进行迭代
            for i, (pattern, simp, ok1, ok2) in enumerate(matchers_division):
                # 如果不符合当前模式，则继续下一个匹配器
                if not _dotrig(expr, pattern):
                    continue

                # 尝试用模式 i 重写表达式
                newexpr = _match_div_rewrite(expr, i)
                if newexpr is not None:
                    # 如果成功重写且结果与原表达式不同，则更新表达式
                    if newexpr != expr:
                        expr = newexpr
                        break
                    else:
                        continue

                # 使用 SymPy 匹配代替手动模式匹配
                res = expr.match(pattern)
                if res and res.get(c, 0):
                    # 如果 c 不是整数，则替换 ok1 和 ok2 并检查是否为正数
                    if not res[c].is_integer:
                        ok = ok1.subs(res)
                        if not ok.is_positive:
                            continue
                        ok = ok2.subs(res)
                        if not ok.is_positive:
                            continue

                    # 如果 a 中包含任何带有参数 b 的三角或双曲函数，则跳过简化
                    if any(w.args[0] == res[b] for w in res[a].atoms(
                            TrigonometricFunction, HyperbolicFunction)):
                        continue

                    # 简化表达式并结束循环
                    expr = simp.subs(res)
                    break  # 继续下一步处理
    # 检查表达式是否是加法表达式
    if expr.is_Add:
        # 初始化空列表用于存储处理后的项
        args = []
        # 遍历表达式中的每一项
        for term in expr.args:
            # 如果当前项不是可交换的，则进行特殊处理
            if not term.is_commutative:
                # 分离项中的可交换部分和不可交换部分
                com, nc = term.args_cnc()
                # 将不可交换部分转换成乘法表达式
                nc = Mul._from_args(nc)
                # 将可交换部分转换成乘法表达式
                term = Mul._from_args(com)
            else:
                # 如果是可交换的，则不可交换部分置为1
                nc = S.One
            # 对当前项进行三角函数简化，根据需要进行深度处理
            term = _trigsimp(term, deep)
            # 尝试匹配当前项与预定义的标识符模式，替换匹配到的部分
            for pattern, result in matchers_identity:
                res = term.match(pattern)
                if res is not None:
                    term = result.subs(res)
                    break
            # 将处理后的项加入到列表中
            args.append(term*nc)
        # 如果处理后的项列表与原始表达式的项列表不同，则重新构造加法表达式
        if args != expr.args:
            expr = Add(*args)
            # 对构造后的表达式进行展开和计算操作数数量的最小化
            expr = min(expr, expand(expr), key=count_ops)
        # 如果处理后的表达式仍然是加法表达式，则进一步匹配和简化
        if expr.is_Add:
            for pattern, result in matchers_add:
                # 如果当前表达式匹配指定模式，且不包含某些三角函数
                if not _dotrig(expr, pattern):
                    continue
                # 对匹配到的表达式应用特定的简化规则
                expr = TR10i(expr)
                # 如果表达式中包含双曲函数，则进一步处理匹配结果
                if expr.has(HyperbolicFunction):
                    res = expr.match(pattern)
                    # 如果匹配结果为空或不包含指定变量，或者在匹配结果的d项中包含三角函数或双曲函数
                    if res is None or not (a in res and b in res) or any(
                        w.args[0] in (res[a], res[b]) for w in res[d].atoms(
                            TrigonometricFunction, HyperbolicFunction)):
                        continue
                    # 应用匹配到的结果进行替换
                    expr = result.subs(res)
                    break

        # 最后处理剩余的不规则项，例如 sin(x)**2 转换为 1 - cos(x)**2
        for pattern, result, ex in artifacts:
            if not _dotrig(expr, pattern):
                continue
            # 替换一个新的通配符来排除某些函数，以帮助更好地匹配
            a_t = Wild('a', exclude=[ex])
            pattern = pattern.subs(a, a_t)
            result = result.subs(a, a_t)

            # 匹配当前表达式与模式，并进行逐步替换，直到不再有变化
            m = expr.match(pattern)
            was = None
            while m and was != expr:
                was = expr
                # 如果匹配结果中的特定条件满足，则中断替换过程
                if m[a_t] == 0 or \
                        -m[a_t] in m[c].args or m[a_t] + m[c] == 0:
                    break
                if d in m and m[a_t]*m[d] + m[c] == 0:
                    break
                # 应用匹配到的结果进行替换
                expr = result.subs(m)
                m = expr.match(pattern)
                m.setdefault(c, S.Zero)

    # 如果表达式是乘法表达式或幂表达式，或者需要进行深度处理且包含多个子项，则进一步简化
    elif expr.is_Mul or expr.is_Pow or deep and expr.args:
        expr = expr.func(*[_trigsimp(a, deep) for a in expr.args])
    try:
        # 尝试执行以下代码块，处理可能出现的异常
        if not expr.has(*_trigs):
            # 如果 expr 对象不包含 _trigs 中的任何一个元素，则抛出 TypeError 异常
            raise TypeError
        e = expr.atoms(exp)
        # 获取表达式 expr 中所有的原子表达式
        new = expr.rewrite(exp, deep=deep)
        # 使用 exp 对表达式 expr 进行重写，可以选择深度或者浅层方式
        if new == e:
            # 如果重写后的表达式与原始表达式 e 相同，则抛出 TypeError 异常
            raise TypeError
        fnew = factor(new)
        # 对新表达式 new 进行因式分解得到 fnew
        if fnew != new:
            # 如果因式分解后的表达式 fnew 不等于原表达式 new，则选择操作数最少的表达式
            new = min([new, factor(new)], key=count_ops)
        # 如果所有引入的 exp 都消失了，则接受这个新表达式
        if not (new.atoms(exp) - e):
            # 如果新表达式中的所有 exp 都不在原始表达式 e 中存在
            expr = new
    except TypeError:
        # 捕获 TypeError 异常，并进行处理，不做任何操作
        pass

    return expr
#------------------- end of old trigsimp routines --------------------

# 定义函数 futrig，用于使用类似 Fu 变换对表达式 e 进行简化
def futrig(e, *, hyper=True, **kwargs):
    """Return simplified ``e`` using Fu-like transformations.
    This is not the "Fu" algorithm. This is called by default
    from ``trigsimp``. By default, hyperbolics subexpressions
    will be simplified, but this can be disabled by setting
    ``hyper=False``.

    Examples
    ========

    >>> from sympy import trigsimp, tan, sinh, tanh
    >>> from sympy.simplify.trigsimp import futrig
    >>> from sympy.abc import x
    >>> trigsimp(1/tan(x)**2)
    tan(x)**(-2)

    >>> futrig(sinh(x)/tanh(x))
    cosh(x)

    """
    # 导入必要的模块和函数
    from sympy.simplify.fu import hyper_as_trig

    # 将输入的表达式 e 转换为 SymPy 的基本表达式类型
    e = sympify(e)

    # 如果 e 不是 Basic 类型，则直接返回 e
    if not isinstance(e, Basic):
        return e

    # 如果 e 没有参数，则直接返回 e
    if not e.args:
        return e

    # 保存原始的表达式 e
    old = e
    # 应用底向上的转换函数 _futrig 对 e 进行简化
    e = bottom_up(e, _futrig)

    # 如果 hyper 参数为 True，并且 e 中包含双曲函数，进行特定处理
    if hyper and e.has(HyperbolicFunction):
        # 将双曲函数表达式转换为三角函数表达式
        e, f = hyper_as_trig(e)
        # 再次对转换后的表达式应用 _futrig 函数进行简化
        e = f(bottom_up(e, _futrig))

    # 如果简化后的 e 与原始 e 不同，并且 e 是乘法表达式且第一个参数是有理数
    if e != old and e.is_Mul and e.args[0].is_Rational:
        # 重新分配乘法表达式的前导系数
        e = Mul(*e.as_coeff_Mul())
    # 返回简化后的表达式 e
    return e


# 定义 _futrig 函数，用于 futrig 函数的辅助处理
def _futrig(e):
    """Helper for futrig."""
    # 导入需要用到的转换函数
    from sympy.simplify.fu import (
        TR1, TR2, TR3, TR2i, TR10, L, TR10i,
        TR8, TR6, TR15, TR16, TR111, TR5, TRmorrie, TR11, _TR11, TR14, TR22,
        TR12)

    # 如果表达式 e 不含三角函数，则直接返回 e
    if not e.has(TrigonometricFunction):
        return e

    # 如果 e 是乘法表达式，则分离出系数和三角函数表达式
    if e.is_Mul:
        coeff, e = e.as_independent(TrigonometricFunction)
    else:
        coeff = None

    # 定义 Lops 和 trigs 作为简化函数的辅助函数
    Lops = lambda x: (L(x), x.count_ops(), _nodes(x), len(x.args), x.is_Add)
    trigs = lambda x: x.has(TrigonometricFunction)
    # 定义变量 `tree`，包含一系列转换函数和函数列表
    tree = [identity,
        (
        TR3,  # 规范角度
        TR1,  # sec-csc -> cos-sin
        TR12,  # 展开和的正切
        lambda x: _eapply(factor, x, trigs),  # 对 x 应用 factor 函数，并传入 trigs 参数
        TR2,  # tan-cot -> sin-cos
        [identity, lambda x: _eapply(_mexpand, x, trigs)],  # 标识函数和对 x 应用 _mexpand 函数
        TR2i,  # sin-cos 比率 -> tan
        lambda x: _eapply(lambda i: factor(i.normal()), x, trigs),  # 对 x 应用 lambda 函数，并传入 trigs 参数
        TR14,  # 分解的恒等式
        TR5,  # sin-pow -> cos_pow
        TR10,  # 和的 sin-cos -> sin-cos 乘积
        TR11, _TR11, TR6,  # 减少双角度并重写 cos 幂
        lambda x: _eapply(factor, x, trigs),  # 对 x 应用 factor 函数，并传入 trigs 参数
        TR14,  # 恒等式的幂次
        [identity, lambda x: _eapply(_mexpand, x, trigs)],  # 标识函数和对 x 应用 _mexpand 函数
        TR10i,  # sin-cos 乘积 > 和的 sin-cos
        TRmorrie,  # TRmorrie 函数调用
        [identity, TR8],  # sin-cos 乘积 -> 和的 sin-cos
        [identity, lambda x: TR2i(TR2(x))],  # tan -> sin-cos -> tan
        [
            lambda x: _eapply(expand_mul, TR5(x), trigs),  # 对 x 应用 expand_mul 函数，并传入 trigs 参数
            lambda x: _eapply(
                expand_mul, TR15(x), trigs)],  # 对 x 应用 expand_mul 函数，并传入 trigs 参数，处理正/负 sin 幂次
        [
            lambda x:  _eapply(expand_mul, TR6(x), trigs),  # 对 x 应用 expand_mul 函数，并传入 trigs 参数
            lambda x:  _eapply(
                expand_mul, TR16(x), trigs)],  # 对 x 应用 expand_mul 函数，并传入 trigs 参数，处理正/负 cos 幂次
        TR111,  # tan, sin, cos 到负幂次 -> cot, csc, sec
        [identity, TR2i],  # sin-cos 比率到 tan
        [identity, lambda x: _eapply(
            expand_mul, TR22(x), trigs)],  # tan-cot 到 sec-csc
        TR1, TR2, TR2i,
        [identity, lambda x: _eapply(
            factor_terms, TR12(x), trigs)],  # 展开和的正切
        )]
    # 通过贪婪算法，利用 `tree` 对象并应用 Lops 目标函数，得到结果 `e`
    e = greedy(tree, objective=Lops)(e)

    # 如果 `coeff` 不为 None，则用 `coeff` 乘以 `e`
    if coeff is not None:
        e = coeff * e

    # 返回最终结果 `e`
    return e
# 判断给定表达式 e 是否为 Expr 类型或其所有参数都是 Expr 类型的辅助函数
def _is_Expr(e):
    """_eapply helper to tell whether ``e`` and all its args
    are Exprs."""
    # 如果 e 是 Derivative 类型，则递归判断其内部表达式 e.expr 是否为 Expr 类型
    if isinstance(e, Derivative):
        return _is_Expr(e.expr)
    # 如果 e 不是 Expr 类型，则返回 False
    if not isinstance(e, Expr):
        return False
    # 对于 e 的所有参数，递归判断它们是否都是 Expr 类型
    return all(_is_Expr(i) for i in e.args)


# 对表达式 e 应用函数 func，条件是其所有参数都是 Expr 类型，否则只应用于是 Expr 类型的参数
def _eapply(func, e, cond=None):
    """Apply ``func`` to ``e`` if all args are Exprs else only
    apply it to those args that *are* Exprs."""
    # 如果 e 不是 Expr 类型，则直接返回 e
    if not isinstance(e, Expr):
        return e
    # 如果 e 是 Expr 类型并且其所有参数都是 Expr 类型，或者它没有参数
    if _is_Expr(e) or not e.args:
        # 对 e 应用 func 函数
        return func(e)
    # 否则，对 e 的参数进行递归处理，根据条件 cond 决定是否对参数应用 func 函数
    return e.func(*[
        _eapply(func, ei) if (cond is None or cond(ei)) else ei
        for ei in e.args])
```