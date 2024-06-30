# `D:\src\scipysrc\sympy\sympy\series\gruntz.py`

```
"""
Limits
======

Implemented according to the PhD thesis
https://www.cybertester.com/data/gruntz.pdf, which contains very thorough
descriptions of the algorithm including many examples.  We summarize here
the gist of it.

All functions are sorted according to how rapidly varying they are at
infinity using the following rules. Any two functions f and g can be
compared using the properties of L:

L=lim  log|f(x)| / log|g(x)|           (for x -> oo)

We define >, < ~ according to::

    1. f > g .... L=+-oo

        we say that:
        - f is greater than any power of g
        - f is more rapidly varying than g
        - f goes to infinity/zero faster than g

    2. f < g .... L=0

        we say that:
        - f is lower than any power of g

    3. f ~ g .... L!=0, +-oo

        we say that:
        - both f and g are bounded from above and below by suitable integral
          powers of the other

Examples
========
::
    2 < x < exp(x) < exp(x**2) < exp(exp(x))
    2 ~ 3 ~ -5
    x ~ x**2 ~ x**3 ~ 1/x ~ x**m ~ -x
    exp(x) ~ exp(-x) ~ exp(2x) ~ exp(x)**2 ~ exp(x+exp(-x))
    f ~ 1/f

So we can divide all the functions into comparability classes (x and x^2
belong to one class, exp(x) and exp(-x) belong to some other class). In
principle, we could compare any two functions, but in our algorithm, we
do not compare anything below the class 2~3~-5 (for example log(x) is
below this), so we set 2~3~-5 as the lowest comparability class.

Given the function f, we find the list of most rapidly varying (mrv set)
subexpressions of it. This list belongs to the same comparability class.
Let's say it is {exp(x), exp(2x)}. Using the rule f ~ 1/f we find an
element "w" (either from the list or a new one) from the same
comparability class which goes to zero at infinity. In our example we
set w=exp(-x) (but we could also set w=exp(-2x) or w=exp(-3x) ...). We
rewrite the mrv set using w, in our case {1/w, 1/w^2}, and substitute it
into f. Then we expand f into a series in w::

    f = c0*w^e0 + c1*w^e1 + ... + O(w^en),       where e0<e1<...<en, c0!=0

but for x->oo, lim f = lim c0*w^e0, because all the other terms go to zero,
because w goes to zero faster than the ci and ei. So::

    for e0>0, lim f = 0
    for e0<0, lim f = +-oo   (the sign depends on the sign of c0)
    for e0=0, lim f = lim c0

We need to recursively compute limits at several places of the algorithm, but
as is shown in the PhD thesis, it always finishes.

Important functions from the implementation:

compare(a, b, x) compares "a" and "b" by computing the limit L.
mrv(e, x) returns list of most rapidly varying (mrv) subexpressions of "e"
rewrite(e, Omega, x, wsym) rewrites "e" in terms of w
leadterm(f, x) returns the lowest power term in the series of f
mrv_leadterm(e, x) returns the lead term (c0, e0) for e
limitinf(e, x) computes lim e  (for x->oo)
limit(e, z, z0) computes any limit by converting it to the case x->oo

All the functions are really simple and straightforward except
"""
from functools import reduce  # 导入 functools 模块中的 reduce 函数

from sympy.core import Basic, S, Mul, PoleError, expand_mul  # 导入 sympy.core 模块中的多个类和异常
from sympy.core.cache import cacheit  # 导入 sympy.core.cache 模块中的 cacheit 函数
from sympy.core.intfunc import ilcm  # 导入 sympy.core.intfunc 模块中的 ilcm 函数
from sympy.core.numbers import I, oo  # 导入 sympy.core.numbers 模块中的常数 I 和 oo
from sympy.core.symbol import Dummy, Wild  # 导入 sympy.core.symbol 模块中的 Dummy 和 Wild 符号类
from sympy.core.traversal import bottom_up  # 导入 sympy.core.traversal 模块中的 bottom_up 函数

from sympy.functions import log, exp, sign as _sign  # 导入 sympy.functions 模块中的 log, exp 和 sign 函数别名为 _sign
from sympy.series.order import Order  # 导入 sympy.series.order 模块中的 Order 类
from sympy.utilities.exceptions import SymPyDeprecationWarning  # 导入 sympy.utilities.exceptions 模块中的 SymPyDeprecationWarning 异常类
from sympy.utilities.misc import debug_decorator as debug  # 导入 sympy.utilities.misc 模块中的 debug_decorator 函数别名为 debug
from sympy.utilities.timeutils import timethis  # 导入 sympy.utilities.timeutils 模块中的 timethis 函数

timeit = timethis('gruntz')  # 使用 timethis 函数装饰 gruntz 函数，以测量其执行时间


def compare(a, b, x):
    """Returns "<" if a<b, "=" for a == b, ">" for a>b"""
    # 如果 a 或 b 是指数函数或幂函数，必须在此处对 log(exp(...)) 进行简化以确保终止性
    la, lb = log(a), log(b)
    if isinstance(a, Basic) and (isinstance(a, exp) or (a.is_Pow and a.base == S.Exp1)):
        la = a.exp
    if isinstance(b, Basic) and (isinstance(b, exp) or (b.is_Pow and b.base == S.Exp1)):
        lb = b.exp

    c = limitinf(la/lb, x)  # 计算 la/lb 在 x 趋向于某个极限的值
    if c == 0:
        return "<"
    elif c.is_infinite:
        return ">"
    else:
        return "="


class SubsSet(dict):
    """
    Stores (expr, dummy) pairs, and how to rewrite expr-s.

    Explanation
    ===========

    The gruntz algorithm needs to rewrite certain expressions in term of a new
    variable w. We cannot use subs, because it is just too smart for us. For
    example::

        > Omega=[exp(exp(_p - exp(-_p))/(1 - 1/_p)), exp(exp(_p))]
        > O2=[exp(-exp(_p) + exp(-exp(-_p))*exp(_p)/(1 - 1/_p))/_w, 1/_w]
        > e = exp(exp(_p - exp(-_p))/(1 - 1/_p)) - exp(exp(_p))
        > e.subs(Omega[0],O2[0]).subs(Omega[1],O2[1])
        -1/w + exp(exp(p)*exp(-exp(-p))/(1 - 1/p))

    is really not what we want!
    """
    """This class manages substitutions of expressions with dummy variables.

    So we do it the hard way and keep track of all the things we potentially
    want to substitute by dummy variables. Consider the expression::

        exp(x - exp(-x)) + exp(x) + x.

    The mrv set is {exp(x), exp(-x), exp(x - exp(-x))}.
    We introduce corresponding dummy variables d1, d2, d3 and rewrite::

        d3 + d1 + x.

    This class first of all keeps track of the mapping expr->variable, i.e.
    will at this stage be a dictionary::

        {exp(x): d1, exp(-x): d2, exp(x - exp(-x)): d3}.

    [It turns out to be more convenient this way round.]
    But sometimes expressions in the mrv set have other expressions from the
    mrv set as subexpressions, and we need to keep track of that as well. In
    this case, d3 is really exp(x - d2), so rewrites at this stage is::

        {d3: exp(x-d2)}.

    The function rewrite uses all this information to correctly rewrite our
    expression in terms of w. In this case w can be chosen to be exp(-x),
    i.e. d2. The correct rewriting then is::

        exp(-w)/w + 1/w + x.
    """
    # 初始化方法，创建一个空的重写字典
    def __init__(self):
        self.rewrites = {}

    # 返回对象的字符串表示，包含重写字典的内容
    def __repr__(self):
        return super().__repr__() + ', ' + self.rewrites.__repr__()

    # 获取指定键的值，如果键不存在则创建一个新的 Dummy 变量
    def __getitem__(self, key):
        if key not in self:
            self[key] = Dummy()
        return dict.__getitem__(self, key)

    # 对表达式 e 进行变量替换，用本对象中的表达式对应的变量替换
    def do_subs(self, e):
        """Substitute the variables with expressions"""
        for expr, var in self.items():
            e = e.xreplace({var: expr})
        return e

    # 判断当前对象和另一个 SubsSet 对象 s2 是否有非空交集
    def meets(self, s2):
        """Tell whether or not self and s2 have non-empty intersection"""
        return set(self.keys()).intersection(list(s2.keys())) != set()

    # 计算当前对象和另一个 SubsSet 对象 s2 的并集，调整重写表达式
    def union(self, s2, exps=None):
        """Compute the union of self and s2, adjusting exps"""
        res = self.copy()
        tr = {}
        for expr, var in s2.items():
            if expr in self:
                if exps:
                    exps = exps.xreplace({var: res[expr]})
                tr[var] = res[expr]
            else:
                res[expr] = var
        for var, rewr in s2.rewrites.items():
            res.rewrites[var] = rewr.xreplace(tr)
        return res, exps

    # 创建当前对象的浅拷贝
    def copy(self):
        """Create a shallow copy of SubsSet"""
        r = SubsSet()
        r.rewrites = self.rewrites.copy()
        for expr, var in self.items():
            r[expr] = var
        return r
@debug
# 装饰器，用于调试目的，在函数定义时提供额外的调试功能
def mrv(e, x):
    """Returns a SubsSet of most rapidly varying (mrv) subexpressions of 'e',
       and e rewritten in terms of these"""
    # 导入必要的函数模块，简化幂运算，深度优化，合并指数
    from sympy.simplify.powsimp import powsimp
    # 对表达式e进行幂简化，深度优化，合并指数操作
    e = powsimp(e, deep=True, combine='exp')
    # 检查e是否为Basic类的实例，若不是则引发类型错误
    if not isinstance(e, Basic):
        raise TypeError("e should be an instance of Basic")
    # 如果e不包含变量x，则返回空的SubsSet和原始表达式e
    if not e.has(x):
        return SubsSet(), e
    # 如果e恰好等于变量x，则返回一个空的SubsSet和变量x
    elif e == x:
        s = SubsSet()
        return s, s[x]
    # 如果e是乘法或加法表达式
    elif e.is_Mul or e.is_Add:
        # 将e分解为独立于x的项i和依赖于x的部分d
        i, d = e.as_independent(x)  # throw away x-independent terms
        # 若d的类型与e不同，则递归计算mrv(d, x)
        if d.func != e.func:
            s, expr = mrv(d, x)
            return s, e.func(i, expr)
        # 将d分解为两个项a和b，分别计算它们的mrv
        a, b = d.as_two_terms()
        s1, e1 = mrv(a, x)
        s2, e2 = mrv(b, x)
        # 返回两个mrv集合的最大值
        return mrv_max1(s1, s2, e.func(i, e1, e2), x)
    # 如果e是幂运算且基数不是自然指数
    elif e.is_Pow and e.base != S.Exp1:
        e1 = S.One
        # 将e连续的幂运算拆解为单一幂运算
        while e.is_Pow:
            b1 = e.base
            e1 *= e.exp
            e = b1
        # 如果b1为1，则返回空的SubsSet和1
        if b1 == 1:
            return SubsSet(), b1
        # 如果e1包含变量x，则计算exp(e1*log(b1))的mrv
        if e1.has(x):
            return mrv(exp(e1*log(b1)), x)
        else:
            # 否则递归计算mrv(b1, x)
            s, expr = mrv(b1, x)
            return s, expr**e1
    # 如果e是对数函数
    elif isinstance(e, log):
        # 计算对数函数参数的mrv
        s, expr = mrv(e.args[0], x)
        return s, log(expr)
    # 如果e是指数函数或自然指数的幂运算
    elif isinstance(e, exp) or (e.is_Pow and e.base == S.Exp1):
        # 根据算法的理论，exp(log(...))总是可以在此处简化，并且这对终止是至关重要的
        if isinstance(e.exp, log):
            return mrv(e.exp.args[0], x)
        # 如果乘积有一个无限因子，则考虑结果为无限，否则为NaN
        li = limitinf(e.exp, x)
        if any(_.is_infinite for _ in Mul.make_args(li)):
            s1 = SubsSet()
            e1 = s1[e]
            s2, e2 = mrv(e.exp, x)
            su = s1.union(s2)[0]
            su.rewrites[e1] = exp(e2)
            return mrv_max3(s1, e1, s2, exp(e2), su, e1, x)
        else:
            # 否则递归计算mrv(e.exp, x)
            s, expr = mrv(e.exp, x)
            return s, exp(expr)
    # 如果e是函数
    elif e.is_Function:
        # 递归计算函数参数的mrv
        l = [mrv(a, x) for a in e.args]
        l2 = [s for (s, _) in l if s != SubsSet()]
        # 如果函数参数的mrv集合不止一个，则抛出未实现的错误
        if len(l2) != 1:
            raise NotImplementedError("MRV set computation for functions in"
                                      " several variables not implemented.")
        s, ss = l2[0], SubsSet()
        args = [ss.do_subs(x[1]) for x in l]
        return s, e.func(*args)
    # 如果e是导数
    elif e.is_Derivative:
        # 抛出未实现的错误，导数的mrv计算尚未实现
        raise NotImplementedError("MRV set computation for derivatives"
                                  " not implemented yet.")
    # 对于其他情况，抛出未实现的错误，无法计算e的mrv
    raise NotImplementedError(
        "Don't know how to calculate the mrv of '%s'" % e)


def mrv_max3(f, expsf, g, expsg, union, expsboth, x):
    """
    Computes the maximum of two sets of expressions f and g, which
    are in the same comparability class, i.e. max() compares (two elements of)
    """
    # 计算两个表达式集合f和g的最大值，这些表达式处于相同的可比较类中
    # 即计算f和g的最大值
    # 检查 f 是否为 SubsSet 类的实例，如果不是则抛出类型错误
    if not isinstance(f, SubsSet):
        raise TypeError("f should be an instance of SubsSet")
    
    # 检查 g 是否为 SubsSet 类的实例，如果不是则抛出类型错误
    if not isinstance(g, SubsSet):
        raise TypeError("g should be an instance of SubsSet")
    
    # 如果 f 是空的 SubsSet 对象，则返回 g 和 expsg
    if f == SubsSet():
        return g, expsg
    
    # 如果 g 是空的 SubsSet 对象，则返回 f 和 expsf
    elif g == SubsSet():
        return f, expsf
    
    # 如果 f 和 g 有交集，则返回 union 和 expsboth
    elif f.meets(g):
        return union, expsboth

    # 比较 f 和 g 第一个键的值，使用 x 作为参数
    c = compare(list(f.keys())[0], list(g.keys())[0], x)
    
    # 如果比较结果为大于号 (>)
    if c == ">":
        return f, expsf
    
    # 如果比较结果为小于号 (<)
    elif c == "<":
        return g, expsg
    
    # 如果比较结果为等于号 (=)，否则抛出值错误异常
    else:
        if c != "=":
            raise ValueError("c should be =")
        return union, expsboth
# 计算两组表达式 f 和 g 的最大值，这两组表达式属于同一比较类别。mrv_max1() 比较 f 和 g，
# 返回在它们的并集中较高比较类别中的集合，并返回适当进行替换的 exps。
def mrv_max1(f, g, exps, x):
    u, b = f.union(g, exps)  # 将 f 和 g 与 exps 进行联合，返回联合结果和联合后的约束条件
    return mrv_max3(f, g.do_subs(exps), g, f.do_subs(exps),  # 调用 mrv_max3 函数，传入参数 f, g 替换后的结果，g, f 替换后的结果
                    u, b, x)


# 为表达式 e(x) 返回在 x 趋向无穷时的符号。
@debug  # 调试装饰器，用于调试该函数
@cacheit  # 缓存装饰器，用于缓存该函数的计算结果
@timeit  # 计时装饰器，用于记录该函数的执行时间
def sign(e, x):
    """
    返回表达式 e(x) 在 x 趋向无穷时的符号。

    ::

        e >  0 当 x 足够大时 ...  1
        e == 0 当 x 足够大时 ...  0
        e <  0 当 x 足够大时 ... -1

    如果 e 在 x 足够大时任意变化符号，则此函数的结果目前未定义（例如 sin(x)）。

    注意，仅当 e 在 x 足够大时始终为零时返回零。[如果 e 是常数，当然，这与 e 的符号相同。]
    """
    if not isinstance(e, Basic):
        raise TypeError("e should be an instance of Basic")  # 如果 e 不是 Basic 类型则抛出类型错误

    if e.is_positive:
        return 1  # 如果 e 是正数，返回 1
    elif e.is_negative:
        return -1  # 如果 e 是负数，返回 -1
    elif e.is_zero:
        return 0  # 如果 e 是零，返回 0

    elif not e.has(x):
        from sympy.simplify import logcombine
        e = logcombine(e)  # 对 e 进行对数合并操作
        return _sign(e)  # 调用 _sign 函数计算 e 的符号
    elif e == x:
        return 1  # 如果 e 等于 x，返回 1
    elif e.is_Mul:
        a, b = e.as_two_terms()
        sa = sign(a, x)
        if not sa:
            return 0
        return sa * sign(b, x)  # 如果 e 是乘法，递归计算 a 和 b 的符号并相乘返回结果
    elif isinstance(e, exp):
        return 1  # 如果 e 是指数函数，返回 1
    elif e.is_Pow:
        if e.base == S.Exp1:
            return 1
        s = sign(e.base, x)
        if s == 1:
            return 1
        if e.exp.is_Integer:
            return s**e.exp
    elif isinstance(e, log):
        return sign(e.args[0] - 1, x)  # 如果 e 是对数函数，计算其参数减 1 的符号

    # 如果所有条件均不符合，以硬编码方式计算
    c0, e0 = mrv_leadterm(e, x)  # 调用 mrv_leadterm 函数获取 e 的主导项和常数项
    return sign(c0, x)  # 返回 c0 在 x 趋向无穷时的符号


# 返回 e(x) 当 x 趋向无穷时的极限。
@debug  # 调试装饰器，用于调试该函数
@timeit  # 计时装饰器，用于记录该函数的执行时间
@cacheit  # 缓存装饰器，用于缓存该函数的计算结果
def limitinf(e, x):
    """Limit e(x) as x-> oo."""
    # 将 e 重写为仅包含易处理的函数

    old = e
    if not e.has(x):
        return e  # 如果 e 不包含 x，e 是常数，直接返回 e
    from sympy.simplify.powsimp import powdenest
    from sympy.calculus.util import AccumBounds
    if e.has(Order):
        e = e.expand().removeO()  # 如果 e 含有 Order，展开并移除 O 符号
    if not x.is_positive or x.is_integer:
        # 确保 x 是正数且不是整数，以确保从表达式中得到正确的数学行为
        # 需要一个新的变量。
        p = Dummy('p', positive=True)
        e = e.subs(x, p)
        x = p
    e = e.rewrite('tractable', deep=True, limitvar=x)  # 使用 tractable 重写 e，限制变量为 x
    e = powdenest(e)  # 对 e 进行幂简化
    if isinstance(e, AccumBounds):
        if mrv_leadterm(e.min, x) != mrv_leadterm(e.max, x):
            raise NotImplementedError
        c0, e0 = mrv_leadterm(e.min, x)
    else:
        c0, e0 = mrv_leadterm(e, x)  # 获取 e 的主导项和常数项
    sig = sign(e0, x)  # 计算 e0 在 x 趋向无穷时的符号
    if sig == 1:
        return S.Zero  # 如果 sig 等于 1，返回 S.Zero，表示极限 f = 0
    elif sig == -1:  # 如果 sig 等于 -1，即 e0<0，表示极限 f = +-oo（正负无穷，取决于 c0 的符号）
        if c0.match(I*Wild("a", exclude=[I])):
            return c0*oo  # 如果 c0 匹配到形如 I*Wild("a", exclude=[I]) 的模式，返回 c0 乘以无穷大 oo
        s = sign(c0, x)
        # 主导项不应为 0：
        if s == 0:
            raise ValueError("Leading term should not be 0")  # 如果主导项 s 为 0，抛出数值错误异常
        return s*oo  # 否则返回 s 乘以无穷大 oo
    elif sig == 0:
        if c0 == old:
            c0 = c0.cancel()
        return limitinf(c0, x)  # 如果 sig 等于 0，即 e0=0，返回 limitinf(c0, x)，表示极限 f = lim c0
    else:
        raise ValueError("{} could not be evaluated".format(sig))  # 如果 sig 不属于上述情况，则抛出值错误异常，提示无法评估该值
# 定义一个函数 moveup2，接受字典 s 和符号 x 作为参数
def moveup2(s, x):
    # 创建一个空的 SubsSet 对象
    r = SubsSet()
    # 遍历 s 字典中的每个表达式 expr 和对应的变量 var
    for expr, var in s.items():
        # 使用表达式 expr 中的符号 x，替换为 exp(x)，并添加到 r 字典中
        r[expr.xreplace({x: exp(x)})] = var
    # 遍历 s 对象的重写字典中的每个变量 var 和对应的表达式 expr
    for var, expr in s.rewrites.items():
        # 使用表达式 expr 中的符号 x，替换为 exp(x)，并添加到 r 对象的重写字典中
        r.rewrites[var] = s.rewrites[var].xreplace({x: exp(x)})
    # 返回结果对象 r
    return r


# 定义一个函数 moveup，接受列表 l 和符号 x 作为参数
def moveup(l, x):
    # 对列表 l 中的每个元素 e，使用表达式 e 中的符号 x，替换为 exp(x)，构成新的列表并返回
    return [e.xreplace({x: exp(x)}) for e in l]


# 定义一个装饰器函数，用于调试和计时的组合
@debug
@timeit
# 定义一个计算级数的函数 calculate_series，接受表达式 e、符号 x 和可选的 logx 参数
def calculate_series(e, x, logx=None):
    """ Calculates at least one term of the series of ``e`` in ``x``.

    This is a place that fails most often, so it is in its own function.
    """

    # 发出关于计算级数功能即将废弃的警告信息
    SymPyDeprecationWarning(
        feature="calculate_series",
        useinstead="series() with suitable n, or as_leading_term",
        issue=21838,
        deprecated_since_version="1.12"
    ).warn()

    # 导入 sympy.simplify.powsimp 模块中的 powdenest 函数
    from sympy.simplify.powsimp import powdenest

    # 对表达式 e 的 x 级数展开进行迭代计算
    for t in e.lseries(x, logx=logx):
        # 对每个 t 进行 bottom_up 函数的调用，用于特定情况下的正常化处理
        t = bottom_up(t, lambda w:
            getattr(w, 'normal', lambda: w)())
        # 如果表达式包含特定的 exp(x) 和 log(x)，则进行 powdenest 处理
        if t.has(exp) and t.has(log):
            t = powdenest(t)
        # 如果 t 不是零表达式，则结束迭代
        if not t.is_zero:
            break

    # 返回计算后的结果 t
    return t


# 定义一个装饰器函数，用于调试、计时和缓存结果的组合
@debug
@timeit
@cacheit
# 定义一个计算主导项的函数 mrv_leadterm，接受表达式 e 和符号 x 作为参数
def mrv_leadterm(e, x):
    """Returns (c0, e0) for e."""
    # 创建一个空的 SubsSet 对象 Omega
    Omega = SubsSet()
    # 如果表达式 e 不含符号 x，则直接返回 e 和 0
    if not e.has(x):
        return (e, S.Zero)
    # 如果 Omega 为空 SubsSet 对象，则调用 mrv 函数计算 Omega 和 exps
    if Omega == SubsSet():
        Omega, exps = mrv(e, x)
    # 如果 Omega 为空，说明在简化后 e 不再依赖于 x，返回 exps 和 0
    if not Omega:
        return exps, S.Zero
    # 如果符号 x 在 Omega 中，将整个 Omega 向上移动（对每个项取指数）
    Omega_up = moveup2(Omega, x)
    exps_up = moveup([exps], x)[0]
    # 注意：这里不需要向下移动！

    # 使用正 Dummy 变量 w，以确保 log(w*2) 等可以展开；这个算法需要一个唯一的虚拟变量
    w = Dummy("w", positive=True)
    # 调用 rewrite 函数，重写 exps 使用 Omega、x 和 w，返回结果 f 和 logw
    f, logw = rewrite(exps, Omega, x, w)
    # 尝试计算 f 的主导项，使用 w 和 logw 作为参数
    lt = f.leadterm(w, logx=logw)
    # 尝试捕获特定异常：NotImplementedError, PoleError, ValueError
    except (NotImplementedError, PoleError, ValueError):
        # 设置初始值 n0 为 1
        n0 = 1
        # 创建 Order 对象，并赋值给 _series
        _series = Order(1)
        # 初始化增量 incr 为 S.One
        incr = S.One
        # 循环直到 _series 不再是 Order 对象
        while _series.is_Order:
            # 尝试计算 f 的 n-series 展开，使用 n=n0+incr，logx=logw
            _series = f._eval_nseries(w, n=n0+incr, logx=logw)
            # 增加增量的值为原来的两倍
            incr *= 2
        # 对 _series 展开并去除高阶无穷小项，赋值给 series
        series = _series.expand().removeO()
        # 再次尝试捕获特定异常：NotImplementedError, PoleError, ValueError
        try:
            # 尝试获取 series 的主导项，使用变量 w，logx=logw
            lt = series.leadterm(w, logx=logw)
        # 如果捕获到异常，则执行以下代码块
        except (NotImplementedError, PoleError, ValueError):
            # 获取 f 的 w 系数和指数，并赋值给 lt
            lt = f.as_coeff_exponent(w)
            # 如果 lt[0] 中包含变量 w
            if lt[0].has(w):
                # 获取 f 的基数和指数
                base = f.as_base_exp()[0].as_coeff_exponent(w)
                ex = f.as_base_exp()[1]
                # 重新定义 lt 为 (base[0]**ex, base[1]*ex)
                lt = (base[0]**ex, base[1]*ex)
    # 返回 lt 的第一个元素中的 log(w) 替换为 logw，以及 lt 的第二个元素
    return (lt[0].subs(log(w), logw), lt[1])
def build_expression_tree(Omega, rewrites):
    r""" Helper function for rewrite.

    We need to sort Omega (mrv set) so that we replace an expression before
    we replace any expression in terms of which it has to be rewritten::

        e1 ---> e2 ---> e3
                 \
                  -> e4

    Here we can do e1, e2, e3, e4 or e1, e2, e4, e3.
    To do this we assemble the nodes into a tree, and sort them by height.

    This function builds the tree, rewrites then sorts the nodes.
    """
    class Node:
        def __init__(self):
            self.before = []  # List to store nodes that precede this node
            self.expr = None   # Expression associated with this node
            self.var = None    # Variable associated with this node
        
        def ht(self):
            """Calculate the height of the node in the tree."""
            # Calculate height recursively by summing up heights of 'before' nodes
            return reduce(lambda x, y: x + y, [x.ht() for x in self.before], 1)

    nodes = {}  # Dictionary to hold Node objects, keyed by variables
    for expr, v in Omega:
        n = Node()
        n.var = v
        n.expr = expr
        nodes[v] = n  # Store Node object in dictionary using variable 'v'

    # Link nodes based on the rewrites mapping
    for _, v in Omega:
        if v in rewrites:
            n = nodes[v]
            r = rewrites[v]
            for _, v2 in Omega:
                if r.has(v2):  # Check if 'v2' is in the rewrite rules of 'v'
                    n.before.append(nodes[v2])  # Append node of 'v2' to 'before' list of 'v'

    return nodes


@debug
@timeit
def rewrite(e, Omega, x, wsym):
    """e(x) ... the function
    Omega ... the mrv set
    wsym ... the symbol which is going to be used for w

    Returns the rewritten e in terms of w and log(w). See test_rewrite1()
    for examples and correct results.
    """

    from sympy import AccumBounds
    if not isinstance(Omega, SubsSet):
        raise TypeError("Omega should be an instance of SubsSet")
    if len(Omega) == 0:
        raise ValueError("Length cannot be 0")
    
    # Validate that all items in Omega are instances of exp
    for t in Omega.keys():
        if not isinstance(t, exp):
            raise ValueError("Value should be exp")
    
    rewrites = Omega.rewrites
    Omega = list(Omega.items())

    nodes = build_expression_tree(Omega, rewrites)  # Build expression tree using Omega and rewrites
    Omega.sort(key=lambda x: nodes[x[1]].ht(), reverse=True)  # Sort Omega by node height

    # Ensure we know the sign of each exp() term
    for g, _ in Omega:
        sig = sign(g.exp, x)
        if sig != 1 and sig != -1 and not sig.has(AccumBounds):
            raise NotImplementedError('Result depends on the sign of %s' % sig)
    
    if sig == 1:
        wsym = 1/wsym  # Adjust wsym if g goes to infinity
    
    O2 = []  # Initialize list to store rewritten expressions
    denominators = []  # Initialize list to store denominators for rational coefficients
    
    for f, var in Omega:
        c = limitinf(f.exp/g.exp, x)  # Calculate limit of f.exp / g.exp as x approaches infinity
        if c.is_Rational:
            denominators.append(c.q)  # Store the denominator of rational coefficient 'c'
        arg = f.exp
        if var in rewrites:
            if not isinstance(rewrites[var], exp):
                raise ValueError("Value should be exp")
            arg = rewrites[var].args[0]
        O2.append((var, exp((arg - c*g.exp).expand())*wsym**c))  # Append rewritten expression to O2

    # Substitute subexpressions of 'e' with the corresponding rewrites in 'O2'
    # 以下的 powsimp 操作是必要的，用于自动合并指数项，
    # 以便下面的 .xreplace() 操作成功执行：
    # TODO 这个步骤实际上不应该是必要的
    from sympy.simplify.powsimp import powsimp
    # 对表达式 e 进行指数项的简化操作
    f = powsimp(e, deep=True, combine='exp')

    # 对于 O2 中的每对 (a, b)，用 b 替换 f 中的所有 a
    for a, b in O2:
        f = f.xreplace({a: b})

    # 对于 Omega 中的每个 (_, var)，确保 f 中不包含 var
    for _, var in Omega:
        assert not f.has(var)

    # 计算对数 w 的值，将结果存储在 logw 中
    logw = g.exp
    # 如果 sig 等于 1，则取相反数，即 log(w) 变为 log(1/w)=-log(w)
    if sig == 1:
        logw = -logw

    # SymPy 的某些部分在计算具有非整数指数的级数展开时可能遇到困难。
    # 下面的启发式方法可以改善这种情况：
    # 计算所有分母的最小公倍数作为指数
    exponent = reduce(ilcm, denominators, 1)
    # 将 wsym 替换为 wsym 的 exponent 次方
    f = f.subs({wsym: wsym**exponent})
    # 将 logw 除以 exponent
    logw /= exponent

    # 对于特定情况下的 f，需要使用 bottom_up 函数进行处理，
    # 以确保正确简化，而不扩展多项式。
    f = bottom_up(f, lambda w: getattr(w, 'normal', lambda: w)())
    # 将 f 进行乘法和加法的展开
    f = expand_mul(f)

    # 返回计算结果 f 和 logw
    return f, logw
# 定义一个函数 gruntz，用于计算在点 z0 处使用 Gruntz 算法计算 e(z) 的极限。

def gruntz(e, z, z0, dir="+"):
    """
    Compute the limit of e(z) at the point z0 using the Gruntz algorithm.

    Explanation
    ===========

    ``z0`` can be any expression, including oo and -oo.

    For ``dir="+"`` (default) it calculates the limit from the right
    (z->z0+) and for ``dir="-"`` the limit from the left (z->z0-). For infinite z0
    (oo or -oo), the dir argument does not matter.

    This algorithm is fully described in the module docstring in the gruntz.py
    file. It relies heavily on the series expansion. Most frequently, gruntz()
    is only used if the faster limit() function (which uses heuristics) fails.
    """
    
    # 如果 z 不是符号，则抛出 NotImplementedError
    if not z.is_symbol:
        raise NotImplementedError("Second argument must be a Symbol")

    # 将所有的极限转换为 z->oo 的极限；z 的符号在 limitinf 函数中处理
    r = None
    if z0 in (oo, I*oo):
        e0 = e
    elif z0 in (-oo, -I*oo):
        e0 = e.subs(z, -z)
    else:
        # 根据 dir 参数的值来选择替换 z0 的方式，计算 e0
        if str(dir) == "-":
            e0 = e.subs(z, z0 - 1/z)
        elif str(dir) == "+":
            e0 = e.subs(z, z0 + 1/z)
        else:
            raise NotImplementedError("dir must be '+' or '-'")

    # 调用 limitinf 函数计算极限 r
    r = limitinf(e0, z)

    # 这是一个启发式的处理方式，以获得良好的结果... 我们总是将可处理的函数重写为熟悉的难处理函数。
    # 可能更好的方式是将其完全重写为最初的样子，但那需要一些实现工作。
    # 使用 rewrite 方法将结果 r 深度重写为 'intractable' 形式
    return r.rewrite('intractable', deep=True)
```