# `D:\src\scipysrc\sympy\sympy\series\limits.py`

```
from sympy.calculus.accumulationbounds import AccumBounds
from sympy.core import S, Symbol, Add, sympify, Expr, PoleError, Mul
from sympy.core.exprtools import factor_terms
from sympy.core.numbers import Float, _illegal
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import (Abs, sign, arg, re)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.special.gamma_functions import gamma
from sympy.polys import PolynomialError, factor
from sympy.series.order import Order
from .gruntz import gruntz

def limit(e, z, z0, dir="+"):
    """Computes the limit of ``e(z)`` at the point ``z0``.

    Parameters
    ==========

    e : expression, the limit of which is to be taken

    z : symbol representing the variable in the limit.
        Other symbols are treated as constants. Multivariate limits
        are not supported.

    z0 : the value toward which ``z`` tends. Can be any expression,
        including ``oo`` and ``-oo``.

    dir : string, optional (default: "+")
        The limit is bi-directional if ``dir="+-"``, from the right
        (z->z0+) if ``dir="+"``, and from the left (z->z0-) if
        ``dir="-"``. For infinite ``z0`` (``oo`` or ``-oo``), the ``dir``
        argument is determined from the direction of the infinity
        (i.e., ``dir="-"`` for ``oo``).

    Examples
    ========

    >>> from sympy import limit, sin, oo
    >>> from sympy.abc import x
    >>> limit(sin(x)/x, x, 0)
    1
    >>> limit(1/x, x, 0) # default dir='+'
    oo
    >>> limit(1/x, x, 0, dir="-")
    -oo
    >>> limit(1/x, x, 0, dir='+-')
    zoo
    >>> limit(1/x, x, oo)
    0

    Notes
    =====

    First we try some heuristics for easy and frequent cases like "x", "1/x",
    "x**2" and similar, so that it's fast. For all other cases, we use the
    Gruntz algorithm (see the gruntz() function).

    See Also
    ========

    limit_seq : returns the limit of a sequence.
    """

    # 调用 Limit 类的 doit 方法计算表达式 e 关于变量 z 在点 z0 处的极限
    return Limit(e, z, z0, dir).doit(deep=False)


def heuristics(e, z, z0, dir):
    """Computes the limit of an expression term-wise.
    Parameters are the same as for the ``limit`` function.
    Works with the arguments of expression ``e`` one by one, computing
    the limit of each and then combining the results. This approach
    works only for simple limits, but it is fast.
    """

    # 如果 z0 是正无穷，则计算 e(z) 在 z=1/z 的情况下的极限
    if z0 is S.Infinity:
        rv = limit(e.subs(z, 1/z), z, S.Zero, "+")
        # 如果结果是 Limit 对象，则返回 None
        if isinstance(rv, Limit):
            return
    # 如果表达式 e 是乘法、加法、幂运算或函数类型的实例，则执行以下操作
    elif e.is_Mul or e.is_Add or e.is_Pow or e.is_Function:
        # 初始化结果列表 r
        r = []
        # 导入 sympy 中的 together 函数用于合并分式
        from sympy.simplify.simplify import together
        # 遍历 e 的所有参数 a
        for a in e.args:
            # 计算参数 a 关于变量 z 在点 z0 处的极限，方向为 dir
            l = limit(a, z, z0, dir)
            # 如果极限结果包含无穷大且不是有限的
            if l.has(S.Infinity) and l.is_finite is None:
                # 如果 e 是加法表达式的实例
                if isinstance(e, Add):
                    # 对 e 进行因式分解
                    m = factor_terms(e)
                    # 如果因式分解结果不是乘法表达式，则尝试合并分式
                    if not isinstance(m, Mul):
                        m = together(m)
                    # 如果合并分式后仍不是乘法表达式，则尝试因式分解
                    if not isinstance(m, Mul):
                        m = factor(e)
                    # 如果最终结果是乘法表达式，则应用启发式算法处理
                    if isinstance(m, Mul):
                        return heuristics(m, z, z0, dir)
                    return  # 返回空值
                return  # 返回空值
            # 如果极限结果是 Limit 类型的实例
            elif isinstance(l, Limit):
                return  # 返回空值
            # 如果极限结果是 NaN (Not a Number)
            elif l is S.NaN:
                return  # 返回空值
            else:
                # 将极限结果加入到结果列表 r 中
                r.append(l)
        # 如果结果列表 r 不为空
        if r:
            # 对参数 e 应用其函数类型的构造函数，并传入结果列表 r 作为参数
            rv = e.func(*r)
            # 如果 rv 是 NaN 并且 e 是乘法表达式，且 r 中有 AccumBounds 类型的参数
            if rv is S.NaN and e.is_Mul and any(isinstance(rr, AccumBounds) for rr in r):
                # 初始化新的结果列表 r2 和表达式列表 e2
                r2 = []
                e2 = []
                # 遍历结果列表 r 中的每个元素 rval
                for ii, rval in enumerate(r):
                    # 如果 rval 是 AccumBounds 类型，则加入 r2
                    if isinstance(rval, AccumBounds):
                        r2.append(rval)
                    else:
                        # 否则加入到表达式列表 e2
                        e2.append(e.args[ii])

                # 如果表达式列表 e2 的长度大于 0
                if len(e2) > 0:
                    # 将表达式列表 e2 中的元素相乘并简化
                    e3 = Mul(*e2).simplify()
                    # 计算简化后表达式 e3 关于变量 z 在点 z0 处的极限
                    l = limit(e3, z, z0, dir)
                    # 计算最终结果 rv
                    rv = l * Mul(*r2)

            # 如果最终结果 rv 是 NaN
            if rv is S.NaN:
                try:
                    # 导入 sympy 中的 ratsimp 函数进行有理化简
                    from sympy.simplify.ratsimp import ratsimp
                    # 对表达式 e 进行有理化简
                    rat_e = ratsimp(e)
                except PolynomialError:
                    return  # 返回空值
                # 如果有理化简后仍然是 NaN 或者等于原表达式 e
                if rat_e is S.NaN or rat_e == e:
                    return  # 返回空值
                # 计算有理化简后的表达式关于变量 z 在点 z0 处的极限
                return limit(rat_e, z, z0, dir)
    # 返回最终结果 rv
    return rv
class Limit(Expr):
    """Represents an unevaluated limit.

    Examples
    ========

    >>> from sympy import Limit, sin
    >>> from sympy.abc import x
    >>> Limit(sin(x)/x, x, 0)
    Limit(sin(x)/x, x, 0, dir='+')
    >>> Limit(1/x, x, 0, dir="-")
    Limit(1/x, x, 0, dir='-')

    """

    def __new__(cls, e, z, z0, dir="+"):
        # 将输入参数转换为符号表达式
        e = sympify(e)
        z = sympify(z)
        z0 = sympify(z0)

        # 根据极限点的类型调整方向
        if z0 in (S.Infinity, S.ImaginaryUnit*S.Infinity):
            dir = "-"
        elif z0 in (S.NegativeInfinity, S.ImaginaryUnit*S.NegativeInfinity):
            dir = "+"

        # 检查极限点是否包含自变量，这种情况不支持
        if(z0.has(z)):
            raise NotImplementedError("Limits approaching a variable point are"
                    " not supported (%s -> %s)" % (z, z0))
        # 检查方向参数的类型和取值
        if isinstance(dir, str):
            dir = Symbol(dir)
        elif not isinstance(dir, Symbol):
            raise TypeError("direction must be of type basestring or "
                    "Symbol, not %s" % type(dir))
        if str(dir) not in ('+', '-', '+-'):
            raise ValueError("direction must be one of '+', '-' "
                    "or '+-', not %s" % dir)

        # 创建新的 Limit 对象
        obj = Expr.__new__(cls)
        obj._args = (e, z, z0, dir)
        return obj


    @property
    def free_symbols(self):
        # 获取自由符号集合
        e = self.args[0]
        isyms = e.free_symbols
        isyms.difference_update(self.args[1].free_symbols)
        isyms.update(self.args[2].free_symbols)
        return isyms


    def pow_heuristics(self, e):
        # 根据幂函数的特性处理表达式
        _, z, z0, _ = self.args
        b1, e1 = e.base, e.exp
        if not b1.has(z):
            # 若基数不包含自变量，则计算幂函数的极限
            res = limit(e1*log(b1), z, z0)
            return exp(res)

        # 计算基数和指数函数在极限情况下的表达式
        ex_lim = limit(e1, z, z0)
        base_lim = limit(b1, z, z0)

        if base_lim is S.One:
            if ex_lim in (S.Infinity, S.NegativeInfinity):
                # 特殊情况：基数为1，指数为无穷时的处理
                res = limit(e1*(b1 - 1), z, z0)
                return exp(res)
        if base_lim is S.NegativeInfinity and ex_lim is S.Infinity:
            # 特殊情况：基数为负无穷，指数为正无穷时的处理
            return S.ComplexInfinity
```