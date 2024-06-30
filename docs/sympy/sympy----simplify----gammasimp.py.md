# `D:\src\scipysrc\sympy\sympy\simplify\gammasimp.py`

```
from sympy.core import Function, S, Mul, Pow, Add  # 导入需要的 SymPy 核心模块及函数类
from sympy.core.sorting import ordered, default_sort_key  # 导入排序相关函数
from sympy.core.function import expand_func  # 导入函数展开相关函数
from sympy.core.symbol import Dummy  # 导入符号相关类
from sympy.functions import gamma, sqrt, sin  # 导入 gamma 函数及其它数学函数
from sympy.polys import factor, cancel  # 导入多项式相关函数
from sympy.utilities.iterables import sift, uniq  # 导入迭代工具函数


def gammasimp(expr):
    r"""
    Simplify expressions with gamma functions.

    Explanation
    ===========

    This function takes as input an expression containing gamma
    functions or functions that can be rewritten in terms of gamma
    functions and tries to minimize the number of those functions and
    reduce the size of their arguments.

    The algorithm works by rewriting all gamma functions as expressions
    involving rising factorials (Pochhammer symbols) and applies
    recurrence relations and other transformations applicable to rising
    factorials, to reduce their arguments, possibly letting the resulting
    rising factorial to cancel. Rising factorials with the second argument
    being an integer are expanded into polynomial forms and finally all
    other rising factorial are rewritten in terms of gamma functions.

    Then the following two steps are performed.

    1. Reduce the number of gammas by applying the reflection theorem
       gamma(x)*gamma(1-x) == pi/sin(pi*x).
    2. Reduce the number of gammas by applying the multiplication theorem
       gamma(x)*gamma(x+1/n)*...*gamma(x+(n-1)/n) == C*gamma(n*x).

    It then reduces the number of prefactors by absorbing them into gammas
    where possible and expands gammas with rational argument.

    All transformation rules can be found (or were derived from) here:

    .. [1] https://functions.wolfram.com/GammaBetaErf/Pochhammer/17/01/02/
    .. [2] https://functions.wolfram.com/GammaBetaErf/Pochhammer/27/01/0005/

    Examples
    ========

    >>> from sympy.simplify import gammasimp
    >>> from sympy import gamma, Symbol
    >>> from sympy.abc import x
    >>> n = Symbol('n', integer = True)

    >>> gammasimp(gamma(x)/gamma(x - 3))
    (x - 3)*(x - 2)*(x - 1)
    >>> gammasimp(gamma(n + 3))
    gamma(n + 3)

    """

    expr = expr.rewrite(gamma)  # 将表达式重写为包含 gamma 函数的形式

    # compute_ST will be looking for Functions and we don't want
    # it looking for non-gamma functions: issue 22606
    # so we mask free, non-gamma functions
    f = expr.atoms(Function)  # 获取表达式中的所有函数对象
    # take out gammas
    gammas = {i for i in f if isinstance(i, gamma)}  # 找出所有的 gamma 函数
    if not gammas:
        return expr  # 如果没有 gamma 函数，直接返回原表达式以避免影响如因式分解等副作用
    f -= gammas  # 去除 gamma 函数，只保留非 gamma 函数
    # keep only those without bound symbols
    f = f & expr.as_dummy().atoms(Function)  # 保留没有绑定符号的函数
    if f:
        dum, fun, simp = zip(*[
            (Dummy(), fi, fi.func(*[
                _gammasimp(a, as_comb=False) for a in fi.args]))
            for fi in ordered(f)])
        d = expr.xreplace(dict(zip(fun, dum)))  # 用虚拟符号替换函数
        return _gammasimp(d, as_comb=False).xreplace(dict(zip(dum, simp)))

    return _gammasimp(expr, as_comb=False)  # 对表达式应用 gamma 简化算法
# 定义一个辅助函数，用于简化与 gamma 函数相关的表达式，被 gammasimp 和 combsimp 调用

def _gammasimp(expr, as_comb):
    """
    Helper function for gammasimp and combsimp.

    Explanation
    ===========

    Simplifies expressions written in terms of gamma function. If
    as_comb is True, it tries to preserve integer arguments. See
    docstring of gammasimp for more information. This was part of
    combsimp() in combsimp.py.
    """
    # 替换 expr 中的 gamma 函数调用，使用 lambda 函数对每个 n 进行替换
    expr = expr.replace(gamma,
        lambda n: _rf(1, (n - 1).expand()))

    # 如果 as_comb 为 True，则尝试保留整数参数的组合函数
    if as_comb:
        # 替换 expr 中的 _rf 函数调用，使用 lambda 函数对每对 a, b 进行替换
        expr = expr.replace(_rf,
            lambda a, b: gamma(b + 1))
    else:
        # 替换 expr 中的 _rf 函数调用，使用 lambda 函数对每对 a, b 进行替换
        expr = expr.replace(_rf,
            lambda a, b: gamma(a + b)/gamma(a))

    # 对 expr 进行因式分解
    was = factor(expr)
    # (由于某些原因，这里无法使用 Basic.replace)
    # 使用 rule_gamma 处理 was，并与 expr 比较
    expr = rule_gamma(was)
    if expr != was:
        # 如果结果改变，则再次对 expr 进行因式分解
        expr = factor(expr)

    # 替换 expr 中的 gamma 函数调用，使用 lambda 函数对每个 n 进行替换
    expr = expr.replace(gamma,
        lambda n: expand_func(gamma(n)) if n.is_Rational else gamma(n))

    # 返回简化后的表达式
    return expr


class _rf(Function):
    @classmethod
    def eval(cls, a, b):
        # 如果 b 是整数
        if b.is_Integer:
            # 如果 b 是零
            if not b:
                return S.One

            # 将 b 转换为整数
            n = int(b)

            # 如果 n 大于 0
            if n > 0:
                # 返回乘积的表达式
                return Mul(*[a + i for i in range(n)])
            # 如果 n 小于 0
            elif n < 0:
                # 返回倒数的乘积的表达式
                return 1/Mul(*[a - i for i in range(1, -n + 1)])
        else:
            # 如果 b 是加法
            if b.is_Add:
                # 将 b 分解为系数 c 和 _b
                c, _b = b.as_coeff_Add()

                # 如果 c 是整数
                if c.is_Integer:
                    # 如果 c 大于 0
                    if c > 0:
                        # 返回递归调用 _rf 的乘积的表达式
                        return _rf(a, _b)*_rf(a + _b, c)
                    # 如果 c 小于 0
                    elif c < 0:
                        # 返回递归调用 _rf 的倒数乘积的表达式
                        return _rf(a, _b)/_rf(a + _b + c, -c)

            # 如果 a 是加法
            if a.is_Add:
                # 将 a 分解为系数 c 和 _a
                c, _a = a.as_coeff_Add()

                # 如果 c 是整数
                if c.is_Integer:
                    # 如果 c 大于 0
                    if c > 0:
                        # 返回递归调用 _rf 的三个 _rf 函数的乘积的表达式
                        return _rf(_a, b)*_rf(_a + b, c)/_rf(_a, c)
                    # 如果 c 小于 0
                    elif c < 0:
                        # 返回递归调用 _rf 的三个 _rf 函数的倒数乘积的表达式
                        return _rf(_a, b)*_rf(_a + c, -c)/_rf(_a + b + c, -c)
```