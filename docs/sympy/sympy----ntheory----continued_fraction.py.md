# `D:\src\scipysrc\sympy\sympy\ntheory\continued_fraction.py`

```
# 导入必要的库
from __future__ import annotations
import itertools
from sympy.core.exprtools import factor_terms  # 导入用于因式分解的函数
from sympy.core.numbers import Integer, Rational  # 导入整数和有理数类
from sympy.core.singleton import S  # 导入单例类
from sympy.core.symbol import Dummy  # 导入虚拟符号类
from sympy.core.sympify import _sympify  # 导入用于将输入转换为SymPy表达式的函数
from sympy.utilities.misc import as_int  # 导入用于将输入转换为整数的函数

# 定义函数：计算有理数或二次无理数的连分数表示
def continued_fraction(a) -> list:
    """Return the continued fraction representation of a Rational or
    quadratic irrational.

    Examples
    ========

    >>> from sympy.ntheory.continued_fraction import continued_fraction
    >>> from sympy import sqrt
    >>> continued_fraction((1 + 2*sqrt(3))/5)
    [0, 1, [8, 3, 34, 3]]

    See Also
    ========
    continued_fraction_periodic, continued_fraction_reduce, continued_fraction_convergents
    """
    e = _sympify(a)  # 将输入参数转换为SymPy表达式

    # 检查所有子表达式是否为有理数
    if all(i.is_Rational for i in e.atoms()):
        # 如果整个表达式是整数，则调用特定函数计算其连分数
        if e.is_Integer:
            return continued_fraction_periodic(e, 1, 0)
        # 如果整个表达式是有理数，则调用特定函数计算其连分数
        elif e.is_Rational:
            return continued_fraction_periodic(e.p, e.q, 0)
        # 如果表达式是指数为1/2的整数幂，则调用特定函数计算其连分数
        elif e.is_Pow and e.exp is S.Half and e.base.is_Integer:
            return continued_fraction_periodic(0, 1, e.base)
        # 如果表达式是两个参数的乘积，第一个是有理数，第二个是指数为1/2的整数幂，则调用特定函数计算其连分数
        elif e.is_Mul and len(e.args) == 2 and (
                e.args[0].is_Rational and
                e.args[1].is_Pow and
                e.args[1].base.is_Integer and
                e.args[1].exp is S.Half):
            a, b = e.args
            return continued_fraction_periodic(0, a.q, b.base, a.p)
        else:
            # 如果无法简化，则展开表达式并计算其分子和分母
            p, d = e.expand().as_numer_denom()
            # 如果分母是整数，则调用特定函数计算其连分数
            if d.is_Integer:
                if p.is_Rational:
                    return continued_fraction_periodic(p, d)
                # 如果表达式是一个加法操作，包含两个参数，则检查是否可以处理
                if p.is_Add and len(p.args) == 2:
                    a, bc = p.args
                else:
                    a = S.Zero
                    bc = p
                if a.is_Integer:
                    b = S.NaN
                    if bc.is_Mul and len(bc.args) == 2:
                        b, c = bc.args
                    elif bc.is_Pow:
                        b = Integer(1)
                        c = bc
                    if b.is_Integer and (
                            c.is_Pow and c.exp is S.Half and
                            c.base.is_Integer):
                        # 如果可以简化为形式(a + b*sqrt(c))/d，则调用特定函数计算其连分数
                        c = c.base
                        return continued_fraction_periodic(a, d, c, b)
    # 如果表达式不是有理数或二次无理数，则抛出值错误
    raise ValueError(
        'expecting a rational or quadratic irrational, not %s' % e)


# 定义函数：计算二次无理数的周期性连分数展开
def continued_fraction_periodic(p, q, d=0, s=1) -> list:
    r"""
    Find the periodic continued fraction expansion of a quadratic irrational.

    Compute the continued fraction expansion of a rational or a
    """
    Compute the continued fraction representation for a quadratic irrational number.

    Returns the continued fraction representation (canonical form) as
    a list of integers, optionally ending (for quadratic irrationals)
    with a list of integers representing the repeating digits.

    Parameters
    ==========

    p : int
        the rational part of the number's numerator
    q : int
        the denominator of the number
    d : int, optional
        the irrational part (discriminator) of the number's numerator
    s : int, optional
        the coefficient of the irrational part

    Examples
    ========

    >>> from sympy.ntheory.continued_fraction import continued_fraction_periodic
    >>> continued_fraction_periodic(3, 2, 7)
    [2, [1, 4, 1, 1]]

    Golden ratio has the simplest continued fraction expansion:

    >>> continued_fraction_periodic(1, 2, 5)
    [[1]]

    If the discriminator is zero or a perfect square then the number will be a
    rational number:

    >>> continued_fraction_periodic(4, 3, 0)
    [1, 3]
    >>> continued_fraction_periodic(4, 3, 49)
    [3, 1, 2]

    See Also
    ========

    continued_fraction_iterator, continued_fraction_reduce

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Periodic_continued_fraction
    .. [2] K. Rosen. Elementary Number theory and its applications.
           Addison-Wesley, 3 Sub edition, pages 379-381, January 1992.

    """
    from sympy.functions import sqrt, floor  # 导入所需的数学函数

    p, q, d, s = list(map(as_int, [p, q, d, s]))  # 将输入参数转换为整数类型

    if d < 0:
        raise ValueError("expected non-negative for `d` but got %s" % d)  # 如果 d 小于 0，抛出异常

    if q == 0:
        raise ValueError("The denominator cannot be 0.")  # 如果 q 等于 0，抛出异常

    if not s:
        d = 0  # 如果 s 为 0，将 d 设为 0

    # 检查是否为有理数情况
    sd = sqrt(d)
    if sd.is_Integer:
        return list(continued_fraction_iterator(Rational(p + s*sd, q)))  # 如果 sd 是整数，返回简化的连分数形式

    # 非有理数情况，且 sd 不是整数
    if q < 0:
        p, q, s = -p, -q, -s  # 如果 q 小于 0，调整 p, q, s 的符号

    n = (p + s*sd)/q
    if n < 0:
        w = floor(-n)  # 如果 n 小于 0，向下取整得到 w
        f = -n - w  # 计算小数部分 f
        one_f = continued_fraction(1 - f)  # 计算 1-f 的连分数
        one_f[0] -= w + 1  # 调整第一个项
        return one_f  # 返回调整后的连分数列表

    d *= s**2
    sd *= s

    if (d - p**2)%q:
        d *= q**2
        sd *= q
        p *= q
        q *= q  # 调整 p, q, d, sd 以确保余数为零

    terms: list[int] = []
    pq = {}

    while (p, q) not in pq:
        pq[(p, q)] = len(terms)  # 记录当前的 p, q 对应的位置
        terms.append((p + sd)//q)  # 计算下一个项并添加到 terms 中
        p = terms[-1]*q - p  # 更新 p
        q = (d - p**2)//q  # 更新 q

    i = pq[(p, q)]  # 获取重复项的起始位置
    return terms[:i] + [terms[i:]]  # 返回连分数的列表表示，包括重复部分
# 将一个连分数化简为有理数或二次无理数。

def continued_fraction_reduce(cf):
    """
    Reduce a continued fraction to a rational or quadratic irrational.

    Compute the rational or quadratic irrational number from its
    terminating or periodic continued fraction expansion.  The
    continued fraction expansion (cf) should be supplied as a
    terminating iterator supplying the terms of the expansion.  For
    terminating continued fractions, this is equivalent to
    ``list(continued_fraction_convergents(cf))[-1]``, only a little more
    efficient.  If the expansion has a repeating part, a list of the
    repeating terms should be returned as the last element from the
    iterator.  This is the format returned by
    continued_fraction_periodic.

    For quadratic irrationals, returns the largest solution found,
    which is generally the one sought, if the fraction is in canonical
    form (all terms positive except possibly the first).

    Examples
    ========

    >>> from sympy.ntheory.continued_fraction import continued_fraction_reduce
    >>> continued_fraction_reduce([1, 2, 3, 4, 5])
    225/157
    >>> continued_fraction_reduce([-2, 1, 9, 7, 1, 2])
    -256/233
    >>> continued_fraction_reduce([2, 1, 2, 1, 1, 4, 1, 1, 6, 1, 1, 8]).n(10)
    2.718281835
    >>> continued_fraction_reduce([1, 4, 2, [3, 1]])
    (sqrt(21) + 287)/238
    >>> continued_fraction_reduce([[1]])
    (1 + sqrt(5))/2
    >>> from sympy.ntheory.continued_fraction import continued_fraction_periodic
    >>> continued_fraction_reduce(continued_fraction_periodic(8, 5, 13))
    (sqrt(13) + 8)/5

    See Also
    ========

    continued_fraction_periodic

    """

    # 导入所需的 solve 函数和符号运算相关的类
    from sympy.solvers import solve
    from sympy import Dummy, S
    from sympy.core.expr import factor_terms

    # 初始化周期部分为空列表和符号变量 x
    period = []
    x = Dummy('x')

    # 定义将迭代器转换为列表的函数
    def untillist(cf):
        for nxt in cf:
            if isinstance(nxt, list):
                period.extend(nxt)
                yield x
                break
            yield nxt

    a = S.Zero
    # 计算连分数收敛值
    for a in continued_fraction_convergents(untillist(cf)):
        pass

    # 如果存在周期部分
    if period:
        # 定义符号变量 y 并解方程
        y = Dummy('y')
        solns = solve(continued_fraction_reduce(period + [y]) - y, y)
        solns.sort()
        pure = solns[-1]
        # 替换 a 中的 x 为纯解，并进行有理化简
        rv = a.subs(x, pure).radsimp()
    else:
        rv = a
    
    # 如果 rv 是加法表达式，则因式分解并检查是否需要处理负数情况
    if rv.is_Add:
        rv = factor_terms(rv)
        if rv.is_Mul and rv.args[0] == -1:
            rv = rv.func(*rv.args)
    
    # 返回结果 rv
    return rv


def continued_fraction_iterator(x):
    """
    Return continued fraction expansion of x as iterator.

    Examples
    ========

    >>> from sympy import Rational, pi
    >>> from sympy.ntheory.continued_fraction import continued_fraction_iterator

    >>> list(continued_fraction_iterator(Rational(3, 8)))
    [0, 2, 1, 2]
    >>> list(continued_fraction_iterator(Rational(-3, 8)))
    [-1, 1, 1, 1, 2]

    >>> for i, v in enumerate(continued_fraction_iterator(pi)):
    ...     if i > 7:
    ...         break
    ...     print(v)
    3
    7
    15
    1
    292
    1
    1
    1

    References
    ==========


    """
    """
        .. [1] https://en.wikipedia.org/wiki/Continued_fraction
    
        从 sympy 库中导入 floor 函数
        """
        from sympy.functions import floor
        # 无限循环，生成器函数，用于实现连分数的计算
        while True:
            # 取 x 的向下取整值
            i = floor(x)
            # 生成器返回当前的整数部分 i
            yield i
            # 更新 x 为原始 x 减去取整后的值
            x -= i
            # 如果 x 等于 0，则跳出循环
            if not x:
                break
            # 更新 x 为 1/x，用于下一次迭代
            x = 1/x
# 返回一个迭代器，生成给定连分数（cf）的收敛值（convergents）

def continued_fraction_convergents(cf):
    """
    Return an iterator over the convergents of a continued fraction (cf).

    The parameter should be in either of the following to forms:
    - A list of partial quotients, possibly with the last element being a list
    of repeating partial quotients, such as might be returned by
    continued_fraction and continued_fraction_periodic.
    - An iterable returning successive partial quotients of the continued
    fraction, such as might be returned by continued_fraction_iterator.

    In computing the convergents, the continued fraction need not be strictly
    in canonical form (all integers, all but the first positive).
    Rational and negative elements may be present in the expansion.

    Examples
    ========

    >>> from sympy.core import pi
    >>> from sympy import S
    >>> from sympy.ntheory.continued_fraction import \
            continued_fraction_convergents, continued_fraction_iterator

    >>> list(continued_fraction_convergents([0, 2, 1, 2]))
    [0, 1/2, 1/3, 3/8]

    >>> list(continued_fraction_convergents([1, S('1/2'), -7, S('1/4')]))
    [1, 3, 19/5, 7]

    >>> it = continued_fraction_convergents(continued_fraction_iterator(pi))
    >>> for n in range(7):
    ...     print(next(it))
    3
    22/7
    333/106
    355/113
    103993/33102
    104348/33215
    208341/66317

    >>> it = continued_fraction_convergents([1, [1, 2]])  # sqrt(3)
    >>> for n in range(7):
    ...     print(next(it))
    1
    2
    5/3
    7/4
    19/11
    26/15
    71/41

    See Also
    ========

    continued_fraction_iterator, continued_fraction, continued_fraction_periodic

    """
    # 如果cf是一个列表并且最后一个元素是列表，将cf处理为可循环的迭代器
    if isinstance(cf, list) and isinstance(cf[-1], list):
        cf = itertools.chain(cf[:-1], itertools.cycle(cf[-1]))
    
    # 初始化连分数的计算变量
    p_2, q_2 = S.Zero, S.One  # 上上个部分的分子和分母
    p_1, q_1 = S.One, S.Zero  # 上个部分的分子和分母
    
    # 遍历连分数的每个部分，并计算对应的收敛值，使用生成器返回结果
    for a in cf:
        p, q = a*p_1 + p_2, a*q_1 + q_2
        p_2, q_2 = p_1, q_1
        p_1, q_1 = p, q
        yield p/q
```