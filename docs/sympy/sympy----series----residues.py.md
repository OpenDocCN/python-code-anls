# `D:\src\scipysrc\sympy\sympy\series\residues.py`

```
"""
This module implements the Residue function and related tools for working
with residues.
"""

from sympy.core.mul import Mul  # 导入 sympy 核心模块中的乘法类
from sympy.core.singleton import S  # 导入 sympy 核心模块中的单例类 S
from sympy.core.sympify import sympify  # 导入 sympy 核心模块中的 sympify 函数
from sympy.utilities.timeutils import timethis  # 导入 sympy 工具模块中的 timethis 函数


@timethis('residue')
def residue(expr, x, x0):
    """
    Finds the residue of ``expr`` at the point x=x0.

    The residue is defined as the coefficient of ``1/(x-x0)`` in the power series
    expansion about ``x=x0``.

    Examples
    ========

    >>> from sympy import Symbol, residue, sin
    >>> x = Symbol("x")
    >>> residue(1/x, x, 0)
    1
    >>> residue(1/x**2, x, 0)
    0
    >>> residue(2/sin(x), x, 0)
    2

    This function is essential for the Residue Theorem [1].

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Residue_theorem
    """
    # The current implementation uses series expansion to
    # calculate it. A more general implementation is explained in
    # the section 5.6 of the Bronstein's book {M. Bronstein:
    # Symbolic Integration I, Springer Verlag (2005)}. For purely
    # rational functions, the algorithm is much easier. See
    # sections 2.4, 2.5, and 2.7 (this section actually gives an
    # algorithm for computing any Laurent series coefficient for
    # a rational function). The theory in section 2.4 will help to
    # understand why the resultant works in the general algorithm.
    # For the definition of a resultant, see section 1.4 (and any
    # previous sections for more review).

    from sympy.series.order import Order  # 导入 sympy 系列模块中的 Order 类
    from sympy.simplify.radsimp import collect  # 导入 sympy 简化模块中的 collect 函数
    expr = sympify(expr)  # 将输入表达式转换为 sympy 的表达式对象
    if x0 != 0:
        expr = expr.subs(x, x + x0)  # 替换表达式中的 x 为 x + x0
    for n in (0, 1, 2, 4, 8, 16, 32):
        s = expr.nseries(x, n=n)  # 对表达式进行关于 x 展开至 n 阶的级数展开
        if not s.has(Order) or s.getn() >= 0:  # 如果级数 s 中不包含 Order 类，或者级数阶数大于等于 0
            break
    s = collect(s.removeO(), x)  # 移除高阶无穷小 O(x^n)，并且根据 x 对表达式 s 进行整理
    if s.is_Add:
        args = s.args  # 将整理后的表达式 s 拆解为项的列表
    else:
        args = [s]
    res = S.Zero  # 初始化结果为 0
    for arg in args:
        c, m = arg.as_coeff_mul(x)  # 将项 arg 拆解为系数 c 和乘法形式的因子 m
        m = Mul(*m)  # 将乘法形式的因子 m 还原为乘积对象
        if not (m in (S.One, x) or (m.is_Pow and m.exp.is_Integer)):
            raise NotImplementedError('term of unexpected form: %s' % m)
        if m == 1/x:  # 如果因子 m 是 1/x
            res += c  # 累加系数 c 到结果 res
    return res  # 返回计算得到的残留值
```