# `D:\src\scipysrc\sympy\sympy\solvers\decompogen.py`

```
# 从 sympy 库中导入所需模块和函数
from sympy.core import (Function, Pow, sympify, Expr)
from sympy.core.relational import Relational
from sympy.core.singleton import S
from sympy.polys import Poly, decompose  # 导入多项式相关模块和函数
from sympy.utilities.misc import func_name  # 导入辅助函数 func_name
from sympy.functions.elementary.miscellaneous import Min, Max  # 导入最小值和最大值函数


def decompogen(f, symbol):
    """
    Computes General functional decomposition of ``f``.
    Given an expression ``f``, returns a list ``[f_1, f_2, ..., f_n]``,
    where::
              f = f_1 o f_2 o ... f_n = f_1(f_2(... f_n))

    Note: This is a General decomposition function. It also decomposes
    Polynomials. For only Polynomial decomposition see ``decompose`` in polys.

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy import decompogen, sqrt, sin, cos
    >>> decompogen(sin(cos(x)), x)
    [sin(x), cos(x)]
    >>> decompogen(sin(x)**2 + sin(x) + 1, x)
    [x**2 + x + 1, sin(x)]
    >>> decompogen(sqrt(6*x**2 - 5), x)
    [sqrt(x), 6*x**2 - 5]
    >>> decompogen(sin(sqrt(cos(x**2 + 1))), x)
    [sin(x), sqrt(x), cos(x), x**2 + 1]
    >>> decompogen(x**4 + 2*x**3 - x - 1, x)
    [x**2 - x - 1, x**2 + x]

    """
    # 将输入的表达式 f 转换为 sympy 表达式对象
    f = sympify(f)
    # 检查 f 是否为表达式类型，并且不是关系表达式类型
    if not isinstance(f, Expr) or isinstance(f, Relational):
        raise TypeError('expecting Expr but got: `%s`' % func_name(f))
    # 如果符号 symbol 不在 f 的自由符号中，则直接返回包含 f 的列表
    if symbol not in f.free_symbols:
        return [f]

    # ===== Simple Functions ===== #
    # 处理简单函数和幂函数
    if isinstance(f, (Function, Pow)):
        # 对指数函数特别处理
        if f.is_Pow and f.base == S.Exp1:
            arg = f.exp
        else:
            arg = f.args[0]
        # 如果参数是 symbol，则直接返回函数 f
        if arg == symbol:
            return [f]
        # 否则替换参数并递归地进行分解
        return [f.subs(arg, symbol)] + decompogen(arg, symbol)

    # ===== Min/Max Functions ===== #
    # 处理最小值和最大值函数
    if isinstance(f, (Min, Max)):
        args = list(f.args)
        d0 = None
        for i, a in enumerate(args):
            # 如果参数 a 中不含有 symbol，则跳过
            if not a.has_free(symbol):
                continue
            # 对含有 symbol 的参数进行分解
            d = decompogen(a, symbol)
            # 如果分解结果长度为 1，则加入 symbol 到结果列表
            if len(d) == 1:
                d = [symbol] + d
            # 如果是第一次迭代，则将结果存入 d0
            if d0 is None:
                d0 = d[1:]
            # 如果分解结果不一致，则返回 symbol
            elif d[1:] != d0:
                # decomposition is not the same for each arg:
                # mark as having no decomposition
                d = [symbol]
                break
            args[i] = d[0]
        # 如果结果的第一个元素是 symbol，则直接返回函数 f
        if d[0] == symbol:
            return [f]
        # 否则返回组合后的函数和分解结果
        return [f.func(*args)] + d0

    # ===== Convert to Polynomial ===== #
    # 将表达式转换为多项式
    fp = Poly(f)
    # 获取多项式中的生成元
    gens = list(filter(lambda x: symbol in x.free_symbols, fp.gens))

    # 如果生成元个数为 1 且不等于 symbol，则进行替换和递归分解
    if len(gens) == 1 and gens[0] != symbol:
        f1 = f.subs(gens[0], symbol)
        f2 = gens[0]
        return [f1] + decompogen(f2, symbol)

    # ===== Polynomial decompose() ====== #
    # 尝试使用 polys 模块中的 decompose() 函数进行分解
    try:
        return decompose(f)
    except ValueError:
        return [f]


def compogen(g_s, symbol):
    """
    Returns the composition of functions.
    Given a list of functions ``g_s``, returns their composition ``f``,
    where:
        f = g_1 o g_2 o .. o g_n
    """
    # 函数未实现完整，仅提供函数的目的和输出形式
    pass
    """
    Note: This function recursively composes symbolic expressions in a generalized manner, including polynomial compositions.
    For specific polynomial composition, refer to the `compose` function in the polys module.
    
    Examples
    ========
    
    >>> from sympy.solvers.decompogen import compogen
    >>> from sympy.abc import x
    >>> from sympy import sqrt, sin, cos
    >>> compogen([sin(x), cos(x)], x)
    sin(cos(x))
    >>> compogen([x**2 + x + 1, sin(x)], x)
    sin(x)**2 + sin(x) + 1
    >>> compogen([sqrt(x), 6*x**2 - 5], x)
    sqrt(6*x**2 - 5)
    >>> compogen([sin(x), sqrt(x), cos(x), x**2 + 1], x)
    sin(sqrt(cos(x**2 + 1)))
    >>> compogen([x**2 - x - 1, x**2 + x], x)
    -x**2 - x + (x**2 + x)**2 - 1
    """
    
    # 如果给定的符号表达式列表长度为1，直接返回该表达式
    if len(g_s) == 1:
        return g_s[0]
    
    # 将第一个表达式中的符号用第二个表达式替换，得到组合后的新表达式
    foo = g_s[0].subs(symbol, g_s[1])
    
    # 如果给定的符号表达式列表长度为2，直接返回组合后的表达式
    if len(g_s) == 2:
        return foo
    
    # 递归调用compogen函数，将组合后的表达式和剩余的符号表达式列表继续组合
    return compogen([foo] + g_s[2:], symbol)
```