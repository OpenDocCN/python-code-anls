# `D:\src\scipysrc\sympy\sympy\polys\rationaltools.py`

```
"""
Tools for manipulation of rational expressions.
"""

# 从 sympy.core 中导入 Basic, Add, sympify 等模块
from sympy.core import Basic, Add, sympify
# 从 sympy.core.exprtools 中导入 gcd_terms 函数
from sympy.core.exprtools import gcd_terms
# 从 sympy.utilities 中导入 public 模块
from sympy.utilities import public
# 从 sympy.utilities.iterables 中导入 iterable 函数
from sympy.utilities.iterables import iterable

# 使用 @public 装饰器声明 together 函数为公共函数
@public
def together(expr, deep=False, fraction=True):
    """
    Denest and combine rational expressions using symbolic methods.

    This function takes an expression or a container of expressions
    and puts it (them) together by denesting and combining rational
    subexpressions. No heroic measures are taken to minimize degree
    of the resulting numerator and denominator. To obtain completely
    reduced expression use :func:`~.cancel`. However, :func:`~.together`
    can preserve as much as possible of the structure of the input
    expression in the output (no expansion is performed).

    A wide variety of objects can be put together including lists,
    tuples, sets, relational objects, integrals and others. It is
    also possible to transform interior of function applications,
    by setting ``deep`` flag to ``True``.

    By definition, :func:`~.together` is a complement to :func:`~.apart`,
    so ``apart(together(expr))`` should return expr unchanged. Note
    however, that :func:`~.together` uses only symbolic methods, so
    it might be necessary to use :func:`~.cancel` to perform algebraic
    simplification and minimize degree of the numerator and denominator.

    Examples
    ========

    >>> from sympy import together, exp
    >>> from sympy.abc import x, y, z

    >>> together(1/x + 1/y)
    (x + y)/(x*y)
    >>> together(1/x + 1/y + 1/z)
    (x*y + x*z + y*z)/(x*y*z)

    >>> together(1/(x*y) + 1/y**2)
    (x + y)/(x*y**2)

    >>> together(1/(1 + 1/x) + 1/(1 + 1/y))
    (x*(y + 1) + y*(x + 1))/((x + 1)*(y + 1))

    >>> together(exp(1/x + 1/y))
    exp(1/y + 1/x)
    >>> together(exp(1/x + 1/y), deep=True)
    exp((x + y)/(x*y))

    >>> together(1/exp(x) + 1/(x*exp(x)))
    (x + 1)*exp(-x)/x

    >>> together(1/exp(2*x) + 1/(x*exp(3*x)))
    (x*exp(x) + 1)*exp(-3*x)/x

    """
    # 定义内部函数 _together，用于实际处理表达式的合并
    def _together(expr):
        # 如果 expr 是 Basic 类型的对象
        if isinstance(expr, Basic):
            # 如果是原子或者不需要深度处理的函数，则直接返回表达式
            if expr.is_Atom or (expr.is_Function and not deep):
                return expr
            # 如果是加法表达式，则递归地对每个子表达式应用 _together 函数，并使用 gcd_terms 合并
            elif expr.is_Add:
                return gcd_terms(list(map(_together, Add.make_args(expr))), fraction=fraction)
            # 如果是幂函数表达式，则处理其底数和指数
            elif expr.is_Pow:
                base = _together(expr.base)

                # 根据 deep 标志决定是否递归处理指数
                if deep:
                    exp = _together(expr.exp)
                else:
                    exp = expr.exp

                return expr.func(base, exp)
            # 对于其他类型的函数，递归地应用 _together 函数到每个参数上
            else:
                return expr.func(*[_together(arg) for arg in expr.args])
        # 如果 expr 是可迭代对象，则递归地应用 _together 函数到每个元素上
        elif iterable(expr):
            return expr.__class__([_together(ex) for ex in expr])

        return expr

    # 对输入的表达式先进行 sympify 处理，然后应用 _together 函数
    return _together(sympify(expr))
```