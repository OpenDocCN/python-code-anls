# `D:\src\scipysrc\sympy\sympy\unify\rewrite.py`

```
""" Functions to support rewriting of SymPy expressions """

# 导入需要的模块和函数
from sympy.core.expr import Expr
from sympy.assumptions import ask
from sympy.strategies.tools import subs
from sympy.unify.usympy import rebuild, unify

# 定义重写规则函数
def rewriterule(source, target, variables=(), condition=None, assume=None):
    """ Rewrite rule.

    Transform expressions that match source into expressions that match target
    treating all ``variables`` as wilds.

    Examples
    ========

    >>> from sympy.abc import w, x, y, z
    >>> from sympy.unify.rewrite import rewriterule
    >>> from sympy import default_sort_key

    # 创建重写规则对象，将符合条件的表达式转换为目标表达式
    rl = rewriterule(x + y, x**y, [x, y])

    # 对结果进行排序并输出
    >>> sorted(rl(z + 3), key=default_sort_key)
    [3**z, z**3]

    # 使用条件函数指定额外要求
    >>> rl = rewriterule(x + y, x**y, [x, y], lambda x, y: x.is_integer)

    # 输出匹配结果
    >>> list(rl(z + 3))
    [3**z]

    # 使用假设条件指定额外要求
    >>> from sympy.assumptions import Q
    >>> rl = rewriterule(x + y, x**y, [x, y], assume=Q.integer(x))

    # 输出匹配结果
    >>> list(rl(z + 3))
    [3**z]

    # 在规则运行时提供本地上下文的假设条件
    >>> list(rl(w + z, Q.integer(z)))
    [z**w]
    """

    # 定义内部的重写函数
    def rewrite_rl(expr, assumptions=True):
        # 遍历所有匹配 source 的表达式
        for match in unify(source, expr, {}, variables=variables):
            # 检查条件函数是否满足
            if (condition and
                not condition(*[match.get(var, var) for var in variables])):
                continue
            # 检查假设条件是否满足
            if (assume and not ask(assume.xreplace(match), assumptions)):
                continue
            # 对匹配的表达式进行替换，并重建表达式结构
            expr2 = subs(match)(target)
            if isinstance(expr2, Expr):
                expr2 = rebuild(expr2)
            yield expr2

    # 返回内部定义的重写函数
    return rewrite_rl
```