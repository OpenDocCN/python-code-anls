# `D:\src\scipysrc\sympy\sympy\strategies\tools.py`

```
# 导入当前目录下的 rl 模块
from . import rl
# 从 core 模块导入 do_one, exhaust, switch 函数
from .core import do_one, exhaust, switch
# 从 traverse 模块导入 top_down 函数
from .traverse import top_down


def subs(d, **kwargs):
    """ Full simultaneous exact substitution.

    Examples
    ========

    >>> from sympy.strategies.tools import subs
    >>> from sympy import Basic, S
    >>> mapping = {S(1): S(4), S(4): S(1), Basic(S(5)): Basic(S(6), S(7))}
    >>> expr = Basic(S(1), Basic(S(2), S(3)), Basic(S(4), Basic(S(5))))
    >>> subs(mapping)(expr)
    Basic(4, Basic(2, 3), Basic(1, Basic(6, 7)))
    """
    # 如果字典 d 非空，则进行全局的替换操作，并返回结果
    if d:
        # 使用 top_down 函数应用 rl.subs 规则映射到 d 的每一对键值对
        return top_down(do_one(*map(rl.subs, *zip(*d.items()))), **kwargs)
    else:
        # 如果字典 d 为空，则返回一个恒等函数，不对输入做任何修改
        return lambda x: x


def canon(*rules, **kwargs):
    """ Strategy for canonicalization.

    Explanation
    ===========

    Apply each rule in a bottom_up fashion through the tree.
    Do each one in turn.
    Keep doing this until there is no change.
    """
    # 应用 rules 中的每一个规则，通过树的 bottom_up 方式进行规范化处理
    return exhaust(top_down(exhaust(do_one(*rules)), **kwargs))


def typed(ruletypes):
    """ Apply rules based on the expression type

    inputs:
        ruletypes -- a dict mapping {Type: rule}

    Examples
    ========

    >>> from sympy.strategies import rm_id, typed
    >>> from sympy import Add, Mul
    >>> rm_zeros = rm_id(lambda x: x==0)
    >>> rm_ones  = rm_id(lambda x: x==1)
    >>> remove_idents = typed({Add: rm_zeros, Mul: rm_ones})
    """
    # 根据表达式的类型应用对应的规则
    return switch(type, ruletypes)
```