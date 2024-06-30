# `D:\src\scipysrc\sympy\sympy\sets\contains.py`

```
from sympy.core import S
from sympy.core.sympify import sympify
from sympy.core.relational import Eq, Ne
from sympy.core.parameters import global_parameters
from sympy.logic.boolalg import Boolean
from sympy.utilities.misc import func_name
from .sets import Set

class Contains(Boolean):
    """
    Asserts that x is an element of the set S.

    Examples
    ========

    >>> from sympy import Symbol, Integer, S, Contains
    >>> Contains(Integer(2), S.Integers)
    True
    >>> Contains(Integer(-2), S.Naturals)
    False
    >>> i = Symbol('i', integer=True)
    >>> Contains(i, S.Naturals)
    Contains(i, Naturals)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Element_%28mathematics%29
    """

    # 定义一个新的包含关系，验证 x 是否属于集合 s
    def __new__(cls, x, s, evaluate=None):
        # 将 x 和 s 转换为 Sympy 对象
        x = sympify(x)
        s = sympify(s)

        # 如果未指定 evaluate 参数，则使用全局参数中的设置
        if evaluate is None:
            evaluate = global_parameters.evaluate

        # 如果 s 不是 Set 类型，则抛出类型错误异常
        if not isinstance(s, Set):
            raise TypeError('expecting Set, not %s' % func_name(s))

        # 如果需要评估结果
        if evaluate:
            # _contains 可能返回符号布尔值，用于 s.contains(x) 的情况，
            # 但是在 Contains(x, s) 的上下文中，我们只需评估为 true、false 或者返回未评估的 Contains。
            result = s._contains(x)

            # 如果结果是布尔类型
            if isinstance(result, Boolean):
                if result in (S.true, S.false):
                    return result
            # 如果结果不是 None，则抛出类型错误异常
            elif result is not None:
                raise TypeError("_contains() should return Boolean or None")

        # 调用父类的 __new__ 方法创建新的 Contains 对象
        return super().__new__(cls, x, s)

    # 返回该包含关系中涉及的二进制符号集合
    @property
    def binary_symbols(self):
        return set().union(*[i.binary_symbols
            for i in self.args[1].args
            if i.is_Boolean or i.is_Symbol or
            isinstance(i, (Eq, Ne))])

    # 返回作为集合的参数
    def as_set(self):
        return self.args[1]
```