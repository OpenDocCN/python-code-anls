# `D:\src\scipysrc\sympy\sympy\core\mod.py`

```
# 导入从其他模块导入的类和函数
from .add import Add
from .exprtools import gcd_terms
from .function import Function
from .kind import NumberKind
from .logic import fuzzy_and, fuzzy_not
from .mul import Mul
from .numbers import equal_valued
from .singleton import S

# 定义 Mod 类，继承自 Function 类
class Mod(Function):
    """Represents a modulo operation on symbolic expressions.

    Parameters
    ==========

    p : Expr
        Dividend.

    q : Expr
        Divisor.

    Notes
    =====

    The convention used is the same as Python's: the remainder always has the
    same sign as the divisor.

    Many objects can be evaluated modulo ``n`` much faster than they can be
    evaluated directly (or at all).  For this, ``evaluate=False`` is
    necessary to prevent eager evaluation:

    >>> from sympy import binomial, factorial, Mod, Pow
    >>> Mod(Pow(2, 10**16, evaluate=False), 97)
    61
    >>> Mod(factorial(10**9, evaluate=False), 10**9 + 9)
    712524808
    >>> Mod(binomial(10**18, 10**12, evaluate=False), (10**5 + 3)**2)
    3744312326

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> x**2 % y
    Mod(x**2, y)
    >>> _.subs({x: 5, y: 6})
    1

    """

    kind = NumberKind  # 设置类属性 kind 为 NumberKind

    @classmethod
    def _eval_is_integer(self):
        # 检查 Mod 对象的参数是否都是整数且除数不为零
        p, q = self.args
        if fuzzy_and([p.is_integer, q.is_integer, fuzzy_not(q.is_zero)]):
            return True

    def _eval_is_nonnegative(self):
        # 检查 Mod 对象的除数是否为正数
        if self.args[1].is_positive:
            return True

    def _eval_is_nonpositive(self):
        # 检查 Mod 对象的除数是否为负数
        if self.args[1].is_negative:
            return True

    def _eval_rewrite_as_floor(self, a, b, **kwargs):
        # 将 Mod 表达式重写为 floor 函数的形式
        from sympy.functions.elementary.integers import floor
        return a - b*floor(a/b)

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        # 返回 Mod 表达式的主导项，利用 floor 函数进行重写
        from sympy.functions.elementary.integers import floor
        return self.rewrite(floor)._eval_as_leading_term(x, logx=logx, cdir=cdir)

    def _eval_nseries(self, x, n, logx, cdir=0):
        # 对 Mod 表达式进行 n 级数展开，利用 floor 函数进行重写
        from sympy.functions.elementary.integers import floor
        return self.rewrite(floor)._eval_nseries(x, n, logx=logx, cdir=cdir)
```