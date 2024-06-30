# `D:\src\scipysrc\sympy\sympy\assumptions\wrapper.py`

```
"""
Functions and wrapper object to call assumption property and predicate
query with same syntax.

In SymPy, there are two assumption systems. Old assumption system is
defined in sympy/core/assumptions, and it can be accessed by attribute
such as ``x.is_even``. New assumption system is defined in
sympy/assumptions, and it can be accessed by predicates such as
``Q.even(x)``.

Old assumption is fast, while new assumptions can freely take local facts.
In general, old assumption is used in evaluation method and new assumption
is used in refinement method.

In most cases, both evaluation and refinement follow the same process, and
the only difference is which assumption system is used. This module provides
``is_[...]()`` functions and ``AssumptionsWrapper()`` class which allows
using two systems with same syntax so that parallel code implementation can be
avoided.

Examples
========

For multiple use, use ``AssumptionsWrapper()``.

>>> from sympy import Q, Symbol
>>> from sympy.assumptions.wrapper import AssumptionsWrapper
>>> x = Symbol('x')
>>> _x = AssumptionsWrapper(x, Q.even(x))
>>> _x.is_integer
True
>>> _x.is_odd
False

For single use, use ``is_[...]()`` functions.

>>> from sympy.assumptions.wrapper import is_infinite
>>> a = Symbol('a')
>>> print(is_infinite(a))
None
>>> is_infinite(a, Q.finite(a))
False

"""

from sympy.assumptions import ask, Q  # 导入 ask 函数和 Q 对象
from sympy.core.basic import Basic  # 导入 Basic 类
from sympy.core.sympify import _sympify  # 导入 _sympify 函数


def make_eval_method(fact):
    """
    Create a method that evaluates a fact using the ask function.

    Parameters
    ==========
    fact : str
        The fact to evaluate.

    Returns
    =======
    function
        A function that evaluates the fact using the ask function.

    """
    def getit(self):
        pred = getattr(Q, fact)  # 获取 Q 对象中对应的属性
        ret = ask(pred(self.expr), self.assumptions)  # 使用 ask 函数进行查询
        return ret
    return getit


# we subclass Basic to use the fact deduction and caching
class AssumptionsWrapper(Basic):
    """
    Wrapper over ``Basic`` instances to call predicate query by
    ``.is_[...]`` property

    Parameters
    ==========

    expr : Basic
        The expression to apply assumptions to.

    assumptions : Boolean, optional
        The assumptions to apply to the expression.

    Examples
    ========

    >>> from sympy import Q, Symbol
    >>> from sympy.assumptions.wrapper import AssumptionsWrapper
    >>> x = Symbol('x', even=True)
    >>> AssumptionsWrapper(x).is_integer
    True
    >>> y = Symbol('y')
    >>> AssumptionsWrapper(y, Q.even(y)).is_integer
    True

    With ``AssumptionsWrapper``, both evaluation and refinement can be supported
    by single implementation.

    >>> from sympy import Function
    >>> class MyAbs(Function):
    ...     @classmethod
    ...     def eval(cls, x, assumptions=True):
    ...         _x = AssumptionsWrapper(x, assumptions)
    ...         if _x.is_nonnegative:
    ...             return x
    ...         if _x.is_negative:
    ...             return -x
    ...     def _eval_refine(self, assumptions):
    ...         return MyAbs.eval(self.args[0], assumptions)
    >>> MyAbs(x)
    MyAbs(x)
    >>> MyAbs(x).refine(Q.positive(x))
    x
    >>> MyAbs(Symbol('y', negative=True))
    -y

    """
    # 定义一个特殊方法 __new__，用于创建新对象
    def __new__(cls, expr, assumptions=None):
        # 如果没有传入 assumptions 参数，则直接返回表达式 expr
        if assumptions is None:
            return expr
        # 使用父类的 __new__ 方法创建新对象 obj，并使用 _sympify 处理 assumptions
        obj = super().__new__(cls, expr, _sympify(assumptions))
        # 设置对象的 expr 属性为传入的表达式 expr
        obj.expr = expr
        # 设置对象的 assumptions 属性为处理后的 assumptions
        obj.assumptions = assumptions
        # 返回创建的对象 obj
        return obj

    # 定义一个方法 _eval_is_algebraic，使用 make_eval_method 创建"algebraic"评估方法
    _eval_is_algebraic = make_eval_method("algebraic")
    # 定义一个方法 _eval_is_antihermitian，使用 make_eval_method 创建"antihermitian"评估方法
    _eval_is_antihermitian = make_eval_method("antihermitian")
    # 定义一个方法 _eval_is_commutative，使用 make_eval_method 创建"commutative"评估方法
    _eval_is_commutative = make_eval_method("commutative")
    # 定义一个方法 _eval_is_complex，使用 make_eval_method 创建"complex"评估方法
    _eval_is_complex = make_eval_method("complex")
    # 定义一个方法 _eval_is_composite，使用 make_eval_method 创建"composite"评估方法
    _eval_is_composite = make_eval_method("composite")
    # 定义一个方法 _eval_is_even，使用 make_eval_method 创建"even"评估方法
    _eval_is_even = make_eval_method("even")
    # 定义一个方法 _eval_is_extended_negative，使用 make_eval_method 创建"extended_negative"评估方法
    _eval_is_extended_negative = make_eval_method("extended_negative")
    # 定义一个方法 _eval_is_extended_nonnegative，使用 make_eval_method 创建"extended_nonnegative"评估方法
    _eval_is_extended_nonnegative = make_eval_method("extended_nonnegative")
    # 定义一个方法 _eval_is_extended_nonpositive，使用 make_eval_method 创建"extended_nonpositive"评估方法
    _eval_is_extended_nonpositive = make_eval_method("extended_nonpositive")
    # 定义一个方法 _eval_is_extended_nonzero，使用 make_eval_method 创建"extended_nonzero"评估方法
    _eval_is_extended_nonzero = make_eval_method("extended_nonzero")
    # 定义一个方法 _eval_is_extended_positive，使用 make_eval_method 创建"extended_positive"评估方法
    _eval_is_extended_positive = make_eval_method("extended_positive")
    # 定义一个方法 _eval_is_extended_real，使用 make_eval_method 创建"extended_real"评估方法
    _eval_is_extended_real = make_eval_method("extended_real")
    # 定义一个方法 _eval_is_finite，使用 make_eval_method 创建"finite"评估方法
    _eval_is_finite = make_eval_method("finite")
    # 定义一个方法 _eval_is_hermitian，使用 make_eval_method 创建"hermitian"评估方法
    _eval_is_hermitian = make_eval_method("hermitian")
    # 定义一个方法 _eval_is_imaginary，使用 make_eval_method 创建"imaginary"评估方法
    _eval_is_imaginary = make_eval_method("imaginary")
    # 定义一个方法 _eval_is_infinite，使用 make_eval_method 创建"infinite"评估方法
    _eval_is_infinite = make_eval_method("infinite")
    # 定义一个方法 _eval_is_integer，使用 make_eval_method 创建"integer"评估方法
    _eval_is_integer = make_eval_method("integer")
    # 定义一个方法 _eval_is_irrational，使用 make_eval_method 创建"irrational"评估方法
    _eval_is_irrational = make_eval_method("irrational")
    # 定义一个方法 _eval_is_negative，使用 make_eval_method 创建"negative"评估方法
    _eval_is_negative = make_eval_method("negative")
    # 定义一个方法 _eval_is_noninteger，使用 make_eval_method 创建"noninteger"评估方法
    _eval_is_noninteger = make_eval_method("noninteger")
    # 定义一个方法 _eval_is_nonnegative，使用 make_eval_method 创建"nonnegative"评估方法
    _eval_is_nonnegative = make_eval_method("nonnegative")
    # 定义一个方法 _eval_is_nonpositive，使用 make_eval_method 创建"nonpositive"评估方法
    _eval_is_nonpositive = make_eval_method("nonpositive")
    # 定义一个方法 _eval_is_nonzero，使用 make_eval_method 创建"nonzero"评估方法
    _eval_is_nonzero = make_eval_method("nonzero")
    # 定义一个方法 _eval_is_odd，使用 make_eval_method 创建"odd"评估方法
    _eval_is_odd = make_eval_method("odd")
    # 定义一个方法 _eval_is_polar，使用 make_eval_method 创建"polar"评估方法
    _eval_is_polar = make_eval_method("polar")
    # 定义一个方法 _eval_is_positive，使用 make_eval_method 创建"positive"评估方法
    _eval_is_positive = make_eval_method("positive")
    # 定义一个方法 _eval_is_prime，使用 make_eval_method 创建"prime"评估方法
    _eval_is_prime = make_eval_method("prime")
    # 定义一个方法 _eval_is_rational，使用 make_eval_method 创建"rational"评估方法
    _eval_is_rational = make_eval_method("rational")
    # 定义一个方法 _eval_is_real，使用 make_eval_method 创建"real"评估方法
    _eval_is_real = make_eval_method("real")
    # 定义一个方法 _eval_is_transcendental，使用 make_eval_method 创建"transcendental"评估方法
    _eval_is_transcendental = make_eval_method("transcendental")
    # 定义一个方法 _eval_is_zero，使用 make_eval_method 创建"zero"评估方法
    _eval_is_zero = make_eval_method("zero")
# 判断对象是否为无穷的函数，比 AssumptionsWrapper 更快的一次性函数

def is_infinite(obj, assumptions=None):
    # 如果未提供假设条件，则直接返回对象的无穷属性
    if assumptions is None:
        return obj.is_infinite
    # 否则，使用提供的假设条件询问对象是否为无穷
    return ask(Q.infinite(obj), assumptions)


# 判断对象是否为扩展实数的函数，比 AssumptionsWrapper 更快的一次性函数
def is_extended_real(obj, assumptions=None):
    # 如果未提供假设条件，则直接返回对象的扩展实数属性
    if assumptions is None:
        return obj.is_extended_real
    # 否则，使用提供的假设条件询问对象是否为扩展实数
    return ask(Q.extended_real(obj), assumptions)


# 判断对象是否为扩展非负实数的函数，比 AssumptionsWrapper 更快的一次性函数
def is_extended_nonnegative(obj, assumptions=None):
    # 如果未提供假设条件，则直接返回对象的扩展非负实数属性
    if assumptions is None:
        return obj.is_extended_nonnegative
    # 否则，使用提供的假设条件询问对象是否为扩展非负实数
    return ask(Q.extended_nonnegative(obj), assumptions)
```