# `D:\src\scipysrc\sympy\sympy\assumptions\predicates\order.py`

```
# 导入符号计算库中的谓词类 Predicate
from sympy.assumptions import Predicate
# 导入多分派方法的调度器 Dispatcher
from sympy.multipledispatch import Dispatcher

# 定义 NegativePredicate 类，继承自 Predicate
class NegativePredicate(Predicate):
    r"""
    Negative number predicate.

    Explanation
    ===========

    ``Q.negative(x)`` is true iff ``x`` is a real number and :math:`x < 0`, that is,
    it is in the interval :math:`(-\infty, 0)`.  Note in particular that negative
    infinity is not negative.

    A few important facts about negative numbers:

    - Note that ``Q.nonnegative`` and ``~Q.negative`` are *not* the same
        thing. ``~Q.negative(x)`` simply means that ``x`` is not negative,
        whereas ``Q.nonnegative(x)`` means that ``x`` is real and not
        negative, i.e., ``Q.nonnegative(x)`` is logically equivalent to
        ``Q.zero(x) | Q.positive(x)``.  So for example, ``~Q.negative(I)`` is
        true, whereas ``Q.nonnegative(I)`` is false.

    - See the documentation of ``Q.real`` for more information about
        related facts.

    Examples
    ========

    >>> from sympy import Q, ask, symbols, I
    >>> x = symbols('x')
    >>> ask(Q.negative(x), Q.real(x) & ~Q.positive(x) & ~Q.zero(x))
    True
    >>> ask(Q.negative(-1))
    True
    >>> ask(Q.nonnegative(I))
    False
    >>> ask(~Q.negative(I))
    True

    """
    # 设置谓词名称为 'negative'
    name = 'negative'
    # 定义多分派调度器，用于处理 Q.negative 的情况
    handler = Dispatcher(
        "NegativeHandler",
        doc=("Handler for Q.negative. Test that an expression is strictly less"
        " than zero.")
    )


# 定义 NonNegativePredicate 类，继承自 Predicate
class NonNegativePredicate(Predicate):
    """
    Nonnegative real number predicate.

    Explanation
    ===========

    ``ask(Q.nonnegative(x))`` is true iff ``x`` belongs to the set of
    positive numbers including zero.

    - Note that ``Q.nonnegative`` and ``~Q.negative`` are *not* the same
        thing. ``~Q.negative(x)`` simply means that ``x`` is not negative,
        whereas ``Q.nonnegative(x)`` means that ``x`` is real and not
        negative, i.e., ``Q.nonnegative(x)`` is logically equivalent to
        ``Q.zero(x) | Q.positive(x)``.  So for example, ``~Q.negative(I)`` is
        true, whereas ``Q.nonnegative(I)`` is false.

    Examples
    ========

    >>> from sympy import Q, ask, I
    >>> ask(Q.nonnegative(1))
    True
    >>> ask(Q.nonnegative(0))
    True
    >>> ask(Q.nonnegative(-1))
    False
    >>> ask(Q.nonnegative(I))
    False
    >>> ask(Q.nonnegative(-I))
    False

    """
    # 设置谓词名称为 'nonnegative'
    name = 'nonnegative'
    # 定义多分派调度器，用于处理 Q.nonnegative 的情况
    handler = Dispatcher(
        "NonNegativeHandler",
        doc=("Handler for Q.nonnegative.")
    )


# 定义 NonZeroPredicate 类，继承自 Predicate
class NonZeroPredicate(Predicate):
    """
    Nonzero real number predicate.

    Explanation
    ===========

    ``ask(Q.nonzero(x))`` is true iff ``x`` is real and ``x`` is not zero.  Note in
    particular that ``Q.nonzero(x)`` is false if ``x`` is not real.  Use
    ``~Q.zero(x)`` if you want the negation of being zero without any real
    assumptions.

    A few important facts about nonzero numbers:

    - ``Q.nonzero`` is logically equivalent to ``Q.positive | Q.negative``.


    """
    # 设置谓词名称为 'nonzero'
    name = 'nonzero'
    # 没有定义 handler，因此这个谓词没有多分派方法的处理器
    # 通常在这里应该定义一个处理 Q.nonzero 的多分派调度器，但在示例中没有展示出来
    """
    name = 'nonzero'
    handler = Dispatcher(
        "NonZeroHandler",
        doc=("Handler for key 'nonzero'. Test that an expression is not identically"
        " zero.")
    )



    # 定义名字为 'nonzero' 的处理器名称
    name = 'nonzero'
    # 创建一个调度器 Dispatcher 对象，用于处理 'nonzero' 的逻辑
    handler = Dispatcher(
        "NonZeroHandler",
        # 设置该处理器的文档字符串，说明它用于检测表达式不是完全为零
        doc=("Handler for key 'nonzero'. Test that an expression is not identically"
        " zero.")
    )
class ZeroPredicate(Predicate):
    """
    Zero number predicate.

    Explanation
    ===========

    ``ask(Q.zero(x))`` is true iff the value of ``x`` is zero.

    Examples
    ========

    >>> from sympy import ask, Q, oo, symbols
    >>> x, y = symbols('x, y')
    >>> ask(Q.zero(0))
    True
    >>> ask(Q.zero(1/oo))
    True
    >>> print(ask(Q.zero(0*oo)))
    None
    >>> ask(Q.zero(1))
    False
    >>> ask(Q.zero(x*y), Q.zero(x) | Q.zero(y))
    True

    """
    name = 'zero'
    # 定义处理器，用于处理'zero'关键字的查询
    handler = Dispatcher(
        "ZeroHandler",
        doc="Handler for key 'zero'."
    )


class NonPositivePredicate(Predicate):
    """
    Nonpositive real number predicate.

    Explanation
    ===========

    ``ask(Q.nonpositive(x))`` is true iff ``x`` belongs to the set of
    negative numbers including zero.

    - Note that ``Q.nonpositive`` and ``~Q.positive`` are *not* the same
        thing. ``~Q.positive(x)`` simply means that ``x`` is not positive,
        whereas ``Q.nonpositive(x)`` means that ``x`` is real and not
        positive, i.e., ``Q.nonpositive(x)`` is logically equivalent to
        `Q.negative(x) | Q.zero(x)`.  So for example, ``~Q.positive(I)`` is
        true, whereas ``Q.nonpositive(I)`` is false.

    Examples
    ========

    >>> from sympy import Q, ask, I

    >>> ask(Q.nonpositive(-1))
    True
    >>> ask(Q.nonpositive(0))
    True
    >>> ask(Q.nonpositive(1))
    False
    >>> ask(Q.nonpositive(I))
    False
    >>> ask(Q.nonpositive(-I))
    False

    """
    name = 'nonpositive'
    # 定义处理器，用于处理'nonpositive'关键字的查询
    handler = Dispatcher(
        "NonPositiveHandler",
        doc="Handler for key 'nonpositive'."
    )


class PositivePredicate(Predicate):
    r"""
    Positive real number predicate.

    Explanation
    ===========

    ``Q.positive(x)`` is true iff ``x`` is real and `x > 0`, that is if ``x``
    is in the interval `(0, \infty)`.  In particular, infinity is not
    positive.

    A few important facts about positive numbers:

    - Note that ``Q.nonpositive`` and ``~Q.positive`` are *not* the same
        thing. ``~Q.positive(x)`` simply means that ``x`` is not positive,
        whereas ``Q.nonpositive(x)`` means that ``x`` is real and not
        positive, i.e., ``Q.nonpositive(x)`` is logically equivalent to
        `Q.negative(x) | Q.zero(x)`.  So for example, ``~Q.positive(I)`` is
        true, whereas ``Q.nonpositive(I)`` is false.

    - See the documentation of ``Q.real`` for more information about
        related facts.

    Examples
    ========

    >>> from sympy import Q, ask, symbols, I
    >>> x = symbols('x')
    >>> ask(Q.positive(x), Q.real(x) & ~Q.negative(x) & ~Q.zero(x))
    True
    >>> ask(Q.positive(1))
    True
    >>> ask(Q.nonpositive(I))
    False
    >>> ask(~Q.positive(I))
    True

    """
    name = 'positive'
    # 定义处理器，用于处理'positive'关键字的查询
    handler = Dispatcher(
        "PositiveHandler",
        doc=("Handler for key 'positive'. Test that an expression is strictly"
        " greater than zero.")
    )
class ExtendedPositivePredicate(Predicate):
    r"""
    Positive extended real number predicate.

    Explanation
    ===========

    ``Q.extended_positive(x)`` is true iff ``x`` is extended real and
    `x > 0`, that is if ``x`` is in the interval `(0, \infty]`.

    Examples
    ========

    >>> from sympy import ask, I, oo, Q
    >>> ask(Q.extended_positive(1))
    True
    >>> ask(Q.extended_positive(oo))
    True
    >>> ask(Q.extended_positive(I))
    False

    """
    # 设置谓词名称为 'extended_positive'
    name = 'extended_positive'
    # 设置处理程序为 ExtendedPositiveHandler 的调度器
    handler = Dispatcher("ExtendedPositiveHandler")


class ExtendedNegativePredicate(Predicate):
    r"""
    Negative extended real number predicate.

    Explanation
    ===========

    ``Q.extended_negative(x)`` is true iff ``x`` is extended real and
    `x < 0`, that is if ``x`` is in the interval `[-\infty, 0)`.

    Examples
    ========

    >>> from sympy import ask, I, oo, Q
    >>> ask(Q.extended_negative(-1))
    True
    >>> ask(Q.extended_negative(-oo))
    True
    >>> ask(Q.extended_negative(-I))
    False

    """
    # 设置谓词名称为 'extended_negative'
    name = 'extended_negative'
    # 设置处理程序为 ExtendedNegativeHandler 的调度器
    handler = Dispatcher("ExtendedNegativeHandler")


class ExtendedNonZeroPredicate(Predicate):
    """
    Nonzero extended real number predicate.

    Explanation
    ===========

    ``ask(Q.extended_nonzero(x))`` is true iff ``x`` is extended real and
    ``x`` is not zero.

    Examples
    ========

    >>> from sympy import ask, I, oo, Q
    >>> ask(Q.extended_nonzero(-1))
    True
    >>> ask(Q.extended_nonzero(oo))
    True
    >>> ask(Q.extended_nonzero(I))
    False

    """
    # 设置谓词名称为 'extended_nonzero'
    name = 'extended_nonzero'
    # 设置处理程序为 ExtendedNonZeroHandler 的调度器
    handler = Dispatcher("ExtendedNonZeroHandler")


class ExtendedNonPositivePredicate(Predicate):
    """
    Nonpositive extended real number predicate.

    Explanation
    ===========

    ``ask(Q.extended_nonpositive(x))`` is true iff ``x`` is extended real and
    ``x`` is not positive.

    Examples
    ========

    >>> from sympy import ask, I, oo, Q
    >>> ask(Q.extended_nonpositive(-1))
    True
    >>> ask(Q.extended_nonpositive(oo))
    False
    >>> ask(Q.extended_nonpositive(0))
    True
    >>> ask(Q.extended_nonpositive(I))
    False

    """
    # 设置谓词名称为 'extended_nonpositive'
    name = 'extended_nonpositive'
    # 设置处理程序为 ExtendedNonPositiveHandler 的调度器
    handler = Dispatcher("ExtendedNonPositiveHandler")


class ExtendedNonNegativePredicate(Predicate):
    """
    Nonnegative extended real number predicate.

    Explanation
    ===========

    ``ask(Q.extended_nonnegative(x))`` is true iff ``x`` is extended real and
    ``x`` is not negative.

    Examples
    ========

    >>> from sympy import ask, I, oo, Q
    >>> ask(Q.extended_nonnegative(-1))
    False
    >>> ask(Q.extended_nonnegative(oo))
    True
    >>> ask(Q.extended_nonnegative(0))
    True
    >>> ask(Q.extended_nonnegative(I))
    False

    """
    # 设置谓词名称为 'extended_nonnegative'
    name = 'extended_nonnegative'
    # 设置处理程序为 ExtendedNonNegativeHandler 的调度器
    handler = Dispatcher("ExtendedNonNegativeHandler")
```