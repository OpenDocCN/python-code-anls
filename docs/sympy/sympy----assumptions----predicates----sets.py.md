# `D:\src\scipysrc\sympy\sympy\assumptions\predicates\sets.py`

```
from sympy.assumptions import Predicate  # 导入 Predicate 类，用于创建谓词
from sympy.multipledispatch import Dispatcher  # 导入 Dispatcher 类，用于多重分派


class IntegerPredicate(Predicate):
    """
    Integer predicate.

    Explanation
    ===========

    ``Q.integer(x)`` is true iff ``x`` belongs to the set of integer
    numbers.

    Examples
    ========

    >>> from sympy import Q, ask, S
    >>> ask(Q.integer(5))
    True
    >>> ask(Q.integer(S(1)/2))
    False

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Integer

    """
    name = 'integer'  # 定义谓词名称为 'integer'
    handler = Dispatcher(
        "IntegerHandler",  # 创建名为 'IntegerHandler' 的 Dispatcher 对象
        doc=("Handler for Q.integer.\n\n"
        "Test that an expression belongs to the field of integer numbers.")
    )


class NonIntegerPredicate(Predicate):
    """
    Non-integer extended real predicate.
    """
    name = 'noninteger'  # 定义谓词名称为 'noninteger'
    handler = Dispatcher(
        "NonIntegerHandler",  # 创建名为 'NonIntegerHandler' 的 Dispatcher 对象
        doc=("Handler for Q.noninteger.\n\n"
        "Test that an expression is a non-integer extended real number.")
    )


class RationalPredicate(Predicate):
    """
    Rational number predicate.

    Explanation
    ===========

    ``Q.rational(x)`` is true iff ``x`` belongs to the set of
    rational numbers.

    Examples
    ========

    >>> from sympy import ask, Q, pi, S
    >>> ask(Q.rational(0))
    True
    >>> ask(Q.rational(S(1)/2))
    True
    >>> ask(Q.rational(pi))
    False

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Rational_number

    """
    name = 'rational'  # 定义谓词名称为 'rational'
    handler = Dispatcher(
        "RationalHandler",  # 创建名为 'RationalHandler' 的 Dispatcher 对象
        doc=("Handler for Q.rational.\n\n"
        "Test that an expression belongs to the field of rational numbers.")
    )


class IrrationalPredicate(Predicate):
    """
    Irrational number predicate.

    Explanation
    ===========

    ``Q.irrational(x)`` is true iff ``x``  is any real number that
    cannot be expressed as a ratio of integers.

    Examples
    ========

    >>> from sympy import ask, Q, pi, S, I
    >>> ask(Q.irrational(0))
    False
    >>> ask(Q.irrational(S(1)/2))
    False
    >>> ask(Q.irrational(pi))
    True
    >>> ask(Q.irrational(I))
    False

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Irrational_number

    """
    name = 'irrational'  # 定义谓词名称为 'irrational'
    handler = Dispatcher(
        "IrrationalHandler",  # 创建名为 'IrrationalHandler' 的 Dispatcher 对象
        doc=("Handler for Q.irrational.\n\n"
        "Test that an expression is irrational numbers.")
    )


class RealPredicate(Predicate):
    r"""
    Real number predicate.

    Explanation
    ===========

    ``Q.real(x)`` is true iff ``x`` is a real number, i.e., it is in the
    interval `(-\infty, \infty)`.  Note that, in particular the
    infinities are not real. Use ``Q.extended_real`` if you want to
    consider those as well.

    A few important facts about reals:

    - Every real number is positive, negative, or zero.  Furthermore,
        because these sets are pairwise disjoint, each real number is
        exactly one of those three.
    ```
   ```python
# 导入 Predicate 类，用于创建谓词
from sympy.assumptions import Predicate  
# 导入 Dispatcher 类，用于多重分派
from sympy.multipledispatch import Dispatcher  


class IntegerPredicate(Predicate):
    """
    Integer predicate.

    Explanation
    ===========

    ``Q.integer(x)`` is true iff ``x`` belongs to the set of integer
    numbers.

    Examples
    ========

    >>> from sympy import Q, ask, S
    >>> ask(Q.integer(5))
    True
    >>> ask(Q.integer(S(1)/2))
    False

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Integer

    """
    name = 'integer'  # 定义谓词名称为 'integer'
    handler = Dispatcher(
        "IntegerHandler",  # 创建名为 'IntegerHandler' 的 Dispatcher 对象
        doc=("Handler for Q.integer.\n\n"
        "Test that an expression belongs to the field of integer numbers.")
    )


class NonIntegerPredicate(Predicate):
    """
    Non-integer extended real predicate.
    """
    name = 'noninteger'  # 定义谓词名称为 'noninteger'
    handler = Dispatcher(
        "NonIntegerHandler",  # 创建名为 'NonIntegerHandler' 的 Dispatcher 对象
        doc=("Handler for Q.noninteger.\n\n"
        "Test that an expression is a non-integer extended real number.")
    )


class RationalPredicate(Predicate):
    """
    Rational number predicate.

    Explanation
    ===========

    ``Q.rational(x)`` is true iff ``x`` belongs to the set of
    rational numbers.

    Examples
    ========

    >>> from sympy import ask, Q, pi, S
    >>> ask(Q.rational(0))
    True
    >>> ask(Q.rational(S(1)/2))
    True
    >>> ask(Q.rational(pi))
    False

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Rational_number

    """
    name = 'rational'  # 定义谓词名称为 'rational'
    handler = Dispatcher(
        "RationalHandler",  # 创建名为 'RationalHandler' 的 Dispatcher 对象
        doc=("Handler for Q.rational.\n\n"
        "Test that an expression belongs to the field of rational numbers.")
    )


class IrrationalPredicate(Predicate):
    """
    Irrational number predicate.

    Explanation
    ===========

    ``Q.irrational(x)`` is true iff ``x``  is any real number that
    cannot be expressed as a ratio of integers.

    Examples
    ========

    >>> from sympy import ask, Q, pi, S, I
    >>> ask(Q.irrational(0))
    False
    >>> ask(Q.irrational(S(1)/2))
    False
    >>> ask(Q.irrational(pi))
    True
    >>> ask(Q.irrational(I))
    False

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Irrational_number

    """
    name = 'irrational'  # 定义谓词名称为 'irrational'
    handler = Dispatcher(
        "IrrationalHandler",  # 创建名为 'IrrationalHandler' 的 Dispatcher 对象
        doc=("Handler for Q.irrational.\n\n"
        "Test that an expression is irrational numbers.")
    )


class RealPredicate(Predicate):
    r"""
    Real number predicate.

    Explanation
    ===========

    ``Q.real(x)`` is true iff ``x`` is a real number, i.e., it is in the
    interval `(-\infty, \infty)`.  Note that, in particular the
    infinities are not real. Use ``Q.extended_real`` if you want to
    consider those as well.

    A few important facts about reals:

    - Every real number is positive, negative, or zero.  Furthermore,
        because these sets are pairwise disjoint, each real number is
        exactly one of those three.
    """
   These comments annotate each section of the provided Python code, explaining the purpose and functionality of the classes and their components. If you have any more questions or need further clarification, feel free to ask!
    # 设定名称为'real'，用于标识 Q.real 的属性或特性
    name = 'real'
    # 创建一个分发器对象，用于处理 Q.real 的相关属性
    handler = Dispatcher(
        "RealHandler",
        doc=("Handler for Q.real.\n\n"
        "Test that an expression belongs to the field of real numbers.")
    )
# 定义一个扩展实数谓词类，继承自 Predicate 类
class ExtendedRealPredicate(Predicate):
    r"""
    Extended real predicate.

    Explanation
    ===========

    ``Q.extended_real(x)`` is true iff ``x`` is a real number or
    `\{-\infty, \infty\}`.

    See documentation of ``Q.real`` for more information about related
    facts.

    Examples
    ========

    >>> from sympy import ask, Q, oo, I
    >>> ask(Q.extended_real(1))
    True
    >>> ask(Q.extended_real(I))
    False
    >>> ask(Q.extended_real(oo))
    True

    """
    # 设置谓词的名称为 'extended_real'
    name = 'extended_real'
    # 定义一个调度程序，用于处理 Q.extended_real 的请求
    handler = Dispatcher(
        "ExtendedRealHandler",
        doc=("Handler for Q.extended_real.\n\n"
        "Test that an expression belongs to the field of extended real\n"
        "numbers, that is real numbers union {Infinity, -Infinity}.")
    )


# 定义一个厄米特算子谓词类，继承自 Predicate 类
class HermitianPredicate(Predicate):
    """
    Hermitian predicate.

    Explanation
    ===========

    ``ask(Q.hermitian(x))`` is true iff ``x`` belongs to the set of
    Hermitian operators.

    References
    ==========

    .. [1] https://mathworld.wolfram.com/HermitianOperator.html

    """
    # TODO: Add examples (添加示例的待办事项)
    # 设置谓词的名称为 'hermitian'
    name = 'hermitian'
    # 定义一个调度程序，用于处理 Q.hermitian 的请求
    handler = Dispatcher(
        "HermitianHandler",
        doc=("Handler for Q.hermitian.\n\n"
        "Test that an expression belongs to the field of Hermitian operators.")
    )


# 定义一个复数谓词类，继承自 Predicate 类
class ComplexPredicate(Predicate):
    """
    Complex number predicate.

    Explanation
    ===========

    ``Q.complex(x)`` is true iff ``x`` belongs to the set of complex
    numbers. Note that every complex number is finite.

    Examples
    ========

    >>> from sympy import Q, Symbol, ask, I, oo
    >>> x = Symbol('x')
    >>> ask(Q.complex(0))
    True
    >>> ask(Q.complex(2 + 3*I))
    True
    >>> ask(Q.complex(oo))
    False

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Complex_number

    """
    # 设置谓词的名称为 'complex'
    name = 'complex'
    # 定义一个调度程序，用于处理 Q.complex 的请求
    handler = Dispatcher(
        "ComplexHandler",
        doc=("Handler for Q.complex.\n\n"
        "Test that an expression belongs to the field of complex numbers.")
    )


# 定义一个虚数谓词类，继承自 Predicate 类
class ImaginaryPredicate(Predicate):
    """
    Imaginary number predicate.

    Explanation
    ===========

    ``Q.imaginary(x)`` is true iff ``x`` can be written as a real
    number multiplied by the imaginary unit ``I``. Please note that ``0``
    is not considered to be an imaginary number.

    Examples
    ========

    >>> from sympy import Q, ask, I
    >>> ask(Q.imaginary(3*I))
    True
    >>> ask(Q.imaginary(2 + 3*I))
    False
    >>> ask(Q.imaginary(0))
    False

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Imaginary_number

    """
    # 设置谓词的名称为 'imaginary'
    name = 'imaginary'
    # 定义一个调度程序，用于处理 Q.imaginary 的请求
    handler = Dispatcher(
        "ImaginaryHandler",
        doc=("Handler for Q.imaginary.\n\n"
        "Test that an expression belongs to the field of imaginary numbers,\n"
        "that is, numbers in the form x*I, where x is real.")
    )


# 定义一个反厄米特算子谓词类，继承自 Predicate 类
class AntihermitianPredicate(Predicate):
    """
    Antihermitian predicate.

    Explanation
    ===========
    # Q.antihermitian(x) 是真的当且仅当 x 属于反厄米算子的域，即形式为 x*I 的算子，其中 x 是厄米算子。
    # 这里引用了厄米算子的定义和相关信息，可以参考 [1] 中的链接。
    """
    # TODO: Add examples
    # 定义处理器名称为 'AntiHermitianHandler'，用于处理 Q.antihermitian。
    # 文档字符串描述了这个处理器的功能，即测试表达式是否属于反厄米算子的域，
    # 即形式为 x*I 的算子，其中 x 是厄米算子。
    name = 'antihermitian'
    handler = Dispatcher(
        "AntiHermitianHandler",
        doc=("Handler for Q.antihermitian.\n\n"
        "Test that an expression belongs to the field of anti-Hermitian\n"
        "operators, that is, operators in the form x*I, where x is Hermitian.")
    )
class AlgebraicPredicate(Predicate):
    r"""
    Algebraic number predicate.

    Explanation
    ===========

    ``Q.algebraic(x)`` is true iff ``x`` belongs to the set of
    algebraic numbers. ``x`` is algebraic if there is some polynomial
    in ``p(x)\in \mathbb\{Q\}[x]`` such that ``p(x) = 0``.

    Examples
    ========

    >>> from sympy import ask, Q, sqrt, I, pi
    >>> ask(Q.algebraic(sqrt(2)))
    True
    >>> ask(Q.algebraic(I))
    True
    >>> ask(Q.algebraic(pi))
    False

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Algebraic_number

    """
    # 定义谓词名称为 'algebraic'
    name = 'algebraic'
    # 创建一个分发器对象，用于处理 'Q.algebraic' 的键
    AlgebraicHandler = Dispatcher(
        "AlgebraicHandler",
        doc="""Handler for Q.algebraic key."""
    )


class TranscendentalPredicate(Predicate):
    """
    Transcedental number predicate.

    Explanation
    ===========

    ``Q.transcendental(x)`` is true iff ``x`` belongs to the set of
    transcendental numbers. A transcendental number is a real
    or complex number that is not algebraic.

    """
    # TODO: Add examples
    # 定义谓词名称为 'transcendental'
    name = 'transcendental'
    # 创建一个分发器对象，用于处理 'Q.transcendental' 的键
    handler = Dispatcher(
        "Transcendental",
        doc="""Handler for Q.transcendental key."""
    )
```