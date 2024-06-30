# `D:\src\scipysrc\sympy\sympy\assumptions\relation\equality.py`

```
"""
Module for mathematical equality [1] and inequalities [2].

The purpose of this module is to provide the instances which represent the
binary predicates in order to combine the relationals into logical inference
system. Objects such as ``Q.eq``, ``Q.lt`` should remain internal to
assumptions module, and user must use the classes such as :obj:`~.Eq()`,
:obj:`~.Lt()` instead to construct the relational expressions.

References
==========

.. [1] https://en.wikipedia.org/wiki/Equality_(mathematics)
.. [2] https://en.wikipedia.org/wiki/Inequality_(mathematics)
"""
from sympy.assumptions import Q
from sympy.core.relational import is_eq, is_neq, is_gt, is_ge, is_lt, is_le

from .binrel import BinaryRelation

__all__ = ['EqualityPredicate', 'UnequalityPredicate', 'StrictGreaterThanPredicate',
    'GreaterThanPredicate', 'StrictLessThanPredicate', 'LessThanPredicate']


class EqualityPredicate(BinaryRelation):
    """
    Binary predicate for $=$.

    The purpose of this class is to provide the instance which represent
    the equality predicate in order to allow the logical inference.
    This class must remain internal to assumptions module and user must
    use :obj:`~.Eq()` instead to construct the equality expression.

    Evaluating this predicate to ``True`` or ``False`` is done by
    :func:`~.core.relational.is_eq()`

    Examples
    ========

    >>> from sympy import ask, Q
    >>> Q.eq(0, 0)
    Q.eq(0, 0)
    >>> ask(_)
    True

    See Also
    ========

    sympy.core.relational.Eq

    """
    is_reflexive = True
    is_symmetric = True

    name = 'eq'
    handler = None  # Do not allow dispatching by this predicate

    @property
    def negated(self):
        return Q.ne

    def eval(self, args, assumptions=True):
        if assumptions == True:
            # default assumptions for is_eq is None
            assumptions = None
        return is_eq(*args, assumptions)


class UnequalityPredicate(BinaryRelation):
    r"""
    Binary predicate for $\neq$.

    The purpose of this class is to provide the instance which represent
    the inequation predicate in order to allow the logical inference.
    This class must remain internal to assumptions module and user must
    use :obj:`~.Ne()` instead to construct the inequation expression.

    Evaluating this predicate to ``True`` or ``False`` is done by
    :func:`~.core.relational.is_neq()`

    Examples
    ========

    >>> from sympy import ask, Q
    >>> Q.ne(0, 0)
    Q.ne(0, 0)
    >>> ask(_)
    False

    See Also
    ========

    sympy.core.relational.Ne

    """
    is_reflexive = False
    is_symmetric = True

    name = 'ne'
    handler = None

    @property
    def negated(self):
        return Q.eq

    def eval(self, args, assumptions=True):
        if assumptions == True:
            # default assumptions for is_neq is None
            assumptions = None
        return is_neq(*args, assumptions)


class StrictGreaterThanPredicate(BinaryRelation):
    """
    Binary predicate for $>$.

    The purpose of this class is to provide the instance which represents
    the strict greater than predicate in order to allow the logical inference.
    This class must remain internal to the assumptions module.

    Evaluating this predicate to ``True`` or ``False`` is done by
    :func:`~.core.relational.is_gt()`

    Examples
    ========

    >>> from sympy import ask, Q
    >>> Q.gt(2, 1)
    Q.gt(2, 1)
    >>> ask(_)
    True

    See Also
    ========

    sympy.core.relational.Gt

    """
    Binary predicate for $>$.

    The purpose of this class is to represent the ">" predicate for logical inference.
    This class is intended to be used internally within the assumptions module.
    Users should construct the greater-than expression using :obj:`~.Gt()` instead.

    Evaluating this predicate to ``True`` or ``False`` is handled by
    :func:`~.core.relational.is_gt()`

    Examples
    ========

    >>> from sympy import ask, Q
    >>> Q.gt(0, 0)
    Q.gt(0, 0)
    >>> ask(_)
    False

    See Also
    ========

    sympy.core.relational.Gt
class GreaterThanPredicate(BinaryRelation):
    """
    Binary predicate for $>=$.

    The purpose of this class is to provide the instance which represents
    the ">=" predicate to enable logical inference.

    This class is intended to remain internal to the assumptions module, and users should
    use :obj:`~.Ge()` instead to construct the equality expression.

    Evaluating this predicate to ``True`` or ``False`` is done by
    :func:`~.core.relational.is_ge()`

    Examples
    ========

    >>> from sympy import ask, Q
    >>> Q.ge(0, 0)
    Q.ge(0, 0)
    >>> ask(_)
    True

    See Also
    ========

    sympy.core.relational.Ge

    """
    is_reflexive = True   # Indicates if the relation is reflexive (True for >=)
    is_symmetric = False  # Indicates if the relation is symmetric (False for >=)

    name = 'ge'           # String representation of the predicate
    handler = None        # Placeholder for a handler function

    @property
    def reversed(self):
        return Q.le       # Property that returns the reversed predicate Q.le (<=)

    @property
    def negated(self):
        return Q.lt       # Property that returns the negated predicate Q.lt (<)

    def eval(self, args, assumptions=True):
        if assumptions == True:
            # default assumptions for is_ge is None
            assumptions = None  # Set assumptions to None if it's True
        return is_ge(*args, assumptions)


class StrictLessThanPredicate(BinaryRelation):
    """
    Binary predicate for $<$.

    The purpose of this class is to provide the instance which represents
    the "<" predicate to enable logical inference.

    This class is intended to remain internal to the assumptions module, and users should
    use :obj:`~.Lt()` instead to construct the equality expression.

    Evaluating this predicate to ``True`` or ``False`` is done by
    :func:`~.core.relational.is_lt()`

    Examples
    ========

    >>> from sympy import ask, Q
    >>> Q.lt(0, 0)
    Q.lt(0, 0)
    >>> ask(_)
    False

    See Also
    ========

    sympy.core.relational.Lt

    """
    is_reflexive = False  # Indicates if the relation is reflexive (False for <)
    is_symmetric = False  # Indicates if the relation is symmetric (False for <)

    name = 'lt'           # String representation of the predicate
    handler = None        # Placeholder for a handler function

    @property
    def reversed(self):
        return Q.gt       # Property that returns the reversed predicate Q.gt (>)

    @property
    def negated(self):
        return Q.ge       # Property that returns the negated predicate Q.ge (>=)

    def eval(self, args, assumptions=True):
        if assumptions == True:
            # default assumptions for is_lt is None
            assumptions = None  # Set assumptions to None if it's True
        return is_lt(*args, assumptions)


class LessThanPredicate(BinaryRelation):
    """
    Binary predicate for $<=$.

    The purpose of this class is to provide the instance which represents
    the "<=" predicate to enable logical inference.

    This class is intended to remain internal to the assumptions module, and users should
    use :obj:`~.Le()` instead to construct the equality expression.

    Evaluating this predicate to ``True`` or ``False`` is done by
    :func:`~.core.relational.is_le()`

    Examples
    ========

    >>> from sympy import ask, Q
    >>> Q.le(0, 0)
    Q.le(0, 0)
    >>> ask(_)
    True

    See Also
    ========

    sympy.core.relational.Le

    """
    is_reflexive = True   # Indicates if the relation is reflexive (True for <=)
    is_symmetric = False  # Indicates if the relation is symmetric (False for <=)

    name = 'le'           # String representation of the predicate
    handler = None        # Placeholder for a handler function

    @property
    def reversed(self):
        return Q.ge       # Property that returns the reversed predicate Q.ge (>=)

    @property
    def negated(self):
        return Q.gt       # Property that returns the negated predicate Q.gt (>)
    # 定义一个方法 eval，接收参数 self, args, assumptions=True
    def eval(self, args, assumptions=True):
        # 如果 assumptions 等于 True，则将 assumptions 设置为 None
        if assumptions == True:
            assumptions = None
        # 调用 is_le 函数，并传递 args 和 assumptions 参数
        return is_le(*args, assumptions)
```