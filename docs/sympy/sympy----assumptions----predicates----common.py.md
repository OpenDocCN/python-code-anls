# `D:\src\scipysrc\sympy\sympy\assumptions\predicates\common.py`

```
# 导入从 sympy.assumptions 模块中所需的类和函数
from sympy.assumptions import Predicate, AppliedPredicate, Q
# 导入从 sympy.core.relational 模块中的关系运算符类
from sympy.core.relational import Eq, Ne, Gt, Lt, Ge, Le
# 导入从 sympy.multipledispatch 模块的 Dispatcher 类
from sympy.multipledispatch import Dispatcher

# 定义 CommutativePredicate 类，继承自 Predicate 类
class CommutativePredicate(Predicate):
    """
    Commutative predicate.

    Explanation
    ===========

    ``ask(Q.commutative(x))`` is true iff ``x`` commutes with any other
    object with respect to multiplication operation.

    """
    # 用于表示谓词的名称
    name = 'commutative'
    # 创建一个分发器对象，用于处理 'commutative' 谓词
    handler = Dispatcher("CommutativeHandler", doc="Handler for key 'commutative'.")

# 定义 binrelpreds 字典，将关系运算符类映射到对应的 Q 谓词
binrelpreds = {Eq: Q.eq, Ne: Q.ne, Gt: Q.gt, Lt: Q.lt, Ge: Q.ge, Le: Q.le}

# 定义 IsTruePredicate 类，继承自 Predicate 类
class IsTruePredicate(Predicate):
    """
    Generic predicate.

    Explanation
    ===========

    ``ask(Q.is_true(x))`` is true iff ``x`` is true. This only makes
    sense if ``x`` is a boolean object.

    Examples
    ========

    >>> from sympy import ask, Q
    >>> from sympy.abc import x, y
    >>> ask(Q.is_true(True))
    True

    Wrapping another applied predicate just returns the applied predicate.

    >>> Q.is_true(Q.even(x))
    Q.even(x)

    Wrapping binary relation classes in SymPy core returns applied binary
    relational predicates.

    >>> from sympy import Eq, Gt
    >>> Q.is_true(Eq(x, y))
    Q.eq(x, y)
    >>> Q.is_true(Gt(x, y))
    Q.gt(x, y)

    Notes
    =====

    This class is designed to wrap the boolean objects so that they can
    behave as if they are applied predicates. Consequently, wrapping another
    applied predicate is unnecessary and thus it just returns the argument.
    Also, binary relation classes in SymPy core have binary predicates to
    represent themselves and thus wrapping them with ``Q.is_true`` converts them
    to these applied predicates.

    """
    # 用于表示谓词的名称
    name = 'is_true'
    # 创建一个分发器对象，用于处理 'is_true' 谓词
    handler = Dispatcher(
        "IsTrueHandler",
        doc="Wrapper allowing to query the truth value of a boolean expression."
    )

    # 定义谓词的调用方法，用于处理谓词的参数
    def __call__(self, arg):
        # 如果参数已经是一个应用谓词，则直接返回该参数
        if isinstance(arg, AppliedPredicate):
            return arg
        # 如果参数是一个关系运算符类的实例，则将其转换为对应的 Q 谓词应用
        if getattr(arg, "is_Relational", False):
            pred = binrelpreds[type(arg)]
            return pred(*arg.args)
        # 否则调用父类的调用方法处理参数
        return super().__call__(arg)
```