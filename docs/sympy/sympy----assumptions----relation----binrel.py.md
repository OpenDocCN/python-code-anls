# `D:\src\scipysrc\sympy\sympy\assumptions\relation\binrel.py`

```
"""
General binary relations.
"""
# 从 typing 模块导入 Optional 类型
from typing import Optional

# 导入 sympy 库中的相关模块和类
from sympy.core.singleton import S
from sympy.assumptions import AppliedPredicate, ask, Predicate, Q  # type: ignore
from sympy.core.kind import BooleanKind
from sympy.core.relational import Eq, Ne, Gt, Lt, Ge, Le
from sympy.logic.boolalg import conjuncts, Not

# 声明 __all__ 变量，指定公开的类和函数
__all__ = ["BinaryRelation", "AppliedBinaryRelation"]

# 定义 BinaryRelation 类，继承自 Predicate 类
class BinaryRelation(Predicate):
    """
    Base class for all binary relational predicates.

    Explanation
    ===========

    Binary relation takes two arguments and returns ``AppliedBinaryRelation``
    instance. To evaluate it to boolean value, use :obj:`~.ask()` or
    :obj:`~.refine()` function.

    You can add support for new types by registering the handler to dispatcher.
    See :obj:`~.Predicate()` for more information about predicate dispatching.

    Examples
    ========

    Applying and evaluating to boolean value:

    >>> from sympy import Q, ask, sin, cos
    >>> from sympy.abc import x
    >>> Q.eq(sin(x)**2+cos(x)**2, 1)
    Q.eq(sin(x)**2 + cos(x)**2, 1)
    >>> ask(_)
    True

    You can define a new binary relation by subclassing and dispatching.
    Here, we define a relation $R$ such that $x R y$ returns true if
    $x = y + 1$.

    >>> from sympy import ask, Number, Q
    >>> from sympy.assumptions import BinaryRelation
    >>> class MyRel(BinaryRelation):
    ...     name = "R"
    ...     is_reflexive = False
    >>> Q.R = MyRel()
    >>> @Q.R.register(Number, Number)
    ... def _(n1, n2, assumptions):
    ...     return ask(Q.zero(n1 - n2 - 1), assumptions)
    >>> Q.R(2, 1)
    Q.R(2, 1)

    Now, we can use ``ask()`` to evaluate it to boolean value.

    >>> ask(Q.R(2, 1))
    True
    >>> ask(Q.R(1, 2))
    False

    ``Q.R`` returns ``False`` with minimum cost if two arguments have same
    structure because it is antireflexive relation [1] by
    ``is_reflexive = False``.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Reflexive_relation
    """

    # 可选属性：是否是自反关系
    is_reflexive: Optional[bool] = None
    # 可选属性：是否是对称关系
    is_symmetric: Optional[bool] = None

    # 实例调用方法，接受两个参数并返回 AppliedBinaryRelation 实例
    def __call__(self, *args):
        if not len(args) == 2:
            raise ValueError("Binary relation takes two arguments, but got %s." % len(args))
        return AppliedBinaryRelation(self, *args)

    # 属性方法：返回反转后的二元关系，若为对称关系则返回自身，否则返回 None
    @property
    def reversed(self):
        if self.is_symmetric:
            return self
        return None

    # 属性方法：返回否定后的二元关系，始终返回 None
    @property
    def negated(self):
        return None
    # 比较函数，检查两个参数是否反射性相等
    def _compare_reflexive(self, lhs, rhs):
        # 对结构相同的参数进行快速退出
        # 这里不使用 != 检查，因为它无法捕捉到结构不同但等效的参数。

        # 若其中任一参数为 NaN，则不具备反射性
        if lhs is S.NaN or rhs is S.NaN:
            return None

        # 获取反射性属性
        reflexive = self.is_reflexive
        if reflexive is None:
            pass
        elif reflexive and (lhs == rhs):
            return True
        elif not reflexive and (lhs == rhs):
            return False
        return None

    # 对关系进行求值
    def eval(self, args, assumptions=True):
        # 对结构相同的参数进行快速退出
        ret = self._compare_reflexive(*args)
        if ret is not None:
            return ret

        # 不在这里对参数进行简化（由 AppliedBinaryRelation._eval_ask 完成）
        # 通过多分派进行评估
        lhs, rhs = args
        ret = self.handler(lhs, rhs, assumptions=assumptions)
        if ret is not None:
            return ret

        # 如果关系是反射性的，则检查反向顺序
        if self.is_reflexive:
            types = (type(lhs), type(rhs))
            # 如果处理程序的分派不相同，则检查反向分派
            if self.handler.dispatch(*types) is not self.handler.dispatch(*reversed(types)):
                ret = self.handler(rhs, lhs, assumptions=assumptions)

        return ret
class AppliedBinaryRelation(AppliedPredicate):
    """
    The class of expressions resulting from applying ``BinaryRelation``
    to the arguments.

    """

    @property
    def lhs(self):
        """The left-hand side of the relation."""
        return self.arguments[0]  # 返回关系表达式的左侧参数

    @property
    def rhs(self):
        """The right-hand side of the relation."""
        return self.arguments[1]  # 返回关系表达式的右侧参数

    @property
    def reversed(self):
        """
        Try to return the relationship with sides reversed.
        """
        revfunc = self.function.reversed  # 获取反转函数
        if revfunc is None:
            return self
        return revfunc(self.rhs, self.lhs)  # 如果反转函数存在，返回反转后的关系

    @property
    def reversedsign(self):
        """
        Try to return the relationship with signs reversed.
        """
        revfunc = self.function.reversed  # 获取反转函数
        if revfunc is None:
            return self
        if not any(side.kind is BooleanKind for side in self.arguments):
            return revfunc(-self.lhs, -self.rhs)  # 如果参数不是布尔类型，则返回带符号反转的关系
        return self

    @property
    def negated(self):
        neg_rel = self.function.negated  # 获取否定函数
        if neg_rel is None:
            return Not(self, evaluate=False)  # 如果否定函数不存在，返回对当前关系的否定
        return neg_rel(*self.arguments)  # 否则，对关系的参数应用否定函数

    def _eval_ask(self, assumptions):
        conj_assumps = set()
        binrelpreds = {Eq: Q.eq, Ne: Q.ne, Gt: Q.gt, Lt: Q.lt, Ge: Q.ge, Le: Q.le}
        for a in conjuncts(assumptions):
            if a.func in binrelpreds:
                conj_assumps.add(binrelpreds[type(a)](*a.args))  # 将二元关系谓词添加到假设集合中
            else:
                conj_assumps.add(a)  # 添加非二元关系谓词到假设集合中

        # After CNF in assumptions module is modified to take polyadic
        # predicate, this will be removed
        if any(rel in conj_assumps for rel in (self, self.reversed)):
            return True  # 如果当前关系或其反转存在于假设集合中，返回真

        neg_rels = (self.negated, self.reversed.negated, Not(self, evaluate=False),
                    Not(self.reversed, evaluate=False))
        if any(rel in conj_assumps for rel in neg_rels):
            return False  # 如果当前关系或其否定存在于假设集合中，返回假

        # evaluation using multipledispatching
        ret = self.function.eval(self.arguments, assumptions)  # 使用多分派评估函数对参数和假设进行评估
        if ret is not None:
            return ret  # 如果评估结果不为空，返回结果

        # simplify the args and try again
        args = tuple(a.simplify() for a in self.arguments)  # 对参数进行简化
        return self.function.eval(args, assumptions)  # 使用简化后的参数再次进行评估

    def __bool__(self):
        ret = ask(self)  # 调用ask函数获取布尔值
        if ret is None:
            raise TypeError("Cannot determine truth value of %s" % self)  # 如果无法确定真值，则引发类型错误
        return ret  # 返回ask函数的结果
```