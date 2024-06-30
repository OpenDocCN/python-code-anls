# `D:\src\scipysrc\sympy\sympy\core\relational.py`

```
from __future__ import annotations

# 导入必要的模块和类
from .basic import Atom, Basic  # 导入Atom和Basic类
from .coreerrors import LazyExceptionMessage  # 导入LazyExceptionMessage异常类
from .sorting import ordered  # 导入ordered函数
from .evalf import EvalfMixin  # 导入EvalfMixin类
from .function import AppliedUndef  # 导入AppliedUndef类
from .numbers import int_valued  # 导入int_valued函数
from .singleton import S  # 导入S单例
from .sympify import _sympify, SympifyError  # 导入_sympify函数和SympifyError异常类
from .parameters import global_parameters  # 导入global_parameters全局参数
from .logic import fuzzy_bool, fuzzy_xor, fuzzy_and, fuzzy_not  # 导入布尔逻辑相关函数
from sympy.logic.boolalg import Boolean, BooleanAtom  # 导入布尔逻辑相关类
from sympy.utilities.iterables import sift  # 导入sift函数
from sympy.utilities.misc import filldedent  # 导入filldedent函数
from sympy.utilities.exceptions import sympy_deprecation_warning  # 导入sympy_deprecation_warning警告


__all__ = (
    'Rel', 'Eq', 'Ne', 'Lt', 'Le', 'Gt', 'Ge',  # 将以下类名加入到导出列表中
    'Relational', 'Equality', 'Unequality', 'StrictLessThan', 'LessThan',
    'StrictGreaterThan', 'GreaterThan',
)

# 导入Expr类和dispatch装饰器
from .expr import Expr
from sympy.multipledispatch import dispatch
# 导入Tuple类和Symbol类
from .containers import Tuple
from .symbol import Symbol


def _nontrivBool(side):
    # 判断side是否为Boolean类型且不是Atom类型的对象
    return isinstance(side, Boolean) and \
           not isinstance(side, Atom)


# 注意，参见问题编号4986。理想情况下，我们不希望同时继承Boolean和Expr类。
# 这里从..导入Expr类


def _canonical(cond):
    # 返回一个将所有关系操作符变为规范形式的条件
    reps = {r: r.canonical for r in cond.atoms(Relational)}
    return cond.xreplace(reps)
    # XXX: 这里曾经捕获到AttributeError异常，但是没有在任何测试中触发，所以我将其移除了...


def _canonical_coeff(rel):
    # 返回 -2*x + 1 < 0 的规范形式，例如 x > 1/2
    # XXX 这部分是否应该属于Relational.canonical?
    rel = rel.canonical
    if not rel.is_Relational or rel.rhs.is_Boolean:
        return rel  # 如果rel是形如Eq(x, True)的情况，则直接返回
    b, l = rel.lhs.as_coeff_Add(rational=True)
    m, lhs = l.as_coeff_Mul(rational=True)
    rhs = (rel.rhs - b)/m
    if m < 0:
        return rel.reversed.func(lhs, rhs)  # 如果m小于0，返回关系操作符的反向形式
    return rel.func(lhs, rhs)


class Relational(Boolean, EvalfMixin):
    """所有关系类型的基类。

    说明
    ===========

    Relational的子类通常应直接实例化，但可以使用有效的``rop``值实例化Relational，
    以分派到适当的子类。

    参数
    ==========

    rop : str or None
        指示要实例化的子类。Relational.ValidRelationOperator的键中可以找到有效值。

    示例
    ========

    >>> from sympy import Rel
    >>> from sympy.abc import x, y
    >>> Rel(y, x + x**2, '==')
    Eq(y, x**2 + x)

    可以在创建时定义关系的类型使用``rop``。
    可以使用其``rel_op``属性获取现有表达式的关系类型。
    下表显示了所有关系类型以及它们的``rop``和``rel_op``值：

    +---------------------+----------------------------+------------+
    |Relation             |``rop``                     |``rel_op``  |
    +=====================+============================+============+
    # 定义一个名称为 `Equality` 的关系运算符，可以使用 `==`、`eq` 或未指定来产生相等关系
    # 例如，将 `rop` 设置为 `==` 将产生一个 `Equality` 关系，即 `Eq()`。
    # 同样，设置 `rop` 为 `eq` 或不指定 `rop` 也会产生相同的结果。
    # 使用表中不同行的 `rop` 会产生不同的关系类型。
    # 例如，使用 `lt` 作为 `rop` 的第四个 `Rel()` 会产生一个 `StrictLessThan` 不等式：
    # ```
    # >>> from sympy import Rel
    # >>> from sympy.abc import x, y
    # >>> Rel(y, x + x**2, '==')
    # Eq(y, x**2 + x)
    # >>> Rel(y, x + x**2, 'eq')
    # Eq(y, x**2 + x)
    # >>> Rel(y, x + x**2)
    # Eq(y, x**2 + x)
    # >>> Rel(y, x + x**2, 'lt')
    # y < x**2 + x
    # ```
    # 要获取现有表达式的关系类型，请使用其 `rel_op` 属性。
    # 例如，上面的 `Equality` 关系的 `rel_op` 是 `==`，严格小于不等式的 `rel_op` 是 `<`：
    # ```
    # >>> from sympy import Rel
    # >>> from sympy.abc import x, y
    # >>> my_equality = Rel(y, x + x**2, '==')
    # >>> my_equality.rel_op
    # '=='
    # >>> my_inequality = Rel(y, x + x**2, 'lt')
    # >>> my_inequality.rel_op
    # '<'
    # ```
    
    """
    __slots__ = ()
    
    ValidRelationOperator: dict[str | None, type[Relational]] = {}
    
    is_Relational = True
    
    # ValidRelationOperator - Defined below, because the necessary classes
    #   have not yet been defined
    def __new__(cls, lhs, rhs, rop=None, **assumptions):
        # 如果被子类调用，则不做特殊处理，继续传递给 Basic 类处理。
        if cls is not Relational:
            return Basic.__new__(cls, lhs, rhs, **assumptions)

        # XXX: 为什么要这样做？应该有一个单独的函数从字符串创建特定的 Relational 子类。
        #
        # 如果直接使用运算符调用，则查找对应于该运算符的子类，并委托给它
        cls = cls.ValidRelationOperator.get(rop, None)
        if cls is None:
            raise ValueError("Invalid relational operator symbol: %r" % rop)

        if not issubclass(cls, (Eq, Ne)):
            # 验证在关系运算中除了 Eq/Ne 外不能使用布尔值；
            # 注意：Symbol 是 Boolean 的子类，但在这里被视为可接受的。
            if any(map(_nontrivBool, (lhs, rhs))):
                raise TypeError(filldedent('''
                    A Boolean argument can only be used in
                    Eq and Ne; all other relationals expect
                    real expressions.
                '''))

        return cls(lhs, rhs, **assumptions)

    @property
    def lhs(self):
        """关系的左侧."""
        return self._args[0]

    @property
    def rhs(self):
        """关系的右侧."""
        return self._args[1]

    @property
    def reversed(self):
        """返回反转了两侧关系的关系对象.

        Examples
        ========

        >>> from sympy import Eq
        >>> from sympy.abc import x
        >>> Eq(x, 1)
        Eq(x, 1)
        >>> _.reversed
        Eq(1, x)
        >>> x < 1
        x < 1
        >>> _.reversed
        1 > x
        """
        ops = {Eq: Eq, Gt: Lt, Ge: Le, Lt: Gt, Le: Ge, Ne: Ne}
        a, b = self.args
        return Relational.__new__(ops.get(self.func, self.func), b, a)

    @property
    def reversedsign(self):
        """返回反转了符号的关系对象.

        Examples
        ========

        >>> from sympy import Eq
        >>> from sympy.abc import x
        >>> Eq(x, 1)
        Eq(x, 1)
        >>> _.reversedsign
        Eq(-x, -1)
        >>> x < 1
        x < 1
        >>> _.reversedsign
        -x > -1
        """
        a, b = self.args
        if not (isinstance(a, BooleanAtom) or isinstance(b, BooleanAtom)):
            ops = {Eq: Eq, Gt: Lt, Ge: Le, Lt: Gt, Le: Ge, Ne: Ne}
            return Relational.__new__(ops.get(self.func, self.func), -a, -b)
        else:
            return self

    @property
    def negated(self):
        """Return the negated relationship.

        Examples
        ========

        >>> from sympy import Eq
        >>> from sympy.abc import x
        >>> Eq(x, 1)
        Eq(x, 1)
        >>> _.negated
        Ne(x, 1)
        >>> x < 1
        x < 1
        >>> _.negated
        x >= 1

        Notes
        =====

        This works more or less identical to ``~``/``Not``. The difference is
        that ``negated`` returns the relationship even if ``evaluate=False``.
        Hence, this is useful in code when checking for e.g. negated relations
        to existing ones as it will not be affected by the `evaluate` flag.

        """
        ops = {Eq: Ne, Ge: Lt, Gt: Le, Le: Gt, Lt: Ge, Ne: Eq}
        # 如果将来有新的关系子类，下面这行代码会继续工作，直到它被适当处理
        # return ops.get(self.func, lambda a, b, evaluate=False: ~(self.func(a,
        #      b, evaluate=evaluate)))(*self.args, evaluate=False)
        # 根据操作符映射表 ops，创建新的关系对象并返回
        return Relational.__new__(ops.get(self.func), *self.args)

    @property
    def weak(self):
        """return the non-strict version of the inequality or self

        EXAMPLES
        ========

        >>> from sympy.abc import x
        >>> (x < 1).weak
        x <= 1
        >>> _.weak
        x <= 1
        """
        # 返回不严格版本的不等式或者自身
        return self

    @property
    def strict(self):
        """return the strict version of the inequality or self

        EXAMPLES
        ========

        >>> from sympy.abc import x
        >>> (x <= 1).strict
        x < 1
        >>> _.strict
        x < 1
        """
        # 返回严格版本的不等式或者自身
        return self

    def _eval_evalf(self, prec):
        # 对于给定的精度，对关系对象的每个参数进行浮点数评估
        return self.func(*[s._evalf(prec) for s in self.args])
    def canonical(self):
        """
        Return a canonical form of the relational by putting a
        number on the rhs, canonically removing a sign or else
        ordering the args canonically. No other simplification is
        attempted.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> x < 2
        x < 2
        >>> _.reversed.canonical
        x < 2
        >>> (-y < x).canonical
        x > -y
        >>> (-y > x).canonical
        x < -y
        >>> (-y < -x).canonical
        x < y

        The canonicalization is recursively applied:

        >>> from sympy import Eq
        >>> Eq(x < y, y > x).canonical
        True
        """
        # 获取所有参数的规范形式，如果参数是关系表达式则递归地获取其规范形式
        args = tuple([i.canonical if isinstance(i, Relational) else i for i in self.args])
        # 如果规范化后的参数与原参数不同，则创建新的关系表达式
        if args != self.args:
            r = self.func(*args)
            # 如果结果不是关系表达式，则直接返回结果
            if not isinstance(r, Relational):
                return r
        else:
            r = self
        # 如果右侧是数值
        if r.rhs.is_number:
            # 如果左侧和右侧都是数值，并且左侧大于右侧，则反转关系运算符
            if r.rhs.is_Number and r.lhs.is_Number and r.lhs > r.rhs:
                r = r.reversed
        elif r.lhs.is_number:
            # 如果左侧是数值，则反转关系运算符
            r = r.reversed
        elif tuple(ordered(args)) != args:
            # 如果参数未按规范顺序排列，则反转关系运算符
            r = r.reversed

        # 获取左侧和右侧是否能提取负号的函数
        LHS_CEMS = getattr(r.lhs, 'could_extract_minus_sign', None)
        RHS_CEMS = getattr(r.rhs, 'could_extract_minus_sign', None)

        # 如果左侧或右侧是布尔原子，则直接返回结果
        if isinstance(r.lhs, BooleanAtom) or isinstance(r.rhs, BooleanAtom):
            return r

        # 检查左侧是否有负号
        if LHS_CEMS and LHS_CEMS():
            return r.reversedsign  # 返回反转符号后的关系表达式
        elif not r.rhs.is_number and RHS_CEMS and RHS_CEMS():
            # 如果右侧有负号但左侧没有，则比较反转符号后的表达式是否与原表达式相同
            expr1, _ = ordered([r.lhs, -r.rhs])
            if expr1 != r.lhs:
                return r.reversed.reversedsign  # 返回反转符号后的反转表达式

        return r  # 返回最终的规范形式的关系表达式
    def equals(self, other, failing_expression=False):
        """Return True if the sides of the relationship are mathematically
        identical and the type of relationship is the same.
        If failing_expression is True, return the expression whose truth value
        was unknown."""
        # 检查参数other是否是Relational类的实例
        if isinstance(other, Relational):
            # 检查other是否是self或者self.reversed的实例
            if other in (self, self.reversed):
                return True
            # 设置a和b为当前对象和参数other
            a, b = self, other
            # 检查当前对象或者参数是否为等式（Eq）或不等式（Ne）
            if a.func in (Eq, Ne) or b.func in (Eq, Ne):
                # 如果当前对象和参数的函数类型不同，则返回False
                if a.func != b.func:
                    return False
                # 比较当前对象和参数的参数列表的每一项是否相等
                left, right = [i.equals(j,
                                        failing_expression=failing_expression)
                               for i, j in zip(a.args, b.args)]
                # 如果左侧比较结果为True，则返回右侧结果
                if left is True:
                    return right
                # 如果右侧比较结果为True，则返回左侧结果
                if right is True:
                    return left
                # 否则，尝试反向比较当前对象的参数列表和参数对象的反向参数列表
                lr, rl = [i.equals(j, failing_expression=failing_expression)
                          for i, j in zip(a.args, b.reversed.args)]
                # 如果左右反向比较结果为True，则返回右反向比较结果
                if lr is True:
                    return rl
                # 如果右左反向比较结果为True，则返回左反向比较结果
                if rl is True:
                    return lr
                # 如果以上比较结果都为False，则返回未知的表达式
                e = (left, right, lr, rl)
                if all(i is False for i in e):
                    return False
                # 返回第一个不是True也不是False的结果
                for i in e:
                    if i not in (True, False):
                        return i
            else:
                # 如果当前对象和参数的函数类型不同，则交换参数b为其反向对象
                if b.func != a.func:
                    b = b.reversed
                # 再次检查当前对象和参数的函数类型是否一致
                if a.func != b.func:
                    return False
                # 递归调用equals方法比较当前对象和参数的左侧和右侧
                left = a.lhs.equals(b.lhs,
                                    failing_expression=failing_expression)
                # 如果左侧比较结果为False，则返回False
                if left is False:
                    return False
                # 递归调用equals方法比较当前对象和参数的右侧
                right = a.rhs.equals(b.rhs,
                                     failing_expression=failing_expression)
                # 如果右侧比较结果为False，则返回False
                if right is False:
                    return False
                # 如果左侧比较结果为True，则返回右侧比较结果
                if left is True:
                    return right
                # 否则返回左侧比较结果
                return left

    def _eval_trigsimp(self, **opts):
        """Simplify each side of the relational expression using trigonometric simplification."""
        # 导入trigsimp函数并对当前对象的左侧和右侧进行三角函数简化
        from sympy.simplify.trigsimp import trigsimp
        return self.func(trigsimp(self.lhs, **opts), trigsimp(self.rhs, **opts))

    def expand(self, **kwargs):
        """Expand each argument of the relational expression."""
        # 对当前对象的所有参数进行展开操作
        args = (arg.expand(**kwargs) for arg in self.args)
        return self.func(*args)

    def __bool__(self):
        """Raise a TypeError indicating that the truth value of Relational expressions cannot be determined."""
        # 抛出TypeError异常，指示无法确定关系表达式的真值
        raise TypeError(
            LazyExceptionMessage(
                lambda: f"cannot determine truth value of Relational: {self}"
            )
        )
    # 计算作为集合的表达式
    def _eval_as_set(self):
        # 引入解不等式的单变量求解器
        from sympy.solvers.inequalities import solve_univariate_inequality
        # 引入条件集合
        from sympy.sets.conditionset import ConditionSet
        # 获取表达式中的自由符号集合
        syms = self.free_symbols
        # 断言只有一个自由符号
        assert len(syms) == 1
        # 取出唯一的符号
        x = syms.pop()
        try:
            # 尝试求解单变量不等式
            xset = solve_univariate_inequality(self, x, relational=False)
        except NotImplementedError:
            # 如果求解不了，则返回一个条件集合，表示在实数域上的解
            xset = ConditionSet(x, self, S.Reals)
        # 返回计算得到的集合
        return xset

    @property
    def binary_symbols(self):
        # 覆盖需要覆盖的地方
        return set()
Rel = Relational



class Equality(Relational):

定义了一个名为 `Equality` 的类，它继承自 `Relational` 类。


"""
An equal relation between two objects.

Explanation
===========

Represents that two objects are equal.  If they can be easily shown
to be definitively equal (or unequal), this will reduce to True (or
False).  Otherwise, the relation is maintained as an unevaluated
Equality object.  Use the ``simplify`` function on this object for
more nontrivial evaluation of the equality relation.

As usual, the keyword argument ``evaluate=False`` can be used to
prevent any evaluation.

Examples
========

>>> from sympy import Eq, simplify, exp, cos
>>> from sympy.abc import x, y
>>> Eq(y, x + x**2)
Eq(y, x**2 + x)
>>> Eq(2, 5)
False
>>> Eq(2, 5, evaluate=False)
Eq(2, 5)
>>> _.doit()
False
>>> Eq(exp(x), exp(x).rewrite(cos))
Eq(exp(x), sinh(x) + cosh(x))
>>> simplify(_)
True

See Also
========

sympy.logic.boolalg.Equivalent : for representing equality between two
    boolean expressions

Notes
=====

Python treats 1 and True (and 0 and False) as being equal; SymPy
does not. And integer will always compare as unequal to a Boolean:

>>> Eq(True, 1), True == 1
(False, True)

This class is not the same as the == operator.  The == operator tests
for exact structural equality between two expressions; this class
compares expressions mathematically.

If either object defines an ``_eval_Eq`` method, it can be used in place of
the default algorithm.  If ``lhs._eval_Eq(rhs)`` or ``rhs._eval_Eq(lhs)``
returns anything other than None, that return value will be substituted for
the Equality.  If None is returned by ``_eval_Eq``, an Equality object will
be created as usual.

Since this object is already an expression, it does not respond to
the method ``as_expr`` if one tries to create `x - y` from ``Eq(x, y)``.
If ``eq = Eq(x, y)`` then write `eq.lhs - eq.rhs` to get ``x - y``.

.. deprecated:: 1.5

   ``Eq(expr)`` with a single argument is a shorthand for ``Eq(expr, 0)``,
   but this behavior is deprecated and will be removed in a future version
   of SymPy.

"""

这是 `Equality` 类的文档字符串，提供了关于此类的详细说明、示例、使用说明和注意事项。


rel_op = '=='

设置了一个类变量 `rel_op`，其值为字符串 `'=='`。


__slots__ = ()

定义了空元组 `__slots__`，这用于限制该类的实例只能拥有指定的属性，但在此处未定义具体的实例属性。


is_Equality = True

定义了一个类变量 `is_Equality`，其值为 `True`。


def __new__(cls, lhs, rhs, **options):

定义了一个特殊方法 `__new__`，用于创建新的 `Equality` 对象，接受左右两个操作数 `lhs` 和 `rhs`，以及其他选项。


evaluate = options.pop('evaluate', global_parameters.evaluate)

从 `options` 字典中获取 `evaluate` 键的值，如果不存在则使用 `global_parameters.evaluate` 的值，默认为 `False`。


lhs = _sympify(lhs)
rhs = _sympify(rhs)

将 `lhs` 和 `rhs` 使用 `_sympify` 函数转换为 SymPy 表达式对象。


if evaluate:
    val = is_eq(lhs, rhs)
    if val is None:
        return cls(lhs, rhs, evaluate=False)
    else:
        return _sympify(val)

如果 `evaluate` 为 `True`，则调用 `is_eq` 函数判断 `lhs` 和 `rhs` 是否相等，如果无法确定则返回一个新的 `Equality` 对象；否则直接返回计算后的结果。


return Relational.__new__(cls, lhs, rhs)

如果不进行求值，则调用父类 `Relational` 的 `__new__` 方法创建一个新的 `Equality` 对象。


@classmethod
def _eval_relation(cls, lhs, rhs):

定义了一个类方法 `_eval_relation`，用于计算两个对象 `lhs` 和 `rhs` 是否相等。


return _sympify(lhs == rhs)

返回 `lhs == rhs` 的结果转换为 SymPy 表达式。
    def _eval_rewrite_as_Add(self, L, R, evaluate=True, **kwargs):
        """
        将 Eq(L, R) 重写为 L - R。通过设置 `evaluate=True` 控制结果的评估，返回 L - R；
        如果设置为 `evaluate=None`，则 L 和 R 中的项不会被取消，但会按规范顺序列出；
        否则返回非规范参数。如果一侧为 0，则返回非零一侧。

        .. deprecated:: 1.13

           方法 ``Eq.rewrite(Add)`` 已弃用。
           详见 :ref:`eq-rewrite-Add` 获取详情。

        Examples
        ========

        >>> from sympy import Eq, Add
        >>> from sympy.abc import b, x
        >>> eq = Eq(x + b, x - b)
        >>> eq.rewrite(Add)  #doctest: +SKIP
        2*b
        >>> eq.rewrite(Add, evaluate=None).args  #doctest: +SKIP
        (b, b, x, -x)
        >>> eq.rewrite(Add, evaluate=False).args  #doctest: +SKIP
        (b, x, b, -x)
        """
        sympy_deprecation_warning("""
        Eq.rewrite(Add) 已弃用。

        对于 ``eq = Eq(a, b)``，使用 ``eq.lhs - eq.rhs`` 可得到 ``a - b``。
        """,
            deprecated_since_version="1.13",
            active_deprecations_target="eq-rewrite-Add",
            stacklevel=5,
        )
        from .add import _unevaluated_Add, Add
        if L == 0:
            # 如果 L 为 0，则返回 R
            return R
        if R == 0:
            # 如果 R 为 0，则返回 L
            return L
        if evaluate:
            # 允许参数的抵消
            return L - R
        args = Add.make_args(L) + Add.make_args(-R)
        if evaluate is None:
            # 不进行抵消，但保持规范顺序
            return _unevaluated_Add(*args)
        # 不进行抵消，不保持规范顺序
        return Add._from_args(args)

    @property
    def binary_symbols(self):
        if S.true in self.args or S.false in self.args:
            if self.lhs.is_Symbol:
                # 如果左侧是符号，则返回包含左侧的集合
                return {self.lhs}
            elif self.rhs.is_Symbol:
                # 如果右侧是符号，则返回包含右侧的集合
                return {self.rhs}
        # 默认返回空集合
        return set()

    def _eval_simplify(self, **kwargs):
        # 标准简化
        e = super()._eval_simplify(**kwargs)
        if not isinstance(e, Equality):
            # 如果 e 不是等式对象，则直接返回 e
            return e
        from .expr import Expr
        if not isinstance(e.lhs, Expr) or not isinstance(e.rhs, Expr):
            # 如果 e 的左侧或右侧不是表达式对象，则直接返回 e
            return e
        free = self.free_symbols
        if len(free) == 1:
            try:
                from .add import Add
                from sympy.solvers.solveset import linear_coeffs
                x = free.pop()
                m, b = linear_coeffs(
                    Add(e.lhs, -e.rhs, evaluate=False), x)
                if m.is_zero is False:
                    enew = e.func(x, -b / m)
                else:
                    enew = e.func(m * x, -b)
                measure = kwargs['measure']
                if measure(enew) <= kwargs['ratio'] * measure(e):
                    e = enew
            except ValueError:
                pass
        # 返回规范化的 e
        return e.canonical
    # 使用 sympy 库中的 integrate 函数来进行积分计算
    def integrate(self, *args, **kwargs):
        """See the integrate function in sympy.integrals"""
        # 导入 sympy.integrals.integrals 模块中的 integrate 函数
        from sympy.integrals.integrals import integrate
        # 调用 integrate 函数，对当前对象（self）进行积分计算，并返回结果
        return integrate(self, *args, **kwargs)

    # 将等式左右两边表达式作为多项式返回
    def as_poly(self, *gens, **kwargs):
        '''Returns lhs-rhs as a Poly

        Examples
        ========

        >>> from sympy import Eq
        >>> from sympy.abc import x
        >>> Eq(x**2, 1).as_poly(x)
        Poly(x**2 - 1, x, domain='ZZ')
        '''
        # 将等式左边减去右边的表达式，然后作为多项式返回
        return (self.lhs - self.rhs).as_poly(*gens, **kwargs)
Eq = Equality
# 将 Equality 类赋值给 Eq，用于表示相等关系

class Unequality(Relational):
    """An unequal relation between two objects.

    Explanation
    ===========

    Represents that two objects are not equal.  If they can be shown to be
    definitively equal, this will reduce to False; if definitively unequal,
    this will reduce to True.  Otherwise, the relation is maintained as an
    Unequality object.

    Examples
    ========

    >>> from sympy import Ne
    >>> from sympy.abc import x, y
    >>> Ne(y, x+x**2)
    Ne(y, x**2 + x)

    See Also
    ========
    Equality

    Notes
    =====
    This class is not the same as the != operator.  The != operator tests
    for exact structural equality between two expressions; this class
    compares expressions mathematically.

    This class is effectively the inverse of Equality.  As such, it uses the
    same algorithms, including any available `_eval_Eq` methods.

    """
    rel_op = '!='
    # 设置不等关系的操作符为 '!='

    __slots__ = ()

    def __new__(cls, lhs, rhs, **options):
        lhs = _sympify(lhs)
        rhs = _sympify(rhs)
        evaluate = options.pop('evaluate', global_parameters.evaluate)
        # 将左右两侧表达式转换成符号表达式
        if evaluate:
            val = is_neq(lhs, rhs)
            if val is None:
                return cls(lhs, rhs, evaluate=False)
            else:
                return _sympify(val)
        # 如果开启了求值选项，根据是否不相等来返回 Unequality 对象或相应的布尔值

        return Relational.__new__(cls, lhs, rhs, **options)
        # 调用父类 Relational 的构造方法生成 Unequality 对象

    @classmethod
    def _eval_relation(cls, lhs, rhs):
        return _sympify(lhs != rhs)
        # 使用 _sympify 方法判断左右两侧表达式是否不相等，并返回结果

    @property
    def binary_symbols(self):
        if S.true in self.args or S.false in self.args:
            if self.lhs.is_Symbol:
                return {self.lhs}
            elif self.rhs.is_Symbol:
                return {self.rhs}
        return set()
    # 返回表达式中包含的符号的集合

    def _eval_simplify(self, **kwargs):
        # simplify as an equality
        eq = Equality(*self.args)._eval_simplify(**kwargs)
        # 将 Unequality 对象转换为对应的 Equality 对象，并简化
        if isinstance(eq, Equality):
            # send back Ne with the new args
            return self.func(*eq.args)
            # 返回新的 Unequality 对象，参数为简化后的 Equality 对象的参数
        return eq.negated  # result of Ne is the negated Eq
        # 返回 Ne 对象的否定结果

Ne = Unequality
# 将 Unequality 类赋值给 Ne，Ne 表示不等关系

class _Inequality(Relational):
    """Internal base class for all *Than types.

    Each subclass must implement _eval_relation to provide the method for
    comparing two real numbers.

    """
    __slots__ = ()
    # 为所有 *Than 类型的内部基类
    # 定义一个新的类方法 `__new__`，用于创建对象实例，接收 `lhs` 和 `rhs` 作为参数，还有可选的参数集合 `options`
    def __new__(cls, lhs, rhs, **options):
        
        # 尝试将 `lhs` 和 `rhs` 转换为符号表达式（sympify），如果无法转换则返回 NotImplemented
        try:
            lhs = _sympify(lhs)
            rhs = _sympify(rhs)
        except SympifyError:
            return NotImplemented
        
        # 从 `options` 中获取 `evaluate` 参数，如果不存在则使用全局参数中的默认值
        evaluate = options.pop('evaluate', global_parameters.evaluate)
        
        # 如果 `evaluate` 为真，进行以下操作：
        if evaluate:
            # 检查 `lhs` 和 `rhs` 是否为非实数，如果是则抛出类型错误异常
            for me in (lhs, rhs):
                if me.is_extended_real is False:
                    raise TypeError("Invalid comparison of non-real %s" % me)
                # 如果 `me` 是 NaN，则抛出类型错误异常
                if me is S.NaN:
                    raise TypeError("Invalid NaN comparison")
            
            # 首先调用 `lhs` 的相应不等式方法（例如 `lhs.__lt__`），该方法尝试将结果缩减为布尔值或引发异常。
            # 它可能会继续调用超类，直到达到 `Expr` 类（例如 `Expr.__lt__`）。
            # 在某些情况下，`Expr` 类将再次调用当前方法（如果它和其子类都无法将结果缩减为布尔值或引发异常）。
            # 在这种情况下，它必须以 `evaluate=False` 调用我们，以防止无限递归。
            return cls._eval_relation(lhs, rhs, **options)
        
        # 如果 `evaluate` 不为真，创建一个未评估的不等式表达式 `Expr` 实例
        return Relational.__new__(cls, lhs, rhs, **options)

    # 类方法 `_eval_relation`，用于评估两个表达式 `lhs` 和 `rhs` 之间的模糊关系
    @classmethod
    def _eval_relation(cls, lhs, rhs, **options):
        # 调用 `_eval_fuzzy_relation` 方法获取模糊关系的值
        val = cls._eval_fuzzy_relation(lhs, rhs)
        
        # 如果值为 None，返回未评估的不等式表达式 `Expr` 实例
        if val is None:
            return cls(lhs, rhs, evaluate=False)
        else:
            # 否则，将值转换为符号表达式并返回
            return _sympify(val)
class _Greater(_Inequality):
    """Not intended for general use

    _Greater is only used so that GreaterThan and StrictGreaterThan may
    subclass it for the .gts and .lts properties.

    """
    __slots__ = ()

    @property
    def gts(self):
        # 返回不等式左侧的对象
        return self._args[0]

    @property
    def lts(self):
        # 返回不等式右侧的对象
        return self._args[1]


class _Less(_Inequality):
    """Not intended for general use.

    _Less is only used so that LessThan and StrictLessThan may subclass it for
    the .gts and .lts properties.

    """
    __slots__ = ()

    @property
    def gts(self):
        # 返回不等式右侧的对象
        return self._args[1]

    @property
    def lts(self):
        # 返回不等式左侧的对象
        return self._args[0]


class GreaterThan(_Greater):
    r"""Class representations of inequalities.

    Explanation
    ===========

    The ``*Than`` classes represent inequal relationships, where the left-hand
    side is generally bigger or smaller than the right-hand side.  For example,
    the GreaterThan class represents an inequal relationship where the
    left-hand side is at least as big as the right side, if not bigger.  In
    mathematical notation:

    lhs $\ge$ rhs

    In total, there are four ``*Than`` classes, to represent the four
    inequalities:

    +-----------------+--------+
    |Class Name       | Symbol |
    +=================+========+
    |GreaterThan      | ``>=`` |
    +-----------------+--------+
    |LessThan         | ``<=`` |
    +-----------------+--------+
    |StrictGreaterThan| ``>``  |
    +-----------------+--------+
    |StrictLessThan   | ``<``  |
    +-----------------+--------+

    All classes take two arguments, lhs and rhs.

    +----------------------------+-----------------+
    |Signature Example           | Math Equivalent |
    +============================+=================+
    |GreaterThan(lhs, rhs)       |   lhs $\ge$ rhs |
    +----------------------------+-----------------+
    |LessThan(lhs, rhs)          |   lhs $\le$ rhs |
    +----------------------------+-----------------+
    |StrictGreaterThan(lhs, rhs) |   lhs $>$ rhs   |
    +----------------------------+-----------------+
    |StrictLessThan(lhs, rhs)    |   lhs $<$ rhs   |
    +----------------------------+-----------------+

    In addition to the normal .lhs and .rhs of Relations, ``*Than`` inequality
    objects also have the .lts and .gts properties, which represent the "less
    than side" and "greater than side" of the operator.  Use of .lts and .gts
    in an algorithm rather than .lhs and .rhs as an assumption of inequality
    direction will make more explicit the intent of a certain section of code,
    and will make it similarly more robust to client code changes:

    >>> from sympy import GreaterThan, StrictGreaterThan
    >>> from sympy import LessThan, StrictLessThan
    >>> from sympy import And, Ge, Gt, Le, Lt, Rel, S
    >>> from sympy.abc import x, y, z
    >>> from sympy.core.relational import Relational

    >>> e = GreaterThan(x, 1)
    >>> e
    x >= 1
    # 使用字符串格式化输出，将 e.gts、e.lts、e.lts、e.gts 四个变量插入字符串中
    >>> '%s >= %s is the same as %s <= %s' % (e.gts, e.lts, e.lts, e.gts)
    'x >= 1 is the same as 1 <= x'
    
    Examples
    ========
    
    One generally does not instantiate these classes directly, but uses various
    convenience methods:
    
    # 使用循环遍历列表中的 Ge, Gt, Le, Lt 四个类，通过其 convenience 方法进行实例化和调用
    >>> for f in [Ge, Gt, Le, Lt]:  # convenience wrappers
    ...     print(f(x, 2))
    x >= 2
    x > 2
    x <= 2
    x < 2
    
    Another option is to use the Python inequality operators (``>=``, ``>``,
    ``<=``, ``<``) directly.  Their main advantage over the ``Ge``, ``Gt``,
    ``Le``, and ``Lt`` counterparts, is that one can write a more
    "mathematical looking" statement rather than littering the math with
    oddball function calls.  However there are certain (minor) caveats of
    which to be aware (search for 'gotcha', below).
    
    # 直接使用 Python 不等式运算符进行比较，可以更直观地表达数学上的关系
    >>> x >= 2
    x >= 2
    >>> _ == Ge(x, 2)
    True
    
    However, it is also perfectly valid to instantiate a ``*Than`` class less
    succinctly and less conveniently:
    
    # 可以选择以更冗长和不那么方便的方式实例化 ``*Than`` 类
    >>> Rel(x, 1, ">")
    x > 1
    >>> Relational(x, 1, ">")
    x > 1
    
    >>> StrictGreaterThan(x, 1)
    x > 1
    >>> GreaterThan(x, 1)
    x >= 1
    >>> LessThan(x, 1)
    x <= 1
    >>> StrictLessThan(x, 1)
    x < 1
    
    Notes
    =====
    
    There are a couple of "gotchas" to be aware of when using Python's
    operators.
    
    The first is that what your write is not always what you get:
    
    # Python 解析语句的顺序可能导致输出与输入顺序不同
    >>> 1 < x
    x > 1
    
    Due to the order that Python parses a statement, it may
    not immediately find two objects comparable.  When ``1 < x``
    is evaluated, Python recognizes that the number 1 is a native
    number and that x is *not*.  Because a native Python number does
    not know how to compare itself with a SymPy object
    Python will try the reflective operation, ``x > 1`` and that is the
    form that gets evaluated, hence returned.
    
    If the order of the statement is important (for visual output to
    the console, perhaps), one can work around this annoyance in a
    couple ways:
    
    (1) "sympify" the literal before comparison
    
    # 在比较之前，使用 sympify 将字面量转换为 SymPy 对象
    >>> S(1) < x
    1 < x
    
    (2) use one of the wrappers or less succinct methods described
    above
    
    # 使用上述提到的包装器或不太简洁的方法
    >>> Lt(1, x)
    1 < x
    >>> Relational(1, x, "<")
    1 < x
    
    The second gotcha involves writing equality tests between relationals
    # 当测试的一侧或两侧涉及文字关系运算符时的注意事项：
    
        >>> e = x < 1; e
        x < 1
        >>> e == e  # 两侧都不是文字关系运算符
        True
        >>> e == x < 1  # 预期结果为 True
        False
        >>> e != x < 1  # 预期结果为 False
        x < 1
        >>> x < 1 != x < 1  # 预期结果为 False 或与之前相同
        Traceback (most recent call last):
        ...
        TypeError: cannot determine truth value of Relational
    
        在这种情况下的解决方案是将文字关系运算符用括号括起来：
    
        >>> e == (x < 1)
        True
        >>> e != (x < 1)
        False
        >>> (x < 1) != (x < 1)
        False
    
    第三个问题涉及链式不等式，不涉及 ``==`` 或 ``!=``。偶尔会尝试编写如下代码：
    
        >>> e = x < y < z
        Traceback (most recent call last):
        ...
        TypeError: symbolic boolean expression has no truth value.
    
        由于 Python 的实现细节或决策 [1]_，SymPy 无法使用这种语法创建链式不等式，因此必须使用 And：
    
        >>> e = And(x < y, y < z)
        >>> type( e )
        And
        >>> e
        (x < y) & (y < z)
    
        虽然可以使用 '&' 操作符来实现相同效果，但无法使用 'and' 操作符：
    
        >>> (x < y) & (y < z)
        (x < y) & (y < z)
        >>> (x < y) and (y < z)
        Traceback (most recent call last):
        ...
        TypeError: cannot determine truth value of Relational
    """
    [1] This implementation detail is that Python provides no reliable
       method to determine that a chained inequality is being built.
       Chained comparison operators are evaluated pairwise, using "and"
       logic (see
       https://docs.python.org/3/reference/expressions.html#not-in). This
       is done in an efficient way, so that each object being compared
       is only evaluated once and the comparison can short-circuit. For
       example, ``1 > 2 > 3`` is evaluated by Python as ``(1 > 2) and (2
       > 3)``. The ``and`` operator coerces each side into a bool,
       returning the object itself when it short-circuits. The bool of
       the --Than operators will raise TypeError on purpose, because
       SymPy cannot determine the mathematical ordering of symbolic
       expressions. Thus, if we were to compute ``x > y > z``, with
       ``x``, ``y``, and ``z`` being Symbols, Python converts the
       statement (roughly) into these steps:
    
        (1) x > y > z
        (2) (x > y) and (y > z)
        (3) (GreaterThanObject) and (y > z)
        (4) (GreaterThanObject.__bool__()) and (y > z)
        (5) TypeError
    
       Because of the ``and`` added at step 2, the statement gets turned into a
       weak ternary statement, and the first object's ``__bool__`` method will
       raise TypeError.  Thus, creating a chained inequality is not possible.
    
           In Python, there is no way to override the ``and`` operator, or to
           control how it short circuits, so it is impossible to make something
           like ``x > y > z`` work.  There was a PEP to change this,
           :pep:`335`, but it was officially closed in March, 2012.
    
    """
    # 空的 __slots__ 定义，用于限制类的实例属性
    __slots__ = ()
    
    # 设置关系操作符为 '>='
    rel_op = '>='
    
    # 定义类方法 _eval_fuzzy_relation，用于模糊关系的评估
    @classmethod
    def _eval_fuzzy_relation(cls, lhs, rhs):
        return is_ge(lhs, rhs)
    
    # 定义 strict 属性，返回用参数构造的 Gt 对象
    @property
    def strict(self):
        return Gt(*self.args)
# 引入 GreaterThan 类并重命名为 Ge
Ge = GreaterThan

# 定义 LessThan 类，继承自 _Less 类
class LessThan(_Less):
    # 继承 GreaterThan 类的文档字符串
    __doc__ = GreaterThan.__doc__
    # 空的 __slots__ 列表，用于声明实例属性
    __slots__ = ()

    # 关系运算符设为 '<='
    rel_op = '<='

    # 类方法，模糊关系的评估，判断 lhs 是否小于等于 rhs
    @classmethod
    def _eval_fuzzy_relation(cls, lhs, rhs):
        return is_le(lhs, rhs)

    # 属性方法，返回严格小于关系的实例
    @property
    def strict(self):
        return Lt(*self.args)

# 将 LessThan 类重命名为 Le
Le = LessThan

# 定义 StrictGreaterThan 类，继承自 _Greater 类
class StrictGreaterThan(_Greater):
    # 继承 GreaterThan 类的文档字符串
    __doc__ = GreaterThan.__doc__
    # 空的 __slots__ 列表，用于声明实例属性
    __slots__ = ()

    # 关系运算符设为 '>'
    rel_op = '>'

    # 类方法，模糊关系的评估，判断 lhs 是否严格大于 rhs
    @classmethod
    def _eval_fuzzy_relation(cls, lhs, rhs):
        return is_gt(lhs, rhs)

    # 属性方法，返回弱大于关系的实例
    @property
    def weak(self):
        return Ge(*self.args)

# 将 StrictGreaterThan 类重命名为 Gt
Gt = StrictGreaterThan

# 定义 StrictLessThan 类，继承自 _Less 类
class StrictLessThan(_Less):
    # 继承 GreaterThan 类的文档字符串
    __doc__ = GreaterThan.__doc__
    # 空的 __slots__ 列表，用于声明实例属性
    __slots__ = ()

    # 关系运算符设为 '<'
    rel_op = '<'

    # 类方法，模糊关系的评估，判断 lhs 是否严格小于 rhs
    @classmethod
    def _eval_fuzzy_relation(cls, lhs, rhs):
        return is_lt(lhs, rhs)

    # 属性方法，返回弱小于关系的实例
    @property
    def weak(self):
        return Le(*self.args)

# 将 StrictLessThan 类重命名为 Lt
Lt = StrictLessThan

# Relational.ValidRelationOperator 是一个字典，将字符串关系运算符映射到对应的关系类
Relational.ValidRelationOperator = {
    None: Equality,        # 等于关系
    '==': Equality,        # 等于关系
    'eq': Equality,        # 等于关系
    '!=': Unequality,      # 不等于关系
    '<>': Unequality,      # 不等于关系
    'ne': Unequality,      # 不等于关系
    '>=': GreaterThan,     # 大于等于关系
    'ge': GreaterThan,     # 大于等于关系
    '<=': LessThan,        # 小于等于关系
    'le': LessThan,        # 小于等于关系
    '>': StrictGreaterThan, # 严格大于关系
    'gt': StrictGreaterThan, # 严格大于关系
    '<': StrictLessThan,    # 严格小于关系
    'lt': StrictLessThan,   # 严格小于关系
}

# 函数 _n2(a, b) 计算 a 和 b 的差值保留两位小数，如果不可比较则返回 None
def _n2(a, b):
    """Return (a - b).evalf(2) if a and b are comparable, else None.
    This should only be used when a and b are already sympified.
    """
    # 如果 a 和 b 都可比较
    if a.is_comparable and b.is_comparable:
        # 计算差值 dif，保留两位小数
        dif = (a - b).evalf(2)
        # 如果 dif 可比较，则返回 dif
        if dif.is_comparable:
            return dif

# 函数 _eval_is_ge(lhs, rhs) 评估 lhs 是否大于等于 rhs，返回 None
@dispatch(Expr, Expr)
def _eval_is_ge(lhs, rhs):
    return None

# 函数 _eval_is_eq(lhs, rhs) 评估 lhs 是否等于 rhs，返回 None
@dispatch(Basic, Basic)
def _eval_is_eq(lhs, rhs):
    return None

# 函数 _eval_is_eq(lhs, rhs) 评估 Tuple lhs 是否等于 Expr rhs，返回 False
@dispatch(Tuple, Expr) # type: ignore
def _eval_is_eq(lhs, rhs):  # noqa:F811
    return False

# 函数 _eval_is_eq(lhs, rhs) 评估 Tuple lhs 是否等于 AppliedUndef rhs，返回 None
@dispatch(Tuple, AppliedUndef) # type: ignore
def _eval_is_eq(lhs, rhs):  # noqa:F811
    return None

# 函数 _eval_is_eq(lhs, rhs) 评估 Tuple lhs 是否等于 Symbol rhs，返回 None
@dispatch(Tuple, Symbol) # type: ignore
def _eval_is_eq(lhs, rhs):  # noqa:F811
    return None

# 函数 _eval_is_eq(lhs, rhs) 评估 Tuple lhs 是否等于 Tuple rhs，如果长度不同则返回 False
@dispatch(Tuple, Tuple) # type: ignore
def _eval_is_eq(lhs, rhs):  # noqa:F811
    if len(lhs) != len(rhs):
        return False

    # 遍历元组的每个元素，检查是否逐一相等
    return fuzzy_and(fuzzy_bool(is_eq(s, o)) for s, o in zip(lhs, rhs))

# 函数 is_lt(lhs, rhs, assumptions=None) 判断 lhs 是否严格小于 rhs，返回模糊布尔值
def is_lt(lhs, rhs, assumptions=None):
    """Fuzzy bool for lhs is strictly less than rhs.

    See the docstring for :func:`~.is_ge` for more.
    """
    return fuzzy_not(is_ge(lhs, rhs, assumptions))

# 函数 is_gt(lhs, rhs, assumptions=None) 判断 lhs 是否严格大于 rhs，返回模糊布尔值
def is_gt(lhs, rhs, assumptions=None):
    """Fuzzy bool for lhs is strictly greater than rhs.

    See the docstring for :func:`~.is_ge` for more.
    """
    return fuzzy_not(is_le(lhs, rhs, assumptions))

# 函数 is_le(lhs, rhs, assumptions=None) 判断 lhs 是否小于等于 rhs，返回模糊布尔值
def is_le(lhs, rhs, assumptions=None):
    """Fuzzy bool for lhs is less than or equal to rhs.

    See the docstring for :func:`~.is_ge` for more.
    """
    # 调用 is_ge 函数，判断 lhs 是否小于等于 rhs，使用 assumptions 参数
    return is_ge(rhs, lhs, assumptions)
def is_ge(lhs, rhs, assumptions=None):
    """
    Fuzzy bool for *lhs* is greater than or equal to *rhs*.

    Parameters
    ==========

    lhs : Expr
        The left-hand side of the expression, must be sympified,
        and an instance of expression. Throws an exception if
        lhs is not an instance of expression.

    rhs : Expr
        The right-hand side of the expression, must be sympified
        and an instance of expression. Throws an exception if
        lhs is not an instance of expression.

    assumptions: Boolean, optional
        Assumptions taken to evaluate the inequality.

    Returns
    =======

    ``True`` if *lhs* is greater than or equal to *rhs*, ``False`` if *lhs*
    is less than *rhs*, and ``None`` if the comparison between *lhs* and
    *rhs* is indeterminate.

    Explanation
    ===========

    This function is intended to give a relatively fast determination and
    deliberately does not attempt slow calculations that might help in
    obtaining a determination of True or False in more difficult cases.

    The four comparison functions ``is_le``, ``is_lt``, ``is_ge``, and ``is_gt`` are
    each implemented in terms of ``is_ge`` in the following way:

    is_ge(x, y) := is_ge(x, y)
    is_le(x, y) := is_ge(y, x)
    is_lt(x, y) := fuzzy_not(is_ge(x, y))
    is_gt(x, y) := fuzzy_not(is_ge(y, x))

    Therefore, supporting new type with this function will ensure behavior for
    other three functions as well.

    To maintain these equivalences in fuzzy logic it is important that in cases where
    either x or y is non-real all comparisons will give None.

    Examples
    ========

    >>> from sympy import S, Q
    >>> from sympy.core.relational import is_ge, is_le, is_gt, is_lt
    >>> from sympy.abc import x
    >>> is_ge(S(2), S(0))
    True
    >>> is_ge(S(0), S(2))
    False
    >>> is_le(S(0), S(2))
    True
    >>> is_gt(S(0), S(2))
    False
    >>> is_lt(S(2), S(0))
    False

    Assumptions can be passed to evaluate the quality which is otherwise
    indeterminate.

    >>> print(is_ge(x, S(0)))
    None
    >>> is_ge(x, S(0), assumptions=Q.positive(x))
    True

    New types can be supported by dispatching to ``_eval_is_ge``.

    >>> from sympy import Expr, sympify
    >>> from sympy.multipledispatch import dispatch
    >>> class MyExpr(Expr):
    ...     def __new__(cls, arg):
    ...         return super().__new__(cls, sympify(arg))
    ...     @property
    ...     def value(self):
    ...         return self.args[0]
    >>> @dispatch(MyExpr, MyExpr)
    ... def _eval_is_ge(a, b):
    ...     return is_ge(a.value, b.value)
    >>> a = MyExpr(1)
    >>> b = MyExpr(2)
    >>> is_ge(b, a)
    True
    >>> is_le(a, b)
    True
    """
    from sympy.assumptions.wrapper import AssumptionsWrapper, is_extended_nonnegative

    # 检查 lhs 和 rhs 是否为符号表达式，如果不是则引发 TypeError 异常
    if not (isinstance(lhs, Expr) and isinstance(rhs, Expr)):
        raise TypeError("Can only compare inequalities with Expr")

    # 调用内部函数 _eval_is_ge 来计算 lhs 是否大于等于 rhs 的结果
    retval = _eval_is_ge(lhs, rhs)
    # 如果 retval 不是 None，则直接返回它
    if retval is not None:
        return retval
    # 否则，计算 _n2(lhs, rhs) 的值
    else:
        n2 = _n2(lhs, rhs)
        # 如果 n2 不是 None
        if n2 is not None:
            # 对于无穷大的情况，使用浮点数比较
            # 否则会陷入无限递归
            if n2 in (S.Infinity, S.NegativeInfinity):
                n2 = float(n2)
            # 返回 n2 是否大于等于 0 的比较结果
            return n2 >= 0
        
        # 将 lhs 和 rhs 封装成带有 assumptions 的 AssumptionsWrapper 对象
        _lhs = AssumptionsWrapper(lhs, assumptions)
        _rhs = AssumptionsWrapper(rhs, assumptions)
        # 如果 lhs 和 rhs 都是扩展实数
        if _lhs.is_extended_real and _rhs.is_extended_real:
            # 如果 _lhs 是无穷大且是扩展正数，或者 _rhs 是无穷大且是扩展负数
            if (_lhs.is_infinite and _lhs.is_extended_positive) or (_rhs.is_infinite and _rhs.is_extended_negative):
                # 返回 True
                return True
            # 计算 lhs - rhs 的差值
            diff = lhs - rhs
            # 如果差值不是 NaN
            if diff is not S.NaN:
                # 判断差值是否是扩展非负数，并返回结果
                rv = is_extended_nonnegative(diff, assumptions)
                if rv is not None:
                    return rv
def is_neq(lhs, rhs, assumptions=None):
    """Fuzzy bool for lhs does not equal rhs.

    See the docstring for :func:`~.is_eq` for more.
    """
    # 返回 lhs 不等于 rhs 的模糊布尔值
    return fuzzy_not(is_eq(lhs, rhs, assumptions))


def is_eq(lhs, rhs, assumptions=None):
    """
    Fuzzy bool representing mathematical equality between *lhs* and *rhs*.

    Parameters
    ==========

    lhs : Expr
        The left-hand side of the expression, must be sympified.

    rhs : Expr
        The right-hand side of the expression, must be sympified.

    assumptions: Boolean, optional
        Assumptions taken to evaluate the equality.

    Returns
    =======

    ``True`` if *lhs* is equal to *rhs*, ``False`` is *lhs* is not equal to *rhs*,
    and ``None`` if the comparison between *lhs* and *rhs* is indeterminate.

    Explanation
    ===========

    This function is intended to give a relatively fast determination and
    deliberately does not attempt slow calculations that might help in
    obtaining a determination of True or False in more difficult cases.

    :func:`~.is_neq` calls this function to return its value, so supporting
    new type with this function will ensure correct behavior for ``is_neq``
    as well.

    Examples
    ========

    >>> from sympy import Q, S
    >>> from sympy.core.relational import is_eq, is_neq
    >>> from sympy.abc import x
    >>> is_eq(S(0), S(0))
    True
    >>> is_neq(S(0), S(0))
    False
    >>> is_eq(S(0), S(2))
    False
    >>> is_neq(S(0), S(2))
    True

    Assumptions can be passed to evaluate the equality which is otherwise
    indeterminate.

    >>> print(is_eq(x, S(0)))
    None
    >>> is_eq(x, S(0), assumptions=Q.zero(x))
    True

    New types can be supported by dispatching to ``_eval_is_eq``.

    >>> from sympy import Basic, sympify
    >>> from sympy.multipledispatch import dispatch
    >>> class MyBasic(Basic):
    ...     def __new__(cls, arg):
    ...         return Basic.__new__(cls, sympify(arg))
    ...     @property
    ...     def value(self):
    ...         return self.args[0]
    ...
    >>> @dispatch(MyBasic, MyBasic)
    ... def _eval_is_eq(a, b):
    ...     return is_eq(a.value, b.value)
    ...
    >>> a = MyBasic(1)
    >>> b = MyBasic(1)
    >>> is_eq(a, b)
    True
    >>> is_neq(a, b)
    False

    """
    # 这里只为了向后兼容性而调用 _eval_Eq
    # 新代码应该使用多分派的 is_eq，如文档字符串中所述
    for side1, side2 in (lhs, rhs), (rhs, lhs):
        eval_func = getattr(side1, '_eval_Eq', None)
        if eval_func is not None:
            retval = eval_func(side2)
            if retval is not None:
                return retval

    # 调用 _eval_is_eq 函数来进行具体的相等性评估
    retval = _eval_is_eq(lhs, rhs)
    if retval is not None:
        return retval

    # 如果类型不同，尝试反向评估
    if dispatch(type(lhs), type(rhs)) != dispatch(type(rhs), type(lhs)):
        retval = _eval_is_eq(rhs, lhs)
        if retval is not None:
            return retval
    # 如果 retval 仍然是 None，则进入等式逻辑判断
    # 如果表达式具有相同的结构，则它们必须相等。
    if lhs == rhs:
        return True  # 例如 True == True
    elif all(isinstance(i, BooleanAtom) for i in (rhs, lhs)):
        return False  # 例如 True != False
    elif not (lhs.is_Symbol or rhs.is_Symbol) and (
        isinstance(lhs, Boolean) !=
        isinstance(rhs, Boolean)):
        return False  # 只有布尔类型可以等于布尔类型

    # 导入必要的模块和函数
    from sympy.assumptions.wrapper import (AssumptionsWrapper,
        is_infinite, is_extended_real)
    from .add import Add

    # 对 lhs 和 rhs 应用假设包装器
    _lhs = AssumptionsWrapper(lhs, assumptions)
    _rhs = AssumptionsWrapper(rhs, assumptions)

    # 如果 lhs 或 rhs 是无穷大，则它们不相等
    if _lhs.is_infinite or _rhs.is_infinite:
        if fuzzy_xor([_lhs.is_infinite, _rhs.is_infinite]):
            return False
        # 如果其中一个是无限扩展实数，而另一个不是，则它们不相等
        if fuzzy_xor([_lhs.is_extended_real, _rhs.is_extended_real]):
            return False
        # 如果两个都是无限扩展实数，并且一个是正的而另一个不是，则它们不相等
        if fuzzy_and([_lhs.is_extended_real, _rhs.is_extended_real]):
            return fuzzy_xor([_lhs.is_extended_positive, fuzzy_not(_rhs.is_extended_positive)])

        # 尝试分离实部和虚部并进行比较
        I = S.ImaginaryUnit

        def split_real_imag(expr):
            # 判断表达式是实部还是虚部
            real_imag = lambda t: (
                'real' if is_extended_real(t, assumptions) else
                'imag' if is_extended_real(I*t, assumptions) else None)
            return sift(Add.make_args(expr), real_imag)

        # 分离 lhs 的实部和虚部
        lhs_ri = split_real_imag(lhs)
        if not lhs_ri[None]:
            # 分离 rhs 的实部和虚部
            rhs_ri = split_real_imag(rhs)
            if not rhs_ri[None]:
                # 比较实部和虚部是否相等
                eq_real = is_eq(Add(*lhs_ri['real']), Add(*rhs_ri['real']), assumptions)
                eq_imag = is_eq(I * Add(*lhs_ri['imag']), I * Add(*rhs_ri['imag']), assumptions)
                return fuzzy_and(map(fuzzy_bool, [eq_real, eq_imag]))

        # 导入必要的模块
        from sympy.functions.elementary.complexes import arg
        # 比较例如 zoo 和 1+I*oo 的参数
        arglhs = arg(lhs)
        argrhs = arg(rhs)
        # 防止 Eq(nan, nan) -> False 的情况
        if not (arglhs == S.NaN and argrhs == S.NaN):
            return fuzzy_bool(is_eq(arglhs, argrhs, assumptions))
    # 检查 lhs 和 rhs 是否都是 Expr 类型的实例
    if all(isinstance(i, Expr) for i in (lhs, rhs)):
        # 计算差异 dif = lhs - rhs
        dif = lhs - rhs
        # 将 dif 与给定的假设进行包装
        _dif = AssumptionsWrapper(dif, assumptions)
        # 检查 dif 是否为零
        z = _dif.is_zero
        if z is not None:
            # 如果 dif 不为零且是可交换的，则返回 False（参考 issue 10728）
            if z is False and _dif.is_commutative:
                return False
            # 如果 dif 为零，则返回 True
            if z:
                return True

        # 对于包含 Float 的情况，is_zero 无法帮助确定整数/有理数
        c, t = dif.as_coeff_Add()
        # 如果 c 是 Float 类型
        if c.is_Float:
            # 如果 c 是整数值且 t 不是整数，则返回 False
            if int_valued(c):
                if t.is_integer is False:
                    return False
            # 如果 c 不是有理数，则返回 False
            elif t.is_rational is False:
                return False

        # 计算 n2 = _n2(lhs, rhs)
        n2 = _n2(lhs, rhs)
        # 如果 n2 不为 None，则返回 _sympify(n2 == 0) 的结果
        if n2 is not None:
            return _sympify(n2 == 0)

        # 将 dif 分解为分子 n 和分母 d
        n, d = dif.as_numer_denom()
        rv = None
        # 对分子 n 和分母 d 分别进行假设包装
        _n = AssumptionsWrapper(n, assumptions)
        _d = AssumptionsWrapper(d, assumptions)
        # 如果 _n 是零，则 rv 取 _d 是否非零的结果
        if _n.is_zero:
            rv = _d.is_nonzero
        # 如果 _n 是有限的
        elif _n.is_finite:
            # 如果 _d 是无限的，则 rv 为 True
            if _d.is_infinite:
                rv = True
            # 否则如果 _n 不为零，则 rv 取 _d 是否为无限的结果
            elif _n.is_zero is False:
                rv = _d.is_infinite
                if rv is None:
                    # 如果使得分母无限的条件不能使原始表达式成为真，则返回 False
                    from sympy.simplify.simplify import clear_coefficients
                    l, r = clear_coefficients(d, S.Infinity)
                    args = [_.subs(l, r) for _ in (lhs, rhs)]
                    if args != [lhs, rhs]:
                        # 模糊布尔运算结果，检查是否相等
                        rv = fuzzy_bool(is_eq(*args, assumptions))
                        # 如果相等则返回 None
                        if rv is True:
                            rv = None
        # 否则如果分子 n 中任何一部分为无限，则返回 False
        elif any(is_infinite(a, assumptions) for a in Add.make_args(n)):
            rv = False
        # 返回 rv 结果，可能为 True、False 或 None
        if rv is not None:
            return rv
```