# `D:\src\scipysrc\sympy\sympy\sets\sets.py`

```
frompython
from typing import Any, Callable  # 导入类型提示相关的模块
from functools import reduce  # 导入用于函数式编程的reduce函数
from collections import defaultdict  # 导入默认字典
import inspect  # 导入用于检查对象的属性和方法的模块

from sympy.core.kind import Kind, UndefinedKind, NumberKind  # 导入Sympy库中与类型相关的模块
from sympy.core.basic import Basic  # 导入Sympy库中基本表达式的模块
from sympy.core.containers import Tuple, TupleKind  # 导入Sympy库中元组和元组类型相关的模块
from sympy.core.decorators import sympify_method_args, sympify_return  # 导入Sympy库中用于符号化的装饰器
from sympy.core.evalf import EvalfMixin  # 导入Sympy库中用于数值估算的Mixin类
from sympy.core.expr import Expr  # 导入Sympy库中表达式相关的模块
from sympy.core.function import Lambda  # 导入Sympy库中表示Lambda函数的模块
from sympy.core.logic import (FuzzyBool, fuzzy_bool, fuzzy_or, fuzzy_and,  # 导入Sympy库中模糊布尔逻辑相关的模块
    fuzzy_not)
from sympy.core.numbers import Float, Integer  # 导入Sympy库中浮点数和整数类型的模块
from sympy.core.operations import LatticeOp  # 导入Sympy库中Lattice操作相关的模块
from sympy.core.parameters import global_parameters  # 导入Sympy库中全局参数相关的模块
from sympy.core.relational import Eq, Ne, is_lt  # 导入Sympy库中关系运算相关的模块
from sympy.core.singleton import Singleton, S  # 导入Sympy库中单例模式相关的模块和单例对象S
from sympy.core.sorting import ordered  # 导入Sympy库中排序相关的模块
from sympy.core.symbol import symbols, Symbol, Dummy, uniquely_named_symbol  # 导入Sympy库中符号相关的模块和函数
from sympy.core.sympify import _sympify, sympify, _sympy_converter  # 导入Sympy库中符号化相关的模块和函数
from sympy.functions.elementary.exponential import exp, log  # 导入Sympy库中指数和对数函数相关的模块
from sympy.functions.elementary.miscellaneous import Max, Min  # 导入Sympy库中最大最小函数相关的模块
from sympy.logic.boolalg import And, Or, Not, Xor, true, false  # 导入Sympy库中布尔逻辑运算相关的模块和常量
from sympy.utilities.decorator import deprecated  # 导入Sympy库中用于标记过时函数的装饰器
from sympy.utilities.exceptions import sympy_deprecation_warning  # 导入Sympy库中用于Sympy特定异常的模块
from sympy.utilities.iterables import (iproduct, sift, roundrobin, iterable,  # 导入Sympy库中用于迭代操作的模块
                                       subsets)
from sympy.utilities.misc import func_name, filldedent  # 导入Sympy库中用于函数名和缩进处理的函数

from mpmath import mpi, mpf  # 导入mpmath库中的多精度实现相关的模块

from mpmath.libmp.libmpf import prec_to_dps  # 导入mpmath库中的精度转换函数


tfn = defaultdict(lambda: None, {  # 创建一个默认字rval = False

    is_FiniteSet = False
    is_Interval = False
    is_ProductSet = False
    is_Union = False
    is_Intersection: FuzzyBool = None
    is_UniversalSet: FuzzyBool = None
    is_Complement: FuzzyBool = None
    is_ComplexRegion = False

    is_empty: FuzzyBool = None
    is_finite_set: FuzzyBool = None

    @property  # type: ignore
    @deprecated(
        """
        The is_EmptySet attribute of Set objects is deprecated.
        Use 's is S.EmptySet" or 's.is_empty' instead.
        """,
        deprecated_since_version="1.5",
        active_deprecations_target="deprecated-is-emptyset",
    )
    # 定义 is_EmptySet 属性为只读属性，用于判断当前集合是否为空集
    def is_EmptySet(self):
        return None

    @staticmethod
    def _infimum_key(expr):
        """
        Return infimum (if possible) else S.Infinity.
        """
        try:
            # 尝试获取表达式的下确界
            infimum = expr.inf
            # 断言下确界可以比较
            assert infimum.is_comparable
            # 对下确界进行数值计算，解决问题 #18505
            infimum = infimum.evalf()  # issue #18505
        except (NotImplementedError,
                AttributeError, AssertionError, ValueError):
            # 如果出现错误，将下确界设为正无穷大
            infimum = S.Infinity
        # 返回下确界
        return infimum

    def union(self, other):
        """
        Returns the union of ``self`` and ``other``.

        Examples
        ========

        As a shortcut it is possible to use the ``+`` operator:

        >>> from sympy import Interval, FiniteSet
        >>> Interval(0, 1).union(Interval(2, 3))
        Union(Interval(0, 1), Interval(2, 3))
        >>> Interval(0, 1) + Interval(2, 3)
        Union(Interval(0, 1), Interval(2, 3))
        >>> Interval(1, 2, True, True) + FiniteSet(2, 3)
        Union({3}, Interval.Lopen(1, 2))

        Similarly it is possible to use the ``-`` operator for set differences:

        >>> Interval(0, 2) - Interval(0, 1)
        Interval.Lopen(1, 2)
        >>> Interval(1, 3) - FiniteSet(2)
        Union(Interval.Ropen(1, 2), Interval.Lopen(2, 3))

        """
        return Union(self, other)

    def intersect(self, other):
        """
        Returns the intersection of 'self' and 'other'.

        Examples
        ========

        >>> from sympy import Interval

        >>> Interval(1, 3).intersect(Interval(1, 2))
        Interval(1, 2)

        >>> from sympy import imageset, Lambda, symbols, S
        >>> n, m = symbols('n m')
        >>> a = imageset(Lambda(n, 2*n), S.Integers)
        >>> a.intersect(imageset(Lambda(m, 2*m + 1), S.Integers))
        EmptySet

        """
        return Intersection(self, other)

    def intersection(self, other):
        """
        Alias for :meth:`intersect()`
        """
        return self.intersect(other)

    def is_disjoint(self, other):
        """
        Returns True if ``self`` and ``other`` are disjoint.

        Examples
        ========

        >>> from sympy import Interval
        >>> Interval(0, 2).is_disjoint(Interval(1, 2))
        False
        >>> Interval(0, 2).is_disjoint(Interval(3, 4))
        True

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Disjoint_sets
        """
        # 判断两个集合是否互不相交
        return self.intersect(other) == S.EmptySet

    def isdisjoint(self, other):
        """
        Alias for :meth:`is_disjoint()`
        """
        return self.is_disjoint(other)

    def complement(self, universe):
        r"""
        The complement of 'self' w.r.t the given universe.

        Examples
        ========

        >>> from sympy import Interval, S
        >>> Interval(0, 1).complement(S.Reals)
        Union(Interval.open(-oo, 0), Interval.open(1, oo))

        >>> Interval(0, 1).complement(S.UniversalSet)
        Complement(UniversalSet, Interval(0, 1))

        """
        return Complement(universe, self)
    def _complement(self, other):
        # 行为类似于 other - self 的补集操作

        if isinstance(self, ProductSet) and isinstance(other, ProductSet):
            # 如果 self 和 other 都是 ProductSet 类型，则处理如下：
            # 如果 self 和 other 的集合数量不同，则 other - self 等于 other
            if len(self.sets) != len(other.sets):
                return other

            # 使用如下方式表示补集操作：
            # (A x B) - (C x D) = ((A - C) x B) U (A x (B - D))
            overlaps = []
            pairs = list(zip(self.sets, other.sets))
            for n in range(len(pairs)):
                sets = (o if i != n else o - s for i, (s, o) in enumerate(pairs))
                overlaps.append(ProductSet(*sets))
            return Union(*overlaps)

        elif isinstance(other, Interval):
            # 如果 other 是 Interval 类型
            if isinstance(self, (Interval, FiniteSet)):
                # 如果 self 是 Interval 或 FiniteSet 类型，则返回如下补集操作
                return Intersection(other, self.complement(S.Reals))

        elif isinstance(other, Union):
            # 如果 other 是 Union 类型，则返回 other 中每个元素减去 self 的结果的并集
            return Union(*(o - self for o in other.args))

        elif isinstance(other, Complement):
            # 如果 other 是 Complement 类型，则返回其他类型和 self 的补集组合
            return Complement(other.args[0], Union(other.args[1], self), evaluate=False)

        elif other is S.EmptySet:
            # 如果 other 是空集，则返回空集
            return S.EmptySet

        elif isinstance(other, FiniteSet):
            # 如果 other 是 FiniteSet 类型，则进行以下操作
            sifted = sift(other, lambda x: fuzzy_bool(self.contains(x)))
            # 过滤掉那些包含在 self 中的元素
            return Union(FiniteSet(*(sifted[False])),
                         Complement(FiniteSet(*(sifted[None])), self, evaluate=False)
                         if sifted[None] else S.EmptySet)

    def symmetric_difference(self, other):
        """
        返回 ``self`` 和 ``other`` 的对称差集。

        Examples
        ========

        >>> from sympy import Interval, S
        >>> Interval(1, 3).symmetric_difference(S.Reals)
        Union(Interval.open(-oo, 1), Interval.open(3, oo))
        >>> Interval(1, 10).symmetric_difference(S.Reals)
        Union(Interval.open(-oo, 1), Interval.open(10, oo))

        >>> from sympy import S, EmptySet
        >>> S.Reals.symmetric_difference(EmptySet)
        Reals

        References
        ==========
        .. [1] https://en.wikipedia.org/wiki/Symmetric_difference

        """
        return SymmetricDifference(self, other)

    def _symmetric_difference(self, other):
        # 返回 self 和 other 的对称差集
        return Union(Complement(self, other), Complement(other, self))

    @property
    def inf(self):
        """
        ``self`` 的下确界。

        Examples
        ========

        >>> from sympy import Interval, Union
        >>> Interval(0, 1).inf
        0
        >>> Union(Interval(0, 1), Interval(2, 3)).inf
        0

        """
        return self._inf

    @property
    def _inf(self):
        # 抛出未实现异常，表明需要子类实现该属性
        raise NotImplementedError("(%s)._inf" % self)

    @property
    def sup(self):
        """
        返回 ``self`` 的上确界。

        Examples
        ========

        >>> from sympy import Interval, Union
        >>> Interval(0, 1).sup
        1
        >>> Union(Interval(0, 1), Interval(2, 3)).sup
        3

        """
        返回 self 对象的上确界值
        return self._sup

    @property
    def _sup(self):
        抛出一个未实现错误，用于子类覆盖实现
        raise NotImplementedError("(%s)._sup" % self)

    def contains(self, other):
        """
        返回一个 SymPy 值，指示 ``other`` 是否包含在 ``self`` 中：
        如果包含则为 ``true``，不包含则为 ``false``，否则为一个未计算的 ``Contains`` 表达式
        （或者在 ConditionSet 和 FiniteSet/Interval 的联合情况下，表示包含条件的表达式）。

        Examples
        ========

        >>> from sympy import Interval, S
        >>> from sympy.abc import x

        >>> Interval(0, 1).contains(0.5)
        True

        可以使用 ``in`` 运算符作为快捷方式，但是除非得到肯定的 true 或 false，否则会引发错误。

        >>> Interval(0, 1).contains(x)
        (0 <= x) & (x <= 1)
        >>> x in Interval(0, 1)
        Traceback (most recent call last):
        ...
        TypeError: did not evaluate to a bool: None

        'in' 的结果是一个布尔值，不是 SymPy 值

        >>> 1 in Interval(0, 2)
        True
        >>> _ is S.true
        False
        """
        从 .contains 导入 Contains
        将 other 转换为 SymPy 表达式，严格模式
        other = sympify(other, strict=True)

        调用 self 的内部方法 _contains，判断 other 是否包含在 self 中
        c = self._contains(other)
        如果 c 是 Contains 类型的实例，则直接返回 c
        if isinstance(c, Contains):
            return c
        如果 c 是 None，则返回一个新的 Contains 对象，表示未计算的表达式
        if c is None:
            return Contains(other, self, evaluate=False)
        否则，根据 c 在 tfn 中的映射结果返回相应的值
        b = tfn[c]
        如果 b 也是 None，则返回 c
        if b is None:
            return c
        否则返回 b
        return b

    def _contains(self, other):
        """
        测试 ``other`` 是否是集合 ``self`` 的元素。

        这是一个内部方法，预期由 ``Set`` 的子类覆盖实现，并由公共的 :func:`Set.contains` 方法或者 :class:`Contains` 表达式调用。

        Parameters
        ==========

        other: 被 SymPy 化的 :class:`Basic` 实例
            要测试其在 ``self`` 中的成员资格的对象。

        Returns
        =======

        符号逻辑 :class:`Boolean` 或者 ``None``。

        返回 ``None`` 表示未知 ``other`` 是否包含在 ``self`` 中。从这里返回 ``None`` 可确保
        ``self.contains(other)`` 或者 ``Contains(self, other)`` 将返回一个未计算的 :class:`Contains` 表达式。

        如果不是 ``None``，则返回的值是一个逻辑上等价于 ``other`` 是 ``self`` 的元素的 :class:`Boolean`。
        通常这可能是 ``S.true`` 或 ``S.false``，但不总是这样。
        """
        抛出一个未实现错误，用于子类覆盖实现
        raise NotImplementedError(f"{type(self).__name__}._contains")
    # 判断当前集合是否为参数集合的子集
    def is_subset(self, other):
        """
        Returns True if ``self`` is a subset of ``other``.

        Examples
        ========

        >>> from sympy import Interval
        >>> Interval(0, 0.5).is_subset(Interval(0, 1))
        True
        >>> Interval(0, 1).is_subset(Interval(0, 1, left_open=True))
        False

        """
        # 如果参数 `other` 不是 Set 类型，则抛出 ValueError 异常
        if not isinstance(other, Set):
            raise ValueError("Unknown argument '%s'" % other)

        # 处理特殊情况
        # 如果当前集合与参数集合相等，则当前集合是参数集合的子集
        if self == other:
            return True
        # 如果当前集合为空集，则它是任何集合的子集
        is_empty = self.is_empty
        if is_empty is True:
            return True
        # 如果当前集合非空而参数集合为空，则当前集合不是参数集合的子集
        elif fuzzy_not(is_empty) and other.is_empty:
            return False
        # 如果当前集合是无限集合而参数集合是有限集合，则当前集合不是参数集合的子集
        if self.is_finite_set is False and other.is_finite_set:
            return False

        # 根据子类规则进行分派
        ret = self._eval_is_subset(other)
        if ret is not None:
            return ret
        ret = other._eval_is_superset(self)
        if ret is not None:
            return ret

        # 使用多分派的配对规则
        from sympy.sets.handlers.issubset import is_subset_sets
        ret = is_subset_sets(self, other)
        if ret is not None:
            return ret

        # 退回到计算交集来确定子集关系
        # XXX: 我们不应该这样做。应该避免创建新的 Set 对象来处理这样的查询。
        # 应该是交集方法使用 is_subset 进行评估。
        if self.intersect(other) == self:
            return True

    # 返回当前集合是否是参数集合的子集的模糊布尔值
    def _eval_is_subset(self, other):
        '''Returns a fuzzy bool for whether self is a subset of other.'''
        return None

    # 返回当前集合是否是参数集合的超集的模糊布尔值
    def _eval_is_superset(self, other):
        '''Returns a fuzzy bool for whether self is a subset of other.'''
        return None

    # 应该废弃这个方法：
    # 返回当前集合是否是参数集合的子集，是 is_subset 方法的别名
    def issubset(self, other):
        """
        Alias for :meth:`is_subset()`
        """
        return self.is_subset(other)

    # 返回当前集合是否是参数集合的真子集
    def is_proper_subset(self, other):
        """
        Returns True if ``self`` is a proper subset of ``other``.

        Examples
        ========

        >>> from sympy import Interval
        >>> Interval(0, 0.5).is_proper_subset(Interval(0, 1))
        True
        >>> Interval(0, 1).is_proper_subset(Interval(0, 1))
        False

        """
        # 如果参数 `other` 是 Set 类型，则返回当前集合是否是参数集合的真子集
        if isinstance(other, Set):
            return self != other and self.is_subset(other)
        else:
            # 如果参数 `other` 不是 Set 类型，则抛出 ValueError 异常
            raise ValueError("Unknown argument '%s'" % other)
    def is_superset(self, other):
        """
        Returns True if ``self`` is a superset of ``other``.

        Examples
        ========

        >>> from sympy import Interval
        >>> Interval(0, 0.5).is_superset(Interval(0, 1))
        False
        >>> Interval(0, 1).is_superset(Interval(0, 1, left_open=True))
        True

        """
        # 如果 ``other`` 是 Set 类的实例，则调用其 is_subset 方法检查是否为子集
        if isinstance(other, Set):
            return other.is_subset(self)
        else:
            # 若不是 Set 类实例，则引发 ValueError 异常
            raise ValueError("Unknown argument '%s'" % other)

    # This should be deprecated:
    def issuperset(self, other):
        """
        Alias for :meth:`is_superset()`
        """
        # 调用 is_superset 方法的别名，返回其结果
        return self.is_superset(other)

    def is_proper_superset(self, other):
        """
        Returns True if ``self`` is a proper superset of ``other``.

        Examples
        ========

        >>> from sympy import Interval
        >>> Interval(0, 1).is_proper_superset(Interval(0, 0.5))
        True
        >>> Interval(0, 1).is_proper_superset(Interval(0, 1))
        False

        """
        # 如果 ``other`` 是 Set 类的实例，且 self 不等于 other，并且 self 是 other 的超集，则返回 True
        if isinstance(other, Set):
            return self != other and self.is_superset(other)
        else:
            # 若不是 Set 类实例，则引发 ValueError 异常
            raise ValueError("Unknown argument '%s'" % other)

    def _eval_powerset(self):
        # 导入 PowerSet 类并返回以 self 为参数的实例
        from .powerset import PowerSet
        return PowerSet(self)

    def powerset(self):
        """
        Find the Power set of ``self``.

        Examples
        ========

        >>> from sympy import EmptySet, FiniteSet, Interval

        A power set of an empty set:

        >>> A = EmptySet
        >>> A.powerset()
        {EmptySet}

        A power set of a finite set:

        >>> A = FiniteSet(1, 2)
        >>> a, b, c = FiniteSet(1), FiniteSet(2), FiniteSet(1, 2)
        >>> A.powerset() == FiniteSet(a, b, c, EmptySet)
        True

        A power set of an interval:

        >>> Interval(1, 2).powerset()
        PowerSet(Interval(1, 2))

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Power_set

        """
        # 调用 _eval_powerset 方法获取 self 的 PowerSet
        return self._eval_powerset()

    @property
    def measure(self):
        """
        The (Lebesgue) measure of ``self``.

        Examples
        ========

        >>> from sympy import Interval, Union
        >>> Interval(0, 1).measure
        1
        >>> Union(Interval(0, 1), Interval(2, 3)).measure
        2

        """
        # 返回 self 的测度属性 _measure 的值
        return self._measure

    @property
    def kind(self):
        """
        The kind of a Set

        Explanation
        ===========

        Any :class:`Set` will have kind :class:`SetKind` which is
        parametrised by the kind of the elements of the set. For example
        most sets are sets of numbers and will have kind
        ``SetKind(NumberKind)``. If elements of sets are different in kind than
        their kind will ``SetKind(UndefinedKind)``. See
        :class:`sympy.core.kind.Kind` for an explanation of the kind system.

        Examples
        ========

        >>> from sympy import Interval, Matrix, FiniteSet, EmptySet, ProductSet, PowerSet

        >>> FiniteSet(Matrix([1, 2])).kind
        SetKind(MatrixKind(NumberKind))

        >>> Interval(1, 2).kind
        SetKind(NumberKind)

        >>> EmptySet.kind
        SetKind()

        A :class:`sympy.sets.powerset.PowerSet` is a set of sets:

        >>> PowerSet({1, 2, 3}).kind
        SetKind(SetKind(NumberKind))

        A :class:`ProductSet` represents the set of tuples of elements of
        other sets. Its kind is :class:`sympy.core.containers.TupleKind`
        parametrised by the kinds of the elements of those sets:

        >>> p = ProductSet(FiniteSet(1, 2), FiniteSet(3, 4))
        >>> list(p)
        [(1, 3), (2, 3), (1, 4), (2, 4)]
        >>> p.kind
        SetKind(TupleKind(NumberKind, NumberKind))

        When all elements of the set do not have same kind, the kind
        will be returned as ``SetKind(UndefinedKind)``:

        >>> FiniteSet(0, Matrix([1, 2])).kind
        SetKind(UndefinedKind)

        The kind of the elements of a set are given by the ``element_kind``
        attribute of ``SetKind``:

        >>> Interval(1, 2).kind.element_kind
        NumberKind

        See Also
        ========

        NumberKind
        sympy.core.kind.UndefinedKind
        sympy.core.containers.TupleKind
        MatrixKind
        sympy.matrices.expressions.sets.MatrixSet
        sympy.sets.conditionset.ConditionSet
        Rationals
        Naturals
        Integers
        sympy.sets.fancysets.ImageSet
        sympy.sets.fancysets.Range
        sympy.sets.fancysets.ComplexRegion
        sympy.sets.powerset.PowerSet
        sympy.sets.sets.ProductSet
        sympy.sets.sets.Interval
        sympy.sets.sets.Union
        sympy.sets.sets.Intersection
        sympy.sets.sets.Complement
        sympy.sets.sets.EmptySet
        sympy.sets.sets.UniversalSet
        sympy.sets.sets.FiniteSet
        sympy.sets.sets.SymmetricDifference
        sympy.sets.sets.DisjointUnion
        """
        return self._kind()

    @property


注释：
    def boundary(self):
        """
        The boundary or frontier of a set.

        Explanation
        ===========

        A point x is on the boundary of a set S if

        1.  x is in the closure of S.
            I.e. Every neighborhood of x contains a point in S.
        2.  x is not in the interior of S.
            I.e. There does not exist an open set centered on x contained
            entirely within S.

        There are the points on the outer rim of S.  If S is open then these
        points need not actually be contained within S.

        For example, the boundary of an interval is its start and end points.
        This is true regardless of whether or not the interval is open.

        Examples
        ========

        >>> from sympy import Interval
        >>> Interval(0, 1).boundary
        {0, 1}
        >>> Interval(0, 1, True, False).boundary
        {0, 1}
        """
        return self._boundary

    @property
    def is_open(self):
        """
        Property method to check whether a set is open.

        Explanation
        ===========

        A set is open if and only if it has an empty intersection with its
        boundary. In particular, a subset A of the reals is open if and only
        if each one of its points is contained in an open interval that is a
        subset of A.

        Examples
        ========
        >>> from sympy import S
        >>> S.Reals.is_open
        True
        >>> S.Rationals.is_open
        False
        """
        return Intersection(self, self.boundary).is_empty

    @property
    def is_closed(self):
        """
        A property method to check whether a set is closed.

        Explanation
        ===========

        A set is closed if its complement is an open set. The closedness of a
        subset of the reals is determined with respect to R and its standard
        topology.

        Examples
        ========
        >>> from sympy import Interval
        >>> Interval(0, 1).is_closed
        True
        """
        return self.boundary.is_subset(self)

    @property
    def closure(self):
        """
        Property method which returns the closure of a set.
        The closure is defined as the union of the set itself and its
        boundary.

        Examples
        ========
        >>> from sympy import S, Interval
        >>> S.Reals.closure
        Reals
        >>> Interval(0, 1).closure
        Interval(0, 1)
        """
        return self + self.boundary

    @property
    def interior(self):
        """
        Property method which returns the interior of a set.
        The interior of a set S consists all points of S that do not
        belong to the boundary of S.

        Examples
        ========
        >>> from sympy import Interval
        >>> Interval(0, 1).interior
        Interval.open(0, 1)
        >>> Interval(0, 1).boundary.interior
        EmptySet
        """
        return self - self.boundary
    # 定义一个抽象方法 _boundary，子类需要实现具体逻辑
    def _boundary(self):
        raise NotImplementedError()

    # 定义一个属性 _measure，子类需要实现具体逻辑，返回对象的度量值
    @property
    def _measure(self):
        raise NotImplementedError("(%s)._measure" % self)

    # 定义一个方法 _kind，返回一个 SetKind 对象，其类型为 UndefinedKind
    def _kind(self):
        return SetKind(UndefinedKind)

    # 定义一个方法 _eval_evalf，接受一个精度参数 prec，返回对象在给定精度下的数值估算结果
    def _eval_evalf(self, prec):
        # 将精度转换为小数位数 dps
        dps = prec_to_dps(prec)
        # 对对象的每个参数进行数值估算，并返回结果
        return self.func(*[arg.evalf(n=dps) for arg in self.args])

    # 定义特殊方法 __add__，接受一个 Set 类型的参数 other，返回当前对象与参数的并集
    @sympify_return([('other', 'Set')], NotImplemented)
    def __add__(self, other):
        return self.union(other)

    # 定义特殊方法 __or__，接受一个 Set 类型的参数 other，返回当前对象与参数的并集
    @sympify_return([('other', 'Set')], NotImplemented)
    def __or__(self, other):
        return self.union(other)

    # 定义特殊方法 __and__，接受一个 Set 类型的参数 other，返回当前对象与参数的交集
    @sympify_return([('other', 'Set')], NotImplemented)
    def __and__(self, other):
        return self.intersect(other)

    # 定义特殊方法 __mul__，接受一个 Set 类型的参数 other，返回当前对象与参数的笛卡尔积
    @sympify_return([('other', 'Set')], NotImplemented)
    def __mul__(self, other):
        return ProductSet(self, other)

    # 定义特殊方法 __xor__，接受一个 Set 类型的参数 other，返回当前对象与参数的对称差
    @sympify_return([('other', 'Set')], NotImplemented)
    def __xor__(self, other):
        return SymmetricDifference(self, other)

    # 定义特殊方法 __pow__，接受一个 Expr 类型的参数 exp，返回当前对象的 exp 次幂
    @sympify_return([('exp', Expr)], NotImplemented)
    def __pow__(self, exp):
        # 如果 exp 不是正整数，抛出 ValueError 异常
        if not (exp.is_Integer and exp >= 0):
            raise ValueError("%s: Exponent must be a positive Integer" % exp)
        # 返回当前对象的 exp 次幂，即当前对象的笛卡尔积的 exp 次方
        return ProductSet(*[self]*exp)

    # 定义特殊方法 __sub__，接受一个 Set 类型的参数 other，返回当前对象与参数的补集
    @sympify_return([('other', 'Set')], NotImplemented)
    def __sub__(self, other):
        return Complement(self, other)

    # 定义特殊方法 __contains__，检查参数 other 是否属于当前对象
    def __contains__(self, other):
        # 将参数 other 转换为符号表达式
        other = _sympify(other)
        # 调用对象的 _contains 方法，检查 other 是否属于当前对象
        c = self._contains(other)
        # 从 tfn 字典获取结果
        b = tfn[c]
        # 如果结果为 None，抛出 TypeError 异常，要求结果必须是布尔值
        if b is None:
            raise TypeError('did not evaluate to a bool: %r' % c)
        # 返回结果布尔值
        return b
class ProductSet(Set):
    """
    Represents a Cartesian Product of Sets.

    Explanation
    ===========

    Returns a Cartesian product given several sets as either an iterable
    or individual arguments.

    Can use ``*`` operator on any sets for convenient shorthand.

    Examples
    ========

    >>> from sympy import Interval, FiniteSet, ProductSet
    >>> I = Interval(0, 5); S = FiniteSet(1, 2, 3)
    >>> ProductSet(I, S)
    ProductSet(Interval(0, 5), {1, 2, 3})

    >>> (2, 2) in ProductSet(I, S)
    True

    >>> Interval(0, 1) * Interval(0, 1) # The unit square
    ProductSet(Interval(0, 1), Interval(0, 1))

    >>> coin = FiniteSet('H', 'T')
    >>> set(coin**2)
    {(H, H), (H, T), (T, H), (T, T)}

    The Cartesian product is not commutative or associative e.g.:

    >>> I*S == S*I
    False
    >>> (I*I)*I == I*(I*I)
    False

    Notes
    =====

    - Passes most operations down to the argument sets

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Cartesian_product
    """
    is_ProductSet = True  # 设置一个类变量标志，表示这是一个ProductSet类的对象

    def __new__(cls, *sets, **assumptions):
        # 如果只传入一个集合的可迭代对象并且不是Set类型，则发出警告并转换为多个集合的元组形式
        if len(sets) == 1 and iterable(sets[0]) and not isinstance(sets[0], (Set, set)):
            sympy_deprecation_warning(
                """
ProductSet(iterable) is deprecated. Use ProductSet(*iterable) instead.
                """,
                deprecated_since_version="1.5",
                active_deprecations_target="deprecated-productset-iterable",
            )
            sets = tuple(sets[0])

        # 将所有集合元素sympify化，确保它们是Sympy表达式
        sets = [sympify(s) for s in sets]

        # 如果不是所有参数都是Set类型，则抛出TypeError
        if not all(isinstance(s, Set) for s in sets):
            raise TypeError("Arguments to ProductSet should be of type Set")

        # 如果没有传入任何集合，则返回空的有限集合
        if len(sets) == 0:
            return FiniteSet(())

        # 如果集合中包含EmptySet，则返回EmptySet
        if S.EmptySet in sets:
            return S.EmptySet

        # 调用基类的__new__方法创建对象
        return Basic.__new__(cls, *sets, **assumptions)

    @property
    def sets(self):
        # 返回ProductSet对象的参数列表
        return self.args

    def flatten(self):
        # 定义一个内部函数，用于展开ProductSet中的嵌套结构
        def _flatten(sets):
            for s in sets:
                if s.is_ProductSet:
                    yield from _flatten(s.sets)
                else:
                    yield s
        return ProductSet(*_flatten(self.sets))

    def _contains(self, element):
        """
        ``in`` operator for ProductSets.

        Examples
        ========

        >>> from sympy import Interval
        >>> (2, 3) in Interval(0, 5) * Interval(0, 5)
        True

        >>> (10, 10) in Interval(0, 5) * Interval(0, 5)
        False

        Passes operation on to constituent sets
        """
        # 如果元素是Symbol类型，则返回None
        if element.is_Symbol:
            return None

        # 如果元素不是Tuple类型或者长度与集合数量不符，则返回False
        if not isinstance(element, Tuple) or len(element) != len(self.sets):
            return S.false

        # 使用And逻辑运算符检查元素是否属于各个集合
        return And(*[s.contains(e) for s, e in zip(self.sets, element)])
    # 将输入的符号参数转换为符号对象
    symbols = [_sympify(s) for s in symbols]
    # 检查符号数量是否与集合数量相同，并且所有元素都是符号对象
    if len(symbols) != len(self.sets) or not all(
            i.is_Symbol for i in symbols):
        # 如果不满足条件，抛出值错误异常
        raise ValueError(
            'number of symbols must match the number of sets')
    # 对每个集合和对应的符号调用其 as_relational 方法，返回逻辑与连接结果
    return And(*[s.as_relational(i) for s, i in zip(self.sets, symbols)])

@property
def _boundary(self):
    # 构建并返回所有集合的边界的并集
    return Union(*(ProductSet(*(b + b.boundary if i != j else b.boundary
                            for j, b in enumerate(self.sets)))
                            for i, a in enumerate(self.sets)))

@property
def is_iterable(self):
    """
    A property method which tests whether a set is iterable or not.
    Returns True if set is iterable, otherwise returns False.

    Examples
    ========

    >>> from sympy import FiniteSet, Interval
    >>> I = Interval(0, 1)
    >>> A = FiniteSet(1, 2, 3, 4, 5)
    >>> I.is_iterable
    False
    >>> A.is_iterable
    True

    """
    # 检查所有集合是否可迭代
    return all(set.is_iterable for set in self.sets)

def __iter__(self):
    """
    A method which implements is_iterable property method.
    If self.is_iterable returns True (both constituent sets are iterable),
    then return the Cartesian Product. Otherwise, raise TypeError.
    """
    # 如果 self.is_iterable 返回 True，则返回集合的笛卡尔积迭代器
    return iproduct(*self.sets)

@property
def is_empty(self):
    # 返回所有集合是否为空的逻辑或结果
    return fuzzy_or(s.is_empty for s in self.sets)

@property
def is_finite_set(self):
    # 检查所有集合是否都是有限集合的模糊与结果
    all_finite = fuzzy_and(s.is_finite_set for s in self.sets)
    return fuzzy_or([self.is_empty, all_finite])

@property
def _measure(self):
    # 计算所有集合的度量值的乘积
    measure = 1
    for s in self.sets:
        measure *= s.measure
    return measure

def _kind(self):
    # 返回集合元素类型的元组类型
    return SetKind(TupleKind(*(i.kind.element_kind for i in self.args)))

def __len__(self):
    # 返回所有集合中元素数量的乘积
    return reduce(lambda a, b: a*b, (len(s) for s in self.args))

def __bool__(self):
    # 检查所有集合是否都为真
    return all(self.sets)
class Interval(Set):
    """
    Represents a real interval as a Set.

    Usage:
        Returns an interval with end points ``start`` and ``end``.

        For ``left_open=True`` (default ``left_open`` is ``False``) the interval
        will be open on the left. Similarly, for ``right_open=True`` the interval
        will be open on the right.

    Examples
    ========

    >>> from sympy import Symbol, Interval
    >>> Interval(0, 1)
    Interval(0, 1)
    >>> Interval.Ropen(0, 1)
    Interval.Ropen(0, 1)
    >>> Interval.Ropen(0, 1)
    Interval.Ropen(0, 1)
    >>> Interval.Lopen(0, 1)
    Interval.Lopen(0, 1)
    >>> Interval.open(0, 1)
    Interval.open(0, 1)

    >>> a = Symbol('a', real=True)
    >>> Interval(0, a)
    Interval(0, a)

    Notes
    =====
    - Only real end points are supported
    - ``Interval(a, b)`` with $a > b$ will return the empty set
    - Use the ``evalf()`` method to turn an Interval into an mpmath
      ``mpi`` interval instance

    References
    ==========
    .. [1] https://en.wikipedia.org/wiki/Interval_%28mathematics%29
    """
    is_Interval = True  # 标志这个类是一个 Interval 类

    def __new__(cls, start, end, left_open=False, right_open=False):
        """
        Creates a new Interval object with specified start and end points.

        Parameters:
        - start: The starting point of the interval
        - end: The ending point of the interval
        - left_open: Boolean indicating if the interval is open on the left (default False)
        - right_open: Boolean indicating if the interval is open on the right (default False)

        Returns:
        - An Interval object representing the specified interval.

        Raises:
        - NotImplementedError: If left_open or right_open are not boolean values
        - ValueError: If the interval is not real or if start > end

        Notes:
        - Handles special cases for infinite endpoints.
        """
        start = _sympify(start)  # 将 start 转换为 sympy 符号表达式
        end = _sympify(end)  # 将 end 转换为 sympy 符号表达式
        left_open = _sympify(left_open)  # 将 left_open 转换为 sympy 符号表达式
        right_open = _sympify(right_open)  # 将 right_open 转换为 sympy 符号表达式

        if not all(isinstance(a, (type(true), type(false))) for a in [left_open, right_open]):
            # 如果 left_open 或 right_open 不是布尔值，抛出异常
            raise NotImplementedError(
                "left_open and right_open can have only true/false values, "
                "got %s and %s" % (left_open, right_open))

        # 只支持实数区间
        if fuzzy_not(fuzzy_and(i.is_extended_real for i in (start, end, end-start))):
            raise ValueError("Non-real intervals are not supported")

        # 处理起始点大于结束点的情况
        if is_lt(end, start):
            return S.EmptySet
        elif (end - start).is_negative:
            return S.EmptySet

        # 处理起始点等于结束点的情况
        if end == start and (left_open or right_open):
            return S.EmptySet
        if end == start and not (left_open or right_open):
            if start is S.Infinity or start is S.NegativeInfinity:
                return S.EmptySet
            return FiniteSet(end)

        # 确保无限区间的端点是开放的
        if start is S.NegativeInfinity:
            left_open = true
        if end is S.Infinity:
            right_open = true
        if start == S.Infinity or end == S.NegativeInfinity:
            return S.EmptySet

        # 返回新的 Interval 对象
        return Basic.__new__(cls, start, end, left_open, right_open)

    @property
    def start(self):
        """
        The left end point of the interval.

        This property takes the same value as the ``inf`` property.

        Examples
        ========

        >>> from sympy import Interval
        >>> Interval(0, 1).start
        0

        """
        return self._args[0]  # 返回区间的起始点
    # 返回区间的右端点，与属性 `sup` 的值相同
    def end(self):
        return self._args[1]

    # 返回区间是否左开的布尔值
    @property
    def left_open(self):
        return self._args[2]

    # 返回区间是否右开的布尔值
    @property
    def right_open(self):
        return self._args[3]

    # 类方法：返回一个不包括任何边界的区间对象
    @classmethod
    def open(cls, a, b):
        return cls(a, b, True, True)

    # 类方法：返回一个不包括左边界的区间对象
    @classmethod
    def Lopen(cls, a, b):
        return cls(a, b, True, False)

    # 类方法：返回一个不包括右边界的区间对象
    @classmethod
    def Ropen(cls, a, b):
        return cls(a, b, False, True)

    # 返回区间的下确界（起始点）
    @property
    def _inf(self):
        return self.start

    # 返回区间的上确界（结束点）
    @property
    def _sup(self):
        return self.end

    # 返回区间的左端点（起始点）
    @property
    def left(self):
        return self.start

    # 返回区间的右端点（结束点）
    @property
    def right(self):
        return self.end

    # 返回区间是否为空集的布尔值
    @property
    def is_empty(self):
        # 判断区间是否为空集
        if self.left_open or self.right_open:
            cond = self.start >= self.end  # 至少有一个边界开放
        else:
            cond = self.start > self.end  # 两个边界均闭合
        return fuzzy_bool(cond)

    # 返回区间是否为有限集的布尔值
    @property
    def is_finite_set(self):
        return self.measure.is_zero

    # 返回与另一个区间的补集
    def _complement(self, other):
        if other == S.Reals:
            # 如果补集是实数集，则返回左开和右开的两个区间的并集
            a = Interval(S.NegativeInfinity, self.start,
                         True, not self.left_open)
            b = Interval(self.end, S.Infinity, not self.right_open, True)
            return Union(a, b)

        if isinstance(other, FiniteSet):
            # 如果补集是有限集，则返回空值
            nums = [m for m in other.args if m.is_number]
            if nums == []:
                return None

        return Set._complement(self, other)

    # 返回区间的边界点集合
    @property
    def _boundary(self):
        # 获取区间内有限的端点并构成一个有限集合
        finite_points = [p for p in (self.start, self.end)
                         if abs(p) != S.Infinity]
        return FiniteSet(*finite_points)
    # 检查给定对象是否为表达式类型，并且不是NaN，且是实数且不包含ComplexInfinity
    def _contains(self, other):
        if (not isinstance(other, Expr) or other is S.NaN
            or other.is_real is False or other.has(S.ComplexInfinity)):
                # 如果表达式包含无穷大或者NaN，则不是实数
                return false

        # 如果区间的起始点是负无穷，终点是正无穷
        if self.start is S.NegativeInfinity and self.end is S.Infinity:
            # 如果给定对象是实数，则返回相应的真假值
            if other.is_real is not None:
                return tfn[other.is_real]

        # 创建一个虚拟符号来代替表达式，并返回该区间与给定表达式之间的关系
        d = Dummy()
        return self.as_relational(d).subs(d, other)

    def as_relational(self, x):
        """将区间重写为不等式和逻辑运算符的形式。"""
        x = sympify(x)
        # 根据区间是否右开设置右侧不等式
        if self.right_open:
            right = x < self.end
        else:
            right = x <= self.end
        # 根据区间是否左开设置左侧不等式
        if self.left_open:
            left = self.start < x
        else:
            left = self.start <= x
        return And(left, right)

    @property
    def _measure(self):
        """返回区间的测量值，即区间长度。"""
        return self.end - self.start

    def _kind(self):
        """返回区间的类型，这里是 NumberKind。"""
        return SetKind(NumberKind)

    def to_mpi(self, prec=53):
        """将区间转换为精确的多精度区间。"""
        return mpi(mpf(self.start._eval_evalf(prec)),
            mpf(self.end._eval_evalf(prec)))

    def _eval_evalf(self, prec):
        """返回区间的浮点数估值。"""
        return Interval(self.left._evalf(prec), self.right._evalf(prec),
            left_open=self.left_open, right_open=self.right_open)

    def _is_comparable(self, other):
        """检查两个区间是否可以比较。"""
        is_comparable = self.start.is_comparable
        is_comparable &= self.end.is_comparable
        is_comparable &= other.start.is_comparable
        is_comparable &= other.end.is_comparable

        return is_comparable

    @property
    def is_left_unbounded(self):
        """如果区间左端点为负无穷，则返回True。"""
        return self.left is S.NegativeInfinity or self.left == Float("-inf")

    @property
    def is_right_unbounded(self):
        """如果区间右端点为正无穷，则返回True。"""
        return self.right is S.Infinity or self.right == Float("+inf")

    def _eval_Eq(self, other):
        """检查区间是否等于给定对象。"""
        if not isinstance(other, Interval):
            if isinstance(other, FiniteSet):
                return false
            elif isinstance(other, Set):
                return None
            return false
    # 表示一个集合的并集，继承自 Set 和 LatticeOp
    """
    Represents a union of sets as a :class:`Set`.

    Examples
    ========

    >>> from sympy import Union, Interval
    >>> Union(Interval(1, 2), Interval(3, 4))
    Union(Interval(1, 2), Interval(3, 4))

    The Union constructor will always try to merge overlapping intervals,
    if possible. For example:

    >>> Union(Interval(1, 2), Interval(2, 3))
    Interval(1, 3)

    See Also
    ========

    Intersection

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Union_%28set_theory%29
    """
    # 标志表示这是一个 Union 类
    is_Union = True

    @property
    # 返回空集作为标识元素
    def identity(self):
        return S.EmptySet

    @property
    # 返回全集作为零元素
    def zero(self):
        return S.UniversalSet

    def __new__(cls, *args, **kwargs):
        # 是否对输入进行评估，默认为全局设置中的 evaluate
        evaluate = kwargs.get('evaluate', global_parameters.evaluate)

        # 将输入参数转化为 Sympy 对象
        args = _sympify(args)

        # 使用已知规则简化集合
        if evaluate:
            args = list(cls._new_args_filter(args))
            return simplify_union(args)

        # 对参数列表进行排序，按 Set._infimum_key 排序
        args = list(ordered(args, Set._infimum_key))

        # 创建新的 Union 对象
        obj = Basic.__new__(cls, *args)
        obj._argset = frozenset(args)
        return obj

    @property
    # 返回 Union 对象的参数列表
    def args(self):
        return self._args

    def _complement(self, universe):
        # 返回当前 Union 对象在 universe 中的补集，应用 DeMorgan 定律
        return Intersection(s.complement(universe) for s in self.args)

    @property
    # 返回所有参数的 inf 的最小值，用于符号区间端点的组合
    def _inf(self):
        return Min(*[set.inf for set in self.args])

    @property
    # 返回所有参数的 sup 的最大值，用于符号端点的组合
    def _sup(self):
        return Max(*[set.sup for set in self.args])

    @property
    # 判断 Union 对象是否为空集合
    def is_empty(self):
        return fuzzy_and(set.is_empty for set in self.args)

    @property
    # 判断 Union 对象是否为有限集合
    def is_finite_set(self):
        return fuzzy_and(set.is_finite_set for set in self.args)

    @property
    def _measure(self):
        # Measure of a union is the sum of the measures of the sets minus
        # the sum of their pairwise intersections plus the sum of their
        # triple-wise intersections minus ... etc...

        # Sets is a collection of intersections and a set of elementary
        # sets which made up those intersections (called "sos" for set of sets)
        # An example element might of this list might be:
        #    ( {A,B,C}, A.intersect(B).intersect(C) )

        # Start with just elementary sets (  ({A}, A), ({B}, B), ... )
        # Then get and subtract (  ({A,B}, (A int B), ... ) while non-zero
        sets = [(FiniteSet(s), s) for s in self.args]  # 初始为元素集合的列表
        measure = 0  # 初始化测度为零
        parity = 1  # 初始奇偶性为正数
        while sets:
            # Add up the measure of these sets and add or subtract it to total
            measure += parity * sum(inter.measure for sos, inter in sets)

            # For each intersection in sets, compute the intersection with every
            # other set not already part of the intersection.
            sets = ((sos + FiniteSet(newset), newset.intersect(intersection))
                    for sos, intersection in sets for newset in self.args
                    if newset not in sos)

            # Clear out sets with no measure
            sets = [(sos, inter) for sos, inter in sets if inter.measure != 0]

            # Clear out duplicates
            sos_list = []
            sets_list = []
            for _set in sets:
                if _set[0] in sos_list:
                    continue
                else:
                    sos_list.append(_set[0])
                    sets_list.append(_set)
            sets = sets_list

            # Flip Parity - next time subtract/add if we added/subtracted here
            parity *= -1
        return measure  # 返回计算得到的测度值

    def _kind(self):
        kinds = tuple(arg.kind for arg in self.args if arg is not S.EmptySet)
        if not kinds:
            return SetKind()  # 如果没有定义类型，返回空集合的类型
        elif all(i == kinds[0] for i in kinds):
            return kinds[0]  # 如果所有元素的类型相同，则返回该类型
        else:
            return SetKind(UndefinedKind)  # 否则返回未定义类型

    @property
    def _boundary(self):
        def boundary_of_set(i):
            """ The boundary of set i minus interior of all other sets """
            b = self.args[i].boundary  # 计算集合 i 的边界
            for j, a in enumerate(self.args):
                if j != i:
                    b = b - a.interior  # 减去除集合 i 外所有其他集合的内部
            return b
        return Union(*map(boundary_of_set, range(len(self.args))))  # 返回所有集合边界的并集

    def _contains(self, other):
        return Or(*[s.contains(other) for s in self.args])  # 返回所有集合是否包含元素 other 的逻辑或

    def is_subset(self, other):
        return fuzzy_and(s.is_subset(other) for s in self.args)  # 返回所有集合是否为给定集合的模糊与
    # 将 Union 表达式重新写成等式和逻辑运算符的形式
    def as_relational(self, symbol):
        """Rewrite a Union in terms of equalities and logic operators. """
        # 如果 Union 只有两个 Interval 类型的参数
        if (len(self.args) == 2 and
                all(isinstance(i, Interval) for i in self.args)):
            # 优化，将三个参数优化为 (x > 1) & (x < 5) & Ne(x, 3)
            # 而不是作为四个参数 ((1 <= x) & (x < 3)) | ((x <= 5) & (3 < x))
            # XXX: 理想情况应当改进以处理任意数量的区间，并且不假定区间按特定顺序排序。
            a, b = self.args
            # 如果一个区间的上界等于另一个区间的下界，并且左开右开
            if a.sup == b.inf and a.right_open and b.left_open:
                # 判断最小条件
                mincond = symbol > a.inf if a.left_open else symbol >= a.inf
                # 判断最大条件
                maxcond = symbol < b.sup if b.right_open else symbol <= b.sup
                # 不等条件
                necond = Ne(symbol, a.sup)
                # 返回逻辑与的结果：Ne(symbol, a.sup) & mincond & maxcond
                return And(necond, mincond, maxcond)
        # 如果不满足上述条件，返回参数列表中各项的逻辑或结果
        return Or(*[i.as_relational(symbol) for i in self.args])

    # 检查对象是否可迭代的属性
    @property
    def is_iterable(self):
        return all(arg.is_iterable for arg in self.args)

    # 实现迭代器接口，将多个参数合并为一个迭代器
    def __iter__(self):
        return roundrobin(*(iter(arg) for arg in self.args))
class Intersection(Set, LatticeOp):
    """
    Represents an intersection of sets as a :class:`Set`.

    Examples
    ========

    >>> from sympy import Intersection, Interval
    >>> Intersection(Interval(1, 3), Interval(2, 4))
    Interval(2, 3)

    We often use the .intersect method

    >>> Interval(1,3).intersect(Interval(2,4))
    Interval(2, 3)

    See Also
    ========

    Union

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Intersection_%28set_theory%29
    """
    
    # 设置类属性，表示这个类是 Intersection 类型的对象
    is_Intersection = True

    @property
    # 返回 Intersection 的 identity 属性，即全集
    def identity(self):
        return S.UniversalSet

    @property
    # 返回 Intersection 的 zero 属性，即空集
    def zero(self):
        return S.EmptySet

    def __new__(cls, *args , evaluate=None):
        # 如果 evaluate 未指定，则使用全局参数中的 evaluate
        if evaluate is None:
            evaluate = global_parameters.evaluate

        # 将输入的参数 args 扁平化，并去重
        args = list(ordered(set(_sympify(args))))

        # 如果 evaluate 为真，则应用已知的规则来简化集合交集
        if evaluate:
            args = list(cls._new_args_filter(args))
            return simplify_intersection(args)

        # 根据 Set._infimum_key 对 args 进行排序
        args = list(ordered(args, Set._infimum_key))

        # 创建一个新的 Intersection 对象
        obj = Basic.__new__(cls, *args)
        obj._argset = frozenset(args)
        return obj

    @property
    # 返回 Intersection 的参数列表
    def args(self):
        return self._args

    @property
    # 返回 Intersection 是否是可迭代的
    def is_iterable(self):
        return any(arg.is_iterable for arg in self.args)

    @property
    # 返回 Intersection 是否是有限集合
    def is_finite_set(self):
        if fuzzy_or(arg.is_finite_set for arg in self.args):
            return True

    def _kind(self):
        # 返回 Intersection 中各个元素的种类，如果所有元素种类相同则返回该种类
        kinds = tuple(arg.kind for arg in self.args if arg is not S.UniversalSet)
        if not kinds:
            return SetKind(UndefinedKind)
        elif all(i == kinds[0] for i in kinds):
            return kinds[0]
        else:
            return SetKind()

    @property
    # 返回 Intersection 的下确界，但此处未实现
    def _inf(self):
        raise NotImplementedError()

    @property
    # 返回 Intersection 的上确界，但此处未实现
    def _sup(self):
        raise NotImplementedError()

    def _contains(self, other):
        # 检查 other 是否属于 Intersection 中的所有集合的交集
        return And(*[set.contains(other) for set in self.args])
    # 定义一个迭代方法，用于迭代Intersection对象的元素
    def __iter__(self):
        # 使用 sift 函数筛选出可迭代的参数集合
        sets_sift = sift(self.args, lambda x: x.is_iterable)

        # 初始化完成标志
        completed = False
        # 将可迭代的集合和未知是否可迭代的集合作为候选集合
        candidates = sets_sift[True] + sets_sift[None]

        # 初始化有限候选集合和其他集合
        finite_candidates, others = [], []
        # 遍历候选集合
        for candidate in candidates:
            length = None
            # 尝试获取候选集合的长度
            try:
                length = len(candidate)
            except TypeError:
                # 若无法获取长度，则将候选集合添加到其他集合
                others.append(candidate)

            # 若成功获取长度，则将候选集合添加到有限候选集合
            if length is not None:
                finite_candidates.append(candidate)
        # 根据候选集合的长度排序有限候选集合
        finite_candidates.sort(key=len)

        # 遍历有限候选集合和其他集合的并集
        for s in finite_candidates + others:
            # 计算其他集合的交集
            other_sets = set(self.args) - {s}
            other = Intersection(*other_sets, evaluate=False)
            completed = True
            # 遍历当前集合 s 的元素
            for x in s:
                try:
                    # 若元素 x 在其他集合的交集中，则产生该元素
                    if x in other:
                        yield x
                except TypeError:
                    # 若无法比较，则标记为未完成
                    completed = False
            # 若当前集合完成了交集检查，则结束迭代
            if completed:
                return

        # 若没有完成交集检查，则抛出异常
        if not completed:
            if not candidates:
                raise TypeError("None of the constituent sets are iterable")
            raise TypeError(
                "The computation had not completed because of the "
                "undecidable set membership is found in every candidates.")

    @staticmethod
    # 将 Intersection 对象转换为基于相等性和逻辑运算符的关系表达式
    def as_relational(self, symbol):
        """Rewrite an Intersection in terms of equalities and logic operators"""
        return And(*[set.as_relational(symbol) for set in self.args])
# 表示一个集合的补集，即一个集合相对于另一个集合的差集
class Complement(Set):
    r"""Represents the set difference or relative complement of a set with
    another set.

    $$A - B = \{x \in A \mid x \notin B\}$$


    Examples
    ========

    >>> from sympy import Complement, FiniteSet
    >>> Complement(FiniteSet(0, 1, 2), FiniteSet(1))
    {0, 2}

    See Also
    =========

    Intersection, Union

    References
    ==========

    .. [1] https://mathworld.wolfram.com/ComplementSet.html
    """

    is_Complement = True  # 表示这是一个 Complement 类的实例

    def __new__(cls, a, b, evaluate=True):
        a, b = map(_sympify, (a, b))  # 将 a 和 b 转换为符号表达式
        if evaluate:
            return Complement.reduce(a, b)  # 如果 evaluate 为 True，则进行化简操作

        return Basic.__new__(cls, a, b)

    @staticmethod
    def reduce(A, B):
        """
        Simplify a :class:`Complement`.

        """
        if B == S.UniversalSet or A.is_subset(B):  # 如果 B 是全集或者 A 是 B 的子集，返回空集
            return S.EmptySet

        if isinstance(B, Union):  # 如果 B 是 Union 类型，则对每个集合求补集的交集
            return Intersection(*(s.complement(A) for s in B.args))

        result = B._complement(A)  # 否则调用 B 的 _complement 方法
        if result is not None:
            return result
        else:
            return Complement(A, B, evaluate=False)

    def _contains(self, other):
        A = self.args[0]  # 取得 Complement 对象的第一个参数 A
        B = self.args[1]  # 取得 Complement 对象的第二个参数 B
        return And(A.contains(other), Not(B.contains(other)))  # 返回 A 包含 other 且 B 不包含 other 的逻辑与结果

    def as_relational(self, symbol):
        """Rewrite a complement in terms of equalities and logic
        operators"""
        A, B = self.args  # 取得 Complement 对象的两个参数 A 和 B

        A_rel = A.as_relational(symbol)  # 使用 symbol 将 A 表达为关系形式
        B_rel = Not(B.as_relational(symbol))  # 使用 symbol 将 B 表达为关系形式的否定

        return And(A_rel, B_rel)  # 返回 A_rel 和 B_rel 的逻辑与结果

    def _kind(self):
        return self.args[0].kind  # 返回 Complement 对象的第一个参数 A 的 kind 属性

    @property
    def is_iterable(self):
        if self.args[0].is_iterable:  # 如果 Complement 对象的第一个参数 A 是可迭代的，则返回 True
            return True

    @property
    def is_finite_set(self):
        A, B = self.args  # 取得 Complement 对象的两个参数 A 和 B
        a_finite = A.is_finite_set  # 检查 A 是否是有限集
        if a_finite is True:
            return True
        elif a_finite is False and B.is_finite_set:  # 如果 A 不是有限集且 B 是有限集，则返回 False
            return False

    def __iter__(self):
        A, B = self.args  # 取得 Complement 对象的两个参数 A 和 B
        for a in A:
            if a not in B:  # 遍历 A 中的元素，如果元素不在 B 中，则 yield 返回该元素
                yield a
            else:
                continue


class EmptySet(Set, metaclass=Singleton):
    """
    Represents the empty set. The empty set is available as a singleton
    as ``S.EmptySet``.

    Examples
    ========

    >>> from sympy import S, Interval
    >>> S.EmptySet
    EmptySet

    >>> Interval(1, 2).intersect(S.EmptySet)
    EmptySet

    See Also
    ========

    UniversalSet

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Empty_set
    """
    is_empty = True  # 表示这是一个 EmptySet 类的实例
    is_finite_set = True  # 表示这是一个有限集
    is_FiniteSet = True  # 表示这是一个 FiniteSet 类的实例

    @property  # type: ignore
    @deprecated(
        """
        The is_EmptySet attribute of Set objects is deprecated.
        Use 's is S.EmptySet" or 's.is_empty' instead.
        """,
        deprecated_since_version="1.5",
        active_deprecations_target="deprecated-is-emptyset",
    )
    def is_EmptySet(self):
        return True  # 返回 True 表示这是一个 EmptySet 对象的实例

    @property
    def _measure(self):
        return 0  # 返回 0，表示 EmptySet 的度量为 0
    # 定义一个方法 `_contains`，用于判断当前对象是否包含另一个对象，始终返回假
    def _contains(self, other):
        return false

    # 定义一个方法 `as_relational`，返回假，未使用 `symbol` 参数
    def as_relational(self, symbol):
        return false

    # 实现特殊方法 `__len__`，返回集合的长度，这里始终返回0
    def __len__(self):
        return 0

    # 实现特殊方法 `__iter__`，返回一个空迭代器，表示集合为空
    def __iter__(self):
        return iter([])

    # 实现方法 `_eval_powerset`，返回包含自身的幂集合
    def _eval_powerset(self):
        return FiniteSet(self)

    # 属性装饰器 `@property` 用于定义 `_boundary` 属性，返回自身
    @property
    def _boundary(self):
        return self

    # 定义方法 `_complement`，返回另一个集合 `other`，表示补集操作
    def _complement(self, other):
        return other

    # 定义方法 `_kind`，返回 `SetKind()` 的实例，表示集合类型
    def _kind(self):
        return SetKind()

    # 定义方法 `_symmetric_difference`，返回另一个集合 `other`，表示对称差集操作
    def _symmetric_difference(self, other):
        return other
    """
    Represents a finite set of Sympy expressions.

    Examples
    ========

    >>> from sympy import FiniteSet, Symbol, Interval, Naturals0
    >>> FiniteSet(1, 2, 3, 4)
    {1, 2, 3, 4}
    >>> 3 in FiniteSet(1, 2, 3, 4)
    True
    >>> FiniteSet(1, (1, 2), Symbol('x'))
    {1, x, (1, 2)}
    >>> FiniteSet(Interval(1, 2), Naturals0, {1, 2})
    FiniteSet({1, 2}, Interval(1, 2), Naturals0)
    >>> members = [1, 2, 3, 4]
    >>> f = FiniteSet(*members)
    >>> f
    {1, 2, 3, 4}
    >>> f - FiniteSet(2)
    {1, 3, 4}
    >>> f + FiniteSet(2, 5)
    {1, 2, 3, 4, 5}

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Finite_set
    """
    # 表示有限集合的类，继承自 Sympy 的 Set 类

    is_FiniteSet = True
    # 表示这个类的实例是有限集合

    is_iterable = True
    # 表示这个类的实例可以迭代

    is_empty = False
    # 表示这个类的实例不是空集合

    is_finite_set = True
    # 表示这个类的实例是有限集合

    def __new__(cls, *args, **kwargs):
        # 定义创建新对象的方法，允许额外的关键字参数 'evaluate'

        evaluate = kwargs.get('evaluate', global_parameters.evaluate)
        # 获取 'evaluate' 参数，默认使用全局参数 global_parameters.evaluate

        if evaluate:
            # 如果 evaluate 是 True，表示需要评估表达式

            args = list(map(sympify, args))
            # 将所有参数转换为 Sympy 表达式

            if len(args) == 0:
                return S.EmptySet
                # 如果参数列表为空，则返回空集合
        else:
            # 如果 evaluate 是 False，表示不需要评估表达式

            args = list(map(sympify, args))
            # 将所有参数转换为 Sympy 表达式

        dargs = {}
        # 创建一个空字典用于存储参数

        for i in reversed(list(ordered(args))):
            # 反向遍历有序的参数列表

            if i.is_Symbol:
                dargs[i] = i
                # 如果参数是符号，则直接存储在字典中
            else:
                try:
                    dargs[i.as_dummy()] = i
                    # 尝试将非符号参数作为虚拟符号存储在字典中
                except TypeError:
                    dargs[i] = i
                    # 处理无法作为虚拟符号的参数，直接存储在字典中

        _args_set = set(dargs.values())
        # 使用字典中的值创建集合

        args = list(ordered(_args_set, Set._infimum_key))
        # 使用有序集合中的值排序参数

        obj = Basic.__new__(cls, *args)
        # 创建一个新的 Basic 类实例

        obj._args_set = _args_set
        # 设置实例属性 _args_set 为创建的有序集合

        return obj
        # 返回创建的对象

    def __iter__(self):
        # 定义迭代器方法，返回对象的迭代器
        return iter(self.args)
        # 返回对象的参数的迭代器
    def _complement(self, other):
        if isinstance(other, Interval):
            # 如果 other 是 Interval 类型
            # 仅对于 S.Reals（实数集合）才进行子区间分割；
            # 需要分割的其它情况将首先通过 Set._complement() 处理。
            nums, syms = [], []
            for m in self.args:
                if m.is_number and m.is_real:
                    nums.append(m)
                elif m.is_real == False:
                    pass  # 丢弃非实数
                else:
                    syms.append(m)  # 各种符号表达式
            if other == S.Reals and nums != []:
                nums.sort()
                intervals = []  # 构建元素之间区间的列表
                intervals += [Interval(S.NegativeInfinity, nums[0], True, True)]
                for a, b in zip(nums[:-1], nums[1:]):
                    intervals.append(Interval(a, b, True, True))  # 两端开放
                intervals.append(Interval(nums[-1], S.Infinity, True, True))
                if syms != []:
                    return Complement(Union(*intervals, evaluate=False),
                            FiniteSet(*syms), evaluate=False)
                else:
                    return Union(*intervals, evaluate=False)
            elif nums == []:  # 不需要或不可能进行分割：
                if syms:
                    return Complement(other, FiniteSet(*syms), evaluate=False)
                else:
                    return other

        elif isinstance(other, FiniteSet):
            unk = []
            for i in self:
                c = sympify(other.contains(i))
                if c is not S.true and c is not S.false:
                    unk.append(i)
            unk = FiniteSet(*unk)
            if unk == self:
                return
            not_true = []
            for i in other:
                c = sympify(self.contains(i))
                if c is not S.true:
                    not_true.append(i)
            return Complement(FiniteSet(*not_true), unk)

        return Set._complement(self, other)

    def _contains(self, other):
        """
        Tests whether an element, other, is in the set.

        Explanation
        ===========

        The actual test is for mathematical equality (as opposed to
        syntactical equality). In the worst case all elements of the
        set must be checked.

        Examples
        ========

        >>> from sympy import FiniteSet
        >>> 1 in FiniteSet(1, 2)
        True
        >>> 5 in FiniteSet(1, 2)
        False

        """
        if other in self._args_set:
            return S.true
        else:
            # evaluate=True 是为了覆盖 evaluate=False 的上下文；
            # 我们需要 Eq 来执行评估
            return Or(*[Eq(e, other, evaluate=True) for e in self.args])

    def _eval_is_subset(self, other):
        return fuzzy_and(other._contains(e) for e in self.args)

    @property


注释已按要求添加到代码块中，详细解释了每行代码的作用和功能。
    # 定义 _boundary 方法，返回对象本身，这是一个占位方法
    def _boundary(self):
        return self

    # 定义 _inf 属性，返回集合中的最小元素
    @property
    def _inf(self):
        return Min(*self)

    # 定义 _sup 属性，返回集合中的最大元素
    @property
    def _sup(self):
        return Max(*self)

    # 定义 measure 方法，返回集合的测度，这里总是返回 0
    def measure(self):
        return 0

    # 定义 _kind 方法，根据集合元素的种类返回集合的种类
    def _kind(self):
        if not self.args:
            return SetKind()  # 若集合为空，则返回空集合种类
        elif all(i.kind == self.args[0].kind for i in self.args):
            return SetKind(self.args[0].kind)  # 若所有元素的种类相同，则返回相同的种类
        else:
            return SetKind(UndefinedKind)  # 否则返回未定义的种类

    # 定义 __len__ 方法，返回集合中元素的数量
    def __len__(self):
        return len(self.args)

    # 定义 as_relational 方法，将有限集合转换为关于相等性和逻辑运算的表达式
    def as_relational(self, symbol):
        """Rewrite a FiniteSet in terms of equalities and logic operators. """
        return Or(*[Eq(symbol, elem) for elem in self])

    # 定义 compare 方法，比较两个对象的哈希值
    def compare(self, other):
        return (hash(self) - hash(other))

    # 定义 _eval_evalf 方法，对集合中的每个元素进行指定精度的数值计算
    def _eval_evalf(self, prec):
        dps = prec_to_dps(prec)
        return FiniteSet(*[elem.evalf(n=dps) for elem in self])

    # 定义 _eval_simplify 方法，简化集合中的每个元素
    def _eval_simplify(self, **kwargs):
        from sympy.simplify import simplify
        return FiniteSet(*[simplify(elem, **kwargs) for elem in self])

    # 定义 _sorted_args 属性，返回集合的排序后的元素
    @property
    def _sorted_args(self):
        return self.args

    # 定义 _eval_powerset 方法，计算集合的幂集
    def _eval_powerset(self):
        return self.func(*[self.func(*s) for s in subsets(self.args)])

    # 定义 _eval_rewrite_as_PowerSet 方法，将有限集合重写为幂集的形式
    def _eval_rewrite_as_PowerSet(self, *args, **kwargs):
        """Rewriting method for a finite set to a power set."""
        from .powerset import PowerSet

        # 检查集合元素数量是否为 2 的幂
        is2pow = lambda n: bool(n and not n & (n - 1))
        if not is2pow(len(self)):
            return None

        # 检查参数是否为有限集合，并且都是有限集合
        fs_test = lambda arg: isinstance(arg, Set) and arg.is_FiniteSet
        if not all(fs_test(arg) for arg in args):
            return None

        # 找到参数中元素最多的集合
        biggest = max(args, key=len)
        # 检查最多元素的集合是否为其子集的每个子集
        for arg in subsets(biggest.args):
            arg_set = FiniteSet(*arg)
            if arg_set not in args:
                return None
        # 返回重写后的幂集对象
        return PowerSet(biggest)

    # 定义 __ge__ 方法，判断集合是否包含另一个集合
    def __ge__(self, other):
        if not isinstance(other, Set):
            raise TypeError("Invalid comparison of set with %s" % func_name(other))
        return other.is_subset(self)

    # 定义 __gt__ 方法，判断集合是否严格包含另一个集合
    def __gt__(self, other):
        if not isinstance(other, Set):
            raise TypeError("Invalid comparison of set with %s" % func_name(other))
        return self.is_proper_superset(other)

    # 定义 __le__ 方法，判断集合是否被另一个集合包含
    def __le__(self, other):
        if not isinstance(other, Set):
            raise TypeError("Invalid comparison of set with %s" % func_name(other))
        return self.is_subset(other)

    # 定义 __lt__ 方法，判断集合是否被另一个集合严格包含
    def __lt__(self, other):
        if not isinstance(other, Set):
            raise TypeError("Invalid comparison of set with %s" % func_name(other))
        return self.is_proper_subset(other)

    # 定义 __eq__ 方法，判断集合是否等于另一个对象
    def __eq__(self, other):
        if isinstance(other, (set, frozenset)):
            return self._args_set == other
        return super().__eq__(other)

    # 定义 __hash__ 方法，设置对象的哈希函数
    __hash__ : Callable[[Basic], Any] = Basic.__hash__
# 将集合转换为有限集合的函数定义
_sympy_converter[set] = lambda x: FiniteSet(*x)
# 将不可变集合转换为有限集合的函数定义
_sympy_converter[frozenset] = lambda x: FiniteSet(*x)

# 表示对称差集的类继承自集合类 Set
class SymmetricDifference(Set):
    """Represents the set of elements which are in either of the
    sets and not in their intersection.

    Examples
    ========

    >>> from sympy import SymmetricDifference, FiniteSet
    >>> SymmetricDifference(FiniteSet(1, 2, 3), FiniteSet(3, 4, 5))
    {1, 2, 4, 5}

    See Also
    ========

    Complement, Union

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Symmetric_difference
    """

    # 设置类属性标记为 SymmetricDifference
    is_SymmetricDifference = True

    # 构造函数，用于创建 SymmetricDifference 对象
    def __new__(cls, a, b, evaluate=True):
        # 如果 evaluate 为 True，则调用 reduce 方法进行化简
        if evaluate:
            return SymmetricDifference.reduce(a, b)

        # 否则调用基类 Basic 的构造函数创建对象
        return Basic.__new__(cls, a, b)

    @staticmethod
    def reduce(A, B):
        # 使用 B 对象的 _symmetric_difference 方法对 A 进行对称差运算
        result = B._symmetric_difference(A)
        # 如果结果不为空，则返回对称差结果
        if result is not None:
            return result
        else:
            # 否则递归调用 SymmetricDifference 类的构造函数
            return SymmetricDifference(A, B, evaluate=False)

    # 将对称差表示为关系操作的方法
    def as_relational(self, symbol):
        # 获取对称差集的两个参数 A 和 B
        A, B = self.args
        # 将 A 和 B 转换为关系表达式
        A_rel = A.as_relational(symbol)
        B_rel = B.as_relational(symbol)
        # 返回 A_rel 和 B_rel 的异或逻辑运算结果
        return Xor(A_rel, B_rel)

    # 属性判断是否可迭代
    @property
    def is_iterable(self):
        # 如果所有参数都是可迭代的集合，则返回 True
        if all(arg.is_iterable for arg in self.args):
            return True

    # 迭代器方法，用于生成对称差集的元素
    def __iter__(self):
        # 获取参数列表 args
        args = self.args
        # 将参数 args 中的所有集合进行轮转并生成并集 union
        union = roundrobin(*(iter(arg) for arg in args))

        # 遍历 union 中的每个元素
        for item in union:
            count = 0
            # 计算当前元素 item 在 args 中的集合中出现的次数
            for s in args:
                if item in s:
                    count += 1

            # 如果出现次数为奇数，则生成该元素
            if count % 2 == 1:
                yield item


class DisjointUnion(Set):
    """ Represents the disjoint union (also known as the external disjoint union)
    of a finite number of sets.

    Examples
    ========

    >>> from sympy import DisjointUnion, FiniteSet, Interval, Union, Symbol
    >>> A = FiniteSet(1, 2, 3)
    >>> B = Interval(0, 5)
    >>> DisjointUnion(A, B)
    DisjointUnion({1, 2, 3}, Interval(0, 5))
    >>> DisjointUnion(A, B).rewrite(Union)
    Union(ProductSet({1, 2, 3}, {0}), ProductSet(Interval(0, 5), {1}))
    >>> C = FiniteSet(Symbol('x'), Symbol('y'), Symbol('z'))
    >>> DisjointUnion(C, C)
    DisjointUnion({x, y, z}, {x, y, z})
    >>> DisjointUnion(C, C).rewrite(Union)
    ProductSet({x, y, z}, {0, 1})

    References
    ==========

    https://en.wikipedia.org/wiki/Disjoint_union
    """

    # 构造函数，用于创建 DisjointUnion 对象
    def __new__(cls, *sets):
        # 初始化一个空列表，用于存储有效的集合对象
        dj_collection = []
        # 遍历输入参数 sets
        for set_i in sets:
            # 如果 set_i 是集合对象，则添加到 dj_collection 中
            if isinstance(set_i, Set):
                dj_collection.append(set_i)
            else:
                # 否则抛出类型错误异常
                raise TypeError("Invalid input: '%s', input args \
                    to DisjointUnion must be Sets" % set_i)
        # 使用基类 Basic 的构造函数创建对象并返回
        obj = Basic.__new__(cls, *dj_collection)
        return obj

    # 返回 DisjointUnion 对象的所有集合
    @property
    def sets(self):
        return self.args

    # 判断 DisjointUnion 对象是否为空集合的属性
    @property
    def is_empty(self):
        # 使用模糊逻辑与运算判断所有子集合是否为空集合
        return fuzzy_and(s.is_empty for s in self.sets)
    @property
    def is_finite_set(self):
        # 判断是否为有限集合，通过对所有子集合应用模糊 AND 操作得出结果
        all_finite = fuzzy_and(s.is_finite_set for s in self.sets)
        # 返回当前集合是否为空或者所有子集合是否都是有限集合的模糊 OR 操作的结果
        return fuzzy_or([self.is_empty, all_finite])

    @property
    def is_iterable(self):
        # 如果当前集合为空集，则不可迭代
        if self.is_empty:
            return False
        # 初始化迭代标志为 True
        iter_flag = True
        # 遍历当前集合的所有子集合
        for set_i in self.sets:
            # 如果子集合不为空，则判断其是否可迭代
            if not set_i.is_empty:
                iter_flag = iter_flag and set_i.is_iterable
        # 返回迭代标志，表示当前集合是否可迭代
        return iter_flag

    def _eval_rewrite_as_Union(self, *sets, **kwargs):
        """
        Rewrites the disjoint union as the union of (``set`` x {``i``})
        where ``set`` is the element in ``sets`` at index = ``i``
        """
        # 初始化空的不相交并集
        dj_union = S.EmptySet
        # 初始化索引
        index = 0
        # 遍历传入的集合
        for set_i in sets:
            # 如果集合是 Set 类型的实例
            if isinstance(set_i, Set):
                # 构建集合 set_i 和单元素集 {index} 的直积
                cross = ProductSet(set_i, FiniteSet(index))
                # 将直积并入不相交并集中
                dj_union = Union(dj_union, cross)
                # 索引递增
                index = index + 1
        # 返回重写后的不相交并集
        return dj_union

    def _contains(self, element):
        """
        ``in`` operator for DisjointUnion

        Examples
        ========

        >>> from sympy import Interval, DisjointUnion
        >>> D = DisjointUnion(Interval(0, 1), Interval(0, 2))
        >>> (0.5, 0) in D
        True
        >>> (0.5, 1) in D
        True
        >>> (1.5, 0) in D
        False
        >>> (1.5, 1) in D
        True

        Passes operation on to constituent sets
        """
        # 如果元素不是二元组或长度不为 2，则返回 False
        if not isinstance(element, Tuple) or len(element) != 2:
            return S.false

        # 如果第二个元素不是整数，则返回 False
        if not element[1].is_Integer:
            return S.false

        # 如果第二个元素超出了集合列表的索引范围，则返回 False
        if element[1] >= len(self.sets) or element[1] < 0:
            return S.false

        # 调用对应索引的集合的 _contains 方法，判断第一个元素是否在其中
        return self.sets[element[1]]._contains(element[0])

    def _kind(self):
        # 如果没有参数，则返回一个 SetKind 实例
        if not self.args:
            return SetKind()
        # 如果所有参数的 kind 都相同，则返回第一个参数的 kind
        elif all(i.kind == self.args[0].kind for i in self.args):
            return self.args[0].kind
        # 否则返回一个包含 UndefinedKind 的 SetKind 实例
        else:
            return SetKind(UndefinedKind)

    def __iter__(self):
        # 如果当前集合可迭代
        if self.is_iterable:
            # 初始化一个空列表用于存放迭代器
            iters = []
            # 遍历当前集合的所有子集合及其索引
            for i, s in enumerate(self.sets):
                # 将子集合 s 和单元素集 {Integer(i)} 的直积迭代器加入列表
                iters.append(iproduct(s, {Integer(i)}))
            # 返回合并了所有子集合的迭代器
            return iter(roundrobin(*iters))
        else:
            # 如果当前集合不可迭代，则抛出 ValueError 异常
            raise ValueError("'%s' is not iterable." % self)
    # 返回不相交并集的长度，即集合中元素的数量。
    def __len__(self):
        """
        Returns the length of the disjoint union, i.e., the number of elements in the set.

        Examples
        ========

        >>> from sympy import FiniteSet, DisjointUnion, EmptySet
        >>> D1 = DisjointUnion(FiniteSet(1, 2, 3, 4), EmptySet, FiniteSet(3, 4, 5))
        >>> len(D1)
        7
        >>> D2 = DisjointUnion(FiniteSet(3, 5, 7), EmptySet, FiniteSet(3, 5, 7))
        >>> len(D2)
        6
        >>> D3 = DisjointUnion(EmptySet, EmptySet)
        >>> len(D3)
        0

        Adds up the lengths of the constituent sets.
        """

        # 如果集合是有限集，则计算所有子集的长度之和。
        if self.is_finite_set:
            size = 0
            for set in self.sets:
                size += len(set)
            return size
        else:
            # 如果集合不是有限集，抛出异常。
            raise ValueError("'%s' is not a finite set." % self)
# 导入必要的模块和类
from .fancysets import ImageSet
from .setexpr import set_function

# 检查参数数量是否足够
if len(args) < 2:
    # 如果参数少于2个，抛出数值错误异常
    raise ValueError('imageset expects at least 2 args, got: %s' % len(args))

# 检查第一个参数是否是符号或元组，并且参数数量大于2
if isinstance(args[0], (Symbol, tuple)) and len(args) > 2:
    # 创建 Lambda 函数作为变换函数 f，接受第一个和第二个参数
    f = Lambda(args[0], args[1])
    # 其余参数作为集合列表
    set_list = args[2:]
else:
    # 直接将第一个参数作为变换函数 f
    f = args[0]
    # 其余参数作为集合列表
    set_list = args[1:]

# 如果 f 是 Lambda 函数，直接通过
if isinstance(f, Lambda):
    pass
# 如果 f 是可调用函数
elif callable(f):
    # 获取函数的参数信息
    nargs = getattr(f, 'nargs', {})
    # 如果 nargs 存在
    if nargs:
        # 如果 nargs 不等于1，暂未实现支持多参数情况
        if len(nargs) != 1:
            raise NotImplementedError(filldedent('''
                This function can take more than 1 arg
                but the potentially complicated set input
                has not been analyzed at this point to
                know its dimensions. TODO
                '''))
        # 获取参数数量 N
        N = nargs.args[0]
        # 根据参数数量 N 设置默认符号 'x' 或符号列表
        if N == 1:
            s = 'x'
        else:
            s = [Symbol('x%i' % i) for i in range(1, N + 1)]
    else:
        # 获取函数的签名参数
        s = inspect.signature(f).parameters

    # 对函数应用虚拟变量，获取函数表达式
    dexpr = _sympify(f(*[Dummy() for i in s]))
    # 根据表达式创建唯一命名的符号变量
    var = tuple(uniquely_named_symbol(Symbol(i), dexpr) for i in s)
    # 创建 Lambda 函数并应用变量
    f = Lambda(var, f(*var))
else:
    # 如果 f 既不是 Lambda 也不是可调用函数，抛出类型错误异常
    raise TypeError(filldedent('''
        expecting lambda, Lambda, or FunctionClass,
        not \'%s\'.''' % func_name(f)))

# 检查集合列表中的每个元素是否为集合
if any(not isinstance(s, Set) for s in set_list):
    # 如果有非集合元素，抛出值错误异常
    name = [func_name(s) for s in set_list]
    raise ValueError('arguments after mapping should be sets, not %s' % name)
    # 检查集合列表是否只有一个集合
    if len(set_list) == 1:
        # 如果只有一个集合，将其赋值给变量 set
        set = set_list[0]
        try:
            # 尝试使用 set_function 对 f 和 set 进行操作
            # 如果参数数量与集合维度不符，会引发 TypeError
            r = set_function(f, set)
            # 如果返回结果为 None，也引发 TypeError
            if r is None:
                raise TypeError
            # 如果 r 为假值，直接返回 r
            if not r:
                return r
        except TypeError:
            # 如果出现 TypeError，创建一个新的 ImageSet 对象
            r = ImageSet(f, set)
        # 如果 r 是 ImageSet 类型的对象
        if isinstance(r, ImageSet):
            # 将 r 的参数赋值给 f 和 set
            f, set = r.args

        # 如果函数的第一个变量是其表达式本身，则返回 set
        if f.variables[0] == f.expr:
            return set

        # 如果 set 是 ImageSet 类型的对象
        if isinstance(set, ImageSet):
            # 检查 set.lamda 变量的变量数是否为 1，同时 f 的变量数也为 1
            if len(set.lamda.variables) == 1 and len(f.variables) == 1:
                # 取出 set.lamda 的变量 x 和 f 的变量 y
                x = set.lamda.variables[0]
                y = f.variables[0]
                # 构造一个新的 Lambda 函数，并返回 imageset 的结果
                return imageset(
                    Lambda(x, f.expr.subs(y, set.lamda.expr)), *set.base_sets)

        # 如果 r 不为空，则返回 r
        if r is not None:
            return r

    # 如果集合列表中不止一个集合或没有集合，则创建一个新的 ImageSet 对象
    return ImageSet(f, *set_list)
def is_function_invertible_in_set(func, setv):
    """
    Checks whether function ``func`` is invertible when the domain is
    restricted to set ``setv``.
    """
    # Functions known to always be invertible:
    # 如果函数是指数函数或对数函数，则总是可逆的
    if func in (exp, log):
        return True
    
    # 创建一个虚拟变量 u
    u = Dummy("u")
    # 计算函数 func 在变量 u 上的导数
    fdiff = func(u).diff(u)
    
    # 如果函数单调递增或单调递减，则认为可逆
    if (fdiff > 0) == True or (fdiff < 0) == True:
        return True
    
    # 对于更多情况的支持尚未实现，暂时返回 None
    # TODO: 支持更多情况
    return None


def simplify_union(args):
    """
    Simplify a :class:`Union` using known rules.

    Explanation
    ===========

    We first start with global rules like 'Merge all FiniteSets'

    Then we iterate through all pairs and ask the constituent sets if they
    can simplify themselves with any other constituent.  This process depends
    on ``union_sets(a, b)`` functions.
    """
    from sympy.sets.handlers.union import union_sets
    
    # ===== Global Rules =====
    # 如果输入参数为空集合，则返回空集合
    if not args:
        return S.EmptySet
    
    # 检查输入参数是否都是集合类型
    for arg in args:
        if not isinstance(arg, Set):
            raise TypeError("Input args to Union must be Sets")
    
    # 合并所有有限集合
    finite_sets = [x for x in args if x.is_FiniteSet]
    if len(finite_sets) > 1:
        # 将所有有限集合中的元素合并成一个新的有限集合
        a = (x for set in finite_sets for x in set)
        finite_set = FiniteSet(*a)
        args = [finite_set] + [x for x in args if not x.is_FiniteSet]
    
    # ===== Pair-wise Rules =====
    # 这里依赖于集合元素内建的规则
    args = set(args)
    new_args = True
    while new_args:
        for s in args:
            new_args = False
            for t in args - {s}:
                # 尝试对集合 s 和 t 进行并集操作
                new_set = union_sets(s, t)
                # 如果并集操作成功，更新集合 args
                # 如果 s 和 t 不能进行并集操作，则返回 None
                if new_set is not None:
                    if not isinstance(new_set, set):
                        new_set = {new_set}
                    new_args = (args - {s, t}).union(new_set)
                    break
            if new_args:
                args = new_args
                break
    
    # 如果最终集合 args 中只剩一个元素，则返回该元素
    if len(args) == 1:
        return args.pop()
    else:
        # 否则返回一个 Union 对象，不进行求值
        return Union(*args, evaluate=False)


def simplify_intersection(args):
    """
    Simplify an intersection using known rules.

    Explanation
    ===========

    We first start with global rules like
    'if any empty sets return empty set' and 'distribute any unions'

    Then we iterate through all pairs and ask the constituent sets if they
    can simplify themselves with any other constituent
    """

    # ===== Global Rules =====
    # 如果输入参数为空集合，则返回全集
    if not args:
        return S.UniversalSet
    
    # 检查输入参数是否都是集合类型
    for arg in args:
        if not isinstance(arg, Set):
            raise TypeError("Input args to Union must be Sets")
    
    # 如果输入参数中包含空集，则返回空集
    if S.EmptySet in args:
        return S.EmptySet
    
    # 处理有限集合的情况
    rv = Intersection._handle_finite_sets(args)
    # 如果 rv 不为空，则直接返回 rv
    if rv is not None:
        return rv

    # 如果任何集合是并集，返回交集的并集
    for s in args:
        # 检查当前集合是否为并集
        if s.is_Union:
            # 从参数中排除当前集合，剩余集合进行交集操作
            other_sets = set(args) - {s}
            if len(other_sets) > 0:
                other = Intersection(*other_sets)
                # 返回当前并集中各元素与其他集合交集后的并集
                return Union(*(Intersection(arg, other) for arg in s.args))
            else:
                # 如果没有其他集合，直接返回当前并集的元素
                return Union(*s.args)

    for s in args:
        # 如果当前集合是补集
        if s.is_Complement:
            # 从参数中移除当前集合，并加入其补集的另一部分进行交集操作
            args.remove(s)
            other_sets = args + [s.args[0]]
            return Complement(Intersection(*other_sets), s.args[1])

    from sympy.sets.handlers.intersection import intersection_sets

    # 此时保证交集中不含有空集、有限集或并集

    # ===== 配对规则 =====
    # 这里依赖于各个集合自身内置的交集规则
    args = set(args)
    new_args = True
    while new_args:
        for s in args:
            new_args = False
            for t in args - {s}:
                new_set = intersection_sets(s, t)
                # 如果 s 无法与 t 进行交集操作，则返回 None；否则返回新的交集集合

                if new_set is not None:
                    new_args = (args - {s, t}).union({new_set})
                    break
            if new_args:
                args = new_args
                break

    # 如果 args 中只剩下一个集合，则直接返回该集合
    if len(args) == 1:
        return args.pop()
    else:
        # 否则返回所有集合的交集
        return Intersection(*args, evaluate=False)
# 处理有限集合操作的函数，根据给定的操作符和参数 x, y，以及交换律标志 commutative
def _handle_finite_sets(op, x, y, commutative):
    # 使用 sift 函数分离参数列表中的 FiniteSet 对象和其他对象
    fs_args, other = sift([x, y], lambda x: isinstance(x, FiniteSet), binary=True)
    
    # 如果有两个 FiniteSet 对象，对它们执行操作 op，并返回结果的并集
    if len(fs_args) == 2:
        return FiniteSet(*[op(i, j) for i in fs_args[0] for j in fs_args[1]])
    # 如果只有一个 FiniteSet 对象，将其他参数依次与该集合中的每个元素执行操作 op，并返回结果的并集
    elif len(fs_args) == 1:
        sets = [_apply_operation(op, other[0], i, commutative) for i in fs_args[0]]
        return Union(*sets)
    # 如果没有 FiniteSet 对象，则返回 None
    else:
        return None


# 应用给定的操作符 op 到参数 x, y 上，根据 commutative 参数决定是否考虑交换律
def _apply_operation(op, x, y, commutative):
    # 导入 ImageSet 类
    from .fancysets import ImageSet
    # 创建一个虚拟符号 'd'
    d = Dummy('d')

    # 调用 _handle_finite_sets 处理有限集合操作，并获取结果
    out = _handle_finite_sets(op, x, y, commutative)
    
    # 如果结果为 None，则直接应用操作 op 到 x, y 上
    if out is None:
        out = op(x, y)
    
    # 如果结果仍为 None 并且 commutative 为 True，则尝试应用操作 op 到 y, x 上
    if out is None and commutative:
        out = op(y, x)
    
    # 如果结果仍为 None，则根据 x, y 的类型来创建 ImageSet 对象，并执行其中的表达式
    if out is None:
        _x, _y = symbols("x y")
        if isinstance(x, Set) and not isinstance(y, Set):
            out = ImageSet(Lambda(d, op(d, y)), x).doit()
        elif not isinstance(x, Set) and isinstance(y, Set):
            out = ImageSet(Lambda(d, op(x, d)), y).doit()
        else:
            out = ImageSet(Lambda((_x, _y), op(_x, _y)), x, y)
    
    # 返回最终计算结果
    return out


# 对两个集合执行加法操作，使用 _set_add 函数处理
def set_add(x, y):
    # 导入 _set_add 函数并调用 _apply_operation 处理加法操作
    from sympy.sets.handlers.add import _set_add
    return _apply_operation(_set_add, x, y, commutative=True)


# 对两个集合执行减法操作，使用 _set_sub 函数处理
def set_sub(x, y):
    # 导入 _set_sub 函数并调用 _apply_operation 处理减法操作
    from sympy.sets.handlers.add import _set_sub
    return _apply_operation(_set_sub, x, y, commutative=False)


# 对两个集合执行乘法操作，使用 _set_mul 函数处理
def set_mul(x, y):
    # 导入 _set_mul 函数并调用 _apply_operation 处理乘法操作
    from sympy.sets.handlers.mul import _set_mul
    return _apply_operation(_set_mul, x, y, commutative=True)


# 对两个集合执行除法操作，使用 _set_div 函数处理
def set_div(x, y):
    # 导入 _set_div 函数并调用 _apply_operation 处理除法操作
    from sympy.sets.handlers.mul import _set_div
    return _apply_operation(_set_div, x, y, commutative=False)


# 对两个集合执行幂运算操作，使用 _set_pow 函数处理
def set_pow(x, y):
    # 导入 _set_pow 函数并调用 _apply_operation 处理幂运算操作
    from sympy.sets.handlers.power import _set_pow
    return _apply_operation(_set_pow, x, y, commutative=False)


# 对集合和函数应用函数操作，使用 _set_function 函数处理
def set_function(f, x):
    # 导入 _set_function 函数并调用，将函数 f 应用到集合 x 上
    from sympy.sets.handlers.functions import _set_function
    return _set_function(f, x)


# SetKind 类，用于所有集合的种类
class SetKind(Kind):
    """
    SetKind is kind for all Sets

    Every instance of Set will have kind ``SetKind`` parametrised by the kind
    of the elements of the ``Set``. The kind of the elements might be
    ``NumberKind``, or ``TupleKind`` or something else. When not all elements
    have the same kind then the kind of the elements will be given as
    ``UndefinedKind``.

    Parameters
    ==========

    element_kind: Kind (optional)
        The kind of the elements of the set. In a well defined set all elements
        will have the same kind. Otherwise the kind should
        :class:`sympy.core.kind.UndefinedKind`. The ``element_kind`` argument is optional but
        should only be omitted in the case of ``EmptySet`` whose kind is simply
        ``SetKind()``

    Examples
    ========

    >>> from sympy import Interval
    >>> Interval(1, 2).kind
    SetKind(NumberKind)
    >>> Interval(1,2).kind.element_kind
    NumberKind

    See Also
    ========

    sympy.core.kind.NumberKind
    sympy.matrices.kind.MatrixKind
    sympy.core.containers.TupleKind
    """
    # 定义一个特殊方法 __new__，用于创建对象实例
    def __new__(cls, element_kind=None):
        # 调用父类的 __new__ 方法创建对象实例
        obj = super().__new__(cls, element_kind)
        # 将 element_kind 属性赋值给创建的对象实例
        obj.element_kind = element_kind
        # 返回创建的对象实例
        return obj

    # 定义一个特殊方法 __repr__，用于返回对象的字符串表示形式
    def __repr__(self):
        # 如果 element_kind 属性为 None，则返回默认字符串 "SetKind()"
        if not self.element_kind:
            return "SetKind()"
        else:
            # 如果 element_kind 属性不为 None，则返回带有 element_kind 值的字符串表示形式
            return "SetKind(%s)" % self.element_kind
```