# `D:\src\scipysrc\sympy\sympy\sets\tests\test_sets.py`

```
# 导入所需的 SymPy 模块中的具体类和函数
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.containers import TupleKind
from sympy.core.function import Lambda
from sympy.core.kind import NumberKind, UndefinedKind
from sympy.core.numbers import (Float, I, Rational, nan, oo, pi, zoo)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.elementary.miscellaneous import (Max, Min, sqrt)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.logic.boolalg import (false, true)
from sympy.matrices.kind import MatrixKind
from sympy.matrices.dense import Matrix
from sympy.polys.rootoftools import rootof
from sympy.sets.contains import Contains
from sympy.sets.fancysets import (ImageSet, Range)
from sympy.sets.sets import (Complement, DisjointUnion, FiniteSet, Intersection, Interval, ProductSet, Set, SymmetricDifference, Union, imageset, SetKind)
from mpmath import mpi

# 导入 SymPy 核心表达式模块中的具体类和函数
from sympy.core.expr import unchanged
from sympy.core.relational import Eq, Ne, Le, Lt, LessThan
from sympy.logic import And, Or, Xor
from sympy.testing.pytest import raises, XFAIL, warns_deprecated_sympy
from sympy.utilities.iterables import cartes

# 从 sympy.abc 模块中导入常见数学符号
from sympy.abc import x, y, z, m, n

# 定义常量 EmptySet 为 S.EmptySet
EmptySet = S.EmptySet

# 定义测试函数 test_imageset
def test_imageset():
    # 定义变量 ints 为 S.Integers
    ints = S.Integers
    # 断言 imageset(x, x - 1, S.Naturals) 返回 S.Naturals0
    assert imageset(x, x - 1, S.Naturals) is S.Naturals0
    # 断言 imageset(x, x + 1, S.Naturals0) 返回 S.Naturals
    assert imageset(x, x + 1, S.Naturals0) is S.Naturals
    # 断言 imageset(x, abs(x), S.Naturals0) 返回 S.Naturals0
    assert imageset(x, abs(x), S.Naturals0) is S.Naturals0
    # 断言 imageset(x, abs(x), S.Naturals) 返回 S.Naturals
    assert imageset(x, abs(x), S.Naturals) is S.Naturals
    # 断言 imageset(x, abs(x), S.Integers) 返回 S.Naturals0
    assert imageset(x, abs(x), S.Integers) is S.Naturals0
    
    # issue 16878a
    r = symbols('r', real=True)
    # 断言 imageset(x, (x, x), S.Reals)._contains((1, r)) 返回 None
    assert imageset(x, (x, x), S.Reals)._contains((1, r)) == None
    # 断言 imageset(x, (x, x), S.Reals)._contains((1, 2)) 返回 False
    assert imageset(x, (x, x), S.Reals)._contains((1, 2)) == False
    # 断言 (r, r) 在 imageset(x, (x, x), S.Reals) 中
    assert (r, r) in imageset(x, (x, x), S.Reals)
    # 断言 1 + I 在 imageset(x, x + I, S.Reals) 中
    assert 1 + I in imageset(x, x + I, S.Reals)
    # 断言 {1} 不在 imageset(x, (x,), S.Reals) 中
    assert {1} not in imageset(x, (x,), S.Reals)
    # 断言 (1, 1) 不在 imageset(x, (x,), S.Reals) 中
    assert (1, 1) not in imageset(x, (x,), S.Reals)
    # 断言 imageset(x, ints) 引发 TypeError 异常
    raises(TypeError, lambda: imageset(x, ints))
    # 断言 imageset(x, y, z, ints) 引发 ValueError 异常
    raises(ValueError, lambda: imageset(x, y, z, ints))
    # 断言 imageset(Lambda(x, cos(x)), y) 引发 ValueError 异常
    raises(ValueError, lambda: imageset(Lambda(x, cos(x)), y))
    # 断言 (1, 2) 在 imageset(Lambda((x, y), (x, y)), ints, ints) 中
    assert (1, 2) in imageset(Lambda((x, y), (x, y)), ints, ints)
    # 断言 imageset(Lambda(x, x), ints, ints) 引发 ValueError 异常
    raises(ValueError, lambda: imageset(Lambda(x, x), ints, ints))
    # 断言 imageset(cos, ints) 等于 ImageSet(Lambda(x, cos(x)), ints)
    assert imageset(cos, ints) == ImageSet(Lambda(x, cos(x)), ints)
    
    # 定义函数 f(x) = cos(x)
    def f(x):
        return cos(x)
    # 断言 imageset(f, ints) 等于 imageset(x, cos(x), ints)
    assert imageset(f, ints) == imageset(x, cos(x), ints)
    
    # 使用 lambda 表达式定义函数 f(x) = cos(x)
    f = lambda x: cos(x)
    # 断言 imageset(f, ints) 等于 ImageSet(Lambda(x, cos(x)), ints)
    assert imageset(f, ints) == ImageSet(Lambda(x, cos(x)), ints)
    
    # 断言 imageset(x, 1, ints) 等于 FiniteSet(1)
    assert imageset(x, 1, ints) == FiniteSet(1)
    # 断言 imageset(x, y, ints) 等于 {y}
    assert imageset(x, y, ints) == {y}
    # 断言 imageset((x, y), (1, z), ints, S.Reals) 等于 {(1, z)}
    assert imageset((x, y), (1, z), ints, S.Reals) == {(1, z)}
    
    # 定义符号变量 clash 为 Symbol('x', integer=true)
    clash = Symbol('x', integer=true)
    # 断言 str(imageset(lambda x: x + clash, Interval(-2, 1)).lamda.expr) 包含 ('x0 + x', 'x + x0')
    assert (str(imageset(lambda x: x + clash, Interval(-2, 1)).lamda.expr)
        in ('x0 + x', 'x + x0'))

    # 定义符号变量 x1, x2
    x1, x2 = symbols("x1, x2")
    # 使用断言验证两个表达式是否相等：
    # 第一个表达式是调用 `imageset` 函数，传入一个函数 lambda(x, y): Add(x, y)，以及两个区间 Interval(1, 2) 和 Interval(2, 3)；
    # 第二个表达式是调用 `ImageSet` 类的构造函数，传入一个 lambda 表达式 Lambda((x1, x2), x1 + x2)，以及两个区间 Interval(1, 2) 和 Interval(2, 3)。
    assert imageset(lambda x, y:
        Add(x, y), Interval(1, 2), Interval(2, 3)).dummy_eq(
        ImageSet(Lambda((x1, x2), x1 + x2),
        Interval(1, 2), Interval(2, 3)))
def test_is_empty():
    # 对于给定的集合类型，验证其非空性
    for s in [S.Naturals, S.Naturals0, S.Integers, S.Rationals, S.Reals,
            S.UniversalSet]:
        # 断言每个集合类型不为空集
        assert s.is_empty is False

    # 断言空集是空集
    assert S.EmptySet.is_empty is True


def test_is_finiteset():
    # 对于给定的集合类型，验证其有限性
    for s in [S.Naturals, S.Naturals0, S.Integers, S.Rationals, S.Reals,
            S.UniversalSet]:
        # 断言每个集合类型不是有限集
        assert s.is_finite_set is False

    # 断言空集是有限集
    assert S.EmptySet.is_finite_set is True

    # 验证具体集合的有限性
    assert FiniteSet(1, 2).is_finite_set is True
    assert Interval(1, 2).is_finite_set is False
    assert Interval(x, y).is_finite_set is None
    assert ProductSet(FiniteSet(1), FiniteSet(2)).is_finite_set is True
    assert ProductSet(FiniteSet(1), Interval(1, 2)).is_finite_set is False
    assert ProductSet(FiniteSet(1), Interval(x, y)).is_finite_set is None
    assert Union(Interval(0, 1), Interval(2, 3)).is_finite_set is False
    assert Union(FiniteSet(1), Interval(2, 3)).is_finite_set is False
    assert Union(FiniteSet(1), FiniteSet(2)).is_finite_set is True
    assert Union(FiniteSet(1), Interval(x, y)).is_finite_set is None
    assert Intersection(Interval(x, y), FiniteSet(1)).is_finite_set is True
    assert Intersection(Interval(x, y), Interval(1, 2)).is_finite_set is None
    assert Intersection(FiniteSet(x), FiniteSet(y)).is_finite_set is True
    assert Complement(FiniteSet(1), Interval(x, y)).is_finite_set is True
    assert Complement(Interval(x, y), FiniteSet(1)).is_finite_set is None
    assert Complement(Interval(1, 2), FiniteSet(x)).is_finite_set is False
    assert DisjointUnion(Interval(-5, 3), FiniteSet(x, y)).is_finite_set is False
    assert DisjointUnion(S.EmptySet, FiniteSet(x, y), S.EmptySet).is_finite_set is True


def test_deprecated_is_EmptySet():
    # 测试对于已弃用的属性 is_EmptySet 的警告
    with warns_deprecated_sympy():
        S.EmptySet.is_EmptySet

    with warns_deprecated_sympy():
        FiniteSet(1).is_EmptySet


def test_interval_arguments():
    # 验证区间的构造参数
    assert Interval(0, oo) == Interval(0, oo, False, True)
    assert Interval(0, oo).right_open is true  # 这里应为 True
    assert Interval(-oo, 0) == Interval(-oo, 0, True, False)
    assert Interval(-oo, 0).left_open is true  # 这里应为 True
    assert Interval(oo, -oo) == S.EmptySet
    assert Interval(oo, oo) == S.EmptySet
    assert Interval(-oo, -oo) == S.EmptySet
    assert Interval(oo, x) == S.EmptySet
    assert Interval(oo, oo) == S.EmptySet
    assert Interval(x, -oo) == S.EmptySet
    assert Interval(x, x) == {x}

    # 验证区间是否为有限集
    assert isinstance(Interval(1, 1), FiniteSet)
    e = Sum(x, (x, 1, 3))
    assert isinstance(Interval(e, e), FiniteSet)

    assert Interval(1, 1) == S.EmptySet  # 这里应为 Interval(1, 1)
    assert Interval(1, 1).measure == 0

    assert Interval(1, 1, False, True) == S.EmptySet
    assert Interval(1, 1, True, False) == S.EmptySet
    assert Interval(1, 1, True, True) == S.EmptySet

    # 验证带符号的区间参数
    assert isinstance(Interval(0, Symbol('a')), Interval)
    assert Interval(Symbol('a', positive=True), 0) == S.EmptySet
    raises(ValueError, lambda: Interval(0, S.ImaginaryUnit))
    # 调用 raises 函数验证 Interval 类在特定条件下是否会引发 ValueError 异常
    raises(ValueError, lambda: Interval(0, Symbol('z', extended_real=False)))
    # 调用 raises 函数验证 Interval 类在特定条件下是否会引发 ValueError 异常
    raises(ValueError, lambda: Interval(x, x + S.ImaginaryUnit))

    # 调用 raises 函数验证 Interval 类在特定条件下是否会引发 NotImplementedError 异常
    raises(NotImplementedError, lambda: Interval(0, 1, And(x, y)))
    # 调用 raises 函数验证 Interval 类在特定条件下是否会引发 NotImplementedError 异常
    raises(NotImplementedError, lambda: Interval(0, 1, False, And(x, y)))
    # 调用 raises 函数验证 Interval 类在特定条件下是否会引发 NotImplementedError 异常
    raises(NotImplementedError, lambda: Interval(0, 1, z, And(x, y)))
# 定义一个测试函数，用于验证区间符号结束点的符号表达式行为
def test_interval_symbolic_end_points():
    # 创建一个实数符号变量 a
    a = Symbol('a', real=True)

    # 断言：联合区间 [0, a] 和 [0, 3] 的上确界应为 Max(a, 3)
    assert Union(Interval(0, a), Interval(0, 3)).sup == Max(a, 3)

    # 断言：联合区间 [a, 0] 和 [-3, 0] 的下确界应为 Min(-3, a)
    assert Union(Interval(a, 0), Interval(-3, 0)).inf == Min(-3, a)

    # 断言：区间 [0, a] 是否包含数值 1，应为 LessThan(1, a)
    assert Interval(0, a).contains(1) == LessThan(1, a)


# 定义一个测试函数，用于验证区间是否为空集的行为
def test_interval_is_empty():
    # 创建符号变量 x, y
    x, y = symbols('x, y')
    # 创建实数符号变量 r，正数符号变量 p，负数符号变量 n，非负数符号变量 nn
    r = Symbol('r', real=True)
    p = Symbol('p', positive=True)
    n = Symbol('n', negative=True)
    nn = Symbol('nn', nonnegative=True)

    # 下面是一系列断言，验证不同类型区间是否为空集
    assert Interval(1, 2).is_empty == False
    assert Interval(3, 3).is_empty == False  # 有限集
    assert Interval(r, r).is_empty == False  # 有限集
    assert Interval(r, r + nn).is_empty == False
    assert Interval(x, x).is_empty == False
    assert Interval(1, oo).is_empty == False
    assert Interval(-oo, oo).is_empty == False
    assert Interval(-oo, 1).is_empty == False
    assert Interval(x, y).is_empty == None
    assert Interval(r, oo).is_empty == False  # 实数区间为有限集
    assert Interval(n, 0).is_empty == False
    assert Interval(n, 0, left_open=True).is_empty == False
    assert Interval(p, 0).is_empty == True  # 空集
    assert Interval(nn, 0).is_empty == None
    assert Interval(n, p).is_empty == False
    assert Interval(0, p, left_open=True).is_empty == False
    assert Interval(0, p, right_open=True).is_empty == False
    assert Interval(0, nn, left_open=True).is_empty == None
    assert Interval(0, nn, right_open=True).is_empty == None


# 定义一个测试函数，用于验证区间的并集操作
def test_union():
    # 断言：区间 [1, 2] 和 [2, 3] 的并集应为 [1, 3]
    assert Union(Interval(1, 2), Interval(2, 3)) == Interval(1, 3)
    assert Union(Interval(1, 2), Interval(2, 3, True)) == Interval(1, 3)
    assert Union(Interval(1, 3), Interval(2, 4)) == Interval(1, 4)
    assert Union(Interval(1, 2), Interval(1, 3)) == Interval(1, 3)
    assert Union(Interval(1, 3), Interval(1, 2)) == Interval(1, 3)
    assert Union(Interval(1, 3, False, True), Interval(1, 2)) == \
        Interval(1, 3, False, True)
    assert Union(Interval(1, 3), Interval(1, 2, False, True)) == Interval(1, 3)
    assert Union(Interval(1, 2, True), Interval(1, 3)) == Interval(1, 3)
    assert Union(Interval(1, 2, True), Interval(1, 3, True)) == \
        Interval(1, 3, True)
    assert Union(Interval(1, 2, True), Interval(1, 3, True, True)) == \
        Interval(1, 3, True, True)
    assert Union(Interval(1, 2, True, True), Interval(1, 3, True)) == \
        Interval(1, 3, True)
    assert Union(Interval(1, 3), Interval(2, 3)) == Interval(1, 3)
    assert Union(Interval(1, 3, False, True), Interval(2, 3)) == \
        Interval(1, 3)
    assert Union(Interval(1, 2, False, True), Interval(2, 3, True)) != \
        Interval(1, 3)
    assert Union(Interval(1, 2), S.EmptySet) == Interval(1, 2)
    assert Union(S.EmptySet) == S.EmptySet

    # 断言：区间 [0, 1] 与 [1.0/n] (n 从 1 到 9) 的并集应为 [0, 1]
    assert Union(Interval(0, 1), *[FiniteSet(1.0/n) for n in range(1, 10)]) == \
        Interval(0, 1)
    
    # issue #18241 的断言
    x = Symbol('x')
    assert Union(Interval(0, 1), FiniteSet(1, x)) == Union(
        Interval(0, 1), FiniteSet(x))
    # 断言：确保 unchanged 函数对 Union、Interval(0, 1)、FiniteSet(2, x) 的调用没有改变
    assert unchanged(Union, Interval(0, 1), FiniteSet(2, x))
    
    # 断言：测试 Interval(1, 2) 和 Interval(2, 3) 的并集是否等于 Interval(1, 2) + Interval(2, 3)
    assert Interval(1, 2).union(Interval(2, 3)) == \
        Interval(1, 2) + Interval(2, 3)
    
    # 断言：测试 Interval(1, 2) 和 Interval(2, 3) 的并集是否等于 Interval(1, 3)
    assert Interval(1, 2).union(Interval(2, 3)) == Interval(1, 3)
    
    # 断言：测试空集合的并集是否等于空集合
    assert Union(Set()) == Set()
    
    # 断言：测试 FiniteSet(1)、FiniteSet(2) 和 FiniteSet(3) 的并集是否等于 FiniteSet(1, 2, 3)
    assert FiniteSet(1) + FiniteSet(2) + FiniteSet(3) == FiniteSet(1, 2, 3)
    
    # 断言：测试 FiniteSet('ham') 和 FiniteSet('eggs') 的并集是否等于 FiniteSet('ham', 'eggs')
    assert FiniteSet('ham') + FiniteSet('eggs') == FiniteSet('ham', 'eggs')
    
    # 断言：测试 FiniteSet(1, 2, 3) 与 S.EmptySet 的并集是否等于 FiniteSet(1, 2, 3)
    assert FiniteSet(1, 2, 3) + S.EmptySet == FiniteSet(1, 2, 3)
    
    # 断言：测试 FiniteSet(1, 2, 3) 和 FiniteSet(2, 3, 4) 的交集是否等于 FiniteSet(2, 3)
    assert FiniteSet(1, 2, 3) & FiniteSet(2, 3, 4) == FiniteSet(2, 3)
    
    # 断言：测试 FiniteSet(1, 2, 3) 和 FiniteSet(2, 3, 4) 的并集是否等于 FiniteSet(1, 2, 3, 4)
    assert FiniteSet(1, 2, 3) | FiniteSet(2, 3, 4) == FiniteSet(1, 2, 3, 4)
    
    # 断言：测试 FiniteSet(1, 2, 3) 与 S.EmptySet 的交集是否等于 S.EmptySet
    assert FiniteSet(1, 2, 3) & S.EmptySet == S.EmptySet
    
    # 断言：测试 FiniteSet(1, 2, 3) 与 S.EmptySet 的并集是否等于 FiniteSet(1, 2, 3)
    assert FiniteSet(1, 2, 3) | S.EmptySet == FiniteSet(1, 2, 3)
    
    # 定义符号变量 x, y, z
    x = Symbol("x")
    y = Symbol("y")
    z = Symbol("z")
    
    # 断言：测试空集合 S.EmptySet 与 FiniteSet(x, FiniteSet(y, z)) 的并集是否等于 FiniteSet(x, FiniteSet(y, z))
    assert S.EmptySet | FiniteSet(x, FiniteSet(y, z)) == \
        FiniteSet(x, FiniteSet(y, z))
    
    # 测试 Interval 和 FiniteSet 的相互作用
    assert Interval(1, 3) + FiniteSet(2) == Interval(1, 3)
    assert Interval(1, 3, True, True) + FiniteSet(3) == \
        Interval(1, 3, True, False)
    X = Interval(1, 3) + FiniteSet(5)
    Y = Interval(1, 2) + FiniteSet(3)
    XandY = X.intersect(Y)
    assert 2 in X and 3 in X and 3 in XandY
    assert XandY.is_subset(X) and XandY.is_subset(Y)
    
    # 断言：测试 Union(1, 2, 3) 是否会引发 TypeError 异常
    raises(TypeError, lambda: Union(1, 2, 3))
    
    # 断言：测试 X 是否可迭代的属性 is_iterable 是否为 False
    assert X.is_iterable is False
    
    # issue 7843 的问题验证
    assert Union(S.EmptySet, FiniteSet(-sqrt(-I), sqrt(-I))) == \
        FiniteSet(-sqrt(-I), sqrt(-I))
    
    # 断言：测试 Union(S.Reals, S.Integers) 的并集是否等于 S.Reals
    assert Union(S.Reals, S.Integers) == S.Reals
def test_union_iter():
    # 创建一个 Union 对象 u，包含多个 Range 对象，不进行求值
    u = Union(Range(3), Range(5), Range(4), evaluate=False)

    # 断言 u 的迭代结果与预期的列表相同
    assert list(u) == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4]


def test_union_is_empty():
    # 断言 Interval(x, y) + FiniteSet(1) 不为空集
    assert (Interval(x, y) + FiniteSet(1)).is_empty == False
    # 断言 Interval(x, y) + Interval(-x, y) 的空集性质为 None
    assert (Interval(x, y) + Interval(-x, y)).is_empty == None


def test_difference():
    # 测试 Interval(1, 3) 减去 Interval(1, 2) 的结果为 Interval(2, 3, True)
    assert Interval(1, 3) - Interval(1, 2) == Interval(2, 3, True)
    # 测试 Interval(1, 3) 减去 Interval(2, 3) 的结果为 Interval(1, 2, False, True)
    assert Interval(1, 3) - Interval(2, 3) == Interval(1, 2, False, True)
    # 测试 Interval(1, 3, True) 减去 Interval(2, 3) 的结果为 Interval(1, 2, True, True)
    assert Interval(1, 3, True) - Interval(2, 3) == Interval(1, 2, True, True)
    # 测试 Interval(1, 3, True) 减去 Interval(2, 3, True) 的结果为 Interval(1, 2, True, False)
    assert Interval(1, 3, True) - Interval(2, 3, True) == Interval(1, 2, True, False)
    # 测试 Interval(0, 2) 减去 FiniteSet(1) 的结果为 Union(Interval(0, 1, False, True), Interval(1, 2, True, False))
    assert Interval(0, 2) - FiniteSet(1) == Union(Interval(0, 1, False, True), Interval(1, 2, True, False))

    # issue #18119
    # 其他特定的减法运算
    assert S.Reals - FiniteSet(I) == S.Reals
    assert S.Reals - FiniteSet(-I, I) == S.Reals
    assert Interval(0, 10) - FiniteSet(-I, I) == Interval(0, 10)
    assert Interval(0, 10) - FiniteSet(1, I) == Union(Interval.Ropen(0, 1), Interval.Lopen(1, 10))
    assert S.Reals - FiniteSet(1, 2 + I, x, y**2) == Complement(Union(Interval.open(-oo, 1), Interval.open(1, oo)), FiniteSet(x, y**2), evaluate=False)

    # 其他 FiniteSet 之间的减法运算
    assert FiniteSet(1, 2, 3) - FiniteSet(2) == FiniteSet(1, 3)
    assert FiniteSet('ham', 'eggs') - FiniteSet('eggs') == FiniteSet('ham')
    assert FiniteSet(1, 2, 3, 4) - Interval(2, 10, True, False) == FiniteSet(1, 2)
    assert FiniteSet(1, 2, 3, 4) - S.EmptySet == FiniteSet(1, 2, 3, 4)
    assert Union(Interval(0, 2), FiniteSet(2, 3, 4)) - Interval(1, 3) == Union(Interval(0, 1, False, True), FiniteSet(4))

    # 断言 -1 不在 S.Reals 减去 S.Naturals 后的结果中
    assert -1 in S.Reals - S.Naturals


def test_Complement():
    # 定义一些集合对象 A, B, C, D
    A = FiniteSet(1, 3, 4)
    B = FiniteSet(3, 4)
    C = Interval(1, 3)
    D = Interval(1, 2)

    # 断言 Complement(A, B) 的可迭代属性为 True
    assert Complement(A, B, evaluate=False).is_iterable is True
    # 断言 Complement(A, C) 的可迭代属性为 True
    assert Complement(A, C, evaluate=False).is_iterable is True
    # 断言 Complement(C, D) 的可迭代属性为 None
    assert Complement(C, D, evaluate=False).is_iterable is None

    # 断言 Complement(A, B) 的结果集合为 FiniteSet(1)
    assert FiniteSet(*Complement(A, B, evaluate=False)) == FiniteSet(1)
    # 断言 Complement(A, C) 的结果集合为 FiniteSet(4)
    assert FiniteSet(*Complement(A, C, evaluate=False)) == FiniteSet(4)
    # 检查 Complement(C, A) 的操作引发 TypeError
    raises(TypeError, lambda: FiniteSet(*Complement(C, A, evaluate=False)))

    # 其他特定的 Complement 运算
    assert Complement(Interval(1, 3), Interval(1, 2)) == Interval(2, 3, True)
    assert Complement(FiniteSet(1, 3, 4), FiniteSet(3, 4)) == FiniteSet(1)
    assert Complement(Union(Interval(0, 2), FiniteSet(2, 3, 4)), Interval(1, 3)) == Union(Interval(0, 1, False, True), FiniteSet(4))

    # 断言 3 不在 Complement(Interval(0, 5), Interval(1, 4)) 的结果中
    assert 3 not in Complement(Interval(0, 5), Interval(1, 4), evaluate=False)
    # 断言 -1 在 Complement(S.Reals, S.Naturals) 的结果中
    assert -1 in Complement(S.Reals, S.Naturals, evaluate=False)
    # 断言 1 不在 Complement(S.Reals, S.Naturals) 的结果中
    assert 1 not in Complement(S.Reals, S.Naturals, evaluate=False)

    # 断言 Complement(S.Integers, S.UniversalSet) 结果为空集
    assert Complement(S.Integers, S.UniversalSet) == EmptySet
    # 断言 S.UniversalSet 相对于 S.Integers 的补集为空集
    assert S.UniversalSet.complement(S.Integers) == EmptySet
    # 断言：实数集 S.Reals 与整数集 S.Integers - {0} 的交集不包含 0
    assert (0 not in S.Reals.intersect(S.Integers - FiniteSet(0)))

    # 断言：空集 S.EmptySet 减去整数集 S.Integers 结果仍然是空集
    assert S.EmptySet - S.Integers == S.EmptySet

    # 断言：整数集 S.Integers 减去 {0} 再减去 {1} 的结果等于整数集减去 {0, 1} 的结果
    assert (S.Integers - FiniteSet(0)) - FiniteSet(1) == S.Integers - FiniteSet(0, 1)

    # 断言：实数集 S.Reals 减去 S.Naturals 和有限集 {pi} 的并集，等于实数集减去 S.Naturals 和 {pi} 的交集
    assert S.Reals - Union(S.Naturals, FiniteSet(pi)) == \
            Intersection(S.Reals - S.Naturals, S.Reals - FiniteSet(pi))

    # issue 12712 的问题断言：有限集 {x, y, 2} 相对于区间 (-10, 10) 的补集，等于有限集 {x, y} 相对于区间 (-10, 10) 的补集
    assert Complement(FiniteSet(x, y, 2), Interval(-10, 10)) == \
            Complement(FiniteSet(x, y), Interval(-10, 10))

    # 定义集合 A 和 B，分别为符号 'a' 到 'c' 和 'd' 到 'f' 的有限集
    A = FiniteSet(*symbols('a:c'))
    B = FiniteSet(*symbols('d:f'))
    # 断言：Complement 函数应用于 A 和 A 的笛卡尔积，相对于 B，结果不变
    assert unchanged(Complement, ProductSet(A, A), B)

    # 定义 A2 为 A 与 A 的笛卡尔积，B3 为 B 与 B 与 B 的笛卡尔积
    A2 = ProductSet(A, A)
    B3 = ProductSet(B, B, B)
    # 断言：A2 减去 B3 的结果等于 A2 自身
    assert A2 - B3 == A2
    # 断言：B3 减去 A2 的结果等于 B3 自身
    assert B3 - A2 == B3
# 定义一个测试函数，用于测试非集合的操作是否引发 TypeError 异常
def test_set_operations_nonsets():
    # 定义一组操作函数，每个函数接受两个参数并执行不同的操作
    ops = [
        lambda a, b: a + b,      # 加法操作
        lambda a, b: a - b,      # 减法操作
        lambda a, b: a * b,      # 乘法操作
        lambda a, b: a / b,      # 除法操作
        lambda a, b: a // b,     # 地板除操作
        lambda a, b: a | b,      # 并集操作
        lambda a, b: a & b,      # 交集操作
        lambda a, b: a ^ b,      # 对称差操作
        # FiniteSet(1) ** 2 产生一个 ProductSet
        #lambda a, b: a ** b,   # 幂操作，暂时注释掉
    ]

    # 创建两个单元素的有限集合
    Sx = FiniteSet(x)
    Sy = FiniteSet(y)

    # 定义一组不同的集合和数字作为测试用例
    sets = [
        {1},                             # 普通集合
        FiniteSet(1),                    # 有限集合
        Interval(1, 2),                  # 区间
        Union(Sx, Interval(1, 2)),       # 联合集
        Intersection(Sx, Sy),            # 交集
        Complement(Sx, Sy),              # 补集
        ProductSet(Sx, Sy),              # 乘积集
        S.EmptySet,                      # 空集
    ]

    # 数字作为测试用例
    nums = [0, 1, 2, S(0), S(1), S(2)]

    # 对于每个集合和数字的组合，以及每个操作函数，验证是否会引发 TypeError 异常
    for si in sets:
        for ni in nums:
            for op in ops:
                raises(TypeError, lambda: op(si, ni))    # 测试 si 作为第一个参数
                raises(TypeError, lambda: op(ni, si))    # 测试 si 作为第二个参数
        raises(TypeError, lambda: si ** object())        # 测试幂操作使用 object 类型
        raises(TypeError, lambda: si ** {1})             # 测试幂操作使用集合 {1}


# 测试补集操作的函数
def test_complement():
    # 验证补集操作的几个基本用例
    assert Complement({1, 2}, {1}) == {2}
    assert Interval(0, 1).complement(S.Reals) == \
        Union(Interval(-oo, 0, True, True), Interval(1, oo, True, True))
    assert Interval(0, 1, True, False).complement(S.Reals) == \
        Union(Interval(-oo, 0, True, False), Interval(1, oo, True, True))
    assert Interval(0, 1, False, True).complement(S.Reals) == \
        Union(Interval(-oo, 0, True, True), Interval(1, oo, False, True))
    assert Interval(0, 1, True, True).complement(S.Reals) == \
        Union(Interval(-oo, 0, True, False), Interval(1, oo, False, True))

    # 验证补集操作在不同集合之间的特殊情况
    assert S.UniversalSet.complement(S.EmptySet) == S.EmptySet
    assert S.UniversalSet.complement(S.Reals) == S.EmptySet
    assert S.UniversalSet.complement(S.UniversalSet) == S.EmptySet
    assert S.EmptySet.complement(S.Reals) == S.Reals

    # 验证补集操作在多个区间的情况下的结果
    assert Union(Interval(0, 1), Interval(2, 3)).complement(S.Reals) == \
        Union(Interval(-oo, 0, True, True), Interval(1, 2, True, True),
              Interval(3, oo, True, True))

    # 验证有限集合的补集操作
    assert FiniteSet(0).complement(S.Reals) ==  \
        Union(Interval(-oo, 0, True, True), Interval(0, oo, True, True))

    # 验证带有变量的有限集合的补集操作
    assert (FiniteSet(5) + Interval(S.NegativeInfinity,
                                    0)).complement(S.Reals) == \
        Interval(0, 5, True, True) + Interval(5, S.Infinity, True, True)

    # 验证包含多个元素的有限集合的补集操作
    assert FiniteSet(1, 2, 3).complement(S.Reals) == \
        Interval(S.NegativeInfinity, 1, True, True) + \
        Interval(1, 2, True, True) + Interval(2, 3, True, True) +\
        Interval(3, S.Infinity, True, True)

    # 验证带有变量的有限集合的补集操作
    assert FiniteSet(x).complement(S.Reals) == Complement(S.Reals, FiniteSet(x))

    # 验证带有变量的有限集合的补集操作
    assert FiniteSet(0, x).complement(S.Reals) == Complement(Interval(-oo, 0, True, True) +
                                                             Interval(0, oo, True, True)
                                                             , FiniteSet(x), evaluate=False)

    # 创建一个表示二维正方形的区间
    square = Interval(0, 1) * Interval(0, 1)
    # 计算正方形的补集，即不属于 S.Reals*S.Reals 的部分
    notsquare = square.complement(S.Reals*S.Reals)

    # 断言：所有给定点 (0, 0), (.5, .5), (1, 0), (1, 1) 都应该在正方形内
    assert all(pt in square for pt in [(0, 0), (.5, .5), (1, 0), (1, 1)])
    
    # 断言：给定点 (0, 0), (.5, .5), (1, 0), (1, 1) 中不存在任何一个在非正方形的补集中
    assert not any(
        pt in notsquare for pt in [(0, 0), (.5, .5), (1, 0), (1, 1)])
    
    # 断言：给定点 (-1, 0), (1.5, .5), (10, 10) 中不存在任何一个在正方形内
    assert not any(pt in square for pt in [(-1, 0), (1.5, .5), (10, 10)])
    
    # 断言：所有给定点 (-1, 0), (1.5, .5), (10, 10) 都应该在非正方形的补集中
    assert all(pt in notsquare for pt in [(-1, 0), (1.5, .5), (10, 10)])
# 定义一个测试函数，用于验证交集运算的正确性
def test_intersect1():
    # 断言对于所有自然数和非负自然数，它们与整数集的交集应该等于自身
    assert all(S.Integers.intersection(i) is i for i in
        (S.Naturals, S.Naturals0))
    # 断言对于所有自然数和非负自然数，它们与整数集的交集应该等于自身
    assert all(i.intersection(S.Integers) is i for i in
        (S.Naturals, S.Naturals0))
    # 定义变量 s 为非负自然数集
    s =  S.Naturals0
    # 断言非负整数集与自然数集的交集应该等于自然数集
    assert S.Naturals.intersection(s) is S.Naturals
    # 断言非负自然数集与自然数集的交集应该等于自然数集
    assert s.intersection(S.Naturals) is S.Naturals
    # 创建一个符号变量 x
    x = Symbol('x')
    # 断言区间[0, 2]与区间[1, 2]的交集应该是区间[1, 2]
    assert Interval(0, 2).intersect(Interval(1, 2)) == Interval(1, 2)
    # 断言区间[0, 2]与区间[1, 2)的交集应该是区间[1, 2)
    assert Interval(0, 2).intersect(Interval(1, 2, True)) == \
        Interval(1, 2, True)
    # 断言区间(0, 2]与区间[1, 2]的交集应该是区间(1, 2)
    assert Interval(0, 2, True).intersect(Interval(1, 2)) == \
        Interval(1, 2, False, False)
    # 断言区间(0, 2)与区间[1, 2]的交集应该是区间(1, 2]
    assert Interval(0, 2, True, True).intersect(Interval(1, 2)) == \
        Interval(1, 2, False, True)
    # 断言区间[0, 2]与集合{[0, 1]并[2, 3]}的交集应该是{[0, 1]并[2, 2]}
    assert Interval(0, 2).intersect(Union(Interval(0, 1), Interval(2, 3))) == \
        Union(Interval(0, 1), Interval(2, 2))

    # 断言集合{1, 2}与集合{1, 2, 3}的交集应该是{1, 2}
    assert FiniteSet(1, 2).intersect(FiniteSet(1, 2, 3)) == FiniteSet(1, 2)
    # 断言集合{1, 2, x}与集合{x}的交集应该是{x}
    assert FiniteSet(1, 2, x).intersect(FiniteSet(x)) == FiniteSet(x)
    # 断言集合{'ham', 'eggs'}与集合{'ham'}的交集应该是{'ham'}
    assert FiniteSet('ham', 'eggs').intersect(FiniteSet('ham')) == \
        FiniteSet('ham')
    # 断言集合{1, 2, 3, 4, 5}与空集的交集应该是空集
    assert FiniteSet(1, 2, 3, 4, 5).intersect(S.EmptySet) == S.EmptySet

    # 断言区间[0, 5]与集合{1, 3}的交集应该是{1, 3}
    assert Interval(0, 5).intersect(FiniteSet(1, 3)) == FiniteSet(1, 3)
    # 断言区间(0, 1)与集合{1}的交集应该是空集
    assert Interval(0, 1, True, True).intersect(FiniteSet(1)) == S.EmptySet

    # 断言{[0, 1]并[2, 3]}与区间[1, 2]的交集应该是{[1, 1]并[2, 2]}
    assert Union(Interval(0, 1), Interval(2, 3)).intersect(Interval(1, 2)) == \
        Union(Interval(1, 1), Interval(2, 2))
    # 断言{[0, 1]并[2, 3]}与区间[0, 2]的交集应该是{[0, 1]并[2, 2]}
    assert Union(Interval(0, 1), Interval(2, 3)).intersect(Interval(0, 2)) == \
        Union(Interval(0, 1), Interval(2, 2))
    # 断言{[0, 1]并[2, 3]}与区间[1, 2)的交集应该是空集
    assert Union(Interval(0, 1), Interval(2, 3)).intersect(Interval(1, 2, True, True)) == \
        S.EmptySet
    # 断言{[0, 5]并{'ham'}}与集合{2, 3, 4, 5, 6}的交集应该是交集{2, 3, 4, 5, 6}和并集{'ham'并[0, 5]}
    assert Union(Interval(0, 5), FiniteSet('ham')).intersect(FiniteSet(2, 3, 4, 5, 6)) == \
        Intersection(FiniteSet(2, 3, 4, 5, 6), Union(FiniteSet('ham'), Interval(0, 5)))
    # 断言交集{1, 2, 3}与区间[2, x]并区间[3, y]应该是交集{3}并区间[2, x]并区间[3, y]
    assert Intersection(FiniteSet(1, 2, 3), Interval(2, x), Interval(3, y)) == \
        Intersection(FiniteSet(3), Interval(2, x), Interval(3, y), evaluate=False)
    # 断言交集{1, 2}与区间[0, 3]并区间[x, y]应该是交集{1, 2}并区间[x, y]
    assert Intersection(FiniteSet(1, 2), Interval(0, 3), Interval(x, y)) == \
        Intersection({1, 2}, Interval(x, y), evaluate=False)
    # 断言交集{1, 2, 4}与区间[0, 3]并区间[x, y]应该是交集{1, 2}并区间[x, y]
    assert Intersection(FiniteSet(1, 2, 4), Interval(0, 3), Interval(x, y)) == \
        Intersection({1, 2}, Interval(x, y), evaluate=False)
    # XXX: 这里真的需要 real=True 吗？
    # https://github.com/sympy/sympy/issues/17532
    # 创建符号变量 m, n，并断言交集{m}与交集{m, n}并区间[m, m+1]应该是交集{m}
    m, n = symbols('m, n', real=True)
    assert Intersection(FiniteSet(m), FiniteSet(m, n), Interval(m, m+1)) == \
        FiniteSet(m)

    # issue 8217
    # 断言交集{x}与交集{y}应该是交集{x}与交集{y}
    assert Intersection(FiniteSet(x), FiniteSet(y)) == \
        Intersection(FiniteSet(x), FiniteSet(y), evaluate=False)
    # 断言交集{x}与实数集应该是交集{实数集, x}
    assert FiniteSet(x).intersect(S.Reals) == \
        Intersection(S.Reals, FiniteSet(x), evaluate=False)

    # 测试交集的别名函数
    # 断言区间[0, 5]与集合{1, 3}的交集应该是{1, 3}
    assert Interval(0, 5).intersection(FiniteSet(1, 3)) == FiniteSet(1, 3)
    # 断言：空集合与区间 (0, 1] 和有限集合 {1} 的交集为空集合
    assert Interval(0, 1, True, True).intersection(FiniteSet(1)) == S.EmptySet
    
    # 断言：区间 (0, 1) 和区间 (2, 3) 的并集与区间 (1, 2) 的交集等于并集 (1, 1) 和 (2, 2)
    assert Union(Interval(0, 1), Interval(2, 3)).intersection(Interval(1, 2)) == \
        Union(Interval(1, 1), Interval(2, 2))
    
    # 选择规范的边界
    a = sqrt(2*sqrt(6) + 5)
    b = sqrt(2) + sqrt(3)
    # 断言：区间 [a, 4] 和区间 [b, 5] 的交集等于区间 [b, 4]
    assert Interval(a, 4).intersection(Interval(b, 5)) == Interval(b, 4)
    # 断言：区间 [1, a] 和区间 [0, b] 的交集等于区间 [1, b]
    assert Interval(1, a).intersection(Interval(0, b)) == Interval(1, b)
def test_intersection_interval_float():
    # intersection of Intervals with mixed Rational/Float boundaries should
    # lead to Float boundaries in all cases regardless of which Interval is
    # open or closed.
    # 定义不同类型的 Interval 元组
    typs = [
        (Interval, Interval, Interval),
        (Interval, Interval.open, Interval.open),
        (Interval, Interval.Lopen, Interval.Lopen),
        (Interval, Interval.Ropen, Interval.Ropen),
        (Interval.open, Interval.open, Interval.open),
        (Interval.open, Interval.Lopen, Interval.open),
        (Interval.open, Interval.Ropen, Interval.open),
        (Interval.Lopen, Interval.Lopen, Interval.Lopen),
        (Interval.Lopen, Interval.Ropen, Interval.open),
        (Interval.Ropen, Interval.Ropen, Interval.Ropen),
    ]

    # Lambda 函数，返回第二个参数（a2）若为 float 类型，否则返回第一个参数（a1）
    as_float = lambda a1, a2: a2 if isinstance(a2, float) else a1

    # 迭代处理每一种 Interval 元组组合
    for t1, t2, t3 in typs:
        for t1i, t2i in [(t1, t2), (t2, t1)]:
            # 迭代处理每一种边界的组合
            for a1, a2, b1, b2 in cartes([2, 2.0], [2, 2.0], [3, 3.0], [3, 3.0]):
                # 创建 Interval 对象 I1 和 I2
                I1 = t1(a1, b1)
                I2 = t2(a2, b2)
                # 创建预期的结果 Interval 对象 I3
                I3 = t3(as_float(a1, a2), as_float(b1, b2))
                # 断言两个 Interval 对象的交集应与预期的结果相等
                assert I1.intersect(I2) == I3


def test_intersection():
    # iterable
    # 创建 Intersection 对象 i，包含 FiniteSet 和 Interval 的交集，禁止求值
    i = Intersection(FiniteSet(1, 2, 3), Interval(2, 5), evaluate=False)
    # 断言 Intersection 对象 i 可迭代
    assert i.is_iterable
    # 断言 Intersection 对象 i 的集合表示应该等于 {2, 3}
    assert set(i) == {S(2), S(3)}

    # challenging intervals
    # 创建符号变量 x
    x = Symbol('x', real=True)
    # 创建 Interval 对象 i，包含 Interval(0, 3) 和 Interval(x, 6) 的交集
    i = Intersection(Interval(0, 3), Interval(x, 6))
    # 断言 5 不在 Intersection 对象 i 中
    assert (5 in i) is False
    # 断言在 Intersection 对象 i 中判断 2 的类型会引发 TypeError 异常
    raises(TypeError, lambda: 2 in i)

    # Singleton special cases
    # 断言空集合与 Interval(0, 1) 的交集应为 EmptySet
    assert Intersection(Interval(0, 1), S.EmptySet) == S.EmptySet
    # 断言 Interval(-oo, oo) 与 Interval(-oo, x) 的交集应为 Interval(-oo, x)
    assert Intersection(Interval(-oo, oo), Interval(-oo, x)) == Interval(-oo, x)

    # Products
    # 创建 Interval 对象 line
    line = Interval(0, 5)
    # 创建 Intersection 对象 i，包含 line 的 2 次方和 3 次方，禁止求值
    i = Intersection(line**2, line**3, evaluate=False)
    # 断言 (2, 2) 不在 Intersection 对象 i 中
    assert (2, 2) not in i
    # 断言 (2, 2, 2) 不在 Intersection 对象 i 中
    assert (2, 2, 2) not in i
    # 断言对 Intersection 对象 i 调用 list 方法会引发 TypeError 异常
    raises(TypeError, lambda: list(i))

    # 创建 Intersection 对象 a，包含 S.Integers 和 S.Naturals 的交集，禁止求值
    a = Intersection(Intersection(S.Integers, S.Naturals, evaluate=False), S.Reals, evaluate=False)
    # 断言 Intersection 对象 a 的参数集合应为 {Intersection(S.Naturals, S.Integers, evaluate=False), S.Reals}
    assert a._argset == frozenset([Intersection(S.Naturals, S.Integers, evaluate=False), S.Reals])

    # 断言 Intersection 对象包含 S.Complexes 和 FiniteSet(S.ComplexInfinity) 的交集应为 EmptySet
    assert Intersection(S.Complexes, FiniteSet(S.ComplexInfinity)) == S.EmptySet

    # issue 12178
    # 断言空 Intersection 对象应为 UniversalSet
    assert Intersection() == S.UniversalSet

    # issue 16987
    # 断言 Intersection 对象包含 {1}, {1}, {x} 的交集应为 {1}, {x}
    assert Intersection({1}, {1}, {x}) == Intersection({1}, {x})


def test_issue_9623():
    # 创建符号变量 n
    n = Symbol('n')

    a = S.Reals
    b = Interval(0, oo)
    c = FiniteSet(n)

    # 断言 Intersection(a, b, c) 应等于 Intersection(b, c)
    assert Intersection(a, b, c) == Intersection(b, c)
    # 断言 Intersection(Interval(1, 2), Interval(3, 4), FiniteSet(n)) 应为空集 EmptySet
    assert Intersection(Interval(1, 2), Interval(3, 4), FiniteSet(n)) == EmptySet


def test_is_disjoint():
    # 断言 Interval(0, 2) 和 Interval(1, 2) 不是不相交的
    assert Interval(0, 2).is_disjoint(Interval(1, 2)) == False
    # 断言 Interval(0, 2) 和 Interval(3, 4) 是不相交的
    assert Interval(0, 2).is_disjoint(Interval(3, 4)) == True


def test_ProductSet__len__():
    A = FiniteSet(1, 2)
    B = FiniteSet(1, 2, 3)
    # 断言 ProductSet(A) 的长度应为 2
    assert ProductSet(A).__len__() == 2
    # 断言 ProductSet(A) 的长度不应为 S(2)
    assert ProductSet(A).__len__() is not S(2)
    # 断言 ProductSet(A, B) 的长度应为 6
    assert ProductSet(A, B).__len__() == 6
    # 断言 ProductSet(A, B) 的长度不应为 S(6)
    assert ProductSet(A, B).__len__() is not S(6)


def test_ProductSet():
    pass  # 这个函数没有具体的代码实现，只是一个占位符
    # ProductSet 总是包含一组元组
    assert ProductSet(S.Reals) == S.Reals ** 1
    # ProductSet 包含两个实数的笛卡尔积
    assert ProductSet(S.Reals, S.Reals) == S.Reals ** 2
    # ProductSet 包含三个实数的笛卡尔积
    assert ProductSet(S.Reals, S.Reals, S.Reals) == S.Reals ** 3

    # ProductSet 不等于单个集合 S.Reals
    assert ProductSet(S.Reals) != S.Reals
    # ProductSet 等于 S.Reals 乘以 S.Reals
    assert ProductSet(S.Reals, S.Reals) == S.Reals * S.Reals
    # ProductSet 不等于 S.Reals 乘以 S.Reals 乘以 S.Reals
    assert ProductSet(S.Reals, S.Reals, S.Reals) != S.Reals * S.Reals * S.Reals
    # ProductSet 等于扁平化后的 S.Reals 乘以 S.Reals 乘以 S.Reals
    assert ProductSet(S.Reals, S.Reals, S.Reals) == (S.Reals * S.Reals * S.Reals).flatten()

    # 1 不在 ProductSet(S.Reals) 中
    assert 1 not in ProductSet(S.Reals)
    # (1,) 在 ProductSet(S.Reals) 中
    assert (1,) in ProductSet(S.Reals)

    # 1 不在 ProductSet(S.Reals, S.Reals) 中
    assert 1 not in ProductSet(S.Reals, S.Reals)
    # (1, 2) 在 ProductSet(S.Reals, S.Reals) 中
    assert (1, 2) in ProductSet(S.Reals, S.Reals)
    # (1, I) 不在 ProductSet(S.Reals, S.Reals) 中
    assert (1, I) not in ProductSet(S.Reals, S.Reals)

    # (1, 2, 3) 在 ProductSet(S.Reals, S.Reals, S.Reals) 中
    assert (1, 2, 3) in ProductSet(S.Reals, S.Reals, S.Reals)
    # (1, 2, 3) 在 S.Reals 的三次方中
    assert (1, 2, 3) in S.Reals ** 3
    # (1, 2, 3) 不在 S.Reals 乘以 S.Reals 乘以 S.Reals 中
    assert (1, 2, 3) not in S.Reals * S.Reals * S.Reals
    # ((1, 2), 3) 在 S.Reals 乘以 S.Reals 乘以 S.Reals 中
    assert ((1, 2), 3) in S.Reals * S.Reals * S.Reals
    # (1, (2, 3)) 不在 S.Reals 乘以 S.Reals 乘以 S.Reals 中
    assert (1, (2, 3)) not in S.Reals * S.Reals * S.Reals
    # (1, (2, 3)) 在 S.Reals 乘以 (S.Reals 乘以 S.Reals) 中
    assert (1, (2, 3)) in S.Reals * (S.Reals * S.Reals)

    # 空 ProductSet 等于只包含空元组的有限集
    assert ProductSet() == FiniteSet(())
    # ProductSet 包含 S.Reals 和 S.EmptySet 的结果是 S.EmptySet
    assert ProductSet(S.Reals, S.EmptySet) == S.EmptySet

    # 查看 GitHub 问题 GH-17458

    for ni in range(5):
        # Rn 是 n 个 S.Reals 的笛卡尔积
        Rn = ProductSet(*(S.Reals,) * ni)
        # (1,) * ni 在 Rn 中
        assert (1,) * ni in Rn
        # 1 不在 Rn 中
        assert 1 not in Rn

    # (S.Reals * S.Reals) * S.Reals 不等于 S.Reals * (S.Reals * S.Reals)
    assert (S.Reals * S.Reals) * S.Reals != S.Reals * (S.Reals * S.Reals)

    # 定义 S1 和 S2 为 S.Reals 和 S.Integers
    S1 = S.Reals
    S2 = S.Integers
    x1 = pi
    x2 = 3
    # x1 在 S1 中
    assert x1 in S1
    # x2 在 S2 中
    assert x2 in S2
    # (x1, x2) 在 S1 乘以 S2 中
    assert (x1, x2) in S1 * S2
    # 定义 S3 为 S1 乘以 S2
    S3 = S1 * S2
    x3 = (x1, x2)
    # x3 在 S3 中
    assert x3 in S3
    # (x3, x3) 在 S3 乘以 S3 中
    assert (x3, x3) in S3 * S3
    # x3 + x3 不在 S3 乘以 S3 中
    assert x3 + x3 not in S3 * S3

    # 抛出 ValueError 异常，因为 S.Reals 的负指数不合法
    raises(ValueError, lambda: S.Reals**-1)
    # 使用 warns_deprecated_sympy() 上下文管理器捕获警告
    with warns_deprecated_sympy():
        # 产生 TypeError 异常，因为 ProductSet 中包含 None
        ProductSet(FiniteSet(s) for s in range(2))
    # 抛出 TypeError 异常，因为 ProductSet 的参数不能为 None
    raises(TypeError, lambda: ProductSet(None))

    # 定义 S1 和 S2 分别为包含元素 1 和 2 的有限集
    S1 = FiniteSet(1, 2)
    S2 = FiniteSet(3, 4)
    # 定义 S3 为 S1 和 S2 的笛卡尔积
    S3 = ProductSet(S1, S2)
    # 检查 S3 的关系表示与 S1 和 S2 的关系表示相等
    assert (S3.as_relational(x, y)
            == And(S1.as_relational(x), S2.as_relational(y))
            == And(Or(Eq(x, 1), Eq(x, 2)), Or(Eq(y, 3), Eq(y, 4))))
    # 抛出 ValueError 异常，因为只能为 S3 提供一个变量来生成关系表示
    raises(ValueError, lambda: S3.as_relational(x))
    # 抛出 ValueError 异常，因为为 S3 提供了多个变量来生成关系表示
    raises(ValueError, lambda: S3.as_relational(x, 1))
    # 抛出 ValueError 异常，因为不能为区间(Interval)创建关系表示
    raises(ValueError, lambda: ProductSet(Interval(0, 1)).as_relational(x, y))

    # 定义 Z2 为 S.Integers 乘以 S.Integers
    Z2 = ProductSet(S.Integers, S.Integers)
    # 检查 (1, 2) 是否在 Z2 中，应返回 S.true
    assert Z2.contains((1, 2)) is S.true
    # 检查 (1,) 是否在 Z2 中，应返回 S.false
    assert Z2.contains((1,)) is S.false
    # 检查 x 是否在 Z2 中的布尔表达式
    assert Z2.contains(x) == Contains(x, Z2, evaluate=False)
    # 检查替换 x 为 1 后，是否在 Z2 中，应返回 S.false
    assert Z2.contains(x).subs(x, 1) is S.false
    # 检查替换 (x, 1) 为 (2, 1) 后，是否在 Z2 中，应返回 S.true
    assert Z2.contains((x, 1)).subs(x, 2) is S.true
    # 检查 (x, y) 是否在 Z2 中的逻辑与表达式
    assert Z2.contains((x, y)) == Contains(x, S.Integers) & Contains(y, S.Integers)
    # 检查函数 unchanged 是否保持 Contains(x, y, Z2) 的一致性
    assert unchanged(Contains, (x, y), Z2)
    # 检查 (1, 2) 是否在 Z2 中
# 测试单个参数的 ProductSet 不变性，期望不变
def test_ProductSet_of_single_arg_is_not_arg():
    assert unchanged(ProductSet, Interval(0, 1))  # 检查 ProductSet 的行为是否不受参数的影响
    assert unchanged(ProductSet, ProductSet(Interval(0, 1)))  # 检查 ProductSet 的行为是否不受自身的影响


# 测试 ProductSet 是否为空集
def test_ProductSet_is_empty():
    assert ProductSet(S.Integers, S.Reals).is_empty == False  # 检查非空 ProductSet
    assert ProductSet(Interval(x, 1), S.Reals).is_empty == None  # 检查无法判断是否为空集的 ProductSet


# 测试 Interval 对象的替换功能
def test_interval_subs():
    a = Symbol('a', real=True)

    assert Interval(0, a).subs(a, 2) == Interval(0, 2)  # 替换 Interval 的上限
    assert Interval(a, 0).subs(a, 2) == S.EmptySet  # 替换后的 Interval 为 EmptySet


# 测试 Interval 对象转换为 mpi 格式
def test_interval_to_mpi():
    assert Interval(0, 1).to_mpi() == mpi(0, 1)  # 检查 Interval 转为 mpi 格式的正确性
    assert Interval(0, 1, True, False).to_mpi() == mpi(0, 1)  # 检查带有边界信息的 Interval 转为 mpi 的正确性
    assert type(Interval(0, 1).to_mpi()) == type(mpi(0, 1))  # 检查转换结果类型的正确性


# 测试集合的数值求解功能
def test_set_evalf():
    assert Interval(S(11)/64, S.Half).evalf() == Interval(
        Float('0.171875'), Float('0.5'))  # 检查 Interval 的 evalf() 结果是否正确
    assert Interval(x, S.Half, right_open=True).evalf() == Interval(
        x, Float('0.5'), right_open=True)  # 检查带有变量的 Interval evalf() 结果
    assert Interval(-oo, S.Half).evalf() == Interval(-oo, Float('0.5'))  # 检查无限区间的 evalf() 结果
    assert FiniteSet(2, x).evalf() == FiniteSet(Float('2.0'), x)  # 检查有限集合的 evalf() 结果


# 测试测量功能
def test_measure():
    a = Symbol('a', real=True)

    assert Interval(1, 3).measure == 2  # 检查 Interval 的测量结果
    assert Interval(0, a).measure == a  # 检查带有变量的 Interval 的测量结果
    assert Interval(1, a).measure == a - 1  # 检查带有变量的 Interval 的测量结果

    assert Union(Interval(1, 2), Interval(3, 4)).measure == 2  # 检查 Union 的测量结果
    assert Union(Interval(1, 2), Interval(3, 4), FiniteSet(5, 6, 7)).measure \
        == 2  # 检查 Union 包含 FiniteSet 的测量结果

    assert FiniteSet(1, 2, oo, a, -oo, -5).measure == 0  # 检查有限集合的测量结果

    assert S.EmptySet.measure == 0  # 检查空集的测量结果

    square = Interval(0, 10) * Interval(0, 10)
    offsetsquare = Interval(5, 15) * Interval(5, 15)
    band = Interval(-oo, oo) * Interval(2, 4)

    assert square.measure == offsetsquare.measure == 100  # 检查正方形和偏移正方形的测量结果
    assert (square + offsetsquare).measure == 175  # 检查正方形和偏移正方形的并集的测量结果
    assert (square - offsetsquare).measure == 75  # 检查正方形和偏移正方形的差集的测量结果
    assert (square * FiniteSet(1, 2, 3)).measure == 0  # 检查正方形和有限集合的交集的测量结果
    assert (square.intersect(band)).measure == 20  # 检查正方形和带状区间的交集的测量结果
    assert (square + band).measure is oo  # 检查正方形和带状区间的并集的测量结果
    assert (band * FiniteSet(1, 2, 3)).measure is nan  # 检查带状区间和有限集合的交集的测量结果


# 测试子集关系判断功能
def test_is_subset():
    assert Interval(0, 1).is_subset(Interval(0, 2)) is True  # 检查 Interval 的子集关系判断
    assert Interval(0, 3).is_subset(Interval(0, 2)) is False  # 检查 Interval 的子集关系判断
    assert Interval(0, 1).is_subset(FiniteSet(0, 1)) is False  # 检查 Interval 和 FiniteSet 的子集关系判断

    assert FiniteSet(1, 2).is_subset(FiniteSet(1, 2, 3, 4))  # 检查 FiniteSet 和更大 FiniteSet 的子集关系
    assert FiniteSet(4, 5).is_subset(FiniteSet(1, 2, 3, 4)) is False  # 检查 FiniteSet 和更大 FiniteSet 的子集关系
    assert FiniteSet(1).is_subset(Interval(0, 2))  # 检查 FiniteSet 和 Interval 的子集关系
    assert FiniteSet(1, 2).is_subset(Interval(0, 2, True, True)) is False  # 检查 FiniteSet 和带边界的 Interval 的子集关系
    assert (Interval(1, 2) + FiniteSet(3)).is_subset(
        Interval(0, 2, False, True) + FiniteSet(2, 3))  # 检查 Interval 和带有 FiniteSet 的子集关系

    assert Interval(3, 4).is_subset(Union(Interval(0, 1), Interval(2, 5))) is True  # 检查 Interval 和 Union 的子集关系
    assert Interval(3, 6).is_subset(Union(Interval(0, 1), Interval(2, 5))) is False  # 检查 Interval 和 Union 的子集关系

    assert FiniteSet(1, 2, 3, 4).is_subset(Interval(0, 5)) is True  # 检查 FiniteSet 和 Interval 的子集关系
    assert S.EmptySet.is_subset(FiniteSet(1, 2, 3)) is True  # 检查空集和 FiniteSet 的子集关系

    assert Interval(0, 1).is_subset(S.EmptySet) is False  # 检查 Interval 和空集的子集关系
    assert S.EmptySet.is_subset(S.EmptySet) is True  # 检查空集和空集的子集关系
    # 调用 raises 函数，期望抛出 ValueError 异常，检查 S.EmptySet 是否是整数 1 的子集
    raises(ValueError, lambda: S.EmptySet.is_subset(1))

    # 测试 issubset 的别名，确保有限集合中的元素是否是区间 [0, 5) 的子集
    assert FiniteSet(1, 2, 3, 4).issubset(Interval(0, 5)) is True
    # 确保空集合 S.EmptySet 是有限集合 {1, 2, 3} 的子集
    assert S.EmptySet.issubset(FiniteSet(1, 2, 3)) is True

    # 确保自然数集 S.Naturals 是整数集 S.Integers 的子集
    assert S.Naturals.is_subset(S.Integers)
    # 确保非负整数集 S.Naturals0 是整数集 S.Integers 的子集
    assert S.Naturals0.is_subset(S.Integers)

    # 确保单元素集合 FiniteSet(x) 是否是单元素集合 FiniteSet(y) 的子集，应返回 None
    assert FiniteSet(x).is_subset(FiniteSet(y)) is None
    # 确保单元素集合 FiniteSet(x) 是否是替换变量 y 后的单元素集合 FiniteSet(x) 的子集，应返回 True
    assert FiniteSet(x).is_subset(FiniteSet(y).subs(y, x)) is True
    # 确保单元素集合 FiniteSet(x) 是否是替换变量 y 后的单元素集合 FiniteSet(x+1) 的子集，应返回 False
    assert FiniteSet(x).is_subset(FiniteSet(y).subs(y, x+1)) is False

    # 确保区间 [0, 1) 不是区间 (0, 1] 的子集，应返回 False
    assert Interval(0, 1).is_subset(Interval(0, 1, left_open=True)) is False
    # 确保区间 [-2, 3] 不是区间 (-∞, -2) ∪ (3, ∞) 的子集，应返回 False
    assert Interval(-2, 3).is_subset(Union(Interval(-oo, -2), Interval(3, oo))) is False

    # 创建整数符号 n
    n = Symbol('n', integer=True)
    # 确保整数范围 [-3, 4) 不是有限集合 {-10, 10} 的子集，应返回 False
    assert Range(-3, 4, 1).is_subset(FiniteSet(-10, 10)) is False
    # 确保整数范围 [10^100] 不是有限集合 {0, 1, 2} 的子集，应返回 False
    assert Range(S(10)**100).is_subset(FiniteSet(0, 1, 2)) is False
    # 确保整数范围 [6, 0, -2) 是有限集合 {2, 4, 6} 的子集，应返回 True
    assert Range(6, 0, -2).is_subset(FiniteSet(2, 4, 6)) is True
    # 确保整数范围 [1, ∞) 不是有限集合 {1, 2} 的子集，应返回 False
    assert Range(1, oo).is_subset(FiniteSet(1, 2)) is False
    # 确保整数范围 (-∞, 1) 不是有限集合 {1} 的子集，应返回 False
    assert Range(-oo, 1).is_subset(FiniteSet(1)) is False
    # 确保整数范围 [0, 1, n] 的子集性质未定，应返回 None
    assert Range(3).is_subset(FiniteSet(0, 1, n)) is None
    # 确保整数范围 [n, n+2) 是有限集合 {n, n+1} 的子集，应返回 True
    assert Range(n, n + 2).is_subset(FiniteSet(n, n + 1)) is True
    # 确保整数范围 [0, 5) 不是区间 [0, 4) 的子集，应返回 False
    assert Range(5).is_subset(Interval(0, 4, right_open=True)) is False
    # 问题 19513：确保 Lambda 函数生成的映射集 imageset(Lambda(n, 1/n), S.Integers) 不是实数集 S.Reals 的子集，应返回 None
    assert imageset(Lambda(n, 1/n), S.Integers).is_subset(S.Reals) is None
def test_is_proper_subset():
    # 检查区间(0, 1)是否是区间(0, 2)的真子集，应返回True
    assert Interval(0, 1).is_proper_subset(Interval(0, 2)) is True
    # 检查区间(0, 3)是否是区间(0, 2)的真子集，应返回False
    assert Interval(0, 3).is_proper_subset(Interval(0, 2)) is False
    # 检查空集是否是有限集{1, 2, 3}的真子集，应返回True
    assert S.EmptySet.is_proper_subset(FiniteSet(1, 2, 3)) is True

    # 检查传递给函数的非法参数，应引发ValueError异常
    raises(ValueError, lambda: Interval(0, 1).is_proper_subset(0))


def test_is_superset():
    # 检查区间(0, 1)是否是区间(0, 2)的超集，应返回False
    assert Interval(0, 1).is_superset(Interval(0, 2)) == False
    # 检查区间(0, 3)是否是区间(0, 2)的超集，应返回True
    assert Interval(0, 3).is_superset(Interval(0, 2))

    # 检查有限集{1, 2}是否是有限集{1, 2, 3, 4}的超集，应返回False
    assert FiniteSet(1, 2).is_superset(FiniteSet(1, 2, 3, 4)) == False
    # 检查有限集{4, 5}是否是有限集{1, 2, 3, 4}的超集，应返回False
    assert FiniteSet(4, 5).is_superset(FiniteSet(1, 2, 3, 4)) == False
    # 检查有限集{1}是否是区间(0, 2)的超集，应返回False
    assert FiniteSet(1).is_superset(Interval(0, 2)) == False
    # 检查有限集{1, 2}是否是区间[0, 2]的超集，应返回False
    assert FiniteSet(1, 2).is_superset(Interval(0, 2, True, True)) == False
    # 检查区间(1, 2)与有限集{3}的并集是否是区间(0, 2)与有限集{2, 3}的并集的超集，应返回False
    assert (Interval(1, 2) + FiniteSet(3)).is_superset(
        Interval(0, 2, False, True) + FiniteSet(2, 3)) == False

    # 检查区间(3, 4)是否是区间(0, 1)和区间(2, 5)的并集的超集，应返回False
    assert Interval(3, 4).is_superset(Union(Interval(0, 1), Interval(2, 5))) == False

    # 检查有限集{1, 2, 3, 4}是否是区间[0, 5]的超集，应返回False
    assert FiniteSet(1, 2, 3, 4).is_superset(Interval(0, 5)) == False
    # 检查空集是否是有限集{1, 2, 3}的超集，应返回False
    assert S.EmptySet.is_superset(FiniteSet(1, 2, 3)) == False

    # 检查区间(0, 1)是否是空集的超集，应返回True
    assert Interval(0, 1).is_superset(S.EmptySet) == True
    # 检查空集是否是空集的超集，应返回True
    assert S.EmptySet.is_superset(S.EmptySet) == True

    # 检查传递给函数的非法参数，应引发ValueError异常
    raises(ValueError, lambda: S.EmptySet.is_superset(1))

    # issuperset别名的测试
    assert Interval(0, 1).issuperset(S.EmptySet) == True
    assert S.EmptySet.issuperset(S.EmptySet) == True


def test_is_proper_superset():
    # 检查区间(0, 1)是否是区间(0, 2)的真超集，应返回False
    assert Interval(0, 1).is_proper_superset(Interval(0, 2)) is False
    # 检查区间(0, 3)是否是区间(0, 2)的真超集，应返回True
    assert Interval(0, 3).is_proper_superset(Interval(0, 2)) is True
    # 检查有限集{1, 2, 3}是否是空集的真超集，应返回True
    assert FiniteSet(1, 2, 3).is_proper_superset(S.EmptySet) is True

    # 检查传递给函数的非法参数，应引发ValueError异常
    raises(ValueError, lambda: Interval(0, 1).is_proper_superset(0))


def test_contains():
    # 检查区间(0, 2)是否包含数值1，应返回S.true
    assert Interval(0, 2).contains(1) is S.true
    # 检查区间(0, 2)是否包含数值3，应返回S.false
    assert Interval(0, 2).contains(3) is S.false
    # 检查区间(0, 2)是否包含数值0（包括开区间0），应返回S.false
    assert Interval(0, 2, True, False).contains(0) is S.false
    # 检查区间(0, 2)是否包含数值2（不包括开区间2），应返回S.true
    assert Interval(0, 2, True, False).contains(2) is S.true
    # 检查区间(0, 2)是否包含数值0（不包括开区间0），应返回S.true
    assert Interval(0, 2, False, True).contains(0) is S.true
    # 检查区间(0, 2)是否包含数值2（包括开区间2），应返回S.false
    assert Interval(0, 2, False, True).contains(2) is S.false
    # 检查区间(0, 2)是否包含数值0（包括开区间0和开区间2），应返回S.false
    assert Interval(0, 2, True, True).contains(0) is S.false
    # 检查区间(0, 2)是否包含数值2（包括开区间0和开区间2），应返回S.false
    assert Interval(0, 2, True, True).contains(2) is S.false

    # 检查区间(0, 2)是否包含区间(0, 2)，应返回False
    assert (Interval(0, 2) in Interval(0, 2)) is False

    # 检查有限集{1, 2, 3}是否包含数值2，应返回S.true
    assert FiniteSet(1, 2, 3).contains(2) is S.true
    # 检查有限集{1, 2, x}是否包含符号x，应返回S.true
    assert FiniteSet(1, 2, Symbol('x')).contains(Symbol('x')) is S.true

    # 检查有限集{y}是否包含符号x，应返回y == x，不进行求值
    assert FiniteSet(y)._contains(x) == Eq(y, x, evaluate=False)
    # 检查传递给函数的非法参数，应引发TypeError异常
    raises(TypeError, lambda: x in FiniteSet(y))
    # 检查有限集{{x, y}}是否包含集合{x}，应返回{x, y} == {x}，不进行求值
    assert FiniteSet({x, y})._contains({x}) == Eq({x, y}, {x}, evaluate=False)
    # 检查有限集{{x, y}}替换y为x后是否包含集合{x}，应返回True
    assert FiniteSet({x, y}).
    # 计算表达式 Pow(Pow(2, 1/3) - 1, 1/3)，即先计算内层次幂，再计算外层次幂
    rad1 = Pow(Pow(2, Rational(1, 3)) - 1, Rational(1, 3))
    # 计算表达式 Pow(1/9, 1/3) - Pow(2/9, 1/3) + Pow(4/9, 1/3)，按顺序计算并相加
    rad2 = Pow(Rational(1, 9), Rational(1, 3)) - Pow(Rational(2, 9), Rational(1, 3)) + Pow(Rational(4, 9), Rational(1, 3))
    # 将 rad1 封装成一个有限集合
    s1 = FiniteSet(rad1)
    # 将 rad2 封装成一个有限集合
    s2 = FiniteSet(rad2)
    # 断言 s1 减去 s2 等于空集 S.EmptySet
    assert s1 - s2 == S.EmptySet

    # 创建包含特定项的有限集合
    items = [1, 2, S.Infinity, S('ham'), -1.1]
    fset = FiniteSet(*items)
    # 断言所有 items 中的项都在 fset 中
    assert all(item in fset for item in items)
    # 断言 fset 对每个 item 的包含状态为 S.true
    assert all(fset.contains(item) is S.true for item in items)

    # 断言 Union(Interval(0, 1), Interval(2, 5)) 包含数字 3
    assert Union(Interval(0, 1), Interval(2, 5)).contains(3) is S.true
    # 断言 Union(Interval(0, 1), Interval(2, 5)) 不包含数字 6
    assert Union(Interval(0, 1), Interval(2, 5)).contains(6) is S.false
    # 断言 Union(Interval(0, 1), FiniteSet(2, 5)) 不包含数字 3
    assert Union(Interval(0, 1), FiniteSet(2, 5)).contains(3) is S.false

    # 断言空集合 S.EmptySet 不包含数字 1
    assert S.EmptySet.contains(1) is S.false
    # 断言包含根号下的方程 x**3 + x - 1 的根的有限集合不包含无穷大 S.Infinity
    assert FiniteSet(rootof(x**3 + x - 1, 0)).contains(S.Infinity) is S.false

    # 断言根号下的方程 x**5 + x**3 + 1 的根索引为 0 的根在实数集合 S.Reals 中
    assert rootof(x**5 + x**3 + 1, 0) in S.Reals
    # 断言根号下的方程 x**5 + x**3 + 1 的根索引为 1 的根不在实数集合 S.Reals 中
    assert not rootof(x**5 + x**3 + 1, 1) in S.Reals

    # 非布尔结果断言，计算 Union(Interval(1, 2), Interval(3, 4)) 是否包含变量 x 的表达式
    assert Union(Interval(1, 2), Interval(3, 4)).contains(x) == \
        Or(And(S.One <= x, x <= 2), And(S(3) <= x, x <= 4))
    # 计算 Intersection(Interval(1, x), Interval(2, 3)) 是否包含变量 y 的表达式
    assert Intersection(Interval(1, x), Interval(2, 3)).contains(y) == \
        And(y <= 3, y <= x, S.One <= y, S(2) <= y)

    # 断言复数集合 S.Complexes 不包含复无穷 S.ComplexInfinity
    assert (S.Complexes).contains(S.ComplexInfinity) == S.false
def test_interval_symbolic():
    # 创建一个符号变量 x
    x = Symbol('x')
    # 创建一个闭区间 [0, 1]
    e = Interval(0, 1)
    # 断言区间 e 是否包含变量 x，返回一个逻辑与表达式
    assert e.contains(x) == And(S.Zero <= x, x <= 1)
    # 断言在区间 e 中判断 x 的成员资格会引发 TypeError 异常
    raises(TypeError, lambda: x in e)
    # 创建一个开区间 (0, 1)
    e = Interval(0, 1, True, True)
    # 断言区间 e 是否包含变量 x，返回一个逻辑与表达式
    assert e.contains(x) == And(S.Zero < x, x < 1)
    # 创建一个非实数符号变量 c
    c = Symbol('c', real=False)
    # 断言区间 [x, x+1] 不包含符号变量 c，返回 False
    assert Interval(x, x + 1).contains(c) == False
    # 创建一个扩展实数符号变量 e
    e = Symbol('e', extended_real=True)
    # 断言区间 (-oo, oo) 是否包含符号变量 e，返回一个逻辑与表达式
    assert Interval(-oo, oo).contains(e) == And(
        S.NegativeInfinity < e, e < S.Infinity)


def test_union_contains():
    # 创建一个符号变量 x
    x = Symbol('x')
    # 创建闭区间 [0, 1]
    i1 = Interval(0, 1)
    # 创建闭区间 [2, 3]
    i2 = Interval(2, 3)
    # 创建 i1 和 i2 的并集
    i3 = Union(i1, i2)
    # 断言 i3 表示为关系表达式关于变量 x 的逻辑或
    assert i3.as_relational(x) == Or(And(S.Zero <= x, x <= 1), And(S(2) <= x, x <= 3))
    # 断言在 i3 中判断 x 的成员资格会引发 TypeError 异常
    raises(TypeError, lambda: x in i3)
    # 获取 i3 是否包含变量 x 的逻辑表达式
    e = i3.contains(x)
    assert e == i3.as_relational(x)
    # 断言在不同的 x 值下，e 的求值
    assert e.subs(x, -0.5) is false
    assert e.subs(x, 0.5) is true
    assert e.subs(x, 1.5) is false
    assert e.subs(x, 2.5) is true
    assert e.subs(x, 3.5) is false

    # 创建一个合集 U 包含多个区间和有限集合
    U = Interval(0, 2, True, True) + Interval(10, oo) + FiniteSet(-1, 2, 5, 6)
    # 断言 U 中不存在特定元素的真假表达式
    assert all(el not in U for el in [0, 4, -oo])
    # 断言 U 中存在特定元素的真假表达式
    assert all(el in U for el in [2, 5, 10])


def test_is_number():
    # 断言闭区间 [0, 1] 的 is_number 属性为 False
    assert Interval(0, 1).is_number is False
    # 断言空集合的 is_number 属性为 False


def test_Interval_is_left_unbounded():
    # 断言区间 [3, 4] 的 is_left_unbounded 属性为 False
    assert Interval(3, 4).is_left_unbounded is False
    # 断言区间 (-oo, 3] 的 is_left_unbounded 属性为 True
    assert Interval(-oo, 3).is_left_unbounded is True
    # 断言区间 (-oo, 3] 的 is_left_unbounded 属性为 True
    assert Interval(Float("-inf"), 3).is_left_unbounded is True


def test_Interval_is_right_unbounded():
    # 断言区间 [3, 4] 的 is_right_unbounded 属性为 False
    assert Interval(3, 4).is_right_unbounded is False
    # 断言区间 [3, oo) 的 is_right_unbounded 属性为 True
    assert Interval(3, oo).is_right_unbounded is True
    # 断言区间 [3, oo) 的 is_right_unbounded 属性为 True
    assert Interval(3, Float("+inf")).is_right_unbounded is True


def test_Interval_as_relational():
    x = Symbol('x')

    # 断言区间 [-1, 2) 的关系表达式为 (x <= 2) ∧ (-1 <= x)
    assert Interval(-1, 2, False, False).as_relational(x) == \
        And(Le(-1, x), Le(x, 2))
    # 断言区间 (-1, 2) 的关系表达式为 (x < 2) ∧ (-1 <= x)
    assert Interval(-1, 2, True, False).as_relational(x) == \
        And(Lt(-1, x), Le(x, 2))
    # 断言区间 [-1, 2) 的关系表达式为 (x <= 2) ∧ (-1 < x)
    assert Interval(-1, 2, False, True).as_relational(x) == \
        And(Le(-1, x), Lt(x, 2))
    # 断言区间 (-1, 2) 的关系表达式为 (x < 2) ∧ (-1 < x)
    assert Interval(-1, 2, True, True).as_relational(x) == \
        And(Lt(-1, x), Lt(x, 2))

    # 断言区间 (-oo, 2] 的关系表达式为 (-oo < x) ∧ (x <= 2)
    assert Interval(-oo, 2, right_open=False).as_relational(x) == And(Lt(-oo, x), Le(x, 2))
    # 断言区间 (-oo, 2) 的关系表达式为 (-oo < x) ∧ (x < 2)
    assert Interval(-oo, 2, right_open=True).as_relational(x) == And(Lt(-oo, x), Lt(x, 2))

    # 断言区间 [-2, oo) 的关系表达式为 (-2 <= x) ∧ (x < oo)
    assert Interval(-2, oo, left_open=False).as_relational(x) == And(Le(-2, x), Lt(x, oo))
    # 断言区间 (-2, oo) 的关系表达式为 (-2 < x) ∧ (x < oo)
    assert Interval(-2, oo, left_open=True).as_relational(x) == And(Lt(-2, x), Lt(x, oo))

    # 断言区间 (-oo, oo) 的关系表达式为 (-oo < x) ∧ (x < oo)
    assert Interval(-oo, oo).as_relational(x) == And(Lt(-oo, x), Lt(x, oo))
    # 创建一个实数符号变量 x
    x = Symbol('x', real=True)
    # 创建一个实数符号变量 y
    y = Symbol('y', real=True)
    # 断言区间 [x, y] 的关系表达式为 x <= y
    assert Interval(x, y).as_relational(x) == (x <= y)
    # 断言区间 [y, x] 的关系表达式为 y <= x
    assert Interval(y, x).as_relational(x) == (y <= x)


def test_Finite_as_relational():
    # 创建一个符号变量 x
    x = Symbol('x')
    # 创建一个有限集合 {1, 2} 的关系表达式为 x == 1 ∨ x == 2
    assert FiniteSet(1, 2).as_relational(x) == Or(Eq(x, 1), Eq(x, 2))
    # 创建一个有限集合 {y, -5} 的关系表达式为 x == y ∨ x == -5
    assert FiniteSet(y, -5).as_relational(x) == Or(Eq(x, y), Eq(x, -5))


def test_Union_as_relational():
    # 创建一个符号变量 x
    x = Symbol('x')
    # 确保将区间 [0, 1] 和集合 {2} 的运算结果转换为逻辑表达式，并验证是否等于逻辑表达式 (0 <= x <= 1) 或 x == 2
    assert (Interval(0, 1) + FiniteSet(2)).as_relational(x) == \
        Or(And(Le(0, x), Le(x, 1)), Eq(x, 2))
    
    # 确保将区间 (0, 1) 和集合 {1} 的运算结果转换为逻辑表达式，并验证是否等于逻辑表达式 (0 < x <= 1)
    assert (Interval(0, 1, True, True) + FiniteSet(1)).as_relational(x) == \
        And(Lt(0, x), Le(x, 1))
    
    # 确保将逻辑表达式 x < 0 或 x > 0 转换为集合形式，然后再转换回逻辑表达式，并验证是否等于逻辑表达式 (-oo < x < oo) 且 x != 0
    assert Or(x < 0, x > 0).as_set().as_relational(x) == \
        And((x > -oo), (x < oo), Ne(x, 0))
    
    # 确保将开区间 [1, 3) 和开区间 (3, 5] 的运算结果转换为逻辑表达式，并验证是否等于逻辑表达式 (x != 3) 且 (1 <= x <= 5)
    assert (Interval.Ropen(1, 3) + Interval.Lopen(3, 5)).as_relational(x) == \
        And(Ne(x, 3), (x >= 1), (x <= 5))
# 定义测试函数，测试 Intersection 对象作为关系表达式时的行为
def test_Intersection_as_relational():
    # 定义符号变量 x
    x = Symbol('x')
    # 断言 Intersection(Interval(0, 1), FiniteSet(2), evaluate=False) 转换为关系表达式时的结果
    assert (Intersection(Interval(0, 1), FiniteSet(2),
            evaluate=False).as_relational(x)
            == And(And(Le(0, x), Le(x, 1)), Eq(x, 2)))


# 定义测试函数，测试 Complement 对象作为关系表达式时的行为
def test_Complement_as_relational():
    # 定义符号变量 x
    x = Symbol('x')
    # 创建 Complement 对象表达式
    expr = Complement(Interval(0, 1), FiniteSet(2), evaluate=False)
    # 断言 Complement 对象转换为关系表达式时的结果
    assert expr.as_relational(x) == \
        And(Le(0, x), Le(x, 1), Ne(x, 2))


# 定义一个预期失败的测试函数
@XFAIL
def test_Complement_as_relational_fail():
    # 定义符号变量 x
    x = Symbol('x')
    # 创建 Complement 对象表达式
    expr = Complement(Interval(0, 1), FiniteSet(2), evaluate=False)
    # XXX 这个例子预期失败，因为 0 <= x 在评估时会变成 x >= 0
    # 断言 Complement 对象转换为关系表达式时的结果
    assert expr.as_relational(x) == \
            (0 <= x) & (x <= 1) & Ne(x, 2)


# 定义测试函数，测试 SymmetricDifference 对象作为关系表达式时的行为
def test_SymmetricDifference_as_relational():
    # 定义符号变量 x
    x = Symbol('x')
    # 创建 SymmetricDifference 对象表达式
    expr = SymmetricDifference(Interval(0, 1), FiniteSet(2), evaluate=False)
    # 断言 SymmetricDifference 对象转换为关系表达式时的结果
    assert expr.as_relational(x) == Xor(Eq(x, 2), Le(0, x) & Le(x, 1))


# 定义测试函数，测试 EmptySet 对象的属性和方法
def test_EmptySet():
    # 断言 S.EmptySet 对象转换为关系表达式时为 S.false
    assert S.EmptySet.as_relational(Symbol('x')) is S.false
    # 断言 S.EmptySet 与 S.UniversalSet 的交集为 S.EmptySet
    assert S.EmptySet.intersect(S.UniversalSet) == S.EmptySet
    # 断言 S.EmptySet 的边界为 S.EmptySet
    assert S.EmptySet.boundary == S.EmptySet


# 定义测试函数，测试 FiniteSet 对象的基本功能
def test_finite_basic():
    # 定义符号变量 x
    x = Symbol('x')
    # 创建 FiniteSet 对象 A 和 B
    A = FiniteSet(1, 2, 3)
    B = FiniteSet(3, 4, 5)
    # 计算 A 和 B 的并集和交集
    AorB = Union(A, B)
    AandB = A.intersect(B)
    # 断言 A 是 AorB 的子集，B 也是 AorB 的子集
    assert A.is_subset(AorB) and B.is_subset(AorB)
    # 断言 AandB 是 A 的子集
    assert AandB.is_subset(A)
    # 断言 AandB 等于 FiniteSet(3)
    assert AandB == FiniteSet(3)

    # 断言 A 的最小值和最大值
    assert A.inf == 1 and A.sup == 3
    # 断言 AorB 的最小值和最大值
    assert AorB.inf == 1 and AorB.sup == 5
    # 断言 FiniteSet(x, 1, 5) 的最大值和最小值
    assert FiniteSet(x, 1, 5).sup == Max(x, 5)
    assert FiniteSet(x, 1, 5).inf == Min(x, 1)

    # issue 7335 的相关断言
    assert FiniteSet(S.EmptySet) != S.EmptySet
    assert FiniteSet(FiniteSet(1, 2, 3)) != FiniteSet(1, 2, 3)
    assert FiniteSet((1, 2, 3)) != FiniteSet(1, 2, 3)

    # 断言 FiniteSet 可以包含不同类型的元素
    assert FiniteSet((1, 2), A, -5, x, 'eggs', x**2)

    # 各种比较运算的断言
    assert (A > B) is False
    assert (A >= B) is False
    assert (A < B) is False
    assert (A <= B) is False
    assert AorB > A and AorB > B
    assert AorB >= A and AorB >= B
    assert A >= A and A <= A
    assert A >= AandB and B >= AandB
    assert A > AandB and B > AandB


# 定义测试函数，测试 CartesianProduct 对象的基本功能
def test_product_basic():
    # 定义硬币的头和尾
    H, T = 'H', 'T'
    # 创建单位线段和骰子的有限集
    unit_line = Interval(0, 1)
    d6 = FiniteSet(1, 2, 3, 4, 5, 6)
    d4 = FiniteSet(1, 2, 3, 4)
    coin = FiniteSet(H, T)

    # 创建单位线段的笛卡尔积
    square = unit_line * unit_line

    # 各种断言测试笛卡尔积的包含关系
    assert (0, 0) in square
    assert 0 not in square
    assert (H, T) in coin ** 2
    assert (.5, .5, .5) in (square * unit_line).flatten()
    assert ((.5, .5), .5) in square * unit_line
    assert (H, 3, 3) in (coin * d6 * d6).flatten()
    assert ((H, 3), 3) in coin * d6 * d6
    HH, TT = sympify(H), sympify(T)
    assert set(coin**2) == {(HH, HH), (HH, TT), (TT, HH), (TT, TT)}

    # 断言 d4*d4 是 d6*d6 的子集
    assert (d4*d4).is_subset(d6*d6)
    # 断言：使用 square 对象的 complement 方法，应该等于 Union 结果
    assert square.complement(Interval(-oo, oo) * Interval(-oo, oo)) == Union(
        # 第一个子集：(-∞, 0) U (1, ∞) × (-∞, ∞)
        (Interval(-oo, 0, True, True) + Interval(1, oo, True, True)) * Interval(-oo, oo),
        # 第二个子集：(-∞, ∞) × ((-∞, 0) U (1, ∞))
        Interval(-oo, oo) * (Interval(-oo, 0, True, True) + Interval(1, oo, True, True))
    )
    
    # 断言：Interval(-5, 5) 的立方是 Interval(-10, 10) 的立方的子集
    assert (Interval(-5, 5)**3).is_subset(Interval(-10, 10)**3)
    
    # 断言：Interval(-10, 10) 的立方不是 Interval(-5, 5) 的立方的子集
    assert not (Interval(-10, 10)**3).is_subset(Interval(-5, 5)**3)
    
    # 断言：Interval(-5, 5) 的平方不是 Interval(-10, 10) 的立方的子集
    assert not (Interval(-5, 5)**2).is_subset(Interval(-10, 10)**3)
    
    # 断言：Interval(.2, .5) × FiniteSet(.5) 是 square 的子集，表示线段在正方形内
    assert (Interval(.2, .5) * FiniteSet(.5)).is_subset(square)
    
    # 断言：coin * coin * coin 的长度为 8
    assert len(coin * coin * coin) == 8
    
    # 断言：空集 S.EmptySet 与自身的乘积的长度为 0
    assert len(S.EmptySet * S.EmptySet) == 0
    
    # 断言：空集 S.EmptySet 与 coin 的乘积的长度为 0
    assert len(S.EmptySet * coin) == 0
    
    # 断言：尝试对 coin 与 Interval(0, 2) 进行乘积运算会引发 TypeError 异常
    raises(TypeError, lambda: len(coin * Interval(0, 2)))
def test_real():
    # 定义实数符号变量 x
    x = Symbol('x', real=True)

    # 定义闭区间 I 和 J
    I = Interval(0, 5)
    J = Interval(10, 20)

    # 定义有限集 A、B、C 和 D，包含不同类型的元素
    A = FiniteSet(1, 2, 30, x, S.Pi)
    B = FiniteSet(-4, 0)
    C = FiniteSet(100)
    D = FiniteSet('Ham', 'Eggs')

    # 断言：所有集合都是实数的子集
    assert all(s.is_subset(S.Reals) for s in [I, J, A, B, C])
    # 断言：集合 D 不是实数的子集
    assert not D.is_subset(S.Reals)

    # 断言：所有可能的两两集合元素相加后仍为实数的子集
    assert all((a + b).is_subset(S.Reals) for a in [I, J, A, B, C] for b in [I, J, A, B, C])
    # 断言：至少存在一个集合与集合 D 的并集不是实数的子集
    assert not any((a + D).is_subset(S.Reals) for a in [I, J, A, B, C, D])

    # 断言：多个集合相加后的并集不是实数的子集
    assert not (I + A + D).is_subset(S.Reals)


def test_supinf():
    # 定义实数符号变量 x 和 y
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)

    # 断言：计算 Interval(0, 1) + FiniteSet(2) 的上确界和下确界
    assert (Interval(0, 1) + FiniteSet(2)).sup == 2
    assert (Interval(0, 1) + FiniteSet(2)).inf == 0

    # 断言：计算 Interval(0, 1) + FiniteSet(x) 的上确界和下确界
    assert (Interval(0, 1) + FiniteSet(x)).sup == Max(1, x)
    assert (Interval(0, 1) + FiniteSet(x)).inf == Min(0, x)

    # 断言：计算 FiniteSet(5, 1, x) 的上确界和下确界
    assert FiniteSet(5, 1, x).sup == Max(5, x)
    assert FiniteSet(5, 1, x).inf == Min(1, x)

    # 断言：计算 FiniteSet(5, 1, x, y) 的上确界和下确界
    assert FiniteSet(5, 1, x, y).sup == Max(5, x, y)
    assert FiniteSet(5, 1, x, y).inf == Min(1, x, y)

    # 断言：计算包含无穷大和负无穷大的集合的上确界和下确界
    assert FiniteSet(5, 1, x, y, S.Infinity, S.NegativeInfinity).sup == S.Infinity
    assert FiniteSet(5, 1, x, y, S.Infinity, S.NegativeInfinity).inf == S.NegativeInfinity

    # 断言：计算包含字符串的集合的上确界
    assert FiniteSet('Ham', 'Eggs').sup == Max('Ham', 'Eggs')


def test_universalset():
    # 定义全集 U 和符号变量 x
    U = S.UniversalSet
    x = Symbol('x')

    # 断言：全集 U 作为一个关系集合包含符号变量 x
    assert U.as_relational(x) is S.true
    # 断言：全集 U 与区间 [2, 4] 的并集仍为全集 U
    assert U.union(Interval(2, 4)) == U

    # 断言：全集 U 与区间 [2, 4] 的交集为区间 [2, 4]
    assert U.intersect(Interval(2, 4)) == Interval(2, 4)
    # 断言：全集 U 的度量为无穷大
    assert U.measure is S.Infinity
    # 断言：全集 U 的边界为空集
    assert U.boundary == S.EmptySet
    # 断言：全集 U 包含整数 0
    assert U.contains(0) is S.true


def test_Union_of_ProductSets_shares():
    # 定义区间 line 和点集 points
    line = Interval(0, 2)
    points = FiniteSet(0, 1, 2)

    # 断言：线段乘积集合的并集与线段乘积集合相等
    assert Union(line * line, line * points) == line * line


def test_Interval_free_symbols():
    # issue 6211
    # 断言：区间 [0, 1] 的自由符号集为空集
    assert Interval(0, 1).free_symbols == set()
    # 定义实数符号变量 x
    x = Symbol('x', real=True)
    # 断言：区间 [0, x] 的自由符号集为 {x}
    assert Interval(0, x).free_symbols == {x}


def test_image_interval():
    # 定义实数符号变量 x
    x = Symbol('x', real=True)
    a = Symbol('a', real=True)

    # 断言：图像集 imageset(x, 2*x, Interval(-2, 1)) 的结果为区间 [-4, 2]
    assert imageset(x, 2*x, Interval(-2, 1)) == Interval(-4, 2)
    # 断言：图像集 imageset(x, 2*x, Interval(-2, 1, True, False)) 的结果为区间 (-4, 2]
    assert imageset(x, 2*x, Interval(-2, 1, True, False)) == Interval(-4, 2, True, False)
    # 断言：图像集 imageset(x, x**2, Interval(-2, 1, True, False)) 的结果为区间 (0, 4]
    assert imageset(x, x**2, Interval(-2, 1, True, False)) == Interval(0, 4, False, True)
    # 断言：图像集 imageset(x, x**2, Interval(-2, 1)) 的结果为区间 [0, 4]
    assert imageset(x, x**2, Interval(-2, 1)) == Interval(0, 4)
    # 断言：图像集 imageset(x, x**2, Interval(-2, 1, True, False)) 的结果为区间 (0, 4]
    assert imageset(x, x**2, Interval(-2, 1, True, False)) == Interval(0, 4, False, True)
    # 断言：图像集 imageset(x, x**2, Interval(-2, 1, True, True)) 的结果为区间 (0, 4]
    assert imageset(x, x**2, Interval(-2, 1, True, True)) == Interval(0, 4, False, True)
    # 断言：图像集 imageset(x, (x - 2)**2, Interval(1, 3)) 的结果为区间 [0, 1]
    assert imageset(x, (x - 2)**2, Interval(1, 3)) == Interval(0, 1)
    # 断言：图像集 imageset(x, 3*x**4 - 26*x**3 + 78*x**2 - 90*x, Interval(0, 4)) 的结果为区间 [-35, 0]
    assert imageset(x, 3*x**4 - 26*x**3 + 78*x**2 - 90*x, Interval(0, 4)) == Interval(-35, 0)
    # 断言：图像集 imageset(x, x + 1/x, Interval(-oo, oo)) 的结果为区间 (-oo, -2) ∪ (2, oo)
    assert imageset(x, x + 1/x, Interval(-oo, oo)) == Interval(-oo, -2) + Interval(2, oo)
    # 断言测试函数 `imageset` 对于给定函数在指定区间的映像集合是否符合预期结果
    assert imageset(x, 1/x + 1/(x-1)**2, Interval(0, 2, True, False)) == \
        Interval(Rational(3, 2), oo, False)  # 多重无穷不连续点
    
    # 测试 Python lambda 表达式的使用
    assert imageset(lambda x: 2*x, Interval(-2, 1)) == Interval(-4, 2)
    
    # 断言测试函数 `imageset` 对于给定 lambda 函数在指定区间的映像集合是否符合预期结果
    assert imageset(Lambda(x, a*x), Interval(0, 1)) == \
            ImageSet(Lambda(x, a*x), Interval(0, 1))
    
    # 断言测试函数 `imageset` 对于给定 lambda 函数在指定区间的映像集合是否符合预期结果
    assert imageset(Lambda(x, sin(cos(x))), Interval(0, 1)) == \
            ImageSet(Lambda(x, sin(cos(x))), Interval(0, 1))
# 定义测试函数 test_image_piecewise，用于测试 Piecewise 对象的图像集合
def test_image_piecewise():
    # 定义 Piecewise 函数 f，根据不同条件返回不同的表达式
    f = Piecewise((x, x <= -1), (1/x**2, x <= 5), (x**3, True))
    # 定义 Piecewise 函数 f1，根据不同条件返回不同的表达式
    f1 = Piecewise((0, x <= 1), (1, x <= 2), (2, True))
    # 断言计算 imageset(x, f, Interval(-5, 5)) 的结果是否为 Union(Interval(-5, -1), Interval(Rational(1, 25), oo))
    assert imageset(x, f, Interval(-5, 5)) == Union(Interval(-5, -1), Interval(Rational(1, 25), oo))
    # 断言计算 imageset(x, f1, Interval(1, 2)) 的结果是否为 FiniteSet(0, 1)
    assert imageset(x, f1, Interval(1, 2)) == FiniteSet(0, 1)


# 标记为 XFAIL 的测试函数，链接提供了进一步的讨论
@XFAIL  # See: https://github.com/sympy/sympy/pull/2723#discussion_r8659826
# 定义测试函数 test_image_Intersection，测试 Intersection 的图像集合计算
def test_image_Intersection():
    # 定义实数符号 x 和 y
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    # 断言计算 imageset(x, x**2, Interval(-2, 0).intersect(Interval(x, y))) 的结果是否为
    # Interval(0, 4).intersect(Interval(Min(x**2, y**2), Max(x**2, y**2)))
    assert imageset(x, x**2, Interval(-2, 0).intersect(Interval(x, y))) == \
           Interval(0, 4).intersect(Interval(Min(x**2, y**2), Max(x**2, y**2)))


# 定义测试函数 test_image_FiniteSet，测试有限集的图像集合计算
def test_image_FiniteSet():
    # 定义实数符号 x
    x = Symbol('x', real=True)
    # 断言计算 imageset(x, 2*x, FiniteSet(1, 2, 3)) 的结果是否为 FiniteSet(2, 4, 6)
    assert imageset(x, 2*x, FiniteSet(1, 2, 3)) == FiniteSet(2, 4, 6)


# 定义测试函数 test_image_Union，测试并集的图像集合计算
def test_image_Union():
    # 定义实数符号 x
    x = Symbol('x', real=True)
    # 断言计算 imageset(x, x**2, Interval(-2, 0) + FiniteSet(1, 2, 3)) 的结果是否为
    # (Interval(0, 4) + FiniteSet(9))
    assert imageset(x, x**2, Interval(-2, 0) + FiniteSet(1, 2, 3)) == \
            (Interval(0, 4) + FiniteSet(9))


# 定义测试函数 test_image_EmptySet，测试空集的图像集合计算
def test_image_EmptySet():
    # 定义实数符号 x
    x = Symbol('x', real=True)
    # 断言计算 imageset(x, 2*x, S.EmptySet) 的结果是否为 S.EmptySet
    assert imageset(x, 2*x, S.EmptySet) == S.EmptySet


# 定义测试函数 test_issue_5724_7680，测试问题 5724 和 7680 的相关问题
def test_issue_5724_7680():
    # 断言 I 不在实数集合 S.Reals 中，这是 issue 7680
    assert I not in S.Reals
    # 断言 Interval(-oo, oo).contains(I) 的结果为 S.false
    assert Interval(-oo, oo).contains(I) is S.false


# 定义测试函数 test_boundary，测试有界性质的计算
def test_boundary():
    # 断言 FiniteSet(1).boundary 的结果为 FiniteSet(1)
    assert FiniteSet(1).boundary == FiniteSet(1)
    # 使用列表推导式测试 Interval(0, 1, left_open, right_open).boundary 的计算结果
    assert all(Interval(0, 1, left_open, right_open).boundary == FiniteSet(0, 1)
            for left_open in (true, false) for right_open in (true, false))


# 定义测试函数 test_boundary_Union，测试并集边界的计算
def test_boundary_Union():
    # 断言 (Interval(0, 1) + Interval(2, 3)).boundary 的结果为 FiniteSet(0, 1, 2, 3)
    assert (Interval(0, 1) + Interval(2, 3)).boundary == FiniteSet(0, 1, 2, 3)
    # 断言 (Interval(0, 1, False, True) + Interval(1, 2, True, False)).boundary 的结果为 FiniteSet(0, 1, 2)
    assert ((Interval(0, 1, False, True)
           + Interval(1, 2, True, False)).boundary == FiniteSet(0, 1, 2))

    # 断言 (Interval(0, 1) + FiniteSet(2)).boundary 的结果为 FiniteSet(0, 1, 2)
    assert (Interval(0, 1) + FiniteSet(2)).boundary == FiniteSet(0, 1, 2)
    # 断言 Union(Interval(0, 10), Interval(5, 15), evaluate=False).boundary 的结果为 FiniteSet(0, 15)
    assert Union(Interval(0, 10), Interval(5, 15), evaluate=False).boundary \
            == FiniteSet(0, 15)

    # 断言 Union(Interval(0, 10), Interval(0, 1), evaluate=False).boundary 的结果为 FiniteSet(0, 10)
    assert Union(Interval(0, 10), Interval(0, 1), evaluate=False).boundary \
            == FiniteSet(0, 10)
    # 断言 Union(Interval(0, 10, True, True), Interval(10, 15, True, True), evaluate=False).boundary 的结果为 FiniteSet(0, 10, 15)
    assert Union(Interval(0, 10, True, True),
                 Interval(10, 15, True, True), evaluate=False).boundary \
            == FiniteSet(0, 10, 15)


# 标记为 XFAIL 的测试函数，注释指出了测试困难的问题
@XFAIL
# 定义测试函数 test_union_boundary_of_joining_sets，测试联合集合的边界计算
def test_union_boundary_of_joining_sets():
    # 断言 Union(Interval(0, 10), Interval(10, 15), evaluate=False).boundary 的结果为 FiniteSet(0, 15)
    assert Union(Interval(0, 10), Interval(10, 15), evaluate=False).boundary \
            == FiniteSet(0, 15)


# 定义测试函数 test_boundary_ProductSet，测试直积集的边界计算
def test_boundary_ProductSet():
    # 定义 open_square 为 Interval(0, 1, True, True) 的二维直积
    open_square = Interval(0, 1, True, True) ** 2
    # 断言 open_square.boundary 的结果为 (FiniteSet(0, 1) * Interval(0, 1) + Interval(0, 1) * FiniteSet(0, 1))
    assert open_square.boundary == (FiniteSet(0, 1) * Interval(0, 1)
                                  + Interval(0, 1) * FiniteSet(0, 1))

    # 定义 second_square 为 Interval(1, 2, True, True) * Interval(0, 1, True, True) 的直积
    second_square = Interval(1, 2, True, True) * Interval(0, 1, True, True)
    # 断言 (open_square + second_square).boundary 的结果为 (FiniteSet(0, 1) * Interval(0, 1) + FiniteSet(1, 2) * Interval(0, 1) + Interval(0, 1) * FiniteSet(0, 1) + Interval(1, 2) * FiniteSet(0, 1))
    assert (open
# 定义一个测试函数，用于验证 Interval 对象的 is_open 属性
def test_is_open():
    # 断言指定的半开半闭区间对象 is_open 属性为 False
    assert Interval(0, 1, False, False).is_open is False
    # 断言指定的左开右闭区间对象 is_open 属性为 False
    assert Interval(0, 1, True, False).is_open is False
    # 断言指定的左右开区间对象 is_open 属性为 True
    assert Interval(0, 1, True, True).is_open is True
    # 断言指定的有限集合对象 is_open 属性为 False
    assert FiniteSet(1, 2, 3).is_open is False


# 定义一个测试函数，用于验证 Interval 对象的 is_closed 属性
def test_is_closed():
    # 断言指定的闭区间对象 is_closed 属性为 True
    assert Interval(0, 1, False, False).is_closed is True
    # 断言指定的左开右闭区间对象 is_closed 属性为 False
    assert Interval(0, 1, True, False).is_closed is False
    # 断言指定的有限集合对象 is_closed 属性为 True
    assert FiniteSet(1, 2, 3).is_closed is True


# 定义一个测试函数，用于验证 Interval 对象的 closure 方法
def test_closure():
    # 断言半开半闭区间的闭包等于左闭右闭区间对象
    assert Interval(0, 1, False, True).closure == Interval(0, 1, False, False)


# 定义一个测试函数，用于验证 Interval 对象的 interior 方法
def test_interior():
    # 断言半开半闭区间的内部等于左开右开区间对象
    assert Interval(0, 1, False, True).interior == Interval(0, 1, True, True)


# 定义一个测试函数，用于验证在 S.Reals 中引发 TypeError 异常
def test_issue_7841():
    # 断言 lambda 表达式引发 TypeError 异常，验证 x 是否在 S.Reals 中
    raises(TypeError, lambda: x in S.Reals)


# 定义一个测试函数，用于验证 Eq 对象的各种情况
def test_Eq():
    # 断言两个相同的区间对象相等
    assert Eq(Interval(0, 1), Interval(0, 1))
    # 断言两个不同的区间对象不相等
    assert Eq(Interval(0, 1), Interval(0, 2)) == False

    # 定义两个有限集合对象 s1 和 s2
    s1 = FiniteSet(0, 1)
    s2 = FiniteSet(1, 2)

    # 断言两个相同的有限集合对象相等
    assert Eq(s1, s1)
    # 断言两个不同的有限集合对象不相等
    assert Eq(s1, s2) == False

    # 断言两个乘积集合对象相等
    assert Eq(s1*s2, s1*s2)
    # 断言两个不同的乘积集合对象不相等
    assert Eq(s1*s2, s2*s1) == False

    # 断言未改变的 Eq 对象与替换 y 为 x 后的有限集合对象相等
    assert unchanged(Eq, FiniteSet({x, y}), FiniteSet({x}))
    assert Eq(FiniteSet({x, y}).subs(y, x), FiniteSet({x})) is S.true
    assert Eq(FiniteSet({x, y}), FiniteSet({x})).subs(y, x) is S.true
    assert Eq(FiniteSet({x, y}).subs(y, x+1), FiniteSet({x})) is S.false
    assert Eq(FiniteSet({x, y}), FiniteSet({x})).subs(y, x+1) is S.false

    # 断言两个不同类型的集合对象不相等
    assert Eq(ProductSet({1}, {2}), Interval(1, 2)) is S.false
    assert Eq(ProductSet({1}), ProductSet({1}, {2})) is S.false

    # 断言两个不同的有限集合对象不相等
    assert Eq(FiniteSet(()), FiniteSet(1)) is S.false
    assert Eq(ProductSet(), FiniteSet(1)) is S.false

    # 断言未改变的 Eq 对象与替换 x, y 为区间对象后的乘积集合对象相等
    i1 = Interval(0, 1)
    i2 = Interval(x, y)
    assert unchanged(Eq, ProductSet(i1, i1), ProductSet(i2, i2))


# 定义一个测试函数，用于验证 SymmetricDifference 对象的各种情况
def test_SymmetricDifference():
    # 定义集合对象 A、B 和区间对象 C
    A = FiniteSet(0, 1, 2, 3, 4, 5)
    B = FiniteSet(2, 4, 6, 8, 10)
    C = Interval(8, 10)

    # 断言对称差集 A 和 B 的结果可以迭代
    assert SymmetricDifference(A, B, evaluate=False).is_iterable is True
    # 断言对称差集 A 和 C 的结果不可以迭代
    assert SymmetricDifference(A, C, evaluate=False).is_iterable is None
    # 断言对称差集 A 和 B 的结果与预期的有限集合对象相等
    assert FiniteSet(*SymmetricDifference(A, B, evaluate=False)) == \
        FiniteSet(0, 1, 3, 5, 6, 8, 10)
    # 断言对称差集 A 和 C 引发 TypeError 异常
    raises(TypeError,
        lambda: FiniteSet(*SymmetricDifference(A, C, evaluate=False)))

    # 断言对称差集两个有限集合对象的计算结果与预期的有限集合对象相等
    assert SymmetricDifference(FiniteSet(0, 1, 2, 3, 4, 5), \
            FiniteSet(2, 4, 6, 8, 10)) == FiniteSet(0, 1, 3, 5, 6, 8, 10)
    assert SymmetricDifference(FiniteSet(2, 3, 4), FiniteSet(2, 3, 4 ,5)) \
            == FiniteSet(5)
    assert FiniteSet(1, 2, 3, 4, 5) ^ FiniteSet(1, 2, 5, 6) == \
            FiniteSet(3, 4, 6)
    assert Set(S(1), S(2), S(3)) ^ Set(S(2), S(3), S(4)) == Union(Set(S(1), S(2), S(3)) - Set(S(2), S(3), S(4)), \
            Set(S(2), S(3), S(4)) - Set(S(1), S(2), S(3)))
    assert Interval(0, 4) ^ Interval(2, 5) == Union(Interval(0, 4) - \
            Interval(2, 5), Interval(2, 5) - Interval(0, 4))


# 定义一个测试函数，用于验证在 SymPy 中的问题 9536
def test_issue_9536():
    # 导入对数函数 log 和实数符号 a
    from sympy.functions.elementary.exponential import log
    a = Symbol('a', real=True)
    # 断言语句，用于检查 log(a) 的对数值集合与实数集的交集是否等于 log(a) 的有限集
    assert FiniteSet(log(a)).intersect(S.Reals) == Intersection(S.Reals, FiniteSet(log(a)))
def test_issue_9637():
    # 创建符号 'n'
    n = Symbol('n')
    # 创建包含单个符号 'n' 的有限集合
    a = FiniteSet(n)
    # 创建包含常数 2 和符号 'n' 的有限集合
    b = FiniteSet(2, n)
    # 验证补集操作的评估结果与未评估结果是否相等
    assert Complement(S.Reals, a) == Complement(S.Reals, a, evaluate=False)
    # 验证补集操作的评估结果与未评估结果是否相等
    assert Complement(Interval(1, 3), a) == Complement(Interval(1, 3), a, evaluate=False)
    # 验证补集操作结果与预期的联合区间是否相等
    assert Complement(Interval(1, 3), b) == \
        Complement(Union(Interval(1, 2, False, True), Interval(2, 3, True, False)), a)
    # 验证补集操作的评估结果与未评估结果是否相等
    assert Complement(a, S.Reals) == Complement(a, S.Reals, evaluate=False)
    # 验证补集操作的评估结果与未评估结果是否相等
    assert Complement(a, Interval(1, 3)) == Complement(a, Interval(1, 3), evaluate=False)


def test_issue_9808():
    # 参考 https://github.com/sympy/sympy/issues/16342
    # 验证补集操作的评估结果与未评估结果是否相等
    assert Complement(FiniteSet(y), FiniteSet(1)) == Complement(FiniteSet(y), FiniteSet(1), evaluate=False)
    # 验证补集操作的评估结果与未评估结果是否相等
    assert Complement(FiniteSet(1, 2, x), FiniteSet(x, y, 2, 3)) == \
        Complement(FiniteSet(1), FiniteSet(y), evaluate=False)


def test_issue_9956():
    # 验证 Union 操作的结果与预期的区间是否相等
    assert Union(Interval(-oo, oo), FiniteSet(1)) == Interval(-oo, oo)
    # 验证区间对象是否包含给定元素 1
    assert Interval(-oo, oo).contains(1) is S.true


def test_issue_Symbol_inter():
    # 创建半开区间对象 [0, oo)
    i = Interval(0, oo)
    # 创建实数集对象
    r = S.Reals
    # 创建包含零矩阵的矩阵对象
    mat = Matrix([0, 0, 0])
    # 验证交集操作的结果是否与预期的交集对象相等
    assert Intersection(r, i, FiniteSet(m), FiniteSet(m, n)) == \
        Intersection(i, FiniteSet(m))
    # 验证交集操作的结果是否与预期的交集对象相等
    assert Intersection(FiniteSet(1, m, n), FiniteSet(m, n, 2), i) == \
        Intersection(i, FiniteSet(m, n))
    # 验证交集操作的结果是否与预期的交集对象相等
    assert Intersection(FiniteSet(m, n, x), FiniteSet(m, z), r) == \
        Intersection(Intersection({m, z}, {m, n, x}), r)
    # 验证交集操作的结果是否与预期的交集对象相等
    assert Intersection(FiniteSet(m, n, 3), FiniteSet(m, n, x), r) == \
        Intersection(FiniteSet(3, m, n), FiniteSet(m, n, x), r, evaluate=False)
    # 验证交集操作的结果是否与预期的交集对象相等
    assert Intersection(FiniteSet(m, n, 3), FiniteSet(m, n, 2, 3), r) == \
        Intersection(FiniteSet(3, m, n), r)
    # 验证交集操作的结果是否与预期的交集对象相等
    assert Intersection(r, FiniteSet(mat, 2, n), FiniteSet(0, mat, n)) == \
        Intersection(r, FiniteSet(n))
    # 验证交集操作的结果是否与预期的交集对象相等
    assert Intersection(FiniteSet(sin(x), cos(x)), FiniteSet(sin(x), cos(x), 1), r) == \
        Intersection(r, FiniteSet(sin(x), cos(x)))
    # 验证交集操作的结果是否与预期的交集对象相等
    assert Intersection(FiniteSet(x**2, 1, sin(x)), FiniteSet(x**2, 2, sin(x)), r) == \
        Intersection(r, FiniteSet(x**2, sin(x)))


def test_issue_11827():
    # 验证自然数非负集合的四次幂
    assert S.Naturals0**4


def test_issue_10113():
    # 创建符号函数
    f = x**2/(x**2 - 4)
    # 验证映射集操作的结果是否与预期的区间对象相等
    assert imageset(x, f, S.Reals) == Union(Interval(-oo, 0), Interval(1, oo, True, True))
    # 验证映射集操作的结果是否与预期的区间对象相等
    assert imageset(x, f, Interval(-2, 2)) == Interval(-oo, 0)
    # 验证映射集操作的结果是否与预期的区间对象相等
    assert imageset(x, f, Interval(-2, 3)) == Union(Interval(-oo, 0), Interval(Rational(9, 5), oo))


def test_issue_10248():
    # 验证 TypeError 是否被正确地引发
    raises(
        TypeError, lambda: list(Intersection(S.Reals, FiniteSet(x)))
    )
    # 创建具有实数属性的符号 'A'
    A = Symbol('A', real=True)
    # 验证交集操作的结果列表是否与预期的列表相等
    assert list(Intersection(S.Reals, FiniteSet(A))) == [A]


def test_issue_9447():
    # 创建两个区间的并集对象
    a = Interval(0, 1) + Interval(2, 3)
    # 验证补集操作的评估结果与未评估结果是否相等
    assert Complement(S.UniversalSet, a) == Complement(
            S.UniversalSet, Union(Interval(0, 1), Interval(2, 3)), evaluate=False)
    # 断言：验证集合 a 相对于 S.Naturals 的补集是否等于
    # S.Naturals 相对于 Union(Interval(0, 1), Interval(2, 3)) 的补集，
    # 并且不进行表达式求值。
    assert Complement(S.Naturals, a) == Complement(
            S.Naturals, Union(Interval(0, 1), Interval(2, 3)), evaluate=False)
def test_issue_10337():
    # 断言 FiniteSet(2) 等于 3 是假的
    assert (FiniteSet(2) == 3) is False
    # 断言 FiniteSet(2) 不等于 3 是真的
    assert (FiniteSet(2) != 3) is True
    # 引发 TypeError，因为 FiniteSet(2) 不能与整数 3 进行小于比较
    raises(TypeError, lambda: FiniteSet(2) < 3)
    # 引发 TypeError，因为 FiniteSet(2) 不能与整数 3 进行小于等于比较
    raises(TypeError, lambda: FiniteSet(2) <= 3)
    # 引发 TypeError，因为 FiniteSet(2) 不能与整数 3 进行大于比较
    raises(TypeError, lambda: FiniteSet(2) > 3)
    # 引发 TypeError，因为 FiniteSet(2) 不能与整数 3 进行大于等于比较
    raises(TypeError, lambda: FiniteSet(2) >= 3)


def test_issue_10326():
    # 定义一个包含各种对象的列表 bad
    bad = [
        EmptySet,
        FiniteSet(1),
        Interval(1, 2),
        S.ComplexInfinity,
        S.ImaginaryUnit,
        S.Infinity,
        S.NaN,
        S.NegativeInfinity,
    ]
    # 创建一个区间对象 interval，范围是 [0, 5]
    interval = Interval(0, 5)
    # 遍历 bad 中的每个对象 i，断言 i 不在 interval 中
    for i in bad:
        assert i not in interval

    # 创建一个实数符号 x
    x = Symbol('x', real=True)
    # 创建一个非扩展实数符号 nr
    nr = Symbol('nr', extended_real=False)
    # 断言 x + 1 在区间 [x, x + 4] 内
    assert x + 1 in Interval(x, x + 4)
    # 断言 nr 不在区间 [x, x + 4] 内
    assert nr not in Interval(x, x + 4)
    # 断言区间 [1, 2] 在有限集 {Interval(0, 5), Interval(1, 2)} 中
    assert Interval(1, 2) in FiniteSet(Interval(0, 5), Interval(1, 2))
    # 断言区间 [-oo, oo] 不包含 oo
    assert Interval(-oo, oo).contains(oo) is S.false
    # 断言区间 [-oo, oo] 不包含 -oo
    assert Interval(-oo, oo).contains(-oo) is S.false


def test_issue_2799():
    # 创建一个表示全集的符号 U
    U = S.UniversalSet
    # 创建一个实数符号 a
    a = Symbol('a', real=True)
    # 创建一个从 a 到 oo 的区间对象 inf_interval
    inf_interval = Interval(a, oo)
    # 创建一个表示实数集的符号 R
    R = S.Reals

    # 断言全集 U 与 inf_interval 的并集等于 inf_interval 与 U 的并集
    assert U + inf_interval == inf_interval + U
    # 断言全集 U 与实数集 R 的并集等于实数集 R 与 U 的并集
    assert U + R == R + U
    # 断言实数集 R 与 inf_interval 的并集等于 inf_interval 与实数集 R 的并集
    assert R + inf_interval == inf_interval + R


def test_issue_9706():
    # 断言区间 [-oo, 0] 的闭包等于区间 [-oo, 0] 的表示（包含 -oo，不包含 0）
    assert Interval(-oo, 0).closure == Interval(-oo, 0, True, False)
    # 断言区间 [0, oo] 的闭包等于区间 [0, oo] 的表示（不包含 0，包含 oo）
    assert Interval(0, oo).closure == Interval(0, oo, False, True)
    # 断言区间 [-oo, oo] 的闭包等于区间 [-oo, oo] 的表示
    assert Interval(-oo, oo).closure == Interval(-oo, oo)


def test_issue_8257():
    # 创建一个表示实数集和无穷的并集对象 reals_plus_infinity
    reals_plus_infinity = Union(Interval(-oo, oo), FiniteSet(oo))
    # 创建一个表示实数集和负无穷的并集对象 reals_plus_negativeinfinity
    reals_plus_negativeinfinity = Union(Interval(-oo, oo), FiniteSet(-oo))
    # 断言区间 [-oo, oo] 与有限集 {oo} 的并集等于 reals_plus_infinity
    assert Interval(-oo, oo) + FiniteSet(oo) == reals_plus_infinity
    # 断言有限集 {oo} 与区间 [-oo, oo] 的并集等于 reals_plus_infinity
    assert FiniteSet(oo) + Interval(-oo, oo) == reals_plus_infinity
    # 断言区间 [-oo, oo] 与有限集 {-oo} 的并集等于 reals_plus_negativeinfinity
    assert Interval(-oo, oo) + FiniteSet(-oo) == reals_plus_negativeinfinity
    # 断言有限集 {-oo} 与区间 [-oo, oo] 的并集等于 reals_plus_negativeinfinity
    assert FiniteSet(-oo) + Interval(-oo, oo) == reals_plus_negativeinfinity


def test_issue_10931():
    # 断言整数集 S.Integers 减去整数集 S.Integers 得到空集 EmptySet
    assert S.Integers - S.Integers == EmptySet
    # 断言整数集 S.Integers 减去实数集 S.Reals 得到空集 EmptySet
    assert S.Integers - S.Reals == EmptySet


def test_issue_11174():
    # 创建一个求交集的对象 soln，表示为 Intersection(FiniteSet(-x), S.Reals)，禁止求值
    soln = Intersection(Interval(-oo, oo), FiniteSet(-x), evaluate=False)
    # 断言 Intersection(FiniteSet(-x), S.Reals) 等于 soln
    assert Intersection(FiniteSet(-x), S.Reals) == soln

    # 创建一个求交集的对象 soln，表示为 Intersection(S.Reals, FiniteSet(x)，禁止求值
    soln = Intersection(S.Reals, FiniteSet(x), evaluate=False)
    # 断言 Intersection(FiniteSet(x), S.Reals) 等于 soln
    assert Intersection(FiniteSet(x), S.Reals) == soln


def test_issue_18505():
    # 断言 ImageSet(Lambda(n, sqrt(pi*n/2 - 1 + pi/2)), S.Integers) 包含 0
    assert ImageSet(Lambda(n, sqrt(pi*n/2 - 1 + pi/2)), S.Integers).contains(0) == \
            Contains(0, ImageSet(Lambda(n, sqrt(pi*n/2 - 1 + pi/2)), S.Integers))


def test_finite_set_intersection():
    # 下列代码不应产生递归错误
    # 注意：有些断言并不完全正确。详见 https://github.com/sympy/sympy/issues/16342。
    # 断言 FiniteSet(-oo, x) 与 FiniteSet(x) 的交集等于 FiniteSet(x)
    assert Intersection(FiniteSet(-oo, x), FiniteSet(x)) == FiniteSet(x)
    # 断言交集处理函数 Intersection._handle_finite_sets([FiniteSet(-oo, x), FiniteSet(0, x)]) 返回 FiniteSet(x)
    assert Intersection._handle_finite_sets([FiniteSet(-oo, x), FiniteSet(0, x)]) == FiniteSet(x)

    # 断言交集处理函数 Intersection._handle_finite_sets([FiniteSet(-oo, x), FiniteSet(x)]) 返回 FiniteSet(x)
    assert Intersection._handle_finite_sets([FiniteSet(-oo, x), FiniteSet(x)]) == FiniteSet(x)
    # 断言语句，验证 Intersection 类中 _handle_finite_sets 方法的返回结果
    assert Intersection._handle_finite_sets([FiniteSet(2, 3, x, y), FiniteSet(1, 2, x)]) == \
        Intersection._handle_finite_sets([FiniteSet(1, 2, x), FiniteSet(2, 3, x, y)]) == \
        Intersection(FiniteSet(1, 2, x), FiniteSet(2, 3, x, y)) == \
        Intersection(FiniteSet(1, 2, x), FiniteSet(2, x, y))
    
    # 断言语句，验证 FiniteSet 对象的交集运算
    assert FiniteSet(1+x-y) & FiniteSet(1) == \
        FiniteSet(1) & FiniteSet(1+x-y) == \
        Intersection(FiniteSet(1+x-y), FiniteSet(1), evaluate=False)
    
    # 断言语句，验证 FiniteSet 对象的交集运算
    assert FiniteSet(1) & FiniteSet(x) == FiniteSet(x) & FiniteSet(1) == \
        Intersection(FiniteSet(1), FiniteSet(x), evaluate=False)
    
    # 断言语句，验证 FiniteSet 对象的交集运算，其中集合中包含集合的情况
    assert FiniteSet({x}) & FiniteSet({x, y}) == \
        Intersection(FiniteSet({x}), FiniteSet({x, y}), evaluate=False)
def test_union_intersection_constructor():
    # 创建一个包含两个 FiniteSet 对象的列表
    sets = [FiniteSet(1), FiniteSet(2)]
    # 测试 Union 和 Intersection 构造函数是否会引发异常
    raises(Exception, lambda: Union(sets))
    raises(Exception, lambda: Intersection(sets))
    # 将列表转换为元组，再次测试 Union 和 Intersection 构造函数是否会引发异常
    raises(Exception, lambda: Union(tuple(sets)))
    raises(Exception, lambda: Intersection(tuple(sets)))
    # 使用生成器表达式测试 Union 和 Intersection 构造函数是否会引发异常
    raises(Exception, lambda: Union(i for i in sets))
    raises(Exception, lambda: Intersection(i for i in sets))

    # Python 中的集合对象与 FiniteSet 相同处理方式
    # 单个集合的并集仍然是该集合本身
    assert Union(set(sets)) == FiniteSet(*sets)
    assert Intersection(set(sets)) == FiniteSet(*sets)

    # 测试两个集合对象的并集和交集
    assert Union({1}, {2}) == FiniteSet(1, 2)
    assert Intersection({1, 2}, {2, 3}) == FiniteSet(2)


def test_Union_contains():
    # 测试是否存在 zoo 元素不在 (-oo, 0) 和 (0, oo) 两个区间的并集中
    assert zoo not in Union(
        Interval.open(-oo, 0), Interval.open(0, oo))


@XFAIL
def test_issue_16878b():
    # 在 intersection_sets 中，对于 (ImageSet, Set) 类型，没有处理 S.Reals 基础集合的代码
    # 与处理整数集合的代码不同
    assert imageset(x, (x, x), S.Reals).is_subset(S.Reals**2) is True

def test_DisjointUnion():
    # 测试 DisjointUnion 对象的重写方法
    assert DisjointUnion(FiniteSet(1, 2, 3), FiniteSet(1, 2, 3), FiniteSet(1, 2, 3)).rewrite(Union) == (FiniteSet(1, 2, 3) * FiniteSet(0, 1, 2))
    assert DisjointUnion(Interval(1, 3), Interval(2, 4)).rewrite(Union) == Union(Interval(1, 3) * FiniteSet(0), Interval(2, 4) * FiniteSet(1))
    assert DisjointUnion(Interval(0, 5), Interval(0, 5)).rewrite(Union) == Union(Interval(0, 5) * FiniteSet(0), Interval(0, 5) * FiniteSet(1))
    assert DisjointUnion(Interval(-1, 2), S.EmptySet, S.EmptySet).rewrite(Union) == Interval(-1, 2) * FiniteSet(0)
    assert DisjointUnion(Interval(-1, 2)).rewrite(Union) == Interval(-1, 2) * FiniteSet(0)
    assert DisjointUnion(S.EmptySet, Interval(-1, 2), S.EmptySet).rewrite(Union) == Interval(-1, 2) * FiniteSet(1)
    assert DisjointUnion(Interval(-oo, oo)).rewrite(Union) == Interval(-oo, oo) * FiniteSet(0)
    assert DisjointUnion(S.EmptySet).rewrite(Union) == S.EmptySet
    assert DisjointUnion().rewrite(Union) == S.EmptySet
    # 测试传入 Symbol('n') 是否会引发 TypeError 异常
    raises(TypeError, lambda: DisjointUnion(Symbol('n')))

    x = Symbol("x")
    y = Symbol("y")
    z = Symbol("z")
    # 测试包含不同 FiniteSet 对象的 DisjointUnion 对象的重写方法
    assert DisjointUnion(FiniteSet(x), FiniteSet(y, z)).rewrite(Union) == (FiniteSet(x) * FiniteSet(0)) + (FiniteSet(y, z) * FiniteSet(1))

def test_DisjointUnion_is_empty():
    # 测试空集合的 DisjointUnion 对象是否为空
    assert DisjointUnion(S.EmptySet).is_empty is True
    assert DisjointUnion(S.EmptySet, S.EmptySet).is_empty is True
    assert DisjointUnion(S.EmptySet, FiniteSet(1, 2, 3)).is_empty is False

def test_DisjointUnion_is_iterable():
    # 测试 DisjointUnion 对象是否可迭代
    assert DisjointUnion(S.Integers, S.Naturals, S.Rationals).is_iterable is True
    assert DisjointUnion(S.EmptySet, S.Reals).is_iterable is False
    assert DisjointUnion(FiniteSet(1, 2, 3), S.EmptySet, FiniteSet(x, y)).is_iterable is True
    assert DisjointUnion(S.EmptySet, S.EmptySet).is_iterable is False
# 定义测试函数 test_DisjointUnion_contains，用于测试 DisjointUnion 类的 contains 方法
def test_DisjointUnion_contains():
    # 断言元组 (0, 0) 是否在以 FiniteSet(0, 1, 2) 为基础集合的 DisjointUnion 中
    assert (0, 0) in DisjointUnion(FiniteSet(0, 1, 2), FiniteSet(0, 1, 2), FiniteSet(0, 1, 2))
    # 断言元组 (0, 1) 是否在同样的 DisjointUnion 中
    assert (0, 1) in DisjointUnion(FiniteSet(0, 1, 2), FiniteSet(0, 1, 2), FiniteSet(0, 1, 2))
    # 断言元组 (0, 2) 是否在同样的 DisjointUnion 中
    assert (0, 2) in DisjointUnion(FiniteSet(0, 1, 2), FiniteSet(0, 1, 2), FiniteSet(0, 1, 2))
    # 断言元组 (1, 0) 是否在同样的 DisjointUnion 中
    assert (1, 0) in DisjointUnion(FiniteSet(0, 1, 2), FiniteSet(0, 1, 2), FiniteSet(0, 1, 2))
    # 断言元组 (1, 1) 是否在同样的 DisjointUnion 中
    assert (1, 1) in DisjointUnion(FiniteSet(0, 1, 2), FiniteSet(0, 1, 2), FiniteSet(0, 1, 2))
    # 断言元组 (1, 2) 是否在同样的 DisjointUnion 中
    assert (1, 2) in DisjointUnion(FiniteSet(0, 1, 2), FiniteSet(0, 1, 2), FiniteSet(0, 1, 2))
    # 断言元组 (2, 0) 是否在同样的 DisjointUnion 中
    assert (2, 0) in DisjointUnion(FiniteSet(0, 1, 2), FiniteSet(0, 1, 2), FiniteSet(0, 1, 2))
    # 断言元组 (2, 1) 是否在同样的 DisjointUnion 中
    assert (2, 1) in DisjointUnion(FiniteSet(0, 1, 2), FiniteSet(0, 1, 2), FiniteSet(0, 1, 2))
    # 断言元组 (2, 2) 是否在同样的 DisjointUnion 中
    assert (2, 2) in DisjointUnion(FiniteSet(0, 1, 2), FiniteSet(0, 1, 2), FiniteSet(0, 1, 2))
    # 断言元组 (0, 1, 2) 是否不在同样的 DisjointUnion 中
    assert (0, 1, 2) not in DisjointUnion(FiniteSet(0, 1, 2), FiniteSet(0, 1, 2), FiniteSet(0, 1, 2))
    # 断言元组 (0, 0.5) 是否不在以 FiniteSet(0.5) 为基础集合的 DisjointUnion 中
    assert (0, 0.5) not in DisjointUnion(FiniteSet(0.5))
    # 断言元组 (0, 5) 是否不在同样的 DisjointUnion 中
    assert (0, 5) not in DisjointUnion(FiniteSet(0, 1, 2), FiniteSet(0, 1, 2), FiniteSet(0, 1, 2))
    # 断言元组 (x, 0) 是否在以 FiniteSet(x, y, z), S.EmptySet, FiniteSet(y) 为基础集合的 DisjointUnion 中
    assert (x, 0) in DisjointUnion(FiniteSet(x, y, z), S.EmptySet, FiniteSet(y))
    # 断言元组 (y, 0) 是否在同样的 DisjointUnion 中
    assert (y, 0) in DisjointUnion(FiniteSet(x, y, z), S.EmptySet, FiniteSet(y))
    # 断言元组 (z, 0) 是否在同样的 DisjointUnion 中
    assert (z, 0) in DisjointUnion(FiniteSet(x, y, z), S.EmptySet, FiniteSet(y))
    # 断言元组 (y, 2) 是否在同样的 DisjointUnion 中
    assert (y, 2) in DisjointUnion(FiniteSet(x, y, z), S.EmptySet, FiniteSet(y))
    # 断言元组 (0.5, 0) 是否在以 Interval(0, 1), Interval(0, 2) 为基础集合的 DisjointUnion 中
    assert (0.5, 0) in DisjointUnion(Interval(0, 1), Interval(0, 2))
    # 断言元组 (0.5, 1) 是否在同样的 DisjointUnion 中
    assert (0.5, 1) in DisjointUnion(Interval(0, 1), Interval(0, 2))
    # 断言元组 (1.5, 0) 是否不在同样的 DisjointUnion 中
    assert (1.5, 0) not in DisjointUnion(Interval(0, 1), Interval(0, 2))
    # 断言元组 (1.5, 1) 是否在同样的 DisjointUnion 中
    assert (1.5, 1) in DisjointUnion(Interval(0, 1), Interval(0, 2))

# 定义测试函数 test_DisjointUnion_iter，用于测试 DisjointUnion 类的 iter 方法
def test_DisjointUnion_iter():
    # 创建 DisjointUnion 对象 D，基础集合分别为 FiniteSet(3, 5, 7, 9) 和 FiniteSet(x, y, z)
    D = DisjointUnion(FiniteSet(3, 5, 7, 9), FiniteSet(x, y, z))
    # 获取 DisjointUnion 对象的迭代器
    it = iter(D)
    # 定义两个预期结果列表
    L1 = [(x, 1), (y, 1), (z, 1)]
    L2 = [(3, 0), (5, 0), (7, 0), (9, 0)]
    # 获取迭代器的下一个元素，并断言它在 L2 中
    nxt = next(it)
    assert nxt in L2
    # 从 L2 中移除已获取的元素
    L2.remove(nxt)
    # 获取迭代器的下一个元素，并断言它在 L1 中
    nxt = next(it)
    assert nxt in L1
    # 从 L1 中移除已获取的元素
    L1.remove(nxt)
    # 重复以上步骤，直到迭代完成
    # 最后断言迭代器抛出 StopIteration 异常
    nxt = next(it)
    assert nxt in L2
    L2.remove(nxt)
    nxt = next(it)
    assert nxt in L1
    L1.remove(nxt)
    nxt = next(it)
    assert nxt in L2
    L2.remove(nxt)
    nxt = next(it)
    assert nxt in L1
    L1.remove(nxt)
    nxt = next(it)
    assert nxt in L2
    L2.remove(nxt)
    raises(StopIteration, lambda: next(it))

    # 断言试图迭代包含 Interval(0, 1) 和 S.EmptySet 的 DisjointUnion 时引发 ValueError 异常
    raises(ValueError, lambda: iter(DisjointUnion(Interval(0, 1), S.EmptySet)))

# 定义测试函数 test_DisjointUnion_len，用于测试 DisjointUnion 类的 len 方法
def test_DisjointUnion_len():
    # 断言 DisjointUnion 对象的长度为 7，基础集合分别为 FiniteSet(3, 5, 7, 9) 和 FiniteSet(x, y, z)
    assert len(DisjointUnion(FiniteSet(3, 5, 7, 9), FiniteSet(x, y, z))) == 7
    # 断言 DisjointUnion 对象的长度为 3，基础集合分别为 S.EmptySet, S.EmptySet, FiniteSet(x, y, z), S.EmptySet
    assert len(DisjointUnion(S.EmptySet, S.EmptySet, FiniteSet(x, y, z), S.EmptySet)) == 3
    # 断言试图获取包
    # 断言：验证给定的对象是否符合特定条件
    assert ProductSet(Interval(1, 2), FiniteSet(Matrix([1, 2]))).kind is SetKind(TupleKind(NumberKind, mk))
    # 创建一个 ProductSet 对象，该对象由一个 Interval 和一个 FiniteSet 组成，用来表示集合的乘积
    # Interval(1, 2) 表示闭区间 [1, 2]
    # FiniteSet(Matrix([1, 2])) 表示包含一个 Matrix 对象 [1, 2] 的有限集合
    # .kind 访问 ProductSet 对象的种类属性
    # SetKind(TupleKind(NumberKind, mk)) 表示一个特定种类的集合，其元素为元组，元组的第一个元素是 NumberKind 类型，第二个元素是变量 mk 的值
    # 最终断言表达式验证 ProductSet 对象的种类是否与预期的 SetKind(TupleKind(NumberKind, mk)) 相符
# 测试函数，用于检查 Interval 对象的 kind 属性是否为 NumberKind
def test_SetKind_Interval():
    assert Interval(1, 2).kind is SetKind(NumberKind)

# 测试函数，验证 EmptySet 和 UniversalSet 的 kind 属性
def test_SetKind_EmptySet_UniversalSet():
    assert S.UniversalSet.kind is SetKind(UndefinedKind)
    assert EmptySet.kind is SetKind()

# 测试函数，验证 FiniteSet 对象的 kind 属性
def test_SetKind_FiniteSet():
    assert FiniteSet(1, Matrix([1, 2])).kind is SetKind(UndefinedKind)
    assert FiniteSet(1, 2).kind is SetKind(NumberKind)

# 测试函数，验证 Union 对象的 kind 属性
def test_SetKind_Unions():
    assert Union(FiniteSet(Matrix([1, 2])), Interval(1, 2)).kind is SetKind(UndefinedKind)
    assert Union(Interval(1, 2), Interval(1, 7)).kind is SetKind(NumberKind)

# 测试函数，验证 DisjointUnion 对象的 kind 属性
def test_SetKind_DisjointUnion():
    A = FiniteSet(1, 2, 3)
    B = Interval(0, 5)
    assert DisjointUnion(A, B).kind is SetKind(NumberKind)

# 测试函数，验证 evaluate=False 参数下的 Union、Intersection、Complement 对象的 kind 属性
def test_SetKind_evaluate_False():
    U = lambda *args: Union(*args, evaluate=False)
    assert U({1}, EmptySet).kind is SetKind(NumberKind)
    assert U(Interval(1, 2), EmptySet).kind is SetKind(NumberKind)
    assert U({1}, S.UniversalSet).kind is SetKind(UndefinedKind)
    assert U(Interval(1, 2), Interval(4, 5), FiniteSet(1)).kind is SetKind(NumberKind)
    
    I = lambda *args: Intersection(*args, evaluate=False)
    assert I({1}, S.UniversalSet).kind is SetKind(NumberKind)
    assert I({1}, EmptySet).kind is SetKind()
    
    C = lambda *args: Complement(*args, evaluate=False)
    assert C(S.UniversalSet, {1, 2, 4, 5}).kind is SetKind(UndefinedKind)
    assert C({1, 2, 3, 4, 5}, EmptySet).kind is SetKind(NumberKind)
    assert C(EmptySet, {1, 2, 3, 4, 5}).kind is SetKind()

# 测试函数，验证 ImageSet 对象的 kind 属性
def test_SetKind_ImageSet_Special():
    f = ImageSet(Lambda(n, n ** 2), Interval(1, 4))
    assert (f - FiniteSet(3)).kind is SetKind(NumberKind)
    assert (f + Interval(16, 17)).kind is SetKind(NumberKind)
    assert (f + FiniteSet(17)).kind is SetKind(NumberKind)

# 测试函数，验证 FiniteSet 对象的一些特定情况
def test_issue_20089():
    B = FiniteSet(FiniteSet(1, 2), FiniteSet(1))
    assert 1 not in B
    assert 1.0 not in B
    assert not Eq(1, FiniteSet(1, 2))
    assert FiniteSet(1) in B
    A = FiniteSet(1, 2)
    assert A in B
    assert B.issubset(B)
    assert not A.issubset(B)
    assert 1 in A
    C = FiniteSet(FiniteSet(1, 2), FiniteSet(1), 1, 2)
    assert A.issubset(C)
    assert B.issubset(C)

# 测试函数，验证 FiniteSet 和 ProductSet 对象的 kind 属性
def test_issue_19378():
    a = FiniteSet(1, 2)
    b = ProductSet(a, a)
    c = FiniteSet((1, 1), (1, 2), (2, 1), (2, 2))
    assert b.is_subset(c) is True
    d = FiniteSet(1)
    assert b.is_subset(d) is False
    assert Eq(c, b).simplify() is S.true
    assert Eq(a, c).simplify() is S.false
    assert Eq({1}, {x}).simplify() == Eq({1}, {x})

# 测试函数，验证 Intersection 对象在符号计算下的使用
def test_intersection_symbolic():
    n = Symbol('n')
    # 这些情况不应该抛出错误
    assert isinstance(Intersection(Range(n), Range(100)), Intersection)
    assert isinstance(Intersection(Range(n), Interval(1, 100)), Intersection)
    assert isinstance(Intersection(Range(100), Interval(1, n)), Intersection)

# 标记为预期失败的测试函数，验证 Intersection 对象在特定符号情况下的行为
@XFAIL
def test_intersection_symbolic_failing():
    n = Symbol('n', integer=True, positive=True)
    # 断言两个集合的交集应该相等
    assert Intersection(Range(10, n), Range(4, 500, 5)) == Intersection(
        Range(14, n), Range(14, 500, 5))
    
    # 断言两个集合的交集应该相等
    assert Intersection(Interval(10, n), Range(4, 500, 5)) == Intersection(
        Interval(14, n), Range(14, 500, 5))
# 测试 GitHub 问题编号 20379 的问题
def test_issue_20379():
    # 定义 x，其值为 π 减去近似值 3.14159265358979
    x = pi - 3.14159265358979
    # 断言使用有限集合对 x 求值保留两位小数后等于指定的浮点数
    assert FiniteSet(x).evalf(2) == FiniteSet(Float('3.23108914886517e-15', 2))

# 测试 FiniteSet 的 simplify 方法
def test_finiteset_simplify():
    # 创建一个包含 1 和 cos(1)**2 + sin(1)**2 的有限集合 S
    S = FiniteSet(1, cos(1)**2 + sin(1)**2)
    # 断言对集合 S 调用 simplify 方法后结果为 {1}
    assert S.simplify() == {1}

# 测试 GitHub 问题编号 14336 的问题
def test_issue_14336():
    # 定义 U，其为复数集合 S.Complexes
    U = S.Complexes
    # 定义符号 x
    x = Symbol("x")
    # 从集合 U 中去除与 Ne(x, 1) 相交的部分
    U -= U.intersect(Ne(x, 1).as_set())
    # 从集合 U 中去除与 S.true 相交的部分
    U -= U.intersect(S.true.as_set())

# 测试 GitHub 问题编号 9855 的问题
def test_issue_9855():
    # 定义实数符号 x, y, z
    x, y, z = symbols('x, y, z', real=True)
    # 创建区间 s1 和 s2
    s1 = Interval(1, x) & Interval(y, 2)
    s2 = Interval(1, 2)
    # 断言区间 s1 是否是区间 s2 的子集，期望结果为 None
    assert s1.is_subset(s2) == None
```