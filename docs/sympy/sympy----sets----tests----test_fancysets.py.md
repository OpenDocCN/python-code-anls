# `D:\src\scipysrc\sympy\sympy\sets\tests\test_fancysets.py`

```
from sympy.core.expr import unchanged  # 导入 unchanged 函数
from sympy.sets.contains import Contains  # 导入 Contains 类
from sympy.sets.fancysets import (ImageSet, Range, normalize_theta_set,
                                  ComplexRegion)  # 导入多个 fancysets 模块
from sympy.sets.sets import (FiniteSet, Interval, Union, imageset,
                             Intersection, ProductSet, SetKind)  # 导入多个 sets 模块
from sympy.sets.conditionset import ConditionSet  # 导入 ConditionSet 类
from sympy.simplify.simplify import simplify  # 导入 simplify 函数
from sympy.core.basic import Basic  # 导入 Basic 类
from sympy.core.containers import Tuple, TupleKind  # 导入 Tuple 和 TupleKind 类
from sympy.core.function import Lambda  # 导入 Lambda 类
from sympy.core.kind import NumberKind  # 导入 NumberKind 类
from sympy.core.numbers import (I, Rational, oo, pi)  # 导入多个数学常数和类型
from sympy.core.relational import Eq  # 导入 Eq 类
from sympy.core.singleton import S  # 导入 S 单例
from sympy.core.symbol import (Dummy, Symbol, symbols)  # 导入符号相关类和函数
from sympy.functions.elementary.complexes import Abs  # 导入 Abs 函数
from sympy.functions.elementary.exponential import (exp, log)  # 导入 exp 和 log 函数
from sympy.functions.elementary.integers import floor  # 导入 floor 函数
from sympy.functions.elementary.miscellaneous import sqrt  # 导入 sqrt 函数
from sympy.functions.elementary.trigonometric import (cos, sin, tan)  # 导入三角函数
from sympy.logic.boolalg import And  # 导入 And 函数
from sympy.matrices.dense import eye  # 导入 eye 函数
from sympy.testing.pytest import XFAIL, raises  # 导入测试相关函数和装饰器
from sympy.abc import x, y, t, z  # 导入符号变量
from sympy.core.mod import Mod  # 导入 Mod 类

import itertools  # 导入 itertools 模块


def test_naturals():
    N = S.Naturals  # 创建自然数集合对象 N
    assert 5 in N  # 测试 5 是否在 N 中
    assert -5 not in N  # 测试 -5 是否不在 N 中
    assert 5.5 not in N  # 测试 5.5 是否不在 N 中
    ni = iter(N)  # 创建 N 的迭代器 ni
    a, b, c, d = next(ni), next(ni), next(ni), next(ni)  # 从 ni 中获取四个值并赋给 a, b, c, d
    assert (a, b, c, d) == (1, 2, 3, 4)  # 断言 a, b, c, d 的值分别为 1, 2, 3, 4
    assert isinstance(a, Basic)  # 断言 a 是 Basic 类的实例

    assert N.intersect(Interval(-5, 5)) == Range(1, 6)  # 断言 N 与区间 (-5, 5) 的交集为 Range(1, 6)
    assert N.intersect(Interval(-5, 5, True, True)) == Range(1, 5)  # 断言 N 与开区间 (-5, 5) 的交集为 Range(1, 5)

    assert N.boundary == N  # 断言 N 的边界等于 N 本身
    assert N.is_open == False  # 断言 N 不是开集
    assert N.is_closed == True  # 断言 N 是闭集

    assert N.inf == 1  # 断言 N 的下界为 1
    assert N.sup is oo  # 断言 N 的上界为无穷大
    assert not N.contains(oo)  # 断言 N 不包含无穷大
    for s in (S.Naturals0, S.Naturals):
        assert s.intersection(S.Reals) is s  # 断言 s 与实数集合的交集等于 s 自身
        assert s.is_subset(S.Reals)  # 断言 s 是实数集合的子集

    assert N.as_relational(x) == And(Eq(floor(x), x), x >= 1, x < oo)  # 断言 N 的关系表达式为 floor(x) = x 且 x 在 [1, oo) 内


def test_naturals0():
    N = S.Naturals0  # 创建非负整数集合对象 N
    assert 0 in N  # 断言 0 在 N 中
    assert -1 not in N  # 断言 -1 不在 N 中
    assert next(iter(N)) == 0  # 断言 N 的迭代器的下一个值为 0
    assert not N.contains(oo)  # 断言 N 不包含无穷大
    assert N.contains(sin(x)) == Contains(sin(x), N)  # 断言 N 包含 sin(x) 是 Contains(sin(x), N) 的实例


def test_integers():
    Z = S.Integers  # 创建整数集合对象 Z
    assert 5 in Z  # 断言 5 在 Z 中
    assert -5 in Z  # 断言 -5 在 Z 中
    assert 5.5 not in Z  # 断言 5.5 不在 Z 中
    assert not Z.contains(oo)  # 断言 Z 不包含无穷大
    assert not Z.contains(-oo)  # 断言 Z 不包含负无穷大

    zi = iter(Z)  # 创建 Z 的迭代器 zi
    a, b, c, d = next(zi), next(zi), next(zi), next(zi)  # 从 zi 中获取四个值并赋给 a, b, c, d
    assert (a, b, c, d) == (0, 1, -1, 2)  # 断言 a, b, c, d 的值分别为 0, 1, -1, 2
    assert isinstance(a, Basic)  # 断言 a 是 Basic 类的实例

    assert Z.intersect(Interval(-5, 5)) == Range(-5, 6)  # 断言 Z 与区间 (-5, 5) 的交集为 Range(-5, 6)
    assert Z.intersect(Interval(-5, 5, True, True)) == Range(-4, 5)  # 断言 Z 与开区间 (-5, 5) 的交集为 Range(-4, 5)
    assert Z.intersect(Interval(5, S.Infinity)) == Range(5, S.Infinity)  # 断言 Z 与区间 [5, oo) 的交集为 Range(5, oo)
    assert Z.intersect(Interval.Lopen(5, S.Infinity)) == Range(6, S.Infinity)  # 断言 Z 与开区间 (5, oo) 的交集为 Range(6, oo)

    assert Z.inf is -oo  # 断言 Z 的下界为负无穷大
    assert Z.sup is oo  # 断言 Z 的上界为无穷大

    assert Z.boundary == Z  # 断言 Z 的边界等于 Z 本身
    assert Z.is_open == False  # 断言 Z 不是开集
    assert Z.is_closed == True  # 断言 Z 是闭集

    assert Z.as_relational(x) == And(Eq(floor(x), x), -oo < x, x < oo)  # 断言 Z 的关系表达式为 floor(x) = x 且 x 在 (-oo, oo) 内
def test_ImageSet():
    # 测试是否会引发 ValueError，期望函数抛出异常当 ImageSet 的第一个参数不是 Lambda 函数时
    raises(ValueError, lambda: ImageSet(x, S.Integers))
    
    # 测试 Lambda 函数返回常数时，ImageSet 是否正确返回这个常数的有限集
    assert ImageSet(Lambda(x, 1), S.Integers) == FiniteSet(1)
    
    # 测试 Lambda 函数返回变量时，ImageSet 是否正确返回这个变量的单元素集合
    assert ImageSet(Lambda(x, y), S.Integers) == {y}
    
    # 测试 Lambda 函数返回常数时，ImageSet 是否正确处理空集
    assert ImageSet(Lambda(x, 1), S.EmptySet) == S.EmptySet
    
    # 创建一个包含 log(2)/pi 的有限集和整数集的交集
    empty = Intersection(FiniteSet(log(2)/pi), S.Integers)
    
    # 测试 ImageSet 函数在 issue #17471 中的问题
    assert unchanged(ImageSet, Lambda(x, 1), empty)
    
    # 创建一个由自然数的平方组成的 ImageSet
    squares = ImageSet(Lambda(x, x**2), S.Naturals)
    
    # 测试是否 4 在 squares 中
    assert 4 in squares
    
    # 测试是否 5 不在 squares 中
    assert 5 not in squares
    
    # 测试有限集和 squares 的交集是否包含 1, 4, 9
    assert FiniteSet(*range(10)).intersect(squares) == FiniteSet(1, 4, 9)
    
    # 测试 squares 与区间 (0, 10) 的交集中是否不包含 16
    assert 16 not in squares.intersect(Interval(0, 10))
    
    # 创建 squares 的迭代器
    si = iter(squares)
    
    # 测试迭代器 si 中的前四个值是否分别是 1, 4, 9, 16
    a, b, c, d = next(si), next(si), next(si), next(si)
    assert (a, b, c, d) == (1, 4, 9, 16)
    
    # 创建一个由自然数的倒数组成的 ImageSet
    harmonics = ImageSet(Lambda(x, 1/x), S.Naturals)
    
    # 测试 1/5 是否在 harmonics 中
    assert Rational(1, 5) in harmonics
    
    # 测试 0.25 是否在 harmonics 中
    assert Rational(.25) in harmonics
    
    # 测试 harmonics 是否包含 0.25，使用 evaluate=False 来验证
    assert harmonics.contains(.25) == Contains(
        0.25, ImageSet(Lambda(x, 1/x), S.Naturals), evaluate=False)
    
    # 测试 0.3 是否不在 harmonics 中
    assert Rational(.3) not in harmonics
    
    # 测试 (1, 2) 是否不在 harmonics 中
    assert (1, 2) not in harmonics
    
    # 测试 harmonics 是否可迭代
    assert harmonics.is_iterable
    
    # 测试 imageset 函数的结果是否正确
    assert imageset(x, -x, Interval(0, 1)) == Interval(-1, 0)
    
    # 测试 ImageSet(Lambda(x, x**2), Interval(0, 2)) 的结果是否正确计算为 Interval(0, 4)
    assert ImageSet(Lambda(x, x**2), Interval(0, 2)).doit() == Interval(0, 4)
    
    # 测试 ImageSet(Lambda((x, y), 2*x), {4}, {3}) 的结果是否正确计算为 FiniteSet(8)
    assert ImageSet(Lambda((x, y), 2*x), {4}, {3}).doit() == FiniteSet(8)
    
    # 测试 ImageSet(Lambda((x, y), x+y), {1, 2, 3}, {10, 20, 30}) 的结果是否正确计算为 FiniteSet(11, 12, 13, 21, 22, 23, 31, 32, 33)
    assert (ImageSet(Lambda((x, y), x+y), {1, 2, 3}, {10, 20, 30}).doit() ==
                FiniteSet(11, 12, 13, 21, 22, 23, 31, 32, 33))
    
    # 创建一个三个区间的乘积集合
    c = Interval(1, 3) * Interval(1, 3)
    
    # 测试 Tuple(2, 6) 是否在 ImageSet(Lambda(((x, y),), (x, 2*y)), c) 中
    assert Tuple(2, 6) in ImageSet(Lambda(((x, y),), (x, 2*y)), c)
    
    # 测试 Tuple(2, S.Half) 是否在 ImageSet(Lambda(((x, y),), (x, 1/y)), c) 中
    assert Tuple(2, S.Half) in ImageSet(Lambda(((x, y),), (x, 1/y)), c)
    
    # 测试 Tuple(2, -2) 是否不在 ImageSet(Lambda(((x, y),), (x, y**2)), c) 中
    assert Tuple(2, -2) not in ImageSet(Lambda(((x, y),), (x, y**2)), c)
    
    # 测试 Tuple(2, -2) 是否在 ImageSet(Lambda(((x, y),), (x, -2)), c) 中
    assert Tuple(2, -2) in ImageSet(Lambda(((x, y),), (x, -2)), c)
    
    # 创建一个三个区间的乘积集合 c3
    c3 = ProductSet(Interval(3, 7), Interval(8, 11), Interval(5, 9))
    
    # 测试 Tuple(8, 3, 9) 是否在 ImageSet(Lambda(((t, y, x),), (y, t, x)), c3) 中
    assert Tuple(8, 3, 9) in ImageSet(Lambda(((t, y, x),), (y, t, x)), c3)
    
    # 测试 Tuple(Rational(1, 8), 3, 9) 是否在 ImageSet(Lambda(((t, y, x),), (1/y, t, x)), c3) 中
    assert Tuple(Rational(1, 8), 3, 9) in ImageSet(Lambda(((t, y, x),), (1/y, t, x)), c3)
    
    # 测试 2/pi 是否不在 ImageSet(Lambda(((x, y),), 2/x), c) 中
    assert 2/pi not in ImageSet(Lambda(((x, y),), 2/x), c)
    
    # 测试 2/S(100) 是否不在 ImageSet(Lambda(((x, y),), 2/x), c) 中
    assert 2/S(100) not in ImageSet(Lambda(((x, y),), 2/x), c)
    
    # 测试 Rational(2, 3) 是否在 ImageSet(Lambda(((x, y),), 2/x), c) 中
    assert Rational(2, 3) in ImageSet(Lambda(((x, y),), 2/x), c)
    
    # 测试 imageset 函数的结果 S1 的 base_pset 和 base_sets 是否正确
    S1 = imageset(lambda x, y: x + y, S.Integers, S.Naturals)
    assert S1.base_pset == ProductSet(S.Integers, S.Naturals)
    assert S1.base_sets == (S.Integers, S.Naturals)
    
    # 测试 ImageSet(Lambda(x, x**2), {1, 2, 3}) 中传递集合而不是有限集合是否不会引发异常
    assert unchanged(ImageSet, Lambda(x, x**2), {1, 2, 3})
    
    # 创建一个由元组组成的 ImageSet
    S2 = ImageSet(Lambda(((x, y),), x+y), {(1, 2), (3, 4)})
    
    # 测试 3 是否在 S2.doit() 的结果中
    assert 3 in S2.doit()
    
    # FIXME: 这个测试目前无法工作:
    #assert 3 in S2
    
    # 测试 S2._contains(3) 是否为 None
    assert S2._contains(3) is None
    
    # 测试 ImageSet(Lambda(x, x**2), 1) 是否引发 TypeError
    raises(TypeError, lambda: ImageSet(Lambda(x, x**2), 1))


def test_image_is_ImageSet():
    # 创建一个半圆的图像集合对象 `halfcircle`，其包含从左到右 L 长度的一半圆弧的点集。
    halfcircle = ImageSet(L, Interval(0, 1)*Interval(0, pi))
    
    # 断言：点 (1, 0) 在 `halfcircle` 中。
    assert (1, 0) in halfcircle
    
    # 断言：点 (0, -1) 不在 `halfcircle` 中。
    assert (0, -1) not in halfcircle
    
    # 断言：原点 (0, 0) 在 `halfcircle` 中。
    assert (0, 0) in halfcircle
    
    # 断言：`halfcircle` 对象中不包含特定点 (r, 0)。
    assert halfcircle._contains((r, 0)) is None
    
    # 断言：`halfcircle` 对象不可迭代。
    assert not halfcircle.is_iterable
@XFAIL
# 声明一个测试函数，用来测试半圆失败的情况
def test_halfcircle_fail():
    # 声明符号变量 r, th，限定为实数
    r, th = symbols('r, theta', real=True)
    # 创建一个 Lambda 函数 L，将 (r, th) 映射到 (r*cos(th), r*sin(th))
    L = Lambda(((r, th),), (r*cos(th), r*sin(th)))
    # 创建一个半圆 ImageSet，表示 r 在 [0, 1] 而 th 在 [0, pi] 之间的图像集合
    halfcircle = ImageSet(L, Interval(0, 1)*Interval(0, pi))
    # 断言 (r, 2*pi) 不在半圆图像集合中
    assert (r, 2*pi) not in halfcircle


def test_ImageSet_iterator_not_injective():
    # 创建一个 Lambda 函数 L，将 x 映射到 x - x % 2，即产生 0, 2, 2, 4, 4, 6, 6, ...
    L = Lambda(x, x - x % 2)
    # 创建一个偶数的 ImageSet，表示所有偶数集合
    evens = ImageSet(L, S.Naturals)
    # 创建 evens 的迭代器 i
    i = iter(evens)
    # 断言迭代器中依次取出的四个元素分别为 0, 2, 4, 6
    assert (next(i), next(i), next(i), next(i)) == (0, 2, 4, 6)


def test_inf_Range_len():
    # 断言对 Range(0, oo, 2) 的长度抛出 ValueError 异常
    raises(ValueError, lambda: len(Range(0, oo, 2)))
    # 断言 Range(0, oo, 2) 的大小为无穷大
    assert Range(0, oo, 2).size is S.Infinity
    # 断言 Range(0, -oo, -2) 的大小为无穷大
    assert Range(0, -oo, -2).size is S.Infinity
    # 断言 Range(oo, 0, -2) 的大小为无穷大
    assert Range(oo, 0, -2).size is S.Infinity
    # 断言 Range(-oo, 0, 2) 的大小为无穷大
    assert Range(-oo, 0, 2).size is S.Infinity


def test_Range_set():
    # 创建一个空的 Range 对象
    empty = Range(0)

    # 断言 Range(5) 等于 Range(0, 5) 且等于 Range(0, 5, 1)
    assert Range(5) == Range(0, 5) == Range(0, 5, 1)

    # 创建一个 Range 对象 r，从 10 到 20，步长为 2
    r = Range(10, 20, 2)
    # 断言 12 存在于 r 中
    assert 12 in r
    # 断言 8 不在 r 中
    assert 8 not in r
    # 断言 11 不在 r 中
    assert 11 not in r
    # 断言 30 不在 r 中
    assert 30 not in r

    # 断言 Range(0, 5) 的列表形式与内置 range(5) 的列表形式相同
    assert list(Range(0, 5)) == list(range(5))
    # 断言 Range(5, 0, -1) 的列表形式与内置 range(5, 0, -1) 的列表形式相同
    assert list(Range(5, 0, -1)) == list(range(5, 0, -1))

    # 断言 Range(5, 15) 的上界为 14
    assert Range(5, 15).sup == 14
    # 断言 Range(5, 15) 的下界为 5
    assert Range(5, 15).inf == 5
    # 断言 Range(15, 5, -1) 的上界为 15
    assert Range(15, 5, -1).sup == 15
    # 断言 Range(15, 5, -1) 的下界为 6
    assert Range(15, 5, -1).inf == 6
    # 断言 Range(10, 67, 10) 的上界为 60
    assert Range(10, 67, 10).sup == 60

    # 断言 Range(10, 38, 10) 的长度为 3
    assert len(Range(10, 38, 10)) == 3

    # 断言 Range(0, 0, 5) 等于 empty
    assert Range(0, 0, 5) == empty
    # 断言 Range(oo, oo, 1) 等于 empty
    assert Range(oo, oo, 1) == empty
    # 断言 Range(oo, 1, 1) 等于 empty
    assert Range(oo, 1, 1) == empty
    # 断言 Range(-oo, 1, -1) 等于 empty
    assert Range(-oo, 1, -1) == empty
    # 断言 Range(1, oo, -1) 等于 empty
    assert Range(1, oo, -1) == empty
    # 断言 Range(1, -oo, 1) 等于 empty
    assert Range(1, -oo, 1) == empty
    # 断言 Range(1, -4, oo) 等于 empty
    assert Range(1, -4, oo) == empty

    # 声明一个正的符号变量 ip
    ip = symbols('ip', positive=True)
    # 断言 Range(0, ip, -1) 等于 empty
    assert Range(0, ip, -1) == empty
    # 断言 Range(0, -ip, 1) 等于 empty
    assert Range(0, -ip, 1) == empty

    # 断言 Range(1, -4, -oo) 等于 Range(1, 2)
    assert Range(1, -4, -oo) == Range(1, 2)
    # 断言 Range(1, 4, oo) 等于 Range(1, 2)
    assert Range(1, 4, oo) == Range(1, 2)

    # 断言 Range(-oo, oo).size 等于 oo
    assert Range(-oo, oo).size == oo
    # 断言 Range(oo, -oo, -1).size 等于 oo
    assert Range(oo, -oo, -1).size == oo

    # 断言抛出 ValueError 异常，因为步长为 2 的 Range(-oo, oo, 2) 无效
    raises(ValueError, lambda: Range(-oo, oo, 2))
    # 断言抛出 ValueError 异常，因为参数 x, pi, y 无效
    raises(ValueError, lambda: Range(x, pi, y))
    # 断言抛出 ValueError 异常，因为步长为 0 无效
    raises(ValueError, lambda: Range(x, y, 0))

    # 断言 5 存在于 Range(0, oo, 5) 中
    assert 5 in Range(0, oo, 5)
    # 断言 -5 存在于 Range(-oo, 0, 5) 中
    assert -5 in Range(-oo, 0, 5)
    # 断言 oo 不在 Range(0, oo) 中
    assert oo not in Range(0, oo)

    # 声明一个非整数符号变量 ni
    ni = symbols('ni', integer=False)
    # 断言 ni 不在 Range(oo) 中
    assert ni not in Range(oo)

    # 声明一个整数或无穷的符号变量 u
    u = symbols('u', integer=None)
    # 断言 Range(oo) 包含 u
    assert Range(oo).contains(u) is not False

    # 声明一个无穷大的符号变量 inf
    inf = symbols('inf', infinite=True)
    # 断言 inf 不在 Range(-oo, oo) 中
    assert inf not in Range(-oo, oo)

    # 断言 Range(0, oo, 2)[-1]
    # 断言：Range(-1, 10, 1) 与 S.Integers 的交集应该等于 Range(-1, 10, 1)
    assert Range(-1, 10, 1).intersect(S.Integers) == Range(-1, 10, 1)
    # 断言：Range(-1, 10, 1) 与 S.Naturals 的交集应该等于 Range(1, 10, 1)
    assert Range(-1, 10, 1).intersect(S.Naturals) == Range(1, 10, 1)
    # 断言：Range(-1, 10, 1) 与 S.Naturals0 的交集应该等于 Range(0, 10, 1)
    assert Range(-1, 10, 1).intersect(S.Naturals0) == Range(0, 10, 1)

    # 测试切片操作
    # 断言：从 Range(1, 10, 1) 中取索引为 5 的元素应该是 6
    assert Range(1, 10, 1)[5] == 6
    # 断言：从 Range(1, 12, 2) 中取索引为 5 的元素应该是 11
    assert Range(1, 12, 2)[5] == 11
    # 断言：从 Range(1, 10, 1) 中取倒数第一个元素应该是 9
    assert Range(1, 10, 1)[-1] == 9
    # 断言：从 Range(1, 10, 3) 中取倒数第一个元素应该是 7
    assert Range(1, 10, 3)[-1] == 7
    # 断言：Range(oo,0,-1) 中切片 [1:3:0] 应该引发 ValueError 异常
    raises(ValueError, lambda: Range(oo,0,-1)[1:3:0])
    # 断言：从 Range(oo, 0, -1) 中取切片 [:1] 应该引发 ValueError 异常
    raises(ValueError, lambda: Range(oo, 0, -1)[:1])
    # 断言：从 Range(1, oo) 中取索引为 -2 的元素应该引发 ValueError 异常
    raises(ValueError, lambda: Range(1, oo)[-2])
    # 断言：从 Range(-oo, 1) 中取索引为 2 的元素应该引发 ValueError 异常
    raises(ValueError, lambda: Range(-oo, 1)[2])
    # 断言：从 Range(10) 中取索引为 -20 的元素应该引发 IndexError 异常
    raises(IndexError, lambda: Range(10)[-20])
    # 断言：从 Range(10) 中取索引为 20 的元素应该引发 IndexError 异常
    raises(IndexError, lambda: Range(10)[20])
    # 断言：从 Range(2, -oo, -2) 中切片 [2:2:0] 应该引发 ValueError 异常
    raises(ValueError, lambda: Range(2, -oo, -2)[2:2:0])
    # 断言：Range(2, -oo, -2) 中切片 [2:2:2] 应该等于 empty
    assert Range(2, -oo, -2)[2:2:2] == empty
    # 断言：Range(2, -oo, -2) 中切片 [:2:2] 应该等于 Range(2, -2, -4)
    assert Range(2, -oo, -2)[:2:2] == Range(2, -2, -4)
    # 断言：从 Range(-oo, 4, 2) 中切片 [:2:2] 应该引发 ValueError 异常
    raises(ValueError, lambda: Range(-oo, 4, 2)[:2:2])
    # 断言：Range(-oo, 4, 2) 中切片 [::-2] 应该等于 Range(2, -oo, -4)
    assert Range(-oo, 4, 2)[::-2] == Range(2, -oo, -4)
    # 断言：从 Range(-oo, 4, 2) 中切片 [::2] 应该引发 ValueError 异常
    raises(ValueError, lambda: Range(-oo, 4, 2)[::2])
    # 断言：Range(oo, 2, -2) 中切片 [::] 应该等于 Range(oo, 2, -2)
    assert Range(oo, 2, -2)[::] == Range(oo, 2, -2)
    # 断言：Range(-oo, 4, 2) 中切片 [:-2:-2] 应该等于 Range(2, 0, -4)
    assert Range(-oo, 4, 2)[:-2:-2] == Range(2, 0, -4)
    # 断言：Range(-oo, 4, 2) 中切片 [:-2:2] 应该等于 Range(-oo, 0, 4)
    assert Range(-oo, 4, 2)[:-2:2] == Range(-oo, 0, 4)
    # 断言：从 Range(-oo, 4, 2) 中切片 [:0:-2] 应该引发 ValueError 异常
    raises(ValueError, lambda: Range(-oo, 4, 2)[:0:-2])
    # 断言：从 Range(-oo, 4, 2) 中切片 [:2:-2] 应该引发 ValueError 异常
    raises(ValueError, lambda: Range(-oo, 4, 2)[:2:-2])
    # 断言：Range(-oo, 4, 2) 中切片 [-2::-2] 应该等于 Range(0, -oo, -4)
    assert Range(-oo, 4, 2)[-2::-2] == Range(0, -oo, -4)
    # 断言：从 Range(-oo, 4, 2) 中切片 [-2:0:-2] 应该引发 ValueError 异常
    raises(ValueError, lambda: Range(-oo, 4, 2)[-2:0:-2])
    # 断言：从 Range(-oo, 4, 2) 中切片 [0::2] 应该引发 ValueError 异常
    raises(ValueError, lambda: Range(-oo, 4, 2)[0::2])
    # 断言：Range(oo, 2, -2) 中切片 [0::] 应该等于 Range(oo, 2, -2)
    assert Range(oo, 2, -2)[0::] == Range(oo, 2, -2)
    # 断言：从 Range(-oo, 4, 2) 中切片 [0:-2:2] 应该引发 ValueError 异常
    raises(ValueError, lambda: Range(-oo, 4, 2)[0:-2:2])
    # 断言：Range(oo, 2, -2) 中切片 [0:-2:] 应该等于 Range(oo, 6, -2)
    assert Range(oo, 2, -2)[0:-2:] == Range(oo, 6, -2)
    # 断言：从 Range(oo, 2, -2) 中切片 [0:2:] 应该引发 ValueError 异常
    raises(ValueError, lambda: Range(oo, 2, -2)[0:2:])
    # 断言：从 Range(-oo, 4, 2) 中切片 [2::-1] 应该引发 ValueError 异常
    raises(ValueError, lambda: Range(-oo, 4, 2)[2::-1])
    # 断言：Range(-oo, 4, 2) 中切片 [-2::2] 应该等于 Range(0, 4, 4)
    assert Range(-oo, 4, 2)[-2::2] == Range(0, 4, 4)
    # 断言：Range(oo, 0, -2) 中切片 [-10:0:2] 应该等于 empty
    assert Range(oo, 0, -2)[-10:0:2] == empty
    # 断言：从 Range(oo, 0, -2) 中取索引为 0 的元素应该引发 ValueError 异常
    raises(ValueError, lambda: Range(oo, 0, -2)[0])
    # 断言：从 Range
    # 遍历两个 Range 对象的迭代器：
    # - 第一个 Range 对象从 1 到 9（不包括 10）
    # - 第二个 Range 对象从 1 到 9，步长为 2
    for R in [
            Range(1, 10),
            Range(1, 10, 2),
        ]:
        # 将 Range 对象 R 转换为列表 r
        r = list(R)
        # 使用 itertools.product 生成器生成三元组 (a, b, c)，其中 a 和 b 取自 AB 列表，c 取自 [-3, -1, None, 1, 3]
        for a, b, c in itertools.product(AB, AB, [-3, -1, None, 1, 3]):
            # 遍历两次，一次正序，一次倒序
            for reverse in range(2):
                # 将列表 r 反转
                r = list(reversed(r))
                # 将 Range 对象 R 反转
                R = R.reversed
                # 使用 Range 对象 R 的切片方法获取结果列表 result
                result = list(R[a:b:c])
                # 获取预期的结果列表 ans
                ans = r[a:b:c]
                # 格式化输出测试结果的文本
                txt = ('\n%s[%s:%s:%s] = %s -> %s' % (
                R, a, b, c, result, ans))
                # 检查结果是否符合预期
                check = ans == result
                # 使用断言确保 check 为 True，否则输出 txt 提示错误
                assert check, txt

    # 断言两个相同参数的 Range 对象的边界相同
    assert Range(1, 10, 1).boundary == Range(1, 10, 1)

    # 遍历两个 Range 对象：
    # - 第一个 Range 对象从 1 到 9，步长为 2
    # - 第二个 Range 对象从 1 到正无穷，步长为 2
    for r in (Range(1, 10, 2), Range(1, oo, 2)):
        # 获取 r 对象的反转对象 rev
        rev = r.reversed
        # 断言 r 对象和 rev 对象的 inf（起始值）和 sup（结束值）相同
        assert r.inf == rev.inf and r.sup == rev.sup
        # 断言 r 对象的步长等于 rev 对象步长的负值
        assert r.step == -rev.step

    # 将内置的 range 函数赋值给 builtin_range
    builtin_range = range

    # 使用 lambda 表达式测试 TypeError 是否被 Range 对象正确引发
    raises(TypeError, lambda: Range(builtin_range(1)))
    # 断言将内置的 range 对象转换为 Range 对象后得到的结果正确
    assert S(builtin_range(10)) == Range(10)
    assert S(builtin_range(1000000000000)) == Range(1000000000000)

    # 测试 Range 对象的 as_relational 方法
    assert Range(1, 4).as_relational(x) == (x >= 1) & (x <= 3)  & Eq(Mod(x, 1), 0)
    assert Range(oo, 1, -2).as_relational(x) == (x >= 3) & (x < oo)  & Eq(Mod(x + 1, -2), 0)
def test_Range_symbolic():
    # symbolic Range
    # 创建一个符号范围对象 xr，起始于 x，结束于 x + 4，步长为 5
    xr = Range(x, x + 4, 5)
    # 创建一个符号范围对象 sr，起始于 x，结束于 y，步长为 t
    sr = Range(x, y, t)
    # 创建一个整数符号 i
    i = Symbol('i', integer=True)
    # 创建一个正整数符号 ip
    ip = Symbol('i', integer=True, positive=True)
    # 创建一个符号范围对象 ipr，起始于 1，步长为 1，结束于 ip-1
    ipr = Range(ip)
    # 创建一个符号范围对象 inr，起始于 0，步长为 -1，结束于 -ip
    inr = Range(0, -ip, -1)
    # 创建一个符号范围对象 ir，起始于 i，结束于 i + 19，步长为 2
    ir = Range(i, i + 19, 2)
    # 创建一个符号范围对象 ir2，起始于 i，结束于 i*8，步长为 3*i
    ir2 = Range(i, i*8, 3*i)
    # 创建一个整数符号 i
    i = Symbol('i', integer=True)
    # 创建一个无穷大符号 inf
    inf = symbols('inf', infinite=True)
    # 抛出 ValueError 异常，因为无法生成以无穷大为起点的范围
    raises(ValueError, lambda: Range(inf))
    # 抛出 ValueError 异常，因为无法生成以无穷大为起点，步长为 -1 的范围
    raises(ValueError, lambda: Range(inf, 0, -1))
    # 抛出 ValueError 异常，因为无法生成步长为 1 的无穷大范围
    raises(ValueError, lambda: Range(inf, inf, 1))
    # 抛出 ValueError 异常，因为无法生成以 1 为起点且步长为无穷大的范围
    raises(ValueError, lambda: Range(1, 1, inf))
    # 验证 xr 的参数为 (x, x + 5, 5)
    assert xr.args == (x, x + 5, 5)
    # 验证 sr 的参数为 (x, y, t)
    assert sr.args == (x, y, t)
    # 验证 ir 的参数为 (i, i + 20, 2)
    assert ir.args == (i, i + 20, 2)
    # 验证 ir2 的参数为 (i, 10*i, 3*i)
    assert ir2.args == (i, 10*i, 3*i)
    # 抛出 ValueError 异常，因为无法反转一个符号范围对象 xr
    raises(ValueError, lambda: xr.reversed)
    # 抛出 ValueError 异常，因为无法反转一个符号范围对象 sr
    raises(ValueError, lambda: sr.reversed)
    # 验证 ipr 反转后的参数为 (ip - 1, -1, -1)
    assert ipr.reversed.args == (ip - 1, -1, -1)
    # 验证 inr 反转后的参数为 (-ip + 1, 1, 1)
    assert inr.reversed.args == (-ip + 1, 1, 1)
    # 验证 ir 反转后的参数为 (i + 18, i - 2, -2)
    assert ir.reversed.args == (i + 18, i - 2, -2)
    # 验证 ir2 反转后的参数为 (7*i, -2*i, -3*i)
    assert ir2.reversed.args == (7*i, -2*i, -3*i)
    # 验证无穷大 inf 不在 sr 范围内
    assert inf not in sr
    # 验证无穷大 inf 不在 ir 范围内
    assert inf not in ir
    # 验证 0 在 ipr 范围内
    assert 0 in ipr
    # 验证 0 在 inr 范围内
    assert 0 in inr
    # 抛出 TypeError 异常，因为无法在 ipr 范围中使用整数判定
    raises(TypeError, lambda: 1 in ipr)
    # 抛出 TypeError 异常，因为无法在 inr 范围中使用整数判定
    raises(TypeError, lambda: -1 in inr)
    # 验证 0.1 不在 sr 范围内
    assert .1 not in sr
    # 验证 0.1 不在 ir 范围内
    assert .1 not in ir
    # 验证 i + 1 不在 ir 范围内
    assert i + 1 not in ir
    # 验证 i + 2 在 ir 范围内
    assert i + 2 in ir
    # 抛出 TypeError 异常，因为无法在 xr 中使用符号判定
    raises(TypeError, lambda: x in xr)  # XXX is this what contains is supposed to do?
    # 抛出 TypeError 异常，因为无法在 sr 中使用整数判定
    raises(TypeError, lambda: 1 in sr)  # XXX is this what contains is supposed to do?
    # 抛出 ValueError 异常，因为无法对 xr 进行迭代
    raises(ValueError, lambda: next(iter(xr)))
    # 抛出 ValueError 异常，因为无法对 sr 进行迭代
    raises(ValueError, lambda: next(iter(sr)))
    # 验证 ir 迭代后的第一个值为 i
    assert next(iter(ir)) == i
    # 验证 ir2 迭代后的第一个值为 i
    assert next(iter(ir2)) == i
    # 验证 sr 与 S.Integers 的交集为 sr
    assert sr.intersect(S.Integers) == sr
    # 验证 sr 与 {x} 的交集为 Intersection({x}, sr)
    assert sr.intersect(FiniteSet(x)) == Intersection({x}, sr)
    # 抛出 ValueError 异常，因为无法对 sr 的切片进行操作
    raises(ValueError, lambda: sr[:2])
    # 抛出 ValueError 异常，因为无法对 xr 的索引进行操作
    raises(ValueError, lambda: xr[0])
    # 抛出 ValueError 异常，因为无法对 sr 的索引进行操作
    raises(ValueError, lambda: sr[0])
    # 验证 ir 的长度为 ir.size 为 10
    assert len(ir) == ir.size == 10
    # 验证 ir2 的长度为 ir2.size 为 3
    assert len(ir2) == ir2.size == 3
    # 抛出 ValueError 异常，因为无法获取 xr 的长度
    raises(ValueError, lambda: len(xr))
    # 抛出 ValueError 异常，因为无法获取 xr 的 size
    raises(ValueError, lambda: xr.size)
    # 抛出 ValueError 异常，因为无法获取 sr 的长度
    raises(ValueError, lambda: len(sr))
    # 抛出 ValueError 异常，因为无法获取 sr 的 size
    raises(ValueError, lambda: sr.size)
    # 验证 Range(0) 的布尔值为 False
    assert bool(Range(0)) == False
    # 验证 xr 的布尔值为 True
    assert bool(xr)
    # 验证 ir 的布尔值为 True
    assert bool(ir)
    # 验证 ipr 的布尔值为 True
    assert bool(ipr)
    # 验证 inr 的布尔值为 True
    assert bool(inr)
    # 抛出 ValueError 异常，因为无法获取 sr 的布尔值
    raises(ValueError, lambda: bool(sr))
    # 抛出 ValueError 异常，因为无法获取 ir2 的布尔值
    raises(ValueError, lambda: bool(ir2))
    # 抛出 ValueError 异常，因为无法获取 xr 的 inf 属性
    raises(ValueError, lambda: xr.inf)
    # 抛出 ValueError 异常，因为无法获取 sr 的 inf 属性
    raises(ValueError, lambda: sr.inf)
    # 验证 ipr 的 inf 属性为 0
    assert ipr.inf == 0
    # 验证 inr
    # 确保ir的最后一个元素等于i + 18
    assert ir[-1] == i + 18
    # 确保ir2的前两个元素等于Range(i, 7*i, 3*i)的前两个元素
    assert ir2[:2] == Range(i, 7*i, 3*i)
    # 确保ir2的第一个元素等于i
    assert ir2[0] == i
    # 确保ir2的倒数第二个元素等于4*i
    assert ir2[-2] == 4*i
    # 确保ir2的最后一个元素等于7*i
    assert ir2[-1] == 7*i
    # 测试Range(i)生成的Range对象不支持负索引
    raises(ValueError, lambda: Range(i)[-1])
    # 确保ipr的第一个元素等于ipr.inf等于0
    assert ipr[0] == ipr.inf == 0
    # 确保ipr的最后一个元素等于ipr.sup等于ip - 1
    assert ipr[-1] == ipr.sup == ip - 1
    # 确保inr的第一个元素等于inr.sup等于0
    assert inr[0] == inr.sup == 0
    # 确保inr的最后一个元素等于inr.inf等于-ip + 1
    assert inr[-1] == inr.inf == -ip + 1
    # 测试ipr对象不支持负索引
    raises(ValueError, lambda: ipr[-2])
    # 确保ir对象的最小值为i
    assert ir.inf == i
    # 确保ir对象的最大值为i + 18
    assert ir.sup == i + 18
    # 测试ir对象不支持inf方法
    raises(ValueError, lambda: Range(i).inf)
    # 测试ir对象转换为关系表达式的准确性
    assert ir.as_relational(x) == ((x >= i) & (x <= i + 18) &
        Eq(Mod(-i + x, 2), 0))
    # 测试ir2对象转换为关系表达式的准确性
    assert ir2.as_relational(x) == Eq(
        Mod(-i + x, 3*i), 0) & (((x >= i) & (x <= 7*i) & (3*i >= 1)) |
        ((x <= i) & (x >= 7*i) & (3*i <= -1)))
    # 测试Range(i, i + 1)对象转换为关系表达式的准确性
    assert Range(i, i + 1).as_relational(x) == Eq(x, i)
    # 测试sr对象转换为关系表达式的准确性
    assert sr.as_relational(z) == Eq(
        Mod(t, 1), 0) & Eq(Mod(x, 1), 0) & Eq(Mod(-x + z, t), 0
        ) & (((z >= x) & (z <= -t + y) & (t >= 1)) |
        ((z <= x) & (z >= -t + y) & (t <= -1)))
    # 测试xr对象转换为关系表达式的准确性
    assert xr.as_relational(z) == Eq(z, x) & Eq(Mod(x, 1), 0)
    # 测试xr对象转换为关系表达式时，符号冲突的情况（用户想要的情况，但必须是整数）
    assert xr.as_relational(x) == Eq(Mod(x, 1), 0)
    # 测试contains()方法对符号值的处理（问题#18146）
    e = Symbol('e', integer=True, even=True)
    o = Symbol('o', integer=True, odd=True)
    assert Range(5).contains(i) == And(i >= 0, i <= 4)
    assert Range(1).contains(i) == Eq(i, 0)
    assert Range(-oo, 5, 1).contains(i) == (i <= 4)
    assert Range(-oo, oo).contains(i) == True
    assert Range(0, 8, 2).contains(i) == Contains(i, Range(0, 8, 2))
    assert Range(0, 8, 2).contains(e) == And(e >= 0, e <= 6)
    assert Range(0, 8, 2).contains(2*i) == And(2*i >= 0, 2*i <= 6)
    assert Range(0, 8, 2).contains(o) == False
    assert Range(1, 9, 2).contains(e) == False
    assert Range(1, 9, 2).contains(o) == And(o >= 1, o <= 7)
    assert Range(8, 0, -2).contains(o) == False
    assert Range(9, 1, -2).contains(o) == And(o >= 3, o <= 9)
    assert Range(-oo, 8, 2).contains(i) == Contains(i, Range(-oo, 8, 2))
# 定义测试函数 test_range_range_intersection，用于测试 Range 类的区间交集功能
def test_range_range_intersection():
    # 遍历以下参数组合进行测试
    for a, b, r in [
            (Range(0), Range(1), S.EmptySet),  # 当一个区间为 [0]，另一个为 [1]，它们的交集为空集
            (Range(3), Range(4, oo), S.EmptySet),  # 区间 [3] 和 [4, oo) 的交集为空集
            (Range(3), Range(-3, -1), S.EmptySet),  # 区间 [3] 和 [-3, -1) 的交集为空集
            (Range(1, 3), Range(0, 3), Range(1, 3)),  # 区间 [1, 3) 和 [0, 3) 的交集为 [1, 3)
            (Range(1, 3), Range(1, 4), Range(1, 3)),  # 区间 [1, 3) 和 [1, 4) 的交集为 [1, 3)
            (Range(1, oo, 2), Range(2, oo, 2), S.EmptySet),  # 区间 [1, oo) 步长为 2 和 [2, oo) 步长为 2 的交集为空集
            (Range(0, oo, 2), Range(oo), Range(0, oo, 2)),  # 区间 [0, oo) 步长为 2 和 [oo) 的交集为 [0, oo) 步长为 2
            (Range(0, oo, 2), Range(100), Range(0, 100, 2)),  # 区间 [0, oo) 步长为 2 和 [0, 100) 的交集为 [0, 100, 2)
            (Range(2, oo, 2), Range(oo), Range(2, oo, 2)),  # 区间 [2, oo) 步长为 2 和 [oo) 的交集为 [2, oo) 步长为 2
            (Range(0, oo, 2), Range(5, 6), S.EmptySet),  # 区间 [0, oo) 步长为 2 和 [5, 6) 的交集为空集
            (Range(2, 80, 1), Range(55, 71, 4), Range(55, 71, 4)),  # 区间 [2, 80) 步长为 1 和 [55, 71) 步长为 4 的交集为 [55, 71, 4)
            (Range(0, 6, 3), Range(-oo, 5, 3), S.EmptySet),  # 区间 [0, 6) 步长为 3 和 [-oo, 5) 步长为 3 的交集为空集
            (Range(0, oo, 2), Range(5, oo, 3), Range(8, oo, 6)),  # 区间 [0, oo) 步长为 2 和 [5, oo) 步长为 3 的交集为 [8, oo) 步长为 6
            (Range(4, 6, 2), Range(2, 16, 7), S.EmptySet),  # 区间 [4, 6) 步长为 2 和 [2, 16) 步长为 7 的交集为空集
        ]:
        # 断言区间 a 和 b 的交集等于预期结果 r
        assert a.intersect(b) == r
        # 断言区间 a 和 b 的反向交集等于预期结果 r
        assert a.intersect(b.reversed) == r
        # 断言区间 a 的反向和 b 的交集等于预期结果 r
        assert a.reversed.intersect(b) == r
        # 断言区间 a 和 b 的反向交集的反向等于预期结果 r
        assert a.reversed.intersect(b.reversed) == r
        # 交换区间 a 和 b
        a, b = b, a
        # 重新进行相同的断言测试，保证交换后的结果依然正确
        assert a.intersect(b) == r
        assert a.intersect(b.reversed) == r
        assert a.reversed.intersect(b) == r
        assert a.reversed.intersect(b.reversed) == r


# 定义测试函数 test_range_interval_intersection，用于测试 Range 类和 Interval 类的交集功能
def test_range_interval_intersection():
    # 声明一个正数符号变量 p
    p = symbols('p', positive=True)
    # 断言区间 [3] 和 Interval(p, p + 2) 的交集是 Intersection 类型的对象
    assert isinstance(Range(3).intersect(Interval(p, p + 2)), Intersection)
    # 断言区间 [4] 和 Interval(0, 3) 的交集是 [4]
    assert Range(4).intersect(Interval(0, 3)) == Range(4)
    # 断言区间 [4] 和 Interval(-oo, oo) 的交集是 [4]
    assert Range(4).intersect(Interval(-oo, oo)) == Range(4)
    # 断言区间 [4] 和 Interval(1, oo) 的交集是 [1, 4]
    assert Range(4).intersect(Interval(1, oo)) == Range(1, 4)
    # 断言区间 [4] 和 Interval(1.1, oo) 的交集是 [2, 4]
    assert Range(4).intersect(Interval(1.1, oo)) == Range(2, 4)
    # 断言区间 [4] 和 Interval(0.1, 3) 的交集是 [1, 4]
    assert Range(4).intersect(Interval(0.1, 3)) == Range(1, 4)
    # 断言区间 [4] 和 Interval(0.1, 3.1) 的交集是 [1, 4]
    assert Range(4).intersect(Interval(0.1, 3.1)) == Range(1, 4)
    # 断言区间 [4] 和 Interval.open(0, 3) 的交集是 (0, 3)
    assert Range(4).intersect(Interval.open(0, 3)) == Range(1, 3)
    # 断言区间 [4] 和 Interval.open(0.1, 0.5) 的交集是 EmptySet
    assert Range(4).intersect(Interval.open(0.1, 0.5)) is S.EmptySet
    # 断言区间 [-1, 5] 和 Complexes 的交集是 [-1, 5]
    assert Interval(-1, 5).intersect(S.Complexes) == Interval(-1, 5)
    # 断言区间 [-1, 5] 和 Reals 的交集是 [-1, 5]
    assert Interval(-1, 5).intersect(S.Reals) == Interval(-1, 5)
    # 断言区间 [-1, 5] 和 Integers 的交集是 [-1, 5]
    assert Interval(-1, 5).intersect(S.Integers) == Range(-1, 6)
    # 断言区间 [-1, 5] 和 Naturals 的交集是 [1, 5]
    assert Interval(-1, 5).intersect(S.Naturals) == Range(1, 6)
    # 断言区间 [-1, 5] 和 Naturals0 的交集是 [0, 5]

    assert Interval(-1, 5).intersect(S.Naturals0) == Range(0, 6)

    # 断言空区间 [0] 和 Interval(0.2, 0.8) 的交集是 EmptySet
    assert Range(0).intersect(Interval(0.2, 0.8)) is S.EmptySet
    # 断言空区间 [0] 和 Interval(-oo, oo) 的交集是 EmptySet
    assert Range(0).intersect(Interval(-oo, oo)) is S.EmptySet


# 定义测试函数 test_range_is_finite_set，用于测试 Range 类的有限集判断功能
def test_range_is_finite_set():
    # 断言区间 [-100, 100] 是有限集
    assert Range(-100, 100).is_finite_set is True
    # 断言区间 [2, oo) 不是有限集
    assert Range(2, oo).is_finite_set is False
    # 断言区间 [-oo, 50] 不是有限集
    # 断言：检查指定范围是否为有限集合
    assert Range(-3, n + 7).is_finite_set is True
    
    # 断言：检查指定范围是否为有限集合
    assert Range(n, m).is_finite_set is True
    
    # 断言：检查指定范围是否为有限集合
    assert Range(n + m, m - n).is_finite_set is True
    
    # 断言：检查指定范围是否为有限集合
    assert Range(n, n + m + n).is_finite_set is True
    
    # 断言：检查指定范围是否为有限集合
    assert Range(n, oo).is_finite_set is False
    
    # 断言：检查指定范围是否为有限集合
    assert Range(-oo, n).is_finite_set is False
    
    # 断言：检查指定范围是否为有限集合
    assert Range(n, -oo).is_finite_set is True
    
    # 断言：检查指定范围是否为有限集合
    assert Range(oo, n).is_finite_set is True
# 定义测试函数，用于验证 Range 类的 is_iterable 方法
def test_Range_is_iterable():
    # 验证从 -100 到 100 的 Range 对象是否可迭代
    assert Range(-100, 100).is_iterable is True
    # 验证从 2 到正无穷的 Range 对象是否可迭代（应为 False）
    assert Range(2, oo).is_iterable is False
    # 验证从负无穷到 50 的 Range 对象是否可迭代（应为 False）
    assert Range(-oo, 50).is_iterable is False
    # 验证从负无穷到正无穷的 Range 对象是否可迭代（应为 False）
    assert Range(-oo, oo).is_iterable is False
    # 验证从正无穷到负无穷的 Range 对象是否可迭代
    assert Range(oo, -oo).is_iterable is True
    # 验证从 0 到 0 的 Range 对象是否可迭代
    assert Range(0, 0).is_iterable is True
    # 验证从正无穷到正无穷的 Range 对象是否可迭代
    assert Range(oo, oo).is_iterable is True
    # 验证从负无穷到负无穷的 Range 对象是否可迭代
    assert Range(-oo, -oo).is_iterable is True
    
    # 创建整数符号变量 n, m, p
    n = Symbol('n', integer=True)
    m = Symbol('m', integer=True)
    p = Symbol('p', integer=True, positive=True)
    
    # 验证从 n 到 n + 49 的 Range 对象是否可迭代
    assert Range(n, n + 49).is_iterable is True
    # 验证从 n 到 0 的 Range 对象是否可迭代（应为 False）
    assert Range(n, 0).is_iterable is False
    # 验证从 -3 到 n + 7 的 Range 对象是否可迭代（应为 False）
    assert Range(-3, n + 7).is_iterable is False
    # 验证从 -3 到 p + 7 的 Range 对象是否可迭代（应为 False，应改进 __iter__ 方法）
    assert Range(-3, p + 7).is_iterable is False
    # 验证从 n 到 m 的 Range 对象是否可迭代（应为 False）
    assert Range(n, m).is_iterable is False
    # 验证从 n + m 到 m - n 的 Range 对象是否可迭代（应为 False）
    assert Range(n + m, m - n).is_iterable is False
    # 验证从 n 到 n + m + n 的 Range 对象是否可迭代（应为 False）
    assert Range(n, n + m + n).is_iterable is False
    # 验证从 n 到正无穷的 Range 对象是否可迭代（应为 False）
    assert Range(n, oo).is_iterable is False
    # 验证从负无穷到 n 的 Range 对象是否可迭代（应为 False）
    assert Range(-oo, n).is_iterable is False
    
    # 创建符号变量 x
    x = Symbol('x')
    
    # 验证从 x 到 x + 49 的 Range 对象是否可迭代（应为 False）
    assert Range(x, x + 49).is_iterable is False
    # 验证从 x 到 0 的 Range 对象是否可迭代（应为 False）
    assert Range(x, 0).is_iterable is False
    # 验证从 -3 到 x + 7 的 Range 对象是否可迭代（应为 False）
    assert Range(-3, x + 7).is_iterable is False
    # 验证从 x 到 m 的 Range 对象是否可迭代（应为 False）
    assert Range(x, m).is_iterable is False
    # 验证从 x + m 到 m - x 的 Range 对象是否可迭代（应为 False）
    assert Range(x + m, m - x).is_iterable is False
    # 验证从 x 到 x + m + x 的 Range 对象是否可迭代（应为 False）
    assert Range(x, x + m + x).is_iterable is False
    # 验证从 x 到正无穷的 Range 对象是否可迭代（应为 False）
    assert Range(x, oo).is_iterable is False
    # 验证从负无穷到 x 的 Range 对象是否可迭代（应为 False）
    assert Range(-oo, x).is_iterable is False
    # 断言：检查负整数到0范围不为空
    assert not Range(-1, 0).is_empty
    
    # 断言：检查整数 i 范围是否为空，期望返回 None
    assert Range(i).is_empty is None
    
    # 断言：检查整数 n 范围是否为空，期望返回 True
    assert Range(n).is_empty
    
    # 断言：检查整数 p 范围是否为空，期望返回 False
    assert Range(p).is_empty is False
    
    # 断言：检查从 n 到 0 的范围是否为空，期望返回 False
    assert Range(n, 0).is_empty is False
    
    # 断言：检查从 n 到 p 的范围是否为空，期望返回 False
    assert Range(n, p).is_empty is False
    
    # 断言：检查从 p 到 n 的范围是否为空，期望返回 True
    assert Range(p, n).is_empty
    
    # 断言：检查从 n 到 -1 的范围是否为空，期望返回 None
    assert Range(n, -1).is_empty is None
    
    # 断言：检查从 p 到 n，步长为 -1 的范围是否为空，期望返回 False
    assert Range(p, n, -1).is_empty is False
# 定义测试函数，用于测试 S.Reals 中的数学性质
def test_Reals():
    # 断言 5 是实数集 S.Reals 的一个元素
    assert 5 in S.Reals
    # 断言 π 是实数集 S.Reals 的一个元素
    assert S.Pi in S.Reals
    # 断言 -sqrt(2) 是实数集 S.Reals 的一个元素
    assert -sqrt(2) in S.Reals
    # 断言 (2, 5) 不是实数集 S.Reals 的一个元素
    assert (2, 5) not in S.Reals
    # 断言 sqrt(-1) 不是实数集 S.Reals 的一个元素
    assert sqrt(-1) not in S.Reals
    # 断言 S.Reals 等于区间 (-∞, ∞)
    assert S.Reals == Interval(-oo, oo)
    # 断言 S.Reals 不等于区间 (0, ∞)
    assert S.Reals != Interval(0, oo)
    # 断言 S.Reals 是区间 (-∞, ∞) 的一个子集
    assert S.Reals.is_subset(Interval(-oo, oo))
    # 断言 S.Reals 与 Range(-∞, ∞) 的交集等于 Range(-∞, ∞)
    assert S.Reals.intersect(Range(-oo, oo)) == Range(-oo, oo)
    # 断言 S.ComplexInfinity 不是实数集 S.Reals 的一个元素
    assert S.ComplexInfinity not in S.Reals
    # 断言 S.NaN 不是实数集 S.Reals 的一个元素
    assert S.NaN not in S.Reals
    # 断言 x + S.ComplexInfinity 不是实数集 S.Reals 的一个元素

# 定义测试函数，用于测试 S.Complexes 中的数学性质
def test_Complex():
    # 断言 5 是复数集 S.Complexes 的一个元素
    assert 5 in S.Complexes
    # 断言 5 + 4*I 是复数集 S.Complexes 的一个元素
    assert 5 + 4*I in S.Complexes
    # 断言 π 是复数集 S.Complexes 的一个元素
    assert S.Pi in S.Complexes
    # 断言 -sqrt(2) 是复数集 S.Complexes 的一个元素
    assert -sqrt(2) in S.Complexes
    # 断言 -I 是复数集 S.Complexes 的一个元素
    assert -I in S.Complexes
    # 断言 sqrt(-1) 是复数集 S.Complexes 的一个元素
    assert sqrt(-1) in S.Complexes
    # 断言 S.Complexes 与 S.Reals 的交集等于 S.Reals
    assert S.Complexes.intersect(S.Reals) == S.Reals
    # 断言 S.Complexes 与 S.Reals 的并集等于 S.Complexes
    assert S.Complexes.union(S.Reals) == S.Complexes
    # 断言 S.Complexes 等于 ComplexRegion(S.Reals * S.Reals)
    assert S.Complexes == ComplexRegion(S.Reals * S.Reals)
    # 断言 (S.Complexes == ComplexRegion(Interval(1, 2) * Interval(3, 4))) 的结果为 False
    assert (S.Complexes == ComplexRegion(Interval(1, 2) * Interval(3, 4))) == False
    # 断言 str(S.Complexes) 的输出为 "Complexes"
    assert str(S.Complexes) == "Complexes"
    # 断言 repr(S.Complexes) 的输出为 "Complexes"
    assert repr(S.Complexes) == "Complexes"

# 定义函数 take，用于从可迭代对象中取出前 n 个元素并返回列表
def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(itertools.islice(iterable, n))

# 定义测试函数，用于测试整数集合与实数集合的交集
def test_intersections():
    # 断言 S.Integers 与 S.Reals 的交集等于 S.Integers
    assert S.Integers.intersect(S.Reals) == S.Integers
    # 断言 5 是 S.Integers 与 S.Reals 的交集的一个元素
    assert 5 in S.Integers.intersect(S.Reals)
    # 断言 5 是 S.Integers 与 S.Reals 的交集的一个元素
    assert 5 in S.Integers.intersect(S.Reals)
    # 断言 -5 不是 S.Naturals 与 S.Reals 的交集的一个元素
    assert -5 not in S.Naturals.intersect(S.Reals)
    # 断言 5.5 不是 S.Integers 与 S.Reals 的交集的一个元素
    assert 5.5 not in S.Integers.intersect(S.Reals)
    # 断言 5 是 S.Integers 与 区间 (3, ∞) 的交集的一个元素
    assert 5 in S.Integers.intersect(Interval(3, oo))
    # 断言 -5 是 S.Integers 与 区间 (-∞, 3) 的交集的一个元素
    assert -5 in S.Integers.intersect(Interval(-oo, 3))
    # 断言 S.Integers 与 区间 (3, ∞) 的交集中的所有元素都是整数
    assert all(x.is_Integer
               for x in take(10, S.Integers.intersect(Interval(3, oo))))

# 定义测试函数，用于测试无限索引集合
def test_infinitely_indexed_set_1():
    from sympy.abc import n, m
    # 断言 Lambda(n, n) 映射下的 S.Integers 的像与 Lambda(m, m) 映射下的 S.Integers 的像相等
    assert imageset(Lambda(n, n), S.Integers) == imageset(Lambda(m, m), S.Integers)
    # 断言 Lambda(n, 2*n) 映射下的 S.Integers 的像与 Lambda(m, 2*m + 1) 映射下的 S.Integers 的像的交集为空集
    assert imageset(Lambda(n, 2*n), S.Integers).intersect(
            imageset(Lambda(m, 2*m + 1), S.Integers)) is S.EmptySet
    # 断言 Lambda(n, 2*n) 映射下的 S.Integers 的像与 Lambda(n, 2*n + 1) 映射下的 S.Integers 的像的交集为空集
    assert imageset(Lambda(n, 2*n), S.Integers).intersect(
            imageset(Lambda(n, 2*n + 1), S.Integers)) is S.EmptySet
    # 断言 Lambda(m, 2*m) 映射下的 S.Integers 的像与 Lambda(n, 3*n) 映射下的 S.Integers 的像相等
    assert imageset(Lambda(m, 2*m), S.Integers).intersect(
                imageset(Lambda(n, 3*n), S.Integers)).dummy_eq(
            ImageSet(Lambda(t, 6*t), S.Integers))
    # 断言 imageset(x, x/2 + Rational(1, 3), S.Integers) 与 S.Integers 的交集为空集
    assert imageset(x, x/2 + Rational(1, 3), S.Integers).intersect(S.Integers) is S.EmptySet
    # 断言 imageset(x, x/2 + S.Half, S.Integers) 与 S.Integers 相等
    assert imageset(x, x/2 + S.Half, S.Integers).intersect(S.Integers) is S.Integers
    # 断言 ImageSet(Lambda(n, 5*n + 3), S.Integers) 与 S.Integers 的交集等于 ImageSet(Lambda(t, 6*t), S.Integers)
    S53 = ImageSet(Lambda(n, 5*n + 3), S.Integers)
    assert S53.intersect(S.Integers) == S53

# 定义测试函数，用于测试另一种无限索引集合
def test_infinitely_indexed_set_2():
    from sympy.abc import n
    a = Symbol('a', integer=True)
    # 断言 Lambda(n, n) 映射下的 S.Integers 的像与 Lambda(n, n + a) 映射下的 S.Integers 的像相等
    assert imageset(Lambda(n, n), S.Integers) == \
        imageset
    # 断言：对于 Lambda 函数 Lambda(n, -6*n)，应用于整数集 S.Integers 的图像集应该等于 Lambda 函数 Lambda(n, 6*n) 应用于整数集 S.Integers 的图像集
    assert imageset(Lambda(n, -6*n), S.Integers) == \
        ImageSet(Lambda(n, 6*n), S.Integers)
    
    # 断言：对于 Lambda 函数 Lambda(n, 2*n + pi)，应用于整数集 S.Integers 的图像集应该等于 Lambda 函数 Lambda(n, 2*n + pi - 2) 应用于整数集 S.Integers 的图像集
    assert imageset(Lambda(n, 2*n + pi), S.Integers) == \
        ImageSet(Lambda(n, 2*n + pi - 2), S.Integers)
# 定义测试函数 test_imageset_intersect_real
def test_imageset_intersect_real():
    # 导入 sympy 库中的符号 n
    from sympy.abc import n
    # 断言：使用 Lambda 表达式创建的 imageset 对象，应与 S.Integers 交集后得到 FiniteSet(-1, 1)
    assert imageset(Lambda(n, n + (n - 1)*(n + 1)*I), S.Integers).intersect(S.Reals) == FiniteSet(-1, 1)
    
    # 定义 im 为表达式 (n - 1)*(n + S.Half)
    im = (n - 1)*(n + S.Half)
    # 断言：使用 Lambda 表达式创建的 imageset 对象，应与 S.Integers 交集后得到 FiniteSet(1)
    assert imageset(Lambda(n, n + im*I), S.Integers).intersect(S.Reals) == FiniteSet(1)
    
    # 断言：使用 Lambda 表达式创建的 imageset 对象，应与 S.Naturals0 交集后得到 FiniteSet(1)
    assert imageset(Lambda(n, n + im*(n + 1)*I), S.Naturals0).intersect(S.Reals) == FiniteSet(1)
    
    # 断言：使用 Lambda 表达式创建的 imageset 对象，应与 S.Integers 交集后得到 ImageSet 对象
    assert imageset(Lambda(n, n/2 + im.expand()*I), S.Integers).intersect(S.Reals) == ImageSet(Lambda(x, x/2), ConditionSet(
        n, Eq(n**2 - n/2 - S(1)/2, 0), S.Integers))
    
    # 断言：使用 Lambda 表达式创建的 imageset 对象，应与 S.Integers 交集后得到 FiniteSet(S.Half)
    assert imageset(Lambda(n, n/(1/n - 1) + im*(n + 1)*I), S.Integers).intersect(S.Reals) == FiniteSet(S.Half)
    
    # 断言：使用 Lambda 表达式创建的 imageset 对象，应与 S.Integers 交集后得到 FiniteSet(-1)
    assert imageset(Lambda(n, n/(n - 6) + (n - 3)*(n + 1)*I/(2*n + 2)), S.Integers).intersect(S.Reals) == FiniteSet(-1)
    
    # 断言：使用 Lambda 表达式创建的 imageset 对象，应与 S.Integers 交集后得到 EmptySet
    assert imageset(Lambda(n, n/(n**2 - 9) + (n - 3)*(n + 1)*I/(2*n + 2)), S.Integers).intersect(S.Reals) is S.EmptySet
    
    # 创建 ImageSet 对象 s，其中 Lambda 表达式包含复杂的符号计算
    s = ImageSet(
        Lambda(n, -I*(I*(2*pi*n - pi/4) + log(Abs(sqrt(-I))))),
        S.Integers)
    # 断言：s 与 S.Reals 交集后应得到规范化后的 ImageSet 对象
    assert s.intersect(S.Reals) == imageset(Lambda(n, 2*n*pi - pi/4), S.Integers) == ImageSet(
        Lambda(n, 2*pi*n + pi*Rational(7, 4)), S.Integers)


# 定义测试函数 test_imageset_intersect_interval
def test_imageset_intersect_interval():
    # 导入 sympy 库中的符号 n
    from sympy.abc import n
    
    # 定义多个 ImageSet 对象，每个对象包含 Lambda 表达式和集合对象
    f1 = ImageSet(Lambda(n, n*pi), S.Integers)
    f2 = ImageSet(Lambda(n, 2*n), Interval(0, pi))
    f3 = ImageSet(Lambda(n, 2*n*pi + pi/2), S.Integers)
    # 复杂表达式
    f4 = ImageSet(Lambda(n, n*I*pi), S.Integers)
    f5 = ImageSet(Lambda(n, 2*I*n*pi + pi/2), S.Integers)
    # 非线性表达式
    f6 = ImageSet(Lambda(n, log(n)), S.Integers)
    f7 = ImageSet(Lambda(n, n**2), S.Integers)
    f8 = ImageSet(Lambda(n, Abs(n)), S.Integers)
    f9 = ImageSet(Lambda(n, exp(n)), S.Naturals0)
    
    # 断言：f1 与 Interval(-1, 1) 交集后得到 FiniteSet(0)
    assert f1.intersect(Interval(-1, 1)) == FiniteSet(0)
    
    # 断言：f1 与 Interval(0, 2*pi, False, True) 交集后得到 FiniteSet(0, pi)
    assert f1.intersect(Interval(0, 2*pi, False, True)) == FiniteSet(0, pi)
    
    # 断言：f2 与 Interval(1, 2) 交集后得到 Interval(1, 2)
    assert f2.intersect(Interval(1, 2)) == Interval(1, 2)
    
    # 断言：f3 与 Interval(-1, 1) 交集后得到 EmptySet
    assert f3.intersect(Interval(-1, 1)) == S.EmptySet
    
    # 断言：f3 与 Interval(-5, 5) 交集后得到 FiniteSet(pi*Rational(-3, 2), pi/2)
    assert f3.intersect(Interval(-5, 5)) == FiniteSet(pi*Rational(-3, 2), pi/2)
    
    # 断言：f4 与 Interval(-1, 1) 交集后得到 FiniteSet(0)
    assert f4.intersect(Interval(-1, 1)) == FiniteSet(0)
    
    # 断言：f4 与 Interval(1, 2) 交集后得到 EmptySet
    assert f4.intersect(Interval(1, 2)) == S.EmptySet
    
    # 断言：f5 与 Interval(0, 1) 交集后得到 EmptySet
    assert f5.intersect(Interval(0, 1)) == S.EmptySet
    
    # 断言：f6 与 Interval(0, 1) 交集后得到 FiniteSet(0, log(2))
    assert f6.intersect(Interval(0, 1)) == FiniteSet(S.Zero, log(2))
    
    # 断言：f7 与 Interval(0, 10) 交集后得到 Intersection(f7, Interval(0, 10))
    assert f7.intersect(Interval(0, 10)) == Intersection(f7, Interval(0, 10))
    
    # 断言：f8 与 Interval(0, 2) 交集后得到 Intersection(f8, Interval(0, 2))
    assert f8.intersect(Interval(0, 2)) == Intersection(f8, Interval(0, 2))
    
    # 断言：f9 与 Interval(1, 2) 交集后得到 Intersection(f9, Interval(1, 2))
    assert f9.intersect(Interval(1, 2)) == Intersection(f9, Interval(1, 2))


# 定义测试函数 test_imageset_intersect_diophantine
def test_imageset_intersect_diophantine():
    # 导入 sympy 库中的符号 m 和 n
    from sympy.abc import m, n
    # 创建两个 ImageSet 对象，每个对象包含 Lambda 表达式和集合对象
    img1 = ImageSet(Lambda(n, 2*n + 1), S.Integers)
    img2 = ImageSet(Lambda(n, 4*n + 1), S.Integers)
    # 断言：img1 和 img2 的交集应该等于 img2 自身
    assert img1.intersect(img2) == img2

    # 断言：diophantine 返回空的解集
    assert ImageSet(Lambda(n, 2*n), S.Integers).intersect(
            ImageSet(Lambda(n, 2*n + 1), S.Integers)) == S.EmptySet

    # 断言：检查与 S.Integers 的交集
    assert ImageSet(Lambda(n, 9/n + 20*n/3), S.Integers).intersect(
            S.Integers) == FiniteSet(-61, -23, 23, 61)

    # 断言：diophantine 方程的解只有一个 (2, 3)
    assert ImageSet(Lambda(n, (n - 2)**2), S.Integers).intersect(
            ImageSet(Lambda(n, -(n - 3)**2), S.Integers)) == FiniteSet(0)

    # 断言：diophantine 方程有单参数解
    assert ImageSet(Lambda(n, n**2 + 5), S.Integers).intersect(
            ImageSet(Lambda(m, 2*m), S.Integers)).dummy_eq(ImageSet(
            Lambda(n, 4*n**2 + 4*n + 6), S.Integers))

    # 断言：diophantine 方程有四对非参数解
    assert ImageSet(Lambda(n, n**2 - 9), S.Integers).intersect(
            ImageSet(Lambda(m, -m**2), S.Integers)) == FiniteSet(-9, 0)

    # 断言：diophantine 方程有双参数解
    assert ImageSet(Lambda(m, m**2 + 40), S.Integers).intersect(
            ImageSet(Lambda(n, 41*n), S.Integers)).dummy_eq(Intersection(
            ImageSet(Lambda(m, m**2 + 40), S.Integers),
            ImageSet(Lambda(n, 41*n), S.Integers)))

    # 断言：diophantine 返回所有 (8) 解（permute=True）
    assert ImageSet(Lambda(n, n**4 - 2**4), S.Integers).intersect(
            ImageSet(Lambda(m, -m**4 + 3**4), S.Integers)) == FiniteSet(0, 65)

    # 断言：检查 pi 相关的解集
    assert ImageSet(Lambda(n, pi/12 + n*5*pi/12), S.Integers).intersect(
            ImageSet(Lambda(n, 7*pi/12 + n*11*pi/12), S.Integers)).dummy_eq(ImageSet(
            Lambda(n, 55*pi*n/12 + 17*pi/4), S.Integers))

    # 断言：TypeError 应该被 diophantine 抛出 (#18081)
    assert ImageSet(Lambda(n, n*log(2)), S.Integers).intersection(
        S.Integers).dummy_eq(Intersection(ImageSet(
        Lambda(n, n*log(2)), S.Integers), S.Integers))

    # 断言：NotImplementedError 应该被 diophantine 抛出（对于 cubic_thue 没有求解器）
    assert ImageSet(Lambda(n, n**3 + 1), S.Integers).intersect(
            ImageSet(Lambda(n, n**3), S.Integers)).dummy_eq(Intersection(
            ImageSet(Lambda(n, n**3 + 1), S.Integers),
            ImageSet(Lambda(n, n**3), S.Integers)))
# 定义测试函数 test_infinitely_indexed_set_3，引入符号 n 和 m
def test_infinitely_indexed_set_3():
    from sympy.abc import n, m
    # 断言：两个像集的交集，其中第一个是以 m 为参数的函数，第二个是以 n 为参数的函数，都与 Lambda(t, 6*pi*t) 的像集相等
    assert imageset(Lambda(m, 2*pi*m), S.Integers).intersect(
            imageset(Lambda(n, 3*pi*n), S.Integers)).dummy_eq(
        ImageSet(Lambda(t, 6*pi*t), S.Integers))
    # 断言：对于像集 Lambda(n, 2*n + 1) 和 Lambda(n, 2*n - 1)，它们在整数集 S.Integers 上是相等的
    assert imageset(Lambda(n, 2*n + 1), S.Integers) == \
        imageset(Lambda(n, 2*n - 1), S.Integers)
    # 断言：对于像集 Lambda(n, 3*n + 2) 和 Lambda(n, 3*n - 1)，它们在整数集 S.Integers 上是相等的
    assert imageset(Lambda(n, 3*n + 2), S.Integers) == \
        imageset(Lambda(n, 3*n - 1), S.Integers)


# 定义测试函数 test_ImageSet_simplification，引入符号 n 和 m
def test_ImageSet_simplification():
    from sympy.abc import n, m
    # 断言：Lambda(n, n) 的像集在整数集 S.Integers 上等于整数集 S.Integers 自身
    assert imageset(Lambda(n, n), S.Integers) == S.Integers
    # 断言：Lambda(n, sin(n)) 的像集与像集 Lambda(m, sin(tan(m))) 在整数集 S.Integers 上相等
    assert imageset(Lambda(n, sin(n)),
                    imageset(Lambda(m, tan(m)), S.Integers)) == \
            imageset(Lambda(m, sin(tan(m))), S.Integers)
    # 断言：像集 Lambda(n, 1 + 2*n) 在自然数集 S.Naturals 上等于 Range(3, oo, 2)
    assert imageset(n, 1 + 2*n, S.Naturals) == Range(3, oo, 2)
    # 断言：像集 Lambda(n, 1 + 2*n) 在非负整数集 S.Naturals0 上等于 Range(1, oo, 2)
    assert imageset(n, 1 + 2*n, S.Naturals0) == Range(1, oo, 2)
    # 断言：像集 Lambda(n, 1 - 2*n) 在自然数集 S.Naturals 上等于 Range(-1, -oo, -2)
    assert imageset(n, 1 - 2*n, S.Naturals) == Range(-1, -oo, -2)


# 定义测试函数 test_ImageSet_contains
def test_ImageSet_contains():
    # 断言：元组 (2, S.Half) 在像集 x, (x, 1/x), S.Integers 中
    assert (2, S.Half) in imageset(x, (x, 1/x), S.Integers)
    # 断言：像集 x, x + I*3, S.Integers 与实数集的交集是空集
    assert imageset(x, x + I*3, S.Integers).intersection(S.Reals) is S.EmptySet
    # 创建一个虚拟整数 i
    i = Dummy(integer=True)
    # q 是像集 x, x + I*y, S.Integers 与实数集的交集
    q = imageset(x, x + I*y, S.Integers).intersection(S.Reals)
    # 断言：q 替换 y 为 I*i 后与整数集 S.Integers 的交集是整数集 S.Integers
    assert q.subs(y, I*i).intersection(S.Integers) is S.Integers
    # q 是像集 x, x + I*y/x, S.Integers 与实数集的交集
    q = imageset(x, x + I*y/x, S.Integers).intersection(S.Reals)
    # 断言：q 替换 y 为 0 后与整数集 S.Integers 的交集是整数集 S.Integers
    assert q.subs(y, 0) is S.Integers
    # 断言：q 替换 y 为 I*i*x 后与整数集 S.Integers 的交集是整数集 S.Integers
    assert q.subs(y, I*i*x).intersection(S.Integers) is S.Integers
    # 计算 z = cos(1)**2 + sin(1)**2 - 1
    z = cos(1)**2 + sin(1)**2 - 1
    # q 是像集 x, x + I*z, S.Integers 与实数集的交集
    q = imageset(x, x + I*z, S.Integers).intersection(S.Reals)
    # 断言：q 不是空集
    assert q is not S.EmptySet


# 定义测试函数 test_ComplexRegion_contains
def test_ComplexRegion_contains():
    # 创建一个实数符号 r
    r = Symbol('r', real=True)
    # 创建区间 a、b、c
    a = Interval(2, 3)
    b = Interval(4, 6)
    c = Interval(7, 9)
    # 创建复杂区域 c1 和 c2
    c1 = ComplexRegion(a*b)
    c2 = ComplexRegion(Union(a*b, c*a))
    # 断言：2.5 + 4.5*I 在复杂区域 c1 中
    assert 2.5 + 4.5*I in c1
    # 断言：2 + 4*I 在复杂区域 c1 中
    assert 2 + 4*I in c1
    # 断言：3 + 4*I 在复杂区域 c1 中
    assert 3 + 4*I in c1
    # 断言：8 + 2.5*I 在复杂区域 c2 中
    assert 8 + 2.5*I in c2
    # 断言：2.5 + 6.1*I 不在复杂区域 c1 中
    assert 2.5 + 6.1*I not in c1
    # 断言：4.5 + 3.2*I 不在复杂区域 c1 中
    assert 4.5 + 3.2*I not in c1
    # 断言：复杂区域 c1 是否包含符号 x 的判定结果为 Contains(x, c1, evaluate=False)
    assert c1.contains(x) == Contains(x, c1, evaluate=False)
    # 断言：复杂区域 c1 不包含符号 r
    assert c1.contains(r) == False
    # 断言：复杂区域 c2 是否包含符号 x 的判定结果为 Contains(x, c2, evaluate=False)
    assert c2.contains(x) == Contains(x, c2, evaluate=False)
    # 断言：复杂区域 c2 不包含符号 r
    assert c2.contains(r) == False

    # 创建区间 r1 和 theta1
    r1 = Interval(0, 1)
    theta1 = Interval(0, 2*S.Pi)
    # 创建极坐标复杂区域 c3
    c3 = ComplexRegion(r1*theta1, polar=True)
    # 断言：(0.5 + I*6/10) 在复杂区域 c3 中
    assert (0.5 + I*6/10) in c3
    # 断言：S.Half + I*6/10 在复杂区域 c3 中
    assert (S.Half + I*6/10) in c3
    # 断言：S.Half + .6*I 在复杂区域 c3 中
    assert (S.Half + .6*I) in c3
    # 断言：0.5 + .6*I 在复杂区域 c3 中
    assert (0.5 + .6*I) in c3
    # 断言：I 在复杂区域 c3 中
    assert I in c3
    # 断言：1 在复杂区域 c3 中
    assert 1 in c3
    # 断言：0 在复杂区域 c3 中
    assert 0 in
    # 断言测试：验证复数区域 c4 是否包含特定复数 -2 + I，预期结果为 False
    assert c4.contains(-2 + I) == False
    # 断言测试：验证复数区域 c4 是否包含特定复数 -2 - I，预期结果为 True
    assert c4.contains(-2 - I) == True
    # 断言测试：验证复数区域 c4 是否包含特定复数 2 - I，预期结果为 True
    assert c4.contains(2 - I) == True
    # 断言测试：验证复数区域 c4 是否包含特定实数 -2，预期结果为 False
    assert c4.contains(-2) == False
    # 断言测试：验证复数区域 c4 是否包含特定实数 2，预期结果为 True
    assert c4.contains(2) == True
    # 断言测试：验证复数区域 c4 是否包含未求值的符号 x，预期结果为 Contains(x, c4, evaluate=False)
    assert c4.contains(x) == Contains(x, c4, evaluate=False)
    # 断言测试：验证复数区域 c4 是否包含特定表达式 3/(1 + r**2)，预期结果为 Contains(3/(1 + r**2), c4, evaluate=False)，实际上为 True
    assert c4.contains(3/(1 + r**2)) == Contains(3/(1 + r**2), c4, evaluate=False)
    
    # 引发异常测试：尝试创建极坐标形式半径为 2 的复数区域，预期引发 ValueError
    raises(ValueError, lambda: ComplexRegion(r1*theta1, polar=2))
def test_symbolic_Range():
    # 创建符号变量 n
    n = Symbol('n')
    # 测试：期望抛出 ValueError 异常，因为 Range(n) 需要 n 是整数
    raises(ValueError, lambda: Range(n)[0])
    # 测试：期望抛出 IndexError 异常，因为 Range(n, n) 的范围为空
    raises(IndexError, lambda: Range(n, n)[0])
    # 测试：期望抛出 ValueError 异常，因为 Range(n, n+1) 的范围为空
    raises(ValueError, lambda: Range(n, n+1)[0])
    # 测试：期望抛出 ValueError 异常，因为 Range(n).size 需要 n 是整数
    raises(ValueError, lambda: Range(n).size)

    # 将 n 设置为整数符号变量
    n = Symbol('n', integer=True)
    # 测试：期望抛出 ValueError 异常，因为 Range(n) 的范围为空
    raises(ValueError, lambda: Range(n)[0])
    # 测试：期望抛出 IndexError 异常，因为 Range(n, n) 的范围为空
    raises(IndexError, lambda: Range(n, n)[0])
    # 测试：Range(n, n+1) 的第一个元素应该是 n
    assert Range(n, n+1)[0] == n
    # 测试：期望抛出 ValueError 异常，因为 Range(n).size 需要 n 是非负整数
    raises(ValueError, lambda: Range(n).size)
    # 测试：Range(n, n+1) 的大小应该是 1
    assert Range(n, n+1).size == 1

    # 将 n 设置为非负整数符号变量
    n = Symbol('n', integer=True, nonnegative=True)
    # 测试：Range(n) 的第一个元素应该是 0
    assert Range(n+1)[0] == 0
    # 测试：Range(n, n+1) 的第一个元素应该是 n
    assert Range(n, n+1)[0] == n
    # 测试：Range(n) 的大小应该是 n
    assert Range(n).size == n
    # 测试：Range(n+1) 的大小应该是 n+1
    assert Range(n+1).size == n+1
    # 测试：Range(n, n+1) 的大小应该是 1
    assert Range(n, n+1).size == 1

    # 将 n 设置为正整数符号变量
    n = Symbol('n', integer=True, positive=True)
    # 测试：Range(n) 的第一个元素应该是 0
    assert Range(n)[0] == 0
    # 测试：Range(n, n+1) 的第一个元素应该是 n
    assert Range(n, n+1)[0] == n
    # 测试：Range(n) 的大小应该是 n
    assert Range(n).size == n
    # 测试：Range(n, n+1) 的大小应该是 1
    assert Range(n, n+1).size == 1

    # 创建正整数符号变量 m
    m = Symbol('m', integer=True, positive=True)

    # 测试：Range(n, n+m) 的第一个元素应该是 n
    assert Range(n, n+m)[0] == n
    # 测试：Range(n, n+m) 的大小应该是 m
    assert Range(n, n+m).size == m
    # 测试：Range(n, n+1) 的大小应该是 1
    assert Range(n, n+1).size == 1
    # 测试：Range(n, n+m, 2) 的大小应该是 m/2 的下取整
    assert Range(n, n+m, 2).size == floor(m/2)

    # 将 m 设置为正偶数符号变量
    m = Symbol('m', integer=True, positive=True, even=True)
    # 测试：Range(n, n+m, 2) 的大小应该是 m/2
    assert Range(n, n+m, 2).size == m/2


def test_issue_18400():
    # 创建整数符号变量 n
    n = Symbol('n', integer=True)
    # 测试：期望抛出 ValueError 异常，因为 Range(n) 需要 n 是整数
    raises(ValueError, lambda: imageset(lambda x: x*2, Range(n)))

    # 将 n 设置为正整数符号变量
    n = Symbol('n', integer=True, positive=True)
    # 测试：imageset(lambda x: x*2, Range(n)) 应该等于 imageset(lambda x: x*2, Range(n))
    assert imageset(lambda x: x*2, Range(n)) == imageset(lambda x: x*2, Range(n))


def test_ComplexRegion_intersect():
    # 极坐标形式
    X_axis = ComplexRegion(Interval(0, oo)*FiniteSet(0, S.Pi), polar=True)

    unit_disk = ComplexRegion(Interval(0, 1)*Interval(0, 2*S.Pi), polar=True)
    upper_half_unit_disk = ComplexRegion(Interval(0, 1)*Interval(0, S.Pi), polar=True)
    upper_half_disk = ComplexRegion(Interval(0, oo)*Interval(0, S.Pi), polar=True)
    lower_half_disk = ComplexRegion(Interval(0, oo)*Interval(S.Pi, 2*S.Pi), polar=True)
    right_half_disk = ComplexRegion(Interval(0, oo)*Interval(-S.Pi/2, S.Pi/2), polar=True)
    first_quad_disk = ComplexRegion(Interval(0, oo)*Interval(0, S.Pi/2), polar=True)

    # 测试：upper_half_disk 和 unit_disk 的交集应该是 upper_half_unit_disk
    assert upper_half_disk.intersect(unit_disk) == upper_half_unit_disk
    # 测试：right_half_disk 和 first_quad_disk 的交集应该是 first_quad_disk
    assert right_half_disk.intersect(first_quad_disk) == first_quad_disk
    # 测试：upper_half_disk 和 right_half_disk 的交集应该是 first_quad_disk
    assert upper_half_disk.intersect(right_half_disk) == first_quad_disk
    # 测试：upper_half_disk 和 lower_half_disk 的交集应该是 X_axis
    assert upper_half_disk.intersect(lower_half_disk) == X_axis

    c1 = ComplexRegion(Interval(0, 4)*Interval(0, 2*S.Pi), polar=True)
    # 测试：c1 和 Interval(1, 5) 的交集应该是 Interval(1, 4)
    assert c1.intersect(Interval(1, 5)) == Interval(1, 4)
    # 测试：c1 和 Interval(4, 9) 的交集应该是 FiniteSet(4)
    assert c1.intersect(Interval(4, 9)) == FiniteSet(4)
    # 测试：c1 和 Interval(5, 12) 的交集应该是 S.EmptySet
    assert c1.intersect(Interval(5, 12)) is S.EmptySet

    # 矩形形式
    X_axis = ComplexRegion(Interval(-oo, oo)*FiniteSet(0))

    unit_square = ComplexRegion(Interval(-1, 1)*Interval(-1, 1))
    upper_half_unit_square = ComplexRegion(Interval(-1, 1)*Interval(0, 1))
    upper_half_plane = ComplexRegion(Interval(-oo, oo)*Interval(0, oo))
    # 创建一个复平面区域对象，表示复平面的下半平面
    lower_half_plane = ComplexRegion(Interval(-oo, oo)*Interval(-oo, 0))
    
    # 创建一个复平面区域对象，表示复平面的右半平面
    right_half_plane = ComplexRegion(Interval(0, oo)*Interval(-oo, oo))
    
    # 创建一个复平面区域对象，表示复平面的第一象限
    first_quad_plane = ComplexRegion(Interval(0, oo)*Interval(0, oo))
    
    # 断言：计算上半平面与单位正方形的交集，应得到上半单位正方形
    assert upper_half_plane.intersect(unit_square) == upper_half_unit_square
    
    # 断言：计算右半平面与第一象限的交集，应得到第一象限
    assert right_half_plane.intersect(first_quad_plane) == first_quad_plane
    
    # 断言：计算上半平面与右半平面的交集，应得到第一象限
    assert upper_half_plane.intersect(right_half_plane) == first_quad_plane
    
    # 断言：计算上半平面与下半平面的交集，应得到实轴
    assert upper_half_plane.intersect(lower_half_plane) == X_axis
    
    # 创建一个复平面区域对象，表示包含指定区间的复平面区域
    c1 = ComplexRegion(Interval(-5, 5)*Interval(-10, 10))
    
    # 断言：计算复平面区域与指定实数区间的交集，应得到实数区间的一部分
    assert c1.intersect(Interval(2, 7)) == Interval(2, 5)
    
    # 断言：计算复平面区域与指定实数区间的交集，应得到包含单个实数的有限集合
    assert c1.intersect(Interval(5, 7)) == FiniteSet(5)
    
    # 断言：计算复平面区域与指定实数区间的交集，应为空集
    assert c1.intersect(Interval(6, 9)) is S.EmptySet
    
    # 创建一个复平面区域对象，使用极坐标表示，但不对其进行评估
    C1 = ComplexRegion(Interval(0, 1)*Interval(0, 2*S.Pi), polar=True)
    
    # 创建一个复平面区域对象，表示包含指定区间的复平面区域
    C2 = ComplexRegion(Interval(-1, 1)*Interval(-1, 1))
    
    # 断言：计算两个复平面区域的交集，得到交集对象但不进行评估
    assert C1.intersect(C2) == Intersection(C1, C2, evaluate=False)
# 定义测试函数 test_ComplexRegion_union
def test_ComplexRegion_union():
    # 极坐标形式的复杂区域 c1
    c1 = ComplexRegion(Interval(0, 1)*Interval(0, 2*S.Pi), polar=True)
    # 极坐标形式的复杂区域 c2
    c2 = ComplexRegion(Interval(0, 1)*Interval(0, S.Pi), polar=True)
    # 极坐标形式的复杂区域 c3
    c3 = ComplexRegion(Interval(0, oo)*Interval(0, S.Pi), polar=True)
    # 极坐标形式的复杂区域 c4
    c4 = ComplexRegion(Interval(0, oo)*Interval(S.Pi, 2*S.Pi), polar=True)

    # 构建并集 p1
    p1 = Union(Interval(0, 1)*Interval(0, 2*S.Pi), Interval(0, 1)*Interval(0, S.Pi))
    # 构建并集 p2
    p2 = Union(Interval(0, oo)*Interval(0, S.Pi), Interval(0, oo)*Interval(S.Pi, 2*S.Pi))

    # 断言 c1 和 c2 的并集等于以极坐标形式定义的 p1
    assert c1.union(c2) == ComplexRegion(p1, polar=True)
    # 断言 c3 和 c4 的并集等于以极坐标形式定义的 p2
    assert c3.union(c4) == ComplexRegion(p2, polar=True)

    # 矩形形式的复杂区域 c5
    c5 = ComplexRegion(Interval(2, 5)*Interval(6, 9))
    # 矩形形式的复杂区域 c6
    c6 = ComplexRegion(Interval(4, 6)*Interval(10, 12))
    # 矩形形式的复杂区域 c7
    c7 = ComplexRegion(Interval(0, 10)*Interval(-10, 0))
    # 矩形形式的复杂区域 c8
    c8 = ComplexRegion(Interval(12, 16)*Interval(14, 20))

    # 构建并集 p3
    p3 = Union(Interval(2, 5)*Interval(6, 9), Interval(4, 6)*Interval(10, 12))
    # 构建并集 p4
    p4 = Union(Interval(0, 10)*Interval(-10, 0), Interval(12, 16)*Interval(14, 20))

    # 断言 c5 和 c6 的并集等于以矩形形式定义的 p3
    assert c5.union(c6) == ComplexRegion(p3)
    # 断言 c7 和 c8 的并集等于以矩形形式定义的 p4
    assert c7.union(c8) == ComplexRegion(p4)

    # 断言 c1 和 Interval(2, 4) 的并集，不进行求值
    assert c1.union(Interval(2, 4)) == Union(c1, Interval(2, 4), evaluate=False)
    # 断言 c5 和 Interval(2, 4) 的并集，通过实数区间创建复杂区域
    assert c5.union(Interval(2, 4)) == Union(c5, ComplexRegion.from_real(Interval(2, 4)))


# 定义测试函数 test_ComplexRegion_from_real
def test_ComplexRegion_from_real():
    # 极坐标形式的复杂区域 c1
    c1 = ComplexRegion(Interval(0, 1) * Interval(0, 2 * S.Pi), polar=True)

    # 引发值错误，lambda 表达式检查 c1 是否能从自身创建
    raises(ValueError, lambda: c1.from_real(c1))
    # 断言 c1 从实数区间创建的结果
    assert c1.from_real(Interval(-1, 1)) == ComplexRegion(Interval(-1, 1) * FiniteSet(0), False)


# 定义测试函数 test_ComplexRegion_measure
def test_ComplexRegion_measure():
    # 区间定义 a, b
    a, b = Interval(2, 5), Interval(4, 8)
    # 极角区间定义 theta1, theta2
    theta1, theta2 = Interval(0, 2*S.Pi), Interval(0, S.Pi)
    # 矩形形式的复杂区域 c1
    c1 = ComplexRegion(a*b)
    # 极坐标形式的复杂区域 c2
    c2 = ComplexRegion(Union(a*theta1, b*theta2), polar=True)

    # 断言 c1 的测量结果为 12
    assert c1.measure == 12
    # 断言 c2 的测量结果为 9*pi


# 定义测试函数 test_normalize_theta_set
def test_normalize_theta_set():
    # 区间
    assert normalize_theta_set(Interval(pi, 2*pi)) == \
        Union(FiniteSet(0), Interval.Ropen(pi, 2*pi))
    assert normalize_theta_set(Interval(pi*Rational(9, 2), 5*pi)) == Interval(pi/2, pi)
    assert normalize_theta_set(Interval(pi*Rational(-3, 2), pi/2)) == Interval.Ropen(0, 2*pi)
    assert normalize_theta_set(Interval.open(pi*Rational(-3, 2), pi/2)) == \
        Union(Interval.Ropen(0, pi/2), Interval.open(pi/2, 2*pi))
    assert normalize_theta_set(Interval.open(pi*Rational(-7, 2), pi*Rational(-3, 2))) == \
        Union(Interval.Ropen(0, pi/2), Interval.open(pi/2, 2*pi))
    assert normalize_theta_set(Interval(-pi/2, pi/2)) == \
        Union(Interval(0, pi/2), Interval.Ropen(pi*Rational(3, 2), 2*pi))
    assert normalize_theta_set(Interval.open(-pi/2, pi/2)) == \
        Union(Interval.Ropen(0, pi/2), Interval.open(pi*Rational(3, 2), 2*pi))
    assert normalize_theta_set(Interval(-4*pi, 3*pi)) == Interval.Ropen(0, 2*pi)
    assert normalize_theta_set(Interval(pi*Rational(-3, 2), -pi/2)) == Interval(pi/2, pi*Rational(3, 2))
    # 校验函数 normalize_theta_set 的功能是否符合预期，下面是一系列断言语句
    assert normalize_theta_set(Interval.open(0, 2*pi)) == Interval.open(0, 2*pi)
    assert normalize_theta_set(Interval.Ropen(-pi/2, pi/2)) == \
        Union(Interval.Ropen(0, pi/2), Interval.Ropen(pi*Rational(3, 2), 2*pi))
    assert normalize_theta_set(Interval.Lopen(-pi/2, pi/2)) == \
        Union(Interval(0, pi/2), Interval.open(pi*Rational(3, 2), 2*pi))
    assert normalize_theta_set(Interval(-pi/2, pi/2)) == \
        Union(Interval(0, pi/2), Interval.Ropen(pi*Rational(3, 2), 2*pi))
    assert normalize_theta_set(Interval.open(4*pi, pi*Rational(9, 2))) == Interval.open(0, pi/2)
    assert normalize_theta_set(Interval.Lopen(4*pi, pi*Rational(9, 2))) == Interval.Lopen(0, pi/2)
    assert normalize_theta_set(Interval.Ropen(4*pi, pi*Rational(9, 2))) == Interval.Ropen(0, pi/2)
    assert normalize_theta_set(Interval.open(3*pi, 5*pi)) == \
        Union(Interval.Ropen(0, pi), Interval.open(pi, 2*pi))

    # 对于 FiniteSet 的测试
    assert normalize_theta_set(FiniteSet(0, pi, 3*pi)) == FiniteSet(0, pi)
    assert normalize_theta_set(FiniteSet(0, pi/2, pi, 2*pi)) == FiniteSet(0, pi/2, pi)
    assert normalize_theta_set(FiniteSet(0, -pi/2, -pi, -2*pi)) == FiniteSet(0, pi, pi*Rational(3, 2))
    assert normalize_theta_set(FiniteSet(pi*Rational(-3, 2), pi/2)) == \
        FiniteSet(pi/2)
    assert normalize_theta_set(FiniteSet(2*pi)) == FiniteSet(0)

    # 对于 Union 的测试
    assert normalize_theta_set(Union(Interval(0, pi/3), Interval(pi/2, pi))) == \
        Union(Interval(0, pi/3), Interval(pi/2, pi))
    assert normalize_theta_set(Union(Interval(0, pi), Interval(2*pi, pi*Rational(7, 3)))) == \
        Interval(0, pi)

    # 针对非实数集合的数值错误异常
    raises(ValueError, lambda: normalize_theta_set(S.Complexes))

    # 针对实数子集的未实现异常
    raises(NotImplementedError, lambda: normalize_theta_set(Interval(0, 1)))

    # 针对没有 pi 作为系数的未实现异常
    raises(NotImplementedError, lambda: normalize_theta_set(Interval(1, 2*pi)))
    raises(NotImplementedError, lambda: normalize_theta_set(Interval(2*pi, 10)))
    raises(NotImplementedError, lambda: normalize_theta_set(FiniteSet(0, 3, 3*pi)))
def test_ComplexRegion_FiniteSet():
    # 定义符号变量
    x, y, z, a, b, c = symbols('x y z a b c')

    # Issue #9669 的测试用例
    assert ComplexRegion(FiniteSet(a, b, c)*FiniteSet(x, y, z)) == \
        FiniteSet(a + I*x, a + I*y, a + I*z, b + I*x, b + I*y,
                  b + I*z, c + I*x, c + I*y, c + I*z)
    assert ComplexRegion(FiniteSet(2)*FiniteSet(3)) == FiniteSet(2 + 3*I)


def test_union_RealSubSet():
    # 测试 S.Complexes 和 Interval(1, 2) 的联合
    assert (S.Complexes).union(Interval(1, 2)) == S.Complexes
    # 测试 S.Complexes 和 S.Integers 的联合
    assert (S.Complexes).union(S.Integers) == S.Complexes


def test_SetKind_fancySet():
    # 定义一个函数 G，并测试其返回结果的 kind 类型
    G = lambda *args: ImageSet(Lambda(x, x ** 2), *args)
    assert G(Interval(1, 4)).kind is SetKind(NumberKind)
    assert G(FiniteSet(1, 4)).kind is SetKind(NumberKind)
    assert S.Rationals.kind is SetKind(NumberKind)
    assert S.Naturals.kind is SetKind(NumberKind)
    assert S.Integers.kind is SetKind(NumberKind)
    assert Range(3).kind is SetKind(NumberKind)
    # 创建 Interval 对象 a 和 b
    a = Interval(2, 3)
    b = Interval(4, 6)
    # 创建 ComplexRegion 对象 c1，并测试其 kind 类型
    c1 = ComplexRegion(a*b)
    assert c1.kind is SetKind(TupleKind(NumberKind, NumberKind))


def test_issue_9980():
    # 创建 ComplexRegion 对象 c1 和 c2
    c1 = ComplexRegion(Interval(1, 2)*Interval(2, 3))
    c2 = ComplexRegion(Interval(1, 5)*Interval(1, 3))
    # 创建 Union 对象 R，并测试其简化后的结果
    R = Union(c1, c2)
    assert simplify(R) == ComplexRegion(Union(Interval(1, 2)*Interval(2, 3), \
                                    Interval(1, 5)*Interval(1, 3)), False)
    # 测试 ComplexRegion 对象 c1 的构造函数
    assert c1.func(*c1.args) == c1
    # 测试 Union 对象 R 的构造函数
    assert R.func(*R.args) == R


def test_issue_11732():
    # 创建 Interval 对象 interval12 和 FiniteSet 对象 finiteset1234，以及 Tuple 对象 pointComplex
    interval12 = Interval(1, 2)
    finiteset1234 = FiniteSet(1, 2, 3, 4)
    pointComplex = Tuple(1, 5)

    # 测试 interval12 是否在 S.Naturals, S.Naturals0, S.Integers, S.Complexes 中
    assert (interval12 in S.Naturals) == False
    assert (interval12 in S.Naturals0) == False
    assert (interval12 in S.Integers) == False
    assert (interval12 in S.Complexes) == False

    # 测试 finiteset1234 是否在 S.Naturals, S.Naturals0, S.Integers, S.Complexes 中
    assert (finiteset1234 in S.Naturals) == False
    assert (finiteset1234 in S.Naturals0) == False
    assert (finiteset1234 in S.Integers) == False
    assert (finiteset1234 in S.Complexes) == False

    # 测试 pointComplex 是否在 S.Naturals, S.Naturals0, S.Integers, S.Complexes 中
    assert (pointComplex in S.Naturals) == False
    assert (pointComplex in S.Naturals0) == False
    assert (pointComplex in S.Integers) == False
    assert (pointComplex in S.Complexes) == True


def test_issue_11730():
    # 创建 Interval 对象 unit 和 ComplexRegion 对象 square
    unit = Interval(0, 1)
    square = ComplexRegion(unit ** 2)

    # 测试 Union(S.Complexes, FiniteSet(oo)) 和 Union(S.Complexes, FiniteSet(eye(4))) 的结果
    assert Union(S.Complexes, FiniteSet(oo)) != S.Complexes
    assert Union(S.Complexes, FiniteSet(eye(4))) != S.Complexes
    # 测试 Union(unit, square) 和 Intersection(S.Reals, square) 的结果
    assert Union(unit, square) == square
    assert Intersection(S.Reals, square) == unit


def test_issue_11938():
    # 创建 Interval 对象 unit 和 ival，以及 ComplexRegion 对象 cr1
    unit = Interval(0, 1)
    ival = Interval(1, 2)
    cr1 = ComplexRegion(ival * unit)

    # 测试 Intersection(cr1, S.Reals) 和 Intersection(cr1, unit) 的结果
    assert Intersection(cr1, S.Reals) == ival
    assert Intersection(cr1, unit) == FiniteSet(1)

    # 创建 Interval 对象 arg1, arg2, arg3 和 ComplexRegion 对象 cp1, cp2, cp3
    arg1 = Interval(0, S.Pi)
    arg2 = FiniteSet(S.Pi)
    arg3 = Interval(S.Pi / 4, 3 * S.Pi / 4)
    cp1 = ComplexRegion(unit * arg1, polar=True)
    cp2 = ComplexRegion(unit * arg2, polar=True)
    cp3 = ComplexRegion(unit * arg3, polar=True)

    # 测试 Intersection(cp1, S.Reals) 的结果
    assert Intersection(cp1, S.Reals) == Interval(-1, 1)
    # 断言：检查两个集合的交集是否等于区间(-1, 0)
    assert Intersection(cp2, S.Reals) == Interval(-1, 0)
    
    # 断言：检查两个集合的交集是否等于包含单个元素0的有限集合
    assert Intersection(cp3, S.Reals) == FiniteSet(0)
def test_issue_11914():
    # 创建两个区间对象，a=[0, 1]，b=[0, pi]
    a, b = Interval(0, 1), Interval(0, pi)
    # 创建另外两个区间对象，c=[2, 3]，d=[pi, 3*pi/2]
    c, d = Interval(2, 3), Interval(pi, 3 * pi / 2)
    # 创建复平面区域对象cp1，使用极坐标表示
    cp1 = ComplexRegion(a * b, polar=True)
    # 创建复平面区域对象cp2，使用极坐标表示
    cp2 = ComplexRegion(c * d, polar=True)

    # 验证-3是否在cp1和cp2的并集中
    assert -3 in cp1.union(cp2)
    # 验证-3是否在cp2和cp1的并集中
    assert -3 in cp2.union(cp1)
    # 验证-5是否不在cp1和cp2的并集中
    assert -5 not in cp1.union(cp2)


def test_issue_9543():
    # 验证Lambda表达式Lambda(x, x**2)映射的图像集是否是S.Naturals的子集
    assert ImageSet(Lambda(x, x**2), S.Naturals).is_subset(S.Reals)


def test_issue_16871():
    # 验证Lambda表达式Lambda(x, x)映射的图像集与集合{1}是否相等
    assert ImageSet(Lambda(x, x), FiniteSet(1)) == {1}
    # 验证Lambda表达式Lambda(x, x - 3)映射的图像集与S.Integers的交集是否等于S.Integers
    assert ImageSet(Lambda(x, x - 3), S.Integers).intersection(S.Integers) is S.Integers


@XFAIL
def test_issue_16871b():
    # 验证Lambda表达式Lambda(x, x - 3)映射的图像集是否是S.Integers的子集
    assert ImageSet(Lambda(x, x - 3), S.Integers).is_subset(S.Integers)


def test_issue_18050():
    # 验证Lambda表达式Lambda(x, I*x + 1)映射的图像集与S.Integers映射的图像集是否相等
    assert imageset(Lambda(x, I*x + 1), S.Integers) == ImageSet(Lambda(x, I*x + 1), S.Integers)
    # 验证Lambda表达式Lambda(x, 3*I*x + 4 + 8*I)映射的图像集与S.Integers映射的图像集是否相等
    assert imageset(Lambda(x, 3*I*x + 4 + 8*I), S.Integers) == ImageSet(Lambda(x, 3*I*x + 4 + 2*I), S.Integers)
    # 对于接下来的两个测试，没有'Mod'：
    # 验证Lambda表达式Lambda(x, 2*x + 3*I)映射的图像集与S.Integers映射的图像集是否相等
    assert imageset(Lambda(x, 2*x + 3*I), S.Integers) == ImageSet(Lambda(x, 2*x + 3*I), S.Integers)
    # 创建一个正的符号r
    r = Symbol('r', positive=True)
    # 验证Lambda表达式Lambda(x, r*x + 10)映射的图像集与S.Integers映射的图像集是否相等
    assert imageset(Lambda(x, r*x + 10), S.Integers) == ImageSet(Lambda(x, r*x + 10), S.Integers)
    # 减少实部：
    # 验证Lambda表达式Lambda(x, 3*x + 8 + 5*I)映射的图像集与S.Integers映射的图像集是否相等
    assert imageset(Lambda(x, 3*x + 8 + 5*I), S.Integers) == ImageSet(Lambda(x, 3*x + 2 + 5*I), S.Integers)


def test_Rationals():
    # 验证S.Integers是否是S.Rationals的子集
    assert S.Integers.is_subset(S.Rationals)
    # 验证S.Naturals是否是S.Rationals的子集
    assert S.Naturals.is_subset(S.Rationals)
    # 验证S.Naturals0是否是S.Rationals的子集
    assert S.Naturals0.is_subset(S.Rationals)
    # 验证S.Rationals是否是S.Reals的子集
    assert S.Rationals.is_subset(S.Reals)
    # 验证S.Rationals的下确界是否为负无穷
    assert S.Rationals.inf is -oo
    # 验证S.Rationals的上确界是否为正无穷
    assert S.Rationals.sup is oo
    # 创建S.Rationals的迭代器
    it = iter(S.Rationals)
    # 验证S.Rationals的前12个元素是否符合指定顺序
    assert [next(it) for i in range(12)] == [
        0, 1, -1, S.Half, 2, Rational(-1, 2), -2,
        Rational(1, 3), 3, Rational(-1, 3), -3, Rational(2, 3)]
    # 验证Basic()不在S.Rationals中
    assert Basic() not in S.Rationals
    # 验证S.Half是否在S.Rationals中
    assert S.Half in S.Rationals
    # 验证S.Rationals是否包含0.5
    assert S.Rationals.contains(0.5) == Contains(
        0.5, S.Rationals, evaluate=False)
    # 验证2是否在S.Rationals中
    assert 2 in S.Rationals
    # 创建一个有理数符号r
    r = symbols('r', rational=True)
    # 验证r是否在S.Rationals中
    assert r in S.Rationals
    # 引发类型错误，尝试验证x是否在S.Rationals中
    raises(TypeError, lambda: x in S.Rationals)
    # issue #18134:
    # 验证S.Rationals的边界是否等于S.Reals
    assert S.Rationals.boundary == S.Reals
    # 验证S.Rationals的闭包是否等于S.Reals
    assert S.Rationals.closure == S.Reals
    # 验证S.Rationals是否是开集（不是）
    assert S.Rationals.is_open == False
    # 验证S.Rationals是否是闭集（不是）
    assert S.Rationals.is_closed == False


def test_NZQRC_unions():
    # 检查所有平凡数集合的并集是否被简化
    nbrsets = (S.Naturals, S.Naturals0, S.Integers, S.Rationals,
        S.Reals, S.Complexes)
    # 生成所有可能的集合并集
    unions = (Union(a, b) for a in nbrsets for b in nbrsets)
    # 验证所有生成的并集是否都不是Union对象
    assert all(u.is_Union is False for u in unions)


def test_imageset_intersection():
    # 创建一个虚拟变量n
    n = Dummy()
    # 创建一个图像集，映射Lambda(n, -I*(I*(2*pi*n - pi/4) + log(Abs(sqrt(-I)))))到S.Integers
    s = ImageSet(Lambda(n, -I*(I*(2*pi*n - pi/4) + log(Abs(sqrt(-I))))), S.Integers)
    # 验证图像集与S.Reals的交集是否等于Lambda(n, 2*pi*n + pi*7/4)映射的图像集
    assert s.intersect(S.Reals) == ImageSet(
        Lambda(n, 2*pi*n + pi*Rational(7, 4)), S.Integers)


def test_issue_17858():
    # 验证1是否在Range(-oo, oo)范围内
    assert 1 in Range(-oo, oo)
    # 验证0是否在Range(oo, -oo, -1)范围内
    assert 0 in Range(oo, -oo, -1)
    # 验证oo是否不在Range(-oo, oo)范围内
    assert oo not in Range(-oo,
    # 断言语句，用于检查表达式 `-oo not in Range(-oo, oo)` 的真假
    assert -oo not in Range(-oo, oo)
# 定义测试函数 test_issue_17859，用于验证 Range 对象的行为
def test_issue_17859():
    # 创建一个无限范围对象 r，从负无穷到正无穷
    r = Range(-oo,oo)
    # 断言：尝试使用步长为 2 访问该范围时会引发 ValueError 异常
    raises(ValueError, lambda: r[::2])
    # 断言：尝试使用步长为 -2 访问该范围时会引发 ValueError 异常
    raises(ValueError, lambda: r[::-2])
    
    # 创建一个范围对象 r，从正无穷到负无穷，步长为 -1
    r = Range(oo,-oo,-1)
    # 断言：尝试使用步长为 2 访问该范围时会引发 ValueError 异常
    raises(ValueError, lambda: r[::2])
    # 断言：尝试使用步长为 -2 访问该范围时会引发 ValueError 异常
    raises(ValueError, lambda: r[::-2])
```