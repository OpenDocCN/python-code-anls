# `D:\src\scipysrc\sympy\sympy\calculus\tests\test_singularities.py`

```
# 导入必要的符号和函数
from sympy.core.numbers import (I, Rational, pi, oo)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol, Dummy
from sympy.core.function import Lambda
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.trigonometric import sec, csc
from sympy.functions.elementary.hyperbolic import (coth, sech,
                                                   atanh, asech, acoth, acsch)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.calculus.singularities import (
    singularities,
    is_increasing,
    is_strictly_increasing,
    is_decreasing,
    is_strictly_decreasing,
    is_monotonic
)
from sympy.sets import Interval, FiniteSet, Union, ImageSet
from sympy.testing.pytest import raises
from sympy.abc import x, y


def test_singularities():
    # 创建符号变量 x
    x = Symbol('x')
    # 断言 x**2 的奇点集是空集
    assert singularities(x**2, x) == S.EmptySet
    # 断言 x/(x**2 + 3*x + 2) 的奇点集是 {-2, -1}
    assert singularities(x/(x**2 + 3*x + 2), x) == FiniteSet(-2, -1)
    # 断言 1/(x**2 + 1) 的奇点集是 {I, -I}
    assert singularities(1/(x**2 + 1), x) == FiniteSet(I, -I)
    # 断言 x/(x**3 + 1) 的奇点集是 {-1, (1 - sqrt(3)*I)/2, (1 + sqrt(3)*I)/2}
    assert singularities(x/(x**3 + 1), x) == \
        FiniteSet(-1, (1 - sqrt(3)*I)/2, (1 + sqrt(3)*I)/2)
    # 断言 1/(y**2 + 2*I*y + 1) 的奇点集是 {-I + sqrt(2)*I, -I - sqrt(2)*I}
    assert singularities(1/(y**2 + 2*I*y + 1), y) == \
        FiniteSet(-I + sqrt(2)*I, -I - sqrt(2)*I)
    # 创建虚拟变量 _n
    _n = Dummy('n')
    # 断言 sech(x) 的奇点集是 Union(ImageSet(Lambda(_n, 2*_n*I*pi + I*pi/2), S.Integers),
    #                        ImageSet(Lambda(_n, 2*_n*I*pi + 3*I*pi/2), S.Integers))
    assert singularities(sech(x), x).dummy_eq(Union(
        ImageSet(Lambda(_n, 2*_n*I*pi + I*pi/2), S.Integers),
        ImageSet(Lambda(_n, 2*_n*I*pi + 3*I*pi/2), S.Integers)))
    # 断言 coth(x) 的奇点集是 Union(ImageSet(Lambda(_n, 2*_n*I*pi + I*pi), S.Integers),
    #                        ImageSet(Lambda(_n, 2*_n*I*pi), S.Integers))
    assert singularities(coth(x), x).dummy_eq(Union(
        ImageSet(Lambda(_n, 2*_n*I*pi + I*pi), S.Integers),
        ImageSet(Lambda(_n, 2*_n*I*pi), S.Integers)))
    # 断言 atanh(x) 的奇点集是 {-1, 1}
    assert singularities(atanh(x), x) == FiniteSet(-1, 1)
    # 断言 acoth(x) 的奇点集是 {-1, 1}
    assert singularities(acoth(x), x) == FiniteSet(-1, 1)
    # 断言 asech(x) 的奇点集是 {0}
    assert singularities(asech(x), x) == FiniteSet(0)
    # 断言 acsch(x) 的奇点集是 {0}
    assert singularities(acsch(x), x) == FiniteSet(0)

    # 创建实数变量 x
    x = Symbol('x', real=True)
    # 断言 1/(x**2 + 1) 在实数集上的奇点集是空集
    assert singularities(1/(x**2 + 1), x) == S.EmptySet
    # 断言 exp(1/x) 在实数集上的奇点集是 {0}
    assert singularities(exp(1/x), x, S.Reals) == FiniteSet(0)
    # 断言 exp(1/x) 在区间 (1, 2) 上的奇点集是空集
    assert singularities(exp(1/x), x, Interval(1, 2)) == S.EmptySet
    # 断言 log((x - 2)**2) 在区间 (1, 3) 上的奇点集是 {2}
    assert singularities(log((x - 2)**2), x, Interval(1, 3)) == FiniteSet(2)
    # 断言 raises(NotImplementedError, lambda: singularities(x**-oo, x)) 抛出 NotImplementedError 异常
    raises(NotImplementedError, lambda: singularities(x**-oo, x))
    # 断言 sec(x) 在区间 (0, 3*pi) 上的奇点集是 {pi/2, 3*pi/2, 5*pi/2}
    assert singularities(sec(x), x, Interval(0, 3*pi)) == FiniteSet(
        pi/2, 3*pi/2, 5*pi/2)
    # 断言 csc(x) 在区间 (0, 3*pi) 上的奇点集是 {0, pi, 2*pi, 3*pi}


def test_is_increasing():
    """Test whether is_increasing returns correct value."""
    # 创建负数符号变量 a
    a = Symbol('a', negative=True)

    # 断言 x**3 - 3*x**2 + 4*x 在实数集上是单调递增的
    assert is_increasing(x**3 - 3*x**2 + 4*x, S.Reals)
    # 断言 -x**2 在区间 (-oo, 0) 上是单调递增的
    assert is_increasing(-x**2, Interval(-oo, 0))
    # 断言 -x**2 在区间 (0, oo) 上不是单调递增的
    assert not is_increasing(-x**2, Interval(0, oo))
    # 断言 4*x**3 - 6*x**2 - 72*x + 30 在区间 (-2, 3) 上不是单调递增的
    assert not is_increasing(4*x**3 - 6*x**2 - 72*x + 30, Interval(-2, 3))
    # 断言 x**2 + y 在区间 (1, oo) 上关于 x 是单调递增的
    assert is_increasing(x**2 + y, Interval(1, oo), x)
    # 断言 -x**2*a 在区间 (1, oo) 上关于 x 是单调递增的
    assert is_increasing(-x**2*a, Interval(1, oo), x)
    # 断言 1 是单调递增的
    assert is_increasing(1)

    # 断言 4*x**3 - 6*x**2 - 72*x + 30 在区间 (-2, 3) 上不是单调递增的
    assert is_increasing(4*x**3 - 6*x**2 - 72*x + 30, Interval(-2, 3)) is False
def test_is_strictly_increasing():
    """Test whether is_strictly_increasing returns correct value."""
    # 第一个断言：检查 4*x**3 - 6*x**2 - 72*x + 30 在区间 (-oo, -2] 上是否严格增加
    assert is_strictly_increasing(4*x**3 - 6*x**2 - 72*x + 30, Interval.Ropen(-oo, -2))
    # 第二个断言：检查 4*x**3 - 6*x**2 - 72*x + 30 在区间 (3, oo) 上是否严格增加
    assert is_strictly_increasing(4*x**3 - 6*x**2 - 72*x + 30, Interval.Lopen(3, oo))
    # 第三个断言：检查 4*x**3 - 6*x**2 - 72*x + 30 在区间 (-2, 3) 上是否非严格增加
    assert not is_strictly_increasing(4*x**3 - 6*x**2 - 72*x + 30, Interval.open(-2, 3))
    # 第四个断言：检查 -x**2 在区间 (0, oo) 上是否非严格减少
    assert not is_strictly_increasing(-x**2, Interval(0, oo))
    # 第五个断言：检查是否非严格减少 1
    assert not is_strictly_decreasing(1)
    # 第六个断言：检查 4*x**3 - 6*x**2 - 72*x + 30 在区间 (-2, 3) 上是否不是严格增加
    assert is_strictly_increasing(4*x**3 - 6*x**2 - 72*x + 30, Interval.open(-2, 3)) is False


def test_is_decreasing():
    """Test whether is_decreasing returns correct value."""
    # 符号 b 被定义为正数
    b = Symbol('b', positive=True)
    
    # 第一个断言：检查 1/(x**2 - 3*x) 在区间 (3/2, 3) 上是否严格减少
    assert is_decreasing(1/(x**2 - 3*x), Interval.open(Rational(3,2), 3))
    # 第二个断言：检查 1/(x**2 - 3*x) 在区间 (1.5, 3) 上是否严格减少
    assert is_decreasing(1/(x**2 - 3*x), Interval.open(1.5, 3))
    # 第三个断言：检查 1/(x**2 - 3*x) 在区间 (3, oo) 上是否严格减少
    assert is_decreasing(1/(x**2 - 3*x), Interval.Lopen(3, oo))
    # 第四个断言：检查 1/(x**2 - 3*x) 在区间 (-oo, 3/2) 上是否非严格减少
    assert not is_decreasing(1/(x**2 - 3*x), Interval.Ropen(-oo, Rational(3, 2)))
    # 第五个断言：检查 -x**2 在区间 (-oo, 0) 上是否非严格减少
    assert not is_decreasing(-x**2, Interval(-oo, 0))
    # 第六个断言：检查 -x**2*b 在区间 (-oo, 0) 上是否非严格减少，这里的 x 是自由变量
    assert not is_decreasing(-x**2*b, Interval(-oo, 0), x)


def test_is_strictly_decreasing():
    """Test whether is_strictly_decreasing returns correct value."""
    # 第一个断言：检查 1/(x**2 - 3*x) 在区间 (3, oo) 上是否严格减少
    assert is_strictly_decreasing(1/(x**2 - 3*x), Interval.Lopen(3, oo))
    # 第二个断言：检查 1/(x**2 - 3*x) 在区间 (-oo, 3/2) 上是否非严格减少
    assert not is_strictly_decreasing(1/(x**2 - 3*x), Interval.Ropen(-oo, Rational(3, 2)))
    # 第三个断言：检查 -x**2 在区间 (-oo, 0) 上是否非严格减少
    assert not is_strictly_decreasing(-x**2, Interval(-oo, 0))
    # 第四个断言：检查是否非严格减少 1
    assert not is_strictly_decreasing(1)
    # 第五个断言：检查 1/(x**2 - 3*x) 在区间 (3/2, 3) 上是否严格减少
    assert is_strictly_decreasing(1/(x**2 - 3*x), Interval.open(Rational(3,2), 3))
    # 第六个断言：检查 1/(x**2 - 3*x) 在区间 (1.5, 3) 上是否严格减少
    assert is_strictly_decreasing(1/(x**2 - 3*x), Interval.open(1.5, 3))


def test_is_monotonic():
    """Test whether is_monotonic returns correct value."""
    # 第一个断言：检查 1/(x**2 - 3*x) 在区间 (3/2, 3) 上是否单调
    assert is_monotonic(1/(x**2 - 3*x), Interval.open(Rational(3,2), 3))
    # 第二个断言：检查 1/(x**2 - 3*x) 在区间 (1.5, 3) 上是否单调
    assert is_monotonic(1/(x**2 - 3*x), Interval.open(1.5, 3))
    # 第三个断言：检查 1/(x**2 - 3*x) 在区间 (3, oo) 上是否单调
    assert is_monotonic(1/(x**2 - 3*x), Interval.Lopen(3, oo))
    # 第四个断言：检查 x**3 - 3*x**2 + 4*x 在全体实数上是否单调
    assert is_monotonic(x**3 - 3*x**2 + 4*x, S.Reals)
    # 第五个断言：检查 -x**2 在全体实数上是否非单调
    assert not is_monotonic(-x**2, S.Reals)
    # 第六个断言：检查 x**2 + y + 1 在区间 [1, 2] 上关于 x 是否单调，y 是自由变量
    assert is_monotonic(x**2 + y + 1, Interval(1, 2), x)
    # 第七个断言：尝试对未实现的函数调用，预期抛出 NotImplementedError
    raises(NotImplementedError, lambda: is_monotonic(x**2 + y + 1))


def test_issue_23401():
    x = Symbol('x')
    # 构建表达式
    expr = (x + 1)/(-1.0e-3*x**2 + 0.1*x + 0.1)
    # 断言：检查表达式在区间 [1, 2] 上是否单调递增，其中 x 是自由变量
    assert is_increasing(expr, Interval(1,2), x)
```