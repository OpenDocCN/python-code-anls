# `D:\src\scipysrc\sympy\sympy\geometry\tests\test_curve.py`

```
# 导入需要的符号和函数模块
from sympy.core.containers import Tuple
from sympy.core.numbers import (Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.hyperbolic import asinh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.geometry import Curve, Line, Point, Ellipse, Ray, Segment, Circle, Polygon, RegularPolygon
from sympy.testing.pytest import raises, slow


# 定义测试函数 test_curve
def test_curve():
    # 定义实数符号变量
    x = Symbol('x', real=True)
    # 定义符号变量 s, z
    s = Symbol('s')
    z = Symbol('z')

    # 创建一个参数为 z 的 Curve 对象，函数为 (2*s, s**2)
    c = Curve([2*s, s**2], (z, 0, 2))

    # 断言参数为 z
    assert c.parameter == z
    # 断言函数为 (2*s, s**2)
    assert c.functions == (2*s, s**2)
    # 断言任意点的返回值为 Point(2*s, s**2)
    assert c.arbitrary_point() == Point(2*s, s**2)
    # 断言指定参数的任意点的返回值为 Point(2*s, s**2)
    assert c.arbitrary_point(z) == Point(2*s, s**2)

    # 使用参数 s 创建 Curve 对象，函数为 (2*s, s**2)
    c = Curve([2*s, s**2], (s, 0, 2))

    # 断言参数为 s
    assert c.parameter == s
    # 断言函数为 (2*s, s**2)
    assert c.functions == (2*s, s**2)
    
    # 定义符号变量 t，并断言未指定符号 t 的任意点不等于 Point(2*t, t**2)
    t = Symbol('t')
    assert c.arbitrary_point() != Point(2*t, t**2)
    
    # 现在指定 t 为实数符号变量，断言任意点为 Point(2*t, t**2)
    t = Symbol('t', real=True)
    assert c.arbitrary_point() == Point(2*t, t**2)
    # 断言任意点的返回值为 Point(2*z, z**2)
    assert c.arbitrary_point(z) == Point(2*z, z**2)
    # 断言任意点的返回值为 Point(2*s, s**2)
    assert c.arbitrary_point(c.parameter) == Point(2*s, s**2)
    # 断言任意点的返回值为 Point(2*s, s**2)
    assert c.arbitrary_point(None) == Point(2*s, s**2)
    # 断言绘图区间的返回值为 [t, 0, 2]
    assert c.plot_interval() == [t, 0, 2]
    # 断言指定参数的绘图区间的返回值为 [z, 0, 2]
    assert c.plot_interval(z) == [z, 0, 2]

    # 断言 Curve([x, x], (x, 0, 1)) 绕 pi/2 弧度旋转后与 Curve([-x, x], (x, 0, 1)) 相等
    assert Curve([x, x], (x, 0, 1)).rotate(pi/2) == Curve([-x, x], (x, 0, 1))
    
    # 断言 Curve([x, x], (x, 0, 1)) 绕 (1, 2) 点旋转 pi/2 弧度，缩放 2 倍和 3 倍，平移 (1, 3) 后的任意点与预期值 Point(-2*s + 7, 3*s + 6) 相等
    assert Curve([x, x], (x, 0, 1)).rotate(pi/2, (1, 2)).scale(2, 3).translate(
        1, 3).arbitrary_point(s) == \
        Line((0, 0), (1, 1)).rotate(pi/2, (1, 2)).scale(2, 3).translate(
            1, 3).arbitrary_point(s) == \
        Point(-2*s + 7, 3*s + 6)

    # 断言 Curve((s), (s, 1, 2)) 抛出 ValueError 异常
    raises(ValueError, lambda: Curve((s), (s, 1, 2)))
    # 断言 Curve((x, x * 2), (1, x)) 抛出 ValueError 异常
    raises(ValueError, lambda: Curve((x, x * 2), (1, x)))

    # 断言 Curve((s, s + t), (s, 1, 2)) 在未指定参数时抛出 ValueError 异常
    raises(ValueError, lambda: Curve((s, s + t), (s, 1, 2)).arbitrary_point())
    # 断言 Curve((s, s + t), (t, 1, 2)) 在指定参数 s 时抛出 ValueError 异常
    raises(ValueError, lambda: Curve((s, s + t), (t, 1, 2)).arbitrary_point(s))


# 带有 slow 标记的测试函数 test_free_symbols
@slow
def test_free_symbols():
    # 定义符号变量 a, b, c, d, e, f, s
    a, b, c, d, e, f, s = symbols('a:f,s')
    
    # 断言 Point(a, b) 的自由符号为 {a, b}
    assert Point(a, b).free_symbols == {a, b}
    # 断言 Line((a, b), (c, d)) 的自由符号为 {a, b, c, d}
    assert Line((a, b), (c, d)).free_symbols == {a, b, c, d}
    # 断言 Ray((a, b), (c, d)) 的自由符号为 {a, b, c, d}
    assert Ray((a, b), (c, d)).free_symbols == {a, b, c, d}
    # 断言 Ray((a, b), angle=c) 的自由符号为 {a, b, c}
    assert Ray((a, b), angle=c).free_symbols == {a, b, c}
    # 断言 Segment((a, b), (c, d)) 的自由符号为 {a, b, c, d}
    assert Segment((a, b), (c, d)).free_symbols == {a, b, c, d}
    # 断言 Line((a, b), slope=c) 的自由符号为 {a, b, c}
    assert Line((a, b), slope=c).free_symbols == {a, b, c}
    # 断言 Curve((a*s, b*s), (s, c, d)) 的自由符号为 {a, b, c, d}
    assert Curve((a*s, b*s), (s, c, d)).free_symbols == {a, b, c, d}
    # 断言 Ellipse((a, b), c, d) 的自由符号为 {a, b, c, d}
    assert Ellipse((a, b), c, d).free_symbols == {a, b, c, d}
    # 断言 Ellipse((a, b), c, eccentricity=d) 的自由符号为 {a, b, c, d}
    assert Ellipse((a, b), c, eccentricity=d).free_symbols == \
        {a, b, c, d}
    # 断言 Ellipse((a, b), vradius=c, eccentricity=d) 的自由符号为 {a, b, c, d}
    assert Ellipse((a, b), vradius=c, eccentricity=d).free_symbols == \
        {a, b, c, d}
    # 断言 Circle((a, b), c) 的自由符号为 {a, b, c}
    assert Circle((a, b), c).free_symbols == {a, b, c}
    # 断言 Circle((a, b), (c, d), (e, f)) 的自由符号为 {e, d, c, b, f, a}
    assert Circle((a, b), (c, d), (e, f)).free_symbols == \
        {e, d, c, b, f, a}
    # 断言：创建一个由三个点构成的多边形对象，并检查其自由符号集合是否等于指定的集合
    assert Polygon((a, b), (c, d), (e, f)).free_symbols == \
        {e, b, d, f, a, c}
    
    # 断言：创建一个正多边形对象，并检查其自由符号集合是否等于指定的集合
    assert RegularPolygon((a, b), c, d, e).free_symbols == {e, a, b, c, d}
# 定义一个名为 test_transform 的测试函数
def test_transform():
    # 定义实数域上的符号变量 x 和 y
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    # 创建一条曲线对象 c，表示为 (x, x**2)，定义域为 x 从 0 到 1
    c = Curve((x, x**2), (x, 0, 1))
    # 创建另一条曲线对象 cout，表示为 (2*x - 4, 3*x**2 - 10)，定义域为 x 从 0 到 1
    cout = Curve((2*x - 4, 3*x**2 - 10), (x, 0, 1))
    # 创建点集合 pts，包含 Point(0, 0)，Point(S.Half, Rational(1, 4))，Point(1, 1)
    pts = [Point(0, 0), Point(S.Half, Rational(1, 4)), Point(1, 1)]
    # 创建点集合 pts_out，包含 Point(-4, -10)，Point(-3, Rational(-37, 4))，Point(-2, -7)
    pts_out = [Point(-4, -10), Point(-3, Rational(-37, 4)), Point(-2, -7)]

    # 断言 c 缩放为 (2, 3)，平移为 (4, 5) 后等于 cout
    assert c.scale(2, 3, (4, 5)) == cout
    # 断言 c 在将 x 替换为 xi/2 后的结果集合等于 pts
    assert [c.subs(x, xi/2) for xi in Tuple(0, 1, 2)] == pts
    # 断言 cout 在将 x 替换为 xi/2 后的结果集合等于 pts_out
    assert [cout.subs(x, xi/2) for xi in Tuple(0, 1, 2)] == pts_out
    # 断言曲线 (x + y, 3*x) 在将 y 替换为 S.Half 后等于曲线 (x + S.Half, 3*x)
    assert Curve((x + y, 3*x), (x, 0, 1)).subs(y, S.Half) == \
        Curve((x + S.Half, 3*x), (x, 0, 1))
    # 断言曲线 (x, 3*x) 平移为 (4, 5) 后等于曲线 (x + 4, 3*x + 5)
    assert Curve((x, 3*x), (x, 0, 1)).translate(4, 5) == \
        Curve((x + 4, 3*x + 5), (x, 0, 1))


# 定义一个名为 test_length 的测试函数
def test_length():
    # 定义实数域上的符号变量 t
    t = Symbol('t', real=True)
    
    # 创建一条曲线对象 c1，表示为 (t, 0)，定义域为 t 从 0 到 1
    c1 = Curve((t, 0), (t, 0, 1))
    # 断言 c1 的长度为 1
    assert c1.length == 1
    
    # 创建一条曲线对象 c2，表示为 (t, t)，定义域为 t 从 0 到 1
    c2 = Curve((t, t), (t, 0, 1))
    # 断言 c2 的长度为 sqrt(2)
    assert c2.length == sqrt(2)
    
    # 创建一条曲线对象 c3，表示为 (t**2, t)，定义域为 t 从 2 到 5
    c3 = Curve((t ** 2, t), (t, 2, 5))
    # 断言 c3 的长度为 -sqrt(17) - asinh(4) / 4 + asinh(10) / 4 + 5 * sqrt(101) / 2
    assert c3.length == -sqrt(17) - asinh(4) / 4 + asinh(10) / 4 + 5 * sqrt(101) / 2


# 定义一个名为 test_parameter_value 的测试函数
def test_parameter_value():
    # 定义符号变量 t
    t = Symbol('t')
    # 创建曲线对象 C，表示为 (2*t, t**2)，定义域为 t 从 0 到 2
    C = Curve([2*t, t**2], (t, 0, 2))
    # 断言 C 在参数值为 (2, 1) 时的参数 t 为 1
    assert C.parameter_value((2, 1), t) == {t: 1}
    # 断言曲线对象 C 在参数值为 (2, 0) 时抛出 ValueError 异常
    raises(ValueError, lambda: C.parameter_value((2, 0), t))


# 定义一个名为 test_issue_17997 的测试函数
def test_issue_17997():
    # 定义符号变量 t 和 s
    t, s = symbols('t s')
    # 创建曲线对象 c，表示为 (t, t**2)，定义域为 t 从 0 到 10
    c = Curve((t, t**2), (t, 0, 10))
    # 创建曲线对象 p，表示为 (2*s, s**2)，定义域为 s 从 0 到 2
    p = Curve([2*s, s**2], (s, 0, 2))
    # 断言曲线对象 c 在参数值为 2 时的点为 Point(2, 4)
    assert c(2) == Point(2, 4)
    # 断言曲线对象 p 在参数值为 1 时的点为 Point(2, 1)
    assert p(1) == Point(2, 1)
```