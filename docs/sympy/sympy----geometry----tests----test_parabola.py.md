# `D:\src\scipysrc\sympy\sympy\geometry\tests\test_parabola.py`

```
from sympy.core.numbers import (Rational, oo)  # 导入有理数和无穷大
from sympy.core.singleton import S  # 导入SymPy单例S
from sympy.core.symbol import symbols  # 导入符号变量
from sympy.functions.elementary.complexes import sign  # 导入符号函数
from sympy.functions.elementary.miscellaneous import sqrt  # 导入平方根函数
from sympy.geometry.ellipse import (Circle, Ellipse)  # 导入椭圆和圆
from sympy.geometry.line import (Line, Ray2D, Segment2D)  # 导入线段、射线和线段
from sympy.geometry.parabola import Parabola  # 导入抛物线类
from sympy.geometry.point import (Point, Point2D)  # 导入点和二维点
from sympy.testing.pytest import raises  # 导入测试函数raises

from sympy.abc import x, y  # 导入符号x和y

def test_parabola_geom():
    a, b = symbols('a b')  # 定义符号变量a和b

    # 定义一些点
    p1 = Point(0, 0)
    p2 = Point(3, 7)
    p3 = Point(0, 4)
    p4 = Point(6, 0)
    p5 = Point(a, a)

    # 定义一些直线
    d1 = Line(Point(4, 0), Point(4, 9))
    d2 = Line(Point(7, 6), Point(3, 6))
    d3 = Line(Point(4, 0), slope=oo)
    d4 = Line(Point(7, 6), slope=0)
    d5 = Line(Point(b, a), slope=oo)
    d6 = Line(Point(a, b), slope=0)

    half = S.Half  # 定义SymPy的1/2常量

    # 创建不同参数的抛物线对象
    pa1 = Parabola(None, d2)
    pa2 = Parabola(directrix=d1)
    pa3 = Parabola(p1, d1)
    pa4 = Parabola(p2, d2)
    pa5 = Parabola(p2, d4)
    pa6 = Parabola(p3, d2)
    pa7 = Parabola(p2, d1)
    pa8 = Parabola(p4, d1)
    pa9 = Parabola(p4, d3)
    pa10 = Parabola(p5, d5)
    pa11 = Parabola(p5, d6)

    # 创建带有对称轴的抛物线对象
    d = Line(Point(3, 7), Point(2, 9))
    pa12 = Parabola(Point(7, 8), d)
    pa12r = Parabola(Point(7, 8).reflect(d), d)

    # 检查异常情况
    raises(ValueError, lambda: Parabola(Point(7, 8, 9), Line(Point(6, 7), Point(7, 7))))
    raises(ValueError, lambda: Parabola(Point(0, 2), Line(Point(7, 2), Point(6, 2))))
    raises(ValueError, lambda: Parabola(Point(7, 8), Point(3, 8)))

    # 基本验证
    assert pa1.focus == Point(0, 0)
    assert pa1.ambient_dimension == S(2)
    assert pa2 == pa3
    assert pa4 != pa7
    assert pa6 != pa7
    assert pa6.focus == Point2D(0, 4)
    assert pa6.focal_length == 1
    assert pa6.p_parameter == -1
    assert pa6.vertex == Point2D(0, 5)
    assert pa6.eccentricity == 1
    assert pa7.focus == Point2D(3, 7)
    assert pa7.focal_length == half
    assert pa7.p_parameter == -half
    assert pa7.vertex == Point2D(7*half, 7)
    assert pa4.focal_length == half
    assert pa4.p_parameter == half
    assert pa4.vertex == Point2D(3, 13*half)
    assert pa8.focal_length == 1
    assert pa8.p_parameter == 1
    assert pa8.vertex == Point2D(5, 0)
    assert pa4.focal_length == pa5.focal_length
    assert pa4.p_parameter == pa5.p_parameter
    assert pa4.vertex == pa5.vertex
    assert pa4.equation() == pa5.equation()
    assert pa8.focal_length == pa9.focal_length
    assert pa8.p_parameter == pa9.p_parameter
    assert pa8.vertex == pa9.vertex
    assert pa8.equation() == pa9.equation()
    assert pa10.focal_length == pa11.focal_length == sqrt((a - b) ** 2) / 2  # 如果a, b是实数，则为abs(a - b)/2
    assert pa11.vertex == Point(*pa10.vertex[::-1]) == Point(a,
                            a - sqrt((a - b)**2)*sign(a - b)/2)  # 将轴改变为x->y, y->x在pa10

    aos = pa12.axis_of_symmetry  # 获取对称轴
    # 验证焦点在 (7, 8) 且经过 (5, 7) 的直线
    assert aos == Line(Point(7, 8), Point(5, 7))
    
    # 验证焦点在 (3, 7) 和 (2, 9) 之间的直线
    assert pa12.directrix == Line(Point(3, 7), Point(2, 9))
    
    # 验证焦点在 (3, 7) 和 (2, 9) 之间的直线与 aos 相互垂直
    assert pa12.directrix.angle_between(aos) == S.Pi/2
    
    # 验证离心率为 1
    assert pa12.eccentricity == 1
    
    # 验证椭圆的方程式
    assert pa12.equation(x, y) == (x - 7)**2 + (y - 8)**2 - (-2*x - y + 13)**2/5
    
    # 验证焦距长度
    assert pa12.focal_length == 9*sqrt(5)/10
    
    # 验证焦点位置
    assert pa12.focus == Point(7, 8)
    
    # 验证参数 p
    assert pa12.p_parameter == 9*sqrt(5)/10
    
    # 验证顶点位置
    assert pa12.vertex == Point2D(S(26)/5, S(71)/10)
    
    # 验证逆向椭圆的焦距长度
    assert pa12r.focal_length == 9*sqrt(5)/10
    
    # 验证逆向椭圆的焦点位置
    assert pa12r.focus == Point(-S(1)/5, S(22)/5)
    
    # 验证逆向椭圆的参数 p
    assert pa12r.p_parameter == -9*sqrt(5)/10
    
    # 验证逆向椭圆的顶点位置
    assert pa12r.vertex == Point(S(8)/5, S(53)/10)
# 定义一个用于测试抛物线相交功能的函数
def test_parabola_intersection():
    # 创建三条直线对象，用于构造抛物线对象
    l1 = Line(Point(1, -2), Point(-1,-2))
    l2 = Line(Point(1, 2), Point(-1,2))
    l3 = Line(Point(1, 0), Point(-1,0))

    # 创建三个点对象，用于构造抛物线对象和进行相交测试
    p1 = Point(0,0)
    p2 = Point(0, -2)
    p3 = Point(120, -12)

    # 创建抛物线对象parabola1，基于点p1和直线l1
    parabola1 = Parabola(p1, l1)

    # 测试抛物线与自身的相交情况
    assert parabola1.intersection(parabola1) == [parabola1]

    # 测试抛物线与另一抛物线的相交情况
    assert parabola1.intersection(Parabola(p1, l2)) == [Point2D(-2, 0), Point2D(2, 0)]
    assert parabola1.intersection(Parabola(p2, l3)) == [Point2D(0, -1)]
    assert parabola1.intersection(Parabola(Point(16, 0), l1)) == [Point2D(8, 15)]
    assert parabola1.intersection(Parabola(Point(0, 16), l1)) == [Point2D(-6, 8), Point2D(6, 8)]
    assert parabola1.intersection(Parabola(p3, l3)) == []

    # 测试抛物线与点的相交情况
    assert parabola1.intersection(p1) == []
    assert parabola1.intersection(Point2D(0, -1)) == [Point2D(0, -1)]
    assert parabola1.intersection(Point2D(4, 3)) == [Point2D(4, 3)]

    # 测试抛物线与直线的相交情况
    assert parabola1.intersection(Line(Point2D(-7, 3), Point(12, 3))) == [Point2D(-4, 3), Point2D(4, 3)]
    assert parabola1.intersection(Line(Point(-4, -1), Point(4, -1))) == [Point(0, -1)]
    assert parabola1.intersection(Line(Point(2, 0), Point(0, -2))) == [Point2D(2, 0)]
    raises(TypeError, lambda: parabola1.intersection(Line(Point(0, 0, 0), Point(1, 1, 1))))

    # 测试抛物线与线段的相交情况
    assert parabola1.intersection(Segment2D((-4, -5), (4, 3))) == [Point2D(0, -1), Point2D(4, 3)]
    assert parabola1.intersection(Segment2D((0, -5), (0, 6))) == [Point2D(0, -1)]
    assert parabola1.intersection(Segment2D((-12, -65), (14, -68))) == []

    # 测试抛物线与射线的相交情况
    assert parabola1.intersection(Ray2D((-4, -5), (4, 3))) == [Point2D(0, -1), Point2D(4, 3)]
    assert parabola1.intersection(Ray2D((0, 7), (1, 14))) == [Point2D(14 + 2*sqrt(57), 105 + 14*sqrt(57))]
    assert parabola1.intersection(Ray2D((0, 7), (0, 14))) == []

    # 测试抛物线与椭圆/圆的相交情况
    assert parabola1.intersection(Circle(p1, 2)) == [Point2D(-2, 0), Point2D(2, 0)]
    assert parabola1.intersection(Circle(p2, 1)) == [Point2D(0, -1)]
    assert parabola1.intersection(Ellipse(p2, 2, 1)) == [Point2D(0, -1)]
    assert parabola1.intersection(Ellipse(Point(0, 19), 5, 7)) == []
    assert parabola1.intersection(Ellipse((0, 3), 12, 4)) == [
           Point2D(0, -1),
           Point2D(-4*sqrt(17)/3, Rational(59, 9)),
           Point2D(4*sqrt(17)/3, Rational(59, 9))]

    # 测试抛物线与不支持的类型的相交情况
    raises(TypeError, lambda: parabola1.intersection(2))
```