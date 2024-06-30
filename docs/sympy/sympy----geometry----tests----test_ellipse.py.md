# `D:\src\scipysrc\sympy\sympy\geometry\tests\test_ellipse.py`

```
# 导入Sympy库中的expand函数，用于展开表达式
from sympy.core import expand
# 导入Sympy库中的Rational、oo（无穷大）、pi等数值常量
from sympy.core.numbers import (Rational, oo, pi)
# 导入Sympy库中的Eq函数，用于创建方程式
from sympy.core.relational import Eq
# 导入Sympy库中的S（Singleton）类，用于创建单例对象
from sympy.core.singleton import S
# 导入Sympy库中的Symbol和symbols函数，用于创建符号变量
from sympy.core.symbol import (Symbol, symbols)
# 导入Sympy库中的Abs函数，用于计算绝对值
from sympy.functions.elementary.complexes import Abs
# 导入Sympy库中的sqrt函数，用于计算平方根
from sympy.functions.elementary.miscellaneous import sqrt
# 导入Sympy库中的sec函数，用于计算secant函数
from sympy.functions.elementary.trigonometric import sec
# 导入Sympy库中的Segment2D类，用于表示二维线段
from sympy.geometry.line import Segment2D
# 导入Sympy库中的Point2D类，用于表示二维点
from sympy.geometry.point import Point2D
# 导入Sympy库中的各种几何图形类，用于表示圆、椭圆、线段等
from sympy.geometry import (Circle, Ellipse, GeometryError, Line, Point,
                            Polygon, Ray, RegularPolygon, Segment,
                            Triangle, intersection)
# 导入Sympy库中的raises和slow装饰器，用于测试中的异常检查和标记慢速测试
from sympy.testing.pytest import raises, slow
# 导入Sympy库中的积分函数integrate
from sympy.integrals.integrals import integrate
# 导入Sympy库中的elliptic_e函数，用于计算椭圆积分
from sympy.functions.special.elliptic_integrals import elliptic_e
# 导入Sympy库中的Max函数，用于求取最大值
from sympy.functions.elementary.miscellaneous import Max


def test_ellipse_equation_using_slope():
    # 导入Sympy库中的符号x和y
    from sympy.abc import x, y

    # 创建一个以点(1, 0)为中心，长轴为3，短轴为2的椭圆对象e1
    e1 = Ellipse(Point(1, 0), 3, 2)
    # 断言e1对象的方程，其中斜率_slope设为1时的表达式
    assert str(e1.equation(_slope=1)) == str((-x + y + 1)**2/8 + (x + y - 1)**2/18 - 1)

    # 创建一个以原点为中心，长轴为4，短轴为1的椭圆对象e2
    e2 = Ellipse(Point(0, 0), 4, 1)
    # 断言e2对象的方程，其中斜率_slope设为1时的表达式
    assert str(e2.equation(_slope=1)) == str((-x + y)**2/2 + (x + y)**2/32 - 1)

    # 创建一个以点(1, 5)为中心，长轴为6，短轴为2的椭圆对象e3
    e3 = Ellipse(Point(1, 5), 6, 2)
    # 断言e3对象的方程，其中斜率_slope设为2时的表达式
    assert str(e3.equation(_slope=2)) == str((-2*x + y - 3)**2/20 + (x + 2*y - 11)**2/180 - 1)


def test_object_from_equation():
    # 导入Sympy库中的符号x、y、a、b、c、d、e
    from sympy.abc import x, y, a, b, c, d, e
    # 断言根据方程x**2 + y**2 + 3*x + 4*y - 8创建的圆对象与预期的圆相等
    assert Circle(x**2 + y**2 + 3*x + 4*y - 8) == Circle(Point2D(S(-3) / 2, -2), sqrt(57) / 2)
    # 断言根据方程x**2 + y**2 + 6*x + 8*y + 25创建的圆对象与预期的圆相等
    assert Circle(x**2 + y**2 + 6*x + 8*y + 25) == Circle(Point2D(-3, -4), 0)
    # 断言根据方程a**2 + b**2 + 6*a + 8*b + 25创建的圆对象与预期的圆相等，同时指定x和y的符号表示
    assert Circle(a**2 + b**2 + 6*a + 8*b + 25, x='a', y='b') == Circle(Point2D(-3, -4), 0)
    # 断言根据方程x**2 + y**2 - 25创建的圆对象与预期的圆相等
    assert Circle(x**2 + y**2 - 25) == Circle(Point2D(0, 0), 5)
    # 断言根据方程x**2 + y**2创建的圆对象与预期的圆相等
    assert Circle(x**2 + y**2) == Circle(Point2D(0, 0), 0)
    # 断言根据方程a**2 + b**2创建的圆对象与预期的圆相等，同时指定x和y的符号表示
    assert Circle(a**2 + b**2, x='a', y='b') == Circle(Point2D(0, 0), 0)
    # 断言根据方程x**2 + y**2 + 6*x + 8创建的圆对象与预期的圆相等
    assert Circle(x**2 + y**2 + 6*x + 8) == Circle(Point2D(-3, 0), 1)
    # 断言根据方程x**2 + y**2 + 6*y + 8创建的圆对象与预期的圆相等
    assert Circle(x**2 + y**2 + 6*y + 8) == Circle(Point2D(0, -3), 1)
    # 断言根据方程(x - 1)**2 + y**2 - 9创建的圆对象与预期的圆相等
    assert Circle((x - 1)**2 + y**2 - 9) == Circle(Point2D(1, 0), 3)
    # 断言根据方程6*(x**2) + 6*(y**2) + 6*x + 8*y - 25创建的圆对象与预期的圆相等
    assert Circle(6*(x**2) + 6*(y**2) + 6*x + 8*y - 25) == Circle(Point2D(Rational(-1, 2), Rational(-2, 3)), 5*sqrt(7)/6)
    # 断言根据方程Eq(a**2 + b**2, 25)创建的圆对象与预期的圆相等，同时指定x和y的符号表示
    assert Circle(Eq(a**2 + b**2, 25), x='a', y=b) == Circle(Point2D(0, 0), 5)
    # 断言创建Circle对象时引发GeometryError异常，预期为x**2 + y**2 + 3*x + 4*y + 26无法创建圆
    raises(GeometryError, lambda: Circle(x**2 + y**2 + 3*x + 4*y + 26))
    # 断言创建Circle对象时引发GeometryError异常，预期为x**2 + y**2 + 25无法创建圆
    raises(GeometryError, lambda: Circle(x**2 + y**2 + 25))
    # 断言创建Circle对象时引发GeometryError异常，预期为a**2 + b**2 + 25无法创建圆，同时指定x和y的符号表示
    raises(Ge
    # 创建一个实数符号 t
    t = Symbol('t', real=True)
    # 创建一个实数符号 y1
    y1 = Symbol('y1', real=True)
    # 创建有理数常量 1/2
    half = S.Half
    # 创建点对象 p1，坐标为 (0, 0)
    p1 = Point(0, 0)
    # 创建点对象 p2，坐标为 (1, 1)
    p2 = Point(1, 1)
    # 创建点对象 p4，坐标为 (0, 1)
    p4 = Point(0, 1)

    # 创建椭圆对象 e1，中心为 p1，长半轴 1，短半轴 1
    e1 = Ellipse(p1, 1, 1)
    # 创建椭圆对象 e2，中心为 p2，长半轴 1/2，短半轴 1
    e2 = Ellipse(p2, half, 1)
    # 创建椭圆对象 e3，中心为 p1，长半轴 y1，短半轴 y1
    e3 = Ellipse(p1, y1, y1)
    # 创建圆对象 c1，中心为 p1，半径 1
    c1 = Circle(p1, 1)
    # 创建圆对象 c2，中心为 p2，半径 1
    c2 = Circle(p2, 1)
    # 创建圆对象 c3，中心为 (sqrt(2), sqrt(2))，半径 1
    c3 = Circle(Point(sqrt(2), sqrt(2)), 1)
    # 创建直线对象 l1，通过点 p1 和 p2
    l1 = Line(p1, p2)

    # 使用三个点来测试创建圆的情况
    cen, rad = Point(3*half, 2), 5*half
    assert Circle(Point(0, 0), Point(3, 0), Point(0, 4)) == Circle(cen, rad)
    assert Circle(Point(0, 0), Point(1, 1), Point(2, 2)) == Segment2D(Point2D(0, 0), Point2D(2, 2))

    # 验证错误情况的抛出
    raises(ValueError, lambda: Ellipse(None, None, None, 1))
    raises(ValueError, lambda: Ellipse())
    raises(GeometryError, lambda: Circle(Point(0, 0)))
    raises(GeometryError, lambda: Circle(Symbol('x')*Symbol('y')))

    # 基础功能验证
    assert Ellipse(None, 1, 1).center == Point(0, 0)
    assert e1 == c1
    assert e1 != e2
    assert e1 != l1
    assert p4 in e1
    assert e1 in e1
    assert e2 in e2
    assert 1 not in e2
    assert p2 not in e2
    assert e1.area == pi
    assert e2.area == pi/2
    assert e3.area == pi*y1*abs(y1)
    assert c1.area == e1.area
    assert c1.circumference == e1.circumference
    assert e3.circumference == 2*pi*y1
    assert e1.plot_interval() == e2.plot_interval() == [t, -pi, pi]
    assert e1.plot_interval(x) == e2.plot_interval(x) == [x, -pi, pi]

    # 验证圆的属性
    assert c1.minor == 1
    assert c1.major == 1
    assert c1.hradius == 1
    assert c1.vradius == 1

    # 验证椭圆根据不同参数的创建
    assert Ellipse((1, 1), 0, 0) == Point(1, 1)
    assert Ellipse((1, 1), 1, 0) == Segment(Point(0, 1), Point(2, 1))
    assert Ellipse((1, 1), 0, 1) == Segment(Point(1, 0), Point(1, 2))

    # 私有函数验证
    assert hash(c1) == hash(Circle(Point(1, 0), Point(0, 1), Point(0, -1)))
    assert c1 in e1
    assert (Line(p1, p2) in e1) is False
    assert e1.__cmp__(e1) == 0
    assert e1.__cmp__(Point(0, 0)) > 0

    # 包含关系验证
    assert e1.encloses(Segment(Point(-0.5, -0.5), Point(0.5, 0.5))) is True
    assert e1.encloses(Line(p1, p2)) is False
    assert e1.encloses(Ray(p1, p2)) is False
    assert e1.encloses(e1) is False
    assert e1.encloses(
        Polygon(Point(-0.5, -0.5), Point(-0.5, 0.5), Point(0.5, 0.5))) is True
    assert e1.encloses(RegularPolygon(p1, 0.5, 3)) is True
    assert e1.encloses(RegularPolygon(p1, 5, 3)) is False
    assert e1.encloses(RegularPolygon(p2, 5, 3)) is False

    # 获取任意点验证
    assert e2.arbitrary_point() in e2
    raises(ValueError, lambda: Ellipse(Point(x, y), 1, 1).arbitrary_point(parameter='x'))

    # 焦点验证
    f1, f2 = Point(sqrt(12), 0), Point(-sqrt(12), 0)
    ef = Ellipse(Point(0, 0), 4, 2)
    assert ef.foci in [(f1, f2), (f2, f1)]

    # 切线验证
    v = sqrt(2) / 2
    p1_1 = Point(v, v)
    p1_2 = p2 + Point(half, 0)
    p1_3 = p2 + Point(0, 1)
    assert e1.tangent_lines(p4) == c1.tangent_lines(p4)
    assert e2.tangent_lines(p1_2) == [Line(Point(Rational(3, 2), 1), Point(Rational(3, 2), S.Half))]
    # 断言：检查 e2 对象在点 p1_3 处的切线是否为 [Line(Point(1, 2), Point(Rational(5, 4), 2))]
    assert e2.tangent_lines(p1_3) == [Line(Point(1, 2), Point(Rational(5, 4), 2))]
    
    # 断言：检查 c1 对象在点 p1_1 处的切线是否不等于 [Line(p1_1, Point(0, sqrt(2)))]
    assert c1.tangent_lines(p1_1) != [Line(p1_1, Point(0, sqrt(2)))]

    # 断言：检查 c1 对象在点 p1 处的切线是否为空列表 []
    assert c1.tangent_lines(p1) == []

    # 断言：检查 e2 对象是否与 Line(p1_2, p2 + Point(half, 1)) 的切线相切
    assert e2.is_tangent(Line(p1_2, p2 + Point(half, 1)))

    # 断言：检查 e2 对象是否与 Line(p1_3, p2 + Point(half, 1)) 的切线相切
    assert e2.is_tangent(Line(p1_3, p2 + Point(half, 1)))

    # 断言：检查 c1 对象是否与 Line(p1_1, Point(0, sqrt(2))) 的切线相切
    assert c1.is_tangent(Line(p1_1, Point(0, sqrt(2))))

    # 断言：检查 e1 对象是否与 Line(Point(0, 0), Point(1, 1)) 的切线不相切
    assert e1.is_tangent(Line(Point(0, 0), Point(1, 1))) is False

    # 断言：检查 c1 对象是否与 e1 对象的切线相切
    assert c1.is_tangent(e1) is True

    # 断言：检查 c1 对象是否与 Ellipse(Point(2, 0), 1, 1) 的切线相切
    assert c1.is_tangent(Ellipse(Point(2, 0), 1, 1)) is True

    # 断言：检查 c1 对象是否与 Polygon(Point(1, 1), Point(1, -1), Point(2, 0)) 的切线不相切
    assert c1.is_tangent(Polygon(Point(1, 1), Point(1, -1), Point(2, 0))) is False

    # 断言：检查 c1 对象是否与 Polygon(Point(1, 1), Point(1, 0), Point(2, 0)) 的切线不相切
    assert c1.is_tangent(Polygon(Point(1, 1), Point(1, 0), Point(2, 0))) is False

    # 断言：检查 Circle(Point(5, 5), 3) 和 Circle(Point(0, 5), 1) 的切线是否不相切
    assert Circle(Point(5, 5), 3).is_tangent(Circle(Point(0, 5), 1)) is False

    # 断言：检查 Ellipse(Point(5, 5), 2, 1) 在 Point(0, 0) 处的法线是否为 [Line(Point(0, 0), Point(Rational(77, 25), Rational(132, 25))), Line(Point(0, 0), Point(Rational(33, 5), Rational(22, 5)))]
    assert Ellipse(Point(5, 5), 2, 1).tangent_lines(Point(0, 0)) == \
        [Line(Point(0, 0), Point(Rational(77, 25), Rational(132, 25))),
         Line(Point(0, 0), Point(Rational(33, 5), Rational(22, 5)))]

    # 断言：检查 Ellipse(Point(5, 5), 2, 1) 在 Point(3, 4) 处的法线是否为 [Line(Point(3, 4), Point(4, 4)), Line(Point(3, 4), Point(3, 5))]
    assert Ellipse(Point(5, 5), 2, 1).tangent_lines(Point(3, 4)) == \
        [Line(Point(3, 4), Point(4, 4)), Line(Point(3, 4), Point(3, 5))]

    # 断言：检查 Circle(Point(5, 5), 2) 在 Point(3, 3) 处的法线是否为 [Line(Point(3, 3), Point(4, 3)), Line(Point(3, 3), Point(3, 4))]
    assert Circle(Point(5, 5), 2).tangent_lines(Point(3, 3)) == \
        [Line(Point(3, 3), Point(4, 3)), Line(Point(3, 3), Point(3, 4))]

    # 断言：检查 Circle(Point(5, 5), 2) 在 Point(5 - 2*sqrt(2), 5) 处的法线是否为 [Line(Point(5 - 2*sqrt(2), 5), Point(5 - sqrt(2), 5 - sqrt(2))), Line(Point(5 - 2*sqrt(2), 5), Point(5 - sqrt(2), 5 + sqrt(2)))]
    assert Circle(Point(5, 5), 2).tangent_lines(Point(5 - 2*sqrt(2), 5)) == \
        [Line(Point(5 - 2*sqrt(2), 5), Point(5 - sqrt(2), 5 - sqrt(2))),
         Line(Point(5 - 2*sqrt(2), 5), Point(5 - sqrt(2), 5 + sqrt(2)))]

    # 断言：检查 Circle(Point(5, 5), 5) 在 Point(4, 0) 处的法线是否为 [Line(Point(4, 0), Point(Rational(40, 13), Rational(5, 13))), Line(Point(4, 0), Point(5, 0))]
    assert Circle(Point(5, 5), 5).tangent_lines(Point(4, 0)) == \
        [Line(Point(4, 0), Point(Rational(40, 13), Rational(5, 13))),
         Line(Point(4, 0), Point(5, 0))]

    # 断言：检查 Circle(Point(5, 5), 5) 在 Point(0, 6) 处的法线是否为 [Line(Point(0, 6), Point(0, 7)), Line(Point(0, 6), Point(Rational(5, 13), Rational(90, 13)))]
    assert Circle(Point(5, 5), 5).tangent_lines(Point(0, 6)) == \
        [Line(Point(0, 6), Point(0, 7)),
         Line(Point(0, 6), Point(Rational(5, 13), Rational(90, 13)))]

    # 函数定义：用于数值计算，检查两条线段是否在给定精度内相等
    def lines_close(l1, l2, prec):
        """ tests whether l1 and 12 are within 10**(-prec)
        of each other """
        return abs(l1.p1 - l2.p1) < 10**(-prec) and abs(l1.p2 - l2.p2) < 10**(-prec)

    # 函数定义：检查两个线段列表是否在给定精度内相等
    def line_list_close(ll1, ll2, prec):
        return all(lines_close(l1, l2, prec) for l1, l2 in zip(ll1, ll2))

    # 创建 Ellipse 对象 e
    e = Ellipse(Point(0, 0), 2, 1)

    # 断言：检查 Ellipse 对象 e 在 Point(0, 0) 处的法线是否为 [Line(Point(0, 0), Point(0, 1)), Line(Point(0, 0), Point(1, 0))]
    assert e.normal_lines(Point(0, 0)) == \
        [Line(Point(0, 0), Point(0, 1)), Line(Point(0, 0), Point(1, 0))]

    # 断言：检查 Ellipse 对象 e 在 Point(1, 0) 处的法线是否为 [Line(Point(0, 0), Point(1, 0))]
    assert e.normal_lines(Point(1, 0)) == \
        [Line(Point(0, 0), Point(1,
    # 确保在边界上使用不是未定义的斜率
    assert line_list_close(e.normal_lines(p, 2), [
        Line(Point(Rational(-341, 171), Rational(-1, 13)), Point(Rational(-170, 171), Rational(5, 64))),
        Line(Point(Rational(26, 15), Rational(-1, 2)), Point(Rational(41, 15), Rational(-43, 26)))], 2)

    # 创建一个椭圆对象e，以中心(0, 0)，长轴2，短轴2*sqrt(3)/3初始化
    e = Ellipse((0, 0), 2, 2*sqrt(3)/3)
    
    # 确保在一般椭圆形状下的法线失败，除非在特定条件下
    assert line_list_close(e.normal_lines((1, 1), 2), [
        Line(Point(Rational(-64, 33), Rational(-20, 71)), Point(Rational(-31, 33), Rational(2, 13))),
        Line(Point(1, -1), Point(2, -4))], 2)
    
    # 创建一个椭圆对象e，以中心(0, 0)，长轴为x，短轴为1初始化
    e = Ellipse((0, 0), x, 1)
    
    # 确保在特定条件下，与(x + 1, 0)的法线是一个点(0, 0)到点(1, 0)的直线
    assert e.normal_lines((x + 1, 0)) == [Line(Point(0, 0), Point(1, 0))]
    
    # 断言e.normal_lines((x + 1, 1))抛出NotImplementedError异常
    raises(NotImplementedError, lambda: e.normal_lines((x + 1, 1)))
    
    # 属性
    major = 3
    minor = 1
    
    # 创建一个椭圆对象e4，以中心p2，长轴为minor，短轴为major初始化
    e4 = Ellipse(p2, minor, major)
    
    # 断言e4的焦距为sqrt(major**2 - minor**2)
    assert e4.focus_distance == sqrt(major**2 - minor**2)
    
    # 计算e4的离心率为焦距除以长轴
    ecc = e4.focus_distance / major
    
    # 断言e4的离心率为ecc
    assert e4.eccentricity == ecc
    
    # 断言e4的近日点为major*(1 - ecc)
    assert e4.periapsis == major*(1 - ecc)
    
    # 断言e4的远日点为major*(1 + ecc)
    assert e4.apoapsis == major*(1 + ecc)
    
    # 断言e4的半矩形为major*(1 - ecc ** 2)
    assert e4.semilatus_rectum == major*(1 - ecc ** 2)
    
    # 创建一个椭圆对象e4，以中心p2，长轴为major，短轴为minor初始化
    e4 = Ellipse(p2, major, minor)
    
    # 断言e4的焦距为sqrt(major**2 - minor**2)
    assert e4.focus_distance == sqrt(major**2 - minor**2)
    
    # 计算e4的离心率为焦距除以长轴
    ecc = e4.focus_distance / major
    
    # 断言e4的离心率为ecc
    assert e4.eccentricity == ecc
    
    # 断言e4的近日点为major*(1 - ecc)
    assert e4.periapsis == major*(1 - ecc)
    
    # 断言e4的远日点为major*(1 + ecc)
    assert e4.apoapsis == major*(1 + ecc)

    # 交点
    l1 = Line(Point(1, -5), Point(1, 5))
    l2 = Line(Point(-5, -1), Point(5, -1))
    l3 = Line(Point(-1, -1), Point(1, 1))
    l4 = Line(Point(-10, 0), Point(0, 10))
    pts_c1_l3 = [Point(sqrt(2)/2, sqrt(2)/2), Point(-sqrt(2)/2, -sqrt(2)/2)]
    
    # 断言椭圆e2与直线l4没有交点
    assert intersection(e2, l4) == []
    
    # 断言圆c1与点(1, 0)的交点为[Point(1, 0)]
    assert intersection(c1, Point(1, 0)) == [Point(1, 0)]
    
    # 断言圆c1与直线l1的交点为[Point(1, 0)]
    assert intersection(c1, l1) == [Point(1, 0)]
    
    # 断言圆c1与直线l2的交点为[Point(0, -1)]
    assert intersection(c1, l2) == [Point(0, -1)]
    
    # 断言圆c1与直线l3的交点在pts_c1_l3或者[pts_c1_l3[1], pts_c1_l3[0]]中
    assert intersection(c1, l3) in [pts_c1_l3, [pts_c1_l3[1], pts_c1_l3[0]]]
    
    # 断言圆c1与椭圆c2的交点为[Point(0, 1), Point(1, 0)]
    assert intersection(c1, c2) == [Point(0, 1), Point(1, 0)]
    
    # 断言圆c1与椭圆c3的交点为[Point(sqrt(2)/2, sqrt(2)/2)]
    assert intersection(c1, c3) == [Point(sqrt(2)/2, sqrt(2)/2)]
    
    # 断言椭圆e1与直线l1的交点为[Point(1, 0)]
    assert e1.intersection(l1) == [Point(1, 0)]
    
    # 断言椭圆e2与直线l4没有交点
    assert e2.intersection(l4) == []
    
    # 断言椭圆e1与圆Circle(Point(0, 2), 1)的交点为[Point(0, 1)]
    assert e1.intersection(Circle(Point(0, 2), 1)) == [Point(0, 1)]
    
    # 断言椭圆e1与圆Circle(Point(5, 0), 1)没有交点
    assert e1.intersection(Circle(Point(5, 0), 1)) == []
    
    # 断言椭圆e1与椭圆Ellipse(Point(2, 0), 1, 1)的交点为[Point(1, 0)]
    assert e1.intersection(Ellipse(Point(2, 0), 1, 1)) == [Point(1, 0)]
    
    # 断言椭圆e1与椭圆Ellipse(Point(5, 0), 1, 1)没有交点
    assert e1.intersection(Ellipse(Point(5, 0), 1, 1)) == []
    
    # 断言椭圆e1与点(2, 0)没有交点
    assert e1.intersection(Point(2, 0)) == []
    
    # 断言椭圆e1与自身的交点为e1
    assert e1.intersection(e1) == e1
    
    # 断言椭圆Ellipse(Point(0, 0), 2, 1)与椭圆Ellipse(Point(3, 0), 1, 2)的交点为[Point(2, 0)]
    assert intersection(Ellipse(Point(0, 0), 2, 1), Ellipse(Point(3, 0), 1, 2)) == [Point(2, 0)]
    
    # 断言圆Circle(Point(0, 0), 2)与圆Circle(Point(3, 0), 1)的交点为[Point(2, 0)]
    assert intersection(Circle(Point(0, 0), 2), Circle(Point(3, 0), 1)) == [Point(2, 0)]
    
    # 断言圆Circle(Point(0, 0), 2)与
    # 确保椭圆和椭圆没有交点
    assert intersection(Ellipse(Point(0, 0), 5, 17), Ellipse(Point(4, 0), 0.999, 0.2)) == []

    # 确保圆与三角形的交点为特定的两个点
    assert Circle((0, 0), S.Half).intersection(
        Triangle((-1, 0), (1, 0), (0, 1))) == [
        Point(Rational(-1, 2), 0), Point(S.Half, 0)]

    # 确保类型错误引发异常，期望椭圆和线段之间没有交点
    raises(TypeError, lambda: intersection(e2, Line((0, 0, 0), (0, 0, 1))))

    # 确保类型错误引发异常，期望椭圆和有理数之间没有交点
    raises(TypeError, lambda: intersection(e2, Rational(12)))

    # 确保类型错误引发异常，期望椭圆和整数之间没有交点
    raises(TypeError, lambda: Ellipse.intersection(e2, 1))

    # 创建一些特殊情况下的圆的实例
    csmall = Circle(p1, 3)
    cbig = Circle(p1, 5)
    cout = Circle(Point(5, 5), 1)

    # 确保一个圆完全包含在另一个圆内，没有交点
    assert csmall.intersection(cbig) == []

    # 确保两个分离的圆没有交点
    assert csmall.intersection(cout) == []

    # 确保两个重合的圆返回其自身
    assert csmall.intersection(csmall) == csmall

    # 创建一个特定三角形实例
    v = sqrt(2)
    t1 = Triangle(Point(0, v), Point(0, -v), Point(v, 0))

    # 确保三角形与圆的交点数为4个，并包含特定的四个点
    points = intersection(t1, c1)
    assert len(points) == 4
    assert Point(0, 1) in points
    assert Point(0, -1) in points
    assert Point(v/2, v/2) in points
    assert Point(v/2, -v/2) in points

    # 创建圆和椭圆的实例
    circ = Circle(Point(0, 0), 5)
    elip = Ellipse(Point(0, 0), 5, 20)

    # 确保圆和椭圆的交点为特定的两组点之一
    assert intersection(circ, elip) in \
        [[Point(5, 0), Point(-5, 0)], [Point(-5, 0), Point(5, 0)]]

    # 确保椭圆在原点处没有切线
    assert elip.tangent_lines(Point(0, 0)) == []

    # 创建另一个椭圆实例
    elip = Ellipse(Point(0, 0), 3, 2)

    # 确保椭圆在特定点处有一条切线
    assert elip.tangent_lines(Point(3, 0)) == \
        [Line(Point(3, 0), Point(3, -12))]

    # 创建两个椭圆实例和一些数学表达式
    e1 = Ellipse(Point(0, 0), 5, 10)
    e2 = Ellipse(Point(2, 1), 4, 8)
    a = Rational(53, 17)
    c = 2*sqrt(3991)/17

    # 确保两个椭圆的交点是预期的两个点
    ans = [Point(a - c/8, a/2 + c), Point(a + c/8, a/2 - c)]
    assert e1.intersection(e2) == ans

    # 创建另一个椭圆实例和一些数学表达式
    e2 = Ellipse(Point(x, y), 4, 8)
    c = sqrt(3991)

    # 确保两个椭圆在特定参数下的交点是预期的两个点
    ans = [Point(-c/68 + a, c*Rational(2, 17) + a/2), Point(c/68 + a, c*Rational(-2, 17) + a/2)]
    assert [p.subs({x: 2, y:1}) for p in e1.intersection(e2)] == ans

    # 组合上述情况
    assert e3.is_tangent(e3.tangent_lines(p1 + Point(y1, 0))[0])

    # 创建一个椭圆实例和一些数学表达式
    e = Ellipse((1, 2), 3, 2)

    # 确保椭圆在特定点处的切线是预期的两条直线
    assert e.tangent_lines(Point(10, 0)) == \
        [Line(Point(10, 0), Point(1, 0)),
        Line(Point(10, 0), Point(Rational(14, 5), Rational(18, 5)))]

    # 创建一个椭圆实例和一些数学表达式
    e = Ellipse((0, 0), 1, 2)

    # 确保椭圆能正确包含特定点
    assert e.encloses_point(e.center)
    assert e.encloses_point(e.center + Point(0, e.vradius - Rational(1, 10)))
    assert e.encloses_point(e.center + Point(e.hradius - Rational(1, 10), 0))
    assert e.encloses_point(e.center + Point(e.hradius, 0)) is False
    assert e.encloses_point(
        e.center + Point(e.hradius + Rational(1, 10), 0)) is False

    # 创建另一个椭圆实例和一些数学表达式
    e = Ellipse((0, 0), 2, 1)

    # 确保椭圆能正确包含特定点
    assert e.encloses_point(e.center)
    assert e.encloses_point(e.center + Point(0, e.vradius - Rational(1, 10)))
    assert e.encloses_point(e.center + Point(e.hradius - Rational(1, 10), 0))
    assert e.encloses_point(e.center + Point(e.hradius, 0)) is False
    assert e.encloses_point(
        e.center + Point(e.hradius + Rational(1, 10), 0)) is False

    # 确保圆不包含特定点
    assert c1.encloses_point(Point(1, 0)) is False
    # 断言：检查圆形 c1 是否包含点 (0.3, 0.4)，应该返回 True
    assert c1.encloses_point(Point(0.3, 0.4)) is True

    # 断言：检查椭圆 e 缩放后是否符合预期
    assert e.scale(2, 3) == Ellipse((0, 0), 4, 3)
    # 断言：再次检查椭圆 e 缩放后是否符合预期
    assert e.scale(3, 6) == Ellipse((0, 0), 6, 6)
    # 断言：检查椭圆 e 绕其原点旋转 π 后是否保持不变
    assert e.rotate(pi) == e
    # 断言：检查椭圆 e 绕点 (1, 2) 旋转 π 后是否符合预期
    assert e.rotate(pi, (1, 2)) == Ellipse(Point(2, 4), 2, 1)
    # 断言：检查尝试对椭圆 e 使用 π/3 的角度进行旋转是否会引发 NotImplementedError 异常
    raises(NotImplementedError, lambda: e.rotate(pi/3))

    # Circle rotation tests (Issue #11743)
    # Link - https://github.com/sympy/sympy/issues/11743
    # 创建一个圆 cir，中心为 (1, 0)，半径为 1，检查其绕 π/2 的角度旋转后是否符合预期
    assert cir.rotate(pi/2) == Circle(Point(0, 1), 1)
    # 检查圆 cir 绕 π/3 的角度旋转后是否符合预期
    assert cir.rotate(pi/3) == Circle(Point(S.Half, sqrt(3)/2), 1)
    # 检查圆 cir 绕 π/3 的角度以点 (1, 0) 为中心旋转后是否保持不变
    assert cir.rotate(pi/3, Point(1, 0)) == Circle(Point(1, 0), 1)
    # 检查圆 cir 绕 π/3 的角度以点 (0, 1) 为中心旋转后是否符合预期
    assert cir.rotate(pi/3, Point(0, 1)) == Circle(Point(S.Half + sqrt(3)/2, S.Half + sqrt(3)/2), 1)
# 定义一个测试函数，用于测试椭圆和圆形对象的构建和属性
def test_construction():
    # 创建一个椭圆对象 e1，水平半径为 2，垂直半径为 1，离心率为 None
    e1 = Ellipse(hradius=2, vradius=1, eccentricity=None)
    # 断言 e1 的离心率为 sqrt(3)/2
    assert e1.eccentricity == sqrt(3)/2

    # 创建一个椭圆对象 e2，水平半径为 2，垂直半径为 None，离心率为 sqrt(3)/2
    e2 = Ellipse(hradius=2, vradius=None, eccentricity=sqrt(3)/2)
    # 断言 e2 的垂直半径为 1
    assert e2.vradius == 1

    # 创建一个椭圆对象 e3，水平半径为 None，垂直半径为 1，离心率为 sqrt(3)/2
    e3 = Ellipse(hradius=None, vradius=1, eccentricity=sqrt(3)/2)
    # 断言 e3 的水平半径为 2
    assert e3.hradius == 2

    # 创建一个椭圆对象 e4，中心点为 (0, 0)，水平半径为 1，离心率为 0
    # 由于离心率为 0，v半径为 1，构造函数会抛出错误
    e4 = Ellipse(Point(0, 0), hradius=1, eccentricity=0)
    # 断言 e4 的垂直半径为 1
    assert e4.vradius == 1

    # 测试离心率大于 1 的情况，应该引发 GeometryError
    raises(GeometryError, lambda: Ellipse(Point(3, 1), hradius=3, eccentricity = S(3)/2))
    raises(GeometryError, lambda: Ellipse(Point(3, 1), hradius=3, eccentricity=sec(5)))
    raises(GeometryError, lambda: Ellipse(Point(3, 1), hradius=3, eccentricity=S.Pi-S(2)))

    # 测试离心率等于 1 的情况
    # 如果垂直半径未定义，预期椭圆周长为 2
    assert Ellipse(None, 1, None, 1).length == 2
    # 如果水平半径未定义，构造函数应该抛出 GeometryError
    raises(GeometryError, lambda: Ellipse(None, None, 1, eccentricity = 1))

    # 测试离心率小于 0 的情况，应该引发 GeometryError
    raises(GeometryError, lambda: Ellipse(Point(3, 1), hradius=3, eccentricity = -3))
    raises(GeometryError, lambda: Ellipse(Point(3, 1), hradius=3, eccentricity = -0.5))


def test_ellipse_random_point():
    # 定义一个实数符号 y1
    y1 = Symbol('y1', real=True)
    # 创建一个以 (0, 0) 为中心，水平和垂直半径均为 y1 的椭圆 e3
    e3 = Ellipse(Point(0, 0), y1, y1)
    # 定义实数符号 rx, ry
    rx, ry = Symbol('rx'), Symbol('ry')
    # 循环测试随机点的生成，替换后应为 0*y1**2
    for ind in range(0, 5):
        r = e3.random_point()
        # 断言椭圆方程 e3.equation(rx, ry) 在随机点 r 上的替换结果为 0
        assert e3.equation(rx, ry).subs(zip((rx, ry), r.args)).equals(0)
    # 测试具有种子的情况下的随机点生成
    r = e3.random_point(seed=1)
    # 断言椭圆方程 e3.equation(rx, ry) 在随机点 r 上的替换结果为 0
    assert e3.equation(rx, ry).subs(zip((rx, ry), r.args)).equals(0)


def test_repr():
    # 断言圆形对象的字符串表示应该是 'Circle(Point2D(0, 1), 2)'
    assert repr(Circle((0, 1), 2)) == 'Circle(Point2D(0, 1), 2)'


def test_transform():
    # 创建一个圆形 c，中心点为 (1, 1)，半径为 2
    c = Circle((1, 1), 2)
    # 断言 c 按照比例缩放 -1 后的结果为 Circle((-1, 1), 2)
    assert c.scale(-1) == Circle((-1, 1), 2)
    # 断言 c 按照 y 轴比例缩放 -1 后的结果为 Circle((1, -1), 2)
    assert c.scale(y=-1) == Circle((1, -1), 2)
    # 断言 c 按照 2 的比例缩放后的结果为 Ellipse((2, 1), 4, 2)
    assert c.scale(2) == Ellipse((2, 1), 4, 2)

    # 断言 Ellipse((0, 0), 2, 3) 按照给定参数缩放后的结果为 Ellipse(Point(-4, -10), 4, 9)
    assert Ellipse((0, 0), 2, 3).scale(2, 3, (4, 5)) == Ellipse(Point(-4, -10), 4, 9)
    # 断言 Circle((0, 0), 2) 按照给定参数缩放后的结果为 Ellipse(Point(-4, -10), 4, 6)
    assert Circle((0, 0), 2).scale(2, 3, (4, 5)) == Ellipse(Point(-4, -10), 4, 6)
    # 断言 Ellipse((0, 0), 2, 3) 按照给定参数缩放后的结果为 Ellipse(Point(-8, -10), 6, 9)
    assert Ellipse((0, 0), 2, 3).scale(3, 3, (4, 5)) == Ellipse(Point(-8, -10), 6, 9)
    # 断言 Circle((0, 0), 2) 按照给定参数缩放后的结果为 Circle(Point(-8, -10), 6)
    assert Circle((0, 0), 2).scale(3, 3, (4, 5)) == Circle(Point(-8, -10), 6)
    # 断言 Circle(Point(-8, -10), 6) 按照给定参数缩放后的结果为 Circle((0, 0), 2)
    assert Circle(Point(-8, -10), 6).scale(Rational(1, 3), Rational(1, 3), (4, 5)) == Circle((0, 0), 2)
    # 断言 Circle((0, 0), 2) 按照给定参数平移后的结果为 Circle((4, 5), 2)
    assert Circle((0, 0), 2).translate(4, 5) == Circle((4, 5), 2)
    # 断言 Circle((0, 0), 2) 按照给定参数缩放后的结果为 Circle((0, 0), 6)
    assert Circle((0, 0), 2).scale(3, 3) == Circle((0, 0), 6)


def test_bounds():
    # 创建一个椭圆 e1，中心点为 (0, 0)，水平半径为 3，垂直半径为 5
    e1 = Ellipse(Point(0, 0), 3, 5)
    # 创建一个椭圆 e2，中心点为 (2, -2)，水平半径和垂直半径均为 7
    e2 = Ellipse(Point(2, -2), 7, 7)
    # 创建一个圆形 c
#`
def test_reflect():
    # 创建一个符号 b
    b = Symbol('b')
    # 创建一个符号 m
    m = Symbol('m')
    # 创建一条经过点 (0, b) 且斜率为 m 的直线
    l = Line((0, b), slope=m)
    # 创建一个三角形，顶点为 (0, 0), (1, 0), (2, 3)
    t1 = Triangle((0, 0), (1, 0), (2, 3))
    # 断言三角形的面积等于其关于直线 l 对称后的面积的相反数
    assert t1.area == -t1.reflect(l).area
    # 创建一个椭圆，中心为 (1, 0)，长半轴为 1，短半轴为 2
    e = Ellipse((1, 0), 1, 2)
    # 断言椭圆的面积等于其关于斜率为 0 的直线对称后的面积的相反数
    assert e.area == -e.reflect(Line((1, 0), slope=0)).area
    # 断言椭圆的面积等于其关于斜率为无穷大的直线对称后的面积的相反数
    assert e.area == -e.reflect(Line((1, 0), slope=oo)).area
    # 断言在尝试对斜率为 m 的直线进行对称反射时，抛出 NotImplementedError 异常
    raises(NotImplementedError, lambda: e.reflect(Line((1, 0), slope=m)))
    # 断言圆心为 (0, 1)，半径为 1 的圆，关于直线通过 (0, 0) 和 (1, 1) 的对称结果等于新圆
    assert Circle((0, 1), 1).reflect(Line((0, 0), (1, 1))) == Circle(Point2D(1, 0), -1)


def test_is_tangent():
    # 创建一个椭圆，中心为 (0, 0)，长半轴为 3，短半轴为 5
    e1 = Ellipse(Point(0, 0), 3, 5)
    # 创建一个圆，中心为 (2, -2)，半径为 7
    c1 = Circle(Point(2, -2), 7)
    # 断言椭圆在点 (0, 0) 处不是切线
    assert e1.is_tangent(Point(0, 0)) is False
    # 断言椭圆在点 (3, 0) 处不是切线
    assert e1.is_tangent(Point(3, 0)) is False
    # 断言椭圆与自身是切线
    assert e1.is_tangent(e1) is True
    # 断言椭圆与一个长半轴为 1，短半轴为 2 的椭圆不是切线
    assert e1.is_tangent(Ellipse((0, 0), 1, 2)) is False
    # 断言椭圆与长半轴为 3，短半轴为 2 的椭圆是切线
    assert e1.is_tangent(Ellipse((0, 0), 3, 2)) is True
    # 断言圆与一个长半轴为 7，短半轴为 1 的椭圆是切线
    assert c1.is_tangent(Ellipse((2, -2), 7, 1)) is True
    # 断言圆与中心为 (11, -2)，半径为 2 的圆是切线
    assert c1.is_tangent(Circle((11, -2), 2)) is True
    # 断言圆与中心为 (7, -2)，半径为 2 的圆是切线
    assert c1.is_tangent(Circle((7, -2), 2)) is True
    # 断言圆与一条起点 (-5, -2)，终点 (-15, -20) 的射线不是切线
    assert c1.is_tangent(Ray((-5, -2), (-15, -20))) is False
    # 断言圆与一条起点 (-3, -2)，终点 (-15, -20) 的射线不是切线
    assert c1.is_tangent(Ray((-3, -2), (-15, -20))) is False
    # 断言圆与一条起点 (-3, -22)，终点 (15, 20) 的射线不是切线
    assert c1.is_tangent(Ray((-3, -22), (15, 20))) is False
    # 断言圆与一条起点 (9, 20)，终点 (9, -20) 的射线是切线
    assert c1.is_tangent(Ray((9, 20), (9, -20))) is True
    # 断言圆与一条起点 (2, 5)，终点 (9, 5) 的射线是切线
    assert c1.is_tangent(Ray((2, 5), (9, 5))) is True
    # 断言圆与一条起点 (2, 5)，终点 (9, 5) 的线段是切线
    assert c1.is_tangent(Segment((2, 5), (9, 5))) is True
    # 断言椭圆与一条起点 (2, 2)，终点 (-7, 7) 的线段不是切线
    assert e1.is_tangent(Segment((2, 2), (-7, 7))) is False
    # 断言椭圆与一条起点 (0, 0)，终点 (1, 2) 的线段不是切线
    assert e1.is_tangent(Segment((0, 0), (1, 2))) is False
    # 断言圆与一条起点 (0, 0)，终点 (-5, -2) 的线段不是切线
    assert c1.is_tangent(Segment((0, 0), (-5, -2))) is False
    # 断言椭圆与一条起点 (3, 0)，终点 (12, 12) 的线段不是切线
    assert e1.is_tangent(Segment((3, 0), (12, 12))) is False
    # 断言椭圆与一条起点 (12, 12)，终点 (3, 0) 的线段不是切线
    assert e1.is_tangent(Segment((12, 12), (3, 0))) is False
    # 断言椭圆与一条起点 (-3, 0)，终点 (3, 0) 的线段不是切线
    assert e1.is_tangent(Segment((-3, 0), (3, 0))) is False
    # 断言椭圆与一条起点 (-3, 5)，终点 (3, 5) 的线段是切线
    assert e1.is_tangent(Segment((-3, 5), (3, 5))) is True
    # 断言椭圆与一条起点 (10, 0)，终点 (10, 10) 的直线不是切线
    assert e1.is_tangent(Line((10, 0), (10, 10))) is False
    # 断言椭圆与一条通过点 (0, 0) 和 (1, 1) 的直线不是切线
    assert e1.is_tangent(Line((0, 0), (1, 1))) is False
    # 断言椭圆与一条通过点 (-3, 0) 和 (-2.99, -0.001) 的直线不是切线
    assert e1.is_tangent(Line((-3, 0), (-2.99, -0.001))) is False
    # 断言椭圆与一条垂直于 x 轴，经过点 (-3, 0) 的直线是切线
    assert e1.is_tangent(Line((-3, 0), (-3, 1))) is True
    # 断言椭圆与一个多边形的切线判定测试
    assert e1.is_tangent(Polygon((0, 0), (5, 5), (5, -5))) is False
    assert e1.is_tangent(Polygon((-100, -50), (-40, -334), (-70, -52))) is False
    assert e1.is_tangent(Polygon((-3, 0), (3, 0), (0, 1))) is False
    assert e1.is_tangent(Polygon((-3, 0), (3, 0), (0, 5))) is False
    assert e1.is_tangent(Polygon((-3, 0), (0, -```
# 定义一个测试函数，用于测试几何形状的反射和切线
def test_reflect():
    # 创建一个符号 b
    b = Symbol('b')
    # 创建一个符号 m
    m = Symbol('m')
    # 创建一条直线 l，通过指定斜率 m 和点 (0, b)
    l = Line((0, b), slope=m)
    # 创建一个三角形 t1，顶点分别为 (0, 0), (1, 0), (2, 3)
    t1 = Triangle((0, 0), (1, 0), (2, 3))
    # 断言 t1 的面积等于其关于直线 l 的反射的面积的相反数
    assert t1.area == -t1.reflect(l).area
    # 创建一个椭圆 e，中心点为 (1, 0)，长轴为 1，短轴为 2
    e = Ellipse((1, 0), 1, 2)
    # 断言 e 的面积等于其关于通过点 (1, 0) 斜率为 0 的直线的反射的面积的相反数
    assert e.area == -e.reflect(Line((1, 0), slope=0)).area
    # 断言 e 的面积等于其关于通过点 (1, 0) 斜率为无穷大的直线的反射的面积的相反数
    assert e.area == -e.reflect(Line((1, 0), slope=oo)).area
    # 断言对于尚未实现的操作，反射椭圆 e 关于直线 (1, 0) 斜率为 m 会抛出 NotImplementedError
    raises(NotImplementedError, lambda: e.reflect(Line((1, 0), slope=m)))
    # 断言圆的反射结果等于以 Point2D(1, 0) 为圆心、半径为 -1 的圆
    assert Circle((0, 1), 1).reflect(Line((0, 0), (1, 1))) == Circle(Point2D(1, 0), -1)


# 定义一个测试函数，用于测试几何形状的切线关系
def test_is_tangent():
    # 创建一个椭圆 e1，中心点为 (0, 0)，长轴为 3，短轴为 5
    e1 = Ellipse(Point(0, 0), 3, 5)
    # 创建一个圆 c1，中心点为 (2, -2)，半径为 7
    c1 = Circle(Point(2, -2), 7)
    # 断言 e1 不与点 (0, 0) 相切
    assert e1.is_tangent(Point(0, 0)) is False
    # 断言 e1 不与点 (3, 0) 相切
    assert e1.is_tangent(Point(3, 0)) is False
    # 断言 e1 与自身相切
    assert e1.is_tangent(e1) is True
    # 断言 e1 不与另一个椭圆相切，该椭圆中心为 (0, 0)，长轴为 1，短轴为 2
    assert e1.is_tangent(Ellipse((0, 0), 1, 2)) is False
    # 断言 e1 与另一个椭圆相切，该椭圆中心为 (0, 0)，长轴为 3，短轴为 2
    assert e1.is_tangent(Ellipse((0, 0), 3, 2)) is True
    # 断言 c1 与另一个椭圆相切，该椭圆中心为 (2, -2)，长轴为 7，短轴为 1
    assert c1.is_tangent(Ellipse((2, -2), 7, 1)) is True
    # 断言 c1 与另一个圆相切，该圆中心为 (11, -2)，半径为 2
    assert c1.is_tangent(Circle((11, -2), 2)) is True
    # 断言 c1 与另一个圆相切，该圆```
# 定义一个测试函数，用于测试几何形状的反射和切线
def test_reflect():
    # 创建一个符号 b
    b = Symbol('b')
    # 创建一个符号 m
    m = Symbol('m')
    # 创建一条直线 l，通过指定斜率 m 和点 (0, b)
    l = Line((0, b), slope=m)
    # 创建一个三角形 t1，顶点分别为 (0, 0), (1, 0), (2, 3)
    t1 = Triangle((0, 0), (1, 0), (2, 3))
    # 断言 t1 的面积等于其关于直线 l 的反射的面积的相反数
    assert t1.area == -t1.reflect(l).area
    # 创建一个椭圆 e，中心点为 (1, 0)，长轴为 1，短轴为 2
    e = Ellipse((1, 0), 1, 2)
    # 断言 e 的面积等于其关于通过点 (1, 0) 斜率为 0 的直线的反射的面积的相反数
    assert e.area == -e.reflect(Line((1, 0), slope=0)).area
    # 断言 e 的面积等于其关于通过点 (1, 0) 斜率为无穷大的直线的反射的面积的相反数
    assert e.area == -e.reflect(Line((1, 0), slope=oo)).area
    # 断言对于尚未实现的操作，反射椭圆 e 关于直线 (1, 0) 斜率为 m 会抛出 NotImplementedError
    raises(NotImplementedError, lambda: e.reflect(Line((1, 0), slope=m)))
    # 断言圆的反射结果等于以 Point2D(1, 0) 为圆心、半径为 -1 的圆
    assert Circle((0, 1), 1).reflect(Line((0, 0), (1, 1))) == Circle(Point2D(1, 0), -1)


# 定义一个测试函数，用于测试几何形状的切线关系
def test_is_tangent():
    # 创建一个椭圆 e1，中心点为 (0, 0)，长轴为 3，短轴为 5
    e1 = Ellipse(Point(0, 0), 3, 5)
    # 创建一个圆 c1，中心点为 (2, -2)，半径为 7
    c1 = Circle(Point(2, -2), 7)
    # 断言 e1 不与点 (0, 0) 相切
    assert e1.is_tangent(Point(0, 0)) is False
    # 断言 e1 不与点 (3, 0) 相切
    assert e1.is_tangent(Point(3, 0)) is False
    # 断言 e1 与自身相切
    assert e1.is_tangent(e1) is True
    # 断言 e1 不与另一个椭圆相切，该椭圆中心为 (0, 0)，长轴为 1，短轴为 2
    assert e1.is_tangent(Ellipse((0, 0), 1, 2)) is False
    # 断言 e1 与另一个椭圆相切，该椭圆中心为 (0, 0)，长轴为 3，短轴为 2
    assert e1.is_tangent(Ellipse((0, 0), 3, 2)) is True
    # 断言 c1 与另一个椭圆相切，该椭圆中心为 (2, -2)，长轴为 7，短轴为 1
    assert c1.is_tangent(Ellipse((2, -2), 7, 1)) is True
    # 断言 c1 与另一个圆相切，该圆中心为 (11, -2)，半径为 2
    assert c1.is_tangent(Circle((11, -2), 2)) is True
    # 断言 c1 与另一个圆相切，该圆中心为 (7, -2)，半径为 2
    assert c1.is_tangent(Circle((7, -2), 2)) is True
    # 断言 c1 不与射线相切，该射线起点 (-5, -2)，终点 (-15, -20)
    assert c1.is_tangent(Ray((-5, -2), (-15, -20))) is False
    # 断言 c1 不与射线相切，该射线起点 (-3, -2)，终点 (-15, -20)
    assert c1.is_tangent(Ray((-3, -2), (-15, -20))) is False
    # 断言 c1 不与射线相切，该射线起点 (-3, -22)，终点 (15, 20)
    assert c1.is_tangent(Ray((-3, -22), (15, 20))) is False
    # 断言 c1 与射线相切，该射线起点 (9, 20)，终点 (9, -20)
    assert c1.is_tangent(Ray((9, 20), (9, -20))) is True
    # 断言 c1 与射线相切，该射线起点 (2, 5)，终点 (9, 5)
    assert c1.is_tangent(Ray((2, 5), (9, 5))) is True
    # 断言 c1 与线段相切，该线段起点 (2, 5)，终点 (9, 5)
    assert c1.is_tangent(Segment((2, 5), (9, 5))) is True
    # 断言 e1 不与线段相切，该线段起点 (2, 2)，终点 (-7, 7)
    assert e1.is_tangent(Segment((2, 2), (-7, 7))) is False
    # 断言 e1 不与线段相切，该线段起点 (0, 0)，终点 (1, 2)
    assert e1.is_tangent(Segment((0, 0), (1, 2))) is False
    # 断言 c1 不与线段相切，该线段起点 (0, 0)，终点 (-5, -2)
    assert c1.is_tangent(Segment((0, 0), (-5, -2))) is False
    # 断言 e1 不与线段相切，该线段起点 (3, 0)，终点 (12, 12)
    assert e1.is_tangent(Segment((3, 0), (12, 12))) is False
    # 断言 e1 不与线段相切，该线段起点 (12, 12)，终点 (3, 0)
    assert e1.is_tangent(Segment((12, 12), (3, 0))) is False
    # 断言 e1 不与线段相切，该线段起点 (-3, 0)，终点 (3, 0)
    assert e1.is_tangent(Segment((-3, 0), (3, 0))) is False
    # 断言 e1 与线段相切，该线段起点 (-3, 5)，终点 (3, 5)
    assert e1.is_tangent(Segment((-3, 5), (3, 5))) is True
    # 断言 e1 不与直线相切，该直线通过点 (10, 0)，方向 (10, 10)
    assert e1.is_tangent(Line((10, 0), (10, 10))) is False
    # 断言 e1 不与直线相切，该直线通过点 (0, 0)，方向 (1, 1)
    assert e1.is_tangent(Line((0, 0), (1, 1))) is False
    # 断言 e1 不与直线相切，该直线通过点 (-3, 0)，方向 (-2.99, -0.001)
    assert e
    # 断言：检查边缘 e1 是否与多边形 (0, 0), (5, 5), (5, -5) 不相切
    assert e1.is_tangent(Polygon((0, 0), (5, 5), (5, -5))) is False
    # 断言：检查边缘 e1 是否与多边形 (-100, -50), (-40, -334), (-70, -52) 不相切
    assert e1.is_tangent(Polygon((-100, -50), (-40, -334), (-70, -52))) is False
    # 断言：检查边缘 e1 是否与多边形 (-3, -5), (-3, 5), (3, 5), (3, -5) 相切
    assert e1.is_tangent(Polygon((-3, -5), (-3, 5), (3, 5), (3, -5))) is True
    # 断言：检查圆 c1 是否与多边形 (-3, -5), (-3, 5), (3, 5), (3, -5) 不相切
    assert c1.is_tangent(Polygon((-3, -5), (-3, 5), (3, 5), (3, -5))) is False
    # 断言：检查边缘 e1 是否与多边形 (3, 12), (3, -12), (0, -5), (0, 5) 不相切
    assert e1.is_tangent(Polygon((3, 12), (3, -12), (0, -5), (0, 5))) is False
    # 断言：检查边缘 e1 是否与多边形 (3, 0), (5, 7), (6, -5) 不相切
    assert e1.is_tangent(Polygon((3, 0), (5, 7), (6, -5))) is False
    # 检查是否引发 TypeError 异常，lambda 函数传入 Point(0, 0, 0) 作为参数调用 e1.is_tangent
    raises(TypeError, lambda: e1.is_tangent(Point(0, 0, 0)))
    # 检查是否引发 TypeError 异常，lambda 函数传入 Rational(5) 作为参数调用 e1.is_tangent
    raises(TypeError, lambda: e1.is_tangent(Rational(5)))
def test_parameter_value():
    # 创建符号变量 t
    t = Symbol('t')
    # 创建椭圆对象 e，中心在 (0, 0)，水平半径为 3，垂直半径为 5
    e = Ellipse(Point(0, 0), 3, 5)
    # 断言椭圆 e 在点 (3, 0) 处的参数值为 {t: 0}
    assert e.parameter_value((3, 0), t) == {t: 0}
    # 断言在点 (4, 0) 处调用 parameter_value 方法会引发 ValueError 异常
    raises(ValueError, lambda: e.parameter_value((4, 0), t))


@slow
def test_second_moment_of_area():
    # 创建符号变量 x 和 y
    x, y = symbols('x, y')
    # 创建椭圆对象 e，中心在 (0, 0)，水平半径为 5，垂直半径为 4
    e = Ellipse(Point(0, 0), 5, 4)
    # 计算 I_yy 的表达式
    I_yy = 2 * 4 * integrate(sqrt(25 - x**2) * x**2, (x, -5, 5)) / 5
    # 计算 I_xx 的表达式
    I_xx = 2 * 5 * integrate(sqrt(16 - y**2) * y**2, (y, -4, 4)) / 4
    # 计算 Y 的表达式
    Y = 3 * sqrt(1 - x**2 / 5**2)
    # 计算 I_xy 的表达式
    I_xy = integrate(integrate(y, (y, -Y, Y)) * x, (x, -5, 5))
    # 断言计算得到的 I_yy 等于椭圆 e 的二阶矩 I_yy
    assert I_yy == e.second_moment_of_area()[1]
    # 断言计算得到的 I_xx 等于椭圆 e 的二阶矩 I_xx
    assert I_xx == e.second_moment_of_area()[0]
    # 断言计算得到的 I_xy 等于椭圆 e 的二阶矩 I_xy
    assert I_xy == e.second_moment_of_area()[2]
    # 检查在其他点 (6, 5) 处调用 second_moment_of_area 方法的结果
    t1 = e.second_moment_of_area(Point(6, 5))
    t2 = (580*pi, 845*pi, 600*pi)
    assert t1 == t2


def test_section_modulus_and_polar_second_moment_of_area():
    # 创建符号变量 d
    d = Symbol('d', positive=True)
    # 创建圆对象 c，中心在 (3, 7)，半径为 8
    c = Circle((3, 7), 8)
    # 断言圆 c 的极化面积矩为 2048*pi
    assert c.polar_second_moment_of_area() == 2048*pi
    # 断言圆 c 的截面模量为 (128*pi, 128*pi)
    assert c.section_modulus() == (128*pi, 128*pi)
    # 创建圆对象 c，中心在 (2, 9)，半径为 d/2
    c = Circle((2, 9), d/2)
    # 断言圆 c 的极化面积矩为 pi*d**3*Abs(d)/64 + pi*d*Abs(d)**3/64
    assert c.polar_second_moment_of_area() == pi*d**3*Abs(d)/64 + pi*d*Abs(d)**3/64
    # 断言圆 c 的截面模量为 (pi*d**3/S(32), pi*d**3/S(32))
    assert c.section_modulus() == (pi*d**3/S(32), pi*d**3/S(32))

    # 创建符号变量 a 和 b
    a, b = symbols('a, b', positive=True)
    # 创建椭圆对象 e，中心在 (4, 6)，水平半径为 a，垂直半径为 b
    e = Ellipse((4, 6), a, b)
    # 断言椭圆 e 的截面模量为 (pi*a*b**2/S(4), pi*a**2*b/S(4))
    assert e.section_modulus() == (pi*a*b**2/S(4), pi*a**2*b/S(4))
    # 断言椭圆 e 的极化面积矩为 pi*a**3*b/S(4) + pi*a*b**3/S(4)
    assert e.polar_second_moment_of_area() == pi*a**3*b/S(4) + pi*a*b**3/S(4)
    # 将椭圆 e 绕中心旋转 π/2 弧度
    e = e.rotate(pi/2) # 极化面积矩和截面模量不变
    # 断言旋转后椭圆 e 的截面模量为 (pi*a**2*b/S(4), pi*a*b**2/S(4))
    assert e.section_modulus() == (pi*a**2*b/S(4), pi*a*b**2/S(4))
    # 断言旋转后椭圆 e 的极化面积矩为 pi*a**3*b/S(4) + pi*a*b**3/S(4)
    assert e.polar_second_moment_of_area() == pi*a**3*b/S(4) + pi*a*b**3/S(4)

    # 创建椭圆对象 e，中心在 (a, b)，水平半径为 2，垂直半径为 6
    e = Ellipse((a, b), 2, 6)
    # 断言椭圆 e 的截面模量为 (18*pi, 6*pi)
    assert e.section_modulus() == (18*pi, 6*pi)
    # 断言椭圆 e 的极化面积矩为 120*pi

    # 创建椭圆对象 e，中心在 (0, 0)，水平半径为 2，垂直半径为 2
    e = Ellipse(Point(0, 0), 2, 2)
    # 断言椭圆 e 的截面模量为 (2*pi, 2*pi)
    assert e.section_modulus() == (2*pi, 2*pi)
    # 断言在点 (2, 2) 处调用 section_modulus 方法的结果为 (2*pi, 2*pi)


def test_circumference():
    # 创建符号变量 M 和 m
    M = Symbol('M')
    m = Symbol('m')
    # 断言椭圆对象的周长为 4 * M * elliptic_e((M ** 2 - m ** 2) / M**2)
    assert Ellipse(Point(0, 0), M, m).circumference == 4 * M * elliptic_e((M ** 2 - m ** 2) / M**2)

    # 断言椭圆对象的周长为 20 * elliptic_e(S(9) / 25)
    assert Ellipse(Point(0, 0), 5, 4).circumference == 20 * elliptic_e(S(9) / 25)

    # 断言圆对象的周长为 2*pi
    assert Ellipse(None, 1, None, 0).circumference == 2*pi

    # 数值测试，断言椭圆对象的周长数值近似等于 25.52699886339813
    assert abs(Ellipse(None, hradius=5, vradius=3).circumference.evalf(16) - 25.52699886339813) < 1e-10


def test_issue_15259():
    # 断言圆对象与点 (1, 2) 相等
    assert Circle((1, 2), 0) == Point(1, 2)


def test_issue_15797_equals():
    # 定义数值常量 Ri, Ci, A
    Ri = 0.024127189424130748
    Ci = (0.0864931002830291, 0.0819863295239654)
    A = Point(0, 0.0578591400998346)
    # 导入符号变量 x, y, a, b，并创建椭圆对象 e
    x, y, a, b = symbols('x y a b')
    e = Ellipse((x, y), a, b)
    
    # 测试一般情况下的辅助圆结果是否正确
    assert e.auxiliary_circle() == Circle((x, y), Max(a, b))
    
    # 测试特殊情况：当椭圆是一个圆的时候，辅助圆的计算是否正确
    assert Circle((3, 4), 8).auxiliary_circle() == Circle((3, 4), 8)
# 测试椭圆对象的导向圆方法
def test_director_circle():
    # 定义符号变量 x, y, a, b
    x, y, a, b = symbols('x y a b')
    # 创建一个椭圆对象 e，中心为 (x, y)，长轴为 a，短轴为 b
    e = Ellipse((x, y), a, b)
    # 断言：一般情况下，椭圆的导向圆应为以 (x, y) 为中心，半径为 sqrt(a**2 + b**2) 的圆
    assert e.director_circle() == Circle((x, y), sqrt(a**2 + b**2))
    # 断言：特殊情况下，当椭圆为圆时，导向圆应保持为半径为 8*sqrt(2)，中心为 (3, 4) 的圆
    assert Circle((3, 4), 8).director_circle() == Circle((3, 4), 8*sqrt(2))


# 测试椭圆对象的摄动曲线方法
def test_evolute():
    # 定义符号变量 x, y, h, k，并限定为实数
    x, y, h, k = symbols('x y h k', real=True)
    # 定义符号变量 a, b
    a, b = symbols('a b')
    # 创建一个以 (h, k) 为中心，长轴为 a，短轴为 b 的椭圆对象 e
    e = Ellipse(Point(h, k), a, b)
    # 计算椭圆的摄动曲线 E
    t1 = (e.hradius*(x - e.center.x))**Rational(2, 3)
    t2 = (e.vradius*(y - e.center.y))**Rational(2, 3)
    E = t1 + t2 - (e.hradius**2 - e.vradius**2)**Rational(2, 3)
    # 断言：椭圆的摄动曲线应与计算得到的 E 相等
    assert e.evolute() == E
    # 数值示例：创建一个以 (1, 1) 为中心，长轴为 6，短轴为 3 的椭圆对象 e
    e = Ellipse(Point(1, 1), 6, 3)
    # 计算椭圆的摄动曲线 E
    t1 = (6*(x - 1))**Rational(2, 3)
    t2 = (3*(y - 1))**Rational(2, 3)
    E = t1 + t2 - (27)**Rational(2, 3)
    # 断言：椭圆的摄动曲线应与计算得到的 E 相等
    assert e.evolute() == E


# 测试椭圆对象的 SVG 方法
def test_svg():
    # 创建一个以 (1, 0) 为中心，长轴为 3，短轴为 2 的椭圆对象 e1
    e1 = Ellipse(Point(1, 0), 3, 2)
    # 断言：调用椭圆对象 e1 的 _svg 方法，期望得到指定参数和颜色的 SVG 字符串
    assert e1._svg(2, "#FFAAFF") == '<ellipse fill="#FFAAFF" stroke="#555555" stroke-width="4.0" opacity="0.6" cx="1.00000000000000" cy="0" rx="3.00000000000000" ry="2.00000000000000"/>'
```