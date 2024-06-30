# `D:\src\scipysrc\sympy\sympy\geometry\tests\test_polygon.py`

```
# 导入 sympy 库中的特定模块和函数
from sympy.core.numbers import (Float, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, cos, sin)
from sympy.functions.elementary.trigonometric import tan
from sympy.geometry import (Circle, Ellipse, GeometryError, Point, Point2D,
                            Polygon, Ray, RegularPolygon, Segment, Triangle,
                            are_similar, convex_hull, intersection, Line, Ray2D)
from sympy.testing.pytest import raises, slow, warns  # 导入测试相关模块
from sympy.core.random import verify_numerically  # 导入随机数生成相关模块
from sympy.geometry.polygon import rad, deg  # 导入多边形相关模块
from sympy.integrals.integrals import integrate  # 导入积分相关模块
from sympy.utilities.iterables import rotate_left  # 导入可迭代对象相关模块

# 定义浮点数相等测试函数
def feq(a, b):
    """Test if two floating point values are 'equal'."""
    t_float = Float("1.0E-10")
    return -t_float < a - b < t_float

@slow
# 定义多边形测试函数
def test_polygon():
    # 定义实数符号变量
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    q = Symbol('q', real=True)
    u = Symbol('u', real=True)
    v = Symbol('v', real=True)
    w = Symbol('w', real=True)
    x1 = Symbol('x1', real=True)
    half = S.Half  # 定义 S.Half 为 half
    a, b, c = Point(0, 0), Point(2, 0), Point(3, 3)  # 定义三个点 a, b, c
    t = Triangle(a, b, c)  # 创建三角形 t

    # 断言多边形与点的比较结果
    assert Polygon(Point(0, 0)) == Point(0, 0)
    assert Polygon(a, Point(1, 0), b, c) == t
    assert Polygon(Point(1, 0), b, c, a) == t
    assert Polygon(b, c, a, Point(1, 0)) == t

    # 两个“remove folded”测试
    assert Polygon(a, Point(3, 0), b, c) == t
    assert Polygon(a, b, Point(3, -1), b, c) == t

    # 移除多个共线点
    assert Polygon(Point(-4, 15), Point(-11, 15), Point(-15, 15),
        Point(-15, Rational(33, 5)), Point(-15, Rational(-87, 10)), Point(-15, -15),
        Point(Rational(-42, 5), -15), Point(-2, -15), Point(7, -15), Point(15, -15),
        Point(15, -3), Point(15, 10), Point(15, 15)) == \
        Polygon(Point(-15, -15), Point(15, -15), Point(15, 15), Point(-15, 15))

    # 创建多边形对象 p1 到 p11
    p1 = Polygon(
        Point(0, 0), Point(3, -1),
        Point(6, 0), Point(4, 5),
        Point(2, 3), Point(0, 3))
    p2 = Polygon(
        Point(6, 0), Point(3, -1),
        Point(0, 0), Point(0, 3),
        Point(2, 3), Point(4, 5))
    p3 = Polygon(
        Point(0, 0), Point(3, 0),
        Point(5, 2), Point(4, 4))
    p4 = Polygon(
        Point(0, 0), Point(4, 4),
        Point(5, 2), Point(3, 0))
    p5 = Polygon(
        Point(0, 0), Point(4, 4),
        Point(0, 4))
    p6 = Polygon(
        Point(-11, 1), Point(-9, 6.6),
        Point(-4, -3), Point(-8.4, -8.7))
    p7 = Polygon(
        Point(x, y), Point(q, u),
        Point(v, w))
    p8 = Polygon(
        Point(x, y), Point(v, w),
        Point(q, u))
    p9 = Polygon(
        Point(0, 0), Point(4, 4),
        Point(3, 0), Point(5, 2))
    p10 = Polygon(
        Point(0, 2), Point(2, 2),
        Point(0, 0), Point(2, 0))
    p11 = Polygon(Point(0, 0), 1, n=3)  # 创建正 n 边形
    # 创建一个正三角形，中心在原点，边长为1，顶角为0，总共3个顶点
    p12 = Polygon(Point(0, 0), 1, 0, n=3)
    
    # 创建一个四边形，顶点按顺时针方向依次是(0,0)，(8,8)，(23,20)，(0,20)
    p13 = Polygon(
        Point(0, 0), Point(8, 8),
        Point(23, 20), Point(0, 20))
    
    # 对p13进行逆时针旋转1个位置，生成新的多边形对象p14
    p14 = Polygon(*rotate_left(p13.args, 1))
    
    # 创建一条射线，起点(-9, 6.6)，终点(-9, 5.5)
    r = Ray(Point(-9, 6.6), Point(-9, 5.5))
    
    #
    # 一般多边形
    #
    
    # 断言p1与p2相等
    assert p1 == p2
    
    # 断言p1的顶点数为6
    assert len(p1.args) == 6
    
    # 断言p1的边数为6
    assert len(p1.sides) == 6
    
    # 断言p1的周长为5 + 2*sqrt(10) + sqrt(29) + sqrt(8)
    assert p1.perimeter == 5 + 2*sqrt(10) + sqrt(29) + sqrt(8)
    
    # 断言p1的面积为22
    assert p1.area == 22
    
    # 断言p1不是凸多边形
    assert not p1.is_convex()
    
    # 断言给定点按顺时针或逆时针顺序生成的多边形不是凸多边形
    assert Polygon((-1, 1), (2, -1), (2, 1), (-1, -1), (3, 0)).is_convex() is False
    
    # 断言p3是凸多边形
    assert p3.is_convex()
    
    # 断言p4是凸多边形
    assert p4.is_convex()
    
    # 获取p5的各个顶点的角度字典
    dict5 = p5.angles
    
    # 断言字典中Point(0, 0)对应的角度为pi / 4
    assert dict5[Point(0, 0)] == pi / 4
    
    # 断言字典中Point(0, 4)对应的角度为pi / 2
    assert dict5[Point(0, 4)] == pi / 2
    
    # 断言p5不包含点(Point(x, y))
    assert p5.encloses_point(Point(x, y)) is None
    
    # 断言p5包含点(Point(1, 3))
    assert p5.encloses_point(Point(1, 3))
    
    # 断言p5不包含点(Point(0, 0))
    assert p5.encloses_point(Point(0, 0)) is False
    
    # 断言p5不包含点(Point(4, 0))
    assert p5.encloses_point(Point(4, 0)) is False
    
    # 断言p1不包含给定圆形
    assert p1.encloses(Circle(Point(2.5, 2.5), 5)) is False
    
    # 断言p1不包含给定椭圆
    assert p1.encloses(Ellipse(Point(2.5, 2), 5, 6)) is False
    
    # 断言p5关于'x'轴的绘图间隔为[x, 0, 1]
    assert p5.plot_interval('x') == [x, 0, 1]
    
    # 断言p5到指定多边形的距离为6 * sqrt(2)
    assert p5.distance(
        Polygon(Point(10, 10), Point(14, 14), Point(10, 14))) == 6 * sqrt(2)
    
    # 断言p5到指定多边形的距离为4
    assert p5.distance(
        Polygon(Point(1, 8), Point(5, 8), Point(8, 12), Point(1, 12))) == 4
    
    # 检查警告信息是否匹配，可能会产生交叉的多边形距离计算错误的输出
    with warns(UserWarning, match="Polygons may intersect producing erroneous output"):
        Polygon(Point(0, 0), Point(1, 0), Point(1, 1)).distance(
                Polygon(Point(0, 0), Point(0, 1), Point(1, 1)))
    
    # 断言p5的哈希值与另一个相同顶点的多边形对象的哈希值相等
    assert hash(p5) == hash(Polygon(Point(0, 0), Point(4, 4), Point(0, 4)))
    
    # 断言p1的哈希值与p2相等
    assert hash(p1) == hash(p2)
    
    # 断言p7的哈希值与p8相等
    assert hash(p7) == hash(p8)
    
    # 断言p3的哈希值与p9不相等
    assert hash(p3) != hash(p9)
    
    # 断言p5等于指定的多边形
    assert p5 == Polygon(Point(4, 4), Point(0, 4), Point(0, 0))
    
    # 断言指定的多边形对象在p5中
    assert Polygon(Point(4, 4), Point(0, 4), Point(0, 0)) in p5
    
    # 断言p5不等于Point(0, 4)
    assert p5 != Point(0, 4)
    
    # 断言Point(0, 1)在p5中
    assert Point(0, 1) in p5
    
    # 断言任意点't'的替代值为0时，等于Point(0, 0)
    assert p5.arbitrary_point('t').subs(Symbol('t', real=True), 0) == \
        Point(0, 0)
    
    # 断言创建包含参数值错误的正三角形时会引发ValueError异常
    raises(ValueError, lambda: Polygon(
        Point(x, 0), Point(0, y), Point(x, y)).arbitrary_point('x'))
    
    # 断言p6与射线r的交点为[Point(-9, -84/13), Point(-9, 33/5)]
    assert p6.intersection(r) == [Point(-9, Rational(-84, 13)), Point(-9, Rational(33, 5))]
    
    # 断言p10的面积为0
    assert p10.area == 0
    
    # 断言p11与指定参数创建的正三角形相等
    assert p11 == RegularPolygon(Point(0, 0), 1, 3, 0)
    
    # 断言p11与p12相等
    assert p11 == p12
    
    # 断言p11的第一个顶点为Point(1, 0)
    assert p11.vertices[0] == Point(1, 0)
    
    # 断言p11的第一个参数为Point(0, 0)
    assert p11.args[0] == Point(0, 0)
    
    # 以pi/2角度旋转p11
    p11.spin(pi/2)
    
    # 断言p11的第一个顶点为Point(0, 1)
    assert p11.vertices[0] == Point(0, 1)
    
    #
    # 正多边形
    #
    
    # 创建一个中心在原点，边长为10，顶点数为5的正多边形p1
    p1 = RegularPolygon(Point(0, 0), 10, 5)
    
    # 创建一个中心在原点，边长为5，顶点数为5的正多边形p2
    p2 = RegularPolygon(Point(0, 0), 5, 5)
    
    # 断言p1不等于p2
    assert p1 != p2
    
    # 断言p1的内角为pi*Rational(3,
    # 确认点 p2、p1 的外接圆心均为 (0, 0)
    assert p2.circumcenter == p1.circumcenter == Point(0, 0)
    # 确认点 p1 的外接圆半径和内接圆半径均为 10
    assert p1.circumradius == p1.radius == 10
    # 确认点 p2 的外接圆为以 (0, 0) 为中心，半径为 5 的圆
    assert p2.circumcircle == Circle(Point(0, 0), 5)
    # 确认点 p2 的内切圆为以 (0, 0) 为中心，半径为 p2.apothem 的圆
    assert p2.incircle == Circle(Point(0, 0), p2.apothem)
    # 确认点 p2 的内切圆半径等于 p2.apothem，计算公式为 5 * (1 + sqrt(5)) / 4
    assert p2.inradius == p2.apothem == (5 * (1 + sqrt(5)) / 4)
    # 绕原点旋转点 p2，旋转角度为 pi / 10
    p2.spin(pi / 10)
    # 获取点 p2 的角度字典
    dict1 = p2.angles
    # 确认点 (0, 5) 对应的角度为 3 * pi / 5
    assert dict1[Point(0, 5)] == 3 * pi / 5
    # 确认点 p1 是凸多边形
    assert p1.is_convex()
    # 确认点 p1 的旋转角度为 0
    assert p1.rotation == 0
    # 确认点 p1 包含点 (0, 0)
    assert p1.encloses_point(Point(0, 0))
    # 确认点 p1 不包含点 (11, 0)
    assert p1.encloses_point(Point(11, 0)) is False
    # 确认点 p2 包含点 (0, 4.9)
    assert p2.encloses_point(Point(0, 4.9))
    # 绕原点旋转点 p1，旋转角度为 pi / 3
    p1.spin(pi/3)
    # 确认点 p1 的旋转角度为 pi / 3
    assert p1.rotation == pi / 3
    # 确认点 p1 的第一个顶点为 (5, 5 * sqrt(3))
    assert p1.vertices[0] == Point(5, 5 * sqrt(3))
    # 遍历 p1 的参数列表，确认每个点均为 (0, 0)，其他参数为 5、10 或 pi / 3
    for var in p1.args:
        if isinstance(var, Point):
            assert var == Point(0, 0)
        else:
            assert var in (5, 10, pi / 3)
    # 确认点 p1 不等于点 (0, 0)
    assert p1 != Point(0, 0)
    # 确认点 p1 不等于点 p5
    assert p1 != p5

    # p1.rotate(pi/3) 返回一个新对象，而不是在原地旋转（下面的旋转角度为 2pi/3）
    p1_old = p1
    # 确认旋转点 p1 后等于一个新的 RegularPolygon 对象
    assert p1.rotate(pi/3) == RegularPolygon(Point(0, 0), 10, 5, pi*Rational(2, 3))
    # 确认旋转后的 p1 与旋转前的 p1 相等
    assert p1 == p1_old

    # 确认点 p1 的面积计算结果
    assert p1.area == (-250*sqrt(5) + 1250)/(4*tan(pi/5))
    # 确认点 p1 的周长计算结果
    assert p1.length == 20*sqrt(-sqrt(5)/8 + Rational(5, 8))
    # 确认点 p1 放大两倍后与原始点 p1 具有相同的属性
    assert p1.scale(2, 2) == \
        RegularPolygon(p1.center, p1.radius*2, p1._n, p1.rotation)
    # 确认 RegularPolygon((0, 0), 1, 4) 放大为 RegularPolygon((0, 0), 2, 4)
    assert RegularPolygon((0, 0), 1, 4).scale(2, 3) == \
        Polygon(Point(2, 0), Point(0, 3), Point(-2, 0), Point(0, -3))

    # 确认点 p1 的字符串表示与其正常字符串表示相同
    assert repr(p1) == str(p1)

    #
    # 角度
    #
    # 获取点 p4 的角度字典，确认各点的角度值近似相等
    angles = p4.angles
    assert feq(angles[Point(0, 0)].evalf(), Float("0.7853981633974483"))
    assert feq(angles[Point(4, 4)].evalf(), Float("1.2490457723982544"))
    assert feq(angles[Point(5, 2)].evalf(), Float("1.8925468811915388"))
    assert feq(angles[Point(3, 0)].evalf(), Float("2.3561944901923449"))

    # 获取点 p3 的角度字典，确认各点的角度值近似相等
    angles = p3.angles
    assert feq(angles[Point(0, 0)].evalf(), Float("0.7853981633974483"))
    assert feq(angles[Point(4, 4)].evalf(), Float("1.2490457723982544"))
    assert feq(angles[Point(5, 2)].evalf(), Float("1.8925468811915388"))
    assert feq(angles[Point(3, 0)].evalf(), Float("2.3561944901923449"))

    # 确认多边形 p13 和 p14 内角之和近似等于 (点数 - 2) * pi
    interior_angles_sum = sum(p13.angles.values())
    assert feq(interior_angles_sum, (len(p13.angles) - 2)*pi )
    interior_angles_sum = sum(p14.angles.values())
    assert feq(interior_angles_sum, (len(p14.angles) - 2)*pi )

    #
    # 三角形
    #
    # 创建三角形 t1、t2、t3，并验证其性质和方法的正确性
    p1 = Point(0, 0)
    p2 = Point(5, 0)
    p3 = Point(0, 5)
    t1 = Triangle(p1, p2, p3)
    t2 = Triangle(p1, p2, Point(Rational(5, 2), sqrt(Rational(75, 4))))
    t3 = Triangle(p1, Point(x1, 0), Point(0, x1))
    # 确认 t1 与 Polygon(p1, p2, p1) 以及 Segment(p1, p2) 等效
    s1 = t1.sides
    assert Triangle(p1, p2, p1) == Polygon(p1, p2, p1) == Segment(p1, p2)
    # 确认创建 Triangle(Point(0, 0)) 引发 GeometryError
    raises(GeometryError, lambda: Triangle(Point(0, 0)))

    # 基本验证
    # 确认三点相等的 Triangle(p1, p1, p1) 等于点 p1
    assert Triangle(p1, p1, p1) == p1
    # 确认 Triangle(p2, p2*2, p2*3) 等于 Segment(p2, p2*3)
    assert Triangle(p2, p2*2, p2*3) == Segment(p2, p2*3)
    # 确认 t1 的面积为 25/2
    assert t1.area == Rational(25, 2)
    # 确认 t1 为直角三角形
    assert t1.is_right()
    # 确认三角形 t2 不是直角三角形
    assert t2.is_right() is False
    # 确认三角形 t3 是直角三角形
    assert t3.is_right()
    # 确认点 p1 在三角形 t1 内部
    assert p1 in t1
    # 确认三角形 t1 的第一个边在三角形 t1 内部
    assert t1.sides[0] in t1
    # 确认线段 ((0, 0), (1, 0)) 在三角形 t1 内部
    assert Segment((0, 0), (1, 0)) in t1
    # 确认点 (5, 5) 不在三角形 t2 内部
    assert Point(5, 5) not in t2
    # 确认三角形 t1 是凸多边形
    assert t1.is_convex()
    # 确认 t1 中 p1 对应的角度近似等于 pi/2
    assert feq(t1.angles[p1].evalf(), pi.evalf()/2)

    # 确认三角形 t1 不是等边三角形
    assert t1.is_equilateral() is False
    # 确认三角形 t2 是等边三角形
    assert t2.is_equilateral()
    # 确认三角形 t3 不是等边三角形
    assert t3.is_equilateral() is False
    # 确认三角形 t1 和 t2 不相似
    assert are_similar(t1, t2) is False
    # 确认三角形 t1 和 t3 相似
    assert are_similar(t1, t3)
    # 确认三角形 t2 和 t3 不相似
    assert are_similar(t2, t3) is False
    # 确认点 (0, 0) 不和三角形 t1 相似
    assert t1.is_similar(Point(0, 0)) is False
    # 确认三角形 t1 和 t2 不相似
    assert t1.is_similar(t2) is False

    # 生成 t1 的角平分线
    bisectors = t1.bisectors()
    # 确认 t1 中 p1 对应的角平分线为指定的线段
    assert bisectors[p1] == Segment(
        p1, Point(Rational(5, 2), Rational(5, 2)))
    # 确认 t2 中 p2 对应的角平分线为指定的线段
    assert t2.bisectors()[p2] == Segment(
        Point(5, 0), Point(Rational(5, 4), 5*sqrt(3)/4))
    # 确认 t3 中 p4 对应的角平分线为指定的线段
    p4 = Point(0, x1)
    assert t3.bisectors()[p4] == Segment(p4, Point(x1*(sqrt(2) - 1), 0))
    # 计算三角形 t1 的内切圆心的坐标 ic 并确认
    ic = (250 - 125*sqrt(2))/50
    assert t1.incenter == Point(ic, ic)

    # 确认三角形 t1 的内接圆半径和外接圆半径是否正确
    assert t1.inradius == t1.incircle.radius == 5 - 5*sqrt(2)/2
    assert t2.inradius == t2.incircle.radius == 5*sqrt(3)/6
    assert t3.inradius == t3.incircle.radius == x1**2/((2 + sqrt(2))*Abs(x1))

    # 确认三角形 t1 的外接圆的某些特性
    assert t1.exradii[t1.sides[2]] == 5*sqrt(2)/2

    # 确认三角形 t1 的外心位置
    assert t1.excenters[t1.sides[2]] == Point2D(25*sqrt(2), -5*sqrt(2)/2)

    # 确认三角形 t1 的外接圆圆心位置
    assert t1.circumcircle.center == Point(2.5, 2.5)

    # 计算 t1 的中位线和重心
    m = t1.medians
    assert t1.centroid == Point(Rational(5, 3), Rational(5, 3))
    assert m[p1] == Segment(p1, Point(Rational(5, 2), Rational(5, 2)))
    assert t3.medians[p1] == Segment(p1, Point(x1/2, x1/2))
    # 确认中位线的交点为 t1 的重心
    assert intersection(m[p1], m[p2], m[p3]) == [t1.centroid]
    # 确认 t1 的中位三角形
    assert t1.medial == Triangle(Point(2.5, 0), Point(0, 2.5), Point(2.5, 2.5))

    # 确认 t1 的九点圆
    assert t1.nine_point_circle == Circle(Point(2.5, 0),
                                          Point(0, 2.5), Point(2.5, 2.5))
    assert t1.nine_point_circle == Circle(Point(0, 0),
                                          Point(0, 2.5), Point(2.5, 2.5))

    # 确认 t1 的高线和垂心
    altitudes = t1.altitudes
    assert altitudes[p1] == Segment(p1, Point(Rational(5, 2), Rational(5, 2)))
    assert altitudes[p2].equals(s1[0])
    assert altitudes[p3] == s1[2]
    # 确认 t1 的垂心位置
    assert t1.orthocenter == p1
    # 针对具有复杂表达式的三角形 t 进行垂心计算确认
    t = S('''Triangle(
    Point(100080156402737/5000000000000, 79782624633431/500000000000),
    Point(39223884078253/2000000000000, 156345163124289/1000000000000),
    Point(31241359188437/1250000000000, 338338270939941/1000000000000000))''')
    assert t.orthocenter == S('''Point(-780660869050599840216997'''
    '''79471538701955848721853/80368430960602242240789074233100000000000000,'''
    '''20151573611150265741278060334545897615974257/16073686192120448448157'''
    '''8148466200000000000)''')

    # 确认 t1 的角平分线、高线和中位线的交点数量为 1
    assert len(intersection(*bisectors.values())) == 1
    assert len(intersection(*altitudes.values())) == 1
    assert len(intersection(*m.values())) == 1
    # 创建四个多边形对象，分别表示不同的几何形状
    p1 = Polygon(
        Point(0, 0), Point(1, 0),
        Point(1, 1), Point(0, 1))
    p2 = Polygon(
        Point(0, Rational(5)/4), Point(1, Rational(5)/4),
        Point(1, Rational(9)/4), Point(0, Rational(9)/4))
    p3 = Polygon(
        Point(1, 2), Point(2, 2),
        Point(2, 1))
    p4 = Polygon(
        Point(1, 1), Point(Rational(6)/5, 1),
        Point(1, Rational(6)/5))
    
    # 创建两个点对象，表示不同的几何点
    pt1 = Point(half, half)
    pt2 = Point(1, 1)

    '''Polygon to Point'''
    # 断言：计算多边形 p1 到点 pt1 的距离为 half
    assert p1.distance(pt1) == half
    # 断言：计算多边形 p1 到点 pt2 的距离为 0
    assert p1.distance(pt2) == 0
    # 断言：计算多边形 p2 到点 pt1 的距离为 Rational(3)/4
    assert p2.distance(pt1) == Rational(3)/4
    # 断言：计算多边形 p3 到点 pt2 的距离为 sqrt(2)/2
    assert p3.distance(pt2) == sqrt(2)/2

    '''Polygon to Polygon'''
    # 使用警告上下文管理器，断言：计算多边形 p1 到 p2 的距离为 half/2，有可能会有交集导致错误的输出
    with warns(UserWarning, match="Polygons may intersect producing erroneous output"):
        assert p1.distance(p2) == half/2

    # 断言：计算多边形 p1 到 p3 的距离为 sqrt(2)/2
    assert p1.distance(p3) == sqrt(2)/2

    # 使用警告上下文管理器，断言：计算多边形 p3 到 p4 的距离为 (sqrt(2)/2 - sqrt(Rational(2)/25)/2)，有可能会有交集导致错误的输出
    with warns(UserWarning, match="Polygons may intersect producing erroneous output"):
        assert p3.distance(p4) == (sqrt(2)/2 - sqrt(Rational(2)/25)/2)
def test_convex_hull():
    # 创建一个点列表
    p = [Point(-5, -1), Point(-2, 1), Point(-2, -1), Point(-1, -3), \
         Point(0, 0), Point(1, 1), Point(2, 2), Point(2, -1), Point(3, 1), \
         Point(4, -1), Point(6, 2)]
    # 使用部分点创建一个多边形对象作为凸包的预期输出
    ch = Polygon(p[0], p[3], p[9], p[10], p[6], p[1])
    # 测试处理重复点的情况
    p.append(p[3])

    # 多于3个共线点的情况
    another_p = [Point(-45, -85), Point(-45, 85), Point(-45, 26), \
                 Point(-45, -24)]
    # 使用部分点创建线段对象作为凸包的另一个预期输出
    ch2 = Segment(another_p[0], another_p[1])

    # 断言：验证凸包函数对于给定点集的输出是否符合预期
    assert convex_hull(*another_p) == ch2
    assert convex_hull(*p) == ch
    assert convex_hull(p[0]) == p[0]
    assert convex_hull(p[0], p[1]) == Segment(p[0], p[1])

    # 断言：验证对于无唯一点的情况，凸包函数返回预期结果
    assert convex_hull(*[p[-1]]*3) == p[-1]

    # 断言：验证对于包含不同类型对象的集合，凸包函数返回预期的多边形对象
    assert convex_hull(*[Point(0, 0), \
                        Segment(Point(1, 0), Point(1, 1)), \
                        RegularPolygon(Point(2, 0), 2, 4)]) == \
        Polygon(Point(0, 0), Point(2, -2), Point(4, 0), Point(2, 2))


def test_encloses():
    # 创建一个具有凹点左侧的正方形多边形对象
    s = Polygon(Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1), \
        Point(S.Half, S.Half))
    # 断言：验证多边形对象对于特定点的包含关系是否符合预期
    assert s.encloses(Point(0, S.Half)) is False
    assert s.encloses(Point(S.Half, S.Half)) is False  # it's a vertex
    assert s.encloses(Point(Rational(3, 4), S.Half)) is True


def test_triangle_kwargs():
    # 断言：验证根据不同的角度参数化三角形对象的创建是否符合预期
    assert Triangle(sss=(3, 4, 5)) == \
        Triangle(Point(0, 0), Point(3, 0), Point(3, 4))
    assert Triangle(asa=(30, 2, 30)) == \
        Triangle(Point(0, 0), Point(2, 0), Point(1, sqrt(3)/3))
    assert Triangle(sas=(1, 45, 2)) == \
        Triangle(Point(0, 0), Point(2, 0), Point(sqrt(2)/2, sqrt(2)/2))
    assert Triangle(sss=(1, 2, 5)) is None
    assert deg(rad(180)) == 180


def test_transform():
    # 创建一个点列表，用于测试变换函数
    pts = [Point(0, 0), Point(S.Half, Rational(1, 4)), Point(1, 1)]
    # 创建期望输出的变换后的点列表
    pts_out = [Point(-4, -10), Point(-3, Rational(-37, 4)), Point(-2, -7)]
    # 断言：验证三角形对象的缩放变换是否得到预期结果
    assert Triangle(*pts).scale(2, 3, (4, 5)) == Triangle(*pts_out)
    # 断言：验证正多边形对象的缩放变换是否得到预期结果
    assert RegularPolygon((0, 0), 1, 4).scale(2, 3, (4, 5)) == \
        Polygon(Point(-2, -10), Point(-4, -7), Point(-6, -10), Point(-4, -13))
    # 断言：验证对称缩放的情况下正多边形对象是否得到预期结果
    assert RegularPolygon((0, 0), 1, 4).scale(2, 2) == \
        RegularPolygon(Point2D(0, 0), 2, 4, 0)


def test_reflect():
    # 创建符号变量
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    b = Symbol('b')
    m = Symbol('m')
    # 创建直线对象
    l = Line((0, b), slope=m)
    # 创建点对象
    p = Point(x, y)
    # 计算点关于直线的镜像点
    r = p.reflect(l)
    # 计算点到直线垂线段的长度
    dp = l.perpendicular_segment(p).length
    # 计算镜像点到直线垂线段的长度
    dr = l.perpendicular_segment(r).length

    # 断言：验证垂线段长度是否数值上相等
    assert verify_numerically(dp, dr)

    # 断言：验证多边形对象关于给定直线的镜像是否符合预期
    assert Polygon((1, 0), (2, 0), (2, 2)).reflect(Line((3, 0), slope=oo)) \
        == Triangle(Point(5, 0), Point(4, 0), Point(4, 2))
    assert Polygon((1, 0), (2, 0), (2, 2)).reflect(Line((0, 3), slope=oo)) \
        == Triangle(Point(-1, 0), Point(-2, 0), Point(-2, 2))
    # 断言：验证多边形在给定直线的反射是否等于三角形的特定配置
    
    # 第一个断言：验证多边形 (1, 0), (2, 0), (2, 2) 相对于直线 (0, 3) 上斜率为0的反射是否等于三角形 (1, 6), (2, 6), (2, 4)
    assert Polygon((1, 0), (2, 0), (2, 2)).reflect(Line((0, 3), slope=0)) \
        == Triangle(Point(1, 6), Point(2, 6), Point(2, 4))
    
    # 第二个断言：验证多边形 (1, 0), (2, 0), (2, 2) 相对于直线 (3, 0) 上斜率为0的反射是否等于三角形 (1, 0), (2, 0), (2, -2)
    assert Polygon((1, 0), (2, 0), (2, 2)).reflect(Line((3, 0), slope=0)) \
        == Triangle(Point(1, 0), Point(2, 0), Point(2, -2))
# 定义测试函数 test_bisectors，用于测试多边形的角平分线方法
def test_bisectors():
    # 创建三个点对象 p1, p2, p3，并定义三角形 t
    p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, 1)
    t = Triangle(p1, p2, p3)
    # 断言验证 t 的角平分线方法 bisectors 的返回结果
    assert t.bisectors()[p2] == Segment(Point(1, 0), Point(0, sqrt(2) - 1))

    # 创建多边形对象 p 和 q
    p = Polygon(Point(0, 0), Point(2, 0), Point(1, 1), Point(0, 3))
    q = Polygon(Point(1, 0), Point(2, 0), Point(3, 3), Point(-1, 5))
    # 断言验证 p 和 q 的角平分线方法 bisectors 的返回结果
    assert p.bisectors()[Point2D(0, 3)] == Ray2D(Point2D(0, 3), \
        Point2D(sin(acos(2*sqrt(5)/5)/2), 3 - cos(acos(2*sqrt(5)/5)/2)))
    assert q.bisectors()[Point2D(-1, 5)] == \
        Ray2D(Point2D(-1, 5), Point2D(-1 + sqrt(29)*(5*sin(acos(9*sqrt(145)/145)/2) + \
        2*cos(acos(9*sqrt(145)/145)/2))/29, sqrt(29)*(-5*cos(acos(9*sqrt(145)/145)/2) + \
        2*sin(acos(9*sqrt(145)/145)/2))/29 + 5))

    # 创建具有五个顶点的多边形对象 poly
    poly = Polygon(Point(3, 4), Point(0, 0), Point(8, 7), Point(-1, 1), Point(19, -19))
    # 断言验证 poly 的角平分线方法 bisectors 的返回结果
    assert poly.bisectors()[Point2D(-1, 1)] == Ray2D(Point2D(-1, 1), \
        Point2D(-1 + sin(acos(sqrt(26)/26)/2 + pi/4), 1 - sin(-acos(sqrt(26)/26)/2 + pi/4)))

# 定义测试函数 test_incenter，验证三角形的内心方法
def test_incenter():
    # 断言验证三角形的内心坐标
    assert Triangle(Point(0, 0), Point(1, 0), Point(0, 1)).incenter \
        == Point(1 - sqrt(2)/2, 1 - sqrt(2)/2)

# 定义测试函数 test_inradius，验证三角形的内切圆半径
def test_inradius():
    # 断言验证三角形的内切圆半径
    assert Triangle(Point(0, 0), Point(4, 0), Point(0, 3)).inradius == 1

# 定义测试函数 test_incircle，验证三角形的内切圆
def test_incircle():
    # 断言验证三角形的内切圆
    assert Triangle(Point(0, 0), Point(2, 0), Point(0, 2)).incircle \
        == Circle(Point(2 - sqrt(2), 2 - sqrt(2)), 2 - sqrt(2))

# 定义测试函数 test_exradii，验证三角形的外接圆半径
def test_exradii():
    # 创建三角形 t，并验证其外接圆半径
    t = Triangle(Point(0, 0), Point(6, 0), Point(0, 2))
    assert t.exradii[t.sides[2]] == (-2 + sqrt(10))

# 定义测试函数 test_medians，验证三角形的中位线
def test_medians():
    # 创建三角形 t，并验证其中位线
    t = Triangle(Point(0, 0), Point(1, 0), Point(0, 1))
    assert t.medians[Point(0, 0)] == Segment(Point(0, 0), Point(S.Half, S.Half))

# 定义测试函数 test_medial，验证三角形的中位三角形
def test_medial():
    # 断言验证三角形的中位三角形
    assert Triangle(Point(0, 0), Point(1, 0), Point(0, 1)).medial \
        == Triangle(Point(S.Half, 0), Point(S.Half, S.Half), Point(0, S.Half))

# 定义测试函数 test_nine_point_circle，验证三角形的九点圆
def test_nine_point_circle():
    # 断言验证三角形的九点圆
    assert Triangle(Point(0, 0), Point(1, 0), Point(0, 1)).nine_point_circle \
        == Circle(Point2D(Rational(1, 4), Rational(1, 4)), sqrt(2)/4)

# 定义测试函数 test_eulerline，验证三角形的欧拉线
def test_eulerline():
    # 断言验证三角形的欧拉线
    assert Triangle(Point(0, 0), Point(1, 0), Point(0, 1)).eulerline \
        == Line(Point2D(0, 0), Point2D(S.Half, S.Half))
    assert Triangle(Point(0, 0), Point(10, 0), Point(5, 5*sqrt(3))).eulerline \
        == Point2D(5, 5*sqrt(3)/3)
    assert Triangle(Point(4, -6), Point(4, -1), Point(-3, 3)).eulerline \
        == Line(Point2D(Rational(64, 7), 3), Point2D(Rational(-29, 14), Rational(-7, 2)))

# 定义测试函数 test_intersection，验证多边形的交点
def test_intersection():
    # 创建两个多边形对象 poly1 和 poly2
    poly1 = Triangle(Point(0, 0), Point(1, 0), Point(0, 1))
    poly2 = Polygon(Point(0, 1), Point(-5, 0),
                    Point(0, -4), Point(0, Rational(1, 5)),
                    Point(S.Half, -0.1), Point(1, 0), Point(0, 1))
    # 断言验证 poly1 和 poly2 的交点
    assert poly1.intersection(poly2) == [Point2D(Rational(1, 3), 0),
        Segment(Point(0, Rational(1, 5)), Point(0, 0)),
        Segment(Point(1, 0), Point(0, 1))]
    # 断言：poly2 和 poly1 多边形的交集应为特定的几何对象列表
    assert poly2.intersection(poly1) == [Point(Rational(1, 3), 0),  # 交点为有理数点 (1/3, 0)
        Segment(Point(0, 0), Point(0, Rational(1, 5))),  # 交线段为从 (0, 0) 到 (0, 1/5)
        Segment(Point(1, 0), Point(0, 1))]  # 另一个交线段从 (1, 0) 到 (0, 1)
    
    # 断言：poly1 和 Point(0, 0) 的交集应为点 (0, 0)
    assert poly1.intersection(Point(0, 0)) == [Point(0, 0)]
    
    # 断言：poly1 和 Point(-12, -43) 的交集应为空列表，表示无交点
    assert poly1.intersection(Point(-12,  -43)) == []
    
    # 断言：poly2 和从 (-12, 0) 到 (12, 0) 的直线的交集应为特定的几何对象列表
    assert poly2.intersection(Line((-12, 0), (12, 0))) == [Point(-5, 0),  # 交点 (-5, 0)
        Point(0, 0),  # 交点 (0, 0)
        Point(Rational(1, 3), 0),  # 交点 (1/3, 0)
        Point(1, 0)]  # 交点 (1, 0)
    
    # 断言：poly2 和从 (-12, 12) 到 (12, 12) 的直线的交集应为空列表，表示无交点
    assert poly2.intersection(Line((-12, 12), (12, 12))) == []
    
    # 断言：poly2 和从 (-3, 4) 开始，向 (1, 0) 方向的射线的交集应为特定的几何对象列表
    assert poly2.intersection(Ray((-3, 4), (1, 0))) == [Segment(Point(1, 0),  # 交线段从 (1, 0)
        Point(0, 1))]  # 到 (0, 1)
    
    # 断言：poly2 和以 (0, -1) 为中心、半径为 1 的圆的交集应为特定的几何对象列表
    assert poly2.intersection(Circle((0, -1), 1)) == [Point(0, -2),  # 交点 (0, -2)
        Point(0, 0)]  # 交点 (0, 0)
    
    # 断言：poly1 和 poly1 多边形的交集应为特定的几何对象列表
    assert poly1.intersection(poly1) == [Segment(Point(0, 0), Point(1, 0)),  # 交线段从 (0, 0) 到 (1, 0)
        Segment(Point(0, 1), Point(0, 0)),  # 交线段从 (0, 1) 到 (0, 0)
        Segment(Point(1, 0), Point(0, 1))]  # 交线段从 (1, 0) 到 (0, 1)
    
    # 断言：poly2 和 poly2 多边形的交集应为特定的几何对象列表
    assert poly2.intersection(poly2) == [Segment(Point(-5, 0), Point(0, -4)),  # 交线段从 (-5, 0) 到 (0, -4)
        Segment(Point(0, -4), Point(0, Rational(1, 5))),  # 交线段从 (0, -4) 到 (0, 1/5)
        Segment(Point(0, Rational(1, 5)), Point(S.Half, Rational(-1, 10))),  # 交线段从 (0, 1/5) 到 (1/2, -1/10)
        Segment(Point(0, 1), Point(-5, 0)),  # 交线段从 (0, 1) 到 (-5, 0)
        Segment(Point(S.Half, Rational(-1, 10)), Point(1, 0)),  # 交线段从 (1/2, -1/10) 到 (1, 0)
        Segment(Point(1, 0), Point(0, 1))]  # 交线段从 (1, 0) 到 (0, 1)
    
    # 断言：poly2 和以 (0, 1)，(1, 0)，(-1, 1) 为顶点的三角形的交集应为特定的几何对象列表
    assert poly2.intersection(Triangle(Point(0, 1), Point(1, 0), Point(-1, 1))) \
        == [Point(Rational(-5, 7), Rational(6, 7)),  # 交点 (-5/7, 6/7)
        Segment(Point(0, 1), Point(1, 0))]  # 交线段从 (0, 1) 到 (1, 0)
    
    # 断言：poly1 和以 (-12, -15) 为中心、边数为 3、边长为 3 的正多边形的交集应为空列表，表示无交点
    assert poly1.intersection(RegularPolygon((-12, -15), 3, 3)) == []
# 定义一个测试函数，用于测试多边形对象的参数值方法
def test_parameter_value():
    # 创建符号变量 t
    t = Symbol('t')
    # 创建一个正方形对象 sq，顶点为 (0, 0), (0, 1), (1, 1), (1, 0)
    sq = Polygon((0, 0), (0, 1), (1, 1), (1, 0))
    # 断言正方形 sq 在点 (0.5, 1) 处的参数值为 {t: Rational(3, 8)}
    assert sq.parameter_value((0.5, 1), t) == {t: Rational(3, 8)}
    
    # 创建一个四边形对象 q，顶点为 (0, 0), (2, 1), (2, 4), (4, 0)
    q = Polygon((0, 0), (2, 1), (2, 4), (4, 0))
    # 断言四边形 q 在点 (4, 0) 处的参数值为 {t: -6 + 3*sqrt(5)}
    assert q.parameter_value((4, 0), t) == {t: -6 + 3*sqrt(5)}  # 大约为 0.708

    # 断言调用参数值方法对于超出多边形的点会引发 ValueError 异常
    raises(ValueError, lambda: sq.parameter_value((5, 6), t))
    # 断言调用参数值方法对于不是点的输入（圆形对象）会引发 ValueError 异常
    raises(ValueError, lambda: sq.parameter_value(Circle(Point(0, 0), 1), t))


# 定义一个测试函数，用于测试多边形对象的任意点方法
def test_issue_12966():
    # 创建一个多边形 poly，顶点为 (0, 0), (0, 10), (5, 10), (5, 5), (10, 5), (10, 0)
    poly = Polygon(Point(0, 0), Point(0, 10), Point(5, 10), Point(5, 5),
                   Point(10, 5), Point(10, 0))
    # 创建符号变量 t
    t = Symbol('t')
    # 获取多边形 poly 在参数 t 处的任意点
    pt = poly.arbitrary_point(t)
    # 计算步长 DELTA
    DELTA = 5 / poly.perimeter
    # 断言多边形 poly 在每个步长处替换参数 t 后的点列表是否等于指定的点列表
    assert [pt.subs(t, DELTA*i) for i in range(int(1/DELTA))] == [
        Point(0, 0), Point(0, 5), Point(0, 10), Point(5, 10),
        Point(5, 5), Point(10, 5), Point(10, 0), Point(5, 0)]


# 定义一个测试函数，用于测试多边形对象的二阶矩（面积矩）方法
def test_second_moment_of_area():
    # 创建符号变量 x, y
    x, y = symbols('x, y')
    
    # 定义三角形的顶点 p1, p2, p3 和一个基点 p
    p1, p2, p3 = [(0, 0), (4, 0), (0, 2)]
    p = (0, 0)
    # 定义斜边的方程式 eq_y
    eq_y = (1 - x/4) * 2
    # 计算 I_xx, I_yy, I_xy
    I_yy = integrate((x**2) * (integrate(1, (y, 0, eq_y))), (x, 0, 4))
    I_xx = integrate(1 * (integrate(y**2, (y, 0, eq_y))), (x, 0, 4))
    I_xy = integrate(x * (integrate(y, (y, 0, eq_y))), (x, 0, 4))

    # 创建三角形对象 triangle
    triangle = Polygon(p1, p2, p3)

    # 断言计算得到的二阶矩与三角形对象的方法计算的二阶矩是否相等
    assert (I_xx - triangle.second_moment_of_area(p)[0]) == 0
    assert (I_yy - triangle.second_moment_of_area(p)[1]) == 0
    assert (I_xy - triangle.second_moment_of_area(p)[2]) == 0

    # 定义矩形的顶点 p1, p2, p3, p4
    p1, p2, p3, p4 = [(0, 0), (4, 0), (4, 2), (0, 2)]
    # 计算 I_xx, I_yy, I_xy
    I_yy = integrate((x**2) * integrate(1, (y, 0, 2)), (x, 0, 4))
    I_xx = integrate(1 * integrate(y**2, (y, 0, 2)), (x, 0, 4))
    I_xy = integrate(x * integrate(y, (y, 0, 2)), (x, 0, 4))

    # 创建矩形对象 rectangle
    rectangle = Polygon(p1, p2, p3, p4)

    # 断言计算得到的二阶矩与矩形对象的方法计算的二阶矩是否相等
    assert (I_xx - rectangle.second_moment_of_area(p)[0]) == 0
    assert (I_yy - rectangle.second_moment_of_area(p)[1]) == 0
    assert (I_xy - rectangle.second_moment_of_area(p)[2]) == 0

    # 创建一个正六边形对象 r
    r = RegularPolygon(Point(0, 0), 5, 3)
    # 断言正六边形对象 r 的二阶矩是否为 (1875*sqrt(3)/S(32), 1875*sqrt(3)/S(32), 0)
    assert r.second_moment_of_area() == (1875*sqrt(3)/S(32), 1875*sqrt(3)/S(32), 0)


# 定义一个测试函数，用于测试多边形对象的一阶矩（面积矩）方法
def test_first_moment():
    # 创建正数符号变量 a, b
    a, b = symbols('a, b', positive=True)
    
    # 创建矩形对象 p1，顶点为 (0, 0), (a, 0), (a, b), (0, b)
    p1 = Polygon((0, 0), (a, 0), (a, b), (0, b))
    # 断言矩形对象 p1 的一阶矩是否为 (a*b**2/8, a**2*b/8)
    assert p1.first_moment_of_area() == (a*b**2/8, a**2*b/8)
    # 断言矩形对象 p1 在指定点 (a/3, b/4) 处的一阶矩是否为 (-3*a*b**2/32, -a**2*b/9)

    assert p1.first_moment_of_area((a/3, b/4)) == (-3*a*b**2/32, -a**2*b/9)

    # 创建矩形对象 p1，顶点为 (0, 0), (40, 0), (40, 30), (0, 30)
    p1 = Polygon((0, 0), (40, 0), (40, 30), (0, 30))
    # 断言矩形对象 p1 的一阶矩是否为 (4500, 6000)

    assert p1.first_moment_of_area() == (4500, 6000)

    # 创建三角形对象 p2，顶点为 (0, 0), (a, 0), (a/2, b)
    p2 = Polygon((0, 0), (a, 0), (a/2, b))
    # 断言三角形对象 p2 的一阶矩是否为 (4*a*b**2/81, a**2*b/24)
    assert p2.first_moment_of_area() == (4*a*b**2/81, a**2*b/24)
    # 断言
    # 断言计算矩形的截面模量，使用给定点(x, y)，期望的结果是(a*b**3/12/(-b/2 + y), a**3*b/12/(-a/2 + x))
    assert rectangle.section_modulus(Point(x, y)) == (a*b**3/12/(-b/2 + y), a**3*b/12/(-a/2 + x))
    
    # 断言计算矩形的极惯性矩，期望的结果是a**3*b/12 + a*b**3/12
    assert rectangle.polar_second_moment_of_area() == a**3*b/12 + a*b**3/12
    
    # 创建一个六边形的正多边形对象convex，中心在(0, 0)，半径为1
    convex = RegularPolygon((0, 0), 1, 6)
    # 断言计算正多边形的截面模量，期望的结果是(Rational(5, 8), sqrt(3)*Rational(5, 16))
    assert convex.section_modulus() == (Rational(5, 8), sqrt(3)*Rational(5, 16))
    # 断言计算正多边形的极惯性矩，期望的结果是5*sqrt(3)/S(8)
    assert convex.polar_second_moment_of_area() == 5*sqrt(3)/S(8)
    
    # 创建一个凹多边形对象concave，顶点分别为(0, 0), (1, 8), (3, 4), (4, 6), (7, 1)
    concave = Polygon((0, 0), (1, 8), (3, 4), (4, 6), (7, 1))
    # 断言计算凹多边形的截面模量，期望的结果是(Rational(-6371, 429), Rational(-9778, 519))
    assert concave.section_modulus() == (Rational(-6371, 429), Rational(-9778, 519))
    # 断言计算凹多边形的极惯性矩，期望的结果是Rational(-38669, 252)
    assert concave.polar_second_moment_of_area() == Rational(-38669, 252)
def test_cut_section():
    # 定义一个凹多边形
    p = Polygon((-1, -1), (1, Rational(5, 2)), (2, 1), (3, Rational(5, 2)), (4, 2), (5, 3), (-1, 3))
    # 定义一条直线
    l = Line((0, 0), (Rational(9, 2), 3))
    # 对凹多边形使用给定直线进行切割，并得到切割后的两个多边形
    p1 = p.cut_section(l)[0]
    p2 = p.cut_section(l)[1]
    # 断言切割后的第一个多边形符合预期
    assert p1 == Polygon(
        Point2D(Rational(-9, 13), Rational(-6, 13)), Point2D(1, Rational(5, 2)), Point2D(Rational(24, 13), Rational(16, 13)),
        Point2D(Rational(12, 5), Rational(8, 5)), Point2D(3, Rational(5, 2)), Point2D(Rational(24, 7), Rational(16, 7)),
        Point2D(Rational(9, 2), 3), Point2D(-1, 3), Point2D(-1, Rational(-2, 3)))
    # 断言切割后的第二个多边形符合预期
    assert p2 == Polygon(Point2D(-1, -1), Point2D(Rational(-9, 13), Rational(-6, 13)), Point2D(Rational(24, 13), Rational(16, 13)),
        Point2D(2, 1), Point2D(Rational(12, 5), Rational(8, 5)), Point2D(Rational(24, 7), Rational(16, 7)), Point2D(4, 2), Point2D(5, 3),
        Point2D(Rational(9, 2), 3), Point2D(-1, Rational(-2, 3)))

    # 定义一个凸多边形
    p = RegularPolygon(Point2D(0, 0), 6, 6)
    # 对凸多边形使用斜率为1的直线进行切割，并得到切割后的两个多边形
    s = p.cut_section(Line((0, 0), slope=1))
    # 断言切割后的第一个多边形符合预期
    assert s[0] == Polygon(Point2D(-3*sqrt(3) + 9, -3*sqrt(3) + 9), Point2D(3, 3*sqrt(3)),
        Point2D(-3, 3*sqrt(3)), Point2D(-6, 0), Point2D(-9 + 3*sqrt(3), -9 + 3*sqrt(3)))
    # 断言切割后的第二个多边形符合预期
    assert s[1] == Polygon(Point2D(6, 0), Point2D(-3*sqrt(3) + 9, -3*sqrt(3) + 9),
        Point2D(-9 + 3*sqrt(3), -9 + 3*sqrt(3)), Point2D(-3, -3*sqrt(3)), Point2D(3, -3*sqrt(3)))

    # 情况：直线与多边形的边重合但不相交
    a, b = 20, 10
    t1, t2, t3, t4 = [(0, b), (0, 0), (a, 0), (a, b)]
    p = Polygon(t1, t2, t3, t4)
    # 使用斜率为0的直线切割多边形，并得到切割后的两个多边形
    p1, p2 = p.cut_section(Line((0, b), slope=0))
    # 断言切割后的第一个多边形符合预期
    assert p1 == None
    # 断言切割后的第二个多边形符合预期
    assert p2 == Polygon(Point2D(0, 10), Point2D(0, 0), Point2D(20, 0), Point2D(20, 10))

    # 使用斜率为0的直线再次切割多边形，并得到切割后的两个多边形
    p3, p4 = p.cut_section(Line((0, 0), slope=0))
    # 断言切割后的第一个多边形符合预期
    assert p3 == Polygon(Point2D(0, 10), Point2D(0, 0), Point2D(20, 0), Point2D(20, 10))
    # 断言切割后的第二个多边形符合预期
    assert p4 == None

    # 情况：直线与多边形完全不相交
    raises(ValueError, lambda: p.cut_section(Line((0, a), slope=0)))

def test_type_of_triangle():
    # 等腰三角形
    p1 = Polygon(Point(0, 0), Point(5, 0), Point(2, 4))
    # 断言判断是否为等腰三角形
    assert p1.is_isosceles() == True
    # 断言判断是否为不等边三角形
    assert p1.is_scalene() == False
    # 断言判断是否为等边三角形
    assert p1.is_equilateral() == False

    # 不等边三角形
    p2 = Polygon (Point(0, 0), Point(0, 2), Point(4, 0))
    # 断言判断是否为等腰三角形
    assert p2.is_isosceles() == False
    # 断言判断是否为不等边三角形
    assert p2.is_scalene() == True
    # 断言判断是否为等边三角形
    assert p2.is_equilateral() == False

    # 等边三角形
    p3 = Polygon(Point(0, 0), Point(6, 0), Point(3, sqrt(27)))
    # 断言判断是否为等腰三角形
    assert p3.is_isosceles() == True
    # 断言判断是否为不等边三角形
    assert p3.is_scalene() == False
    # 断言判断是否为等边三角形
    assert p3.is_equilateral() == True

def test_do_poly_distance():
    # 不相交的多边形
    square1 = Polygon (Point(0, 0), Point(0, 1), Point(1, 1), Point(1, 0))
    triangle1 = Polygon(Point(1, 2), Point(2, 2), Point(2, 1))
    # 断言计算多边形之间的距离
    assert square1._do_poly_distance(triangle1) == sqrt(2)/2

    # 多边形的边相交的情况
    # 创建一个正方形 Polygon 对象，顶点分别为 (1, 0), (2, 0), (2, 1), (1, 1)
    square2 = Polygon(Point(1, 0), Point(2, 0), Point(2, 1), Point(1, 1))
    
    # 使用 warns 上下文管理器捕获 UserWarning，并匹配特定字符串以测试警告
    with warns(UserWarning, \
               match="Polygons may intersect producing erroneous output", test_stacklevel=False):
        # 断言 square1 和 square2 的多边形距离为 0
        assert square1._do_poly_distance(square2) == 0

    # 创建一个三角形 Polygon 对象，顶点分别为 (0, -1), (2, -1), (1/2, 1/2)
    # 该三角形与 square1 的边界交叉
    triangle2 = Polygon(Point(0, -1), Point(2, -1), Point(S.Half, S.Half))
    
    # 使用 warns 上下文管理器捕获 UserWarning，并匹配特定字符串以测试警告
    with warns(UserWarning, \
               match="Polygons may intersect producing erroneous output", test_stacklevel=False):
        # 断言 triangle2 和 square1 的多边形距离为 0
        assert triangle2._do_poly_distance(square1) == 0
```