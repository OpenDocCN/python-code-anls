# `D:\src\scipysrc\sympy\sympy\geometry\tests\test_plane.py`

```
from sympy.core.numbers import (Rational, pi)  # 导入有理数和圆周率 pi
from sympy.core.singleton import S  # 导入符号 S
from sympy.core.symbol import (Dummy, symbols)  # 导入符号 Dummy 和 symbols
from sympy.functions.elementary.miscellaneous import sqrt  # 导入平方根函数 sqrt
from sympy.functions.elementary.trigonometric import (asin, cos, sin)  # 导入反正弦、余弦、正弦函数
from sympy.geometry import Line, Point, Ray, Segment, Point3D, Line3D, Ray3D, Segment3D, Plane, Circle  # 导入几何模块中的各种对象
from sympy.geometry.util import are_coplanar  # 导入判断共面的函数 are_coplanar
from sympy.testing.pytest import raises  # 导入测试框架中的 raises 函数


def test_plane():
    x, y, z, u, v = symbols('x y z u v', real=True)  # 定义实数域的符号变量
    p1 = Point3D(0, 0, 0)  # 创建三维空间的点 p1
    p2 = Point3D(1, 1, 1)  # 创建三维空间的点 p2
    p3 = Point3D(1, 2, 3)  # 创建三维空间的点 p3
    pl3 = Plane(p1, p2, p3)  # 创建由三点确定的平面对象 pl3
    pl4 = Plane(p1, normal_vector=(1, 1, 1))  # 创建由点 p1 和法向量确定的平面对象 pl4
    pl4b = Plane(p1, p2)  # 创建由两点确定的平面对象 pl4b
    pl5 = Plane(p3, normal_vector=(1, 2, 3))  # 创建由点 p3 和法向量确定的平面对象 pl5
    pl6 = Plane(Point3D(2, 3, 7), normal_vector=(2, 2, 2))  # 创建由点和法向量确定的平面对象 pl6
    pl7 = Plane(Point3D(1, -5, -6), normal_vector=(1, -2, 1))  # 创建由点和法向量确定的平面对象 pl7
    pl8 = Plane(p1, normal_vector=(0, 0, 1))  # 创建由点 p1 和法向量确定的平面对象 pl8
    pl9 = Plane(p1, normal_vector=(0, 12, 0))  # 创建由点 p1 和法向量确定的平面对象 pl9
    pl10 = Plane(p1, normal_vector=(-2, 0, 0))  # 创建由点 p1 和法向量确定的平面对象 pl10
    pl11 = Plane(p2, normal_vector=(0, 0, 1))  # 创建由点 p2 和法向量确定的平面对象 pl11
    l1 = Line3D(Point3D(5, 0, 0), Point3D(1, -1, 1))  # 创建三维空间的直线对象 l1
    l2 = Line3D(Point3D(0, -2, 0), Point3D(3, 1, 1))  # 创建三维空间的直线对象 l2
    l3 = Line3D(Point3D(0, -1, 0), Point3D(5, -1, 9))  # 创建三维空间的直线对象 l3

    raises(ValueError, lambda: Plane(p1, p1, p1))  # 断言抛出 ValueError 异常

    assert Plane(p1, p2, p3) != Plane(p1, p3, p2)  # 断言不同的三点确定的平面对象不相等
    assert Plane(p1, p2, p3).is_coplanar(Plane(p1, p3, p2))  # 断言两个平面对象共面
    assert Plane(p1, p2, p3).is_coplanar(p1)  # 断言平面对象与点 p1 共面
    assert Plane(p1, p2, p3).is_coplanar(Circle(p1, 1)) is False  # 断言平面对象与圆不共面
    assert Plane(p1, normal_vector=(0, 0, 1)).is_coplanar(Circle(p1, 1))  # 断言平面对象与圆共面

    assert pl3 == Plane(Point3D(0, 0, 0), normal_vector=(1, -2, 1))  # 断言平面对象 pl3 与给定的点和法向量定义的平面相等
    assert pl3 != pl4  # 断言平面对象 pl3 不等于平面对象 pl4
    assert pl4 == pl4b  # 断言平面对象 pl4 等于平面对象 pl4b
    assert pl5 == Plane(Point3D(1, 2, 3), normal_vector=(1, 2, 3))  # 断言平面对象 pl5 与给定的点和法向量定义的平面相等

    assert pl5.equation(x, y, z) == x + 2*y + 3*z - 14  # 断言平面对象 pl5 的方程式为 x + 2*y + 3*z - 14
    assert pl3.equation(x, y, z) == x - 2*y + z  # 断言平面对象 pl3 的方程式为 x - 2*y + z

    assert pl3.p1 == p1  # 断言平面对象 pl3 的第一个点为 p1
    assert pl4.p1 == p1  # 断言平面对象 pl4 的第一个点为 p1
    assert pl5.p1 == p3  # 断言平面对象 pl5 的第一个点为 p3

    assert pl4.normal_vector == (1, 1, 1)  # 断言平面对象 pl4 的法向量为 (1, 1, 1)
    assert pl5.normal_vector == (1, 2, 3)  # 断言平面对象 pl5 的法向量为 (1, 2, 3)

    assert p1 in pl3  # 断言点 p1 在平面对象 pl3 上
    assert p1 in pl4  # 断言点 p1 在平面对象 pl4 上
    assert p3 in pl5  # 断言点 p3 在平面对象 pl5 上

    assert pl3.projection(Point(0, 0)) == p1  # 断言点 (0, 0) 在平面对象 pl3 的投影为 p1
    p = pl3.projection(Point3D(1, 1, 0))
    assert p == Point3D(Rational(7, 6), Rational(2, 3), Rational(1, 6))  # 断言平面对象 pl3 投影点的精确值

    l = pl3.projection_line(Line(Point(0, 0), Point(1, 1)))  # 获取平面对象 pl3 在给定直线上的投影线
    assert l == Line3D(Point3D(0, 0, 0), Point3D(Rational(7, 6), Rational(2, 3), Rational(1, 6)))  # 断言投影线的精确值

    # get a segment that does not intersect the plane which is also
    # parallel to pl3's normal vector
    t = Dummy()  # 创建一个虚拟符号变量 t
    r = pl3.random_point()  # 获取平面对象 pl3 的随机点
    a = pl3.perpendicular_line(r).arbitrary_point(t)  # 获取与平面对象 pl3 法线平行且不与平面相交的线段的任意一点
    s = Segment3D(a.subs(t, 1), a.subs(t, 2))  # 创建由任意点确定的线段对象 s
    assert s.p1 not in pl3 and s.p2 not in pl3  # 断言线段 s 的两个端点都不在平面对象 pl3 上
    assert pl3.projection_line(s).equals(r)  # 断言线段 s 在平面对象 pl3 上的投影线等于 r
    assert pl3.projection_line(Segment(Point(1, 0), Point(1, 1))) == \
    # 断言：检查投影线函数的计算结果是否与预期的射线相同
    assert pl6.projection_line(Ray(Point(1, 0), Point(1, 1))) == \
               Ray3D(Point3D(Rational(14, 3), Rational(11, 3), Rational(11, 3)), Point3D(Rational(13, 3), Rational(13, 3), Rational(10, 3)))
    # 断言：检查垂直线函数对参数 r.args 的计算结果是否与直接使用 r 参数的结果相同
    assert pl3.perpendicular_line(r.args) == pl3.perpendicular_line(r)

    # 断言：检查两个平面 pl3 和 pl6 是否不平行
    assert pl3.is_parallel(pl6) is False
    # 断言：检查 pl4 和 pl6 是否平行
    assert pl4.is_parallel(pl6)
    # 断言：检查 pl3 是否与通过两点 p1, p2 定义的直线平行
    assert pl3.is_parallel(Line(p1, p2))
    # 断言：检查 pl6 是否与直线 l1 不平行
    assert pl6.is_parallel(l1) is False

    # 断言：检查 pl3 和 pl6 是否垂直
    assert pl3.is_perpendicular(pl6)
    # 断言：检查 pl4 和 pl7 是否垂直
    assert pl4.is_perpendicular(pl7)
    # 断言：检查 pl6 和 pl7 是否垂直
    assert pl6.is_perpendicular(pl7)
    # 断言：检查 pl6 和 pl4 是否不垂直
    assert pl6.is_perpendicular(pl4) is False
    # 断言：检查 pl6 和直线 l1 是否不垂直
    assert pl6.is_perpendicular(l1) is False
    # 断言：检查 pl6 和通过点 (0, 0, 0) 和 (1, 1, 1) 定义的直线是否垂直
    assert pl6.is_perpendicular(Line((0, 0, 0), (1, 1, 1)))
    # 断言：检查 pl6 和点 (1, 1) 是否不垂直
    assert pl6.is_perpendicular((1, 1)) is False

    # 断言：检查平面 pl6 到其参数 u, v 定义的任意点的距离是否为 0
    assert pl6.distance(pl6.arbitrary_point(u, v)) == 0
    # 断言：检查平面 pl7 到其参数 u, v 定义的任意点的距离是否为 0
    assert pl7.distance(pl7.arbitrary_point(u, v)) == 0
    # 断言：检查平面 pl6 到其参数 t 定义的任意点的距离是否为 0
    assert pl6.distance(pl6.arbitrary_point(t)) == 0
    # 断言：检查平面 pl7 到其参数 t 定义的任意点的距离是否为 0
    assert pl7.distance(pl7.arbitrary_point(t)) == 0
    # 断言：检查 pl6 到其参数 t 定义的任意点与 pl6.p1 之间的距离是否为 1（经过简化）
    assert pl6.p1.distance(pl6.arbitrary_point(t)).simplify() == 1
    # 断言：检查 pl7 到其参数 t 定义的任意点与 pl7.p1 之间的距离是否为 1（经过简化）
    assert pl7.p1.distance(pl7.arbitrary_point(t)).simplify() == 1
    # 断言：检查平面 pl3 到其参数 t 定义的任意点是否等于给定的三维点
    assert pl3.arbitrary_point(t) == Point3D(-sqrt(30)*sin(t)/30 + \
        2*sqrt(5)*cos(t)/5, sqrt(30)*sin(t)/15 + sqrt(5)*cos(t)/5, sqrt(30)*sin(t)/6)
    # 断言：检查平面 pl3 到其参数 u, v 定义的任意点是否等于给定的三维点
    assert pl3.arbitrary_point(u, v) == Point3D(2*u - v, u + 2*v, 5*v)

    # 断言：检查平面 pl7 到给定的三维点 (1, 3, 5) 的距离是否为 5*sqrt(6)/6
    assert pl7.distance(Point3D(1, 3, 5)) == 5*sqrt(6)/6
    # 断言：检查平面 pl6 到给定的三维点 (0, 0, 0) 的距离是否为 4*sqrt(3)
    assert pl6.distance(Point3D(0, 0, 0)) == 4*sqrt(3)
    # 断言：检查平面 pl6 到其起点 pl6.p1 的距离是否为 0
    assert pl6.distance(pl6.p1) == 0
    # 断言：检查平面 pl7 到平面 pl6 的距离是否为 0
    assert pl7.distance(pl6) == 0
    # 断言：检查平面 pl7 到直线 l1 的距离是否为 0
    assert pl7.distance(l1) == 0
    # 断言：检查平面 pl6 到线段 Segment3D(Point3D(2, 3, 1), Point3D(1, 3, 4)) 的距离是否等于 pl6 到点 Point3D(1, 3, 4) 的距离，均为 4*sqrt(3)/3
    assert pl6.distance(Segment3D(Point3D(2, 3, 1), Point3D(1, 3, 4))) == \
        pl6.distance(Point3D(1, 3, 4)) == 4*sqrt(3)/3
    # 断言：检查平面 pl6 到线段 Segment3D(Point3D(1, 3, 4), Point3D(0, 3, 7)) 的距离是否等于 pl6 到点 Point3D(0, 3, 7) 的距离，均为 2*sqrt(3)/3
    assert pl6.distance(Segment3D(Point3D(1, 3, 4), Point3D(0, 3, 7))) == \
        pl6.distance(Point3D(0, 3, 7)) == 2*sqrt(3)/3
    # 断言：检查平面 pl6 到线段 Segment3D(Point3D(0, 3, 7), Point3D(-1, 3, 10)) 的距离是否为 0
    assert pl6.distance(Segment3D(Point3D(0, 3, 7), Point3D(-1, 3, 10))) == 0
    # 断言：检查平面 pl6 到线段 Segment3D(Point3D(-1, 3, 10), Point3D(-2, 3, 13)) 的距离是否等于 pl6 到点 Point3D(-2, 3, 13) 的距离，均为 2*sqrt(3)/3
    assert pl6.distance(Segment3D(Point3D(-1, 3, 10), Point3D(-2, 3, 13))) == \
        pl6.distance(Point3D(-2, 3, 13)) == 2*sqrt(3)/3
    # 断言：检查平面 pl6 到给定点 (5, 5, 5) 和法向量 (8, 8, 8) 定义的平面的距离是否为 sqrt(3)
    assert pl6.distance(Plane(Point3D(5, 5, 5), normal_vector=(8, 8, 8))) == sqrt(3)
    # 断言：检查平面 pl6 到给定射线 Ray3D(Point3D(1, 3, 4), direction_ratio=[1, 0, -3]) 的距离是否等于 4*sqrt(3)/3
    assert pl6.distance(Ray3D(Point3D(1, 3, 4), direction_ratio=[1, 0
    # 使用 are_coplanar 函数检查给定的三个平面是否共面
    assert are_coplanar(Plane(p1, p2, p3), Plane(p1, p3, p2))

    # 使用 are_concurrent 函数检查三个平面是否共点，预期返回 False
    assert Plane.are_concurrent(pl3, pl4, pl5) is False

    # 使用 are_concurrent 函数检查单个平面是否与其他平面共点，预期返回 False
    assert Plane.are_concurrent(pl6) is False

    # 使用 raises 函数确保在给定参数下，调用 are_concurrent 函数会抛出 ValueError 异常
    raises(ValueError, lambda: Plane.are_concurrent(Point3D(0, 0, 0)))

    # 使用 raises 函数确保在给定参数下，调用 Plane 构造函数会抛出 ValueError 异常
    raises(ValueError, lambda: Plane((1, 2, 3), normal_vector=(0, 0, 0)))

    # 测试 parallel_plane 方法，预期返回一个与指定点和法向量平行的平面对象
    assert pl3.parallel_plane(Point3D(1, 2, 5)) == Plane(Point3D(1, 2, 5), \
                                                      normal_vector=(1, -2, 1))

    # 测试 perpendicular_plane 方法

    # 默认情况下，使用 (0, 0, 0) 和 (1, 0, 0) 构造平面对象，并测试其垂直平面的计算
    assert p.perpendicular_plane() == Plane(Point3D(0, 0, 0), (0, 1, 0))

    # 使用单个点 (1, 0, 1) 测试 perpendicular_plane 方法
    assert p.perpendicular_plane(Point3D(1, 0, 1)) == \
        Plane(Point3D(1, 0, 1), (0, 1, 0))

    # 使用点元组 (1, 0, 1) 和 (1, 1, 1) 测试 perpendicular_plane 方法
    assert p.perpendicular_plane((1, 0, 1), (1, 1, 1)) == \
        Plane(Point3D(1, 0, 1), (0, 0, -1))

    # 使用三个以上的平面点来测试 perpendicular_plane 方法，预期引发 ValueError 异常
    raises(ValueError, lambda: p.perpendicular_plane((1, 0, 1), (1, 1, 1), (1, 1, 0)))

    # 测试更多 perpendicular_plane 方法的各种情况
    # case 4
    assert p.perpendicular_plane(a, b) == Plane(a, (1, 0, 0))
    # case 1
    assert p.perpendicular_plane(a, n) == Plane(a, (-1, 0, 0))
    # case 2
    assert Plane(a, normal_vector=b.args).perpendicular_plane(a, a + b) == \
        Plane(Point3D(0, 0, 0), (1, 0, 0))
    # case 1&3
    assert Plane(b, normal_vector=Z).perpendicular_plane(b, b + n) == \
        Plane(Point3D(0, 1, 0), (-1, 0, 0))
    # case 2&3
    assert Plane(b, normal_vector=b.args).perpendicular_plane(n, n + b) == \
        Plane(Point3D(0, 0, 1), (1, 0, 0))

    # 使用法向量为 (0, 0, 1) 的平面对象测试 perpendicular_plane 方法
    p = Plane(a, normal_vector=(0, 0, 1))
    assert p.perpendicular_plane() == Plane(a, normal_vector=(1, 0, 0))

    # 测试 intersection 方法

    # 测试平面与自身的交点，预期返回包含自身的列表
    assert pl6.intersection(pl6) == [pl6]

    # 测试平面与其一个点的交点，预期返回包含该点的列表
    assert pl4.intersection(pl4.p1) == [pl4.p1]

    # 测试平面与另一个平面的交点，预期返回一个线对象列表
    assert pl3.intersection(pl6) == [
        Line3D(Point3D(8, 4, 0), Point3D(2, 4, 6))]

    # 测试平面与三维线的交点，预期返回一个点对象列表
    assert pl3.intersection(Line3D(Point3D(1,2,4), Point3D(4,4,2))) == [
        Point3D(2, Rational(8, 3), Rational(10, 3))]

    # 测试平面与另一个平面的交点，预期返回一个线对象列表
    assert pl3.intersection(Plane(Point3D(6, 0, 0), normal_vector=(2, -5, 3))
        ) == [Line3D(Point3D(-24, -12, 0), Point3D(-25, -13, -1))]

    # 测试平面与三维射线的交点，预期返回一个点对象列表
    assert pl6.intersection(Ray3D(Point3D(2, 3, 1), Point3D(1, 3, 4))) == [
        Point3D(-1, 3, 10)]

    # 测试平面与三维线段的交点，预期返回空列表
    assert pl6.intersection(Segment3D(Point3D(2, 3, 1), Point3D(1, 3, 4))) == []

    # 测试平面与二维线的交点，预期返回一个点对象列表
    assert pl7.intersection(Line(Point(2, 3), Point(4, 2))) == [
        Point3D(Rational(13, 2), Rational(3, 4), 0)]

    # 使用二维射线测试平面与其交点，预期返回一个射线对象列表
    r = Ray(Point(2, 3), Point(4, 2))
    assert Plane((1,2,0), normal_vector=(0,0,1)).intersection(r) == [
        Ray3D(Point(2, 3), Point(4, 2))]

    # 测试平面与另一个平面的交点，预期返回一个线对象列表
    assert pl9.intersection(pl8) == [Line3D(Point3D(0, 0, 0), Point3D(12, 0, 0))]

    # 测试平面与另一个平面的交点，预期返回一个线对象列表
    assert pl10.intersection(pl11) == [Line3D(Point3D(0, 0, 1), Point3D(0, 2, 1))]

    # 测试平面与另一个平面的交点，预期返回一个线对象列表
    assert pl4.intersection(pl8) == [Line3D(Point3D(0, 0, 0), Point3D(1, -1, 0))]

    # 测试平面与另一个平面的交点，预期返回空列表
    assert pl11.intersection(pl8) == []
    # 断言检查两个平面 pl9 和 pl11 的交点是否为给定的线段列表
    assert pl9.intersection(pl11) == [Line3D(Point3D(0, 0, 1), Point3D(12, 0, 1))]
    
    # 断言检查两个平面 pl9 和 pl4 的交点是否为给定的线段列表
    assert pl9.intersection(pl4) == [Line3D(Point3D(0, 0, 0), Point3D(12, 0, -12))]
    
    # 断言检查平面 pl3 的随机点是否在平面内
    assert pl3.random_point() in pl3
    
    # 断言检查设置了种子的情况下，平面 pl3 的随机点是否在平面内
    assert pl3.random_point(seed=1) in pl3

    # 使用 equals 方法测试几何实体
    assert pl4.intersection(pl4.p1)[0].equals(pl4.p1)
    
    # 断言检查平面 pl3 和 pl6 的交点是否为给定的线段，且它等于预期的线段
    assert pl3.intersection(pl6)[0].equals(Line3D(Point3D(8, 4, 0), Point3D(2, 4, 6)))
    
    # 创建新的平面 pl8，基于给定的点和法向量
    pl8 = Plane((1, 2, 0), normal_vector=(0, 0, 1))
    
    # 断言检查平面 pl8 和给定的线段的交点是否为给定的线段，且它等于预期的线段
    assert pl8.intersection(Line3D(p1, (1, 12, 0)))[0].equals(Line((0, 0, 0), (0.1, 1.2, 0)))
    
    # 断言检查平面 pl8 和给定的射线的交点是否为给定的射线，且它等于预期的射线
    assert pl8.intersection(Ray3D(p1, (1, 12, 0)))[0].equals(Ray((0, 0, 0), (1, 12, 0)))
    
    # 断言检查平面 pl8 和给定的线段的交点是否为给定的线段，且它等于预期的线段
    assert pl8.intersection(Segment3D(p1, (21, 1, 0)))[0].equals(Segment3D(p1, (21, 1, 0)))
    
    # 断言检查平面 pl8 和给定的平面的交点是否为给定的平面，且它等于预期的平面
    assert pl8.intersection(Plane(p1, normal_vector=(0, 0, 112)))[0].equals(pl8)
    
    # 断言检查平面 pl8 和给定的平面的交点是否为给定的线段，且它等于预期的线段
    assert pl8.intersection(Plane(p1, normal_vector=(0, 12, 0)))[0].equals(
        Line3D(p1, direction_ratio=(112 * pi, 0, 0)))
    
    # 断言检查平面 pl8 和给定的平面的交点是否为给定的线段，且它等于预期的线段
    assert pl8.intersection(Plane(p1, normal_vector=(11, 0, 1)))[0].equals(
        Line3D(p1, direction_ratio=(0, -11, 0)))
    
    # 断言检查平面 pl8 和给定的平面的交点是否为给定的线段，且它等于预期的线段
    assert pl8.intersection(Plane(p1, normal_vector=(1, 0, 11)))[0].equals(
        Line3D(p1, direction_ratio=(0, 11, 0)))
    
    # 断言检查平面 pl8 和给定的平面的交点是否为给定的线段，且它等于预期的线段
    assert pl8.intersection(Plane(p1, normal_vector=(-1, -1, -11)))[0].equals(
        Line3D(p1, direction_ratio=(1, -1, 0)))
    
    # 再次断言检查平面 pl3 的随机点是否在平面内
    assert pl3.random_point() in pl3
    
    # 断言检查平面 pl8 和给定的射线是否没有交点
    assert len(pl8.intersection(Ray3D(Point3D(0, 2, 3), Point3D(1, 0, 3)))) == 0
    
    # 断言检查平面 pl6 和自身的交点是否为给定的平面，且它等于预期的平面
    assert pl6.intersection(pl6)[0].equals(pl6)
    
    # 断言检查平面 pl8 和给定的平面是否不相等
    assert pl8.equals(Plane(p1, normal_vector=(0, 12, 0))) is False
    
    # 断言检查平面 pl8 和自身是否相等
    assert pl8.equals(pl8)
    
    # 断言检查平面 pl8 和给定的平面是否相等
    assert pl8.equals(Plane(p1, normal_vector=(0, 0, -12)))
    
    # 断言检查平面 pl8 和给定的平面是否相等
    assert pl8.equals(Plane(p1, normal_vector=(0, 0, -12*sqrt(3))))
    
    # 断言检查平面 pl8 和给定的点是否不相等
    assert pl8.equals(p1) is False
    
    # Issue 8570 的测试情况下，创建线 l2 和平面 p2
    l2 = Line3D(Point3D(Rational(50000004459633, 5000000000000),
                        Rational(-891926590718643, 1000000000000000),
                        Rational(231800966893633, 100000000000000)),
                Point3D(Rational(50000004459633, 50000000000000),
                        Rational(-222981647679771, 250000000000000),
                        Rational(231800966893633, 100000000000000)))
    
    p2 = Plane(Point3D(Rational(402775636372767, 100000000000000),
                       Rational(-97224357654973, 100000000000000),
                       Rational(216793600814789, 100000000000000)),
               (-S('9.00000087501922'), -S('4.81170658872543e-13'),
                S('0.0')))
    
    # 断言检查平面 p2 和线 l2 的交点的数值近似是否等于预期的列表
    assert str([i.n(2) for i in p2.intersection(l2)]) == \
           '[Point3D(4.0, -0.89, 2.3)]'
# 定义一个测试函数，用于测试平面对象的维度归一化方法
def test_dimension_normalization():
    # 创建一个平面对象 A，指定其点和法向量
    A = Plane(Point3D(1, 1, 2), normal_vector=(1, 1, 1))
    # 创建一个点 b
    b = Point(1, 1)
    # 断言：A 对 b 的投影结果应为 Rational(5, 3), Rational(5, 3), Rational(2, 3) 所表示的点
    assert A.projection(b) == Point(Rational(5, 3), Rational(5, 3), Rational(2, 3))

    # 创建两个点 a, b
    a, b = Point(0, 0), Point3D(0, 1)
    # 创建一个法向量 Z
    Z = (0, 0, 1)
    # 创建一个平面 p，指定其点 a 和法向量 Z
    p = Plane(a, normal_vector=Z)
    # 断言：p 对 (a, b) 的垂直平面应为以 Point3D(0, 0, 0) 为点，(1, 0, 0) 为法向量的平面
    assert p.perpendicular_plane(a, b) == Plane(Point3D(0, 0, 0), (1, 0, 0))
    
    # 断言：创建一个平面对象，其通过三个点 (1, 2, 1), (2, 1, 0), (3, 1, 2) 构成
    assert Plane((1, 2, 1), (2, 1, 0), (3, 1, 2)).intersection((2, 1)) == [Point(2, 1, 0)]


# 定义一个测试函数，用于测试平面对象的参数数值方法
def test_parameter_value():
    # 创建符号变量 t, u, v
    t, u, v = symbols("t, u v")
    # 创建三个点 p1, p2, p3
    p1, p2, p3 = Point(0, 0, 0), Point(0, 0, 1), Point(0, 1, 0)
    # 创建一个平面对象 p，通过 p1, p2, p3 构成
    p = Plane(p1, p2, p3)
    # 断言：p 对 (0, -3, 2) 的参数 t 的数值应为 {t: asin(2*sqrt(13)/13)}
    assert p.parameter_value((0, -3, 2), t) == {t: asin(2*sqrt(13)/13)}
    # 断言：p 对 (0, -3, 2) 的参数 u, v 的数值应为 {u: 3, v: 2}
    assert p.parameter_value((0, -3, 2), u, v) == {u: 3, v: 2}
    # 断言：p 对 p1 的参数 t 的数值应为 p1 自身
    assert p.parameter_value(p1, t) == p1
    # 断言：调用参数值方法时，对于不在平面上的点 (1, 0, 0)，应引发 ValueError 异常
    raises(ValueError, lambda: p.parameter_value((1, 0, 0), t))
    # 断言：调用参数值方法时，对于不是点的参数，应引发 ValueError 异常
    raises(ValueError, lambda: p.parameter_value(Line(Point(0, 0), Point(1, 1)), t))
    # 断言：调用参数值方法时，提供了额外的参数 1，应引发 ValueError 异常
    raises(ValueError, lambda: p.parameter_value((0, -3, 2), t, 1))
```