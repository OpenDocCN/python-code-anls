# `D:\src\scipysrc\sympy\sympy\vector\tests\test_integrals.py`

```
# 导入符号计算库中的特定模块和函数
from sympy.core.numbers import pi
from sympy.core.singleton import S
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.testing.pytest import raises
from sympy.vector.coordsysrect import CoordSys3D
from sympy.vector.integrals import ParametricIntegral, vector_integrate
from sympy.vector.parametricregion import ParametricRegion
from sympy.vector.implicitregion import ImplicitRegion
from sympy.abc import x, y, z, u, v, r, t, theta, phi
from sympy.geometry import Point, Segment, Curve, Circle, Polygon, Plane

# 创建一个三维坐标系对象C
C = CoordSys3D('C')

# 定义测试函数，测试参数化线积分
def test_parametric_lineintegrals():
    # 定义半圆的参数化区域
    halfcircle = ParametricRegion((4*cos(theta), 4*sin(theta)), (theta, -pi/2, pi/2))
    # 断言计算参数化积分，验证结果为8192/5
    assert ParametricIntegral(C.x*C.y**4, halfcircle) == S(8192)/5

    # 定义曲线的参数化区域
    curve = ParametricRegion((t, t**2, t**3), (t, 0, 1))
    # 定义场field1
    field1 = 8*C.x**2*C.y*C.z*C.i + 5*C.z*C.j - 4*C.x*C.y*C.k
    # 断言计算参数化积分，验证结果为1
    assert ParametricIntegral(field1, curve) == 1

    # 定义直线的参数化区域
    line = ParametricRegion((4*t - 1, 2 - 2*t, t), (t, 0, 1))
    # 断言计算参数化积分，验证结果为3
    assert ParametricIntegral(C.x*C.z*C.i - C.y*C.z*C.k, line) == 3

    # 断言计算参数化积分，验证结果为8
    assert ParametricIntegral(4*C.x**3, ParametricRegion((1, t), (t, 0, 2))) == 8

    # 定义螺旋线的参数化区域
    helix = ParametricRegion((cos(t), sin(t), 3*t), (t, 0, 4*pi))
    # 断言计算参数化积分，验证结果为-3*sqrt(10)*pi
    assert ParametricIntegral(C.x*C.y*C.z, helix) == -3*sqrt(10)*pi

    # 定义场field2
    field2 = C.y*C.i + C.z*C.j + C.z*C.k
    # 断言计算参数化积分，验证结果为-5*pi/2 + pi**4/2
    assert ParametricIntegral(field2, ParametricRegion((cos(t), sin(t), t**2), (t, 0, pi))) == -5*pi/2 + pi**4/2

# 定义测试函数，测试参数化表面积分
def test_parametric_surfaceintegrals():

    # 定义半球面的参数化区域
    semisphere = ParametricRegion((2*sin(phi)*cos(theta), 2*sin(phi)*sin(theta), 2*cos(phi)),\
                            (theta, 0, 2*pi), (phi, 0, pi/2))
    # 断言计算参数化积分，验证结果为8*pi
    assert ParametricIntegral(C.z, semisphere) == 8*pi

    # 定义圆柱体的参数化区域
    cylinder = ParametricRegion((sqrt(3)*cos(theta), sqrt(3)*sin(theta), z), (z, 0, 6), (theta, 0, 2*pi))
    # 断言计算参数化积分，验证结果为0
    assert ParametricIntegral(C.y, cylinder) == 0

    # 定义圆锥体的参数化区域
    cone = ParametricRegion((v*cos(u), v*sin(u), v), (u, 0, 2*pi), (v, 0, 1))
    # 断言计算参数化积分，验证结果为pi/3
    assert ParametricIntegral(C.x*C.i + C.y*C.j + C.z**4*C.k, cone) == pi/3

    # 定义三角形区域的参数化区域
    triangle1 = ParametricRegion((x, y), (x, 0, 2), (y, 0, 10 - 5*x))
    triangle2 = ParametricRegion((x, y), (y, 0, 10 - 5*x), (x, 0, 2))
    # 断言两个三角形区域的参数化积分相等
    assert ParametricIntegral(-15.6*C.y*C.k, triangle1) == ParametricIntegral(-15.6*C.y*C.k, triangle2)
    # 断言计算参数化积分，验证结果为10*C.z
    assert ParametricIntegral(C.z, triangle1) == 10*C.z

# 定义测试函数，测试参数化体积积分
def test_parametric_volumeintegrals():

    # 定义立方体的参数化区域
    cube = ParametricRegion((x, y, z), (x, 0, 1), (y, 0, 1), (z, 0, 1))
    # 断言计算参数化积分，验证结果为1
    assert ParametricIntegral(1, cube) == 1

    # 定义实心球体的参数化区域，考虑两种参数化次序
    solidsphere1 = ParametricRegion((r*sin(phi)*cos(theta), r*sin(phi)*sin(theta), r*cos(phi)),\
                            (r, 0, 2), (theta, 0, 2*pi), (phi, 0, pi))
    solidsphere2 = ParametricRegion((r*sin(phi)*cos(theta), r*sin(phi)*sin(theta), r*cos(phi)),\
                            (r, 0, 2), (phi, 0, pi), (theta, 0, 2*pi))
    # 断言计算参数化积分，验证结果分别为-256*pi/15和256*pi/15
    assert ParametricIntegral(C.x**2 + C.y**2, solidsphere1) == -256*pi/15
    assert ParametricIntegral(C.x**2 + C.y**2, solidsphere2) == 256*pi/15
    # 创建参数化区域对象，表示一个平面以下的区域，使用参数 (x, y, z)
    region_under_plane1 = ParametricRegion((x, y, z), (x, 0, 3), (y, 0, -2*x/3 + 2),\
                                    (z, 0, 6 - 2*x - 3*y))
    
    # 创建另一个参数化区域对象，表示另一个平面以下的区域，使用参数 (x, y, z)
    region_under_plane2 = ParametricRegion((x, y, z), (x, 0, 3), (z, 0, 6 - 2*x - 3*y),\
                                    (y, 0, -2*x/3 + 2))
    
    # 断言：两个参数化积分对象的结果相等，使用参数 (C.x*C.i + C.j - 100*C.k)，分别在上述两个区域中计算
    assert ParametricIntegral(C.x*C.i + C.j - 100*C.k, region_under_plane1) == \
        ParametricIntegral(C.x*C.i + C.j - 100*C.k, region_under_plane2)
    
    # 断言：参数化积分对象的结果等于 -9，使用参数 2*C.x，在第二个区域中计算
    assert ParametricIntegral(2*C.x, region_under_plane2) == -9
# 定义测试函数 test_vector_integrate，用于测试 vector_integrate 函数的不同输入情况
def test_vector_integrate():
    # 创建半圆盘 ParametricRegion 对象，其参数为 (r*cos(theta), r*sin(theta))，其中 r 范围为 -2 到 2，theta 范围为 0 到 pi
    halfdisc = ParametricRegion((r*cos(theta), r* sin(theta)), (r, -2, 2), (theta, 0, pi))
    # 断言向量积分 C.x**2 在 halfdisc 区域上的结果为 4*pi
    assert vector_integrate(C.x**2, halfdisc) == 4*pi
    # 断言向量积分 C.x 在 ParametricRegion((t, t**2), (t, 2, 3)) 区域上的结果为 -17*sqrt(17)/12 + 37*sqrt(37)/12
    assert vector_integrate(C.x, ParametricRegion((t, t**2), (t, 2, 3))) == -17*sqrt(17)/12 + 37*sqrt(37)/12

    # 断言向量积分 C.y**3*C.z 在区间 (C.x, 0, 3), (C.y, -1, 4) 上的结果为 765*C.z/4
    assert vector_integrate(C.y**3*C.z, (C.x, 0, 3), (C.y, -1, 4)) == 765*C.z/4

    # 创建线段对象 s1，起点为 (0, 0)，终点为 (0, 1)
    s1 = Segment(Point(0, 0), Point(0, 1))
    # 断言向量积分 -15*C.y 在线段 s1 上的结果为 -15/2
    assert vector_integrate(-15*C.y, s1) == S(-15)/2
    # 创建线段对象 s2，起点为 (4, 3, 9)，终点为 (1, 1, 7)
    s2 = Segment(Point(4, 3, 9), Point(1, 1, 7))
    # 断言向量积分 C.y*C.i 在线段 s2 上的结果为 -6
    assert vector_integrate(C.y*C.i, s2) == -6

    # 创建曲线对象 curve，其参数为 ((sin(t), cos(t)), (t, 0, 2))
    curve = Curve((sin(t), cos(t)), (t, 0, 2))
    # 断言向量积分 5*C.z 在曲线 curve 上的结果为 10*C.z
    assert vector_integrate(5*C.z, curve) == 10*C.z

    # 创建圆对象 c1，中心点为 (2, 3)，半径为 6
    c1 = Circle(Point(2, 3), 6)
    # 断言向量积分 C.x*C.y 在圆 c1 上的结果为 72*pi
    assert vector_integrate(C.x*C.y, c1) == 72*pi
    # 创建圆对象 c2，其定义通过三个点 (0, 0), (1, 1), (1, 0)
    c2 = Circle(Point(0, 0), Point(1, 1), Point(1, 0))
    # 断言向量积分 1 在圆 c2 上的结果为 c2 的周长
    assert vector_integrate(1, c2) == c2.circumference

    # 创建多边形对象 triangle，顶点为 (0, 0), (1, 0), (1, 1)
    triangle = Polygon((0, 0), (1, 0), (1, 1))
    # 断言向量积分 C.x*C.i - 14*C.y*C.j 在三角形 triangle 上的结果为 0
    assert vector_integrate(C.x*C.i - 14*C.y*C.j, triangle) == 0
    # 定义四个顶点 p1, p2, p3, p4 组成的多边形 poly
    p1, p2, p3, p4 = [(0, 0), (1, 0), (5, 1), (0, 1)]
    poly = Polygon(p1, p2, p3, p4)
    # 断言向量积分 -23*C.z 在多边形 poly 上的结果为 -161*C.z - 23*sqrt(17)*C.z
    assert vector_integrate(-23*C.z, poly) == -161*C.z - 23*sqrt(17)*C.z

    # 创建点对象 point，坐标为 (2, 3)
    point = Point(2, 3)
    # 断言向量积分 C.i*C.y - C.z 在点 point 上的结果为 ParametricIntegral(C.y*C.i, ParametricRegion((2, 3)))
    assert vector_integrate(C.i*C.y - C.z, point) == ParametricIntegral(C.y*C.i, ParametricRegion((2, 3)))

    # 创建隐式区域 c3，其定义为 x**2 + y**2 - 4
    c3 = ImplicitRegion((x, y), x**2 + y**2 - 4)
    # 断言向量积分 45 在隐式区域 c3 上的结果为 180*pi
    assert vector_integrate(45, c3) == 180*pi
    # 创建隐式区域 c4，其定义为 (x - 3)**2 + (y - 4)**2 - 9
    c4 = ImplicitRegion((x, y), (x - 3)**2 + (y - 4)**2 - 9)
    # 断言向量积分 1 在隐式区域 c4 上的结果为 6*pi
    assert vector_integrate(1, c4) == 6*pi

    # 创建平面对象 pl，通过三个点 (1, 1, 1), (2, 3, 4), (2, 2, 2) 定义
    pl = Plane(Point(1, 1, 1), Point(2, 3, 4), Point(2, 2, 2))
    # 使用 lambda 函数断言调用 vector_integrate(C.x*C.z*C.i + C.k, pl) 抛出 ValueError 异常
    raises(ValueError, lambda: vector_integrate(C.x*C.z*C.i + C.k, pl))
```