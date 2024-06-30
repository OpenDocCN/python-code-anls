# `D:\src\scipysrc\sympy\sympy\vector\tests\test_parametricregion.py`

```
from sympy.core.numbers import pi
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.vector.coordsysrect import CoordSys3D
from sympy.vector.parametricregion import ParametricRegion, parametric_region_list
from sympy.geometry import Point, Segment, Curve, Ellipse, Line, Parabola, Polygon
from sympy.testing.pytest import raises
from sympy.abc import a, b, r, t, x, y, z, theta, phi

# 定义三维笛卡尔坐标系
C = CoordSys3D('C')

def test_ParametricRegion():

    # 创建一个零维参数区域，表示一个点
    point = ParametricRegion((3, 4))
    assert point.definition == (3, 4)
    assert point.parameters == ()
    assert point.limits == {}
    assert point.dimensions == 0

    # 创建一维参数区域，表示直线 x = y
    line_xy = ParametricRegion((y, y), (y, 1, 5))
    assert line_xy.definition == (y, y)
    assert line_xy.parameters == (y,)
    assert line_xy.dimensions == 1

    # 创建一维参数区域，表示直线 y = z
    line_yz = ParametricRegion((x, t, t), x, (t, 1, 2))
    assert line_yz.definition == (x, t, t)
    assert line_yz.parameters == (x, t)
    assert line_yz.limits == {t: (1, 2)}
    assert line_yz.dimensions == 1

    # 创建二维参数区域，参数为 a 和 b
    p1 = ParametricRegion((9*a, -16*b), (a, 0, 2), (b, -1, 5))
    assert p1.definition == (9*a, -16*b)
    assert p1.parameters == (a, b)
    assert p1.limits == {a: (0, 2), b: (-1, 5)}
    assert p1.dimensions == 2

    # 创建零维参数区域，表示一个点
    p2 = ParametricRegion((t, t**3), t)
    assert p2.parameters == (t,)
    assert p2.limits == {}
    assert p2.dimensions == 0

    # 创建一维参数区域，表示圆的边界
    circle = ParametricRegion((r*cos(theta), r*sin(theta)), r, (theta, 0, 2*pi))
    assert circle.definition == (r*cos(theta), r*sin(theta))
    assert circle.dimensions == 1

    # 创建二维参数区域，表示半圆盘的边界
    halfdisc = ParametricRegion((r*cos(theta), r*sin(theta)), (r, -2, 2), (theta, 0, pi))
    assert halfdisc.definition == (r*cos(theta), r*sin(theta))
    assert halfdisc.parameters == (r, theta)
    assert halfdisc.limits == {r: (-2, 2), theta: (0, pi)}
    assert halfdisc.dimensions == 2

    # 创建一维参数区域，表示椭圆的边界
    ellipse = ParametricRegion((a*cos(t), b*sin(t)), (t, 0, 8))
    assert ellipse.parameters == (t,)
    assert ellipse.limits == {t: (0, 8)}
    assert ellipse.dimensions == 1

    # 创建三维参数区域，表示圆柱体的边界
    cylinder = ParametricRegion((r*cos(theta), r*sin(theta), z), (r, 0, 1), (theta, 0, 2*pi), (z, 0, 4))
    assert cylinder.parameters == (r, theta, z)
    assert cylinder.dimensions == 3

    # 创建二维参数区域，表示球体的边界
    sphere = ParametricRegion((r*sin(phi)*cos(theta), r*sin(phi)*sin(theta), r*cos(phi)),
                                r, (theta, 0, 2*pi), (phi, 0, pi))
    assert sphere.definition == (r*sin(phi)*cos(theta), r*sin(phi)*sin(theta), r*cos(phi))
    assert sphere.parameters == (r, theta, phi)
    assert sphere.dimensions == 2

    # 检查是否会引发 ValueError 异常
    raises(ValueError, lambda: ParametricRegion((a*t**2, 2*a*t), (a, -2)))
    raises(ValueError, lambda: ParametricRegion((a, b), (a**2, sin(b)), (a, 2, 4, 6)))


def test_parametric_region_list():

    # 创建一个点对象
    point = Point(-5, 12)
    assert parametric_region_list(point) == [ParametricRegion((-5, 12))]

    # 创建一个椭圆对象
    e = Ellipse(Point(2, 8), 2, 6)
    # 断言：调用 parametric_region_list 函数并检查返回值是否与预期列表相等
    assert parametric_region_list(e, t) == [ParametricRegion((2*cos(t) + 2, 6*sin(t) + 8), (t, 0, 2*pi))]

    # 创建 Curve 对象 c，使用参数 t 的范围 (5, 3) 来定义参数化区域
    c = Curve((t, t**3), (t, 5, 3))
    # 断言：调用 parametric_region_list 函数并检查返回值是否包含预期的 ParametricRegion 对象
    assert parametric_region_list(c) == [ParametricRegion((t, t**3), (t, 5, 3))]

    # 创建 Segment 对象 s，使用参数 t 的范围 (0, 1) 来定义参数化区域
    s = Segment(Point(2, 11, -6), Point(0, 2, 5))
    # 断言：调用 parametric_region_list 函数并检查返回值是否包含预期的 ParametricRegion 对象
    assert parametric_region_list(s, t) == [ParametricRegion((2 - 2*t, 11 - 9*t, 11*t - 6), (t, 0, 1))]

    # 创建 Segment 对象 s1，使用参数 t 的范围 (0, 1) 来定义参数化区域
    s1 = Segment(Point(0, 0), (1, 0))
    # 断言：调用 parametric_region_list 函数并检查返回值是否包含预期的 ParametricRegion 对象
    assert parametric_region_list(s1, t) == [ParametricRegion((t, 0), (t, 0, 1))]

    # 创建 Segment 对象 s2，使用参数 t 的范围 (0, 1) 来定义参数化区域
    s2 = Segment(Point(1, 2, 3), Point(1, 2, 5))
    # 断言：调用 parametric_region_list 函数并检查返回值是否包含预期的 ParametricRegion 对象
    assert parametric_region_list(s2, t) == [ParametricRegion((1, 2, 2*t + 3), (t, 0, 1))]

    # 创建 Segment 对象 s3，不指定参数 t，表示 s3 是一个固定的点，无法参数化
    s3 = Segment(Point(12, 56), Point(12, 56))
    # 断言：调用 parametric_region_list 函数并检查返回值是否包含预期的 ParametricRegion 对象
    assert parametric_region_list(s3) == [ParametricRegion((12, 56))]

    # 创建 Polygon 对象 poly，使用参数 t 的范围 (0, 1) 来定义多个参数化区域
    poly = Polygon((1,3), (-3, 8), (2, 4))
    # 断言：调用 parametric_region_list 函数并检查返回值是否包含预期的 ParametricRegion 对象列表
    assert parametric_region_list(poly, t) == [
        ParametricRegion((1 - 4*t, 5*t + 3), (t, 0, 1)),
        ParametricRegion((5*t - 3, 8 - 4*t), (t, 0, 1)),
        ParametricRegion((2 - t, 4 - t), (t, 0, 1))
    ]

    # 创建 Parabola 对象 p1，由于 Parabola 类不能表示为参数化区域，预期抛出 ValueError 异常
    p1 = Parabola(Point(0, 0), Line(Point(5, 8), Point(7,8)))
    # 断言：调用 parametric_region_list 函数并期望抛出 ValueError 异常
    raises(ValueError, lambda: parametric_region_list(p1))
```