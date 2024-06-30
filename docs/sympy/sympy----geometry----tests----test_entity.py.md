# `D:\src\scipysrc\sympy\sympy\geometry\tests\test_entity.py`

```
from sympy.core.numbers import (Rational, pi)  # 导入 Rational 和 pi 两个数学常数
from sympy.core.singleton import S  # 导入符号 S，用于表示特殊值
from sympy.core.symbol import Symbol  # 导入符号类 Symbol
from sympy.geometry import (Circle, Ellipse, Point, Line, Parabola,  # 导入几何图形类
    Polygon, Ray, RegularPolygon, Segment, Triangle, Plane, Curve)
from sympy.geometry.entity import scale, GeometryEntity  # 导入缩放函数和几何实体类
from sympy.testing.pytest import raises  # 导入 pytest 的 raises 断言


def test_entity():
    x = Symbol('x', real=True)  # 创建一个实数符号 x
    y = Symbol('y', real=True)  # 创建一个实数符号 y

    assert GeometryEntity(x, y) in GeometryEntity(x, y)  # 断言 (x, y) 几何实体在 (x, y) 几何实体中
    raises(NotImplementedError, lambda: Point(0, 0) in GeometryEntity(x, y))  # 断言尝试在 (x, y) 几何实体中查找 (0, 0) 点时会抛出 NotImplementedError

    assert GeometryEntity(x, y) == GeometryEntity(x, y)  # 断言 (x, y) 几何实体等于 (x, y) 几何实体
    assert GeometryEntity(x, y).equals(GeometryEntity(x, y))  # 断言 (x, y) 几何实体等于 (x, y) 几何实体


    c = Circle((0, 0), 5)  # 创建一个圆 c，中心在 (0, 0)，半径为 5
    assert GeometryEntity.encloses(c, Point(0, 0))  # 断言圆 c 包含点 (0, 0)
    assert GeometryEntity.encloses(c, Segment((0, 0), (1, 1)))  # 断言圆 c 包含线段 (0, 0) 到 (1, 1)
    assert GeometryEntity.encloses(c, Line((0, 0), (1, 1))) is False  # 断言圆 c 不包含直线 (0, 0) 到 (1, 1)
    assert GeometryEntity.encloses(c, Circle((0, 0), 4))  # 断言圆 c 包含圆心在 (0, 0)，半径为 4 的圆
    assert GeometryEntity.encloses(c, Polygon(Point(0, 0), Point(1, 0), Point(0, 1)))  # 断言圆 c 包含由点 (0, 0), (1, 0), (0, 1) 构成的多边形
    assert GeometryEntity.encloses(c, RegularPolygon(Point(8, 8), 1, 3)) is False  # 断言圆 c 不包含中心在 (8, 8)，边数为 3 的正多边形


def test_svg():
    a = Symbol('a')  # 创建符号 a
    b = Symbol('b')  # 创建符号 b
    d = Symbol('d')  # 创建符号 d

    entity = Circle(Point(a, b), d)  # 创建一个圆实体，中心为 (a, b)，半径为 d
    assert entity._repr_svg_() is None  # 断言该实体的 SVG 表示为 None

    entity = Circle(Point(0, 0), S.Infinity)  # 创建一个半径为无穷大的圆实体，中心在 (0, 0)
    assert entity._repr_svg_() is None  # 断言该实体的 SVG 表示为 None


def test_subs():
    x = Symbol('x', real=True)  # 创建一个实数符号 x
    y = Symbol('y', real=True)  # 创建一个实数符号 y
    p = Point(x, 2)  # 创建一个点 p，x 坐标为 x，y 坐标为 2
    q = Point(1, 1)  # 创建一个点 q，坐标为 (1, 1)
    r = Point(3, 4)  # 创建一个点 r，坐标为 (3, 4)
    for o in [p,
              Segment(p, q),
              Ray(p, q),
              Line(p, q),
              Triangle(p, q, r),
              RegularPolygon(p, 3, 6),
              Polygon(p, q, r, Point(5, 4)),
              Circle(p, 3),
              Ellipse(p, 3, 4)]:
        assert 'y' in str(o.subs(x, y))  # 对于集合中的每个几何实体 o，断言在将 x 替换为 y 后，字符串表达式中包含 'y'
    assert p.subs({x: 1}) == Point(1, 2)  # 断言将点 p 中的 x 替换为 1 后得到点 (1, 2)
    assert Point(1, 2).subs(Point(1, 2), Point(3, 4)) == Point(3, 4)  # 断言将点 (1, 2) 替换为点 (3, 4) 后得到点 (3, 4)
    assert Point(1, 2).subs((1, 2), Point(3, 4)) == Point(3, 4)  # 断言将点 (1, 2) 替换为点 (3, 4) 后得到点 (3, 4)
    assert Point(1, 2).subs(Point(1, 2), Point(3, 4)) == Point(3, 4)  # 断言将点 (1, 2) 替换为点 (3, 4) 后得到点 (3, 4)
    assert Point(1, 2).subs({(1, 2)}) == Point(2, 2)  # 断言将点 (1, 2) 替换为 (1, 2) 后得到点 (2, 2)
    raises(ValueError, lambda: Point(1, 2).subs(1))  # 断言替换点 (1, 2) 中的一个非法值 1 会引发 ValueError
    raises(ValueError, lambda: Point(1, 1).subs((Point(1, 1), Point(1,
           2)), 1, 2))  # 断言替换点 (1, 1) 中的一个非法元组会引发 ValueError


def test_transform():
    assert scale(1, 2, (3, 4)).tolist() == \
        [[1, 0, 0], [0, 2, 0], [0, -4, 1]]  # 断言将原始坐标 (3, 4) 缩放为 (1, 2) 后得到的变换矩阵


def test_reflect_entity_overrides():
    x = Symbol('x', real=True)  # 创建一个实数符号 x
    y = Symbol('y', real=True)  # 创建一个实数符号 y
    b = Symbol('b')  # 创建一个符号 b
    m = Symbol('m')  # 创建一个符号 m
    l = Line((0, b), slope=m)  # 创建一条经过点 (0, b) 且斜率为 m 的直线 l
    p = Point(x, y)  # 创建一个点 p，坐标为 (x, y)
    r = p.reflect(l)  # 将点 p 关于直线 l 反射得到点 r
    c = Circle((x, y), 3)  # 创建一个圆 c，中心在 (x, y)，半径为 3
    cr = c.reflect(l)  # 将圆 c 关于直线 l 反射得到圆 cr
    assert cr == Circle(r, -3)  # 断言圆 cr 等于以点 r 为中心、半径为 -3 的圆
    assert c.area == -cr.area  # 断言圆 c 的面积等于圆 cr 的面积的相反数

    pent = RegularPolygon((1, 2), 1, 5)  # 创建一个中心在 (1, 2)，边长为 1，边数为 5 的正多边形 pent
    slope
    # 对每个五边形顶点进行反射，得到反射后的顶点列表 rvert
    rvert = [i.reflect(l) for i in pent.vertices]

    # 遍历原始五边形 rpent 的顶点
    for v in rpent.vertices:
        # 遍历 rvert 列表中的每个顶点
        for i in range(len(rvert)):
            # 获取当前的反射后的顶点 ri
            ri = rvert[i]
            # 如果 ri 等于 rpent 的当前顶点 v
            if ri.equals(v):
                # 从 rvert 中移除 ri
                rvert.remove(ri)
                # 跳出内层循环
                break

    # 断言 rvert 列表为空，即所有原始顶点都能在反射后的顶点中找到对应项
    assert not rvert

    # 断言原始五边形 pent 的面积与反射后五边形 rpent 的面积相等且为负数
    assert pent.area.equals(-rpent.area)
# 定义测试函数，用于测试几何图形类的 EvalfMixin 功能
def test_geometry_EvalfMixin():
    # 定义 π 的值为 x
    x = pi
    # 定义符号变量 t
    t = Symbol('t')
    # 遍历各种几何图形对象
    for g in [
            # 创建一个点对象，坐标为 (π, π)
            Point(x, x),
            # 创建一个平面对象，通过一个点 (0, π, 0) 和法向量 (0, 0, π) 定义
            Plane(Point(0, x, 0), (0, 0, x)),
            # 创建一个曲线对象，参数为 (π*t, π)，范围为 t 从 0 到 π
            Curve((x*t, x), (t, 0, x)),
            # 创建一个椭圆对象，中心为 (π, π)，长轴为 π，短轴为 -π
            Ellipse((x, x), x, -x),
            # 创建一个圆对象，中心为 (π, π)，半径为 π
            Circle((x, x), x),
            # 创建一条直线对象，通过两点 (0, π) 和 (π, 0) 定义
            Line((0, x), (x, 0)),
            # 创建一条线段对象，通过两点 (0, π) 和 (π, 0) 定义
            Segment((0, x), (x, 0)),
            # 创建一个射线对象，通过起点 (0, π) 和方向向量 (π, 0) 定义
            Ray((0, x), (x, 0)),
            # 创建一个抛物线对象，通过焦点 (0, π) 和直线 y = 0 的定义
            Parabola((0, x), Line((-x, 0), (x, 0))),
            # 创建一个多边形对象，顶点为 (0, 0), (0, π), (π, 0), (π, π)
            Polygon((0, 0), (0, x), (x, 0), (x, x)),
            # 创建一个正多边形对象，中心为 (0, π)，边长为 π，边数为 4，旋转角度为 π
            RegularPolygon((0, x), x, 4, x),
            # 创建一个三角形对象，顶点为 (0, 0), (π, 0), (π, π)
            Triangle((0, 0), (x, 0), (x, x)),
            ]:
        # 断言：将每个几何对象转换成字符串形式并将 π 替换为 3.1，与保留两位小数后的字符串形式进行比较
        assert str(g).replace('pi', '3.1') == str(g.n(2))
```