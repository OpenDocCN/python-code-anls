# `D:\src\scipysrc\sympy\sympy\geometry\tests\test_geometrysets.py`

```
from sympy.core.numbers import Rational  # 导入有理数类
from sympy.core.singleton import S  # 导入单例 S
from sympy.geometry import Circle, Line, Point, Polygon, Segment  # 导入几何图形相关类
from sympy.sets import FiniteSet, Union, Intersection, EmptySet  # 导入集合相关类


def test_booleans():
    """ 测试基本的并集和交集 """

    half = S.Half  # 定义 S.Half 为 half

    # 定义几个点
    p1, p2, p3, p4 = map(Point, [(0, 0), (1, 0), (5, 1), (0, 1)])
    p5, p6, p7 = map(Point, [(3, 2), (1, -1), (0, 2)])
    
    # 定义几条线
    l1 = Line(Point(0,0), Point(1,1))
    l2 = Line(Point(half, half), Point(5,5))
    l3 = Line(p2, p3)
    l4 = Line(p3, p4)
    
    # 定义几个多边形
    poly1 = Polygon(p1, p2, p3, p4)
    poly2 = Polygon(p5, p6, p7)
    poly3 = Polygon(p1, p2, p5)
    
    # 断言语句，检查交集和并集的计算结果
    assert Union(l1, l2).equals(l1)  # 检查 l1 和 l2 的并集是否等于 l1
    assert Intersection(l1, l2).equals(l1)  # 检查 l1 和 l2 的交集是否等于 l1
    assert Intersection(l1, l4) == FiniteSet(Point(1,1))  # 检查 l1 和 l4 的交集是否为 Point(1,1)
    assert Intersection(Union(l1, l4), l3) == FiniteSet(Point(Rational(-1, 3), Rational(-1, 3)), Point(5, 1))  # 检查 Union(l1, l4) 和 l3 的交集
    assert Intersection(l1, FiniteSet(Point(7,-7))) == EmptySet  # 检查 l1 和 FiniteSet(Point(7,-7)) 的交集是否为空集
    assert Intersection(Circle(Point(0,0), 3), Line(p1,p2)) == FiniteSet(Point(-3,0), Point(3,0))  # 检查 Circle(Point(0,0), 3) 和 Line(p1,p2) 的交集
    assert Intersection(l1, FiniteSet(p1)) == FiniteSet(p1)  # 检查 l1 和 FiniteSet(p1) 的交集是否为 FiniteSet(p1)
    assert Union(l1, FiniteSet(p1)) == l1  # 检查 l1 和 FiniteSet(p1) 的并集是否等于 l1

    # 定义一个有限集合 fs
    fs = FiniteSet(Point(Rational(1, 3), 1), Point(Rational(2, 3), 0), Point(Rational(9, 5), Rational(1, 5)), Point(Rational(7, 3), 1))
    
    # 检查多边形的交集
    assert Intersection(poly1, poly2) == fs
    
    # 检查多边形和其子集的并集
    assert Union(poly1, poly2, fs) == Union(poly1, poly2)
    
    # 检查与不是子集的有限集合的并集
    assert Union(poly1, FiniteSet(Point(0,0), Point(3,5))) == Union(poly1, FiniteSet(Point(3,5)))
    
    # 检查共享边的两个多边形的交集
    assert Intersection(poly1, poly3) == Union(FiniteSet(Point(Rational(3, 2), 1), Point(2, 1)), Segment(Point(0, 0), Point(1, 0)))
```