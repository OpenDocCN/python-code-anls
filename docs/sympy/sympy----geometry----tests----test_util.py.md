# `D:\src\scipysrc\sympy\sympy\geometry\tests\test_util.py`

```
# 导入 pytest 模块，用于单元测试
import pytest
# 导入 SymPy 中的相关模块和类
from sympy.core.numbers import Float
from sympy.core.function import (Derivative, Function)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions import exp, cos, sin, tan, cosh, sinh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.geometry import Point, Point2D, Line, Polygon, Segment, convex_hull,\
    intersection, centroid, Point3D, Line3D, Ray, Ellipse
# 导入 SymPy 几何模块中的实用函数
from sympy.geometry.util import idiff, closest_points, farthest_points, _ordered_points, are_coplanar
# 导入 SymPy 求解器模块中的 solve 函数
from sympy.solvers.solvers import solve
# 导入 SymPy 测试模块中的 raises 函数
from sympy.testing.pytest import raises


# 定义单元测试函数 test_idiff
def test_idiff():
    # 定义符号变量 x, y, t，并声明为实数
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    t = Symbol('t', real=True)
    # 定义函数对象 f 和 g
    f = Function('f')
    g = Function('g')
    # 定义圆形方程 circ: x^2 + y^2 - 4
    circ = x**2 + y**2 - 4
    # 计算 idiff(circ, y, x, 3) 的预期结果，存入 ans
    ans = -3*x*(x**2/y**2 + 1)/y**3
    # 断言 idiff(circ, y, x, 3) 等于 ans
    assert ans == idiff(circ, y, x, 3), idiff(circ, y, x, 3)
    # 断言 idiff(circ, [y], x, 3) 等于 ans
    assert ans == idiff(circ, [y], x, 3)
    # 断言 idiff(circ, y, x, 3) 等于 ans
    assert idiff(circ, y, x, 3) == ans
    # 计算 ans 替换 y 后的显式表达式，存入 explicit
    explicit = 12*x/sqrt(-x**2 + 4)**5
    # 断言 ans 在 y = solve(circ, y)[0] 时等于 explicit
    assert ans.subs(y, solve(circ, y)[0]).equals(explicit)
    # 断言 solve(circ, y) 中每个解的三阶导数关于 x 的结果均等于 explicit
    assert True in [sol.diff(x, 3).equals(explicit) for sol in solve(circ, y)]
    # 断言 idiff(x + t + y, [y, t], x) 的结果为 -Derivative(t, x) - 1
    assert idiff(x + t + y, [y, t], x) == -Derivative(t, x) - 1
    # 断言 idiff(f(x) * exp(f(x)) - x * exp(x), f(x), x) 的结果
    assert idiff(f(x) * exp(f(x)) - x * exp(x), f(x), x) == (x + 1)*exp(x)*exp(-f(x))/(f(x) + 1)
    # 断言 idiff(f(x) - y * exp(x), [f(x), y], x) 的结果
    assert idiff(f(x) - y * exp(x), [f(x), y], x) == (y + Derivative(y, x))*exp(x)
    # 断言 idiff(f(x) - y * exp(x), [y, f(x)], x) 的结果
    assert idiff(f(x) - y * exp(x), [y, f(x)], x) == -y + Derivative(f(x), x)*exp(-x)
    # 断言 idiff(f(x) - g(x), [f(x), g(x)], x) 的结果
    assert idiff(f(x) - g(x), [f(x), g(x)], x) == Derivative(g(x), x)
    # 断言 idiff(fxy, y, x) 的结果，其中 fxy 是复杂表达式
    fxy = y - (-10*(-sin(x) + 1/x)**2 + tan(x)**2 + 2*cosh(x/10))
    assert idiff(fxy, y, x) == -20*sin(x)*cos(x) + 2*tan(x)**3 + \
        2*tan(x) + sinh(x/10)/5 + 20*cos(x)/x - 20*sin(x)/x**2 + 20/x**3


# 定义单元测试函数 test_intersection
def test_intersection():
    # 断言 intersection(Point(0, 0)) 的结果为空列表
    assert intersection(Point(0, 0)) == []
    # 使用 lambda 函数断言 intersection(Point(0, 0), 3) 抛出 TypeError 异常
    raises(TypeError, lambda: intersection(Point(0, 0), 3))
    # 断言 intersection 的结果与给定线段相交的点和线段的列表
    assert intersection(
            Segment((0, 0), (2, 0)),
            Segment((-1, 0), (1, 0)),
            Line((0, 0), (0, 1)), pairwise=True) == [
        Point(0, 0), Segment((0, 0), (1, 0))]
    # 断言 intersection 的结果与给定线段、线、与线段相交的点和线段的列表
    assert intersection(
            Line((0, 0), (0, 1)),
            Segment((0, 0), (2, 0)),
            Segment((-1, 0), (1, 0)), pairwise=True) == [
        Point(0, 0), Segment((0, 0), (1, 0))]
    # 断言 intersection 的结果与给定线段、线、直线相交的点和线段的列表
    assert intersection(
            Line((0, 0), (0, 1)),
            Segment((0, 0), (2, 0)),
            Segment((-1, 0), (1, 0)),
            Line((0, 0), slope=1), pairwise=True) == [
        Point(0, 0), Segment((0, 0), (1, 0))]
    # 定义 Ray 对象，与给定椭圆相交，并断言交点的坐标近似于预期值
    R = 4.0
    c = intersection(
            Ray(Point2D(0.001, -1),
            Point2D(0.0008, -1.7)),
            Ellipse(center=Point2D(0, 0), hradius=R, vradius=2.0), pairwise=True)[0].coordinates
    assert c == pytest.approx(
            Point2D(0.000714285723396502, -1.99999996811224, evaluate=False).coordinates)
    # 检查此测试对较低精度参数响应
    # 创建一个浮点数对象 R，精度为 4 位小数，总位数为 5
    R = Float(4, 5)
    # 使用 intersection 函数计算射线和椭圆的交点
    c2 = intersection(
            # 创建起点 (0.001, -1) 和终点 (0.0008, -1.7) 的射线
            Ray(Point2D(0.001, -1),
            Point2D(0.0008, -1.7)),
            # 创建中心在 (0, 0)，水平半径为 R，垂直半径为 2.0 的椭圆
            Ellipse(center=Point2D(0, 0), hradius=R, vradius=2.0), pairwise=True)[0].coordinates
    # 使用 pytest 的 approx 函数断言 c2 等于给定的点坐标，evaluate=False 表示不计算精度
    assert c2 == pytest.approx(
            Point2D(0.000714285723396502, -1.99999996811224, evaluate=False).coordinates)
    # 断言 c[0]._prec 等于 53，_prec 表示精度属性
    assert c[0]._prec == 53
    # 断言 c2[0]._prec 等于 20，_prec 表示精度属性
    assert c2[0]._prec == 20
# 定义一个测试函数，用于测试凸包算法的正确性
def test_convex_hull():
    # 确保当传入参数类型不正确时，会引发 TypeError 异常
    raises(TypeError, lambda: convex_hull(Point(0, 0), 3))
    
    # 准备测试数据点集
    points = [(1, -1), (1, -2), (3, -1), (-5, -2), (15, -4)]
    
    # 断言调用凸包算法函数时，使用特定参数返回预期结果
    assert convex_hull(*points, **{"polygon": False}) == (
        [Point2D(-5, -2), Point2D(1, -1), Point2D(3, -1), Point2D(15, -4)],
        [Point2D(-5, -2), Point2D(15, -4)])
        

# 定义一个测试函数，用于测试质心（centroid）计算函数的正确性
def test_centroid():
    # 创建一个多边形对象 p
    p = Polygon((0, 0), (10, 0), (10, 10))
    # 将 p 平移后得到 q
    q = p.translate(0, 20)
    
    # 断言调用质心计算函数时，使用特定参数返回预期结果
    assert centroid(p, q) == Point(20, 40)/3
    
    # 创建线段对象 p 和 q
    p = Segment((0, 0), (2, 0))
    q = Segment((0, 0), (2, 2))
    
    # 断言调用质心计算函数时，使用特定参数返回预期结果
    assert centroid(p, q) == Point(1, -sqrt(2) + 2)
    
    # 断言调用质心计算函数时，使用特定参数返回预期结果
    assert centroid(Point(0, 0), Point(2, 0)) == Point(2, 0)/2
    
    # 断言调用质心计算函数时，使用特定参数返回预期结果
    assert centroid(Point(0, 0), Point(0, 0), Point(2, 0)) == Point(2, 0)/3


# 定义一个测试函数，用于测试最远点和最近点计算函数的正确性
def test_farthest_points_closest_points():
    # 导入所需的模块和函数
    from sympy.core.random import randint
    from sympy.utilities.iterables import subsets
    
    # 针对最远点和最近点计算函数，分别进行测试
    for how in (min, max):
        if how == min:
            func = closest_points
        else:
            func = farthest_points
        
        # 断言调用最远点或最近点计算函数时，使用特定参数抛出 ValueError 异常
        raises(ValueError, lambda: func(Point2D(0, 0), Point2D(0, 0)))
        
        # 定义不同的测试数据集合
        p1 = [Point2D(0, 0), Point2D(3, 0), Point2D(1, 1)]
        p2 = [Point2D(0, 0), Point2D(3, 0), Point2D(2, 1)]
        p3 = [Point2D(0, 0), Point2D(3, 0), Point2D(1, 10)]
        p4 = [Point2D(0, 0), Point2D(3, 0), Point2D(4, 0)]
        p5 = [Point2D(0, 0), Point2D(3, 0), Point2D(-1, 0)]
        dup = [Point2D(0, 0), Point2D(3, 0), Point2D(3, 0), Point2D(-1, 0)]
        x = Symbol('x', positive=True)
        s = [Point2D(a) for a in ((x, 1), (x + 3, 2), (x + 2, 2))]
        
        # 对每个测试数据集进行遍历，进行断言验证
        for points in (p1, p2, p3, p4, p5, dup, s):
            # 计算点集中所有点两两之间的距离，并取其最大值或最小值
            d = how(i.distance(j) for i, j in subsets(set(points), 2))
            # 调用函数 func 计算最远点或最近点，并验证返回的结果是否符合预期
            ans = a, b = list(func(*points))[0]
            assert a.distance(b) == d
            assert ans == _ordered_points(ans)
        
        # 如果以下断言失败，说明上面的测试不足以覆盖所有逻辑，需要修复算法中的逻辑错误
        points = set()
        while len(points) != 7:
            points.add(Point2D(randint(1, 100), randint(1, 100)))
        points = list(points)
        d = how(i.distance(j) for i, j in subsets(points, 2))
        ans = a, b = list(func(*points))[0]
        assert a.distance(b) == d
        assert ans == _ordered_points(ans)
    
    # 针对具有等距点的情况，进行验证
    a, b, c = (
        Point2D(0, 0), Point2D(1, 0), Point2D(S.Half, sqrt(3)/2))
    ans = {_ordered_points((i, j)) for i, j in subsets((a, b, c), 2)}
    assert closest_points(b, c, a) == ans
    assert farthest_points(b, c, a) == ans
    
    # 针对最远点的独特情况进行验证
    points = [(1, 1), (1, 2), (3, 1), (-5, 2), (15, 4)]
    # 断言测试函数 farthest_points 对于给定点集是否返回预期的结果集合
    assert farthest_points(*points) == {
        (Point2D(-5, 2), Point2D(15, 4))
    }
    
    # 定义一个新的点集合 points，包含特定的点坐标
    points = [(1, -1), (1, -2), (3, -1), (-5, -2), (15, -4)]
    
    # 断言测试函数 farthest_points 对新的点集合是否返回预期的结果集合
    assert farthest_points(*points) == {
        (Point2D(-5, -2), Point2D(15, -4))
    }
    
    # 断言测试函数 farthest_points 对包含两个点的元组是否返回预期的结果集合
    assert farthest_points((1, 1), (0, 0)) == {
        (Point2D(0, 0), Point2D(1, 1))
    }
    
    # 断言测试函数 farthest_points 是否能够引发 ValueError 异常，因为点集数目少于两个
    raises(ValueError, lambda: farthest_points((1, 1)))
# 定义一个测试函数，用于测试三维和二维几何对象的共面性检查函数
def test_are_coplanar():
    # 创建三维空间中的直线对象 a，连接点 (5, 0, 0) 和 (1, -1, 1)
    a = Line3D(Point3D(5, 0, 0), Point3D(1, -1, 1))
    # 创建三维空间中的直线对象 b，连接点 (0, -2, 0) 和 (3, 1, 1)
    b = Line3D(Point3D(0, -2, 0), Point3D(3, 1, 1))
    # 创建三维空间中的直线对象 c，连接点 (0, -1, 0) 和 (5, -1, 9)
    c = Line3D(Point3D(0, -1, 0), Point3D(5, -1, 9))
    # 创建二维空间中的直线对象 d，连接点 (0, 3) 和 (1, 5)
    d = Line(Point2D(0, 3), Point2D(1, 5))

    # 断言：检查直线 a、b、c 是否共面，预期结果为 False
    assert are_coplanar(a, b, c) == False
    # 断言：检查直线 a 和 d 是否共面，预期结果为 False
    assert are_coplanar(a, d) == False
```