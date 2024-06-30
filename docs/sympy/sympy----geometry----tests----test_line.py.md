# `D:\src\scipysrc\sympy\sympy\geometry\tests\test_line.py`

```
from sympy.core.numbers import (Float, Rational, oo, pi)  # 导入数值相关模块和常数
from sympy.core.relational import Eq  # 导入等式模块
from sympy.core.singleton import S  # 导入单例模块
from sympy.core.symbol import (Symbol, symbols)  # 导入符号相关模块
from sympy.functions.elementary.miscellaneous import sqrt  # 导入平方根函数
from sympy.functions.elementary.trigonometric import (acos, cos, sin)  # 导入三角函数
from sympy.sets import EmptySet  # 导入空集合模块
from sympy.simplify.simplify import simplify  # 导入简化函数
from sympy.functions.elementary.trigonometric import tan  # 导入正切函数
from sympy.geometry import (Circle, GeometryError, Line, Point, Ray,  # 导入几何相关模块
    Segment, Triangle, intersection, Point3D, Line3D, Ray3D, Segment3D,
    Point2D, Line2D, Plane)
from sympy.geometry.line import Undecidable  # 导入不可判断模块
from sympy.geometry.polygon import _asa as asa  # 导入多边形相关模块
from sympy.utilities.iterables import cartes  # 导入可迭代模块
from sympy.testing.pytest import raises, warns  # 导入测试相关模块


x = Symbol('x', real=True)  # 定义实数符号 x
y = Symbol('y', real=True)  # 定义实数符号 y
z = Symbol('z', real=True)  # 定义实数符号 z
k = Symbol('k', real=True)  # 定义实数符号 k
x1 = Symbol('x1', real=True)  # 定义实数符号 x1
y1 = Symbol('y1', real=True)  # 定义实数符号 y1
t = Symbol('t', real=True)  # 定义实数符号 t
a, b = symbols('a,b', real=True)  # 定义实数符号 a, b
m = symbols('m', real=True)  # 定义实数符号 m


def test_object_from_equation():
    from sympy.abc import x, y, a, b  # 从 sympy.abc 中导入符号 x, y, a, b
    assert Line(3*x + y + 18) == Line2D(Point2D(0, -18), Point2D(1, -21))  # 测试线对象与线段的等式
    assert Line(3*x + 5 * y + 1) == Line2D(
        Point2D(0, Rational(-1, 5)), Point2D(1, Rational(-4, 5)))  # 测试线对象与线段的等式
    assert Line(3*a + b + 18, x="a", y="b") == Line2D(
        Point2D(0, -18), Point2D(1, -21))  # 测试线对象与线段的等式
    assert Line(3*x + y) == Line2D(Point2D(0, 0), Point2D(1, -3))  # 测试线对象与线段的等式
    assert Line(x + y) == Line2D(Point2D(0, 0), Point2D(1, -1))  # 测试线对象与线段的等式
    assert Line(Eq(3*a + b, -18), x="a", y=b) == Line2D(
        Point2D(0, -18), Point2D(1, -21))  # 测试线对象与线段的等式
    # issue 22361
    assert Line(x - 1) == Line2D(Point2D(1, 0), Point2D(1, 1))  # 测试线对象与线段的等式
    assert Line(2*x - 2, y=x) == Line2D(Point2D(0, 1), Point2D(1, 1))  # 测试线对象与线段的等式
    assert Line(y) == Line2D(Point2D(0, 0), Point2D(1, 0))  # 测试线对象与线段的等式
    assert Line(2*y, x=y) == Line2D(Point2D(0, 0), Point2D(0, 1))  # 测试线对象与线段的等式
    assert Line(y, x=y) == Line2D(Point2D(0, 0), Point2D(0, 1))  # 测试线对象与线段的等式
    raises(ValueError, lambda: Line(x / y))  # 测试是否引发值错误异常
    raises(ValueError, lambda: Line(a / b, x='a', y='b'))  # 测试是否引发值错误异常
    raises(ValueError, lambda: Line(y / x))  # 测试是否引发值错误异常
    raises(ValueError, lambda: Line(b / a, x='a', y='b'))  # 测试是否引发值错误异常
    raises(ValueError, lambda: Line((x + 1)**2 + y))  # 测试是否引发值错误异常


def feq(a, b):
    """Test if two floating point values are 'equal'."""  # 测试两个浮点数是否相等的函数
    t_float = Float("1.0E-10")  # 定义浮点数精度
    return -t_float < a - b < t_float  # 返回浮点数之差在精度范围内的判断结果


def test_angle_between():
    a = Point(1, 2, 3, 4)  # 创建点对象 a
    b = a.orthogonal_direction  # 计算点对象 a 的正交方向
    o = a.origin  # 获取点对象 a 的原点
    assert feq(Line.angle_between(Line(Point(0, 0), Point(1, 1)),  # 测试线与线之间的角度
                                  Line(Point(0, 0), Point(5, 0))).evalf(), pi.evalf() / 4)  # 断言角度相等
    assert Line(a, o).angle_between(Line(b, o)) == pi / 2  # 测试线与线之间的角度
    z = Point3D(0, 0, 0)  # 创建三维点对象 z
    assert Line3D.angle_between(Line3D(z, Point3D(1, 1, 1)),  # 测试三维线与三维线之间的角度
                                Line3D(z, Point3D(5, 0, 0))) == acos(sqrt(3) / 3)  # 断言角度相等
    # direction of points is used to determine angle  # 使用点的方向确定角度
    # 使用 Line3D 类中的 angle_between 方法计算两条三维直线之间的夹角
    assert Line3D.angle_between(Line3D(z, Point3D(1, 1, 1)),
                                Line3D(Point3D(5, 0, 0), z)) == acos(-sqrt(3) / 3)
# 定义一个测试函数，测试 Ray 类的 closing_angle 方法
def test_closing_angle():
    # 创建两个 Ray 对象，一个水平，一个垂直
    a = Ray((0, 0), angle=0)
    b = Ray((1, 2), angle=pi/2)
    # 测试 a 对 b 的夹角，应为 -π/2
    assert a.closing_angle(b) == -pi/2
    # 测试 b 对 a 的夹角，应为 π/2
    assert b.closing_angle(a) == pi/2
    # 测试 a 对自身的夹角，应为 0
    assert a.closing_angle(a) == 0


# 定义一个测试函数，测试 Line 类的 smallest_angle_between 方法
def test_smallest_angle():
    # 创建两个 Line 对象
    a = Line(Point(1, 1), Point(1, 2))
    b = Line(Point(1, 1), Point(2, 3))
    # 测试两条直线之间的最小夹角，应为 acos(2*sqrt(5)/5)
    assert a.smallest_angle_between(b) == acos(2*sqrt(5)/5)


# 定义一个测试函数，测试 Line 和 Segment 类的 _svg 方法
def test_svg():
    # 测试 Line 类的 _svg 方法生成的 SVG 字符串
    a = Line(Point(1, 1), Point(1, 2))
    assert a._svg() == '<path fill-rule="evenodd" fill="#66cc99" stroke="#555555" stroke-width="2.0" opacity="0.6" d="M 1.00000000000000,1.00000000000000 L 1.00000000000000,2.00000000000000" marker-start="url(#markerReverseArrow)" marker-end="url(#markerArrow)"/>'
    # 测试 Segment 类的 _svg 方法生成的 SVG 字符串
    a = Segment(Point(1, 0), Point(1, 1))
    assert a._svg() == '<path fill-rule="evenodd" fill="#66cc99" stroke="#555555" stroke-width="2.0" opacity="0.6" d="M 1.00000000000000,0 L 1.00000000000000,1.00000000000000" />'  
    # 测试 Ray 类的 _svg 方法生成的 SVG 字符串
    a = Ray(Point(2, 3), Point(3, 5))
    assert a._svg() == '<path fill-rule="evenodd" fill="#66cc99" stroke="#555555" stroke-width="2.0" opacity="0.6" d="M 2.00000000000000,3.00000000000000 L 3.00000000000000,5.00000000000000" marker-start="url(#markerCircle)" marker-end="url(#markerArrow)"/>'


# 定义一个测试函数，测试 Line3D、Ray、Segment3D 等类的方法
def test_arbitrary_point():
    # 创建 Line3D 对象
    l1 = Line3D(Point3D(0, 0, 0), Point3D(1, 1, 1))
    # 创建 Line 对象，其中使用变量 x1 和 y1
    l2 = Line(Point(x1, x1), Point(y1, y1))
    # 测试 arbitrary_point 方法返回的点是否在 l2 上
    assert l2.arbitrary_point() in l2
    # 测试 Ray 类的 arbitrary_point 方法返回的点是否正确
    assert Ray((1, 1), angle=pi / 4).arbitrary_point() == Point(t + 1, t + 1)
    # 测试 Segment 类的 arbitrary_point 方法返回的点是否正确
    assert Segment((1, 1), (2, 3)).arbitrary_point() == Point(1 + t, 1 + 2 * t)
    # 测试 Line3D 类的 perpendicular_segment 方法返回的点是否正确
    assert l1.perpendicular_segment(l1.arbitrary_point()) == l1.arbitrary_point()
    # 测试 Ray3D 类的 arbitrary_point 方法返回的点是否正确
    assert Ray3D((1, 1, 1), direction_ratio=[1, 2, 3]).arbitrary_point() == Point3D(t + 1, 2 * t + 1, 3 * t + 1)
    # 测试 Segment3D 类的 midpoint 属性返回的点是否正确
    assert Segment3D(Point3D(0, 0, 0), Point3D(1, 1, 1)).midpoint == Point3D(S.Half, S.Half, S.Half)
    # 测试 Segment3D 类的 length 属性返回的长度是否正确
    assert Segment3D(Point3D(x1, x1, x1), Point3D(y1, y1, y1)).length == sqrt(3) * sqrt((x1 - y1) ** 2)
    # 测试 Segment3D 类的 arbitrary_point 方法返回的点是否正确
    assert Segment3D((1, 1, 1), (2, 3, 4)).arbitrary_point() == Point3D(t + 1, 2 * t + 1, 3 * t + 1)
    # 测试 Line 类的 arbitrary_point 方法在给定 x 值时抛出 ValueError 异常
    raises(ValueError, (lambda: Line((x, 1), (2, 3)).arbitrary_point(x)))


# 定义一个测试函数，测试 Line 类和 Line3D 类的 are_concurrent 方法
def test_are_concurrent_2d():
    # 创建两个 Line 对象
    l1 = Line(Point(0, 0), Point(1, 1))
    l2 = Line(Point(x1, x1), Point(x1, 1 + x1))
    # 测试 are_concurrent 方法判断 l1 是否与其他对象都不共线
    assert Line.are_concurrent(l1) is False
    # 测试 are_concurrent 方法判断 l1 和 l2 是否共线
    assert Line.are_concurrent(l1, l2)
    # 测试 are_concurrent 方法判断多个 Line 对象是否共线
    assert Line.are_concurrent(l1, l1, l1, l2)
    # 测试 are_concurrent 方法判断多个 Line 对象是否共线，其中包含参数变换
    assert Line.are_concurrent(l1, l2, Line(Point(5, x1), Point(Rational(-3, 5), x1)))
    # 测试 are_concurrent 方法判断多个 Line 对象是否共线，其中包含参数变换
    assert Line.are_concurrent(l1, Line(Point(0, 0), Point(-x1, x1)), l2) is False


# 定义一个测试函数，测试 Line3D 类的 are_concurrent 方法
def test_are_concurrent_3d():
    # 创建 3D 点和 Line3D 对象
    p1 = Point3D(0, 0, 0)
    l1 = Line3D(p1, Point3D(1, 1, 1))
    # 创建与 l1 平行的 Line3D 对象
    parallel_1 = Line3D(Point3D(0, 0, 0), Point3D(1, 0, 0))
    parallel_2 = Line3D(Point3D(0, 1, 0), Point3D(1, 1, 0))
    # 测试 are_concurrent 方法判断 l1 是否与其他对象都不共线
    assert Line3D.are_concurrent(l1) is False
    # 测试 are_concurrent 方法判断 l1 和另一个 Line3D 对象是否共线
    assert Line3D.are_concurrent(l1, Line(Point3D(x1, x1, x1), Point3D(y1, y1, y1))) is False
    # 使用 assert 断言两个三维空间中的直线是否共线，预期返回结果为 True
    assert Line3D.are_concurrent(l1, Line3D(p1, Point3D(x1, x1, x1)),
                                 Line(Point3D(x1, x1, x1), Point3D(x1, 1 + x1, 1))) is True
    
    # 使用 assert 断言两个平行的三维空间中的直线是否共线，预期返回结果为 False
    assert Line3D.are_concurrent(parallel_1, parallel_2) is False
# 测试函数，用于验证接受 `Point` 对象的 `geometry` 函数是否能自动接受元组、列表和生成器，并将它们自动转换为点。
def test_arguments():
    # 导入 subsets 函数
    from sympy.utilities.iterables import subsets

    # 定义二维单点集合
    singles2d = ((1, 2), [1, 3], Point(1, 5))
    # 计算二维单点集合的子集组合
    doubles2d = subsets(singles2d, 2)
    # 创建二维直线对象
    l2d = Line(Point2D(1, 2), Point2D(2, 3))
    
    # 定义三维单点集合
    singles3d = ((1, 2, 3), [1, 2, 4], Point(1, 2, 6))
    # 计算三维单点集合的子集组合
    doubles3d = subsets(singles3d, 2)
    # 创建三维直线对象
    l3d = Line(Point3D(1, 2, 3), Point3D(1, 1, 2))
    
    # 定义四维单点集合
    singles4d = ((1, 2, 3, 4), [1, 2, 3, 5], Point(1, 2, 3, 7))
    # 计算四维单点集合的子集组合
    doubles4d = subsets(singles4d, 2)
    # 创建四维直线对象
    l4d = Line(Point(1, 2, 3, 4), Point(2, 2, 2, 2))
    
    # 测试二维情况
    test_single = ['contains', 'distance', 'equals', 'parallel_line', 'perpendicular_line', 'perpendicular_segment',
                   'projection', 'intersection']
    # 遍历二维单点集合的子集组合，创建二维直线对象
    for p in doubles2d:
        Line2D(*p)
    # 遍历测试函数列表
    for func in test_single:
        # 遍历二维单点集合，对每个点执行指定的测试函数
        for p in singles2d:
            getattr(l2d, func)(p)
    
    # 测试三维情况
    for p in doubles3d:
        Line3D(*p)
    for func in test_single:
        for p in singles3d:
            getattr(l3d, func)(p)
    
    # 测试四维情况
    for p in doubles4d:
        Line(*p)
    for func in test_single:
        for p in singles4d:
            getattr(l4d, func)(p)


# 测试二维基本属性
def test_basic_properties_2d():
    # 创建点对象
    p1 = Point(0, 0)
    p2 = Point(1, 1)
    p10 = Point(2000, 2000)
    # 创建射线对象并生成随机点
    p_r3 = Ray(p1, p2).random_point()
    p_r4 = Ray(p2, p1).random_point()
    
    # 创建二维直线对象
    l1 = Line(p1, p2)
    # 尝试创建二维直线对象（出现 x1 未定义的错误）
    l3 = Line(Point(x1, x1), Point(x1, 1 + x1))
    # 创建二维直线对象
    l4 = Line(p1, Point(1, 0))
    
    # 创建射线对象
    r1 = Ray(p1, Point(0, 1))
    r2 = Ray(Point(0, 1), p1)
    
    # 创建线段对象
    s1 = Segment(p1, p10)
    # 在线段上生成随机点
    p_s1 = s1.random_point()
    
    # 断言语句，用于验证二维直线对象的各种属性
    assert Line((1, 1), slope=1) == Line((1, 1), (2, 2))
    assert Line((1, 1), slope=oo) == Line((1, 1), (1, 2))
    assert Line((1, 1), slope=oo).bounds == (1, 1, 1, 2)
    assert Line((1, 1), slope=-oo) == Line((1, 1), (1, 2))
    assert Line(p1, p2).scale(2, 1) == Line(p1, Point(2, 1))
    assert Line(p1, p2) == Line(p1, p2)
    assert Line(p1, p2) != Line(p2, p1)
    assert l1 != Line(Point(x1, x1), Point(y1, y1))
    assert l1 != l3
    assert Line(p1, p10) != Line(p10, p1)
    assert Line(p1, p10) != p1
    assert p1 in l1  # is p1 on the line l1?
    assert p1 not in l3
    assert s1 in Line(p1, p10)
    assert Ray(Point(0, 0), Point(0, 1)) in Ray(Point(0, 0), Point(0, 2))
    assert Ray(Point(0, 0), Point(0, 2)) in Ray(Point(0, 0), Point(0, 1))
    assert Ray(Point(0, 0), Point(0, 2)).xdirection == S.Zero
    assert Ray(Point(0, 0), Point(1, 2)).xdirection == S.Infinity
    assert Ray(Point(0, 0), Point(-1, 2)).xdirection == S.NegativeInfinity
    assert Ray(Point(0, 0), Point(2, 0)).ydirection == S.Zero
    assert Ray(Point(0, 0), Point(2, 2)).ydirection == S.Infinity
    assert Ray(Point(0, 0), Point(2, -2)).ydirection == S.NegativeInfinity
    assert (r1 in s1) is False
    assert Segment(p1, p2) in s1
    assert Ray(Point(x1, x1), Point(x1, 1 + x1)) != Ray(p1, Point(-1, 5))
    # 确认通过两个点创建的线段的中点是否等于坐标为 (S.Half, S.Half) 的点
    assert Segment(p1, p2).midpoint == Point(S.Half, S.Half)
    
    # 确认通过两个点创建的线段的长度是否等于 sqrt(2 * (x1 ** 2))
    assert Segment(p1, Point(-x1, x1)).length == sqrt(2 * (x1 ** 2))
    
    # 确认直线 l1 的斜率是否为 1
    assert l1.slope == 1
    
    # 确认 l3 是一条垂直线
    assert l3.slope is oo  # oo 表示正无穷，即垂直线的斜率
    
    # 确认 l4 是一条水平线
    assert l4.slope == 0
    
    # 确认通过点 p1 和 (0, 1) 创建的直线斜率为正无穷
    assert Line(p1, Point(0, 1)).slope is oo
    
    # 确认随机生成的点是否在射线 r1 上，并且具有相同的斜率
    assert Line(r1.source, r1.random_point()).slope == r1.slope
    
    # 确认随机生成的点是否在射线 r2 上，并且具有相同的斜率
    assert Line(r2.source, r2.random_point()).slope == r2.slope
    
    # 确认随机生成的点是否在由点 (0, -1) 和线段 p1 到 (0, 1) 的斜率相同的线段上
    assert Segment(Point(0, -1), Segment(p1, Point(0, 1)).random_point()).slope == Segment(p1, Point(0, 1)).slope
    
    # 确认直线 l4 的系数为 (0, 1, 0)
    assert l4.coefficients == (0, 1, 0)
    
    # 确认通过两个点 (-x, x) 和 (-x + 1, x - 1) 创建的直线的系数为 (1, 1, 0)
    assert Line((-x, x), (-x + 1, x - 1)).coefficients == (1, 1, 0)
    
    # 确认通过点 p1 和 (0, 1) 创建的直线的系数为 (1, 0, 0)
    assert Line(p1, Point(0, 1)).coefficients == (1, 0, 0)
    
    # issue 7963
    # 确认射线 r 在角度 x 替换为 3*pi/4 后的结果为 Ray((0, 0), (-1, 1))
    assert r.subs(x, 3 * pi / 4) == Ray((0, 0), (-1, 1))
    
    # 确认射线 r 在角度 x 替换为 5*pi/4 后的结果为 Ray((0, 0), (-1, -1))
    assert r.subs(x, 5 * pi / 4) == Ray((0, 0), (-1, -1))
    
    # 确认射线 r 在角度 x 替换为 -pi/4 后的结果为 Ray((0, 0), (1, -1))
    assert r.subs(x, -pi / 4) == Ray((0, 0), (1, -1))
    
    # 确认射线 r 在角度 x 替换为 pi/2 后的结果为 Ray((0, 0), (0, 1))
    assert r.subs(x, pi / 2) == Ray((0, 0), (0, 1))
    
    # 确认射线 r 在角度 x 替换为 -pi/2 后的结果为 Ray((0, 0), (0, -1))
    assert r.subs(x, -pi / 2) == Ray((0, 0), (0, -1))
    
    # 确认直线 l3 随机点生成函数 random_point() 生成的点确实位于直线上，进行五次检查
    for ind in range(0, 5):
        assert l3.random_point() in l3
    
    # 确认点 p_r3 的 x 坐标大于等于 p1 的 x 坐标且 y 坐标大于等于 p1 的 y 坐标
    assert p_r3.x >= p1.x and p_r3.y >= p1.y
    
    # 确认点 p_r4 的 x 坐标小于等于 p2 的 x 坐标且 y 坐标小于等于 p2 的 y 坐标
    assert p_r4.x <= p2.x and p_r4.y <= p2.y
    
    # 确认点 p_s1 的 x 坐标在 p1.x 和 p10.x 之间，且 y 坐标在 p1.y 和 p10.y 之间
    assert p1.x <= p_s1.x <= p10.x and p1.y <= p_s1.y <= p10.y
    
    # 确认线段 s1 的哈希值不等于由点 p10 到 p1 创建的线段的哈希值
    assert hash(s1) != hash(Segment(p10, p1))
    
    # 确认线段 s1 的绘图区间为 [t, 0, 1]
    assert s1.plot_interval() == [t, 0, 1]
    
    # 确认通过点 p1 和 p10 创建的直线的绘图区间为 [t, -5, 5]
    assert Line(p1, p10).plot_interval() == [t, -5, 5]
    
    # 确认角度为 pi/4 的射线的绘图区间为 [t, 0, 10]
    assert Ray((0, 0), angle=pi / 4).plot_interval() == [t, 0, 10]
def test_basic_properties_3d():
    # 创建三维点对象
    p1 = Point3D(0, 0, 0)
    p2 = Point3D(1, 1, 1)
    # 使用未定义的变量 x1 来创建第三个三维点对象 p3
    p3 = Point3D(x1, x1, x1)
    # 使用未定义的变量 x1 来创建第四个三维点对象 p5
    p5 = Point3D(x1, 1 + x1, 1)

    # 创建三维线对象 l1 和 l3
    l1 = Line3D(p1, p2)
    l3 = Line3D(p3, p5)

    # 创建三维射线对象 r1 和 r3
    r1 = Ray3D(p1, Point3D(-1, 5, 0))
    r3 = Ray3D(p1, p2)

    # 创建三维线段对象 s1
    s1 = Segment3D(p1, p2)

    # 断言语句，验证属性和对象是否符合预期
    assert Line3D((1, 1, 1), direction_ratio=[2, 3, 4]) == Line3D(Point3D(1, 1, 1), Point3D(3, 4, 5))
    assert Line3D((1, 1, 1), direction_ratio=[1, 5, 7]) == Line3D(Point3D(1, 1, 1), Point3D(2, 6, 8))
    assert Line3D((1, 1, 1), direction_ratio=[1, 2, 3]) == Line3D(Point3D(1, 1, 1), Point3D(2, 3, 4))
    assert Line3D(Point3D(0, 0, 0), Point3D(1, 0, 0)).direction_cosine == [1, 0, 0]
    assert Line3D(Line3D(p1, Point3D(0, 1, 0))) == Line3D(p1, Point3D(0, 1, 0))
    assert Ray3D(Line3D(Point3D(0, 0, 0), Point3D(1, 0, 0))) == Ray3D(p1, Point3D(1, 0, 0))
    assert Line3D(p1, p2) != Line3D(p2, p1)
    assert l1 != l3
    assert l1 != Line3D(p3, Point3D(y1, y1, y1))
    assert r3 != r1
    assert Ray3D(Point3D(0, 0, 0), Point3D(1, 1, 1)) in Ray3D(Point3D(0, 0, 0), Point3D(2, 2, 2))
    assert Ray3D(Point3D(0, 0, 0), Point3D(2, 2, 2)) in Ray3D(Point3D(0, 0, 0), Point3D(1, 1, 1))
    assert Ray3D(Point3D(0, 0, 0), Point3D(2, 2, 2)).xdirection == S.Infinity
    assert Ray3D(Point3D(0, 0, 0), Point3D(2, 2, 2)).ydirection == S.Infinity
    assert Ray3D(Point3D(0, 0, 0), Point3D(2, 2, 2)).zdirection == S.Infinity
    assert Ray3D(Point3D(0, 0, 0), Point3D(-2, 2, 2)).xdirection == S.NegativeInfinity
    assert Ray3D(Point3D(0, 0, 0), Point3D(2, -2, 2)).ydirection == S.NegativeInfinity
    assert Ray3D(Point3D(0, 0, 0), Point3D(2, 2, -2)).zdirection == S.NegativeInfinity
    assert Ray3D(Point3D(0, 0, 0), Point3D(0, 2, 2)).xdirection == S.Zero
    assert Ray3D(Point3D(0, 0, 0), Point3D(2, 0, 2)).ydirection == S.Zero
    assert Ray3D(Point3D(0, 0, 0), Point3D(2, 2, 0)).zdirection == S.Zero
    assert p1 in l1
    assert p1 not in l3

    # 验证三维线对象的方向比率
    assert l1.direction_ratio == [1, 1, 1]

    # 验证三维线段对象的中点
    assert s1.midpoint == Point3D(S.Half, S.Half, S.Half)
    # 测试 zdirection 属性
    assert Ray3D(p1, Point3D(0, 0, -1)).zdirection is S.NegativeInfinity


def test_contains():
    p1 = Point(0, 0)

    r = Ray(p1, Point(4, 4))
    r1 = Ray3D(p1, Point3D(0, 0, -1))
    r2 = Ray3D(p1, Point3D(0, 1, 0))
    r3 = Ray3D(p1, Point3D(0, 0, 1))

    l = Line(Point(0, 1), Point(3, 4))
    # Segment contains
    assert Point(0, (a + b) / 2) in Segment((0, a), (0, b))
    assert Point((a + b) / 2, 0) in Segment((a, 0), (b, 0))
    assert Point3D(0, 1, 0) in Segment3D((0, 1, 0), (0, 1, 0))
    assert Point3D(1, 0, 0) in Segment3D((1, 0, 0), (1, 0, 0))
    assert Segment3D(Point3D(0, 0, 0), Point3D(1, 0, 0)).contains([]) is True
    assert Segment3D(Point3D(0, 0, 0), Point3D(1, 0, 0)).contains(
        Segment3D(Point3D(2, 2, 2), Point3D(3, 2, 2))) is False
    # Line contains
    assert l.contains(Point(0, 1)) is True
    assert l.contains((0, 1)) is True
    assert l.contains((0, 0)) is False
    # Ray contains
    # 断言矩形 r 包含点 p1，返回 True
    assert r.contains(p1) is True
    # 断言矩形 r 包含点 (1, 1)，返回 True
    assert r.contains((1, 1)) is True
    # 断言矩形 r 包含点 (1, 3)，返回 False
    assert r.contains((1, 3)) is False
    # 断言矩形 r 包含线段 Segment((1, 1), (2, 2))，返回 True
    assert r.contains(Segment((1, 1), (2, 2))) is True
    # 断言矩形 r 包含线段 Segment((1, 2), (2, 5))，返回 False
    assert r.contains(Segment((1, 2), (2, 5))) is False
    # 断言矩形 r 包含射线 Ray((2, 2), (3, 3))，返回 True
    assert r.contains(Ray((2, 2), (3, 3))) is True
    # 断言矩形 r 包含射线 Ray((2, 2), (3, 5))，返回 False
    assert r.contains(Ray((2, 2), (3, 5))) is False
    # 断言立体 r1 包含 3D 线段 Segment3D(p1, Point3D(0, 0, -10))，返回 True
    assert r1.contains(Segment3D(p1, Point3D(0, 0, -10))) is True
    # 断言立体 r1 包含 3D 线段 Segment3D(Point3D(1, 1, 1), Point3D(2, 2, 2))，返回 False
    assert r1.contains(Segment3D(Point3D(1, 1, 1), Point3D(2, 2, 2))) is False
    # 断言立体 r2 包含 3D 点 Point3D(0, 0, 0)，返回 True
    assert r2.contains(Point3D(0, 0, 0)) is True
    # 断言立体 r3 包含 3D 点 Point3D(0, 0, 0)，返回 True
    assert r3.contains(Point3D(0, 0, 0)) is True
    # 断言 3D 射线 Ray3D(Point3D(1, 1, 1), Point3D(1, 0, 0)) 不包含空列表，返回 False
    assert Ray3D(Point3D(1, 1, 1), Point3D(1, 0, 0)).contains([]) is False
    # 断言 3D 直线 Line3D((0, 0, 0), (x, y, z)) 包含点 (2 * x, 2 * y, 2 * z)
    with warns(UserWarning, test_stacklevel=False):
        # 在警告情况下，断言 Line3D(p1, Point3D(0, 1, 0)) 不包含二维点 Point(1.0, 1.0)，返回 False
        assert Line3D(p1, Point3D(0, 1, 0)).contains(Point(1.0, 1.0)) is False

    with warns(UserWarning, test_stacklevel=False):
        # 在警告情况下，断言立体 r3 不包含二维点 Point(1.0, 1.0)，返回 False
        assert r3.contains(Point(1.0, 1.0)) is False
def test_contains_nonreal_symbols():
    # 创建符号变量 u, v, w, z
    u, v, w, z = symbols('u, v, w, z')
    # 创建线段对象 l，起点为 Point(u, w)，终点为 Point(v, z)
    l = Segment(Point(u, w), Point(v, z))
    # 创建点对象 p，位置为 u*2/3 + v/3, w*2/3 + z/3
    p = Point(u*Rational(2, 3) + v/3, w*Rational(2, 3) + z/3)
    # 断言线段 l 是否包含点 p
    assert l.contains(p)


def test_distance_2d():
    # 创建点对象 p1，位置为 (0, 0)
    p1 = Point(0, 0)
    # 创建点对象 p2，位置为 (1, 1)
    p2 = Point(1, 1)
    # 创建 S.Half 的别名 half
    half = S.Half

    # 创建线段对象 s1，起点为 Point(0, 0)，终点为 Point(1, 1)
    s1 = Segment(Point(0, 0), Point(1, 1))
    # 创建线段对象 s2，起点为 Point(half, half)，终点为 Point(1, 0)
    s2 = Segment(Point(half, half), Point(1, 0))

    # 创建射线对象 r，起点为 p1，终点为 p2
    r = Ray(p1, p2)

    # 断言 s1 到点 (0, 0) 的距离为 0
    assert s1.distance(Point(0, 0)) == 0
    # 断言 s1 到元组 (0, 0) 的距离为 0
    assert s1.distance((0, 0)) == 0
    # 断言 s2 到点 (0, 0) 的距离为 sqrt(2)/2
    assert s2.distance(Point(0, 0)) == 2 ** half / 2
    # 断言 s2 到点 (3/2, 3/2) 的距离为 sqrt(2)
    assert s2.distance(Point(Rational(3) / 2, Rational(3) / 2)) == 2 ** half
    # 断言直线 Line(p1, p2) 到点 (-1, 1) 的距离为 sqrt(2)
    assert Line(p1, p2).distance(Point(-1, 1)) == sqrt(2)
    # 断言直线 Line(p1, p2) 到点 (1, -1) 的距离为 sqrt(2)
    assert Line(p1, p2).distance(Point(1, -1)) == sqrt(2)
    # 断言直线 Line(p1, p2) 到点 (2, 2) 的距离为 0
    assert Line(p1, p2).distance(Point(2, 2)) == 0
    # 断言直线 Line(p1, p2) 到元组 (-1, 1) 的距离为 sqrt(2)
    assert Line(p1, p2).distance((-1, 1)) == sqrt(2)
    # 断言直线 Line((0, 0), (0, 1)) 到点 p1 的距离为 0
    assert Line((0, 0), (0, 1)).distance(p1) == 0
    # 断言直线 Line((0, 0), (0, 1)) 到点 p2 的距离为 1
    assert Line((0, 0), (0, 1)).distance(p2) == 1
    # 断言直线 Line((0, 0), (1, 0)) 到点 p1 的距离为 0
    assert Line((0, 0), (1, 0)).distance(p1) == 0
    # 断言直线 Line((0, 0), (1, 0)) 到点 p2 的距离为 1
    assert Line((0, 0), (1, 0)).distance(p2) == 1
    # 断言射线 r 到点 (-1, -1) 的距离为 sqrt(2)
    assert r.distance(Point(-1, -1)) == sqrt(2)
    # 断言射线 r 到点 (1, 1) 的距离为 0
    assert r.distance(Point(1, 1)) == 0
    # 断言射线 r 到点 (-1, 1) 的距离为 sqrt(2)
    assert r.distance(Point(-1, 1)) == sqrt(2)
    # 断言射线 Ray((1, 1), (2, 2)) 到点 (1.5, 3) 的距离为 3 * sqrt(2) / 4
    assert Ray((1, 1), (2, 2)).distance(Point(1.5, 3)) == 3 * sqrt(2) / 4
    # 断言射线 r 到元组 (1, 1) 的距离为 0
    assert r.distance((1, 1)) == 0


def test_dimension_normalization():
    # 使用 with 语句捕获 UserWarning，忽略堆栈信息
    with warns(UserWarning, test_stacklevel=False):
        # 断言射线 Ray((1, 1), (2, 1, 2)) 是否等于 Ray((1, 1, 0), (2, 1, 2))
        assert Ray((1, 1), (2, 1, 2)) == Ray((1, 1, 0), (2, 1, 2))


def test_distance_3d():
    # 创建 3D 点对象 p1 和 p2，位置分别为 (0, 0, 0) 和 (1, 1, 1)
    p1, p2 = Point3D(0, 0, 0), Point3D(1, 1, 1)
    # 创建 3D 点对象 p3，位置为 (3/2, 3/2, 3/2)
    p3 = Point3D(Rational(3) / 2, Rational(3) / 2, Rational(3) / 2)

    # 创建 3D 线段对象 s1，起点为 Point3D(0, 0, 0)，终点为 Point3D(1, 1, 1)
    s1 = Segment3D(Point3D(0, 0, 0), Point3D(1, 1, 1))
    # 创建 3D 线段对象 s2，起点为 Point3D(S.Half, S.Half, S.Half)，终点为 Point3D(1, 0, 1)
    s2 = Segment3D(Point3D(S.Half, S.Half, S.Half), Point3D(1, 0, 1))

    # 创建 3D 射线对象 r，起点为 p1，终点为 p2
    r = Ray3D(p1, p2)

    # 断言 s1 到点 p1 的距离为 0
    assert s1.distance(p1) == 0
    # 断言 s2 到点 p1 的距离为 sqrt(3)/2
    assert s2.distance(p1) == sqrt(3) / 2
    # 断言 s2 到点 p3 的距离为 2 * sqrt(6)/3
    assert s2.distance(p3) == 2 * sqrt(6) / 3
    # 断言 s1 到元组 (0, 0, 0) 的距离为 0
    assert s1.distance((0, 0, 0)) == 0
    # 断言 s2 到元组 (0, 0, 0) 的距离为 sqrt(3)/2
    assert s2.distance((0, 0, 0)) == sqrt(3) / 2
    # Line3D 到点 Point3D(-1, 1, 1) 的距离为 2 * sqrt(6)/3
    assert Line3D(p1, p2).distance(Point3D(-1, 1, 1)) == 2 * sqrt(6) / 3
    # Line3D 到点 Point3D(1, -1, 1) 的距离为 2 * sqrt
    # 验证两条三维直线之间的距离
    assert Line3D((0, 0, 0), (1, 0, 0)).distance(Line3D((0, 1, 0), (0, 1, 1))) == 1
    
    # 验证三维直线与平面之间的距离
    assert Line3D((0, 0, 0), (1, 0, 0)).distance(Plane((2, 0, 0), (0, 0, 1))) == 0
    assert Line3D((0, 0, 0), (1, 0, 0)).distance(Plane((0, 1, 0), (0, 1, 0))) == 1
    assert Line3D((0, 0, 0), (1, 0, 0)).distance(Plane((1, 1, 3), (1, 0, 0))) == 0
    
    # 验证射线与点之间的距离
    assert r.distance(Point3D(-1, -1, -1)) == sqrt(3)
    assert r.distance(Point3D(1, 1, 1)) == 0
    assert r.distance((-1, -1, -1)) == sqrt(3)
    assert r.distance((1, 1, 1)) == 0
    
    # 验证射线与点之间的距离
    assert Ray3D((0, 0, 0), (1, 1, 2)).distance((-1, -1, 2)) == 4 * sqrt(3) / 3
    assert Ray3D((1, 1, 1), (2, 2, 2)).distance(Point3D(1.5, -3, -1)) == Rational(9) / 2
    assert Ray3D((1, 1, 1), (2, 2, 2)).distance(Point3D(1.5, 3, 1)) == sqrt(78) / 6
# 定义测试函数 test_equals
def test_equals():
    # 创建两个二维点对象 p1 和 p2
    p1 = Point(0, 0)
    p2 = Point(1, 1)

    # 创建两条线对象 l1 和 l2，分别通过两点 p1, p2 和 (0, 5)，且斜率为 m
    l1 = Line(p1, p2)
    l2 = Line((0, 5), slope=m)

    # 创建一条线对象 l3，通过两点 (x1, x1) 和 (x1, 1 + x1)
    l3 = Line(Point(x1, x1), Point(x1, 1 + x1))

    # 断言：l1 垂直线经过点 p1 的参数表达式，与给定的直线相等
    assert l1.perpendicular_line(p1.args).equals(Line(Point(0, 0), Point(1, -1)))
    
    # 断言：l1 垂直线经过点 p1，与给定的直线相等
    assert l1.perpendicular_line(p1).equals(Line(Point(0, 0), Point(1, -1)))
    
    # 断言：线 l3 平行线经过点 p1 的参数表达式，与给定的直线相等
    assert Line(Point(x1, x1), Point(y1, y1)).parallel_line(Point(-x1, x1)). \
        equals(Line(Point(-x1, x1), Point(-y1, 2 * x1 - y1)))
    
    # 断言：l3 平行线经过点 p1，与给定的直线相等
    assert l3.parallel_line(p1.args).equals(Line(Point(0, 0), Point(0, -1)))
    
    # 断言：l3 平行线经过点 p1，与给定的直线相等
    assert l3.parallel_line(p1).equals(Line(Point(0, 0), Point(0, -1)))
    
    # 断言：l2 到点 (2, 3) 的距离，与给定值相等
    assert (l2.distance(Point(2, 3)) - 2 * abs(m + 1) / sqrt(m ** 2 + 1)).equals(0)
    
    # 断言：三维线对象 Line3D(p1, Point3D(0, 1, 0)) 不等于点 (1.0, 1.0)
    assert Line3D(p1, Point3D(0, 1, 0)).equals(Point(1.0, 1.0)) is False
    
    # 断言：三维线对象 Line3D(Point3D(0, 0, 0), Point3D(1, 0, 0)) 等于另一条线对象 Line3D(Point3D(-5, 0, 0), Point3D(-1, 0, 0))
    assert Line3D(Point3D(0, 0, 0), Point3D(1, 0, 0)).equals(Line3D(Point3D(-5, 0, 0), Point3D(-1, 0, 0))) is True
    
    # 断言：三维线对象 Line3D(Point3D(0, 0, 0), Point3D(1, 0, 0)) 不等于线对象 Line3D(p1, Point3D(0, 1, 0))
    assert Line3D(Point3D(0, 0, 0), Point3D(1, 0, 0)).equals(Line3D(p1, Point3D(0, 1, 0))) is False
    
    # 断言：Ray3D(p1, Point3D(0, 0, -1)) 不等于点 (1.0, 1.0)
    assert Ray3D(p1, Point3D(0, 0, -1)).equals(Point(1.0, 1.0)) is False
    
    # 断言：Ray3D(p1, Point3D(0, 0, -1)) 等于另一条射线对象 Ray3D(p1, Point3D(0, 0, -1))
    assert Ray3D(p1, Point3D(0, 0, -1)).equals(Ray3D(p1, Point3D(0, 0, -1))) is True
    
    # 断言：三维线对象 Line3D((0, 0), (t, t)) 垂直线通过点 (0, 1, 0)，与给定的直线相等
    assert Line3D((0, 0), (t, t)).perpendicular_line(Point(0, 1, 0)).equals(
        Line3D(Point3D(0, 1, 0), Point3D(S.Half, S.Half, 0)))
    
    # 断言：三维线对象 Line3D((0, 0), (t, t)) 垂直线段通过点 (0, 1, 0)，与给定的线段相等
    assert Line3D((0, 0), (t, t)).perpendicular_segment(Point(0, 1, 0)).equals(Segment3D((0, 1), (S.Half, S.Half)))
    
    # 断言：三维线对象 Line3D(p1, Point3D(0, 1, 0)) 不等于点 (1.0, 1.0)
    assert Line3D(p1, Point3D(0, 1, 0)).equals(Point(1.0, 1.0)) is False


# 定义测试函数 test_equation
def test_equation():
    # 创建两个二维点对象 p1 和 p2
    p1 = Point(0, 0)
    p2 = Point(1, 1)
    
    # 创建两条线对象 l1 和 l3，分别通过两点 p1, p2 和 (x1, x1), (x1, 1 + x1)
    l1 = Line(p1, p2)
    l3 = Line(Point(x1, x1), Point(x1, 1 + x1))
    
    # 断言：简化后的 l1 的方程在 (x - y, y - x) 中
    assert simplify(l1.equation()) in (x - y, y - x)
    
    # 断言：简化后的 l3 的方程在 (x - x1, x1 - x) 中
    assert simplify(l3.equation()) in (x - x1, x1 - x)
    
    # 断言：线 l1 在给定的坐标 (x, y) 下的方程为 y
    assert Line(p1, Point(1, 0)).equation(x=x, y=y) == y
    
    # 断言：线 l1 在默认情况下的方程为 x
    assert Line(p1, Point(0, 1)).equation() == x
    
    # 断言：线通过点 (2, 0) 和 (2, 1) 的方程为 x - 2
    assert Line(Point(2, 0), Point(2, 1)).equation() == x - 2
    
    # 断言：线通过点 p2 和 (2, 1) 的方程为 y - 1
    assert Line(p2, Point(2, 1)).equation() == y - 1
    
    # 断言：三维线对象 Line3D(Point(x1, x1, x1), Point(y1, y1, y1)) 的方程在 (-x + y, -x + z) 中
    assert Line3D(Point(x1, x1, x1), Point(y1, y1, y1)).equation() == (-x + y, -x + z)
    
    # 断言：三维线对象 Line3D(Point(1, 2, 3), Point(2, 3, 4)) 的方程在 (-x + y - 1, -x + z - 2) 中
    assert Line3D(Point(1, 2, 3), Point(2, 3, 4)).equation() == (-x + y - 1, -x + z - 2)
    
    # 断言：三维线对象 Line3D(Point(1, 2, 3), Point(1, 3, 4)) 的方程在 (x - 1, -y + z - 1) 中
    assert Line3D(Point(1, 2, 3), Point(1, 3, 4)).equation() == (x - 1, -y + z - 1)
    # 创建射线对象 r1，起点 (1, 1)，终点 (2, 2)
    r1 = Ray(Point(1, 1), Point(2, 2))
    # 创建射线对象 r2，起点 (0, 0)，终点 (3, 4)
    r2 = Ray(Point(0, 0), Point(3, 4))
    # 创建射线对象 r4，起点为 p1，终点为 p2
    r4 = Ray(p1, p2)
    # 创建射线对象 r6，起点 (0, 1)，终点 (1, 2)
    r6 = Ray(Point(0, 1), Point(1, 2))
    # 创建射线对象 r7，起点 (0.5, 0.5)，终点 (1, 1)
    r7 = Ray(Point(0.5, 0.5), Point(1, 1))

    # 创建线段对象 s1，起点为 p1，终点为 p2
    s1 = Segment(p1, p2)
    # 创建线段对象 s2，起点 (0.25, 0.25)，终点 (0.5, 0.5)
    s2 = Segment(Point(0.25, 0.25), Point(0.5, 0.5))
    # 创建线段对象 s3，起点 (0, 0)，终点 (3, 4)
    s3 = Segment(Point(0, 0), Point(3, 4))

    # 断言：直线对象 l1 与点 p1 的交点为 [p1]
    assert intersection(l1, p1) == [p1]
    # 断言：直线对象 l1 与点 (x1, 1 + x1) 的交点为空列表
    assert intersection(l1, Point(x1, 1 + x1)) == []
    # 断言：直线对象 l1 与直线对象 Line(p3, p4) 的交点为 [l1] 或 [Line(p3, p4)]
    assert intersection(l1, Line(p3, p4)) in [[l1], [Line(p3, p4)]]
    # 断言：直线对象 l1 与与 l1 平行的 l1.parallel_line(Point(x1, 1 + x1)) 的交点为空列表
    assert intersection(l1, l1.parallel_line(Point(x1, 1 + x1))) == []
    # 断言：直线对象 l3 与自身的交点为 [l3]
    assert intersection(l3, l3) == [l3]
    # 断言：直线对象 l3 与射线对象 r2 的交点为 [r2]
    assert intersection(l3, r2) == [r2]
    # 断言：直线对象 l3 与线段对象 s3 的交点为 [s3]
    assert intersection(l3, s3) == [s3]
    # 断言：线段对象 s3 与直线对象 l3 的交点为 [s3]
    assert intersection(s3, l3) == [s3]
    # 断言：线段对象之间的相交情况，应为空列表
    assert intersection(Segment(Point(-10, 10), Point(10, 10)), Segment(Point(-5, -5), Point(-5, 5))) == []
    # 断言：射线对象 r2 与直线对象 l3 的交点为 [r2]
    assert intersection(r2, l3) == [r2]
    # 断言：射线对象 r1 与起点 (2, 2)、终点 (0, 0) 的射线的交点为 [Segment(Point(1, 1), Point(2, 2))]
    assert intersection(r1, Ray(Point(2, 2), Point(0, 0))) == [Segment(Point(1, 1), Point(2, 2))]
    # 断言：射线对象 r1 与起点 (1, 1) 的射线的交点为 [Point(1, 1)]
    assert intersection(r1, Ray(Point(1, 1), Point(-1, -1))) == [Point(1, 1)]
    # 断言：射线对象 r1 与线段对象 Segment(Point(0, 0), Point(2, 2)) 的交点为 [Segment(Point(1, 1), Point(2, 2))]
    assert intersection(r1, Segment(Point(0, 0), Point(2, 2))) == [Segment(Point(1, 1), Point(2, 2))]

    # 断言：射线对象 r4 与线段对象 s2 的交点为 [s2]
    assert r4.intersection(s2) == [s2]
    # 断言：射线对象 r4 与线段对象 Segment(Point(2, 3), Point(3, 4)) 的交点为空列表
    assert r4.intersection(Segment(Point(2, 3), Point(3, 4))) == []
    # 断言：射线对象 r4 与线段对象 Segment(Point(-1, -1), Point(0.5, 0.5)) 的交点为 [Segment(p1, Point(0.5, 0.5))]
    assert r4.intersection(Segment(Point(-1, -1), Point(0.5, 0.5))) == [Segment(p1, Point(0.5, 0.5))]
    # 断言：射线对象 r4 与射线对象 Ray(p2, p1) 的交点为 [s1]
    assert r4.intersection(Ray(p2, p1)) == [s1]
    # 断言：射线对象 Ray(p2, p1) 与射线对象 r6 的交点为空列表
    assert Ray(p2, p1).intersection(r6) == []
    # 断言：射线对象 r4 与射线对象 r7 的交点为 r7.intersection(r4) 的结果，即 [r7]
    assert r4.intersection(r7) == r7.intersection(r4) == [r7]
    # 断言：三维射线对象 Ray3D((0, 0), (3, 0)) 与三维射线对象 Ray3D((1, 0), (3, 0)) 的交点为 [Ray3D((1, 0), (3, 0))]
    assert Ray3D((0, 0), (3, 0)).intersection(Ray3D((1, 0), (3, 0))) == [Ray3D((1, 0), (3, 0))]
    # 断言：三维射线对象 Ray3D((1, 0), (3, 0)) 与三维射线对象 Ray3D((0, 0), (3, 0)) 的交点为 [Ray3D((1, 0), (3, 0))]
    assert Ray3D((1, 0), (3, 0)).intersection(Ray3D((0, 0), (3, 0))) == [Ray3D((1, 0), (3, 0))]
    # 断言：二维射线对象 Ray(Point(0, 0), Point(0, 4)) 与二维射线对象 Ray(Point(0, 1), Point(0, -1)) 的交点为 [Segment(Point(0, 0), Point(0, 1))]
    assert Ray(Point(0, 0), Point(0, 4)).intersection(Ray(Point(0, 1), Point(0, -1))) == [Segment(Point(0, 0), Point(0, 1))]

    # 断言：三维线段对象 Segment3D((0, 0), (3, 0)) 与三维线段对象 Segment3D((1, 0), (2, 0)) 的交点为 [Segment3D((1, 0), (2, 0))]
    assert Segment3D((0, 0), (3, 0)).intersection(Segment3D((1, 0), (2, 0))) == [Segment3D((1, 0), (2, 0))]
    # 断言：三维线段对象 Segment3D((1, 0), (2, 0)) 与三维线段对象 Segment3D((0, 0), (3, 0)) 的交点为 [Segment3D((1, 0), (2, 0))]
    assert Segment3D((1, 0), (2, 0)).intersection(Segment3D((0, 0), (3, 0))) == [Segment3D((
    # 断言：验证集合 s1 和 s2 的交集是否等于列表 [s2]
    assert s1.intersection(s2) == [s2]
    # 断言：验证集合 s2 和 s1 的交集是否等于列表 [s2]
    assert s2.intersection(s1) == [s2]

    # 断言：验证调用 asa 函数的返回值是否等于预期的三角形对象
    assert asa(120, 8, 52) == \
           Triangle(
               Point(0, 0),
               Point(8, 0),
               Point(-4 * cos(19 * pi / 90) / sin(2 * pi / 45),
                     4 * sqrt(3) * cos(19 * pi / 90) / sin(2 * pi / 45)))
    
    # 断言：验证 Line 对象和 Ray 对象的交点是否为预期的 Point(1, 1)
    assert Line((0, 0), (1, 1)).intersection(Ray((1, 0), (1, 2))) == [Point(1, 1)]
    # 断言：验证 Line 对象和 Segment 对象的交点是否为预期的 Point(1, 1)
    assert Line((0, 0), (1, 1)).intersection(Segment((1, 0), (1, 2))) == [Point(1, 1)]
    # 断言：验证 Ray 对象和 Ray 对象的交点是否为预期的 Point(1, 1)
    assert Ray((0, 0), (1, 1)).intersection(Ray((1, 0), (1, 2))) == [Point(1, 1)]
    # 断言：验证 Ray 对象和 Segment 对象的交点是否为预期的 Point(1, 1)
    assert Ray((0, 0), (1, 1)).intersection(Segment((1, 0), (1, 2))) == [Point(1, 1)]
    # 断言：验证 Ray 对象是否包含 Segment 对象
    assert Ray((0, 0), (10, 10)).contains(Segment((1, 1), (2, 2))) is True
    # 断言：验证 Segment 对象是否在 Line 对象内部
    assert Segment((1, 1), (2, 2)) in Line((0, 0), (10, 10))
    # 断言：验证集合 s1 和 Ray 对象的交集是否等于列表 [Point(1, 1)]
    assert s1.intersection(Ray((1, 1), (4, 4))) == [Point(1, 1)]
def test_line_intersection():
    # 示例中的代码块
    # see also test_issue_11238 in test_matrices.py
    # 使用有理数角度的正切函数计算 x0
    x0 = tan(pi*Rational(13, 45))
    # 计算 x1 为根号3
    x1 = sqrt(3)
    # 计算 x2 为 x0 的平方
    x2 = x0**2
    # 计算 x, y，使用列表解析式计算两个表达式的值
    x, y = [8*x0/(x0 + x1), (24*x0 - 8*x1*x2)/(x2 - 3)]
    # 断言检查直线是否包含特定点 (x, y)
    assert Line(Point(0, 0), Point(1, -sqrt(3))).contains(Point(x, y)) is True


def test_intersection_3d():
    # 创建 3D 点对象 p1 和 p2
    p1 = Point3D(0, 0, 0)
    p2 = Point3D(1, 1, 1)

    # 创建两条 3D 线对象 l1 和 l2
    l1 = Line3D(p1, p2)
    l2 = Line3D(Point3D(0, 0, 0), Point3D(3, 4, 0))

    # 创建 3D 射线对象 r1 和 r2，以及 3D 线段对象 s1
    r1 = Ray3D(Point3D(1, 1, 1), Point3D(2, 2, 2))
    r2 = Ray3D(Point3D(0, 0, 0), Point3D(3, 4, 0))
    s1 = Segment3D(Point3D(0, 0, 0), Point3D(3, 4, 0))

    # 进行多个断言，检查不同几何对象的交点或平行性关系
    assert intersection(l1, p1) == [p1]
    assert intersection(l1, Point3D(x1, 1 + x1, 1)) == []
    assert intersection(l1, l1.parallel_line(p1)) == [Line3D(Point3D(0, 0, 0), Point3D(1, 1, 1))]
    assert intersection(l2, r2) == [r2]
    assert intersection(l2, s1) == [s1]
    assert intersection(r2, l2) == [r2]
    assert intersection(r1, Ray3D(Point3D(1, 1, 1), Point3D(-1, -1, -1))) == [Point3D(1, 1, 1)]
    assert intersection(r1, Segment3D(Point3D(0, 0, 0), Point3D(2, 2, 2))) == [
        Segment3D(Point3D(1, 1, 1), Point3D(2, 2, 2))]
    assert intersection(Ray3D(Point3D(1, 0, 0), Point3D(-1, 0, 0)), Ray3D(Point3D(0, 1, 0), Point3D(0, -1, 0))) \
           == [Point3D(0, 0, 0)]
    assert intersection(r1, Ray3D(Point3D(2, 2, 2), Point3D(0, 0, 0))) == \
           [Segment3D(Point3D(1, 1, 1), Point3D(2, 2, 2))]
    assert intersection(s1, r2) == [s1]

    # 检查两条 3D 线的交点
    assert Line3D(Point3D(4, 0, 1), Point3D(0, 4, 1)).intersection(Line3D(Point3D(0, 0, 1), Point3D(4, 4, 1))) == \
           [Point3D(2, 2, 1)]
    assert Line3D((0, 1, 2), (0, 2, 3)).intersection(Line3D((0, 1, 2), (0, 1, 1))) == [Point3D(0, 1, 2)]
    assert Line3D((0, 0), (t, t)).intersection(Line3D((0, 1), (t, t))) == \
           [Point3D(t, t)]

    # 检查两条 3D 射线的交点
    assert Ray3D(Point3D(0, 0, 0), Point3D(0, 4, 0)).intersection(Ray3D(Point3D(0, 1, 1), Point3D(0, -1, 1))) == []


def test_is_parallel():
    # 创建 3D 点对象 p1, p2, p3
    p1 = Point3D(0, 0, 0)
    p2 = Point3D(1, 1, 1)
    p3 = Point3D(x1, x1, x1)

    # 创建 2D 线对象 l2 和 l2_1
    l2 = Line(Point(x1, x1), Point(y1, y1))
    l2_1 = Line(Point(x1, x1), Point(x1, 1 + x1))

    # 进行多个断言，检查不同线的平行性关系
    assert Line.is_parallel(Line(Point(0, 0), Point(1, 1)), l2)
    assert Line.is_parallel(l2, Line(Point(x1, x1), Point(x1, 1 + x1))) is False
    assert Line.is_parallel(l2, l2.parallel_line(Point(-x1, x1)))
    assert Line.is_parallel(l2_1, l2_1.parallel_line(Point(0, 0)))
    assert Line3D(p1, p2).is_parallel(Line3D(p1, p2))  # same as in 2D
    assert Line3D(Point3D(4, 0, 1), Point3D(0, 4, 1)).is_parallel(Line3D(Point3D(0, 0, 1), Point3D(4, 4, 1))) is False
    assert Line3D(p1, p2).parallel_line(p3) == Line3D(Point3D(x1, x1, x1),
                                                      Point3D(x1 + 1, x1 + 1, x1 + 1))
    assert Line3D(p1, p2).parallel_line(p3.args) == \
           Line3D(Point3D(x1, x1, x1), Point3D(x1 + 1, x1 + 1, x1 + 1))
    # 使用 assert 断言来验证两条三维空间中的直线是否不平行
    assert Line3D(Point3D(4, 0, 1), Point3D(0, 4, 1)).is_parallel(Line3D(Point3D(0, 0, 1), Point3D(4, 4, 1))) is False
def test_is_perpendicular():
    p1 = Point(0, 0)  # 创建二维点 p1
    p2 = Point(1, 1)  # 创建二维点 p2

    l1 = Line(p1, p2)  # 使用 p1 和 p2 创建线段 l1
    l2 = Line(Point(x1, x1), Point(y1, y1))  # 使用给定的点创建线段 l2
    l1_1 = Line(p1, Point(-x1, x1))  # 使用 p1 和另一个点创建线段 l1_1
    # 2D
    assert Line.is_perpendicular(l1, l1_1)  # 检查 l1 和 l1_1 是否垂直
    assert Line.is_perpendicular(l1, l2) is False  # 检查 l1 和 l2 是否不垂直
    p = l1.random_point()  # 获取 l1 的随机点 p
    assert l1.perpendicular_segment(p) == p  # 检查 p 是否为 l1 的垂直线段
    # 3D
    assert Line3D.is_perpendicular(Line3D(Point3D(0, 0, 0), Point3D(1, 0, 0)),
                                   Line3D(Point3D(0, 0, 0), Point3D(0, 1, 0))) is True  # 检查两个三维线是否垂直
    assert Line3D.is_perpendicular(Line3D(Point3D(0, 0, 0), Point3D(1, 0, 0)),
                                   Line3D(Point3D(0, 1, 0), Point3D(1, 1, 0))) is False  # 检查两个三维线是否不垂直
    assert Line3D.is_perpendicular(Line3D(Point3D(0, 0, 0), Point3D(1, 1, 1)),
                                   Line3D(Point3D(x1, x1, x1), Point3D(y1, y1, y1))) is False  # 检查两个三维线是否不垂直


def test_is_similar():
    p1 = Point(2000, 2000)  # 创建二维点 p1
    p2 = p1.scale(2, 2)  # 缩放 p1 生成点 p2

    r1 = Ray3D(Point3D(1, 1, 1), Point3D(1, 0, 0))  # 创建三维射线 r1
    r2 = Ray(Point(0, 0), Point(0, 1))  # 创建二维射线 r2

    s1 = Segment(Point(0, 0), p1)  # 使用点创建线段 s1

    assert s1.is_similar(Segment(p1, p2))  # 检查 s1 和由 p1, p2 创建的线段是否相似
    assert s1.is_similar(r2) is False  # 检查 s1 和 r2 是否不相似
    assert r1.is_similar(Line3D(Point3D(1, 1, 1), Point3D(1, 0, 0))) is True  # 检查 r1 是否与给定三维线相似
    assert r1.is_similar(Line3D(Point3D(0, 0, 0), Point3D(0, 1, 0))) is False  # 检查 r1 是否与另一三维线不相似


def test_length():
    s2 = Segment3D(Point3D(x1, x1, x1), Point3D(y1, y1, y1))  # 使用给定点创建三维线段 s2
    assert Line(Point(0, 0), Point(1, 1)).length is oo  # 检查二维线段长度是否为无穷大
    assert s2.length == sqrt(3) * sqrt((x1 - y1) ** 2)  # 检查三维线段 s2 的长度是否正确
    assert Line3D(Point3D(0, 0, 0), Point3D(1, 1, 1)).length is oo  # 检查三维线长度是否为无穷大


def test_projection():
    p1 = Point(0, 0)  # 创建二维点 p1
    p2 = Point3D(0, 0, 0)  # 创建三维点 p2
    p3 = Point(-x1, x1)  # 创建二维点 p3

    l1 = Line(p1, Point(1, 1))  # 使用 p1 和另一点创建线段 l1
    l2 = Line3D(Point3D(0, 0, 0), Point3D(1, 0, 0))  # 使用两个三维点创建三维线 l2
    l3 = Line3D(p2, Point3D(1, 1, 1))  # 使用 p2 和另一三维点创建三维线 l3

    r1 = Ray(Point(1, 1), Point(2, 2))  # 创建二维射线 r1

    s1 = Segment(Point2D(0, 0), Point2D(0, 1))  # 使用二维点创建二维线段 s1
    s2 = Segment(Point2D(1, 0), Point2D(2, 1/2))  # 使用二维点创建二维线段 s2

    assert Line(Point(x1, x1), Point(y1, y1)).projection(Point(y1, y1)) == Point(y1, y1)  # 检查二维线段投影的正确性
    assert Line(Point(x1, x1), Point(x1, 1 + x1)).projection(Point(1, 1)) == Point(x1, 1)  # 检查二维线段投影的正确性
    assert Segment(Point(-2, 2), Point(0, 4)).projection(r1) == Segment(Point(-1, 3), Point(0, 4))  # 检查二维线段投影的正确性
    assert Segment(Point(0, 4), Point(-2, 2)).projection(r1) == Segment(Point(0, 4), Point(-1, 3))  # 检查二维线段投影的正确性
    assert s2.projection(s1) == EmptySet  # 检查二维线段投影的正确性
    assert l1.projection(p3) == p1  # 检查二维线段投影的正确性
    assert l1.projection(Ray(p1, Point(-1, 5))) == Ray(Point(0, 0), Point(2, 2))  # 检查二维线段投影的正确性
    assert l1.projection(Ray(p1, Point(-1, 1))) == p1  # 检查二维线段投影的正确性
    assert r1.projection(Ray(Point(1, 1), Point(-1, -1))) == Point(1, 1)  # 检查二维射线投影的正确性
    assert r1.projection(Ray(Point(0, 4), Point(-1, -5))) == Segment(Point(1, 1), Point(2, 2))  # 检查二维射线投影的正确性
    assert r1.projection(Segment(Point(-1, 5), Point(-5, -10))) == Segment(Point(1, 1), Point(2, 2))  # 检查二维射线投影的正确性
    assert r1.projection(Ray(Point(1, 1), Point(-1, -1))) == Point(1, 1)  # 检查二维射线投影的正确性
    assert r1.projection(Ray(Point(0, 4), Point(-1, -5))) == Segment(Point(1, 1), Point(2, 2))  # 检查二维射线投影的正确性
    # 断言，验证r1投影到给定线段的结果是否符合预期
    assert r1.projection(Segment(Point(-1, 5), Point(-5, -10))) == Segment(Point(1, 1), Point(2, 2))
    
    # 断言，验证l3投影到给定射线的结果是否符合预期
    assert l3.projection(Ray3D(p2, Point3D(-1, 5, 0))) == Ray3D(Point3D(0, 0, 0), Point3D(Rational(4, 3), Rational(4, 3), Rational(4, 3)))
    
    # 断言，验证l3投影到另一个给定射线的结果是否符合预期
    assert l3.projection(Ray3D(p2, Point3D(-1, 1, 1))) == Ray3D(Point3D(0, 0, 0), Point3D(Rational(1, 3), Rational(1, 3), Rational(1, 3)))
    
    # 断言，验证l2投影到给定点的结果是否符合预期
    assert l2.projection(Point3D(5, 5, 0)) == Point3D(5, 0)
    
    # 断言，验证l2投影到给定直线的结果是否与l2相等
    assert l2.projection(Line3D(Point3D(0, 1, 0), Point3D(1, 1, 0))).equals(l2)
def test_perpendicular_line():
    # 测试垂直线的功能
    # 3D情况下，需要选择特定的直角
    p1, p2, p3 = Point(0, 0, 0), Point(2, 3, 4), Point(-2, 2, 0)
    # 创建线对象，连接 p1 和 p2
    l = Line(p1, p2)
    # 获取从 p1 到 p3 的垂直线
    p = l.perpendicular_line(p3)
    # 断言 p 的起点是 p3
    assert p.p1 == p3
    # 断言 p 的终点在 l 上
    assert p.p2 in l

    # 2D情况下，不需要特别选择
    p1, p2, p3 = Point(0, 0), Point(2, 3), Point(-2, 2)
    # 创建线对象，连接 p1 和 p2
    l = Line(p1, p2)
    # 获取从 p1 到 p3 的垂直线
    p = l.perpendicular_line(p3)
    # 断言 p 的起点是 p3
    assert p.p1 == p3
    # 断言 p 的方向是从 l 到 p3
    assert p.direction.unit == (p3 - l.projection(p3)).unit


def test_perpendicular_bisector():
    # 测试垂直平分线的功能
    s1 = Segment(Point(0, 0), Point(1, 1))
    # 创建一个具有特定斜率的线对象
    aline = Line(Point(S.Half, S.Half), Point(Rational(3, 2), Rational(-1, 2)))
    # 在指定直线上创建一个点
    on_line = Segment(Point(S.Half, S.Half), Point(Rational(3, 2), Rational(-1, 2))).midpoint

    # 断言 s1 的垂直平分线与 aline 相等
    assert s1.perpendicular_bisector().equals(aline)
    # 断言 s1 关于 on_line 的垂直平分线与指定的线段相等
    assert s1.perpendicular_bisector(on_line).equals(Segment(s1.midpoint, on_line))
    # 断言 s1 关于 on_line + (1, 0) 的垂直平分线与 aline 相等
    assert s1.perpendicular_bisector(on_line + (1, 0)).equals(aline)


def test_raises():
    d, e = symbols('a,b', real=True)
    s = Segment((d, 0), (e, 0))

    # 断言以下操作会引发特定的异常
    raises(TypeError, lambda: Line((1, 1), 1))
    raises(ValueError, lambda: Line(Point(0, 0), Point(0, 0)))
    raises(Undecidable, lambda: Point(2 * d, 0) in s)
    raises(ValueError, lambda: Ray3D(Point(1.0, 1.0)))
    raises(ValueError, lambda: Line3D(Point3D(0, 0, 0), Point3D(0, 0, 0)))
    raises(TypeError, lambda: Line3D((1, 1), 1))
    raises(ValueError, lambda: Line3D(Point3D(0, 0, 0)))
    raises(TypeError, lambda: Ray((1, 1), 1))
    raises(GeometryError, lambda: Line(Point(0, 0), Point(1, 0))
           .projection(Circle(Point(0, 0), 1)))


def test_ray_generation():
    # 测试射线生成的功能
    assert Ray((1, 1), angle=pi / 4) == Ray((1, 1), (2, 2))
    assert Ray((1, 1), angle=pi / 2) == Ray((1, 1), (1, 2))
    assert Ray((1, 1), angle=-pi / 2) == Ray((1, 1), (1, 0))
    assert Ray((1, 1), angle=-3 * pi / 2) == Ray((1, 1), (1, 2))
    assert Ray((1, 1), angle=5 * pi / 2) == Ray((1, 1), (1, 2))
    assert Ray((1, 1), angle=5.0 * pi / 2) == Ray((1, 1), (1, 2))
    assert Ray((1, 1), angle=pi) == Ray((1, 1), (0, 1))
    assert Ray((1, 1), angle=3.0 * pi) == Ray((1, 1), (0, 1))
    assert Ray((1, 1), angle=4.0 * pi) == Ray((1, 1), (2, 1))
    assert Ray((1, 1), angle=0) == Ray((1, 1), (2, 1))
    assert Ray((1, 1), angle=4.05 * pi) == Ray(Point(1, 1),
                                               Point(2, -sqrt(5) * sqrt(2 * sqrt(5) + 10) / 4 - sqrt(
                                                   2 * sqrt(5) + 10) / 4 + 2 + sqrt(5)))
    assert Ray((1, 1), angle=4.02 * pi) == Ray(Point(1, 1),
                                               Point(2, 1 + tan(4.02 * pi)))
    assert Ray((1, 1), angle=5) == Ray((1, 1), (2, 1 + tan(5)))

    assert Ray3D((1, 1, 1), direction_ratio=[4, 4, 4]) == Ray3D(Point3D(1, 1, 1), Point3D(5, 5, 5))
    assert Ray3D((1, 1, 1), direction_ratio=[1, 2, 3]) == Ray3D(Point3D(1, 1, 1), Point3D(2, 3, 4))
    # 使用断言验证 Ray3D 类的行为：从起始点 (1, 1, 1) 出发，沿着方向比率 [1, 1, 1] 的方向，创建的 Ray3D 对象
    # 应该与另一个 Ray3D 对象 Point3D(1, 1, 1) 到 Point3D(2, 2, 2) 相等。
    assert Ray3D((1, 1, 1), direction_ratio=[1, 1, 1]) == Ray3D(Point3D(1, 1, 1), Point3D(2, 2, 2))
# 定义测试函数，用于检查 issue 7814
def test_issue_7814():
    # 创建一个圆对象，圆心为 Point(x, 0)，半径为 y
    circle = Circle(Point(x, 0), y)
    # 创建一条直线对象，通过 Point(k, z) 点，且斜率为 0
    line = Line(Point(k, z), slope=0)
    # 计算 _s，即 sqrt((y - z)*(y + z))
    _s = sqrt((y - z)*(y + z))
    # 断言直线和圆的交点应为 [Point2D(x + _s, z), Point2D(x - _s, z)]
    assert line.intersection(circle) == [Point2D(x + _s, z), Point2D(x - _s, z)]


# 定义测试函数，用于检查 issue 2941
def test_issue_2941():
    # 定义内部函数 _check
    def _check():
        # 对于所有的函数对 (f, g)，每个函数为 Line、Ray、Segment 的组合
        for f, g in cartes(*[(Line, Ray, Segment)] * 2):
            # 创建线段 l1 = f(a, b)，线段 l2 = g(c, d)
            l1 = f(a, b)
            l2 = g(c, d)
            # 断言 l1 和 l2 的交点应该与 l2 和 l1 的交点相同
            assert l1.intersection(l2) == l2.intersection(l1)
    
    # 设置第一组参数 c, d 和 a, b，这两组参数用于 _check 函数的测试
    c, d = (-2, -2), (-2, 0)
    a, b = (0, 0), (1, 1)
    # 执行 _check 函数进行测试
    _check()
    
    # 设置第二组参数 c, d 和 a, b，这两组参数用于 _check 函数的测试
    c, d = (-2, -3), (-2, 0)
    # 再次执行 _check 函数进行测试
    _check()


# 定义测试函数，用于检查 parameter_value 方法
def test_parameter_value():
    # 创建符号变量 t
    t = Symbol('t')
    # 创建两个点 p1 和 p2
    p1, p2 = Point(0, 1), Point(5, 6)
    # 创建直线对象 l，通过 p1 和 p2 两点
    l = Line(p1, p2)
    # 断言直线 l 在点 (5, 6) 处的参数值为 {t: 1}
    assert l.parameter_value((5, 6), t) == {t: 1}
    # 使用 lambda 表达式断言在 (0, 0) 点处调用 parameter_value 方法会引发 ValueError
    raises(ValueError, lambda: l.parameter_value((0, 0), t))


# 定义测试函数，用于检查 Line3D 对象的 bisectors 方法
def test_bisectors():
    # 创建两条三维直线 r1 和 r2
    r1 = Line3D(Point3D(0, 0, 0), Point3D(1, 0, 0))
    r2 = Line3D(Point3D(0, 0, 0), Point3D(0, 1, 0))
    # 计算 r1 和 r2 的角平分线
    bisections = r1.bisectors(r2)
    # 断言计算结果与预期相同
    assert bisections == [Line3D(Point3D(0, 0, 0), Point3D(1, 1, 0)),
                          Line3D(Point3D(0, 0, 0), Point3D(1, -1, 0))]
    
    # 设置预期的角平分线列表
    ans = [Line3D(Point3D(0, 0, 0), Point3D(1, 0, 1)),
           Line3D(Point3D(0, 0, 0), Point3D(-1, 0, 1))]
    # 定义两条二维线段的起始和终止点
    l1 = (0, 0, 0), (0, 0, 1)
    l2 = (0, 0), (1, 0)
    # 对于所有的 (a, b) 函数对，分别为 Line、Segment、Ray 的组合
    for a, b in cartes((Line, Segment, Ray), repeat=2):
        # 断言 a(l1).bisectors(b(l2)) 的结果与预期相同
        assert a(*l1).bisectors(b(*l2)) == ans


# 定义测试函数，用于检查 issue 8615
def test_issue_8615():
    # 创建两条三维直线 a 和 b
    a = Line3D(Point3D(6, 5, 0), Point3D(6, -6, 0))
    b = Line3D(Point3D(6, -1, 19/10), Point3D(6, -1, 0))
    # 断言直线 a 和 b 的交点为 [Point3D(6, -1, 0)]
    assert a.intersection(b) == [Point3D(6, -1, 0)]


# 定义测试函数，用于检查 issue 12598
def test_issue_12598():
    # 创建两条射线 r1 和 r2
    r1 = Ray(Point(0, 1), Point(0.98, 0.79).n(2))
    r2 = Ray(Point(0, 0), Point(0.71, 0.71).n(2))
    # 断言 r1 和 r2 的交点的字符串表示为 'Point2D(0.82, 0.82)'
    assert str(r1.intersection(r2)[0]) == 'Point2D(0.82, 0.82)'
    
    # 创建一条直线 l1 和线段 l2
    l1 = Line((0, 0), (1, 1))
    l2 = Segment((-1, 1), (0, -1)).n(2)
    # 断言 l1 和 l2 的交点的字符串表示为 'Point2D(-0.33, -0.33)'
    assert str(l1.intersection(l2)[0]) == 'Point2D(-0.33, -0.33)'
    
    # 更新线段 l2 的端点，使其无交点
    l2 = Segment((-1, 1), (-1/2, 1/2)).n(2)
    # 断言 l1 和 l2 无交点
    assert not l1.intersection(l2)
```