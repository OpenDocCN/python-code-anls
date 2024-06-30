# `D:\src\scipysrc\sympy\sympy\geometry\tests\test_point.py`

```
# 导入 sympy 库的基本模块和函数
from sympy.core.basic import Basic
# 导入 sympy 库的数值常数和符号
from sympy.core.numbers import (I, Rational, pi)
# 导入 sympy 库的参数求值函数
from sympy.core.parameters import evaluate
# 导入 sympy 库的单例对象
from sympy.core.singleton import S
# 导入 sympy 库的符号类
from sympy.core.symbol import Symbol
# 导入 sympy 库的符号化函数
from sympy.core.sympify import sympify
# 导入 sympy 库的数学函数，如平方根
from sympy.functions.elementary.miscellaneous import sqrt
# 导入 sympy 几何模块中的几何对象：直线、点（2D和3D）、三维空间中的直线和平面
from sympy.geometry import Line, Point, Point2D, Point3D, Line3D, Plane
# 导入 sympy 几何模块中的几何变换函数：旋转、缩放、平移
from sympy.geometry.entity import rotate, scale, translate, GeometryEntity
# 导入 sympy 的矩阵模块
from sympy.matrices import Matrix
# 导入 sympy 的集合运算函数：子集、排列、笛卡尔积
from sympy.utilities.iterables import subsets, permutations, cartes
# 导入 sympy 的其他实用函数：Undecidable
from sympy.utilities.misc import Undecidable
# 导入 sympy 的测试框架函数：raises（抛出异常检测）、warns（警告检测）
from sympy.testing.pytest import raises, warns


# 定义测试函数 test_point()
def test_point():
    # 定义实数符号 x, y, x1, x2, y1, y2
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    x1 = Symbol('x1', real=True)
    x2 = Symbol('x2', real=True)
    y1 = Symbol('y1', real=True)
    y2 = Symbol('y2', real=True)
    # 定义有理数 S.Half 作为变量 half
    half = S.Half
    # 创建点对象 p1, p2, p3, p4, p5
    p1 = Point(x1, x2)
    p2 = Point(y1, y2)
    p3 = Point(0, 0)
    p4 = Point(1, 1)
    p5 = Point(0, 1)
    # 创建直线对象 line，通过给定一点和斜率来定义
    line = Line(Point(1, 0), slope=1)

    # 断言：p1 在 p1 中
    assert p1 in p1
    # 断言：p1 不在 p2 中
    assert p1 not in p2
    # 断言：p2 的 y 坐标等于 y2
    assert p2.y == y2
    # 断言：点 p3 和 p4 的中点等于 p4
    assert (p3 + p4) == p4
    # 断言：点 p2 减去点 p1 等于由坐标差构成的新点
    assert (p2 - p1) == Point(y1 - x1, y2 - x2)
    # 断言：点 p2 的相反点等于由坐标的相反数构成的新点
    assert -p2 == Point(-y1, -y2)
    # 引发异常断言：创建点时，只提供一个参数
    raises(TypeError, lambda: Point(1))
    # 引发异常断言：创建点时，使用列表作为参数
    raises(ValueError, lambda: Point([1]))
    # 引发异常断言：创建点时，使用复数作为参数
    raises(ValueError, lambda: Point(3, I))
    # 引发异常断言：创建点时，使用复数作为参数
    raises(ValueError, lambda: Point(2*I, I))
    # 引发异常断言：创建点时，使用复数作为参数
    raises(ValueError, lambda: Point(3 + I, I))

    # 断言：创建点时使用浮点数参数
    assert Point(34.05, sqrt(3)) == Point(Rational(681, 20), sqrt(3))
    # 断言：计算点 p3 和 p4 的中点为 (1/2, 1/2)
    assert Point.midpoint(p3, p4) == Point(half, half)
    # 断言：计算点 p1 和 p4 的中点
    assert Point.midpoint(p1, p4) == Point(half + half*x1, half + half*x2)
    # 断言：点 p2 和自身的中点等于点 p2
    assert Point.midpoint(p2, p2) == p2
    # 断言：点 p2 调用 midpont 方法和自身的中点等于点 p2
    assert p2.midpoint(p2) == p2
    # 断言：点 p1 的原点为 (0, 0)
    assert p1.origin == Point(0, 0)

    # 断言：计算点 p3 和 p4 之间的距离为 sqrt(2)
    assert Point.distance(p3, p4) == sqrt(2)
    # 断言：计算点 p1 和自身的距离为 0
    assert Point.distance(p1, p1) == 0
    # 断言：计算点 p3 和 p2 之间的距离
    assert Point.distance(p3, p2) == sqrt(p2.x**2 + p2.y**2)
    # 引发异常断言：计算点 p1 和数字 0 之间的距离
    raises(TypeError, lambda: Point.distance(p1, 0))
    # 引发异常断言：计算点 p1 和几何实体之间的距离
    raises(TypeError, lambda: Point.distance(p1, GeometryEntity()))

    # 断言：点 p1 到直线的距离应该与直线到点 p1 的距离相等
    assert p1.distance(line) == line.distance(p1)
    # 断言：点 p4 到直线的距离应该与直线到点 p4 的距离相等
    assert p4.distance(line) == line.distance(p4)

    # 断言：点 p4 和 p3 之间的曼哈顿距离为 2
    assert Point.taxicab_distance(p4, p3) == 2

    # 断言：点 p4 和 p5 之间的坎贝拉距离为 1
    assert Point.canberra_distance(p4, p5) == 1
    # 引发异常断言：计算点 p3 和 p3 之间的坎贝拉距离
    raises(ValueError, lambda: Point.canberra_distance(p3, p3))

    # 创建点 p1_1, p1_2, p1_3
    p1_1 = Point(x1, x1)
    p1_2 = Point(y2, y2)
    p1_3 = Point(x1 + 1, x1)
    # 断言：检查点 p3 是否共线
    assert Point.is_collinear(p3)

    # 引发警告断言：检查点 p3 和维度为 4 的点是否共线
    with warns(UserWarning, test_stacklevel=False):
        assert Point.is_collinear(p3, Point(p3, dim=4))
    # 断言：检查点 p3 自身是否共线
    assert p3.is_collinear()
    # 断言：检查点 p3 和 p4 是否共线
    assert Point.is_collinear(p3, p4)
    # 断言：检查点 p3, p1_1, p1_2 是否共线
    assert Point.is_collinear(p3, p4, p1_1, p1_2)
    # 断言：检查点 p3, p1_1, p1_3 是否共线，预期为 False
    assert Point.is_collinear(p3, p4, p1_1, p1_3) is False
    # 断言：检查点 p3, p3, p4, p5 是否共线，预期为 False
    assert Point.is_collinear
    # 使用警告上下文管理器确保 UserWarning 不会引发测试失败
    with warns(UserWarning, test_stacklevel=False):
        # 断言两个 Point 对象的交点，预期结果是包含 Point(0, 0, 0) 的列表
        assert Point.intersection(Point(0, 0, 0), Point(0, 0)) == [Point(0, 0, 0)]

    # 创建一个正数约束的符号变量 'x'
    x_pos = Symbol('x', positive=True)
    # 创建几个 Point 对象，使用 x_pos 和其他参数
    p2_1 = Point(x_pos, 0)
    p2_2 = Point(0, x_pos)
    p2_3 = Point(-x_pos, 0)
    p2_4 = Point(0, -x_pos)
    p2_5 = Point(x_pos, 5)
    # 断言这些点是否共圆
    assert Point.is_concyclic(p2_1)
    assert Point.is_concyclic(p2_1, p2_2)
    assert Point.is_concyclic(p2_1, p2_2, p2_3, p2_4)
    # 对点的排列进行排列组合，验证是否不共圆
    for pts in permutations((p2_1, p2_2, p2_3, p2_5)):
        assert Point.is_concyclic(*pts) is False
    # 断言三个点不共圆
    assert Point.is_concyclic(p4, p4 * 2, p4 * 3) is False
    # 断言 Point(0, 0) 与其他三个点不共圆
    assert Point(0, 0).is_concyclic((1, 1), (2, 2), (2, 1)) is False
    # 断言四个不同维度的点不共圆
    assert Point.is_concyclic(Point(0, 0, 0, 0), Point(1, 0, 0, 0), Point(1, 1, 0, 0), Point(1, 1, 1, 0)) is False

    # 断言一个点与自身的标量倍乘结果为真
    assert p1.is_scalar_multiple(p1)
    # 断言一个点与其两倍的标量倍乘结果为真
    assert p1.is_scalar_multiple(2*p1)
    # 断言一个点与另一个不同点的标量倍乘结果为假
    assert not p1.is_scalar_multiple(p2)
    # 断言两个给定点是否是标量倍乘关系
    assert Point.is_scalar_multiple(Point(1, 1), (-1, -1))
    assert Point.is_scalar_multiple(Point(0, 0), (0, -1))
    # 测试当无法确定标量倍乘关系时引发 Undecidable 异常
    raises(Undecidable, lambda: Point.is_scalar_multiple(Point(sympify("x1%y1"), sympify("x2%y2")), Point(0, 1)))

    # 断言 Point(0, 1) 的正交方向是 Point(1, 0)
    assert Point(0, 1).orthogonal_direction == Point(1, 0)
    assert Point(1, 0).orthogonal_direction == Point(0, 1)

    # 断言 p1 的 is_zero 属性为 None
    assert p1.is_zero is None
    # 断言 p3 的 is_zero 属性为 True
    assert p3.is_zero
    # 断言 p4 的 is_zero 属性为 False
    assert p4.is_zero is False
    # 断言 p1 的 is_nonzero 属性为 None
    assert p1.is_nonzero is None
    # 断言 p3 的 is_nonzero 属性为 False
    assert p3.is_nonzero is False
    # 断言 p4 的 is_nonzero 属性为 True
    assert p4.is_nonzero

    # 断言 p4 缩放后与期望的 Point(2, 3) 相等
    assert p4.scale(2, 3) == Point(2, 3)
    # 断言 p3 缩放后与自身相等
    assert p3.scale(2, 3) == p3

    # 断言 p4 绕 Point(0.5, 0.5) 旋转 π 弧度后与 p3 相等
    assert p4.rotate(pi, Point(0.5, 0.5)) == p3
    # 断言 p1 与 p2 的 __radd__ 运算结果与 p1 和 p2 的中点再扩展为 2 倍后相等
    assert p1.__radd__(p2) == p1.midpoint(p2).scale(2, 2)
    # 断言 (-p3) 与 p4 的 __rsub__ 运算结果与 p3 和 p4 的中点再扩展为 2 倍后相等
    assert (-p3).__rsub__(p4) == p3.midpoint(p4).scale(2, 2)

    # 断言 p4 乘以 5 等于 Point(5, 5)
    assert p4 * 5 == Point(5, 5)
    # 断言 p4 除以 5 等于 Point(0.2, 0.2)
    assert p4 / 5 == Point(0.2, 0.2)
    # 断言 5 乘以 p4 等于 Point(5, 5)
    assert 5 * p4 == Point(5, 5)

    # 测试当尝试将 Point(0, 0) 加上 10 时引发 ValueError 异常
    raises(ValueError, lambda: Point(0, 0) + 10)

    # 断言 Point 的差应被简化为 Point(0, -1)
    assert Point(x*(x - 1), y) - Point(x**2 - x, y + 1) == Point(0, -1)

    # 定义两个有理数 a 和 b
    a, b = S.Half, Rational(1, 3)
    # 断言 Point(a, b) 的 evalf(2) 结果与 Point(a.n(2), b.n(2), evaluate=False) 相等
    assert Point(a, b).evalf(2) == \
        Point(a.n(2), b.n(2), evaluate=False)
    # 测试当尝试对 Point(1, 2) 执行无效运算时引发 ValueError 异常
    raises(ValueError, lambda: Point(1, 2) + 1)

    # 测试投影函数 project
    assert Point.project((0, 1), (1, 0)) == Point(0, 0)
    assert Point.project((1, 1), (1, 0)) == Point(1, 0)
    # 测试当输入参数不是 Point 对象时引发 ValueError 异常
    raises(ValueError, lambda: Point.project(p1, Point(0, 0)))

    # 测试各种变换函数
    p = Point(1, 0)
    # 断言 p 绕原点逆时针旋转 π/2 弧度后结果为 Point(0, 1)
    assert p.rotate(pi/2) == Point(0, 1)
    # 断言 p 绕自身旋转 π/2 弧度后结果仍为 p
    assert p.rotate(pi/2, p) == p
    p = Point(1, 1)
    # 断言 p 缩放后结果为 Point(2, 3)
    assert p.scale(2, 3) == Point(2, 3)
    # 断言 p 平移后结果为 Point(2, 3)
    assert p.translate(1, 2) == Point(2, 3)
    # 断言 p 在 x 方向平移后结果为 Point(2, 1)
    assert p.translate(1) == Point(2, 1)
    # 断言 p 在 y 方向平移后结果为 Point(1, 2)
    assert p.translate(y=1) == Point(1, 2)
    # 断言 p 沿自身参数进行平移后结果为 Point(2, 2)
    assert p.translate(*p.args) == Point(2, 2)

    # 测试不合法的变换输入时是否引发 ValueError 异常
    raises(ValueError, lambda: p3.transform(p3))
    raises(ValueError, lambda: p.transform(Matrix([[1, 0
    # 测试 Point 类的静态方法 affine_rank() 的结果是否等于 -1
    assert Point.affine_rank() == -1
# 定义一个测试函数，用于测试 Point3D 类的各种方法和属性
def test_point3D():
    # 定义多个实数符号变量
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    x1 = Symbol('x1', real=True)
    x2 = Symbol('x2', real=True)
    x3 = Symbol('x3', real=True)
    y1 = Symbol('y1', real=True)
    y2 = Symbol('y2', real=True)
    y3 = Symbol('y3', real=True)
    # 定义 S.Half 为 1/2
    half = S.Half
    # 创建多个 Point3D 对象，每个对象包含三个坐标参数
    p1 = Point3D(x1, x2, x3)
    p2 = Point3D(y1, y2, y3)
    p3 = Point3D(0, 0, 0)
    p4 = Point3D(1, 1, 1)
    p5 = Point3D(0, 1, 2)

    # 测试点是否包含于自身
    assert p1 in p1
    # 测试点是否不包含于另一个点
    assert p1 not in p2
    # 测试点的某个坐标属性
    assert p2.y == y2
    # 测试点的加法操作
    assert (p3 + p4) == p4
    # 测试点的减法操作
    assert (p2 - p1) == Point3D(y1 - x1, y2 - x2, y3 - x3)
    # 测试点的取负操作
    assert -p2 == Point3D(-y1, -y2, -y3)

    # 测试 Point 类的实例化和相等性
    assert Point(34.05, sqrt(3)) == Point(Rational(681, 20), sqrt(3))
    # 测试 Point3D 类的 midpoint 方法
    assert Point3D.midpoint(p3, p4) == Point3D(half, half, half)
    # 测试 Point3D 类的 midpoint 方法
    assert Point3D.midpoint(p1, p4) == Point3D(half + half*x1, half + half*x2,
                                         half + half*x3)
    # 测试 Point3D 类的 midpoint 方法，输入相同点时返回该点自身
    assert Point3D.midpoint(p2, p2) == p2
    # 测试 Point3D 实例的 midpoint 方法，与类方法等效
    assert p2.midpoint(p2) == p2

    # 测试 Point3D 类的 distance 方法
    assert Point3D.distance(p3, p4) == sqrt(3)
    # 测试 Point3D 类的 distance 方法，两点相同时距离为0
    assert Point3D.distance(p1, p1) == 0
    # 测试 Point3D 类的 distance 方法，计算两点间的欧氏距离
    assert Point3D.distance(p3, p2) == sqrt(p2.x**2 + p2.y**2 + p2.z**2)

    # 创建多个具有相同坐标的 Point3D 对象
    p1_1 = Point3D(x1, x1, x1)
    p1_2 = Point3D(y2, y2, y2)
    p1_3 = Point3D(x1 + 1, x1, x1)
    # 测试 Point3D 类的 are_collinear 静态方法，检查共线性
    Point3D.are_collinear(p3)
    assert Point3D.are_collinear(p3, p4)
    assert Point3D.are_collinear(p3, p4, p1_1, p1_2)
    assert Point3D.are_collinear(p3, p4, p1_1, p1_3) is False
    assert Point3D.are_collinear(p3, p3, p4, p5) is False

    # 测试 Point3D 类的 intersection 方法，两点相交返回空列表
    assert p3.intersection(Point3D(0, 0, 0)) == [p3]
    assert p3.intersection(p4) == []

    # 测试 Point3D 类的乘法和除法操作
    assert p4 * 5 == Point3D(5, 5, 5)
    assert p4 / 5 == Point3D(0.2, 0.2, 0.2)
    assert 5 * p4 == Point3D(5, 5, 5)

    # 测试 Point3D 类的异常处理
    raises(ValueError, lambda: Point3D(0, 0, 0) + 10)

    # 测试点对象的坐标属性
    assert p1.coordinates == (x1, x2, x3)
    assert p2.coordinates == (y1, y2, y3)
    assert p3.coordinates == (0, 0, 0)
    assert p4.coordinates == (1, 1, 1)
    assert p5.coordinates == (0, 1, 2)
    assert p5.x == 0
    assert p5.y == 1
    assert p5.z == 2

    # 测试点之间的坐标差异，应简化为相应差值
    assert Point3D(x*(x - 1), y, 2) - Point3D(x**2 - x, y + 1, 1) == \
        Point3D(0, -1, 1)

    # 测试 Point3D 类的 evalf 方法
    a, b, c = S.Half, Rational(1, 3), Rational(1, 4)
    assert Point3D(a, b, c).evalf(2) == \
        Point(a.n(2), b.n(2), c.n(2), evaluate=False)
    raises(ValueError, lambda: Point3D(1, 2, 3) + 1)

    # 测试点的平移和缩放变换
    p = Point3D(1, 1, 1)
    assert p.scale(2, 3) == Point3D(2, 3, 1)
    assert p.translate(1, 2) == Point3D(2, 3, 1)
    assert p.translate(1) == Point3D(2, 1, 1)
    assert p.translate(z=1) == Point3D(1, 1, 2)
    assert p.translate(*p.args) == Point3D(2, 2, 2)

    # 测试 Point3D 类的 __new__ 方法
    assert Point3D(0.1, 0.2, evaluate=False, on_morph='ignore').args[0].is_Float

    # 测试 length 属性是否正确返回
    assert p.length == 0
    assert p1_1.length == 0
    assert p1_2.length == 0

    # 测试 are_collinear 方法的类型错误处理
    raises(TypeError, lambda: Point3D.are_collinear(p, x))
    # 使用 lambda 函数测试 Point3D 类中的 are_collinear 方法是否引发 TypeError 异常

    # Test are_coplanar
    assert Point.are_coplanar()
    # 检查 Point 类中的 are_coplanar 方法是否返回 True
    assert Point.are_coplanar((1, 2, 0), (1, 2, 0), (1, 3, 0))
    # 检查给定三个点是否共面，预期结果为 True
    assert Point.are_coplanar((1, 2, 0), (1, 2, 3))
    # 检查给定两个点是否共面，预期结果为 False
    with warns(UserWarning, test_stacklevel=False):
        raises(ValueError, lambda: Point2D.are_coplanar((1, 2), (1, 2, 3)))
    # 使用 lambda 函数测试 Point2D 类中的 are_coplanar 方法是否引发 ValueError 异常，并忽略 UserWarning
    assert Point3D.are_coplanar((1, 2, 0), (1, 2, 3))
    # 检查给定三个点是否共面，预期结果为 True
    assert Point.are_coplanar((0, 0, 0), (1, 1, 0), (1, 1, 1), (1, 2, 1)) is False
    # 检查给定四个点是否共面，预期结果为 False
    planar2 = Point3D(1, -1, 1)
    planar3 = Point3D(-1, 1, 1)
    assert Point3D.are_coplanar(p, planar2, planar3) == True
    # 检查给定四个点是否共面，预期结果为 True
    assert Point3D.are_coplanar(p, planar2, planar3, p3) == False
    # 检查给定四个点是否共面，预期结果为 False
    assert Point.are_coplanar(p, planar2)
    # 检查给定两个点是否共面，预期结果为 True
    planar2 = Point3D(1, 1, 2)
    planar3 = Point3D(1, 1, 3)
    assert Point3D.are_coplanar(p, planar2, planar3)  # line, not plane
    # 检查给定三个点是否共面，预期结果为 False（这是一条线而不是一个平面）
    plane = Plane((1, 2, 1), (2, 1, 0), (3, 1, 2))
    assert Point.are_coplanar(*[plane.projection(((-1)**i, i)) for i in range(4)])
    # 检查平面投影后的四个点是否共面

    # all 2D points are coplanar
    assert Point.are_coplanar(Point(x, y), Point(x, x + y), Point(y, x + 2)) is True
    # 检查给定的三个二维点是否共面，预期结果为 True

    # Test Intersection
    assert planar2.intersection(Line3D(p, planar3)) == [Point3D(1, 1, 2)]
    # 检查两个对象的交点是否为 [Point3D(1, 1, 2)]

    # Test Scale
    assert planar2.scale(1, 1, 1) == planar2
    # 检查对象在不变缩放情况下是否与自身相等
    assert planar2.scale(2, 2, 2, planar3) == Point3D(1, 1, 1)
    # 检查对象按指定比例缩放后是否等于给定点
    assert planar2.scale(1, 1, 1, p3) == planar2
    # 检查对象在不变缩放情况下是否与自身相等

    # Test Transform
    identity = Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    assert p.transform(identity) == p
    # 检查对象应用单位矩阵变换后是否等于自身
    trans = Matrix([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 1]])
    assert p.transform(trans) == Point3D(2, 2, 2)
    # 检查对象应用指定变换矩阵后是否等于给定点
    raises(ValueError, lambda: p.transform(p))
    # 使用 lambda 函数测试对象在应用非法变换时是否引发 ValueError 异常
    raises(ValueError, lambda: p.transform(Matrix([[1, 0], [0, 1]])))
    # 使用 lambda 函数测试对象在应用非法变换矩阵时是否引发 ValueError 异常

    # Test Equals
    assert p.equals(x1) == False
    # 检查对象是否与给定对象不相等

    # Test __sub__
    p_4d = Point(0, 0, 0, 1)
    with warns(UserWarning, test_stacklevel=False):
        assert p - p_4d == Point(1, 1, 1, -1)
    # 检查对象与另一个点相减后是否等于给定点
    p_4d3d = Point(0, 0, 1, 0)
    with warns(UserWarning, test_stacklevel=False):
        assert p - p_4d3d == Point(1, 1, 0, 0)
    # 检查对象与另一个点相减后是否等于给定点
def test_Point2D():

    # Test Distance
    # 创建两个二维点对象，分别在坐标 (1, 5) 和 (4, 2.5)
    p1 = Point2D(1, 5)
    p2 = Point2D(4, 2.5)
    # 创建一个元组表示第三个点的坐标 (6, 3)
    p3 = (6, 3)
    # 断言计算点之间的距离
    assert p1.distance(p2) == sqrt(61)/2
    assert p2.distance(p3) == sqrt(17)/2

    # Test coordinates
    # 断言点对象的坐标值
    assert p1.x == 1
    assert p1.y == 5
    assert p2.x == 4
    assert p2.y == S(5)/2
    # 断言点对象的坐标元组
    assert p1.coordinates == (1, 5)
    assert p2.coordinates == (4, S(5)/2)

    # test bounds
    # 断言点对象的边界值
    assert p1.bounds == (1, 5, 1, 5)

def test_issue_9214():
    # 创建三维点对象
    p1 = Point3D(4, -2, 6)
    p2 = Point3D(1, 2, 3)
    p3 = Point3D(7, 2, 3)

    # 断言三维点是否共线
    assert Point3D.are_collinear(p1, p2, p3) is False


def test_issue_11617():
    # 创建三维和二维点对象
    p1 = Point3D(1,0,2)
    p2 = Point2D(2,0)

    # 断言两个点之间的距离，带有警告
    with warns(UserWarning, test_stacklevel=False):
        assert p1.distance(p2) == sqrt(5)


def test_transform():
    # 创建二维点对象
    p = Point(1, 1)
    # 断言点对象经过旋转变换后的结果
    assert p.transform(rotate(pi/2)) == Point(-1, 1)
    # 断言点对象经过缩放变换后的结果
    assert p.transform(scale(3, 2)) == Point(3, 2)
    # 断言点对象经过平移变换后的结果
    assert p.transform(translate(1, 2)) == Point(2, 3)
    # 断言点对象经过scale方法的缩放变换后的结果
    assert Point(1, 1).scale(2, 3, (4, 5)) == \
        Point(-2, -7)
    # 断言点对象经过translate方法的平移变换后的结果
    assert Point(1, 1).translate(4, 5) == \
        Point(5, 6)


def test_concyclic_doctest_bug():
    # 创建四个二维点对象
    p1, p2 = Point(-1, 0), Point(1, 0)
    p3, p4 = Point(0, 1), Point(-1, 2)
    # 断言四个点是否共圆
    assert Point.is_concyclic(p1, p2, p3)
    assert not Point.is_concyclic(p1, p2, p3, p4)


def test_arguments():
    """Functions accepting `Point` objects in `geometry`
    should also accept tuples and lists and
    automatically convert them to points."""

    # 定义不同类型的二维点对象和操作
    singles2d = ((1,2), [1,2], Point(1,2))
    singles2d2 = ((1,3), [1,3], Point(1,3))
    doubles2d = cartes(singles2d, singles2d2)
    p2d = Point2D(1,2)
    # 定义不同类型的三维点对象和操作
    singles3d = ((1,2,3), [1,2,3], Point(1,2,3))
    doubles3d = subsets(singles3d, 2)
    p3d = Point3D(1,2,3)
    # 定义不同类型的四维点对象和操作
    singles4d = ((1,2,3,4), [1,2,3,4], Point(1,2,3,4))
    doubles4d = subsets(singles4d, 2)
    p4d = Point(1,2,3,4)

    # test 2D
    # 测试二维点对象的单参数函数和双参数函数
    test_single = ['distance', 'is_scalar_multiple', 'taxicab_distance', 'midpoint', 'intersection', 'dot', 'equals', '__add__', '__sub__']
    test_double = ['is_concyclic', 'is_collinear']
    for p in singles2d:
        Point2D(p)
    for func in test_single:
        for p in singles2d:
            getattr(p2d, func)(p)
    for func in test_double:
        for p in doubles2d:
            getattr(p2d, func)(*p)

    # test 3D
    # 测试三维点对象的单参数函数和双参数函数
    test_double = ['is_collinear']
    for p in singles3d:
        Point3D(p)
    for func in test_single:
        for p in singles3d:
            getattr(p3d, func)(p)
    for func in test_double:
        for p in doubles3d:
            getattr(p3d, func)(*p)

    # test 4D
    # 测试四维点对象的单参数函数和双参数函数
    test_double = ['is_collinear']
    for p in singles4d:
        Point(p)
    for func in test_single:
        for p in singles4d:
            getattr(p4d, func)(p)
    for func in test_double:
        for p in doubles4d:
            getattr(p4d, func)(*p)

    # test evaluate=False for ops
    # 测试操作时evaluate=False的情况
    x = Symbol('x')
    a = Point(0, 1)
    assert a + (0.1, x) == Point(0.1, 1 + x, evaluate=False)
    a = Point(0, 1)
    # 断言验证 a/10.0 是否等于 Point(0, 0.1, evaluate=False)
    assert a/10.0 == Point(0, 0.1, evaluate=False)
    # 创建一个 Point 对象 a，坐标为 (0, 1)
    a = Point(0, 1)
    # 断言验证 a*10.0 是否等于 Point(0.0, 10.0, evaluate=False)
    assert a*10.0 == Point(0.0, 10.0, evaluate=False)

    # 测试 evaluate=False 在改变维度时的行为
    # 创建一个 Point 对象 u，坐标为 (0.1, 0.2)，evaluate=False
    u = Point(.1, .2, evaluate=False)
    # 将 u 转换为四维空间的 Point 对象 u4，dim=4，on_morph='ignore'
    u4 = Point(u, dim=4, on_morph='ignore')
    # 断言验证 u4.args 的值为 (.1, .2, 0, 0)
    assert u4.args == (.1, .2, 0, 0)
    # 断言验证 u4.args 的前两个元素是否都是浮点数
    assert all(i.is_Float for i in u4.args[:2])
    # 即使在不改变维度时，也要保持 evaluate=False 的行为
    assert all(i.is_Float for i in Point(u).args)

    # 创建一个三维空间的 Point 对象，on_morph='error'，不应该抛出错误
    assert Point(dim=3, on_morph='error')

    # 创建一个四维空间的 Point 对象，dim=3，on_morph='error' 应该抛出 ValueError 错误
    raises(ValueError, lambda: Point(1, 1, dim=3, on_morph='error'))
    # 测试未知的 on_morph 参数 'unknown' 应该抛出 ValueError 错误
    raises(ValueError, lambda: Point(1, 1, dim=3, on_morph='unknown'))
    # 测试无效的表达式作为参数应该抛出 TypeError 错误
    raises(TypeError, lambda: Point(Basic(), Basic()))
# 测试单位向量的计算
def test_unit():
    assert Point(1, 1).unit == Point(sqrt(2)/2, sqrt(2)/2)

# 测试点与线的点积操作是否抛出类型错误异常
def test_dot():
    raises(TypeError, lambda: Point(1, 2).dot(Line((0, 0), (1, 1))))

# 测试点类的_normalize_dimension方法，验证对不同维度点的处理是否正确
def test__normalize_dimension():
    assert Point._normalize_dimension(Point(1, 2), Point(3, 4)) == [
        Point(1, 2), Point(3, 4)]
    assert Point._normalize_dimension(
        Point(1, 2), Point(3, 4, 0), on_morph='ignore') == [
        Point(1, 2, 0), Point(3, 4, 0)]

# 测试处理问题编号22684，检查是否会触发错误
def test_issue_22684():
    # Used to give an error
    with evaluate(False):
        Point(1, 2)

# 测试三维点的方向余弦计算功能
def test_direction_cosine():
    p1 = Point3D(0, 0, 0)
    p2 = Point3D(1, 1, 1)

    assert p1.direction_cosine(Point3D(1, 0, 0)) == [1, 0, 0]
    assert p1.direction_cosine(Point3D(0, 1, 0)) == [0, 1, 0]
    assert p1.direction_cosine(Point3D(0, 0, pi)) == [0, 0, 1]

    assert p1.direction_cosine(Point3D(5, 0, 0)) == [1, 0, 0]
    assert p1.direction_cosine(Point3D(0, sqrt(3), 0)) == [0, 1, 0]
    assert p1.direction_cosine(Point3D(0, 0, 5)) == [0, 0, 1]

    assert p1.direction_cosine(Point3D(2.4, 2.4, 0)) == [sqrt(2)/2, sqrt(2)/2, 0]
    assert p1.direction_cosine(Point3D(1, 1, 1)) == [sqrt(3) / 3, sqrt(3) / 3, sqrt(3) / 3]
    assert p1.direction_cosine(Point3D(-12, 0, -15)) == [-4*sqrt(41)/41, -5*sqrt(41)/41, 0]

    assert p2.direction_cosine(Point3D(0, 0, 0)) == [-sqrt(3) / 3, -sqrt(3) / 3, -sqrt(3) / 3]
    assert p2.direction_cosine(Point3D(1, 1, 12)) == [0, 0, 1]
    assert p2.direction_cosine(Point3D(12, 1, 12)) == [sqrt(2) / 2, 0, sqrt(2) / 2]
```