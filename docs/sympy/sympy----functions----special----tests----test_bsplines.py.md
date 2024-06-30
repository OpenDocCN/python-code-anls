# `D:\src\scipysrc\sympy\sympy\functions\special\tests\test_bsplines.py`

```
# 导入需要的函数和类
from sympy.functions import bspline_basis_set, interpolating_spline
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.piecewise import Piecewise
from sympy.logic.boolalg import And
from sympy.sets.sets import Interval
from sympy.testing.pytest import slow

# 定义符号变量 x 和 y
x, y = symbols('x,y')

# 测试函数：测试零阶 B 样条基函数
def test_basic_degree_0():
    # 设置 B 样条的阶数为 0
    d = 0
    # 设置结点向量为 [0, 1, 2, 3, 4]
    knots = range(5)
    # 计算 B 样条基函数集合
    splines = bspline_basis_set(d, knots, x)
    # 遍历基函数集合
    for i in range(len(splines)):
        # 断言每个基函数的定义
        assert splines[i] == Piecewise((1, Interval(i, i + 1).contains(x)),
                                       (0, True))

# 测试函数：测试一阶 B 样条基函数
def test_basic_degree_1():
    # 设置 B 样条的阶数为 1
    d = 1
    # 设置结点向量为 [0, 1, 2, 3, 4]
    knots = range(5)
    # 计算 B 样条基函数集合
    splines = bspline_basis_set(d, knots, x)
    # 断言每个基函数的定义
    assert splines[0] == Piecewise((x, Interval(0, 1).contains(x)),
                                   (2 - x, Interval(1, 2).contains(x)),
                                   (0, True))
    assert splines[1] == Piecewise((-1 + x, Interval(1, 2).contains(x)),
                                   (3 - x, Interval(2, 3).contains(x)),
                                   (0, True))
    assert splines[2] == Piecewise((-2 + x, Interval(2, 3).contains(x)),
                                   (4 - x, Interval(3, 4).contains(x)),
                                   (0, True))

# 测试函数：测试二阶 B 样条基函数
def test_basic_degree_2():
    # 设置 B 样条的阶数为 2
    d = 2
    # 设置结点向量为 [0, 1, 2, 3, 4]
    knots = range(5)
    # 计算 B 样条基函数集合
    splines = bspline_basis_set(d, knots, x)
    # 定义预期的基函数
    b0 = Piecewise((x**2/2, Interval(0, 1).contains(x)),
                   (Rational(-3, 2) + 3*x - x**2, Interval(1, 2).contains(x)),
                   (Rational(9, 2) - 3*x + x**2/2, Interval(2, 3).contains(x)),
                   (0, True))
    b1 = Piecewise((S.Half - x + x**2/2, Interval(1, 2).contains(x)),
                   (Rational(-11, 2) + 5*x - x**2, Interval(2, 3).contains(x)),
                   (8 - 4*x + x**2/2, Interval(3, 4).contains(x)),
                   (0, True))
    # 断言每个基函数的定义
    assert splines[0] == b0
    assert splines[1] == b1

# 测试函数：测试三阶 B 样条基函数
def test_basic_degree_3():
    # 设置 B 样条的阶数为 3
    d = 3
    # 设置结点向量为 [0, 1, 2, 3, 4]
    knots = range(5)
    # 计算 B 样条基函数集合
    splines = bspline_basis_set(d, knots, x)
    # 定义预期的基函数
    b0 = Piecewise(
        (x**3/6, Interval(0, 1).contains(x)),
        (Rational(2, 3) - 2*x + 2*x**2 - x**3/2, Interval(1, 2).contains(x)),
        (Rational(-22, 3) + 10*x - 4*x**2 + x**3/2, Interval(2, 3).contains(x)),
        (Rational(32, 3) - 8*x + 2*x**2 - x**3/6, Interval(3, 4).contains(x)),
        (0, True)
    )
    # 断言基函数的定义
    assert splines[0] == b0

# 测试函数：测试重复结点的一阶 B 样条基函数
def test_repeated_degree_1():
    # 设置 B 样条的阶数为 1
    d = 1
    # 设置带有重复结点的结点向量
    knots = [0, 0, 1, 2, 2, 3, 4, 4]
    # 计算 B 样条基函数集合
    splines = bspline_basis_set(d, knots, x)
    # 断言每个基函数的定义
    assert splines[0] == Piecewise((1 - x, Interval(0, 1).contains(x)),
                                   (0, True))
    assert splines[1] == Piecewise((x, Interval(0, 1).contains(x)),
                                   (2 - x, Interval(1, 2).contains(x)),
                                   (0, True))
    assert splines[2] == Piecewise((-1 + x, Interval(1, 2).contains(x)),
                                   (0, True))
    # 断言，验证 splines 列表中索引为 3 的元素是否等于 Piecewise 对象
    assert splines[3] == Piecewise((3 - x, Interval(2, 3).contains(x)),
                                   (0, True))
    # 断言，验证 splines 列表中索引为 4 的元素是否等于 Piecewise 对象
    assert splines[4] == Piecewise((-2 + x, Interval(2, 3).contains(x)),
                                   (4 - x, Interval(3, 4).contains(x)),
                                   (0, True))
    # 断言，验证 splines 列表中索引为 5 的元素是否等于 Piecewise 对象
    assert splines[5] == Piecewise((-3 + x, Interval(3, 4).contains(x)),
                                   (0, True))
# 定义一个测试函数，用于测试二次重复次数的 B-spline 基函数集合
def test_repeated_degree_2():
    # 设置 B-spline 的阶数为 2
    d = 2
    # 设置 B-spline 的节点序列
    knots = [0, 0, 1, 2, 2, 3, 4, 4]
    # 调用 bspline_basis_set 函数生成 B-spline 基函数集合
    splines = bspline_basis_set(d, knots, x)

    # 断言：验证生成的 B-spline 基函数集合的各个分段函数
    assert splines[0] == Piecewise(((-3*x**2/2 + 2*x), And(x <= 1, x >= 0)),
                                   (x**2/2 - 2*x + 2, And(x <= 2, x >= 1)),
                                   (0, True))
    assert splines[1] == Piecewise((x**2/2, And(x <= 1, x >= 0)),
                                   (-3*x**2/2 + 4*x - 2, And(x <= 2, x >= 1)),
                                   (0, True))
    assert splines[2] == Piecewise((x**2 - 2*x + 1, And(x <= 2, x >= 1)),
                                   (x**2 - 6*x + 9, And(x <= 3, x >= 2)),
                                   (0, True))
    assert splines[3] == Piecewise((-3*x**2/2 + 8*x - 10, And(x <= 3, x >= 2)),
                                   (x**2/2 - 4*x + 8, And(x <= 4, x >= 3)),
                                   (0, True))
    assert splines[4] == Piecewise((x**2/2 - 2*x + 2, And(x <= 3, x >= 2)),
                                   (-3*x**2/2 + 10*x - 16, And(x <= 4, x >= 3)),
                                   (0, True))


# Tests for interpolating_spline


# 测试使用 10 个点的线性插值样条
def test_10_points_degree_1():
    # 设置插值样条的阶数为 1
    d = 1
    # 设置输入数据点的 X 坐标
    X = [-5, 2, 3, 4, 7, 9, 10, 30, 31, 34]
    # 设置输入数据点的 Y 坐标
    Y = [-10, -2, 2, 4, 7, 6, 20, 45, 19, 25]
    # 调用 interpolating_spline 函数生成线性插值样条
    spline = interpolating_spline(d, x, X, Y)

    # 断言：验证生成的线性插值样条函数
    assert spline == Piecewise((x*Rational(8, 7) - Rational(30, 7), (x >= -5) & (x <= 2)),
                               (4*x - 10, (x >= 2) & (x <= 3)),
                               (2*x - 4, (x >= 3) & (x <= 4)),
                               (x, (x >= 4) & (x <= 7)),
                               (-x/2 + Rational(21, 2), (x >= 7) & (x <= 9)),
                               (14*x - 120, (x >= 9) & (x <= 10)),
                               (x*Rational(5, 4) + Rational(15, 2), (x >= 10) & (x <= 30)),
                               (-26*x + 825, (x >= 30) & (x <= 31)),
                               (2*x - 43, (x >= 31) & (x <= 34)))


# 测试使用 3 个点的二次插值样条
def test_3_points_degree_2():
    # 设置插值样条的阶数为 2
    d = 2
    # 设置输入数据点的 X 坐标
    X = [-3, 10, 19]
    # 设置输入数据点的 Y 坐标
    Y = [3, -4, 30]
    # 调用 interpolating_spline 函数生成二次插值样条
    spline = interpolating_spline(d, x, X, Y)

    # 断言：验证生成的二次插值样条函数
    assert spline == Piecewise((505*x**2/2574 - x*Rational(4921, 2574) - Rational(1931, 429), (x >= -3) & (x <= 19)))


# 测试使用 5 个点的二次插值样条
def test_5_points_degree_2():
    # 设置插值样条的阶数为 2
    d = 2
    # 设置输入数据点的 X 坐标
    X = [-3, 2, 4, 5, 10]
    # 设置输入数据点的 Y 坐标
    Y = [-1, 2, 5, 10, 14]
    # 调用 interpolating_spline 函数生成二次插值样条
    spline = interpolating_spline(d, x, X, Y)

    # 断言：验证生成的二次插值样条函数
    assert spline == Piecewise((4*x**2/329 + x*Rational(1007, 1645) + Rational(1196, 1645), (x >= -3) & (x <= 3)),
                               (2701*x**2/1645 - x*Rational(15079, 1645) + Rational(5065, 329), (x >= 3) & (x <= Rational(9, 2))),
                               (-1319*x**2/1645 + x*Rational(21101, 1645) - Rational(11216, 329), (x >= Rational(9, 2)) & (x <= 10)))


# 测试使用 6 个点的三次插值样条，标记为 @slow 表示这个测试较慢
@slow
def test_6_points_degree_3():
    # 设置插值样条的阶数为 3
    d = 3
    # 设置输入数据点的 X 坐标
    X = [-1, 0, 2, 3, 9, 12]
    # 设置输入数据点的 Y 坐标
    Y = [-4, 3, 3, 7, 9, 20]
    # 调用 interpolating_spline 函数生成三次插值样条
    spline = interpolating_spline(d, x, X, Y)
    # 使用 assert 语句验证 spline 变量是否等于给定的分段函数表达式
    assert spline == Piecewise((6058*x**3/5301 - 18427*x**2/5301 + x*Rational(12622, 5301) + 3, (x >= -1) & (x <= 2)),
                               (-8327*x**3/5301 + 67883*x**2/5301 - x*Rational(159998, 5301) + Rational(43661, 1767), (x >= 2) & (x <= 3)),
                               (5414*x**3/47709 - 1386*x**2/589 + x*Rational(4267, 279) - Rational(12232, 589), (x >= 3) & (x <= 12)))
# 定义一个测试函数，用于测试与问题编号19262相关的功能
def test_issue_19262():
    # 创建一个正数符号 Delta
    Delta = symbols('Delta', positive=True)
    # 根据 Delta 创建一组结点列表，每个结点为 Delta 的倍数，共计4个结点
    knots = [i*Delta for i in range(4)]
    # 使用给定的结点和变量 x，创建 B-spline 基函数集合，阶数为1
    basis = bspline_basis_set(1, knots, x)
    # 创建一个非负数符号 y
    y = symbols('y', nonnegative=True)
    # 使用相同的结点和变量 y，再次创建 B-spline 基函数集合，阶数为1
    basis2 = bspline_basis_set(1, knots, y)
    # 断言：两个不同变量（x 和 y）对应的第一个基函数的值相等
    assert basis[0].subs(x, y) == basis2[0]
    # 断言：使用给定参数创建的插值样条在特定区间的定义与期望的 Piecewise 函数相等
    assert interpolating_spline(1, x,
        [Delta*i for i in [1, 2, 4, 7]], [3, 6, 5, 7]
        )  == Piecewise((3*x/Delta, (Delta <= x) & (x <= 2*Delta)),
        (7 - x/(2*Delta), (x >= 2*Delta) & (x <= 4*Delta)),
        (Rational(7, 3) + 2*x/(3*Delta), (x >= 4*Delta) & (x <= 7*Delta)))
```