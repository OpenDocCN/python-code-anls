# `D:\src\scipysrc\sympy\sympy\integrals\tests\test_intpoly.py`

```
# 导入符号数学库中的复数和平方根函数
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.miscellaneous import sqrt

# 导入符号数学库的核心模块，包括符号、有理数等
from sympy.core import S, Rational

# 导入符号数学库中的积分多项式相关函数
from sympy.integrals.intpoly import (decompose, best_origin, distance_to_side,
                                     polytope_integrate, point_sort,
                                     hyperplane_parameters, main_integrate3d,
                                     main_integrate, polygon_integrate,
                                     lineseg_integrate, integration_reduction,
                                     integration_reduction_dynamic, is_vertex)

# 导入符号几何库中的线段、多边形、点等对象和函数
from sympy.geometry.line import Segment2D
from sympy.geometry.polygon import Polygon
from sympy.geometry.point import Point, Point2D
from sympy.abc import x, y, z

# 导入符号数学库中的测试模块，特别标记为慢速测试
from sympy.testing.pytest import slow


# 定义测试函数，用于测试多项式分解函数 decompose
def test_decompose():
    assert decompose(x) == {1: x}
    assert decompose(x**2) == {2: x**2}
    assert decompose(x*y) == {2: x*y}
    assert decompose(x + y) == {1: x + y}
    assert decompose(x**2 + y) == {1: y, 2: x**2}
    assert decompose(8*x**2 + 4*y + 7) == {0: 7, 1: 4*y, 2: 8*x**2}
    assert decompose(x**2 + 3*y*x) == {2: x**2 + 3*x*y}
    assert decompose(9*x**2 + y + 4*x + x**3 + y**2*x + 3) ==\
        {0: 3, 1: 4*x + y, 2: 9*x**2, 3: x**3 + x*y**2}

    assert decompose(x, True) == {x}
    assert decompose(x ** 2, True) == {x**2}
    assert decompose(x * y, True) == {x * y}
    assert decompose(x + y, True) == {x, y}
    assert decompose(x ** 2 + y, True) == {y, x ** 2}
    assert decompose(8 * x ** 2 + 4 * y + 7, True) == {7, 4*y, 8*x**2}
    assert decompose(x ** 2 + 3 * y * x, True) == {x ** 2, 3 * x * y}
    assert decompose(9 * x ** 2 + y + 4 * x + x ** 3 + y ** 2 * x + 3, True) == \
           {3, y, 4*x, 9*x**2, x*y**2, x**3}


# 定义测试函数，用于测试最佳原点函数 best_origin
def test_best_origin():
    expr1 = y ** 2 * x ** 5 + y ** 5 * x ** 7 + 7 * x + x ** 12 + y ** 7 * x

    l1 = Segment2D(Point(0, 3), Point(1, 1))
    l2 = Segment2D(Point(S(3) / 2, 0), Point(S(3) / 2, 3))
    l3 = Segment2D(Point(0, S(3) / 2), Point(3, S(3) / 2))
    l4 = Segment2D(Point(0, 2), Point(2, 0))
    l5 = Segment2D(Point(0, 2), Point(1, 1))
    l6 = Segment2D(Point(2, 0), Point(1, 1))

    assert best_origin((2, 1), 3, l1, expr1) == (0, 3)
    # XXX: 这些是否应该返回精确的有理数输出？也许 best_origin 应该 sympify 其参数...
    assert best_origin((2, 0), 3, l2, x ** 7) == (1.5, 0)
    assert best_origin((0, 2), 3, l3, x ** 7) == (0, 1.5)
    assert best_origin((1, 1), 2, l4, x ** 7 * y ** 3) == (0, 2)
    assert best_origin((1, 1), 2, l4, x ** 3 * y ** 7) == (2, 0)
    assert best_origin((1, 1), 2, l5, x ** 2 * y ** 9) == (0, 2)
    assert best_origin((1, 1), 2, l6, x ** 9 * y ** 2) == (2, 0)


# 标记为慢速测试的函数，用于测试多边形积分函数 polytope_integrate
@slow
def test_polytope_integrate():
    #  凸2多边形
    #  顶点表示法
    assert polytope_integrate(Polygon(Point(0, 0), Point(0, 2),
                                      Point(4, 0)), 1) == 4
    # 确认多边形在给定函数 x * y 下的积分结果等于 1/4
    assert polytope_integrate(Polygon(Point(0, 0), Point(0, 1),
                                      Point(1, 1), Point(1, 0)), x * y) ==\
                                      Rational(1, 4)
    
    # 确认多边形在给定函数 6*x**2 - 40*y 下的积分结果等于 -935/3
    assert polytope_integrate(Polygon(Point(0, 3), Point(5, 3), Point(1, 1)),
                              6*x**2 - 40*y) == Rational(-935, 3)

    # 确认多边形在给定常数函数 1 下的积分结果等于 3
    assert polytope_integrate(Polygon(Point(0, 0), Point(0, sqrt(3)),
                                      Point(sqrt(3), sqrt(3)),
                                      Point(sqrt(3), 0)), 1) == 3

    # 创建六边形对象 hexagon
    hexagon = Polygon(Point(0, 0), Point(-sqrt(3) / 2, S.Half),
                      Point(-sqrt(3) / 2, S(3) / 2), Point(0, 2),
                      Point(sqrt(3) / 2, S(3) / 2), Point(sqrt(3) / 2, S.Half))

    # 确认 hexagon 在给定常数函数 1 下的积分结果等于 3*sqrt(3)/2
    assert polytope_integrate(hexagon, 1) == S(3*sqrt(3)) / 2

    # 使用超平面表示确认在给定常数函数 1 下的积分结果等于 4
    assert polytope_integrate([((-1, 0), 0), ((1, 2), 4),
                               ((0, -1), 0)], 1) == 4
    
    # 使用超平面表示确认在给定函数 x * y 下的积分结果等于 1/4
    assert polytope_integrate([((-1, 0), 0), ((0, 1), 1),
                               ((1, 0), 1), ((0, -1), 0)], x * y) == Rational(1, 4)
    
    # 使用超平面表示确认在给定函数 6*x**2 - 40*y 下的积分结果等于 -935/3
    assert polytope_integrate([((0, 1), 3), ((1, -2), -1),
                               ((-2, -1), -3)], 6*x**2 - 40*y) == Rational(-935, 3)
    
    # 使用超平面表示确认在给定常数函数 1 下的积分结果等于 3
    assert polytope_integrate([((-1, 0), 0), ((0, sqrt(3)), 3),
                               ((sqrt(3), 0), 3), ((0, -1), 0)], 1) == 3

    # 创建六边形对象 hexagon
    hexagon = [((Rational(-1, 2), -sqrt(3) / 2), 0),
               ((-1, 0), sqrt(3) / 2),
               ((Rational(-1, 2), sqrt(3) / 2), sqrt(3)),
               ((S.Half, sqrt(3) / 2), sqrt(3)),
               ((1, 0), sqrt(3) / 2),
               ((S.Half, -sqrt(3) / 2), 0)]
    
    # 确认 hexagon 在给定常数函数 1 下的积分结果等于 3*sqrt(3)/2
    assert polytope_integrate(hexagon, 1) == S(3*sqrt(3)) / 2

    # 非凸多边形的测试
    # 使用顶点表示确认在给定常数函数 1 下的积分结果等于 3
    assert polytope_integrate(Polygon(Point(-1, -1), Point(-1, 1),
                                      Point(1, 1), Point(0, 0),
                                      Point(1, -1)), 1) == 3
    
    # 使用顶点表示确认在给定常数函数 1 下的积分结果等于 2
    assert polytope_integrate(Polygon(Point(-1, -1), Point(-1, 1),
                                      Point(0, 0), Point(1, 1),
                                      Point(1, -1), Point(0, 0)), 1) == 2
    
    # 使用超平面表示确认在给定常数函数 1 下的积分结果等于 3
    assert polytope_integrate([((-1, 0), 1), ((0, 1), 1), ((1, -1), 0),
                               ((1, 1), 0), ((0, -1), 1)], 1) == 3
    
    # 使用超平面表示确认在给定常数函数 1 下的积分结果等于 2
    assert polytope_integrate([((-1, 0), 1), ((1, 1), 0), ((-1, 1), 0),
                               ((1, 0), 1), ((-1, -1), 0),
                               ((1, -1), 0)], 1) == 2

    # Chin 等人在第 10 页提到的二维多边形测试
    fig1 = Polygon(Point(1.220, -0.827), Point(-1.490, -4.503),
                   Point(-3.766, -1.622), Point(-4.240, -0.091),
                   Point(-3.160, 4), Point(-0.981, 4.447),
                   Point(0.132, 4.027))
    # 确保对于给定的多边形 fig1，使用 polytope_integrate 函数计算二次多项式 x**2 + x*y + y**2 的积分值，并进行断言比较
    assert polytope_integrate(fig1, x**2 + x*y + y**2) ==\
        S(2031627344735367)/(8*10**12)

    # 创建一个包含五个顶点的多边形 fig2，并使用 polytope_integrate 函数计算其对于二次多项式 x**2 + x*y + y**2 的积分值，并进行断言比较
    fig2 = Polygon(Point(4.561, 2.317), Point(1.491, -1.315),
                   Point(-3.310, -3.164), Point(-4.845, -3.110),
                   Point(-4.569, 1.867))
    assert polytope_integrate(fig2, x**2 + x*y + y**2) ==\
        S(517091313866043)/(16*10**11)

    # 创建一个包含四个顶点的多边形 fig3，并使用 polytope_integrate 函数计算其对于二次多项式 x**2 + x*y + y**2 的积分值，并进行断言比较
    fig3 = Polygon(Point(-2.740, -1.888), Point(-3.292, 4.233),
                   Point(-2.723, -0.697), Point(-0.643, -3.151))
    assert polytope_integrate(fig3, x**2 + x*y + y**2) ==\
        S(147449361647041)/(8*10**12)

    # 创建一个包含五个顶点的多边形 fig4，并使用 polytope_integrate 函数计算其对于二次多项式 x**2 + x*y + y**2 的积分值，并进行断言比较
    fig4 = Polygon(Point(0.211, -4.622), Point(-2.684, 3.851),
                   Point(0.468, 4.879), Point(4.630, -1.325),
                   Point(-0.411, -1.044))
    assert polytope_integrate(fig4, x**2 + x*y + y**2) ==\
        S(180742845225803)/(10**12)

    # 创建一个三角形 tri，并定义多项式列表 polys，计算多项式在 tri 上的积分值，并将结果存储在 result_dict 中，然后进行断言比较
    tri = Polygon(Point(0, 3), Point(5, 3), Point(1, 1))
    polys = []
    expr1 = x**9*y + x**7*y**3 + 2*x**2*y**8
    expr2 = x**6*y**4 + x**5*y**5 + 2*y**10
    expr3 = x**10 + x**9*y + x**8*y**2 + x**5*y**5
    polys.extend((expr1, expr2, expr3))
    result_dict = polytope_integrate(tri, polys, max_degree=10)
    assert result_dict[expr1] == Rational(615780107, 594)
    assert result_dict[expr2] == Rational(13062161, 27)
    assert result_dict[expr3] == Rational(1946257153, 924)

    # 重新定义三角形 tri 和多项式列表 polys，计算多项式在 tri 上的积分值（限制最大次数为 9），并进行断言比较
    tri = Polygon(Point(0, 3), Point(5, 3), Point(1, 1))
    expr1 = x**7*y**1 + 2*x**2*y**6
    expr2 = x**6*y**4 + x**5*y**5 + 2*y**10
    expr3 = x**10 + x**9*y + x**8*y**2 + x**5*y**5
    polys.extend((expr1, expr2, expr3))
    assert polytope_integrate(tri, polys, max_degree=9) == \
        {x**7*y + 2*x**2*y**6: Rational(489262, 9)}

    # 计算一个四边形的积分值，其中计算所有最高次数为 4 的单项式，将结果存储在字典中，并进行断言比较
    assert polytope_integrate(Polygon(Point(0, 0), Point(0, 1),
                                      Point(1, 1), Point(1, 0)),
                              max_degree=4) == {0: 0, 1: 1, x: S.Half,
                                                x ** 2 * y ** 2: S.One / 9,
                                                x ** 4: S.One / 5,
                                                y ** 4: S.One / 5,
                                                y: S.Half,
                                                x * y ** 2: S.One / 6,
                                                y ** 2: S.One / 3,
                                                x ** 3: S.One / 4,
                                                x ** 2 * y: S.One / 6,
                                                x ** 3 * y: S.One / 8,
                                                x * y: S.One / 4,
                                                y ** 3: S.One / 4,
                                                x ** 2: S.One / 3,
                                                x * y ** 3: S.One / 8}

    # 用于测试三维多面体的情况
    # 定义一个立方体的顶点坐标和面的顶点索引列表
    cube1 = [[(0, 0, 0), (0, 6, 6), (6, 6, 6), (3, 6, 0),
              (0, 6, 0), (6, 0, 6), (3, 0, 0), (0, 0, 6)],
             [1, 2, 3, 4], [3, 2, 5, 6], [1, 7, 5, 2], [0, 6, 5, 7],
             [1, 4, 0, 7], [0, 4, 3, 6]]
    # 断言polytope_integrate函数应用于cube1返回结果为162
    assert polytope_integrate(cube1, 1) == S(162)

    #  从Chin et al(2015)中引用的三维测试案例
    # 定义第二个立方体的顶点坐标和面的顶点索引列表
    cube2 = [[(0, 0, 0), (0, 0, 5), (0, 5, 0), (0, 5, 5), (5, 0, 0),
             (5, 0, 5), (5, 5, 0), (5, 5, 5)],
             [3, 7, 6, 2], [1, 5, 7, 3], [5, 4, 6, 7], [0, 4, 5, 1],
             [2, 0, 1, 3], [2, 6, 4, 0]]

    # 定义第三个立方体的顶点坐标和面的顶点索引列表
    cube3 = [[(0, 0, 0), (5, 0, 0), (5, 4, 0), (3, 2, 0), (3, 5, 0),
              (0, 5, 0), (0, 0, 5), (5, 0, 5), (5, 4, 5), (3, 2, 5),
              (3, 5, 5), (0, 5, 5)],
             [6, 11, 5, 0], [1, 7, 6, 0], [5, 4, 3, 2, 1, 0], [11, 10, 4, 5],
             [10, 9, 3, 4], [9, 8, 2, 3], [8, 7, 1, 2], [7, 8, 9, 10, 11, 6]]

    # 定义第四个立方体的顶点坐标和面的顶点索引列表
    cube4 = [[(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
              (S.One / 4, S.One / 4, S.One / 4)],
             [0, 2, 1], [1, 3, 0], [4, 2, 3], [4, 3, 1],
             [0, 1, 2], [2, 4, 1], [0, 3, 2]]

    # 断言polytope_integrate函数应用于cube2、cube3、cube4分别返回指定的有理数或平方根表达式结果
    assert polytope_integrate(cube2, x ** 2 + y ** 2 + x * y + z ** 2) ==\
           Rational(15625, 4)
    assert polytope_integrate(cube3, x ** 2 + y ** 2 + x * y + z ** 2) ==\
           S(33835) / 12
    assert polytope_integrate(cube4, x ** 2 + y ** 2 + x * y + z ** 2) ==\
           S(37) / 960

    #  从Mathematica的PolyhedronData库中引用的测试案例
    # 定义一个八面体的顶点坐标和面的顶点索引列表
    octahedron = [[(S.NegativeOne / sqrt(2), 0, 0), (0, S.One / sqrt(2), 0),
                   (0, 0, S.NegativeOne / sqrt(2)), (0, 0, S.One / sqrt(2)),
                   (0, S.NegativeOne / sqrt(2), 0), (S.One / sqrt(2), 0, 0)],
                  [3, 4, 5], [3, 5, 1], [3, 1, 0], [3, 0, 4], [4, 0, 2],
                  [4, 2, 5], [2, 0, 1], [5, 2, 1]]

    # 断言polytope_integrate函数应用于octahedron返回结果为平方根2除以3
    assert polytope_integrate(octahedron, 1) == sqrt(2) / 3
    # 定义一个包含坐标和面索引的复杂结构，表示一个大星形十二面体
    great_stellated_dodecahedron =\
        [[(-0.32491969623290634095, 0, 0.42532540417601993887),
          (0.32491969623290634095, 0, -0.42532540417601993887),
          (-0.52573111211913359231, 0, 0.10040570794311363956),
          (0.52573111211913359231, 0, -0.10040570794311363956),
          (-0.10040570794311363956, -0.3090169943749474241, 0.42532540417601993887),
          (-0.10040570794311363956, 0.30901699437494742410, 0.42532540417601993887),
          (0.10040570794311363956, -0.3090169943749474241, -0.42532540417601993887),
          (0.10040570794311363956, 0.30901699437494742410, -0.42532540417601993887),
          (-0.16245984811645317047, -0.5, 0.10040570794311363956),
          (-0.16245984811645317047,  0.5, 0.10040570794311363956),
          (0.16245984811645317047,  -0.5, -0.10040570794311363956),
          (0.16245984811645317047,   0.5, -0.10040570794311363956),
          (-0.42532540417601993887, -0.3090169943749474241, -0.10040570794311363956),
          (-0.42532540417601993887, 0.30901699437494742410, -0.10040570794311363956),
          (-0.26286555605956679615, 0.1909830056250525759, -0.42532540417601993887),
          (-0.26286555605956679615, -0.1909830056250525759, -0.42532540417601993887),
          (0.26286555605956679615, 0.1909830056250525759, 0.42532540417601993887),
          (0.26286555605956679615, -0.1909830056250525759, 0.42532540417601993887),
          (0.42532540417601993887, -0.3090169943749474241, 0.10040570794311363956),
          (0.42532540417601993887, 0.30901699437494742410, 0.10040570794311363956)],
         [12, 3, 0, 6, 16], [17, 7, 0, 3, 13],
         [9, 6, 0, 7, 8], [18, 2, 1, 4, 14],
         [15, 5, 1, 2, 19], [11, 4, 1, 5, 10],
         [8, 19, 2, 18, 9], [10, 13, 3, 12, 11],
         [16, 14, 4, 11, 12], [13, 10, 5, 15, 17],
         [14, 16, 6, 9, 18], [19, 8, 7, 17, 15]]
    # 断言计算得到的多面体积为约 0.163118960624632
    assert Abs(polytope_integrate(great_stellated_dodecahedron, 1) -\
        0.163118960624632) < 1e-12

    # 定义一个表达式 expr，表示 x^2 + y^2 + z^2
    expr = x **2 + y ** 2 + z ** 2
    # 断言计算得到的多面体积使用表达式 expr 为约 0.353553
    assert Abs(polytope_integrate(octahedron_five_compound, expr)) - 0.353553\
        < 1e-6
    # 定义一个复合列表 cube_five_compound，包含了多个顶点的坐标和多个面的顶点索引
    cube_five_compound = [
        # 第一个顶点坐标组
        [(-0.1624598481164531631, -0.5, -0.6881909602355867691),
         (-0.1624598481164531631, 0.5, -0.6881909602355867691),
         (0.1624598481164531631, -0.5, 0.68819096023558676910),
         (0.1624598481164531631, 0.5, 0.68819096023558676910),
         (-0.52573111211913359231, 0, -0.6881909602355867691),
         (0.52573111211913359231, 0, 0.68819096023558676910),
         (-0.26286555605956679615, -0.8090169943749474241, -0.1624598481164531631),
         (-0.26286555605956679615, 0.8090169943749474241, -0.1624598481164531631),
         (0.26286555605956680301, -0.8090169943749474241, 0.1624598481164531631),
         (0.26286555605956680301, 0.8090169943749474241, 0.1624598481164531631),
         (-0.42532540417601993887, -0.3090169943749474241, 0.68819096023558676910),
         (-0.42532540417601993887, 0.30901699437494742410, 0.68819096023558676910),
         (0.42532540417601996609, -0.3090169943749474241, -0.6881909602355867691),
         (0.42532540417601996609, 0.30901699437494742410, -0.6881909602355867691),
         (-0.6881909602355867691, -0.5, 0.1624598481164531631),
         (-0.6881909602355867691, 0.5,  0.1624598481164531631),
         (0.68819096023558676910, -0.5, -0.1624598481164531631),
         (0.68819096023558676910, 0.5, -0.1624598481164531631),
         (-0.85065080835203998877, 0, -0.1624598481164531631),
         (0.85065080835203993218, 0, 0.1624598481164531631)],
        
        # 多个面的顶点索引
        [18, 10, 3, 7], [13, 19, 8, 0], [18, 0, 8, 10],
        [3, 19, 13, 7], [18, 7, 13, 0], [8, 19, 3, 10],
        [6, 2, 11, 18], [1, 9, 19, 12], [11, 9, 1, 18],
        [6, 12, 19, 2], [1, 12, 6, 18], [11, 2, 19, 9],
        [4, 14, 11, 7], [17, 5, 8, 12], [4, 12, 8, 14],
        [11, 5, 17, 7], [4, 7, 17, 12], [8, 5, 11, 14],
        [6, 10, 15, 4], [13, 9, 5, 16], [15, 9, 13, 4],
        [6, 16, 5, 10], [13, 16, 6, 4], [15, 10, 5, 9],
        [14, 15, 1, 0], [16, 17, 3, 2], [14, 2, 3, 15],
        [1, 17, 16, 0], [14, 0, 16, 2], [3, 17, 1, 15]
    ]
    
    # 使用断言检查多面体 polytope_integrate 函数计算的结果是否接近 1.25
    assert Abs(polytope_integrate(cube_five_compound, expr) - 1.25) < 1e-12
    
    # 实际体积为：51.405764746872634，使用断言检查多面体 polytope_integrate 函数计算的结果是否接近该值
    assert Abs(polytope_integrate(echidnahedron, 1) - 51.4057647468726) < 1e-12
    # 断言检查多面体积分结果与预期值的差异是否在指定精度内
    assert Abs(polytope_integrate(echidnahedron, expr) - 253.569603474519) <\
    1e-12

    # 测试在二维情况下，对多个多项式进行积分，最大次数为2
    assert polytope_integrate(cube2, [x**2, y*z], max_degree=2) == \
        {y * z: 3125 / S(4), x ** 2: 3125 / S(3)}

    # 测试在二维情况下，对多面体 cube2 进行积分，最大次数为2
    assert polytope_integrate(cube2, max_degree=2) == \
        {1: 125, x: 625 / S(2), x * z: 3125 / S(4), y: 625 / S(2),
         y * z: 3125 / S(4), z ** 2: 3125 / S(3), y ** 2: 3125 / S(3),
         z: 625 / S(2), x * y: 3125 / S(4), x ** 2: 3125 / S(3)}
# 定义用于测试点排序的函数
def test_point_sort():
    # 断言对于给定的点列表，排序后的结果是否符合预期
    assert point_sort([Point(0, 0), Point(1, 0), Point(1, 1)]) == \
        [Point2D(1, 1), Point2D(1, 0), Point2D(0, 0)]

    # 创建一个六边形对象
    fig6 = Polygon((0, 0), (1, 0), (1, 1))
    # 断言多边形在指定区域上的积分结果是否等于预期值
    assert polytope_integrate(fig6, x*y) == Rational(-1, 8)
    # 断言多边形在指定区域上的积分结果是否等于预期值，且按顺时针方向积分
    assert polytope_integrate(fig6, x*y, clockwise=True) == Rational(1, 8)


# 定义用于测试多边形交集边的函数
def test_polytopes_intersecting_sides():
    # 创建一个具有八个顶点的多边形对象
    fig5 = Polygon(Point(-4.165, -0.832), Point(-3.668, 1.568),
                   Point(-3.266, 1.279), Point(-1.090, -2.080),
                   Point(3.313, -0.683), Point(3.033, -4.845),
                   Point(-4.395, 4.840), Point(-1.007, -3.328))
    # 断言多边形在指定区域上的积分结果是否等于预期有理数
    assert polytope_integrate(fig5, x**2 + x*y + y**2) == \
        S(1633405224899363)/(24*10**12)

    # 创建另一个具有五个顶点的多边形对象
    fig6 = Polygon(Point(-3.018, -4.473), Point(-0.103, 2.378),
                   Point(-1.605, -2.308), Point(4.516, -0.771),
                   Point(4.203, 0.478))
    # 断言多边形在指定区域上的积分结果是否等于预期有理数
    assert polytope_integrate(fig6, x**2 + x*y + y**2) == \
        S(88161333955921)/(3*10**12)


# 定义用于测试最大次数的函数
def test_max_degree():
    # 创建一个四边形对象
    polygon = Polygon((0, 0), (0, 1), (1, 1), (1, 0))
    # 创建一个包含多个多项式的列表
    polys = [1, x, y, x*y, x**2*y, x*y**2]
    # 断言多边形在指定区域上的积分结果是否等于预期字典
    assert polytope_integrate(polygon, polys, max_degree=3) == \
        {1: 1, x: S.Half, y: S.Half, x*y: Rational(1, 4), x**2*y: Rational(1, 6), x*y**2: Rational(1, 6)}
    # 断言多边形在指定区域上的积分结果是否等于预期字典，且最大次数为2
    assert polytope_integrate(polygon, polys, max_degree=2) == \
        {1: 1, x: S.Half, y: S.Half, x*y: Rational(1, 4)}
    # 断言多边形在指定区域上的积分结果是否等于预期字典，且最大次数为1
    assert polytope_integrate(polygon, polys, max_degree=1) == \
        {1: 1, x: S.Half, y: S.Half}


# 定义用于测试三维积分的主函数
def test_main_integrate3d():
    # 创建一个立方体的顶点和面列表
    cube = [[(0, 0, 0), (0, 0, 5), (0, 5, 0), (0, 5, 5), (5, 0, 0),\
             (5, 0, 5), (5, 5, 0), (5, 5, 5)],\
            [2, 6, 7, 3], [3, 7, 5, 1], [7, 6, 4, 5], [1, 5, 4, 0],\
            [3, 1, 0, 2], [0, 4, 6, 2]]
    # 提取立方体的顶点
    vertices = cube[0]
    # 提取立方体的面
    faces = cube[1:]
    # 计算超平面参数
    hp_params = hyperplane_parameters(faces, vertices)
    # 断言在指定区域内的三维积分结果是否等于预期值
    assert main_integrate3d(1, faces, vertices, hp_params) == -125
    # 断言在指定区域内的三维积分结果是否等于预期字典，且最大次数为1
    assert main_integrate3d(1, faces, vertices, hp_params, max_degree=1) == \
        {1: -125, y: Rational(-625, 2), z: Rational(-625, 2), x: Rational(-625, 2)}


# 定义用于测试二维积分的主函数
def test_main_integrate():
    # 创建一个三角形对象
    triangle = Polygon((0, 3), (5, 3), (1, 1))
    # 提取三角形的边
    facets = triangle.sides
    # 计算超平面参数
    hp_params = hyperplane_parameters(triangle)
    # 断言在指定区域内的二维积分结果是否等于预期有理数
    assert main_integrate(x**2 + y**2, facets, hp_params) == Rational(325, 6)
    # 断言在指定区域内的二维积分结果是否等于预期字典，且最大次数为1
    assert main_integrate(x**2 + y**2, facets, hp_params, max_degree=1) == \
        {0: 0, 1: 5, y: Rational(35, 3), x: 10}


# 定义用于测试多边形积分的函数
def test_polygon_integrate():
    # 创建一个立方体的顶点和面列表
    cube = [[(0, 0, 0), (0, 0, 5), (0, 5, 0), (0, 5, 5), (5, 0, 0),\
             (5, 0, 5), (5, 5, 0), (5, 5, 5)],\
            [2, 6, 7, 3], [3, 7, 5, 1], [7, 6, 4, 5], [1, 5, 4, 0],\
            [3, 1, 0, 2], [0, 4, 6, 2]]
    # 提取一个面的顶点列表
    facet = cube[1]
    # 提取立方体的所有面
    facets = cube[1:]
    # 提取立方体的所有顶点
    vertices = cube[0]
    # 断言在指定面的区域内的多边形积分结果是否等于预期值
    assert polygon_integrate(facet, [(0, 1, 0), 5], 0, facets, vertices, 1, 0) == -25


# 定义用于测试到边距离的函数
def test_distance_to_side():
    # 创建一个点
    point = (0, 0, 0)
    # 断言点到指定边的距离是否等于预期值
    assert distance_to_side(point, [(0, 0, 1), (0, 1, 0)], (1, 0, 0)) == -sqrt(2)/2
# 定义测试函数 `test_lineseg_integrate()`，用于测试 `lineseg_integrate` 函数
def test_lineseg_integrate():
    # 定义一个多边形，包含四个顶点
    polygon = [(0, 5, 0), (5, 5, 0), (5, 5, 5), (0, 5, 5)]
    # 定义一个线段，包含两个端点
    line_seg = [(0, 5, 0), (5, 5, 0)]
    # 断言调用 `lineseg_integrate` 函数返回结果为 5
    assert lineseg_integrate(polygon, 0, line_seg, 1, 0) == 5
    # 断言调用 `lineseg_integrate` 函数返回结果为 0
    assert lineseg_integrate(polygon, 0, line_seg, 0, 0) == 0


# 定义测试函数 `test_integration_reduction()`，用于测试 `integration_reduction` 函数
def test_integration_reduction():
    # 创建一个三角形对象
    triangle = Polygon(Point(0, 3), Point(5, 3), Point(1, 1))
    # 获取三角形的边
    facets = triangle.sides
    # 计算三角形的超平面参数，取第一个参数值
    a, b = hyperplane_parameters(triangle)[0]
    # 断言调用 `integration_reduction` 函数返回结果为 5
    assert integration_reduction(facets, 0, a, b, 1, (x, y), 0) == 5
    # 断言调用 `integration_reduction` 函数返回结果为 0
    assert integration_reduction(facets, 0, a, b, 0, (x, y), 0) == 0


# 定义测试函数 `test_integration_reduction_dynamic()`，用于测试 `integration_reduction_dynamic` 函数
def test_integration_reduction_dynamic():
    # 创建一个三角形对象
    triangle = Polygon(Point(0, 3), Point(5, 3), Point(1, 1))
    # 获取三角形的边
    facets = triangle.sides
    # 计算三角形的超平面参数，取第一个参数值
    a, b = hyperplane_parameters(triangle)[0]
    # 获取三角形的第一个边的第一个点
    x0 = facets[0].points[0]
    # 定义一个多项式值的列表
    monomial_values = [[0, 0, 0, 0], [1, 0, 0, 5],\
                       [y, 0, 1, 15], [x, 1, 0, None]]
    # 断言调用 `integration_reduction_dynamic` 函数返回结果为 25/2
    assert integration_reduction_dynamic(facets, 0, a, b, x, 1, (x, y), 1,\
                                         0, 1, x0, monomial_values, 3) == Rational(25, 2)
    # 断言调用 `integration_reduction_dynamic` 函数返回结果为 0
    assert integration_reduction_dynamic(facets, 0, a, b, 0, 1, (x, y), 1,\
                                         0, 1, x0, monomial_values, 3) == 0


# 定义测试函数 `test_is_vertex()`，用于测试 `is_vertex` 函数
def test_is_vertex():
    # 断言调用 `is_vertex` 函数，参数为整数 2 返回 False
    assert is_vertex(2) is False
    # 断言调用 `is_vertex` 函数，参数为元组 (2, 3) 返回 True
    assert is_vertex((2, 3)) is True
    # 断言调用 `is_vertex` 函数，参数为点对象 Point(2, 3) 返回 True
    assert is_vertex(Point(2, 3)) is True
    # 断言调用 `is_vertex` 函数，参数为元组 (2, 3, 4) 返回 True
    assert is_vertex((2, 3, 4)) is True
    # 断言调用 `is_vertex` 函数，参数为元组 (2, 3, 4, 5) 返回 False
    assert is_vertex((2, 3, 4, 5)) is False


# 定义测试函数 `test_issue_19234()`，用于测试 `polytope_integrate` 函数
def test_issue_19234():
    # 创建一个四边形对象
    polygon = Polygon(Point(0, 0), Point(0, 1), Point(1, 1), Point(1, 0))
    # 定义多项式列表
    polys =  [ 1, x, y, x*y, x**2*y, x*y**2]
    # 断言调用 `polytope_integrate` 函数返回结果为特定的字典
    assert polytope_integrate(polygon, polys) == \
        {1: 1, x: S.Half, y: S.Half, x*y: Rational(1, 4), x**2*y: Rational(1, 6), x*y**2: Rational(1, 6)}
    # 重新定义多项式列表
    polys =  [ 1, x, y, x*y, 3 + x**2*y, x + x*y**2]
    # 断言调用 `polytope_integrate` 函数返回结果为特定的字典
    assert polytope_integrate(polygon, polys) == \
        {1: 1, x: S.Half, y: S.Half, x*y: Rational(1, 4), x**2*y + 3: Rational(19, 6), x*y**2 + x: Rational(2, 3)}
```