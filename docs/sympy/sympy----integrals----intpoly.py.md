# `D:\src\scipysrc\sympy\sympy\integrals\intpoly.py`

```
"""
Module to implement integration of uni/bivariate polynomials over
2D Polytopes and uni/bi/trivariate polynomials over 3D Polytopes.

Uses evaluation techniques as described in Chin et al. (2015) [1].


References
===========

.. [1] Chin, Eric B., Jean B. Lasserre, and N. Sukumar. "Numerical integration
of homogeneous functions on convex and nonconvex polygons and polyhedra."
Computational Mechanics 56.6 (2015): 967-981

PDF link : http://dilbert.engr.ucdavis.edu/~suku/quadrature/cls-integration.pdf
"""

from functools import cmp_to_key

from sympy.abc import x, y, z
from sympy.core import S, diff, Expr, Symbol
from sympy.core.sympify import _sympify
from sympy.geometry import Segment2D, Polygon, Point, Point2D
from sympy.polys.polytools import LC, gcd_list, degree_list, Poly
from sympy.simplify.simplify import nsimplify


def polytope_integrate(poly, expr=None, *, clockwise=False, max_degree=None):
    """Integrates polynomials over 2/3-Polytopes.

    Explanation
    ===========

    This function accepts the polytope in ``poly`` and the function in ``expr``
    (uni/bi/trivariate polynomials are implemented) and returns
    the exact integral of ``expr`` over ``poly``.

    Parameters
    ==========

    poly : The input Polygon.

    expr : The input polynomial.

    clockwise : Binary value to sort input points of 2-Polytope clockwise.(Optional)

    max_degree : The maximum degree of any monomial of the input polynomial.(Optional)

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy import Point, Polygon
    >>> from sympy.integrals.intpoly import polytope_integrate
    >>> polygon = Polygon(Point(0, 0), Point(0, 1), Point(1, 1), Point(1, 0))
    >>> polys = [1, x, y, x*y, x**2*y, x*y**2]
    >>> expr = x*y
    >>> polytope_integrate(polygon, expr)
    1/4
    >>> polytope_integrate(polygon, polys, max_degree=3)
    {1: 1, x: 1/2, y: 1/2, x*y: 1/4, x*y**2: 1/6, x**2*y: 1/6}
    """
    if clockwise:
        # 如果需要按顺时针方向排序输入点，而且输入为多边形类型，就对其顶点进行排序
        if isinstance(poly, Polygon):
            poly = Polygon(*point_sort(poly.vertices), evaluate=False)
        else:
            raise TypeError("clockwise=True works for only 2-Polytope"
                            "V-representation input")

    if isinstance(poly, Polygon):
        # 对于顶点表示法（2D 情况）
        # 计算多边形的超平面参数
        hp_params = hyperplane_parameters(poly)
        # 获取多边形的所有边
        facets = poly.sides
    # 如果多边形的第一个子列表的长度为2，则处理二维超平面表示法的情况
    elif len(poly[0]) == 2:
        # 获取多边形的长度
        plen = len(poly)
        # 如果多边形的第一个顶点的长度也为2，则计算交点
        if len(poly[0][0]) == 2:
            # 计算多边形各边与二维平面的交点
            intersections = [intersection(poly[(i - 1) % plen], poly[i],
                                          "plane2D")
                             for i in range(0, plen)]
            # 复制多边形参数作为超平面参数
            hp_params = poly
            # 计算交点的数量
            lints = len(intersections)
            # 创建二维线段对象列表，连接相邻的交点
            facets = [Segment2D(intersections[i],
                                intersections[(i + 1) % lints])
                      for i in range(lints)]
        else:
            # 抛出未实现错误，因为三维H表示法的情况尚未实现
            raise NotImplementedError("Integration for H-representation 3D "
                                      "case not implemented yet.")
    else:
        # 处理三维顶点表示法的情况
        vertices = poly[0]
        facets = poly[1:]
        # 计算超平面参数
        hp_params = hyperplane_parameters(facets, vertices)

        # 如果未提供最大次数，则进行三维积分计算
        if max_degree is None:
            if expr is None:
                # 如果未提供表达式，则抛出类型错误
                raise TypeError('Input expression must be a valid SymPy expression')
            # 调用主要的三维积分函数进行计算
            return main_integrate3d(expr, facets, vertices, hp_params)

    # 如果提供了最大次数限制，则进行以下操作
    if max_degree is not None:
        # 初始化结果字典
        result = {}
        # 如果提供了表达式，则处理表达式
        if expr is not None:
            f_expr = []
            # 遍历表达式列表
            for e in expr:
                # 对表达式进行分解
                _ = decompose(e)
                # 如果表达式只有一个元素且为零，则将其添加到新表达式列表中
                if len(_) == 1 and not _.popitem()[0]:
                    f_expr.append(e)
                # 否则，检查多项式的总次数是否小于等于最大次数限制
                elif Poly(e).total_degree() <= max_degree:
                    f_expr.append(e)
            # 更新表达式列表
            expr = f_expr

        # 如果表达式不是列表类型且不为空，则抛出类型错误
        if not isinstance(expr, list) and expr is not None:
            raise TypeError('Input polynomials must be list of expressions')

        # 检查超平面参数的第一个顶点长度是否为3，选择对应的积分函数进行计算
        if len(hp_params[0][0]) == 3:
            result_dict = main_integrate3d(0, facets, vertices, hp_params,
                                           max_degree)
        else:
            result_dict = main_integrate(0, facets, hp_params, max_degree)

        # 如果未提供表达式，则直接返回结果字典
        if expr is None:
            return result_dict

        # 遍历每个多项式，并计算其积分值
        for poly in expr:
            # 将多项式转换为SymPy表达式
            poly = _sympify(poly)
            # 如果多项式不在结果字典中，则进行计算
            if poly not in result:
                # 如果多项式为零，则结果也为零
                if poly.is_zero:
                    result[S.Zero] = S.Zero
                    continue
                # 初始化积分值
                integral_value = S.Zero
                # 分解多项式成单项式，并分别计算每个单项式的积分值
                monoms = decompose(poly, separate=True)
                for monom in monoms:
                    monom = nsimplify(monom)
                    coeff, m = strip(monom)
                    integral_value += result_dict[m] * coeff
                # 将计算得到的积分值存入结果字典
                result[poly] = integral_value
        # 返回最终的结果字典
        return result

    # 如果未提供表达式，则抛出类型错误
    if expr is None:
        raise TypeError('Input expression must be a valid SymPy expression')

    # 使用主要的积分函数计算结果并返回
    return main_integrate(expr, facets, hp_params)
def strip(monom):
    # 如果单项式为零，返回 (0, 0)
    if monom.is_zero:
        return S.Zero, S.Zero
    # 如果单项式为数值，返回 (单项式值, 1)
    elif monom.is_number:
        return monom, S.One
    else:
        # 获取单项式的首项系数
        coeff = LC(monom)
        # 返回首项系数和单项式除以首项系数后的结果
        return coeff, monom / coeff

def _polynomial_integrate(polynomials, facets, hp_params):
    # 定义变量维度为 (x, y)
    dims = (x, y)
    dim_length = len(dims)
    # 初始化积分值为 0
    integral_value = S.Zero
    # 遍历多项式的每个阶次
    for deg in polynomials:
        # 初始化多项式的贡献为 0
        poly_contribute = S.Zero
        # 初始化面计数为 0
        facet_count = 0
        # 遍历超平面参数列表
        for hp in hp_params:
            # 计算在给定超平面上的积分值
            value_over_boundary = integration_reduction(facets,
                                                        facet_count,
                                                        hp[0], hp[1],
                                                        polynomials[deg],
                                                        dims, deg)
            # 计算当前超平面的贡献，并累加到多项式的贡献中
            poly_contribute += value_over_boundary * (hp[1] / norm(hp[0]))
            # 增加面计数
            facet_count += 1
        # 将多项式的贡献除以 (变量维度数 + 多项式阶次)，并累加到积分值中
        poly_contribute /= (dim_length + deg)
        integral_value += poly_contribute

    # 返回积分值
    return integral_value


def main_integrate3d(expr, facets, vertices, hp_params, max_degree=None):
    """Function to translate the problem of integrating uni/bi/tri-variate
    polynomials over a 3-Polytope to integrating over its faces.
    This is done using Generalized Stokes' Theorem and Euler's Theorem.

    Parameters
    ==========

    expr :
        The input polynomial.
    facets :
        Faces of the 3-Polytope(expressed as indices of `vertices`).
    vertices :
        Vertices that constitute the Polytope.
    hp_params :
        Hyperplane Parameters of the facets.
    max_degree : optional
        Max degree of constituent monomial in given list of polynomial.

    Examples
    ========

    >>> from sympy.integrals.intpoly import main_integrate3d, \
    hyperplane_parameters
    >>> cube = [[(0, 0, 0), (0, 0, 5), (0, 5, 0), (0, 5, 5), (5, 0, 0),\
                (5, 0, 5), (5, 5, 0), (5, 5, 5)],\
                [2, 6, 7, 3], [3, 7, 5, 1], [7, 6, 4, 5], [1, 5, 4, 0],\
                [3, 1, 0, 2], [0, 4, 6, 2]]
    >>> vertices = cube[0]
    >>> faces = cube[1:]
    >>> hp_params = hyperplane_parameters(faces, vertices)
    >>> main_integrate3d(1, faces, vertices, hp_params)
    -125
    """
    # 初始化结果字典为空
    result = {}
    # 定义变量维度为 (x, y, z)
    dims = (x, y, z)
    dim_length = len(dims)
    # 如果 max_degree 不为零，执行以下代码块
    if max_degree:
        # 使用 gradient_terms 函数生成指定最大次数的梯度项列表
        grad_terms = gradient_terms(max_degree, 3)
        # 将梯度项列表展平成一维列表
        flat_list = [term for z_terms in grad_terms
                     for x_term in z_terms
                     for term in x_term]

        # 遍历 flat_list 中的每个项，将其第一个元素作为 result 字典的键，值初始化为 0
        for term in flat_list:
            result[term[0]] = 0

        # 遍历 hp_params 列表中的元素，每个元素包含两个值 a 和 b
        for facet_count, hp in enumerate(hp_params):
            a, b = hp[0], hp[1]
            # 获取顶点 vertices 中 facets[facet_count][0] 对应的坐标 x0
            x0 = vertices[facets[facet_count][0]]

            # 遍历 flat_list 中的每个元素，每个元素是一个长度为 8 的元组
            for i, monom in enumerate(flat_list):
                # 解包 monom 元组，分别获取表达式 expr 和 x_d, y_d, z_d 的度数
                expr, x_d, y_d, z_d, z_index, y_index, x_index, _ = monom
                # 计算当前单项式的总度数
                degree = x_d + y_d + z_d

                # 如果 b 是零，则 value_over_face 等于 0，否则调用 integration_reduction_dynamic 函数计算值
                if b.is_zero:
                    value_over_face = S.Zero
                else:
                    value_over_face = \
                        integration_reduction_dynamic(facets, facet_count, a,
                                                      b, expr, degree, dims,
                                                      x_index, y_index,
                                                      z_index, x0, grad_terms,
                                                      i, vertices, hp)
                # 更新 monom 的第 7 个元素为计算得到的 value_over_face
                monom[7] = value_over_face
                # 将计算结果加到 result[expr] 中
                result[expr] += value_over_face * \
                    (b / norm(a)) / (dim_length + x_d + y_d + z_d)
        # 返回最终的 result 字典
        return result
    else:
        # 如果 max_degree 是零，则执行以下代码块
        integral_value = S.Zero
        # 对表达式 expr 进行分解，得到多项式列表 polynomials
        polynomials = decompose(expr)
        # 遍历多项式列表中的每个元素 deg
        for deg in polynomials:
            poly_contribute = S.Zero
            facet_count = 0
            # 遍历 facets 列表中的每个元素 facet，同时获取对应的 hp 参数
            for i, facet in enumerate(facets):
                hp = hp_params[i]
                # 如果 hp 的第二个元素是零，跳过当前循环
                if hp[1].is_zero:
                    continue
                # 调用 polygon_integrate 函数计算多边形 facet 上的积分 pi
                pi = polygon_integrate(facet, hp, i, facets, vertices, expr, deg)
                # 计算当前多项式对积分的贡献，并累加到 poly_contribute
                poly_contribute += pi *\
                    (hp[1] / norm(tuple(hp[0])))
                facet_count += 1
            # 将 poly_contribute 除以 dim_length + deg，并累加到 integral_value 中
            poly_contribute /= (dim_length + deg)
            integral_value += poly_contribute
        # 返回最终的积分结果 integral_value
        return integral_value
# 定义主函数，用于将多变量多项式在二维多面体的边界面上积分，应用广义斯托克斯定理和欧拉定理
def main_integrate(expr, facets, hp_params, max_degree=None):
    """Function to translate the problem of integrating univariate/bivariate
    polynomials over a 2-Polytope to integrating over its boundary facets.
    This is done using Generalized Stokes's Theorem and Euler's Theorem.

    Parameters
    ==========

    expr :
        The input polynomial.
    facets :
        Facets(Line Segments) of the 2-Polytope.
    hp_params :
        Hyperplane Parameters of the facets.
    max_degree : optional
        The maximum degree of any monomial of the input polynomial.

    >>> from sympy.abc import x, y
    >>> from sympy.integrals.intpoly import main_integrate,\
    hyperplane_parameters
    >>> from sympy import Point, Polygon
    >>> triangle = Polygon(Point(0, 3), Point(5, 3), Point(1, 1))
    >>> facets = triangle.sides
    >>> hp_params = hyperplane_parameters(triangle)
    >>> main_integrate(x**2 + y**2, facets, hp_params)
    325/6
    """
    # 定义变量维度
    dims = (x, y)
    # 计算变量维度的长度
    dim_length = len(dims)
    # 初始化结果字典
    result = {}

    # 如果指定了最大次数
    if max_degree:
        # 计算梯度项
        grad_terms = [[0, 0, 0, 0]] + gradient_terms(max_degree)

        # 遍历超平面参数列表
        for facet_count, hp in enumerate(hp_params):
            a, b = hp[0], hp[1]
            # 获取当前面片的第一个点
            x0 = facets[facet_count].points[0]

            # 遍历梯度项列表
            for i, monom in enumerate(grad_terms):
                # 每个单项式是一个元组：
                # (项，x次数，y次数，边界上的值)
                m, x_d, y_d, _ = monom
                # 获取当前项在结果字典中的值
                value = result.get(m, None)
                # 初始化次数为零
                degree = S.Zero
                # 如果 b 是零，则边界上的值为零
                if b.is_zero:
                    value_over_boundary = S.Zero
                else:
                    # 动态降维积分计算边界上的值
                    degree = x_d + y_d
                    value_over_boundary = \
                        integration_reduction_dynamic(facets, facet_count, a,
                                                      b, m, degree, dims, x_d,
                                                      y_d, max_degree, x0,
                                                      grad_terms, i)
                # 更新当前单项式的边界上的值
                monom[3] = value_over_boundary
                # 如果结果字典中已经存在该项，则加上边界上的值乘以系数
                if value is not None:
                    result[m] += value_over_boundary * \
                                        (b / norm(a)) / (dim_length + degree)
                # 否则将该项的边界上的值乘以系数加入结果字典
                else:
                    result[m] = value_over_boundary * \
                                (b / norm(a)) / (dim_length + degree)
        return result
    else:
        # 如果表达式不是列表，则进行分解后积分
        if not isinstance(expr, list):
            polynomials = decompose(expr)
            return _polynomial_integrate(polynomials, facets, hp_params)
        else:
            # 如果表达式是列表，则分别对每个表达式进行分解后积分
            return {e: _polynomial_integrate(decompose(e), facets, hp_params) for e in expr}


def polygon_integrate(facet, hp_param, index, facets, vertices, expr, degree):
    """Helper function to integrate the input uni/bi/trivariate polynomial
    over a certain face of the 3-Polytope.

    Parameters
    ==========

    facet :
        Particular face of the 3-Polytope over which ``expr`` is integrated.
    expr = S(expr)
    # 将输入的表达式转换为 SymPy 表达式对象

    if expr.is_zero:
        # 如果表达式为零，直接返回零
        return S.Zero
    
    result = S.Zero
    # 初始化结果为零
    
    x0 = vertices[facet[0]]
    # 取出 facet 中第一个顶点的坐标作为起始点 x0
    
    facet_len = len(facet)
    # 获取 facet 的长度，即顶点数量
    
    for i, fac in enumerate(facet):
        # 遍历 facet 中的每个顶点及其索引
        
        side = (vertices[fac], vertices[facet[(i + 1) % facet_len]])
        # 计算当前顶点 fac 与下一个顶点之间的边 side
        
        result += distance_to_side(x0, side, hp_param[0]) *\
            lineseg_integrate(facet, i, side, expr, degree)
        # 将边 side 的距离乘以边上的线段积分结果累加到 result 中
    
    if not expr.is_number:
        # 如果表达式不是数值
        
        expr = diff(expr, x) * x0[0] + diff(expr, y) * x0[1] +\
            diff(expr, z) * x0[2]
        # 对表达式在 x, y, z 方向上求偏导数，并分别乘以 x0 的对应坐标
        
        result += polygon_integrate(facet, hp_param, index, facets, vertices,
                                    expr, degree - 1)
        # 递归调用 polygon_integrate 函数，计算当前多边形 facet 的积分
        
    result /= (degree + 2)
    # 将结果除以 degree + 2
    
    return result
    # 返回计算得到的结果
# 计算给定3D点到线段的有向距离的辅助函数

def distance_to_side(point, line_seg, A):
    """Helper function to compute the signed distance between given 3D point
    and a line segment.

    Parameters
    ==========

    point : 3D Point
        给定的三维点
    line_seg : Line Segment
        线段表示为两个端点的元组列表
    A : Vector
        给定的向量

    Examples
    ========

    >>> from sympy.integrals.intpoly import distance_to_side
    >>> point = (0, 0, 0)
    >>> distance_to_side(point, [(0, 0, 1), (0, 1, 0)], (1, 0, 0))
    -sqrt(2)/2
    """
    x1, x2 = line_seg
    # 计算A的归一化反向向量
    rev_normal = [-1 * S(i)/norm(A) for i in A]
    # 计算线段的向量并归一化
    vector = [x2[i] - x1[i] for i in range(0, 3)]
    vector = [vector[i]/norm(vector) for i in range(0, 3)]

    # 计算法向量
    n_side = cross_product((0, 0, 0), rev_normal, vector)
    # 计算点到线段的有向距离
    vectorx0 = [line_seg[0][i] - point[i] for i in range(0, 3)]
    dot_product = sum(vectorx0[i] * n_side[i] for i in range(0, 3))

    return dot_product


# 计算表达式在给定线段上的线积分的辅助函数

def lineseg_integrate(polygon, index, line_seg, expr, degree):
    """Helper function to compute the line integral of ``expr`` over ``line_seg``.

    Parameters
    ===========

    polygon :
        3-Polytope的一个面。
    index :
        line_seg在polygon中的索引。
    line_seg :
        线段表示为两个端点的元组列表。
    expr :
        要在线段上积分的表达式。
    degree :
        齐次多项式的次数。

    Examples
    ========

    >>> from sympy.integrals.intpoly import lineseg_integrate
    >>> polygon = [(0, 5, 0), (5, 5, 0), (5, 5, 5), (0, 5, 5)]
    >>> line_seg = [(0, 5, 0), (5, 5, 0)]
    >>> lineseg_integrate(polygon, 0, line_seg, 1, 0)
    5
    """
    expr = _sympify(expr)
    if expr.is_zero:
        return S.Zero
    result = S.Zero
    x0 = line_seg[0]
    # 计算线段长度
    distance = norm(tuple([line_seg[1][i] - line_seg[0][i] for i in range(3)]))

    if isinstance(expr, Expr):
        # 对表达式进行符号化并替换变量
        expr_dict = {x: line_seg[1][0],
                     y: line_seg[1][1],
                     z: line_seg[1][2]}
        result += distance * expr.subs(expr_dict)
    else:
        result += distance * expr

    # 计算表达式在线段上的梯度并积分
    expr = diff(expr, x) * x0[0] + diff(expr, y) * x0[1] +\
        diff(expr, z) * x0[2]

    result += lineseg_integrate(polygon, index, line_seg, expr, degree - 1)
    result /= (degree + 1)
    return result


# 主积分的辅助方法，返回在给定索引处的多面体面上计算的表达式值

def integration_reduction(facets, index, a, b, expr, dims, degree):
    """Helper method for main_integrate. Returns the value of the input
    expression evaluated over the polytope facet referenced by a given index.

    Parameters
    ===========

    facets :
        多面体的面列表。
    index :
        引用要在其上积分表达式的面的索引。
    a :
        表示方向的超平面参数。
    b :
        表示距离的超平面参数。
    expr :
        要在面上积分的表达式。
    dims :
        表示轴的符号列表。
    degree :
        齐次多项式的次数。

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy.integrals.intpoly import integration_reduction,\
    hyperplane_parameters
    >>> from sympy import Point, Polygon

    """
    """
    将表达式转换为符号表达式对象（如果不是的话）
    """
    expr = _sympify(expr)
    
    """
    如果表达式为零，直接返回表达式对象
    """
    if expr.is_zero:
        return expr
    
    """
    初始化值为零
    """
    value = S.Zero
    
    """
    获取多边形边的第一个点
    """
    x0 = facets[index].points[0]
    
    """
    多边形的边数
    """
    m = len(facets)
    
    """
    生成器变量，用于求导
    """
    gens = (x, y)
    
    """
    计算表达式关于 gens[0] 的偏导数乘以 x0[0]，以及关于 gens[1] 的偏导数乘以 x0[1] 的内积
    """
    inner_product = diff(expr, gens[0]) * x0[0] + diff(expr, gens[1]) * x0[1]
    
    """
    如果内积不为零，则进行积分减少操作
    """
    if inner_product != 0:
        value += integration_reduction(facets, index, a, b,
                                       inner_product, dims, degree - 1)
    
    """
    对左侧的二维积分进行计算
    """
    value += left_integral2D(m, index, facets, x0, expr, gens)
    
    """
    返回计算结果除以 dims 的长度加上 degree 减一的值
    """
    return value/(len(dims) + degree - 1)
# 定义函数，计算 Chin 等人文章中方程 10 的二维左侧积分
def left_integral2D(m, index, facets, x0, expr, gens):
    """
    计算 Chin 等人文章中方程 10 的左侧积分。对于二维情况，积分仅是多项式在两个面片交点处的评估，
    乘以面片第一个点与该交点之间的距离。

    Parameters
    ==========

    m :
        超平面的数量。
    index :
        要查找与之交点的面片的索引。
    facets :
        面片列表（在二维情况下是线段）。
    x0 :
        与索引相关的面片上的第一个点。
    expr :
        输入多项式。
    gens :
        生成多项式的变量。

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy.integrals.intpoly import left_integral2D
    >>> from sympy import Point, Polygon
    >>> triangle = Polygon(Point(0, 3), Point(5, 3), Point(1, 1))
    >>> facets = triangle.sides
    >>> left_integral2D(3, 0, facets, facets[0].points[0], 1, (x, y))
    5
    """
    # 初始化积分值为零
    value = S.Zero
    # 遍历所有超平面
    for j in range(m):
        intersect = ()
        # 如果 j 是 index 前一个或后一个超平面的索引
        if j in ((index - 1) % m, (index + 1) % m):
            # 计算 facets[index] 和 facets[j] 的交点
            intersect = intersection(facets[index], facets[j], "segment2D")
        # 如果存在交点
        if intersect:
            # 计算交点与 x0 的距离
            distance_origin = norm(tuple(map(lambda x, y: x - y,
                                             intersect, x0)))
            # 如果交点是顶点
            if is_vertex(intersect):
                # 如果 expr 是表达式类型
                if isinstance(expr, Expr):
                    # 根据 gens 的长度创建相应的字典
                    if len(gens) == 3:
                        expr_dict = {gens[0]: intersect[0],
                                     gens[1]: intersect[1],
                                     gens[2]: intersect[2]}
                    else:
                        expr_dict = {gens[0]: intersect[0],
                                     gens[1]: intersect[1]}
                    # 计算并累加积分值
                    value += distance_origin * expr.subs(expr_dict)
                else:
                    # 如果 expr 是常量，直接计算并累加积分值
                    value += distance_origin * expr
    # 返回积分值
    return value


def integration_reduction_dynamic(facets, index, a, b, expr, degree, dims,
                                  x_index, y_index, max_index, x0,
                                  monomial_values, monom_index, vertices=None,
                                  hp_param=None):
    """
    使用动态规划方法计算积分约化函数，该函数使用先前计算的积分值来计算项的值。

    Parameters
    ==========

    facets :
        多面体的面片。
    index :
        要查找与之交点的面片的索引。（在 left_integral() 中使用）
    a, b :
        超平面参数。
    expr :
        输入的单项式。
    degree :
        ``expr`` 的总次数。
    dims :
        表示轴变量的元组。
    x_index :
        ``expr`` 中 'x' 的指数。
    y_index :
        ``expr`` 中 'y' 的指数。
    max_index :
        ``monomial_values`` 中任何单项式的最大指数。
    x0 :
        ``facets[index]`` 上的第一个点。

    vertices : optional
        多面体的顶点。
    hp_param : optional
        超平面的参数。
    """
    # 函数体还没有给出，不在这里注释
    pass
    """
    Calculate the integration of a polynomial over a 2D or 3D domain represented by facets.

    Parameters
    ----------
    monomial_values : list
        List of monomial values constituting the polynomial.
    monom_index : int
        Index of monomial whose integration is being found.
    vertices : optional
        Coordinates of vertices constituting the 3-Polytope.
    hp_param : optional
        Hyperplane Parameter of the face of the facets[index].

    Returns
    -------
    value : sympy expression
        Result of the integration.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy.integrals.intpoly import (integration_reduction_dynamic, \
            hyperplane_parameters)
    >>> from sympy import Point, Polygon
    >>> triangle = Polygon(Point(0, 3), Point(5, 3), Point(1, 1))
    >>> facets = triangle.sides
    >>> a, b = hyperplane_parameters(triangle)[0]
    >>> x0 = facets[0].points[0]
    >>> monomial_values = [[0, 0, 0, 0], [1, 0, 0, 5],\
                           [y, 0, 1, 15], [x, 1, 0, None]]
    >>> integration_reduction_dynamic(facets, 0, a, b, x, 1, (x, y), 1, 0, 1,\
                                      x0, monomial_values, 3)
    25/2
    """
    # Initialize the result value to zero
    value = S.Zero
    # Number of facets
    m = len(facets)

    # If the expression is zero, return zero
    if expr == S.Zero:
        return expr

    # If the dimensions are 2 (2D case)
    if len(dims) == 2:
        # If expression is not a number, compute the contribution of x and y degrees
        if not expr.is_number:
            _, x_degree, y_degree, _ = monomial_values[monom_index]
            # Compute indices based on degrees
            x_index = monom_index - max_index + x_index - 2 if x_degree > 0 else 0
            y_index = monom_index - 1 if y_degree > 0 else 0
            # Extract x and y values from monomial_values
            x_value, y_value = \
                monomial_values[x_index][3], monomial_values[y_index][3]

            # Accumulate contribution to the value based on degrees and values
            value += x_degree * x_value * x0[0] + y_degree * y_value * x0[1]

        # Add contribution from left integral in 2D
        value += left_integral2D(m, index, facets, x0, expr, dims)
    else:
        # For 3D case, extract degrees from indices
        z_index = max_index
        if not expr.is_number:
            x_degree, y_degree, z_degree = y_index, \
                                           z_index - x_index - y_index, x_index
            # Extract x, y, z values from monomial_values
            x_value = monomial_values[z_index - 1][y_index - 1][x_index][7] \
                if x_degree > 0 else 0
            y_value = monomial_values[z_index - 1][y_index][x_index][7] \
                if y_degree > 0 else 0
            z_value = monomial_values[z_index - 1][y_index][x_index - 1][7] \
                if z_degree > 0 else 0

            # Accumulate contribution to the value based on degrees and values
            value += x_degree * x_value * x0[0] + y_degree * y_value * x0[1] \
                + z_degree * z_value * x0[2]

        # Add contribution from left integral in 3D
        value += left_integral3D(facets, index, expr,
                                 vertices, hp_param, degree)

    # Return the computed value normalized by the sum of dimensions and degree
    return value / (len(dims) + degree - 1)
def left_integral3D(facets, index, expr, vertices, hp_param, degree):
    """Computes the left integral of Eq 10 in Chin et al.

    Explanation
    ===========

    For the 3D case, this is the sum of the integral values over constituting
    line segments of the face (which is accessed by facets[index]) multiplied
    by the distance between the first point of facet and that line segment.

    Parameters
    ==========

    facets :
        List of faces of the 3-Polytope.
    index :
        Index of face over which integral is to be calculated.
    expr :
        Input polynomial.
    vertices :
        List of vertices that constitute the 3-Polytope.
    hp_param :
        The hyperplane parameters of the face.
    degree :
        Degree of the ``expr``.

    Examples
    ========

    >>> from sympy.integrals.intpoly import left_integral3D
    >>> cube = [[(0, 0, 0), (0, 0, 5), (0, 5, 0), (0, 5, 5), (5, 0, 0),\
                 (5, 0, 5), (5, 5, 0), (5, 5, 5)],\
                 [2, 6, 7, 3], [3, 7, 5, 1], [7, 6, 4, 5], [1, 5, 4, 0],\
                 [3, 1, 0, 2], [0, 4, 6, 2]]
    >>> facets = cube[1:]
    >>> vertices = cube[0]
    >>> left_integral3D(facets, 3, 1, vertices, ([0, -1, 0], -5), 0)
    -50
    """
    value = S.Zero  # 初始化积分结果为零
    facet = facets[index]  # 获取第index个面
    x0 = vertices[facet[0]]  # 获取面的第一个顶点
    facet_len = len(facet)  # 获取面的顶点数
    for i, fac in enumerate(facet):  # 遍历面的每个顶点
        side = (vertices[fac], vertices[facet[(i + 1) % facet_len]])  # 获取当前边的两个顶点
        # 计算距离当前边的距离乘以该边的积分值，累加到总积分结果中
        value += distance_to_side(x0, side, hp_param[0]) * \
            lineseg_integrate(facet, i, side, expr, degree)
    return value
    else:
        # 生成一个三重嵌套列表推导式，用于生成组合数学术语的列表
        terms = [[[[x ** x_count * y ** y_count *
                    z ** (z_count - y_count - x_count),
                    x_count, y_count, z_count - y_count - x_count,
                    z_count, x_count, z_count - y_count - x_count, 0]
                 # 第二个内部循环，遍历 y_count 从 z_count - x_count 到 0
                 for y_count in range(z_count - x_count, -1, -1)]
                 # 第一个内部循环，遍历 x_count 从 0 到 z_count
                 for x_count in range(0, z_count + 1)]
                 # 外部循环，遍历 z_count 从 0 到 binomial_power
                 for z_count in range(0, binomial_power + 1)]
    # 返回生成的 terms 列表
    return terms
def hyperplane_parameters(poly, vertices=None):
    """A helper function to return the hyperplane parameters
    of which the facets of the polytope are a part of.

    Parameters
    ==========

    poly :
        The input 2/3-Polytope.
    vertices :
        Vertex indices of 3-Polytope.

    Examples
    ========

    >>> from sympy import Point, Polygon
    >>> from sympy.integrals.intpoly import hyperplane_parameters
    >>> hyperplane_parameters(Polygon(Point(0, 3), Point(5, 3), Point(1, 1)))
    [((0, 1), 3), ((1, -2), -1), ((-2, -1), -3)]
    >>> cube = [[(0, 0, 0), (0, 0, 5), (0, 5, 0), (0, 5, 5), (5, 0, 0),\
                (5, 0, 5), (5, 5, 0), (5, 5, 5)],\
                [2, 6, 7, 3], [3, 7, 5, 1], [7, 6, 4, 5], [1, 5, 4, 0],\
                [3, 1, 0, 2], [0, 4, 6, 2]]
    >>> hyperplane_parameters(cube[1:], cube[0])
    [([0, -1, 0], -5), ([0, 0, -1], -5), ([-1, 0, 0], -5),
    ([0, 1, 0], 0), ([1, 0, 0], 0), ([0, 0, 1], 0)]
    """
    if isinstance(poly, Polygon):
        vertices = list(poly.vertices) + [poly.vertices[0]]  # Close the polygon
        params = [None] * (len(vertices) - 1)

        for i in range(len(vertices) - 1):
            v1 = vertices[i]
            v2 = vertices[i + 1]

            a1 = v1[1] - v2[1]  # Compute difference in y coordinates
            a2 = v2[0] - v1[0]  # Compute difference in x coordinates
            b = v2[0] * v1[1] - v2[1] * v1[0]  # Compute determinant for the plane equation

            factor = gcd_list([a1, a2, b])  # Compute gcd of coefficients

            b = S(b) / factor  # Normalize b coefficient
            a = (S(a1) / factor, S(a2) / factor)  # Normalize a coefficients
            params[i] = (a, b)  # Store normalized parameters
    else:
        params = [None] * len(poly)
        for i, polygon in enumerate(poly):
            v1, v2, v3 = [vertices[vertex] for vertex in polygon[:3]]

            # Compute the normal vector of the plane
            normal = cross_product(v1, v2, v3)
            b = sum(normal[j] * v1[j] for j in range(0, 3))  # Compute the b coefficient
            fac = gcd_list(normal)  # Compute gcd of normal vector coefficients
            if fac.is_zero:
                fac = 1
            normal = [j / fac for j in normal]  # Normalize normal vector
            b = b / fac  # Normalize b coefficient
            params[i] = (normal, b)  # Store normalized parameters
    return params


def cross_product(v1, v2, v3):
    """Returns the cross-product of vectors (v2 - v1) and (v3 - v1)
    That is : (v2 - v1) X (v3 - v1)
    """
    v2 = [v2[j] - v1[j] for j in range(0, 3)]  # Compute difference for vector v2
    v3 = [v3[j] - v1[j] for j in range(0, 3)]  # Compute difference for vector v3
    return [v3[2] * v2[1] - v3[1] * v2[2],  # Compute cross-product components
            v3[0] * v2[2] - v3[2] * v2[0],
            v3[1] * v2[0] - v3[0] * v2[1]]


def best_origin(a, b, lineseg, expr):
    """Helper method for polytope_integrate. Currently not used in the main
    algorithm.

    Explanation
    ===========

    Returns a point on the lineseg whose vector inner product with the
    divergence of `expr` yields an expression with the least maximum
    total power.

    Parameters
    ==========

    a :
        Hyperplane parameter denoting direction.
    b :
        Hyperplane parameter denoting distance.
    lineseg :
        Line segment on which to find the origin.
    expr :
        The expression which determines the best point.

    Algorithm(currently works only for 2D use case)
    a1, b1 = lineseg.points[0]
    ```
    # 从线段对象中获取第一个点的坐标，分别赋给 a1 和 b1

    def x_axis_cut(ls):
        """
        Returns the point where the input line segment
        intersects the x-axis.

        Parameters
        ==========

        ls :
            Line segment
        """
        # 获取线段的两个端点
        p, q = ls.points
        # 如果第一个端点的纵坐标为零，则返回该端点的坐标
        if p.y.is_zero:
            return tuple(p)
        # 如果第二个端点的纵坐标为零，则返回该端点的坐标
        elif q.y.is_zero:
            return tuple(q)
        # 如果第一个端点的纵坐标与第二个端点的纵坐标异号，则计算 x 轴交点坐标
        elif p.y/q.y < S.Zero:
            return p.y * (p.x - q.x)/(q.y - p.y) + p.x, S.Zero
        else:
            return ()

    def y_axis_cut(ls):
        """
        Returns the point where the input line segment
        intersects the y-axis.

        Parameters
        ==========

        ls :
            Line segment
        """
        # 获取线段的两个端点
        p, q = ls.points
        # 如果第一个端点的横坐标为零，则返回该端点的坐标
        if p.x.is_zero:
            return tuple(p)
        # 如果第二个端点的横坐标为零，则返回该端点的坐标
        elif q.x.is_zero:
            return tuple(q)
        # 如果第一个端点的横坐标与第二个端点的横坐标异号，则计算 y 轴交点坐标
        elif p.x/q.x < S.Zero:
            return S.Zero, p.x * (p.y - q.y)/(q.x - p.x) + p.y
        else:
            return ()

    gens = (x, y)
    # 初始化一个空字典用于存储每个生成器的幂次数
    power_gens = {}

    for i in gens:
        # 将每个生成器初始化为幂次数为零
        power_gens[i] = S.Zero
    # 如果生成器列表长度大于1
    if len(gens) > 1:
        # 处理生成器列表长度为2的特殊情况（垂直和水平线）
        if len(gens) == 2:
            # 如果第一个系数a的第一个元素为0
            if a[0] == 0:
                # 如果与y轴有交点，则返回零常量和b除以a的第二个元素
                if y_axis_cut(lineseg):
                    return S.Zero, b / a[1]
                else:
                    return a1, b1
            # 如果第一个系数a的第二个元素为0
            elif a[1] == 0:
                # 如果与x轴有交点，则返回b除以a的第一个元素和零常量
                if x_axis_cut(lineseg):
                    return b / a[0], S.Zero
                else:
                    return a1, b1

        # 如果表达式是表达式类型，找到每个生成器的幂次和，并存储在一个字典中
        if isinstance(expr, Expr):
            # 如果表达式是加法类型，对每个单项式进行处理
            if expr.is_Add:
                for monomial in expr.args:
                    # 如果单项式是幂次类型且第一个参数在生成器列表中
                    if monomial.is_Pow:
                        if monomial.args[0] in gens:
                            power_gens[monomial.args[0]] += monomial.args[1]
                    else:
                        for univariate in monomial.args:
                            term_type = len(univariate.args)
                            # 如果单项式没有参数且在生成器列表中
                            if term_type == 0 and univariate in gens:
                                power_gens[univariate] += 1
                            # 如果单项式有两个参数且第一个参数在生成器列表中
                            elif term_type == 2 and univariate.args[0] in gens:
                                power_gens[univariate.args[0]] += univariate.args[1]
            # 如果表达式是乘法类型，对每个项进行处理
            elif expr.is_Mul:
                for term in expr.args:
                    term_type = len(term.args)
                    # 如果项没有参数且在生成器列表中
                    if term_type == 0 and term in gens:
                        power_gens[term] += 1
                    # 如果项有两个参数且第一个参数在生成器列表中
                    elif term_type == 2 and term.args[0] in gens:
                        power_gens[term.args[0]] += term.args[1]
            # 如果表达式是幂次类型，直接存储其幂次
            elif expr.is_Pow:
                power_gens[expr.args[0]] = expr.args[1]
            # 如果表达式是符号类型，在字典中对应生成器的幂次加一
            elif expr.is_Symbol:
                power_gens[expr] += 1
        else:
            # 如果表达式是常数，则返回线段的第一个顶点
            return a1, b1

        # 对生成器的幂次按生成器名称进行排序
        power_gens = sorted(power_gens.items(), key=lambda k: str(k[0]))
        # 比较两个生成器的幂次，选择相应的返回值
        if power_gens[0][1] >= power_gens[1][1]:
            if y_axis_cut(lineseg):
                x0 = (S.Zero, b / a[1])
            elif x_axis_cut(lineseg):
                x0 = (b / a[0], S.Zero)
            else:
                x0 = (a1, b1)
        else:
            if x_axis_cut(lineseg):
                x0 = (b / a[0], S.Zero)
            elif y_axis_cut(lineseg):
                x0 = (S.Zero, b / a[1])
            else:
                x0 = (a1, b1)
    else:
        # 如果生成器列表长度为1，返回b除以a的第一个元素
        x0 = (b / a[0])
    return x0
# 将输入的多项式分解为同次的较小或相等次数的多项式

def decompose(expr, separate=False):
    # 初始化一个空字典来存储分解后的多项式
    poly_dict = {}

    # 检查输入是否是多项式表达式并且不是单个数值
    if isinstance(expr, Expr) and not expr.is_number:
        # 如果表达式是符号，则其次数为1
        if expr.is_Symbol:
            poly_dict[1] = expr
        # 如果表达式是加法操作
        elif expr.is_Add:
            # 获取表达式中所有符号
            symbols = expr.atoms(Symbol)
            # 计算每个单项式的次数和单项式本身，并组成一个列表
            degrees = [(sum(degree_list(monom, *symbols)), monom)
                       for monom in expr.args]
            # 如果需要单独返回单项式列表，则直接返回
            if separate:
                return {monom[1] for monom in degrees}
            else:
                # 将每个单项式添加到对应次数的字典项中
                for monom in degrees:
                    degree, term = monom
                    if poly_dict.get(degree):
                        poly_dict[degree] += term
                    else:
                        poly_dict[degree] = term
        # 如果表达式是幂操作
        elif expr.is_Pow:
            # 获取幂操作的次数，并将其作为键，表达式作为值存储到字典中
            _, degree = expr.args
            poly_dict[degree] = expr
        else:  # 现在表达式只能是乘法操作类型
            degree = 0
            # 遍历乘法操作的每个项，根据项的类型确定其次数并存储到字典中
            for term in expr.args:
                term_type = len(term.args)
                if term_type == 0 and term.is_Symbol:
                    degree += 1
                elif term_type == 2:
                    degree += term.args[1]
            poly_dict[degree] = expr
    else:
        # 如果表达式不是多项式，则将其次数设置为0，并存储到字典中
        poly_dict[0] = expr

    # 如果需要单独返回单项式列表，则将字典的值转换为集合返回
    if separate:
        return set(poly_dict.values())
    # 否则返回包含各次数多项式的字典
    return poly_dict

# 返回排序后的多边形，点的顺序可以是顺时针或逆时针

def point_sort(poly, normal=None, clockwise=True):
    # 注意：为了使积分算法正常工作，输入点集必须按某一顺序排序（顺时针或逆时针）
    # 作为约定，该算法按照顺时针方向进行实现

    # 参数poly：2D或3D多边形
    # 参数normal：平面的法向量（可选）
    # 参数clockwise：如果为True，则返回顺时针排序的点；如果为False，则返回逆时针排序的点

    # 例子：返回顺时针排序的点集
    # point_sort([Point(0, 0), Point(1, 0), Point(1, 1)])
    [Point2D(1, 1), Point2D(1, 0), Point2D(0, 0)]
    """
    # 如果给定的多边形是 Polygon 类型，则使用其顶点列表；否则直接使用多边形本身
    pts = poly.vertices if isinstance(poly, Polygon) else poly
    # 获取顶点数量
    n = len(pts)
    # 如果顶点数小于2，则直接返回顶点列表
    if n < 2:
        return list(pts)

    # 根据顺时针或逆时针方向确定排序顺序
    order = S.One if clockwise else S.NegativeOne
    # 确定顶点的维度（2D 或 3D）
    dim = len(pts[0])
    # 计算中心点坐标
    if dim == 2:
        center = Point(sum((vertex.x for vertex in pts)) / n,
                        sum((vertex.y for vertex in pts)) / n)
    else:
        center = Point(sum((vertex.x for vertex in pts)) / n,
                        sum((vertex.y for vertex in pts)) / n,
                        sum((vertex.z for vertex in pts)) / n)

    # 定义用于比较排序的函数，针对2D和3D分别实现
    def compare(a, b):
        if a.x - center.x >= S.Zero and b.x - center.x < S.Zero:
            return -order
        elif a.x - center.x < 0 and b.x - center.x >= 0:
            return order
        elif a.x - center.x == 0 and b.x - center.x == 0:
            if a.y - center.y >= 0 or b.y - center.y >= 0:
                return -order if a.y > b.y else order
            return -order if b.y > a.y else order

        det = (a.x - center.x) * (b.y - center.y) -\
              (b.x - center.x) * (a.y - center.y)
        if det < 0:
            return -order
        elif det > 0:
            return order

        first = (a.x - center.x) * (a.x - center.x) +\
                (a.y - center.y) * (a.y - center.y)
        second = (b.x - center.x) * (b.x - center.x) +\
                 (b.y - center.y) * (b.y - center.y)
        return -order if first > second else order

    # 定义用于3D情况下的比较排序函数
    def compare3d(a, b):
        # 计算跨越中心点的向量叉乘结果
        det = cross_product(center, a, b)
        # 计算法向量与点积
        dot_product = sum(det[i] * normal[i] for i in range(0, 3))
        if dot_product < 0:
            return -order
        elif dot_product > 0:
            return order

    # 返回根据定义的比较函数进行排序后的顶点列表
    return sorted(pts, key=cmp_to_key(compare if dim==2 else compare3d))
# 计算给定点的欧几里得范数（即点到原点的距离）
def norm(point):
    half = S.Half  # 定义分数 1/2
    if isinstance(point, (list, tuple)):  # 检查点是否为列表或元组
        return sum(coord ** 2 for coord in point) ** half  # 如果是列表或元组，计算其范数
    elif isinstance(point, Point):  # 检查点是否为 sympy 中的 Point 对象
        if isinstance(point, Point2D):  # 检查点是否为二维点对象
            return (point.x ** 2 + point.y ** 2) ** half  # 如果是二维点，计算其范数
        else:
            return (point.x ** 2 + point.y ** 2 + point.z) ** half  # 如果是其他维度的点，计算其范数
    elif isinstance(point, dict):  # 检查点是否为字典
        return sum(i**2 for i in point.values()) ** half  # 如果是字典，计算其值的范数


# 计算两个几何对象的交点
def intersection(geom_1, geom_2, intersection_type):
    if intersection_type[:-2] == "segment":  # 检查交点类型是否为线段
        if intersection_type == "segment2D":  # 如果是二维线段
            x1, y1 = geom_1.points[0]  # 提取第一个线段的起点坐标
            x2, y2 = geom_1.points[1]  # 提取第一个线段的终点坐标
            x3, y3 = geom_2.points[0]  # 提取第二个线段的起点坐标
            x4, y4 = geom_2.points[1]  # 提取第二个线段的终点坐标
        elif intersection_type == "segment3D":  # 如果是三维线段
            x1, y1, z1 = geom_1.points[0]  # 提取第一个线段的起点坐标
            x2, y2, z2 = geom_1.points[1]  # 提取第一个线段的终点坐标
            x3, y3, z3 = geom_2.points[0]  # 提取第二个线段的起点坐标
            x4, y4, z4 = geom_2.points[1]  # 提取第二个线段的终点坐标

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)  # 计算分母
        if denom:
            t1 = x1 * y2 - y1 * x2  # 辅助计算参数 t1
            t2 = x3 * y4 - x4 * y3  # 辅助计算参数 t2
            return (S(t1 * (x3 - x4) - t2 * (x1 - x2)) / denom,  # 返回交点的 x 坐标
                    S(t1 * (y3 - y4) - t2 * (y1 - y2)) / denom)  # 返回交点的 y 坐标
    if intersection_type[:-2] == "plane":  # 检查交点类型是否为平面
        if intersection_type == "plane2D":  # 如果是二维平面（超平面）的交点计算
            a1x, a1y = geom_1[0]  # 提取第一个平面的法向量
            a2x, a2y = geom_2[0]  # 提取第二个平面的法向量
            b1, b2 = geom_1[1], geom_2[1]  # 提取第一个平面和第二个平面的常量

            denom = a1x * a2y - a2x * a1y  # 计算分母
            if denom:
                return (S(b1 * a2y - b2 * a1y) / denom,  # 返回交点的 x 坐标
                        S(b2 * a1x - b1 * a2x) / denom)  # 返回交点的 y 坐标


def is_vertex(ent):
    """检查输入实体是否为顶点，如果是则返回True。"""
    # 判断给定的实体是否是顶点
    # 如果实体是一个元组，检查其长度是否为2或3，表示二维或三维点
    if isinstance(ent, tuple):
        if len(ent) in [2, 3]:
            return True
    # 如果实体是 sympy 中的 Point 对象，直接返回 True
    elif isinstance(ent, Point):
        return True
    # 如果实体不是上述类型，则返回 False
    return False
# 绘制二维多面体的函数，使用绘图模块中基于 matplotlib 的功能
def plot_polytope(poly):
    # 导入绘图所需的类和函数
    from sympy.plotting.plot import Plot, List2DSeries

    # 提取多面体顶点的 x 坐标列表
    xl = [vertex.x for vertex in poly.vertices]
    # 提取多面体顶点的 y 坐标列表
    yl = [vertex.y for vertex in poly.vertices]

    # 添加第一个顶点以闭合多边形
    xl.append(poly.vertices[0].x)
    yl.append(poly.vertices[0].y)

    # 创建 2D 列表数据系列对象
    l2ds = List2DSeries(xl, yl)
    # 创建绘图对象，并指定使用标签化的坐标轴
    p = Plot(l2ds, axes='label_axes=True')
    # 显示绘图
    p.show()


# 绘制多项式函数的函数，使用绘图模块中基于 matplotlib 的功能
def plot_polynomial(expr):
    # 导入绘图所需的类和函数
    from sympy.plotting.plot import plot3d, plot

    # 提取表达式中的自由符号（变量）
    gens = expr.free_symbols
    # 如果表达式包含两个变量，则绘制三维图形
    if len(gens) == 2:
        plot3d(expr)
    # 否则绘制二维图形
    else:
        plot(expr)
```