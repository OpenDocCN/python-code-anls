# `D:\src\scipysrc\sympy\sympy\vector\parametricregion.py`

```
from functools import singledispatch
from sympy.core.numbers import pi
from sympy.functions.elementary.trigonometric import tan
from sympy.simplify import trigsimp
from sympy.core import Basic, Tuple
from sympy.core.symbol import _symbol
from sympy.solvers import solve
from sympy.geometry import Point, Segment, Curve, Ellipse, Polygon
from sympy.vector import ImplicitRegion

class ParametricRegion(Basic):
    """
    Represents a parametric region in space.

    Examples
    ========

    >>> from sympy import cos, sin, pi
    >>> from sympy.abc import r, theta, t, a, b, x, y
    >>> from sympy.vector import ParametricRegion

    >>> ParametricRegion((t, t**2), (t, -1, 2))
    ParametricRegion((t, t**2), (t, -1, 2))
    >>> ParametricRegion((x, y), (x, 3, 4), (y, 5, 6))
    ParametricRegion((x, y), (x, 3, 4), (y, 5, 6))
    >>> ParametricRegion((r*cos(theta), r*sin(theta)), (r, -2, 2), (theta, 0, pi))
    ParametricRegion((r*cos(theta), r*sin(theta)), (r, -2, 2), (theta, 0, pi))
    >>> ParametricRegion((a*cos(t), b*sin(t)), t)
    ParametricRegion((a*cos(t), b*sin(t)), t)

    >>> circle = ParametricRegion((r*cos(theta), r*sin(theta)), r, (theta, 0, pi))
    >>> circle.parameters
    (r, theta)
    >>> circle.definition
    (r*cos(theta), r*sin(theta))
    >>> circle.limits
    {theta: (0, pi)}

    Dimension of a parametric region determines whether a region is a curve, surface
    or volume region. It does not represent its dimensions in space.

    >>> circle.dimensions
    1

    Parameters
    ==========

    definition : tuple to define base scalars in terms of parameters.

    bounds : Parameter or a tuple of length 3 to define parameter and corresponding lower and upper bound.

    """
    def __new__(cls, definition, *bounds):
        parameters = ()  # 初始化空元组，用于存放参数
        limits = {}      # 初始化空字典，用于存放参数的上下限

        if not isinstance(bounds, Tuple):
            bounds = Tuple(*bounds)  # 将 bounds 转换为 Tuple 类型

        for bound in bounds:
            if isinstance(bound, (tuple, Tuple)):
                if len(bound) != 3:
                    raise ValueError("Tuple should be in the form (parameter, lowerbound, upperbound)")
                parameters += (bound[0],)  # 将参数加入 parameters 元组
                limits[bound[0]] = (bound[1], bound[2])  # 记录参数及其上下限
            else:
                parameters += (bound,)  # 处理单个参数的情况

        if not isinstance(definition, (tuple, Tuple)):
            definition = (definition,)  # 将 definition 转换为 Tuple 类型

        obj = super().__new__(cls, Tuple(*definition), *bounds)  # 创建 ParametricRegion 对象
        obj._parameters = parameters  # 记录参数
        obj._limits = limits  # 记录参数的上下限

        return obj

    @property
    def definition(self):
        return self.args[0]  # 返回定义部分的参数元组

    @property
    def limits(self):
        return self._limits  # 返回参数的上下限字典

    @property
    def parameters(self):
        return self._parameters  # 返回参数元组

    @property
    def dimensions(self):
        return len(self.limits)  # 返回参数的数量，即维度


@singledispatch
def parametric_region_list(reg):
    """
    Returns a list of ParametricRegion objects representing the geometric region.

    Examples
    ========

    >>> from sympy.abc import t
    """
    >>> from sympy.vector import parametric_region_list
    >>> from sympy.geometry import Point, Curve, Ellipse, Segment, Polygon

    >>> p = Point(2, 5)
    >>> parametric_region_list(p)
    [ParametricRegion((2, 5))]

    >>> c = Curve((t**3, 4*t), (t, -3, 4))
    >>> parametric_region_list(c)
    [ParametricRegion((t**3, 4*t), (t, -3, 4))]

    >>> e = Ellipse(Point(1, 3), 2, 3)
    >>> parametric_region_list(e)
    [ParametricRegion((2*cos(t) + 1, 3*sin(t) + 3), (t, 0, 2*pi))]

    >>> s = Segment(Point(1, 3), Point(2, 6))
    >>> parametric_region_list(s)
    [ParametricRegion((t + 1, 3*t + 3), (t, 0, 1))]

    >>> p1, p2, p3, p4 = [(0, 1), (2, -3), (5, 3), (-2, 3)]
    >>> poly = Polygon(p1, p2, p3, p4)
    >>> parametric_region_list(poly)
    [ParametricRegion((2*t, 1 - 4*t), (t, 0, 1)), ParametricRegion((3*t + 2, 6*t - 3), (t, 0, 1)),\
     ParametricRegion((5 - 7*t, 3), (t, 0, 1)), ParametricRegion((2*t - 2, 3 - 2*t),  (t, 0, 1))]

    """
    raise ValueError("SymPy cannot determine parametric representation of the region.")


注释：


    >>> from sympy.vector import parametric_region_list  # 导入 parametric_region_list 函数
    >>> from sympy.geometry import Point, Curve, Ellipse, Segment, Polygon  # 导入几何模块中的特定类

    >>> p = Point(2, 5)  # 创建一个二维点对象 p

    >>> parametric_region_list(p)  # 调用 parametric_region_list 函数，传入点对象 p
    [ParametricRegion((2, 5))]  # 返回一个包含参数化区域的列表，这里只有一个点 (2, 5)

    >>> c = Curve((t**3, 4*t), (t, -3, 4))  # 创建一个曲线对象 c，其参数为 (t**3, 4*t)，t 的范围为 -3 到 4

    >>> parametric_region_list(c)  # 调用 parametric_region_list 函数，传入曲线对象 c
    [ParametricRegion((t**3, 4*t), (t, -3, 4))]  # 返回一个包含参数化区域的列表，表示曲线 c 的参数化形式

    >>> e = Ellipse(Point(1, 3), 2, 3)  # 创建一个椭圆对象 e，中心为 (1, 3)，长轴为 2，短轴为 3

    >>> parametric_region_list(e)  # 调用 parametric_region_list 函数，传入椭圆对象 e
    [ParametricRegion((2*cos(t) + 1, 3*sin(t) + 3), (t, 0, 2*pi))]  # 返回一个包含参数化区域的列表，表示椭圆 e 的参数化形式

    >>> s = Segment(Point(1, 3), Point(2, 6))  # 创建一个线段对象 s，起点为 (1, 3)，终点为 (2, 6)

    >>> parametric_region_list(s)  # 调用 parametric_region_list 函数，传入线段对象 s
    [ParametricRegion((t + 1, 3*t + 3), (t, 0, 1))]  # 返回一个包含参数化区域的列表，表示线段 s 的参数化形式

    >>> p1, p2, p3, p4 = [(0, 1), (2, -3), (5, 3), (-2, 3)]  # 定义四个二维点的坐标
    >>> poly = Polygon(p1, p2, p3, p4)  # 创建一个多边形对象 poly，由定义的四个点组成

    >>> parametric_region_list(poly)  # 调用 parametric_region_list 函数，传入多边形对象 poly
    [ParametricRegion((2*t, 1 - 4*t), (t, 0, 1)), ParametricRegion((3*t + 2, 6*t - 3), (t, 0, 1)),\
     ParametricRegion((5 - 7*t, 3), (t, 0, 1)), ParametricRegion((2*t - 2, 3 - 2*t),  (t, 0, 1))]
    # 返回一个包含参数化区域的列表，表示多边形 poly 的每条边的参数化形式

    """
    raise ValueError("SymPy cannot determine parametric representation of the region.")  # 抛出 ValueError 异常，说明 SymPy 无法确定区域的参数化表示
@parametric_region_list.register(Point)
def _(obj):
    # 对于点对象，创建一个包含其参数的 ParametricRegion 列表
    return [ParametricRegion(obj.args)]


@parametric_region_list.register(Curve)  # type: ignore
def _(obj):
    # 对于曲线对象，计算任意点的定义和限制，并创建 ParametricRegion 列表
    definition = obj.arbitrary_point(obj.parameter).args
    bounds = obj.limits
    return [ParametricRegion(definition, bounds)]


@parametric_region_list.register(Ellipse) # type: ignore
def _(obj, parameter='t'):
    # 对于椭圆对象，计算任意点的定义和默认边界，创建 ParametricRegion 列表
    definition = obj.arbitrary_point(parameter).args
    t = _symbol(parameter, real=True)
    bounds = (t, 0, 2*pi)
    return [ParametricRegion(definition, bounds)]


@parametric_region_list.register(Segment) # type: ignore
def _(obj, parameter='t'):
    # 对于线段对象，计算任意点的定义和边界，创建 ParametricRegion 列表
    t = _symbol(parameter, real=True)
    definition = obj.arbitrary_point(t).args

    for i in range(0, 3):
        lower_bound = solve(definition[i] - obj.points[0].args[i], t)
        upper_bound = solve(definition[i] - obj.points[1].args[i], t)

        if len(lower_bound) == 1 and len(upper_bound) == 1:
            bounds = t, lower_bound[0], upper_bound[0]
            break

    definition_tuple = obj.arbitrary_point(parameter).args
    return [ParametricRegion(definition_tuple, bounds)]


@parametric_region_list.register(Polygon) # type: ignore
def _(obj, parameter='t'):
    # 对于多边形对象，递归调用 parametric_region_list 处理每个边的 ParametricRegion 列表
    l = [parametric_region_list(side, parameter)[0] for side in obj.sides]
    return l


@parametric_region_list.register(ImplicitRegion) # type: ignore
def _(obj, parameters=('t', 's')):
    # 对于隐式区域对象，计算有理参数化和替换参数的正切，创建 ParametricRegion 列表
    definition = obj.rational_parametrization(parameters)
    bounds = []

    for i in range(len(obj.variables) - 1):
        parameter = _symbol(parameters[i], real=True)
        definition = [trigsimp(elem.subs(parameter, tan(parameter/2))) for elem in definition]
        bounds.append((parameter, 0, 2*pi),)

    definition = Tuple(*definition)
    return [ParametricRegion(definition, *bounds)]
```