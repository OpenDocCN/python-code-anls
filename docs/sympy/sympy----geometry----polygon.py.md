# `D:\src\scipysrc\sympy\sympy\geometry\polygon.py`

```
# 导入 SymPy 库中所需的模块和函数
from sympy.core import Expr, S, oo, pi, sympify
from sympy.core.evalf import N
from sympy.core.sorting import default_sort_key, ordered
from sympy.core.symbol import _symbol, Dummy, Symbol
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import cos, sin, tan
from .ellipse import Circle  # 导入 Circle 类，位于当前目录的 ellipse 模块中
from .entity import GeometryEntity, GeometrySet  # 导入几何实体类和集合类
from .exceptions import GeometryError  # 导入几何异常类
from .line import Line, Segment, Ray  # 导入线类、线段类和射线类
from .point import Point  # 导入点类
from sympy.logic import And  # 导入逻辑运算 And 函数
from sympy.matrices import Matrix  # 导入矩阵类
from sympy.simplify.simplify import simplify  # 导入简化函数 simplify
from sympy.solvers.solvers import solve  # 导入求解函数 solve
from sympy.utilities.iterables import (  # 导入迭代工具函数
    has_dups,
    has_variety,
    uniq,
    rotate_left,
    least_rotation,
)
from sympy.utilities.misc import as_int, func_name  # 导入杂项工具函数

from mpmath.libmp.libmpf import prec_to_dps  # 从 mpmath 库中导入精度转换函数

import warnings  # 导入警告模块

# 创建三个虚拟变量 x, y, T，用于多边形类的定义
x, y, T = [Dummy('polygon_dummy', real=True) for i in range(3)]


class Polygon(GeometrySet):
    """A two-dimensional polygon.

    A simple polygon in space. Can be constructed from a sequence of points
    or from a center, radius, number of sides and rotation angle.

    Parameters
    ==========

    vertices
        A sequence of points.

    n : int, optional
        If $> 0$, an n-sided RegularPolygon is created.
        Default value is $0$.

    Attributes
    ==========

    area
    angles
    perimeter
    vertices
    centroid
    sides

    Raises
    ======

    GeometryError
        If all parameters are not Points.

    See Also
    ========

    sympy.geometry.point.Point, sympy.geometry.line.Segment, Triangle

    Notes
    =====

    Polygons are treated as closed paths rather than 2D areas so
    some calculations can be be negative or positive (e.g., area)
    based on the orientation of the points.

    Any consecutive identical points are reduced to a single point
    and any points collinear and between two points will be removed
    unless they are needed to define an explicit intersection (see examples).

    A Triangle, Segment or Point will be returned when there are 3 or
    fewer points provided.

    Examples
    ========

    >>> from sympy import Polygon, pi
    >>> p1, p2, p3, p4, p5 = [(0, 0), (1, 0), (5, 1), (0, 1), (3, 0)]
    >>> Polygon(p1, p2, p3, p4)
    Polygon(Point2D(0, 0), Point2D(1, 0), Point2D(5, 1), Point2D(0, 1))
    >>> Polygon(p1, p2)
    Segment2D(Point2D(0, 0), Point2D(1, 0))
    >>> Polygon(p1, p2, p5)
    Segment2D(Point2D(0, 0), Point2D(3, 0))

    The area of a polygon is calculated as positive when vertices are
    traversed in a ccw direction. When the sides of a polygon cross the
    area will have positive and negative contributions. The following
    defines a Z shape where the bottom right connects back to the top
    left.

    >>> Polygon((0, 2), (2, 2), (0, 0), (2, 0)).area
    0

    When the keyword `n` is used to define the number of sides of the
    Polygon then a RegularPolygon is created and the other arguments are
    interpreted as center, radius and rotation. The unrotated RegularPolygon
    will always have a vertex at Point(r, 0) where `r` is the radius of the
    circle that circumscribes the RegularPolygon. Its method `spin` can be
    used to increment that angle.

    >>> p = Polygon((0,0), 1, n=3)
    >>> p
    RegularPolygon(Point2D(0, 0), 1, 3, 0)
    >>> p.vertices[0]
    Point2D(1, 0)
    >>> p.args[0]
    Point2D(0, 0)
    >>> p.spin(pi/2)
    >>> p.vertices[0]
    Point2D(0, 1)

    """

    __slots__ = ()

    # 定义一个新的多边形对象构造函数，可以根据输入参数创建不同类型的几何实体
    def __new__(cls, *args, n = 0, **kwargs):
        if n:
            args = list(args)
            # 如果参数 n 不为零，创建一个具有 n 条边的正多边形
            if len(args) == 2:  # 中心点, 半径
                args.append(n)
            elif len(args) == 3:  # 中心点, 半径, 旋转角度
                args.insert(2, n)
            return RegularPolygon(*args, **kwargs)

        # 创建一个顶点列表，每个顶点都是一个 Point2D 对象
        vertices = [Point(a, dim=2, **kwargs) for a in args]

        # 去除连续重复的顶点
        nodup = []
        for p in vertices:
            if nodup and p == nodup[-1]:
                continue
            nodup.append(p)
        
        # 如果最后一个顶点与第一个顶点相同，则移除最后一个顶点
        if len(nodup) > 1 and nodup[-1] == nodup[0]:
            nodup.pop()

        # 去除共线的顶点
        i = -3
        while i < len(nodup) - 3 and len(nodup) > 2:
            a, b, c = nodup[i], nodup[i + 1], nodup[i + 2]
            if Point.is_collinear(a, b, c):
                nodup.pop(i + 1)
                if a == c:
                    nodup.pop(i)
            else:
                i += 1

        vertices = list(nodup)

        # 根据顶点数量返回对应的几何实体
        if len(vertices) > 3:
            return GeometryEntity.__new__(cls, *vertices, **kwargs)
        elif len(vertices) == 3:
            return Triangle(*vertices, **kwargs)
        elif len(vertices) == 2:
            return Segment(*vertices, **kwargs)
        else:
            return Point(*vertices, **kwargs)

    @property
    def area(self):
        """
        The area of the polygon.

        Notes
        =====

        The area calculation can be positive or negative based on the
        orientation of the points. If any side of the polygon crosses
        any other side, there will be areas having opposite signs.

        See Also
        ========

        sympy.geometry.ellipse.Ellipse.area

        Examples
        ========

        >>> from sympy import Point, Polygon
        >>> p1, p2, p3, p4 = map(Point, [(0, 0), (1, 0), (5, 1), (0, 1)])
        >>> poly = Polygon(p1, p2, p3, p4)
        >>> poly.area
        3

        In the Z shaped polygon (with the lower right connecting back
        to the upper left) the areas cancel out:

        >>> Z = Polygon((0, 1), (1, 1), (0, 0), (1, 0))
        >>> Z.area
        0

        In the M shaped polygon, areas do not cancel because no side
        crosses any other (though there is a point of contact).

        >>> M = Polygon((0, 0), (0, 1), (2, 0), (3, 1), (3, 0))
        >>> M.area
        -3/2

        """
        area = 0  # Initialize the area accumulator
        args = self.args  # Get the vertices of the polygon
        for i in range(len(args)):
            x1, y1 = args[i - 1].args  # Previous vertex coordinates
            x2, y2 = args[i].args      # Current vertex coordinates
            # Add the signed area of the triangle formed by (0, 0), (x1, y1), (x2, y2)
            area += x1*y2 - x2*y1
        # Return the area divided by 2, and simplified using sympy's simplify function
        return simplify(area) / 2

    @staticmethod
    def _is_clockwise(a, b, c):
        """Return True/False for cw/ccw orientation.

        Examples
        ========

        >>> from sympy import Point, Polygon
        >>> a, b, c = [Point(i) for i in [(0, 0), (1, 1), (1, 0)]]
        >>> Polygon._is_clockwise(a, b, c)
        True
        >>> Polygon._is_clockwise(a, c, b)
        False
        """
        ba = b - a  # Vector from a to b
        ca = c - a  # Vector from a to c
        t_area = simplify(ba.x*ca.y - ca.x*ba.y)  # Twice the signed area of triangle abc
        res = t_area.is_nonpositive  # Check if the area is non-positive
        if res is None:
            raise ValueError("Can't determine orientation")  # Raise error if orientation cannot be determined
        return res  # Return True if clockwise or collinear, False if counterclockwise

    @property
    def angles(self):
        """The internal angle at each vertex.

        Returns
        =======

        angles : dict
            A dictionary where each key is a vertex and each value is the
            internal angle at that vertex. The vertices are represented as
            Points.

        See Also
        ========

        sympy.geometry.point.Point, sympy.geometry.line.LinearEntity.angle_between

        Examples
        ========

        >>> from sympy import Point, Polygon
        >>> p1, p2, p3, p4 = map(Point, [(0, 0), (1, 0), (5, 1), (0, 1)])
        >>> poly = Polygon(p1, p2, p3, p4)
        >>> poly.angles[p1]
        pi/2
        >>> poly.angles[p2]
        acos(-4*sqrt(17)/17)

        """

        # 获取多边形的顶点列表
        args = self.vertices
        # 多边形顶点数量
        n = len(args)
        # 初始化返回的角度字典
        ret = {}
        # 遍历每个顶点
        for i in range(n):
            # 获取当前顶点及其前两个顶点
            a, b, c = args[i - 2], args[i - 1], args[i]
            # 计算夹角，使用射线类计算
            reflex_ang = Ray(b, a).angle_between(Ray(b, c))
            # 检查顶点组成的角度是否顺时针
            if self._is_clockwise(a, b, c):
                ret[b] = 2*S.Pi - reflex_ang
            else:
                ret[b] = reflex_ang

        # 检查多边形内角之和是否正确
        wrong = ((sum(ret.values())/S.Pi-1)/(n - 2) - 1).is_positive
        # 如果角度和计算错误，则修正
        if wrong:
            two_pi = 2*S.Pi
            for b in ret:
                ret[b] = two_pi - ret[b]
        elif wrong is None:
            # 如果无法确定多边形方向，则引发错误
            raise ValueError("could not determine Polygon orientation.")
        # 返回计算得到的角度字典
        return ret

    @property
    def ambient_dimension(self):
        # 返回多边形的维度，即第一个顶点的环境维度
        return self.vertices[0].ambient_dimension

    @property
    def perimeter(self):
        """The perimeter of the polygon.

        Returns
        =======

        perimeter : number or Basic instance

        See Also
        ========

        sympy.geometry.line.Segment.length

        Examples
        ========

        >>> from sympy import Point, Polygon
        >>> p1, p2, p3, p4 = map(Point, [(0, 0), (1, 0), (5, 1), (0, 1)])
        >>> poly = Polygon(p1, p2, p3, p4)
        >>> poly.perimeter
        sqrt(17) + 7
        """
        # 计算多边形的周长
        p = 0
        # 获取多边形的顶点列表
        args = self.vertices
        # 遍历顶点列表，计算相邻顶点之间的距离总和
        for i in range(len(args)):
            p += args[i - 1].distance(args[i])
        # 简化并返回周长
        return simplify(p)

    @property
    def vertices(self):
        """The vertices of the polygon.

        Returns
        =======

        vertices : list of Points
            返回多边形的顶点列表

        Notes
        =====

        When iterating over the vertices, it is more efficient to index self
        rather than to request the vertices and index them. Only use the
        vertices when you want to process all of them at once. This is even
        more important with RegularPolygons that calculate each vertex.
            迭代顶点时，通过索引 self 更有效率，而不是请求顶点并对其进行索引。
            只有当需要一次处理所有顶点时才使用顶点列表。这对于计算每个顶点的 RegularPolygon 尤为重要。

        See Also
        ========

        sympy.geometry.point.Point
            参见 sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Point, Polygon
        >>> p1, p2, p3, p4 = map(Point, [(0, 0), (1, 0), (5, 1), (0, 1)])
        >>> poly = Polygon(p1, p2, p3, p4)
        >>> poly.vertices
        [Point2D(0, 0), Point2D(1, 0), Point2D(5, 1), Point2D(0, 1)]
        >>> poly.vertices[0]
        Point2D(0, 0)
            示例用法，创建一个多边形并展示其顶点的使用方式

        """
        return list(self.args)

    @property
    def centroid(self):
        """The centroid of the polygon.

        Returns
        =======

        centroid : Point
            返回多边形的质心点

        See Also
        ========

        sympy.geometry.point.Point, sympy.geometry.util.centroid
            参见 sympy.geometry.point.Point 和 sympy.geometry.util.centroid

        Examples
        ========

        >>> from sympy import Point, Polygon
        >>> p1, p2, p3, p4 = map(Point, [(0, 0), (1, 0), (5, 1), (0, 1)])
        >>> poly = Polygon(p1, p2, p3, p4)
        >>> poly.centroid
        Point2D(31/18, 11/18)
            示例用法，计算并展示多边形的质心点坐标

        """
        A = 1/(6*self.area)
        cx, cy = 0, 0
        args = self.args
        for i in range(len(args)):
            x1, y1 = args[i - 1].args
            x2, y2 = args[i].args
            v = x1*y2 - x2*y1
            cx += v*(x1 + x2)
            cy += v*(y1 + y2)
        return Point(simplify(A*cx), simplify(A*cy))
            计算并返回多边形的质心点坐标
    # 计算二维多边形的二阶矩和面积矩

    I_xx, I_yy, I_xy = 0, 0, 0  # 初始化二阶矩和产品矩为零
    args = self.vertices  # 获取多边形的顶点坐标
    for i in range(len(args)):
        x1, y1 = args[i-1].args  # 获取当前顶点和前一顶点的坐标
        x2, y2 = args[i].args  # 获取当前顶点的坐标
        v = x1*y2 - x2*y1  # 计算顶点坐标的叉乘值
        # 计算二阶矩和产品矩的累加值
        I_xx += (y1**2 + y1*y2 + y2**2)*v
        I_yy += (x1**2 + x1*x2 + x2**2)*v
        I_xy += (x1*y2 + 2*x1*y1 + 2*x2*y2 + x2*y1)*v
    A = self.area  # 获取多边形的面积
    c_x = self.centroid[0]  # 获取多边形的质心 x 坐标
    c_y = self.centroid[1]  # 获取多边形的质心 y 坐标

    # 使用平行轴定理计算关于给定点的二阶矩和产品矩
    I_xx_c = (I_xx/12) - (A*(c_y**2))
    I_yy_c = (I_yy/12) - (A*(c_x**2))
    I_xy_c = (I_xy/24) - (A*(c_x*c_y))

    if point is None:
        # 如果未指定点，则返回关于多边形质心的二阶矩和产品矩
        return I_xx_c, I_yy_c, I_xy_c

    # 计算关于给定点的二阶矩和产品矩
    I_xx = (I_xx_c + A*((point[1]-c_y)**2))
    I_yy = (I_yy_c + A*((point[0]-c_x)**2))
    I_xy = (I_xy_c + A*((point[0]-c_x)*(point[1]-c_y)))

    return I_xx, I_yy, I_xy
    def first_moment_of_area(self, point=None):
        """
        Returns the first moment of area of a two-dimensional polygon with
        respect to a certain point of interest.

        First moment of area is a measure of the distribution of the area
        of a polygon in relation to an axis. The first moment of area of
        the entire polygon about its own centroid is always zero. Therefore,
        here it is calculated for an area, above or below a certain point
        of interest, that makes up a smaller portion of the polygon. This
        area is bounded by the point of interest and the extreme end
        (top or bottom) of the polygon. The first moment for this area is
        is then determined about the centroidal axis of the initial polygon.

        References
        ==========

        .. [1] https://skyciv.com/docs/tutorials/section-tutorials/calculating-the-statical-or-first-moment-of-area-of-beam-sections/?cc=BMD
        .. [2] https://mechanicalc.com/reference/cross-sections

        Parameters
        ==========

        point: Point, two-tuple of sympifyable objects, or None (default=None)
            point is the point above or below which the area of interest lies
            If ``point=None`` then the centroid acts as the point of interest.

        Returns
        =======

        Q_x, Q_y: number or SymPy expressions
            Q_x is the first moment of area about the x-axis
            Q_y is the first moment of area about the y-axis
            A negative sign indicates that the section modulus is
            determined for a section below (or left of) the centroidal axis

        Examples
        ========

        >>> from sympy import Point, Polygon
        >>> a, b = 50, 10
        >>> p1, p2, p3, p4 = [(0, b), (0, 0), (a, 0), (a, b)]
        >>> p = Polygon(p1, p2, p3, p4)
        >>> p.first_moment_of_area()
        (625, 3125)
        >>> p.first_moment_of_area(point=Point(30, 7))
        (525, 3000)
        """
        # If a specific point of interest is provided, use its coordinates
        # Otherwise, use the centroid of the polygon as the point of interest
        if point:
            xc, yc = self.centroid
        else:
            point = self.centroid
            xc, yc = point

        # Create horizontal and vertical lines passing through the point of interest
        h_line = Line(point, slope=0)
        v_line = Line(point, slope=S.Infinity)

        # Cut the polygon into sections using the horizontal and vertical lines
        h_poly = self.cut_section(h_line)
        v_poly = self.cut_section(v_line)

        # Determine which sub-section of the polygon has smaller area
        poly_1 = h_poly[0] if h_poly[0].area <= h_poly[1].area else h_poly[1]
        poly_2 = v_poly[0] if v_poly[0].area <= v_poly[1].area else v_poly[1]

        # Calculate the first moments of area about the x-axis and y-axis
        Q_x = (poly_1.centroid.y - yc) * poly_1.area
        Q_y = (poly_2.centroid.x - xc) * poly_2.area

        # Return the calculated first moments of area
        return Q_x, Q_y
    # 计算二维多边形的极性面积第二矩（极点模量）

    # 获取多边形的平面截面惯性矩
    second_moment = self.second_moment_of_area()
    # 返回多边形对极点（垂直于中心轴的平面）的第二矩，通过平行轴定理与平面截面惯性矩相关联
    return second_moment[0] + second_moment[1]
    def section_modulus(self, point=None):
        """
        Returns a tuple with the section modulus of a two-dimensional
        polygon.

        Section modulus is a geometric property of a polygon defined as the
        ratio of second moment of area to the distance of the extreme end of
        the polygon from the centroidal axis.

        Parameters
        ==========

        point : Point, two-tuple of sympifyable objects, or None (default=None)
            point is the point at which section modulus is to be found.
            If "point=None" it will be calculated for the point farthest from the
            centroidal axis of the polygon.

        Returns
        =======

        S_x, S_y: numbers or SymPy expressions
                  S_x is the section modulus with respect to the x-axis
                  S_y is the section modulus with respect to the y-axis
                  A negative sign indicates that the section modulus is
                  determined for a point below the centroidal axis

        Examples
        ========

        >>> from sympy import symbols, Polygon, Point
        >>> a, b = symbols('a, b', positive=True)
        >>> rectangle = Polygon((0, 0), (a, 0), (a, b), (0, b))
        >>> rectangle.section_modulus()
        (a*b**2/6, a**2*b/6)
        >>> rectangle.section_modulus(Point(a/4, b/4))
        (-a*b**2/3, -a**2*b/3)

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Section_modulus

        """
        x_c, y_c = self.centroid
        if point is None:
            # taking x and y as maximum distances from centroid
            x_min, y_min, x_max, y_max = self.bounds
            y = max(y_c - y_min, y_max - y_c)
            x = max(x_c - x_min, x_max - x_c)
        else:
            # taking x and y as distances of the given point from the centroid
            y = point.y - y_c
            x = point.x - x_c

        second_moment = self.second_moment_of_area()
        S_x = second_moment[0] / y
        S_y = second_moment[1] / x

        return S_x, S_y


    @property
    def sides(self):
        """
        The directed line segments that form the sides of the polygon.

        Returns
        =======

        sides : list of sides
            Each side is a directed Segment.

        See Also
        ========

        sympy.geometry.point.Point, sympy.geometry.line.Segment

        Examples
        ========

        >>> from sympy import Point, Polygon
        >>> p1, p2, p3, p4 = map(Point, [(0, 0), (1, 0), (5, 1), (0, 1)])
        >>> poly = Polygon(p1, p2, p3, p4)
        >>> poly.sides
        [Segment2D(Point2D(0, 0), Point2D(1, 0)),
        Segment2D(Point2D(1, 0), Point2D(5, 1)),
        Segment2D(Point2D(5, 1), Point2D(0, 1)), Segment2D(Point2D(0, 1), Point2D(0, 0))]

        """
        res = []
        args = self.vertices
        for i in range(-len(args), 0):
            # Creating directed segments between consecutive vertices
            res.append(Segment(args[i], args[i + 1]))
        return res
    def bounds(self):
        """Return a tuple (xmin, ymin, xmax, ymax) representing the bounding
        rectangle for the geometric figure.

        """
        # 获取顶点列表
        verts = self.vertices
        # 提取所有顶点的 x 坐标
        xs = [p.x for p in verts]
        # 提取所有顶点的 y 坐标
        ys = [p.y for p in verts]
        # 返回边界框的坐标范围：(最小 x, 最小 y, 最大 x, 最大 y)
        return (min(xs), min(ys), max(xs), max(ys))

    def is_convex(self):
        """Is the polygon convex?

        A polygon is convex if all its interior angles are less than 180
        degrees and there are no intersections between sides.

        Returns
        =======

        is_convex : boolean
            True if this polygon is convex, False otherwise.

        See Also
        ========

        sympy.geometry.util.convex_hull

        Examples
        ========

        >>> from sympy import Point, Polygon
        >>> p1, p2, p3, p4 = map(Point, [(0, 0), (1, 0), (5, 1), (0, 1)])
        >>> poly = Polygon(p1, p2, p3, p4)
        >>> poly.is_convex()
        True

        """
        # 确定顶点的方向
        args = self.vertices
        # 检查第一个顶点相对于最后一个和第一个顶点的方向
        cw = self._is_clockwise(args[-2], args[-1], args[0])
        # 遍历剩余的顶点，检查相邻顶点之间的方向
        for i in range(1, len(args)):
            if cw ^ self._is_clockwise(args[i - 2], args[i - 1], args[i]):
                return False
        # 检查多边形的边是否相交
        sides = self.sides
        for i, si in enumerate(sides):
            pts = si.args
            # 排除与 si 相连的边
            for j in range(1 if i == len(sides) - 1 else 0, i - 1):
                sj = sides[j]
                if sj.p1 not in pts and sj.p2 not in pts:
                    hit = si.intersection(sj)
                    if hit:
                        return False
        return True
    def encloses_point(self, p):
        """
        Return True if p is enclosed by (is inside of) self.

        Notes
        =====

        Being on the border of self is considered False.

        Parameters
        ==========

        p : Point
            The point to check whether it is inside the polygon.

        Returns
        =======

        encloses_point : True, False or None
            True if the point is inside the polygon, False if it is outside but on the border,
            None if it cannot be determined (e.g., vertices or sides have symbolic expressions).

        See Also
        ========

        sympy.geometry.point.Point, sympy.geometry.ellipse.Ellipse.encloses_point
            Related functions and classes for geometry calculations.

        Examples
        ========

        >>> from sympy import Polygon, Point
        >>> p = Polygon((0, 0), (4, 0), (4, 4))
        >>> p.encloses_point(Point(2, 1))
        True
        >>> p.encloses_point(Point(2, 2))
        False
        >>> p.encloses_point(Point(5, 5))
        False

        References
        ==========

        .. [1] https://paulbourke.net/geometry/polygonmesh/#insidepoly
            Reference link for polygon containment algorithms.

        """
        p = Point(p, dim=2)  # Convert point p to a 2-dimensional Point object

        # Check if point p is a vertex of the polygon or lies on any of its sides
        if p in self.vertices or any(p in s for s in self.sides):
            return False

        lit = []
        # Calculate vectors from point p to each vertex of the polygon
        for v in self.vertices:
            lit.append(v - p)  # Compute vector from vertex v to point p
            if lit[-1].free_symbols:
                return None  # Return None if any vector has symbolic components

        poly = Polygon(*lit)  # Create a new polygon from the vectors lit

        args = poly.args
        indices = list(range(-len(args), 1))

        if poly.is_convex():  # Check if the polygon is convex
            orientation = None
            # Check orientation of each side of the polygon
            for i in indices:
                a = args[i]
                b = args[i + 1]
                test = ((-a.y)*(b.x - a.x) - (-a.x)*(b.y - a.y)).is_negative
                if orientation is None:
                    orientation = test
                elif test is not orientation:
                    return False  # Return False if orientations are inconsistent
            return True  # Return True if all sides have consistent orientation (polygon encloses point)

        # Check using the ray-casting method for concave polygons
        hit_odd = False
        p1x, p1y = args[0].args
        for i in indices[1:]:
            p2x, p2y = args[i].args
            if 0 > min(p1y, p2y):
                if 0 <= max(p1y, p2y):
                    if 0 <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (-p1y)*(p2x - p1x)/(p2y - p1y) + p1x
                            if p1x == p2x or 0 <= xinters:
                                hit_odd = not hit_odd  # Toggle hit_odd if ray crosses the polygon side
            p1x, p1y = p2x, p2y
        return hit_odd  # Return whether the point is inside the polygon (True if odd number of hits)
    def arbitrary_point(self, parameter='t'):
        """返回多边形上的一个参数化点。

        参数 `parameter` 的取值范围是 0 到 1，表示多边形周长上的某一点，其数值
        是总周长的分数部分。例如，当 t=1/2 时，返回多边形周长上第一个顶点的一半处
        的点。

        Parameters
        ==========

        parameter : str, optional
            默认值为 't'。

        Returns
        =======

        arbitrary_point : Point
            返回一个点对象。

        Raises
        ======

        ValueError
            当 `parameter` 已经在多边形的定义中出现时。

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Polygon, Symbol
        >>> t = Symbol('t', real=True)
        >>> tri = Polygon((0, 0), (1, 0), (1, 1))
        >>> p = tri.arbitrary_point('t')
        >>> perimeter = tri.perimeter
        >>> s1, s2 = [s.length for s in tri.sides[:2]]
        >>> p.subs(t, (s1 + s2/2)/perimeter)
        Point2D(1, 1/2)

        """
        t = _symbol(parameter, real=True)  # 创建一个实数符号对象 t
        if t.name in (f.name for f in self.free_symbols):  # 检查 t 是否已经在自由符号中出现过
            raise ValueError('Symbol %s already appears in object and cannot be used as a parameter.' % t.name)
        sides = []  # 初始化一个空列表来存储边界条件
        perimeter = self.perimeter  # 获取多边形的周长
        perim_fraction_start = 0  # 初始化周长分数起点为 0
        for s in self.sides:  # 遍历多边形的每一条边
            side_perim_fraction = s.length/perimeter  # 计算当前边的周长占比
            perim_fraction_end = perim_fraction_start + side_perim_fraction  # 计算当前边的结束周长分数
            pt = s.arbitrary_point(parameter).subs(  # 获取当前边上的参数化点
                t, (t - perim_fraction_start)/side_perim_fraction)
            sides.append(  # 将点和其对应的边界条件添加到列表中
                (pt, (And(perim_fraction_start <= t, t < perim_fraction_end))))
            perim_fraction_start = perim_fraction_end  # 更新起始周长分数
        return Piecewise(*sides)  # 返回多段函数对象，表示多边形上的参数化点

    def parameter_value(self, other, t):
        if not isinstance(other, GeometryEntity):  # 如果 other 不是几何实体
            other = Point(other, dim=self.ambient_dimension)  # 将其转换为一个点
        if not isinstance(other, Point):  # 如果 other 不是 Point 类型，则抛出错误
            raise ValueError("other must be a point")
        if other.free_symbols:  # 如果 other 中包含自由符号，则抛出未实现的错误
            raise NotImplementedError('non-numeric coordinates')
        unknown = False  # 初始化未知标志为 False
        p = self.arbitrary_point(T)  # 获取多边形上的参数化点
        for pt, cond in p.args:  # 遍历参数化点和其条件
            sol = solve(pt - other, T, dict=True)  # 解方程 pt - other = 0，求解 T
            if not sol:  # 如果没有解，则继续下一个循环
                continue
            value = sol[0][T]  # 获取解的值
            if simplify(cond.subs(T, value)) == True:  # 简化并检查条件是否为真
                return {t: value}  # 返回参数值的字典
            unknown = True  # 设置未知标志为 True
        if unknown:  # 如果存在未知点，则抛出给定点可能不在多边形上的错误
            raise ValueError("Given point may not be on %s" % func_name(self))
        raise ValueError("Given point is not on %s" % func_name(self))  # 给定点不在多边形上的错误
    def plot_interval(self, parameter='t'):
        """定义多边形的默认几何图形绘制间隔。

        Parameters
        ==========

        parameter : str, optional
            默认值为 't'。

        Returns
        =======

        plot_interval : list
            [parameter, lower_bound, upper_bound]

        Examples
        ========

        >>> from sympy import Polygon
        >>> p = Polygon((0, 0), (1, 0), (1, 1))
        >>> p.plot_interval()
        [t, 0, 1]

        """
        # 使用参数名创建一个实数符号对象
        t = Symbol(parameter, real=True)
        # 返回一个列表，包含参数名、下界和上界
        return [t, 0, 1]

    def intersection(self, o):
        """计算多边形与几何实体的交集。

        交集可能为空，也可能包含单独的点和完整的线段。

        Parameters
        ==========

        other: GeometryEntity
            另一个几何实体对象

        Returns
        =======

        intersection : list
            包含线段和点的列表

        See Also
        ========

        sympy.geometry.point.Point, sympy.geometry.line.Segment

        Examples
        ========

        >>> from sympy import Point, Polygon, Line
        >>> p1, p2, p3, p4 = map(Point, [(0, 0), (1, 0), (5, 1), (0, 1)])
        >>> poly1 = Polygon(p1, p2, p3, p4)
        >>> p5, p6, p7 = map(Point, [(3, 2), (1, -1), (0, 2)])
        >>> poly2 = Polygon(p5, p6, p7)
        >>> poly1.intersection(poly2)
        [Point2D(1/3, 1), Point2D(2/3, 0), Point2D(9/5, 1/5), Point2D(7/3, 1)]
        >>> poly1.intersection(Line(p1, p2))
        [Segment2D(Point2D(0, 0), Point2D(1, 0))]
        >>> poly1.intersection(p1)
        [Point2D(0, 0)]
        """
        # 初始化一个空列表用于存储交集结果
        intersection_result = []
        # 如果 o 是 Polygon 对象，则获取其边；否则将 o 视为单个几何实体
        k = o.sides if isinstance(o, Polygon) else [o]
        # 遍历当前多边形的边
        for side in self.sides:
            # 遍历其他几何实体的边或者单个几何实体
            for side1 in k:
                # 计算当前边与其他边或几何实体的交集并扩展结果列表
                intersection_result.extend(side.intersection(side1))

        # 去除重复的交集实体
        intersection_result = list(uniq(intersection_result))
        # 将交集结果分为点和线段
        points = [entity for entity in intersection_result if isinstance(entity, Point)]
        segments = [entity for entity in intersection_result if isinstance(entity, Segment)]

        # 如果同时存在点和线段
        if points and segments:
            # 找出点同时在线段中的情况，并从点列表中移除这些点
            points_in_segments = list(uniq([point for point in points for segment in segments if point in segment]))
            if points_in_segments:
                for i in points_in_segments:
                    points.remove(i)
            # 返回按顺序排列的线段和点组成的列表
            return list(ordered(segments + points))
        else:
            # 如果没有同时存在点和线段，则返回按顺序排列的交集结果列表
            return list(ordered(intersection_result))
    def distance(self, o):
        """
        Returns the shortest distance between self and o.

        If o is a point, then self does not need to be convex.
        If o is another polygon self and o must be convex.

        Examples
        ========

        >>> from sympy import Point, Polygon, RegularPolygon
        >>> p1, p2 = map(Point, [(0, 0), (7, 5)])
        >>> poly = Polygon(*RegularPolygon(p1, 1, 3).vertices)
        >>> poly.distance(p2)
        sqrt(61)
        """
        # 如果 o 是一个点
        if isinstance(o, Point):
            # 初始化距离为正无穷
            dist = oo
            # 遍历多边形的边
            for side in self.sides:
                # 计算当前边到点 o 的距离
                current = side.distance(o)
                # 如果距离为零，直接返回零
                if current == 0:
                    return S.Zero
                # 如果当前距离小于已记录的最短距离，则更新最短距离
                elif current < dist:
                    dist = current
            # 返回最短距离
            return dist
        # 如果 o 是一个多边形，并且 self 和 o 都是凸多边形
        elif isinstance(o, Polygon) and self.is_convex() and o.is_convex():
            # 调用 _do_poly_distance 方法计算多边形之间的距离
            return self._do_poly_distance(o)
        # 如果以上条件都不满足，则抛出 NotImplementedError
        raise NotImplementedError()

    def _svg(self, scale_factor=1., fill_color="#66cc99"):
        """Returns SVG path element for the Polygon.

        Parameters
        ==========

        scale_factor : float
            Multiplication factor for the SVG stroke-width.  Default is 1.
        fill_color : str, optional
            Hex string for fill color. Default is "#66cc99".
        """
        # 将多边形顶点映射为数值类型
        verts = map(N, self.vertices)
        # 生成顶点坐标字符串列表
        coords = ["{},{}".format(p.x, p.y) for p in verts]
        # 构建 SVG 路径字符串
        path = "M {} L {} z".format(coords[0], " L ".join(coords[1:]))
        # 返回 SVG path 元素字符串，包括填充颜色、边框颜色和路径
        return (
            '<path fill-rule="evenodd" fill="{2}" stroke="#555555" '
            'stroke-width="{0}" opacity="0.6" d="{1}" />'
            ).format(2. * scale_factor, path, fill_color)

    def _hashable_content(self):

        D = {}
        # 定义内部函数 ref_list 用于生成点列表的参考键列表
        def ref_list(point_list):
            kee = {}
            # 使用有序集合去除重复点，并为每个点分配一个唯一的索引
            for i, p in enumerate(ordered(set(point_list))):
                kee[p] = i
                D[i] = p
            # 返回点列表的参考键列表
            return [kee[p] for p in point_list]

        # 生成正向和反向排序后的点列表的参考键列表
        S1 = ref_list(self.args)
        r_nor = rotate_left(S1, least_rotation(S1))
        S2 = ref_list(list(reversed(self.args)))
        r_rev = rotate_left(S2, least_rotation(S2))
        # 根据字典序比较选择更小的旋转结果
        if r_nor < r_rev:
            r = r_nor
        else:
            r = r_rev
        # 根据最小旋转结果重新排序点列表，生成规范化参数列表
        canonical_args = [ D[order] for order in r ]
        # 返回规范化参数的元组
        return tuple(canonical_args)
   `
# 定义一个方法用于检查几何实体是否包含在当前多边形内部
def __contains__(self, o):
    """
    如果 o 被包含在 self.altitudes 的边界线内部，则返回 True。

    Parameters
    ==========

    o : GeometryEntity
        被检查是否包含在当前多边形内的几何实体

    Returns
    =======

    contained in : bool
        点（如果适用，还有边）是否包含在当前多边形内部的布尔值

    See Also
    ========

    sympy.geometry.entity.GeometryEntity.encloses
        查看更多有关几何实体包含关系的信息

    Examples
    ========

    >>> from sympy import Line, Segment, Point
    >>> p = Point(0, 0)
    >>> q = Point(1, 1)
    >>> s = Segment(p, q*2)
    >>> l = Line(p, q)
    >>> p in q
    False
    >>> p in s
    True
    >>> q*3 in s
    False
    >>> s in l
    True

    """

    if isinstance(o, Polygon):  # 如果 o 是多边形类型
        return self == o  # 直接比较当前多边形和 o 是否相等
    elif isinstance(o, Segment):  # 如果 o 是线段类型
        return any(o in s for s in self.sides)  # 检查是否有任何一个边包含该线段
    elif isinstance(o, Point):  # 如果 o 是点类型
        if o in self.vertices:  # 如果点在当前多边形的顶点列表中
            return True
        for side in self.sides:  # 遍历多边形的边
            if o in side:  # 如果点在某条边上
                return True

    return False  # 默认情况下，返回 False 表示 o 不在当前多边形内部


def bisectors(p, prec=None):
    """
    返回多边形的角平分线。如果提供了 prec，则以该精度近似定义射线的点。

    定义角平分线的点之间的距离为 1。

    Examples
    ========

    >>> from sympy import Polygon, Point
    >>> p = Polygon(Point(0, 0), Point(2, 0), Point(1, 1), Point(0, 3))
    >>> p.bisectors(2)
    {Point2D(0, 0): Ray2D(Point2D(0, 0), Point2D(0.71, 0.71)),
     Point2D(0, 3): Ray2D(Point2D(0, 3), Point2D(0.23, 2.0)),
     Point2D(1, 1): Ray2D(Point2D(1, 1), Point2D(0.19, 0.42)),
     Point2D(2, 0): Ray2D(Point2D(2, 0), Point2D(1.1, 0.38))}
    """
    b = {}  # 初始化一个空字典用于存放角平分线
    pts = list(p.args)  # 将多边形的顶点转换为列表
    pts.append(pts[0])  # 将第一个顶点添加到列表末尾，形成闭合多边形
    cw = Polygon._is_clockwise(*pts[:3])  # 检查多边形的顶点是否顺时针排列
    if cw:
        pts = list(reversed(pts))  # 如果是顺时针排列，反转顶点列表
    for v, a in p.angles.items():  # 遍历多边形的角度字典
        i = pts.index(v)  # 找到当前角对应的顶点在列表中的索引
        p1, p2 = Point._normalize_dimension(pts[i], pts[i + 1])  # 标准化顶点维度
        ray = Ray(p1, p2).rotate(a/2, v)  # 创建旋转后的射线
        dir = ray.direction  # 获取射线的方向向量
        ray = Ray(ray.p1, ray.p1 + dir/dir.distance((0, 0)))  # 根据方向向量调整射线
p.bisectors(2)
        {Point2D(0, 0): Ray2D(Point2D(0, 0), Point2D(0.71, 0.71)),
         Point2D(0, 3): Ray2D(Point2D(0, 3), Point2D(0.23, 2.0)),
         Point2D(1, 1): Ray2D(Point2D(1, 1), Point2D(0.19, 0.42)),
         Point2D(2, 0): Ray2D(Point2D(2, 0), Point2D(1.1, 0.38))}
        """

        b = {}  # Initialize an empty dictionary for storing bisectors
        pts = list(p.args)  # Extract vertices of the polygon
        pts.append(pts[0])  # Close the polygon by appending the first vertex at the end
        cw = Polygon._is_clockwise(*pts[:3])  # Check if the polygon vertices are in clockwise order
        if cw:
            pts = list(reversed(pts))  # Reverse the order of vertices if they are clockwise

        # Iterate over angles of the polygon vertices
        for v, a in p.angles.items():
            i = pts.index(v)
            p1, p2 = Point._normalize_dimension(pts[i], pts[i + 1])  # Normalize points defining the bisector ray
            ray = Ray(p1, p2).rotate(a/2, v)  # Create a ray bisecting the angle at vertex 'v'
            dir = ray.direction  # Get the direction vector of the ray
            ray = Ray(ray.p1, ray.p1 + dir/dir.distance((0, 0)))  # Normalize the ray direction
            if prec is not None:
                ray = Ray(ray.p1, ray.p2.n(prec))  # Approximate ray points to given precision if specified
            b[v] = ray  # Store the bisector ray in the dictionary

        return b  # Return the dictionary of angle bisectors
class RegularPolygon(Polygon):
    """
    A regular polygon.

    Such a polygon has all internal angles equal and all sides the same length.

    Parameters
    ==========

    center : Point
        The center of the regular polygon.
    radius : number or Basic instance
        The distance from the center to a vertex.
    n : int
        The number of sides.

    Attributes
    ==========

    vertices
        Vertices of the regular polygon.
    center
        Center of the regular polygon.
    radius
        Radius of the circumcircle of the regular polygon.
    rotation
        Rotation angle of the regular polygon.
    apothem
        Apothem (distance from the center to the midpoint of a side) of the regular polygon.
    interior_angle
        Interior angle of each side of the regular polygon.
    exterior_angle
        Exterior angle of each side of the regular polygon.
    circumcircle
        Circumcircle (circle passing through all vertices) of the regular polygon.
    incircle
        Incircle (largest circle that can be inscribed inside the polygon) of the regular polygon.
    angles
        Angles formed at the center by adjacent sides of the regular polygon.

    Raises
    ======

    GeometryError
        If the `center` is not a Point, or the `radius` is not a number or Basic instance,
        or the number of sides, `n`, is less than three.

    Notes
    =====

    A RegularPolygon can be instantiated with Polygon using the `n` keyword argument.

    Regular polygons are instantiated with a center, radius, number of sides,
    and a rotation angle. Whereas the arguments of a Polygon are vertices, the
    vertices of the RegularPolygon must be obtained with the vertices method.

    See Also
    ========

    sympy.geometry.point.Point, Polygon

    Examples
    ========

    >>> from sympy import RegularPolygon, Point
    >>> r = RegularPolygon(Point(0, 0), 5, 3)
    >>> r
    RegularPolygon(Point2D(0, 0), 5, 3, 0)
    >>> r.vertices[0]
    Point2D(5, 0)

    """

    __slots__ = ('_n', '_center', '_radius', '_rot')

    def __new__(self, c, r, n, rot=0, **kwargs):
        # Convert `r`, `n`, and `rot` to SymPy expressions if they are not already
        r, n, rot = map(sympify, (r, n, rot))
        # Ensure `c` is a Point object in 2D space
        c = Point(c, dim=2, **kwargs)
        # Check if `r` is an Expr object; raise GeometryError if not
        if not isinstance(r, Expr):
            raise GeometryError("r must be an Expr object, not %s" % r)
        # Check if `n` is a number; raise GeometryError if `n` < 3
        if n.is_Number:
            as_int(n)  # Ensure `n` is an integer; let an error raise if necessary
            if n < 3:
                raise GeometryError("n must be >= 3, not %s" % n)

        # Create a new instance of RegularPolygon with validated parameters
        obj = GeometryEntity.__new__(self, c, r, n, **kwargs)
        obj._n = n
        obj._center = c
        obj._radius = r
        # Normalize rotation angle `rot` within [0, 2π/n) range if `rot` is a number
        obj._rot = rot % (2*S.Pi/n) if rot.is_number else rot
        return obj

    def _eval_evalf(self, prec=15, **options):
        # Evaluate the center, radius, and angle to a given precision
        c, r, n, a = self.args
        dps = prec_to_dps(prec)
        c, r, a = [i.evalf(n=dps, **options) for i in (c, r, a)]
        return self.func(c, r, n, a)

    @property
    def args(self):
        """
        Returns the center point, the radius,
        the number of sides, and the orientation angle.

        Examples
        ========

        >>> from sympy import RegularPolygon, Point
        >>> r = RegularPolygon(Point(0, 0), 5, 3)
        >>> r.args
        (Point2D(0, 0), 5, 3, 0)
        """
        return self._center, self._radius, self._n, self._rot

    def __str__(self):
        # Return a string representation of the RegularPolygon object
        return 'RegularPolygon(%s, %s, %s, %s)' % tuple(self.args)

    def __repr__(self):
        # Return a detailed string representation of the RegularPolygon object
        return 'RegularPolygon(%s, %s, %s, %s)' % tuple(self.args)
    def area(self):
        """Returns the area.

        Examples
        ========

        >>> from sympy import RegularPolygon
        >>> square = RegularPolygon((0, 0), 1, 4)
        >>> square.area
        2
        >>> _ == square.length**2
        True
        """
        # 解构 RegularPolygon 的参数
        c, r, n, rot = self.args
        # 计算并返回正多边形的面积公式
        return sign(r)*n*self.length**2/(4*tan(pi/n))

    @property
    def length(self):
        """Returns the length of the sides.

        The half-length of the side and the apothem form two legs
        of a right triangle whose hypotenuse is the radius of the
        regular polygon.

        Examples
        ========

        >>> from sympy import RegularPolygon
        >>> from sympy import sqrt
        >>> s = square_in_unit_circle = RegularPolygon((0, 0), 1, 4)
        >>> s.length
        sqrt(2)
        >>> sqrt((_/2)**2 + s.apothem**2) == s.radius
        True

        """
        # 返回正多边形边长的计算公式
        return self.radius*2*sin(pi/self._n)

    @property
    def center(self):
        """The center of the RegularPolygon

        This is also the center of the circumscribing circle.

        Returns
        =======

        center : Point

        See Also
        ========

        sympy.geometry.point.Point, sympy.geometry.ellipse.Ellipse.center

        Examples
        ========

        >>> from sympy import RegularPolygon, Point
        >>> rp = RegularPolygon(Point(0, 0), 5, 4)
        >>> rp.center
        Point2D(0, 0)
        """
        # 返回正多边形的中心点
        return self._center

    centroid = center

    @property
    def circumcenter(self):
        """
        Alias for center.

        Examples
        ========

        >>> from sympy import RegularPolygon, Point
        >>> rp = RegularPolygon(Point(0, 0), 5, 4)
        >>> rp.circumcenter
        Point2D(0, 0)
        """
        # 返回正多边形的外心，即中心点
        return self.center

    @property
    def radius(self):
        """Radius of the RegularPolygon

        This is also the radius of the circumscribing circle.

        Returns
        =======

        radius : number or instance of Basic

        See Also
        ========

        sympy.geometry.line.Segment.length, sympy.geometry.ellipse.Circle.radius

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy import RegularPolygon, Point
        >>> radius = Symbol('r')
        >>> rp = RegularPolygon(Point(0, 0), radius, 4)
        >>> rp.radius
        r

        """
        # 返回正多边形的半径
        return self._radius

    @property
    def circumradius(self):
        """
        Alias for radius.

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy import RegularPolygon, Point
        >>> radius = Symbol('r')
        >>> rp = RegularPolygon(Point(0, 0), radius, 4)
        >>> rp.circumradius
        r
        """
        # 返回正多边形的外接圆半径，即半径
        return self.radius

    @property
    def rotation(self):
        """
        CCW angle by which the RegularPolygon is rotated

        Returns
        =======
        rotation : number or instance of Basic

        Examples
        ========
        >>> from sympy import pi
        >>> from sympy.abc import a
        >>> from sympy import RegularPolygon, Point
        >>> RegularPolygon(Point(0, 0), 3, 4, pi/4).rotation
        pi/4

        Numerical rotation angles are made canonical:

        >>> RegularPolygon(Point(0, 0), 3, 4, a).rotation
        a
        >>> RegularPolygon(Point(0, 0), 3, 4, pi).rotation
        0
        """
        return self._rot

    @property
    def apothem(self):
        """
        The inradius of the RegularPolygon.

        The apothem/inradius is the radius of the inscribed circle.

        Returns
        =======
        apothem : number or instance of Basic

        See Also
        ========
        sympy.geometry.line.Segment.length, sympy.geometry.ellipse.Circle.radius

        Examples
        ========
        >>> from sympy import Symbol
        >>> from sympy import RegularPolygon, Point
        >>> radius = Symbol('r')
        >>> rp = RegularPolygon(Point(0, 0), radius, 4)
        >>> rp.apothem
        sqrt(2)*r/2
        """
        return self.radius * cos(S.Pi/self._n)

    @property
    def inradius(self):
        """
        Alias for apothem.

        Examples
        ========
        >>> from sympy import Symbol
        >>> from sympy import RegularPolygon, Point
        >>> radius = Symbol('r')
        >>> rp = RegularPolygon(Point(0, 0), radius, 4)
        >>> rp.inradius
        sqrt(2)*r/2
        """
        return self.apothem

    @property
    def interior_angle(self):
        """
        Measure of the interior angles.

        Returns
        =======
        interior_angle : number

        See Also
        ========
        sympy.geometry.line.LinearEntity.angle_between

        Examples
        ========
        >>> from sympy import RegularPolygon, Point
        >>> rp = RegularPolygon(Point(0, 0), 4, 8)
        >>> rp.interior_angle
        3*pi/4
        """
        return (self._n - 2)*S.Pi/self._n

    @property
    def exterior_angle(self):
        """
        Measure of the exterior angles.

        Returns
        =======
        exterior_angle : number

        See Also
        ========
        sympy.geometry.line.LinearEntity.angle_between

        Examples
        ========
        >>> from sympy import RegularPolygon, Point
        >>> rp = RegularPolygon(Point(0, 0), 4, 8)
        >>> rp.exterior_angle
        pi/4
        """
        return 2*S.Pi/self._n
    def circumcircle(self):
        """
        The circumcircle of the RegularPolygon.

        Returns
        =======

        circumcircle : Circle

        See Also
        ========

        circumcenter, sympy.geometry.ellipse.Circle

        Examples
        ========

        >>> from sympy import RegularPolygon, Point
        >>> rp = RegularPolygon(Point(0, 0), 4, 8)
        >>> rp.circumcircle
        Circle(Point2D(0, 0), 4)

        """
        # 返回正多边形的外接圆对象，圆心为正多边形的中心，半径为正多边形的外接圆半径
        return Circle(self.center, self.radius)

    @property
    def incircle(self):
        """
        The incircle of the RegularPolygon.

        Returns
        =======

        incircle : Circle

        See Also
        ========

        inradius, sympy.geometry.ellipse.Circle

        Examples
        ========

        >>> from sympy import RegularPolygon, Point
        >>> rp = RegularPolygon(Point(0, 0), 4, 7)
        >>> rp.incircle
        Circle(Point2D(0, 0), 4*cos(pi/7))

        """
        # 返回正多边形的内切圆对象，圆心为正多边形的中心，半径为正多边形的内切圆半径
        return Circle(self.center, self.apothem)

    @property
    def angles(self):
        """
        Returns a dictionary with keys, the vertices of the Polygon,
        and values, the interior angle at each vertex.

        Examples
        ========

        >>> from sympy import RegularPolygon, Point
        >>> r = RegularPolygon(Point(0, 0), 5, 3)
        >>> r.angles
        {Point2D(-5/2, -5*sqrt(3)/2): pi/3,
         Point2D(-5/2, 5*sqrt(3)/2): pi/3,
         Point2D(5, 0): pi/3}
        """
        # 创建一个字典，以正多边形的顶点为键，对应的内角为值
        ret = {}
        ang = self.interior_angle
        for v in self.vertices:
            ret[v] = ang
        return ret
    def encloses_point(self, p):
        """
        Return True if p is enclosed by (is inside of) self.

        Notes
        =====

        Being on the border of self is considered False.

        The general Polygon.encloses_point method is called only if
        a point is not within or beyond the incircle or circumcircle,
        respectively.

        Parameters
        ==========

        p : Point
            The point to check if it is enclosed by the polygon.

        Returns
        =======

        encloses_point : True, False or None
            Returns True if the point is inside the polygon, False otherwise.

        See Also
        ========

        sympy.geometry.ellipse.Ellipse.encloses_point

        Examples
        ========

        >>> from sympy import RegularPolygon, S, Point, Symbol
        >>> p = RegularPolygon((0, 0), 3, 4)
        >>> p.encloses_point(Point(0, 0))
        True
        >>> r, R = p.inradius, p.circumradius
        >>> p.encloses_point(Point((r + R)/2, 0))
        True
        >>> p.encloses_point(Point(R/2, R/2 + (R - r)/10))
        False
        >>> t = Symbol('t', real=True)
        >>> p.encloses_point(p.arbitrary_point().subs(t, S.Half))
        False
        >>> p.encloses_point(Point(5, 5))
        False

        """

        c = self.center  # Get the center of the polygon
        d = Segment(c, p).length  # Calculate the distance between center and point p
        if d >= self.radius:  # Check if the distance is greater than or equal to the polygon's radius
            return False  # If so, point p is outside the polygon
        elif d < self.inradius:  # Check if the distance is less than the polygon's inradius
            return True  # If so, point p is inside the polygon
        else:
            # If neither condition is met, defer to the general Polygon.encloses_point method
            return Polygon.encloses_point(self, p)

    def spin(self, angle):
        """Increment *in place* the virtual Polygon's rotation by ccw angle.

        See also: rotate method which moves the center.

        Parameters
        ==========

        angle : float
            The angle by which to rotate the polygon in radians.

        Returns
        =======

        None

        See Also
        ========

        rotation
        rotate : Creates a copy of the RegularPolygon rotated about a Point

        Examples
        ========

        >>> from sympy import Polygon, Point, pi
        >>> r = Polygon(Point(0,0), 1, n=3)
        >>> r.vertices[0]
        Point2D(1, 0)
        >>> r.spin(pi/6)
        >>> r.vertices[0]
        Point2D(sqrt(3)/2, 1/2)

        """

        self._rot += angle  # Increment the rotation angle of the polygon in place

    def rotate(self, angle, pt=None):
        """Override GeometryEntity.rotate to first rotate the RegularPolygon
        about its center.

        Parameters
        ==========

        angle : float
            The angle by which to rotate the polygon in radians.

        pt : Point, optional
            The point about which to rotate. If None, rotate about the center of the polygon.

        Returns
        =======

        RegularPolygon
            A new RegularPolygon object rotated by the specified angle.

        See Also
        ========

        rotation
        spin : Rotates a RegularPolygon in place

        Examples
        ========

        >>> from sympy import Point, RegularPolygon, pi
        >>> t = RegularPolygon(Point(1, 0), 1, 3)
        >>> t.vertices[0] # vertex on x-axis
        Point2D(2, 0)
        >>> t.rotate(pi/2).vertices[0] # vertex on y axis now
        Point2D(0, 2)

        """

        r = type(self)(*self.args)  # Create a copy of the polygon to perform non-inplace rotation
        r._rot += angle  # Increment the rotation angle of the copied polygon
        return GeometryEntity.rotate(r, angle, pt)  # Rotate the copied polygon about the specified point
    def scale(self, x=1, y=1, pt=None):
        """Override GeometryEntity.scale since it is the radius that must be
        scaled (if x == y) or else a new Polygon must be returned.

        >>> from sympy import RegularPolygon

        Symmetric scaling returns a RegularPolygon:

        >>> RegularPolygon((0, 0), 1, 4).scale(2, 2)
        RegularPolygon(Point2D(0, 0), 2, 4, 0)

        Asymmetric scaling returns a kite as a Polygon:

        >>> RegularPolygon((0, 0), 1, 4).scale(2, 1)
        Polygon(Point2D(2, 0), Point2D(0, 1), Point2D(-2, 0), Point2D(0, -1))

        """
        # 如果指定了缩放中心点
        if pt:
            # 将pt转换为Point对象
            pt = Point(pt, dim=2)
            # 先将多边形移动使得pt成为新的原点，然后按照指定比例缩放，最后再移动回原来位置
            return self.translate(*(-pt).args).scale(x, y).translate(*pt.args)
        # 如果缩放比例不相等
        if x != y:
            # 返回一个新的Polygon对象，使用当前多边形的顶点
            return Polygon(*self.vertices).scale(x, y)
        # 否则，执行对称缩放，更新多边形的半径信息
        c, r, n, rot = self.args
        r *= x
        # 返回一个新的RegularPolygon对象，更新了半径信息
        return self.func(c, r, n, rot)

    def reflect(self, line):
        """Override GeometryEntity.reflect since this is not made of only
        points.

        Examples
        ========

        >>> from sympy import RegularPolygon, Line

        >>> RegularPolygon((0, 0), 1, 4).reflect(Line((0, 1), slope=-2))
        RegularPolygon(Point2D(4/5, 2/5), -1, 4, atan(4/3))

        """
        c, r, n, rot = self.args
        v = self.vertices[0]
        d = v - c
        cc = c.reflect(line)
        vv = v.reflect(line)
        dd = vv - cc
        # 计算围绕新中心旋转的角度，以便使顶点对齐
        l1 = Ray((0, 0), dd)
        l2 = Ray((0, 0), d)
        ang = l1.closing_angle(l2)
        rot += ang
        # 修改半径的符号，因为顶点遍历方向已反转
        return self.func(cc, -r, n, rot)

    @property
    def vertices(self):
        """The vertices of the RegularPolygon.

        Returns
        =======

        vertices : list
            Each vertex is a Point.

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import RegularPolygon, Point
        >>> rp = RegularPolygon(Point(0, 0), 5, 4)
        >>> rp.vertices
        [Point2D(5, 0), Point2D(0, 5), Point2D(-5, 0), Point2D(0, -5)]

        """
        c = self._center
        r = abs(self._radius)
        rot = self._rot
        v = 2*S.Pi/self._n

        # 计算并返回多边形的顶点列表，每个顶点是一个Point对象
        return [Point(c.x + r*cos(k*v + rot), c.y + r*sin(k*v + rot))
                for k in range(self._n)]

    def __eq__(self, o):
        # 检查是否为同一类型的多边形
        if not isinstance(o, Polygon):
            return False
        elif not isinstance(o, RegularPolygon):
            return Polygon.__eq__(o, self)
        # 比较多边形的参数是否相等
        return self.args == o.args

    def __hash__(self):
        # 调用父类的哈希方法
        return super().__hash__()
    """
    A polygon with three vertices and three sides.

    Parameters
    ==========

    points : sequence of Points
        The vertices of the triangle.
    keyword: asa, sas, or sss to specify sides/angles of the triangle
        Specifies how the triangle is defined using side lengths and/or angles.

    Attributes
    ==========

    vertices
        Vertices of the triangle.
    altitudes
        Altitudes of the triangle.
    orthocenter
        Orthocenter of the triangle.
    circumcenter
        Circumcenter of the triangle.
    circumradius
        Circumradius of the triangle.
    circumcircle
        Circumcircle of the triangle.
    inradius
        Inradius of the triangle.
    incircle
        Incircle of the triangle.
    exradii
        Exradii of the triangle.
    medians
        Medians of the triangle.
    medial
        Medial triangle of the triangle.
    nine_point_circle
        Nine-point circle of the triangle.

    Raises
    ======

    GeometryError
        If the number of vertices is not equal to three, or one of the vertices
        is not a Point, or a valid keyword is not given.

    See Also
    ========

    sympy.geometry.point.Point, Polygon

    Examples
    ========

    >>> from sympy import Triangle, Point
    >>> Triangle(Point(0, 0), Point(4, 0), Point(4, 3))
    Triangle(Point2D(0, 0), Point2D(4, 0), Point2D(4, 3))

    Keywords sss, sas, or asa can be used to give the desired
    side lengths (in order) and interior angles (in degrees) that
    define the triangle:

    >>> Triangle(sss=(3, 4, 5))
    Triangle(Point2D(0, 0), Point2D(3, 0), Point2D(3, 4))
    >>> Triangle(asa=(30, 1, 30))
    Triangle(Point2D(0, 0), Point2D(1, 0), Point2D(1/2, sqrt(3)/6))
    >>> Triangle(sas=(1, 45, 2))
    Triangle(Point2D(0, 0), Point2D(2, 0), Point2D(sqrt(2)/2, sqrt(2)/2))

    """

    def __new__(cls, *args, **kwargs):
        # Check if exactly three points are provided or a valid keyword for triangle definition
        if len(args) != 3:
            if 'sss' in kwargs:
                return _sss(*[simplify(a) for a in kwargs['sss']])
            if 'asa' in kwargs:
                return _asa(*[simplify(a) for a in kwargs['asa']])
            if 'sas' in kwargs:
                return _sas(*[simplify(a) for a in kwargs['sas']])
            msg = "Triangle instantiates with three points or a valid keyword."
            raise GeometryError(msg)

        # Create Point objects from given arguments
        vertices = [Point(a, dim=2, **kwargs) for a in args]

        # Remove consecutive duplicate vertices
        nodup = []
        for p in vertices:
            if nodup and p == nodup[-1]:
                continue
            nodup.append(p)
        if len(nodup) > 1 and nodup[-1] == nodup[0]:
            nodup.pop()  # last point was same as first

        # Remove collinear points
        i = -3
        while i < len(nodup) - 3 and len(nodup) > 2:
            a, b, c = sorted(
                [nodup[i], nodup[i + 1], nodup[i + 2]], key=default_sort_key)
            if Point.is_collinear(a, b, c):
                nodup[i] = a
                nodup[i + 1] = None
                nodup.pop(i + 1)
            i += 1

        # Filter out None values (removed collinear points)
        vertices = list(filter(lambda x: x is not None, nodup))

        # Determine the type of geometric entity to return based on vertices count
        if len(vertices) == 3:
            return GeometryEntity.__new__(cls, *vertices, **kwargs)
        elif len(vertices) == 2:
            return Segment(*vertices, **kwargs)
        else:
            return Point(*vertices, **kwargs)
    def vertices(self):
        """返回三角形的顶点

        返回
        =======
        
        vertices : tuple
            元组中的每个元素都是一个 Point 对象

        参见
        ======

        sympy.geometry.point.Point

        示例
        ======

        >>> from sympy import Triangle, Point
        >>> t = Triangle(Point(0, 0), Point(4, 0), Point(4, 3))
        >>> t.vertices
        (Point2D(0, 0), Point2D(4, 0), Point2D(4, 3))

        """
        return self.args

    def is_similar(t1, t2):
        """判断另一个三角形是否与此三角形相似。

        如果两个三角形可以等比缩放到彼此，则它们是相似的。

        参数
        ==========
        
        other: Triangle
            另一个三角形对象

        返回
        =======

        is_similar : boolean
            如果相似返回 True，否则返回 False

        参见
        ======

        sympy.geometry.entity.GeometryEntity.is_similar

        示例
        ======

        >>> from sympy import Triangle, Point
        >>> t1 = Triangle(Point(0, 0), Point(4, 0), Point(4, 3))
        >>> t2 = Triangle(Point(0, 0), Point(-4, 0), Point(-4, -3))
        >>> t1.is_similar(t2)
        True

        >>> t2 = Triangle(Point(0, 0), Point(-4, 0), Point(-4, -4))
        >>> t1.is_similar(t2)
        False

        """
        if not isinstance(t2, Polygon):
            return False

        s1_1, s1_2, s1_3 = [side.length for side in t1.sides]
        s2 = [side.length for side in t2.sides]

        def _are_similar(u1, u2, u3, v1, v2, v3):
            e1 = simplify(u1/v1)
            e2 = simplify(u2/v2)
            e3 = simplify(u3/v3)
            return bool(e1 == e2) and bool(e2 == e3)

        # 只有6种排列方式，因此分别进行判断
        return _are_similar(s1_1, s1_2, s1_3, *s2) or \
            _are_similar(s1_1, s1_3, s1_2, *s2) or \
            _are_similar(s1_2, s1_1, s1_3, *s2) or \
            _are_similar(s1_2, s1_3, s1_1, *s2) or \
            _are_similar(s1_3, s1_1, s1_2, *s2) or \
            _are_similar(s1_3, s1_2, s1_1, *s2)

    def is_equilateral(self):
        """判断三角形是否为等边三角形。

        返回
        =======

        is_equilateral : boolean
            如果是等边三角形返回 True，否则返回 False

        参见
        ======

        sympy.geometry.entity.GeometryEntity.is_similar, RegularPolygon
        is_isosceles, is_right, is_scalene

        示例
        ======

        >>> from sympy import Triangle, Point
        >>> t1 = Triangle(Point(0, 0), Point(4, 0), Point(4, 3))
        >>> t1.is_equilateral()
        False

        >>> from sympy import sqrt
        >>> t2 = Triangle(Point(0, 0), Point(10, 0), Point(5, 5*sqrt(3)))
        >>> t2.is_equilateral()
        True

        """
        return not has_variety(s.length for s in self.sides)
    def is_isosceles(self):
        """Are two or more of the sides the same length?

        Returns
        =======

        is_isosceles : boolean
            True if the triangle has at least two sides of equal length, False otherwise.

        See Also
        ========

        is_equilateral, is_right, is_scalene

        Examples
        ========

        >>> from sympy import Triangle, Point
        >>> t1 = Triangle(Point(0, 0), Point(4, 0), Point(2, 4))
        >>> t1.is_isosceles()
        True

        """
        # Check if there are duplicate lengths among the triangle sides
        return has_dups(s.length for s in self.sides)

    def is_scalene(self):
        """Are all the sides of the triangle of different lengths?

        Returns
        =======

        is_scalene : boolean
            True if all sides of the triangle have different lengths, False otherwise.

        See Also
        ========

        is_equilateral, is_isosceles, is_right

        Examples
        ========

        >>> from sympy import Triangle, Point
        >>> t1 = Triangle(Point(0, 0), Point(4, 0), Point(1, 4))
        >>> t1.is_scalene()
        True

        """
        # Check if there are no duplicate lengths among the triangle sides
        return not has_dups(s.length for s in self.sides)

    def is_right(self):
        """Is the triangle right-angled.

        Returns
        =======

        is_right : boolean
            True if the triangle is right-angled, False otherwise.

        See Also
        ========

        sympy.geometry.line.LinearEntity.is_perpendicular
        is_equilateral, is_isosceles, is_scalene

        Examples
        ========

        >>> from sympy import Triangle, Point
        >>> t1 = Triangle(Point(0, 0), Point(4, 0), Point(4, 3))
        >>> t1.is_right()
        True

        """
        s = self.sides
        # Check if any pair of sides are perpendicular to each other
        return Segment.is_perpendicular(s[0], s[1]) or \
            Segment.is_perpendicular(s[1], s[2]) or \
            Segment.is_perpendicular(s[0], s[2])

    @property
    def altitudes(self):
        """The altitudes of the triangle.

        An altitude of a triangle is a segment through a vertex,
        perpendicular to the opposite side, with length being the
        height of the vertex measured from the line containing the side.

        Returns
        =======

        altitudes : dict
            A dictionary mapping each vertex to its corresponding altitude (Segment).

        See Also
        ========

        sympy.geometry.point.Point, sympy.geometry.line.Segment.length

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, 1)
        >>> t = Triangle(p1, p2, p3)
        >>> t.altitudes[p1]
        Segment2D(Point2D(0, 0), Point2D(1/2, 1/2))

        """
        s = self.sides
        v = self.vertices
        # Compute altitudes for each vertex by finding perpendicular segments
        return {v[0]: s[1].perpendicular_segment(v[0]),
                v[1]: s[2].perpendicular_segment(v[1]),
                v[2]: s[0].perpendicular_segment(v[2])}

    @property
    def orthocenter(self):
        """
        The orthocenter of the triangle.

        The orthocenter is the intersection of the altitudes of a triangle.
        It may lie inside, outside or on the triangle.

        Returns
        =======

        orthocenter : Point
            The Point representing the orthocenter of the triangle.

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, 1)
        >>> t = Triangle(p1, p2, p3)
        >>> t.orthocenter
        Point2D(0, 0)
        """
        # Compute the altitudes and vertices of the triangle
        a = self.altitudes
        v = self.vertices
        # Calculate and return the intersection of altitudes as a Point
        return Line(a[v[0]]).intersection(Line(a[v[1]]))[0]

    @property
    def circumcenter(self):
        """
        The circumcenter of the triangle.

        The circumcenter is the center of the circumcircle.

        Returns
        =======

        circumcenter : Point
            The Point representing the circumcenter of the triangle.

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, 1)
        >>> t = Triangle(p1, p2, p3)
        >>> t.circumcenter
        Point2D(1/2, 1/2)
        """
        # Compute perpendicular bisectors of sides and return their intersection as a Point
        a, b, c = [x.perpendicular_bisector() for x in self.sides]
        return a.intersection(b)[0]

    @property
    def circumradius(self):
        """
        The radius of the circumcircle of the triangle.

        Returns
        =======

        circumradius : number or Basic instance
            The radius of the circumcircle.

        See Also
        ========

        sympy.geometry.ellipse.Circle.radius

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy import Point, Triangle
        >>> a = Symbol('a')
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, a)
        >>> t = Triangle(p1, p2, p3)
        >>> t.circumradius
        sqrt(a**2/4 + 1/4)
        """
        # Calculate the distance between circumcenter and any vertex to get the circumradius
        return Point.distance(self.circumcenter, self.vertices[0])

    @property
    def circumcircle(self):
        """
        The circle passing through the three vertices of the triangle.

        Returns
        =======

        circumcircle : Circle
            The Circle object representing the circumcircle of the triangle.

        See Also
        ========

        sympy.geometry.ellipse.Circle

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, 1)
        >>> t = Triangle(p1, p2, p3)
        >>> t.circumcircle
        Circle(Point2D(1/2, 1/2), sqrt(2)/2)

        """
        # Return a Circle centered at circumcenter with radius circumradius
        return Circle(self.circumcenter, self.circumradius)
    def bisectors(self):
        """The angle bisectors of the triangle.

        An angle bisector of a triangle is a straight line through a vertex
        which cuts the corresponding angle in half.

        Returns
        =======

        bisectors : dict
            Each key is a vertex (Point) and each value is the corresponding
            bisector (Segment).

        See Also
        ========

        sympy.geometry.point.Point, sympy.geometry.line.Segment

        Examples
        ========

        >>> from sympy import Point, Triangle, Segment
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, 1)
        >>> t = Triangle(p1, p2, p3)
        >>> from sympy import sqrt
        >>> t.bisectors()[p2] == Segment(Point(1, 0), Point(0, sqrt(2) - 1))
        True

        """
        # 将三角形的边转换为线段对象，以便在计算角平分线时避免包含检查的复杂性
        s = [Line(l) for l in self.sides]
        # 获取三角形的顶点
        v = self.vertices
        # 获取三角形的内心
        c = self.incenter
        # 计算每个顶点到其对应的角平分线的线段
        l1 = Segment(v[0], Line(v[0], c).intersection(s[1])[0])
        l2 = Segment(v[1], Line(v[1], c).intersection(s[2])[0])
        l3 = Segment(v[2], Line(v[2], c).intersection(s[0])[0])
        # 返回每个顶点与其对应角的平分线段的字典
        return {v[0]: l1, v[1]: l2, v[2]: l3}

    @property
    def incenter(self):
        """The center of the incircle.

        The incircle is the circle which lies inside the triangle and touches
        all three sides.

        Returns
        =======

        incenter : Point

        See Also
        ========

        incircle, sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, 1)
        >>> t = Triangle(p1, p2, p3)
        >>> t.incenter
        Point2D(1 - sqrt(2)/2, 1 - sqrt(2)/2)

        """
        # 获取三角形的边长列表
        s = self.sides
        # 计算三角形三条边的长度
        l = Matrix([s[i].length for i in [1, 2, 0]])
        # 计算周长
        p = sum(l)
        # 获取三角形的顶点
        v = self.vertices
        # 计算内心的 x 和 y 坐标
        x = simplify(l.dot(Matrix([vi.x for vi in v]))/p)
        y = simplify(l.dot(Matrix([vi.y for vi in v]))/p)
        # 返回内心的 Point 对象
        return Point(x, y)

    @property
    def inradius(self):
        """The radius of the incircle.

        Returns
        =======

        inradius : number of Basic instance

        See Also
        ========

        incircle, sympy.geometry.ellipse.Circle.radius

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(4, 0), Point(0, 3)
        >>> t = Triangle(p1, p2, p3)
        >>> t.inradius
        1

        """
        # 返回内切圆的半径，通过三角形的面积和周长计算得到
        return simplify(2 * self.area / self.perimeter)
    def incircle(self):
        """The incircle of the triangle.

        The incircle is the circle which lies inside the triangle and touches
        all three sides.

        Returns
        =======

        incircle : Circle
            Returns a Circle object representing the incircle of the triangle.

        See Also
        ========

        sympy.geometry.ellipse.Circle
            More information about the Circle class in SymPy geometry.

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(2, 0), Point(0, 2)
        >>> t = Triangle(p1, p2, p3)
        >>> t.incircle
        Circle(Point2D(2 - sqrt(2), 2 - sqrt(2)), 2 - sqrt(2))
            Returns an example of the incircle for a specific triangle.

        """
        return Circle(self.incenter, self.inradius)

    @property
    def exradii(self):
        """The radius of excircles of a triangle.

        An excircle of the triangle is a circle lying outside the triangle,
        tangent to one of its sides and tangent to the extensions of the
        other two.

        Returns
        =======

        exradii : dict
            Returns a dictionary where keys are triangle sides and values are
            the radii of the corresponding excircles.

        See Also
        ========

        sympy.geometry.polygon.Triangle.inradius
            Related information about the inradius of triangles in SymPy.

        Examples
        ========

        The exradius touches the side of the triangle to which it is keyed, e.g.
        the exradius touching side 2 is:

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(6, 0), Point(0, 2)
        >>> t = Triangle(p1, p2, p3)
        >>> t.exradii[t.sides[2]]
        -2 + sqrt(10)
            Shows an example of calculating the exradius for a specific triangle side.

        References
        ==========

        .. [1] https://mathworld.wolfram.com/Exradius.html
               Background information on exradius in geometry.
        .. [2] https://mathworld.wolfram.com/Excircles.html
               Additional details on excircles in geometry.

        """

        side = self.sides
        a = side[0].length
        b = side[1].length
        c = side[2].length
        s = (a+b+c)/2
        area = self.area
        exradii = {self.sides[0]: simplify(area/(s-a)),
                   self.sides[1]: simplify(area/(s-b)),
                   self.sides[2]: simplify(area/(s-c))}
            Calculates and returns a dictionary mapping each triangle side to its
            corresponding exradius using the triangle's area and side lengths.

        return exradii

    @property
    def excenters(self):
        """Excenters of the triangle.

        An excenter is the center of a circle that is tangent to a side of the
        triangle and the extensions of the other two sides.

        Returns
        =======

        excenters : dict
            A dictionary where keys are sides of the triangle (Line) and values are
            excenter points (Point).

        Examples
        ========

        The excenters are keyed to the side of the triangle to which their corresponding
        excircle is tangent: The center is keyed, e.g. the excenter of a circle touching
        side 0 is:

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(6, 0), Point(0, 2)
        >>> t = Triangle(p1, p2, p3)
        >>> t.excenters[t.sides[0]]
        Point2D(12*sqrt(10), 2/3 + sqrt(10)/3)

        See Also
        ========

        sympy.geometry.polygon.Triangle.exradii

        References
        ==========

        .. [1] https://mathworld.wolfram.com/Excircles.html

        """

        s = self.sides  # 获取三角形的边对象列表
        v = self.vertices  # 获取三角形的顶点对象列表
        a = s[0].length  # 获取第一条边的长度
        b = s[1].length  # 获取第二条边的长度
        c = s[2].length  # 获取第三条边的长度
        x = [v[0].x, v[1].x, v[2].x]  # 获取顶点的 x 坐标列表
        y = [v[0].y, v[1].y, v[2].y]  # 获取顶点的 y 坐标列表

        # 计算三角形三个外心的坐标
        exc_coords = {
            "x1": simplify(-a*x[0]+b*x[1]+c*x[2]/(-a+b+c)),
            "x2": simplify(a*x[0]-b*x[1]+c*x[2]/(a-b+c)),
            "x3": simplify(a*x[0]+b*x[1]-c*x[2]/(a+b-c)),
            "y1": simplify(-a*y[0]+b*y[1]+c*y[2]/(-a+b+c)),
            "y2": simplify(a*y[0]-b*y[1]+c*y[2]/(a-b+c)),
            "y3": simplify(a*y[0]+b*y[1]-c*y[2]/(a+b-c))
        }

        # 构建三角形的外心字典
        excenters = {
            s[0]: Point(exc_coords["x1"], exc_coords["y1"]),
            s[1]: Point(exc_coords["x2"], exc_coords["y2"]),
            s[2]: Point(exc_coords["x3"], exc_coords["y3"])
        }

        return excenters

    @property
    def medians(self):
        """The medians of the triangle.

        A median of a triangle is a straight line through a vertex and the
        midpoint of the opposite side, and divides the triangle into two
        equal areas.

        Returns
        =======

        medians : dict
            Each key is a vertex (Point) and each value is the median (Segment)
            at that point.

        See Also
        ========

        sympy.geometry.point.Point.midpoint, sympy.geometry.line.Segment.midpoint

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, 1)
        >>> t = Triangle(p1, p2, p3)
        >>> t.medians[p1]
        Segment2D(Point2D(0, 0), Point2D(1/2, 1/2))

        """
        s = self.sides  # 获取三角形的边对象列表
        v = self.vertices  # 获取三角形的顶点对象列表
        return {v[0]: Segment(v[0], s[1].midpoint),  # 返回以各顶点为键，中位线为值的字典
                v[1]: Segment(v[1], s[2].midpoint),
                v[2]: Segment(v[2], s[0].midpoint)}
    def medial(self):
        """The medial triangle of the triangle.

        The triangle which is formed from the midpoints of the three sides.

        Returns
        =======

        medial : Triangle

        See Also
        ========

        sympy.geometry.line.Segment.midpoint

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, 1)
        >>> t = Triangle(p1, p2, p3)
        >>> t.medial
        Triangle(Point2D(1/2, 0), Point2D(1/2, 1/2), Point2D(0, 1/2))

        """
        # 获取三角形的三条边
        s = self.sides
        # 返回由三条边中点构成的三角形对象
        return Triangle(s[0].midpoint, s[1].midpoint, s[2].midpoint)

    @property
    def nine_point_circle(self):
        """The nine-point circle of the triangle.

        Nine-point circle is the circumcircle of the medial triangle, which
        passes through the feet of altitudes and the middle points of segments
        connecting the vertices and the orthocenter.

        Returns
        =======

        nine_point_circle : Circle

        See also
        ========

        sympy.geometry.line.Segment.midpoint
        sympy.geometry.polygon.Triangle.medial
        sympy.geometry.polygon.Triangle.orthocenter

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, 1)
        >>> t = Triangle(p1, p2, p3)
        >>> t.nine_point_circle
        Circle(Point2D(1/4, 1/4), sqrt(2)/4)

        """
        # 返回由三角形的中位三角形的顶点构成的圆的对象
        return Circle(*self.medial.vertices)

    @property
    def eulerline(self):
        """The Euler line of the triangle.

        The line which passes through circumcenter, centroid and orthocenter.

        Returns
        =======

        eulerline : Line (or Point for equilateral triangles in which case all
                    centers coincide)

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, 1)
        >>> t = Triangle(p1, p2, p3)
        >>> t.eulerline
        Line2D(Point2D(0, 0), Point2D(1/2, 1/2))

        """
        # 如果三角形是等边三角形，则返回三角形的正交中心
        if self.is_equilateral():
            return self.orthocenter
        # 否则返回通过三角形的正交中心和外接圆心的直线对象
        return Line(self.orthocenter, self.circumcenter)
# 返回给定角度的弧度值（pi = 180度）
def rad(d):
    return d*pi/180

# 返回给定弧度的角度值（pi = 180度）
def deg(r):
    return r/pi*180

# 返回给定角度的正切值
def _slope(d):
    rv = tan(rad(d))
    return rv

# 返回在x轴上具有指定长度为l的边的三角形
def _asa(d1, l, d2):
    # 计算两条直线的交点，形成一个三角形
    xy = Line((0, 0), slope=_slope(d1)).intersection(
        Line((l, 0), slope=_slope(180 - d2)))[0]
    return Triangle((0, 0), (l, 0), xy)

# 返回在x轴上具有指定长度为l1的边的三角形
def _sss(l1, l2, l3):
    # 创建两个圆，找到它们的交点，选择y非负的第一个交点作为三角形的顶点
    c1 = Circle((0, 0), l3)
    c2 = Circle((l1, 0), l2)
    inter = [a for a in c1.intersection(c2) if a.y.is_nonnegative]
    if not inter:
        return None
    pt = inter[0]
    return Triangle((0, 0), (l1, 0), pt)

# 返回在x轴上具有指定长度为l2的边的三角形
def _sas(l1, d, l2):
    # 计算三角形的第三个顶点坐标，根据给定的边长l1和角度d
    p1 = Point(0, 0)
    p2 = Point(l2, 0)
    p3 = Point(cos(rad(d))*l1, sin(rad(d))*l1)
    return Triangle(p1, p2, p3)
```