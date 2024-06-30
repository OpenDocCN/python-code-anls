# `D:\src\scipysrc\sympy\sympy\geometry\parabola.py`

```
"""Parabolic geometrical entity.

Contains
* Parabola

"""

from sympy.core import S  # 导入 Sympy 核心模块 S
from sympy.core.sorting import ordered  # 导入排序函数 ordered
from sympy.core.symbol import _symbol, symbols  # 导入符号相关模块
from sympy.geometry.entity import GeometryEntity, GeometrySet  # 导入几何实体和几何集模块
from sympy.geometry.point import Point, Point2D  # 导入点和二维点模块
from sympy.geometry.line import Line, Line2D, Ray2D, Segment2D, LinearEntity3D  # 导入线段和线的相关模块
from sympy.geometry.ellipse import Ellipse  # 导入椭圆模块
from sympy.functions import sign  # 导入符号函数 sign
from sympy.simplify import simplify  # 导入简化函数 simplify
from sympy.solvers.solvers import solve  # 导入求解函数 solve


class Parabola(GeometrySet):
    """A parabolic GeometryEntity.

    A parabola is declared with a point, that is called 'focus', and
    a line, that is called 'directrix'.
    Only vertical or horizontal parabolas are currently supported.

    Parameters
    ==========

    focus : Point
        Default value is Point(0, 0)
    directrix : Line

    Attributes
    ==========

    focus
    directrix
    axis of symmetry
    focal length
    p parameter
    vertex
    eccentricity

    Raises
    ======
    ValueError
        When `focus` is not a two dimensional point.
        When `focus` is a point of directrix.
    NotImplementedError
        When `directrix` is neither horizontal nor vertical.

    Examples
    ========

    >>> from sympy import Parabola, Point, Line
    >>> p1 = Parabola(Point(0, 0), Line(Point(5, 8), Point(7,8)))
    >>> p1.focus
    Point2D(0, 0)
    >>> p1.directrix
    Line2D(Point2D(5, 8), Point2D(7, 8))

    """

    def __new__(cls, focus=None, directrix=None, **kwargs):
        # 如果提供了 focus 参数，则将其转换为二维点，否则使用默认点 (0, 0)
        if focus:
            focus = Point(focus, dim=2)
        else:
            focus = Point(0, 0)

        # 将 directrix 参数转换为 Line 对象
        directrix = Line(directrix)

        # 如果 directrix 包含 focus 点，则抛出 ValueError 异常
        if directrix.contains(focus):
            raise ValueError('The focus must not be a point of directrix')

        # 调用父类 GeometryEntity 的构造方法，返回 Parabola 实例
        return GeometryEntity.__new__(cls, focus, directrix, **kwargs)

    @property
    def ambient_dimension(self):
        """Returns the ambient dimension of parabola.

        Returns
        =======

        ambient_dimension : integer

        Examples
        ========

        >>> from sympy import Parabola, Point, Line
        >>> f1 = Point(0, 0)
        >>> p1 = Parabola(f1, Line(Point(5, 8), Point(7, 8)))
        >>> p1.ambient_dimension
        2

        """
        return 2

    @property
    def axis_of_symmetry(self):
        """Return the axis of symmetry of the parabola: a line
        perpendicular to the directrix passing through the focus.

        Returns
        =======

        axis_of_symmetry : Line

        See Also
        ========

        sympy.geometry.line.Line

        Examples
        ========

        >>> from sympy import Parabola, Point, Line
        >>> p1 = Parabola(Point(0, 0), Line(Point(5, 8), Point(7, 8)))
        >>> p1.axis_of_symmetry
        Line2D(Point2D(0, 0), Point2D(0, 1))

        """
        return self.directrix.perpendicular_line(self.focus)

    @property
    def focal_length(self):
        """Returns the focal length of the parabola.

        The focal length is the distance between the vertex and the focus.

        Returns
        =======

        focal_length : float or expression

        Examples
        ========

        >>> from sympy import Parabola, Point, Line
        >>> p1 = Parabola(Point(0, 0), Line(Point(5, 8), Point(7, 8)))
        >>> p1.focal_length
        sqrt(29)/2

        """
        return self.focus.distance(self.vertex)
    def directrix(self):
        """
        The directrix of the parabola.

        Returns
        =======

        directrix : Line

        See Also
        ========

        sympy.geometry.line.Line

        Examples
        ========

        >>> from sympy import Parabola, Point, Line
        >>> l1 = Line(Point(5, 8), Point(7, 8))
        >>> p1 = Parabola(Point(0, 0), l1)
        >>> p1.directrix
        Line2D(Point2D(5, 8), Point2D(7, 8))

        """
        # 返回该抛物线的直准线（直线对象）
        return self.args[1]

    @property
    def eccentricity(self):
        """
        The eccentricity of the parabola.

        Returns
        =======

        eccentricity : number

        A parabola may also be characterized as a conic section with an
        eccentricity of 1. As a consequence of this, all parabolas are
        similar, meaning that while they can be different sizes,
        they are all the same shape.

        See Also
        ========

        https://en.wikipedia.org/wiki/Parabola


        Examples
        ========

        >>> from sympy import Parabola, Point, Line
        >>> p1 = Parabola(Point(0, 0), Line(Point(5, 8), Point(7, 8)))
        >>> p1.eccentricity
        1

        Notes
        -----
        The eccentricity for every Parabola is 1 by definition.

        """
        # 返回抛物线的离心率，对于所有的抛物线，离心率都是1
        return S.One

    def equation(self, x='x', y='y'):
        """
        The equation of the parabola.

        Parameters
        ==========
        x : str, optional
            Label for the x-axis. Default value is 'x'.
        y : str, optional
            Label for the y-axis. Default value is 'y'.

        Returns
        =======
        equation : SymPy expression

        Examples
        ========

        >>> from sympy import Parabola, Point, Line
        >>> p1 = Parabola(Point(0, 0), Line(Point(5, 8), Point(7, 8)))
        >>> p1.equation()
        -x**2 - 16*y + 64
        >>> p1.equation('f')
        -f**2 - 16*y + 64
        >>> p1.equation(y='z')
        -x**2 - 16*z + 64

        """
        # 定义符号变量 x 和 y，它们分别代表 x 轴和 y 轴的标签
        x = _symbol(x, real=True)
        y = _symbol(y, real=True)

        # 获取直准线的斜率 m
        m = self.directrix.slope
        if m is S.Infinity:
            # 如果直准线的斜率是无穷大，则使用特定公式计算
            t1 = 4 * (self.p_parameter) * (x - self.vertex.x)
            t2 = (y - self.vertex.y)**2
        elif m == 0:
            # 如果直准线的斜率是 0，则使用另一种特定公式计算
            t1 = 4 * (self.p_parameter) * (y - self.vertex.y)
            t2 = (x - self.vertex.x)**2
        else:
            # 对于其他情况，使用一般公式计算
            a, b = self.focus
            c, d = self.directrix.coefficients[:2]
            t1 = (x - a)**2 + (y - b)**2
            t2 = self.directrix.equation(x, y)**2/(c**2 + d**2)
        # 返回抛物线的方程
        return t1 - t2
    def focal_length(self):
        """The focal length of the parabola.

        Returns
        =======

        focal_lenght : number or symbolic expression
            The distance between the vertex and the focus (or the vertex and directrix),
            measured along the axis of symmetry, is the "focal length".

        Notes
        =====

        The distance between the vertex and the focus
        (or the vertex and directrix), measured along the axis
        of symmetry, is the "focal length".

        See Also
        ========

        https://en.wikipedia.org/wiki/Parabola

        Examples
        ========

        >>> from sympy import Parabola, Point, Line
        >>> p1 = Parabola(Point(0, 0), Line(Point(5, 8), Point(7, 8)))
        >>> p1.focal_length
        4

        """
        # 计算顶点到焦点的距离
        distance = self.directrix.distance(self.focus)
        # 计算焦距（顶点到焦点的距离的一半）
        focal_length = distance / 2

        return focal_length

    @property
    def focus(self):
        """The focus of the parabola.

        Returns
        =======

        focus : Point
            Returns the focus point of the parabola.

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Parabola, Point, Line
        >>> f1 = Point(0, 0)
        >>> p1 = Parabola(f1, Line(Point(5, 8), Point(7, 8)))
        >>> p1.focus
        Point2D(0, 0)

        """
        # 返回抛物线的焦点（抛物线初始化参数的第一个参数，即焦点）
        return self.args[0]
    def intersection(self, o):
        """The intersection of the parabola and another geometrical entity `o`.

        Parameters
        ==========

        o : GeometryEntity, LinearEntity
            Another geometrical entity (e.g., Point, Line, Segment, Ellipse) to find intersections with.

        Returns
        =======

        intersection : list of GeometryEntity objects
            List of points or geometrical entities where the parabola intersects with `o`.

        Examples
        ========

        >>> from sympy import Parabola, Point, Ellipse, Line, Segment
        >>> p1 = Point(0,0)
        >>> l1 = Line(Point(1, -2), Point(-1,-2))
        >>> parabola1 = Parabola(p1, l1)
        >>> parabola1.intersection(Ellipse(Point(0, 0), 2, 5))
        [Point2D(-2, 0), Point2D(2, 0)]
        >>> parabola1.intersection(Line(Point(-7, 3), Point(12, 3)))
        [Point2D(-4, 3), Point2D(4, 3)]
        >>> parabola1.intersection(Segment((-12, -65), (14, -68)))
        []

        """
        # Define symbols x and y as real numbers
        x, y = symbols('x y', real=True)
        # Get the equation of the parabola
        parabola_eq = self.equation()
        
        # Check if `o` is a Parabola
        if isinstance(o, Parabola):
            # Check if `o` is identical to self (this parabola)
            if o in self:
                return [o]
            else:
                # Solve for intersection points between the two parabolas
                return list(ordered([Point(i) for i in solve(
                    [parabola_eq, o.equation()], [x, y], set=True)[1]]))
        
        # Check if `o` is a Point2D
        elif isinstance(o, Point2D):
            # Check if the point lies on the parabola
            if simplify(parabola_eq.subs([(x, o._args[0]), (y, o._args[1])])) == 0:
                return [o]
            else:
                return []
        
        # Check if `o` is a Segment2D or Ray2D
        elif isinstance(o, (Segment2D, Ray2D)):
            # Solve for intersection points between the parabola and the line formed by the segment or ray
            result = solve([parabola_eq,
                Line2D(o.points[0], o.points[1]).equation()],
                [x, y], set=True)[1]
            # Return ordered list of intersection points that lie on the segment or ray
            return list(ordered([Point2D(i) for i in result if i in o]))
        
        # Check if `o` is a Line2D or Ellipse
        elif isinstance(o, (Line2D, Ellipse)):
            # Solve for intersection points between the parabola and `o`
            return list(ordered([Point2D(i) for i in solve(
                [parabola_eq, o.equation()], [x, y], set=True)[1]]))
        
        # If `o` is a LinearEntity3D, raise an error
        elif isinstance(o, LinearEntity3D):
            raise TypeError('Entity must be two dimensional, not three dimensional')
        
        # Raise an error if `o` is of an unrecognized type
        else:
            raise TypeError('Wrong type of argument were put')
    def p_parameter(self):
        """获取抛物线的参数 p。

        Returns
        =======

        p : 数字或符号表达式

        Notes
        =====

        p 的绝对值是焦距。p 的符号决定抛物线的开口方向。上开口的垂直抛物线和右开口的水平抛物线，p 为正值。
        下开口的垂直抛物线和左开口的水平抛物线，p 为负值。

        See Also
        ========

        https://www.sparknotes.com/math/precalc/conicsections/section2/

        Examples
        ========

        >>> from sympy import Parabola, Point, Line
        >>> p1 = Parabola(Point(0, 0), Line(Point(5, 8), Point(7, 8)))
        >>> p1.p_parameter
        -4

        """
        # 获取直线的斜率
        m = self.directrix.slope
        # 根据直线的斜率不同情况计算 p
        if m is S.Infinity:
            x = self.directrix.coefficients[2]
            p = sign(self.focus.args[0] + x)
        elif m == 0:
            y = self.directrix.coefficients[2]
            p = sign(self.focus.args[1] + y)
        else:
            d = self.directrix.projection(self.focus)
            p = sign(self.focus.x - d.x)
        # 返回计算得到的 p 乘以焦距
        return p * self.focal_length

    @property
    def vertex(self):
        """抛物线的顶点。

        Returns
        =======

        vertex : Point

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Parabola, Point, Line
        >>> p1 = Parabola(Point(0, 0), Line(Point(5, 8), Point(7, 8)))
        >>> p1.vertex
        Point2D(0, 4)

        """
        # 获取焦点
        focus = self.focus
        # 获取直线的斜率
        m = self.directrix.slope
        # 根据直线的斜率不同情况计算顶点的坐标
        if m is S.Infinity:
            vertex = Point(focus.args[0] - self.p_parameter, focus.args[1])
        elif m == 0:
            vertex = Point(focus.args[0], focus.args[1] - self.p_parameter)
        else:
            vertex = self.axis_of_symmetry.intersection(self)[0]
        # 返回计算得到的顶点
        return vertex
```