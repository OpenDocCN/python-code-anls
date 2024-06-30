# `D:\src\scipysrc\sympy\sympy\geometry\ellipse.py`

```
# 导入必要的符号计算模块和函数
from sympy.core.expr import Expr
from sympy.core.relational import Eq
from sympy.core import S, pi, sympify
from sympy.core.evalf import N
from sympy.core.parameters import global_parameters
from sympy.core.logic import fuzzy_bool
from sympy.core.numbers import Rational, oo
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy, uniquely_named_symbol, _symbol
from sympy.simplify import simplify, trigsimp
from sympy.functions.elementary.miscellaneous import sqrt, Max
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.functions.special.elliptic_integrals import elliptic_e
# 导入几何实体类和集合类
from .entity import GeometryEntity, GeometrySet
from .exceptions import GeometryError
# 导入几何实体类：直线、线段、射线、二维线段、二维直线、三维线性实体
from .line import Line, Segment, Ray2D, Segment2D, Line2D, LinearEntity3D
# 导入点类：点、二维点、三维点
from .point import Point, Point2D, Point3D
# 导入工具函数：差分、查找
from .util import idiff, find
# 导入多项式和解方程相关模块
from sympy.polys import DomainError, Poly, PolynomialError
from sympy.polys.polyutils import _not_a_coeff, _nsort
from sympy.solvers import solve
from sympy.solvers.solveset import linear_coeffs
from sympy.utilities.misc import filldedent, func_name

# 导入精度和浮点数转换相关函数
from mpmath.libmp.libmpf import prec_to_dps

# 导入随机数生成模块
import random

# 创建符号变量 x 和 y 作为椭圆的虚拟变量
x, y = [Dummy('ellipse_dummy', real=True) for i in range(2)]


class Ellipse(GeometrySet):
    """An elliptical GeometryEntity.

    Parameters
    ==========

    center : Point, optional
        Default value is Point(0, 0)
    hradius : number or SymPy expression, optional
    vradius : number or SymPy expression, optional
    eccentricity : number or SymPy expression, optional
        Two of `hradius`, `vradius` and `eccentricity` must be supplied to
        create an Ellipse. The third is derived from the two supplied.

    Attributes
    ==========

    center
    hradius
    vradius
    area
    circumference
    eccentricity
    periapsis
    apoapsis
    focus_distance
    foci

    Raises
    ======

    GeometryError
        When `hradius`, `vradius` and `eccentricity` are incorrectly supplied
        as parameters.
    TypeError
        When `center` is not a Point.

    See Also
    ========

    Circle

    Notes
    -----
    Constructed from a center and two radii, the first being the horizontal
    radius (along the x-axis) and the second being the vertical radius (along
    the y-axis).

    When symbolic value for hradius and vradius are used, any calculation that
    refers to the foci or the major or minor axis will assume that the ellipse
    has its major radius on the x-axis. If this is not true then a manual
    rotation is necessary.

    Examples
    ========

    >>> from sympy import Ellipse, Point, Rational
    >>> e1 = Ellipse(Point(0, 0), 5, 1)
    >>> e1.hradius, e1.vradius
    (5, 1)
    >>> e2 = Ellipse(Point(3, 1), hradius=3, eccentricity=Rational(4, 5))
    >>> e2
    Ellipse(Point2D(3, 1), 3, 9/5)

    """
    # 检查给定对象是否在当前椭圆内
    def __contains__(self, o):
        # 如果 o 是 Point 类型，则计算点到椭圆方程的值，并检查是否为零
        if isinstance(o, Point):
            res = self.equation(x, y).subs({x: o.x, y: o.y})
            return trigsimp(simplify(res)) is S.Zero
        # 如果 o 是 Ellipse 类型，则比较当前椭圆与 o 是否相等
        elif isinstance(o, Ellipse):
            return self == o
        # 其他情况下返回 False
        return False

    # 判断当前椭圆与另一个 GeometryEntity 是否相等
    def __eq__(self, o):
        """Is the other GeometryEntity the same as this ellipse?"""
        return isinstance(o, Ellipse) and (self.center == o.center and
                                           self.hradius == o.hradius and
                                           self.vradius == o.vradius)

    # 返回当前对象的哈希值
    def __hash__(self):
        return super().__hash__()

    # 构造新的 Ellipse 对象
    def __new__(
        cls, center=None, hradius=None, vradius=None, eccentricity=None, **kwargs):
        # 将 hradius 和 vradius 转换为符号表达式
        hradius = sympify(hradius)
        vradius = sympify(vradius)

        # 如果 center 为 None，则使用默认点 (0, 0)
        if center is None:
            center = Point(0, 0)
        else:
            # 如果 center 不是二维点则抛出错误
            if len(center) != 2:
                raise ValueError('The center of "{}" must be a two dimensional point'.format(cls))
            center = Point(center, dim=2)

        # 检查 hradius、vradius、eccentricity 中恰好有两个不为 None
        if len(list(filter(lambda x: x is not None, (hradius, vradius, eccentricity)))) != 2:
            raise ValueError(filldedent('''
                Exactly two arguments of "hradius", "vradius", and
                "eccentricity" must not be None.'''))

        # 如果给定 eccentricity，则根据它计算 hradius 或 vradius
        if eccentricity is not None:
            eccentricity = sympify(eccentricity)
            if eccentricity.is_negative:
                raise GeometryError("Eccentricity of ellipse/circle should lie between [0, 1)")
            elif hradius is None:
                hradius = vradius / sqrt(1 - eccentricity**2)
            elif vradius is None:
                vradius = hradius * sqrt(1 - eccentricity**2)

        # 如果 hradius 等于 vradius，则返回一个 Circle 对象
        if hradius == vradius:
            return Circle(center, hradius, **kwargs)

        # 如果 hradius 或 vradius 为零，则返回一个 Segment 对象
        if S.Zero in (hradius, vradius):
            return Segment(Point(center[0] - hradius, center[1] - vradius), Point(center[0] + hradius, center[1] + vradius))

        # 如果 hradius 或 vradius 不是实数，则抛出 GeometryError
        if hradius.is_real is False or vradius.is_real is False:
            raise GeometryError("Invalid value encountered when computing hradius / vradius.")

        # 否则返回一个 GeometryEntity 对象
        return GeometryEntity.__new__(cls, center, hradius, vradius, **kwargs)

    # 返回 SVG 格式的椭圆元素表示
    def _svg(self, scale_factor=1., fill_color="#66cc99"):
        """Returns SVG ellipse element for the Ellipse.

        Parameters
        ==========

        scale_factor : float
            Multiplication factor for the SVG stroke-width.  Default is 1.
        fill_color : str, optional
            Hex string for fill color. Default is "#66cc99".
        """
        # 获取当前椭圆的中心坐标和半径
        c = N(self.center)
        h, v = N(self.hradius), N(self.vradius)
        # 构造并返回 SVG 椭圆元素字符串
        return (
            '<ellipse fill="{1}" stroke="#555555" '
            'stroke-width="{0}" opacity="0.6" cx="{2}" cy="{3}" rx="{4}" ry="{5}"/>'
        ).format(2. * scale_factor, fill_color, c.x, c.y, h, v)

    # 返回环境维度，这里固定返回 2
    @property
    def ambient_dimension(self):
        return 2
    @property
    def apoapsis(self):
        """The apoapsis of the ellipse.

        The greatest distance between the focus and the contour.

        Returns
        =======

        apoapsis : number
            The calculated apoapsis value.

        See Also
        ========

        periapsis : Returns shortest distance between foci and contour

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.apoapsis
        2*sqrt(2) + 3

        """
        return self.major * (1 + self.eccentricity)

    def arbitrary_point(self, parameter='t'):
        """A parameterized point on the ellipse.

        Parameters
        ==========

        parameter : str, optional
            Default value is 't'.

        Returns
        =======

        arbitrary_point : Point
            The point on the ellipse corresponding to the given parameter.

        Raises
        ======

        ValueError
            When `parameter` already appears in the functions.

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> e1 = Ellipse(Point(0, 0), 3, 2)
        >>> e1.arbitrary_point()
        Point2D(3*cos(t), 2*sin(t))

        """
        t = _symbol(parameter, real=True)
        if t.name in (f.name for f in self.free_symbols):
            raise ValueError(filldedent('Symbol %s already appears in object '
                                        'and cannot be used as a parameter.' % t.name))
        return Point(self.center.x + self.hradius*cos(t),
                     self.center.y + self.vradius*sin(t))

    @property
    def area(self):
        """The area of the ellipse.

        Returns
        =======

        area : number
            The calculated area of the ellipse.

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.area
        3*pi

        """
        return simplify(S.Pi * self.hradius * self.vradius)

    @property
    def bounds(self):
        """Return a tuple (xmin, ymin, xmax, ymax) representing the bounding
        rectangle for the geometric figure.

        Returns
        =======

        bounds : tuple
            Tuple containing the coordinates of the bounding rectangle.

        """
        h, v = self.hradius, self.vradius
        return (self.center.x - h, self.center.y - v, self.center.x + h, self.center.y + v)

    @property
    def center(self):
        """The center of the ellipse.

        Returns
        =======

        center : number
            The center point (coordinates) of the ellipse.

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.center
        Point2D(0, 0)

        """
        return self.args[0]
    def circumference(self):
        """
        The circumference of the ellipse.

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.circumference
        12*elliptic_e(8/9)

        """
        if self.eccentricity == 1:
            # 如果离心率为1，则椭圆退化为直线段
            return 4*self.major
        elif self.eccentricity == 0:
            # 如果离心率为0，则椭圆为圆
            return 2*pi*self.hradius
        else:
            # 一般情况下，根据椭圆的主轴长度和离心率计算周长
            return 4*self.major*elliptic_e(self.eccentricity**2)

    @property
    def eccentricity(self):
        """
        The eccentricity of the ellipse.

        Returns
        =======

        eccentricity : number

        Examples
        ========

        >>> from sympy import Point, Ellipse, sqrt
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, sqrt(2))
        >>> e1.eccentricity
        sqrt(7)/3

        """
        return self.focus_distance / self.major

    def encloses_point(self, p):
        """
        Return True if p is enclosed by (is inside of) self.

        Notes
        -----
        Being on the border of self is considered False.

        Parameters
        ==========

        p : Point

        Returns
        =======

        encloses_point : True, False or None

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Ellipse, S
        >>> from sympy.abc import t
        >>> e = Ellipse((0, 0), 3, 2)
        >>> e.encloses_point((0, 0))
        True
        >>> e.encloses_point(e.arbitrary_point(t).subs(t, S.Half))
        False
        >>> e.encloses_point((4, 0))
        False

        """
        p = Point(p, dim=2)
        if p in self:
            return False

        if len(self.foci) == 2:
            # 如果从两个焦点到点p的距离之和小于从两个焦点到椭圆短轴的距离之和
            # （即等于椭圆的长轴长度），则点p在椭圆内部
            h1, h2 = [f.distance(p) for f in self.foci]
            test = 2*self.major - (h1 + h2)
        else:
            test = self.radius - self.center.distance(p)

        return fuzzy_bool(test.is_positive)
    def equation(self, x='x', y='y', _slope=None):
        """
        Returns the equation of an ellipse aligned with the x and y axes;
        when slope is given, the equation returned corresponds to an ellipse
        with a major axis having that slope.

        Parameters
        ==========

        x : str, optional
            Label for the x-axis. Default value is 'x'.
        y : str, optional
            Label for the y-axis. Default value is 'y'.
        _slope : Expr, optional
                The slope of the major axis. Ignored when 'None'.

        Returns
        =======

        equation : SymPy expression

        See Also
        ========

        arbitrary_point : Returns parameterized point on ellipse

        Examples
        ========

        >>> from sympy import Point, Ellipse, pi
        >>> from sympy.abc import x, y
        >>> e1 = Ellipse(Point(1, 0), 3, 2)
        >>> eq1 = e1.equation(x, y); eq1
        y**2/4 + (x/3 - 1/3)**2 - 1
        >>> eq2 = e1.equation(x, y, _slope=1); eq2
        (-x + y + 1)**2/8 + (x + y - 1)**2/18 - 1

        A point on e1 satisfies eq1. Let's use one on the x-axis:

        >>> p1 = e1.center + Point(e1.major, 0)
        >>> assert eq1.subs(x, p1.x).subs(y, p1.y) == 0

        When rotated the same as the rotated ellipse, about the center
        point of the ellipse, it will satisfy the rotated ellipse's
        equation, too:

        >>> r1 = p1.rotate(pi/4, e1.center)
        >>> assert eq2.subs(x, r1.x).subs(y, r1.y) == 0

        References
        ==========

        .. [1] https://math.stackexchange.com/questions/108270/what-is-the-equation-of-an-ellipse-that-is-not-aligned-with-the-axis
        .. [2] https://en.wikipedia.org/wiki/Ellipse#Shifted_ellipse

        """

        # Ensure x and y are symbols representing real numbers
        x = _symbol(x, real=True)
        y = _symbol(y, real=True)

        # Calculate differences from the ellipse center
        dx = x - self.center.x
        dy = y - self.center.y

        # If _slope is provided, calculate the equation for a rotated ellipse
        if _slope is not None:
            # Calculate intermediate values
            L = (dy - _slope*dx)**2
            l = (_slope*dy + dx)**2
            h = 1 + _slope**2
            b = h*self.major**2
            a = h*self.minor**2
            # Return the equation of the rotated ellipse
            return l/b + L/a - 1

        else:
            # Calculate the equation of the ellipse aligned with the axes
            t1 = (dx/self.hradius)**2
            t2 = (dy/self.vradius)**2
            # Return the equation of the ellipse aligned with the axes
            return t1 + t2 - 1
    def evolute(self, x='x', y='y'):
        """
        The equation of evolute of the ellipse.

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

        >>> from sympy import Point, Ellipse
        >>> e1 = Ellipse(Point(1, 0), 3, 2)
        >>> e1.evolute()
        2**(2/3)*y**(2/3) + (3*x - 3)**(2/3) - 5**(2/3)
        """
        if len(self.args) != 3:
            raise NotImplementedError('Evolute of arbitrary Ellipse is not supported.')
        x = _symbol(x, real=True)  # Symbol for the x-coordinate, ensuring it's real
        y = _symbol(y, real=True)  # Symbol for the y-coordinate, ensuring it's real
        t1 = (self.hradius*(x - self.center.x))**Rational(2, 3)  # Parameterization part 1
        t2 = (self.vradius*(y - self.center.y))**Rational(2, 3)  # Parameterization part 2
        return t1 + t2 - (self.hradius**2 - self.vradius**2)**Rational(2, 3)  # Evolute equation

    @property
    def foci(self):
        """
        The foci of the ellipse.

        Notes
        -----
        The foci can only be calculated if the major/minor axes are known.

        Raises
        ======

        ValueError
            When the major and minor axis cannot be determined.

        See Also
        ========

        sympy.geometry.point.Point
        focus_distance : Returns the distance between focus and center

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.foci
        (Point2D(-2*sqrt(2), 0), Point2D(2*sqrt(2), 0))

        """
        c = self.center  # Center of the ellipse
        hr, vr = self.hradius, self.vradius  # Horizontal and vertical radii
        if hr == vr:
            return (c, c)  # Circular ellipse case: both foci are at the center

        # Calculate the distance from the center to the focus
        fd = sqrt(self.major**2 - self.minor**2)
        if hr == self.minor:
            # Foci are on the y-axis
            return (c + Point(0, -fd), c + Point(0, fd))
        elif hr == self.major:
            # Foci are on the x-axis
            return (c + Point(-fd, 0), c + Point(fd, 0))

    @property
    def focus_distance(self):
        """
        The focal distance of the ellipse.

        The distance between the center and one focus.

        Returns
        =======

        focus_distance : number

        See Also
        ========

        foci

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.focus_distance
        2*sqrt(2)

        """
        return Point.distance(self.center, self.foci[0])
    def hradius(self):
        """
        The horizontal radius of the ellipse.

        Returns
        =======
        
        hradius : number
            The horizontal radius of the ellipse, accessed through `self.args[1]`.

        See Also
        ========
        
        vradius, major, minor
            Other attributes related to the ellipse dimensions.

        Examples
        ========
        
        >>> from sympy import Point, Ellipse
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.hradius
        3
        """
        return self.args[1]
    def intersection(self, o):
        """
        The intersection of this ellipse and another geometrical entity `o`.

        Parameters
        ==========

        o : GeometryEntity
            Another geometrical entity (e.g., Point, Line, Segment, Circle, Ellipse).

        Returns
        =======

        intersection : list of GeometryEntity objects
            A list containing the intersection points or entities.

        Notes
        -----
        Currently supports intersections with Point, Line, Segment, Ray,
        Circle and Ellipse types.

        See Also
        ========

        sympy.geometry.entity.GeometryEntity

        Examples
        ========

        >>> from sympy import Ellipse, Point, Line
        >>> e = Ellipse(Point(0, 0), 5, 7)
        >>> e.intersection(Point(0, 0))
        []
        >>> e.intersection(Point(5, 0))
        [Point2D(5, 0)]
        >>> e.intersection(Line(Point(0,0), Point(0, 1)))
        [Point2D(0, -7), Point2D(0, 7)]
        >>> e.intersection(Line(Point(5,0), Point(5, 1)))
        [Point2D(5, 0)]
        >>> e.intersection(Line(Point(6,0), Point(6, 1)))
        []
        >>> e = Ellipse(Point(-1, 0), 4, 3)
        >>> e.intersection(Ellipse(Point(1, 0), 4, 3))
        [Point2D(0, -3*sqrt(15)/4), Point2D(0, 3*sqrt(15)/4)]
        >>> e.intersection(Ellipse(Point(5, 0), 4, 3))
        [Point2D(2, -3*sqrt(7)/4), Point2D(2, 3*sqrt(7)/4)]
        >>> e.intersection(Ellipse(Point(100500, 0), 4, 3))
        []
        >>> e.intersection(Ellipse(Point(0, 0), 3, 4))
        [Point2D(3, 0), Point2D(-363/175, -48*sqrt(111)/175), Point2D(-363/175, 48*sqrt(111)/175)]
        >>> e.intersection(Ellipse(Point(-1, 0), 3, 4))
        [Point2D(-17/5, -12/5), Point2D(-17/5, 12/5), Point2D(7/5, -12/5), Point2D(7/5, 12/5)]
        """

        # TODO: Replace solve with nonlinsolve, when nonlinsolve will be able to solve in real domain
        # Determine the type of geometrical entity 'o' and compute intersection accordingly

        if isinstance(o, Point):
            # If 'o' is a Point, check if it lies on the ellipse
            if o in self:
                return [o]  # Return the point if it lies on the ellipse
            else:
                return []  # Otherwise, return an empty list

        elif isinstance(o, (Segment2D, Ray2D)):
            # If 'o' is a Segment2D or Ray2D, compute intersection with the ellipse
            ellipse_equation = self.equation(x, y)
            result = solve([ellipse_equation, Line(
                o.points[0], o.points[1]).equation(x, y)], [x, y],
                set=True)[1]
            return list(ordered([Point(i) for i in result if i in o]))

        elif isinstance(o, Polygon):
            # If 'o' is a Polygon, compute intersection with the ellipse
            return o.intersection(self)

        elif isinstance(o, (Ellipse, Line2D)):
            # If 'o' is another Ellipse or Line2D, compute intersection
            if o == self:
                return self  # Return self if 'o' is the same ellipse
            else:
                ellipse_equation = self.equation(x, y)
                return list(ordered([Point(i) for i in solve(
                    [ellipse_equation, o.equation(x, y)], [x, y],
                    set=True)[1]]))

        elif isinstance(o, LinearEntity3D):
            # Raise an error if 'o' is a 3D linear entity (not supported)
            raise TypeError('Entity must be two dimensional, not three dimensional')

        else:
            # Raise an error if intersection is not handled for the type of 'o'
            raise TypeError('Intersection not handled for %s' % func_name(o))
    def is_tangent(self, o):
        """判断对象 `o` 是否与椭圆相切。

        Parameters
        ==========

        o : GeometryEntity
            椭圆、直线实体或多边形

        Raises
        ======

        NotImplementedError
            当提供错误类型的参数时抛出异常。

        Returns
        =======

        is_tangent: boolean
            如果 `o` 与椭圆相切则返回 True，否则返回 False。

        See Also
        ========

        tangent_lines

        Examples
        ========

        >>> from sympy import Point, Ellipse, Line
        >>> p0, p1, p2 = Point(0, 0), Point(3, 0), Point(3, 3)
        >>> e1 = Ellipse(p0, 3, 2)
        >>> l1 = Line(p1, p2)
        >>> e1.is_tangent(l1)
        True

        """
        # 如果 o 是二维点，则不可能与椭圆相切，直接返回 False
        if isinstance(o, Point2D):
            return False
        # 如果 o 是椭圆，检查与当前椭圆的交点类型
        elif isinstance(o, Ellipse):
            intersect = self.intersection(o)
            # 如果交点是椭圆，则 o 与当前椭圆相切
            if isinstance(intersect, Ellipse):
                return True
            # 如果有交点，检查所有交点处的切线是否相同
            elif intersect:
                return all((self.tangent_lines(i)[0]).equals(o.tangent_lines(i)[0]) for i in intersect)
            else:
                return False
        # 如果 o 是二维直线，检查与当前椭圆的交点情况
        elif isinstance(o, Line2D):
            hit = self.intersection(o)
            # 如果没有交点，则 o 不与椭圆相切
            if not hit:
                return False
            # 如果有且仅有一个交点，则 o 与椭圆相切
            if len(hit) == 1:
                return True
            # 如果有两个交点，比较这两个交点是否相等
            return hit[0].equals(hit[1])
        # 如果 o 是线段或射线，检查与当前椭圆的交点情况
        elif isinstance(o, (Segment2D, Ray2D)):
            intersect = self.intersection(o)
            # 如果有且仅有一个交点，检查该交点处的切线是否包含 o
            if len(intersect) == 1:
                return o in self.tangent_lines(intersect[0])[0]
            else:
                return False
        # 如果 o 是多边形，检查多边形的每条边是否与椭圆相切
        elif isinstance(o, Polygon):
            return all(self.is_tangent(s) for s in o.sides)
        # 如果 o 是三维线性实体或三维点，抛出类型错误
        elif isinstance(o, (LinearEntity3D, Point3D)):
            raise TypeError('实体必须是二维的，不能是三维的')
        else:
            raise TypeError('Is_tangent 对于 %s 类型的对象尚未处理' % func_name(o))

    @property
    def major(self):
        """椭圆的主轴长（如果可以确定）或水平半径。

        Returns
        =======

        major : number or expression
            返回椭圆的主轴长或表达式。

        See Also
        ========

        hradius, vradius, minor

        Examples
        ========

        >>> from sympy import Point, Ellipse, Symbol
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.major
        3

        >>> a = Symbol('a')
        >>> b = Symbol('b')
        >>> Ellipse(p1, a, b).major
        a
        >>> Ellipse(p1, b, a).major
        b

        >>> m = Symbol('m')
        >>> M = m + 1
        >>> Ellipse(p1, m, M).major
        m + 1

        """
        ab = self.args[1:3]
        # 如果只有一个参数，返回该参数作为主轴长
        if len(ab) == 1:
            return ab[0]
        a, b = ab
        o = b - a < 0
        # 如果 b - a 小于 0，则返回 a 作为主轴长
        if o == True:
            return a
        # 如果 b - a 大于等于 0，则返回 b 作为主轴长
        elif o == False:
            return b
        # 默认情况下返回水平半径作为主轴长
        return self.hradius
    def minor(self):
        """Returns the shorter axis of the ellipse if determinable, otherwise vradius.

        Returns
        =======
        
        minor : number or expression
            The shorter axis of the ellipse.
        
        See Also
        ========
        
        hradius, vradius, major
        
        Examples
        ========
        
        >>> from sympy import Point, Ellipse, Symbol
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.minor
        1
        
        >>> a = Symbol('a')
        >>> b = Symbol('b')
        >>> Ellipse(p1, a, b).minor
        b
        >>> Ellipse(p1, b, a).minor
        a
        
        >>> m = Symbol('m')
        >>> M = m + 1
        >>> Ellipse(p1, m, M).minor
        m
        
        """
        ab = self.args[1:3]  # 获取椭圆参数中的长轴半径和短轴半径
        if len(ab) == 1:  # 如果只有一个参数
            return ab[0]  # 返回该参数作为短轴半径
        a, b = ab  # 否则，将两个参数分别赋给a和b
        o = a - b < 0  # 判断长轴半径和短轴半径的大小关系
        if o == True:  # 如果长轴小于短轴
            return a  # 返回长轴作为短轴半径
        elif o == False:  # 如果长轴大于等于短轴
            return b  # 返回短轴作为短轴半径
        return self.vradius  # 如果无法确定，则返回默认的垂直半径

    @property
    def periapsis(self):
        """Returns the periapsis of the ellipse.

        The periapsis is the shortest distance between the focus and the contour.

        Returns
        =======
        
        periapsis : number
            The periapsis of the ellipse.
        
        See Also
        ========
        
        apoapsis : Returns greatest distance between focus and contour
        
        Examples
        ========
        
        >>> from sympy import Point, Ellipse
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.periapsis
        3 - 2*sqrt(2)
        
        """
        return self.major * (1 - self.eccentricity)

    @property
    def semilatus_rectum(self):
        """
        Calculates the semi-latus rectum of the Ellipse.

        Semi-latus rectum is defined as one half of the chord through a
        focus parallel to the conic section directrix of a conic section.

        Returns
        =======
        
        semilatus_rectum : number
            The semi-latus rectum of the ellipse.
        
        See Also
        ========
        
        apoapsis : Returns greatest distance between focus and contour
        
        periapsis : The shortest distance between the focus and the contour
        
        Examples
        ========
        
        >>> from sympy import Point, Ellipse
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.semilatus_rectum
        1/3
        
        References
        ==========
        
        .. [1] https://mathworld.wolfram.com/SemilatusRectum.html
        .. [2] https://en.wikipedia.org/wiki/Ellipse#Semi-latus_rectum
        
        """
        return self.major * (1 - self.eccentricity ** 2)

    def auxiliary_circle(self):
        """Returns a Circle whose diameter is the major axis of the ellipse.

        Examples
        ========
        
        >>> from sympy import Ellipse, Point, symbols
        >>> c = Point(1, 2)
        >>> Ellipse(c, 8, 7).auxiliary_circle()
        Circle(Point2D(1, 2), 8)
        >>> a, b = symbols('a b')
        >>> Ellipse(c, a, b).auxiliary_circle()
        Circle(Point2D(1, 2), Max(a, b))
        """
        return Circle(self.center, Max(self.hradius, self.vradius))
    def director_circle(self):
        """
        Returns a Circle consisting of all points where two perpendicular
        tangent lines to the ellipse cross each other.

        Returns
        =======

        Circle
            A director circle returned as a geometric object.

        Examples
        ========

        >>> from sympy import Ellipse, Point, symbols
        >>> c = Point(3,8)
        >>> Ellipse(c, 7, 9).director_circle()
        Circle(Point2D(3, 8), sqrt(130))
        >>> a, b = symbols('a b')
        >>> Ellipse(c, a, b).director_circle()
        Circle(Point2D(3, 8), sqrt(a**2 + b**2))

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Director_circle

        """
        # 返回椭圆的导向圆，即通过椭圆上任意一点作两条垂直切线交点组成的圆
        return Circle(self.center, sqrt(self.hradius**2 + self.vradius**2))

    def plot_interval(self, parameter='t'):
        """The plot interval for the default geometric plot of the Ellipse.

        Parameters
        ==========

        parameter : str, optional
            Default value is 't'.

        Returns
        =======

        plot_interval : list
            [parameter, lower_bound, upper_bound]

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> e1 = Ellipse(Point(0, 0), 3, 2)
        >>> e1.plot_interval()
        [t, -pi, pi]

        """
        # 返回用于默认椭圆几何图绘制的参数范围列表
        t = _symbol(parameter, real=True)
        return [t, -S.Pi, S.Pi]
    def random_point(self, seed=None):
        """返回椭圆上的随机点。

        Returns
        =======
        
        point : Point
            返回一个 Point 对象，代表椭圆上的一个随机点。

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> e1 = Ellipse(Point(0, 0), 3, 2)
        >>> e1.random_point() # 返回一个随机点
        Point2D(...)
        >>> p1 = e1.random_point(seed=0); p1.n(2)
        Point2D(2.1, 1.4)

        Notes
        =====

        创建随机点时，可以用一个随机数替换参数。然而，这个随机数应该是有理数，
        否则点可能不会被认为在椭圆内：

        >>> from sympy.abc import t
        >>> from sympy import Rational
        >>> arb = e1.arbitrary_point(t); arb
        Point2D(3*cos(t), 2*sin(t))
        >>> arb.subs(t, .1) in e1
        False
        >>> arb.subs(t, Rational(.1)) in e1
        True
        >>> arb.subs(t, Rational('.1')) in e1
        True

        See Also
        ========

        sympy.geometry.point.Point
            返回参数化的椭圆上的点
        arbitrary_point : Returns parameterized point on ellipse
            返回椭圆上的参数化点
        """
        # 定义一个实数符号 t
        t = _symbol('t', real=True)
        # 获取椭圆上的一个任意点的参数
        x, y = self.arbitrary_point(t).args
        # 获取一个在 [-1, 1) 范围内的随机值对应的 cos(t)，并确认它在椭圆内
        if seed is not None:
            # 使用给定的种子初始化随机数生成器
            rng = random.Random(seed)
        else:
            # 使用默认的随机数生成器
            rng = random
        # 将随机数化简为有理数，否则 Float 类型将转换 s 为 Float
        r = Rational(rng.random())
        c = 2*r - 1
        s = sqrt(1 - c**2)
        # 返回一个经过替换后的 Point 对象，以确保点在椭圆内
        return Point(x.subs(cos(t), c), y.subs(sin(t), s))
    def reflect(self, line):
        """Override GeometryEntity.reflect since the radius
        is not a GeometryEntity.

        Examples
        ========

        >>> from sympy import Circle, Line
        >>> Circle((0, 1), 1).reflect(Line((0, 0), (1, 1)))
        Circle(Point2D(1, 0), -1)
        >>> from sympy import Ellipse, Line, Point
        >>> Ellipse(Point(3, 4), 1, 3).reflect(Line(Point(0, -4), Point(5, 0)))
        Traceback (most recent call last):
        ...
        NotImplementedError:
        General Ellipse is not supported but the equation of the reflected
        Ellipse is given by the zeros of: f(x, y) = (9*x/41 + 40*y/41 +
        37/41)**2 + (40*x/123 - 3*y/41 - 364/123)**2 - 1

        Notes
        =====

        Until the general ellipse (with no axis parallel to the x-axis) is
        supported a NotImplemented error is raised and the equation whose
        zeros define the rotated ellipse is given.

        """
        # 检查直线斜率是否为水平或垂直
        if line.slope in (0, oo):
            # 反射圆形或水平椭圆
            c = self.center
            c = c.reflect(line)
            return self.func(c, -self.hradius, self.vradius)
        else:
            # 对于非水平或垂直的情况
            x, y = [uniquely_named_symbol(
                name, (self, line), modify=lambda s: '_' + s, real=True)
                for name in 'xy']
            # 计算椭圆的方程式
            expr = self.equation(x, y)
            # 反射点并计算结果
            p = Point(x, y).reflect(line)
            result = expr.subs(zip((x, y), p.args
                                   ), simultaneous=True)
            # 抛出未实现错误，提供反射椭圆的方程式
            raise NotImplementedError(filldedent(
                'General Ellipse is not supported but the equation '
                'of the reflected Ellipse is given by the zeros of: ' +
                "f(%s, %s) = %s" % (str(x), str(y), str(result))))

    def rotate(self, angle=0, pt=None):
        """Rotate ``angle`` radians counterclockwise about Point ``pt``.

        Note: since the general ellipse is not supported, only rotations that
        are integer multiples of pi/2 are allowed.

        Examples
        ========

        >>> from sympy import Ellipse, pi
        >>> Ellipse((1, 0), 2, 1).rotate(pi/2)
        Ellipse(Point2D(0, 1), 1, 2)
        >>> Ellipse((1, 0), 2, 1).rotate(pi)
        Ellipse(Point2D(-1, 0), 2, 1)
        """
        # 如果水平半径等于垂直半径，直接旋转
        if self.hradius == self.vradius:
            return self.func(self.center.rotate(angle, pt), self.hradius)
        # 如果角度是 pi/2 的整数倍，允许旋转
        if (angle/S.Pi).is_integer:
            return super().rotate(angle, pt)
        # 如果角度是 pi 的整数倍，交换半径再旋转
        if (2*angle/S.Pi).is_integer:
            return self.func(self.center.rotate(angle, pt), self.vradius, self.hradius)
        # 否则，抛出未实现错误，只支持 pi/2 的旋转
        raise NotImplementedError('Only rotations of pi/2 are currently supported for Ellipse.')
    # 覆盖 GeometryEntity.scale 方法，因为需要缩放主轴和副轴，它们不是 GeometryEntities。

    c = self.center
    # 获取椭圆的中心点

    if pt:
        # 如果给定了参考点 pt
        pt = Point(pt, dim=2)
        # 将 pt 转换为二维点对象
        return self.translate(*(-pt).args).scale(x, y).translate(*pt.args)
        # 将椭圆平移到参考点 pt，进行缩放操作，再移回原来的位置

    h = self.hradius
    # 获取椭圆的水平半径
    v = self.vradius
    # 获取椭圆的垂直半径

    return self.func(c.scale(x, y), hradius=h*x, vradius=v*y)
    # 对椭圆中心进行缩放操作，同时更新水平和垂直半径，返回缩放后的新椭圆对象
    def tangent_lines(self, p):
        """计算点 `p` 与椭圆之间的切线。

        如果 `p` 在椭圆上，则返回通过点 `p` 的切线。
        否则，返回从点 `p` 到椭圆的切线（如果有多条），如果不可能找到切线则返回 None（例如，`p` 在椭圆内部）。

        Parameters
        ==========

        p : Point
            指定的点对象，用于计算切线。

        Returns
        =======

        tangent_lines : list with 1 or 2 Lines
            包含 1 或 2 条 Line 对象的列表，表示计算得到的切线。

        Raises
        ======

        NotImplementedError
            当点 `p` 不在椭圆上时抛出，目前仅支持计算点在椭圆上的切线。

        See Also
        ========

        sympy.geometry.point.Point, sympy.geometry.line.Line
            相关的点和直线对象定义。

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> e1 = Ellipse(Point(0, 0), 3, 2)
        >>> e1.tangent_lines(Point(3, 0))
        [Line2D(Point2D(3, 0), Point2D(3, -12))]

        """
        p = Point(p, dim=2)  # 将输入点 `p` 转换为二维点对象
        if self.encloses_point(p):  # 检查点 `p` 是否在椭圆内部
            return []  # 如果在内部，则返回空列表表示无切线

        if p in self:  # 如果点 `p` 在椭圆上
            delta = self.center - p  # 计算椭圆中心到点 `p` 的向量差
            rise = (self.vradius**2) * delta.x  # 垂直方向上的变化量
            run = -(self.hradius**2) * delta.y  # 水平方向上的变化量
            p2 = Point(simplify(p.x + run),
                       simplify(p.y + rise))  # 计算经过点 `p` 的切线上的另一点
            return [Line(p, p2)]  # 返回从点 `p` 到 `p2` 的直线作为切线

        else:  # 如果点 `p` 不在椭圆上
            if len(self.foci) == 2:  # 椭圆有两个焦点
                f1, f2 = self.foci
                maj = self.hradius  # 椭圆的半长轴长度
                test = (2 * maj -
                        Point.distance(f1, p) -
                        Point.distance(f2, p))  # 计算判断点 `p` 是否在椭圆外部的条件
            else:  # 椭圆为圆形，只有一个焦点
                test = self.radius - Point.distance(self.center, p)  # 判断点 `p` 是否在椭圆外部的条件

            if test.is_number and test.is_positive:
                return []  # 如果判断点 `p` 在椭圆外部，则返回空列表表示无切线

            # 否则尝试计算切线方程
            eq = self.equation(x, y)  # 获取椭圆的方程式
            dydx = idiff(eq, y, x)  # 计算方程 `eq` 对于 `y` 对 `x` 的隐函数导数
            slope = Line(p, Point(x, y)).slope  # 计算通过点 `p` 和 `(x, y)` 的直线斜率

            # TODO: 当该行代码测试通过时，用 `solveset` 替换 `solve`
            tangent_points = solve([slope - dydx, eq], [x, y])  # 解方程组以获取切线与椭圆的交点坐标

            # 处理水平和垂直的切线情况
            if len(tangent_points) == 1:
                if tangent_points[0][0] == p.x or tangent_points[0][1] == p.y:
                    return [Line(p, p + Point(1, 0)), Line(p, p + Point(0, 1))]
                else:
                    return [Line(p, p + Point(0, 1)), Line(p, tangent_points[0])]
            
            # 处理其他情况
            return [Line(p, tangent_points[0]), Line(p, tangent_points[1])]
    def vradius(self):
        """The vertical radius of the ellipse.

        Returns
        =======

        vradius : number
            返回椭圆的垂直半径。

        See Also
        ========

        hradius, major, minor
            相关的椭圆参数：水平半径、主轴、副轴。

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.vradius
        1
        示例：创建一个以原点为中心、水平半径为3、垂直半径为1的椭圆，并获取其垂直半径。

        """
        return self.args[2]


    def second_moment_of_area(self, point=None):
        """Returns the second moment and product moment area of an ellipse.

        Parameters
        ==========

        point : Point, two-tuple of sympifiable objects, or None(default=None)
            point是第二矩及面积矩的计算点。
            如果"point=None"，则计算关于椭圆质心通过的轴。

        Returns
        =======

        I_xx, I_yy, I_xy : number or SymPy expression
            I_xx, I_yy是椭圆的第二矩。
            I_xy是椭圆的产品矩。

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.second_moment_of_area()
        (3*pi/4, 27*pi/4, 0)
        示例：创建一个以原点为中心、水平半径为3、垂直半径为1的椭圆，并计算其关于椭圆质心的第二矩及面积矩。

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/List_of_second_moments_of_area
            参考资料：椭圆的第二矩及面积矩的更多信息。

        """

        I_xx = (S.Pi*(self.hradius)*(self.vradius**3))/4
        I_yy = (S.Pi*(self.hradius**3)*(self.vradius))/4
        I_xy = 0

        if point is None:
            return I_xx, I_yy, I_xy

        # parallel axis theorem
        I_xx = I_xx + self.area*((point[1] - self.center.y)**2)
        I_yy = I_yy + self.area*((point[0] - self.center.x)**2)
        I_xy = I_xy + self.area*(point[0] - self.center.x)*(point[1] - self.center.y)

        return I_xx, I_yy, I_xy
    def polar_second_moment_of_area(self):
        """
        Returns the polar second moment of area of an Ellipse

        It is a constituent of the second moment of area, linked through
        the perpendicular axis theorem. While the planar second moment of
        area describes an object's resistance to deflection (bending) when
        subjected to a force applied to a plane parallel to the central
        axis, the polar second moment of area describes an object's
        resistance to deflection when subjected to a moment applied in a
        plane perpendicular to the object's central axis (i.e. parallel to
        the cross-section)

        Examples
        ========

        >>> from sympy import symbols, Circle, Ellipse
        >>> c = Circle((5, 5), 4)
        >>> c.polar_second_moment_of_area()
        128*pi
        >>> a, b = symbols('a, b')
        >>> e = Ellipse((0, 0), a, b)
        >>> e.polar_second_moment_of_area()
        pi*a**3*b/4 + pi*a*b**3/4

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Polar_moment_of_inertia

        """
        # 计算椭圆的二阶矩，即沿两个正交轴的二阶矩之和
        second_moment = self.second_moment_of_area()
        # 返回椭圆的极性二阶矩，即两个正交轴的二阶矩之和
        return second_moment[0] + second_moment[1]
    def section_modulus(self, point=None):
        """Returns a tuple with the section modulus of an ellipse

        Section modulus is a geometric property of an ellipse defined as the
        ratio of second moment of area to the distance of the extreme end of
        the ellipse from the centroidal axis.

        Parameters
        ==========

        point : Point, two-tuple of sympifyable objects, or None(default=None)
            point is the point at which section modulus is to be found.
            If "point=None" section modulus will be calculated for the
            point farthest from the centroidal axis of the ellipse.

        Returns
        =======

        S_x, S_y: numbers or SymPy expressions
                  S_x is the section modulus with respect to the x-axis
                  S_y is the section modulus with respect to the y-axis
                  A negative sign indicates that the section modulus is
                  determined for a point below the centroidal axis.

        Examples
        ========

        >>> from sympy import Symbol, Ellipse, Circle, Point2D
        >>> d = Symbol('d', positive=True)
        >>> c = Circle((0, 0), d/2)
        >>> c.section_modulus()
        (pi*d**3/32, pi*d**3/32)
        >>> e = Ellipse(Point2D(0, 0), 2, 4)
        >>> e.section_modulus()
        (8*pi, 4*pi)
        >>> e.section_modulus((2, 2))
        (16*pi, 4*pi)

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Section_modulus

        """
        # Extract the center coordinates of the ellipse
        x_c, y_c = self.center
        
        if point is None:
            # If point is not provided, calculate section modulus using extreme distances
            # Determine the minimum and maximum distances from centroid to ellipse bounds
            x_min, y_min, x_max, y_max = self.bounds
            # Calculate distances from centroid to extreme ends along x and y axes
            y = max(y_c - y_min, y_max - y_c)
            x = max(x_c - x_min, x_max - x_c)
        else:
            # If point is provided, calculate section modulus using distances from center to point
            point = Point2D(point)
            # Calculate distances from centroid to the given point along x and y axes
            y = point.y - y_c
            x = point.x - x_c

        # Calculate the second moment of area of the ellipse
        second_moment = self.second_moment_of_area()
        # Calculate section modulus along x and y axes
        S_x = second_moment[0] / y
        S_y = second_moment[1] / x

        # Return the calculated section modulus values
        return S_x, S_y
# Circle 类继承自 Ellipse 类，表示平面上的一个圆形。

"""A circle in space.

Constructed simply from a center and a radius, from three
non-collinear points, or the equation of a circle.

Parameters
==========

center : Point
radius : number or SymPy expression
points : sequence of three Points
equation : equation of a circle

Attributes
==========

radius (synonymous with hradius, vradius, major and minor)
circumference
equation

Raises
======

GeometryError
    When the given equation is not that of a circle.
    When trying to construct circle from incorrect parameters.

See Also
========

Ellipse, sympy.geometry.point.Point

Examples
========

>>> from sympy import Point, Circle, Eq
>>> from sympy.abc import x, y, a, b

A circle constructed from a center and radius:

>>> c1 = Circle(Point(0, 0), 5)
>>> c1.hradius, c1.vradius, c1.radius
(5, 5, 5)

A circle constructed from three points:

>>> c2 = Circle(Point(0, 0), Point(1, 1), Point(1, 0))
>>> c2.hradius, c2.vradius, c2.radius, c2.center
(sqrt(2)/2, sqrt(2)/2, sqrt(2)/2, Point2D(1/2, 1/2))

A circle can be constructed from an equation in the form
`a*x**2 + by**2 + gx + hy + c = 0`, too:

>>> Circle(x**2 + y**2 - 25)
Circle(Point2D(0, 0), 5)

If the variables corresponding to x and y are named something
else, their name or symbol can be supplied:

>>> Circle(Eq(a**2 + b**2, 25), x='a', y=b)
Circle(Point2D(0, 0), 5)
"""
    # 定义一个特殊方法 __new__，用于创建 Circle 类的实例
    def __new__(cls, *args, **kwargs):
        # 从 kwargs 或者全局参数中获取 evaluate 的值
        evaluate = kwargs.get('evaluate', global_parameters.evaluate)
        # 如果只有一个参数且是 Expr 或 Eq 类的实例
        if len(args) == 1 and isinstance(args[0], (Expr, Eq)):
            # 从 kwargs 中获取或设置默认的变量 x 和 y
            x = kwargs.get('x', 'x')
            y = kwargs.get('y', 'y')
            # 对方程进行展开处理
            equation = args[0].expand()
            # 如果方程是 Eq 类的实例，转化为其左右两侧的差
            if isinstance(equation, Eq):
                equation = equation.lhs - equation.rhs
            # 分别求解 x 和 y 在方程中的位置
            x = find(x, equation)
            y = find(y, equation)

            try:
                # 获取线性系数 a, b, c, d, e
                a, b, c, d, e = linear_coeffs(equation, x**2, y**2, x, y)
            except ValueError:
                # 如果无法获取系数，则抛出几何错误
                raise GeometryError("The given equation is not that of a circle.")

            # 检查系数是否合法，如果 a 或 b 中有零，或者 a 不等于 b，则抛出几何错误
            if S.Zero in (a, b) or a != b:
                raise GeometryError("The given equation is not that of a circle.")

            # 计算圆心的 x 和 y 坐标
            center_x = -c/a/2
            center_y = -d/b/2
            # 计算半径的平方
            r2 = (center_x**2) + (center_y**2) - e/a

            # 返回一个 Circle 实例，其中心为 (center_x, center_y)，半径为 sqrt(r2)
            return Circle((center_x, center_y), sqrt(r2), evaluate=evaluate)

        else:
            c, r = None, None
            # 如果有三个参数
            if len(args) == 3:
                # 将每个参数转换为 Point 类的实例，并创建 Triangle 对象 t
                args = [Point(a, dim=2, evaluate=evaluate) for a in args]
                t = Triangle(*args)
                # 如果 t 不是 Triangle 的实例，则返回 t
                if not isinstance(t, Triangle):
                    return t
                # 获取 t 的外心和外接圆半径
                c = t.circumcenter
                r = t.circumradius
            # 如果有两个参数
            elif len(args) == 2:
                # 假设为 (center, radius) 对
                c = Point(args[0], dim=2, evaluate=evaluate)
                r = args[1]
                # 禁止使用虚数半径
                try:
                    r = Point(r, 0, evaluate=evaluate).x
                except ValueError:
                    raise GeometryError("Circle with imaginary radius is not permitted")

            # 如果 c 或 r 有一个为 None，则抛出几何错误
            if not (c is None or r is None):
                # 如果 r 等于 0，则返回圆心 c
                if r == 0:
                    return c
                # 否则调用 GeometryEntity 类的 __new__ 方法创建实例
                return GeometryEntity.__new__(cls, c, r, **kwargs)

            # 如果参数无法解析，则抛出几何错误
            raise GeometryError("Circle.__new__ received unknown arguments")

    # 定义一个内部方法 _eval_evalf，用于求取数值近似值
    def _eval_evalf(self, prec=15, **options):
        # 获取圆的中心点和半径
        pt, r = self.args
        # 将精度 prec 转换为小数位数 dps
        dps = prec_to_dps(prec)
        # 对圆的中心点和半径进行数值近似计算
        pt = pt.evalf(n=dps, **options)
        r = r.evalf(n=dps, **options)
        # 返回数值近似后的 Circle 实例
        return self.func(pt, r, evaluate=False)

    # 定义一个 circumference 属性，表示圆的周长
    @property
    def circumference(self):
        """The circumference of the circle.

        Returns
        =======

        circumference : number or SymPy expression
            圆的周长，可以是数值或 SymPy 表达式

        Examples
        ========

        >>> from sympy import Point, Circle
        >>> c1 = Circle(Point(3, 4), 6)
        >>> c1.circumference
        12*pi

        """
        # 返回圆的周长公式
        return 2 * S.Pi * self.radius
    # 定义一个方法，用于返回圆的方程。
    def equation(self, x='x', y='y'):
        """The equation of the circle.

        Parameters
        ==========

        x : str or Symbol, optional
            Default value is 'x'.
        y : str or Symbol, optional
            Default value is 'y'.

        Returns
        =======

        equation : SymPy expression

        Examples
        ========

        >>> from sympy import Point, Circle
        >>> c1 = Circle(Point(0, 0), 5)
        >>> c1.equation()
        x**2 + y**2 - 25

        """
        # 将输入的参数 x 和 y 转换为 SymPy 的符号，确保是实数
        x = _symbol(x, real=True)
        y = _symbol(y, real=True)
        # 计算圆的方程中的两个平方项
        t1 = (x - self.center.x)**2
        t2 = (y - self.center.y)**2
        # 返回圆的方程：两个平方项相加，减去半径的平方
        return t1 + t2 - self.major**2

    # 定义一个方法，用于计算圆与另一个几何实体的交点。
    def intersection(self, o):
        """The intersection of this circle with another geometrical entity.

        Parameters
        ==========

        o : GeometryEntity

        Returns
        =======

        intersection : list of GeometryEntities

        Examples
        ========

        >>> from sympy import Point, Circle, Line, Ray
        >>> p1, p2, p3 = Point(0, 0), Point(5, 5), Point(6, 0)
        >>> p4 = Point(5, 0)
        >>> c1 = Circle(p1, 5)
        >>> c1.intersection(p2)
        []
        >>> c1.intersection(p4)
        [Point2D(5, 0)]
        >>> c1.intersection(Ray(p1, p2))
        [Point2D(5*sqrt(2)/2, 5*sqrt(2)/2)]
        >>> c1.intersection(Line(p2, p3))
        []

        """
        # 调用父类 Ellipse 的 intersection 方法，计算圆与另一个几何实体的交点
        return Ellipse.intersection(self, o)

    # 定义一个属性，用于返回圆的半径。
    @property
    def radius(self):
        """The radius of the circle.

        Returns
        =======

        radius : number or SymPy expression

        See Also
        ========

        Ellipse.major, Ellipse.minor, Ellipse.hradius, Ellipse.vradius

        Examples
        ========

        >>> from sympy import Point, Circle
        >>> c1 = Circle(Point(3, 4), 6)
        >>> c1.radius
        6

        """
        # 返回圆的半径，即第二个参数
        return self.args[1]

    # 定义一个方法，用于实现圆在某条直线上的反射。
    def reflect(self, line):
        """Override GeometryEntity.reflect since the radius
        is not a GeometryEntity.

        Examples
        ========

        >>> from sympy import Circle, Line
        >>> Circle((0, 1), 1).reflect(Line((0, 0), (1, 1)))
        Circle(Point2D(1, 0), -1)
        """
        # 获取圆心的反射后的坐标
        c = self.center
        c = c.reflect(line)
        # 返回反射后的圆，半径取反
        return self.func(c, -self.radius)
    def scale(self, x=1, y=1, pt=None):
        """
        Override GeometryEntity.scale since the radius
        is not a GeometryEntity.

        Examples
        ========

        >>> from sympy import Circle
        >>> Circle((0, 0), 1).scale(2, 2)
        Circle(Point2D(0, 0), 2)
        >>> Circle((0, 0), 1).scale(2, 4)
        Ellipse(Point2D(0, 0), 2, 4)
        """
        # 获取圆心坐标
        c = self.center
        # 如果有指定缩放的中心点 pt
        if pt:
            # 将 pt 转换为 Point2D 对象
            pt = Point(pt, dim=2)
            # 先将圆形移动到 pt 的相反位置，再进行缩放，最后移动回原位置
            return self.translate(*(-pt).args).scale(x, y).translate(*pt.args)
        # 对圆心进行缩放
        c = c.scale(x, y)
        # 分别取 x, y 的绝对值
        x, y = [abs(i) for i in (x, y)]
        # 如果 x 和 y 相等，则返回一个新的圆形
        if x == y:
            return self.func(c, x*self.radius)
        # 否则返回一个椭圆形，水平和垂直半径分别乘以 x 和 y
        h = v = self.radius
        return Ellipse(c, hradius=h*x, vradius=v*y)

    @property
    def vradius(self):
        """
        This Ellipse property is an alias for the Circle's radius.

        Whereas hradius, major and minor can use Ellipse's conventions,
        the vradius does not exist for a circle. It is always a positive
        value in order that the Circle, like Polygons, will have an
        area that can be positive or negative as determined by the sign
        of the hradius.

        Examples
        ========

        >>> from sympy import Point, Circle
        >>> c1 = Circle(Point(3, 4), 6)
        >>> c1.vradius
        6
        """
        # 返回圆的半径的绝对值作为垂直半径
        return abs(self.radius)
# 从当前目录下的 polygon 模块中导入 Polygon 和 Triangle 类
from .polygon import Polygon, Triangle
```