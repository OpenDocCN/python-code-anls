# `D:\src\scipysrc\sympy\sympy\geometry\line.py`

```
"""
Line-like geometrical entities.

Contains
========
LinearEntity
Line
Ray
Segment
LinearEntity2D
Line2D
Ray2D
Segment2D
LinearEntity3D
Line3D
Ray3D
Segment3D

"""

from sympy.core.containers import Tuple  # 导入Tuple类
from sympy.core.evalf import N  # 导入N函数，用于数值计算
from sympy.core.expr import Expr  # 导入Expr类，表示表达式的基类
from sympy.core.numbers import Rational, oo, Float  # 导入有理数、无穷大和浮点数
from sympy.core.relational import Eq  # 导入Eq类，表示等式
from sympy.core.singleton import S  # 导入S对象，表示单例
from sympy.core.sorting import ordered  # 导入ordered函数，用于排序
from sympy.core.symbol import _symbol, Dummy, uniquely_named_symbol  # 导入符号相关类和函数
from sympy.core.sympify import sympify  # 导入sympify函数，用于将字符串转换为表达式
from sympy.functions.elementary.piecewise import Piecewise  # 导入分段函数相关
from sympy.functions.elementary.trigonometric import (_pi_coeff, acos, tan, atan2)  # 导入三角函数相关
from .entity import GeometryEntity, GeometrySet  # 导入几何实体相关类
from .exceptions import GeometryError  # 导入几何异常类
from .point import Point, Point3D  # 导入点和三维点类
from .util import find, intersection  # 导入工具函数
from sympy.logic.boolalg import And  # 导入逻辑与运算
from sympy.matrices import Matrix  # 导入矩阵类
from sympy.sets.sets import Intersection  # 导入交集类
from sympy.simplify.simplify import simplify  # 导入简化函数
from sympy.solvers.solvers import solve  # 导入求解函数
from sympy.solvers.solveset import linear_coeffs  # 导入线性系数函数
from sympy.utilities.misc import Undecidable, filldedent  # 导入未决定的和格式化函数

import random  # 导入随机数模块


t, u = [Dummy('line_dummy') for i in range(2)]  # 创建两个虚拟符号对象

class LinearEntity(GeometrySet):
    """A base class for all linear entities (Line, Ray and Segment)
    in n-dimensional Euclidean space.

    Attributes
    ==========

    ambient_dimension
    direction
    length
    p1
    p2
    points

    Notes
    =====

    This is an abstract class and is not meant to be instantiated.

    See Also
    ========

    sympy.geometry.entity.GeometryEntity

    """
    def __new__(cls, p1, p2=None, **kwargs):
        p1, p2 = Point._normalize_dimension(p1, p2)  # 标准化点的维度
        if p1 == p2:
            # 如果两点相同，抛出值错误
            raise ValueError(
                "%s.__new__ requires two unique Points." % cls.__name__)
        if len(p1) != len(p2):
            # 如果两点维度不相同，抛出值错误
            raise ValueError(
                "%s.__new__ requires two Points of equal dimension." % cls.__name__)

        return GeometryEntity.__new__(cls, p1, p2, **kwargs)  # 调用基类的构造函数创建实例

    def __contains__(self, other):
        """Return a definitive answer or else raise an error if it cannot
        be determined that other is on the boundaries of self."""
        result = self.contains(other)  # 判断其他对象是否在本对象的边界内

        if result is not None:
            return result
        else:
            raise Undecidable(
                "Cannot decide whether '%s' contains '%s'" % (self, other))  # 如果无法确定，则抛出未决定的错误
    def _span_test(self, other):
        """Test whether the point `other` lies in the positive span of `self`.
        A point x is 'in front' of a point y if x.dot(y) >= 0.  Return
        -1 if `other` is behind `self.p1`, 0 if `other` is `self.p1` and
        and 1 if `other` is in front of `self.p1`."""
        # 检查是否与 self.p1 相同，如果相同返回 0
        if self.p1 == other:
            return 0

        # 计算点 other 相对于 self.p1 的相对位置
        rel_pos = other - self.p1
        # 获取线段的方向向量
        d = self.direction
        # 如果 other 在 self.p1 的正方向上，则返回 1；否则返回 -1
        if d.dot(rel_pos) > 0:
            return 1
        return -1

    @property
    def ambient_dimension(self):
        """A property method that returns the dimension of LinearEntity
        object.

        Parameters
        ==========

        p1 : LinearEntity

        Returns
        =======

        dimension : integer

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2 = Point(0, 0), Point(1, 1)
        >>> l1 = Line(p1, p2)
        >>> l1.ambient_dimension
        2

        >>> from sympy import Point, Line
        >>> p1, p2 = Point(0, 0, 0), Point(1, 1, 1)
        >>> l1 = Line(p1, p2)
        >>> l1.ambient_dimension
        3

        """
        # 返回 self.p1 的维度，即其坐标数量
        return len(self.p1)

    def angle_between(l1, l2):
        """Return the non-reflex angle formed by rays emanating from
        the origin with directions the same as the direction vectors
        of the linear entities.

        Parameters
        ==========

        l1 : LinearEntity
        l2 : LinearEntity

        Returns
        =======

        angle : angle in radians

        Notes
        =====

        From the dot product of vectors v1 and v2 it is known that:

            ``dot(v1, v2) = |v1|*|v2|*cos(A)``

        where A is the angle formed between the two vectors. We can
        get the directional vectors of the two lines and readily
        find the angle between the two using the above formula.

        See Also
        ========

        is_perpendicular, Ray2D.closing_angle

        Examples
        ========

        >>> from sympy import Line
        >>> e = Line((0, 0), (1, 0))
        >>> ne = Line((0, 0), (1, 1))
        >>> sw = Line((1, 1), (0, 0))
        >>> ne.angle_between(e)
        pi/4
        >>> sw.angle_between(e)
        3*pi/4

        To obtain the non-obtuse angle at the intersection of lines, use
        the ``smallest_angle_between`` method:

        >>> sw.smallest_angle_between(e)
        pi/4

        >>> from sympy import Point3D, Line3D
        >>> p1, p2, p3 = Point3D(0, 0, 0), Point3D(1, 1, 1), Point3D(-1, 2, 0)
        >>> l1, l2 = Line3D(p1, p2), Line3D(p2, p3)
        >>> l1.angle_between(l2)
        acos(-sqrt(2)/3)
        >>> l1.smallest_angle_between(l2)
        acos(sqrt(2)/3)
        """
        # 检查输入是否为 LinearEntity 对象，如果不是则抛出 TypeError
        if not isinstance(l1, LinearEntity) and not isinstance(l2, LinearEntity):
            raise TypeError('Must pass only LinearEntity objects')

        # 获取线段 l1 和 l2 的方向向量
        v1, v2 = l1.direction, l2.direction
        # 计算并返回两个向量之间的夹角（弧度）
        return acos(v1.dot(v2)/(abs(v1)*abs(v2)))
    # 计算两条线性实体之间形成的最小角度

    def smallest_angle_between(l1, l2):
        """Return the smallest angle formed at the intersection of the
        lines containing the linear entities.

        Parameters
        ==========

        l1 : LinearEntity
            第一条线性实体
        l2 : LinearEntity
            第二条线性实体

        Returns
        =======

        angle : angle in radians
            弧度表示的最小角度

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2, p3 = Point(0, 0), Point(0, 4), Point(2, -2)
        >>> l1, l2 = Line(p1, p2), Line(p1, p3)
        >>> l1.smallest_angle_between(l2)
        pi/4

        See Also
        ========

        angle_between, is_perpendicular, Ray2D.closing_angle
        """
        if not isinstance(l1, LinearEntity) and not isinstance(l2, LinearEntity):
            raise TypeError('Must pass only LinearEntity objects')

        # 获取两条线性实体的方向向量
        v1, v2 = l1.direction, l2.direction
        # 计算它们的夹角（使用点乘和模）
        return acos(abs(v1.dot(v2))/(abs(v1)*abs(v2)))

    def arbitrary_point(self, parameter='t'):
        """A parameterized point on the Line.

        Parameters
        ==========

        parameter : str, optional
            The name of the parameter which will be used for the parametric
            point. The default value is 't'. When this parameter is 0, the
            first point used to define the line will be returned, and when
            it is 1 the second point will be returned.

        Returns
        =======

        point : Point
            参数化表示的点

        Raises
        ======

        ValueError
            When ``parameter`` already appears in the Line's definition.

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2 = Point(1, 0), Point(5, 3)
        >>> l1 = Line(p1, p2)
        >>> l1.arbitrary_point()
        Point2D(4*t + 1, 3*t)
        >>> from sympy import Point3D, Line3D
        >>> p1, p2 = Point3D(1, 0, 0), Point3D(5, 3, 1)
        >>> l1 = Line3D(p1, p2)
        >>> l1.arbitrary_point()
        Point3D(4*t + 1, 3*t, t)

        """
        t = _symbol(parameter, real=True)
        # 如果参数已经在线的定义中出现，则引发错误
        if t.name in (f.name for f in self.free_symbols):
            raise ValueError(filldedent('''
                Symbol %s already appears in object
                and cannot be used as a parameter.
                ''' % t.name))
        # 返回参数化点的表达式
        # 在右侧乘以参数，以便将变量与点的坐标组合起来
        return self.p1 + (self.p2 - self.p1)*t

    @staticmethod
    def are_concurrent(*lines):
        """判断一组线性实体是否共点。

        如果所有线性实体在同一点相交，则它们共点。

        Parameters
        ==========

        lines
            一组线性实体。

        Returns
        =======

        True : 如果线性实体在一个点相交
        False : 否则

        See Also
        ========

        sympy.geometry.util.intersection

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2 = Point(0, 0), Point(3, 5)
        >>> p3, p4 = Point(-2, -2), Point(0, 2)
        >>> l1, l2, l3 = Line(p1, p2), Line(p1, p3), Line(p1, p4)
        >>> Line.are_concurrent(l1, l2, l3)
        True
        >>> l4 = Line(p2, p3)
        >>> Line.are_concurrent(l2, l3, l4)
        False
        >>> from sympy import Point3D, Line3D
        >>> p1, p2 = Point3D(0, 0, 0), Point3D(3, 5, 2)
        >>> p3, p4 = Point3D(-2, -2, -2), Point3D(0, 2, 1)
        >>> l1, l2, l3 = Line3D(p1, p2), Line3D(p1, p3), Line3D(p1, p4)
        >>> Line3D.are_concurrent(l1, l2, l3)
        True
        >>> l4 = Line3D(p2, p3)
        >>> Line3D.are_concurrent(l2, l3, l4)
        False

        """
        common_points = Intersection(*lines)  # 计算所有线性实体的交点
        if common_points.is_FiniteSet and len(common_points) == 1:  # 如果交点是有限集且只有一个点
            return True  # 返回共点
        return False  # 否则返回不共点

    def contains(self, other):
        """子类应实现此方法，如果 other 在 self 的边界上则返回 True；
           如果不在 self 的边界上则返回 False；
           如果无法确定则返回 None。"""
        raise NotImplementedError()

    @property
    def direction(self):
        """返回线性实体的方向向量。

        Returns
        =======

        p : 一个 Point；从原点到这一点的射线是 `self` 的方向向量

        Examples
        ========

        >>> from sympy import Line
        >>> a, b = (1, 1), (1, 3)
        >>> Line(a, b).direction
        Point2D(0, 2)
        >>> Line(b, a).direction
        Point2D(0, -2)

        这可以报告从原点到此点的距离为1：

        >>> Line(b, a).direction.unit
        Point2D(0, -1)

        See Also
        ========

        sympy.geometry.point.Point.unit

        """
        return self.p2 - self.p1
    def is_parallel(l1, l2):
        """判断两个线性实体是否平行的函数。

        Parameters
        ==========

        l1 : LinearEntity
            第一个线性实体对象。
        l2 : LinearEntity
            第二个线性实体对象。

        Returns
        =======

        bool
            如果 l1 和 l2 平行返回 True，否则返回 False。

        Raises
        ======

        TypeError
            如果 l1 或 l2 不是 LinearEntity 类型的对象时引发。

        See Also
        ========

        coefficients

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2 = Point(0, 0), Point(1, 1)
        >>> p3, p4 = Point(3, 4), Point(6, 7)
        >>> l1, l2 = Line(p1, p2), Line(p3, p4)
        >>> Line.is_parallel(l1, l2)
        True
        >>> p5 = Point(6, 6)
        >>> l3 = Line(p3, p5)
        >>> Line.is_parallel(l1, l3)
        False
        >>> from sympy import Point3D, Line3D
        >>> p1, p2 = Point3D(0, 0, 0), Point3D(3, 4, 5)
        >>> p3, p4 = Point3D(2, 1, 1), Point3D(8, 9, 11)
        >>> l1, l2 = Line3D(p1, p2), Line3D(p3, p4)
        >>> Line3D.is_parallel(l1, l2)
        True
        >>> p5 = Point3D(6, 6, 6)
        >>> l3 = Line3D(p3, p5)
        >>> Line3D.is_parallel(l1, l3)
        False

        """
        # 检查输入参数是否为 LinearEntity 类型，否则抛出异常
        if not isinstance(l1, LinearEntity) and not isinstance(l2, LinearEntity):
            raise TypeError('Must pass only LinearEntity objects')

        # 返回两个线性实体的方向向量是否是标量倍数关系
        return l1.direction.is_scalar_multiple(l2.direction)

    def is_perpendicular(l1, l2):
        """判断两个线性实体是否垂直的函数。

        Parameters
        ==========

        l1 : LinearEntity
            第一个线性实体对象。
        l2 : LinearEntity
            第二个线性实体对象。

        Returns
        =======

        bool
            如果 l1 和 l2 垂直返回 True，否则返回 False。

        Raises
        ======

        TypeError
            如果 l1 或 l2 不是 LinearEntity 类型的对象时引发。

        See Also
        ========

        coefficients

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2, p3 = Point(0, 0), Point(1, 1), Point(-1, 1)
        >>> l1, l2 = Line(p1, p2), Line(p1, p3)
        >>> l1.is_perpendicular(l2)
        True
        >>> p4 = Point(5, 3)
        >>> l3 = Line(p1, p4)
        >>> l1.is_perpendicular(l3)
        False
        >>> from sympy import Point3D, Line3D
        >>> p1, p2, p3 = Point3D(0, 0, 0), Point3D(1, 1, 1), Point3D(-1, 2, 0)
        >>> l1, l2 = Line3D(p1, p2), Line3D(p2, p3)
        >>> l1.is_perpendicular(l2)
        False
        >>> p4 = Point3D(5, 3, 7)
        >>> l3 = Line3D(p1, p4)
        >>> l1.is_perpendicular(l3)
        False

        """
        # 检查输入参数是否为 LinearEntity 类型，否则抛出异常
        if not isinstance(l1, LinearEntity) and not isinstance(l2, LinearEntity):
            raise TypeError('Must pass only LinearEntity objects')

        # 返回两个线性实体的方向向量是否互为垂直的判断
        return S.Zero.equals(l1.direction.dot(l2.direction))
    def is_similar(self, other):
        """
        Return True if self and other are contained in the same line.

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2, p3 = Point(0, 1), Point(3, 4), Point(2, 3)
        >>> l1 = Line(p1, p2)
        >>> l2 = Line(p1, p3)
        >>> l1.is_similar(l2)
        True
        """
        # 创建一个新的线对象 l，使用当前对象的两个定义点 self.p1 和 self.p2
        l = Line(self.p1, self.p2)
        # 调用新线对象 l 的 contains 方法，判断是否包含参数 other
        return l.contains(other)

    @property
    def length(self):
        """
        The length of the line.

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2 = Point(0, 0), Point(3, 5)
        >>> l1 = Line(p1, p2)
        >>> l1.length
        oo
        """
        # 返回线的长度，这里使用了 SymPy 的无穷大对象 S.Infinity 表示无限长
        return S.Infinity

    @property
    def p1(self):
        """The first defining point of a linear entity.

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2 = Point(0, 0), Point(5, 3)
        >>> l = Line(p1, p2)
        >>> l.p1
        Point2D(0, 0)

        """
        # 返回线的第一个定义点，这里假设 self.args 是一个包含定义点的元组
        return self.args[0]

    @property
    def p2(self):
        """The second defining point of a linear entity.

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2 = Point(0, 0), Point(5, 3)
        >>> l = Line(p1, p2)
        >>> l.p2
        Point2D(5, 3)

        """
        # 返回线的第二个定义点，假设 self.args 是一个包含定义点的元组
        return self.args[1]

    def parallel_line(self, p):
        """Create a new Line parallel to this linear entity which passes
        through the point `p`.

        Parameters
        ==========

        p : Point

        Returns
        =======

        line : Line

        See Also
        ========

        is_parallel

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2, p3 = Point(0, 0), Point(2, 3), Point(-2, 2)
        >>> l1 = Line(p1, p2)
        >>> l2 = l1.parallel_line(p3)
        >>> p3 in l2
        True
        >>> l1.is_parallel(l2)
        True
        >>> from sympy import Point3D, Line3D
        >>> p1, p2, p3 = Point3D(0, 0, 0), Point3D(2, 3, 4), Point3D(-2, 2, 0)
        >>> l1 = Line3D(p1, p2)
        >>> l2 = l1.parallel_line(p3)
        >>> p3 in l2
        True
        >>> l1.is_parallel(l2)
        True

        """
        # 将参数 p 转换为一个 Point 对象，维度与当前线对象的维度相同
        p = Point(p, dim=self.ambient_dimension)
        # 返回一个与当前线平行且经过点 p 的新 Line 对象
        return Line(p, p + self.direction)
    def perpendicular_line(self, p):
        """创建一个垂直于当前线性实体并通过点 `p` 的新线。

        Parameters
        ==========

        p : Point
            通过该点的直线必须垂直于当前线性实体。

        Returns
        =======

        line : Line
            返回一个新的 Line 对象，表示垂直线。

        See Also
        ========

        sympy.geometry.line.LinearEntity.is_perpendicular, perpendicular_segment

        Examples
        ========

        >>> from sympy import Point3D, Line3D
        >>> p1, p2, p3 = Point3D(0, 0, 0), Point3D(2, 3, 4), Point3D(-2, 2, 0)
        >>> L = Line3D(p1, p2)
        >>> P = L.perpendicular_line(p3); P
        Line3D(Point3D(-2, 2, 0), Point3D(4/29, 6/29, 8/29))
        >>> L.is_perpendicular(P)
        True

        在三维空间中，用于定义线的第一个点是要通过的垂直线的点；
        第二个点（任意）包含在给定的线中：

        >>> P.p2 in L
        True
        """
        p = Point(p, dim=self.ambient_dimension)
        # 如果点 p 已经在当前线性实体上，则找到 p 在当前线性实体上的垂直方向点
        if p in self:
            p = p + self.direction.orthogonal_direction
        # 返回通过点 p 并垂直于当前线性实体的新 Line 对象
        return Line(p, self.projection(p))

    def perpendicular_segment(self, p):
        """创建从点 `p` 到当前线的垂直线段。

        线段的端点分别是 `p` 和包含当前线的最近点。（如果当前线不是一条线，则该点可能不在当前线上。）

        Parameters
        ==========

        p : Point
            与当前线的垂直线段的起点。

        Returns
        =======

        segment : Segment
            返回一个新的 Segment 对象，表示垂直线段。

        Notes
        =====

        如果点 `p` 在当前线性实体上，则直接返回 `p` 本身。

        See Also
        ========

        perpendicular_line

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2, p3 = Point(0, 0), Point(1, 1), Point(0, 2)
        >>> l1 = Line(p1, p2)
        >>> s1 = l1.perpendicular_segment(p3)
        >>> l1.is_perpendicular(s1)
        True
        >>> p3 in s1
        True
        >>> l1.perpendicular_segment(Point(4, 0))
        Segment2D(Point2D(4, 0), Point2D(2, 2))
        >>> from sympy import Point3D, Line3D
        >>> p1, p2, p3 = Point3D(0, 0, 0), Point3D(1, 1, 1), Point3D(0, 2, 0)
        >>> l1 = Line3D(p1, p2)
        >>> s1 = l1.perpendicular_segment(p3)
        >>> l1.is_perpendicular(s1)
        True
        >>> p3 in s1
        True
        >>> l1.perpendicular_segment(Point3D(4, 0, 0))
        Segment3D(Point3D(4, 0, 0), Point3D(4/3, 4/3, 4/3))

        """
        p = Point(p, dim=self.ambient_dimension)
        # 如果点 p 在当前线性实体上，则直接返回 p
        if p in self:
            return p
        # 否则，找到通过点 p 并垂直于当前线性实体的线，并计算其与当前线的交点
        l = self.perpendicular_line(p)
        # 交点应该是唯一的，因此解压缩单例
        p2, = Intersection(Line(self.p1, self.p2), l)

        # 返回从点 p 到交点 p2 的新的线段对象
        return Segment(p, p2)
    def points(self):
        """The two points used to define this linear entity.

        Returns
        =======

        points : tuple of Points
            返回此线性实体定义所用的两个点的元组

        See Also
        ========

        sympy.geometry.point.Point
            参考 sympy 中的点类

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2 = Point(0, 0), Point(5, 11)
        >>> l1 = Line(p1, p2)
        >>> l1.points
        (Point2D(0, 0), Point2D(5, 11))
        示例：创建线段对象 l1，获取其定义的两个点的元组

        """
        return (self.p1, self.p2)

    def random_point(self, seed=None):
        """A random point on a LinearEntity.

        Returns
        =======

        point : Point
            返回线性实体上的一个随机点

        See Also
        ========

        sympy.geometry.point.Point
            参考 sympy 中的点类

        Examples
        ========

        >>> from sympy import Point, Line, Ray, Segment
        >>> p1, p2 = Point(0, 0), Point(5, 3)
        >>> line = Line(p1, p2)
        >>> r = line.random_point(seed=42)  # seed value is optional
        >>> r.n(3)
        Point2D(-0.72, -0.432)
        >>> r in line
        True
        >>> Ray(p1, p2).random_point(seed=42).n(3)
        Point2D(0.72, 0.432)
        >>> Segment(p1, p2).random_point(seed=42).n(3)
        Point2D(3.2, 1.92)
        示例：创建线段、射线或线对象，获取其上的随机点，并展示使用种子值来控制随机性

        """
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
        pt = self.arbitrary_point(t)  # 获取线性实体上的一个任意点
        if isinstance(self, Ray):
            v = abs(rng.gauss(0, 1))  # 如果是射线，使用高斯分布生成随机值
        elif isinstance(self, Segment):
            v = rng.random()  # 如果是线段，生成 [0, 1) 范围内的随机值
        elif isinstance(self, Line):
            v = rng.gauss(0, 1)  # 如果是直线，使用高斯分布生成随机值
        else:
            raise NotImplementedError('unhandled line type')
        return pt.subs(t, Rational(v))  # 将随机值代入任意点表达式，返回具体点坐标
    def bisectors(self, other):
        """Returns the perpendicular lines which pass through the intersections
        of self and other that are in the same plane.

        Parameters
        ==========

        line : Line3D
            Another Line3D object with which to find intersections.

        Returns
        =======

        list: two Line instances
            List containing two Line3D objects representing the perpendicular bisectors.

        Examples
        ========

        >>> from sympy import Point3D, Line3D
        >>> r1 = Line3D(Point3D(0, 0, 0), Point3D(1, 0, 0))
        >>> r2 = Line3D(Point3D(0, 0, 0), Point3D(0, 1, 0))
        >>> r1.bisectors(r2)
        [Line3D(Point3D(0, 0, 0), Point3D(1, 1, 0)), Line3D(Point3D(0, 0, 0), Point3D(1, -1, 0))]

        """
        # Check if the 'other' object is an instance of LinearEntity
        if not isinstance(other, LinearEntity):
            raise GeometryError("Expecting LinearEntity, not %s" % other)

        # Assign self and other to l1 and l2 respectively
        l1, l2 = self, other

        # Ensure dimensions match or normalize them if they don't
        if l1.p1.ambient_dimension != l2.p1.ambient_dimension:
            # Swap l1 and l2 if l1 is Line2D to maintain consistency
            if isinstance(l1, Line2D):
                l1, l2 = l2, l1
            # Normalize dimensions of points p1 and p2
            _, p1 = Point._normalize_dimension(l1.p1, l2.p1, on_morph='ignore')
            _, p2 = Point._normalize_dimension(l1.p2, l2.p2, on_morph='ignore')
            # Create a new Line instance l2 with normalized points
            l2 = Line(p1, p2)

        # Find the intersection point of l1 and l2
        point = intersection(l1, l2)

        # Handle different cases based on the intersection result
        if not point:
            # Raise an error if there's no intersection point
            raise GeometryError("The lines do not intersect")
        else:
            # Extract the first point from the intersection result
            pt = point[0]
            # Check if the intersection point is a Line object
            if isinstance(pt, Line):
                # Return self if the intersection forms a line (lines are coincident)
                return [self]

        # Compute unit directions of l1 and l2
        d1 = l1.direction.unit
        d2 = l2.direction.unit

        # Create perpendicular bisectors bis1 and bis2 passing through pt
        bis1 = Line(pt, pt + d1 + d2)
        bis2 = Line(pt, pt + d1 - d2)

        # Return the list of perpendicular bisectors
        return [bis1, bis2]
# 表示一个空间中的无限直线的类 Line，继承自 LinearEntity

"""An infinite line in space.

A 2D line is declared with two distinct points, point and slope, or
an equation. A 3D line may be defined with a point and a direction ratio.

Parameters
==========

p1 : Point
p2 : Point
slope : SymPy expression
direction_ratio : list
equation : equation of a line

Notes
=====

`Line` will automatically subclass to `Line2D` or `Line3D` based
on the dimension of `p1`.  The `slope` argument is only relevant
for `Line2D` and the `direction_ratio` argument is only relevant
for `Line3D`.

The order of the points will define the direction of the line
which is used when calculating the angle between lines.

See Also
========

sympy.geometry.point.Point
sympy.geometry.line.Line2D
sympy.geometry.line.Line3D

Examples
========

>>> from sympy import Line, Segment, Point, Eq
>>> from sympy.abc import x, y, a, b

>>> L = Line(Point(2,3), Point(3,5))
>>> L
Line2D(Point2D(2, 3), Point2D(3, 5))
>>> L.points
(Point2D(2, 3), Point2D(3, 5))
>>> L.equation()
-2*x + y + 1
>>> L.coefficients
(-2, 1, 1)

Instantiate with keyword ``slope``:

>>> Line(Point(0, 0), slope=0)
Line2D(Point2D(0, 0), Point2D(1, 0))

Instantiate with another linear object

>>> s = Segment((0, 0), (0, 1))
>>> Line(s).equation()
x

The line corresponding to an equation in the for `ax + by + c = 0`,
can be entered:

>>> Line(3*x + y + 18)
Line2D(Point2D(0, -18), Point2D(1, -21))

If `x` or `y` has a different name, then they can be specified, too,
as a string (to match the name) or symbol:

>>> Line(Eq(3*a + b, -18), x='a', y=b)
Line2D(Point2D(0, -18), Point2D(1, -21))
"""
    def __new__(cls, *args, **kwargs):
        # 如果只有一个参数，并且该参数是 Expr 或 Eq 类型的实例
        if len(args) == 1 and isinstance(args[0], (Expr, Eq)):
            # 在 args 中寻找唯一命名的符号 '?'，用于缺失的参数
            missing = uniquely_named_symbol('?', args)
            # 如果 kwargs 为空，则设置默认值 'x' 和 'y'；否则从 kwargs 中获取 'x' 和 'y' 的值，如果不存在则使用 missing
            if not kwargs:
                x = 'x'
                y = 'y'
            else:
                x = kwargs.pop('x', missing)
                y = kwargs.pop('y', missing)
            # 如果 kwargs 非空，则抛出异常，因为只能接受 'x' 和 'y' 作为关键字参数
            if kwargs:
                raise ValueError('expecting only x and y as keywords')

            # 获取方程对象
            equation = args[0]
            # 如果方程是 Eq 类型，则将其转换为等式的左右两边的差
            if isinstance(equation, Eq):
                equation = equation.lhs - equation.rhs

            # 定义一个函数，用于在方程中查找变量 x 或 y，找不到则返回 missing
            def find_or_missing(x):
                try:
                    return find(x, equation)
                except ValueError:
                    return missing
            # 查找变量 x 和 y 在方程中的位置
            x = find_or_missing(x)
            y = find_or_missing(y)

            # 计算方程的线性系数
            a, b, c = linear_coeffs(equation, x, y)

            # 根据系数返回相应的 Line 对象
            if b:
                return Line((0, -c/b), slope=-a/b)
            if a:
                return Line((-c/a, 0), slope=oo)

            # 如果无法计算出合适的直线，则抛出异常
            raise ValueError('not found in equation: %s' % (set('xy') - {x, y}))

        else:
            # 处理参数不符合上述条件的情况
            if len(args) > 0:
                p1 = args[0]
                # 如果有第二个参数，则赋值给 p2；否则 p2 为 None
                if len(args) > 1:
                    p2 = args[1]
                else:
                    p2 = None

                # 如果 p1 是 LinearEntity 类型的实例
                if isinstance(p1, LinearEntity):
                    # 如果 p2 不为 None，则抛出异常，因为 p1 是 LinearEntity 类型时 p2 必须为 None
                    if p2:
                        raise ValueError('If p1 is a LinearEntity, p2 must be None.')
                    # 计算 p1 的维度
                    dim = len(p1.p1)
                else:
                    # 否则将 p1 转换为 Point 对象，并获取其维度
                    p1 = Point(p1)
                    dim = len(p1)
                    # 如果 p2 不为 None，并且 p2 是 Point 类型或者 p2 的维度与 p1 不一致，则将 p2 转换为 Point 对象
                    if p2 is not None or isinstance(p2, Point) and p2.ambient_dimension != dim:
                        p2 = Point(p2)

                # 根据维度创建相应的 Line 对象：二维或三维
                if dim == 2:
                    return Line2D(p1, p2, **kwargs)
                elif dim == 3:
                    return Line3D(p1, p2, **kwargs)
                # 返回 LinearEntity 类的新实例
                return LinearEntity.__new__(cls, p1, p2, **kwargs)

    def contains(self, other):
        """
        Return True if `other` is on this Line, or False otherwise.

        Examples
        ========

        >>> from sympy import Line,Point
        >>> p1, p2 = Point(0, 1), Point(3, 4)
        >>> l = Line(p1, p2)
        >>> l.contains(p1)
        True
        >>> l.contains((0, 1))
        True
        >>> l.contains((0, 0))
        False
        >>> a = (0, 0, 0)
        >>> b = (1, 1, 1)
        >>> c = (2, 2, 2)
        >>> l1 = Line(a, b)
        >>> l2 = Line(b, a)
        >>> l1 == l2
        False
        >>> l1 in l2
        True

        """
        # 如果 other 不是 GeometryEntity 类型的实例，则将其转换为 Point 对象
        if not isinstance(other, GeometryEntity):
            other = Point(other, dim=self.ambient_dimension)
        # 如果 other 是 Point 对象，则检查其是否与本 Line 共线
        if isinstance(other, Point):
            return Point.is_collinear(other, self.p1, self.p2)
        # 如果 other 是 LinearEntity 对象，则检查其与本 Line 是否共线
        if isinstance(other, LinearEntity):
            return Point.is_collinear(self.p1, self.p2, other.p1, other.p2)
        # 其他情况返回 False
        return False
    def distance(self, other):
        """
        Finds the shortest distance between a line and a point.

        Raises
        ======

        NotImplementedError is raised if `other` is not a Point

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2 = Point(0, 0), Point(1, 1)
        >>> s = Line(p1, p2)
        >>> s.distance(Point(-1, 1))
        sqrt(2)
        >>> s.distance((-1, 2))
        3*sqrt(2)/2
        >>> p1, p2 = Point(0, 0, 0), Point(1, 1, 1)
        >>> s = Line(p1, p2)
        >>> s.distance(Point(-1, 1, 1))
        2*sqrt(6)/3
        >>> s.distance((-1, 1, 1))
        2*sqrt(6)/3

        """
        # 检查参数 `other` 是否为 GeometryEntity 的实例，如果不是，则转换为 Point
        if not isinstance(other, GeometryEntity):
            other = Point(other, dim=self.ambient_dimension)
        # 如果线段包含点 `other`，则返回零距离
        if self.contains(other):
            return S.Zero
        # 否则返回线段与点 `other` 之间的垂直线段的长度
        return self.perpendicular_segment(other).length

    def equals(self, other):
        """Returns True if self and other are the same mathematical entities"""
        # 检查 `other` 是否为 Line 类的实例，如果不是，则返回 False
        if not isinstance(other, Line):
            return False
        # 判断两条线段是否共线
        return Point.is_collinear(self.p1, other.p1, self.p2, other.p2)

    def plot_interval(self, parameter='t'):
        """The plot interval for the default geometric plot of line. Gives
        values that will produce a line that is +/- 5 units long (where a
        unit is the distance between the two points that define the line).

        Parameters
        ==========

        parameter : str, optional
            Default value is 't'.

        Returns
        =======

        plot_interval : list (plot interval)
            [parameter, lower_bound, upper_bound]

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2 = Point(0, 0), Point(5, 3)
        >>> l1 = Line(p1, p2)
        >>> l1.plot_interval()
        [t, -5, 5]

        """
        # 创建一个符号变量 `t`，默认为实数
        t = _symbol(parameter, real=True)
        # 返回一个列表，表示默认几何图形绘制线段的区间，长度为 +/- 5 个单位
        return [t, -5, 5]
    """Represents a Ray, a semi-line in space defined by a source point and direction.

    Parameters
    ==========

    p1 : Point
        The source point of the Ray.
    p2 : Point or radian value, optional
        Determines the direction of the Ray:
        - If Point, specifies another point through which the Ray passes.
        - If radian value, interpreted as angle in radians (ccw positive).

    Attributes
    ==========

    source : Point
        The source point of the Ray.

    See Also
    ========

    sympy.geometry.line.Ray2D
    sympy.geometry.line.Ray3D
    sympy.geometry.point.Point
    sympy.geometry.line.Line

    Notes
    =====

    - Automatically subclasses to Ray2D or Ray3D based on `p1` dimension.
    - Provides various geometric properties like direction, slope.

    Examples
    ========

    >>> from sympy import Ray, Point, pi
    >>> r = Ray(Point(2, 3), Point(3, 5))
    >>> r
    Ray2D(Point2D(2, 3), Point2D(3, 5))
    >>> r.points
    (Point2D(2, 3), Point2D(3, 5))
    >>> r.source
    Point2D(2, 3)
    >>> r.xdirection
    oo
    >>> r.ydirection
    oo
    >>> r.slope
    2
    >>> Ray(Point(0, 0), angle=pi/4).slope
    1

    """
    def __new__(cls, p1, p2=None, **kwargs):
        # Ensure p1 is a Point object
        p1 = Point(p1)
        if p2 is not None:
            # Normalize dimensionality of points p1 and p2
            p1, p2 = Point._normalize_dimension(p1, Point(p2))
        # Determine dimensionality of p1
        dim = len(p1)

        # Return Ray2D or Ray3D instance based on dimensionality of p1
        if dim == 2:
            return Ray2D(p1, p2, **kwargs)
        elif dim == 3:
            return Ray3D(p1, p2, **kwargs)
        # If dimension is not 2 or 3, create a LinearEntity instance
        return LinearEntity.__new__(cls, p1, p2, **kwargs)

    def _svg(self, scale_factor=1., fill_color="#66cc99"):
        """Returns SVG path element representation for the Ray.

        Parameters
        ==========

        scale_factor : float
            Scale factor for SVG stroke-width. Default is 1.
        fill_color : str, optional
            Hex string representing fill color. Default is "#66cc99".
        """
        # Convert points p1 and p2 to numeric values
        verts = (N(self.p1), N(self.p2))
        # Extract x, y coordinates of vertices
        coords = ["{},{}".format(p.x, p.y) for p in verts]
        # Construct SVG path string
        path = "M {} L {}".format(coords[0], " L ".join(coords[1:]))

        # Return SVG path element with defined attributes
        return (
            '<path fill-rule="evenodd" fill="{2}" stroke="#555555" '
            'stroke-width="{0}" opacity="0.6" d="{1}" '
            'marker-start="url(#markerCircle)" marker-end="url(#markerArrow)"/>'
        ).format(2.*scale_factor, path, fill_color)
    def contains(self, other):
        """
        判断其他几何实体是否包含在这条射线中。

        示例
        ========

        >>> from sympy import Ray, Point, Segment
        >>> p1, p2 = Point(0, 0), Point(4, 4)
        >>> r = Ray(p1, p2)
        >>> r.contains(p1)
        True
        >>> r.contains((1, 1))
        True
        >>> r.contains((1, 3))
        False
        >>> s = Segment((1, 1), (2, 2))
        >>> r.contains(s)
        True
        >>> s = Segment((1, 2), (2, 5))
        >>> r.contains(s)
        False
        >>> r1 = Ray((2, 2), (3, 3))
        >>> r.contains(r1)
        True
        >>> r1 = Ray((2, 2), (3, 5))
        >>> r.contains(r1)
        False
        """
        if not isinstance(other, GeometryEntity):
            other = Point(other, dim=self.ambient_dimension)
        if isinstance(other, Point):
            if Point.is_collinear(self.p1, self.p2, other):
                # 如果点在射线的方向上，我们的方向向量点乘射线的方向向量应该非负
                return bool((self.p2 - self.p1).dot(other - self.p1) >= S.Zero)
            return False
        elif isinstance(other, Ray):
            if Point.is_collinear(self.p1, self.p2, other.p1, other.p2):
                return bool((self.p2 - self.p1).dot(other.p2 - other.p1) > S.Zero)
            return False
        elif isinstance(other, Segment):
            return other.p1 in self and other.p2 in self

        # 没有其他已知的实体可以包含在射线中
        return False

    def distance(self, other):
        """
        计算射线与点之间的最短距离。

        Raises
        ======

        如果 `other` 不是一个点，则引发 NotImplementedError

        示例
        ========

        >>> from sympy import Point, Ray
        >>> p1, p2 = Point(0, 0), Point(1, 1)
        >>> s = Ray(p1, p2)
        >>> s.distance(Point(-1, -1))
        sqrt(2)
        >>> s.distance((-1, 2))
        3*sqrt(2)/2
        >>> p1, p2 = Point(0, 0, 0), Point(1, 1, 2)
        >>> s = Ray(p1, p2)
        >>> s
        Ray3D(Point3D(0, 0, 0), Point3D(1, 1, 2))
        >>> s.distance(Point(-1, -1, 2))
        4*sqrt(3)/3
        >>> s.distance((-1, -1, 2))
        4*sqrt(3)/3

        """
        if not isinstance(other, GeometryEntity):
            other = Point(other, dim=self.ambient_dimension)
        if self.contains(other):
            return S.Zero

        proj = Line(self.p1, self.p2).projection(other)
        if self.contains(proj):
            return abs(other - proj)
        else:
            return abs(other - self.source)

    def equals(self, other):
        """
        如果 self 和 other 是相同的数学实体，则返回 True。
        """
        if not isinstance(other, Ray):
            return False
        return self.source == other.source and other.p2 in self
    def plot_interval(self, parameter='t'):
        """
        The plot interval for the default geometric plot of the Ray. Gives
        values that will produce a ray that is 10 units long (where a unit is
        the distance between the two points that define the ray).

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

        >>> from sympy import Ray, pi
        >>> r = Ray((0, 0), angle=pi/4)
        >>> r.plot_interval()
        [t, 0, 10]

        """
        # 创建一个符号变量 t，确保其为实数
        t = _symbol(parameter, real=True)
        # 返回包含参数名称、下界0和上界10的列表，描述射线的绘图区间
        return [t, 0, 10]

    @property
    def source(self):
        """
        The point from which the ray emanates.

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Point, Ray
        >>> p1, p2 = Point(0, 0), Point(4, 1)
        >>> r1 = Ray(p1, p2)
        >>> r1.source
        Point2D(0, 0)
        >>> p1, p2 = Point(0, 0, 0), Point(4, 1, 5)
        >>> r1 = Ray(p2, p1)
        >>> r1.source
        Point3D(4, 1, 5)

        """
        # 返回射线的起点，即定义射线的第一个点
        return self.p1
# Segment 类表示空间中的线段，继承自 LinearEntity 类
class Segment(LinearEntity):
    """A line segment in space.

    Parameters
    ==========

    p1 : Point
        起点坐标
    p2 : Point
        终点坐标

    Attributes
    ==========

    length : number or SymPy expression
        线段长度，可以是数值或 SymPy 表达式
    midpoint : Point
        线段的中点坐标

    See Also
    ========

    sympy.geometry.line.Segment2D
    sympy.geometry.line.Segment3D
    sympy.geometry.point.Point
    sympy.geometry.line.Line

    Notes
    =====

    If 2D or 3D points are used to define `Segment`, it will
    be automatically subclassed to `Segment2D` or `Segment3D`.

    Examples
    ========

    >>> from sympy import Point, Segment
    >>> Segment((1, 0), (1, 1)) # tuples are interpreted as pts
    Segment2D(Point2D(1, 0), Point2D(1, 1))
    >>> s = Segment(Point(4, 3), Point(1, 1))
    >>> s.points
    (Point2D(4, 3), Point2D(1, 1))
    >>> s.slope
    2/3
    >>> s.length
    sqrt(13)
    >>> s.midpoint
    Point2D(5/2, 2)
    >>> Segment((1, 0, 0), (1, 1, 1)) # tuples are interpreted as pts
    Segment3D(Point3D(1, 0, 0), Point3D(1, 1, 1))
    >>> s = Segment(Point(4, 3, 9), Point(1, 1, 7)); s
    Segment3D(Point3D(4, 3, 9), Point3D(1, 1, 7))
    >>> s.points
    (Point3D(4, 3, 9), Point3D(1, 1, 7))
    >>> s.length
    sqrt(17)
    >>> s.midpoint
    Point3D(5/2, 2, 8)

    """
    
    def __new__(cls, p1, p2, **kwargs):
        # 规范化点的维度，确保 p1 和 p2 是 Point 对象
        p1, p2 = Point._normalize_dimension(Point(p1), Point(p2))
        dim = len(p1)

        # 根据点的维度选择返回的子类对象
        if dim == 2:
            return Segment2D(p1, p2, **kwargs)
        elif dim == 3:
            return Segment3D(p1, p2, **kwargs)
        
        # 默认情况下返回 LinearEntity 的新实例
        return LinearEntity.__new__(cls, p1, p2, **kwargs)
    def contains(self, other):
        """
        Is the other GeometryEntity contained within this Segment?

        Examples
        ========

        >>> from sympy import Point, Segment
        >>> p1, p2 = Point(0, 1), Point(3, 4)
        >>> s = Segment(p1, p2)
        >>> s2 = Segment(p2, p1)
        >>> s.contains(s2)
        True
        >>> from sympy import Point3D, Segment3D
        >>> p1, p2 = Point3D(0, 1, 1), Point3D(3, 4, 5)
        >>> s = Segment3D(p1, p2)
        >>> s2 = Segment3D(p2, p1)
        >>> s.contains(s2)
        True
        >>> s.contains((p1 + p2)/2)
        True
        """
        # 如果other不是GeometryEntity的实例，则将其视为Point对象
        if not isinstance(other, GeometryEntity):
            other = Point(other, dim=self.ambient_dimension)
        
        # 如果other是Point对象，并且与self.p1、self.p2共线
        if isinstance(other, Point):
            if Point.is_collinear(other, self.p1, self.p2):
                if isinstance(self, Segment2D):
                    # 如果共线并且在段的边界框内，则必须在段上
                    vert = (1/self.slope).equals(0)
                    if vert is False:
                        isin = (self.p1.x - other.x)*(self.p2.x - other.x) <= 0
                        if isin in (True, False):
                            return isin
                    if vert is True:
                        isin = (self.p1.y - other.y)*(self.p2.y - other.y) <= 0
                        if isin in (True, False):
                            return isin
                # 使用三角不等式判断
                d1, d2 = other - self.p1, other - self.p2
                d = self.p2 - self.p1
                # 如果不调用simplify，SymPy无法确定像(a+b)*(a/2+b/2)这样的表达式始终为非负数。
                # 如果无法确定，则引发Undecidable错误
                try:
                    # 三角不等式指出 |d1|+|d2| >= |d| 并且只有当other位于线段上时是严格的相等
                    return bool(simplify(Eq(abs(d1) + abs(d2) - abs(d), 0)))
                except TypeError:
                    raise Undecidable("Cannot determine if {} is in {}".format(other, self))
        
        # 如果other是Segment对象，则检查其端点是否都在self中
        if isinstance(other, Segment):
            return other.p1 in self and other.p2 in self

        # 默认情况下，返回False
        return False

    def equals(self, other):
        """Returns True if self and other are the same mathematical entities"""
        # 返回True，如果self和other是相同的数学实体（即同一类型的对象且参数相同）
        return isinstance(other, self.func) and list(
            ordered(self.args)) == list(ordered(other.args))
    def distance(self, other):
        """
        Finds the shortest distance between a line segment and a point.

        Raises
        ======

        NotImplementedError is raised if `other` is not a Point

        Examples
        ========

        >>> from sympy import Point, Segment
        >>> p1, p2 = Point(0, 1), Point(3, 4)
        >>> s = Segment(p1, p2)
        >>> s.distance(Point(10, 15))
        sqrt(170)
        >>> s.distance((0, 12))
        sqrt(73)
        >>> from sympy import Point3D, Segment3D
        >>> p1, p2 = Point3D(0, 0, 3), Point3D(1, 1, 4)
        >>> s = Segment3D(p1, p2)
        >>> s.distance(Point3D(10, 15, 12))
        sqrt(341)
        >>> s.distance((10, 15, 12))
        sqrt(341)
        """
        # 检查 `other` 是否为 GeometryEntity 类型，如果不是，则转换为 Point 类型
        if not isinstance(other, GeometryEntity):
            other = Point(other, dim=self.ambient_dimension)
        
        # 如果 `other` 是 Point 类型
        if isinstance(other, Point):
            # 计算 other 到 self.p1 和 self.p2 的向量
            vp1 = other - self.p1
            vp2 = other - self.p2

            # 计算两个向量与线段方向向量的点积的符号
            dot_prod_sign_1 = self.direction.dot(vp1) >= 0
            dot_prod_sign_2 = self.direction.dot(vp2) <= 0

            # 根据点积的符号判断 shortest distance 的计算方式
            if dot_prod_sign_1 and dot_prod_sign_2:
                return Line(self.p1, self.p2).distance(other)
            if dot_prod_sign_1 and not dot_prod_sign_2:
                return abs(vp2)
            if not dot_prod_sign_1 and dot_prod_sign_2:
                return abs(vp1)
        
        # 如果其他情况下（即 `other` 既不是 GeometryEntity 也不是 Point），抛出 NotImplementedError 异常
        raise NotImplementedError()

    @property
    def length(self):
        """The length of the line segment.

        See Also
        ========

        sympy.geometry.point.Point.distance

        Examples
        ========

        >>> from sympy import Point, Segment
        >>> p1, p2 = Point(0, 0), Point(4, 3)
        >>> s1 = Segment(p1, p2)
        >>> s1.length
        5
        >>> from sympy import Point3D, Segment3D
        >>> p1, p2 = Point3D(0, 0, 0), Point3D(4, 3, 3)
        >>> s1 = Segment3D(p1, p2)
        >>> s1.length
        sqrt(34)

        """
        # 返回线段的长度，调用 Point 类的 distance 方法
        return Point.distance(self.p1, self.p2)

    @property
    def midpoint(self):
        """The midpoint of the line segment.

        See Also
        ========

        sympy.geometry.point.Point.midpoint

        Examples
        ========

        >>> from sympy import Point, Segment
        >>> p1, p2 = Point(0, 0), Point(4, 3)
        >>> s1 = Segment(p1, p2)
        >>> s1.midpoint
        Point2D(2, 3/2)
        >>> from sympy import Point3D, Segment3D
        >>> p1, p2 = Point3D(0, 0, 0), Point3D(4, 3, 3)
        >>> s1 = Segment3D(p1, p2)
        >>> s1.midpoint
        Point3D(2, 3/2, 3/2)

        """
        # 返回线段的中点，调用 Point 类的 midpoint 方法
        return Point.midpoint(self.p1, self.p2)
    def perpendicular_bisector(self, p=None):
        """
        The perpendicular bisector of this segment.

        If no point is specified or the point specified is not on the
        bisector then the bisector is returned as a Line. Otherwise a
        Segment is returned that joins the point specified and the
        intersection of the bisector and the segment.

        Parameters
        ==========

        p : Point, optional
            A point to check if it lies on the perpendicular bisector.

        Returns
        =======

        bisector : Line or Segment
            If no point is specified or the point is not on the bisector, returns a Line.
            If a point is specified and it lies on the bisector, returns a Segment.

        See Also
        ========

        LinearEntity.perpendicular_segment

        Examples
        ========

        >>> from sympy import Point, Segment
        >>> p1, p2, p3 = Point(0, 0), Point(6, 6), Point(5, 1)
        >>> s1 = Segment(p1, p2)
        >>> s1.perpendicular_bisector()
        Line2D(Point2D(3, 3), Point2D(-3, 9))

        >>> s1.perpendicular_bisector(p3)
        Segment2D(Point2D(5, 1), Point2D(3, 3))

        """
        # Calculate the perpendicular line passing through the midpoint of the segment
        l = self.perpendicular_line(self.midpoint)
        
        # Check if a specific point lies on the perpendicular bisector
        if p is not None:
            p2 = Point(p, dim=self.ambient_dimension)
            if p2 in l:
                # If the point lies on the bisector, return the segment from that point to the midpoint
                return Segment(p2, self.midpoint)
        
        # Return the perpendicular bisector line if no point is specified or the point does not lie on it
        return l

    def plot_interval(self, parameter='t'):
        """
        The plot interval for the default geometric plot of the Segment gives
        values that will produce the full segment in a plot.

        Parameters
        ==========

        parameter : str, optional
            The parameter name used for plotting. Default is 't'.

        Returns
        =======

        plot_interval : list
            A list [parameter, lower_bound, upper_bound] defining the interval for plotting.

        Examples
        ========

        >>> from sympy import Point, Segment
        >>> p1, p2 = Point(0, 0), Point(5, 3)
        >>> s1 = Segment(p1, p2)
        >>> s1.plot_interval()
        [t, 0, 1]

        """
        # Define the symbolic parameter for plotting
        t = _symbol(parameter, real=True)
        
        # Return the plot interval as [parameter, lower_bound, upper_bound]
        return [t, 0, 1]
class LinearEntity2D(LinearEntity):
    """A base class for all linear entities (line, ray and segment)
    in a 2-dimensional Euclidean space.

    Attributes
    ==========

    p1
        The first endpoint or defining point of the linear entity.
    p2
        The second endpoint or defining point of the linear entity.
    coefficients
        Coefficients defining the equation of the linear entity.
    slope
        The slope of the linear entity, or infinity if vertical.
    points
        Points that define the linear entity.

    Notes
    =====

    This is an abstract class and is not meant to be instantiated.

    See Also
    ========

    sympy.geometry.entity.GeometryEntity

    """
    @property
    def bounds(self):
        """Return a tuple (xmin, ymin, xmax, ymax) representing the bounding
        rectangle for the geometric figure.

        """
        # Get the vertices (points) defining the geometric figure
        verts = self.points
        # Extract x-coordinates and y-coordinates from vertices
        xs = [p.x for p in verts]
        ys = [p.y for p in verts]
        # Calculate and return the bounding rectangle coordinates
        return (min(xs), min(ys), max(xs), max(ys))

    def perpendicular_line(self, p):
        """Create a new Line perpendicular to this linear entity which passes
        through the point `p`.

        Parameters
        ==========

        p : Point
            The point through which the perpendicular line should pass.

        Returns
        =======

        line : Line
            The perpendicular Line object.

        See Also
        ========

        sympy.geometry.line.LinearEntity.is_perpendicular, perpendicular_segment

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2, p3 = Point(0, 0), Point(2, 3), Point(-2, 2)
        >>> L = Line(p1, p2)
        >>> P = L.perpendicular_line(p3); P
        Line2D(Point2D(-2, 2), Point2D(-5, 4))
        >>> L.is_perpendicular(P)
        True

        In 2D, the first point of the perpendicular line is the
        point through which was required to pass; the second
        point is arbitrarily chosen. To get a line that explicitly
        uses a point in the line, create a line from the perpendicular
        segment from the line to the point:

        >>> Line(L.perpendicular_segment(p3))
        Line2D(Point2D(-2, 2), Point2D(4/13, 6/13))
        """
        p = Point(p, dim=self.ambient_dimension)
        # Create a line perpendicular to this one through point p
        # Direction of the new line is orthogonal to the direction of this line
        return Line(p, p + self.direction.orthogonal_direction)

    @property
    def slope(self):
        """The slope of this linear entity, or infinity if vertical.

        Returns
        =======

        slope : number or SymPy expression
            The slope of the linear entity.

        See Also
        ========

        coefficients

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2 = Point(0, 0), Point(3, 5)
        >>> l1 = Line(p1, p2)
        >>> l1.slope
        5/3

        >>> p3 = Point(0, 4)
        >>> l2 = Line(p1, p3)
        >>> l2.slope
        oo

        """
        d1, d2 = (self.p1 - self.p2).args
        # Calculate the difference in y and x coordinates of the defining points
        if d1 == 0:
            return S.Infinity
        # Compute and simplify the slope using the differences
        return simplify(d2/d1)


class Line2D(LinearEntity2D, Line):
    """An infinite line in space 2D.

    A line is declared with two distinct points or a point and slope
    as defined using keyword `slope`.

    Parameters
    ==========
    # 定义类成员变量 p1 和 pt，分别表示线段的起点和终点
    p1 : Point
    pt : Point
    # 斜率 slope 是一个 SymPy 表达式
    slope : SymPy expression

    # 参考链接指向 sympy.geometry.point.Point 类的文档

    # 示例用法，展示了如何使用 sympy 的 Line 类和相关方法
    Examples
    ========
    
    >>> from sympy import Line, Segment, Point
    >>> L = Line(Point(2,3), Point(3,5))
    >>> L
    Line2D(Point2D(2, 3), Point2D(3, 5))
    >>> L.points
    (Point2D(2, 3), Point2D(3, 5))
    >>> L.equation()
    -2*x + y + 1
    >>> L.coefficients
    (-2, 1, 1)

    # 使用关键字 "slope" 实例化 Line 对象
    Instantiate with keyword ``slope``:

    >>> Line(Point(0, 0), slope=0)
    Line2D(Point2D(0, 0), Point2D(1, 0))

    # 使用另一个线性对象实例化 Line 对象
    Instantiate with another linear object

    >>> s = Segment((0, 0), (0, 1))
    >>> Line(s).equation()
    x
    """
    # 创建 Line 类的构造函数，根据给定的参数 p1, pt 或 slope 实例化线段对象
    def __new__(cls, p1, pt=None, slope=None, **kwargs):
        # 如果 p1 是 LinearEntity 类的实例
        if isinstance(p1, LinearEntity):
            # 如果同时给定了 pt 参数，则抛出异常
            if pt is not None:
                raise ValueError('When p1 is a LinearEntity, pt should be None')
            # 根据 p1 的参数规范化 p1 和 pt
            p1, pt = Point._normalize_dimension(*p1.args, dim=2)
        else:
            # 否则，将 p1 规范化为 Point 对象
            p1 = Point(p1, dim=2)
        
        # 如果给定了 pt 参数但没有 slope 参数
        if pt is not None and slope is None:
            try:
                # 尝试将 pt 规范化为 Point 对象
                p2 = Point(pt, dim=2)
            except (NotImplementedError, TypeError, ValueError):
                # 如果无法规范化 pt，则抛出异常
                raise ValueError('The 2nd argument was not a valid Point.')
        
        # 如果给定了 slope 参数但没有 pt 参数
        elif slope is not None and pt is None:
            # 将 slope 转换为 SymPy 表达式
            slope = sympify(slope)
            # 如果斜率是无限大，则调整坐标不变
            if slope.is_finite is False:
                dx = 0
                dy = 1
            else:
                # 否则，按斜率增加坐标
                dx = 1
                dy = slope
            # 通过直接添加到坐标来避免简化，创建 Point 对象
            p2 = Point(p1.x + dx, p1.y + dy, evaluate=False)
        
        # 如果既没有给定 pt 参数也没有给定 slope 参数，则抛出异常
        else:
            raise ValueError('A 2nd Point or keyword "slope" must be used.')
        
        # 调用父类 LinearEntity2D 的构造函数创建对象并返回
        return LinearEntity2D.__new__(cls, p1, p2, **kwargs)

    # 返回 SVG 路径元素的字符串表示，用于 LinearEntity
    def _svg(self, scale_factor=1., fill_color="#66cc99"):
        """Returns SVG path element for the LinearEntity.

        Parameters
        ==========

        scale_factor : float
            Multiplication factor for the SVG stroke-width.  Default is 1.
        fill_color : str, optional
            Hex string for fill color. Default is "#66cc99".
        """
        # 获取线段的起点和终点坐标
        verts = (N(self.p1), N(self.p2))
        # 将坐标格式化为 SVG 路径格式
        coords = ["{},{}".format(p.x, p.y) for p in verts]
        path = "M {} L {}".format(coords[0], " L ".join(coords[1:]))
        
        # 返回 SVG 元素的字符串表示，包括路径、填充颜色等信息
        return (
            '<path fill-rule="evenodd" fill="{2}" stroke="#555555" '
            'stroke-width="{0}" opacity="0.6" d="{1}" '
            'marker-start="url(#markerReverseArrow)" marker-end="url(#markerArrow)"/>'
        ).format(2.*scale_factor, path, fill_color)

    # 声明一个 property 方法，但未给出具体实现，因此需要继续完善
    @property
    def coefficients(self):
        """The coefficients (`a`, `b`, `c`) for `ax + by + c = 0`.

        See Also
        ========

        sympy.geometry.line.Line2D.equation

        Examples
        ========

        >>> from sympy import Point, Line
        >>> from sympy.abc import x, y
        >>> p1, p2 = Point(0, 0), Point(5, 3)
        >>> l = Line(p1, p2)
        >>> l.coefficients
        (-3, 5, 0)

        >>> p3 = Point(x, y)
        >>> l2 = Line(p1, p3)
        >>> l2.coefficients
        (-y, x, 0)

        """
        p1, p2 = self.points
        # 如果线段垂直于 x 轴
        if p1.x == p2.x:
            return (S.One, S.Zero, -p1.x)
        # 如果线段水平于 y 轴
        elif p1.y == p2.y:
            return (S.Zero, S.One, -p1.y)
        # 一般情况下，计算一般形式的系数 a, b, c
        return tuple([simplify(i) for i in
                      (self.p1.y - self.p2.y,  # a 的计算
                       self.p2.x - self.p1.x,  # b 的计算
                       self.p1.x*self.p2.y - self.p1.y*self.p2.x)])  # c 的计算

    def equation(self, x='x', y='y'):
        """The equation of the line: ax + by + c.

        Parameters
        ==========

        x : str, optional
            The name to use for the x-axis, default value is 'x'.
        y : str, optional
            The name to use for the y-axis, default value is 'y'.

        Returns
        =======

        equation : SymPy expression

        See Also
        ========

        sympy.geometry.line.Line2D.coefficients

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2 = Point(1, 0), Point(5, 3)
        >>> l1 = Line(p1, p2)
        >>> l1.equation()
        -3*x + 4*y + 3

        """
        x = _symbol(x, real=True)
        y = _symbol(y, real=True)
        p1, p2 = self.points
        # 如果线段垂直于 x 轴
        if p1.x == p2.x:
            return x - p1.x
        # 如果线段水平于 y 轴
        elif p1.y == p2.y:
            return y - p1.y

        # 获取线段的系数 a, b, c
        a, b, c = self.coefficients
        # 返回线段的一般方程
        return a*x + b*y + c
class Ray2D(LinearEntity2D, Ray):
    """
    A Ray is a semi-line in the space with a source point and a direction.

    Parameters
    ==========

    p1 : Point
        The source of the Ray
    p2 : Point or radian value
        This point determines the direction in which the Ray propagates.
        If given as an angle it is interpreted in radians with the positive
        direction being ccw.

    Attributes
    ==========

    source
    xdirection
    ydirection

    See Also
    ========

    sympy.geometry.point.Point, Line

    Examples
    ========

    >>> from sympy import Point, pi, Ray
    >>> r = Ray(Point(2, 3), Point(3, 5))
    >>> r
    Ray2D(Point2D(2, 3), Point2D(3, 5))
    >>> r.points
    (Point2D(2, 3), Point2D(3, 5))
    >>> r.source
    Point2D(2, 3)
    >>> r.xdirection
    oo
    >>> r.ydirection
    oo
    >>> r.slope
    2
    >>> Ray(Point(0, 0), angle=pi/4).slope
    1

    """
    def __new__(cls, p1, pt=None, angle=None, **kwargs):
        # 将 p1 转换为 Point 对象，确保是二维的点
        p1 = Point(p1, dim=2)
        if pt is not None and angle is None:
            try:
                # 将 pt 转换为 Point 对象，确保是二维的点
                p2 = Point(pt, dim=2)
            except (NotImplementedError, TypeError, ValueError):
                # 如果无法转换成有效的 Point 对象，抛出异常
                raise ValueError(filldedent('''
                    The 2nd argument was not a valid Point; if
                    it was meant to be an angle it should be
                    given with keyword "angle".'''))
            if p1 == p2:
                # 如果 p1 等于 p2，则抛出异常，要求射线有两个不同的点
                raise ValueError('A Ray requires two distinct points.')
        elif angle is not None and pt is None:
            # 如果指定了角度但没有指定点 pt
            # 判断角度是否是 pi/2 的奇数倍
            angle = sympify(angle)
            c = _pi_coeff(angle)
            p2 = None
            if c is not None:
                if c.is_Rational:
                    # 根据角度系数确定 p2 的位置
                    if c.q == 2:
                        if c.p == 1:
                            p2 = p1 + Point(0, 1)
                        elif c.p == 3:
                            p2 = p1 + Point(0, -1)
                    elif c.q == 1:
                        if c.p == 0:
                            p2 = p1 + Point(1, 0)
                        elif c.p == 1:
                            p2 = p1 + Point(-1, 0)
                if p2 is None:
                    c *= S.Pi
            else:
                c = angle % (2*S.Pi)
            if not p2:
                # 根据角度计算 p2 的坐标
                m = 2*c/S.Pi
                left = And(1 < m, m < 3)  # 是否在第二或第三象限
                x = Piecewise((-1, left), (Piecewise((0, Eq(m % 1, 0)), (1, True)), True))
                y = Piecewise((-tan(c), left), (Piecewise((1, Eq(m, 1)), (-1, Eq(m, 3)), (tan(c), True)), True))
                p2 = p1 + Point(x, y)
        else:
            # 如果没有正确提供第二点或角度参数，则抛出异常
            raise ValueError('A 2nd point or keyword "angle" must be used.')

        # 使用 LinearEntity2D 类的构造函数创建 Ray2D 对象
        return LinearEntity2D.__new__(cls, p1, p2, **kwargs)

    @property
    def xdirection(self):
        """The x direction of the ray.

        Positive infinity if the ray points in the positive x direction,
        negative infinity if the ray points in the negative x direction,
        or 0 if the ray is vertical.

        See Also
        ========

        ydirection

        Examples
        ========

        >>> from sympy import Point, Ray
        >>> p1, p2, p3 = Point(0, 0), Point(1, 1), Point(0, -1)
        >>> r1, r2 = Ray(p1, p2), Ray(p1, p3)
        >>> r1.xdirection
        oo
        >>> r2.xdirection
        0

        """
        # 判断射线在x方向上的方向性质
        if self.p1.x < self.p2.x:
            return S.Infinity  # 射线指向正x方向，返回正无穷
        elif self.p1.x == self.p2.x:
            return S.Zero  # 射线垂直于x轴，返回0
        else:
            return S.NegativeInfinity  # 射线指向负x方向，返回负无穷

    @property
    def ydirection(self):
        """The y direction of the ray.

        Positive infinity if the ray points in the positive y direction,
        negative infinity if the ray points in the negative y direction,
        or 0 if the ray is horizontal.

        See Also
        ========

        xdirection

        Examples
        ========

        >>> from sympy import Point, Ray
        >>> p1, p2, p3 = Point(0, 0), Point(-1, -1), Point(-1, 0)
        >>> r1, r2 = Ray(p1, p2), Ray(p1, p3)
        >>> r1.ydirection
        -oo
        >>> r2.ydirection
        0

        """
        # 判断射线在y方向上的方向性质
        if self.p1.y < self.p2.y:
            return S.Infinity  # 射线指向正y方向，返回正无穷
        elif self.p1.y == self.p2.y:
            return S.Zero  # 射线垂直于y轴，返回0
        else:
            return S.NegativeInfinity  # 射线指向负y方向，返回负无穷

    def closing_angle(r1, r2):
        """Return the angle by which r2 must be rotated so it faces the same
        direction as r1.

        Parameters
        ==========

        r1 : Ray2D
        r2 : Ray2D

        Returns
        =======

        angle : angle in radians (ccw angle is positive)

        See Also
        ========

        LinearEntity.angle_between

        Examples
        ========

        >>> from sympy import Ray, pi
        >>> r1 = Ray((0, 0), (1, 0))
        >>> r2 = r1.rotate(-pi/2)
        >>> angle = r1.closing_angle(r2); angle
        pi/2
        >>> r2.rotate(angle).direction.unit == r1.direction.unit
        True
        >>> r2.closing_angle(r1)
        -pi/2
        """
        # 求射线r2顺时针旋转的角度，使其方向与r1相同
        if not all(isinstance(r, Ray2D) for r in (r1, r2)):
            # 尽管方向属性对所有线性实体都定义了，但只有射线才是真正的有向对象
            raise TypeError('Both arguments must be Ray2D objects.')

        # 计算r1和r2的方向角度
        a1 = atan2(*list(reversed(r1.direction.args)))
        a2 = atan2(*list(reversed(r2.direction.args)))
        
        # 调整角度范围确保角度差在合理范围内
        if a1 * a2 < 0:
            a1 = 2 * S.Pi + a1 if a1 < 0 else a1
            a2 = 2 * S.Pi + a2 if a2 < 0 else a2
        
        # 返回角度差
        return a1 - a2
class Segment2D(LinearEntity2D, Segment):
    """A line segment in 2D space.

    Parameters
    ==========

    p1 : Point
        The starting point of the line segment.
    p2 : Point
        The ending point of the line segment.

    Attributes
    ==========

    length : number or SymPy expression
        The length of the line segment.
    midpoint : Point
        The midpoint of the line segment.

    See Also
    ========

    sympy.geometry.point.Point, Line

    Examples
    ========

    >>> from sympy import Point, Segment
    >>> Segment((1, 0), (1, 1)) # tuples are interpreted as pts
    Segment2D(Point2D(1, 0), Point2D(1, 1))
    >>> s = Segment(Point(4, 3), Point(1, 1)); s
    Segment2D(Point2D(4, 3), Point2D(1, 1))
    >>> s.points
    (Point2D(4, 3), Point2D(1, 1))
    >>> s.slope
    2/3
    >>> s.length
    sqrt(13)
    >>> s.midpoint
    Point2D(5/2, 2)

    """
    def __new__(cls, p1, p2, **kwargs):
        # Convert p1 and p2 to Point objects with dimension 2
        p1 = Point(p1, dim=2)
        p2 = Point(p2, dim=2)

        if p1 == p2:
            # If p1 and p2 are the same point, return p1 (a degenerate case)
            return p1

        # Otherwise, create a new instance of LinearEntity2D with p1 and p2
        return LinearEntity2D.__new__(cls, p1, p2, **kwargs)

    def _svg(self, scale_factor=1., fill_color="#66cc99"):
        """Returns SVG path element for the LinearEntity.

        Parameters
        ==========

        scale_factor : float
            Multiplication factor for the SVG stroke-width. Default is 1.
        fill_color : str, optional
            Hex string for fill color. Default is "#66cc99".
        """
        # Convert points p1 and p2 to numerical values
        verts = (N(self.p1), N(self.p2))
        # Extract x, y coordinates for each point in verts
        coords = ["{},{}".format(p.x, p.y) for p in verts]
        # Construct the SVG path string
        path = "M {} L {}".format(coords[0], " L ".join(coords[1:]))
        # Format and return the SVG path element as a string
        return (
            '<path fill-rule="evenodd" fill="{2}" stroke="#555555" '
            'stroke-width="{0}" opacity="0.6" d="{1}" />'
        ).format(2.*scale_factor, path, fill_color)


class LinearEntity3D(LinearEntity):
    """An base class for all linear entities (line, ray and segment)
    in a 3-dimensional Euclidean space.

    Attributes
    ==========

    p1
    p2
    direction_ratio
    direction_cosine
    points

    Notes
    =====

    This is a base class and is not meant to be instantiated.
    """
    def __new__(cls, p1, p2, **kwargs):
        # Convert p1 and p2 to Point objects with dimension 3
        p1 = Point3D(p1, dim=3)
        p2 = Point3D(p2, dim=3)
        if p1 == p2:
            # If p1 and p2 are the same point, raise an error
            raise ValueError(
                "%s.__new__ requires two unique Points." % cls.__name__)

        # Otherwise, create a new instance of GeometryEntity with p1 and p2
        return GeometryEntity.__new__(cls, p1, p2, **kwargs)

    ambient_dimension = 3

    @property
    def direction_ratio(self):
        """The direction ratio of a given line in 3D.

        See Also
        ========

        sympy.geometry.line.Line3D.equation

        Examples
        ========

        >>> from sympy import Point3D, Line3D
        >>> p1, p2 = Point3D(0, 0, 0), Point3D(5, 3, 1)
        >>> l = Line3D(p1, p2)
        >>> l.direction_ratio
        [5, 3, 1]
        """
        # Extract points p1 and p2 from self.points and compute direction ratio
        p1, p2 = self.points
        return p1.direction_ratio(p2)
    def direction_cosine(self):
        """
        计算给定三维空间中一条线的方向余弦。

        See Also
        ========

        sympy.geometry.line.Line3D.equation
            参考线的方程式。

        Examples
        ========

        >>> from sympy import Point3D, Line3D
        >>> p1, p2 = Point3D(0, 0, 0), Point3D(5, 3, 1)
        >>> l = Line3D(p1, p2)
        >>> l.direction_cosine
        [sqrt(35)/7, 3*sqrt(35)/35, sqrt(35)/35]
        >>> sum(i**2 for i in _)
        1
        """
        p1, p2 = self.points
        # 返回线段两点 p1 和 p2 的方向余弦
        return p1.direction_cosine(p2)
class Line3D(LinearEntity3D, Line):
    """An infinite 3D line in space.

    A line is declared with two distinct points or a point and direction_ratio
    as defined using keyword `direction_ratio`.

    Parameters
    ==========

    p1 : Point3D
        The first point that defines the line.
    pt : Point3D, optional
        The second point that defines the line (if not using direction_ratio).
    direction_ratio : list, optional
        Ratios used to compute the second point if `pt` is not provided.

    See Also
    ========

    sympy.geometry.point.Point3D
    sympy.geometry.line.Line
    sympy.geometry.line.Line2D

    Examples
    ========

    >>> from sympy import Line3D, Point3D
    >>> L = Line3D(Point3D(2, 3, 4), Point3D(3, 5, 1))
    >>> L
    Line3D(Point3D(2, 3, 4), Point3D(3, 5, 1))
    >>> L.points
    (Point3D(2, 3, 4), Point3D(3, 5, 1))
    """
    def __new__(cls, p1, pt=None, direction_ratio=(), **kwargs):
        """Create a new instance of Line3D.

        Determines how the line is initialized based on the input parameters.

        Parameters
        ==========

        p1 : Point3D or LinearEntity3D
            The starting point of the line or a LinearEntity3D object.
        pt : Point3D, optional
            The ending point of the line (if p1 is a Point3D).
        direction_ratio : list, optional
            Ratios used to compute the ending point if `pt` is not provided.

        Raises
        ======

        ValueError
            If the input parameters are not valid for defining a line.

        Returns
        =======

        Line3D object
            A new instance of Line3D based on the provided parameters.
        """
        if isinstance(p1, LinearEntity3D):
            if pt is not None:
                raise ValueError('if p1 is a LinearEntity, pt must be None.')
            p1, pt = p1.args
        else:
            p1 = Point(p1, dim=3)
        
        if pt is not None and len(direction_ratio) == 0:
            pt = Point(pt, dim=3)
        elif len(direction_ratio) == 3 and pt is None:
            pt = Point3D(p1.x + direction_ratio[0], p1.y + direction_ratio[1],
                         p1.z + direction_ratio[2])
        else:
            raise ValueError('A 2nd Point or keyword "direction_ratio" must '
                             'be used.')

        return LinearEntity3D.__new__(cls, p1, pt, **kwargs)

    def equation(self, x='x', y='y', z='z'):
        """Return the equations that define the line in 3D.

        Parameters
        ==========

        x : str, optional
            The name to use for the x-axis, default value is 'x'.
        y : str, optional
            The name to use for the y-axis, default value is 'y'.
        z : str, optional
            The name to use for the z-axis, default value is 'z'.

        Returns
        =======

        equation : Tuple of simultaneous equations
            The equations that describe the line in terms of x, y, and z.

        Examples
        ========

        >>> from sympy import Point3D, Line3D, solve
        >>> from sympy.abc import x, y, z
        >>> p1, p2 = Point3D(1, 0, 0), Point3D(5, 3, 0)
        >>> l1 = Line3D(p1, p2)
        >>> eq = l1.equation(x, y, z); eq
        (-3*x + 4*y + 3, z)
        >>> solve(eq.subs(z, 0), (x, y, z))
        {x: 4*y/3 + 1}
        """
        x, y, z, k = [_symbol(i, real=True) for i in (x, y, z, 'k')]
        p1, p2 = self.points
        d1, d2, d3 = p1.direction_ratio(p2)
        x1, y1, z1 = p1
        eqs = [-d1*k + x - x1, -d2*k + y - y1, -d3*k + z - z1]
        
        # eliminate k from equations by solving first eq with k for k
        for i, e in enumerate(eqs):
            if e.has(k):
                kk = solve(eqs[i], k)[0]
                eqs.pop(i)
                break
        return Tuple(*[i.subs(k, kk).as_numer_denom()[0] for i in eqs])
    def distance(self, other):
        """
        Finds the shortest distance between a line and another object.

        Parameters
        ==========

        Point3D, Line3D, Plane, tuple, list

        Returns
        =======

        distance

        Notes
        =====

        This method accepts only 3D entities as its parameter.

        Tuples and lists are converted to Point3D and therefore must be of
        length 3, 2, or 1.

        NotImplementedError is raised if `other` is not an instance of one
        of the specified classes: Point3D, Line3D, or Plane.

        Examples
        ========

        >>> from sympy.geometry import Line3D
        >>> l1 = Line3D((0, 0, 0), (0, 0, 1))
        >>> l2 = Line3D((0, 1, 0), (1, 1, 1))
        >>> l1.distance(l2)
        1

        The computed distance may be symbolic, too:

        >>> from sympy.abc import x, y
        >>> l1 = Line3D((0, 0, 0), (0, 0, 1))
        >>> l2 = Line3D((0, x, 0), (y, x, 1))
        >>> l1.distance(l2)
        Abs(x*y)/Abs(sqrt(y**2))

        """

        from .plane import Plane  # Avoid circular import

        # If `other` is a tuple or list, attempt conversion to Point3D
        if isinstance(other, (tuple, list)):
            try:
                other = Point3D(other)
            except ValueError:
                pass

        # If `other` is an instance of Point3D, calculate distance using superclass method
        if isinstance(other, Point3D):
            return super().distance(other)

        # If `other` is an instance of Line3D, compute distance based on relative positions
        if isinstance(other, Line3D):
            if self == other:  # If the lines are identical
                return S.Zero
            if self.is_parallel(other):  # If the lines are parallel
                return super().distance(other.p1)

            # If the lines are skew (neither parallel nor intersecting)
            self_direction = Matrix(self.direction_ratio)
            other_direction = Matrix(other.direction_ratio)
            normal = self_direction.cross(other_direction)  # Calculate normal vector
            plane_through_self = Plane(p1=self.p1, normal_vector=normal)  # Create plane through self line
            return other.p1.distance(plane_through_self)  # Compute distance to the plane through self line

        # If `other` is an instance of Plane, compute distance using Plane's method
        if isinstance(other, Plane):
            return other.distance(self)

        # If `other` is not of any supported type, raise NotImplementedError
        msg = f"{other} has type {type(other)}, which is unsupported"
        raise NotImplementedError(msg)
    def __new__(cls, p1, pt=None, direction_ratio=(), **kwargs):
        # 如果 p1 是 LinearEntity3D 的实例，则检查 pt 必须为 None
        if isinstance(p1, LinearEntity3D):
            if pt is not None:
                raise ValueError('If p1 is a LinearEntity, pt must be None')
            # 解包 LinearEntity3D 的参数
            p1, pt = p1.args
        else:
            # 将 p1 转换为 Point 对象（三维）
            p1 = Point(p1, dim=3)
        
        # 根据参数情况处理 pt 和 direction_ratio
        if pt is not None and len(direction_ratio) == 0:
            pt = Point(pt, dim=3)
        elif len(direction_ratio) == 3 and pt is None:
            # 如果提供了 direction_ratio，根据其值计算第二个点的位置
            pt = Point3D(p1.x + direction_ratio[0], p1.y + direction_ratio[1],
                         p1.z + direction_ratio[2])
        else:
            # 抛出异常，要求使用第二个点或关键字 "direction_ratio"
            raise ValueError(filldedent('''
                A 2nd Point or keyword "direction_ratio" must be used.
            '''))
        
        # 调用父类的 __new__ 方法来创建实例
        return LinearEntity3D.__new__(cls, p1, pt, **kwargs)
    def ydirection(self):
        """The y direction of the ray.

        Positive infinity if the ray points in the positive y direction,
        negative infinity if the ray points in the negative y direction,
        or 0 if the ray is horizontal.

        See Also
        ========

        xdirection

        Examples
        ========

        >>> from sympy import Point3D, Ray3D
        >>> p1, p2, p3 = Point3D(0, 0, 0), Point3D(-1, -1, -1), Point3D(-1, 0, 0)
        >>> r1, r2 = Ray3D(p1, p2), Ray3D(p1, p3)
        >>> r1.ydirection
        -oo
        >>> r2.ydirection
        0

        """
        # 检查射线在 y 方向上的方向
        if self.p1.y < self.p2.y:
            # 如果射线朝着正 y 方向，则返回正无穷
            return S.Infinity
        elif self.p1.y == self.p2.y:
            # 如果射线是水平的，则返回 0
            return S.Zero
        else:
            # 射线朝着负 y 方向，则返回负无穷
            return S.NegativeInfinity

    @property
    def zdirection(self):
        """The z direction of the ray.

        Positive infinity if the ray points in the positive z direction,
        negative infinity if the ray points in the negative z direction,
        or 0 if the ray is horizontal.

        See Also
        ========

        xdirection

        Examples
        ========

        >>> from sympy import Point3D, Ray3D
        >>> p1, p2, p3 = Point3D(0, 0, 0), Point3D(-1, -1, -1), Point3D(-1, 0, 0)
        >>> r1, r2 = Ray3D(p1, p2), Ray3D(p1, p3)
        >>> r1.ydirection
        -oo
        >>> r2.ydirection
        0
        >>> r2.zdirection
        0

        """
        # 检查射线在 z 方向上的方向
        if self.p1.z < self.p2.z:
            # 如果射线朝着正 z 方向，则返回正无穷
            return S.Infinity
        elif self.p1.z == self.p2.z:
            # 如果射线是水平的，则返回 0
            return S.Zero
        else:
            # 射线朝着负 z 方向，则返回负无穷
            return S.NegativeInfinity
# 定义一个表示三维空间中线段的类，继承自LinearEntity3D和Segment类
class Segment3D(LinearEntity3D, Segment):
    """A line segment in a 3D space.

    Parameters
    ==========

    p1 : Point3D     # 线段的一个端点，类型为Point3D
    p2 : Point3D     # 线段的另一个端点，类型为Point3D

    Attributes
    ==========

    length : number or SymPy expression    # 线段的长度，可以是数值或SymPy表达式
    midpoint : Point3D    # 线段的中点，类型为Point3D

    See Also
    ========

    sympy.geometry.point.Point3D, Line3D    # 参见SymPy中的Point3D类和Line3D类

    Examples
    ========

    >>> from sympy import Point3D, Segment3D
    >>> Segment3D((1, 0, 0), (1, 1, 1)) # tuples are interpreted as pts
    Segment3D(Point3D(1, 0, 0), Point3D(1, 1, 1))
    >>> s = Segment3D(Point3D(4, 3, 9), Point3D(1, 1, 7)); s
    Segment3D(Point3D(4, 3, 9), Point3D(1, 1, 7))
    >>> s.points
    (Point3D(4, 3, 9), Point3D(1, 1, 7))
    >>> s.length
    sqrt(17)
    >>> s.midpoint
    Point3D(5/2, 2, 8)

    """
    # 构造函数，接受两个点p1和p2来定义线段
    def __new__(cls, p1, p2, **kwargs):
        # 将p1和p2转换为Point对象，维度为3
        p1 = Point(p1, dim=3)
        p2 = Point(p2, dim=3)

        # 如果p1和p2相等，则返回p1，表示线段为一个点
        if p1 == p2:
            return p1

        # 否则调用父类LinearEntity3D的构造函数创建线段
        return LinearEntity3D.__new__(cls, p1, p2, **kwargs)
```