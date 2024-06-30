# `D:\src\scipysrc\sympy\sympy\geometry\entity.py`

```
"""
The definition of the base geometrical entity with attributes common to
all derived geometrical entities.

Contains
========

GeometryEntity
GeometricSet

Notes
=====

A GeometryEntity is any object that has special geometric properties.
A GeometrySet is a superclass of any GeometryEntity that can also
be viewed as a sympy.sets.Set.  In particular, points are the only
GeometryEntity not considered a Set.

Rn is a GeometrySet representing n-dimensional Euclidean space. R2 and
R3 are currently the only ambient spaces implemented.

"""
from __future__ import annotations  # 引入未来版本的类型注解支持

from sympy.core.basic import Basic  # 导入基本符号计算模块中的 Basic 类
from sympy.core.containers import Tuple  # 导入容器模块中的 Tuple 类
from sympy.core.evalf import EvalfMixin, N  # 导入计算模块中的 EvalfMixin 和 N 函数
from sympy.core.numbers import oo  # 导入数值模块中的 oo（无穷大）对象
from sympy.core.symbol import Dummy  # 导入符号模块中的 Dummy 类
from sympy.core.sympify import sympify  # 导入符号模块中的 sympify 函数
from sympy.functions.elementary.trigonometric import cos, sin, atan  # 导入三角函数模块中的 cos, sin, atan 函数
from sympy.matrices import eye  # 导入矩阵模块中的单位矩阵 eye 函数
from sympy.multipledispatch import dispatch  # 导入多分派模块中的 dispatch 装饰器
from sympy.printing import sstr  # 导入打印模块中的 sstr 函数
from sympy.sets import Set, Union, FiniteSet  # 导入集合模块中的 Set, Union, FiniteSet 类
from sympy.sets.handlers.intersection import intersection_sets  # 导入集合处理模块中的 intersection_sets 函数
from sympy.sets.handlers.union import union_sets  # 导入集合处理模块中的 union_sets 函数
from sympy.solvers.solvers import solve  # 导入求解模块中的 solve 函数
from sympy.utilities.misc import func_name  # 导入实用工具模块中的 func_name 函数
from sympy.utilities.iterables import is_sequence  # 导入实用工具模块中的 is_sequence 函数


# How entities are ordered; used by __cmp__ in GeometryEntity
ordering_of_classes = [  # 定义几何实体的排序顺序列表，用于 GeometryEntity 类中的 __cmp__ 方法
    "Point2D",
    "Point3D",
    "Point",
    "Segment2D",
    "Ray2D",
    "Line2D",
    "Segment3D",
    "Line3D",
    "Ray3D",
    "Segment",
    "Ray",
    "Line",
    "Plane",
    "Triangle",
    "RegularPolygon",
    "Polygon",
    "Circle",
    "Ellipse",
    "Curve",
    "Parabola"
]

x, y = [Dummy('entity_dummy') for i in range(2)]  # 创建两个 Dummy 符号对象 x 和 y
T = Dummy('entity_dummy', real=True)  # 创建一个具有实数属性的 Dummy 符号对象 T


class GeometryEntity(Basic, EvalfMixin):
    """The base class for all geometrical entities.

    This class does not represent any particular geometric entity, it only
    provides the implementation of some methods common to all subclasses.

    """

    __slots__: tuple[str, ...] = ()  # 使用 __slots__ 属性限制实例属性

    def __cmp__(self, other):
        """Comparison of two GeometryEntities."""
        n1 = self.__class__.__name__  # 获取当前对象的类名
        n2 = other.__class__.__name__  # 获取另一个对象的类名
        c = (n1 > n2) - (n1 < n2)  # 比较类名的大小关系

        if not c:
            return 0

        i1 = -1
        for cls in self.__class__.__mro__:  # 遍历当前类的方法解析顺序（MRO）
            try:
                i1 = ordering_of_classes.index(cls.__name__)  # 获取当前类在排序列表中的索引
                break
            except ValueError:
                i1 = -1

        if i1 == -1:
            return c

        i2 = -1
        for cls in other.__class__.__mro__:  # 遍历另一个对象的方法解析顺序（MRO）
            try:
                i2 = ordering_of_classes.index(cls.__name__)  # 获取另一个对象类在排序列表中的索引
                break
            except ValueError:
                i2 = -1

        if i2 == -1:
            return c

        return (i1 > i2) - (i1 < i2)  # 根据索引比较类的排序顺序
    def __contains__(self, other):
        """检查对象是否包含另一个对象，子类应该实现比简单相等性更复杂的逻辑。"""
        # 如果self和other的类型相同，则比较它们是否相等
        if type(self) is type(other):
            return self == other
        # 若类型不同，抛出未实现错误
        raise NotImplementedError()

    def __getnewargs__(self):
        """返回一个元组，用于在反序列化时传递给__new__方法。"""
        return tuple(self.args)

    def __ne__(self, o):
        """检查两个几何实体是否不相等。"""
        # 返回self和o是否不相等的结果
        return not self == o

    def __new__(cls, *args, **kwargs):
        """创建一个新的几何实体对象，参数args将被传递给__new__方法。"""
        # Points是序列，但不应转换为元组，因此使用此检测函数代替。
        def is_seq_and_not_point(a):
            # 不能使用isinstance(a, Point)，因为无法导入Point
            if hasattr(a, 'is_Point') and a.is_Point:
                return False
            return is_sequence(a)

        # 根据条件处理args，如果满足条件，将其转换为Tuple对象；否则，转换为sympify对象
        args = [Tuple(*a) if is_seq_and_not_point(a) else sympify(a) for a in args]
        # 调用父类的__new__方法来创建实例
        return Basic.__new__(cls, *args)

    def __radd__(self, a):
        """反向加法的实现方法。"""
        return a.__add__(self)

    def __rtruediv__(self, a):
        """反向除法的实现方法。"""
        return a.__truediv__(self)

    def __repr__(self):
        """返回一个可以由sympy评估的几何实体的字符串表示形式。"""
        return type(self).__name__ + repr(self.args)

    def __rmul__(self, a):
        """反向乘法的实现方法。"""
        return a.__mul__(self)

    def __rsub__(self, a):
        """反向减法的实现方法。"""
        return a.__sub__(self)

    def __str__(self):
        """返回一个几何实体的字符串表示形式。"""
        return type(self).__name__ + sstr(self.args)

    def _eval_subs(self, old, new):
        """用新值替换对象中的旧值。"""
        from sympy.geometry.point import Point, Point3D
        # 如果old或new是序列，则根据对象类型进行转换
        if is_sequence(old) or is_sequence(new):
            if isinstance(self, Point3D):
                old = Point3D(old)
                new = Point3D(new)
            else:
                old = Point(old)
                new = Point(new)
            # 调用对象的内部替换方法_subs，返回替换后的结果
            return self._subs(old, new)
    def _repr_svg_(self):
        """SVG representation of a GeometryEntity suitable for IPython"""

        try:
            bounds = self.bounds  # 获取几何实体的边界框信息
        except (NotImplementedError, TypeError):
            # 如果无法获取边界框信息，则返回 None，使 IPython 使用下一个表示方式
            return None

        if not all(x.is_number and x.is_finite for x in bounds):
            # 如果边界框信息中有任何一个不是有限的数值，则返回 None
            return None

        # 定义 SVG 的顶部部分，包括命名空间和一些标记定义
        svg_top = '''<svg xmlns="http://www.w3.org/2000/svg"
            xmlns:xlink="http://www.w3.org/1999/xlink"
            width="{1}" height="{2}" viewBox="{0}"
            preserveAspectRatio="xMinYMin meet">
            <defs>
                <marker id="markerCircle" markerWidth="8" markerHeight="8"
                    refx="5" refy="5" markerUnits="strokeWidth">
                    <circle cx="5" cy="5" r="1.5" style="stroke: none; fill:#000000;"/>
                </marker>
                <marker id="markerArrow" markerWidth="13" markerHeight="13" refx="2" refy="4"
                       orient="auto" markerUnits="strokeWidth">
                    <path d="M2,2 L2,6 L6,4" style="fill: #000000;" />
                </marker>
                <marker id="markerReverseArrow" markerWidth="13" markerHeight="13" refx="6" refy="4"
                       orient="auto" markerUnits="strokeWidth">
                    <path d="M6,2 L6,6 L2,4" style="fill: #000000;" />
                </marker>
            </defs>'''

        # 根据边界框的数据计算 SVG 视图框和缩放因子
        xmin, ymin, xmax, ymax = map(N, bounds)
        if xmin == xmax and ymin == ymax:
            # 如果边界框是一个点，则扩展边界框的大小
            xmin, ymin, xmax, ymax = xmin - .5, ymin -.5, xmax + .5, ymax + .5
        else:
            # 否则，按照数据范围的一部分扩展边界框
            expand = 0.1  # 或者使用 10%，确保箭头头部在视野中可见（R 绘图使用 4%）
            widest_part = max([xmax - xmin, ymax - ymin])
            expand_amount = widest_part * expand
            xmin -= expand_amount
            ymin -= expand_amount
            xmax += expand_amount
            ymax += expand_amount

        # 计算 SVG 的宽度和高度，并限制它们在 100 到 300 之间
        dx = xmax - xmin
        dy = ymax - ymin
        width = min([max([100., dx]), 300])
        height = min([max([100., dy]), 300])

        # 计算缩放因子，确保所有数据都能显示在 SVG 中
        scale_factor = 1. if max(width, height) == 0 else max(dx, dy) / max(width, height)
        try:
            svg = self._svg(scale_factor)  # 生成具有给定缩放因子的 SVG 表示
        except (NotImplementedError, TypeError):
            # 如果无法生成 SVG 表示，则返回 None，使 IPython 使用下一个表示方式
            return None

        # 设置 SVG 的视图框和变换矩阵
        view_box = "{} {} {} {}".format(xmin, ymin, dx, dy)
        transform = "matrix(1,0,0,-1,0,{})".format(ymax + ymin)
        svg_top = svg_top.format(view_box, width, height)

        # 返回完整的 SVG 字符串，包括顶部定义和实际的 SVG 内容
        return svg_top + (
            '<g transform="{}">{}</g></svg>'
            ).format(transform, svg)
    def _svg(self, scale_factor=1., fill_color="#66cc99"):
        """Returns SVG path element for the GeometryEntity.

        Parameters
        ==========

        scale_factor : float
            Multiplication factor for the SVG stroke-width.  Default is 1.
        fill_color : str, optional
            Hex string for fill color. Default is "#66cc99".
        """
        # 这是一个未实现的方法，应返回表示几何实体的SVG路径元素
        raise NotImplementedError()

    def _sympy_(self):
        # 返回对象本身
        return self

    @property
    def ambient_dimension(self):
        """What is the dimension of the space that the object is contained in?"""
        # 返回该几何对象所在空间的维度
        raise NotImplementedError()

    @property
    def bounds(self):
        """Return a tuple (xmin, ymin, xmax, ymax) representing the bounding
        rectangle for the geometric figure.

        """
        # 返回表示几何图形边界矩形的元组 (xmin, ymin, xmax, ymax)
        raise NotImplementedError()

    def encloses(self, o):
        """
        Return True if o is inside (not on or outside) the boundaries of self.

        The object will be decomposed into Points and individual Entities need
        only define an encloses_point method for their class.

        See Also
        ========

        sympy.geometry.ellipse.Ellipse.encloses_point
        sympy.geometry.polygon.Polygon.encloses_point

        Examples
        ========

        >>> from sympy import RegularPolygon, Point, Polygon
        >>> t  = Polygon(*RegularPolygon(Point(0, 0), 1, 3).vertices)
        >>> t2 = Polygon(*RegularPolygon(Point(0, 0), 2, 3).vertices)
        >>> t2.encloses(t)
        True
        >>> t.encloses(t2)
        False

        """
        # 检查对象 o 是否完全包含在当前对象的边界内
        from sympy.geometry.point import Point
        from sympy.geometry.line import Segment, Ray, Line
        from sympy.geometry.ellipse import Ellipse
        from sympy.geometry.polygon import Polygon, RegularPolygon

        if isinstance(o, Point):
            return self.encloses_point(o)
        elif isinstance(o, Segment):
            return all(self.encloses_point(x) for x in o.points)
        elif isinstance(o, (Ray, Line)):
            return False
        elif isinstance(o, Ellipse):
            return self.encloses_point(o.center) and \
                self.encloses_point(
                Point(o.center.x + o.hradius, o.center.y)) and \
                not self.intersection(o)
        elif isinstance(o, Polygon):
            if isinstance(o, RegularPolygon):
                if not self.encloses_point(o.center):
                    return False
            return all(self.encloses_point(v) for v in o.vertices)
        # 如果对象类型未知，则抛出未实现错误
        raise NotImplementedError()

    def equals(self, o):
        # 判断当前对象是否等于对象 o
        return self == o
    # 抛出未实现错误，表示该方法需要在子类中进行实现
    def intersection(self, o):
        """
        Returns a list of all of the intersections of self with o.

        Notes
        =====

        An entity is not required to implement this method.

        If two different types of entities can intersect, the item with
        higher index in ordering_of_classes should implement
        intersections with anything having a lower index.

        See Also
        ========

        sympy.geometry.util.intersection

        """
        raise NotImplementedError()

    # 抛出未实现错误，表示该方法需要在子类中进行实现
    def is_similar(self, other):
        """Is this geometrical entity similar to another geometrical entity?

        Two entities are similar if a uniform scaling (enlarging or
        shrinking) of one of the entities will allow one to obtain the other.

        Notes
        =====

        This method is not intended to be used directly but rather
        through the `are_similar` function found in util.py.
        An entity is not required to implement this method.
        If two different types of entities can be similar, it is only
        required that one of them be able to determine this.

        See Also
        ========

        scale

        """
        raise NotImplementedError()
    def reflect(self, line):
        """
        Reflects an object across a line.

        Parameters
        ==========

        line: Line
            The line across which the object is reflected.

        Examples
        ========

        >>> from sympy import pi, sqrt, Line, RegularPolygon
        >>> l = Line((0, pi), slope=sqrt(2))
        >>> pent = RegularPolygon((1, 2), 1, 5)
        >>> rpent = pent.reflect(l)
        >>> rpent
        RegularPolygon(Point2D(-2*sqrt(2)*pi/3 - 1/3 + 4*sqrt(2)/3, 2/3 + 2*sqrt(2)/3 + 2*pi/3), -1, 5, -atan(2*sqrt(2)) + 3*pi/5)

        >>> from sympy import pi, Line, Circle, Point
        >>> l = Line((0, pi), slope=1)
        >>> circ = Circle(Point(0, 0), 5)
        >>> rcirc = circ.reflect(l)
        >>> rcirc
        Circle(Point2D(-pi, pi), -5)

        """
        from sympy.geometry.point import Point  # 导入 Point 类

        g = self  # 当前对象
        l = line  # 给定的反射线
        o = Point(0, 0)  # 原点作为 Point 对象

        if l.slope.is_zero:  # 如果反射线斜率为零
            v = l.args[0].y  # 反射线的 y 坐标
            if not v:  # 如果是 x 轴
                return g.scale(y=-1)  # 对 g 进行 y 轴反向缩放
            reps = [(p, p.translate(y=2*(v - p.y))) for p in g.atoms(Point)]  # 否则进行 y 轴平移
        elif l.slope is oo:  # 如果反射线斜率为无穷大
            v = l.args[0].x  # 反射线的 x 坐标
            if not v:  # 如果是 y 轴
                return g.scale(x=-1)  # 对 g 进行 x 轴反向缩放
            reps = [(p, p.translate(x=2*(v - p.x))) for p in g.atoms(Point)]  # 否则进行 x 轴平移
        else:  # 其他情况
            if not hasattr(g, 'reflect') and not all(
                    isinstance(arg, Point) for arg in g.args):
                raise NotImplementedError(
                    'reflect undefined or non-Point args in %s' % g)  # 抛出未实现异常
            a = atan(l.slope)  # 计算反射角度
            c = l.coefficients  # 反射线的系数
            d = -c[-1]/c[1]  # y 轴截距
            xf = Point(x, y)  # 创建一个新的 Point 对象
            xf = xf.translate(y=-d).rotate(-a, o).scale(y=-1
                ).rotate(a, o).translate(y=d)  # 应用变换到一个点
            reps = [(p, xf.xreplace({x: p.x, y: p.y})) for p in g.atoms(Point)]  # 使用该变换替换每个点
        return g.xreplace(dict(reps))  # 返回替换后的对象

    def rotate(self, angle, pt=None):
        """Rotate ``angle`` radians counterclockwise about Point ``pt``.

        The default pt is the origin, Point(0, 0)

        See Also
        ========

        scale, translate

        Examples
        ========

        >>> from sympy import Point, RegularPolygon, Polygon, pi
        >>> t = Polygon(*RegularPolygon(Point(0, 0), 1, 3).vertices)
        >>> t # vertex on x axis
        Triangle(Point2D(1, 0), Point2D(-1/2, sqrt(3)/2), Point2D(-1/2, -sqrt(3)/2))
        >>> t.rotate(pi/2) # vertex on y axis now
        Triangle(Point2D(0, 1), Point2D(-sqrt(3)/2, -1/2), Point2D(sqrt(3)/2, -1/2))

        """
        newargs = []
        for a in self.args:
            if isinstance(a, GeometryEntity):
                newargs.append(a.rotate(angle, pt))  # 对所有几何实体递归调用旋转
            else:
                newargs.append(a)
        return type(self)(*newargs)  # 返回旋转后的对象
    def scale(self, x=1, y=1, pt=None):
        """Scale the object by multiplying the x,y-coordinates by x and y.

        If pt is given, the scaling is done relative to that point; the
        object is shifted by -pt, scaled, and shifted by pt.

        See Also
        ========

        rotate, translate

        Examples
        ========

        >>> from sympy import RegularPolygon, Point, Polygon
        >>> t = Polygon(*RegularPolygon(Point(0, 0), 1, 3).vertices)
        >>> t
        Triangle(Point2D(1, 0), Point2D(-1/2, sqrt(3)/2), Point2D(-1/2, -sqrt(3)/2))
        >>> t.scale(2)
        Triangle(Point2D(2, 0), Point2D(-1, sqrt(3)/2), Point2D(-1, -sqrt(3)/2))
        >>> t.scale(2, 2)
        Triangle(Point2D(2, 0), Point2D(-1, sqrt(3)), Point2D(-1, -sqrt(3)))

        """
        from sympy.geometry.point import Point  # 导入 Point 类
        if pt:  # 如果给定了 pt 参数
            pt = Point(pt, dim=2)  # 将 pt 转换为 Point 对象，维度为2
            # 对象先向 -pt 平移，然后进行缩放操作，最后再向 pt 平移
            return self.translate(*(-pt).args).scale(x, y).translate(*pt.args)
        # 如果没有给定 pt 参数，对对象中的每个元素应用缩放操作
        return type(self)(*[a.scale(x, y) for a in self.args])  # 如果失败，覆盖此类

    def translate(self, x=0, y=0):
        """Shift the object by adding to the x,y-coordinates the values x and y.

        See Also
        ========

        rotate, scale

        Examples
        ========

        >>> from sympy import RegularPolygon, Point, Polygon
        >>> t = Polygon(*RegularPolygon(Point(0, 0), 1, 3).vertices)
        >>> t
        Triangle(Point2D(1, 0), Point2D(-1/2, sqrt(3)/2), Point2D(-1/2, -sqrt(3)/2))
        >>> t.translate(2)
        Triangle(Point2D(3, 0), Point2D(3/2, sqrt(3)/2), Point2D(3/2, -sqrt(3)/2))
        >>> t.translate(2, 2)
        Triangle(Point2D(3, 2), Point2D(3/2, sqrt(3)/2 + 2), Point2D(3/2, 2 - sqrt(3)/2))

        """
        newargs = []
        for a in self.args:
            if isinstance(a, GeometryEntity):  # 如果元素是几何实体
                newargs.append(a.translate(x, y))  # 对几何实体应用平移操作
            else:
                newargs.append(a)
        return self.func(*newargs)

    def parameter_value(self, other, t):
        """Return the parameter corresponding to the given point.
        Evaluating an arbitrary point of the entity at this parameter
        value will return the given point.

        Examples
        ========

        >>> from sympy import Line, Point
        >>> from sympy.abc import t
        >>> a = Point(0, 0)
        >>> b = Point(2, 2)
        >>> Line(a, b).parameter_value((1, 1), t)
        {t: 1/2}
        >>> Line(a, b).arbitrary_point(t).subs(_)
        Point2D(1, 1)
        """
        from sympy.geometry.point import Point  # 导入 Point 类
        if not isinstance(other, GeometryEntity):  # 如果 other 不是几何实体
            other = Point(other, dim=self.ambient_dimension)  # 将其转换为 Point 对象
        if not isinstance(other, Point):  # 如果 other 不是 Point 对象
            raise ValueError("other must be a point")  # 抛出值错误异常
        # 解方程得到参数 t 的值，使得 entity 的任意点等于给定的点 other
        sol = solve(self.arbitrary_point(T) - other, T, dict=True)
        if not sol:  # 如果没有解
            raise ValueError("Given point is not on %s" % func_name(self))  # 抛出值错误异常
        return {t: sol[0][T]}  # 返回参数 t 的值作为字典的键，对应的解作为值
class GeometrySet(GeometryEntity, Set):
    """Parent class of all GeometryEntity that are also Sets
    (compatible with sympy.sets)
    """
    __slots__ = ()  # 定义空的 __slots__，确保实例没有额外的实例变量

    def _contains(self, other):
        """sympy.sets uses the _contains method, so include it for compatibility."""
        
        if isinstance(other, Set) and other.is_FiniteSet:  # 如果 other 是 FiniteSet
            return all(self.__contains__(i) for i in other)  # 检查是否包含 other 中的所有元素
        
        return self.__contains__(other)  # 否则直接检查是否包含 other

@dispatch(GeometrySet, Set)  # 标记为函数重载的一部分 # type:ignore # noqa:F811
def union_sets(self, o): # noqa:F811
    """ Returns the union of self and o
    for use with sympy.sets.Set, if possible. """

    if o.is_FiniteSet:  # 如果 o 是 FiniteSet
        other_points = [p for p in o if not self._contains(p)]  # 找到 o 中不在 self 中的点
        if len(other_points) == len(o):  # 如果 o 中所有点都不在 self 中
            return None  # 返回空值
        return Union(self, FiniteSet(*other_points))  # 否则返回 self 和剩余点的并集
    if self._contains(o):  # 如果 self 包含 o
        return self  # 返回 self
    return None  # 否则返回空值


@dispatch(GeometrySet, Set)  # 标记为函数重载的一部分 # type: ignore # noqa:F811
def intersection_sets(self, o): # noqa:F811
    """ Returns a sympy.sets.Set of intersection objects,
    if possible. """

    from sympy.geometry.point import Point  # 导入 Point 类

    try:
        if o.is_FiniteSet:  # 如果 o 是 FiniteSet
            inter = FiniteSet(*(p for p in o if self.contains(p)))  # 找到 self 和 o 的交集
        else:
            inter = self.intersection(o)  # 否则找到 self 和 o 的交集
    except NotImplementedError:
        return None  # 如果遇到未实现的情况，返回空值

    points = FiniteSet(*[p for p in inter if isinstance(p, Point)])  # 将交集中的点放入 FiniteSet
    non_points = [p for p in inter if not isinstance(p, Point)]  # 找到交集中非点对象

    return Union(*(non_points + [points]))  # 返回非点对象和点对象的并集


def translate(x, y):
    """Return the matrix to translate a 2-D point by x and y."""
    rv = eye(3)  # 创建一个单位矩阵
    rv[2, 0] = x  # 设置平移矩阵的 x 偏移量
    rv[2, 1] = y  # 设置平移矩阵的 y 偏移量
    return rv  # 返回平移矩阵


def scale(x, y, pt=None):
    """Return the matrix to multiply a 2-D point's coordinates by x and y.

    If pt is given, the scaling is done relative to that point."""
    rv = eye(3)  # 创建一个单位矩阵
    rv[0, 0] = x  # 设置缩放矩阵的 x 缩放比例
    rv[1, 1] = y  # 设置缩放矩阵的 y 缩放比例
    if pt:  # 如果给定了 pt
        from sympy.geometry.point import Point  # 导入 Point 类
        pt = Point(pt, dim=2)  # 创建一个二维的 Point 对象
        tr1 = translate(*(-pt).args)  # 创建将 pt 移动到原点的平移矩阵
        tr2 = translate(*pt.args)  # 创建将原点移动到 pt 的平移矩阵
        return tr1 * rv * tr2  # 返回平移、缩放后的矩阵组合
    return rv  # 否则返回缩放矩阵


def rotate(th):
    """Return the matrix to rotate a 2-D point about the origin by ``angle``.

    The angle is measured in radians. To Point a point about a point other
    then the origin, translate the Point, do the rotation, and
    translate it back:

    >>> from sympy.geometry.entity import rotate, translate
    >>> from sympy import Point, pi
    >>> rot_about_11 = translate(-1, -1)*rotate(pi/2)*translate(1, 1)
    >>> Point(1, 1).transform(rot_about_11)
    Point2D(1, 1)
    >>> Point(0, 0).transform(rot_about_11)
    Point2D(2, 0)
    """
    s = sin(th)  # 计算角度的正弦值
    rv = eye(3) * cos(th)  # 创建旋转矩阵并乘以余弦值
    # 将矩阵 rv 的 (0, 1) 位置设置为变量 s 的值
    rv[0, 1] = s
    # 将矩阵 rv 的 (1, 0) 位置设置为变量 s 的负值
    rv[1, 0] = -s
    # 将矩阵 rv 的 (2, 2) 位置设置为 1
    rv[2, 2] = 1
    # 返回更新后的矩阵 rv
    return rv
```