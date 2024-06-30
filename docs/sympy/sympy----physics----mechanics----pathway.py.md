# `D:\src\scipysrc\sympy\sympy\physics\mechanics\pathway.py`

```
"""Implementations of pathways for use by actuators."""

from abc import ABC, abstractmethod  # 导入抽象基类和抽象方法装饰器

from sympy.core.singleton import S  # 导入单例模块中的 S 对象
from sympy.physics.mechanics.loads import Force  # 导入力加载模块中的 Force 类
from sympy.physics.mechanics.wrapping_geometry import WrappingGeometryBase  # 导入包装几何模块中的 WrappingGeometryBase 类
from sympy.physics.vector import Point, dynamicsymbols  # 导入矢量物理模块中的 Point 类和 dynamicsymbols 函数

__all__ = ['PathwayBase', 'LinearPathway', 'ObstacleSetPathway',  # 导出的类列表
           'WrappingPathway']


class PathwayBase(ABC):
    """Abstract base class for all pathway classes to inherit from.

    Notes
    =====

    Instances of this class cannot be directly instantiated by users. However,
    it can be used to created custom pathway types through subclassing.

    """

    def __init__(self, *attachments):
        """Initializer for ``PathwayBase``."""
        self.attachments = attachments  # 设置路径对象的端点对

    @property
    def attachments(self):
        """The pair of points defining a pathway's ends."""
        return self._attachments  # 返回路径的端点对

    @attachments.setter
    def attachments(self, attachments):
        if hasattr(self, '_attachments'):
            msg = (
                f'Can\'t set attribute `attachments` to {repr(attachments)} '
                f'as it is immutable.'
            )
            raise AttributeError(msg)
        if len(attachments) != 2:
            msg = (
                f'Value {repr(attachments)} passed to `attachments` was an '
                f'iterable of length {len(attachments)}, must be an iterable '
                f'of length 2.'
            )
            raise ValueError(msg)
        for i, point in enumerate(attachments):
            if not isinstance(point, Point):
                msg = (
                    f'Value {repr(point)} passed to `attachments` at index '
                    f'{i} was of type {type(point)}, must be {Point}.'
                )
                raise TypeError(msg)
        self._attachments = tuple(attachments)  # 设置不可变的端点对元组

    @property
    @abstractmethod
    def length(self):
        """An expression representing the pathway's length."""
        pass  # 抽象属性，表示路径长度的表达式

    @property
    @abstractmethod
    def extension_velocity(self):
        """An expression representing the pathway's extension velocity."""
        pass  # 抽象属性，表示路径扩展速度的表达式

    @abstractmethod
    def to_loads(self, force):
        """Loads required by the equations of motion method classes.

        Explanation
        ===========

        ``KanesMethod`` requires a list of ``Point``-``Vector`` tuples to be
        passed to the ``loads`` parameters of its ``kanes_equations`` method
        when constructing the equations of motion. This method acts as a
        utility to produce the correctly-structred pairs of points and vectors
        required so that these can be easily concatenated with other items in
        the list of loads and passed to ``KanesMethod.kanes_equations``. These
        loads are also in the correct form to also be passed to the other
        equations of motion method classes, e.g. ``LagrangesMethod``.

        """
        pass  # 抽象方法，生成运动方程方法类所需的负载列表
    def __repr__(self):
        """Default representation of a pathway."""
        # 将所有附件对象的字符串表示形式连接成一个逗号分隔的字符串
        attachments = ', '.join(str(a) for a in self.attachments)
        # 返回路径对象的默认字符串表示形式，包括类名和附件列表
        return f'{self.__class__.__name__}({attachments})'
# 创建一个名为 LinearPathway 的类，继承自 PathwayBase 类
class LinearPathway(PathwayBase):
    """Linear pathway between a pair of attachment points.

    Explanation
    ===========

    A linear pathway forms a straight-line segment between two points and is
    the simplest pathway that can be formed. It will not interact with any
    other objects in the system, i.e. a ``LinearPathway`` will intersect other
    objects to ensure that the path between its two ends (its attachments) is
    the shortest possible.

    A linear pathway is made up of two points that can move relative to each
    other, and a pair of equal and opposite forces acting on the points. If the
    positive time-varying Euclidean distance between the two points is defined,
    then the "extension velocity" is the time derivative of this distance. The
    extension velocity is positive when the two points are moving away from
    each other and negative when moving closer to each other. The direction for
    the force acting on either point is determined by constructing a unit
    vector directed from the other point to this point. This establishes a sign
    convention such that a positive force magnitude tends to push the points
    apart. The following diagram shows the positive force sense and the
    distance between the points::

       P           Q
       o<--- F --->o
       |           |
       |<--l(t)--->|

    Examples
    ========

    >>> from sympy.physics.mechanics import LinearPathway

    To construct a pathway, two points are required to be passed to the
    ``attachments`` parameter as a ``tuple``.

    >>> from sympy.physics.mechanics import Point
    >>> pA, pB = Point('pA'), Point('pB')
    >>> linear_pathway = LinearPathway(pA, pB)
    >>> linear_pathway
    LinearPathway(pA, pB)

    The pathway created above isn't very interesting without the positions and
    velocities of its attachment points being described. Without this its not
    possible to describe how the pathway moves, i.e. its length or its
    extension velocity.

    >>> from sympy.physics.mechanics import ReferenceFrame
    >>> from sympy.physics.vector import dynamicsymbols
    >>> N = ReferenceFrame('N')
    >>> q = dynamicsymbols('q')
    >>> pB.set_pos(pA, q*N.x)
    >>> pB.pos_from(pA)
    q(t)*N.x

    A pathway's length can be accessed via its ``length`` attribute.

    >>> linear_pathway.length
    sqrt(q(t)**2)

    Note how what appears to be an overly-complex expression is returned. This
    is actually required as it ensures that a pathway's length is always
    positive.

    A pathway's extension velocity can be accessed similarly via its
    ``extension_velocity`` attribute.

    >>> linear_pathway.extension_velocity
    sqrt(q(t)**2)*Derivative(q(t), t)/q(t)

    Parameters
    ==========

    """
    
    # 初始化 LinearPathway 类，接受两个点对象作为参数
    def __init__(self, pointA, pointB):
        # 调用父类 PathwayBase 的初始化方法
        super().__init__()
        # 将点对象 pointA 和 pointB 作为路径的附着点
        self.attachments = (pointA, pointB)
        # 计算路径的长度，使用点对象的距离方法
        self.length = pointA.pos_from(pointB).magnitude()
        # 计算路径的扩展速度，使用点对象的速度方法和距离的时间导数
        self.extension_velocity = self.length * pointA.vel(pointB).magnitude() / self.length.magnitude()
    attachments : tuple[Point, Point]
        Pair of ``Point`` objects between which the linear pathway spans.
        Constructor expects two points to be passed, e.g.
        ``LinearPathway(Point('pA'), Point('pB'))``. More or fewer points will
        cause an error to be thrown.



    """
    定义了一个类变量 `attachments`，它是一个包含两个 `Point` 对象的元组。
    这两个 `Point` 对象表示线性路径的起点和终点。
    构造函数期望传入这两个点，例如 `LinearPathway(Point('pA'), Point('pB'))`。
    如果传入的点数不是两个，将会抛出错误。
    """

    def __init__(self, *attachments):
        """
        Initializer for ``LinearPathway``.

        Parameters
        ==========

        attachments : Point
            Pair of ``Point`` objects between which the linear pathway spans.
            Constructor expects two points to be passed, e.g.
            ``LinearPathway(Point('pA'), Point('pB'))``. More or fewer points
            will cause an error to be thrown.

        """
        super().__init__(*attachments)



    @property
    def length(self):
        """
        Exact analytical expression for the pathway's length.
        """
        return _point_pair_length(*self.attachments)

    @property
    def extension_velocity(self):
        """
        Exact analytical expression for the pathway's extension velocity.
        """
        return _point_pair_extension_velocity(*self.attachments)



"""
定义了两个属性：
- `length` 属性返回路径长度的精确解析表达式。
- `extension_velocity` 属性返回路径延伸速度的精确解析表达式。
这两个属性都依赖于 `attachments` 中的两个 `Point` 对象来计算。
"""
    def to_loads(self, force):
        """
        Loads required by the equations of motion method classes.

        Explanation
        ===========

        ``KanesMethod`` requires a list of ``Point``-``Vector`` tuples to be
        passed to the ``loads`` parameters of its ``kanes_equations`` method
        when constructing the equations of motion. This method acts as a
        utility to produce the correctly-structured pairs of points and vectors
        required so that these can be easily concatenated with other items in
        the list of loads and passed to ``KanesMethod.kanes_equations``. These
        loads are also in the correct form to also be passed to the other
        equations of motion method classes, e.g. ``LagrangesMethod``.

        Examples
        ========

        The below example shows how to generate the loads produced in a linear
        actuator that produces an expansile force ``F``. First, create a linear
        actuator between two points separated by the coordinate ``q`` in the
        ``x`` direction of the global frame ``N``.

        >>> from sympy.physics.mechanics import (LinearPathway, Point,
        ...     ReferenceFrame)
        >>> from sympy.physics.vector import dynamicsymbols
        >>> q = dynamicsymbols('q')
        >>> N = ReferenceFrame('N')
        >>> pA, pB = Point('pA'), Point('pB')
        >>> pB.set_pos(pA, q*N.x)
        >>> linear_pathway = LinearPathway(pA, pB)

        Now create a symbol ``F`` to describe the magnitude of the (expansile)
        force that will be produced along the pathway. The list of loads that
        ``KanesMethod`` requires can be produced by calling the pathway's
        ``to_loads`` method with ``F`` passed as the only argument.

        >>> from sympy import symbols
        >>> F = symbols('F')
        >>> linear_pathway.to_loads(F)
        [(pA, - F*q(t)/sqrt(q(t)**2)*N.x), (pB, F*q(t)/sqrt(q(t)**2)*N.x)]

        Parameters
        ==========

        force : Expr
            Magnitude of the force acting along the length of the pathway. As
            per the sign conventions for the pathway length, pathway extension
            velocity, and pair of point forces, if this ``Expr`` is positive
            then the force will act to push the pair of points away from one
            another (it is expansile).
        """

        # Calculate the relative position vector between the attachment points
        relative_position = _point_pair_relative_position(*self.attachments)
        
        # Create the list of loads using the force and relative position
        loads = [
            Force(self.attachments[0], -force*relative_position/self.length),
            Force(self.attachments[-1], force*relative_position/self.length),
        ]
        
        # Return the generated loads
        return loads
class ObstacleSetPathway(PathwayBase):
    """Obstacle-set pathway between a set of attachment points.

    Explanation
    ===========

    An obstacle-set pathway forms a series of straight-line segment between
    pairs of consecutive points in a set of points. It is similiar to multiple
    linear pathways joined end-to-end. It will not interact with any other
    objects in the system, i.e. an ``ObstacleSetPathway`` will intersect other
    objects to ensure that the path between its pairs of points (its
    attachments) is the shortest possible.

    Examples
    ========

    To construct an obstacle-set pathway, three or more points are required to
    be passed to the ``attachments`` parameter as a ``tuple``.

    >>> from sympy.physics.mechanics import ObstacleSetPathway, Point
    >>> pA, pB, pC, pD = Point('pA'), Point('pB'), Point('pC'), Point('pD')
    >>> obstacle_set_pathway = ObstacleSetPathway(pA, pB, pC, pD)
    >>> obstacle_set_pathway
    ObstacleSetPathway(pA, pB, pC, pD)

    The pathway created above isn't very interesting without the positions and
    velocities of its attachment points being described. Without this its not
    possible to describe how the pathway moves, i.e. its length or its
    extension velocity.

    >>> from sympy import cos, sin
    >>> from sympy.physics.mechanics import ReferenceFrame
    >>> from sympy.physics.vector import dynamicsymbols
    >>> N = ReferenceFrame('N')
    >>> q = dynamicsymbols('q')
    >>> pO = Point('pO')
    >>> pA.set_pos(pO, N.y)
    >>> pB.set_pos(pO, -N.x)
    >>> pC.set_pos(pA, cos(q) * N.x - (sin(q) + 1) * N.y)
    >>> pD.set_pos(pA, sin(q) * N.x + (cos(q) - 1) * N.y)
    >>> pB.pos_from(pA)
    - N.x - N.y
    >>> pC.pos_from(pA)
    cos(q(t))*N.x + (-sin(q(t)) - 1)*N.y
    >>> pD.pos_from(pA)
    sin(q(t))*N.x + (cos(q(t)) - 1)*N.y

    A pathway's length can be accessed via its ``length`` attribute.

    >>> obstacle_set_pathway.length.simplify()
    sqrt(2)*(sqrt(cos(q(t)) + 1) + 2)

    A pathway's extension velocity can be accessed similarly via its
    ``extension_velocity`` attribute.

    >>> obstacle_set_pathway.extension_velocity.simplify()
    -sqrt(2)*sin(q(t))*Derivative(q(t), t)/(2*sqrt(cos(q(t)) + 1))

    Parameters
    ==========

    attachments : tuple[Point, Point]
        The set of ``Point`` objects that define the segmented obstacle-set
        pathway.

    """

    def __init__(self, *attachments):
        """Initializer for ``ObstacleSetPathway``.

        Parameters
        ==========

        attachments : tuple[Point, ...]
            The set of ``Point`` objects that define the segmented obstacle-set
            pathway.

        """
        # 调用父类的初始化方法，将所有的 attachments 参数传递给父类的初始化方法
        super().__init__(*attachments)

    @property
    def attachments(self):
        """The set of points defining a pathway's segmented path."""
        # 返回该路径对象的附着点集合属性 _attachments
        return self._attachments

    @attachments.setter



        # 设置属性 attachments 的 setter 方法，用于设置路径对象的附着点集合
    def attachments(self, attachments):
        # 检查是否已经存在 `_attachments` 属性，若存在则抛出异常
        if hasattr(self, '_attachments'):
            msg = (
                f'Can\'t set attribute `attachments` to {repr(attachments)} '
                f'as it is immutable.'
            )
            raise AttributeError(msg)
        # 检查附件列表的长度是否小于等于2，若是则抛出异常
        if len(attachments) <= 2:
            msg = (
                f'Value {repr(attachments)} passed to `attachments` was an '
                f'iterable of length {len(attachments)}, must be an iterable '
                f'of length 3 or greater.'
            )
            raise ValueError(msg)
        # 遍历附件列表，检查每个附件是否为 Point 类型，若不是则抛出异常
        for i, point in enumerate(attachments):
            if not isinstance(point, Point):
                msg = (
                    f'Value {repr(point)} passed to `attachments` at index '
                    f'{i} was of type {type(point)}, must be {Point}.'
                )
                raise TypeError(msg)
        # 将附件列表转换为不可变元组，并赋值给 `_attachments` 属性
        self._attachments = tuple(attachments)

    @property
    def length(self):
        """Exact analytical expression for the pathway's length."""
        # 初始化路径长度为零
        length = S.Zero
        # 遍历附件点对，计算路径长度并累加
        attachment_pairs = zip(self.attachments[:-1], self.attachments[1:])
        for attachment_pair in attachment_pairs:
            length += _point_pair_length(*attachment_pair)
        return length

    @property
    def extension_velocity(self):
        """Exact analytical expression for the pathway's extension velocity."""
        # 初始化路径延伸速度为零
        extension_velocity = S.Zero
        # 遍历附件点对，计算路径延伸速度并累加
        attachment_pairs = zip(self.attachments[:-1], self.attachments[1:])
        for attachment_pair in attachment_pairs:
            extension_velocity += _point_pair_extension_velocity(*attachment_pair)
        return extension_velocity
# 定义一个类 `WrappingPathway`，它继承自 `PathwayBase`
class WrappingPathway(PathwayBase):
    """Pathway that wraps a geometry object.

    Explanation
    ===========

    A wrapping pathway interacts with a geometry object and forms a path that
    wraps smoothly along its surface. The wrapping pathway along the geometry
    object will be the geodesic that the geometry object defines based on the
    two points. It will not interact with any other objects in the system, i.e.
    a ``WrappingPathway`` will intersect other objects to ensure that the path
    between its two ends (its attachments) is the shortest possible.

    To explain the sign conventions used for pathway length, extension
    velocity, and direction of applied forces, we can ignore the geometry with
    which the wrapping pathway interacts. A wrapping pathway is made up of two
    points that can move relative to each other, and a pair of equal and
    opposite forces acting on the points. If the positive time-varying
    Euclidean distance between the two points is defined, then the "extension
    velocity" is the time derivative of this distance. The extension velocity
    is positive when the two points are moving away from each other and
    negative when moving closer to each other. The direction for the force
    acting on either point is determined by constructing a unit vector directed
    from the other point to this point. This establishes a sign convention such
    that a positive force magnitude tends to push the points apart. The
    following diagram shows the positive force sense and the distance between
    the points::

       P           Q
       o<--- F --->o
       |           |
       |<--l(t)--->|

    Examples
    ========

    >>> from sympy.physics.mechanics import WrappingPathway

    To construct a wrapping pathway, like other pathways, a pair of points must
    be passed, followed by an instance of a wrapping geometry class as a
    keyword argument. We'll use a cylinder with radius ``r`` and its axis
    parallel to ``N.x`` passing through a point ``pO``.

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import Point, ReferenceFrame, WrappingCylinder
    >>> r = symbols('r')
    >>> N = ReferenceFrame('N')
    >>> pA, pB, pO = Point('pA'), Point('pB'), Point('pO')
    >>> cylinder = WrappingCylinder(r, pO, N.x)
    >>> wrapping_pathway = WrappingPathway(pA, pB, cylinder)
    >>> wrapping_pathway
    WrappingPathway(pA, pB, geometry=WrappingCylinder(radius=r, point=pO,
        axis=N.x))

    Parameters
    ==========

    attachment_1 : Point
        First of the pair of ``Point`` objects between which the wrapping
        pathway spans.
    attachment_2 : Point
        Second of the pair of ``Point`` objects between which the wrapping
        pathway spans.
    geometry : WrappingGeometryBase
        Geometry about which the pathway wraps.

    """
    def __init__(self, attachment_1, attachment_2, geometry):
        """Initializer for ``WrappingPathway``.

        Parameters
        ==========

        attachment_1 : Point
            First of the pair of ``Point`` objects between which the wrapping
            pathway spans.
        attachment_2 : Point
            Second of the pair of ``Point`` objects between which the wrapping
            pathway spans.
        geometry : WrappingGeometryBase
            Geometry about which the pathway wraps.
            The geometry about which the pathway wraps.

        """
        # 调用父类的初始化方法，传递两个附件点作为参数
        super().__init__(attachment_1, attachment_2)
        # 将传入的 geometry 参数设置为对象的 geometry 属性
        self.geometry = geometry

    @property
    def geometry(self):
        """Geometry around which the pathway wraps."""
        # 返回对象的 geometry 属性值
        return self._geometry

    @geometry.setter
    def geometry(self, geometry):
        # 如果对象已经有 _geometry 属性，表示 geometry 属性不可更改，抛出 AttributeError
        if hasattr(self, '_geometry'):
            msg = (
                f'Can\'t set attribute `geometry` to {repr(geometry)} as it '
                f'is immutable.'
            )
            raise AttributeError(msg)
        # 如果传入的 geometry 不是 WrappingGeometryBase 类型，抛出 TypeError
        if not isinstance(geometry, WrappingGeometryBase):
            msg = (
                f'Value {repr(geometry)} passed to `geometry` was of type '
                f'{type(geometry)}, must be {WrappingGeometryBase}.'
            )
            raise TypeError(msg)
        # 将传入的 geometry 参数设置为对象的 _geometry 属性
        self._geometry = geometry

    @property
    def length(self):
        """Exact analytical expression for the pathway's length."""
        # 调用 geometry 对象的 geodesic_length 方法计算路径长度
        return self.geometry.geodesic_length(*self.attachments)

    @property
    def extension_velocity(self):
        """Exact analytical expression for the pathway's extension velocity."""
        # 使用路径长度对时间 t 求导数，计算路径的扩展速度
        return self.length.diff(dynamicsymbols._t)

    def __repr__(self):
        """Representation of a ``WrappingPathway``."""
        # 返回对象的字符串表示形式，包括附件点和 geometry 属性的信息
        attachments = ', '.join(str(a) for a in self.attachments)
        return (
            f'{self.__class__.__name__}({attachments}, '
            f'geometry={self.geometry})'
        )
# 计算两个点之间的相对位置向量
def _point_pair_relative_position(point_1, point_2):
    """The relative position between a pair of points."""
    return point_2.pos_from(point_1)

# 计算两点之间直线路径的长度
def _point_pair_length(point_1, point_2):
    """The length of the direct linear path between two points."""
    return _point_pair_relative_position(point_1, point_2).magnitude()

# 计算两点之间直线路径的扩展速度
def _point_pair_extension_velocity(point_1, point_2):
    """The extension velocity of the direct linear path between two points."""
    return _point_pair_length(point_1, point_2).diff(dynamicsymbols._t)
```