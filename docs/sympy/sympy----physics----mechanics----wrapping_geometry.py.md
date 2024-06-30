# `D:\src\scipysrc\sympy\sympy\physics\mechanics\wrapping_geometry.py`

```
# 导入 ABC 抽象基类和 abstractmethod 装饰器
from abc import ABC, abstractmethod

# 导入 sympy 库中的具体数学函数和类
from sympy import Integer, acos, pi, sqrt, sympify, tan

# 导入 sympy 库中的关系运算符 Eq
from sympy.core.relational import Eq

# 导入 sympy 库中的三角函数 atan2
from sympy.functions.elementary.trigonometric import atan2

# 导入 sympy 库中的多项式处理函数 cancel
from sympy.polys.polytools import cancel

# 导入 sympy.physics.vector 库中的 Vector 和 dot
from sympy.physics.vector import Vector, dot

# 导入 sympy.simplify.simplify 库中的 trigsimp
from sympy.simplify.simplify import trigsimp

# 定义模块级别变量 __all__，表示在 from 模块 import * 时导入的符号
__all__ = [
    'WrappingGeometryBase',
    'WrappingCylinder',
    'WrappingSphere',
]

# 定义一个抽象基类 WrappingGeometryBase，继承自 ABC 类
class WrappingGeometryBase(ABC):
    """Abstract base class for all geometry classes to inherit from.

    Notes
    =====

    Instances of this class cannot be directly instantiated by users. However,
    it can be used to created custom geometry types through subclassing.

    """

    # 抽象属性，表示与几何体相关联的点
    @property
    @abstractmethod
    def point(cls):
        """The point with which the geometry is associated."""
        pass

    # 抽象方法，返回一个布尔值表示点是否在几何体表面上
    @abstractmethod
    def point_on_surface(self, point):
        """Returns ``True`` if a point is on the geometry's surface.

        Parameters
        ==========
        point : Point
            The point for which it's to be ascertained if it's on the
            geometry's surface or not.

        """
        pass

    # 抽象方法，返回两点之间在几何体表面上的最短距离
    @abstractmethod
    def geodesic_length(self, point_1, point_2):
        """Returns the shortest distance between two points on a geometry's
        surface.

        Parameters
        ==========

        point_1 : Point
            The point from which the geodesic length should be calculated.
        point_2 : Point
            The point to which the geodesic length should be calculated.

        """
        pass

    # 抽象方法，返回两点处几何体表面上与测地线平行的向量
    @abstractmethod
    def geodesic_end_vectors(self, point_1, point_2):
        """The vectors parallel to the geodesic at the two end points.

        Parameters
        ==========

        point_1 : Point
            The point from which the geodesic originates.
        point_2 : Point
            The point at which the geodesic terminates.

        """
        pass

    # 默认的 __repr__ 方法，返回几何体对象的默认字符串表示形式
    def __repr__(self):
        """Default representation of a geometry object."""
        return f'{self.__class__.__name__}()'


# 定义一个具体的类 WrappingSphere，继承自 WrappingGeometryBase 抽象基类
class WrappingSphere(WrappingGeometryBase):
    """A solid spherical object.

    Explanation
    ===========

    A wrapping geometry that allows for circular arcs to be defined between
    pairs of points. These paths are always geodetic (the shortest possible).

    Examples
    ========

    To create a ``WrappingSphere`` instance, a ``Symbol`` denoting its radius
    and ``Point`` at which its center will be located are needed:

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import Point, WrappingSphere
    >>> r = symbols('r')
    >>> pO = Point('pO')

    A sphere with radius ``r`` centered on ``pO`` can be instantiated with:

    >>> WrappingSphere(r, pO)
    WrappingSphere(radius=r, point=pO)

    Parameters
    ==========


    """

    # 初始化方法，设置球体的半径和中心点
    def __init__(self, radius, point):
        self.radius = radius  # 球体的半径
        self.point = point    # 球体的中心点

    # 实现抽象基类中的 point 属性，返回球体的中心点
    @property
    def point(self):
        """The point with which the geometry is associated."""
        return self._point

    # 实现抽象基类中的 point 属性的 setter 方法
    @point.setter
    def point(self, value):
        if not isinstance(value, Point):
            raise TypeError("Point must be an instance of sympy.physics.vector.Point")
        self._point = value

    # 实现抽象基类中的 point_on_surface 方法，判断给定点是否在球体表面上
    def point_on_surface(self, point):
        """Returns ``True`` if a point is on the geometry's surface.

        Parameters
        ==========
        point : Point
            The point for which it's to be ascertained if it's on the
            geometry's surface or not.

        """
        return Eq(dot(point.pos_from(self.point), point.pos_from(self.point)), self.radius**2)

    # 实现抽象基类中的 geodesic_length 方法，返回两点在球体表面上的最短距离
    def geodesic_length(self, point_1, point_2):
        """Returns the shortest distance between two points on a geometry's
        surface.

        Parameters
        ==========

        point_1 : Point
            The point from which the geodesic length should be calculated.
        point_2 : Point
            The point to which the geodesic length should be calculated.

        """
        return acos(dot(point_1.pos_from(self.point).normalize(), point_2.pos_from(self.point).normalize())) * self.radius

    # 实现抽象基类中的 geodesic_end_vectors 方法，返回两点处球体表面上的测地线平行向量
    def geodesic_end_vectors(self, point_1, point_2):
        """The vectors parallel to the geodesic at the two end points.

        Parameters
        ==========

        point_1 : Point
            The point from which the geodesic originates.
        point_2 : Point
            The point at which the geodesic terminates.

        """
        v1 = point_1.pos_from(self.point).normalize() * self.radius
        v2 = point_2.pos_from(self.point).normalize() * self.radius
        return v1, v2

    # 返回球体对象的字符串表示形式
    def __repr__(self):
        """Default representation of a geometry object."""
        return f'{self.__class__.__name__}(radius={self.radius}, point={self.point})'
    radius : Symbol
        Radius of the sphere. This symbol must represent a value that is
        positive and constant, i.e. it cannot be a dynamic symbol, nor can it
        be an expression.
    point : Point
        A point at which the sphere is centered.

    See Also
    ========

    WrappingCylinder: Cylindrical geometry where the wrapping direction can be
        defined.

    """

    # 定义一个包装球体的类
    def __init__(self, radius, point):
        """Initializer for ``WrappingSphere``.

        Parameters
        ==========

        radius : Symbol
            The radius of the sphere.
        point : Point
            A point on which the sphere is centered.

        """
        # 初始化方法，设置球体的半径和中心点
        self.radius = radius
        self.point = point

    @property
    def radius(self):
        """Radius of the sphere."""
        # 获取球体的半径属性
        return self._radius

    @radius.setter
    def radius(self, radius):
        # 设置球体的半径属性
        self._radius = radius

    @property
    def point(self):
        """A point on which the sphere is centered."""
        # 获取球体的中心点属性
        return self._point

    @point.setter
    def point(self, point):
        # 设置球体的中心点属性
        self._point = point

    def point_on_surface(self, point):
        """Returns ``True`` if a point is on the sphere's surface.

        Parameters
        ==========

        point : Point
            The point for which it's to be ascertained if it's on the sphere's
            surface or not. This point's position relative to the sphere's
            center must be a simple expression involving the radius of the
            sphere, otherwise this check will likely not work.

        """
        # 判断给定点是否在球体表面上
        point_vector = point.pos_from(self.point)
        if isinstance(point_vector, Vector):
            point_radius_squared = dot(point_vector, point_vector)
        else:
            point_radius_squared = point_vector**2
        return Eq(point_radius_squared, self.radius**2) == True

    def geodesic_end_vectors(self, point_1, point_2):
        """The vectors parallel to the geodesic at the two end points.

        Parameters
        ==========

        point_1 : Point
            The point from which the geodesic originates.
        point_2 : Point
            The point at which the geodesic terminates.

        """
        # 计算两个端点处的测地线的平行向量
        pA, pB = point_1, point_2
        pO = self.point
        pA_vec = pA.pos_from(pO)
        pB_vec = pB.pos_from(pO)

        if pA_vec.cross(pB_vec) == 0:
            # 如果两点在球面上是对径的，则测地线未定义
            msg = (
                f'Can\'t compute geodesic end vectors for the pair of points '
                f'{pA} and {pB} on a sphere {self} as they are diametrically '
                f'opposed, thus the geodesic is not defined.'
            )
            raise ValueError(msg)

        return (
            pA_vec.cross(pB.pos_from(pA)).cross(pA_vec).normalize(),
            pB_vec.cross(pA.pos_from(pB)).cross(pB_vec).normalize(),
        )
    def __repr__(self):
        """Representation of a ``WrappingSphere``."""
        # 返回一个表示``WrappingSphere``对象的字符串
        return (
            f'{self.__class__.__name__}(radius={self.radius}, '
            f'point={self.point})'
        )
class WrappingCylinder(WrappingGeometryBase):
    """A solid (infinite) cylindrical object.

    Explanation
    ===========

    A wrapping geometry that allows for circular arcs to be defined between
    pairs of points. These paths are always geodetic (the shortest possible) in
    the sense that they will be a straight line on the unwrapped cylinder's
    surface. However, it is also possible for a direction to be specified, i.e.
    paths can be influenced such that they either wrap along the shortest side
    or the longest side of the cylinder. To define these directions, rotations
    are in the positive direction following the right-hand rule.

    Examples
    ========

    To create a ``WrappingCylinder`` instance, a ``Symbol`` denoting its
    radius, a ``Vector`` defining its axis, and a ``Point`` through which its
    axis passes are needed:

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import (Point, ReferenceFrame,
    ...     WrappingCylinder)
    >>> N = ReferenceFrame('N')
    >>> r = symbols('r')
    >>> pO = Point('pO')
    >>> ax = N.x

    A cylinder with radius ``r``, and axis parallel to ``N.x`` passing through
    ``pO`` can be instantiated with:

    >>> WrappingCylinder(r, pO, ax)
    WrappingCylinder(radius=r, point=pO, axis=N.x)

    Parameters
    ==========

    radius : Symbol
        The radius of the cylinder.
    point : Point
        A point through which the cylinder's axis passes.
    axis : Vector
        The axis along which the cylinder is aligned.

    See Also
    ========

    WrappingSphere: Spherical geometry where the wrapping direction is always
        geodetic.

    """

    def __init__(self, radius, point, axis):
        """Initializer for ``WrappingCylinder``.

        Parameters
        ==========

        radius : Symbol
            The radius of the cylinder. This symbol must represent a value that
            is positive and constant, i.e. it cannot be a dynamic symbol.
        point : Point
            A point through which the cylinder's axis passes.
        axis : Vector
            The axis along which the cylinder is aligned.

        """
        # 将传入的参数分别赋给对象的属性
        self.radius = radius
        self.point = point
        # 对轴向量进行归一化处理，并将结果赋给对象的属性
        self.axis = axis.normalize()

    @property
    def radius(self):
        """Radius of the cylinder."""
        return self._radius

    @radius.setter
    def radius(self, radius):
        # 设置半径属性
        self._radius = radius

    @property
    def point(self):
        """A point through which the cylinder's axis passes."""
        return self._point

    @point.setter
    def point(self, point):
        # 设置穿过圆柱轴的点属性
        self._point = point

    @property
    def axis(self):
        """Axis along which the cylinder is aligned."""
        return self._axis

    @axis.setter
    def axis(self, axis):
        # 设置圆柱沿轴对齐的属性，并将轴向量归一化
        self._axis = axis.normalize()
    def point_on_surface(self, point):
        """Returns ``True`` if a point is on the cylinder's surface.

        Parameters
        ==========

        point : Point
            The point for which it's to be ascertained if it's on the
            cylinder's surface or not. This point's position relative to the
            cylinder's axis must be a simple expression involving the radius of
            the cylinder, otherwise this check will likely not work.

        """
        # 计算点相对于圆柱体中心的相对位置向量
        relative_position = point.pos_from(self.point)
        # 计算相对位置向量在圆柱轴向上的投影向量
        parallel = relative_position.dot(self.axis) * self.axis
        # 计算点相对于轴的垂直向量
        point_vector = relative_position - parallel
        if isinstance(point_vector, Vector):
            # 如果是向量，计算其长度的平方
            point_radius_squared = dot(point_vector, point_vector)
        else:
            # 如果不是向量，直接平方
            point_radius_squared = point_vector**2
        # 返回点是否在圆柱表面的判断结果
        return Eq(trigsimp(point_radius_squared), self.radius**2) == True

    def geodesic_end_vectors(self, point_1, point_2):
        """The vectors parallel to the geodesic at the two end points.

        Parameters
        ==========

        point_1 : Point
            The point from which the geodesic originates.
        point_2 : Point
            The point at which the geodesic terminates.

        """
        # 计算点1和点2相对于圆柱体中心的位置向量
        point_1_from_origin_point = point_1.pos_from(self.point)
        point_2_from_origin_point = point_2.pos_from(self.point)

        # 如果点1和点2重合，则无法计算测地线的端点向量
        if point_1_from_origin_point == point_2_from_origin_point:
            msg = (
                f'Cannot compute geodesic end vectors for coincident points '
                f'{point_1} and {point_2} as no geodesic exists.'
            )
            raise ValueError(msg)

        # 计算点1和点2在圆柱轴向上的投影向量
        point_1_parallel = point_1_from_origin_point.dot(self.axis) * self.axis
        point_2_parallel = point_2_from_origin_point.dot(self.axis) * self.axis
        # 计算点1和点2相对于轴的垂直向量
        point_1_normal = (point_1_from_origin_point - point_1_parallel)
        point_2_normal = (point_2_from_origin_point - point_2_parallel)

        # 如果点1和点2的垂直向量相同，则它们的垂直向量均设为零向量
        if point_1_normal == point_2_normal:
            point_1_perpendicular = Vector(0)
            point_2_perpendicular = Vector(0)
        else:
            # 计算点1和点2的垂直向量
            point_1_perpendicular = self.axis.cross(point_1_normal).normalize()
            point_2_perpendicular = -self.axis.cross(point_2_normal).normalize()

        # 计算测地线的长度
        geodesic_length = self.geodesic_length(point_1, point_2)
        # 计算点2相对于点1的位置向量
        relative_position = point_2.pos_from(point_1)
        # 计算相对位置向量在轴向上的投影长度
        parallel_length = relative_position.dot(self.axis)
        # 计算在平面上的弧长
        planar_arc_length = sqrt(geodesic_length**2 - parallel_length**2)

        # 计算点1和点2处测地线的向量
        point_1_vector = (
            planar_arc_length * point_1_perpendicular
            + parallel_length * self.axis
        ).normalize()
        point_2_vector = (
            planar_arc_length * point_2_perpendicular
            - parallel_length * self.axis
        ).normalize()

        # 返回点1和点2处测地线的向量
        return (point_1_vector, point_2_vector)
    def __repr__(self):
        """Representation of a ``WrappingCylinder``."""
        # 返回一个对象的字符串表示形式，用于调试和日志记录
        return (
            f'{self.__class__.__name__}(radius={self.radius}, '
            f'point={self.point}, axis={self.axis})'
        )
# 计算在几何上需要方向性的反正切函数

def _directional_atan(numerator, denominator):
    """Compute atan in a directional sense as required for geodesics.

    Explanation
    ===========
    
    To be able to control the direction of the geodesic length along the
    surface of a cylinder a dedicated arctangent function is needed that
    properly handles the directionality of different case. This function
    ensures that the central angle is always positive but shifting the case
    where ``atan2`` would return a negative angle to be centered around
    ``2*pi``.
    
    Notes
    =====
    
    This function only handles very specific cases, i.e. the ones that are
    expected to be encountered when calculating symbolic geodesics on uniformly
    curved surfaces. As such, ``NotImplemented`` errors can be raised in many
    cases. This function is named with a leader underscore to indicate that it
    only aims to provide very specific functionality within the private scope
    of this module.
    
    Parameters
    ==========
    numerator : object
        The numerator of the fraction for which the atan is being computed.
        Should be a symbolic or numeric value.
    denominator : object
        The denominator of the fraction for which the atan is being computed.
        Should be a symbolic or numeric value.
    
    Returns
    =======
    angle : object
        The computed directional atan value, ensuring positivity and proper
        handling of symbolic cases.
    
    Raises
    ======
    NotImplementedError
        When the function cannot handle the given combination of numerator
        and denominator types.
    """
    
    # 如果分子和分母都是数值类型，则直接计算 atan2
    if numerator.is_number and denominator.is_number:
        angle = atan2(numerator, denominator)
        # 如果计算出的角度小于0，则加上 2*pi 使其保持正向角度
        if angle < 0:
            angle += 2 * pi
    # 如果分子是数值而分母是符号类型，则抛出 Not Implemented 错误
    elif numerator.is_number:
        msg = (
            f'Cannot compute a directional atan when the numerator {numerator} '
            f'is numeric and the denominator {denominator} is symbolic.'
        )
        raise NotImplementedError(msg)
    # 如果分母是数值而分子是符号类型，则抛出 Not Implemented 错误
    elif denominator.is_number:
        msg = (
            f'Cannot compute a directional atan when the numerator {numerator} '
            f'is symbolic and the denominator {denominator} is numeric.'
        )
        raise NotImplementedError(msg)
    else:
        # 对于其他情况，先计算分子与分母的比值，并简化
        ratio = sympify(trigsimp(numerator / denominator))
        # 如果比值是正切函数，则取其参数作为角度
        if isinstance(ratio, tan):
            angle = ratio.args[0]
        # 如果比值是 -tan(theta)，则计算角度为 2*pi 减去 theta
        elif (
            ratio.is_Mul
            and ratio.args[0] == Integer(-1)
            and isinstance(ratio.args[1], tan)
        ):
            angle = 2 * pi - ratio.args[1].args[0]
        else:
            # 对于其他情况，抛出 Not Implemented 错误
            msg = f'Cannot compute a directional atan for the value {ratio}.'
            raise NotImplementedError(msg)
    
    # 返回计算得到的角度
    return angle
```