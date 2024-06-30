# `D:\src\scipysrc\sympy\sympy\physics\mechanics\loads.py`

```
# 从 abc 模块导入抽象基类 ABC
# namedtuple 是一个创建具名元组的函数，用于创建 LoadBase 类
# BodyBase 是 sympy.physics.mechanics.body_base 模块中的类，表示物体
# Vector, ReferenceFrame, Point 是 sympy.physics.vector 模块中的类

from abc import ABC
from collections import namedtuple
from sympy.physics.mechanics.body_base import BodyBase
from sympy.physics.vector import Vector, ReferenceFrame, Point

# __all__ 是一个列表，包含当前模块导出的公共名称
__all__ = ['LoadBase', 'Force', 'Torque']

# LoadBase 类继承自 ABC 抽象类，并且使用 namedtuple 创建具名元组
# 具名元组的字段包括 'location' 和 'vector'，用于表示载荷的位置和向量
class LoadBase(ABC, namedtuple('LoadBase', ['location', 'vector'])):
    """Abstract base class for the various loading types."""

    # 定义加法操作，不支持加法操作，抛出 TypeError 异常
    def __add__(self, other):
        raise TypeError(f"unsupported operand type(s) for +: "
                        f"'{self.__class__.__name__}' and "
                        f"'{other.__class__.__name__}'")

    # 定义乘法操作，不支持乘法操作，抛出 TypeError 异常
    def __mul__(self, other):
        raise TypeError(f"unsupported operand type(s) for *: "
                        f"'{self.__class__.__name__}' and "
                        f"'{other.__class__.__name__}'")

    # 右加法和右乘法与左加法和左乘法相同
    __radd__ = __add__
    __rmul__ = __mul__


# Force 类继承自 LoadBase 类，表示作用在点上的力
class Force(LoadBase):
    """Force acting upon a point.

    Explanation
    ===========

    A force is a vector that is bound to a line of action. This class stores
    both a point, which lies on the line of action, and the vector. A tuple can
    also be used, with the location as the first entry and the vector as second
    entry.

    Examples
    ========

    A force of magnitude 2 along N.x acting on a point Po can be created as
    follows:

    >>> from sympy.physics.mechanics import Point, ReferenceFrame, Force
    >>> N = ReferenceFrame('N')
    >>> Po = Point('Po')
    >>> Force(Po, 2 * N.x)
    (Po, 2*N.x)

    If a body is supplied, then the center of mass of that body is used.

    >>> from sympy.physics.mechanics import Particle
    >>> P = Particle('P', point=Po)
    >>> Force(P, 2 * N.x)
    (Po, 2*N.x)

    """

    # __new__ 方法用于创建新的 Force 实例
    def __new__(cls, point, force):
        # 如果 point 是 BodyBase 类的实例，则使用其质心作为 point
        if isinstance(point, BodyBase):
            point = point.masscenter
        # 如果 point 不是 Point 类的实例，则抛出 TypeError 异常
        if not isinstance(point, Point):
            raise TypeError('Force location should be a Point.')
        # 如果 force 不是 Vector 类的实例，则抛出 TypeError 异常
        if not isinstance(force, Vector):
            raise TypeError('Force vector should be a Vector.')
        # 调用父类 namedtuple 的 __new__ 方法创建实例
        return super().__new__(cls, point, force)

    # __repr__ 方法返回 Force 实例的字符串表示
    def __repr__(self):
        return (f'{self.__class__.__name__}(point={self.point}, '
                f'force={self.force})')

    # @property 修饰器使 point 方法成为属性访问器，返回位置信息
    @property
    def point(self):
        return self.location

    # @property 修饰器使 force 方法成为属性访问器，返回力向量信息
    @property
    def force(self):
        return self.vector


# Torque 类继承自 LoadBase 类，表示作用在参考框架上的力矩
class Torque(LoadBase):
    """Torque acting upon a frame.

    Explanation
    ===========

    A torque is a free vector that is acting on a reference frame, which is
    associated with a rigid body. This class stores both the frame and the
    vector. A tuple can also be used, with the location as the first item and
    the vector as second item.

    Examples
    ========

    A torque of magnitude 2 about N.x acting on a frame N can be created as
    follows:

    >>> from sympy.physics.mechanics import ReferenceFrame, Torque
    >>> N = ReferenceFrame('N')
    >>> Torque(N, 2 * N.x)
    (N, 2*N.x)

    If a body is supplied, then the frame fixed to that body is used.

    """

    # Torque 类的构造函数，与 Force 类似，但表示力矩的应用
    def __new__(cls, frame, torque):
        # 如果 frame 是 BodyBase 类的实例，则使用其固定的参考框架
        if isinstance(frame, BodyBase):
            frame = frame.frame
        # 如果 frame 不是 ReferenceFrame 类的实例，则抛出 TypeError 异常
        if not isinstance(frame, ReferenceFrame):
            raise TypeError('Torque frame should be a ReferenceFrame.')
        # 如果 torque 不是 Vector 类的实例，则抛出 TypeError 异常
        if not isinstance(torque, Vector):
            raise TypeError('Torque vector should be a Vector.')
        # 调用父类 namedtuple 的 __new__ 方法创建实例
        return super().__new__(cls, frame, torque)
    >>> from sympy.physics.mechanics import RigidBody
    # 导入 RigidBody 类，用于创建刚体对象

    >>> rb = RigidBody('rb', frame=N)
    # 创建名为 'rb' 的刚体对象，使用给定的参考框架 N

    >>> Torque(rb, 2 * N.x)
    # 创建 Torque 对象，作用在刚体 rb 上，力矩方向为 2*N.x

    """

    def __new__(cls, frame, torque):
        # 创建 Torque 类的新实例
        if isinstance(frame, BodyBase):
            # 如果 frame 是 BodyBase 的实例，则获取其关联的 ReferenceFrame
            frame = frame.frame
        if not isinstance(frame, ReferenceFrame):
            # 如果 frame 不是 ReferenceFrame 类型，则引发类型错误
            raise TypeError('Torque location should be a ReferenceFrame.')
        if not isinstance(torque, Vector):
            # 如果 torque 不是 Vector 类型，则引发类型错误
            raise TypeError('Torque vector should be a Vector.')
        return super().__new__(cls, frame, torque)

    def __repr__(self):
        # 返回 Torque 对象的字符串表示形式
        return (f'{self.__class__.__name__}(frame={self.frame}, '
                f'torque={self.torque})')

    @property
    def frame(self):
        # 返回 Torque 对象的 frame 属性，即作用位置的 ReferenceFrame
        return self.location

    @property
    def torque(self):
        # 返回 Torque 对象的 torque 属性，即作用的力矩向量
        return self.vector
# 返回一个列表，包含根据给定重力加速度和任意数量的粒子或刚体计算出的重力力列表
def gravity(acceleration, *bodies):
    gravity_force = []  # 初始化一个空列表，用于存储计算出的重力力
    for body in bodies:  # 遍历传入的所有物体
        if not isinstance(body, BodyBase):  # 检查物体是否为有效的物体类型
            raise TypeError(f'{type(body)} is not a body type')  # 如果不是有效的物体类型，抛出类型错误
        gravity_force.append(Force(body.masscenter, body.mass * acceleration))  # 计算并将重力力添加到列表中
    return gravity_force  # 返回计算出的重力力列表


# 辅助函数，用于解析载荷并将元组转换为载荷对象
def _parse_load(load):
    if isinstance(load, LoadBase):  # 如果载荷已经是有效的载荷对象，直接返回
        return load
    elif isinstance(load, tuple):  # 如果载荷是一个元组
        if len(load) != 2:  # 检查元组长度是否为2
            raise ValueError(f'Load {load} should have a length of 2.')  # 如果不是，抛出值错误
        if isinstance(load[0], Point):  # 如果元组的第一个元素是一个点对象
            return Force(load[0], load[1])  # 创建并返回一个力对象
        elif isinstance(load[0], ReferenceFrame):  # 如果元组的第一个元素是一个参考系对象
            return Torque(load[0], load[1])  # 创建并返回一个力矩对象
        else:
            raise ValueError(f'Load not recognized. The load location {load[0]}'  # 如果不是有效的载荷类型，抛出值错误
                             f' should either be a Point or a ReferenceFrame.')
    raise TypeError(f'Load type {type(load)} not recognized as a load. It '  # 如果载荷类型不被识别，抛出类型错误
                    f'should be a Force, Torque or tuple.')
```