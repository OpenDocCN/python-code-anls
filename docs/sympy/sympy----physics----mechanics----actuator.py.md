# `D:\src\scipysrc\sympy\sympy\physics\mechanics\actuator.py`

```
"""Implementations of actuators for linked force and torque application."""

# 导入必要的模块和类
from abc import ABC, abstractmethod
from sympy import S, sympify
from sympy.physics.mechanics.joint import PinJoint
from sympy.physics.mechanics.loads import Torque
from sympy.physics.mechanics.pathway import PathwayBase
from sympy.physics.mechanics.rigidbody import RigidBody
from sympy.physics.vector import ReferenceFrame, Vector

# 定义可以被外部引用的类列表
__all__ = [
    'ActuatorBase',
    'ForceActuator',
    'LinearDamper',
    'LinearSpring',
    'TorqueActuator',
    'DuffingSpring'
]

# 定义抽象基类 ActuatorBase
class ActuatorBase(ABC):
    """Abstract base class for all actuator classes to inherit from.

    Notes
    =====

    Instances of this class cannot be directly instantiated by users. However,
    it can be used to created custom actuator types through subclassing.

    """

    def __init__(self):
        """Initializer for ``ActuatorBase``."""
        pass

    @abstractmethod
    def to_loads(self):
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
        pass

    def __repr__(self):
        """Default representation of an actuator."""
        return f'{self.__class__.__name__}()'

# 定义力作用器 ForceActuator 类
class ForceActuator(ActuatorBase):
    """Force-producing actuator.

    Explanation
    ===========

    A ``ForceActuator`` is an actuator that produces a (expansile) force along
    its length.

    A force actuator uses a pathway instance to determine the direction and
    number of forces that it applies to a system. Consider the simplest case
    where a ``LinearPathway`` instance is used. This pathway is made up of two
    points that can move relative to each other, and results in a pair of equal
    and opposite forces acting on the endpoints. If the positive time-varying
    Euclidean distance between the two points is defined, then the "extension
    velocity" is the time derivative of this distance. The extension velocity
    is positive when the two points are moving away from each other and
    negative when moving closer to each other. The direction for the force
    acting on either point is determined by constructing a unit vector directed
    from the other point to this point. This establishes a sign convention such
    that a positive force magnitude tends to push the points apart, this is the
    def __init__(self, force, pathway):
        """Initializer for ``ForceActuator``.

        Parameters
        ==========

        force : Expr
            The scalar expression defining the (expansile) force that the
            actuator produces.
        pathway : PathwayBase
            The pathway that the actuator follows. This must be an instance of
            a concrete subclass of ``PathwayBase``, e.g. ``LinearPathway``.

        """
        # 将传入的力和路径赋值给实例变量
        self.force = force
        self.pathway = pathway

    @property
    def force(self):
        """The magnitude of the force produced by the actuator."""
        return self._force

    @force.setter
    def force(self, force):
        # 如果已经设置过力量属性，则抛出不可变属性的错误
        if hasattr(self, '_force'):
            msg = (
                f'Can\'t set attribute `force` to {repr(force)} as it is '
                f'immutable.'
            )
            raise AttributeError(msg)
        # 否则，将传入的力量表达式转换为符号表达式，并赋值给实例变量
        self._force = sympify(force, strict=True)

    @property
    def pathway(self):
        """The ``Pathway`` defining the actuator's line of action."""
        return self._pathway

    @pathway.setter
    def pathway(self, pathway):
        # 将传入的路径对象赋值给实例变量
        self._pathway = pathway
    # 定义一个方法 `pathway`，用于设置对象的路径
    def pathway(self, pathway):
        # 检查对象是否具有 `_pathway` 属性，如果有，说明路径属性是不可变的，抛出异常
        if hasattr(self, '_pathway'):
            msg = (
                f'Can\'t set attribute `pathway` to {repr(pathway)} as it is '
                f'immutable.'
            )
            raise AttributeError(msg)
        # 检查传入的路径 `pathway` 是否是 `PathwayBase` 类型的实例，如果不是，抛出类型错误异常
        if not isinstance(pathway, PathwayBase):
            msg = (
                f'Value {repr(pathway)} passed to `pathway` was of type '
                f'{type(pathway)}, must be {PathwayBase}.'
            )
            raise TypeError(msg)
        # 将传入的路径 `pathway` 赋值给对象的 `_pathway` 属性
        self._pathway = pathway

    # 定义对象的字符串表示形式 `__repr__` 方法
    def __repr__(self):
        """Representation of a ``ForceActuator``."""
        # 返回对象的类名以及对象的 `force` 和 `pathway` 属性的字符串表示
        return f'{self.__class__.__name__}({self.force}, {self.pathway})'
# 定义一个线性弹簧类，继承自 ForceActuator 类
class LinearSpring(ForceActuator):
    """A spring with its spring force as a linear function of its length.

    Explanation
    ===========

    Note that the "linear" in the name ``LinearSpring`` refers to the fact that
    the spring force is a linear function of the springs length. I.e. for a
    linear spring with stiffness ``k``, distance between its ends of ``x``, and
    an equilibrium length of ``0``, the spring force will be ``-k*x``, which is
    a linear function in ``x``. To create a spring that follows a linear, or
    straight, pathway between its two ends, a ``LinearPathway`` instance needs
    to be passed to the ``pathway`` parameter.

    A ``LinearSpring`` is a subclass of ``ForceActuator`` and so follows the
    same sign conventions for length, extension velocity, and the direction of
    the forces it applies to its points of attachment on bodies. The sign
    convention for the direction of forces is such that, for the case where a
    linear spring is instantiated with a ``LinearPathway`` instance as its
    pathway, they act to push the two ends of the spring away from one another.
    Because springs produces a contractile force and acts to pull the two ends
    together towards the equilibrium length when stretched, the scalar portion
    of the forces on the endpoint are negative in order to flip the sign of the
    forces on the endpoints when converted into vector quantities. The
    following diagram shows the positive force sense and the distance between
    the points::

       P           Q
       o<--- F --->o
       |           |
       |<--l(t)--->|

    Examples
    ========

    To construct a linear spring, an expression (or symbol) must be supplied to
    represent the stiffness (spring constant) of the spring, alongside a
    pathway specifying its line of action. Let's also create a global reference
    frame and spatially fix one of the points in it while setting the other to
    be positioned such that it can freely move in the frame's x direction
    specified by the coordinate ``q``.

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import (LinearPathway, LinearSpring,
    ...     Point, ReferenceFrame)
    >>> from sympy.physics.vector import dynamicsymbols
    >>> N = ReferenceFrame('N')
    >>> q = dynamicsymbols('q')
    >>> stiffness = symbols('k')
    >>> pA, pB = Point('pA'), Point('pB')
    >>> pA.set_vel(N, 0)
    >>> pB.set_pos(pA, q*N.x)
    >>> pB.pos_from(pA)
    q(t)*N.x
    >>> linear_pathway = LinearPathway(pA, pB)
    >>> spring = LinearSpring(stiffness, linear_pathway)
    >>> spring
    LinearSpring(k, LinearPathway(pA, pB))

    This spring will produce a force that is proportional to both its stiffness
    and the pathway's length. Note that this force is negative as SymPy's sign
    convention for actuators is that negative forces are contractile.

    >>> spring.force
    -k*sqrt(q(t)**2)
    """

    def __init__(self, stiffness, pathway):
        # 调用父类的构造函数初始化力作用器
        super().__init__(pathway)
        # 弹簧的刚度常数
        self.stiffness = stiffness

    @property
    def force(self):
        # 返回弹簧的力，这里使用了 SymPy 的符号计算来表达力的大小
        return -self.stiffness * sqrt(self.pathway.length**2)
    To create a linear spring with a non-zero equilibrium length, an expression
    (or symbol) can be passed to the ``equilibrium_length`` parameter on
    construction on a ``LinearSpring`` instance. Let's create a symbol ``l``
    to denote a non-zero equilibrium length and create another linear spring.

    >>> l = symbols('l')
    >>> spring = LinearSpring(stiffness, linear_pathway, equilibrium_length=l)
    >>> spring
    LinearSpring(k, LinearPathway(pA, pB), equilibrium_length=l)

    The spring force of this new spring is again proportional to both its
    stiffness and the pathway's length. However, the spring will not produce
    any force when ``q(t)`` equals ``l``. Note that the force will become
    expansile when ``q(t)`` is less than ``l``, as expected.

    >>> spring.force
    -k*(-l + sqrt(q(t)**2))

    Parameters
    ==========

    stiffness : Expr
        The spring constant.
    pathway : PathwayBase
        The pathway that the actuator follows. This must be an instance of a
        concrete subclass of ``PathwayBase``, e.g. ``LinearPathway``.
    equilibrium_length : Expr, optional
        The length at which the spring is in equilibrium, i.e. it produces no
        force. The default value is 0, i.e. the spring force is a linear
        function of the pathway's length with no constant offset.

    See Also
    ========

    ForceActuator: force-producing actuator (superclass of ``LinearSpring``).
    LinearPathway: straight-line pathway between a pair of points.

    """

    # 初始化 LinearSpring 类的构造函数
    def __init__(self, stiffness, pathway, equilibrium_length=S.Zero):
        """Initializer for ``LinearSpring``.

        Parameters
        ==========

        stiffness : Expr
            The spring constant.
        pathway : PathwayBase
            The pathway that the actuator follows. This must be an instance of
            a concrete subclass of ``PathwayBase``, e.g. ``LinearPathway``.
        equilibrium_length : Expr, optional
            The length at which the spring is in equilibrium, i.e. it produces
            no force. The default value is 0, i.e. the spring force is a linear
            function of the pathway's length with no constant offset.

        """
        # 设置弹簧的刚度
        self.stiffness = stiffness
        # 设置弹簧的路径
        self.pathway = pathway
        # 设置弹簧的平衡长度
        self.equilibrium_length = equilibrium_length

    @property
    def force(self):
        """The spring force produced by the linear spring."""
        # 返回弹簧产生的力的计算公式
        return -self.stiffness*(self.pathway.length - self.equilibrium_length)

    @force.setter
    def force(self, force):
        # 强制防止设置计算属性 'force'
        raise AttributeError('Can\'t set computed attribute `force`.')

    @property
    def stiffness(self):
        """The spring constant for the linear spring."""
        return self._stiffness

    @stiffness.setter
    # 定义一个方法用于设置弹簧的刚度
    def stiffness(self, stiffness):
        # 如果对象已经有了 `_stiffness` 属性，说明刚度属性不可变，抛出异常
        if hasattr(self, '_stiffness'):
            msg = (
                f'Can\'t set attribute `stiffness` to {repr(stiffness)} as it '
                f'is immutable.'
            )
            raise AttributeError(msg)
        # 将传入的刚度参数转化为符号表达式，并赋给对象的 `_stiffness` 属性
        self._stiffness = sympify(stiffness, strict=True)

    @property
    # 返回弹簧的平衡长度，即它不产生任何力的长度
    def equilibrium_length(self):
        """The length of the spring at which it produces no force."""
        return self._equilibrium_length

    @equilibrium_length.setter
    # 设置弹簧的平衡长度属性
    def equilibrium_length(self, equilibrium_length):
        # 如果对象已经有了 `_equilibrium_length` 属性，说明平衡长度属性不可变，抛出异常
        if hasattr(self, '_equilibrium_length'):
            msg = (
                f'Can\'t set attribute `equilibrium_length` to '
                f'{repr(equilibrium_length)} as it is immutable.'
            )
            raise AttributeError(msg)
        # 将传入的平衡长度参数转化为符号表达式，并赋给对象的 `_equilibrium_length` 属性
        self._equilibrium_length = sympify(equilibrium_length, strict=True)

    # 返回表示 `LinearSpring` 对象的字符串表示形式
    def __repr__(self):
        """Representation of a ``LinearSpring``."""
        # 根据对象的属性生成对象的字符串表示形式
        string = f'{self.__class__.__name__}({self.stiffness}, {self.pathway}'
        # 如果平衡长度为零，则只添加基本信息，否则添加平衡长度信息
        if self.equilibrium_length == S.Zero:
            string += ')'
        else:
            string += f', equilibrium_length={self.equilibrium_length})'
        return string
class LinearDamper(ForceActuator):
    """A damper whose force is a linear function of its extension velocity.

    Explanation
    ===========

    Note that the "linear" in the name ``LinearDamper`` refers to the fact that
    the damping force is a linear function of the damper's rate of change in
    its length. I.e. for a linear damper with damping ``c`` and extension
    velocity ``v``, the damping force will be ``-c*v``, which is a linear
    function in ``v``. To create a damper that follows a linear, or straight,
    pathway between its two ends, a ``LinearPathway`` instance needs to be
    passed to the ``pathway`` parameter.

    A ``LinearDamper`` is a subclass of ``ForceActuator`` and so follows the
    same sign conventions for length, extension velocity, and the direction of
    the forces it applies to its points of attachment on bodies. The sign
    convention for the direction of forces is such that, for the case where a
    linear damper is instantiated with a ``LinearPathway`` instance as its
    pathway, they act to push the two ends of the damper away from one another.
    Because dampers produce a force that opposes the direction of change in
    length, when extension velocity is positive the scalar portions of the
    forces applied at the two endpoints are negative in order to flip the sign
    of the forces on the endpoints wen converted into vector quantities. When
    extension velocity is negative (i.e. when the damper is shortening), the
    scalar portions of the fofces applied are also negative so that the signs
    cancel producing forces on the endpoints that are in the same direction as
    the positive sign convention for the forces at the endpoints of the pathway
    (i.e. they act to push the endpoints away from one another). The following
    diagram shows the positive force sense and the distance between the
    points::

       P           Q
       o<--- F --->o
       |           |
       |<--l(t)--->|

    Examples
    ========

    To construct a linear damper, an expression (or symbol) must be supplied to
    represent the damping coefficient of the damper (we'll use the symbol
    ``c``), alongside a pathway specifying its line of action. Let's also
    create a global reference frame and spatially fix one of the points in it
    while setting the other to be positioned such that it can freely move in
    the frame's x direction specified by the coordinate ``q``. The velocity
    that the two points move away from one another can be specified by the
    coordinate ``u`` where ``u`` is the first time derivative of ``q``
    (i.e., ``u = Derivative(q(t), t)``).

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import (LinearDamper, LinearPathway,
    ...     Point, ReferenceFrame)
    >>> from sympy.physics.vector import dynamicsymbols
    >>> N = ReferenceFrame('N')
    创建一个名为 'N' 的参考坐标系 N
    >>> q = dynamicsymbols('q')
    创建一个描述位置的动态符号 q
    >>> damping = symbols('c')
    创建一个描述阻尼系数的符号 c
    >>> pA, pB = Point('pA'), Point('pB')
    创建两个名为 'pA' 和 'pB' 的点 pA 和 pB
    # 设置点 P 的速度为零，其中 N 是一个参考坐标系
    >>> pA.set_vel(N, 0)
    # 设置点 P 的位置为点 P_A 的位置加上 q*N.x
    >>> pB.set_pos(pA, q*N.x)
    # 计算点 P_B 相对于点 P_A 的位置向量
    >>> pB.pos_from(pA)
    q(t)*N.x
    # 计算点 P_B 相对于参考坐标系 N 的速度
    >>> pB.vel(N)
    Derivative(q(t), t)*N.x
    # 创建一个线性路径对象 linear_pathway，连接点 P_A 和点 P_B
    >>> linear_pathway = LinearPathway(pA, pB)
    # 创建一个线性阻尼器对象 damper，使用给定的阻尼常数和线性路径对象
    >>> damper = LinearDamper(damping, linear_pathway)
    # 打印 damper 对象的字符串表示形式
    >>> damper
    LinearDamper(c, LinearPathway(pA, pB))

    # 此阻尼器将产生一个力，与其阻尼系数和路径的延伸长度成比例。注意力的方向为负，因为 SymPy 中的执行器符号约定为负力是收缩的，阻尼器的阻尼力将反对长度变化的方向。
    >>> damper.force
    -c*sqrt(q(t)**2)*Derivative(q(t), t)/q(t)

    Parameters
    ==========

    damping : Expr
        阻尼常数。
    pathway : PathwayBase
        执行器遵循的路径。必须是 ``PathwayBase`` 的具体子类的实例，例如 ``LinearPathway``。

    See Also
    ========

    ForceActuator: 生成力的执行器（``LinearDamper`` 的超类）。
    LinearPathway: 两个点之间的直线路径。

    """

    def __init__(self, damping, pathway):
        """``LinearDamper`` 的初始化方法。

        Parameters
        ==========

        damping : Expr
            阻尼常数。
        pathway : PathwayBase
            执行器遵循的路径。必须是 ``PathwayBase`` 的具体子类的实例，例如 ``LinearPathway``。

        """
        self.damping = damping
        self.pathway = pathway

    @property
    def force(self):
        """线性阻尼器产生的阻尼力。"""
        return -self.damping*self.pathway.extension_velocity

    @force.setter
    def force(self, force):
        raise AttributeError('无法设置计算属性 `force`。')

    @property
    def damping(self):
        """线性阻尼器的阻尼常数。"""
        return self._damping

    @damping.setter
    def damping(self, damping):
        if hasattr(self, '_damping'):
            msg = (
                f'无法将属性 `damping` 设置为 {repr(damping)}，因为它是不可变的。'
            )
            raise AttributeError(msg)
        self._damping = sympify(damping, strict=True)

    def __repr__(self):
        """``LinearDamper`` 的字符串表示形式。"""
        return f'{self.__class__.__name__}({self.damping}, {self.pathway})'
# 定义 TorqueActuator 类，继承自 ActuatorBase 类
class TorqueActuator(ActuatorBase):
    """Torque-producing actuator.

    Explanation
    ===========

    A ``TorqueActuator`` is an actuator that produces a pair of equal and
    opposite torques on a pair of bodies.

    Examples
    ========

    To construct a torque actuator, an expression (or symbol) must be supplied
    to represent the torque it can produce, alongside a vector specifying the
    axis about which the torque will act, and a pair of frames on which the
    torque will act.

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import (ReferenceFrame, RigidBody,
    ...     TorqueActuator)
    >>> N = ReferenceFrame('N')
    >>> A = ReferenceFrame('A')
    >>> torque = symbols('T')
    >>> axis = N.z
    >>> parent = RigidBody('parent', frame=N)
    >>> child = RigidBody('child', frame=A)
    >>> bodies = (child, parent)
    >>> actuator = TorqueActuator(torque, axis, *bodies)
    >>> actuator
    TorqueActuator(T, axis=N.z, target_frame=A, reaction_frame=N)

    Note that because torques actually act on frames, not bodies,
    ``TorqueActuator`` will extract the frame associated with a ``RigidBody``
    when one is passed instead of a ``ReferenceFrame``.

    Parameters
    ==========

    torque : Expr
        The scalar expression defining the torque that the actuator produces.
    axis : Vector
        The axis about which the actuator applies torques.
    target_frame : ReferenceFrame | RigidBody
        The primary frame on which the actuator will apply the torque.
    reaction_frame : ReferenceFrame | RigidBody | None
        The secondary frame on which the actuator will apply the torque. Note
        that the (equal and opposite) reaction torque is applied to this frame.

    """

    # 初始化方法，用于创建 TorqueActuator 实例
    def __init__(self, torque, axis, target_frame, reaction_frame=None):
        """Initializer for ``TorqueActuator``.

        Parameters
        ==========

        torque : Expr
            The scalar expression defining the torque that the actuator
            produces.
        axis : Vector
            The axis about which the actuator applies torques.
        target_frame : ReferenceFrame | RigidBody
            The primary frame on which the actuator will apply the torque.
        reaction_frame : ReferenceFrame | RigidBody | None
           The secondary frame on which the actuator will apply the torque.
           Note that the (equal and opposite) reaction torque is applied to
           this frame.

        """
        # 设置 torque 属性，表示该执行器产生的扭矩
        self.torque = torque
        # 设置 axis 属性，表示执行器作用的扭矩轴
        self.axis = axis
        # 设置 target_frame 属性，表示执行器作用的主要参考系或刚体
        self.target_frame = target_frame
        # 设置 reaction_frame 属性，表示执行器作用的次要参考系或刚体
        self.reaction_frame = reaction_frame

    @classmethod
    def at_pin_joint(cls, torque, pin_joint):
        """Alternate construtor to instantiate from a ``PinJoint`` instance.

        Examples
        ========

        To create a pin joint the ``PinJoint`` class requires a name, parent
        body, and child body to be passed to its constructor. It is also
        possible to control the joint axis using the ``joint_axis`` keyword
        argument. In this example let's use the parent body's reference frame's
        z-axis as the joint axis.

        >>> from sympy.physics.mechanics import (PinJoint, ReferenceFrame,
        ...     RigidBody, TorqueActuator)
        >>> N = ReferenceFrame('N')
        >>> A = ReferenceFrame('A')
        >>> parent = RigidBody('parent', frame=N)
        >>> child = RigidBody('child', frame=A)
        >>> pin_joint = PinJoint(
        ...     'pin',
        ...     parent,
        ...     child,
        ...     joint_axis=N.z,
        ... )

        Let's also create a symbol ``T`` that will represent the torque applied
        by the torque actuator.

        >>> from sympy import symbols
        >>> torque = symbols('T')

        To create the torque actuator from the ``torque`` and ``pin_joint``
        variables previously instantiated, these can be passed to the alternate
        constructor class method ``at_pin_joint`` of the ``TorqueActuator``
        class. It should be noted that a positive torque will cause a positive
        displacement of the joint coordinate or that the torque is applied on
        the child body with a reaction torque on the parent.

        >>> actuator = TorqueActuator.at_pin_joint(torque, pin_joint)
        >>> actuator
        TorqueActuator(T, axis=N.z, target_frame=A, reaction_frame=N)

        Parameters
        ==========

        torque : Expr
            The scalar expression defining the torque that the actuator
            produces.
        pin_joint : PinJoint
            The pin joint, and by association the parent and child bodies, on
            which the torque actuator will act. The pair of bodies acted upon
            by the torque actuator are the parent and child bodies of the pin
            joint, with the child acting as the reaction body. The pin joint's
            axis is used as the axis about which the torque actuator will apply
            its torque.

        """
        # 检查传入的 pin_joint 是否为 PinJoint 类型，如果不是则抛出 TypeError 异常
        if not isinstance(pin_joint, PinJoint):
            msg = (
                f'Value {repr(pin_joint)} passed to `pin_joint` was of type '
                f'{type(pin_joint)}, must be {PinJoint}.'
            )
            raise TypeError(msg)
        # 调用类的构造方法，使用 pin_joint 的相关属性来实例化 TorqueActuator 类
        return cls(
            torque,
            pin_joint.joint_axis,  # 使用 pin_joint 的关节轴作为扭矩作用的轴
            pin_joint.child_interframe,  # 子体坐标系相对于其惯性参考系的变换
            pin_joint.parent_interframe,  # 父体坐标系相对于其惯性参考系的变换
        )

    @property
    def torque(self):
        """The magnitude of the torque produced by the actuator."""
        # 返回扭矩的大小，作为属性的 getter 方法
        return self._torque

    @torque.setter
    # 定义一个名为 `torque` 的方法，用于设置扭矩属性
    def torque(self, torque):
        # 检查对象是否具有 `_torque` 属性，如果有则表示属性不可更改，抛出错误
        if hasattr(self, '_torque'):
            msg = (
                f'Can\'t set attribute `torque` to {repr(torque)} as it is '
                f'immutable.'
            )
            raise AttributeError(msg)
        # 使用 sympify 函数将 torque 参数转换为 sympy 的表达式，并将其赋值给 `_torque` 属性
        self._torque = sympify(torque, strict=True)

    @property
    # 定义一个名为 `axis` 的属性，表示扭矩作用的轴
    def axis(self):
        """The axis about which the torque acts."""
        return self._axis

    @axis.setter
    # `axis` 属性的 setter 方法，用于设置扭矩作用的轴
    def axis(self, axis):
        # 检查对象是否具有 `_axis` 属性，如果有则表示属性不可更改，抛出错误
        if hasattr(self, '_axis'):
            msg = (
                f'Can\'t set attribute `axis` to {repr(axis)} as it is '
                f'immutable.'
            )
            raise AttributeError(msg)
        # 检查传入的 axis 参数是否为 Vector 类型，如果不是则抛出类型错误
        if not isinstance(axis, Vector):
            msg = (
                f'Value {repr(axis)} passed to `axis` was of type '
                f'{type(axis)}, must be {Vector}.'
            )
            raise TypeError(msg)
        # 将传入的 axis 参数赋值给 `_axis` 属性
        self._axis = axis

    @property
    # 定义一个名为 `target_frame` 的属性，表示扭矩作用的主要参考系
    def target_frame(self):
        """The primary reference frames on which the torque will act."""
        return self._target_frame

    @target_frame.setter
    # `target_frame` 属性的 setter 方法，用于设置扭矩作用的主要参考系
    def target_frame(self, target_frame):
        # 检查对象是否具有 `_target_frame` 属性，如果有则表示属性不可更改，抛出错误
        if hasattr(self, '_target_frame'):
            msg = (
                f'Can\'t set attribute `target_frame` to {repr(target_frame)} '
                f'as it is immutable.'
            )
            raise AttributeError(msg)
        # 如果传入的 target_frame 参数为 RigidBody 类型，则获取其关联的 frame 属性
        if isinstance(target_frame, RigidBody):
            target_frame = target_frame.frame
        # 如果传入的 target_frame 参数不是 ReferenceFrame 类型，则抛出类型错误
        elif not isinstance(target_frame, ReferenceFrame):
            msg = (
                f'Value {repr(target_frame)} passed to `target_frame` was of '
                f'type {type(target_frame)}, must be {ReferenceFrame}.'
            )
            raise TypeError(msg)
        # 将传入的 target_frame 参数赋值给 `_target_frame` 属性
        self._target_frame = target_frame

    @property
    # 定义一个名为 `reaction_frame` 的属性，表示扭矩作用的主要参考系
    def reaction_frame(self):
        """The primary reference frames on which the torque will act."""
        return self._reaction_frame

    @reaction_frame.setter
    # `reaction_frame` 属性的 setter 方法，用于设置扭矩作用的主要参考系
    def reaction_frame(self, reaction_frame):
        # 检查对象是否具有 `_reaction_frame` 属性，如果有则表示属性不可更改，抛出错误
        if hasattr(self, '_reaction_frame'):
            msg = (
                f'Can\'t set attribute `reaction_frame` to '
                f'{repr(reaction_frame)} as it is immutable.'
            )
            raise AttributeError(msg)
        # 如果传入的 reaction_frame 参数为 RigidBody 类型，则获取其关联的 frame 属性
        elif isinstance(reaction_frame, RigidBody):
            reaction_frame = reaction_frame.frame
        # 如果传入的 reaction_frame 参数既不是 ReferenceFrame 类型也不是 None，则抛出类型错误
        elif (
            not isinstance(reaction_frame, ReferenceFrame)
            and reaction_frame is not None
        ):
            msg = (
                f'Value {repr(reaction_frame)} passed to `reaction_frame` was '
                f'of type {type(reaction_frame)}, must be {ReferenceFrame}.'
            )
            raise TypeError(msg)
        # 将传入的 reaction_frame 参数赋值给 `_reaction_frame` 属性
        self._reaction_frame = reaction_frame
    def to_loads(self):
        """Loads required by the equations of motion method classes.

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

        The below example shows how to generate the loads produced by a torque
        actuator that acts on a pair of bodies attached by a pin joint.

        >>> from sympy import symbols
        >>> from sympy.physics.mechanics import (PinJoint, ReferenceFrame,
        ...     RigidBody, TorqueActuator)
        >>> torque = symbols('T')
        >>> N = ReferenceFrame('N')
        >>> A = ReferenceFrame('A')
        >>> parent = RigidBody('parent', frame=N)
        >>> child = RigidBody('child', frame=A)
        >>> pin_joint = PinJoint(
        ...     'pin',
        ...     parent,
        ...     child,
        ...     joint_axis=N.z,
        ... )
        >>> actuator = TorqueActuator.at_pin_joint(torque, pin_joint)

        The forces produces by the damper can be generated by calling the
        ``to_loads`` method.

        >>> actuator.to_loads()
        [(A, T*N.z), (N, - T*N.z)]

        Alternatively, if a torque actuator is created without a reaction frame
        then the loads returned by the ``to_loads`` method will contain just
        the single load acting on the target frame.

        >>> actuator = TorqueActuator(torque, N.z, N)
        >>> actuator.to_loads()
        [(N, T*N.z)]

        """
        # 创建包含加载项的列表，其中加载项表示在目标框架上施加的扭矩
        loads = [
            Torque(self.target_frame, self.torque*self.axis),
        ]
        # 如果有反作用框架，将其表示为负值的扭矩加载添加到列表中
        if self.reaction_frame is not None:
            loads.append(Torque(self.reaction_frame, -self.torque*self.axis))
        # 返回加载项列表作为结果
        return loads

    def __repr__(self):
        """Representation of a ``TorqueActuator``."""
        # 创建 ``TorqueActuator`` 实例的字符串表示形式
        string = (
            f'{self.__class__.__name__}({self.torque}, axis={self.axis}, '
            f'target_frame={self.target_frame}'
        )
        # 如果有反作用框架，将其包含在表示形式中
        if self.reaction_frame is not None:
            string += f', reaction_frame={self.reaction_frame})'
        else:
            string += ')'
        # 返回表示形式字符串
        return string
    """Represents a nonlinear spring based on the Duffing equation.

    Explanation
    ===========

    The class `DuffingSpring` models a nonlinear spring force described by the equation:
    F = -beta*x - alpha*x**3, where:
    - x is the displacement from the equilibrium position,
    - beta is the linear spring constant (linear_stiffness),
    - alpha is the coefficient for the nonlinear cubic term (nonlinear_stiffness).

    Parameters
    ==========

    linear_stiffness : Expr
        The linear stiffness coefficient (beta).
    nonlinear_stiffness : Expr
        The nonlinear stiffness coefficient (alpha).
    pathway : PathwayBase
        The pathway that the actuator follows.
    equilibrium_length : Expr, optional
        The length at which the spring is in equilibrium (x).
    """

    def __init__(self, linear_stiffness, nonlinear_stiffness, pathway, equilibrium_length=S.Zero):
        """Initialize the DuffingSpring instance.

        Parameters
        ----------
        linear_stiffness : Expr
            The linear stiffness coefficient (beta).
        nonlinear_stiffness : Expr
            The nonlinear stiffness coefficient (alpha).
        pathway : PathwayBase
            The pathway that the actuator follows.
        equilibrium_length : Expr, optional
            The length at which the spring is in equilibrium (x).
        """
        # Initialize the linear stiffness coefficient
        self.linear_stiffness = sympify(linear_stiffness, strict=True)
        # Initialize the nonlinear stiffness coefficient
        self.nonlinear_stiffness = sympify(nonlinear_stiffness, strict=True)
        # Initialize the equilibrium length of the spring
        self.equilibrium_length = sympify(equilibrium_length, strict=True)

        # Validate and set the pathway that the actuator follows
        if not isinstance(pathway, PathwayBase):
            raise TypeError("pathway must be an instance of PathwayBase.")
        self._pathway = pathway

    @property
    def linear_stiffness(self):
        """Getter for the linear stiffness coefficient (beta)."""
        return self._linear_stiffness

    @linear_stiffness.setter
    def linear_stiffness(self, linear_stiffness):
        """Setter for the linear stiffness coefficient (beta).

        Raises
        ------
        AttributeError
            If attempting to set `linear_stiffness` after it has been initialized.
        """
        if hasattr(self, '_linear_stiffness'):
            msg = (
                f'Can\'t set attribute `linear_stiffness` to '
                f'{repr(linear_stiffness)} as it is immutable.'
            )
            raise AttributeError(msg)
        self._linear_stiffness = sympify(linear_stiffness, strict=True)

    @property
    def nonlinear_stiffness(self):
        """Getter for the nonlinear stiffness coefficient (alpha)."""
        return self._nonlinear_stiffness

    @nonlinear_stiffness.setter
    def nonlinear_stiffness(self, nonlinear_stiffness):
        """Setter for the nonlinear stiffness coefficient (alpha).

        Raises
        ------
        AttributeError
            If attempting to set `nonlinear_stiffness` after it has been initialized.
        """
        if hasattr(self, '_nonlinear_stiffness'):
            msg = (
                f'Can\'t set attribute `nonlinear_stiffness` to '
                f'{repr(nonlinear_stiffness)} as it is immutable.'
            )
            raise AttributeError(msg)
        self._nonlinear_stiffness = sympify(nonlinear_stiffness, strict=True)

    @property
    def pathway(self):
        """Getter for the pathway that the actuator follows."""
        return self._pathway

    @pathway.setter
    def pathway(self, pathway):
        """Setter for the pathway that the actuator follows.

        Parameters
        ----------
        pathway : PathwayBase
            The pathway that the actuator follows.

        Raises
        ------
        AttributeError
            If attempting to set `pathway` after it has been initialized.
        TypeError
            If the provided `pathway` is not an instance of PathwayBase.
        """
        if hasattr(self, '_pathway'):
            msg = (
                f'Can\'t set attribute `pathway` to {repr(pathway)} as it is '
                f'immutable.'
            )
            raise AttributeError(msg)
        if not isinstance(pathway, PathwayBase):
            msg = (
                f'Value {repr(pathway)} passed to `pathway` was of type '
                f'{type(pathway)}, must be {PathwayBase}.'
            )
            raise TypeError(msg)
        self._pathway = pathway

    @property
    def equilibrium_length(self):
        """Getter for the equilibrium length of the spring (x)."""
        return self._equilibrium_length
    # 定义属性装饰器 `equilibrium_length` 的 setter 方法
    @equilibrium_length.setter
    def equilibrium_length(self, equilibrium_length):
        # 如果对象已经有 `_equilibrium_length` 属性，说明该属性是不可变的
        if hasattr(self, '_equilibrium_length'):
            # 抛出属性错误，指示无法修改 `equilibrium_length` 属性
            msg = (
                f'Can\'t set attribute `equilibrium_length` to '
                f'{repr(equilibrium_length)} as it is immutable.'
            )
            raise AttributeError(msg)
        # 使用 sympify 函数将 equilibrium_length 转换为符号表达式，并存储在 `_equilibrium_length` 属性中
        self._equilibrium_length = sympify(equilibrium_length, strict=True)

    # 定义属性装饰器 `force` 的 getter 方法
    @property
    def force(self):
        """The force produced by the Duffing spring."""
        # 计算位移，即路径长度与平衡长度的差值
        displacement = self.pathway.length - self.equilibrium_length
        # 根据 Duffing 弹簧模型计算力的表达式，包括线性和非线性刚度
        return -self.linear_stiffness * displacement - self.nonlinear_stiffness * displacement**3

    # 定义属性装饰器 `force` 的 setter 方法
    @force.setter
    def force(self, force):
        # 如果对象已经有 `_force` 属性，说明该属性是不可变的
        if hasattr(self, '_force'):
            # 抛出属性错误，指示无法修改 `force` 属性
            msg = (
                f'Can\'t set attribute `force` to {repr(force)} as it is '
                f'immutable.'
            )
            raise AttributeError(msg)
        # 使用 sympify 函数将 force 转换为符号表达式，并存储在 `_force` 属性中
        self._force = sympify(force, strict=True)

    # 定义对象的字符串表示形式方法
    def __repr__(self):
        # 返回对象的字符串表示，包括类名、线性刚度、非线性刚度、路径对象以及平衡长度信息
        return (f"{self.__class__.__name__}("
                f"{self.linear_stiffness}, {self.nonlinear_stiffness}, {self.pathway}, "
                f"equilibrium_length={self.equilibrium_length})")
```