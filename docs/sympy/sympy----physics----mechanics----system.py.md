# `D:\src\scipysrc\sympy\sympy\physics\mechanics\system.py`

```
from functools import wraps
# 导入装饰器 wraps，用于装饰方法，保留原始方法的元信息

from sympy.core.basic import Basic
# 导入 SymPy 核心基本类 Basic

from sympy.matrices.immutable import ImmutableMatrix
# 导入 SymPy 不可变矩阵类 ImmutableMatrix

from sympy.matrices.dense import Matrix, eye, zeros
# 导入 SymPy 密集矩阵类 Matrix，单位矩阵 eye 和零矩阵 zeros

from sympy.core.containers import OrderedSet
# 导入 SymPy 核心容器类 OrderedSet

from sympy.physics.mechanics.actuator import ActuatorBase
# 导入 SymPy 物理力学中执行器基类 ActuatorBase

from sympy.physics.mechanics.body_base import BodyBase
# 导入 SymPy 物理力学中刚体基类 BodyBase

from sympy.physics.mechanics.functions import (
    Lagrangian, _validate_coordinates, find_dynamicsymbols)
# 导入 SymPy 物理力学中的拉格朗日函数 Lagrangian、验证坐标函数 _validate_coordinates 和查找动力学符号函数 find_dynamicsymbols

from sympy.physics.mechanics.joint import Joint
# 导入 SymPy 物理力学中的关节类 Joint

from sympy.physics.mechanics.kane import KanesMethod
# 导入 SymPy 物理力学中的 Kane 方法类 KanesMethod

from sympy.physics.mechanics.lagrange import LagrangesMethod
# 导入 SymPy 物理力学中的拉格朗日方法类 LagrangesMethod

from sympy.physics.mechanics.loads import _parse_load, gravity
# 导入 SymPy 物理力学中的加载解析函数 _parse_load 和重力加速度 gravity

from sympy.physics.mechanics.method import _Methods
# 导入 SymPy 物理力学中的方法基类 _Methods

from sympy.physics.mechanics.particle import Particle
# 导入 SymPy 物理力学中的质点类 Particle

from sympy.physics.vector import Point, ReferenceFrame, dynamicsymbols
# 导入 SymPy 物理向量类 Point、ReferenceFrame 和动力学符号 dynamicsymbols

from sympy.utilities.iterables import iterable
# 导入 SymPy 工具中的可迭代工具函数 iterable

from sympy.utilities.misc import filldedent
# 导入 SymPy 杂项工具函数 filldedent

__all__ = ['SymbolicSystem', 'System']
# 定义模块的公开接口列表，包括 SymbolicSystem 和 System 类

def _reset_eom_method(method):
    """Decorator to reset the eom_method if a property is changed."""

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        self._eom_method = None
        return method(self, *args, **kwargs)

    return wrapper
# 定义装饰器函数 _reset_eom_method，用于重置 eom_method 属性的方法

class System(_Methods):
    """Class to define a multibody system and form its equations of motion.

    Explanation
    ===========

    A ``System`` instance stores the different objects associated with a model,
    including bodies, joints, constraints, and other relevant information. With
    all the relationships between components defined, the ``System`` can be used
    to form the equations of motion using a backend, such as ``KanesMethod``.
    The ``System`` has been designed to be compatible with third-party
    libraries for greater flexibility and integration with other tools.

    Attributes
    ==========

    frame : ReferenceFrame
        Inertial reference frame of the system.
    fixed_point : Point
        A fixed point in the inertial reference frame.
    x : Vector
        Unit vector fixed in the inertial reference frame.
    y : Vector
        Unit vector fixed in the inertial reference frame.
    z : Vector
        Unit vector fixed in the inertial reference frame.
    q : ImmutableMatrix
        Matrix of all the generalized coordinates, i.e. the independent
        generalized coordinates stacked upon the dependent.
    u : ImmutableMatrix
        Matrix of all the generalized speeds, i.e. the independent generealized
        speeds stacked upon the dependent.
    q_ind : ImmutableMatrix
        Matrix of the independent generalized coordinates.
    q_dep : ImmutableMatrix
        Matrix of the dependent generalized coordinates.
    u_ind : ImmutableMatrix
        Matrix of the independent generalized speeds.
    u_dep : ImmutableMatrix
        Matrix of the dependent generalized speeds.
    u_aux : ImmutableMatrix
        Matrix of auxiliary generalized speeds.
    """

    pass
# 定义类 System，用于定义多体系统并形成其运动方程
    kdes : ImmutableMatrix
        # 表示运动微分方程的矩阵，其中表达式等于零矩阵。
    bodies : tuple of BodyBase subclasses
        # 包含系统中所有构成体的元组。
    joints : tuple of Joint
        # 包含连接系统中体的所有关节的元组。
    loads : tuple of LoadBase subclasses
        # 包含已施加到系统上的所有载荷的元组。
    actuators : tuple of ActuatorBase subclasses
        # 包含系统中所有执行器的元组。
    holonomic_constraints : ImmutableMatrix
        # 表示完整约束的矩阵，其中表达式等于零矩阵。
    nonholonomic_constraints : ImmutableMatrix
        # 表示非完整约束的矩阵，其中表达式等于零矩阵。
    velocity_constraints : ImmutableMatrix
        # 表示速度约束的矩阵，其中表达式等于零矩阵。默认情况下，这些约束是完整约束的时间导数，扩展了非完整约束。
    eom_method : subclass of KanesMethod or LagrangesMethod
        # 用于形成运动方程的后端方法的子类。

    Examples
    ========

    In the example below a cart with a pendulum is created. The cart moves along
    the x axis of the rail and the pendulum rotates about the z axis. The length
    of the pendulum is ``l`` with the pendulum represented as a particle. To
    move the cart a time dependent force ``F`` is applied to the cart.

    We first need to import some functions and create some of our variables.

    >>> from sympy import symbols, simplify
    >>> from sympy.physics.mechanics import (
    ...     mechanics_printing, dynamicsymbols, RigidBody, Particle,
    ...     ReferenceFrame, PrismaticJoint, PinJoint, System)
    >>> mechanics_printing(pretty_print=False)
    >>> g, l = symbols('g l')
    >>> F = dynamicsymbols('F')

    The next step is to create bodies. It is also useful to create a frame for
    locating the particle with respect to the pin joint later on, as a particle
    does not have a body-fixed frame.

    >>> rail = RigidBody('rail')
    >>> cart = RigidBody('cart')
    >>> bob = Particle('bob')
    >>> bob_frame = ReferenceFrame('bob_frame')

    Initialize the system, with the rail as the Newtonian reference. The body is
    also automatically added to the system.

    >>> system = System.from_newtonian(rail)
    >>> print(system.bodies[0])
    rail

    Create the joints, while immediately also adding them to the system.

    >>> system.add_joints(
    ...     PrismaticJoint('slider', rail, cart, joint_axis=rail.x),
    ...     PinJoint('pin', cart, bob, joint_axis=cart.z,
    ...              child_interframe=bob_frame,
    ...              child_point=l * bob_frame.y)
    ... )
    >>> system.joints
    (PrismaticJoint: slider  parent: rail  child: cart,
    PinJoint: pin  parent: cart  child: bob)
    While adding the joints, the associated generalized coordinates, generalized
    speeds, kinematic differential equations and bodies are also added to the
    system.

    >>> system.q
    Matrix([
    [q_slider],
    [   q_pin]])
    >>> system.u
    Matrix([
    [u_slider],
    [   u_pin]])
    >>> system.kdes
    Matrix([
    [u_slider - q_slider'],
    [      u_pin - q_pin']])
    >>> [body.name for body in system.bodies]
    ['rail', 'cart', 'bob']

    # 设置完关节后，将广义坐标、广义速度、运动学微分方程和物体加入系统

    With the kinematics established, we can now apply gravity and the cart force
    ``F``.

    >>> system.apply_uniform_gravity(-g * system.y)
    >>> system.add_loads((cart.masscenter, F * rail.x))
    >>> system.loads
    ((rail_masscenter, - g*rail_mass*rail_frame.y),
     (cart_masscenter, - cart_mass*g*rail_frame.y),
     (bob_masscenter, - bob_mass*g*rail_frame.y),
     (cart_masscenter, F*rail_frame.x))

    # 确定了运动学后，可以施加重力和小车受力 F

    With the entire system defined, we can now form the equations of motion.
    Before forming the equations of motion, one can also run some checks that
    will try to identify some common errors.

    >>> system.validate_system()
    >>> system.form_eoms()
    Matrix([
    [bob_mass*l*u_pin**2*sin(q_pin) - bob_mass*l*cos(q_pin)*u_pin'
     - (bob_mass + cart_mass)*u_slider' + F],
    [                   -bob_mass*g*l*sin(q_pin) - bob_mass*l**2*u_pin'
     - bob_mass*l*cos(q_pin)*u_slider]])

    # 系统定义完成后，可以形成运动方程。在形成运动方程之前，可以运行一些检查以识别常见错误

    >>> simplify(system.mass_matrix)
    Matrix([
    [ bob_mass + cart_mass, bob_mass*l*cos(q_pin)],
    [bob_mass*l*cos(q_pin),         bob_mass*l**2]])
    >>> system.forcing
    Matrix([
    [bob_mass*l*u_pin**2*sin(q_pin) + F],
    [          -bob_mass*g*l*sin(q_pin)]])

    # 简化质量矩阵和强迫项

    The complexity of the above example can be increased if we add a constraint
    to prevent the particle from moving in the horizontal (x) direction. This
    can be done by adding a holonomic constraint. After which we should also
    redefine what our (in)dependent generalized coordinates and speeds are.

    >>> system.add_holonomic_constraints(
    ...     bob.masscenter.pos_from(rail.masscenter).dot(system.x)
    ... )
    >>> system.q_ind = system.get_joint('pin').coordinates
    >>> system.q_dep = system.get_joint('slider').coordinates
    >>> system.u_ind = system.get_joint('pin').speeds
    >>> system.u_dep = system.get_joint('slider').speeds

    # 如果添加一个约束来防止粒子在水平（x）方向上移动，以上示例的复杂性会增加。
    # 可以通过添加完整约束来实现。之后，需要重新定义独立和非独立广义坐标和速度。

    With the updated system the equations of motion can be formed again.

    >>> system.validate_system()
    >>> system.form_eoms()
    Matrix([[-bob_mass*g*l*sin(q_pin)
             - bob_mass*l**2*u_pin'
             - bob_mass*l*cos(q_pin)*u_slider'
             - l*(bob_mass*l*u_pin**2*sin(q_pin)
             - bob_mass*l*cos(q_pin)*u_pin'
             - (bob_mass + cart_mass)*u_slider')*cos(q_pin)
             - l*F*cos(q_pin)]])
    >>> simplify(system.mass_matrix)
    Matrix([
    [bob_mass*l**2*sin(q_pin)**2, -cart_mass*l*cos(q_pin)],
    [               l*cos(q_pin),                       1]])
    >>> simplify(system.forcing)
    Matrix([])

    # 使用更新后的系统再次形成运动方程
    """
    [-l*(bob_mass*g*sin(q_pin) + bob_mass*l*u_pin**2*sin(2*q_pin)/2
     + F*cos(q_pin))],
    [
    l*u_pin**2*sin(q_pin)]])

    """

    def __init__(self, frame=None, fixed_point=None):
        """Initialize the system.

        Parameters
        ==========

        frame : ReferenceFrame, optional
            The inertial frame of the system. If none is supplied, a new frame
            will be created.
        fixed_point : Point, optional
            A fixed point in the inertial reference frame. If none is supplied,
            a new fixed_point will be created.

        """
        # 如果未提供惯性参考系，则创建一个新的惯性参考系
        if frame is None:
            frame = ReferenceFrame('inertial_frame')
        # 如果提供的参考系不是 ReferenceFrame 的实例，则引发类型错误
        elif not isinstance(frame, ReferenceFrame):
            raise TypeError('Frame must be an instance of ReferenceFrame.')
        self._frame = frame
        # 如果未提供固定点，则创建一个新的固定点
        if fixed_point is None:
            fixed_point = Point('inertial_point')
        # 如果提供的固定点不是 Point 的实例，则引发类型错误
        elif not isinstance(fixed_point, Point):
            raise TypeError('Fixed point must be an instance of Point.')
        self._fixed_point = fixed_point
        # 将固定点在惯性参考系中的速度设为零
        self._fixed_point.set_vel(self._frame, 0)
        # 初始化广义坐标和广义速度相关的矩阵
        self._q_ind = ImmutableMatrix(1, 0, []).T
        self._q_dep = ImmutableMatrix(1, 0, []).T
        self._u_ind = ImmutableMatrix(1, 0, []).T
        self._u_dep = ImmutableMatrix(1, 0, []).T
        self._u_aux = ImmutableMatrix(1, 0, []).T
        # 初始化控制方程、哈密顿原理条件、非完整条件相关的矩阵
        self._kdes = ImmutableMatrix(1, 0, []).T
        self._hol_coneqs = ImmutableMatrix(1, 0, []).T
        self._nonhol_coneqs = ImmutableMatrix(1, 0, []).T
        self._vel_constrs = None  # 速度约束条件设为 None
        self._bodies = []  # 初始化一个空列表用于存放系统中的物体
        self._joints = []  # 初始化一个空列表用于存放系统中的连接点
        self._loads = []  # 初始化一个空列表用于存放系统中的加载
        self._actuators = []  # 初始化一个空列表用于存放系统中的驱动器
        self._eom_method = None  # 动力学方程方法设为 None

    @classmethod
    def from_newtonian(cls, newtonian):
        """Constructs the system with respect to a Newtonian body."""
        # 如果提供的 newtonian 是 Particle 的实例，则无法作为 Newtonian，引发类型错误
        if isinstance(newtonian, Particle):
            raise TypeError('A Particle has no frame so cannot act as '
                            'the Newtonian.')
        # 根据 newtonian 的参考系和质心构建系统
        system = cls(frame=newtonian.frame, fixed_point=newtonian.masscenter)
        system.add_bodies(newtonian)
        return system

    @property
    def fixed_point(self):
        """Fixed point in the inertial reference frame."""
        return self._fixed_point

    @property
    def frame(self):
        """Inertial reference frame of the system."""
        return self._frame

    @property
    def x(self):
        """Unit vector fixed in the inertial reference frame."""
        return self._frame.x

    @property
    def y(self):
        """Unit vector fixed in the inertial reference frame."""
        return self._frame.y

    @property
    def z(self):
        """Unit vector fixed in the inertial reference frame."""
        return self._frame.z

    @property
    def bodies(self):
        """Tuple of all bodies that have been added to the system."""
        return tuple(self._bodies)

    @bodies.setter
    @_reset_eom_method
    def bodies(self, bodies):
        # 将输入的bodies转换为列表形式
        bodies = self._objects_to_list(bodies)
        # 检查bodies是否符合要求，要求为空列表，元素为BodyBase类型
        self._check_objects(bodies, [], BodyBase, 'Bodies', 'bodies')
        # 将转换后的bodies赋值给对象的_bodies属性
        self._bodies = bodies

    @property
    def joints(self):
        """Tuple of all joints that have been added to the system."""
        # 返回一个元组，包含所有已添加到系统中的关节对象
        return tuple(self._joints)

    @joints.setter
    @_reset_eom_method
    def joints(self, joints):
        # 将输入的joints转换为列表形式
        joints = self._objects_to_list(joints)
        # 检查joints是否符合要求，要求为空列表，元素为Joint类型
        self._check_objects(joints, [], Joint, 'Joints', 'joints')
        # 清空当前对象的_joints属性，然后添加新的关节对象
        self._joints = []
        self.add_joints(*joints)

    @property
    def loads(self):
        """Tuple of loads that have been applied on the system."""
        # 返回一个元组，包含所有已施加在系统上的负载对象
        return tuple(self._loads)

    @loads.setter
    @_reset_eom_method
    def loads(self, loads):
        # 将输入的loads转换为列表形式
        loads = self._objects_to_list(loads)
        # 解析每个负载对象并存储在对象的_loads属性中
        self._loads = [_parse_load(load) for load in loads]

    @property
    def actuators(self):
        """Tuple of actuators present in the system."""
        # 返回一个元组，包含系统中存在的所有执行器对象
        return tuple(self._actuators)

    @actuators.setter
    @_reset_eom_method
    def actuators(self, actuators):
        # 将输入的actuators转换为列表形式
        actuators = self._objects_to_list(actuators)
        # 检查actuators是否符合要求，要求为空列表，元素为ActuatorBase类型
        self._check_objects(actuators, [], ActuatorBase, 'Actuators',
                            'actuators')
        # 将转换后的actuators赋值给对象的_actuators属性
        self._actuators = actuators

    @property
    def q(self):
        """Matrix of all the generalized coordinates with the independent
        stacked upon the dependent."""
        # 返回所有广义坐标的矩阵，独立的坐标堆叠在依赖的坐标之上
        return self._q_ind.col_join(self._q_dep)

    @property
    def u(self):
        """Matrix of all the generalized speeds with the independent stacked
        upon the dependent."""
        # 返回所有广义速度的矩阵，独立的速度堆叠在依赖的速度之上
        return self._u_ind.col_join(self._u_dep)

    @property
    def q_ind(self):
        """Matrix of the independent generalized coordinates."""
        # 返回独立广义坐标的矩阵
        return self._q_ind

    @q_ind.setter
    @_reset_eom_method
    def q_ind(self, q_ind):
        # 解析输入的独立广义坐标，并分配给对象的_q_ind和_q_dep属性
        self._q_ind, self._q_dep = self._parse_coordinates(
            self._objects_to_list(q_ind), True, [], self.q_dep, 'coordinates')

    @property
    def q_dep(self):
        """Matrix of the dependent generalized coordinates."""
        # 返回依赖广义坐标的矩阵
        return self._q_dep

    @q_dep.setter
    @_reset_eom_method
    def q_dep(self, q_dep):
        # 解析输入的依赖广义坐标，并分配给对象的_q_ind和_q_dep属性
        self._q_ind, self._q_dep = self._parse_coordinates(
            self._objects_to_list(q_dep), False, self.q_ind, [], 'coordinates')

    @property
    def u_ind(self):
        """Matrix of the independent generalized speeds."""
        # 返回独立广义速度的矩阵
        return self._u_ind

    @u_ind.setter
    @_reset_eom_method
    def u_ind(self, u_ind):
        # 解析输入的独立广义速度，并分配给对象的_u_ind和_u_dep属性
        self._u_ind, self._u_dep = self._parse_coordinates(
            self._objects_to_list(u_ind), True, [], self.u_dep, 'speeds')

    @property
    def u_dep(self):
        """Matrix of the dependent generalized speeds."""
        # 返回依赖广义速度的矩阵
        return self._u_dep

    @u_dep.setter
    @_reset_eom_method
    def u_dep(self, u_dep):
        # 解析输入的依赖广义速度，并分配给对象的_u_ind和_u_dep属性
        self._u_ind, self._u_dep = self._parse_coordinates(
            self._objects_to_list(u_dep), False, self.u_ind, [], 'speeds')
    # 定义方法 u_dep，用于设置广义速度相关的信息
    def u_dep(self, u_dep):
        # 将 u_dep 转换为列表后，解析其中的坐标信息，并设置到实例属性中
        self._u_ind, self._u_dep = self._parse_coordinates(
            self._objects_to_list(u_dep), False, self.u_ind, [], 'speeds')

    @property
    # 定义属性 u_aux，返回辅助广义速度的矩阵
    def u_aux(self):
        """Matrix of auxiliary generalized speeds."""
        return self._u_aux

    @u_aux.setter
    @_reset_eom_method
    # 设置属性 u_aux，用于更新辅助广义速度的值
    def u_aux(self, u_aux):
        # 将 u_aux 转换为列表后，解析其中的坐标信息，并设置到实例属性中
        self._u_aux = self._parse_coordinates(
            self._objects_to_list(u_aux), True, [], [], 'u_auxiliary')[0]

    @property
    # 定义属性 kdes，返回运动微分方程为零矩阵的表达式
    def kdes(self):
        """Kinematical differential equations as expressions equated to the zero
        matrix. These equations describe the coupling between the generalized
        coordinates and the generalized speeds."""
        return self._kdes

    @kdes.setter
    @_reset_eom_method
    # 设置属性 kdes，用于更新运动微分方程的表达式
    def kdes(self, kdes):
        # 将 kdes 转换为列表后，解析其中的表达式，并设置到实例属性中
        kdes = self._objects_to_list(kdes)
        self._kdes = self._parse_expressions(
            kdes, [], 'kinematic differential equations')

    @property
    # 定义属性 holonomic_constraints，返回用表达式表示的完整约束矩阵
    def holonomic_constraints(self):
        """Matrix with the holonomic constraints as expressions equated to the
        zero matrix."""
        return self._hol_coneqs

    @holonomic_constraints.setter
    @_reset_eom_method
    # 设置属性 holonomic_constraints，用于更新完整约束矩阵的表达式
    def holonomic_constraints(self, constraints):
        # 将 constraints 转换为列表后，解析其中的表达式，并设置到实例属性中
        constraints = self._objects_to_list(constraints)
        self._hol_coneqs = self._parse_expressions(
            constraints, [], 'holonomic constraints')

    @property
    # 定义属性 nonholonomic_constraints，返回用表达式表示的非完整约束矩阵
    def nonholonomic_constraints(self):
        """Matrix with the nonholonomic constraints as expressions equated to
        the zero matrix."""
        return self._nonhol_coneqs

    @nonholonomic_constraints.setter
    @_reset_eom_method
    # 设置属性 nonholonomic_constraints，用于更新非完整约束矩阵的表达式
    def nonholonomic_constraints(self, constraints):
        # 将 constraints 转换为列表后，解析其中的表达式，并设置到实例属性中
        constraints = self._objects_to_list(constraints)
        self._nonhol_coneqs = self._parse_expressions(
            constraints, [], 'nonholonomic constraints')

    @property
    # 定义属性 velocity_constraints，返回用表达式表示的速度约束矩阵
    def velocity_constraints(self):
        """Matrix with the velocity constraints as expressions equated to the
        zero matrix. The velocity constraints are by default derived from the
        holonomic and nonholonomic constraints unless they are explicitly set.
        """
        if self._vel_constrs is None:
            # 若速度约束矩阵未定义，则默认由完整和非完整约束矩阵的时间导数拼接而成
            return self.holonomic_constraints.diff(dynamicsymbols._t).col_join(
                self.nonholonomic_constraints)
        return self._vel_constrs

    @velocity_constraints.setter
    @_reset_eom_method
    # 设置属性 velocity_constraints，用于更新速度约束矩阵的表达式
    def velocity_constraints(self, constraints):
        if constraints is None:
            # 若传入 None，则将速度约束矩阵设置为 None
            self._vel_constrs = None
            return
        constraints = self._objects_to_list(constraints)
        # 解析传入的约束表达式，并设置到实例属性中
        self._vel_constrs = self._parse_expressions(
            constraints, [], 'velocity constraints')

    @property
    # 定义属性 eom_method，返回用于形成运动方程的后端方法
    def eom_method(self):
        """Backend for forming the equations of motion."""
        return self._eom_method
    def _objects_to_list(lst):
        """
        Helper to convert passed objects to a list.
        """
        if not iterable(lst):  # Only one object
            return [lst]  # Return the object wrapped in a list
        return list(lst[:])  # Return a flattened list of objects

    @staticmethod
    def _check_objects(objects, obj_lst, expected_type, obj_name, type_name):
        """
        Helper to check the objects that are being added to the system.

        Explanation
        ===========
        This method checks that the objects being added to the system
        are of the correct type and have not already been added. If any of the
        objects are not of the correct type or have already been added, an error is raised.

        Parameters
        ==========
        objects : iterable
            The objects that would be added to the system.
        obj_lst : list
            The list of objects that are already in the system.
        expected_type : type
            The type that the objects should be.
        obj_name : str
            The name of the category of objects used in error messages.
        type_name : str
            The name of the type that the objects should be used in error messages.
        """
        seen = set(obj_lst)  # Create a set of already seen objects
        duplicates = set()  # Set to store duplicates
        wrong_types = set()  # Set to store objects of wrong types
        for obj in objects:
            if not isinstance(obj, expected_type):
                wrong_types.add(obj)  # Add objects of wrong type to wrong_types set
            if obj in seen:
                duplicates.add(obj)  # Add duplicates to duplicates set
            else:
                seen.add(obj)  # Add new object to seen set
        if wrong_types:
            raise TypeError(f'{obj_name} {wrong_types} are not {type_name}.')  # Raise TypeError if wrong types found
        if duplicates:
            raise ValueError(f'{obj_name} {duplicates} have already been added '
                             f'to the system.')  # Raise ValueError if duplicates found

    def _parse_coordinates(self, new_coords, independent, old_coords_ind,
                           old_coords_dep, coord_type='coordinates'):
        """
        Helper to parse coordinates and speeds.

        Explanation
        ===========
        This method parses new coordinates into independent and dependent lists,
        checks their types and ensures there are no duplicates.

        Parameters
        ==========
        new_coords : iterable
            New coordinates to be parsed.
        independent : bool or iterable
            Flag or iterable indicating independence of new coordinates.
        old_coords_ind : list
            List of existing independent coordinates.
        old_coords_dep : list
            List of existing dependent coordinates.
        coord_type : str, optional
            Type of coordinates being parsed, default is 'coordinates'.

        Returns
        =======
        tuple
            Tuple containing ImmutableMatrix objects representing parsed coordinates.
        """
        coords_ind, coords_dep = old_coords_ind[:], old_coords_dep[:]  # Copy existing coordinate lists
        if not iterable(independent):
            independent = [independent] * len(new_coords)  # Ensure independent is iterable
        for coord, indep in zip(new_coords, independent):
            if indep:
                coords_ind.append(coord)  # Append to independent if indep is True
            else:
                coords_dep.append(coord)  # Append to dependent if indep is False
        current = {'coordinates': self.q_ind[:] + self.q_dep[:],
                   'speeds': self.u_ind[:] + self.u_dep[:],
                   'u_auxiliary': self._u_aux[:],
                   coord_type: coords_ind + coords_dep}  # Dictionary of current coordinates
        _validate_coordinates(**current)  # Validate coordinates
        return (ImmutableMatrix(1, len(coords_ind), coords_ind).T,
                ImmutableMatrix(1, len(coords_dep), coords_dep).T)  # Return parsed coordinates
    # 定义一个辅助函数，用于解析表达式列表，如约束条件等
    def _parse_expressions(new_expressions, old_expressions, name,
                           check_negatives=False):
        """Helper to parse expressions like constraints."""
        # 复制旧表达式列表，以防对原列表进行修改
        old_expressions = old_expressions[:]
        # 将可能是元组的新表达式列表转换为列表形式
        new_expressions = list(new_expressions)
        # 根据是否检查负表达式，构建检查表达式列表
        if check_negatives:
            check_exprs = old_expressions + [-expr for expr in old_expressions]
        else:
            check_exprs = old_expressions
        # 调用 System 类的内部方法，检查新表达式与检查表达式的类型是否匹配
        System._check_objects(new_expressions, check_exprs, Basic, name,
                              'expressions')
        # 检查新表达式列表中是否存在为零的表达式，若存在则引发 ValueError 异常
        for expr in new_expressions:
            if expr == 0:
                raise ValueError(f'Parsed {name} are zero.')
        # 返回一个不可变矩阵对象，行数为 1，列数为旧表达式数加上新表达式数，转置后的矩阵
        return ImmutableMatrix(1, len(old_expressions) + len(new_expressions),
                               old_expressions + new_expressions).T

    # 修饰符函数，用于将添加坐标的方法重置为方程运动方法的起点
    @_reset_eom_method
    def add_coordinates(self, *coordinates, independent=True):
        """Add generalized coordinate(s) to the system.

        Parameters
        ==========

        *coordinates : dynamicsymbols
            One or more generalized coordinates to be added to the system.
        independent : bool or list of bool, optional
            Boolean whether a coordinate is dependent or independent. The
            default is True, so the coordinates are added as independent by
            default.

        """
        # 使用 _parse_coordinates 方法处理坐标，并更新系统中的独立和依赖坐标
        self._q_ind, self._q_dep = self._parse_coordinates(
            coordinates, independent, self.q_ind, self.q_dep, 'coordinates')

    # 修饰符函数，用于将添加速度的方法重置为方程运动方法的起点
    @_reset_eom_method
    def add_speeds(self, *speeds, independent=True):
        """Add generalized speed(s) to the system.

        Parameters
        ==========

        *speeds : dynamicsymbols
            One or more generalized speeds to be added to the system.
        independent : bool or list of bool, optional
            Boolean whether a speed is dependent or independent. The default is
            True, so the speeds are added as independent by default.

        """
        # 使用 _parse_coordinates 方法处理速度，并更新系统中的独立和依赖速度
        self._u_ind, self._u_dep = self._parse_coordinates(
            speeds, independent, self.u_ind, self.u_dep, 'speeds')

    # 修饰符函数，用于将添加辅助速度的方法重置为方程运动方法的起点
    @_reset_eom_method
    def add_auxiliary_speeds(self, *speeds):
        """Add auxiliary speed(s) to the system.

        Parameters
        ==========

        *speeds : dynamicsymbols
            One or more auxiliary speeds to be added to the system.

        """
        # 使用 _parse_coordinates 方法处理辅助速度，并更新系统中的辅助速度列表
        self._u_aux = self._parse_coordinates(
            speeds, True, self._u_aux, [], 'u_auxiliary')[0]

    # 修饰符函数，用于将添加运动方程方法重置为方程运动方法的起点
    @_reset_eom_method
    def add_kdes(self, *kdes):
        """Add kinematic differential equation(s) to the system.

        Parameters
        ==========

        *kdes : Expr
            One or more kinematic differential equations.

        """
        # 使用 _parse_expressions 方法处理运动方程，并更新系统中的运动方程列表
        self._kdes = self._parse_expressions(
            kdes, self.kdes, 'kinematic differential equations',
            check_negatives=True)
    @_reset_eom_method
    def add_holonomic_constraints(self, *constraints):
        """Add holonomic constraint(s) to the system.

        Parameters
        ==========

        *constraints : Expr
            One or more holonomic constraints, which are expressions that should
            be zero.

        """
        # 解析表达式并将其添加到 holonomic constraints 列表中
        self._hol_coneqs = self._parse_expressions(
            constraints, self._hol_coneqs, 'holonomic constraints',
            check_negatives=True)

    @_reset_eom_method
    def add_nonholonomic_constraints(self, *constraints):
        """Add nonholonomic constraint(s) to the system.

        Parameters
        ==========

        *constraints : Expr
            One or more nonholonomic constraints, which are expressions that
            should be zero.

        """
        # 解析表达式并将其添加到 nonholonomic constraints 列表中
        self._nonhol_coneqs = self._parse_expressions(
            constraints, self._nonhol_coneqs, 'nonholonomic constraints',
            check_negatives=True)

    @_reset_eom_method
    def add_bodies(self, *bodies):
        """Add body(ies) to the system.

        Parameters
        ==========

        bodies : Particle or RigidBody
            One or more bodies.

        """
        # 检查并添加物体到系统中的 bodies 列表中
        self._check_objects(bodies, self.bodies, BodyBase, 'Bodies', 'bodies')
        self._bodies.extend(bodies)

    @_reset_eom_method
    def add_loads(self, *loads):
        """Add load(s) to the system.

        Parameters
        ==========

        *loads : Force or Torque
            One or more loads.

        """
        # 检查并解析负载，然后将其添加到系统的 loads 列表中
        loads = [_parse_load(load) for load in loads]  # Checks the loads
        self._loads.extend(loads)

    @_reset_eom_method
    def apply_uniform_gravity(self, acceleration):
        """Apply uniform gravity to all bodies in the system by adding loads.

        Parameters
        ==========

        acceleration : Vector
            The acceleration due to gravity.

        """
        # 将重力负载应用于系统中的所有物体
        self.add_loads(*gravity(acceleration, *self.bodies))

    @_reset_eom_method
    def add_actuators(self, *actuators):
        """Add actuator(s) to the system.

        Parameters
        ==========

        *actuators : subclass of ActuatorBase
            One or more actuators.

        """
        # 检查并添加执行器到系统中的 actuators 列表中
        self._check_objects(actuators, self.actuators, ActuatorBase,
                            'Actuators', 'actuators')
        self._actuators.extend(actuators)

    @_reset_eom_method
    def add_joints(self, *joints):
        """Add joint(s) to the system.

        Explanation
        ===========

        This methods adds one or more joints to the system including its
        associated objects, i.e. generalized coordinates, generalized speeds,
        kinematic differential equations and the bodies.

        Parameters
        ==========

        *joints : subclass of Joint
            One or more joints.

        Notes
        =====

        For the generalized coordinates, generalized speeds and bodies it is
        checked whether they are already known by the system instance. If they
        are, then they are not added. The kinematic differential equations are
        however always added to the system, so you should not also manually add
        those on beforehand.

        """
        # Check if the provided joints are valid and of the correct type
        self._check_objects(joints, self.joints, Joint, 'Joints', 'joints')
        # Extend the internal list of joints with the new joints
        self._joints.extend(joints)
        
        # Initialize sets for generalized coordinates, speeds, kdes, and bodies
        coordinates, speeds, kdes, bodies = (OrderedSet() for _ in range(4))
        
        # Iterate over each joint to gather associated data
        for joint in joints:
            # Update sets with coordinates, speeds, kdes, and bodies from each joint
            coordinates.update(joint.coordinates)
            speeds.update(joint.speeds)
            kdes.update(joint.kdes)
            bodies.update((joint.parent, joint.child))
        
        # Exclude already existing coordinates, speeds, kdes, and bodies
        coordinates = coordinates.difference(self.q)
        speeds = speeds.difference(self.u)
        kdes = kdes.difference(self.kdes[:] + (-self.kdes)[:])
        bodies = bodies.difference(self.bodies)
        
        # Add the gathered coordinates, speeds, kdes, and bodies to the system
        self.add_coordinates(*tuple(coordinates))
        self.add_speeds(*tuple(speeds))
        self.add_kdes(*(kde for kde in tuple(kdes) if not kde == 0))
        self.add_bodies(*tuple(bodies))

    def get_body(self, name):
        """Retrieve a body from the system by name.

        Parameters
        ==========

        name : str
            The name of the body to retrieve.

        Returns
        =======

        RigidBody or Particle
            The body with the given name, or None if no such body exists.

        """
        # Iterate through each body in the system
        for body in self._bodies:
            # Check if the current body's name matches the requested name
            if body.name == name:
                # Return the body if found
                return body

    def get_joint(self, name):
        """Retrieve a joint from the system by name.

        Parameters
        ==========

        name : str
            The name of the joint to retrieve.

        Returns
        =======

        subclass of Joint
            The joint with the given name, or None if no such joint exists.

        """
        # Iterate through each joint in the system
        for joint in self._joints:
            # Check if the current joint's name matches the requested name
            if joint.name == name:
                # Return the joint if found
                return joint

    def _form_eoms(self):
        # Call the method form_eoms() to formulate equations of motion
        return self.form_eoms()
    # 计算显式形式的运动方程。

    def rhs(self, inv_method=None):
        """Compute the equations of motion in the explicit form.

        Parameters
        ==========

        inv_method : str
            The specific sympy inverse matrix calculation method to use. For a
            list of valid methods, see
            :meth:`~sympy.matrices.matrixbase.MatrixBase.inv`

        Returns
        ========

        ImmutableMatrix
            Equations of motion in the explicit form.

        See Also
        ========

        sympy.physics.mechanics.kane.KanesMethod.rhs:
            KanesMethod's ``rhs`` function.
        sympy.physics.mechanics.lagrange.LagrangesMethod.rhs:
            LagrangesMethod's ``rhs`` function.

        """
        return self.eom_method.rhs(inv_method=inv_method)

    @property
    def mass_matrix(self):
        r"""The mass matrix of the system.

        Explanation
        ===========

        The mass matrix $M_d$ and the forcing vector $f_d$ of a system describe
        the system's dynamics according to the following equations:

        .. math::
            M_d \dot{u} = f_d

        where $\dot{u}$ is the time derivative of the generalized speeds.

        """
        return self.eom_method.mass_matrix

    @property
    def mass_matrix_full(self):
        r"""The mass matrix of the system, augmented by the kinematic
        differential equations in explicit or implicit form.

        Explanation
        ===========

        The full mass matrix $M_m$ and the full forcing vector $f_m$ of a system
        describe the dynamics and kinematics according to the following
        equation:

        .. math::
            M_m \dot{x} = f_m

        where $x$ is the state vector stacking $q$ and $u$.

        """
        return self.eom_method.mass_matrix_full

    @property
    def forcing(self):
        """The forcing vector of the system."""
        return self.eom_method.forcing

    @property
    def forcing_full(self):
        """The forcing vector of the system, augmented by the kinematic
        differential equations in explicit or implicit form."""
        return self.eom_method.forcing_full
# SymbolicSystem 是一个类，用于以符号形式表示系统的所有信息，例如运动方程、系统中的体和载荷等。

class SymbolicSystem:
    """SymbolicSystem is a class that contains all the information about a
    system in a symbolic format such as the equations of motions and the bodies
    and loads in the system.

    There are three ways that the equations of motion can be described for
    Symbolic System:


        [1] Explicit form where the kinematics and dynamics are combined
            x' = F_1(x, t, r, p)

        [2] Implicit form where the kinematics and dynamics are combined
            M_2(x, p) x' = F_2(x, t, r, p)

        [3] Implicit form where the kinematics and dynamics are separate
            M_3(q, p) u' = F_3(q, u, t, r, p)
            q' = G(q, u, t, r, p)

    where

    x : states, e.g. [q, u]
    t : time
    r : specified (exogenous) inputs
    p : constants
    q : generalized coordinates
    u : generalized speeds
    F_1 : right hand side of the combined equations in explicit form
    F_2 : right hand side of the combined equations in implicit form
    F_3 : right hand side of the dynamical equations in implicit form
    M_2 : mass matrix of the combined equations in implicit form
    M_3 : mass matrix of the dynamical equations in implicit form
    Attributes
    ==========

    coordinates : Matrix, shape(n, 1)
        This is a matrix containing the generalized coordinates of the system

    speeds : Matrix, shape(m, 1)
        This is a matrix containing the generalized speeds of the system

    states : Matrix, shape(o, 1)
        This is a matrix containing the state variables of the system

    alg_con : List
        This list contains the indices of the algebraic constraints in the
        combined equations of motion. The presence of these constraints
        requires that a DAE solver be used instead of an ODE solver.
        If the system is given in form [3] the alg_con variable will be
        adjusted such that it is a representation of the combined kinematics
        and dynamics thus make sure it always matches the mass matrix
        entered.

    dyn_implicit_mat : Matrix, shape(m, m)
        This is the M matrix in form [3] of the equations of motion (the mass
        matrix or generalized inertia matrix of the dynamical equations of
        motion in implicit form).

    dyn_implicit_rhs : Matrix, shape(m, 1)
        This is the F vector in form [3] of the equations of motion (the right
        hand side of the dynamical equations of motion in implicit form).

    comb_implicit_mat : Matrix, shape(o, o)
        This is the M matrix in form [2] of the equations of motion.
        This matrix contains a block diagonal structure where the top
        left block (the first rows) represent the matrix in the
        implicit form of the kinematical equations and the bottom right
        block (the last rows) represent the matrix in the implicit form
        of the dynamical equations.
    """
    # comb_implicit_rhs 是形式[2]的运动方程右侧的向量F。向量的顶部表示运动学方程的隐式形式的右手边，
    # 底部表示动力学方程的隐式形式的右手边。
    comb_implicit_rhs : Matrix, shape(o, 1)

    # comb_explicit_rhs 表示合并运动方程的显式形式（从上面的形式[1]）的右手边向量。
    comb_explicit_rhs : Matrix, shape(o, 1)

    # kin_explicit_rhs 是运动学方程显式形式的右手边，可以在形式[3]（矩阵G）中看到。
    kin_explicit_rhs : Matrix, shape(m, 1)

    # output_eqns 是一个字典，存储输出方程，其中键对应于特定方程的名称，值是其符号形式的方程。
    output_eqns : Dictionary

    # bodies 是一个元组，存储系统中的主体，以便将来访问。
    bodies : Tuple

    # loads 是一个元组，存储系统中的负载，以便将来访问。这包括力和力矩，其中力由（作用点，力矢量）给出，
    # 力矩由（作用参考系，扭矩矢量）给出。
    loads : Tuple

    # 示例
    # =======

    # 作为简单的例子，将简单摆的动态输入到SymbolicSystem对象中。首先需要一些导入，并设置摆长（l）、末端质量（m）
    # 和重力常数（g）的符号。::
    # 
    #     >>> from sympy import Matrix, sin, symbols
    #     >>> from sympy.physics.mechanics import dynamicsymbols, SymbolicSystem
    #     >>> l, m, g = symbols('l m g')

    # 系统将由与垂直线的角度theta和广义速度omega定义，其中omega = theta_dot。::
    # 
    #     >>> theta, omega = dynamicsymbols('theta omega')

    # 现在准备形成运动方程，并将其传递给SymbolicSystem对象。::
    # 
    #     >>> kin_explicit_rhs = Matrix([omega])
    #     >>> dyn_implicit_mat = Matrix([l**2 * m])
    #     >>> dyn_implicit_rhs = Matrix([-g * l * m * sin(theta)])
    #     >>> symsystem = SymbolicSystem([theta], dyn_implicit_rhs, [omega],
    #     ...                            dyn_implicit_mat)

    # 注意
    # =====

    # m：广义速度的数量
    # n：广义坐标的数量
    # o：状态的数量
    def __init__(self, coord_states, right_hand_side, speeds=None,
                 mass_matrix=None, coordinate_derivatives=None, alg_con=None,
                 output_eqns={}, coord_idxs=None, speed_idxs=None, bodies=None,
                 loads=None):
        """Initializes a SymbolicSystem object"""

        # Extract information on speeds, coordinates and states
        if speeds is None:
            # If speeds are not provided, initialize states matrix from coord_states
            self._states = Matrix(coord_states)

            if coord_idxs is None:
                # If coord_idxs is not provided, set coordinates as None
                self._coordinates = None
            else:
                # Extract specified coordinates based on coord_idxs
                coords = [coord_states[i] for i in coord_idxs]
                self._coordinates = Matrix(coords)

            if speed_idxs is None:
                # If speed_idxs is not provided, set speeds as None
                self._speeds = None
            else:
                # Extract specified speeds based on speed_idxs
                speeds_inter = [coord_states[i] for i in speed_idxs]
                self._speeds = Matrix(speeds_inter)
        else:
            # If speeds are provided, initialize coordinates and speeds from coord_states and speeds
            self._coordinates = Matrix(coord_states)
            self._speeds = Matrix(speeds)
            self._states = self._coordinates.col_join(self._speeds)

        # Extract equations of motion form
        if coordinate_derivatives is not None:
            # If coordinate_derivatives are provided, set explicit kinematic and implicit dynamic equations
            self._kin_explicit_rhs = coordinate_derivatives
            self._dyn_implicit_rhs = right_hand_side
            self._dyn_implicit_mat = mass_matrix
            self._comb_implicit_rhs = None
            self._comb_implicit_mat = None
            self._comb_explicit_rhs = None
        elif mass_matrix is not None:
            # If mass_matrix is provided but not coordinate_derivatives, set combined implicit equations
            self._kin_explicit_rhs = None
            self._dyn_implicit_rhs = None
            self._dyn_implicit_mat = None
            self._comb_implicit_rhs = right_hand_side
            self._comb_implicit_mat = mass_matrix
            self._comb_explicit_rhs = None
        else:
            # If neither coordinate_derivatives nor mass_matrix are provided, set only combined explicit equations
            self._kin_explicit_rhs = None
            self._dyn_implicit_rhs = None
            self._dyn_implicit_mat = None
            self._comb_implicit_rhs = None
            self._comb_implicit_mat = None
            self._comb_explicit_rhs = right_hand_side

        # Set the remainder of the inputs as instance attributes
        if alg_con is not None and coordinate_derivatives is not None:
            # Adjust alg_con indices if coordinate_derivatives are provided
            alg_con = [i + len(coordinate_derivatives) for i in alg_con]
        self._alg_con = alg_con
        self.output_eqns = output_eqns

        # Change the body and loads iterables to tuples if they are not tuples already
        if not isinstance(bodies, tuple) and bodies is not None:
            bodies = tuple(bodies)
        if not isinstance(loads, tuple) and loads is not None:
            loads = tuple(loads)
        self._bodies = bodies
        self._loads = loads

    @property
    def coordinates(self):
        """Returns the column matrix of the generalized coordinates"""
        if self._coordinates is None:
            # Raise an AttributeError if coordinates were not specified
            raise AttributeError("The coordinates were not specified.")
        else:
            # Return the matrix of coordinates
            return self._coordinates
    def speeds(self):
        """Returns the column matrix of generalized speeds"""
        # 如果 _speeds 属性为 None，则抛出 AttributeError 异常
        if self._speeds is None:
            raise AttributeError("The speeds were not specified.")
        else:
            # 否则返回 _speeds 属性
            return self._speeds

    @property
    def states(self):
        """Returns the column matrix of the state variables"""
        # 返回 _states 属性，作为状态变量的列矩阵
        return self._states

    @property
    def alg_con(self):
        """Returns a list with the indices of the rows containing algebraic
        constraints in the combined form of the equations of motion"""
        # 返回 _alg_con 属性，该属性包含联立形式动力学方程中代数约束的行索引列表
        return self._alg_con

    @property
    def dyn_implicit_mat(self):
        """Returns the matrix, M, corresponding to the dynamic equations in
        implicit form, M x' = F, where the kinematical equations are not
        included"""
        # 如果 _dyn_implicit_mat 属性为 None，则抛出 AttributeError 异常
        if self._dyn_implicit_mat is None:
            raise AttributeError("dyn_implicit_mat is not specified for "
                                 "equations of motion form [1] or [2].")
        else:
            # 否则返回 _dyn_implicit_mat 属性，该属性对应隐式形式的动态方程中的矩阵 M
            return self._dyn_implicit_mat

    @property
    def dyn_implicit_rhs(self):
        """Returns the column matrix, F, corresponding to the dynamic equations
        in implicit form, M x' = F, where the kinematical equations are not
        included"""
        # 如果 _dyn_implicit_rhs 属性为 None，则抛出 AttributeError 异常
        if self._dyn_implicit_rhs is None:
            raise AttributeError("dyn_implicit_rhs is not specified for "
                                 "equations of motion form [1] or [2].")
        else:
            # 否则返回 _dyn_implicit_rhs 属性，该属性对应隐式形式的动态方程中的列矩阵 F
            return self._dyn_implicit_rhs

    @property
    def comb_implicit_mat(self):
        """Returns the matrix, M, corresponding to the equations of motion in
        implicit form (form [2]), M x' = F, where the kinematical equations are
        included"""
        # 如果 _comb_implicit_mat 属性为 None
        if self._comb_implicit_mat is None:
            # 并且 _dyn_implicit_mat 属性不为 None，则计算联立形式动力学方程中的 M 矩阵
            if self._dyn_implicit_mat is not None:
                num_kin_eqns = len(self._kin_explicit_rhs)
                num_dyn_eqns = len(self._dyn_implicit_rhs)
                zeros1 = zeros(num_kin_eqns, num_dyn_eqns)
                zeros2 = zeros(num_dyn_eqns, num_kin_eqns)
                inter1 = eye(num_kin_eqns).row_join(zeros1)
                inter2 = zeros2.row_join(self._dyn_implicit_mat)
                self._comb_implicit_mat = inter1.col_join(inter2)
                return self._comb_implicit_mat
            else:
                # 否则抛出 AttributeError 异常，表示无法为动力学方程形式 [1] 指定 comb_implicit_mat 属性
                raise AttributeError("comb_implicit_mat is not specified for "
                                     "equations of motion form [1].")
        else:
            # 如果 _comb_implicit_mat 属性不为 None，则返回其值，对应动力学方程的联立形式 M 矩阵
            return self._comb_implicit_mat

    @property
    # 返回隐式形式的运动方程对应的列矩阵 F，形式为 M x' = F，其中包括运动学方程
    def comb_implicit_rhs(self):
        if self._comb_implicit_rhs is None:
            # 如果动力学的隐式右手边已经存在，则使用显式运动学右手边和隐式动力学右手边组合
            if self._dyn_implicit_rhs is not None:
                kin_inter = self._kin_explicit_rhs  # 获取显式运动学右手边
                dyn_inter = self._dyn_implicit_rhs  # 获取隐式动力学右手边
                self._comb_implicit_rhs = kin_inter.col_join(dyn_inter)  # 组合成一个列矩阵
                return self._comb_implicit_rhs  # 返回组合的列矩阵
            else:
                # 如果没有指定动力学的隐式右手边，则抛出属性错误异常
                raise AttributeError("comb_implicit_mat is not specified for "
                                     "equations of motion in form [1].")
        else:
            # 如果已经计算过组合的隐式右手边，则直接返回结果
            return self._comb_implicit_rhs

    # 计算显式形式的组合运动方程右手边
    def compute_explicit_form(self):
        if self._comb_explicit_rhs is not None:
            # 如果已经形成了显式右手边，则抛出属性错误异常
            raise AttributeError("comb_explicit_rhs is already formed.")

        inter1 = getattr(self, 'kin_explicit_rhs', None)  # 获取显式运动学右手边
        if inter1 is not None:
            # 如果显式运动学右手边已经定义，则计算隐式动力学右手边，并将两者组合成一个列矩阵
            inter2 = self._dyn_implicit_mat.LUsolve(self._dyn_implicit_rhs)
            out = inter1.col_join(inter2)
        else:
            # 如果显式运动学右手边未定义，则使用组合的隐式矩阵和隐式右手边来计算显式右手边
            out = self._comb_implicit_mat.LUsolve(self._comb_implicit_rhs)

        self._comb_explicit_rhs = out  # 将计算得到的显式右手边保存到属性中

    # 返回显式形式的组合运动方程右手边 F，形式为 x' = F，其中包括运动学方程
    @property
    def comb_explicit_rhs(self):
        if self._comb_explicit_rhs is None:
            # 如果显式右手边尚未计算，则抛出属性错误异常
            raise AttributeError("Please run .compute_explicit_form before "
                                 "attempting to access comb_explicit_rhs.")
        else:
            # 否则，返回已经计算好的显式右手边
            return self._comb_explicit_rhs

    # 返回显式形式的运动学方程右手边 G，形式为 q' = G
    @property
    def kin_explicit_rhs(self):
        if self._kin_explicit_rhs is None:
            # 如果显式运动学右手边未定义，则抛出属性错误异常
            raise AttributeError("kin_explicit_rhs is not specified for "
                                 "equations of motion form [1] or [2].")
        else:
            # 否则，返回已经定义好的显式运动学右手边
            return self._kin_explicit_rhs
    # 返回系统中依赖于时间的所有符号的列矩阵
    def dynamic_symbols(self):
        """Returns a column matrix containing all of the symbols in the system
        that depend on time"""
        # 如果没有显式右手边的组合，则将隐式矩阵和隐式右手边的表达式列表组合起来
        if self._comb_explicit_rhs is None:
            eom_expressions = (self.comb_implicit_mat[:] +
                               self.comb_implicit_rhs[:])
        else:
            eom_expressions = (self._comb_explicit_rhs[:])

        # 查找每个运动方程中的动态符号并取并集
        functions_of_time = set()
        for expr in eom_expressions:
            functions_of_time = functions_of_time.union(
                find_dynamicsymbols(expr))
        # 将状态变量也加入到动态符号集合中
        functions_of_time = functions_of_time.union(self._states)

        # 将集合转换为元组并返回
        return tuple(functions_of_time)

    # 返回系统中不依赖于时间的所有符号的列矩阵
    def constant_symbols(self):
        """Returns a column matrix containing all of the symbols in the system
        that do not depend on time"""
        # 如果没有显式右手边的组合，则将隐式矩阵和隐式右手边的表达式列表组合起来
        if self._comb_explicit_rhs is None:
            eom_expressions = (self.comb_implicit_mat[:] +
                               self.comb_implicit_rhs[:])
        else:
            eom_expressions = (self._comb_explicit_rhs[:])

        # 查找每个运动方程中的自由符号并取并集
        constants = set()
        for expr in eom_expressions:
            constants = constants.union(expr.free_symbols)
        # 移除时间符号（dynamicsymbols._t）
        constants.remove(dynamicsymbols._t)

        # 将集合转换为元组并返回
        return tuple(constants)

    # 返回系统中的物体（bodies）
    @property
    def bodies(self):
        """Returns the bodies in the system"""
        # 如果未指定物体，则抛出属性错误异常
        if self._bodies is None:
            raise AttributeError("bodies were not specified for the system.")
        else:
            return self._bodies

    # 返回系统中的负载（loads）
    @property
    def loads(self):
        """Returns the loads in the system"""
        # 如果未指定负载，则抛出属性错误异常
        if self._loads is None:
            raise AttributeError("loads were not specified for the system.")
        else:
            return self._loads
```