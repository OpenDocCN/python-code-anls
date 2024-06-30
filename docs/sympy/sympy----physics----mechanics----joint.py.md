# `D:\src\scipysrc\sympy\sympy\physics\mechanics\joint.py`

```
# coding=utf-8

# 从abc模块导入ABC和abstractmethod，用于定义抽象基类和抽象方法
from abc import ABC, abstractmethod

# 从sympy库导入pi（圆周率）、Derivative（导数）、Matrix（矩阵）
from sympy import pi, Derivative, Matrix

# 从sympy.core.function模块导入AppliedUndef（应用未定义函数）
from sympy.core.function import AppliedUndef

# 从sympy.physics.mechanics.body_base模块导入BodyBase（物体基类）
from sympy.physics.mechanics.body_base import BodyBase

# 从sympy.physics.mechanics.functions模块导入_validate_coordinates（验证坐标函数）
from sympy.physics.mechanics.functions import _validate_coordinates

# 从sympy.physics.vector模块导入Vector（向量）、dynamicsymbols（动态符号）、cross（叉乘）、Point（点）、ReferenceFrame（参考系）
from sympy.physics.vector import (Vector, dynamicsymbols, cross, Point,
                                  ReferenceFrame)

# 从sympy.utilities.iterables模块导入iterable（可迭代对象判断函数）
from sympy.utilities.iterables import iterable

# 从sympy.utilities.exceptions模块导入sympy_deprecation_warning（SymPy废弃警告）
from sympy.utilities.exceptions import sympy_deprecation_warning

# 指定模块对外可见的符号（类、函数等）
__all__ = ['Joint', 'PinJoint', 'PrismaticJoint', 'CylindricalJoint',
           'PlanarJoint', 'SphericalJoint', 'WeldJoint']

# Joint类的定义，继承自ABC（抽象基类）
class Joint(ABC):
    """Abstract base class for all specific joints.

    Explanation
    ===========

    A joint subtracts degrees of freedom from a body. This is the base class
    for all specific joints and holds all common methods acting as an interface
    for all joints. Custom joint can be created by inheriting Joint class and
    defining all abstract functions.

    The abstract methods are:

    - ``_generate_coordinates``
    - ``_generate_speeds``
    - ``_orient_frames``
    - ``_set_angular_velocity``
    - ``_set_linear_velocity``

    Parameters
    ==========

    name : string
        A unique name for the joint.
    parent : Particle or RigidBody or Body
        The parent body of joint.
    child : Particle or RigidBody or Body
        The child body of joint.
    coordinates : iterable of dynamicsymbols, optional
        Generalized coordinates of the joint.
    speeds : iterable of dynamicsymbols, optional
        Generalized speeds of joint.
    parent_point : Point or Vector, optional
        Attachment point where the joint is fixed to the parent body. If a
        vector is provided, then the attachment point is computed by adding the
        vector to the body's mass center. The default value is the parent's mass
        center.
    child_point : Point or Vector, optional
        Attachment point where the joint is fixed to the child body. If a
        vector is provided, then the attachment point is computed by adding the
        vector to the body's mass center. The default value is the child's mass
        center.
    parent_axis : Vector, optional
        .. deprecated:: 1.12
            Axis fixed in the parent body which aligns with an axis fixed in the
            child body. The default is the x axis of parent's reference frame.
            For more information on this deprecation, see
            :ref:`deprecated-mechanics-joint-axis`.
    child_axis : Vector, optional
        .. deprecated:: 1.12
            Axis fixed in the child body which aligns with an axis fixed in the
            parent body. The default is the x axis of child's reference frame.
            For more information on this deprecation, see
            :ref:`deprecated-mechanics-joint-axis`.
    """
    # 抽象方法定义
    @abstractmethod
    def _generate_coordinates(self):
        pass

    @abstractmethod
    def _generate_speeds(self):
        pass

    @abstractmethod
    def _orient_frames(self):
        pass

    @abstractmethod
    def _set_angular_velocity(self):
        pass

    @abstractmethod
    def _set_linear_velocity(self):
        pass
    parent_interframe : ReferenceFrame, optional
        Intermediate frame of the parent body with respect to which the joint
        transformation is formulated. If a Vector is provided then an interframe
        is created which aligns its X axis with the given vector. The default
        value is the parent's own frame.
    child_interframe : ReferenceFrame, optional
        Intermediate frame of the child body with respect to which the joint
        transformation is formulated. If a Vector is provided then an interframe
        is created which aligns its X axis with the given vector. The default
        value is the child's own frame.
    parent_joint_pos : Point or Vector, optional
        .. deprecated:: 1.12
            This argument is replaced by parent_point and will be removed in a
            future version.
            See :ref:`deprecated-mechanics-joint-pos` for more information.
    child_joint_pos : Point or Vector, optional
        .. deprecated:: 1.12
            This argument is replaced by child_point and will be removed in a
            future version.
            See :ref:`deprecated-mechanics-joint-pos` for more information.

    Attributes
    ==========

    name : string
        The joint's name.
    parent : Particle or RigidBody or Body
        The joint's parent body.
    child : Particle or RigidBody or Body
        The joint's child body.
    coordinates : Matrix
        Matrix of the joint's generalized coordinates.
    speeds : Matrix
        Matrix of the joint's generalized speeds.
    parent_point : Point
        Attachment point where the joint is fixed to the parent body.
    child_point : Point
        Attachment point where the joint is fixed to the child body.
    parent_axis : Vector
        The axis fixed in the parent frame that represents the joint.
    child_axis : Vector
        The axis fixed in the child frame that represents the joint.
    parent_interframe : ReferenceFrame
        Intermediate frame of the parent body with respect to which the joint
        transformation is formulated. If a Vector is provided, it creates an
        intermediate frame aligning its X axis with the provided vector.
    child_interframe : ReferenceFrame
        Intermediate frame of the child body with respect to which the joint
        transformation is formulated. If a Vector is provided, it creates an
        intermediate frame aligning its X axis with the provided vector.
    kdes : Matrix
        Kinematical differential equations of the joint.

    Notes
    =====

    When providing a vector as the intermediate frame, a new intermediate frame
    is created which aligns its X axis with the provided vector. This is done
    with a single fixed rotation about a rotation axis. This rotation axis is
    determined by taking the cross product of the ``body.x`` axis with the
    provided vector. In the case where the provided vector is in the ``-body.x``
    direction, the rotation is done about the ``body.y`` axis.

    """

    def __str__(self):
        # 返回关节的名称作为字符串表示形式
        return self.name

    def __repr__(self):
        # 返回关节的字符串表示形式
        return self.__str__()

    @property
    def name(self):
        """Name of the joint."""
        # 返回关节的名称
        return self._name
    # 返回当前 Joint 的父体
    @property
    def parent(self):
        """Parent body of Joint."""
        return self._parent

    # 返回当前 Joint 的子体
    @property
    def child(self):
        """Child body of Joint."""
        return self._child

    # 返回当前 Joint 的广义坐标的矩阵
    @property
    def coordinates(self):
        """Matrix of the joint's generalized coordinates."""
        return self._coordinates

    # 返回当前 Joint 的广义速度的矩阵
    @property
    def speeds(self):
        """Matrix of the joint's generalized speeds."""
        return self._speeds

    # 返回当前 Joint 的运动学微分方程
    @property
    def kdes(self):
        """Kinematical differential equations of the joint."""
        return self._kdes

    # 返回当前 Joint 的父体坐标系的轴
    @property
    def parent_axis(self):
        """The axis of parent frame."""
        # Will be removed with `deprecated-mechanics-joint-axis`
        return self._parent_axis

    # 返回当前 Joint 的子体坐标系的轴
    @property
    def child_axis(self):
        """The axis of child frame."""
        # Will be removed with `deprecated-mechanics-joint-axis`
        return self._child_axis

    # 返回当前 Joint 的父体固定点的附着点
    @property
    def parent_point(self):
        """Attachment point where the joint is fixed to the parent body."""
        return self._parent_point

    # 返回当前 Joint 的子体固定点的附着点
    @property
    def child_point(self):
        """Attachment point where the joint is fixed to the child body."""
        return self._child_point

    # 返回当前 Joint 的父体间框架
    @property
    def parent_interframe(self):
        return self._parent_interframe

    # 返回当前 Joint 的子体间框架
    @property
    def child_interframe(self):
        return self._child_interframe

    # 抽象方法：生成当前 Joint 的广义坐标矩阵
    @abstractmethod
    def _generate_coordinates(self, coordinates):
        """Generate Matrix of the joint's generalized coordinates."""
        pass

    # 抽象方法：生成当前 Joint 的广义速度矩阵
    @abstractmethod
    def _generate_speeds(self, speeds):
        """Generate Matrix of the joint's generalized speeds."""
        pass

    # 抽象方法：根据 Joint 的方向设置坐标系
    @abstractmethod
    def _orient_frames(self):
        """Orient frames as per the joint."""
        pass

    # 抽象方法：设置 Joint 相关框架的角速度
    @abstractmethod
    def _set_angular_velocity(self):
        """Set angular velocity of the joint related frames."""
        pass

    # 抽象方法：设置 Joint 相关点的线速度
    @abstractmethod
    def _set_linear_velocity(self):
        """Set velocity of related points to the joint."""
        pass

    # 静态方法：将给定坐标系中的矩阵转换为向量
    @staticmethod
    def _to_vector(matrix, frame):
        """Converts a matrix to a vector in the given frame."""
        return Vector([(matrix, frame)])
    def _axis(ax, *frames):
        """Check whether an axis is fixed in one of the frames."""
        # 如果轴为 None，则使用第一个 frame 的 x 轴作为默认轴
        if ax is None:
            ax = frames[0].x
            return ax
        # 如果轴不是 Vector 类型，则抛出类型错误异常
        if not isinstance(ax, Vector):
            raise TypeError("Axis must be a Vector.")
        ref_frame = None  # 找到可以表达轴的参考 frame
        # 在给定的 frames 中寻找合适的参考 frame
        for frame in frames:
            try:
                ax.to_matrix(frame)
            except ValueError:
                pass
            else:
                ref_frame = frame
                break
        # 如果找不到合适的参考 frame，则抛出数值错误异常
        if ref_frame is None:
            raise ValueError("Axis cannot be expressed in one of the body's "
                             "frames.")
        # 如果轴相对于参考 frame 是时间变化的，则抛出数值错误异常
        if not ax.dt(ref_frame) == 0:
            raise ValueError('Axis cannot be time-varying when viewed from the '
                             'associated body.')
        return ax

    @staticmethod
    def _choose_rotation_axis(frame, axis):
        components = axis.to_matrix(frame)
        x, y, z = components[0], components[1], components[2]

        # 根据 axis 的分量决定选择哪个轴进行旋转
        if x != 0:
            if y != 0:
                if z != 0:
                    return cross(axis, frame.x)
            if z != 0:
                return frame.y
            return frame.z
        else:
            if y != 0:
                return frame.x
            return frame.y

    @staticmethod
    def _generate_kdes(self):
        """Generate kinematical differential equations."""
        kdes = []
        t = dynamicsymbols._t
        # 对于 self.coordinates 中的每个坐标生成动力学方程
        for i in range(len(self.coordinates)):
            kdes.append(-self.coordinates[i].diff(t) + self.speeds[i])
        return Matrix(kdes)

    def _locate_joint_pos(self, body, joint_pos, body_frame=None):
        """Returns the attachment point of a body."""
        # 如果没有指定 body_frame，则使用 body 的 frame
        if body_frame is None:
            body_frame = body.frame
        # 如果 joint_pos 为 None，则返回 body 的质心
        if joint_pos is None:
            return body.masscenter
        # 如果 joint_pos 不是 Point 或 Vector 类型，则抛出类型错误异常
        if not isinstance(joint_pos, (Point, Vector)):
            raise TypeError('Attachment point must be a Point or Vector.')
        # 如果 joint_pos 是 Vector 类型，则创建一个新的 Point，并以其命名
        if isinstance(joint_pos, Vector):
            point_name = f'{self.name}_{body.name}_joint'
            joint_pos = body.masscenter.locatenew(point_name, joint_pos)
        # 如果 joint_pos 相对于 body 的质心不是固定的，则抛出数值错误异常
        if not joint_pos.pos_from(body.masscenter).dt(body_frame) == 0:
            raise ValueError('Attachment point must be fixed to the associated '
                             'body.')
        return joint_pos
    # 定义一个方法，用于定位身体的连接帧
    def _locate_joint_frame(self, body, interframe, body_frame=None):
        """Returns the attachment frame of a body."""
        # 如果未指定身体的帧，则使用默认的身体帧
        if body_frame is None:
            body_frame = body.frame
        # 如果未提供连接帧，则返回身体的帧
        if interframe is None:
            return body_frame
        # 如果提供的连接帧是一个向量，则创建对齐的连接帧
        if isinstance(interframe, Vector):
            interframe = Joint._create_aligned_interframe(
                body_frame, interframe,
                frame_name=f'{self.name}_{body.name}_int_frame')
        # 如果提供的连接帧不是参考框架类型，则引发类型错误
        elif not isinstance(interframe, ReferenceFrame):
            raise TypeError('Interframe must be a ReferenceFrame.')
        # 如果连接帧相对于身体帧的角速度不为零，则引发数值错误
        if not interframe.ang_vel_in(body_frame) == 0:
            raise ValueError(f'Interframe {interframe} is not fixed to body '
                             f'{body}.')
        # 将连接帧固定到身体上
        body.masscenter.set_vel(interframe, 0)  # Fixate interframe to body
        # 返回连接帧
        return interframe

    # 定义一个方法，用于填充坐标列表
    def _fill_coordinate_list(self, coordinates, n_coords, label='q', offset=0,
                              number_single=False):
        """Helper method for _generate_coordinates and _generate_speeds.

        Parameters
        ==========

        coordinates : iterable
            Iterable of coordinates or speeds that have been provided.
        n_coords : Integer
            Number of coordinates that should be returned.
        label : String, optional
            Coordinate type either 'q' (coordinates) or 'u' (speeds). The
            Default is 'q'.
        offset : Integer
            Count offset when creating new dynamicsymbols. The default is 0.
        number_single : Boolean
            Boolean whether if n_coords == 1, number should still be used. The
            default is False.

        """

        # 内部方法：创建符号
        def create_symbol(number):
            if n_coords == 1 and not number_single:
                return dynamicsymbols(f'{label}_{self.name}')
            return dynamicsymbols(f'{label}{number}_{self.name}')

        # 检查坐标类型是广义坐标还是广义速度
        name = 'generalized coordinate' if label == 'q' else 'generalized speed'
        # 生成的坐标列表
        generated_coordinates = []
        # 如果未提供坐标，则初始化为空列表
        if coordinates is None:
            coordinates = []
        # 如果提供的坐标不是可迭代对象，则转换为单元素列表
        elif not iterable(coordinates):
            coordinates = [coordinates]
        # 检查提供的坐标数量与所需数量是否匹配
        if not (len(coordinates) == 0 or len(coordinates) == n_coords):
            raise ValueError(f'Expected {n_coords} {name}s, instead got '
                             f'{len(coordinates)} {name}s.')
        # 支持更多类型的可迭代对象，包括矩阵
        for i, coord in enumerate(coordinates):
            # 如果坐标为空，则生成新的符号
            if coord is None:
                generated_coordinates.append(create_symbol(i + offset))
            # 如果坐标是应用未定义或导数类型，则直接添加
            elif isinstance(coord, (AppliedUndef, Derivative)):
                generated_coordinates.append(coord)
            # 否则，引发类型错误
            else:
                raise TypeError(f'The {name} {coord} should have been a '
                                f'dynamicsymbol.')
        # 补充剩余的坐标，以确保生成正确数量的坐标符号
        for i in range(len(coordinates) + offset, n_coords + offset):
            generated_coordinates.append(create_symbol(i))
        # 返回生成的坐标列表作为矩阵
        return Matrix(generated_coordinates)
# 定义一个类 PinJoint，继承自 Joint 类
class PinJoint(Joint):
    """Pin (Revolute) Joint.

    .. raw:: html
        :file: ../../../doc/src/modules/physics/mechanics/api/PinJoint.svg

    Explanation
    ===========

    A pin joint is defined such that the joint rotation axis is fixed in both
    the child and parent and the location of the joint is relative to the mass
    center of each body. The child rotates an angle, θ, from the parent about
    the rotation axis and has a simple angular speed, ω, relative to the
    parent. The direction cosine matrix between the child interframe and
    parent interframe is formed using a simple rotation about the joint axis.
    The page on the joints framework gives a more detailed explanation of the
    intermediate frames.

    Parameters
    ==========

    name : string
        A unique name for the joint.
    parent : Particle or RigidBody or Body
        The parent body of joint.
    child : Particle or RigidBody or Body
        The child body of joint.
    coordinates : dynamicsymbol, optional
        Generalized coordinates of the joint.
    speeds : dynamicsymbol, optional
        Generalized speeds of joint.
    parent_point : Point or Vector, optional
        Attachment point where the joint is fixed to the parent body. If a
        vector is provided, then the attachment point is computed by adding the
        vector to the body's mass center. The default value is the parent's mass
        center.
    child_point : Point or Vector, optional
        Attachment point where the joint is fixed to the child body. If a
        vector is provided, then the attachment point is computed by adding the
        vector to the body's mass center. The default value is the child's mass
        center.
    parent_axis : Vector, optional
        .. deprecated:: 1.12
            Axis fixed in the parent body which aligns with an axis fixed in the
            child body. The default is the x axis of parent's reference frame.
            For more information on this deprecation, see
            :ref:`deprecated-mechanics-joint-axis`.
    child_axis : Vector, optional
        .. deprecated:: 1.12
            Axis fixed in the child body which aligns with an axis fixed in the
            parent body. The default is the x axis of child's reference frame.
            For more information on this deprecation, see
            :ref:`deprecated-mechanics-joint-axis`.
    parent_interframe : ReferenceFrame, optional
        Intermediate frame of the parent body with respect to which the joint
        transformation is formulated. If a Vector is provided then an interframe
        is created which aligns its X axis with the given vector. The default
        value is the parent's own frame.
    """
    child_interframe : ReferenceFrame, optional
        # 子体相对于其间隔帧，用于联合变换的中间帧。如果提供了一个向量，则创建一个间隔帧，其X轴与给定向量对齐。默认值是子体的自身帧。

    joint_axis : Vector
        # 绕其进行旋转的轴。注意这个轴在父间隔帧和子间隔帧中的分量是相同的。

    parent_joint_pos : Point or Vector, optional
        # 已弃用：1.12版本将使用parent_point替代，将来版本中将删除。
        # 更多信息请参见:ref:`deprecated-mechanics-joint-pos`。

    child_joint_pos : Point or Vector, optional
        # 已弃用：1.12版本将使用child_point替代，将来版本中将删除。
        # 更多信息请参见:ref:`deprecated-mechanics-joint-pos`。

    Attributes
    ==========

    name : string
        # 节点的名称。

    parent : Particle or RigidBody or Body
        # 节点的父体。

    child : Particle or RigidBody or Body
        # 节点的子体。

    coordinates : Matrix
        # 节点的广义坐标矩阵。默认值为``dynamicsymbols(f'q_{joint.name}')``。

    speeds : Matrix
        # 节点的广义速度矩阵。默认值为``dynamicsymbols(f'u_{joint.name}')``。

    parent_point : Point
        # 节点固定在父体上的连接点。

    child_point : Point
        # 节点固定在子体上的连接点。

    parent_axis : Vector
        # 在父体坐标系中固定的表示节点的轴。

    child_axis : Vector
        # 在子体坐标系中固定的表示节点的轴。

    parent_interframe : ReferenceFrame
        # 父体相对于其间隔帧，用于联合变换的中间帧。

    child_interframe : ReferenceFrame
        # 子体相对于其间隔帧，用于联合变换的中间帧。

    joint_axis : Vector
        # 绕其进行旋转的轴。注意这个轴在父间隔帧和子间隔帧中的分量是相同的。

    kdes : Matrix
        # 节点的运动微分方程。
    # 访问 joint 对象的 child_point 属性
    >>> joint.child_point
    # 返回 C_masscenter，这是一个名称或者符号，表示质心的位置或者标识
    
    # 访问 joint 对象的 parent_axis 属性
    >>> joint.parent_axis
    # 返回 P_frame.x，表示父级参考框架 P_frame 的 x 轴
    
    # 访问 joint 对象的 child_axis 属性
    >>> joint.child_axis
    # 返回 C_frame.x，表示子级参考框架 C_frame 的 x 轴
    
    # 访问 joint 对象的 coordinates 属性
    >>> joint.coordinates
    # 返回一个矩阵，包含一个名称为 q_PC(t) 的函数
    
    # 访问 joint 对象的 speeds 属性
    >>> joint.speeds
    # 返回一个矩阵，包含一个名称为 u_PC(t) 的函数
    
    # 计算 child.frame 相对于 parent.frame 的角速度
    >>> child.frame.ang_vel_in(parent.frame)
    # 返回 u_PC(t)*P_frame.x，表示相对于父级参考框架 P_frame 的 x 轴的角速度
    
    # 计算 child.frame 相对于 parent.frame 的方向余弦矩阵
    >>> child.frame.dcm(parent.frame)
    # 返回一个 3x3 的矩阵，描述了相对于父级参考框架的方向余弦矩阵，其中包含角度 q_PC(t)
    
    # 计算 joint 对象的 child_point 相对于 joint 对象的 parent_point 的位置
    >>> joint.child_point.pos_from(joint.parent_point)
    # 返回 0，表示两个点之间的位置是零
    
    To further demonstrate the use of the pin joint, the kinematics of simple
    double pendulum that rotates about the Z axis of each connected body can be
    created as follows.
    
    # 导入必要的符号和函数
    >>> from sympy import symbols, trigsimp
    >>> from sympy.physics.mechanics import RigidBody, PinJoint
    >>> l1, l2 = symbols('l1 l2')
    
    First create bodies to represent the fixed ceiling and one to represent
    each pendulum bob.
    
    # 创建用于表示固定天花板和各摆锤的 RigidBody 对象
    >>> ceiling = RigidBody('C')
    >>> upper_bob = RigidBody('U')
    >>> lower_bob = RigidBody('L')
    
    The first joint will connect the upper bob to the ceiling by a distance of
    ``l1`` and the joint axis will be about the Z axis for each body.
    
    # 创建连接天花板和上摆锤之间的 PinJoint 对象
    >>> ceiling_joint = PinJoint('P1', ceiling, upper_bob,
    ... child_point=-l1*upper_bob.frame.x,
    ... joint_axis=ceiling.frame.z)
    
    The second joint will connect the lower bob to the upper bob by a distance
    of ``l2`` and the joint axis will also be about the Z axis for each body.
    
    # 创建连接上摆锤和下摆锤之间的 PinJoint 对象
    >>> pendulum_joint = PinJoint('P2', upper_bob, lower_bob,
    ... child_point=-l2*lower_bob.frame.x,
    ... joint_axis=upper_bob.frame.z)
    
    Once the joints are established the kinematics of the connected bodies can
    be accessed. First the direction cosine matrices of pendulum link relative
    to the ceiling are found:
    
    # 访问上摆锤相对于天花板的方向余弦矩阵
    >>> upper_bob.frame.dcm(ceiling.frame)
    # 返回一个 3x3 的矩阵，描述了相对于天花板的方向余弦矩阵，其中包含角度 q_P1(t)
    
    # 简化下摆锤相对于天花板的方向余弦矩阵
    >>> trigsimp(lower_bob.frame.dcm(ceiling.frame))
    # 返回一个简化后的 3x3 矩阵，描述了相对于天花板的方向余弦矩阵，其中包含角度 q_P1(t) + q_P2(t)
    
    The position of the lower bob's masscenter is found with:
    
    # 计算下摆锤质心相对于天花板质心的位置
    >>> lower_bob.masscenter.pos_from(ceiling.masscenter)
    # 返回 l1*U_frame.x + l2*L_frame.x，表示下摆锤质心相对于天花板质心的位置向量
    
    The angular velocities of the two pendulum links can be computed with
    respect to the ceiling.
    
    # 计算上摆锤相对于天花板的角速度
    >>> upper_bob.frame.ang_vel_in(ceiling.frame)
    # 返回 u_P1(t)*C_frame.z，表示上摆锤相对于天花板的角速度
    
    # 计算下摆锤相对于天花板的角速度
    >>> lower_bob.frame.ang_vel_in(ceiling.frame)
    # 返回 u_P1(t)*C_frame.z + u_P2(t)*U_frame.z，表示下摆锤相对于天花板的角速度
    
    And finally, the linear velocities of the two pendulum bobs can be computed
    with respect to the ceiling.
    
    # 计算上摆锤相对于天花板的线速度
    >>> upper_bob.masscenter.vel(ceiling.frame)
    # 返回 l1*u_P1(t)*U_frame.y，表示上摆锤质心相对于天花板的线速度向量
    
    # 计算下摆锤相对于天花板的线速度
    >>> lower_bob.masscenter.vel(ceiling.frame)
    # 返回 l1*u_P1(t)*U_frame.y + l2*(u_P1(t) + u_P2(t))*L_frame.y，表示下摆锤质心相对于天花板的线速度向量
    def __init__(self, name, parent, child, coordinates=None, speeds=None,
                 parent_point=None, child_point=None, parent_interframe=None,
                 child_interframe=None, parent_axis=None, child_axis=None,
                 joint_axis=None, parent_joint_pos=None, child_joint_pos=None):
        """
        构造函数：初始化 PinJoint 对象的各个属性。

        Args:
        - name: 关节名称
        - parent: 父级刚体对象
        - child: 子级刚体对象
        - coordinates: 关节的坐标（默认为None）
        - speeds: 关节的速度（默认为None）
        - parent_point: 父级点对象（默认为None）
        - child_point: 子级点对象（默认为None）
        - parent_interframe: 父级刚体之间的坐标框架（默认为None）
        - child_interframe: 子级刚体之间的坐标框架（默认为None）
        - parent_axis: 父级刚体的轴（默认为None）
        - child_axis: 子级刚体的轴（默认为None）
        - joint_axis: 关节轴（默认为None）
        - parent_joint_pos: 父级关节位置（默认为None）
        - child_joint_pos: 子级关节位置（默认为None）
        """
        self._joint_axis = joint_axis
        super().__init__(name, parent, child, coordinates, speeds, parent_point,
                         child_point, parent_interframe, child_interframe,
                         parent_axis, child_axis, parent_joint_pos,
                         child_joint_pos)

    def __str__(self):
        """
        返回 PinJoint 对象的字符串表示形式。
        """
        return (f'PinJoint: {self.name}  parent: {self.parent}  '
                f'child: {self.child}')

    @property
    def joint_axis(self):
        """
        返回关节轴的属性，描述子级相对于父级旋转的轴。
        """
        return self._joint_axis

    def _generate_coordinates(self, coordinate):
        """
        生成坐标列表的私有方法。

        Args:
        - coordinate: 坐标值

        Returns:
        - 生成的坐标列表
        """
        return self._fill_coordinate_list(coordinate, 1, 'q')

    def _generate_speeds(self, speed):
        """
        生成速度列表的私有方法。

        Args:
        - speed: 速度值

        Returns:
        - 生成的速度列表
        """
        return self._fill_coordinate_list(speed, 1, 'u')

    def _orient_frames(self):
        """
        设置关节轴方向的私有方法。
        """
        self._joint_axis = self._axis(self.joint_axis, self.parent_interframe)
        self.child_interframe.orient_axis(
            self.parent_interframe, self.joint_axis, self.coordinates[0])

    def _set_angular_velocity(self):
        """
        设置角速度的私有方法。
        """
        self.child_interframe.set_ang_vel(self.parent_interframe, self.speeds[
            0] * self.joint_axis.normalize())

    def _set_linear_velocity(self):
        """
        设置线速度的私有方法。
        """
        self.child_point.set_pos(self.parent_point, 0)
        self.parent_point.set_vel(self._parent_frame, 0)
        self.child_point.set_vel(self._child_frame, 0)
        self.child.masscenter.v2pt_theory(self.parent_point,
                                          self._parent_frame, self._child_frame)
class PrismaticJoint(Joint):
    """Prismatic (Sliding) Joint.

    .. image:: PrismaticJoint.svg

    Explanation
    ===========
    
    It is defined such that the child body translates with respect to the parent
    body along the body-fixed joint axis. The location of the joint is defined
    by two points, one in each body, which coincide when the generalized
    coordinate is zero. The direction cosine matrix between the
    parent_interframe and child_interframe is the identity matrix. Therefore,
    the direction cosine matrix between the parent and child frames is fully
    defined by the definition of the intermediate frames. The page on the joints
    framework gives a more detailed explanation of the intermediate frames.

    Parameters
    ==========

    name : string
        A unique name for the joint.
    parent : Particle or RigidBody or Body
        The parent body of joint.
    child : Particle or RigidBody or Body
        The child body of joint.
    coordinates : dynamicsymbol, optional
        Generalized coordinates of the joint. The default value is
        ``dynamicsymbols(f'q_{joint.name}')``.
    speeds : dynamicsymbol, optional
        Generalized speeds of joint. The default value is
        ``dynamicsymbols(f'u_{joint.name}')``.
    parent_point : Point or Vector, optional
        Attachment point where the joint is fixed to the parent body. If a
        vector is provided, then the attachment point is computed by adding the
        vector to the body's mass center. The default value is the parent's mass
        center.
    child_point : Point or Vector, optional
        Attachment point where the joint is fixed to the child body. If a
        vector is provided, then the attachment point is computed by adding the
        vector to the body's mass center. The default value is the child's mass
        center.
    parent_axis : Vector, optional
        .. deprecated:: 1.12
            Axis fixed in the parent body which aligns with an axis fixed in the
            child body. The default is the x axis of parent's reference frame.
            For more information on this deprecation, see
            :ref:`deprecated-mechanics-joint-axis`.
    child_axis : Vector, optional
        .. deprecated:: 1.12
            Axis fixed in the child body which aligns with an axis fixed in the
            parent body. The default is the x axis of child's reference frame.
            For more information on this deprecation, see
            :ref:`deprecated-mechanics-joint-axis`.
    parent_interframe : ReferenceFrame, optional
        Intermediate frame of the parent body with respect to which the joint
        transformation is formulated. If a Vector is provided then an interframe
        is created which aligns its X axis with the given vector. The default
        value is the parent's own frame.
    """

    def __init__(self, name, parent, child, coordinates=None, speeds=None,
                 parent_point=None, child_point=None, parent_axis=None,
                 child_axis=None, parent_interframe=None):
        # 调用父类的构造函数初始化关节对象
        super().__init__(name, parent, child)
        
        # 如果未指定广义坐标，使用默认值
        self.coordinates = coordinates if coordinates is not None else dynamicsymbols(f'q_{self.name}')
        # 如果未指定广义速度，使用默认值
        self.speeds = speeds if speeds is not None else dynamicsymbols(f'u_{self.name}')
        
        # 如果未指定父体附着点，默认为父体的质心
        self.parent_point = parent_point if parent_point is not None else parent.masscenter
        # 如果未指定子体附着点，默认为子体的质心
        self.child_point = child_point if child_point is not None else child.masscenter
        
        # 设置父体和子体的旧版轴向，默认为各自参考系的X轴
        # （此功能已被弃用，详情请参阅文档）
        self.parent_axis = parent_axis  # Deprecated feature
        self.child_axis = child_axis  # Deprecated feature
        
        # 如果未指定父体中间框架，默认使用父体自身的参考系
        self.parent_interframe = parent_interframe if parent_interframe is not None else parent.frame
    child_interframe : ReferenceFrame, optional
        # 子体的中间参考框架，相对于该框架，关节变换被构建。如果提供一个向量，则创建一个中间框架，其X轴与给定向量对齐。默认值是子体自己的框架。
    joint_axis : Vector
        # 关节运动沿着的轴线。注意这个轴线在父体中间框架和子体中间框架中的分量是相同的。
    parent_joint_pos : Point or Vector, optional
        # 已弃用：1.12版本开始
        # 这个参数被parent_point取代，并且将在未来的版本中移除。
        # 更多信息请参见：:ref:`deprecated-mechanics-joint-pos`
    child_joint_pos : Point or Vector, optional
        # 已弃用：1.12版本开始
        # 这个参数被child_point取代，并且将在未来的版本中移除。
        # 更多信息请参见：:ref:`deprecated-mechanics-joint-pos`

    Attributes
    ==========

    name : string
        # 关节的名称。
    parent : Particle or RigidBody or Body
        # 关节的父体。
    child : Particle or RigidBody or Body
        # 关节的子体。
    coordinates : Matrix
        # 关节的广义坐标矩阵。
    speeds : Matrix
        # 关节的广义速度矩阵。
    parent_point : Point
        # 关节固定在父体上的连接点。
    child_point : Point
        # 关节固定在子体上的连接点。
    parent_axis : Vector
        # 在父体框架中固定的代表关节的轴线。
    child_axis : Vector
        # 在子体框架中固定的代表关节的轴线。
    parent_interframe : ReferenceFrame
        # 父体的中间参考框架，相对于该框架，关节变换被构建。
    child_interframe : ReferenceFrame
        # 子体的中间参考框架，相对于该框架，关节变换被构建。
    kdes : Matrix
        # 关节的运动微分方程。

    Examples
    =========

    A single prismatic joint is created from two bodies and has the following
    basic attributes:

    >>> from sympy.physics.mechanics import RigidBody, PrismaticJoint
    >>> parent = RigidBody('P')
    >>> parent
    P
    >>> child = RigidBody('C')
    >>> child
    C
    >>> joint = PrismaticJoint('PC', parent, child)
    >>> joint
    PrismaticJoint: PC  parent: P  child: C
    >>> joint.name
    'PC'
    >>> joint.parent
    P
    >>> joint.child
    C
    >>> joint.parent_point
    P_masscenter
    >>> joint.child_point
    C_masscenter
    >>> joint.parent_axis
    P_frame.x
    >>> joint.child_axis
    C_frame.x
    >>> joint.coordinates
    Matrix([[q_PC(t)]])
    >>> joint.speeds
    Matrix([[u_PC(t)]])
    >>> child.frame.ang_vel_in(parent.frame)
    0
    """
    构造函数，初始化一个棱柱关节对象。

    Parameters:
    name: 关节名称
    parent: 关节的父物体
    child: 关节的子物体
    coordinates: 坐标
    speeds: 速度
    parent_point: 父物体上的点
    child_point: 子物体上的点
    parent_interframe: 父物体的帧间关系
    child_interframe: 子物体的帧间关系
    parent_axis: 父物体的轴
    child_axis: 子物体的轴
    joint_axis: 关节的轴
    parent_joint_pos: 父物体关节位置
    child_joint_pos: 子物体关节位置
    """
    
    # 设置关节轴
    self._joint_axis = joint_axis
    
    # 调用父类的构造函数初始化关节对象
    super().__init__(name, parent, child, coordinates, speeds, parent_point,
                     child_point, parent_interframe, child_interframe,
                     parent_axis, child_axis, parent_joint_pos,
                     child_joint_pos)

def __str__(self):
    """
    返回描述关节对象的字符串表示。
    """
    return (f'PrismaticJoint: {self.name}  parent: {self.parent}  '
            f'child: {self.child}')

@property
    # 返回子部件相对于父部件进行移动的轴
    def joint_axis(self):
        """Axis along which the child translates with respect to the parent."""
        return self._joint_axis

    # 根据给定的坐标生成一个坐标列表
    def _generate_coordinates(self, coordinate):
        return self._fill_coordinate_list(coordinate, 1, 'q')

    # 根据给定的速度生成一个速度列表
    def _generate_speeds(self, speed):
        return self._fill_coordinate_list(speed, 1, 'u')

    # 设定子帧的方向
    def _orient_frames(self):
        # 将关节轴设定为父帧和子帧之间的轴
        self._joint_axis = self._axis(self.joint_axis, self.parent_interframe)
        # 设定子帧的方向，相对于父帧的轴和关节轴的角度为0
        self.child_interframe.orient_axis(self.parent_interframe, self.joint_axis, 0)

    # 设定角速度
    def _set_angular_velocity(self):
        # 设定子帧的角速度为0，相对于父帧
        self.child_interframe.set_ang_vel(self.parent_interframe, 0)

    # 设定线速度
    def _set_linear_velocity(self):
        # 规范化关节轴
        axis = self.joint_axis.normalize()
        # 设定子点相对于父点的位置，根据坐标和轴的乘积
        self.child_point.set_pos(self.parent_point, self.coordinates[0] * axis)
        # 设定父点的速度为0，相对于父帧
        self.parent_point.set_vel(self._parent_frame, 0)
        # 设定子点的速度为0，相对于子帧
        self.child_point.set_vel(self._child_frame, 0)
        # 设定子点相对于父帧的速度，根据速度和轴的乘积
        self.child_point.set_vel(self._parent_frame, self.speeds[0] * axis)
        # 设定子质心相对于父帧的速度，根据速度和轴的乘积
        self.child.masscenter.set_vel(self._parent_frame, self.speeds[0] * axis)
# 定义一个圆柱形关节，继承自关节基类 Joint
class CylindricalJoint(Joint):
    """Cylindrical Joint.

    .. image:: CylindricalJoint.svg
        :align: center
        :width: 600

    Explanation
    ===========

    A cylindrical joint is defined such that the child body both rotates about
    and translates along the body-fixed joint axis with respect to the parent
    body. The joint axis is both the rotation axis and translation axis. The
    location of the joint is defined by two points, one in each body, which
    coincide when the generalized coordinate corresponding to the translation is
    zero. The direction cosine matrix between the child interframe and parent
    interframe is formed using a simple rotation about the joint axis. The page
    on the joints framework gives a more detailed explanation of the
    intermediate frames.

    Parameters
    ==========

    name : string
        A unique name for the joint.
    parent : Particle or RigidBody or Body
        The parent body of joint.
    child : Particle or RigidBody or Body
        The child body of joint.
    rotation_coordinate : dynamicsymbol, optional
        Generalized coordinate corresponding to the rotation angle. The default
        value is ``dynamicsymbols(f'q0_{joint.name}')``.
    translation_coordinate : dynamicsymbol, optional
        Generalized coordinate corresponding to the translation distance. The
        default value is ``dynamicsymbols(f'q1_{joint.name}')``.
    rotation_speed : dynamicsymbol, optional
        Generalized speed corresponding to the angular velocity. The default
        value is ``dynamicsymbols(f'u0_{joint.name}')``.
    translation_speed : dynamicsymbol, optional
        Generalized speed corresponding to the translation velocity. The default
        value is ``dynamicsymbols(f'u1_{joint.name}')``.
    parent_point : Point or Vector, optional
        Attachment point where the joint is fixed to the parent body. If a
        vector is provided, then the attachment point is computed by adding the
        vector to the body's mass center. The default value is the parent's mass
        center.
    child_point : Point or Vector, optional
        Attachment point where the joint is fixed to the child body. If a
        vector is provided, then the attachment point is computed by adding the
        vector to the body's mass center. The default value is the child's mass
        center.
    parent_interframe : ReferenceFrame, optional
        Intermediate frame of the parent body with respect to which the joint
        transformation is formulated. If a Vector is provided then an interframe
        is created which aligns its X axis with the given vector. The default
        value is the parent's own frame.
    """
    # 初始化方法，定义了圆柱形关节的各个属性和参数
    def __init__(self, name, parent, child, rotation_coordinate=None, translation_coordinate=None,
                 rotation_speed=None, translation_speed=None, parent_point=None, child_point=None,
                 parent_interframe=None):
        # 调用父类的初始化方法
        super().__init__(name, parent, child)
        # 如果未指定 rotation_coordinate，则使用默认值
        self.rotation_coordinate = rotation_coordinate or dynamicsymbols(f'q0_{self.name}')
        # 如果未指定 translation_coordinate，则使用默认值
        self.translation_coordinate = translation_coordinate or dynamicsymbols(f'q1_{self.name}')
        # 如果未指定 rotation_speed，则使用默认值
        self.rotation_speed = rotation_speed or dynamicsymbols(f'u0_{self.name}')
        # 如果未指定 translation_speed，则使用默认值
        self.translation_speed = translation_speed or dynamicsymbols(f'u1_{self.name}')
        # 如果未指定 parent_point，则使用默认值（父体的质心）
        self.parent_point = parent_point if parent_point else self.parent.masscenter
        # 如果未指定 child_point，则使用默认值（子体的质心）
        self.child_point = child_point if child_point else self.child.masscenter
        # 如果未指定 parent_interframe，则使用默认值（父体的参考框架）
        self.parent_interframe = parent_interframe if parent_interframe else self.parent.frame
    child_interframe : ReferenceFrame, optional
        # 子体的中间参考框架，关于它制定关节变换。如果提供了一个 Vector，则创建一个中间框架，其 X 轴与给定向量对齐。默认值为子体自身的框架。
    joint_axis : Vector, optional
        # 旋转和平移轴。注意这个轴的分量在 parent_interframe 和 child_interframe 中是相同的。

    Attributes
    ==========

    name : string
        # 关节的名称。
    parent : Particle or RigidBody or Body
        # 关节的父体。
    child : Particle or RigidBody or Body
        # 关节的子体。
    rotation_coordinate : dynamicsymbol
        # 对应于旋转角度的广义坐标。
    translation_coordinate : dynamicsymbol
        # 对应于平移距离的广义坐标。
    rotation_speed : dynamicsymbol
        # 对应于角速度的广义速度。
    translation_speed : dynamicsymbol
        # 对应于平移速度的广义速度。
    coordinates : Matrix
        # 关节的广义坐标矩阵。
    speeds : Matrix
        # 关节的广义速度矩阵。
    parent_point : Point
        # 关节固定在父体上的连接点。
    child_point : Point
        # 关节固定在子体上的连接点。
    parent_interframe : ReferenceFrame
        # 父体的中间参考框架，关于它制定关节变换。
    child_interframe : ReferenceFrame
        # 子体的中间参考框架，关于它制定关节变换。
    kdes : Matrix
        # 关节的运动微分方程。
    joint_axis : Vector
        # 旋转和平移的轴。

    Examples
    =========

    A single cylindrical joint is created between two bodies and has the
    following basic attributes:

    >>> from sympy.physics.mechanics import RigidBody, CylindricalJoint
    >>> parent = RigidBody('P')
    >>> parent
    P
    >>> child = RigidBody('C')
    >>> child
    C
    >>> joint = CylindricalJoint('PC', parent, child)
    >>> joint
    CylindricalJoint: PC  parent: P  child: C
    >>> joint.name
    'PC'
    >>> joint.parent
    P
    >>> joint.child
    C
    >>> joint.parent_point
    P_masscenter
    >>> joint.child_point
    C_masscenter
    >>> joint.parent_axis
    P_frame.x
    >>> joint.child_axis
    C_frame.x
    >>> joint.coordinates
    Matrix([
    [q0_PC(t)],
    [q1_PC(t)]])
    >>> joint.speeds
    Matrix([
    [u0_PC(t)],
    [u1_PC(t)]])
    >>> child.frame.ang_vel_in(parent.frame)
    u0_PC(t)*P_frame.x
    >>> child.frame.dcm(parent.frame)
    Matrix([
    [1,              0,             0],
    [0,  cos(q0_PC(t)), sin(q0_PC(t))],
    [0, -sin(q0_PC(t)), cos(q0_PC(t))]])
    # 定义一个向量，包含三个元素，分别为0，-sin(q0_PC(t))，cos(q0_PC(t))
    >>> joint.child_point.pos_from(joint.parent_point)
    # 计算一个点相对于另一个点的位置向量
    q1_PC(t)*P_frame.x
    # 用符号表达式 q1_PC(t) 乘以 P_frame.x
    >>> child.masscenter.vel(parent.frame)
    # 计算一个质心的速度，相对于给定的参考框架

    To further demonstrate the use of the cylindrical joint, the kinematics of
    two cylindrical joints perpendicular to each other can be created as follows.

    >>> from sympy import symbols
    # 导入符号运算库 sympy 的 symbols 符号
    >>> from sympy.physics.mechanics import RigidBody, CylindricalJoint
    # 导入 sympy 的物理力学模块中的 RigidBody 刚体和 CylindricalJoint 圆柱关节
    >>> r, l, w = symbols('r l w')
    # 创建符号 r, l, w

    First create bodies to represent the fixed floor with a fixed pole on it.
    The second body represents a freely moving tube around that pole. The third
    body represents a solid flag freely translating along and rotating around
    the Y axis of the tube.

    >>> floor = RigidBody('floor')
    # 创建表示固定地板的刚体
    >>> tube = RigidBody('tube')
    # 创建表示围绕柱子自由移动的管子的刚体
    >>> flag = RigidBody('flag')
    # 创建表示围绕管子 Y 轴自由移动和旋转的固体旗帜的刚体

    The first joint will connect the first tube to the floor with it translating
    along and rotating around the Z axis of both bodies.

    >>> floor_joint = CylindricalJoint('C1', floor, tube, joint_axis=floor.z)
    # 创建一个圆柱关节连接地板和管子，使其沿着 Z 轴翻译和旋转

    The second joint will connect the tube perpendicular to the flag along the Y
    axis of both the tube and the flag, with the joint located at a distance
    ``r`` from the tube's center of mass and a combination of the distances
    ``l`` and ``w`` from the flag's center of mass.

    >>> flag_joint = CylindricalJoint('C2', tube, flag,
    ...                               parent_point=r * tube.y,
    ...                               child_point=-w * flag.y + l * flag.z,
    ...                               joint_axis=tube.y)
    # 创建第二个圆柱关节，将管子垂直连接到旗帜，连接点位于距离管子质心 r 处，并且旗帜质心位置为 -w * flag.y + l * flag.z

    Once the joints are established the kinematics of the connected bodies can
    be accessed. First the direction cosine matrices of both the body and the
    flag relative to the floor are found:

    >>> tube.frame.dcm(floor.frame)
    # 找到管子和旗帜相对地板的方向余弦矩阵
    Matrix([
    [ cos(q0_C1(t)), sin(q0_C1(t)), 0],
    [-sin(q0_C1(t)), cos(q0_C1(t)), 0],
    [             0,             0, 1]])
    >>> flag.frame.dcm(floor.frame)
    # 找到旗帜相对于地板的方向余弦矩阵
    Matrix([
    [cos(q0_C1(t))*cos(q0_C2(t)), sin(q0_C1(t))*cos(q0_C2(t)), -sin(q0_C2(t))],
    [             -sin(q0_C1(t)),               cos(q0_C1(t)),              0],
    [sin(q0_C2(t))*cos(q0_C1(t)), sin(q0_C1(t))*sin(q0_C2(t)),  cos(q0_C2(t))]])

    The position of the flag's center of mass is found with:

    >>> flag.masscenter.pos_from(floor.masscenter)
    # 计算旗帜质心相对于地板质心的位置向量
    q1_C1(t)*floor_frame.z + (r + q1_C2(t))*tube_frame.y + w*flag_frame.y - l*flag_frame.z

    The angular velocities of the two tubes can be computed with respect to the
    floor.

    >>> tube.frame.ang_vel_in(floor.frame)
    # 计算管子相对于地板的角速度
    u0_C1(t)*floor_frame.z
    >>> flag.frame.ang_vel_in(floor.frame)
    # 计算旗帜相对于地板的角速度
    u0_C1(t)*floor_frame.z + u0_C2(t)*tube_frame.y

    Finally, the linear velocities of the two tube centers of mass can be
    computed with respect to the floor, while expressed in the tube's frame.

    >>> tube.masscenter.vel(floor.frame).to_matrix(tube.frame)
    # 计算管子质心相对于地板的线速度，在管子框架中表达
    Matrix([
    [       0],
    [       0],
    """
    >>> flag.masscenter.vel(floor.frame).to_matrix(tube.frame).simplify()
    Matrix([
    [-l*u0_C2(t)*cos(q0_C2(t)) - r*u0_C1(t) - w*u0_C1(t) - q1_C2(t)*u0_C1(t)],
    [                    -l*u0_C1(t)*sin(q0_C2(t)) + Derivative(q1_C2(t), t)],
    [                                    l*u0_C2(t)*sin(q0_C2(t)) + u1_C1(t)]])
    
    """

    def __init__(self, name, parent, child, rotation_coordinate=None,
                 translation_coordinate=None, rotation_speed=None,
                 translation_speed=None, parent_point=None, child_point=None,
                 parent_interframe=None, child_interframe=None,
                 joint_axis=None):
        """
        初始化函数，设置关节名称、父对象、子对象及其坐标和速度。

        Parameters:
        - name: 关节名称
        - parent: 父对象
        - child: 子对象
        - rotation_coordinate: 旋转坐标（广义坐标）
        - translation_coordinate: 平移坐标（广义坐标）
        - rotation_speed: 旋转速度（广义速度）
        - translation_speed: 平移速度（广义速度）
        - parent_point: 父对象点
        - child_point: 子对象点
        - parent_interframe: 父对象间帧
        - child_interframe: 子对象间帧
        - joint_axis: 关节轴
        """
        self._joint_axis = joint_axis
        coordinates = (rotation_coordinate, translation_coordinate)
        speeds = (rotation_speed, translation_speed)
        super().__init__(name, parent, child, coordinates, speeds,
                         parent_point, child_point,
                         parent_interframe=parent_interframe,
                         child_interframe=child_interframe)

    def __str__(self):
        """
        返回关节对象的字符串表示形式，包括关节名称、父对象和子对象。
        """
        return (f'CylindricalJoint: {self.name}  parent: {self.parent}  '
                f'child: {self.child}')

    @property
    def joint_axis(self):
        """
        返回关节轴属性，描述旋转和平移发生的轴线方向。
        """
        return self._joint_axis

    @property
    def rotation_coordinate(self):
        """
        返回旋转坐标属性，对应旋转角度的广义坐标。
        """
        return self.coordinates[0]

    @property
    def translation_coordinate(self):
        """
        返回平移坐标属性，对应平移距离的广义坐标。
        """
        return self.coordinates[1]

    @property
    def rotation_speed(self):
        """
        返回旋转速度属性，对应角速度的广义速度。
        """
        return self.speeds[0]

    @property
    def translation_speed(self):
        """
        返回平移速度属性，对应平移速度的广义速度。
        """
        return self.speeds[1]

    def _generate_coordinates(self, coordinates):
        """
        生成坐标的私有方法，填充坐标列表。

        Parameters:
        - coordinates: 坐标列表

        Returns:
        - 填充后的坐标列表
        """
        return self._fill_coordinate_list(coordinates, 2, 'q')

    def _generate_speeds(self, speeds):
        """
        生成速度的私有方法，填充速度列表。

        Parameters:
        - speeds: 速度列表

        Returns:
        - 填充后的速度列表
        """
        return self._fill_coordinate_list(speeds, 2, 'u')

    def _orient_frames(self):
        """
        定向帧的私有方法，设置关节轴和父子帧的方向。
        """
        self._joint_axis = self._axis(self.joint_axis, self.parent_interframe)
        self.child_interframe.orient_axis(
            self.parent_interframe, self.joint_axis, self.rotation_coordinate)

    def _set_angular_velocity(self):
        """
        设置角速度的私有方法，计算并设置子对象的角速度。
        """
        self.child_interframe.set_ang_vel(
            self.parent_interframe,
            self.rotation_speed * self.joint_axis.normalize())
    # 定义私有方法 `_set_linear_velocity`
    def _set_linear_velocity(self):
        # 将子点设置在父点的位置，使用关节轴向量乘以平移坐标
        self.child_point.set_pos(
            self.parent_point,
            self.translation_coordinate * self.joint_axis.normalize())
        # 设置父点的速度为零
        self.parent_point.set_vel(self._parent_frame, 0)
        # 设置子点相对于子坐标系的速度为零
        self.child_point.set_vel(self._child_frame, 0)
        # 设置子点相对于父坐标系的速度，为平移速度乘以关节轴向量
        self.child_point.set_vel(
            self._parent_frame,
            self.translation_speed * self.joint_axis.normalize())
        # 计算子质心的速度，根据子点、父坐标系和子坐标系之间的关系
        self.child.masscenter.v2pt_theory(self.child_point, self._parent_frame,
                                          self.child_interframe)
class PlanarJoint(Joint):
    """Planar Joint.

    .. raw:: html
        :file: ../../../doc/src/modules/physics/mechanics/api/PlanarJoint.svg

    Explanation
    ===========

    A planar joint is defined such that the child body translates over a fixed
    plane of the parent body as well as rotate about the rotation axis, which
    is perpendicular to that plane. The origin of this plane is the
    ``parent_point`` and the plane is spanned by two nonparallel planar vectors.
    The location of the ``child_point`` is based on the planar vectors
    ($\\vec{v}_1$, $\\vec{v}_2$) and generalized coordinates ($q_1$, $q_2$),
    i.e. $\\vec{r} = q_1 \\hat{v}_1 + q_2 \\hat{v}_2$. The direction cosine
    matrix between the ``child_interframe`` and ``parent_interframe`` is formed
    using a simple rotation ($q_0$) about the rotation axis.

    In order to simplify the definition of the ``PlanarJoint``, the
    ``rotation_axis`` and ``planar_vectors`` are set to be the unit vectors of
    the ``parent_interframe`` according to the table below. This ensures that
    you can only define these vectors by creating a separate frame and supplying
    that as the interframe. If you however would only like to supply the normals
    of the plane with respect to the parent and child bodies, then you can also
    supply those to the ``parent_interframe`` and ``child_interframe``
    arguments. An example of both of these cases is in the examples section
    below and the page on the joints framework provides a more detailed
    explanation of the intermediate frames.

    .. list-table::

        * - ``rotation_axis``
          - ``parent_interframe.x``
        * - ``planar_vectors[0]``
          - ``parent_interframe.y``
        * - ``planar_vectors[1]``
          - ``parent_interframe.z``

    Parameters
    ==========

    name : string
        A unique name for the joint.
    parent : Particle or RigidBody or Body
        The parent body of joint.
    child : Particle or RigidBody or Body
        The child body of joint.
    rotation_coordinate : dynamicsymbol, optional
        Generalized coordinate corresponding to the rotation angle. The default
        value is ``dynamicsymbols(f'q0_{joint.name}')``.
    planar_coordinates : iterable of dynamicsymbols, optional
        Two generalized coordinates used for the planar translation. The default
        value is ``dynamicsymbols(f'q1_{joint.name} q2_{joint.name}')``.
    rotation_speed : dynamicsymbol, optional
        Generalized speed corresponding to the angular velocity. The default
        value is ``dynamicsymbols(f'u0_{joint.name}')``.
    planar_speeds : dynamicsymbols, optional
        Two generalized speeds used for the planar translation velocity. The
        default value is ``dynamicsymbols(f'u1_{joint.name} u2_{joint.name}')``.
    """
    pass


注释：

# 定义一个平面关节类，继承自关节类
class PlanarJoint(Joint):
    """Planar Joint.

    .. raw:: html
        :file: ../../../doc/src/modules/physics/mechanics/api/PlanarJoint.svg

    Explanation
    ===========

    A planar joint is defined such that the child body translates over a fixed
    plane of the parent body as well as rotate about the rotation axis, which
    is perpendicular to that plane. The origin of this plane is the
    ``parent_point`` and the plane is spanned by two nonparallel planar vectors.
    The location of the ``child_point`` is based on the planar vectors
    ($\\vec{v}_1$, $\\vec{v}_2$) and generalized coordinates ($q_1$, $q_2$),
    i.e. $\\vec{r} = q_1 \\hat{v}_1 + q_2 \\hat{v}_2$. The direction cosine
    matrix between the ``child_interframe`` and ``parent_interframe`` is formed
    using a simple rotation ($q_0$) about the rotation axis.

    In order to simplify the definition of the ``PlanarJoint``, the
    ``rotation_axis`` and ``planar_vectors`` are set to be the unit vectors of
    the ``parent_interframe`` according to the table below. This ensures that
    you can only define these vectors by creating a separate frame and supplying
    that as the interframe. If you however would only like to supply the normals
    of the plane with respect to the parent and child bodies, then you can also
    supply those to the ``parent_interframe`` and ``child_interframe``
    arguments. An example of both of these cases is in the examples section
    below and the page on the joints framework provides a more detailed
    explanation of the intermediate frames.

    .. list-table::

        * - ``rotation_axis``
          - ``parent_interframe.x``
        * - ``planar_vectors[0]``
          - ``parent_interframe.y``
        * - ``planar_vectors[1]``
          - ``parent_interframe.z``

    Parameters
    ==========

    name : string
        A unique name for the joint.
    parent : Particle or RigidBody or Body
        The parent body of joint.
    child : Particle or RigidBody or Body
        The child body of joint.
    rotation_coordinate : dynamicsymbol, optional
        Generalized coordinate corresponding to the rotation angle. The default
        value is ``dynamicsymbols(f'q0_{joint.name}')``.
    planar_coordinates : iterable of dynamicsymbols, optional
        Two generalized coordinates used for the planar translation. The default
        value is ``dynamicsymbols(f'q1_{joint.name} q2_{joint.name}')``.
    rotation_speed : dynamicsymbol, optional
        Generalized speed corresponding to the angular velocity. The default
        value is ``dynamicsymbols(f'u0_{joint.name}')``.
    planar_speeds : dynamicsymbols, optional
        Two generalized speeds used for the planar translation velocity. The
        default value is ``dynamicsymbols(f'u1_{joint.name} u2_{joint.name}')``.
    """
    # 空的类定义，不包含任何实际代码，只用于描述平面关节的特性和参数
    pass


这段代码定义了一个平面关节类 `PlanarJoint`，继承自 `Joint` 类，提供了详细的文档字符串说明其特性、参数和用法。
    parent_point : Point or Vector, optional
        # 父体固定点，连接到父体的关节固定点。如果提供了向量，则通过将向量添加到体的质心来计算固定点。默认值为父体的质心。
    child_point : Point or Vector, optional
        # 子体固定点，连接到子体的关节固定点。如果提供了向量，则通过将向量添加到体的质心来计算固定点。默认值为子体的质心。
    parent_interframe : ReferenceFrame, optional
        # 父体的中间框架，关节变换相对于该框架制定。如果提供了一个向量，则创建一个中间框架，使其X轴与给定向量对齐。默认值为父体自身的框架。
    child_interframe : ReferenceFrame, optional
        # 子体的中间框架，关节变换相对于该框架制定。如果提供了一个向量，则创建一个中间框架，使其X轴与给定向量对齐。默认值为子体自身的框架。

    Attributes
    ==========

    name : string
        # 关节的名称。
    parent : Particle or RigidBody or Body
        # 关节的父体。
    child : Particle or RigidBody or Body
        # 关节的子体。
    rotation_coordinate : dynamicsymbol
        # 对应于旋转角度的广义坐标。
    planar_coordinates : Matrix
        # 用于平面平移的两个广义坐标。
    rotation_speed : dynamicsymbol
        # 对应于角速度的广义速度。
    planar_speeds : Matrix
        # 用于平面平移速度的两个广义速度。
    coordinates : Matrix
        # 关节的广义坐标矩阵。
    speeds : Matrix
        # 关节的广义速度矩阵。
    parent_point : Point
        # 关节固定到父体的附着点。
    child_point : Point
        # 关节固定到子体的附着点。
    parent_interframe : ReferenceFrame
        # 关节变换中父体的中间框架。
    child_interframe : ReferenceFrame
        # 关节变换中子体的中间框架。
    kdes : Matrix
        # 关节的运动微分方程。
    rotation_axis : Vector
        # 旋转发生的轴。
    planar_vectors : list
        # 描述平面平移方向的向量列表。

    Examples
    =========

    A single planar joint is created between two bodies and has the following
    basic attributes:

    >>> from sympy.physics.mechanics import RigidBody, PlanarJoint
    >>> parent = RigidBody('P')
    # 访问父体
    >>> parent
    P
    # 创建一个名为 'C' 的刚体子体
    >>> child = RigidBody('C')
    # 访问子体 'C'
    >>> child
    C
    # 创建一个平面关节 'PC'，连接父体 'P' 和子体 'C'
    >>> joint = PlanarJoint('PC', parent, child)
    # 访问关节信息，显示关节名称和连接的父体 'P' 和子体 'C'
    >>> joint
    PlanarJoint: PC  parent: P  child: C
    # 访问关节名称
    >>> joint.name
    'PC'
    # 访问关节的父体 'P'
    >>> joint.parent
    P
    # 访问关节的子体 'C'
    >>> joint.child
    C
    # 访问关节的父体点，通常是父体质心
    >>> joint.parent_point
    P_masscenter
    # 访问关节的子体点，通常是子体质心
    >>> joint.child_point
    C_masscenter
    # 访问关节的旋转轴，通常是父体参考框架的 x 轴
    >>> joint.rotation_axis
    P_frame.x
    # 访问关节的平面向量列表，通常是父体参考框架的 y 轴和 z 轴
    >>> joint.planar_vectors
    [P_frame.y, P_frame.z]
    # 访问关节的旋转坐标，通常是关节变量 q0_PC(t)
    >>> joint.rotation_coordinate
    q0_PC(t)
    # 访问关节的平面坐标，通常是关节变量矩阵 q1_PC(t) 和 q2_PC(t)
    >>> joint.planar_coordinates
    Matrix([
    [q1_PC(t)],
    [q2_PC(t)]])
    # 访问关节的坐标，通常是关节变量矩阵 q0_PC(t), q1_PC(t), q2_PC(t)
    >>> joint.coordinates
    Matrix([
    [q0_PC(t)],
    [q1_PC(t)],
    [q2_PC(t)]])
    # 访问关节的旋转速度，通常是关节速度变量 u0_PC(t)
    >>> joint.rotation_speed
    u0_PC(t)
    # 访问关节的平面速度，通常是关节速度变量矩阵 u1_PC(t) 和 u2_PC(t)
    >>> joint.planar_speeds
    Matrix([
    [u1_PC(t)],
    [u2_PC(t)]])
    # 访问关节的速度，通常是关节速度变量矩阵 u0_PC(t), u1_PC(t), u2_PC(t)
    >>> joint.speeds
    Matrix([
    [u0_PC(t)],
    [u1_PC(t)],
    [u2_PC(t)]])
    # 访问子体在父体参考框架中的角速度
    >>> child.frame.ang_vel_in(parent.frame)
    u0_PC(t)*P_frame.x
    # 访问子体在父体参考框架中的方向余弦矩阵
    >>> child.frame.dcm(parent.frame)
    Matrix([
    [1,              0,             0],
    [0,  cos(q0_PC(t)), sin(q0_PC(t))],
    [0, -sin(q0_PC(t)), cos(q0_PC(t))]])
    # 访问关节子体点相对于父体点的位置向量
    >>> joint.child_point.pos_from(joint.parent_point)
    q1_PC(t)*P_frame.y + q2_PC(t)*P_frame.z
    # 访问子体质心在父体参考框架中的速度
    >>> child.masscenter.vel(parent.frame)
    u1_PC(t)*P_frame.y + u2_PC(t)*P_frame.z

    # 进一步展示平面关节的使用，创建一个在斜坡上滑动的块的运动学模型。

    # 导入必要的符号和模块
    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import PlanarJoint, RigidBody, ReferenceFrame
    # 定义符号
    >>> a, d, h = symbols('a d h')

    # 首先创建表示斜坡和块的刚体。

    # 创建地面刚体 'G'
    >>> ground = RigidBody('G')
    # 创建块刚体 'B'
    >>> block = RigidBody('B')

    # 可以通过定义平面向量 ``planar_vectors`` 或者 ``rotation_axis`` 来定义斜面，但建议创建一个旋转的中间参考框架，
    # 使得 ``parent_vectors`` 和 ``rotation_axis`` 将是这个中间参考框架的单位向量。

    # 创建一个旋转的中间参考框架 'A'，使得其在地面参考框架中绕 'G.y' 轴旋转 'a' 弧度
    >>> slope = ReferenceFrame('A')
    >>> slope.orient_axis(ground.frame, ground.y, a)

    # 使用这些刚体和中间参考框架创建平面关节。
    # 可以指定斜面的原点在斜面质心上方距离 'd'，块的质心在斜面表面上方距离 'h'。注意可以使用旋转轴参数来指定平面的法线。

    # 创建平面关节 'PC'，连接地面 'G' 和块 'B'，父体点为 'd * G.x'，子体点为 '-h * B.x'，父体中间参考框架为 'slope'
    >>> joint = PlanarJoint('PC', ground, block, parent_point=d * ground.x,
    ...                     child_point=-h * block.x, parent_interframe=slope)

    # 一旦关节建立，可以访问刚体的运动学特性。
    # 首先是 ``rotation_axis``，即平面的法线，以及 ``plane_vectors``，即平面的方向向量。

    # 访问关节的旋转轴，通常是中间参考框架 'A' 的 x 轴
    >>> joint.rotation_axis
    A.x
    # 访问关节的平面向量列表，通常是中间参考框架 'A' 的 y 轴和 z 轴
    >>> joint.planar_vectors
    [A.y, A.z]

    # 可以通过以下方式找到块相对于地面的方向余弦矩阵：

    # 访问块参考框架相对于地面参考框架的方向余弦矩阵
    >>> block.frame.dcm(ground.frame)
    """
        The following class defines a PlanarJoint for connecting two rigid bodies in a planar motion system.
    
        """
    
        def __init__(self, name, parent, child, rotation_coordinate=None,
                     planar_coordinates=None, rotation_speed=None,
                     planar_speeds=None, parent_point=None, child_point=None,
                     parent_interframe=None, child_interframe=None):
            # 初始化函数，用于创建一个 PlanarJoint 对象
            # 调用父类的初始化方法，传递名称、父体、子体、旋转坐标、平面坐标、旋转速度、平面速度等参数
            # 设置坐标和速度的元组
            coordinates = (rotation_coordinate, planar_coordinates)
            speeds = (rotation_speed, planar_speeds)
            super().__init__(name, parent, child, coordinates, speeds,
                             parent_point, child_point,
                             parent_interframe=parent_interframe,
                             child_interframe=child_interframe)
    
    """
    # 返回描述对象的字符串表示形式，包括名称、父关节和子关节信息
    def __str__(self):
        return (f'PlanarJoint: {self.name}  parent: {self.parent}  '
                f'child: {self.child}')

    @property
    # 返回与旋转角度对应的广义坐标
    def rotation_coordinate(self):
        """Generalized coordinate corresponding to the rotation angle."""
        return self.coordinates[0]

    @property
    # 返回用于平面平移的两个广义坐标
    def planar_coordinates(self):
        """Two generalized coordinates used for the planar translation."""
        return self.coordinates[1:, 0]

    @property
    # 返回与角速度对应的广义速度
    def rotation_speed(self):
        """Generalized speed corresponding to the angular velocity."""
        return self.speeds[0]

    @property
    # 返回用于平面平移速度的两个广义速度
    def planar_speeds(self):
        """Two generalized speeds used for the planar translation velocity."""
        return self.speeds[1:, 0]

    @property
    # 返回旋转发生的轴线
    def rotation_axis(self):
        """The axis about which the rotation occurs."""
        return self.parent_interframe.x

    @property
    # 返回描述平面平移方向的向量
    def planar_vectors(self):
        """The vectors that describe the planar translation directions."""
        return [self.parent_interframe.y, self.parent_interframe.z]

    # 生成坐标的内部方法，填充旋转速度和平面速度列表
    def _generate_coordinates(self, coordinates):
        rotation_speed = self._fill_coordinate_list(coordinates[0], 1, 'q',
                                                    number_single=True)
        planar_speeds = self._fill_coordinate_list(coordinates[1], 2, 'q', 1)
        return rotation_speed.col_join(planar_speeds)

    # 生成速度的内部方法，填充旋转速度和平面速度列表
    def _generate_speeds(self, speeds):
        rotation_speed = self._fill_coordinate_list(speeds[0], 1, 'u',
                                                    number_single=True)
        planar_speeds = self._fill_coordinate_list(speeds[1], 2, 'u', 1)
        return rotation_speed.col_join(planar_speeds)

    # 定向框架的内部方法，设置旋转轴线和坐标
    def _orient_frames(self):
        self.child_interframe.orient_axis(
            self.parent_interframe, self.rotation_axis,
            self.rotation_coordinate)

    # 设置角速度的内部方法，根据父子框架和旋转轴线设置角速度
    def _set_angular_velocity(self):
        self.child_interframe.set_ang_vel(
            self.parent_interframe,
            self.rotation_speed * self.rotation_axis)

    # 设置线性速度的内部方法，根据父点、子点、父框架和平面坐标设置线性速度
    def _set_linear_velocity(self):
        self.child_point.set_pos(
            self.parent_point,
            self.planar_coordinates[0] * self.planar_vectors[0] +
            self.planar_coordinates[1] * self.planar_vectors[1])
        self.parent_point.set_vel(self.parent_interframe, 0)
        self.child_point.set_vel(self.child_interframe, 0)
        self.child_point.set_vel(
            self._parent_frame, self.planar_speeds[0] * self.planar_vectors[0] +
            self.planar_speeds[1] * self.planar_vectors[1])
        self.child.masscenter.v2pt_theory(self.child_point, self._parent_frame,
                                          self._child_frame)
class SphericalJoint(Joint):
    """Spherical (Ball-and-Socket) Joint.

    .. image:: SphericalJoint.svg
        :align: center
        :width: 600

    Explanation
    ===========

    A spherical joint is defined such that the child body is free to rotate in
    any direction, without allowing a translation of the ``child_point``. As can
    also be seen in the image, the ``parent_point`` and ``child_point`` are
    fixed on top of each other, i.e. the ``joint_point``. This rotation is
    defined using the :func:`parent_interframe.orient(child_interframe,
    rot_type, amounts, rot_order)
    <sympy.physics.vector.frame.ReferenceFrame.orient>` method. The default
    rotation consists of three relative rotations, i.e. body-fixed rotations.
    Based on the direction cosine matrix following from these rotations, the
    angular velocity is computed based on the generalized coordinates and
    generalized speeds.

    Parameters
    ==========

    name : string
        A unique name for the joint.
    parent : Particle or RigidBody or Body
        The parent body of joint.
    child : Particle or RigidBody or Body
        The child body of joint.
    coordinates: iterable of dynamicsymbols, optional
        Generalized coordinates of the joint.
    speeds : iterable of dynamicsymbols, optional
        Generalized speeds of joint.
    parent_point : Point or Vector, optional
        Attachment point where the joint is fixed to the parent body. If a
        vector is provided, then the attachment point is computed by adding the
        vector to the body's mass center. The default value is the parent's mass
        center.
    child_point : Point or Vector, optional
        Attachment point where the joint is fixed to the child body. If a
        vector is provided, then the attachment point is computed by adding the
        vector to the body's mass center. The default value is the child's mass
        center.
    parent_interframe : ReferenceFrame, optional
        Intermediate frame of the parent body with respect to which the joint
        transformation is formulated. If a Vector is provided then an interframe
        is created which aligns its X axis with the given vector. The default
        value is the parent's own frame.
    child_interframe : ReferenceFrame, optional
        Intermediate frame of the child body with respect to which the joint
        transformation is formulated. If a Vector is provided then an interframe
        is created which aligns its X axis with the given vector. The default
        value is the child's own frame.
    rot_type : str, optional
        The method used to generate the direction cosine matrix. Supported
        methods are:

        - ``'Body'``: three successive rotations about new intermediate axes,
          also called "Euler and Tait-Bryan angles"
        - ``'Space'``: three successive rotations about the parent frames' unit
          vectors

        The default method is ``'Body'``.
    """

    def __init__(self, name, parent, child, coordinates=None, speeds=None,
                 parent_point=None, child_point=None, parent_interframe=None,
                 child_interframe=None, rot_type='Body'):
        # 调用父类的构造方法，初始化关节的基本属性
        super().__init__(name, parent, child)
        # 设置关节的广义坐标和广义速度
        self.coordinates = coordinates
        self.speeds = speeds
        # 设置关节固定在父体和子体上的点
        self.parent_point = parent_point if parent_point else parent.masscenter
        self.child_point = child_point if child_point else child.masscenter
        # 设置用于关节变换的父体和子体中间参考框架
        self.parent_interframe = parent_interframe if parent_interframe else parent.frame
        self.child_interframe = child_interframe if child_interframe else child.frame
        # 设置用于计算方向余弦矩阵的旋转类型
        self.rot_type = rot_type
    amounts :
        Expressions defining the rotation angles or direction cosine matrix.
        These must match the ``rot_type``. See examples below for details. The
        input types are:

        - ``'Body'``: 3-tuple of expressions, symbols, or functions
        - ``'Space'``: 3-tuple of expressions, symbols, or functions

        The default amounts are the given ``coordinates``.


    rot_order : str or int, optional
        If applicable, the order of the successive of rotations. The string
        ``'123'`` and integer ``123`` are equivalent, for example. Required for
        ``'Body'`` and ``'Space'``. The default value is ``123``.


    Attributes
    ==========

    name : string
        The joint's name.


    parent : Particle or RigidBody or Body
        The joint's parent body.


    child : Particle or RigidBody or Body
        The joint's child body.


    coordinates : Matrix
        Matrix of the joint's generalized coordinates.


    speeds : Matrix
        Matrix of the joint's generalized speeds.


    parent_point : Point
        Attachment point where the joint is fixed to the parent body.


    child_point : Point
        Attachment point where the joint is fixed to the child body.


    parent_interframe : ReferenceFrame
        Intermediate frame of the parent body with respect to which the joint
        transformation is formulated.


    child_interframe : ReferenceFrame
        Intermediate frame of the child body with respect to which the joint
        transformation is formulated.


    kdes : Matrix
        Kinematical differential equations of the joint.


    Examples
    =========

    A single spherical joint is created from two bodies and has the following
    basic attributes:

    >>> from sympy.physics.mechanics import RigidBody, SphericalJoint
    >>> parent = RigidBody('P')
    >>> parent
    P
    >>> child = RigidBody('C')
    >>> child
    C
    >>> joint = SphericalJoint('PC', parent, child)
    >>> joint
    SphericalJoint: PC  parent: P  child: C
    >>> joint.name
    'PC'
    >>> joint.parent
    P
    >>> joint.child
    C
    >>> joint.parent_point
    P_masscenter
    >>> joint.child_point
    C_masscenter
    >>> joint.parent_interframe
    P_frame
    >>> joint.child_interframe
    C_frame
    >>> joint.coordinates
    Matrix([
    [q0_PC(t)],
    [q1_PC(t)],
    [q2_PC(t)]])
    >>> joint.speeds
    Matrix([
    [u0_PC(t)],
    [u1_PC(t)],
    [u2_PC(t)]])
    >>> child.frame.ang_vel_in(parent.frame).to_matrix(child.frame)
    Matrix([
    [ u0_PC(t)*cos(q1_PC(t))*cos(q2_PC(t)) + u1_PC(t)*sin(q2_PC(t))],
    [-u0_PC(t)*sin(q2_PC(t))*cos(q1_PC(t)) + u1_PC(t)*cos(q2_PC(t))],
    [                             u0_PC(t)*sin(q1_PC(t)) + u2_PC(t)]])
    >>> child.frame.x.to_matrix(parent.frame)
    Matrix([
    [                                            cos(q1_PC(t))*cos(q2_PC(t))],
    [sin(q0_PC(t))*sin(q1_PC(t))*cos(q2_PC(t)) + sin(q2_PC(t))*cos(q0_PC(t))],
    """
    在这段代码中，定义了一个名为SphericalJoint的类，用于表示球形关节。以下是对每个方法和参数的解释：

    定义SphericalJoint类，继承于RigidBody中，用于表示球形关节。
    参数：
    - name: 关节的名称
    - parent: 父物体，即连接点的起始位置
    - child: 子物体，即连接点的终止位置
    - coordinates: 坐标，用于描述关节的位置
    - speeds: 速度，描述关节的运动速度
    - parent_point: 父物体的连接点
    - child_point: 子物体的连接点
    - parent_interframe: 父物体的插入帧
    - child_interframe: 子物体的插入帧
    - rot_type: 旋转类型，默认为'BODY'
    - amounts: 量的数量，默认为None
    - rot_order: 旋转顺序，默认为123

    初始化方法，设置了关节的旋转类型（_rot_type）、旋转顺序（_rot_order）、继承自父类的初始化。
    """
    def __init__(self, name, parent, child, coordinates=None, speeds=None,
                 parent_point=None, child_point=None, parent_interframe=None,
                 child_interframe=None, rot_type='BODY', amounts=None,
                 rot_order=123):
        self._rot_type = rot_type
        self._amounts = amounts
        self._rot_order = rot_order
        super().__init__(name, parent, child, coordinates, speeds,
                         parent_point, child_point,
                         parent_interframe=parent_interframe,
                         child_interframe=child_interframe)

    """
    返回关节对象的字符串表示形式，包括关节名称、父物体和子物体的描述。
    """
    def __str__(self):
        return (f'SphericalJoint: {self.name}  parent: {self.parent}  '
                f'child: {self.child}')

    """
    生成坐标列表，填充坐标以描述三维空间中的关节位置。
    """
    def _generate_coordinates(self, coordinates):
        return self._fill_coordinate_list(coordinates, 3, 'q')

    """
    生成速度列表，填充速度以描述关节的运动速度。
    """
    def _generate_speeds(self, speeds):
        return self._fill_coordinate_list(speeds, len(self.coordinates), 'u')
    # 根据当前的旋转类型调整帧的方向
    supported_rot_types = ('BODY', 'SPACE')
    # 检查当前设置的旋转类型是否在支持的列表中，如果不在则抛出未实现错误
    if self._rot_type.upper() not in supported_rot_types:
        raise NotImplementedError(
            f'Rotation type "{self._rot_type}" is not implemented. '
            f'Implemented rotation types are: {supported_rot_types}')
    # 如果没有指定角度变化量，则使用当前对象的坐标值
    amounts = self.coordinates if self._amounts is None else self._amounts
    # 调用子帧对象的orient方法，根据给定的参数设置方向和旋转顺序
    self.child_interframe.orient(self.parent_interframe, self._rot_type,
                                 amounts, self._rot_order)

    # 设置角速度
    t = dynamicsymbols._t
    # 计算子帧相对于父帧的角速度，并用速度替换所有速度的导数
    vel = self.child_interframe.ang_vel_in(self.parent_interframe).xreplace(
        {q.diff(t): u for q, u in zip(self.coordinates, self.speeds)}
    )
    # 将计算得到的角速度设置为子帧相对于父帧的角速度
    self.child_interframe.set_ang_vel(self.parent_interframe, vel)

    # 设置线速度
    # 将子点相对于父点的位置设置为0，即两点重合
    self.child_point.set_pos(self.parent_point, 0)
    # 将父点相对于父框架的速度设置为0
    self.parent_point.set_vel(self._parent_frame, 0)
    # 将子点相对于子框架的速度设置为0
    self.child_point.set_vel(self._child_frame, 0)
    # 计算质心的速度，并使用理论方法将其转换为子点相对于父点的速度
    self.child.masscenter.v2pt_theory(self.parent_point, self._parent_frame,
                                      self._child_frame)
# 定义了一个名为 WeldJoint 的类，继承自 Joint 类
class WeldJoint(Joint):
    """Weld Joint.

    .. raw:: html
        :file: ../../../doc/src/modules/physics/mechanics/api/WeldJoint.svg

    Explanation
    ===========

    A weld joint is defined such that there is no relative motion between the
    child and parent bodies. The direction cosine matrix between the attachment
    frame (``parent_interframe`` and ``child_interframe``) is the identity
    matrix and the attachment points (``parent_point`` and ``child_point``) are
    coincident. The page on the joints framework gives a more detailed
    explanation of the intermediate frames.

    Parameters
    ==========

    name : string
        A unique name for the joint.
    parent : Particle or RigidBody or Body
        The parent body of joint.
    child : Particle or RigidBody or Body
        The child body of joint.
    parent_point : Point or Vector, optional
        Attachment point where the joint is fixed to the parent body. If a
        vector is provided, then the attachment point is computed by adding the
        vector to the body's mass center. The default value is the parent's mass
        center.
    child_point : Point or Vector, optional
        Attachment point where the joint is fixed to the child body. If a
        vector is provided, then the attachment point is computed by adding the
        vector to the body's mass center. The default value is the child's mass
        center.
    parent_interframe : ReferenceFrame, optional
        Intermediate frame of the parent body with respect to which the joint
        transformation is formulated. If a Vector is provided then an interframe
        is created which aligns its X axis with the given vector. The default
        value is the parent's own frame.
    child_interframe : ReferenceFrame, optional
        Intermediate frame of the child body with respect to which the joint
        transformation is formulated. If a Vector is provided then an interframe
        is created which aligns its X axis with the given vector. The default
        value is the child's own frame.

    Attributes
    ==========

    name : string
        The joint's name.
    parent : Particle or RigidBody or Body
        The joint's parent body.
    child : Particle or RigidBody or Body
        The joint's child body.
    coordinates : Matrix
        Matrix of the joint's generalized coordinates. The default value is
        ``dynamicsymbols(f'q_{joint.name}')``.
    speeds : Matrix
        Matrix of the joint's generalized speeds. The default value is
        ``dynamicsymbols(f'u_{joint.name}')``.
    parent_point : Point
        Attachment point where the joint is fixed to the parent body.
    child_point : Point
        Attachment point where the joint is fixed to the child body.
    parent_interframe : ReferenceFrame
        Intermediate frame of the parent body with respect to which the joint
        transformation is formulated.
    """
    child_interframe : ReferenceFrame
        # 子体相对于其设置联接转换的中间框架。
        Intermediate frame of the child body with respect to which the joint
        transformation is formulated.
    kdes : Matrix
        # 联接的运动微分方程。
        Kinematical differential equations of the joint.

    Examples
    =========

    A single weld joint is created from two bodies and has the following basic
    attributes:

    >>> from sympy.physics.mechanics import RigidBody, WeldJoint
    >>> parent = RigidBody('P')
    >>> parent
    P
    >>> child = RigidBody('C')
    >>> child
    C
    >>> joint = WeldJoint('PC', parent, child)
    >>> joint
    WeldJoint: PC  parent: P  child: C
    >>> joint.name
    'PC'
    >>> joint.parent
    P
    >>> joint.child
    C
    >>> joint.parent_point
    P_masscenter
    >>> joint.child_point
    C_masscenter
    >>> joint.coordinates
    Matrix(0, 0, [])
    >>> joint.speeds
    Matrix(0, 0, [])
    >>> child.frame.ang_vel_in(parent.frame)
    0
    >>> child.frame.dcm(parent.frame)
    Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])
    >>> joint.child_point.pos_from(joint.parent_point)
    0

    To further demonstrate the use of the weld joint, two relatively-fixed
    bodies rotated by a quarter turn about the Y axis can be created as follows:

    >>> from sympy import symbols, pi
    >>> from sympy.physics.mechanics import ReferenceFrame, RigidBody, WeldJoint
    >>> l1, l2 = symbols('l1 l2')

    First create the bodies to represent the parent and rotated child body.

    >>> parent = RigidBody('P')
    >>> child = RigidBody('C')

    Next the intermediate frame specifying the fixed rotation with respect to
    the parent can be created.

    >>> rotated_frame = ReferenceFrame('Pr')
    >>> rotated_frame.orient_axis(parent.frame, parent.y, pi / 2)

    The weld between the parent body and child body is located at a distance
    ``l1`` from the parent's center of mass in the X direction and ``l2`` from
    the child's center of mass in the child's negative X direction.

    >>> weld = WeldJoint('weld', parent, child, parent_point=l1 * parent.x,
    ...                  child_point=-l2 * child.x,
    ...                  parent_interframe=rotated_frame)

    Now that the joint has been established, the kinematics of the bodies can be
    accessed. The direction cosine matrix of the child body with respect to the
    parent can be found:

    >>> child.frame.dcm(parent.frame)
    Matrix([
    [0, 0, -1],
    [0, 1,  0],
    [1, 0,  0]])

    As can also been seen from the direction cosine matrix, the parent X axis is
    aligned with the child's Z axis:
    >>> parent.x == child.z
    True

    The position of the child's center of mass with respect to the parent's
    center of mass can be found with:

    >>> child.masscenter.pos_from(parent.masscenter)
    l1*P_frame.x + l2*C_frame.x

    The angular velocity of the child with respect to the parent is 0 as one
    would expect.

    >>> child.frame.ang_vel_in(parent.frame)
    0
    # 初始化方法，用于创建一个新的对象实例
    def __init__(self, name, parent, child, parent_point=None, child_point=None,
                 parent_interframe=None, child_interframe=None):
        # 调用父类的初始化方法，传递必要的参数，并设置一些默认空列表和可选参数
        super().__init__(name, parent, child, [], [], parent_point,
                         child_point, parent_interframe=parent_interframe,
                         child_interframe=child_interframe)
        # 创建一个新的矩阵对象，用于表示某种属性（这里是用于解决堆叠问题的 #10770）
        self._kdes = Matrix(1, 0, []).T  # Removes stackability problems #10770

    # 返回对象的字符串表示形式，包括对象的名称、父节点和子节点信息
    def __str__(self):
        return (f'WeldJoint: {self.name}  parent: {self.parent}  '
                f'child: {self.child}')

    # 根据坐标生成一个新的矩阵对象
    def _generate_coordinates(self, coordinate):
        return Matrix()

    # 根据速度生成一个新的矩阵对象
    def _generate_speeds(self, speed):
        return Matrix()

    # 调整子坐标系的方向，使其与父坐标系的特定轴对齐
    def _orient_frames(self):
        self.child_interframe.orient_axis(self.parent_interframe,
                                          self.parent_interframe.x, 0)

    # 设置角速度，将子坐标系的角速度设置为父坐标系的角速度
    def _set_angular_velocity(self):
        self.child_interframe.set_ang_vel(self.parent_interframe, 0)

    # 设置线速度，确保子坐标系的位置相对于父坐标系的位置为零
    def _set_linear_velocity(self):
        self.child_point.set_pos(self.parent_point, 0)
        self.parent_point.set_vel(self._parent_frame, 0)
        self.child_point.set_vel(self._child_frame, 0)
        self.child.masscenter.set_vel(self._parent_frame, 0)
```