# `D:\src\scipysrc\sympy\sympy\physics\mechanics\body.py`

```
from sympy import Symbol
from sympy.physics.vector import Point, Vector, ReferenceFrame, Dyadic
from sympy.physics.mechanics import RigidBody, Particle, Inertia
from sympy.physics.mechanics.body_base import BodyBase
from sympy.utilities.exceptions import sympy_deprecation_warning

__all__ = ['Body']

# XXX: We use type:ignore because the classes RigidBody and Particle have
# inconsistent parallel axis methods that take different numbers of arguments.
# Body 类继承自 RigidBody 和 Particle 类，用于表示刚体或质点，具体取决于初始化时传入的参数。
class Body(RigidBody, Particle):  # type: ignore
    """
    Body is a common representation of either a RigidBody or a Particle SymPy
    object depending on what is passed in during initialization. If a mass is
    passed in and central_inertia is left as None, the Particle object is
    created. Otherwise a RigidBody object will be created.

    .. deprecated:: 1.13
        The Body class is deprecated. Its functionality is captured by
        :class:`~.RigidBody` and :class:`~.Particle`.

    Explanation
    ===========

    The attributes that Body possesses will be the same as a Particle instance
    or a Rigid Body instance depending on which was created. Additional
    attributes are listed below.

    Attributes
    ==========

    name : string
        The body's name
    masscenter : Point
        The point which represents the center of mass of the rigid body
    frame : ReferenceFrame
        The reference frame which the body is fixed in
    mass : Sympifyable
        The body's mass
    inertia : (Dyadic, Point)
        The body's inertia around its center of mass. This attribute is specific
        to the rigid body form of Body and is left undefined for the Particle
        form
    loads : iterable
        This list contains information on the different loads acting on the
        Body. Forces are listed as a (point, vector) tuple and torques are
        listed as (reference frame, vector) tuples.

    Parameters
    ==========

    name : String
        Defines the name of the body. It is used as the base for defining
        body specific properties.
    masscenter : Point, optional
        A point that represents the center of mass of the body or particle.
        If no point is given, a point is generated.
    mass : Sympifyable, optional
        A Sympifyable object which represents the mass of the body. If no
        mass is passed, one is generated.
    frame : ReferenceFrame, optional
        The ReferenceFrame that represents the reference frame of the body.
        If no frame is given, a frame is generated.
    central_inertia : Dyadic, optional
        Central inertia dyadic of the body. If none is passed while creating
        RigidBody, a default inertia is generated.

    Examples
    ========

    As Body has been deprecated, the following examples are for illustrative
    purposes only. The functionality of Body is fully captured by
    :class:`~.RigidBody` and :class:`~.Particle`. To ignore the deprecation
    """
    # 导入用于忽略警告的上下文管理器 ignore_warnings
    >>> from sympy.utilities.exceptions import ignore_warnings
    
    # 创建一个 Body 对象，其中包含默认值的 RigidBody 版本，具有默认的质量、质心、参考坐标系和惯性属性。
    >>> from sympy.physics.mechanics import Body
    >>> with ignore_warnings(DeprecationWarning):
    ...     body = Body('name_of_body')
    
    # 创建一个包含所有属性值的 Body 对象，同时也创建其 RigidBody 版本。
    >>> from sympy import Symbol
    >>> from sympy.physics.mechanics import ReferenceFrame, Point, inertia
    >>> from sympy.physics.mechanics import Body
    >>> mass = Symbol('mass')
    >>> masscenter = Point('masscenter')
    >>> frame = ReferenceFrame('frame')
    >>> ixx = Symbol('ixx')
    >>> body_inertia = inertia(frame, ixx, 0, 0)
    >>> with ignore_warnings(DeprecationWarning):
    ...     body = Body('name_of_body', masscenter, mass, frame, body_inertia)
    
    # 创建一个粒子版本的 Body 对象，仅需要传入名称和质量。
    >>> from sympy import Symbol
    >>> from sympy.physics.mechanics import Body
    >>> mass = Symbol('mass')
    >>> with ignore_warnings(DeprecationWarning):
    ...     body = Body('name_of_body', mass=mass)
    
    # 创建一个粒子版本的 Body 对象，可以传入质心点和参考坐标系，但不能传入惯性。
    def __init__(self, name, masscenter=None, mass=None, frame=None,
                 central_inertia=None):
        # 发出 sympy 弃用警告，提醒用户不再支持 Body 类，推荐使用 RigidBody 和 Particle
        sympy_deprecation_warning(
            """
            Support for the Body class has been removed, as its functionality is
            fully captured by RigidBody and Particle.
            """,
            deprecated_since_version="1.13",
            active_deprecations_target="deprecated-mechanics-body-class"
        )

        # 初始化负载列表
        self._loads = []

        # 如果未提供参考坐标系，则创建一个新的参考坐标系
        if frame is None:
            frame = ReferenceFrame(name + '_frame')

        # 如果未提供质心，则创建一个新的质心点
        if masscenter is None:
            masscenter = Point(name + '_masscenter')

        # 如果未提供中心惯性和质量，则创建惯性相关的符号变量
        if central_inertia is None and mass is None:
            ixx = Symbol(name + '_ixx')
            iyy = Symbol(name + '_iyy')
            izz = Symbol(name + '_izz')
            izx = Symbol(name + '_izx')
            ixy = Symbol(name + '_ixy')
            iyz = Symbol(name + '_iyz')
            # 根据惯性标量创建惯性对象
            _inertia = Inertia.from_inertia_scalars(masscenter, frame, ixx, iyy,
                                                    izz, ixy, iyz, izx)
        else:
            # 否则直接使用提供的中心惯性和质心
            _inertia = (central_inertia, masscenter)

        # 如果未提供质量，则创建一个新的质量符号变量
        if mass is None:
            _mass = Symbol(name + '_mass')
        else:
            _mass = mass

        # 将质心的速度设置为零
        masscenter.set_vel(frame, 0)

        # 如果用户提供了质心和质量，则创建一个质点；否则创建一个刚体。
        # 注意：使用 BodyBase.__init__ 来避免由于多重继承导致 Particle 和 RigidBody 中 super() 调用的问题。
        if central_inertia is None and mass is not None:
            BodyBase.__init__(self, name, masscenter, _mass)
            self.frame = frame
            self._central_inertia = Dyadic(0)
        else:
            BodyBase.__init__(self, name, masscenter, _mass)
            self.frame = frame
            self.inertia = _inertia

    def __repr__(self):
        # 如果是刚体，则调用 RigidBody 的 __repr__ 方法；否则调用 Particle 的 __repr__ 方法
        if self.is_rigidbody:
            return RigidBody.__repr__(self)
        return Particle.__repr__(self)

    @property
    def loads(self):
        # 返回负载列表
        return self._loads

    @property
    def x(self):
        """The basis Vector for the Body, in the x direction."""
        # 返回质点的基向量，沿 x 方向
        return self.frame.x

    @property
    def y(self):
        """The basis Vector for the Body, in the y direction."""
        # 返回质点的基向量，沿 y 方向
        return self.frame.y

    @property
    def z(self):
        """The basis Vector for the Body, in the z direction."""
        # 返回质点的基向量，沿 z 方向
        return self.frame.z

    @property
    def inertia(self):
        """The body's inertia about a point; stored as (Dyadic, Point)."""
        # 如果是刚体，则返回 RigidBody 的惯性属性；否则返回自定义的中心惯性和质心
        if self.is_rigidbody:
            return RigidBody.inertia.fget(self)
        return (self.central_inertia, self.masscenter)

    @inertia.setter
    def inertia(self, I):
        # 设置惯性属性，用于刚体的惯性设置
        RigidBody.inertia.fset(self, I)

    @property
    def is_rigidbody(self):
        # 检查是否是刚体，判断依据是是否有 _inertia 属性
        if hasattr(self, '_inertia'):
            return True
        return False
    def kinetic_energy(self, frame):
        """
        Kinetic energy of the body.

        Parameters
        ==========

        frame : ReferenceFrame or Body
            The Body's angular velocity and the velocity of it's mass
            center are typically defined with respect to an inertial frame but
            any relevant frame in which the velocities are known can be supplied.

        Examples
        ========

        As Body has been deprecated, the following examples are for illustrative
        purposes only. The functionality of Body is fully captured by
        :class:`~.RigidBody` and :class:`~.Particle`. To ignore the deprecation
        warning we can use the ignore_warnings context manager.

        >>> from sympy.utilities.exceptions import ignore_warnings
        >>> from sympy.physics.mechanics import Body, ReferenceFrame, Point
        >>> from sympy import symbols
        >>> m, v, r, omega = symbols('m v r omega')
        >>> N = ReferenceFrame('N')
        >>> O = Point('O')
        >>> with ignore_warnings(DeprecationWarning):
        ...     P = Body('P', masscenter=O, mass=m)
        >>> P.masscenter.set_vel(N, v * N.y)
        >>> P.kinetic_energy(N)
        m*v**2/2

        >>> N = ReferenceFrame('N')
        >>> b = ReferenceFrame('b')
        >>> b.set_ang_vel(N, omega * b.x)
        >>> P = Point('P')
        >>> P.set_vel(N, v * N.x)
        >>> with ignore_warnings(DeprecationWarning):
        ...     B = Body('B', masscenter=P, frame=b)
        >>> B.kinetic_energy(N)
        B_ixx*omega**2/2 + B_mass*v**2/2

        See Also
        ========

        sympy.physics.mechanics : Particle, RigidBody
        """
        # 如果 frame 是 Body 类的实例，则将 frame 替换为 Body 的 frame 属性
        if isinstance(frame, Body):
            frame = Body.frame
        # 如果 self 是刚体，则调用 RigidBody 类的 kinetic_energy 方法计算动能
        if self.is_rigidbody:
            return RigidBody(self.name, self.masscenter, self.frame, self.mass,
                            (self.central_inertia, self.masscenter)).kinetic_energy(frame)
        # 否则，调用 Particle 类的 kinetic_energy 方法计算动能
        return Particle(self.name, self.masscenter, self.mass).kinetic_energy(frame)

    def clear_loads(self):
        """
        Clears the Body's loads list.

        Example
        =======

        As Body has been deprecated, the following examples are for illustrative
        purposes only. The functionality of Body is fully captured by
        :class:`~.RigidBody` and :class:`~.Particle`. To ignore the deprecation
        warning we can use the ignore_warnings context manager.

        >>> from sympy.utilities.exceptions import ignore_warnings
        >>> from sympy.physics.mechanics import Body
        >>> with ignore_warnings(DeprecationWarning):
        ...     B = Body('B')
        >>> force = B.x + B.y
        >>> B.apply_force(force)
        >>> B.loads
        [(B_masscenter, B_frame.x + B_frame.y)]
        >>> B.clear_loads()
        >>> B.loads
        []

        """
        # 将 Body 对象的 _loads 属性设置为空列表，清空加载列表
        self._loads = []
    def remove_load(self, about=None):
        """
        Remove load about a point or frame.

        Parameters
        ==========

        about : Point or ReferenceFrame, optional
            The point about which force is applied,
            and is to be removed.
            If about is None, then the torque about
            self's frame is removed.

        Example
        =======

        As Body has been deprecated, the following examples are for illustrative
        purposes only. The functionality of Body is fully captured by
        :class:`~.RigidBody` and :class:`~.Particle`. To ignore the deprecation
        warning we can use the ignore_warnings context manager.

        >>> from sympy.utilities.exceptions import ignore_warnings
        >>> from sympy.physics.mechanics import Body, Point
        >>> with ignore_warnings(DeprecationWarning):
        ...     B = Body('B')
        >>> P = Point('P')
        >>> f1 = B.x
        >>> f2 = B.y
        >>> B.apply_force(f1)
        >>> B.apply_force(f2, P)
        >>> B.loads
        [(B_masscenter, B_frame.x), (P, B_frame.y)]

        >>> B.remove_load(P)
        >>> B.loads
        [(B_masscenter, B_frame.x)]

        """

        # 如果指定了 about 参数
        if about is not None:
            # 检查 about 是否为 Point 类型，如果不是则抛出类型错误
            if not isinstance(about, Point):
                raise TypeError('Load is applied about Point or ReferenceFrame.')
        else:
            # 如果 about 参数为 None，则默认为 self 的参考系
            about = self.frame

        # 遍历当前对象的 _loads 属性
        for load in self._loads:
            # 如果指定的 about 在 load 中
            if about in load:
                # 移除该 load
                self._loads.remove(load)
                break

    def masscenter_vel(self, body):
        """
        Returns the velocity of the mass center with respect to the provided
        rigid body or reference frame.

        Parameters
        ==========

        body: Body or ReferenceFrame
            The rigid body or reference frame to calculate the velocity in.

        Example
        =======

        As Body has been deprecated, the following examples are for illustrative
        purposes only. The functionality of Body is fully captured by
        :class:`~.RigidBody` and :class:`~.Particle`. To ignore the deprecation
        warning we can use the ignore_warnings context manager.

        >>> from sympy.utilities.exceptions import ignore_warnings
        >>> from sympy.physics.mechanics import Body
        >>> with ignore_warnings(DeprecationWarning):
        ...     A = Body('A')
        ...     B = Body('B')
        >>> A.masscenter.set_vel(B.frame, 5*B.frame.x)
        >>> A.masscenter_vel(B)
        5*B_frame.x
        >>> A.masscenter_vel(B.frame)
        5*B_frame.x

        """

        # 如果 body 参数是 ReferenceFrame 类型
        if isinstance(body,  ReferenceFrame):
            frame=body
        # 如果 body 参数是 Body 类型
        elif isinstance(body, Body):
            frame = body.frame
        # 返回质心的速度相对于给定参考系 frame 的值
        return self.masscenter.vel(frame)
    def ang_vel_in(self, body):
        """
        Returns this body's angular velocity with respect to the provided
        rigid body or reference frame.

        Parameters
        ==========

        body: Body or ReferenceFrame
            The rigid body or reference frame to calculate the angular velocity in.

        Example
        =======

        As Body has been deprecated, the following examples are for illustrative
        purposes only. The functionality of Body is fully captured by
        :class:`~.RigidBody` and :class:`~.Particle`. To ignore the deprecation
        warning we can use the ignore_warnings context manager.

        >>> from sympy.utilities.exceptions import ignore_warnings
        >>> from sympy.physics.mechanics import Body, ReferenceFrame
        >>> with ignore_warnings(DeprecationWarning):
        ...     A = Body('A')
        >>> N = ReferenceFrame('N')
        >>> with ignore_warnings(DeprecationWarning):
        ...     B = Body('B', frame=N)
        >>> A.frame.set_ang_vel(N, 5*N.x)
        >>> A.ang_vel_in(B)
        5*N.x
        >>> A.ang_vel_in(N)
        5*N.x

        """

        # 如果提供的 body 是 ReferenceFrame 类型
        if isinstance(body,  ReferenceFrame):
            frame=body
        # 如果提供的 body 是 Body 类型
        elif isinstance(body, Body):
            frame = body.frame
        # 返回当前对象所在参考系的角速度相对于指定参考系的角速度
        return self.frame.ang_vel_in(frame)

    def dcm(self, body):
        """
        Returns the direction cosine matrix of this body relative to the
        provided rigid body or reference frame.

        Parameters
        ==========

        body: Body or ReferenceFrame
            The rigid body or reference frame to calculate the dcm.

        Example
        =======

        As Body has been deprecated, the following examples are for illustrative
        purposes only. The functionality of Body is fully captured by
        :class:`~.RigidBody` and :class:`~.Particle`. To ignore the deprecation
        warning we can use the ignore_warnings context manager.

        >>> from sympy.utilities.exceptions import ignore_warnings
        >>> from sympy.physics.mechanics import Body
        >>> with ignore_warnings(DeprecationWarning):
        ...     A = Body('A')
        ...     B = Body('B')
        >>> A.frame.orient_axis(B.frame, B.frame.x, 5)
        >>> A.dcm(B)
        Matrix([
        [1,       0,      0],
        [0,  cos(5), sin(5)],
        [0, -sin(5), cos(5)]])
        >>> A.dcm(B.frame)
        Matrix([
        [1,       0,      0],
        [0,  cos(5), sin(5)],
        [0, -sin(5), cos(5)]])

        """

        # 如果提供的 body 是 ReferenceFrame 类型
        if isinstance(body,  ReferenceFrame):
            frame=body
        # 如果提供的 body 是 Body 类型
        elif isinstance(body, Body):
            frame = body.frame
        # 返回当前对象所在参考系相对于指定参考系的方向余弦矩阵（DCM）
        return self.frame.dcm(frame)
    def parallel_axis(self, point, frame=None):
        """
        Returns the inertia dyadic of the body with respect to another
        point.

        Parameters
        ==========

        point : sympy.physics.vector.Point
            The point to express the inertia dyadic about.
        frame : sympy.physics.vector.ReferenceFrame, optional
            The reference frame used to construct the dyadic. Defaults to None.

        Returns
        =======

        inertia : sympy.physics.vector.Dyadic
            The inertia dyadic of the rigid body expressed about the provided
            point.

        Example
        =======

        As Body has been deprecated, the following examples are for illustrative
        purposes only. The functionality of Body is fully captured by
        :class:`~.RigidBody` and :class:`~.Particle`. To ignore the deprecation
        warning we can use the ignore_warnings context manager.

        >>> from sympy.utilities.exceptions import ignore_warnings
        >>> from sympy.physics.mechanics import Body
        >>> with ignore_warnings(DeprecationWarning):
        ...     A = Body('A')
        >>> P = A.masscenter.locatenew('point', 3 * A.x + 5 * A.y)
        >>> A.parallel_axis(P).to_matrix(A.frame)
        Matrix([
        [A_ixx + 25*A_mass, A_ixy - 15*A_mass,             A_izx],
        [A_ixy - 15*A_mass,  A_iyy + 9*A_mass,             A_iyz],
        [            A_izx,             A_iyz, A_izz + 34*A_mass]])

        """

        # 如果当前对象是刚体，则调用 RigidBody 类的 parallel_axis 方法
        if self.is_rigidbody:
            return RigidBody.parallel_axis(self, point, frame)
        # 否则，调用 Particle 类的 parallel_axis 方法
        return Particle.parallel_axis(self, point, frame)
```