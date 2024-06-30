# `D:\src\scipysrc\sympy\sympy\physics\mechanics\jointsmethod.py`

```
from sympy.physics.mechanics import (Body, Lagrangian, KanesMethod, LagrangesMethod,
                                    RigidBody, Particle)
from sympy.physics.mechanics.body_base import BodyBase
from sympy.physics.mechanics.method import _Methods
from sympy import Matrix
from sympy.utilities.exceptions import sympy_deprecation_warning

__all__ = ['JointsMethod']

# JointsMethod 类继承自 _Methods，用于处理具有关节连接的多体系统的运动方程
class JointsMethod(_Methods):
    """Method for formulating the equations of motion using a set of interconnected bodies with joints.

    .. deprecated:: 1.13
        The JointsMethod class is deprecated. Its functionality has been
        replaced by the new :class:`~.System` class.

    Parameters
    ==========

    newtonion : Body or ReferenceFrame
        The newtonion(inertial) frame.
    *joints : Joint
        The joints in the system

    Attributes
    ==========

    q, u : iterable
        Iterable of the generalized coordinates and speeds
    bodies : iterable
        Iterable of Body objects in the system.
    loads : iterable
        Iterable of (Point, vector) or (ReferenceFrame, vector) tuples
        describing the forces on the system.
    mass_matrix : Matrix, shape(n, n)
        The system's mass matrix
    forcing : Matrix, shape(n, 1)
        The system's forcing vector
    mass_matrix_full : Matrix, shape(2*n, 2*n)
        The "mass matrix" for the u's and q's
    forcing_full : Matrix, shape(2*n, 1)
        The "forcing vector" for the u's and q's
    method : KanesMethod or Lagrange's method
        Method's object.
    kdes : iterable
        Iterable of kde in they system.

    Examples
    ========

    As Body and JointsMethod have been deprecated, the following examples are
    for illustrative purposes only. The functionality of Body is fully captured
    by :class:`~.RigidBody` and :class:`~.Particle` and the functionality of
    JointsMethod is fully captured by :class:`~.System`. To ignore the
    deprecation warning we can use the ignore_warnings context manager.

    >>> from sympy.utilities.exceptions import ignore_warnings

    This is a simple example for a one degree of freedom translational
    spring-mass-damper.

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import Body, JointsMethod, PrismaticJoint
    >>> from sympy.physics.vector import dynamicsymbols
    >>> c, k = symbols('c k')
    >>> x, v = dynamicsymbols('x v')
    >>> with ignore_warnings(DeprecationWarning):
    ...     wall = Body('W')
    ...     body = Body('B')
    >>> J = PrismaticJoint('J', wall, body, coordinates=x, speeds=v)
    >>> wall.apply_force(c*v*wall.x, reaction_body=body)
    >>> wall.apply_force(k*x*wall.x, reaction_body=body)
    >>> with ignore_warnings(DeprecationWarning):
    ...     method = JointsMethod(wall, J)
    >>> method.form_eoms()
    Matrix([[-B_mass*Derivative(v(t), t) - c*v(t) - k*x(t)]])
    >>> M = method.mass_matrix_full
    >>> F = method.forcing_full
    >>> rhs = M.LUsolve(F)
    >>> rhs
    Matrix([
    [                     v(t)],
    [(-c*v(t) - k*x(t))/B_mass]])

    Notes
    =====

    ``JointsMethod`` currently only works with systems that do not have any
    configuration or motion constraints.

    """

    # JointsMethod 类的初始化函数，已废弃，功能已由新的 System 类取代
    def __init__(self, newtonion, *joints):
        # 打印警告信息，提示 JointsMethod 类已废弃
        sympy_deprecation_warning(
            """
            The JointsMethod class is deprecated.
            Its functionality has been replaced by the new System class.
            """,
            deprecated_since_version="1.13",
            active_deprecations_target="deprecated-mechanics-jointsmethod"
        )
        # 如果 newtonion 是 BodyBase 类的实例，则使用其 frame 属性
        if isinstance(newtonion, BodyBase):
            self.frame = newtonion.frame
        else:
            self.frame = newtonion

        # 将 joints 参数保存到实例属性 _joints 中
        self._joints = joints
        # 生成并保存身体列表到实例属性 _bodies 中
        self._bodies = self._generate_bodylist()
        # 生成并保存载荷列表到实例属性 _loads 中
        self._loads = self._generate_loadlist()
        # 生成并保存广义坐标列表到实例属性 _q 中
        self._q = self._generate_q()
        # 生成并保存广义速度列表到实例属性 _u 中
        self._u = self._generate_u()
        # 生成并保存广义坐标导数列表到实例属性 _kdes 中
        self._kdes = self._generate_kdes()

        # 初始化方法属性 _method 为 None
        self._method = None

    @property
    def bodies(self):
        """List of bodies in they system."""
        # 返回系统中的身体列表
        return self._bodies

    @property
    def loads(self):
        """List of loads on the system."""
        # 返回系统中的载荷列表
        return self._loads

    @property
    def q(self):
        """List of the generalized coordinates."""
        # 返回系统中的广义坐标列表
        return self._q

    @property
    def u(self):
        """List of the generalized speeds."""
        # 返回系统中的广义速度列表
        return self._u

    @property
    def kdes(self):
        """List of the generalized coordinates."""
        # 返回系统中的广义坐标导数列表
        return self._kdes

    @property
    def forcing_full(self):
        """The "forcing vector" for the u's and q's."""
        # 返回方法属性 _method 的 forcing_full 属性
        return self.method.forcing_full

    @property
    def mass_matrix_full(self):
        """The "mass matrix" for the u's and q's."""
        # 返回方法属性 _method 的 mass_matrix_full 属性
        return self.method.mass_matrix_full

    @property
    def mass_matrix(self):
        """The system's mass matrix."""
        # 返回系统的质量矩阵
        return self.method.mass_matrix

    @property
    def forcing(self):
        """The system's forcing vector."""
        # 返回系统的外力向量
        return self.method.forcing

    @property
    def method(self):
        """Object of method used to form equations of systems."""
        # 返回用于形成系统方程的方法对象
        return self._method

    # 生成身体列表的私有方法
    def _generate_bodylist(self):
        bodies = []
        # 遍历所有关节，将其子身体和父身体添加到列表中
        for joint in self._joints:
            if joint.child not in bodies:
                bodies.append(joint.child)
            if joint.parent not in bodies:
                bodies.append(joint.parent)
        return bodies

    # 生成载荷列表的私有方法
    def _generate_loadlist(self):
        load_list = []
        # 遍历所有身体，如果是 Body 类的实例，则将其载荷扩展到列表中
        for body in self.bodies:
            if isinstance(body, Body):
                load_list.extend(body.loads)
        return load_list

    # 生成广义坐标列表的私有方法
    def _generate_q(self):
        q_ind = []
        # 遍历所有关节，将其坐标添加到列表中，确保坐标唯一性
        for joint in self._joints:
            for coordinate in joint.coordinates:
                if coordinate in q_ind:
                    raise ValueError('Coordinates of joints should be unique.')
                q_ind.append(coordinate)
        return Matrix(q_ind)
    # 生成速度矩阵 `u_ind`，其中包含所有关节的速度信息
    def _generate_u(self):
        u_ind = []
        # 遍历所有关节
        for joint in self._joints:
            # 遍历每个关节的速度
            for speed in joint.speeds:
                # 检查速度是否唯一，如果不唯一则抛出数值错误
                if speed in u_ind:
                    raise ValueError('Speeds of joints should be unique.')
                u_ind.append(speed)
        # 将速度列表转换为矩阵并返回
        return Matrix(u_ind)

    # 生成阻尼系数矩阵 `kd_ind`，将所有关节的阻尼系数列连接起来
    def _generate_kdes(self):
        # 初始化阻尼系数矩阵为空列向量
        kd_ind = Matrix(1, 0, []).T
        # 遍历所有关节
        for joint in self._joints:
            # 将当前关节的阻尼系数列连接到 `kd_ind` 矩阵中
            kd_ind = kd_ind.col_join(joint.kdes)
        # 返回连接后的阻尼系数矩阵
        return kd_ind

    # 将所有 `Body` 对象转换为 `Particle` 或 `RigidBody`
    def _convert_bodies(self):
        # 初始化空列表，用于存储转换后的 `Body` 对象
        bodylist = []
        # 遍历所有身体对象
        for body in self.bodies:
            # 如果身体对象不是 `Body` 类型，则直接添加到列表中
            if not isinstance(body, Body):
                bodylist.append(body)
                continue
            # 如果是刚体
            if body.is_rigidbody:
                # 创建对应的 `RigidBody` 对象，并复制相关属性
                rb = RigidBody(body.name, body.masscenter, body.frame, body.mass,
                    (body.central_inertia, body.masscenter))
                rb.potential_energy = body.potential_energy
                bodylist.append(rb)
            else:
                # 创建对应的 `Particle` 对象，并复制相关属性
                part = Particle(body.name, body.masscenter, body.mass)
                part.potential_energy = body.potential_energy
                bodylist.append(part)
        # 返回转换后的身体对象列表
        return bodylist
    def form_eoms(self, method=KanesMethod):
        """Method to form system's equation of motions.

        Parameters
        ==========

        method : Class
            Class name of method.

        Returns
        ========

        Matrix
            Vector of equations of motions.

        Examples
        ========

        As Body and JointsMethod have been deprecated, the following examples
        are for illustrative purposes only. The functionality of Body is fully
        captured by :class:`~.RigidBody` and :class:`~.Particle` and the
        functionality of JointsMethod is fully captured by :class:`~.System`. To
        ignore the deprecation warning we can use the ignore_warnings context
        manager.

        >>> from sympy.utilities.exceptions import ignore_warnings

        This is a simple example for a one degree of freedom translational
        spring-mass-damper.

        >>> from sympy import S, symbols
        >>> from sympy.physics.mechanics import LagrangesMethod, dynamicsymbols, Body
        >>> from sympy.physics.mechanics import PrismaticJoint, JointsMethod
        >>> q = dynamicsymbols('q')
        >>> qd = dynamicsymbols('q', 1)
        >>> m, k, b = symbols('m k b')
        >>> with ignore_warnings(DeprecationWarning):
        ...     wall = Body('W')
        ...     part = Body('P', mass=m)
        >>> part.potential_energy = k * q**2 / S(2)
        >>> J = PrismaticJoint('J', wall, part, coordinates=q, speeds=qd)
        >>> wall.apply_force(b * qd * wall.x, reaction_body=part)
        >>> with ignore_warnings(DeprecationWarning):
        ...     method = JointsMethod(wall, J)
        >>> method.form_eoms(LagrangesMethod)
        Matrix([[b*Derivative(q(t), t) + k*q(t) + m*Derivative(q(t), (t, 2))]])

        We can also solve for the states using the 'rhs' method.

        >>> method.rhs()
        Matrix([
        [                Derivative(q(t), t)],
        [(-b*Derivative(q(t), t) - k*q(t))/m]])

        """

        # Convert bodies to internal representation
        bodylist = self._convert_bodies()
        
        # Check if the provided method is a subclass of LagrangesMethod
        if issubclass(method, LagrangesMethod): #LagrangesMethod or similar
            # Compute the Lagrangian for the system
            L = Lagrangian(self.frame, *bodylist)
            # Initialize the method with LagrangesMethod, system variables, and bodies
            self._method = method(L, self.q, self.loads, bodylist, self.frame)
        else: #KanesMethod or similar
            # Initialize the method with KanesMethod, system variables, and bodies
            self._method = method(self.frame, q_ind=self.q, u_ind=self.u, kd_eqs=self.kdes,
                                    forcelist=self.loads, bodies=bodylist)
        
        # Compute the equations of motion using the selected method
        soln = self.method._form_eoms()
        # Return the computed equations of motion
        return soln
    # 定义一个方法 `rhs`，用于返回可以数值求解的方程组。
    def rhs(self, inv_method=None):
        """Returns equations that can be solved numerically.

        Parameters
        ==========

        inv_method : str
            The specific sympy inverse matrix calculation method to use. For a
            list of valid methods, see
            :meth:`~sympy.matrices.matrixbase.MatrixBase.inv`

        Returns
        ========

        Matrix
            Numerically solvable equations.

        See Also
        ========

        sympy.physics.mechanics.kane.KanesMethod.rhs:
            KanesMethod's rhs function.
        sympy.physics.mechanics.lagrange.LagrangesMethod.rhs:
            LagrangesMethod's rhs function.

        """
        
        # 调用对象的 `method` 属性的 `rhs` 方法，传入 `inv_method` 参数
        return self.method.rhs(inv_method=inv_method)
```