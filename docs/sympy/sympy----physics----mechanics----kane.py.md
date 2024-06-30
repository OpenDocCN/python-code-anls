# `D:\src\scipysrc\sympy\sympy\physics\mechanics\kane.py`

```
from sympy import zeros, Matrix, diff, eye  # 导入sympy库中的zeros, Matrix, diff, eye函数
from sympy.core.sorting import default_sort_key  # 导入sympy库中的default_sort_key函数
from sympy.physics.vector import (ReferenceFrame, dynamicsymbols,  # 导入sympy.physics.vector库中的ReferenceFrame, dynamicsymbols,
                                  partial_velocity)               # partial_velocity函数
from sympy.physics.mechanics.method import _Methods  # 导入sympy.physics.mechanics.method库中的_Methods类
from sympy.physics.mechanics.particle import Particle  # 导入sympy.physics.mechanics.particle库中的Particle类
from sympy.physics.mechanics.rigidbody import RigidBody  # 导入sympy.physics.mechanics.rigidbody库中的RigidBody类
from sympy.physics.mechanics.functions import (msubs, find_dynamicsymbols,  # 导入sympy.physics.mechanics.functions库中的msubs, find_dynamicsymbols,
                                               _f_list_parser,              # _f_list_parser函数
                                               _validate_coordinates,       # _validate_coordinates函数
                                               _parse_linear_solver)        # _parse_linear_solver函数
from sympy.physics.mechanics.linearize import Linearizer  # 导入sympy.physics.mechanics.linearize库中的Linearizer类
from sympy.utilities.iterables import iterable  # 导入sympy.utilities.iterables库中的iterable函数

__all__ = ['KanesMethod']  # 定义__all__列表，指定了导入模块的公共接口

class KanesMethod(_Methods):
    r"""Kane's method object.

    Explanation
    ===========

    This object is used to do the "book-keeping" as you go through and form
    equations of motion in the way Kane presents in:
    Kane, T., Levinson, D. Dynamics Theory and Applications. 1985 McGraw-Hill

    The attributes are for equations in the form [M] udot = forcing.

    Attributes
    ==========

    q, u : Matrix
        Matrices of the generalized coordinates and speeds
    bodies : iterable
        Iterable of Particle and RigidBody objects in the system.
    loads : iterable
        Iterable of (Point, vector) or (ReferenceFrame, vector) tuples
        describing the forces on the system.
    auxiliary_eqs : Matrix
        If applicable, the set of auxiliary Kane's
        equations used to solve for non-contributing
        forces.
    mass_matrix : Matrix
        The system's dynamics mass matrix: [k_d; k_dnh]
    forcing : Matrix
        The system's dynamics forcing vector: -[f_d; f_dnh]
    mass_matrix_kin : Matrix
        The "mass matrix" for kinematic differential equations: k_kqdot
    forcing_kin : Matrix
        The forcing vector for kinematic differential equations: -(k_ku*u + f_k)
    mass_matrix_full : Matrix
        The "mass matrix" for the u's and q's with dynamics and kinematics
    forcing_full : Matrix
        The "forcing vector" for the u's and q's with dynamics and kinematics

    Parameters
    ==========

    frame : ReferenceFrame
        The inertial reference frame for the system.
    q_ind : iterable of dynamicsymbols
        Independent generalized coordinates.
    u_ind : iterable of dynamicsymbols
        Independent generalized speeds.
    kd_eqs : iterable of Expr, optional
        Kinematic differential equations, which linearly relate the generalized
        speeds to the time-derivatives of the generalized coordinates.
    q_dependent : iterable of dynamicsymbols, optional
        Dependent generalized coordinates.
    configuration_constraints : iterable of Expr, optional
        Constraints on the system's configuration, i.e. holonomic constraints.
    # u_dependent 是一个可迭代对象，包含系统中的相关广义速度。
    u_dependent : iterable of dynamicsymbols, optional
    # velocity_constraints 是一个可迭代对象，包含系统速度的约束条件，
    # 即非完整约束和完整约束的时间导数的组合。
    velocity_constraints : iterable of Expr, optional
    # acceleration_constraints 是一个可迭代对象，包含系统加速度的约束条件，
    # 默认为速度约束条件的时间导数。
    acceleration_constraints : iterable of Expr, optional
    # u_auxiliary 是一个可迭代对象，包含系统中的辅助广义速度。
    u_auxiliary : iterable of dynamicsymbols, optional
    # bodies 是一个可迭代对象，包含系统中的粒子和刚体。
    bodies : iterable of Particle and/or RigidBody, optional
    # forcelist 是一个可迭代对象，包含作用在系统上的力和力矩。
    forcelist : iterable of tuple[Point | ReferenceFrame, Vector], optional
    # explicit_kinematics 是一个布尔值，指示质量矩阵和强制向量是否采用显式形式。
    # 默认为 True，即使用显式形式，若设置为 False，则使用隐式形式。
    explicit_kinematics : bool
    # kd_eqs_solver 是一个字符串或可调用对象，用于解解动学微分方程的方法。
    # 若为字符串，则应为与 MatrixBase.solve 方法兼容的有效方法。
    # 若为可调用对象，则应具有格式 "f(A, rhs)"，用于解方程并返回解。默认使用 LU solve。
    kd_eqs_solver : str, callable
    # constraint_solver 是一个字符串或可调用对象，用于解解速度约束条件的方法。
    # 若为字符串，则应为与 MatrixBase.solve 方法兼容的有效方法。
    # 若为可调用对象，则应具有格式 "f(A, rhs)"，用于解方程并返回解。默认使用 LU solve。
    constraint_solver : str, callable

    # 笔记部分说明了动力学方法中的质量矩阵和强制向量默认采用显式形式，
    # 设置 explicit_kinematics 为 False 可以获得更紧凑的非简单动力学方程。
    Notes
    =====
    # 默认情况下，动力学质量矩阵 $\mathbf{k_{k\dot{q}}} = \mathbf{I}$，即单位矩阵。
    # 设置 explicit_kinematics 为 False 将使 $\mathbf{k_{k\dot{q}}}$ 不一定是单位矩阵。
    # 这样可以提供更紧凑的非简单动力学方程。
    The mass matrices and forcing vectors related to kinematic equations
    are given in the explicit form by default. In other words, the kinematic
    mass matrix is $\mathbf{k_{k\dot{q}}} = \mathbf{I}$.
    In order to get the implicit form of those matrices/vectors, you can set the
    ``explicit_kinematics`` attribute to ``False``. So $\mathbf{k_{k\dot{q}}}$
    is not necessarily an identity matrix. This can provide more compact
    equations for non-simple kinematics.

    # KanesMethod 可以使用两种线性求解器：用于解动力学微分方程和解速度约束条件的方法。
    Two linear solvers can be supplied to ``KanesMethod``: one for solving the
    kinematic differential equations and one to solve the velocity constraints.
    # 这些方程组都可以表示为线性系统 "Ax = rhs"，需要求解以获得运动方程。
    Both of these sets of equations can be expressed as a linear system ``Ax = rhs``,
    # 默认求解器为 'LU'，即 LU 分解方法，操作数较低。
    The default solver ``'LU'``, which stands for LU solve, results relatively low
    # 该方法的弱点在于可能导致零除错误。
    number of operations. The weakness of this method is that it can result in zero
    division errors.
    If zero divisions are encountered, a possible solver which may solve the problem
    is ``"CRAMER"``. This method uses Cramer's rule to solve the system. This method
    is slower and results in more operations than the default solver. However it only
    uses a single division by default per entry of the solution.

    While a valid list of solvers can be found at
    :meth:`sympy.matrices.matrixbase.MatrixBase.solve`, it is also possible to supply a
    `callable`. This way it is possible to use a different solver routine. If the
    kinematic differential equations are not too complex it can be worth it to simplify
    the solution by using ``lambda A, b: simplify(Matrix.LUsolve(A, b))``. Another
    option solver one may use is :func:`sympy.solvers.solveset.linsolve`. This can be
    done using `lambda A, b: tuple(linsolve((A, b)))[0]`, where we select the first
    solution as our system should have only one unique solution.

    Examples
    ========

    This is a simple example for a one degree of freedom translational
    spring-mass-damper.

    In this example, we first need to do the kinematics.
    This involves creating generalized speeds and coordinates and their
    derivatives.
    Then we create a point and set its velocity in a frame.

        >>> from sympy import symbols
        >>> from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame
        >>> from sympy.physics.mechanics import Point, Particle, KanesMethod
        >>> q, u = dynamicsymbols('q u')
        >>> qd, ud = dynamicsymbols('q u', 1)
        >>> m, c, k = symbols('m c k')
        >>> N = ReferenceFrame('N')
        >>> P = Point('P')
        >>> P.set_vel(N, u * N.x)

    Next we need to arrange/store information in the way that KanesMethod
    requires. The kinematic differential equations should be an iterable of
    expressions. A list of forces/torques must be constructed, where each entry
    in the list is a (Point, Vector) or (ReferenceFrame, Vector) tuple, where
    the Vectors represent the Force or Torque.
    Next a particle needs to be created, and it needs to have a point and mass
    assigned to it.
    Finally, a list of all bodies and particles needs to be created.

        >>> kd = [qd - u]
        >>> FL = [(P, (-k * q - c * u) * N.x)]
        >>> pa = Particle('pa', P, m)
        >>> BL = [pa]

    Finally we can generate the equations of motion.
    First we create the KanesMethod object and supply an inertial frame,
    coordinates, generalized speeds, and the kinematic differential equations.
    Additional quantities such as configuration and motion constraints,
    dependent coordinates and speeds, and auxiliary speeds are also supplied
    here (see the online documentation).
    Next we form FR* and FR to complete: Fr + Fr* = 0.
    We have the equations of motion at this point.
    It makes sense to rearrange them though, so we calculate the mass matrix and
    """
    the forcing terms, for E.o.M. in the form: [MM] udot = forcing, where MM is
    the mass matrix, udot is a vector of the time derivatives of the
    generalized speeds, and forcing is a vector representing "forcing" terms.

        >>> KM = KanesMethod(N, q_ind=[q], u_ind=[u], kd_eqs=kd)
        >>> (fr, frstar) = KM.kanes_equations(BL, FL)
        >>> MM = KM.mass_matrix
        >>> forcing = KM.forcing
        >>> rhs = MM.inv() * forcing
        >>> rhs
        Matrix([[(-c*u(t) - k*q(t))/m]])
        >>> KM.linearize(A_and_B=True)[0]
        Matrix([
        [   0,    1],
        [-k/m, -c/m]])

    Please look at the documentation pages for more information on how to
    perform linearization and how to deal with dependent coordinates & speeds,
    and how do deal with bringing non-contributing forces into evidence.

    """

    # 初始化函数，设置动力学系统的初始条件和约束
    def __init__(self, frame, q_ind, u_ind, kd_eqs=None, q_dependent=None,
                 configuration_constraints=None, u_dependent=None,
                 velocity_constraints=None, acceleration_constraints=None,
                 u_auxiliary=None, bodies=None, forcelist=None,
                 explicit_kinematics=True, kd_eqs_solver='LU',
                 constraint_solver='LU'):

        """Please read the online documentation. """
        # 如果未指定广义坐标，添加一个虚拟的广义坐标和对应的运动微分方程
        if not q_ind:
            q_ind = [dynamicsymbols('dummy_q')]
            kd_eqs = [dynamicsymbols('dummy_kd')]

        # 确保惯性参考系是惯性参考系对象
        if not isinstance(frame, ReferenceFrame):
            raise TypeError('An inertial ReferenceFrame must be supplied')
        self._inertial = frame

        # 初始化私有属性
        self._fr = None
        self._frstar = None

        # 设置力和刚体列表
        self._forcelist = forcelist
        self._bodylist = bodies

        # 设置是否使用显式运动学
        self.explicit_kinematics = explicit_kinematics
        self._constraint_solver = constraint_solver

        # 初始化广义坐标和速度
        self._initialize_vectors(q_ind, q_dependent, u_ind, u_dependent,
                u_auxiliary)
        _validate_coordinates(self.q, self.u)

        # 初始化运动微分方程的矩阵
        self._initialize_kindiffeq_matrices(kd_eqs, kd_eqs_solver)

        # 初始化约束条件的矩阵
        self._initialize_constraint_matrices(
            configuration_constraints, velocity_constraints,
            acceleration_constraints, constraint_solver)
    # 初始化坐标和速度向量
    def _initialize_vectors(self, q_ind, q_dep, u_ind, u_dep, u_aux):
        """Initialize the coordinate and speed vectors."""
        
        # 处理空值的 lambda 函数
        none_handler = lambda x: Matrix(x) if x else Matrix()

        # 初始化广义坐标
        q_dep = none_handler(q_dep)  # 处理依赖坐标
        if not iterable(q_ind):
            raise TypeError('Generalized coordinates must be an iterable.')  # 抛出类型错误，如果广义坐标不是可迭代对象
        if not iterable(q_dep):
            raise TypeError('Dependent coordinates must be an iterable.')  # 抛出类型错误，如果依赖坐标不是可迭代对象
        q_ind = Matrix(q_ind)  # 将广义坐标转换为矩阵形式
        self._qdep = q_dep  # 设置对象的依赖坐标
        self._q = Matrix([q_ind, q_dep])  # 构造广义坐标向量
        self._qdot = self.q.diff(dynamicsymbols._t)  # 计算广义坐标的时间导数

        # 初始化广义速度
        u_dep = none_handler(u_dep)  # 处理依赖速度
        if not iterable(u_ind):
            raise TypeError('Generalized speeds must be an iterable.')  # 抛出类型错误，如果广义速度不是可迭代对象
        if not iterable(u_dep):
            raise TypeError('Dependent speeds must be an iterable.')  # 抛出类型错误，如果依赖速度不是可迭代对象
        u_ind = Matrix(u_ind)  # 将广义速度转换为矩阵形式
        self._udep = u_dep  # 设置对象的依赖速度
        self._u = Matrix([u_ind, u_dep])  # 构造广义速度向量
        self._udot = self.u.diff(dynamicsymbols._t)  # 计算广义速度的时间导数
        self._uaux = none_handler(u_aux)  # 处理辅助速度

    # 构造广义主动力
    def _form_fr(self, fl):
        """Form the generalized active force."""
        if fl is not None and (len(fl) == 0 or not iterable(fl)):
            raise ValueError('Force pairs must be supplied in an '
                'non-empty iterable or None.')  # 如果力对不是非空可迭代对象或者为 None，则抛出值错误

        N = self._inertial  # 获取惯性对象
        # 解析力对，提取相关速度以构造偏导速度
        vel_list, f_list = _f_list_parser(fl, N)
        vel_list = [msubs(i, self._qdot_u_map) for i in vel_list]  # 替换偏导速度映射中的变量
        f_list = [msubs(i, self._qdot_u_map) for i in f_list]  # 替换偏导速度映射中的变量

        # 用偏导速度和力的点积填充 FR
        o = len(self.u)  # 广义速度向量的长度
        b = len(f_list)  # 力列表的长度
        FR = zeros(o, 1)  # 初始化广义力向量
        partials = partial_velocity(vel_list, self.u, N)  # 计算偏导速度
        for i in range(o):
            FR[i] = sum(partials[j][i].dot(f_list[j]) for j in range(b))  # 计算广义力向量的每个分量

        # 如果存在依赖速度
        if self._udep:
            p = o - len(self._udep)  # 计算非依赖速度的索引范围
            FRtilde = FR[:p, 0]  # 提取非依赖速度部分的广义力
            FRold = FR[p:o, 0]  # 提取依赖速度部分的广义力
            FRtilde += self._Ars.T * FRold  # 根据系统矩阵转置对依赖速度部分的广义力进行调整
            FR = FRtilde  # 更新广义力向量为调整后的值

        self._forcelist = fl  # 存储力对列表
        self._fr = FR  # 存储构造的广义力
        return FR  # 返回构造的广义力

    # TODO : Remove `new_method` after 1.1 has been released.
    def linearize(self, *, new_method=None, linear_solver='LU', **kwargs):
        """
        Linearize the equations of motion about a symbolic operating point.

        Parameters
        ==========
        new_method
            Deprecated, does nothing and will be removed.
        linear_solver : str, callable
            Method used to solve the several symbolic linear systems of the
            form ``A*x=b`` in the linearization process. If a string is
            supplied, it should be a valid method that can be used with the
            :meth:`sympy.matrices.matrixbase.MatrixBase.solve`. If a callable is
            supplied, it should have the format ``x = f(A, b)``, where it
            solves the equations and returns the solution. The default is
            ``'LU'`` which corresponds to SymPy's ``A.LUsolve(b)``.
            ``LUsolve()`` is fast to compute but will often result in
            divide-by-zero and thus ``nan`` results.
        **kwargs
            Extra keyword arguments are passed to
            :meth:`sympy.physics.mechanics.linearize.Linearizer.linearize`.

        Explanation
        ===========
        
        If kwarg A_and_B is False (default), returns M, A, B, r for the
        linearized form, M*[q', u']^T = A*[q_ind, u_ind]^T + B*r.
        
        If kwarg A_and_B is True, returns A, B, r for the linearized form
        dx = A*x + B*r, where x = [q_ind, u_ind]^T. Note that this is
        computationally intensive if there are many symbolic parameters. For
        this reason, it may be more desirable to use the default A_and_B=False,
        returning M, A, and B. Values may then be substituted in to these
        matrices, and the state space form found as
        A = P.T*M.inv()*A, B = P.T*M.inv()*B, where P = Linearizer.perm_mat.
        
        In both cases, r is found as all dynamicsymbols in the equations of
        motion that are not part of q, u, q', or u'. They are sorted in
        canonical form.
        
        The operating points may be also entered using the ``op_point`` kwarg.
        This takes a dictionary of {symbol: value}, or a an iterable of such
        dictionaries. The values may be numeric or symbolic. The more values
        you can specify beforehand, the faster this computation will run.
        
        For more documentation, please see the ``Linearizer`` class.
        """
        
        # Convert self to a Linearizer object using specified linear_solver
        linearizer = self.to_linearizer(linear_solver=linear_solver)
        
        # Call linearize method of the Linearizer object with provided kwargs
        result = linearizer.linearize(**kwargs)
        
        # Return the result tuple along with linearizer.r, which contains
        # additional dynamicsymbols sorted in canonical form
        return result + (linearizer.r,)
    def kanes_equations(self, bodies=None, loads=None):
        """ Method to form Kane's equations, Fr + Fr* = 0.

        Explanation
        ===========
        
        Returns (Fr, Fr*). In the case where auxiliary generalized speeds are
        present (say, s auxiliary speeds, o generalized speeds, and m motion
        constraints) the length of the returned vectors will be o - m + s in
        length. The first o - m equations will be the constrained Kane's
        equations, then the s auxiliary Kane's equations. These auxiliary
        equations can be accessed with the auxiliary_eqs property.

        Parameters
        ==========

        bodies : iterable
            An iterable of all RigidBody's and Particle's in the system.
            A system must have at least one body.
        loads : iterable
            Takes in an iterable of (Particle, Vector) or (ReferenceFrame, Vector)
            tuples which represent the force at a point or torque on a frame.
            Must be either a non-empty iterable of tuples or None which corresponds
            to a system with no constraints.
        """
        # 如果 bodies 参数未提供，则默认为对象自身的 bodies 属性
        if bodies is None:
            bodies = self.bodies
        # 如果 loads 参数未提供且存在 self._forcelist，则将 loads 设置为 self._forcelist
        if loads is None and self._forcelist is not None:
            loads = self._forcelist
        # 如果 loads 为空列表，则将 loads 设置为 None
        if loads == []:
            loads = None
        # 如果未定义 kinematic differential equations (_k_kqdot 为 False)，则引发 AttributeError
        if not self._k_kqdot:
            raise AttributeError('Create an instance of KanesMethod with '
                    'kinematic differential equations to use this method.')
        
        # 计算广义力 Fr
        fr = self._form_fr(loads)
        # 计算广义力 Fr*
        frstar = self._form_frstar(bodies)
        
        # 如果存在辅助广义速度 (_uaux) 的情况下
        if self._uaux:
            # 如果没有依赖速度 (_udep) 的情况下
            if not self._udep:
                # 创建 KanesMethod 实例，仅使用辅助广义速度
                km = KanesMethod(self._inertial, self.q, self._uaux,
                             u_auxiliary=self._uaux, constraint_solver=self._constraint_solver)
            else:
                # 创建 KanesMethod 实例，使用辅助广义速度和依赖速度
                km = KanesMethod(self._inertial, self.q, self._uaux,
                        u_auxiliary=self._uaux, u_dependent=self._udep,
                        velocity_constraints=(self._k_nh * self.u +
                        self._f_nh),
                        acceleration_constraints=(self._k_dnh * self._udot +
                        self._f_dnh),
                        constraint_solver=self._constraint_solver
                        )
            
            # 将对象的速度映射 _qdot_u_map 设置为 KanesMethod 实例的映射
            km._qdot_u_map = self._qdot_u_map
            self._km = km
            
            # 计算辅助广义力 fraux
            fraux = km._form_fr(loads)
            # 计算辅助广义力矩 frstaraux
            frstaraux = km._form_frstar(bodies)
            
            # 将辅助方程组保存到对象的 _aux_eq 属性中
            self._aux_eq = fraux + frstaraux
            # 将主要的广义力和广义力矩与辅助部分连接后保存到对象的 _fr 和 _frstar 属性中
            self._fr = fr.col_join(fraux)
            self._frstar = frstar.col_join(frstaraux)
        
        # 返回主要的广义力和广义力矩
        return (self._fr, self._frstar)

    def _form_eoms(self):
        # 调用 kanes_equations 方法计算广义力和广义力矩，并返回它们的和作为方程组
        fr, frstar = self.kanes_equations(self.bodylist, self.forcelist)
        return fr + frstar
    @property
    def rhs(self, inv_method=None):
        """Returns the system's equations of motion in first order form. The
        output is the right hand side of::

           x' = |q'| =: f(q, u, r, p, t)
                |u'|

        The right hand side is what is needed by most numerical ODE
        integrators.

        Parameters
        ==========

        inv_method : str
            The specific sympy inverse matrix calculation method to use. For a
            list of valid methods, see
            :meth:`~sympy.matrices.matrixbase.MatrixBase.inv`

        """
        # 创建一个长度为 q + u 的零向量
        rhs = zeros(len(self.q) + len(self.u), 1)
        
        # 获取运动方程的差分字典
        kdes = self.kindiffdict()
        
        # 遍历广义坐标 q，将对应的运动方程添加到 rhs 中
        for i, q_i in enumerate(self.q):
            rhs[i] = kdes[q_i.diff()]

        # 根据给定的逆矩阵计算方法，计算系统的强制向量
        if inv_method is None:
            rhs[len(self.q):, 0] = self.mass_matrix.LUsolve(self.forcing)
        else:
            rhs[len(self.q):, 0] = (self.mass_matrix.inv(inv_method,
                                                         try_block_diag=True) *
                                    self.forcing)

        # 返回计算得到的右手边向量 rhs
        return rhs

    def kindiffdict(self):
        """Returns a dictionary mapping q' to u."""
        # 检查是否已经建立了广义速度到广义坐标导数的映射关系
        if not self._qdot_u_map:
            raise AttributeError('Create an instance of KanesMethod with '
                    'kinematic differential equations to use this method.')
        # 返回广义速度到广义坐标导数的映射字典
        return self._qdot_u_map

    @property
    def auxiliary_eqs(self):
        """A matrix containing the auxiliary equations."""
        # 检查是否已经计算了 Fr 和 Fr*
        if not self._fr or not self._frstar:
            raise ValueError('Need to compute Fr, Fr* first.')
        # 检查是否声明了辅助速度，返回辅助方程组的矩阵形式
        if not self._uaux:
            raise ValueError('No auxiliary speeds have been declared.')
        return self._aux_eq

    @property
    def mass_matrix_kin(self):
        r"""The kinematic "mass matrix" $\mathbf{k_{k\dot{q}}}$ of the system."""
        # 如果显式指定了运动学质量矩阵，返回显式的质量矩阵
        return self._k_kqdot if self.explicit_kinematics else self._k_kqdot_implicit

    @property
    def forcing_kin(self):
        """The kinematic "forcing vector" of the system."""
        # 如果使用显式的运动学描述，返回显式的强制向量
        if self.explicit_kinematics:
            return -(self._k_ku * Matrix(self.u) + self._f_k)
        else:
            return -(self._k_ku_implicit * Matrix(self.u) + self._f_k_implicit)

    @property
    def mass_matrix(self):
        """The mass matrix of the system."""
        # 检查是否已经计算了 Fr 和 Fr*
        if not self._fr or not self._frstar:
            raise ValueError('Need to compute Fr, Fr* first.')
        # 返回系统的质量矩阵
        return Matrix([self._k_d, self._k_dnh])

    @property
    def forcing(self):
        """The forcing vector of the system."""
        # 检查是否已经计算了 Fr 和 Fr*
        if not self._fr or not self._frstar:
            raise ValueError('Need to compute Fr, Fr* first.')
        # 返回系统的强制向量
        return -Matrix([self._f_d, self._f_dnh])
    # 返回系统的完整质量矩阵，包括显式或隐式形式的运动学微分方程
    def mass_matrix_full(self):
        """The mass matrix of the system, augmented by the kinematic
        differential equations in explicit or implicit form."""
        # 如果尚未计算 Fr 或 Fr*，则引发数值错误
        if not self._fr or not self._frstar:
            raise ValueError('Need to compute Fr, Fr* first.')
        # 获取运动变量和广义坐标的长度
        o, n = len(self.u), len(self.q)
        # 返回完整的质量矩阵，将运动学微分方程连接到质量矩阵的右侧，零填充补足空白
        return (self.mass_matrix_kin.row_join(zeros(n, o))).col_join(
            zeros(o, n).row_join(self.mass_matrix))

    @property
    # 返回系统的完整强制向量，包括显式或隐式形式的运动学微分方程
    def forcing_full(self):
        """The forcing vector of the system, augmented by the kinematic
        differential equations in explicit or implicit form."""
        # 返回包含运动学微分方程的强制向量
        return Matrix([self.forcing_kin, self.forcing])

    @property
    # 返回系统的广义坐标
    def q(self):
        return self._q

    @property
    # 返回系统的广义速度
    def u(self):
        return self._u

    @property
    # 返回系统的刚体列表
    def bodylist(self):
        return self._bodylist

    @property
    # 返回系统的力列表
    def forcelist(self):
        return self._forcelist

    @property
    # 返回系统的刚体列表（别名）
    def bodies(self):
        return self._bodylist

    @property
    # 返回系统的加载列表（别名）
    def loads(self):
        return self._forcelist
```