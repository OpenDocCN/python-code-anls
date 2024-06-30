# `D:\src\scipysrc\sympy\sympy\physics\mechanics\lagrange.py`

```
# 从 sympy 库导入必要的函数和类
from sympy import diff, zeros, Matrix, eye, sympify
from sympy.core.sorting import default_sort_key
from sympy.physics.vector import dynamicsymbols, ReferenceFrame
from sympy.physics.mechanics.method import _Methods
from sympy.physics.mechanics.functions import (
    find_dynamicsymbols, msubs, _f_list_parser, _validate_coordinates)
from sympy.physics.mechanics.linearize import Linearizer
from sympy.utilities.iterables import iterable

# 设置模块的公开接口，这里只导出 LagrangesMethod 类
__all__ = ['LagrangesMethod']

# LagrangesMethod 类继承自 _Methods 类，用于执行拉格朗日方法生成运动方程
class LagrangesMethod(_Methods):
    """Lagrange's method object.

    Explanation
    ===========

    This object generates the equations of motion in a two step procedure. The
    first step involves the initialization of LagrangesMethod by supplying the
    Lagrangian and the generalized coordinates, at the bare minimum. If there
    are any constraint equations, they can be supplied as keyword arguments.
    The Lagrange multipliers are automatically generated and are equal in
    number to the constraint equations. Similarly any non-conservative forces
    can be supplied in an iterable (as described below and also shown in the
    example) along with a ReferenceFrame. This is also discussed further in the
    __init__ method.

    Attributes
    ==========

    q, u : Matrix
        Matrices of the generalized coordinates and speeds
    loads : iterable
        Iterable of (Point, vector) or (ReferenceFrame, vector) tuples
        describing the forces on the system.
    bodies : iterable
        Iterable containing the rigid bodies and particles of the system.
    mass_matrix : Matrix
        The system's mass matrix
    forcing : Matrix
        The system's forcing vector
    mass_matrix_full : Matrix
        The "mass matrix" for the qdot's, qdoubledot's, and the
        lagrange multipliers (lam)
    forcing_full : Matrix
        The forcing vector for the qdot's, qdoubledot's and
        lagrange multipliers (lam)

    Examples
    ========

    This is a simple example for a one degree of freedom translational
    spring-mass-damper.

    In this example, we first need to do the kinematics.
    This involves creating generalized coordinates and their derivatives.
    Then we create a point and set its velocity in a frame.

        >>> from sympy.physics.mechanics import LagrangesMethod, Lagrangian
        >>> from sympy.physics.mechanics import ReferenceFrame, Particle, Point
        >>> from sympy.physics.mechanics import dynamicsymbols
        >>> from sympy import symbols
        >>> q = dynamicsymbols('q')
        >>> qd = dynamicsymbols('q', 1)
        >>> m, k, b = symbols('m k b')
        >>> N = ReferenceFrame('N')
        >>> P = Point('P')
        >>> P.set_vel(N, qd * N.x)

    We need to then prepare the information as required by LagrangesMethod to
    generate equations of motion.
    First we create the Particle, which has a point attached to it.
    Following this the lagrangian is created from the kinetic and potential
    energies.
    """
    pass  # LagrangesMethod 类的实现将在这里继续完成，这里暂时没有具体的实现内容
    Then, an iterable of nonconservative forces/torques must be constructed,
    where each item is a (Point, Vector) or (ReferenceFrame, Vector) tuple,
    with the Vectors representing the nonconservative forces or torques.

        >>> Pa = Particle('Pa', P, m)
        >>> Pa.potential_energy = k * q**2 / 2.0
        >>> L = Lagrangian(N, Pa)
        >>> fl = [(P, -b * qd * N.x)]

    Finally we can generate the equations of motion.
    First we create the LagrangesMethod object. To do this one must supply
    the Lagrangian, and the generalized coordinates. The constraint equations,
    the forcelist, and the inertial frame may also be provided, if relevant.
    Next we generate Lagrange's equations of motion, such that:
    Lagrange's equations of motion = 0.
    We have the equations of motion at this point.

        >>> l = LagrangesMethod(L, [q], forcelist = fl, frame = N)
        >>> print(l.form_lagranges_equations())
        Matrix([[b*Derivative(q(t), t) + 1.0*k*q(t) + m*Derivative(q(t), (t, 2))]])

    We can also solve for the states using the 'rhs' method.

        >>> print(l.rhs())
        Matrix([[Derivative(q(t), t)], [(-b*Derivative(q(t), t) - 1.0*k*q(t))/m]])

    Please refer to the docstrings on each method for more details.


注释：


# 创建一个粒子 Pa，并定义其势能
>>> Pa = Particle('Pa', P, m)
>>> Pa.potential_energy = k * q**2 / 2.0

# 使用粒子 Pa 创建一个 Lagrangian 对象 L
>>> L = Lagrangian(N, Pa)

# 创建非保守力/力矩的列表 fl，每个元素为 (Point, Vector) 或 (ReferenceFrame, Vector) 元组，
# 其中 Vector 表示非保守力或力矩
>>> fl = [(P, -b * qd * N.x)]

# 创建 LagrangesMethod 对象 l，用于生成运动方程
# 参数包括 Lagrangian 对象 L，广义坐标列表 [q]，非保守力列表 forcelist=fl，惯性参考系 frame=N
>>> l = LagrangesMethod(L, [q], forcelist=fl, frame=N)

# 输出 Lagrange 方程组
>>> print(l.form_lagranges_equations())
Matrix([[b*Derivative(q(t), t) + 1.0*k*q(t) + m*Derivative(q(t), (t, 2))]])

# 使用 'rhs' 方法求解状态方程
>>> print(l.rhs())
Matrix([[Derivative(q(t), t)], [(-b*Derivative(q(t), t) - 1.0*k*q(t))/m]])

# 更多细节请参考各方法的文档字符串。
    # 初始化函数，用于初始化 LagrangesMethod 类
    def __init__(self, Lagrangian, qs, forcelist=None, bodies=None, frame=None,
                 hol_coneqs=None, nonhol_coneqs=None):
        """Supply the following for the initialization of LagrangesMethod.

        Lagrangian : Sympifyable
            The Lagrangian of the system, to be converted into a symbolic expression.

        qs : array_like
            The generalized coordinates that describe the system's configuration.

        hol_coneqs : array_like, optional
            The holonomic constraint equations governing the system's motion.

        nonhol_coneqs : array_like, optional
            The nonholonomic constraint equations governing the system's motion.

        forcelist : iterable, optional
            Represents forces or torques applied to points or frames within the system.

        bodies : iterable, optional
            Contains rigid bodies and particles composing the system.

        frame : ReferenceFrame, optional
            Specifies the inertial frame used for calculations involving forces.

        """

        # Initialize the Lagrangian as a symbolic expression
        self._L = Matrix([sympify(Lagrangian)])

        # Initialize variables for storing equations of motion
        self.eom = None
        self._m_cd = Matrix()           # Mass Matrix of differentiated coneqs
        self._m_d = Matrix()            # Mass Matrix of dynamic equations
        self._f_cd = Matrix()           # Forcing part of the diff coneqs
        self._f_d = Matrix()            # Forcing part of the dynamic equations
        self.lam_coeffs = Matrix()      # The coefficients of the multipliers

        # Handle forcelist input, ensuring it is iterable
        forcelist = forcelist if forcelist else []
        if not iterable(forcelist):
            raise TypeError('Force pairs must be supplied in an iterable.')
        self._forcelist = forcelist

        # Validate and set inertial frame
        if frame and not isinstance(frame, ReferenceFrame):
            raise TypeError('frame must be a valid ReferenceFrame')
        self._bodies = bodies
        self.inertial = frame

        # Initialize the Lagrange multipliers vector
        self.lam_vec = Matrix()

        # Initialize terms for storing intermediate calculations
        self._term1 = Matrix()
        self._term2 = Matrix()
        self._term3 = Matrix()
        self._term4 = Matrix()

        # Creating the generalized coordinates, velocities, and accelerations
        if not iterable(qs):
            raise TypeError('Generalized coordinates must be an iterable')
        self._q = Matrix(qs)
        self._qdots = self.q.diff(dynamicsymbols._t)   # Derivative of q with respect to time
        self._qdoubledots = self._qdots.diff(dynamicsymbols._t)   # Second derivative of q with respect to time
        _validate_coordinates(self.q)   # Validate the generalized coordinates

        # Lambda function to build matrices from equations; handle optional arguments
        mat_build = lambda x: Matrix(x) if x else Matrix()
        hol_coneqs = mat_build(hol_coneqs)
        nonhol_coneqs = mat_build(nonhol_coneqs)

        # Constructing constraint equations matrix
        self.coneqs = Matrix([hol_coneqs.diff(dynamicsymbols._t),
                nonhol_coneqs])
        self._hol_coneqs = hol_coneqs
    def form_lagranges_equations(self):
        """Method to form Lagrange's equations of motion.

        Returns a vector of equations of motion using Lagrange's equations of
        the second kind.
        """

        qds = self._qdots  # 获取广义速度的符号列表
        qdd_zero = dict.fromkeys(self._qdoubledots, 0)  # 创建一个字典，将广义加速度初始化为零
        n = len(self.q)  # 获取广义坐标的数量

        # Internally we represent the EOM as four terms:
        # EOM = term1 - term2 - term3 - term4 = 0

        # First term
        self._term1 = self._L.jacobian(qds)  # 计算拉格朗日量对广义速度的雅可比矩阵
        self._term1 = self._term1.diff(dynamicsymbols._t).T  # 对时间求导数并转置

        # Second term
        self._term2 = self._L.jacobian(self.q).T  # 计算拉格朗日量对广义坐标的雅可比矩阵并转置

        # Third term
        if self.coneqs:
            coneqs = self.coneqs
            m = len(coneqs)
            # Creating the multipliers
            self.lam_vec = Matrix(dynamicsymbols('lam1:' + str(m + 1)))  # 创建拉格朗日乘子向量
            self.lam_coeffs = -coneqs.jacobian(qds)  # 计算约束方程对广义速度的雅可比矩阵的负值
            self._term3 = self.lam_coeffs.T * self.lam_vec  # 计算第三项
            # Extracting the coefficients of the qdds from the differential constraint equations
            diffconeqs = coneqs.diff(dynamicsymbols._t)
            self._m_cd = diffconeqs.jacobian(self._qdoubledots)  # 计算加速度的系数
            # The remaining terms i.e. the 'forcing' terms in differential constraint equations
            self._f_cd = -diffconeqs.subs(qdd_zero)  # 计算剩余项
        else:
            self._term3 = zeros(n, 1)  # 若无约束方程，第三项为零向量

        # Fourth term
        if self.forcelist:
            N = self.inertial
            self._term4 = zeros(n, 1)  # 初始化第四项为零向量
            for i, qd in enumerate(qds):
                flist = zip(*_f_list_parser(self.forcelist, N))
                self._term4[i] = sum(v.diff(qd, N).dot(f) for (v, f) in flist)  # 计算第四项的每一项
        else:
            self._term4 = zeros(n, 1)  # 若无外力列表，第四项为零向量

        # Form the dynamic mass and forcing matrices
        without_lam = self._term1 - self._term2 - self._term4  # 计算不包含拉格朗日乘子的动态质量和强迫矩阵
        self._m_d = without_lam.jacobian(self._qdoubledots)  # 计算动态质量矩阵
        self._f_d = -without_lam.subs(qdd_zero)  # 计算强迫矩阵

        # Form the EOM
        self.eom = without_lam - self._term3  # 计算运动方程
        return self.eom  # 返回运动方程向量

    def _form_eoms(self):
        return self.form_lagranges_equations()  # 调用计算拉格朗日方程的方法

    @property
    def mass_matrix(self):
        """Returns the mass matrix, which is augmented by the Lagrange
        multipliers, if necessary.

        Explanation
        ===========

        If the system is described by 'n' generalized coordinates and there are
        no constraint equations then an n X n matrix is returned.

        If there are 'n' generalized coordinates and 'm' constraint equations
        have been supplied during initialization then an n X (n+m) matrix is
        returned. The (n + m - 1)th and (n + m)th columns contain the
        coefficients of the Lagrange multipliers.
        """

        if self.eom is None:
            raise ValueError('Need to compute the equations of motion first')
        if self.coneqs:
            return (self._m_d).row_join(self.lam_coeffs.T)  # 若有约束方程，返回动态质量矩阵和拉格朗日乘子的转置
        else:
            return self._m_d  # 若无约束方程，只返回动态质量矩阵
    # 计算完整的质量矩阵，将 qdots 的系数增加到质量矩阵中
    def mass_matrix_full(self):
        """Augments the coefficients of qdots to the mass_matrix."""

        # 如果尚未计算运动方程，则引发数值错误
        if self.eom is None:
            raise ValueError('Need to compute the equations of motion first')
        
        # 获取广义坐标 q 的数量
        n = len(self.q)
        # 获取约束条件的数量
        m = len(self.coneqs)
        
        # 构造质量矩阵的第一行，包括单位矩阵和零矩阵
        row1 = eye(n).row_join(zeros(n, n + m))
        # 构造质量矩阵的第二行，包括零矩阵和已有的质量矩阵
        row2 = zeros(n, n).row_join(self.mass_matrix)
        
        # 如果存在约束条件，则构造质量矩阵的第三行
        if self.coneqs:
            # 包括零矩阵、_m_cd 矩阵和另一个零矩阵
            row3 = zeros(m, n).row_join(self._m_cd).row_join(zeros(m, m))
            # 返回三行合并成的完整质量矩阵
            return row1.col_join(row2).col_join(row3)
        else:
            # 没有约束条件时，返回两行合并成的质量矩阵
            return row1.col_join(row2)

    @property
    def forcing(self):
        """Returns the forcing vector from 'lagranges_equations' method."""

        # 如果尚未计算运动方程，则引发数值错误
        if self.eom is None:
            raise ValueError('Need to compute the equations of motion first')
        
        # 返回强制项向量 _f_d
        return self._f_d

    @property
    def forcing_full(self):
        """Augments qdots to the forcing vector above."""

        # 如果尚未计算运动方程，则引发数值错误
        if self.eom is None:
            raise ValueError('Need to compute the equations of motion first')
        
        # 如果存在约束条件，则返回 _qdots、forcing 和 _f_cd 合并而成的向量
        if self.coneqs:
            return self._qdots.col_join(self.forcing).col_join(self._f_cd)
        else:
            # 没有约束条件时，返回 _qdots 和 forcing 合并而成的向量
            return self._qdots.col_join(self.forcing)
    def linearize(self, q_ind=None, qd_ind=None, q_dep=None, qd_dep=None,
                  linear_solver='LU', **kwargs):
        """Linearize the equations of motion about a symbolic operating point.

        Parameters
        ==========
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

        For more documentation, please see the ``Linearizer`` class."""
        
        # Convert the current instance into a linearizer object with specified parameters
        linearizer = self.to_linearizer(q_ind, qd_ind, q_dep, qd_dep,
                                        linear_solver=linear_solver)
        # Perform linearization using the linearizer object, passing any additional keyword arguments
        result = linearizer.linearize(**kwargs)
        # Return the linearized results along with the canonical dynamicsymbols 'r'
        return result + (linearizer.r,)
    def solve_multipliers(self, op_point=None, sol_type='dict'):
        """解析求解拉格朗日乘子在指定操作点的符号值。

        Parameters
        ==========

        op_point : dict or iterable of dicts, optional
            求解的操作点。操作点可以是一个字典或者字典的可迭代集合，格式为{symbol: value}。
            value可以是数值或者符号表达式。

        sol_type : str, optional
            求解结果的返回类型。有效选项包括：
            - 'dict': 返回一个字典 {symbol: value}（默认）
            - 'Matrix': 返回解的有序列向量
        """

        # 确定拉格朗日乘子的数量
        k = len(self.lam_vec)
        if k == 0:
            raise ValueError("System has no lagrange multipliers to solve for.")
        
        # 组成操作条件的字典
        if isinstance(op_point, dict):
            op_point_dict = op_point
        elif iterable(op_point):
            op_point_dict = {}
            for op in op_point:
                op_point_dict.update(op)
        elif op_point is None:
            op_point_dict = {}
        else:
            raise TypeError("op_point must be either a dictionary or an "
                            "iterable of dictionaries.")
        
        # 构建要求解的系统
        mass_matrix = self.mass_matrix.col_join(-self.lam_coeffs.row_join(
                zeros(k, k)))
        force_matrix = self.forcing.col_join(self._f_cd)
        
        # 使用操作点进行替换
        mass_matrix = msubs(mass_matrix, op_point_dict)
        force_matrix = msubs(force_matrix, op_point_dict)
        
        # 求解乘子
        sol_list = mass_matrix.LUsolve(-force_matrix)[-k:]
        if sol_type == 'dict':
            return dict(zip(self.lam_vec, sol_list))
        elif sol_type == 'Matrix':
            return Matrix(sol_list)
        else:
            raise ValueError("Unknown sol_type {:}.".format(sol_type))

    def rhs(self, inv_method=None, **kwargs):
        """返回可以进行数值求解的方程组。

        Parameters
        ==========

        inv_method : str
            指定使用的 sympy 矩阵求逆方法。查看
            :meth:`~sympy.matrices.matrixbase.MatrixBase.inv` 获取有效方法列表。
        """

        if inv_method is None:
            self._rhs = self.mass_matrix_full.LUsolve(self.forcing_full)
        else:
            self._rhs = (self.mass_matrix_full.inv(inv_method,
                         try_block_diag=True) * self.forcing_full)
        return self._rhs

    @property
    def q(self):
        return self._q

    @property
    def u(self):
        return self._qdots

    @property
    def bodies(self):
        return self._bodies

    @property
    def forcelist(self):
        return self._forcelist

    @property
    def loads(self):
        return self._forcelist
```