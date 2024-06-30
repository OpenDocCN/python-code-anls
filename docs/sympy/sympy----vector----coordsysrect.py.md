# `D:\src\scipysrc\sympy\sympy\vector\coordsysrect.py`

```
    from collections.abc import Callable  # 导入标准库中的 Callable 类

    from sympy.core.basic import Basic  # 导入 SymPy 核心基础模块中的 Basic 类
    from sympy.core.cache import cacheit  # 导入 SymPy 核心缓存模块中的 cacheit 函数
    from sympy.core import S, Dummy, Lambda  # 导入 SymPy 核心模块中的 S、Dummy、Lambda 符号
    from sympy.core.symbol import Str  # 导入 SymPy 核心符号模块中的 Str 类
    from sympy.core.symbol import symbols  # 导入 SymPy 核心符号模块中的 symbols 函数
    from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix  # 导入 SymPy 不可变密集矩阵模块中的 ImmutableDenseMatrix 类，并重命名为 Matrix
    from sympy.matrices.matrixbase import MatrixBase  # 导入 SymPy 矩阵基类模块中的 MatrixBase 类
    from sympy.solvers import solve  # 导入 SymPy 求解器模块中的 solve 函数
    from sympy.vector.scalar import BaseScalar  # 导入 SymPy 向量标量模块中的 BaseScalar 类
    from sympy.core.containers import Tuple  # 导入 SymPy 核心容器模块中的 Tuple 类
    from sympy.core.function import diff  # 导入 SymPy 核心函数模块中的 diff 函数
    from sympy.functions.elementary.miscellaneous import sqrt  # 导入 SymPy 基础元函数模块中的 sqrt 函数
    from sympy.functions.elementary.trigonometric import (acos, atan2, cos, sin)  # 导入 SymPy 基础三角函数模块中的 acos、atan2、cos、sin 函数
    from sympy.matrices.dense import eye  # 导入 SymPy 密集矩阵模块中的 eye 函数
    from sympy.matrices.immutable import ImmutableDenseMatrix  # 导入 SymPy 不可变密集矩阵模块中的 ImmutableDenseMatrix 类
    from sympy.simplify.simplify import simplify  # 导入 SymPy 简化模块中的 simplify 函数
    from sympy.simplify.trigsimp import trigsimp  # 导入 SymPy 三角简化模块中的 trigsimp 函数
    import sympy.vector  # 导入 SymPy 向量模块
    from sympy.vector.orienters import (Orienter, AxisOrienter, BodyOrienter,  # 导入 SymPy 向量定向模块中的不同定向器类
                                        SpaceOrienter, QuaternionOrienter)


    class CoordSys3D(Basic):
        """
        Represents a coordinate system in 3-D space.
        """

        def _sympystr(self, printer):
            return self._name  # 返回坐标系名称的字符串表示

        def __iter__(self):
            return iter(self.base_vectors())  # 返回基向量的迭代器

        @staticmethod
        def _check_orthogonality(equations):
            """
            Helper method for _connect_to_cartesian. It checks if
            set of transformation equations create orthogonal curvilinear
            coordinate system

            Parameters
            ==========

            equations : Lambda
                Lambda of transformation equations

            """

            x1, x2, x3 = symbols("x1, x2, x3", cls=Dummy)  # 创建三个虚拟符号 x1, x2, x3
            equations = equations(x1, x2, x3)  # 对方程组应用虚拟符号
            v1 = Matrix([diff(equations[0], x1),
                         diff(equations[1], x1), diff(equations[2], x1)])  # 计算第一个向量的偏导数

            v2 = Matrix([diff(equations[0], x2),
                         diff(equations[1], x2), diff(equations[2], x2)])  # 计算第二个向量的偏导数

            v3 = Matrix([diff(equations[0], x3),
                         diff(equations[1], x3), diff(equations[2], x3)])  # 计算第三个向量的偏导数

            if any(simplify(i[0] + i[1] + i[2]) == 0 for i in (v1, v2, v3)):
                return False  # 如果任意一个向量的分量之和为零，则不是正交的
            else:
                if simplify(v1.dot(v2)) == 0 and simplify(v2.dot(v3)) == 0 \
                    and simplify(v3.dot(v1)) == 0:
                    return True  # 如果所有向量两两点积为零，则是正交的
                else:
                    return False  # 否则不是正交的
    @staticmethod
    def _get_lame_coeff(curv_coord_name):
        """
        Store information about Lame coefficients for pre-defined
        coordinate systems.

        Parameters
        ==========

        curv_coord_name : str
            Name of coordinate system

        """
        # 如果 curv_coord_name 是字符串类型
        if isinstance(curv_coord_name, str):
            # 如果是笛卡尔坐标系，返回常数系数 (1, 1, 1)
            if curv_coord_name == 'cartesian':
                return lambda x, y, z: (S.One, S.One, S.One)
            # 如果是球坐标系，返回 Lame 系数 (1, r, r*sin(theta))
            if curv_coord_name == 'spherical':
                return lambda r, theta, phi: (S.One, r, r*sin(theta))
            # 如果是柱坐标系，返回 Lame 系数 (1, r, 1)
            if curv_coord_name == 'cylindrical':
                return lambda r, theta, h: (S.One, r, S.One)
            # 抛出数值错误，说明坐标系类型未定义
            raise ValueError('Wrong set of parameters.'
                             ' Type of coordinate system is not defined')
        # 如果 curv_coord_name 不是字符串类型，调用 _calculate_lame_coefficients 方法
        return CoordSys3D._calculate_lame_coefficients(curv_coord_name)
    # 定义一个函数，用于计算给定变换方程的拉姆系数
    def _calculate_lame_coeff(equations):
        """
        It calculates Lame coefficients
        for given transformations equations.

        Parameters
        ==========

        equations : Lambda
            Lambda of transformation equations.

        """
        # 返回一个 lambda 函数，该函数接受三个参数 x1, x2, x3，并计算拉姆系数
        return lambda x1, x2, x3: (
                          sqrt(diff(equations(x1, x2, x3)[0], x1)**2 +
                               diff(equations(x1, x2, x3)[1], x1)**2 +
                               diff(equations(x1, x2, x3)[2], x1)**2),
                          sqrt(diff(equations(x1, x2, x3)[0], x2)**2 +
                               diff(equations(x1, x2, x3)[1], x2)**2 +
                               diff(equations(x1, x2, x3)[2], x2)**2),
                          sqrt(diff(equations(x1, x2, x3)[0], x3)**2 +
                               diff(equations(x1, x2, x3)[1], x3)**2 +
                               diff(equations(x1, x2, x3)[2], x3)**2)
                      )

    # 返回逆旋转矩阵的简化结果
    def _inverse_rotation_matrix(self):
        """
        Returns inverse rotation matrix.
        """
        return simplify(self._parent_rotation_matrix**-1)

    # 静态方法，根据给定的曲线坐标系名称返回相应的变换方程 lambda 函数
    @staticmethod
    def _get_transformation_lambdas(curv_coord_name):
        """
        Store information about transformation equations for pre-defined
        coordinate systems.

        Parameters
        ==========

        curv_coord_name : str
            Name of coordinate system

        """
        # 检查参数 curv_coord_name 是否为字符串类型
        if isinstance(curv_coord_name, str):
            # 如果 curv_coord_name 为 'cartesian'，返回直角坐标系的恒等变换
            if curv_coord_name == 'cartesian':
                return lambda x, y, z: (x, y, z)
            # 如果 curv_coord_name 为 'spherical'，返回球坐标系的变换方程 lambda 函数
            if curv_coord_name == 'spherical':
                return lambda r, theta, phi: (
                    r*sin(theta)*cos(phi),
                    r*sin(theta)*sin(phi),
                    r*cos(theta)
                )
            # 如果 curv_coord_name 为 'cylindrical'，返回柱坐标系的变换方程 lambda 函数
            if curv_coord_name == 'cylindrical':
                return lambda r, theta, h: (
                    r*cos(theta),
                    r*sin(theta),
                    h
                )
            # 如果 curv_coord_name 不匹配任何预定义的坐标系名称，抛出 ValueError 异常
            raise ValueError('Wrong set of parameters.'
                             'Type of coordinate system is defined')

    # 类方法，根据给定的旋转矩阵和变换方程计算得到变换方程组
    @classmethod
    def _rotation_trans_equations(cls, matrix, equations):
        """
        Returns the transformation equations obtained from rotation matrix.

        Parameters
        ==========

        matrix : Matrix
            Rotation matrix

        equations : tuple
            Transformation equations

        """
        # 返回一个元组，包含由旋转矩阵与变换方程组相乘得到的结果
        return tuple(matrix * Matrix(equations))

    # 返回属性 _origin 的值
    @property
    def origin(self):
        return self._origin

    # 返回方法 base_vectors 的结果
    def base_vectors(self):
        return self._base_vectors

    # 返回方法 base_scalars 的结果
    def base_scalars(self):
        return self._base_scalars

    # 返回方法 lame_coefficients 的结果
    def lame_coefficients(self):
        return self._lame_coefficients

    # 返回方法 transformation_to_parent 的结果，传递 base_scalars 方法的结果作为参数
    def transformation_to_parent(self):
        return self._transformation_lambda(*self.base_scalars())
    # 如果没有父坐标系，则抛出值错误异常，提示需要使用 transformaton_from_parent_function() 函数
    def transformation_from_parent(self):
        if self._parent is None:
            raise ValueError("no parent coordinate system, use "
                             "`transformation_from_parent_function()`")
        # 调用 _transformation_from_parent_lambda 函数，使用父坐标系的基本标量参数进行变换
        return self._transformation_from_parent_lambda(
                            *self._parent.base_scalars())

    # 返回 _transformation_from_parent_lambda 属性，该属性是一个函数对象，用于执行从父坐标系到当前坐标系的转换
    def transformation_from_parent_function(self):
        return self._transformation_from_parent_lambda

    # 返回与另一个坐标系 'other' 相关的方向余弦矩阵（DCM），也称为旋转矩阵
    def rotation_matrix(self, other):
        """
        Returns the direction cosine matrix(DCM), also known as the
        'rotation matrix' of this coordinate system with respect to
        another system.

        If v_a is a vector defined in system 'A' (in matrix format)
        and v_b is the same vector defined in system 'B', then
        v_a = A.rotation_matrix(B) * v_b.

        A SymPy Matrix is returned.

        Parameters
        ==========

        other : CoordSys3D
            The system which the DCM is generated to.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> from sympy import symbols
        >>> q1 = symbols('q1')
        >>> N = CoordSys3D('N')
        >>> A = N.orient_new_axis('A', q1, N.i)
        >>> N.rotation_matrix(A)
        Matrix([
        [1,       0,        0],
        [0, cos(q1), -sin(q1)],
        [0, sin(q1),  cos(q1)]])

        """
        from sympy.vector.functions import _path
        # 如果 'other' 不是 CoordSys3D 类型，则抛出类型错误异常
        if not isinstance(other, CoordSys3D):
            raise TypeError(str(other) +
                            " is not a CoordSys3D")
        
        # 处理特殊情况
        if other == self:
            return eye(3)  # 返回单位矩阵
        elif other == self._parent:
            return self._parent_rotation_matrix  # 返回父坐标系的旋转矩阵
        elif other._parent == self:
            return other._parent_rotation_matrix.T  # 返回其他坐标系的父坐标系的旋转矩阵的转置
        
        # 否则，使用树结构计算位置
        rootindex, path = _path(self, other)
        result = eye(3)
        # 对于根到目标坐标系的路径中的每个坐标系，依次乘以其父坐标系的旋转矩阵
        for i in range(rootindex):
            result *= path[i]._parent_rotation_matrix
        for i in range(rootindex + 1, len(path)):
            result *= path[i]._parent_rotation_matrix.T
        return result

    # 使用缓存装饰器缓存结果，返回此坐标系原点相对于另一个点或坐标系 'other' 的位置向量
    def position_wrt(self, other):
        """
        Returns the position vector of the origin of this coordinate
        system with respect to another Point/CoordSys3D.

        Parameters
        ==========

        other : Point/CoordSys3D
            If other is a Point, the position of this system's origin
            wrt it is returned. If its an instance of CoordSyRect,
            the position wrt its origin is returned.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> N = CoordSys3D('N')
        >>> N1 = N.locate_new('N1', 10 * N.i)
        >>> N.position_wrt(N1)
        (-10)*N.i

        """
        return self.origin.position_wrt(other)
    def scalar_map(self, other):
        """
        Returns a dictionary which expresses the coordinate variables
        (base scalars) of this frame in terms of the variables of
        another coordinate system (other).

        Parameters
        ==========

        other : CoordSys3D
            The other coordinate system to map the variables to.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> from sympy import Symbol
        >>> A = CoordSys3D('A')
        >>> q = Symbol('q')
        >>> B = A.orient_new_axis('B', q, A.k)
        >>> A.scalar_map(B)
        {A.x: B.x*cos(q) - B.y*sin(q), A.y: B.x*sin(q) + B.y*cos(q), A.z: B.z}

        """

        # Calculate the coordinates of this frame relative to 'other'
        origin_coords = tuple(self.position_wrt(other).to_matrix(other))
        
        # Calculate the differences in base scalars between this frame and 'other'
        relocated_scalars = [x - origin_coords[i]
                             for i, x in enumerate(other.base_scalars())]

        # Compute the transformation matrix and simplify using trigonometric identities
        vars_matrix = (self.rotation_matrix(other) *
                       Matrix(relocated_scalars))
        
        # Return a dictionary mapping base scalars of this frame to their expressions in terms of 'other'
        return {x: trigsimp(vars_matrix[i])
                for i, x in enumerate(self.base_scalars())}

    def locate_new(self, name, position, vector_names=None,
                   variable_names=None):
        """
        Returns a CoordSys3D with its origin located at the given
        position with respect to this coordinate system's origin.

        Parameters
        ==========

        name : str
            The name of the new CoordSys3D instance.

        position : Vector
            The position vector of the new system's origin with respect
            to this coordinate system's origin.

        vector_names, variable_names : iterable (optional)
            Iterables of 3 strings each, providing custom names for base
            vectors and base scalars of the new system respectively,
            used for simple string representation.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> A = CoordSys3D('A')
        >>> B = A.locate_new('B', 10 * A.i)
        >>> B.origin.position_wrt(A.origin)
        10*A.i

        """
        
        # Set default variable names if not provided
        if variable_names is None:
            variable_names = self._variable_names
        
        # Set default vector names if not provided
        if vector_names is None:
            vector_names = self._vector_names
        
        # Return a new CoordSys3D instance with specified attributes
        return CoordSys3D(name, location=position,
                          vector_names=vector_names,
                          variable_names=variable_names,
                          parent=self)
    def orient_new_axis(self, name, angle, axis, location=None,
                        vector_names=None, variable_names=None):
        """
        Axis rotation is a rotation about an arbitrary axis by
        some angle. The angle is supplied as a SymPy expr scalar, and
        the axis is supplied as a Vector.

        Parameters
        ==========

        name : string
            The name of the new coordinate system

        angle : Expr
            The angle by which the new system is to be rotated

        axis : Vector
            The axis around which the rotation has to be performed

        location : Vector(optional)
            The location of the new coordinate system's origin wrt this
            system's origin. If not specified, the origins are taken to
            be coincident.

        vector_names, variable_names : iterable(optional)
            Iterables of 3 strings each, with custom names for base
            vectors and base scalars of the new system respectively.
            Used for simple str printing.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> from sympy import symbols
        >>> q1 = symbols('q1')
        >>> N = CoordSys3D('N')
        >>> B = N.orient_new_axis('B', q1, N.i + 2 * N.j)

        """
        # 如果没有提供 variable_names，则使用 self._variable_names
        if variable_names is None:
            variable_names = self._variable_names
        # 如果没有提供 vector_names，则使用 self._vector_names
        if vector_names is None:
            vector_names = self._vector_names

        # 创建 AxisOrienter 对象，用于处理轴向旋转
        orienter = AxisOrienter(angle, axis)
        # 调用 orient_new 方法进行坐标系的旋转操作，并返回结果
        return self.orient_new(name, orienter,
                               location=location,
                               vector_names=vector_names,
                               variable_names=variable_names)
    def orient_new_body(self, name, angle1, angle2, angle3,
                        rotation_order, location=None,
                        vector_names=None, variable_names=None):
        """
        Body orientation takes this coordinate system through three
        successive simple rotations.

        Body fixed rotations include both Euler Angles and
        Tait-Bryan Angles, see https://en.wikipedia.org/wiki/Euler_angles.

        Parameters
        ==========

        name : string
            The name of the new coordinate system

        angle1, angle2, angle3 : Expr
            Three successive angles to rotate the coordinate system by

        rotation_order : string
            String defining the order of axes for rotation

        location : Vector(optional)
            The location of the new coordinate system's origin wrt this
            system's origin. If not specified, the origins are taken to
            be coincident.

        vector_names, variable_names : iterable(optional)
            Iterables of 3 strings each, with custom names for base
            vectors and base scalars of the new system respectively.
            Used for simple str printing.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> from sympy import symbols
        >>> q1, q2, q3 = symbols('q1 q2 q3')
        >>> N = CoordSys3D('N')

        A 'Body' fixed rotation is described by three angles and
        three body-fixed rotation axes. To orient a coordinate system D
        with respect to N, each sequential rotation is always about
        the orthogonal unit vectors fixed to D. For example, a '123'
        rotation will specify rotations about N.i, then D.j, then
        D.k. (Initially, D.i is same as N.i)
        Therefore,

        >>> D = N.orient_new_body('D', q1, q2, q3, '123')

        is same as

        >>> D = N.orient_new_axis('D', q1, N.i)
        >>> D = D.orient_new_axis('D', q2, D.j)
        >>> D = D.orient_new_axis('D', q3, D.k)

        Acceptable rotation orders are of length 3, expressed in XYZ or
        123, and cannot have a rotation about about an axis twice in a row.

        >>> B = N.orient_new_body('B', q1, q2, q3, '123')
        >>> B = N.orient_new_body('B', q1, q2, 0, 'ZXZ')
        >>> B = N.orient_new_body('B', 0, 0, 0, 'XYX')

        """

        # 创建一个 BodyOrienter 对象，用于处理三维空间中的旋转操作
        orienter = BodyOrienter(angle1, angle2, angle3, rotation_order)
        
        # 调用当前对象的 orient_new 方法，使用上述创建的 orienter 对象进行坐标系的重新定向
        return self.orient_new(name, orienter,
                               location=location,
                               vector_names=vector_names,
                               variable_names=variable_names)
    def orient_new_space(self, name, angle1, angle2, angle3,
                         rotation_order, location=None,
                         vector_names=None, variable_names=None):
        """
        Space rotation is similar to Body rotation, but the rotations
        are applied in the opposite order.

        Parameters
        ==========

        name : string
            The name of the new coordinate system

        angle1, angle2, angle3 : Expr
            Three successive angles to rotate the coordinate system by

        rotation_order : string
            String defining the order of axes for rotation

        location : Vector(optional)
            The location of the new coordinate system's origin wrt this
            system's origin. If not specified, the origins are taken to
            be coincident.

        vector_names, variable_names : iterable(optional)
            Iterables of 3 strings each, with custom names for base
            vectors and base scalars of the new system respectively.
            Used for simple str printing.

        See Also
        ========

        CoordSys3D.orient_new_body : method to orient via Euler
            angles

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> from sympy import symbols
        >>> q1, q2, q3 = symbols('q1 q2 q3')
        >>> N = CoordSys3D('N')

        To orient a coordinate system D with respect to N, each
        sequential rotation is always about N's orthogonal unit vectors.
        For example, a '123' rotation will specify rotations about
        N.i, then N.j, then N.k.
        Therefore,

        >>> D = N.orient_new_space('D', q1, q2, q3, '312')

        is same as

        >>> B = N.orient_new_axis('B', q1, N.i)
        >>> C = B.orient_new_axis('C', q2, N.j)
        >>> D = C.orient_new_axis('D', q3, N.k)

        """

        # 创建 SpaceOrienter 对象，用于执行空间旋转
        orienter = SpaceOrienter(angle1, angle2, angle3, rotation_order)
        
        # 调用 self 的 orient_new 方法，进行新坐标系的定向操作
        return self.orient_new(name, orienter,
                               location=location,
                               vector_names=vector_names,
                               variable_names=variable_names)
    def orient_new_quaternion(self, name, q0, q1, q2, q3, location=None,
                              vector_names=None, variable_names=None):
        """
        Quaternion orientation orients the new CoordSys3D with
        Quaternions, defined as a finite rotation about lambda, a unit
        vector, by some amount theta.

        This orientation is described by four parameters:

        q0 = cos(theta/2)

        q1 = lambda_x sin(theta/2)

        q2 = lambda_y sin(theta/2)

        q3 = lambda_z sin(theta/2)

        Quaternion does not take in a rotation order.

        Parameters
        ==========

        name : string
            The name of the new coordinate system

        q0, q1, q2, q3 : Expr
            The quaternions to rotate the coordinate system by

        location : Vector(optional)
            The location of the new coordinate system's origin wrt this
            system's origin. If not specified, the origins are taken to
            be coincident.

        vector_names, variable_names : iterable(optional)
            Iterables of 3 strings each, with custom names for base
            vectors and base scalars of the new system respectively.
            Used for simple str printing.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> from sympy import symbols
        >>> q0, q1, q2, q3 = symbols('q0 q1 q2 q3')
        >>> N = CoordSys3D('N')
        >>> B = N.orient_new_quaternion('B', q0, q1, q2, q3)

        """

        # 创建一个 QuaternionOrienter 对象，用给定的四元数参数初始化
        orienter = QuaternionOrienter(q0, q1, q2, q3)
        
        # 调用当前对象的 orient_new 方法来使用 QuaternionOrienter 对象来定向新的坐标系
        return self.orient_new(name, orienter,
                               location=location,
                               vector_names=vector_names,
                               variable_names=variable_names)
    def create_new(self, name, transformation, variable_names=None, vector_names=None):
        """
        Returns a CoordSys3D which is connected to self by transformation.

        Parameters
        ==========

        name : str
            The name of the new CoordSys3D instance.

        transformation : Lambda, Tuple, str
            Transformation defined by transformation equations or chosen
            from predefined ones.

        vector_names, variable_names : iterable(optional)
            Iterables of 3 strings each, with custom names for base
            vectors and base scalars of the new system respectively.
            Used for simple str printing.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> a = CoordSys3D('a')
        >>> b = a.create_new('b', transformation='spherical')
        >>> b.transformation_to_parent()
        (b.r*sin(b.theta)*cos(b.phi), b.r*sin(b.phi)*sin(b.theta), b.r*cos(b.theta))
        >>> b.transformation_from_parent()
        (sqrt(a.x**2 + a.y**2 + a.z**2), acos(a.z/sqrt(a.x**2 + a.y**2 + a.z**2)), atan2(a.y, a.x))

        """
        # 创建一个新的 CoordSys3D 实例，连接到当前 CoordSys3D 对象，使用给定的转换
        return CoordSys3D(name, parent=self, transformation=transformation,
                          variable_names=variable_names, vector_names=vector_names)

    def __init__(self, name, location=None, rotation_matrix=None,
                 parent=None, vector_names=None, variable_names=None,
                 latex_vects=None, pretty_vects=None, latex_scalars=None,
                 pretty_scalars=None, transformation=None):
        # Dummy initializer for setting docstring
        pass

    __init__.__doc__ = __new__.__doc__

    @staticmethod
    def _compose_rotation_and_translation(rot, translation, parent):
        # 静态方法：组合旋转和平移操作
        r = lambda x, y, z: CoordSys3D._rotation_trans_equations(rot, (x, y, z))
        if parent is None:
            # 如果没有父坐标系，则返回旋转函数 r
            return r

        dx, dy, dz = [translation.dot(i) for i in parent.base_vectors()]
        # 计算平移量 dx, dy, dz 分别乘以父坐标系的基向量
        t = lambda x, y, z: (
            x + dx,
            y + dy,
            z + dz,
        )
        # 返回一个新的 lambda 函数，先旋转再平移
        return lambda x, y, z: t(*r(x, y, z))
# 检查字符串参数是否符合要求，抛出相应的错误信息
def _check_strings(arg_name, arg):
    # 生成错误信息字符串，指示参数应为包含三个字符串类型元素的可迭代对象
    errorstr = arg_name + " must be an iterable of 3 string-types"
    # 如果参数长度不为3，抛出值错误（ValueError）异常
    if len(arg) != 3:
        raise ValueError(errorstr)
    # 遍历参数中的每个元素
    for s in arg:
        # 如果元素不是字符串类型，抛出类型错误（TypeError）异常
        if not isinstance(s, str):
            raise TypeError(errorstr)


# 延迟导入以避免循环导入问题：
# 从 sympy.vector.vector 模块中导入 BaseVector 类
from sympy.vector.vector import BaseVector
```