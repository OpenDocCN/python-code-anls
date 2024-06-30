# `D:\src\scipysrc\sympy\sympy\physics\vector\functions.py`

```
# 导入 functools 库中的 reduce 函数
from functools import reduce

# 从 sympy 库中导入多个函数和类
from sympy import (sympify, diff, sin, cos, Matrix, symbols,
                                Function, S, Symbol, linear_eq_to_matrix)

# 从 sympy.integrals.integrals 模块导入 integrate 函数
from sympy.integrals.integrals import integrate

# 从 sympy.simplify.trigsimp 模块导入 trigsimp 函数
from sympy.simplify.trigsimp import trigsimp

# 从当前包的 vector 模块中导入 Vector 类和 _check_vector 函数
from .vector import Vector, _check_vector

# 从当前包的 frame 模块中导入 CoordinateSym 类和 _check_frame 函数
from .frame import CoordinateSym, _check_frame

# 从当前包的 dyadic 模块中导入 Dyadic 类
from .dyadic import Dyadic

# 从当前包的 printing 模块中导入 vprint, vsprint, vpprint, vlatex, init_vprinting 函数
from .printing import vprint, vsprint, vpprint, vlatex, init_vprinting

# 从 sympy.utilities.iterables 模块导入 iterable 函数
from sympy.utilities.iterables import iterable

# 从 sympy.utilities.misc 模块导入 translate 函数
from sympy.utilities.misc import translate

# 将以下函数和类添加到模块的 __all__ 列表中，使其在使用 `from module import *` 时可见
__all__ = ['cross', 'dot', 'express', 'time_derivative', 'outer',
           'kinematic_equations', 'get_motion_params', 'partial_velocity',
           'dynamicsymbols', 'vprint', 'vsprint', 'vpprint', 'vlatex',
           'init_vprinting']


def cross(vec1, vec2):
    """Cross product convenience wrapper for Vector.cross(): \n"""
    # 检查 vec1 和 vec2 是否为 Vector 或 Dyadic 类型，否则引发 TypeError
    if not isinstance(vec1, (Vector, Dyadic)):
        raise TypeError('Cross product is between two vectors')
    return vec1 ^ vec2


# 将 Vector.cross 方法的文档字符串追加到 cross 函数的文档字符串末尾
cross.__doc__ += Vector.cross.__doc__  # type: ignore


def dot(vec1, vec2):
    """Dot product convenience wrapper for Vector.dot(): \n"""
    # 检查 vec1 和 vec2 是否为 Vector 或 Dyadic 类型，否则引发 TypeError
    if not isinstance(vec1, (Vector, Dyadic)):
        raise TypeError('Dot product is between two vectors')
    return vec1 & vec2


# 将 Vector.dot 方法的文档字符串追加到 dot 函数的文档字符串末尾
dot.__doc__ += Vector.dot.__doc__  # type: ignore


def express(expr, frame, frame2=None, variables=False):
    """
    Global function for 'express' functionality.

    Re-expresses a Vector, scalar(sympyfiable) or Dyadic in given frame.

    Refer to the local methods of Vector and Dyadic for details.
    If 'variables' is True, then the coordinate variables (CoordinateSym
    instances) of other frames present in the vector/scalar field or
    dyadic expression are also substituted in terms of the base scalars of
    this frame.

    Parameters
    ==========

    expr : Vector/Dyadic/scalar(sympyfiable)
        The expression to re-express in ReferenceFrame 'frame'

    frame: ReferenceFrame
        The reference frame to express expr in

    frame2 : ReferenceFrame
        The other frame required for re-expression(only for Dyadic expr)

    variables : boolean
        Specifies whether to substitute the coordinate variables present
        in expr, in terms of those of frame

    Examples
    ========

    >>> from sympy.physics.vector import ReferenceFrame, outer, dynamicsymbols
    >>> from sympy.physics.vector import init_vprinting
    >>> init_vprinting(pretty_print=False)
    >>> N = ReferenceFrame('N')
    >>> q = dynamicsymbols('q')
    >>> B = N.orientnew('B', 'Axis', [q, N.z])
    >>> d = outer(N.x, N.x)
    >>> from sympy.physics.vector import express
    >>> express(d, B, N)
    cos(q)*(B.x|N.x) - sin(q)*(B.y|N.x)
    >>> express(B.x, N)
    cos(q)*N.x + sin(q)*N.y
    >>> express(N[0], B, variables=True)
    B_x*cos(q) - B_y*sin(q)

    """

    # 检查 frame 是否为 ReferenceFrame 类型
    _check_frame(frame)

    # 如果 expr 为零，直接返回零
    if expr == 0:
        return expr
    # 如果表达式是 Vector 类型
    if isinstance(expr, Vector):
        # 给定的表达式是一个向量

        # 如果 variables 参数为 True，则在向量中替换坐标变量
        if variables:
            # 获取表达式中每个分量的坐标系列表
            frame_list = [x[-1] for x in expr.args]
            subs_dict = {}

            # 遍历每个坐标系，更新变量映射到当前坐标系 frame
            for f in frame_list:
                subs_dict.update(f.variable_map(frame))

            # 使用变量映射对表达式进行符号替换
            expr = expr.subs(subs_dict)

        # 在当前坐标系 frame 中重新表达向量
        outvec = Vector([])

        # 遍历表达式的每个分量
        for v in expr.args:
            # 如果分量不在当前坐标系 frame 中
            if v[1] != frame:
                # 计算该分量在当前坐标系 frame 中的表示
                temp = frame.dcm(v[1]) * v[0]

                # 如果启用简化选项 Vector.simp，则对结果进行三角函数简化
                if Vector.simp:
                    temp = temp.applyfunc(lambda x: trigsimp(x, method='fu'))

                # 将计算得到的向量分量添加到输出向量中
                outvec += Vector([(temp, frame)])
            else:
                # 如果分量已经在当前坐标系 frame 中，则直接添加到输出向量中
                outvec += Vector([v])

        # 返回重新表达后的向量 outvec
        return outvec

    # 如果表达式是 Dyadic 类型
    if isinstance(expr, Dyadic):
        # 给定的表达式是一个 Dyadic 类型的对象

        # 如果 frame2 未指定，则默认与 frame 相同
        if frame2 is None:
            frame2 = frame
        
        # 检查 frame2 是否有效
        _check_frame(frame2)
        
        # 初始化输出 Dyadic 对象 ol
        ol = Dyadic(0)
        
        # 遍历 Dyadic 对象的每个分量
        for v in expr.args:
            # 计算分量的表达式在当前坐标系 frame 下的表达式，并乘以另外两个分量的表达式
            ol += express(v[0], frame, variables=variables) * \
                  (express(v[1], frame, variables=variables) |
                   express(v[2], frame2, variables=variables))
        
        # 返回计算得到的 Dyadic 对象 ol
        return ol

    else:
        # 如果 variables 参数为 True，则假设表达式是一个标量场
        if variables:
            # 初始化一个集合来存储所有涉及的坐标系
            frame_set = set()

            # 将表达式转换为 sympy 的表达式对象
            expr = sympify(expr)

            # 替换所有自由符号中的坐标变量
            for x in expr.free_symbols:
                if isinstance(x, CoordinateSym) and x.frame != frame:
                    frame_set.add(x.frame)

            subs_dict = {}

            # 遍历涉及的每个坐标系，更新变量映射到当前坐标系 frame
            for f in frame_set:
                subs_dict.update(f.variable_map(frame))

            # 使用变量映射对表达式进行符号替换
            return expr.subs(subs_dict)

        # 如果 variables 参数为 False，则直接返回原始表达式
        return expr
# 定义函数来计算给定参考系中向量/标量场函数或偶数表达式的时间导数
def time_derivative(expr, frame, order=1):
    """
    Calculate the time derivative of a vector/scalar field function
    or dyadic expression in given frame.

    References
    ==========

    https://en.wikipedia.org/wiki/Rotating_reference_frame#Time_derivatives_in_the_two_frames

    Parameters
    ==========

    expr : Vector/Dyadic/sympifyable
        The expression whose time derivative is to be calculated

    frame : ReferenceFrame
        The reference frame to calculate the time derivative in

    order : integer
        The order of the derivative to be calculated

    Examples
    ========

    >>> from sympy.physics.vector import ReferenceFrame, dynamicsymbols
    >>> from sympy.physics.vector import init_vprinting
    >>> init_vprinting(pretty_print=False)
    >>> from sympy import Symbol
    >>> q1 = Symbol('q1')
    >>> u1 = dynamicsymbols('u1')
    >>> N = ReferenceFrame('N')
    >>> A = N.orientnew('A', 'Axis', [q1, N.x])
    >>> v = u1 * N.x
    >>> A.set_ang_vel(N, 10*A.x)
    >>> from sympy.physics.vector import time_derivative
    >>> time_derivative(v, N)
    u1'*N.x
    >>> time_derivative(u1*A[0], N)
    N_x*u1'
    >>> B = N.orientnew('B', 'Axis', [u1, N.z])
    >>> from sympy.physics.vector import outer
    >>> d = outer(N.x, N.x)
    >>> time_derivative(d, B)
    - u1'*(N.y|N.x) - u1'*(N.x|N.y)

    """

    t = dynamicsymbols._t  # 获取时间变量
    _check_frame(frame)  # 检查参考系的有效性

    # 如果导数阶数为零，直接返回表达式本身
    if order == 0:
        return expr

    # 如果导数阶数不是整数或小于零，抛出错误
    if order % 1 != 0 or order < 0:
        raise ValueError("Unsupported value of order entered")

    # 如果表达式是向量，则计算其各成分的时间导数
    if isinstance(expr, Vector):
        outlist = []
        for v in expr.args:
            if v[1] == frame:
                outlist += [(express(v[0], frame, variables=True).diff(t),
                             frame)]
            else:
                outlist += (time_derivative(Vector([v]), v[1]) +
                            (v[1].ang_vel_in(frame) ^ Vector([v]))).args
        outvec = Vector(outlist)
        return time_derivative(outvec, frame, order - 1)

    # 如果表达式是偶数，则计算其各项的时间导数
    if isinstance(expr, Dyadic):
        ol = Dyadic(0)
        for v in expr.args:
            ol += (v[0].diff(t) * (v[1] | v[2]))
            ol += (v[0] * (time_derivative(v[1], frame) | v[2]))
            ol += (v[0] * (v[1] | time_derivative(v[2], frame)))
        return time_derivative(ol, frame, order - 1)

    # 如果是其他类型的表达式，则直接计算其时间导数
    else:
        return diff(express(expr, frame, variables=True), t, order)


# 定义一个外积的便捷包装函数，调用 Vector.outer() 方法
def outer(vec1, vec2):
    """Outer product convenience wrapper for Vector.outer():\n"""
    if not isinstance(vec1, Vector):
        raise TypeError('Outer product is between two Vectors')
    return vec1.outer(vec2)


outer.__doc__ += Vector.outer.__doc__  # 将 Vector.outer() 的文档字符串追加到该函数的文档字符串中

# 给定速度、坐标、旋转类型和旋转顺序（可选），返回与旋转类型相关的 q 点的方程组
def kinematic_equations(speeds, coords, rot_type, rot_order=''):
    """Gives equations relating the qdot's to u's for a rotation type.

    Supply rotation type and order as in orient. Speeds are assumed to be
    body-fixed; if we are defining the orientation of B in A using by rot_type,

    """
    """
        the angular velocity of B in A is assumed to be in the form: speed[0]*B.x +
        speed[1]*B.y + speed[2]*B.z
    
        Parameters
        ==========
    
        speeds : list of length 3
            The body fixed angular velocity measure numbers.
        coords : list of length 3 or 4
            The coordinates used to define the orientation of the two frames.
        rot_type : str
            The type of rotation used to create the equations. Body, Space, or
            Quaternion only
        rot_order : str or int
            If applicable, the order of a series of rotations.
    
        Examples
        ========
    
        >>> from sympy.physics.vector import dynamicsymbols
        >>> from sympy.physics.vector import kinematic_equations, vprint
        >>> u1, u2, u3 = dynamicsymbols('u1 u2 u3')
        >>> q1, q2, q3 = dynamicsymbols('q1 q2 q3')
        >>> vprint(kinematic_equations([u1,u2,u3], [q1,q2,q3], 'body', '313'),
        ...     order=None)
        [-(u1*sin(q3) + u2*cos(q3))/sin(q2) + q1', -u1*cos(q3) + u2*sin(q3) + q2', (u1*sin(q3) + u2*cos(q3))*cos(q2)/sin(q2) - u3 + q3']
    
        """
    
        # Code below is checking and sanitizing input
    
        # 允许的旋转顺序，用于验证和转换输入
        approved_orders = ('123', '231', '312', '132', '213', '321', '121', '131',
                           '212', '232', '313', '323', '1', '2', '3', '')
        
        # 将 rot_order 转换为对应的数字序列，确保标准化为 '123'
        rot_order = translate(str(rot_order), 'XYZxyz', '123123')
        
        # 将 rot_type 转换为小写，确保统一格式
        rot_type = rot_type.lower()
    
        # 检查速度输入是否为列表或元组，否则抛出类型错误
        if not isinstance(speeds, (list, tuple)):
            raise TypeError('Need to supply speeds in a list')
        
        # 检查速度列表长度是否为3，否则抛出类型错误
        if len(speeds) != 3:
            raise TypeError('Need to supply 3 body-fixed speeds')
        
        # 检查坐标输入是否为列表或元组，否则抛出类型错误
        if not isinstance(coords, (list, tuple)):
            raise TypeError('Need to supply coordinates in a list')
        
        # 如果是四元数旋转类型，进一步检查条件
        elif rot_type == 'quaternion':
            # 如果 rot_order 不为空字符串，抛出数值错误
            if rot_order != '':
                raise ValueError('Cannot have rotation order for quaternion')
            
            # 如果坐标列表长度不为4，抛出数值错误
            if len(coords) != 4:
                raise ValueError('Need 4 coordinates for quaternion')
            
            # 实际硬编码的运动学微分方程
            e0, e1, e2, e3 = coords
            w = Matrix(speeds + [0])
            E = Matrix([[e0, -e3, e2, e1],
                        [e3, e0, -e1, e2],
                        [-e2, e1, e0, e3],
                        [-e1, -e2, -e3, e0]])
            edots = Matrix([diff(i, dynamicsymbols._t) for i in [e1, e2, e3, e0]])
            return list(edots.T - 0.5 * w.T * E.T)
        
        # 如果不是批准的旋转类型，抛出数值错误
        else:
            raise ValueError('Not an approved rotation type for this function')
# 返回给定帧中时间的三个运动参数 - 加速度、速度和位置，作为时间的矢量函数。
# 如果提供了更高阶的微分函数，则较低阶函数作为边界条件。例如，给定加速度，则速度和位置参数被视为边界条件。
# 边界条件的时间值从timevalue1（用于位置边界条件）和timevalue2（用于速度边界条件）中获取。
# 如果未提供任何边界条件，则默认为零（对于矢量输入，则为零矢量）。如果边界条件也是时间的函数，则通过将时间值替换为dynamicsymbols._t中的时间符号来将其转换为常数。
# 此函数也可用于计算旋转运动参数。详见参数和示例以获取更清晰的说明。

def get_motion_params(frame, **kwargs):
    """
    Returns the three motion parameters - (acceleration, velocity, and
    position) as vectorial functions of time in the given frame.

    If a higher order differential function is provided, the lower order
    functions are used as boundary conditions. For example, given the
    acceleration, the velocity and position parameters are taken as
    boundary conditions.

    The values of time at which the boundary conditions are specified
    are taken from timevalue1(for position boundary condition) and
    timevalue2(for velocity boundary condition).

    If any of the boundary conditions are not provided, they are taken
    to be zero by default (zero vectors, in case of vectorial inputs). If
    the boundary conditions are also functions of time, they are converted
    to constants by substituting the time values in the dynamicsymbols._t
    time Symbol.

    This function can also be used for calculating rotational motion
    parameters. Have a look at the Parameters and Examples for more clarity.

    Parameters
    ==========

    frame : ReferenceFrame
        The frame to express the motion parameters in

    acceleration : Vector
        Acceleration of the object/frame as a function of time

    velocity : Vector
        Velocity as function of time or as boundary condition
        of velocity at time = timevalue1

    position : Vector
        Velocity as function of time or as boundary condition
        of velocity at time = timevalue1

    timevalue1 : sympyfiable
        Value of time for position boundary condition

    timevalue2 : sympyfiable
        Value of time for velocity boundary condition

    Examples
    ========

    >>> from sympy.physics.vector import ReferenceFrame, get_motion_params, dynamicsymbols
    >>> from sympy.physics.vector import init_vprinting
    >>> init_vprinting(pretty_print=False)
    >>> from sympy import symbols
    >>> R = ReferenceFrame('R')
    >>> v1, v2, v3 = dynamicsymbols('v1 v2 v3')
    >>> v = v1*R.x + v2*R.y + v3*R.z
    >>> get_motion_params(R, position = v)
    (v1''*R.x + v2''*R.y + v3''*R.z, v1'*R.x + v2'*R.y + v3'*R.z, v1*R.x + v2*R.y + v3*R.z)
    >>> a, b, c = symbols('a b c')
    >>> v = a*R.x + b*R.y + c*R.z
    >>> get_motion_params(R, velocity = v)
    (0, a*R.x + b*R.y + c*R.z, a*t*R.x + b*t*R.y + c*t*R.z)
    >>> parameters = get_motion_params(R, acceleration = v)
    >>> parameters[1]
    a*t*R.x + b*t*R.y + c*t*R.z
    >>>
    def _process_vector_differential(vectdiff, condition, variable, ordinate,
                                     frame):
        """
        Helper function for get_motion methods. Finds derivative of vectdiff
        wrt variable, and its integral using the specified boundary condition
        at value of variable = ordinate.
        Returns a tuple of - (derivative, function and integral) wrt vectdiff

        """

        # 确保边界条件与 'variable' 无关
        if condition != 0:
            condition = express(condition, frame, variables=True)
        # 处理 vectdiff == Vector(0) 的特殊情况
        if vectdiff == Vector(0):
            return (0, 0, condition)
        # 将 vectdiff 完全表示为 condition 所在的坐标系中的 vectdiff1
        vectdiff1 = express(vectdiff, frame)
        # 求 vectdiff 的导数
        vectdiff2 = time_derivative(vectdiff, frame)
        # 积分并应用边界条件
        vectdiff0 = Vector(0)
        lims = (variable, ordinate, variable)
        for dim in frame:
            function1 = vectdiff1.dot(dim)
            abscissa = dim.dot(condition).subs({variable: ordinate})
            # 对 'function1' 关于 'variable' 进行不定积分，使用给定的初始条件 (ordinate, abscissa)
            vectdiff0 += (integrate(function1, lims) + abscissa) * dim
        # 返回元组
        return (vectdiff2, vectdiff, vectdiff0)

    _check_frame(frame)
    # 根据用户输入确定操作模式
    if 'acceleration' in kwargs:
        mode = 2
    elif 'velocity' in kwargs:
        mode = 1
    else:
        mode = 0
    # kwargs 中的所有可能参数
    # 并非每种情况都需要所有参数
    # 如果未指定，则设置为默认值（可能会用于计算，也可能不会）
    conditions = ['acceleration', 'velocity', 'position',
                  'timevalue', 'timevalue1', 'timevalue2']
    for i, x in enumerate(conditions):
        if x not in kwargs:
            if i < 3:
                kwargs[x] = Vector(0)
            else:
                kwargs[x] = S.Zero
        elif i < 3:
            _check_vector(kwargs[x])
        else:
            kwargs[x] = sympify(kwargs[x])
    if mode == 2:
        vel = _process_vector_differential(kwargs['acceleration'],
                                           kwargs['velocity'],
                                           dynamicsymbols._t,
                                           kwargs['timevalue2'], frame)[2]
        pos = _process_vector_differential(vel, kwargs['position'],
                                           dynamicsymbols._t,
                                           kwargs['timevalue1'], frame)[2]
        return (kwargs['acceleration'], vel, pos)
    elif mode == 1:
        # 如果 mode 等于 1，则执行向量微分操作
        return _process_vector_differential(kwargs['velocity'],
                                            kwargs['position'],
                                            dynamicsymbols._t,
                                            kwargs['timevalue1'], frame)
    else:
        # 否则，计算速度和加速度
        vel = time_derivative(kwargs['position'], frame)
        acc = time_derivative(vel, frame)
        # 返回加速度、速度和位置信息的元组
        return (acc, vel, kwargs['position'])
# 返回一个列表，其中包含关于给定参考系中每个提供的速度向量相对于提供的广义速度的偏导数
def partial_velocity(vel_vecs, gen_speeds, frame):
    """
    Returns a list of partial velocities with respect to the provided
    generalized speeds in the given reference frame for each of the supplied
    velocity vectors.

    The output is a list of lists. The outer list has a number of elements
    equal to the number of supplied velocity vectors. The inner lists are, for
    each velocity vector, the partial derivatives of that velocity vector with
    respect to the generalized speeds supplied.

    Parameters
    ==========

    vel_vecs : iterable
        An iterable of velocity vectors (angular or linear).
    gen_speeds : iterable
        An iterable of generalized speeds.
    frame : ReferenceFrame
        The reference frame that the partial derivatives are going to be taken
        in.
    """

    if not iterable(vel_vecs):  # 检查速度向量是否可迭代
        raise TypeError('Velocity vectors must be contained in an iterable.')

    if not iterable(gen_speeds):  # 检查广义速度是否可迭代
        raise TypeError('Generalized speeds must be contained in an iterable')

    vec_partials = []  # 初始化一个空列表，用于存放偏导数向量
    gen_speeds = list(gen_speeds)  # 将广义速度转换为列表形式
    for vel in vel_vecs:  # 对于每个速度向量进行循环
        partials = [Vector(0) for _ in gen_speeds]  # 初始化与广义速度长度相同的零向量列表
        for components, ref in vel.args:  # 对速度向量的组成部分和参考系进行循环
            mat, _ = linear_eq_to_matrix(components, gen_speeds)  # 将速度向量的组成部分转换为线性方程组的形式
            for i in range(len(gen_speeds)):  # 对于每个广义速度
                for dim, direction in enumerate(ref):  # 对速度向量的参考系方向进行枚举
                    if mat[dim, i] != 0:  # 如果线性方程组中的元素不为零
                        partials[i] += direction * mat[dim, i]  # 计算偏导数向量的部分导数

        vec_partials.append(partials)  # 将计算得到的偏导数向量添加到结果列表中

    return vec_partials  # 返回所有速度向量相对于广义速度的偏导数列表
    Examples
    ========

    >>> from sympy.physics.vector import dynamicsymbols  # 导入动力符号相关的模块
    >>> from sympy import diff, Symbol  # 导入求导和符号操作相关的模块
    >>> q1 = dynamicsymbols('q1')  # 创建动力符号 q1(t)
    >>> q1
    q1(t)  # 打印出 q1(t)
    >>> q2 = dynamicsymbols('q2', real=True)  # 创建实数类型的动力符号 q2(t)
    >>> q2.is_real
    True  # 检查 q2 是否为实数类型
    >>> q3 = dynamicsymbols('q3', positive=True)  # 创建正数类型的动力符号 q3(t)
    >>> q3.is_positive
    True  # 检查 q3 是否为正数类型
    >>> q4, q5 = dynamicsymbols('q4,q5', commutative=False)  # 创建非交换的动力符号 q4(t), q5(t)
    >>> bool(q4*q5 != q5*q4)
    True  # 检查 q4 和 q5 是否不可交换
    >>> q6 = dynamicsymbols('q6', integer=True)  # 创建整数类型的动力符号 q6(t)
    >>> q6.is_integer
    True  # 检查 q6 是否为整数类型
    >>> diff(q1, Symbol('t'))
    Derivative(q1(t), t)  # 对 q1(t) 求关于时间 t 的导数

    """
    # 根据给定的名称和假设创建动力学符号
    esses = symbols(names, cls=Function, **assumptions)
    # 获取动力学符号的时间变量
    t = dynamicsymbols._t
    # 如果 esses 是可迭代的，则对每个符号应用 level 次的时间导数，最终返回列表
    if iterable(esses):
        esses = [reduce(diff, [t] * level, e(t)) for e in esses]
        return esses
    else:
        # 否则，对单个符号应用 level 次的时间导数，并返回结果
        return reduce(diff, [t] * level, esses(t))
# 将 Symbol('t') 赋值给 dynamicsymbols._t，指定类型为 't'，忽略类型检查
dynamicsymbols._t = Symbol('t')  # type: ignore

# 将字符串 '\'' 赋值给 dynamicsymbols._str，表示单引号字符 '\'', 忽略类型检查
dynamicsymbols._str = '\''  # type: ignore
```