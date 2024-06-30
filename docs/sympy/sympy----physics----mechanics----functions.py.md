# `D:\src\scipysrc\sympy\sympy\physics\mechanics\functions.py`

```
# 导入函数 dict_merge 和 iterable 从 sympy.utilities 模块
from sympy.utilities import dict_merge
from sympy.utilities.iterables import iterable
# 导入 Dyadic, Vector, ReferenceFrame, Point, dynamicsymbols 类和相关函数从 sympy.physics.vector 模块
from sympy.physics.vector import (Dyadic, Vector, ReferenceFrame,
                                  Point, dynamicsymbols)
# 导入 vprint, vsprint, vpprint, vlatex, init_vprinting 函数从 sympy.physics.vector.printing 模块
from sympy.physics.vector.printing import (vprint, vsprint, vpprint, vlatex,
                                           init_vprinting)
# 导入 Particle 类从 sympy.physics.mechanics.particle 模块
from sympy.physics.mechanics.particle import Particle
# 导入 RigidBody 类从 sympy.physics.mechanics.rigidbody 模块
from sympy.physics.mechanics.rigidbody import RigidBody
# 导入 simplify 函数从 sympy.simplify.simplify 模块
from sympy.simplify.simplify import simplify
# 导入 Matrix, Mul, Derivative, sin, cos, tan, S 函数从 sympy 模块
from sympy import Matrix, Mul, Derivative, sin, cos, tan, S
# 导入 AppliedUndef 类从 sympy.core.function 模块
from sympy.core.function import AppliedUndef
# 导入 _inertia, _inertia_of_point_mass 函数从 sympy.physics.mechanics.inertia 模块
from sympy.physics.mechanics.inertia import (inertia as _inertia,
    inertia_of_point_mass as _inertia_of_point_mass)
# 导入 sympy_deprecation_warning 函数从 sympy.utilities.exceptions 模块
from sympy.utilities.exceptions import sympy_deprecation_warning

# 定义 __all__ 列表，包含公开的函数名列表
__all__ = ['linear_momentum',
           'angular_momentum',
           'kinetic_energy',
           'potential_energy',
           'Lagrangian',
           'mechanics_printing',
           'mprint',
           'msprint',
           'mpprint',
           'mlatex',
           'msubs',
           'find_dynamicsymbols']

# 将 vprint 函数赋值给 mprint，vsprint 函数赋值给 msprint，vpprint 函数赋值给 mpprint，vlatex 函数赋值给 mlatex
mprint = vprint
msprint = vsprint
mpprint = vpprint
mlatex = vlatex

# 定义 mechanics_printing 函数，用于初始化力学模块中所有 SymPy 对象的时间导数打印设置
def mechanics_printing(**kwargs):
    """
    Initializes time derivative printing for all SymPy objects in
    mechanics module.
    """
    # 调用 init_vprinting 函数，初始化打印设置
    init_vprinting(**kwargs)

# 将 mechanics_printing 函数的文档字符串设为 init_vprinting 函数的文档字符串
mechanics_printing.__doc__ = init_vprinting.__doc__

# 定义 inertia 函数，用于计算惯性张量
def inertia(frame, ixx, iyy, izz, ixy=0, iyz=0, izx=0):
    # 发出 sympy_deprecation_warning 警告，提示 inertia 函数已移动到 sympy.physics.mechanics 模块
    sympy_deprecation_warning(
        """
        The inertia function has been moved.
        Import it from "sympy.physics.mechanics".
        """,
        deprecated_since_version="1.13",
        active_deprecations_target="moved-mechanics-functions"
    )
    # 调用 _inertia 函数计算惯性张量
    return _inertia(frame, ixx, iyy, izz, ixy, iyz, izx)

# 定义 inertia_of_point_mass 函数，用于计算点质量的惯性张量
def inertia_of_point_mass(mass, pos_vec, frame):
    # 发出 sympy_deprecation_warning 警告，提示 inertia_of_point_mass 函数已移动到 sympy.physics.mechanics 模块
    sympy_deprecation_warning(
        """
        The inertia_of_point_mass function has been moved.
        Import it from "sympy.physics.mechanics".
        """,
        deprecated_since_version="1.13",
        active_deprecations_target="moved-mechanics-functions"
    )
    # 调用 _inertia_of_point_mass 函数计算点质量的惯性张量
    return _inertia_of_point_mass(mass, pos_vec, frame)

# 定义 linear_momentum 函数，计算系统的线动量
def linear_momentum(frame, *body):
    """Linear momentum of the system.

    Explanation
    ===========

    This function returns the linear momentum of a system of Particle's and/or
    RigidBody's. The linear momentum of a system is equal to the vector sum of
    the linear momentum of its constituents. Consider a system, S, comprised of
    a rigid body, A, and a particle, P. The linear momentum of the system, L,
    is equal to the vector sum of the linear momentum of the particle, L1, and
    the linear momentum of the rigid body, L2, i.e.

    L = L1 + L2

    Parameters
    ==========

    frame : ReferenceFrame
        The frame in which linear momentum is desired.
    """
    # 函数用于计算系统中所有 Particle 和 RigidBody 的线动量的向量和
    pass
    body1, body2, body3... : Particle and/or RigidBody
        需要计算线动量的物体（粒子和/或刚体）。

    Examples
    ========

    >>> from sympy.physics.mechanics import Point, Particle, ReferenceFrame
    >>> from sympy.physics.mechanics import RigidBody, outer, linear_momentum
    >>> N = ReferenceFrame('N')
    >>> P = Point('P')
    >>> P.set_vel(N, 10 * N.x)
    >>> Pa = Particle('Pa', P, 1)
    >>> Ac = Point('Ac')
    >>> Ac.set_vel(N, 25 * N.y)
    >>> I = outer(N.x, N.x)
    >>> A = RigidBody('A', Ac, N, 20, (I, Ac))
    >>> linear_momentum(N, A, Pa)
    10*N.x + 500*N.y

    """

    # 检查传入的参考系是否为 ReferenceFrame 类型，如果不是则引发 TypeError 异常
    if not isinstance(frame, ReferenceFrame):
        raise TypeError('Please specify a valid ReferenceFrame')
    else:
        # 初始化系统总线动量为零向量
        linear_momentum_sys = Vector(0)
        # 遍历每个给定的物体（粒子或刚体）
        for e in body:
            # 检查物体是否为粒子或刚体类型，如果是则计算其线动量并加到系统总线动量中
            if isinstance(e, (RigidBody, Particle)):
                linear_momentum_sys += e.linear_momentum(frame)
            else:
                # 如果物体类型不符合要求，则引发 TypeError 异常
                raise TypeError('*body must have only Particle or RigidBody')
    # 返回计算得到的系统总线动量
    return linear_momentum_sys
def kinetic_energy(frame, *body):
    """Kinetic energy of a multibody system.

    Explanation
    ===========

    This function returns the kinetic energy of a system of Particle's and/or
    RigidBody's. The kinetic energy of such a system is equal to the sum of
    the kinetic energies of its constituents. Consider a system, S, comprising
    a rigid body, A, and a particle, P. The kinetic energy of the system, T,
    is equal to the vector sum of the kinetic energy of the particle, T1, and
    the kinetic energy of the rigid body, T2, i.e.

    T = T1 + T2

    Kinetic energy is a scalar.

    Parameters
    ==========

    frame : ReferenceFrame
        The frame in which the velocity or angular velocity of the body is
        defined.
    body1, body2, body3... : Particle and/or RigidBody
        The body (or bodies) whose kinetic energy is required.

    Examples
    ========

    >>> from sympy.physics.mechanics import Point, Particle, ReferenceFrame
    >>> from sympy.physics.mechanics import RigidBody, kinetic_energy
    >>> N = ReferenceFrame('N')
    >>> O = Point('O')
    >>> O.set_vel(N, 0 * N.x)
    >>> P = O.locatenew('P', 1 * N.x)
    >>> P.set_vel(N, 10 * N.x)
    >>> Pa = Particle('Pa', P, 1)
    >>> Ac = O.locatenew('Ac', 2 * N.y)
    >>> Ac.set_vel(N, 5 * N.y)
    >>> a = ReferenceFrame('a')
    >>> a.set_ang_vel(N, 10 * N.z)
    >>> I = outer(N.z, N.z)
    >>> A = RigidBody('A', Ac, a, 20, (I, Ac))
    >>> kinetic_energy(N, Pa, A)
    400

    """

    # Check if frame is a valid ReferenceFrame
    if not isinstance(frame, ReferenceFrame):
        raise TypeError('Please enter a valid ReferenceFrame')

    kinetic_energy_sys = 0
    for e in body:
        # Check if each body in *body is a Particle or RigidBody
        if isinstance(e, (RigidBody, Particle)):
            kinetic_energy_sys += e.kinetic_energy(frame)
        else:
            raise TypeError('*body must have only Particle or RigidBody')

    # Return the total kinetic energy of the system
    return kinetic_energy_sys
    # 导入必要的模块和函数
    >>> from sympy.physics.mechanics import Point, Particle, ReferenceFrame
    >>> from sympy.physics.mechanics import RigidBody, outer, kinetic_energy
    
    # 创建一个惯性参考系 N
    >>> N = ReferenceFrame('N')
    
    # 创建一个名为 O 的点对象，并设置其速度为零向量
    >>> O = Point('O')
    >>> O.set_vel(N, 0 * N.x)
    
    # 在 O 点的基础上创建点 P，并设置 P 相对于 N 的速度为 10 * N.x
    >>> P = O.locatenew('P', 1 * N.x)
    >>> P.set_vel(N, 10 * N.x)
    
    # 创建一个名为 Pa 的质点，位于点 P 处，质量为 1 单位
    >>> Pa = Particle('Pa', P, 1)
    
    # 在 O 点的基础上创建点 Ac，并设置 Ac 相对于 N 的速度为 5 * N.y
    >>> Ac = O.locatenew('Ac', 2 * N.y)
    >>> Ac.set_vel(N, 5 * N.y)
    
    # 创建一个名为 a 的参考系，并设置其相对于 N 的角速度为 10 * N.z
    >>> a = ReferenceFrame('a')
    >>> a.set_ang_vel(N, 10 * N.z)
    
    # 创建一个惯性张量 I，作为 RigidBody 的参数之一
    >>> I = outer(N.z, N.z)
    
    # 创建一个名为 A 的刚体对象，位于点 Ac 处，参考系为 a，质量为 20 单位
    >>> A = RigidBody('A', Ac, a, 20, (I, Ac))
    
    # 计算系统的动能，其中 body 包含 Pa 和 A
    >>> kinetic_energy(N, Pa, A)
    350

    """

    # 检查 frame 是否为 ReferenceFrame 类型，如果不是则抛出类型错误
    if not isinstance(frame, ReferenceFrame):
        raise TypeError('Please enter a valid ReferenceFrame')
    
    # 初始化系统的总动能为零
    ke_sys = S.Zero
    
    # 遍历系统中的每一个物体
    for e in body:
        # 如果物体是 RigidBody 或 Particle 类型，则计算其动能并累加到 ke_sys 中
        if isinstance(e, (RigidBody, Particle)):
            ke_sys += e.kinetic_energy(frame)
        else:
            # 如果物体不是期望的类型，则抛出类型错误
            raise TypeError('*body must have only Particle or RigidBody')
    
    # 返回计算得到的系统总动能
    return ke_sys
# 计算多体系统的势能
def potential_energy(*body):
    pe_sys = S.Zero  # 初始化系统势能为零
    for e in body:  # 遍历每个传入的体
        if isinstance(e, (RigidBody, Particle)):  # 检查体是否为RigidBody或Particle类型
            pe_sys += e.potential_energy  # 累加体的势能到系统势能中
        else:
            raise TypeError('*body必须仅包含Particle或RigidBody')  # 若体类型不符合要求，抛出类型错误异常
    return pe_sys  # 返回系统的总势能


# 计算给定加速度下，多个物体的万有引力
def gravity(acceleration, *bodies):
    from sympy.physics.mechanics.loads import gravity as _gravity  # 导入引力函数_gravity
    sympy_deprecation_warning(  # 发出SymPy弃用警告
        """
        gravity函数已移动。
        请从"sympy.physics.mechanics.loads"导入。
        """,
        deprecated_since_version="1.13",  # 弃用版本号
        active_deprecations_target="moved-mechanics-functions"  # 目标弃用功能
    )
    return _gravity(acceleration, *bodies)  # 返回引力函数_gravity的结果


# 计算给定点到多体系统质心的位置向量
def center_of_mass(point, *bodies):
    """
    返回给定多个物体（粒子或刚体）的质心相对于给定点的位置向量。

    示例
    =======

    >>> from sympy import symbols, S
    >>> from sympy.physics.vector import Point
    >>> from sympy.physics.mechanics import Particle, ReferenceFrame, RigidBody, outer
    >>> from sympy.physics.mechanics.functions import center_of_mass
    >>> a = ReferenceFrame('a')
    >>> m = symbols('m', real=True)
    >>> p1 = Particle('p1', Point('p1_pt'), S(1))
    >>> p2 = Particle('p2', Point('p2_pt'), S(2))
    >>> p3 = Particle('p3', Point('p3_pt'), S(3))
    >>> p4 = Particle('p4', Point('p4_pt'), m)
    >>> b_f = ReferenceFrame('b_f')
    >>> b_cm = Point('b_cm')
    >>> mb = symbols('mb')
    >>> b = RigidBody('b', b_cm, b_f, mb, (outer(b_f.x, b_f.x), b_cm))
    """
    >>> p2.point.set_pos(p1.point, a.x)
    # 设置 p2 的位置为相对于 p1 的位置加上 a.x

    >>> p3.point.set_pos(p1.point, a.x + a.y)
    # 设置 p3 的位置为相对于 p1 的位置加上 a.x 加上 a.y

    >>> p4.point.set_pos(p1.point, a.y)
    # 设置 p4 的位置为相对于 p1 的位置加上 a.y

    >>> b.masscenter.set_pos(p1.point, a.y + a.z)
    # 设置 b 的质心位置为相对于 p1 的位置加上 a.y 加上 a.z

    >>> point_o=Point('o')
    # 创建一个名称为 'o' 的点对象 point_o

    >>> point_o.set_pos(p1.point, center_of_mass(p1.point, p1, p2, p3, p4, b))
    # 设置 point_o 的位置为相对于 p1 的位置，使用传入的一组点和质心计算其位置

    >>> expr = 5/(m + mb + 6)*a.x + (m + mb + 3)/(m + mb + 6)*a.y + mb/(m + mb + 6)*a.z
    # 计算给定的表达式，用于后续的数值计算

    >>> point_o.pos_from(p1.point)
    # 计算 point_o 相对于 p1 点的位置向量

    """
    if not bodies:
        raise TypeError("No bodies(instances of Particle or Rigidbody) were passed.")

    total_mass = 0
    vec = Vector(0)
    for i in bodies:
        total_mass += i.mass

        masscenter = getattr(i, 'masscenter', None)
        if masscenter is None:
            masscenter = i.point
        vec += i.mass*masscenter.pos_from(point)

    return vec/total_mass
# 计算多体系统的拉格朗日量

def Lagrangian(frame, *body):
    """Lagrangian of a multibody system.

    Explanation
    ===========

    This function returns the Lagrangian of a system of Particle's and/or
    RigidBody's. The Lagrangian of such a system is equal to the difference
    between the kinetic energies and potential energies of its constituents. If
    T and V are the kinetic and potential energies of a system then it's
    Lagrangian, L, is defined as

    L = T - V

    The Lagrangian is a scalar.

    Parameters
    ==========

    frame : ReferenceFrame
        The frame in which the velocity or angular velocity of the body is
        defined to determine the kinetic energy.

    body1, body2, body3... : Particle and/or RigidBody
        The body (or bodies) whose Lagrangian is required.

    Examples
    ========

    >>> from sympy.physics.mechanics import Point, Particle, ReferenceFrame
    >>> from sympy.physics.mechanics import RigidBody, outer, Lagrangian
    >>> from sympy import symbols
    >>> M, m, g, h = symbols('M m g h')
    >>> N = ReferenceFrame('N')
    >>> O = Point('O')
    >>> O.set_vel(N, 0 * N.x)
    >>> P = O.locatenew('P', 1 * N.x)
    >>> P.set_vel(N, 10 * N.x)
    >>> Pa = Particle('Pa', P, 1)
    >>> Ac = O.locatenew('Ac', 2 * N.y)
    >>> Ac.set_vel(N, 5 * N.y)
    >>> a = ReferenceFrame('a')
    >>> a.set_ang_vel(N, 10 * N.z)
    >>> I = outer(N.z, N.z)
    >>> A = RigidBody('A', Ac, a, 20, (I, Ac))
    >>> Pa.potential_energy = m * g * h
    >>> A.potential_energy = M * g * h
    >>> Lagrangian(N, Pa, A)
    -M*g*h - g*h*m + 350

    """

    # 检查输入的 frame 是否为 ReferenceFrame 类型
    if not isinstance(frame, ReferenceFrame):
        raise TypeError('Please supply a valid ReferenceFrame')
    
    # 遍历所有传入的 body，检查是否为 Particle 或 RigidBody 类型
    for e in body:
        if not isinstance(e, (RigidBody, Particle)):
            raise TypeError('*body must have only Particle or RigidBody')
    
    # 返回系统的动能减去势能，得到拉格朗日量
    return kinetic_energy(frame, *body) - potential_energy(*body)


def find_dynamicsymbols(expression, exclude=None, reference_frame=None):
    """Find all dynamicsymbols in expression.

    Explanation
    ===========

    If the optional ``exclude`` kwarg is used, only dynamicsymbols
    not in the iterable ``exclude`` are returned.
    If we intend to apply this function on a vector, the optional
    ``reference_frame`` is also used to inform about the corresponding frame
    with respect to which the dynamic symbols of the given vector is to be
    determined.

    Parameters
    ==========

    expression : SymPy expression

    exclude : iterable of dynamicsymbols, optional

    reference_frame : ReferenceFrame, optional
        The frame with respect to which the dynamic symbols of the
        given vector is to be determined.

    Examples
    ========

    >>> from sympy.physics.mechanics import dynamicsymbols, find_dynamicsymbols
    >>> from sympy.physics.mechanics import ReferenceFrame
    >>> x, y = dynamicsymbols('x, y')
    >>> expr = x + x.diff()*y
    >>> find_dynamicsymbols(expr)
    {x(t), y(t), Derivative(x(t), t)}

    """

    # 找到表达式中所有的动力符号（dynamicsymbols）
    
    # 如果 exclude 参数被使用，仅返回不在 exclude 中的动力符号
    if exclude is not None:
        dynamicsymbols_set = set(dynamicsymbols(expression)) - set(exclude)
    else:
        dynamicsymbols_set = dynamicsymbols(expression)
    
    # 返回动力符号的集合
    return dynamicsymbols_set
    # 在表达式中查找所有动态符号（即未定义的函数的导数），并返回集合
    >>> find_dynamicsymbols(expr, exclude=[x, y])
    {Derivative(x(t), t)}

    # 定义三个动态符号变量 a, b, c
    >>> a, b, c = dynamicsymbols('a, b, c')

    # 创建一个参考坐标系 A
    >>> A = ReferenceFrame('A')

    # 根据动态符号和参考坐标系 A 创建一个向量 v
    >>> v = a * A.x + b * A.y + c * A.z

    # 在向量 v 中查找所有动态符号，并返回集合
    >>> find_dynamicsymbols(v, reference_frame=A)
    {a(t), b(t), c(t)}

    """
    # 初始化一个集合，包含动态符号对象的时间变量
    t_set = {dynamicsymbols._t}

    # 如果指定了排除的符号集合
    if exclude:
        # 如果排除集合是可迭代的，则转换为集合类型
        if iterable(exclude):
            exclude_set = set(exclude)
        else:
            # 如果不可迭代，则抛出类型错误异常
            raise TypeError("exclude kwarg must be iterable")
    else:
        # 否则，初始化为空集合
        exclude_set = set()

    # 如果表达式是一个向量
    if isinstance(expression, Vector):
        # 如果未提供参考坐标系，则引发值错误异常
        if reference_frame is None:
            raise ValueError("You must provide reference_frame when passing a "
                             "vector expression, got %s." % reference_frame)
        else:
            # 将向量表达式转换为参考坐标系 A 的矩阵表示
            expression = expression.to_matrix(reference_frame)

    # 返回所有属于 AppliedUndef 或 Derivative 类型的对象集合，其自由符号等于 t_set
    return {i for i in expression.atoms(AppliedUndef, Derivative) if
            i.free_symbols == t_set} - exclude_set
def msubs(expr, *sub_dicts, smart=False, **kwargs):
    """A custom subs for use on expressions derived in physics.mechanics.

    Traverses the expression tree once, performing the subs found in sub_dicts.
    Terms inside ``Derivative`` expressions are ignored:

    Examples
    ========

    >>> from sympy.physics.mechanics import dynamicsymbols, msubs
    >>> x = dynamicsymbols('x')
    >>> msubs(x.diff() + x, {x: 1})
    Derivative(x(t), t) + 1

    Note that sub_dicts can be a single dictionary, or several dictionaries:

    >>> x, y, z = dynamicsymbols('x, y, z')
    >>> sub1 = {x: 1, y: 2}
    >>> sub2 = {z: 3, x.diff(): 4}
    >>> msubs(x.diff() + x + y + z, sub1, sub2)
    10

    If smart=True (default False), also checks for conditions that may result
    in ``nan``, but if simplified would yield a valid expression. For example:

    >>> from sympy import sin, tan
    >>> (sin(x)/tan(x)).subs(x, 0)
    nan
    >>> msubs(sin(x)/tan(x), {x: 0}, smart=True)
    1

    It does this by first replacing all ``tan`` with ``sin/cos``. Then each
    node is traversed. If the node is a fraction, subs is first evaluated on
    the denominator. If this results in 0, simplification of the entire
    fraction is attempted. Using this selective simplification, only
    subexpressions that result in 1/0 are targeted, resulting in faster
    performance.

    """

    # Merge all sub_dicts into a single dictionary
    sub_dict = dict_merge(*sub_dicts)
    
    # Decide which substitution function to use based on the 'smart' flag
    if smart:
        func = _smart_subs
    elif hasattr(expr, 'msubs'):
        return expr.msubs(sub_dict)
    else:
        func = lambda expr, sub_dict: _crawl(expr, _sub_func, sub_dict)
    
    # Apply the substitution function to elements based on their type
    if isinstance(expr, (Matrix, Vector, Dyadic)):
        return expr.applyfunc(lambda x: func(x, sub_dict))
    else:
        return func(expr, sub_dict)


def _crawl(expr, func, *args, **kwargs):
    """Crawl the expression tree, and apply func to every node."""
    # Apply the given function to the current expression
    val = func(expr, *args, **kwargs)
    if val is not None:
        return val
    # Recursively apply _crawl to each argument of the expression
    new_args = (_crawl(arg, func, *args, **kwargs) for arg in expr.args)
    return expr.func(*new_args)


def _sub_func(expr, sub_dict):
    """Perform direct matching substitution, ignoring derivatives."""
    # Check if the expression matches directly with any item in sub_dict
    if expr in sub_dict:
        return sub_dict[expr]
    # If the expression has no arguments or is a derivative, return as is
    elif not expr.args or expr.is_Derivative:
        return expr


def _tan_repl_func(expr):
    """Replace tan with sin/cos."""
    # Replace instances of tan with sin/cos in the expression
    if isinstance(expr, tan):
        return sin(*expr.args) / cos(*expr.args)
    # If the expression has no arguments or is a derivative, return as is
    elif not expr.args or expr.is_Derivative:
        return expr


def _smart_subs(expr, sub_dict):
    """Performs subs, checking for conditions that may result in `nan` or
    `oo`, and attempts to simplify them out.

    The expression tree is traversed twice, and the following steps are
    performed on each expression node:
    - First traverse:
        Replace all `tan` with `sin/cos`.
    """
    # 对给定的表达式进行处理，使用指定的替换函数 _tan_repl_func 进行处理
    expr = _crawl(expr, _tan_repl_func)
    
    def _recurser(expr, sub_dict):
        # 将表达式分解为分子和分母
        num, den = _fraction_decomp(expr)
        if den != 1:
            # 如果分母不为1，则需要处理非平凡分母
            denom_subbed = _recurser(den, sub_dict)
            if denom_subbed.evalf() == 0:
                # 如果简化后分母为0，则尝试简化表达式
                expr = simplify(expr)
            else:
                # 分母不会导致结果为NaN，继续处理分子
                num_subbed = _recurser(num, sub_dict)
                return num_subbed / denom_subbed
        # 手动遍历表达式树，因为在简化步骤中 `expr` 可能已被修改。
        # 首先正常进行替换操作：
        val = _sub_func(expr, sub_dict)
        if val is not None:
            return val
        # 为每个参数执行递归处理，并重建表达式
        new_args = (_recurser(arg, sub_dict) for arg in expr.args)
        return expr.func(*new_args)
    
    # 返回使用给定的替换字典 sub_dict 递归处理后的表达式结果
    return _recurser(expr, sub_dict)
# 定义函数 _validate_coordinates，用于验证广义坐标和广义速度
def _validate_coordinates(coordinates=None, speeds=None, check_duplicates=True,
                          is_dynamicsymbols=True, u_auxiliary=None):
    """Validate the generalized coordinates and generalized speeds.

    Parameters
    ==========
    coordinates : iterable, optional
        Generalized coordinates to be validated.
    speeds : iterable, optional
        Generalized speeds to be validated.
    check_duplicates : bool, optional
        Checks if there are duplicates in the generalized coordinates and
        generalized speeds. If so it will raise a ValueError. The default is
        True.
    is_dynamicsymbols : iterable, optional
        Checks if all the generalized coordinates and generalized speeds are
        dynamicsymbols. If any is not a dynamicsymbol, a ValueError will be
        raised. The default is True.
    u_auxiliary : iterable, optional
        Auxiliary generalized speeds to be validated.

    """
    # 设置时间变量集合，包含动力符号模块中的时间变量
    t_set = {dynamicsymbols._t}
    
    # 将输入转换为可迭代对象
    if coordinates is None:
        coordinates = []
    elif not iterable(coordinates):
        coordinates = [coordinates]
    if speeds is None:
        speeds = []
    elif not iterable(speeds):
        speeds = [speeds]
    if u_auxiliary is None:
        u_auxiliary = []
    elif not iterable(u_auxiliary):
        u_auxiliary = [u_auxiliary]

    # 初始化消息列表
    msgs = []
    # 如果需要检查重复项
    if check_duplicates:  # Check for duplicates
        # 创建一个空集合用于记录已经见过的元素
        seen = set()
        # 检查并获取重复的广义坐标
        coord_duplicates = {x for x in coordinates if x in seen or seen.add(x)}
        # 重置已见元素集合
        seen = set()
        # 检查并获取重复的广义速度
        speed_duplicates = {x for x in speeds if x in seen or seen.add(x)}
        # 重置已见元素集合
        seen = set()
        # 检查并获取重复的辅助速度
        aux_duplicates = {x for x in u_auxiliary if x in seen or seen.add(x)}
        
        # 找到重叠的广义坐标和广义速度
        overlap_coords = set(coordinates).intersection(speeds)
        # 找到重叠的广义坐标和辅助速度
        overlap_aux = set(coordinates).union(speeds).intersection(u_auxiliary)
        
        # 如果有重复的广义坐标，添加相应的错误消息
        if coord_duplicates:
            msgs.append(f'The generalized coordinates {coord_duplicates} are '
                        f'duplicated, all generalized coordinates should be '
                        f'unique.')
        # 如果有重复的广义速度，添加相应的错误消息
        if speed_duplicates:
            msgs.append(f'The generalized speeds {speed_duplicates} are '
                        f'duplicated, all generalized speeds should be unique.')
        # 如果有重复的辅助速度，添加相应的错误消息
        if aux_duplicates:
            msgs.append(f'The auxiliary speeds {aux_duplicates} are duplicated,'
                        f' all auxiliary speeds should be unique.')
        # 如果有重叠的广义坐标和广义速度，添加相应的错误消息
        if overlap_coords:
            msgs.append(f'{overlap_coords} are defined as both generalized '
                        f'coordinates and generalized speeds.')
        # 如果有重叠的辅助速度与广义坐标或广义速度，添加相应的错误消息
        if overlap_aux:
            msgs.append(f'The auxiliary speeds {overlap_aux} are also defined '
                        f'as generalized coordinates or generalized speeds.')
    
    # 如果需要检查是否为动态符号
    if is_dynamicsymbols:  # Check whether all coordinates are dynamicsymbols
        # 检查每个广义坐标是否是动态符号
        for coordinate in coordinates:
            if not (isinstance(coordinate, (AppliedUndef, Derivative)) and
                    coordinate.free_symbols == t_set):
                msgs.append(f'Generalized coordinate "{coordinate}" is not a '
                            f'dynamicsymbol.')
        # 检查每个广义速度是否是动态符号
        for speed in speeds:
            if not (isinstance(speed, (AppliedUndef, Derivative)) and
                    speed.free_symbols == t_set):
                msgs.append(
                    f'Generalized speed "{speed}" is not a dynamicsymbol.')
        # 检查每个辅助速度是否是动态符号
        for aux in u_auxiliary:
            if not (isinstance(aux, (AppliedUndef, Derivative)) and
                    aux.free_symbols == t_set):
                msgs.append(
                    f'Auxiliary speed "{aux}" is not a dynamicsymbol.')
    
    # 如果有错误消息被记录，抛出值错误异常
    if msgs:
        raise ValueError('\n'.join(msgs))
# 定义一个辅助函数，用于获取指定的线性求解器
def _parse_linear_solver(linear_solver):
    # 如果 linear_solver 是可调用对象（函数），直接返回它
    if callable(linear_solver):
        return linear_solver
    # 如果 linear_solver 不是可调用对象，返回一个 lambda 函数
    # lambda 函数接受矩阵 A 和向量 b 作为参数，并使用指定的 linear_solver 方法来求解
    return lambda A, b: Matrix.solve(A, b, method=linear_solver)
```