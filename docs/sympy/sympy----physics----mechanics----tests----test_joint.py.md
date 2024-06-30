# `D:\src\scipysrc\sympy\sympy\physics\mechanics\tests\test_joint.py`

```
# 从 sympy.core.function 模块导入 expand_mul 函数
# 从 sympy.core.numbers 模块导入 pi 常数
# 从 sympy.core.singleton 模块导入 S 单例对象
# 从 sympy.functions.elementary.miscellaneous 模块导入 sqrt 函数
# 从 sympy.functions.elementary.trigonometric 模块导入 cos 和 sin 函数
# 从 sympy 模块导入 Matrix, simplify, eye, zeros 函数
# 从 sympy.core.symbol 模块导入 symbols 函数
# 从 sympy.physics.mechanics 模块导入 dynamicsymbols, RigidBody, Particle, JointsMethod,
# PinJoint, PrismaticJoint, CylindricalJoint, PlanarJoint, SphericalJoint, WeldJoint, Body 类
# 从 sympy.physics.mechanics.joint 模块导入 Joint 类
# 从 sympy.physics.vector 模块导入 Vector, ReferenceFrame, Point 类
# 从 sympy.testing.pytest 模块导入 raises, warns_deprecated_sympy 函数
from sympy.core.function import expand_mul
from sympy.core.numbers import pi
from sympy.core.singleton import S
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy import Matrix, simplify, eye, zeros
from sympy.core.symbol import symbols
from sympy.physics.mechanics import (
    dynamicsymbols, RigidBody, Particle, JointsMethod, PinJoint, PrismaticJoint,
    CylindricalJoint, PlanarJoint, SphericalJoint, WeldJoint, Body)
from sympy.physics.mechanics.joint import Joint
from sympy.physics.vector import Vector, ReferenceFrame, Point
from sympy.testing.pytest import raises, warns_deprecated_sympy

# 为动力符号 _t 设置一个别名 t，用于时间符号
t = dynamicsymbols._t  # type: ignore


def _generate_body(interframe=False):
    # 定义一个惯性参考系 N
    N = ReferenceFrame('N')
    # 定义一个附加参考系 A
    A = ReferenceFrame('A')
    # 创建一个以 N 参考系为基础的刚体 P
    P = RigidBody('P', frame=N)
    # 创建一个以 A 参考系为基础的刚体 C
    C = RigidBody('C', frame=A)
    if interframe:
        # 如果需要创建中间参考系，定义 Pint 和 Cint 参考系
        Pint, Cint = ReferenceFrame('P_int'), ReferenceFrame('C_int')
        # 设置 Pint 参考系相对于 N 参考系的旋转轴为 N.x，并旋转角度为 pi
        Pint.orient_axis(N, N.x, pi)
        # 设置 Cint 参考系相对于 A 参考系的旋转轴为 A.y，并旋转角度为 -pi/2
        Cint.orient_axis(A, A.y, -pi / 2)
        # 返回定义的参考系和刚体
        return N, A, P, C, Pint, Cint
    # 如果不需要创建中间参考系，直接返回定义的参考系和刚体
    return N, A, P, C


def test_Joint():
    # 创建一个名为 parent 的惯性参考系
    parent = RigidBody('parent')
    # 创建一个名为 child 的惯性参考系
    child = RigidBody('child')
    # 测试是否会引发 TypeError 异常，预期连接 parent 和 child 的 Joint 对象
    raises(TypeError, lambda: Joint('J', parent, child))


def test_coordinate_generation():
    # 创建动力符号 q, u, qj, uj
    q, u, qj, uj = dynamicsymbols('q u q_J u_J')
    # 创建动力符号 q0j, q1j, q2j, q3j, u0j, u1j, u2j, u3j
    q0j, q1j, q2j, q3j, u0j, u1j, u2j, u3j = dynamicsymbols('q0:4_J u0:4_J')
    # 创建动力符号 q0, q1, q2, q3, u0, u1, u2, u3
    q0, q1, q2, q3, u0, u1, u2, u3 = dynamicsymbols('q0:4 u0:4')
    # 创建 N, A, P, C 四个惯性参考系和刚体
    _, _, P, C = _generate_body()
    # 使用 PinJoint 类访问 Joint 的坐标生成方法
    J = PinJoint('J', P, C)
    # 测试单个给定坐标
    assert J._fill_coordinate_list(q, 1) == Matrix([q])
    assert J._fill_coordinate_list([u], 1) == Matrix([u])
    assert J._fill_coordinate_list([u], 1, offset=2) == Matrix([u])
    # 测试 None
    assert J._fill_coordinate_list(None, 1) == Matrix([qj])
    assert J._fill_coordinate_list([None], 1) == Matrix([qj])
    assert J._fill_coordinate_list([q0, None, None], 3) == Matrix(
        [q0, q1j, q2j])
    # 测试自动填充
    assert J._fill_coordinate_list(None, 3) == Matrix([q0j, q1j, q2j])
    assert J._fill_coordinate_list([], 3) == Matrix([q0j, q1j, q2j])
    # 测试偏移量
    assert J._fill_coordinate_list([], 3, offset=1) == Matrix([q1j, q2j, q3j])
    assert J._fill_coordinate_list([q1, None, q3], 3, offset=1) == Matrix(
        [q1, q2j, q3])
    assert J._fill_coordinate_list(None, 2, offset=2) == Matrix([q2j, q3j])
    # 测试标签
    assert J._fill_coordinate_list(None, 1, 'u') == Matrix([uj])
    assert J._fill_coordinate_list([], 3, 'u') == Matrix([u0j, u1j, u2j])
    # 测试单个编号
    assert J._fill_coordinate_list(None, 1, number_single=True) == Matrix([q0j])
    assert J._fill_coordinate_list([], 1, 'u', 2, True) == Matrix([u2j])
    assert J._fill_coordinate_list([], 3, 'q') == Matrix([q0j, q1j, q2j])
    # 测试提供的坐标数量不合法
    # 调用 J 对象的 _fill_coordinate_list 方法，预期引发 ValueError 异常，测试填充坐标列表时出现错误情况
    raises(ValueError, lambda: J._fill_coordinate_list([q0, q1], 1))
    # 调用 J 对象的 _fill_coordinate_list 方法，预期引发 ValueError 异常，测试填充坐标列表时空间数不匹配的错误情况
    raises(ValueError, lambda: J._fill_coordinate_list([u0, u1, None], 2, 'u'))
    # 调用 J 对象的 _fill_coordinate_list 方法，预期引发 ValueError 异常，测试填充坐标列表时坐标数不匹配的错误情况
    raises(ValueError, lambda: J._fill_coordinate_list([q0, q1], 3))
    # 测试填充坐标列表时遇到错误的坐标类型，预期引发 TypeError 异常
    raises(TypeError, lambda: J._fill_coordinate_list([q0, symbols('q1')], 2))
    # 测试填充坐标列表时遇到错误的坐标类型，预期引发 TypeError 异常
    raises(TypeError, lambda: J._fill_coordinate_list([q0 + q1, q1], 2))
    # 创建 PinJoint 对象，使用 q1 和 q1 对时间 t 的导数作为广义速度，测试是否允许将导数作为广义速度
    _, _, P, C = _generate_body()
    PinJoint('J', P, C, q1, q1.diff(t))
    # 创建 SphericalJoint 对象，测试当存在重复坐标时是否引发 ValueError 异常
    _, _, P, C = _generate_body()
    raises(ValueError, lambda: SphericalJoint('J', P, C, [q1j, None, None]))
    # 创建 SphericalJoint 对象，测试当存在重复广义速度时是否引发 ValueError 异常
    raises(ValueError, lambda: SphericalJoint('J', P, C, speeds=[u0, u0, u1]))
def test_pin_joint():
    # 创建两个刚体对象 P 和 C
    P = RigidBody('P')
    C = RigidBody('C')
    
    # 定义符号变量 l 和 m
    l, m = symbols('l m')
    
    # 定义动力学符号 q 和 u
    q, u = dynamicsymbols('q_J, u_J')
    
    # 创建 PinJoint 对象 Pj，连接 P 和 C
    Pj = PinJoint('J', P, C)
    
    # 断言 PinJoint 对象的属性
    assert Pj.name == 'J'
    assert Pj.parent == P
    assert Pj.child == C
    assert Pj.coordinates == Matrix([q])
    assert Pj.speeds == Matrix([u])
    assert Pj.kdes == Matrix([u - q.diff(t)])
    assert Pj.joint_axis == P.frame.x
    
    # 断言 PinJoint 对象中的点的位置关系
    assert Pj.child_point.pos_from(C.masscenter) == Vector(0)
    assert Pj.parent_point.pos_from(P.masscenter) == Vector(0)
    assert Pj.parent_point.pos_from(Pj._child_point) == Vector(0)
    
    # 断言 C 和 P 的质心之间的位置关系
    assert C.masscenter.pos_from(P.masscenter) == Vector(0)
    
    # 断言 PinJoint 对象的插入参考帧
    assert Pj.parent_interframe == P.frame
    assert Pj.child_interframe == C.frame
    
    # 断言 PinJoint 对象的字符串表示
    assert Pj.__str__() == 'PinJoint: J  parent: P  child: C'

    # 创建另外两个刚体对象 P1 和 C1，以及一个参考帧 Pint
    P1 = RigidBody('P1')
    C1 = RigidBody('C1')
    Pint = ReferenceFrame('P_int')
    
    # 设置参考帧 Pint 的方向
    Pint.orient_axis(P1.frame, P1.y, pi / 2)
    
    # 创建 PinJoint 对象 J1，指定了各种连接参数
    J1 = PinJoint('J1', P1, C1, parent_point=l*P1.frame.x,
                  child_point=m*C1.frame.y, joint_axis=P1.frame.z,
                  parent_interframe=Pint)
    
    # 断言 PinJoint 对象 J1 的各种属性
    assert J1._joint_axis == P1.frame.z
    assert J1._child_point.pos_from(C1.masscenter) == m * C1.frame.y
    assert J1._parent_point.pos_from(P1.masscenter) == l * P1.frame.x
    assert J1._parent_point.pos_from(J1._child_point) == Vector(0)
    
    # 断言 C1 和 P1 的质心之间的位置关系
    assert (P1.masscenter.pos_from(C1.masscenter) ==
            -l*P1.frame.x + m*C1.frame.y)
    
    # 断言 PinJoint 对象 J1 的插入参考帧
    assert J1.parent_interframe == Pint
    assert J1.child_interframe == C1.frame

    # 重新定义动力学符号 q 和 u
    q, u = dynamicsymbols('q, u')
    
    # 生成 N、A、P、C、Pint、Cint 这些物体
    N, A, P, C, Pint, Cint = _generate_body(True)
    
    # 在 P 的质心位置上创建一个新的点 parent_point
    parent_point = P.masscenter.locatenew('parent_point', N.x + N.y)
    
    # 在 C 的质心位置上创建一个新的点 child_point
    child_point = C.masscenter.locatenew('child_point', C.y + C.z)
    
    # 创建 PinJoint 对象 J，指定了各种连接参数
    J = PinJoint('J', P, C, q, u, parent_point=parent_point,
                 child_point=child_point, parent_interframe=Pint,
                 child_interframe=Cint, joint_axis=N.z)
    
    # 断言 PinJoint 对象 J 的各种属性
    assert J.joint_axis == N.z
    assert J.parent_point.vel(N) == 0
    assert J.parent_point == parent_point
    assert J.child_point == child_point
    assert J.child_point.pos_from(P.masscenter) == N.x + N.y
    assert J.parent_point.pos_from(C.masscenter) == C.y + C.z
    assert C.masscenter.pos_from(P.masscenter) == N.x + N.y - C.y - C.z
    
    # 断言 PinJoint 对象 J 的插入参考帧
    assert J.parent_interframe == Pint
    assert J.child_interframe == Cint


def test_particle_compatibility():
    # 定义符号变量 m 和 l
    m, l = symbols('m l')
    
    # 创建参考帧 C_frame
    C_frame = ReferenceFrame('C')
    
    # 创建粒子对象 P 和 C
    P = Particle('P')
    C = Particle('C', mass=m)
    
    # 定义动力学符号 q 和 u
    q, u = dynamicsymbols('q, u')
    
    # 创建 PinJoint 对象 J，指定了各种连接参数
    J = PinJoint('J', P, C, q, u, child_interframe=C_frame,
                 child_point=l * C_frame.y)
    
    # 断言 PinJoint 对象 J 的各种属性
    assert J.child_interframe == C_frame
    assert J.parent_interframe.name == 'J_P_frame'
    assert C.masscenter.pos_from(P.masscenter) == -l * C_frame.y
    # 断言：验证 C_frame.dcm(J.parent_interframe) 的值是否等于给定的旋转矩阵
    assert C_frame.dcm(J.parent_interframe) == Matrix([[1, 0, 0],
                                                       [0, cos(q), sin(q)],
                                                       [0, -sin(q), cos(q)]])
    
    # 断言：验证 C.masscenter.vel(J.parent_interframe) 的值是否等于指定的线速度表达式
    assert C.masscenter.vel(J.parent_interframe) == -l * u * C_frame.z
    
    # 使用指定的参考框架创建粒子 P 和 C
    P_frame = ReferenceFrame('P')
    C_frame = ReferenceFrame('C')
    P = Particle('P')
    C = Particle('C', mass=m)
    
    # 定义动力学符号 q 和 u
    q, u = dynamicsymbols('q, u')
    
    # 创建 PinJoint 关节 J，连接粒子 P 和 C，指定联动框架和联动点
    J = PinJoint('J', P, C, q, u, parent_interframe=P_frame,
                 child_interframe=C_frame, child_point=l * C_frame.y,
                 joint_axis=P_frame.z)
    
    # 断言：验证 J.joint_axis 是否等于 J.parent_interframe.z
    assert J.joint_axis == J.parent_interframe.z
    
    # 断言：验证 C_frame.dcm(J.parent_interframe) 的值是否等于给定的旋转矩阵
    assert C_frame.dcm(J.parent_interframe) == Matrix([[cos(q), sin(q), 0],
                                                       [-sin(q), cos(q), 0],
                                                       [0, 0, 1]])
    
    # 断言：验证 P.masscenter.vel(J.parent_interframe) 的值是否等于 0
    assert P.masscenter.vel(J.parent_interframe) == 0
    
    # 断言：验证 C.masscenter.vel(J.parent_interframe) 的值是否等于指定的线速度表达式
    assert C.masscenter.vel(J.parent_interframe) == l * u * C_frame.x
    
    # 定义更多的动力学符号 q1, q2, q3, u1, u2, u3
    q1, q2, q3, u1, u2, u3 = dynamicsymbols('q1:4 u1:4')
    
    # 创建字典 qdot_to_u，用于表示 q 的导数与 u 之间的映射关系
    qdot_to_u = {qi.diff(t): ui for qi, ui in ((q1, u1), (q2, u2), (q3, u3))}
    
    # 创建 PrismaticJoint 关节 J，连接粒子 P 和 C，使用动力学符号 q 和 u
    J = PrismaticJoint('J', P, C, q, u)
    
    # 断言：验证 J.parent_interframe.dcm(J.child_interframe) 的值是否为单位矩阵
    assert J.parent_interframe.dcm(J.child_interframe) == eye(3)
    
    # 断言：验证 C.masscenter.pos_from(P.masscenter) 的值是否等于指定的位移表达式
    assert C.masscenter.pos_from(P.masscenter) == q * J.parent_interframe.x
    
    # 断言：验证 P.masscenter.vel(J.parent_interframe) 的值是否为 0
    assert P.masscenter.vel(J.parent_interframe) == 0
    
    # 断言：验证 C.masscenter.vel(J.parent_interframe) 的值是否等于指定的线速度表达式
    assert C.masscenter.vel(J.parent_interframe) == u * J.parent_interframe.x
    
    # 创建 CylindricalJoint 关节 J，连接粒子 P 和 C，使用动力学符号 q1, q2, u1, u2，指定联动框架和联动点
    P_frame = ReferenceFrame('P_frame')
    J = CylindricalJoint('J', P, C, q1, q2, u1, u2, parent_interframe=P_frame,
                         parent_point=l * P_frame.x, joint_axis=P_frame.y)
    
    # 断言：验证 J.parent_interframe.dcm(J.child_interframe) 的值是否等于给定的旋转矩阵
    assert J.parent_interframe.dcm(J.child_interframe) == Matrix([
        [cos(q1), 0, sin(q1)], [0, 1, 0], [-sin(q1), 0, cos(q1)]])
    
    # 断言：验证 C.masscenter.pos_from(P.masscenter) 的值是否等于指定的位移表达式
    assert C.masscenter.pos_from(P.masscenter) == l * P_frame.x + q2 * P_frame.y
    
    # 断言：验证 C.masscenter.vel(J.parent_interframe) 的值是否等于指定的线速度表达式
    assert C.masscenter.vel(J.parent_interframe) == u2 * P_frame.y
    
    # 断言：验证 P.masscenter.vel(J.child_interframe) 的 x 分量替换 qdot_to_u 后的值是否等于指定表达式
    assert P.masscenter.vel(J.child_interframe).xreplace(qdot_to_u) == (
        -u2 * P_frame.y - l * u1 * P_frame.z)
    
    # 创建 PlanarJoint 关节 J，连接粒子 P 和 C，使用动力学符号 q1, [q2, q3], u1, [u2, u3]，指定联动框架和联动点
    C_frame = ReferenceFrame('C_frame')
    J = PlanarJoint('J', P, C, q1, [q2, q3], u1, [u2, u3],
                    child_interframe=C_frame, child_point=l * C_frame.z)
    
    # 将 P_frame 设置为 J 的 parent_interframe
    P_frame = J.parent_interframe
    
    # 断言：验证 J.parent_interframe.dcm(J.child_interframe) 的值是否等于给定的旋转矩阵
    assert J.parent_interframe.dcm(J.child_interframe) == Matrix([
        [1, 0, 0], [0, cos(q1), -sin(q1)], [0, sin(q1), cos(q1)]])
    
    # 断言：验证 C.masscenter.pos_from(P.masscenter) 的值是否等于指定的位移表达式
    assert C.masscenter.pos_from(P.masscenter) == (
        -l * C_frame.z + q2 * P_frame.y + q3 * P_frame.z)
    
    # 断言：验证 C.masscenter.vel(J.parent_interframe) 的值是否等于指定的线速度表达式
    assert C.masscenter.vel(J.parent_interframe) == (
        l * u1 * C_frame.y + u2 * P_frame.y + u3 * P_frame.z)
    
    # 创建 WeldJoint 关节 J，连接粒子 P 和 C
    P, C = Particle('P'), Particle('C')
    # 创建名为 C_frame 和 P_frame 的参考框架对象
    C_frame, P_frame = ReferenceFrame('C_frame'), ReferenceFrame('P_frame')
    
    # 创建名为 J 的焊接关节对象，将其连接到 P 和 C 两个点之间
    # P 是父参考框架，C 是子参考框架，父参考框架的相对框架是 P_frame，子参考框架的相对框架是 C_frame
    # parent_point 和 child_point 分别为父框架和子框架的点，这里分别是 l * P_frame.x 和 l * C_frame.y
    J = WeldJoint('J', P, C, parent_interframe=P_frame,
                  child_interframe=C_frame, parent_point=l * P_frame.x,
                  child_point=l * C_frame.y)
    
    # 断言父参考框架相对于子参考框架的方向余弦矩阵为单位矩阵
    assert P_frame.dcm(C_frame) == eye(3)
    
    # 断言质心坐标系相对于 P 和 C 的质心之间的距离与 l * P_frame.x - l * C_frame.y 相等
    assert C.masscenter.pos_from(P.masscenter) == l * P_frame.x - l * C_frame.y
    
    # 断言质心的速度相对于 J 的父参考框架的速度为零
    assert C.masscenter.vel(J.parent_interframe) == 0
def test_pin_joint_chaos_pendulum():
    # 定义符号变量
    mA, mB, lA, lB, h = symbols('mA, mB, lA, lB, h')
    # 定义动力学符号变量
    theta, phi, omega, alpha = dynamicsymbols('theta phi omega alpha')
    # 定义参考系
    N = ReferenceFrame('N')
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    # 计算连接点位置
    lA = (lB - h / 2) / 2
    lC = (lB/2 + h/4)
    # 创建刚体
    rod = RigidBody('rod', frame=A, mass=mA)
    plate = RigidBody('plate', mass=mB, frame=B)
    C = RigidBody('C', frame=N)
    # 创建连接关节
    J1 = PinJoint('J1', C, rod, coordinates=theta, speeds=omega,
                  child_point=lA*A.z, joint_axis=N.y)
    J2 = PinJoint('J2', rod, plate, coordinates=phi, speeds=alpha,
                  parent_point=lC*A.z, joint_axis=A.z)

    # 检查方向性
    # 断言：计算A相对于N的方向余弦矩阵是否等于给定的旋转矩阵
    assert A.dcm(N) == Matrix([[cos(theta), 0, -sin(theta)],
                               [0, 1, 0],
                               [sin(theta), 0, cos(theta)]])
    
    # 断言：计算A相对于B的方向余弦矩阵是否等于给定的旋转矩阵
    assert A.dcm(B) == Matrix([[cos(phi), -sin(phi), 0],
                               [sin(phi), cos(phi), 0],
                               [0, 0, 1]])
    
    # 断言：计算B相对于N的方向余弦矩阵是否等于给定的旋转矩阵
    assert B.dcm(N) == Matrix([
        [cos(phi)*cos(theta), sin(phi), -sin(theta)*cos(phi)],
        [-sin(phi)*cos(theta), cos(phi), sin(phi)*sin(theta)],
        [sin(theta), 0, cos(theta)]])

    # 检查角速度
    assert A.ang_vel_in(N) == omega*N.y
    assert A.ang_vel_in(B) == -alpha*A.z
    assert N.ang_vel_in(B) == -omega*N.y - alpha*A.z

    # 检查 kde
    assert J1.kdes == Matrix([omega - theta.diff(t)])
    assert J2.kdes == Matrix([alpha - phi.diff(t)])

    # 检查质心位置
    assert C.masscenter.pos_from(rod.masscenter) == lA*A.z
    assert rod.masscenter.pos_from(plate.masscenter) == - lC * A.z

    # 检查线速度
    assert rod.masscenter.vel(N) == (h/4 - lB/2)*omega*A.x
    assert plate.masscenter.vel(N) == ((h/4 - lB/2)*omega +
                                       (h/4 + lB/2)*omega)*A.x
def test_pin_joint_interframe():
    q, u = dynamicsymbols('q, u')
    # 定义四个运动的参考系 N, A, P, C
    N, A, P, C = _generate_body()
    
    # 创建两个新的参考系 Pint 和 Cint
    Pint, Cint = ReferenceFrame('Pint'), ReferenceFrame('Cint')
    
    # 测试不连接的情况，应该引发 ValueError 异常
    raises(ValueError, lambda: PinJoint('J', P, C, parent_interframe=Pint))
    raises(ValueError, lambda: PinJoint('J', P, C, child_interframe=Cint))
    
    # 检查不固定的 interframe 的情况
    Pint.orient_axis(N, N.z, q)
    Cint.orient_axis(A, A.z, q)
    raises(ValueError, lambda: PinJoint('J', P, C, parent_interframe=Pint))
    raises(ValueError, lambda: PinJoint('J', P, C, child_interframe=Cint))
    
    # 只检查 parent_interframe 的情况
    N, A, P, C = _generate_body()
    Pint = ReferenceFrame('Pint')
    Pint.orient_body_fixed(N, (pi / 4, pi, pi / 3), 'xyz')
    PinJoint('J', P, C, q, u, parent_point=N.x, child_point=-C.y,
             parent_interframe=Pint, joint_axis=Pint.x)
    
    # 断言N到A的方向余弦矩阵
    assert simplify(N.dcm(A)) - Matrix([
        [-1 / 2, sqrt(3) * cos(q) / 2, -sqrt(3) * sin(q) / 2],
        [sqrt(6) / 4, sqrt(2) * (2 * sin(q) + cos(q)) / 4,
         sqrt(2) * (-sin(q) + 2 * cos(q)) / 4],
        [sqrt(6) / 4, sqrt(2) * (-2 * sin(q) + cos(q)) / 4,
         -sqrt(2) * (sin(q) + 2 * cos(q)) / 4]]) == zeros(3)
    
    # 断言 A 相对于 N 的角速度
    assert A.ang_vel_in(N) == u * Pint.x
    
    # 断言 C 质心相对于 P 质心的位置
    assert C.masscenter.pos_from(P.masscenter) == N.x + A.y
    
    # 断言 C 质心在参考系 N 中的速度
    assert C.masscenter.vel(N) == u * A.z
    
    # 断言 P 质心在 Pint 参考系中的速度为零
    assert P.masscenter.vel(Pint) == Vector(0)
    
    # 断言 C 质心在 Pint 参考系中的速度
    assert C.masscenter.vel(Pint) == u * A.z
    
    # 只检查 child_interframe 的情况
    N, A, P, C = _generate_body()
    Cint = ReferenceFrame('Cint')
    Cint.orient_body_fixed(A, (2 * pi / 3, -pi, pi / 2), 'xyz')
    PinJoint('J', P, C, q, u, parent_point=-N.z, child_point=C.x,
             child_interframe=Cint, joint_axis=P.x + P.z)
    
    # 断言 N 到 A 的方向余弦矩阵
    assert simplify(N.dcm(A)) == Matrix([
        [-sqrt(2) * sin(q) / 2,
         -sqrt(3) * (cos(q) - 1) / 4 - cos(q) / 4 - S(1) / 4,
         sqrt(3) * (cos(q) + 1) / 4 - cos(q) / 4 + S(1) / 4],
        [cos(q), (sqrt(2) + sqrt(6)) * -sin(q) / 4,
         (-sqrt(2) + sqrt(6)) * sin(q) / 4],
        [sqrt(2) * sin(q) / 2,
         sqrt(3) * (cos(q) + 1) / 4 + cos(q) / 4 - S(1) / 4,
         sqrt(3) * (1 - cos(q)) / 4 + cos(q) / 4 + S(1) / 4]])
    
    # 断言 A 相对于 N 的角速度
    assert A.ang_vel_in(N) == sqrt(2) * u / 2 * N.x + sqrt(2) * u / 2 * N.z
    
    # 断言 C 质心相对于 P 质心的位置
    assert C.masscenter.pos_from(P.masscenter) == - N.z - A.x
    
    # 断言 C 质心在 N 参考系中的速度
    assert C.masscenter.vel(N).simplify() == (
        -sqrt(6) - sqrt(2)) * u / 4 * A.y + (
               -sqrt(2) + sqrt(6)) * u / 4 * A.z
    
    # 断言 C 质心在 Cint 参考系中的速度为零
    assert C.masscenter.vel(Cint) == Vector(0)
    
    # 检查组合情况
    N, A, P, C = _generate_body()
    Pint, Cint = ReferenceFrame('Pint'), ReferenceFrame('Cint')
    Pint.orient_body_fixed(N, (-pi / 2, pi, pi / 2), 'xyz')
    Cint.orient_body_fixed(A, (2 * pi / 3, -pi, pi / 2), 'xyz')
    PinJoint('J', P, C, q, u, parent_point=N.x - N.y, child_point=-C.z,
             parent_interframe=Pint, child_interframe=Cint,
             joint_axis=Pint.x + Pint.z)
    # 确保简化后的表达式与给定的矩阵相等
    assert simplify(N.dcm(A)) == Matrix([
        [cos(q), (sqrt(2) + sqrt(6)) * -sin(q) / 4, 
         (-sqrt(2) + sqrt(6)) * sin(q) / 4],
        [-sqrt(2) * sin(q) / 2, 
         -sqrt(3) * (cos(q) + 1) / 4 - cos(q) / 4 + S(1) / 4, 
         sqrt(3) * (cos(q) - 1) / 4 - cos(q) / 4 - S(1) / 4],
        [sqrt(2) * sin(q) / 2, 
         sqrt(3) * (cos(q) - 1) / 4 + cos(q) / 4 + S(1) / 4, 
         -sqrt(3) * (cos(q) + 1) / 4 + cos(q) / 4 - S(1) / 4]])
    
    # 确保 A 相对于 N 的角速度与给定的表达式相等
    assert A.ang_vel_in(N) == sqrt(2) * u / 2 * Pint.x + sqrt(2) * u / 2 * Pint.z
    
    # 确保 C 的质心相对于 P 的质心位置矢量与给定的表达式相等
    assert C.masscenter.pos_from(P.masscenter) == N.x - N.y + A.z
    
    # 计算 N 到 C 质心的速度向量，并简化表达式
    N_v_C = (-sqrt(2) + sqrt(6)) * u / 4 * A.x
    assert C.masscenter.vel(N).simplify() == N_v_C
    
    # 确保 C 质心相对于 Pint 点的速度向量与给定的表达式相等，并简化表达式
    assert C.masscenter.vel(Pint).simplify() == N_v_C
    
    # 确保 C 质心相对于 Cint 点的速度向量为零向量
    assert C.masscenter.vel(Cint) == Vector(0)
# 定义测试函数，测试 PinJoint 类的 joint_axis 参数的各种情况
def test_pin_joint_joint_axis():
    # 定义动力学符号 q 和 u
    q, u = dynamicsymbols('q, u')

    # 创建多个身体和关节对象，返回的变量分别赋值给 N, A, P, C, Pint, Cint
    N, A, P, C, Pint, Cint = _generate_body(True)

    # 创建 PinJoint 对象 pin，关节轴为 P.y
    pin = PinJoint('J', P, C, q, u, parent_interframe=Pint,
                   child_interframe=Cint, joint_axis=P.y)
    # 断言关节轴是否为 P.y
    assert pin.joint_axis == P.y
    # 断言 N 关于 A 的方向余弦矩阵
    assert N.dcm(A) == Matrix([[sin(q), 0, cos(q)], [0, -1, 0],
                               [cos(q), 0, -sin(q)]])

    # 重置多个身体和关节对象的变量值
    N, A, P, C, Pint, Cint = _generate_body(True)

    # 创建 PinJoint 对象 pin，关节轴为 Pint.y
    pin = PinJoint('J', P, C, q, u, parent_interframe=Pint,
                   child_interframe=Cint, joint_axis=Pint.y)
    # 断言关节轴是否为 Pint.y
    assert pin.joint_axis == Pint.y
    # 断言 N 关于 A 的方向余弦矩阵
    assert N.dcm(A) == Matrix([[-sin(q), 0, cos(q)], [0, -1, 0],
                               [cos(q), 0, sin(q)]])

    # 创建 PinJoint 对象 pin，关节轴为 N.z，parent_interframe 为 N.z，child_interframe 为 -C.z
    pin = PinJoint('J', P, C, q, u, parent_interframe=N.z,
                   child_interframe=-C.z, joint_axis=N.z)
    # 断言关节轴是否为 N.z
    assert pin.joint_axis == N.z
    # 断言 N 关于 A 的方向余弦矩阵
    assert N.dcm(A) == Matrix([[-cos(q), -sin(q), 0], [-sin(q), cos(q), 0],
                               [0, 0, -1]])

    # 重置多个身体和关节对象的变量值
    N, A, P, C = _generate_body()

    # 创建 PinJoint 对象 pin，关节轴为 N.x，parent_interframe 为 N.z，child_interframe 为 -C.z
    pin = PinJoint('J', P, C, q, u, parent_interframe=N.z,
                   child_interframe=-C.z, joint_axis=N.x)
    # 断言关节轴是否为 N.x
    assert pin.joint_axis == N.x
    # 断言 N 关于 A 的方向余弦矩阵
    assert N.dcm(A) == Matrix([[-1, 0, 0], [0, cos(q), sin(q)],
                               [0, sin(q), -cos(q)]])

    # 创建 PinJoint 对象 pin，关节轴为 cos(q)*N.x + sin(q)*N.y，这是一个时间变化的轴
    N, A, P, C, Pint, Cint = _generate_body(True)
    # 断言创建 PinJoint 对象时会抛出 ValueError 异常
    raises(ValueError, lambda: PinJoint('J', P, C,
                                        joint_axis=cos(q) * N.x + sin(q) * N.y))

    # 创建 PinJoint 对象 pin，关节轴为 C.x，这在 child frame 中是无效的
    raises(ValueError, lambda: PinJoint('J', P, C, joint_axis=C.x))

    # 创建 PinJoint 对象 pin，关节轴为 P.x + C.y，这是一个无效的组合
    raises(ValueError, lambda: PinJoint('J', P, C, joint_axis=P.x + C.y))

    # 创建 PinJoint 对象 pin，关节轴为 Pint.x + C.y，这也是一个无效的组合
    raises(ValueError, lambda: PinJoint(
        'J', P, C, parent_interframe=Pint, child_interframe=Cint,
        joint_axis=Pint.x + C.y))

    # 创建 PinJoint 对象 pin，关节轴为 P.x + Cint.y，这同样是一个无效的组合
    raises(ValueError, lambda: PinJoint(
        'J', P, C, parent_interframe=Pint, child_interframe=Cint,
        joint_axis=P.x + Cint.y))

    # 创建 PinJoint 对象 pin，关节轴为 Pint.x + P.y，这是一个有效的特殊组合
    N, A, P, C, Pint, Cint = _generate_body(True)
    PinJoint('J', P, C, parent_interframe=Pint, child_interframe=Cint,
             joint_axis=Pint.x + P.y)

    # 创建 PinJoint 对象 pin，关节轴为 Vector(0)，这是一个无效的零向量
    raises(Exception, lambda: PinJoint(
        'J', P, C, parent_interframe=Pint, child_interframe=Cint,
        joint_axis=Vector(0)))

    # 创建 PinJoint 对象 pin，关节轴为 P.y + Pint.y，这是一个无效的组合
    raises(Exception, lambda: PinJoint(
        'J', P, C, parent_interframe=Pint, child_interframe=Cint,
        joint_axis=P.y + Pint.y))


def test_pin_joint_arbitrary_axis():
    # 定义动力学符号 q_J 和 u_J
    q, u = dynamicsymbols('q_J, u_J')

    # 创建 PinJoint 对象 pin，关节轴为 -A.x，其中 A 是一个身体对象
    N, A, P, C = _generate_body()
    PinJoint('J', P, C, child_interframe=-A.x)

    # 断言 -A.x 与 N.x 之间的夹角为 0
    assert (-A.x).angle_between(N.x) == 0
    # 断言：验证负方向矢量 -A.x 在惯性系 N 中的表达是否等于矢量 N.x
    assert -A.x.express(N) == N.x
    
    # 断言：验证 A 相对于 N 的方向余弦矩阵是否为给定的旋转矩阵
    assert A.dcm(N) == Matrix([[-1, 0, 0],
                               [0, -cos(q), -sin(q)],
                               [0, -sin(q), cos(q)]])
    
    # 断言：验证 A 相对于 N 的角速度矢量是否为 u*N.x
    assert A.ang_vel_in(N) == u*N.x
    
    # 断言：验证 A 相对于 N 的角速度矢量大小是否为 sqrt(u**2)
    assert A.ang_vel_in(N).magnitude() == sqrt(u**2)
    
    # 断言：验证质心 C 相对于质心 P 的位置矢量在 N 中的表达是否为 0
    assert C.masscenter.pos_from(P.masscenter) == 0
    
    # 断言：验证质心 C 相对于质心 P 的位置矢量在 N 中的表达简化后是否为 0
    assert C.masscenter.pos_from(P.masscenter).express(N).simplify() == 0
    
    # 断言：验证质心 C 相对于惯性系 N 的速度矢量是否为 0
    assert C.masscenter.vel(N) == 0

    # 当轴线不同时，父关节在质心处，子关节在子质心单位矢量处时的情况
    N, A, P, C = _generate_body()
    PinJoint('J', P, C, child_interframe=A.y, child_point=A.x)

    # 断言：验证 A.y 和 N.x 之间的夹角是否为 0，即轴线是对齐的
    assert A.y.angle_between(N.x) == 0
    
    # 断言：验证 A.y 在 N 中的表达是否等于 N.x
    assert A.y.express(N) == N.x
    
    # 断言：验证 A 相对于 N 的方向余弦矩阵是否为给定的旋转矩阵
    assert A.dcm(N) == Matrix([[0, -cos(q), -sin(q)],
                               [1, 0, 0],
                               [0, -sin(q), cos(q)]])
    
    # 断言：验证 A 相对于 N 的角速度矢量是否为 u*N.x
    assert A.ang_vel_in(N) == u*N.x
    
    # 断言：验证 A 相对于 A 的角速度矢量在 A 中的表达是否为 u*A.y
    assert A.ang_vel_in(N).express(A) == u * A.y
    
    # 断言：验证 A 相对于 N 的角速度矢量大小是否为 sqrt(u**2)
    assert A.ang_vel_in(N).magnitude() == sqrt(u**2)
    
    # 断言：验证 A 相对于 N 的角速度矢量与 A.y 的叉乘是否为 0
    assert A.ang_vel_in(N).cross(A.y) == 0
    
    # 断言：验证质心 C 相对于惯性系 N 的速度矢量是否为 u*A.z
    assert C.masscenter.vel(N) == u*A.z
    
    # 断言：验证质心 C 相对于质心 P 的位置矢量是否为 -A.x
    assert C.masscenter.pos_from(P.masscenter) == -A.x
    
    # 断言：验证质心 C 相对于质心 P 的位置矢量在 N 中的表达简化后是否等于 cos(q)*N.y + sin(q)*N.z
    assert (C.masscenter.pos_from(P.masscenter).express(N).simplify() ==
            cos(q)*N.y + sin(q)*N.z)
    
    # 断言：验证质心 C 相对于惯性系 N 的速度矢量与 A.x 之间的夹角是否为 pi/2
    assert C.masscenter.vel(N).angle_between(A.x) == pi/2

    # 类似于前一个情况，但是相对于父体
    N, A, P, C = _generate_body()
    PinJoint('J', P, C, parent_interframe=N.y, parent_point=N.x)

    # 断言：验证 N.y 和 A.x 之间的夹角是否为 0，即轴线是对齐的
    assert N.y.angle_between(A.x) == 0
    
    # 断言：验证 N.y 在 A 中的表达是否等于 A.x
    assert N.y.express(A) == A.x
    
    # 断言：验证 A 相对于 N 的方向余弦矩阵是否为给定的旋转矩阵
    assert A.dcm(N) == Matrix([[0, 1, 0],
                               [-cos(q), 0, sin(q)],
                               [sin(q), 0, cos(q)]])
    
    # 断言：验证 A 相对于 N 的角速度矢量是否为 u*N.y
    assert A.ang_vel_in(N) == u*N.y
    
    # 断言：验证 A 相对于 A 的角速度矢量在 A 中的表达是否为 u*A.x
    assert A.ang_vel_in(N).express(A) == u*A.x
    
    # 断言：验证 A 相对于 N 的角速度矢量大小是否为 sqrt(u**2)
    assert A.ang_vel_in(N).magnitude() == sqrt(u**2)
    
    # angle 变量用于存储 A 相对于 N 的角速度矢量与 A.x 之间的夹角，替换其中的 u 为 1 后是否等于 0
    angle = A.ang_vel_in(N).angle_between(A.x)
    assert angle.xreplace({u: 1}) == 0
    
    # 断言：验证质心 C 相对于惯性系 N 的速度矢量是否为 0
    assert C.masscenter.vel(N) == 0
    
    # 断言：验证质心 C 相对于质心 P 的位置矢量是否为 N.x
    assert C.masscenter.pos_from(P.masscenter) == N.x

    # 父关节和子关节的位置都被定义，但是轴线不同的情况
    N, A, P, C = _generate_body()
    PinJoint('J', P, C, parent_point=N.x, child_point=A.x,
             child_interframe=A.x + A.y)
    
    # 断言：验证 N.x 和 A.x + A.y 之间的夹角是否为 0，即轴线是对齐的
    assert expand_mul(N.x.angle_between(A.x + A.y)) == 0
    
    # 断言：验证 A.x + A.y 在 N 中的表达是否简化为 sqrt(2)*N.x
    assert (A.x + A.y).express(N).simplify() == sqrt(2)*N.x
    
    # 断言：验证 A 相对于 N 的方向余弦矩阵是否为给定的旋转矩阵
    assert simplify(A.dcm(N)) == Matrix([
        [sqrt(2)/2, -sqrt(2)*cos(q)/2, -sqrt(2)*sin(q)/2],
        [sqrt(2)/2, sqrt(2)*cos(q)/2, sqrt(2)*sin(q)/2],
        [0, -sin(q), cos(q)]])
    
    # 断言：验证 A 相对于 N 的角速度矢量是否为 u*N.x
    assert A.ang_vel_in(N) == u*N.x
    
    # 断言：验证 A 相对于 A 的角速度矢量在 A 中的表达是否为 (u*A.x + u*A.y)/sqrt(2)
    assert (A.ang_vel_in(N).express(A).simplify() ==
            (u*A.x + u*A.y)/sqrt(2))
    
    # 断言：验证 A 相对于 N 的角速度矢量大小是否为 sqrt(u**2)
    assert A.ang_vel_in(N).magnitude() == sqrt(u**2)
    
    # angle 变量用于存储 A 相对于 N 的角速度
    # 第一个断言：验证质心C相对于质心P的位置向量在惯性参考系N中的表达式是否简化为给定的表达式
    assert (C.masscenter.pos_from(P.masscenter).express(N).simplify() ==
            (1 - sqrt(2)/2)*N.x + sqrt(2)*cos(q)/2*N.y +
            sqrt(2)*sin(q)/2*N.z)

    # 第二个断言：验证质心C相对于惯性参考系N的速度在N中的表达式是否简化为给定的表达式
    assert (C.masscenter.vel(N).express(N).simplify() ==
            -sqrt(2)*u*sin(q)/2*N.y + sqrt(2)*u*cos(q)/2*N.z)

    # 第三个断言：验证质心C相对于惯性参考系N的速度与A.x之间的夹角是否为pi/2
    assert C.masscenter.vel(N).angle_between(A.x) == pi/2

    # 调用_generate_body函数，返回四个对象N, A, P, C
    N, A, P, C = _generate_body()

    # 在点P和点C之间建立PinJoint，父点为N.x，子点为A.x，
    # 子框架为A.x + A.y - A.z
    PinJoint('J', P, C, parent_point=N.x, child_point=A.x,
             child_interframe=A.x + A.y - A.z)

    # 第四个断言：验证N.x和A.x + A.y - A.z之间的夹角是否展开乘法后为0，即轴对齐
    assert expand_mul(N.x.angle_between(A.x + A.y - A.z)) == 0  # Axis aligned

    # 第五个断言：验证A.x + A.y - A.z在惯性参考系N中的表达式是否简化为给定的表达式
    assert (A.x + A.y - A.z).express(N).simplify() == sqrt(3)*N.x

    # 第六个断言：验证A相对于N的方向余弦矩阵是否简化为给定的3x3矩阵
    assert simplify(A.dcm(N)) == Matrix([
        [sqrt(3)/3, -sqrt(6)*sin(q + pi/4)/3,
         sqrt(6)*cos(q + pi/4)/3],
        [sqrt(3)/3, sqrt(6)*cos(q + pi/12)/3,
         sqrt(6)*sin(q + pi/12)/3],
        [-sqrt(3)/3, sqrt(6)*cos(q + 5*pi/12)/3,
         sqrt(6)*sin(q + 5*pi/12)/3]])

    # 第七个断言：验证A相对于N的角速度是否为u*N.x
    assert A.ang_vel_in(N) == u*N.x

    # 第八个断言：验证A相对于N的角速度在A系中的表达式是否简化为给定的表达式
    assert A.ang_vel_in(N).express(A).simplify() == (u*A.x + u*A.y -
                                                     u*A.z)/sqrt(3)

    # 第九个断言：验证A相对于N的角速度的大小是否为sqrt(u**2)
    assert A.ang_vel_in(N).magnitude() == sqrt(u**2)

    # 计算A相对于N的角速度与A.x + A.y - A.z之间的夹角
    angle = A.ang_vel_in(N).angle_between(A.x + A.y - A.z)

    # 第十个断言：将u替换为1后，验证角度angle是否简化为0
    assert angle.xreplace({u: 1}).simplify() == 0

    # 第十一个断言：验证质心C相对于N的速度是否简化为给定的表达式
    assert C.masscenter.vel(N).simplify() == (u*A.y + u*A.z)/sqrt(3)

    # 第十二个断言：验证质心C相对于质心P的位置向量是否等于N.x - A.x
    assert C.masscenter.pos_from(P.masscenter) == N.x - A.x

    # 第十三个断言：验证质心C相对于质心P的位置向量在惯性参考系N中的表达式是否简化为给定的表达式
    assert (C.masscenter.pos_from(P.masscenter).express(N).simplify() ==
            (1 - sqrt(3)/3)*N.x + sqrt(6)*sin(q + pi/4)/3*N.y -
            sqrt(6)*cos(q + pi/4)/3*N.z)

    # 第十四个断言：验证质心C相对于N的速度在N中的表达式是否简化为给定的表达式
    assert (C.masscenter.vel(N).express(N).simplify() ==
            sqrt(6)*u*cos(q + pi/4)/3*N.y +
            sqrt(6)*u*sin(q + pi/4)/3*N.z)

    # 第十五个断言：验证质心C相对于N的速度与A.x之间的夹角是否为pi/2
    assert C.masscenter.vel(N).angle_between(A.x) == pi/2

    # 调用_generate_body函数，返回四个对象N, A, P, C
    N, A, P, C = _generate_body()

    # 声明符号变量m, n
    m, n = symbols('m n')

    # 在点P和点C之间建立PinJoint，父点为m * N.x，子点为n * A.x，
    # 子框架为A.x + A.y - A.z，父框架为N.x - N.y + N.z
    PinJoint('J', P, C, parent_point=m * N.x, child_point=n * A.x,
             child_interframe=A.x + A.y - A.z,
             parent_interframe=N.x - N.y + N.z)

    # 计算N.x - N.y + N.z和A.x + A.y - A.z之间的夹角
    angle = (N.x - N.y + N.z).angle_between(A.x + A.y - A.z)

    # 第十六个断言：验证N.x - N.y + N.z与A.x + A.y - A.z之间的夹角是否展开乘法后为0，即轴对齐
    assert expand_mul(angle) == 0  # Axis are aligned

    # 第十七个断言：验证A.x - A.y + A.z在惯性参考系N中的表达式是否简化为给定的表达式
    assert ((A.x-A.y+A.z).express(N).simplify() ==
            (-4*cos(q)/3 - S(1)/3)*N.x + (S(1)/3 - 4*sin(q + pi/6)/3)*N.y +
            (4*cos(q + pi/3)/3 - S(1)/3)*N.z)

    # 第十八个断言：验证A相对于N的方向余弦矩阵是否简化为给定的3x3矩阵
    assert simplify(A.dcm(N)) == Matrix([
        [S(1)/3 - 2*cos(q)/3, -2*sin(q + pi/6)/3 - S(1)/3,
         2*cos(q + pi/3)/3 + S(1)/3],
        [2*cos(q + pi/3)/3 + S(1)/3, 2*cos(q)/3 - S(1)/3,
         2*sin(q + pi/6)/3 + S(1)/3],
        [-2*sin(q + pi/6)/3 - S(1)/3, 2*cos(q + pi/3)/3 + S(1)/3,
         2*cos(q)/3 - S(1)/3]])

    # 第十九个断言：验证A相对于N的角速度是否简化为给定的表达式
    assert (A.ang_vel_in(N) - (u*N.x - u*N.y + u*N.z)/sqrt(3)).simplify()

    # 第二十个断言：验证A相对于N的角速度在A系中的表达式是否简化为给定的表达式
    assert A.ang_vel_in(N).express(A).simplify() == (u*A.x + u*A.y -
                                                     u*A.z)/sqrt(3)

    # 第二十一个断言：验证A相对于N的角速度
    # 断言1：验证质心相对于惯性系N的速度是否满足特定的简化条件
    assert (C.masscenter.vel(N).simplify() ==
            sqrt(3)*n*u/3*A.y + sqrt(3)*n*u/3*A.z)
    
    # 断言2：验证质心相对于P质心的位置向量是否等于给定的表达式
    assert C.masscenter.pos_from(P.masscenter) == m*N.x - n*A.x
    
    # 断言3：验证质心相对于P质心的位置向量在转换到惯性系N后是否满足特定的简化条件
    assert (C.masscenter.pos_from(P.masscenter).express(N).simplify() ==
            (m + n*(2*cos(q) - 1)/3)*N.x + n*(2*sin(q + pi/6) + 1)/3*N.y - n*(2*cos(q + pi/3) + 1)/3*N.z)
    
    # 断言4：验证质心相对于惯性系N的速度在惯性系N中的表示是否满足特定的简化条件
    assert (C.masscenter.vel(N).express(N).simplify() ==
            - 2*n*u*sin(q)/3*N.x + 2*n*u*cos(q + pi/6)/3*N.y + 2*n*u*sin(q + pi/3)/3*N.z)
    
    # 断言5：验证质心相对于惯性系N的速度与单位向量(N.x - N.y + N.z)的点积是否为零
    assert C.masscenter.vel(N).dot(N.x - N.y + N.z).simplify() == 0
def test_create_aligned_frame_pi():
    # 生成身体模型的各个部分的位置和参数
    N, A, P, C = _generate_body()
    # 创建一个与给定向量 P 对齐的插值帧 f，其 x 方向与 -P.x 对齐
    f = Joint._create_aligned_interframe(P, -P.x, P.x)
    # 断言 f 的 z 分量与 P 的 z 分量相等
    assert f.z == P.z
    # 创建一个与给定向量 P 对齐的插值帧 f，其 y 方向与 -P.y 对齐
    f = Joint._create_aligned_interframe(P, -P.y, P.y)
    # 断言 f 的 x 分量与 P 的 x 分量相等
    assert f.x == P.x
    # 创建一个与给定向量 P 对齐的插值帧 f，其 z 方向与 -P.z 对齐
    f = Joint._create_aligned_interframe(P, -P.z, P.z)
    # 断言 f 的 y 分量与 P 的 y 分量相等
    assert f.y == P.y
    # 创建一个与给定向量 P 对齐的插值帧 f，其 x 和 y 方向与 -P.x - P.y 对齐
    f = Joint._create_aligned_interframe(P, -P.x - P.y, P.x + P.y)
    # 断言 f 的 z 分量与 P 的 z 分量相等
    assert f.z == P.z
    # 创建一个与给定向量 P 对齐的插值帧 f，其 y 和 z 方向与 -P.y - P.z 对齐
    f = Joint._create_aligned_interframe(P, -P.y - P.z, P.y + P.z)
    # 断言 f 的 x 分量与 P 的 x 分量相等
    assert f.x == P.x
    # 创建一个与给定向量 P 对齐的插值帧 f，其 x 和 z 方向与 -P.x - P.z 对齐
    f = Joint._create_aligned_interframe(P, -P.x - P.z, P.x + P.z)
    # 断言 f 的 y 分量与 P 的 y 分量相等
    assert f.y == P.y
    # 创建一个与给定向量 P 对齐的插值帧 f，其 x、y 和 z 方向与 -P.x - P.y - P.z 对齐
    f = Joint._create_aligned_interframe(P, -P.x - P.y - P.z, P.x + P.y + P.z)
    # 断言 f 的 y 分量减去 f 的 z 分量等于 P 的 y 分量减去 P 的 z 分量
    assert f.y - f.z == P.y - P.z


def test_pin_joint_axis():
    q, u = dynamicsymbols('q u')
    # 测试默认的关节轴
    N, A, P, C, Pint, Cint = _generate_body(True)
    J = PinJoint('J', P, C, q, u, parent_interframe=Pint, child_interframe=Cint)
    # 断言关节 J 的关节轴与父参考帧 Pint 的 x 轴相等
    assert J.joint_axis == Pint.x
    # 测试表达在不同参考帧中相同的关节轴
    N_R_A = Matrix([[0, sin(q), cos(q)],
                    [0, -cos(q), sin(q)],
                    [1, 0, 0]])
    N, A, P, C, Pint, Cint = _generate_body(True)
    PinJoint('J', P, C, q, u, parent_interframe=Pint, child_interframe=Cint,
             joint_axis=N.z)
    # 断言 N 相对于 A 的方向余弦矩阵与 N_R_A 相等
    assert N.dcm(A) == N_R_A
    N, A, P, C, Pint, Cint = _generate_body(True)
    PinJoint('J', P, C, q, u, parent_interframe=Pint, child_interframe=Cint,
             joint_axis=-Pint.z)
    # 断言 N 相对于 A 的方向余弦矩阵与 N_R_A 相等
    assert N.dcm(A) == N_R_A
    # 测试时间变化的关节轴
    N, A, P, C, Pint, Cint = _generate_body(True)
    raises(ValueError, lambda: PinJoint('J', P, C, joint_axis=q * N.z))


def test_locate_joint_pos():
    # 测试向量和默认位置
    N, A, P, C = _generate_body()
    # 创建一个 PinJoint 关节，其父点为 N.y + N.z
    joint = PinJoint('J', P, C, parent_point=N.y + N.z)
    # 断言 joint 的父点名称为 'J_P_joint'
    assert joint.parent_point.name == 'J_P_joint'
    # 断言 joint 的父点相对于 P 的质心的位置为 N.y + N.z
    assert joint.parent_point.pos_from(P.masscenter) == N.y + N.z
    # 断言 joint 的子点为 C 的质心
    assert joint.child_point == C.masscenter
    # 测试 Point 对象
    N, A, P, C = _generate_body()
    parent_point = P.masscenter.locatenew('p', N.y + N.z)
    # 创建一个 PinJoint 关节，其父点为 parent_point，子点为 C 的质心
    joint = PinJoint('J', P, C, parent_point=parent_point,
                     child_point=C.masscenter)
    # 断言 joint 的父点为 parent_point
    assert joint.parent_point == parent_point
    # 断言 joint 的子点为 C 的质心
    assert joint.child_point == C.masscenter
    # 检查无效的类型
    N, A, P, C = _generate_body()
    raises(TypeError,
           lambda: PinJoint('J', P, C, parent_point=N.x.to_matrix(N)))
    # 测试时间变化的位置
    q = dynamicsymbols('q')
    N, A, P, C = _generate_body()
    raises(ValueError, lambda: PinJoint('J', P, C, parent_point=q * N.x))
    N, A, P, C = _generate_body()
    child_point = C.masscenter.locatenew('p', q * A.y)
    raises(ValueError, lambda: PinJoint('J', P, C, child_point=child_point))
    # 测试未定义的位置
    child_point = Point('p')
    raises(ValueError, lambda: PinJoint('J', P, C, child_point=child_point))
    # 生成身体的标准参考系和位置
    N, A, P, C = _generate_body()
    
    # 创建一个名为'int_frame'的参考系对象作为父参考系中间帧
    parent_interframe = ReferenceFrame('int_frame')
    
    # 在父参考系中的'int_frame'上绕轴N的z轴方向旋转1个单位
    parent_interframe.orient_axis(N, N.z, 1)
    
    # 创建一个固定关节，命名为'J'，连接P和C，并将父参考系中间帧设置为parent_interframe
    joint = PinJoint('J', P, C, parent_interframe=parent_interframe)
    
    # 断言关节的父参考系中间帧确实是设置的parent_interframe
    assert joint.parent_interframe == parent_interframe
    
    # 断言关节的父参考系中间帧在N参考系中的角速度为0
    assert joint.parent_interframe.ang_vel_in(N) == 0
    
    # 断言关节的子参考系中间帧是A
    assert joint.child_interframe == A
    
    # 测试时间变化的方向
    q = dynamicsymbols('q')
    N, A, P, C = _generate_body()
    
    # 创建一个名为'int_frame'的参考系对象作为父参考系中间帧
    parent_interframe = ReferenceFrame('int_frame')
    
    # 在父参考系中的'int_frame'上绕轴N的z轴方向旋转q个单位（动态符号）
    # 由于q是一个动态符号，所以这里会引发一个值错误(ValueError)
    raises(ValueError,
           lambda: PinJoint('J', P, C, parent_interframe=parent_interframe))
    
    # 测试未定义的参考系
    N, A, P, C = _generate_body()
    
    # 创建一个名为'int_frame'的参考系对象作为子参考系中间帧
    child_interframe = ReferenceFrame('int_frame')
    
    # 在子参考系中的'int_frame'上绕轴N的z轴方向旋转1个单位，但是未定义父参考系
    # 这里会引发一个值错误(ValueError)
    raises(ValueError,
           lambda: PinJoint('J', P, C, child_interframe=child_interframe))
# 定义测试函数，测试平移关节的功能
def test_prismatic_joint():
    # 生成机体上的四个点并分配给变量
    _, _, P, C = _generate_body()
    # 定义动力学符号 q_S 和 u_S
    q, u = dynamicsymbols('q_S, u_S')
    # 创建平移关节对象 S，连接 P 和 C
    S = PrismaticJoint('S', P, C)
    # 断言关节对象的名称为 'S'
    assert S.name == 'S'
    # 断言关节对象的父级为 P
    assert S.parent == P
    # 断言关节对象的子级为 C
    assert S.child == C
    # 断言关节对象的坐标为 q 的矩阵形式
    assert S.coordinates == Matrix([q])
    # 断言关节对象的速度为 u 的矩阵形式
    assert S.speeds == Matrix([u])
    # 断言关节对象的动力学方程 kdes 为 u - q 的矩阵形式
    assert S.kdes == Matrix([u - q.diff(t)])
    # 断言关节的旋转轴为 P 参考系的 x 轴
    assert S.joint_axis == P.frame.x
    # 断言子级点相对于质心的位置为零向量
    assert S.child_point.pos_from(C.masscenter) == Vector(0)
    # 断言父级点相对于质心的位置为零向量
    assert S.parent_point.pos_from(P.masscenter) == Vector(0)
    # 断言父级点相对于子级点的位置为 -q * P 参考系的 x 轴
    assert S.parent_point.pos_from(S.child_point) == - q * P.frame.x
    # 断言 P 质心相对于 C 质心的位置为 -q * P 参考系的 x 轴
    assert P.masscenter.pos_from(C.masscenter) == - q * P.frame.x
    # 断言 C 质心相对于 P 参考系的速度为 u * P 参考系的 x 轴
    assert C.masscenter.vel(P.frame) == u * P.frame.x
    # 断言 P 参考系相对于 C 参考系的角速度为零
    assert P.frame.ang_vel_in(C.frame) == 0
    # 断言 C 参考系相对于 P 参考系的角速度为零
    assert C.frame.ang_vel_in(P.frame) == 0
    # 断言关节对象的字符串形式为 'PrismaticJoint: S  parent: P  child: C'

    # 生成机体上的四个点并分配给变量
    N, A, P, C = _generate_body()
    # 定义符号 l 和 m
    l, m = symbols('l m')
    # 创建新的参考系 P_int
    Pint = ReferenceFrame('P_int')
    # 将 P_int 参考系沿着 P 参考系的 y 轴旋转 pi/2 弧度
    Pint.orient_axis(P.frame, P.y, pi / 2)
    # 创建平移关节对象 S，连接 P 和 C，设置父级点和子级点的位置，关节轴为 P 参考系的 z 轴，父级参考系为 P_int
    S = PrismaticJoint('S', P, C, parent_point=l * P.frame.x,
                       child_point=m * C.frame.y, joint_axis=P.frame.z,
                       parent_interframe=Pint)

    # 断言关节对象的旋转轴为 P 参考系的 z 轴
    assert S.joint_axis == P.frame.z
    # 断言子级点相对于质心的位置为 m * C 参考系的 y 轴
    assert S.child_point.pos_from(C.masscenter) == m * C.frame.y
    # 断言父级点相对于质心的位置为 l * P 参考系的 x 轴
    assert S.parent_point.pos_from(P.masscenter) == l * P.frame.x
    # 断言父级点相对于子级点的位置为 -q * P 参考系的 z 轴
    assert S.parent_point.pos_from(S.child_point) == - q * P.frame.z
    # 断言 P 质心相对于 C 质心的位置为 -l * N 参考系的 x 轴 - q * N 参考系的 z 轴 + m * A 参考系的 y 轴
    assert P.masscenter.pos_from(C.masscenter) == - l * N.x - q * N.z + m * A.y
    # 断言 C 质心相对于 P 参考系的速度为 u * P 参考系的 z 轴
    assert C.masscenter.vel(P.frame) == u * P.frame.z
    # 断言 P 质心相对于 P_int 参考系的速度为零向量
    assert P.masscenter.vel(Pint) == Vector(0)
    # 断言 C 参考系相对于 P 参考系的角速度为零
    assert C.frame.ang_vel_in(P.frame) == 0
    # 断言 P 参考系相对于 C 参考系的角速度为零
    assert P.frame.ang_vel_in(C.frame) == 0

    # 生成机体上的四个点并分配给变量
    _, _, P, C = _generate_body()
    # 创建新的参考系 P_int
    Pint = ReferenceFrame('P_int')
    # 将 P_int 参考系沿着 P 参考系的 y 轴旋转 pi/2 弧度
    Pint.orient_axis(P.frame, P.y, pi / 2)
    # 创建平移关节对象 S，连接 P 和 C，设置父级点和子级点的位置，关节轴为 P 参考系的 z 轴，父级参考系为 P_int
    S = PrismaticJoint('S', P, C, parent_point=l * P.frame.z,
                       child_point=m * C.frame.x, joint_axis=P.frame.z,
                       parent_interframe=Pint)
    # 断言关节对象的旋转轴为 P 参考系的 z 轴
    assert S.joint_axis == P.frame.z
    # 断言子级点相对于质心的位置为 m * C 参考系的 x 轴
    assert S.child_point.pos_from(C.masscenter) == m * C.frame.x
    # 断言父级点相对于质心的位置为 l * P 参考系的 z 轴
    assert S.parent_point.pos_from(P.masscenter) == l * P.frame.z
    # 断言父级点相对于子级点的位置为 -q * P 参考系的 z 轴
    assert S.parent_point.pos_from(S.child_point) == - q * P.frame.z
    # 断言 P 质心相对于 C 质心的位置为 (-l - q) * P 参考系的 z 轴 + m * C 参考系的 x 轴
    assert P.masscenter.pos_from(C.masscenter) == (-l - q) * P.frame.z + m * C.frame.x
    # 断言 C 质心相对于 P 参考系的速度为 u * P 参考系的 z 轴
    assert C.masscenter.vel(P.frame) == u * P.frame.z
    # 断言 C 参考系相对于 P 参考系的角速度为零
    assert C.frame.ang_vel_in(P.frame) == 0
    # 断言 P 参考系相对于 C 参考系的角速度为零


# 测试带任意轴的平移关节的函数
def test_prismatic_joint_arbitrary_axis():
    # 定义动力学符号 q_S 和 u_S
    q, u = dynamicsymbols('q_S, u_S')

    # 生成机体上的四个点并分配给变量
    N, A, P, C = _generate_body()
    # 创建带有任意轴的平移关节对象 S，连接 P 和 C，子级参考系为 -A 参考系的 x 轴
    PrismaticJoint('S', P
    # 断言角速度为零
    assert N.ang_vel_in(A) == 0

    # 当轴线不同，并且父关节在质心，子关节在子质心单位向量时
    N, A, P, C = _generate_body()
    PrismaticJoint('S', P, C, child_interframe=A.y, child_point=A.x)

    # 断言 A.y 和 N.x 的夹角为零，表示轴线对齐
    assert A.y.angle_between(N.x) == 0
    # 断言 A.y 在 N 坐标系中的表示等于 N.x
    assert A.y.express(N) == N.x
    # 断言 A 关于 N 的方向余弦矩阵
    assert A.dcm(N) == Matrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    # 断言 C 的质心在 N 坐标系中的速度等于 u * N.x
    assert C.masscenter.vel(N) == u * N.x
    # 断言 C 的质心速度在 A 坐标系中的表示等于 u * A.y
    assert C.masscenter.vel(N).express(A) == u * A.y
    # 断言 C 的质心位置相对于 P 的质心位置等于 q*N.x - A.x
    assert C.masscenter.pos_from(P.masscenter) == q*N.x - A.x
    # 断言 C 的质心位置相对于 P 的质心位置在 N 坐标系中简化后等于 q*N.x + N.y
    assert C.masscenter.pos_from(P.masscenter).express(N).simplify() == q*N.x + N.y
    # 断言 A 关于 N 的角速度为零
    assert A.ang_vel_in(N) == 0
    # 断言 N 关于 A 的角速度为零
    assert N.ang_vel_in(A) == 0

    # 类似于前一个情况，但是相对于父体
    N, A, P, C = _generate_body()
    PrismaticJoint('S', P, C, parent_interframe=N.y, parent_point=N.x)

    # 断言 N.y 和 A.x 的夹角为零，表示轴线对齐
    assert N.y.angle_between(A.x) == 0
    # 断言 N.y 在 A 坐标系中的表示等于 A.x
    assert N.y.express(A) == A.x
    # 断言 A 关于 N 的方向余弦矩阵
    assert A.dcm(N) == Matrix([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    # 断言 C 的质心在 N 坐标系中的速度等于 u * N.y
    assert C.masscenter.vel(N) == u * N.y
    # 断言 C 的质心速度在 A 坐标系中的表示等于 u * A.x
    assert C.masscenter.vel(N).express(A) == u * A.x
    # 断言 C 的质心位置相对于 P 的质心位置等于 N.x + q*N.y
    assert C.masscenter.pos_from(P.masscenter) == N.x + q*N.y
    # 断言 A 关于 N 的角速度为零
    assert A.ang_vel_in(N) == 0
    # 断言 N 关于 A 的角速度为零
    assert N.ang_vel_in(A) == 0

    # 两个关节位置都定义，但轴线不同
    N, A, P, C = _generate_body()
    PrismaticJoint('S', P, C, parent_point=N.x, child_point=A.x,
                   child_interframe=A.x + A.y)

    # 断言 N.x 和 A.x + A.y 的夹角为零，表示轴线对齐
    assert N.x.angle_between(A.x + A.y) == 0
    # 断言 (A.x + A.y) 在 N 坐标系中的表示等于 sqrt(2)*N.x
    assert (A.x + A.y).express(N) == sqrt(2)*N.x
    # 断言 A 关于 N 的方向余弦矩阵
    assert A.dcm(N) == Matrix([[sqrt(2)/2, -sqrt(2)/2, 0], [sqrt(2)/2, sqrt(2)/2, 0], [0, 0, 1]])
    # 断言 C 的质心位置相对于 P 的质心位置等于 (q + 1)*N.x - A.x
    assert C.masscenter.pos_from(P.masscenter) == (q + 1)*N.x - A.x
    # 断言 C 的质心位置相对于 P 的质心位置在 N 坐标系中的表示等于 (q - sqrt(2)/2 + 1)*N.x + sqrt(2)/2*N.y
    assert C.masscenter.pos_from(P.masscenter).express(N) == (q - sqrt(2)/2 + 1)*N.x + sqrt(2)/2*N.y
    # 断言 C 的质心速度在 N 坐标系中的表示等于 u * (A.x + A.y)/sqrt(2)
    assert C.masscenter.vel(N).express(A) == u * (A.x + A.y)/sqrt(2)
    # 断言 C 的质心速度在 N 坐标系中等于 u*N.x
    assert C.masscenter.vel(N) == u*N.x
    # 断言 A 关于 N 的角速度为零
    assert A.ang_vel_in(N) == 0
    # 断言 N 关于 A 的角速度为零
    assert N.ang_vel_in(A) == 0

    N, A, P, C = _generate_body()
    PrismaticJoint('S', P, C, parent_point=N.x, child_point=A.x,
                   child_interframe=A.x + A.y - A.z)

    # 断言 N.x 和 A.x + A.y - A.z 的夹角简化后为零，表示轴线对齐
    assert N.x.angle_between(A.x + A.y - A.z).simplify() == 0
    # 断言 ((A.x + A.y - A.z).express(N) - sqrt(3)*N.x).simplify() 等于零
    assert ((A.x + A.y - A.z).express(N) - sqrt(3)*N.x).simplify() == 0
    # 断言 A 关于 N 的方向余弦矩阵简化后等于特定矩阵
    assert simplify(A.dcm(N)) == Matrix([[sqrt(3)/3, -sqrt(3)/3, sqrt(3)/3],
                                         [sqrt(3)/3, sqrt(3)/6 + S(1)/2, S(1)/2 - sqrt(3)/6],
                                         [-sqrt(3)/3, S(1)/2 - sqrt(3)/6, sqrt(3)/6 + S(1)/2]])
    # 断言 C 的质心位置相对于 P 的质心位置等于 (q + 1)*N.x - A.x
    assert C.masscenter.pos_from(P.masscenter) == (q + 1)*N.x - A.x
    # 断言 (C.masscenter.pos_from(P.masscenter).express(N) - ((q - sqrt(3)/3 + 1)*N.x + sqrt(3)/3*N.y - sqrt(3)/3*N.z)).simplify() 等于零
    assert (C.masscenter.pos_from(P.masscenter).express(N) -
            ((q - sqrt(3)/3 + 1)*N.x + sqrt(3)/3*N.y - sqrt(3)/3*N.z)).simplify() == 0
    # 断言 C 的质心速度在 N 坐标系中等于 u*N.x
    assert C.masscenter.vel(N) == u*N.x
    # 断言 (C.masscenter.vel(N).express(A) - (sqrt(3)*u/3*A.x + sqrt(3)*u/3*A.y - sqrt(3)*u/3*A.z)).simplify() 等于零
    assert (C.masscenter.vel(N).express(A) -
            (sqrt(3)*u/3*A.x + sqrt(3)*u/3*A.y - sqrt(3)*u/3*A.z)).simplify() == 0
    # 断言刚体A在参考系N中的角速度为0
    assert A.ang_vel_in(N) == 0
    # 断言参考系N在刚体A中的角速度为0
    assert N.ang_vel_in(A) == 0

    # 调用_generate_body函数生成四个实体N, A, P, C
    N, A, P, C = _generate_body()
    # 定义符号m和n
    m, n = symbols('m n')
    # 创建一个平移关节PrismaticJoint，连接点S，连接参考系P和C，父点为m*N.x，子点为n*A.x，
    # 子参考系间隔为A.x + A.y - A.z，父参考系间隔为N.x - N.y + N.z
    PrismaticJoint('S', P, C, parent_point=m*N.x, child_point=n*A.x,
                   child_interframe=A.x + A.y - A.z,
                   parent_interframe=N.x - N.y + N.z)
    # 断言父参考系间隔与子参考系间隔的角度之间的简化值为0
    assert (N.x-N.y+N.z).angle_between(A.x+A.y-A.z).simplify() == 0
    # 断言子参考系间隔表达式相对于父参考系的表达式减去父参考系间隔的简化值为0
    assert ((A.x+A.y-A.z).express(N) - (N.x - N.y + N.z)).simplify() == 0
    # 断言刚体A相对于参考系N的方向余弦矩阵简化值为给定的矩阵
    assert simplify(A.dcm(N)) == Matrix([[-S(1)/3, -S(2)/3, S(2)/3],
                                         [S(2)/3, S(1)/3, S(2)/3],
                                         [-S(2)/3, S(2)/3, S(1)/3]])
    # 断言质心C相对于质心P的位置矢量表达式相对于参考系N的表达式减去给定的表达式的简化值为0
    assert (C.masscenter.pos_from(P.masscenter) - (
        (m + sqrt(3)*q/3)*N.x - sqrt(3)*q/3*N.y + sqrt(3)*q/3*N.z - n*A.x)
            ).express(N).simplify() == 0
    # 断言质心C相对于质心P的位置矢量表达式相对于参考系N的表达式减去给定的表达式的简化值为0
    assert (C.masscenter.pos_from(P.masscenter).express(N) - (
        (m + n/3 + sqrt(3)*q/3)*N.x + (2*n/3 - sqrt(3)*q/3)*N.y +
        (-2*n/3 + sqrt(3)*q/3)*N.z)).simplify() == 0
    # 断言质心C相对于参考系N的速度矢量表达式相对于参考系N的表达式减去给定的表达式的简化值为0
    assert (C.masscenter.vel(N).express(N) - (
        sqrt(3)*u/3*N.x - sqrt(3)*u/3*N.y + sqrt(3)*u/3*N.z)).simplify() == 0
    # 断言质心C相对于参考系N的速度矢量表达式相对于参考系A的表达式减去给定的表达式的简化值为0
    assert (C.masscenter.vel(N).express(A) -
            (sqrt(3)*u/3*A.x + sqrt(3)*u/3*A.y - sqrt(3)*u/3*A.z)).simplify() == 0
    # 断言刚体A在参考系N中的角速度为0
    assert A.ang_vel_in(N) == 0
    # 断言参考系N在刚体A中的角速度为0
    assert N.ang_vel_in(A) == 0
# 定义测试函数，测试圆柱形关节的功能
def test_cylindrical_joint():
    # 生成身体的相关符号和位置
    N, A, P, C = _generate_body()
    # 定义动力学符号
    q0_def, q1_def, u0_def, u1_def = dynamicsymbols('q0:2_J, u0:2_J')
    # 创建圆柱形关节对象
    Cj = CylindricalJoint('J', P, C)
    # 断言关节对象的属性
    assert Cj.name == 'J'
    assert Cj.parent == P
    assert Cj.child == C
    assert Cj.coordinates == Matrix([q0_def, q1_def])
    assert Cj.speeds == Matrix([u0_def, u1_def])
    assert Cj.rotation_coordinate == q0_def
    assert Cj.translation_coordinate == q1_def
    assert Cj.rotation_speed == u0_def
    assert Cj.translation_speed == u1_def
    # 计算关节的速度约束
    assert Cj.kdes == Matrix([u0_def - q0_def.diff(t), u1_def - q1_def.diff(t)])
    # 断言关节轴的方向
    assert Cj.joint_axis == N.x
    # 断言关节连接点的位置关系
    assert Cj.child_point.pos_from(C.masscenter) == Vector(0)
    assert Cj.parent_point.pos_from(P.masscenter) == Vector(0)
    assert Cj.parent_point.pos_from(Cj._child_point) == -q1_def * N.x
    # 断言质心之间的位置关系
    assert C.masscenter.pos_from(P.masscenter) == q1_def * N.x
    # 断言关节连接点的速度
    assert Cj.child_point.vel(N) == u1_def * N.x
    assert A.ang_vel_in(N) == u0_def * N.x
    # 断言关节的参考框架
    assert Cj.parent_interframe == N
    assert Cj.child_interframe == A
    # 断言关节对象的字符串表示
    assert Cj.__str__() == 'CylindricalJoint: J  parent: P  child: C'

    # 更改动力学符号的定义
    q0, q1, u0, u1 = dynamicsymbols('q0:2, u0:2')
    # 定义额外的符号
    l, m = symbols('l, m')
    # 生成包含附加点和参考框架的身体
    N, A, P, C, Pint, Cint = _generate_body(True)
    # 创建具有自定义参数的圆柱形关节对象
    Cj = CylindricalJoint('J', P, C, rotation_coordinate=q0, rotation_speed=u0,
                          translation_speed=u1, parent_point=m * N.x,
                          child_point=l * A.y, parent_interframe=Pint,
                          child_interframe=Cint, joint_axis=2 * N.z)
    # 断言关节对象的属性
    assert Cj.coordinates == Matrix([q0, q1_def])
    assert Cj.speeds == Matrix([u0, u1])
    assert Cj.rotation_coordinate == q0
    assert Cj.translation_coordinate == q1_def
    assert Cj.rotation_speed == u0
    assert Cj.translation_speed == u1
    # 计算关节的速度约束
    assert Cj.kdes == Matrix([u0 - q0.diff(t), u1 - q1_def.diff(t)])
    # 断言关节轴的方向
    assert Cj.joint_axis == 2 * N.z
    # 断言关节连接点的位置关系
    assert Cj.child_point.pos_from(C.masscenter) == l * A.y
    assert Cj.parent_point.pos_from(P.masscenter) == m * N.x
    assert Cj.parent_point.pos_from(Cj._child_point) == -q1_def * N.z
    # 断言质心之间的位置关系
    assert C.masscenter.pos_from(
        P.masscenter) == m * N.x + q1_def * N.z - l * A.y
    # 断言质心的速度
    assert C.masscenter.vel(N) == u1 * N.z - u0 * l * A.z
    assert A.ang_vel_in(N) == u0 * N.z


# 定义测试函数，测试平面关节的功能
def test_planar_joint():
    # 生成身体的相关符号和位置
    N, A, P, C = _generate_body()
    # 定义动力学符号
    q0_def, q1_def, q2_def = dynamicsymbols('q0:3_J')
    u0_def, u1_def, u2_def = dynamicsymbols('u0:3_J')
    # 创建平面关节对象
    Cj = PlanarJoint('J', P, C)
    # 断言关节对象的属性
    assert Cj.name == 'J'
    assert Cj.parent == P
    assert Cj.child == C
    assert Cj.coordinates == Matrix([q0_def, q1_def, q2_def])
    assert Cj.speeds == Matrix([u0_def, u1_def, u2_def])
    assert Cj.rotation_coordinate == q0_def
    assert Cj.planar_coordinates == Matrix([q1_def, q2_def])
    assert Cj.rotation_speed == u0_def
    assert Cj.planar_speeds == Matrix([u1_def, u2_def])
    # 确认约束条件是否成立，即 Cj.kdes 是否等于 Matrix([u0_def - q0_def.diff(t), u1_def - q1_def.diff(t), u2_def - q2_def.diff(t)])
    assert Cj.kdes == Matrix([u0_def - q0_def.diff(t), u1_def - q1_def.diff(t), u2_def - q2_def.diff(t)])

    # 确认 Cj 的旋转轴是否为 N.x
    assert Cj.rotation_axis == N.x

    # 确认 Cj 的平面向量列表是否为 [N.y, N.z]
    assert Cj.planar_vectors == [N.y, N.z]

    # 确认 Cj.child_point 相对于 C.masscenter 的位置是否为 Vector(0)
    assert Cj.child_point.pos_from(C.masscenter) == Vector(0)

    # 确认 Cj.parent_point 相对于 P.masscenter 的位置是否为 Vector(0)
    assert Cj.parent_point.pos_from(P.masscenter) == Vector(0)

    # 计算点 P 到点 C 的位置矢量 r_P_C
    r_P_C = q1_def * N.y + q2_def * N.z
    assert Cj.parent_point.pos_from(Cj.child_point) == -r_P_C

    # 确认 C.masscenter 相对于 P.masscenter 的位置是否为 r_P_C
    assert C.masscenter.pos_from(P.masscenter) == r_P_C

    # 确认 Cj.child_point 相对于惯性参考系 N 的速度是否为 u1_def * N.y + u2_def * N.z
    assert Cj.child_point.vel(N) == u1_def * N.y + u2_def * N.z

    # 确认刚体 A 相对于惯性参考系 N 的角速度是否为 u0_def * N.x
    assert A.ang_vel_in(N) == u0_def * N.x

    # 确认 Cj 的父参考系间隔为 N
    assert Cj.parent_interframe == N

    # 确认 Cj 的子参考系间隔为 A
    assert Cj.child_interframe == A

    # 确认 Cj 的字符串表示是否为 'PlanarJoint: J  parent: P  child: C'
    assert Cj.__str__() == 'PlanarJoint: J  parent: P  child: C'

    # 定义动力学符号 q0, q1, q2, u0, u1, u2
    q0, q1, q2, u0, u1, u2 = dynamicsymbols('q0:3, u0:3')

    # 定义符号 l, m
    l, m = symbols('l, m')

    # 生成刚体 N, A, P, C, Pint, Cint
    N, A, P, C, Pint, Cint = _generate_body(True)

    # 创建 PlanarJoint 对象 Cj，指定参数如 rotation_coordinate=q0, planar_coordinates=[q1, q2], planar_speeds=[u1, u2], parent_point=m * N.x, child_point=l * A.y, parent_interframe=Pint, child_interframe=Cint
    Cj = PlanarJoint('J', P, C, rotation_coordinate=q0,
                     planar_coordinates=[q1, q2], planar_speeds=[u1, u2],
                     parent_point=m * N.x, child_point=l * A.y,
                     parent_interframe=Pint, child_interframe=Cint)

    # 确认 Cj 的坐标是否为 Matrix([q0, q1, q2])
    assert Cj.coordinates == Matrix([q0, q1, q2])

    # 确认 Cj 的速度是否为 Matrix([u0_def, u1, u2])
    assert Cj.speeds == Matrix([u0_def, u1, u2])

    # 确认 Cj 的旋转坐标是否为 q0
    assert Cj.rotation_coordinate == q0

    # 确认 Cj 的平面坐标是否为 Matrix([q1, q2])
    assert Cj.planar_coordinates == Matrix([q1, q2])

    # 确认 Cj 的旋转速度是否为 u0_def
    assert Cj.rotation_speed == u0_def

    # 确认 Cj 的平面速度是否为 Matrix([u1, u2])
    assert Cj.planar_speeds == Matrix([u1, u2])

    # 确认 Cj.kdes 是否等于 Matrix([u0_def - q0.diff(t), u1 - q1.diff(t), u2 - q2.diff(t)])
    assert Cj.kdes == Matrix([u0_def - q0.diff(t), u1 - q1.diff(t), u2 - q2.diff(t)])

    # 确认 Cj 的旋转轴是否为 Pint.x
    assert Cj.rotation_axis == Pint.x

    # 确认 Cj 的平面向量列表是否为 [Pint.y, Pint.z]
    assert Cj.planar_vectors == [Pint.y, Pint.z]

    # 确认 Cj.child_point 相对于 C.masscenter 的位置是否为 l * A.y
    assert Cj.child_point.pos_from(C.masscenter) == l * A.y

    # 确认 Cj.parent_point 相对于 P.masscenter 的位置是否为 m * N.x
    assert Cj.parent_point.pos_from(P.masscenter) == m * N.x

    # 确认 Cj.parent_point 相对于 Cj.child_point 的位置是否为 q1 * N.y + q2 * N.z
    assert Cj.parent_point.pos_from(Cj.child_point) == q1 * N.y + q2 * N.z

    # 确认 C.masscenter 相对于 P.masscenter 的位置是否为 m * N.x - q1 * N.y - q2 * N.z - l * A.y
    assert C.masscenter.pos_from(P.masscenter) == m * N.x - q1 * N.y - q2 * N.z - l * A.y

    # 确认 C.masscenter 相对于惯性参考系 N 的速度是否为 -u1 * N.y - u2 * N.z + u0_def * l * A.x
    assert C.masscenter.vel(N) == -u1 * N.y - u2 * N.z + u0_def * l * A.x

    # 确认刚体 A 相对于惯性参考系 N 的角速度是否为 u0_def * N.x
    assert A.ang_vel_in(N) == u0_def * N.x
# 定义一个测试函数，用于测试高级平面关节的功能
def test_planar_joint_advanced():
    # 测试是否能够仅指定两个法线，形成父体和子体的旋转轴
    # 此特定示例是一个在斜坡上的块，斜率为30度，在零配置下，父体和子体的框架实际上是对齐的
    q0, q1, q2, u0, u1, u2 = dynamicsymbols('q0:3, u0:3')
    l1, l2 = symbols('l1:3')
    N, A, P, C = _generate_body()
    # 创建一个平面关节对象J，连接父体P和子体C，使用广义坐标q0和[q1, q2]，广义速度u0和[u1, u2]
    # 设置父体连接点在N.z方向的l1处，子体连接点在-C.z方向的l2处
    # 设置父体内部框架为N.z + N.y / sqrt(3)，子体内部框架为A.z + A.y / sqrt(3)
    J = PlanarJoint('J', P, C, q0, [q1, q2], u0, [u1, u2],
                    parent_point=l1 * N.z,
                    child_point=-l2 * C.z,
                    parent_interframe=N.z + N.y / sqrt(3),
                    child_interframe=A.z + A.y / sqrt(3))
    # 断言旋转轴在惯性参考系N中的表达式等于(N.z + N.y / sqrt(3)).normalize()
    assert J.rotation_axis.express(N) == (N.z + N.y / sqrt(3)).normalize()
    # 断言旋转轴在附体A中的表达式等于(A.z + A.y / sqrt(3)).normalize()
    assert J.rotation_axis.express(A) == (A.z + A.y / sqrt(3)).normalize()
    # 断言旋转轴与惯性参考系N.z之间的夹角为pi / 6
    assert J.rotation_axis.angle_between(N.z) == pi / 6
    # 断言惯性参考系N相对于附体A的方向余弦矩阵在q0, q1, q2为0时等于单位矩阵
    assert N.dcm(A).xreplace({q0: 0, q1: 0, q2: 0}) == eye(3)
    # 计算得到惯性参考系N相对于附体A的方向余弦矩阵N_R_A
    N_R_A = Matrix([
        [cos(q0), -sqrt(3) * sin(q0) / 2, sin(q0) / 2],
        [sqrt(3) * sin(q0) / 2, 3 * cos(q0) / 4 + 1 / 4,
         sqrt(3) * (1 - cos(q0)) / 4],
        [-sin(q0) / 2, sqrt(3) * (1 - cos(q0)) / 4, cos(q0) / 4 + 3 / 4]])
    # 断言简化后的惯性参考系N相对于附体A的方向余弦矩阵与预期的N_R_A相等
    # simplify函数用于简化矩阵表达式
    assert simplify(N.dcm(A) - N_R_A) == zeros(3)


# 定义一个测试函数，用于测试球形关节的功能
def test_spherical_joint():
    N, A, P, C = _generate_body()
    q0, q1, q2, u0, u1, u2 = dynamicsymbols('q0:3_S, u0:3_S')
    # 创建一个球形关节对象S，连接父体P和子体C
    S = SphericalJoint('S', P, C)
    # 断言关节名称为'S'
    assert S.name == 'S'
    # 断言关节的父体为P，子体为C
    assert S.parent == P
    assert S.child == C
    # 断言关节的广义坐标为Matrix([q0, q1, q2])
    assert S.coordinates == Matrix([q0, q1, q2])
    # 断言关节的广义速度为Matrix([u0, u1, u2])
    assert S.speeds == Matrix([u0, u1, u2])
    # 断言关节的广义速度减去广义坐标的时间导数等于kdes
    assert S.kdes == Matrix([u0 - q0.diff(t), u1 - q1.diff(t), u2 - q2.diff(t)])
    # 断言关节的子体连接点相对于质心的位置矢量为零向量
    assert S.child_point.pos_from(C.masscenter) == Vector(0)
    # 断言关节的父体连接点相对于质心的位置矢量为零向量
    assert S.parent_point.pos_from(P.masscenter) == Vector(0)
    # 断言关节的父体连接点相对于子体连接点的位置矢量为零向量
    assert S.parent_point.pos_from(S.child_point) == Vector(0)
    # 断言父体P的质心相对于子体C的质心的位置矢量为零向量
    assert P.masscenter.pos_from(C.masscenter) == Vector(0)
    # 断言子体C的质心在惯性参考系N中的速度为零向量
    assert C.masscenter.vel(N) == Vector(0)
    # 断言惯性参考系N相对于附体A的角速度
    assert N.ang_vel_in(A) == (-u0 * cos(q1) * cos(q2) - u1 * sin(q2)) * A.x + (
            u0 * sin(q2) * cos(q1) - u1 * cos(q2)) * A.y + (
                   -u0 * sin(q1) - u2) * A.z
    # 断言附体A相对于惯性参考系N的角速度
    assert A.ang_vel_in(N) == (u0 * cos(q1) * cos(q2) + u1 * sin(q2)) * A.x + (
            -u0 * sin(q2) * cos(q1) + u1 * cos(q2)) * A.y + (
                   u0 * sin(q1) + u2) * A.z
    # 断言关节对象S的字符串表示为'SphericalJoint: S  parent: P  child: C'
    assert S.__str__() == 'SphericalJoint: S  parent: P  child: C'
    # 断言关节的旋转类型为'BODY'
    assert S._rot_type == 'BODY'
    # 断言关节的旋转顺序为123
    assert S._rot_order == 123
    # 断言关节的额外参数为None
    assert S._amounts is None


# 定义一个测试函数，用于测试球形关节中速度作为广义坐标时间导数的情况
def test_spherical_joint_speeds_as_derivative_terms():
    # 这个测试检查系统是否仍然有效，如果用户选择将广义坐标的导数作为广义速度
    q0, q1, q2 = dynamicsymbols('q0:3')
    u0, u1, u2 = dynamicsymbols('q0:3', 1)
    N, A, P, C = _generate_body()
    # 创建一个球形关节对象S，连接父体P和子体C，使用广义坐标[q0, q1, q2]和广义速度[u0, u1, u2]
    S = SphericalJoint('S', P, C, coordinates=[q0, q1, q2], speeds=[u0, u1, u2])
    # 断言：验证系统 S 的坐标是否为矩阵 [q0, q1, q2]
    assert S.coordinates == Matrix([q0, q1, q2])
    
    # 断言：验证系统 S 的速度是否为矩阵 [u0, u1, u2]
    assert S.speeds == Matrix([u0, u1, u2])
    
    # 断言：验证系统 S 的广义速度的导数（kdes）是否为零矩阵 [0, 0, 0]
    assert S.kdes == Matrix([0, 0, 0])
    
    # 断言：验证参考系 N 在参考系 A 中的角速度
    # 计算方式为 (-u0 * cos(q1) * cos(q2) - u1 * sin(q2)) * A.x +
    #            (u0 * sin(q2) * cos(q1) - u1 * cos(q2)) * A.y +
    #            (-u0 * sin(q1) - u2) * A.z
    assert N.ang_vel_in(A) == (-u0 * cos(q1) * cos(q2) - u1 * sin(q2)) * A.x + (
        u0 * sin(q2) * cos(q1) - u1 * cos(q2)) * A.y + (
        -u0 * sin(q1) - u2) * A.z
def test_spherical_joint_coords():
    # 定义动力学符号变量
    q0s, q1s, q2s, u0s, u1s, u2s = dynamicsymbols('q0:3_S, u0:3_S')
    q0, q1, q2, q3, u0, u1, u2, u4 = dynamicsymbols('q0:4, u0:4')

    # 创建身体和连接点
    N, A, P, C = _generate_body()

    # 测试球面关节对象S，使用列表形式的广义坐标和速度
    S = SphericalJoint('S', P, C, [q0, q1, q2], [u0, u1, u2])
    assert S.coordinates == Matrix([q0, q1, q2])
    assert S.speeds == Matrix([u0, u1, u2])

    # 再次创建身体和连接点，测试球面关节对象S，使用矩阵形式的广义坐标和速度
    N, A, P, C = _generate_body()
    S = SphericalJoint('S', P, C, Matrix([q0, q1, q2]),
                       Matrix([u0, u1, u2]))
    assert S.coordinates == Matrix([q0, q1, q2])
    assert S.speeds == Matrix([u0, u1, u2])

    # 再次创建身体和连接点，测试过少的广义坐标引发错误
    N, A, P, C = _generate_body()
    raises(ValueError,
           lambda: SphericalJoint('S', P, C, Matrix([q0, q1]), Matrix([u0])))

    # 再次创建身体和连接点，测试过多的广义坐标引发错误
    raises(ValueError, lambda: SphericalJoint(
        'S', P, C, Matrix([q0, q1, q2, q3]), Matrix([u0, u1, u2])))
    raises(ValueError, lambda: SphericalJoint(
        'S', P, C, Matrix([q0, q1, q2]), Matrix([u0, u1, u2, u4])))


def test_spherical_joint_orient_body():
    # 定义动力学符号变量
    q0, q1, q2, u0, u1, u2 = dynamicsymbols('q0:3, u0:3')

    # 创建姿态矩阵N_R_A
    N_R_A = Matrix([
        [-sin(q1), -sin(q2) * cos(q1), cos(q1) * cos(q2)],
        [-sin(q0) * cos(q1), sin(q0) * sin(q1) * sin(q2) - cos(q0) * cos(q2),
         -sin(q0) * sin(q1) * cos(q2) - sin(q2) * cos(q0)],
        [cos(q0) * cos(q1), -sin(q0) * cos(q2) - sin(q1) * sin(q2) * cos(q0),
         -sin(q0) * sin(q2) + sin(q1) * cos(q0) * cos(q2)]])

    # 创建角速度矩阵N_w_A
    N_w_A = Matrix([[-u0 * sin(q1) - u2],
                    [-u0 * sin(q2) * cos(q1) + u1 * cos(q2)],
                    [u0 * cos(q1) * cos(q2) + u1 * sin(q2)]])

    # 创建速度矩阵N_v_Co
    N_v_Co = Matrix([
        [-sqrt(2) * (u0 * cos(q2 + pi / 4) * cos(q1) + u1 * sin(q2 + pi / 4))],
        [-u0 * sin(q1) - u2], [-u0 * sin(q1) - u2]])

    # 创建身体和连接点，测试默认的rot_type='BODY', rot_order=123
    N, A, P, C, Pint, Cint = _generate_body(True)
    S = SphericalJoint('S', P, C, coordinates=[q0, q1, q2], speeds=[u0, u1, u2],
                       parent_point=N.x + N.y, child_point=-A.y + A.z,
                       parent_interframe=Pint, child_interframe=Cint,
                       rot_type='body', rot_order=123)
    assert S._rot_type.upper() == 'BODY'
    assert S._rot_order == 123
    assert simplify(N.dcm(A) - N_R_A) == zeros(3)
    assert simplify(A.ang_vel_in(N).to_matrix(A) - N_w_A) == zeros(3, 1)
    assert simplify(C.masscenter.vel(N).to_matrix(A)) == N_v_Co

    # 再次创建身体和连接点，测试改变广义坐标顺序引发的变化
    N, A, P, C, Pint, Cint = _generate_body(True)
    S = SphericalJoint('S', P, C, coordinates=[q0, q1, q2], speeds=[u0, u1, u2],
                       parent_point=N.x + N.y, child_point=-A.y + A.z,
                       parent_interframe=Pint, child_interframe=Cint,
                       rot_type='BODY', amounts=(q1, q0, q2), rot_order=123)
    switch_order = lambda expr: expr.xreplace(
        {q0: q1, q1: q0, q2: q2, u0: u1, u1: u0, u2: u2})
    # 断言：验证 S 对象的旋转类型是否为大写的 'BODY'
    assert S._rot_type.upper() == 'BODY'
    
    # 断言：验证 S 对象的旋转顺序是否为 123
    assert S._rot_order == 123
    
    # 断言：验证 N 相对于 A 的方向余弦矩阵减去 N_R_A 函数的简化结果是否为零矩阵
    assert simplify(N.dcm(A) - switch_order(N_R_A)) == zeros(3)
    
    # 断言：验证 A 相对于 N 的角速度转换到 A 坐标系后减去 switch_order(N_w_A) 函数的简化结果是否为零矩阵
    assert simplify(A.ang_vel_in(N).to_matrix(A) - switch_order(N_w_A)) == zeros(3, 1)
    
    # 断言：验证 C 质心在 N 坐标系下的速度转换到 A 坐标系后是否等于 switch_order(N_v_Co) 函数的简化结果
    assert simplify(C.masscenter.vel(N).to_matrix(A)) == switch_order(N_v_Co)
    
    # 测试不同的旋转顺序
    # 生成关节和刚体，返回 N, A, P, C, Pint, Cint
    N, A, P, C, Pint, Cint = _generate_body(True)
    
    # 创建球形关节 S 对象，并设置其参数
    S = SphericalJoint('S', P, C, coordinates=[q0, q1, q2], speeds=[u0, u1, u2],
                       parent_point=N.x + N.y, child_point=-A.y + A.z,
                       parent_interframe=Pint, child_interframe=Cint,
                       rot_type='BodY', rot_order='yxz')
    
    # 断言：验证 S 对象的旋转类型是否为大写的 'BODY'
    assert S._rot_type.upper() == 'BODY'
    
    # 断言：验证 S 对象的旋转顺序是否为 'yxz'
    assert S._rot_order == 'yxz'
    
    # 断言：验证 N 相对于 A 的方向余弦矩阵减去给定矩阵的简化结果是否为零矩阵
    assert simplify(N.dcm(A) - Matrix([
        [-sin(q0) * cos(q1), sin(q0) * sin(q1) * cos(q2) - sin(q2) * cos(q0),
         sin(q0) * sin(q1) * sin(q2) + cos(q0) * cos(q2)],
        [-sin(q1), -cos(q1) * cos(q2), -sin(q2) * cos(q1)],
        [cos(q0) * cos(q1), -sin(q0) * sin(q2) - sin(q1) * cos(q0) * cos(q2),
         sin(q0) * cos(q2) - sin(q1) * sin(q2) * cos(q0)]])) == zeros(3)
    
    # 断言：验证 A 相对于 N 的角速度转换到 A 坐标系后减去给定矩阵的简化结果是否为零矩阵
    assert simplify(A.ang_vel_in(N).to_matrix(A) - Matrix([
        [u0 * sin(q1) - u2], [u0 * cos(q1) * cos(q2) - u1 * sin(q2)],
        [u0 * sin(q2) * cos(q1) + u1 * cos(q2)]])) == zeros(3, 1)
    
    # 断言：验证 C 质心在 N 坐标系下的速度转换到 A 坐标系后是否等于给定矩阵的简化结果
    assert simplify(C.masscenter.vel(N).to_matrix(A)) == Matrix([
        [-sqrt(2) * (u0 * sin(q2 + pi / 4) * cos(q1) + u1 * cos(q2 + pi / 4))],
        [u0 * sin(q1) - u2], [u0 * sin(q1) - u2]])
def test_spherical_joint_orient_space():
    # 定义动力学符号变量
    q0, q1, q2, u0, u1, u2 = dynamicsymbols('q0:3, u0:3')
    
    # 计算惯性参考系到连接体参考系的方向余弦矩阵
    N_R_A = Matrix([
        [-sin(q0) * sin(q2) - sin(q1) * cos(q0) * cos(q2),
         sin(q0) * sin(q1) * cos(q2) - sin(q2) * cos(q0), cos(q1) * cos(q2)],
        [-sin(q0) * cos(q2) + sin(q1) * sin(q2) * cos(q0),
         -sin(q0) * sin(q1) * sin(q2) - cos(q0) * cos(q2), -sin(q2) * cos(q1)],
        [cos(q0) * cos(q1), -sin(q0) * cos(q1), sin(q1)]])
    
    # 计算连接体角速度在惯性参考系中的表示
    N_w_A = Matrix([
        [u1 * sin(q0) - u2 * cos(q0) * cos(q1)],
        [u1 * cos(q0) + u2 * sin(q0) * cos(q1)], 
        [u0 - u2 * sin(q1)]])
    
    # 计算质心速度在惯性参考系中的表示
    N_v_Co = Matrix([
        [u0 - u2 * sin(q1)], [u0 - u2 * sin(q1)],
        [sqrt(2) * (-u1 * sin(q0 + pi / 4) + u2 * cos(q0 + pi / 4) * cos(q1))]])
    
    # 测试默认参数下的球面关节对象创建，旋转类型为空间旋转，旋转顺序为123
    N, A, P, C, Pint, Cint = _generate_body(True)
    S = SphericalJoint('S', P, C, coordinates=[q0, q1, q2], speeds=[u0, u1, u2],
                       parent_point=N.x + N.z, child_point=-A.x + A.y,
                       parent_interframe=Pint, child_interframe=Cint,
                       rot_type='space', rot_order=123)
    assert S._rot_type.upper() == 'SPACE'
    assert S._rot_order == 123
    assert simplify(N.dcm(A) - N_R_A) == zeros(3)
    assert simplify(A.ang_vel_in(N).to_matrix(A)) == N_w_A
    assert simplify(C.masscenter.vel(N).to_matrix(A)) == N_v_Co
    
    # 测试旋转角度顺序变化后的球面关节对象创建
    switch_order = lambda expr: expr.xreplace(
        {q0: q1, q1: q0, q2: q2, u0: u1, u1: u0, u2: u2})
    N, A, P, C, Pint, Cint = _generate_body(True)
    S = SphericalJoint('S', P, C, coordinates=[q0, q1, q2], speeds=[u0, u1, u2],
                       parent_point=N.x + N.z, child_point=-A.x + A.y,
                       parent_interframe=Pint, child_interframe=Cint,
                       rot_type='SPACE', amounts=(q1, q0, q2), rot_order=123)
    assert S._rot_type.upper() == 'SPACE'
    assert S._rot_order == 123
    assert simplify(N.dcm(A) - switch_order(N_R_A)) == zeros(3)
    assert simplify(A.ang_vel_in(N).to_matrix(A)) == switch_order(N_w_A)
    assert simplify(C.masscenter.vel(N).to_matrix(A)) == switch_order(N_v_Co)
    
    # 测试不同旋转顺序字符串下的球面关节对象创建
    N, A, P, C, Pint, Cint = _generate_body(True)
    S = SphericalJoint('S', P, C, coordinates=[q0, q1, q2], speeds=[u0, u1, u2],
                       parent_point=N.x + N.z, child_point=-A.x + A.y,
                       parent_interframe=Pint, child_interframe=Cint,
                       rot_type='SPaCe', rot_order='zxy')
    assert S._rot_type.upper() == 'SPACE'
    assert S._rot_order == 'zxy'
    assert simplify(N.dcm(A) - Matrix([
        [-sin(q2) * cos(q1), -sin(q0) * cos(q2) + sin(q1) * sin(q2) * cos(q0),
         sin(q0) * sin(q1) * sin(q2) + cos(q0) * cos(q2)],
        [-sin(q1), -cos(q0) * cos(q1), -sin(q0) * cos(q1)],
        [cos(q1) * cos(q2), -sin(q0) * sin(q2) - sin(q1) * cos(q0) * cos(q2),
         -sin(q0) * sin(q1) * cos(q2) + sin(q2) * cos(q0)]]))
    # 断言：验证角速度在参考系A中的矩阵表示是否简化为零向量
    assert simplify(A.ang_vel_in(N).to_matrix(A) - Matrix([
        [-u0 + u2 * sin(q1)], [-u1 * sin(q0) + u2 * cos(q0) * cos(q1)],
        [u1 * cos(q0) + u2 * sin(q0) * cos(q1)]])) == zeros(3, 1)
    # 断言：验证质心速度在参考系A中的矩阵表示是否简化为零向量
    assert simplify(C.masscenter.vel(N).to_matrix(A) - Matrix([
        [u1 * cos(q0) + u2 * sin(q0) * cos(q1)],
        [u1 * cos(q0) + u2 * sin(q0) * cos(q1)],
        [u0 + u1 * sin(q0) - u2 * sin(q1) -
         u2 * cos(q0) * cos(q1)]])) == zeros(3, 1)
# 测试焊接接头的功能
def test_weld_joint():
    # 生成一个身体，返回其四个部分
    _, _, P, C = _generate_body()
    # 创建一个焊接接头对象W，连接P和C
    W = WeldJoint('W', P, C)
    # 断言W的名称为'W'
    assert W.name == 'W'
    # 断言W的父部件为P
    assert W.parent == P
    # 断言W的子部件为C
    assert W.child == C
    # 断言W的坐标为零矩阵
    assert W.coordinates == Matrix()
    # 断言W的速度为零矩阵
    assert W.speeds == Matrix()
    # 断言W的kdes属性为1x0空矩阵的转置
    assert W.kdes == Matrix(1, 0, []).T
    # 断言P的坐标系相对于C的坐标系的方向余弦矩阵为单位矩阵
    assert P.frame.dcm(C.frame) == eye(3)
    # 断言W的子部件相对于C的重心位置向量为零向量
    assert W.child_point.pos_from(C.masscenter) == Vector(0)
    # 断言W的父部件相对于P的重心位置向量为零向量
    assert W.parent_point.pos_from(P.masscenter) == Vector(0)
    # 断言W的父部件相对于W的子部件的位置向量为零向量
    assert W.parent_point.pos_from(W.child_point) == Vector(0)
    # 断言P的重心相对于C的重心位置向量为零向量
    assert P.masscenter.pos_from(C.masscenter) == Vector(0)
    # 断言C的重心相对于P的坐标系的速度为零向量
    assert C.masscenter.vel(P.frame) == Vector(0)
    # 断言P的坐标系相对于C的坐标系的角速度为零
    assert P.frame.ang_vel_in(C.frame) == 0
    # 断言C的坐标系相对于P的坐标系的角速度为零
    assert C.frame.ang_vel_in(P.frame) == 0
    # 断言W对象的字符串表示为'WeldJoint: W  parent: P  child: C'
    assert W.__str__() == 'WeldJoint: W  parent: P  child: C'

    # 生成一个身体，返回其四个部分
    N, A, P, C = _generate_body()
    # 定义符号变量l和m
    l, m = symbols('l m')
    # 创建一个新的参考系Pint
    Pint = ReferenceFrame('P_int')
    # 将Pint参考系沿着P的y轴旋转π/2弧度
    Pint.orient_axis(P.frame, P.y, pi / 2)
    # 创建一个新的焊接接头对象W，连接P和C，指定父部件点和子部件点的位置
    W = WeldJoint('W', P, C, parent_point=l * P.frame.x,
                  child_point=m * C.frame.y, parent_interframe=Pint)

    # 断言W的子部件相对于C的重心位置向量为m乘以C的y轴向量
    assert W.child_point.pos_from(C.masscenter) == m * C.frame.y
    # 断言W的父部件相对于P的重心位置向量为l乘以P的x轴向量
    assert W.parent_point.pos_from(P.masscenter) == l * P.frame.x
    # 断言W的父部件相对于W的子部件的位置向量为零向量
    assert W.parent_point.pos_from(W.child_point) == Vector(0)
    # 断言P的重心相对于C的重心位置向量为-l乘以N的x轴向量加上m乘以A的y轴向量
    assert P.masscenter.pos_from(C.masscenter) == - l * N.x + m * A.y
    # 断言C的重心相对于Pint参考系的速度为零向量
    assert C.masscenter.vel(Pint) == Vector(0)
    # 断言C的坐标系相对于P的坐标系的角速度为零
    assert C.frame.ang_vel_in(P.frame) == 0
    # 断言P的坐标系相对于C的坐标系的角速度为零
    assert P.frame.ang_vel_in(C.frame) == 0
    # 断言P的x轴与A的z轴重合
    assert P.x == A.z

    # 使用warns_deprecated_sympy()上下文
    with warns_deprecated_sympy():
        # 调用JointsMethod函数，传入参数P和W，测试#10770
        JointsMethod(P, W)


# 测试废弃的parent_child_axis方法
def test_deprecated_parent_child_axis():
    # 定义动力符号变量q_J和u_J
    q, u = dynamicsymbols('q_J, u_J')
    # 生成一个身体，返回其四个部分
    N, A, P, C = _generate_body()
    # 使用warns_deprecated_sympy()上下文
    with warns_deprecated_sympy():
        # 创建一个PinJoint对象'J'，连接P和C，指定子部件轴为-A.x
        PinJoint('J', P, C, child_axis=-A.x)
    # 断言-A.x轴与N.x轴之间的夹角为零
    assert (-A.x).angle_between(N.x) == 0
    # 断言-A.x在N坐标系中的表达式为N.x
    assert -A.x.express(N) == N.x
    # 断言A相对于N的方向余弦矩阵为给定矩阵
    assert A.dcm(N) == Matrix([[-1, 0, 0],
                               [0, -cos(q), -sin(q)],
                               [0, -sin(q), cos(q)]])
    # 断言A相对于N的角速度为u乘以N的x轴向量
    assert A.ang_vel_in(N) == u * N.x
    # 断言A相对于N的角速度的大小为u的平方根
    assert A.ang_vel_in(N).magnitude() == sqrt(u ** 2)

    # 生成一个身体，返回其四个部分
    N, A, P, C = _generate_body()
    # 使用warns_deprecated_sympy()上下文
    with warns_deprecated_sympy():
        # 创建一个PrismaticJoint对象'J'，连接P和C，指定父部件轴为P.x + P.y
        PrismaticJoint('J', P, C, parent_axis=P.x + P.y)
    # 断言A.x轴与N.x + N.y的夹角为零
    assert (A.x).angle_between(N.x + N.y) == 0
    # 断言A在N坐标系中的表达式为(N.x + N.y)除以其长度
    assert A.x.express(N) == (N.x + N.y) / sqrt(2)
    # 断言A相对于N的方向余弦矩阵为给定矩阵
    assert A.dcm(N) == Matrix([[sqrt(2) / 2, sqrt(2) / 2, 0],
                               [-sqrt(2) / 2, sqrt(2) / 2, 0], [0, 0, 1]])
    # 断言A相对于N的角速度为零向量
    assert A.ang_vel_in(N) == Vector(0)


# 测试废弃的joint_pos方法
def test_deprecated_joint_pos():
    # 生成一个身体，返回其四个部分
    N, A, P, C = _generate_body()
    # 使用warns_deprecated_sympy()上下文
    with warns_deprecated_sympy():
        # 创建一个PinJoint对象'J'，连接P和C，指定父部件关节位置为N.x + N.y，子部件关节位置为C.y - C.z
    # 使用 warns_deprecated_sympy() 上下文管理器，处理即将废弃的 sympy 相关警告
    with warns_deprecated_sympy():
        # 创建一个名为 'J' 的棱柱关节对象 PrismaticJoint，连接点为 P 和 C
        # 父关节位置为 N.z + N.y
        # 子关节位置为 C.y - C.x
        slider = PrismaticJoint('J', P, C, parent_joint_pos=N.z + N.y,
                                child_joint_pos=C.y - C.x)
    # 断言：滑块关节的父连接点位置与 P 质心的位置之间的相对位置应为 N.z + N.y
    assert slider.parent_point.pos_from(P.masscenter) == N.z + N.y
    # 断言：滑块关节的子连接点位置与 C 质心的位置之间的相对位置应为 C.y - C.x
    assert slider.child_point.pos_from(C.masscenter) == C.y - C.x
```