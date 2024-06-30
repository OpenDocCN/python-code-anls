# `D:\src\scipysrc\sympy\sympy\physics\mechanics\tests\test_jointsmethod.py`

```
# 导入 sympy 库中的特定模块和函数
from sympy.core.function import expand
from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.dense import Matrix
from sympy.simplify.trigsimp import trigsimp
from sympy.physics.mechanics import (
    PinJoint, JointsMethod, RigidBody, Particle, Body, KanesMethod,
    PrismaticJoint, LagrangesMethod, inertia)
from sympy.physics.vector import dynamicsymbols, ReferenceFrame
from sympy.testing.pytest import raises, warns_deprecated_sympy
from sympy import zeros
from sympy.utilities.lambdify import lambdify
from sympy.solvers.solvers import solve

# 定义时间符号 t 作为动力学符号的时间变量
t = dynamicsymbols._t  # type: ignore


# 定义测试函数 test_jointsmethod，用于测试 JointsMethod 类的功能
def test_jointsmethod():
    # 使用 warns_deprecated_sympy 上下文管理器来捕获 sympy 弃用警告
    with warns_deprecated_sympy():
        # 创建两个刚体 P 和 C
        P = Body('P')
        C = Body('C')
    # 创建一个 PinJoint 对象 Pin，连接刚体 P 和 C
    Pin = PinJoint('P1', P, C)
    # 定义符号 C_ixx 和 g
    C_ixx, g = symbols('C_ixx g')
    # 定义动力学符号 q 和 u
    q, u = dynamicsymbols('q_P1, u_P1')
    # 在刚体 P 上施加一个力，力的大小为 g*P.y 方向为 P.y
    P.apply_force(g*P.y)
    # 使用 warns_deprecated_sympy 上下文管理器来捕获 sympy 弃用警告
    with warns_deprecated_sympy():
        # 创建 JointsMethod 对象 method，应用于刚体 P 和 Pin 关节
        method = JointsMethod(P, Pin)
    # 断言 method 对象的帧与 P 的帧相同
    assert method.frame == P.frame
    # 断言 method 对象包含的刚体列表为 [C, P]
    assert method.bodies == [C, P]
    # 断言 method 对象的负载为 [(P.masscenter, g*P.frame.y)]
    assert method.loads == [(P.masscenter, g*P.frame.y)]
    # 断言 method 对象的广义坐标为 Matrix([q])
    assert method.q == Matrix([q])
    # 断言 method 对象的广义速度为 Matrix([u])
    assert method.u == Matrix([u])
    # 断言 method 对象的运动方程为 Matrix([u - q.diff()])
    assert method.kdes == Matrix([u - q.diff()])
    # 计算并断言 method 对象的运动方程的解为 Matrix([[-C_ixx*u.diff()]])
    soln = method.form_eoms()
    assert soln == Matrix([[-C_ixx*u.diff()]])
    # 断言 method 对象的完全强迫项为 Matrix([[u], [0]])
    assert method.forcing_full == Matrix([[u], [0]])
    # 断言 method 对象的完全质量矩阵为 Matrix([[1, 0], [0, C_ixx]])
    assert method.mass_matrix_full == Matrix([[1, 0], [0, C_ixx]])
    # 断言 method 对象的 method 属性是 KanesMethod 类的实例
    assert isinstance(method.method, KanesMethod)


# 定义测试函数 test_rigid_body_particle_compatibility，测试刚体和粒子的兼容性
def test_rigid_body_particle_compatibility():
    # 定义符号 l, m, g
    l, m, g = symbols('l m g')
    # 创建一个 RigidBody 对象 C
    C = RigidBody('C')
    # 创建一个质点 Particle b，质量为 m
    b = Particle('b', mass=m)
    # 创建一个参考框架 b_frame
    b_frame = ReferenceFrame('b_frame')
    # 定义动力学符号 q 和 u
    q, u = dynamicsymbols('q u')
    # 创建一个 PinJoint 对象 P，连接刚体 C 和质点 b
    P = PinJoint('P', C, b, coordinates=q, speeds=u, child_interframe=b_frame,
                 child_point=-l * b_frame.x, joint_axis=C.z)
    # 使用 warns_deprecated_sympy 上下文管理器来捕获 sympy 弃用警告
    with warns_deprecated_sympy():
        # 创建 JointsMethod 对象 method，应用于刚体 C 和 Pin 关节 P
        method = JointsMethod(C, P)
    # 将质点 b 的重力加载添加到 method 对象的负载中
    method.loads.append((b.masscenter, m * g * C.x))
    # 计算并形成 method 对象的运动方程
    method.form_eoms()
    # 计算 method 对象的右手边项 rhs
    rhs = method.rhs()
    # 断言 rhs 的第二个元素为 -g*sin(q)/l
    assert rhs[1] == -g*sin(q)/l


# 定义测试函数 test_jointmethod_duplicate_coordinates_speeds，测试重复广义坐标和速度的情况
def test_jointmethod_duplicate_coordinates_speeds():
    # 使用 warns_deprecated_sympy 上下文管理器来捕获 sympy 弃用警告
    with warns_deprecated_sympy():
        # 创建三个刚体 P, C, T
        P = Body('P')
        C = Body('C')
        T = Body('T')
    # 定义动力学符号 q 和 u
    q, u = dynamicsymbols('q u')
    # 创建一个 PinJoint 对象 P1，连接刚体 P 和 C，指定广义坐标 q
    P1 = PinJoint('P1', P, C, q)
    # 创建一个 PrismaticJoint 对象 P2，连接刚体 C 和 T，指定广义坐标 q
    P2 = PrismaticJoint('P2', C, T, q)
    # 使用 warns_deprecated_sympy 上下文管理器来捕获 sympy 弃用警告，并断言引发 ValueError 异常
    with warns_deprecated_sympy():
        raises(ValueError, lambda: JointsMethod(P, P1, P2))

    # 创建一个 PinJoint 对象 P1，连接刚体 P 和 C，指定广义速度 u
    P1 = PinJoint('P1', P, C, speeds=u)
    # 创建一个 PrismaticJoint 对象 P2，连接刚体 C 和 T，指定广义速度 u
    P2 = PrismaticJoint('P2', C, T, speeds=u)
    # 使用 warns_deprecated_sympy 上下文管理器来捕获 sympy 弃用警告，并断言引发 ValueError 异常
    with warns_deprecated_sympy():
        raises(ValueError, lambda: JointsMethod(P, P1, P2))

    # 创建一个 PinJoint 对象 P1，连接刚体 P 和 C，指定广义坐标 q 和速度 u
    P1 = PinJoint('P1', P, C, q, u)
    # 创建一个 PrismaticJoint 对象 P2，连接刚体 C 和 T，指定广义坐标 q 和速度 u
    P2 = PrismaticJoint('P2', C, T, q, u)
    # 使用 warns_deprecated_sympy 上下文管理器来捕获 sympy 弃用警告，并断言引发 ValueError 异常
    with warns_deprecated_sympy():
        raises(ValueError, lambda: JointsMethod(P, P1, P2))


# 定义测试函数 test_complete_simple_double_pendulum，测试完整的简单双摆系统
def test_complete_simple_double_pendulum():
    # 定义动力学符号 q1, q2 和 u1, u2
    q1, q2 = dynamicsymbols('q1 q2')
    u1, u2 = dynamicsymbols('u1 u2')
    # 定义符号 m, l, g
    m, l, g = symbols('m l g')
    # 在使用 warns_deprecated_sympy 上下文管理器中，执行下列语句时会发出关于 sympy 的弃用警告
    with warns_deprecated_sympy():
        # 创建名为 C 的刚体对象，表示天花板
        C = Body('C')  # ceiling
        # 创建名为 PartP 的刚体对象，表示部件 P，并指定质量为 m
        PartP = Body('P', mass=m)
        # 创建名为 PartR 的刚体对象，表示部件 R，并指定质量为 m
        PartR = Body('R', mass=m)
    
    # 创建名为 J1 的销钉连接（PinJoint），连接 C 和 PartP，指定速度为 u1，坐标为 q1，
    # 子点位于 -l*PartP.x 处，关节轴为 C.z 方向
    J1 = PinJoint('J1', C, PartP, speeds=u1, coordinates=q1,
                  child_point=-l*PartP.x, joint_axis=C.z)
    
    # 创建名为 J2 的销钉连接（PinJoint），连接 PartP 和 PartR，指定速度为 u2，坐标为 q2，
    # 子点位于 -l*PartR.x 处，关节轴为 PartP.z 方向
    J2 = PinJoint('J2', PartP, PartR, speeds=u2, coordinates=q2,
                  child_point=-l*PartR.x, joint_axis=PartP.z)

    # 在 PartP 上施加向下方向的力，大小为 m*g
    PartP.apply_force(m*g*C.x)
    
    # 在 PartR 上施加向下方向的力，大小为 m*g
    PartR.apply_force(m*g*C.x)

    # 在使用 warns_deprecated_sympy 上下文管理器中，执行下列语句时会发出关于 sympy 的弃用警告
    with warns_deprecated_sympy():
        # 创建一个名为 method 的连接方法对象，连接 C、J1 和 J2
        method = JointsMethod(C, J1, J2)
    
    # 计算并设置连接方法的运动方程
    method.form_eoms()

    # 断言连接方法的完整质量矩阵与给定的矩阵相等
    assert expand(method.mass_matrix_full) == Matrix([[1, 0, 0, 0],
                                                      [0, 1, 0, 0],
                                                      [0, 0, 2*l**2*m*cos(q2) + 3*l**2*m, l**2*m*cos(q2) + l**2*m],
                                                      [0, 0, l**2*m*cos(q2) + l**2*m, l**2*m]])
    
    # 断言连接方法的完整强迫矩阵经过简化后与给定的矩阵经过简化后相等
    assert trigsimp(method.forcing_full) == trigsimp(Matrix([[u1], [u2], [-g*l*m*(sin(q1 + q2) + sin(q1)) -
                                           g*l*m*sin(q1) + l**2*m*(2*u1 + u2)*u2*sin(q2)],
                                          [-g*l*m*sin(q1 + q2) - l**2*m*u1**2*sin(q2)]]))
def test_two_dof_joints():
    # 定义动力学符号变量
    q1, q2, u1, u2 = dynamicsymbols('q1 q2 u1 u2')
    # 定义物理参数符号变量
    m, c1, c2, k1, k2 = symbols('m c1 c2 k1 k2')
    # 在使用即将过时的警告环境中创建物体
    with warns_deprecated_sympy():
        W = Body('W')
        B1 = Body('B1', mass=m)
        B2 = Body('B2', mass=m)
    # 创建关节对象 J1 和 J2，分别是直动关节
    J1 = PrismaticJoint('J1', W, B1, coordinates=q1, speeds=u1)
    J2 = PrismaticJoint('J2', B1, B2, coordinates=q2, speeds=u2)
    # 给物体 W 施加力
    W.apply_force(k1*q1*W.x, reaction_body=B1)
    W.apply_force(c1*u1*W.x, reaction_body=B1)
    # 给物体 B1 施加力
    B1.apply_force(k2*q2*W.x, reaction_body=B2)
    B1.apply_force(c2*u2*W.x, reaction_body=B2)
    # 在使用即将过时的警告环境中创建关节方法对象
    with warns_deprecated_sympy():
        method = JointsMethod(W, J1, J2)
    # 形成运动方程
    method.form_eoms()
    # 获取质量矩阵和外力向量
    MM = method.mass_matrix
    forcing = method.forcing
    # 解运动方程得到右手边的结果
    rhs = MM.LUsolve(forcing)
    # 断言第一个方程展开后等于给定的表达式
    assert expand(rhs[0]) == expand((-k1 * q1 - c1 * u1 + k2 * q2 + c2 * u2)/m)
    # 断言第二个方程展开后等于给定的表达式
    assert expand(rhs[1]) == expand((k1 * q1 + c1 * u1 - 2 * k2 * q2 - 2 * c2 * u2) / m)

def test_simple_pedulum():
    # 定义符号变量
    l, m, g = symbols('l m g')
    # 在使用即将过时的警告环境中创建物体
    with warns_deprecated_sympy():
        C = Body('C')
        b = Body('b', mass=m)
    # 定义动态符号变量
    q = dynamicsymbols('q')
    # 创建固定关节 P
    P = PinJoint('P', C, b, speeds=q.diff(t), coordinates=q,
                 child_point=-l * b.x, joint_axis=C.z)
    # 设置物体 b 的势能
    b.potential_energy = - m * g * l * cos(q)
    # 在使用即将过时的警告环境中创建关节方法对象
    with warns_deprecated_sympy():
        method = JointsMethod(C, P)
    # 形成运动方程
    method.form_eoms(LagrangesMethod)
    # 获取右手边的结果
    rhs = method.rhs()
    # 断言结果符合预期
    assert rhs[1] == -g*sin(q)/l

def test_chaos_pendulum():
    # 定义符号变量
    mA, mB, lA, lB, IAxx, IBxx, IByy, IBzz, g = symbols('mA, mB, lA, lB, IAxx, IBxx, IByy, IBzz, g')
    # 定义动态符号变量
    theta, phi, omega, alpha = dynamicsymbols('theta phi omega alpha')

    # 创建参考框架 A 和 B
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')

    # 在使用即将过时的警告环境中创建物体
    with warns_deprecated_sympy():
        rod = Body('rod', mass=mA, frame=A,
                   central_inertia=inertia(A, IAxx, IAxx, 0))
        plate = Body('plate', mass=mB, frame=B,
                     central_inertia=inertia(B, IBxx, IByy, IBzz))
        C = Body('C')
    # 创建固定关节 J1 和 J2
    J1 = PinJoint('J1', C, rod, coordinates=theta, speeds=omega,
                  child_point=-lA * rod.z, joint_axis=C.y)
    J2 = PinJoint('J2', rod, plate, coordinates=phi, speeds=alpha,
                  parent_point=(lB - lA) * rod.z, joint_axis=rod.z)

    # 给物体施加力
    rod.apply_force(mA*g*C.z)
    plate.apply_force(mB*g*C.z)

    # 在使用即将过时的警告环境中创建关节方法对象
    with warns_deprecated_sympy():
        method = JointsMethod(C, J1, J2)
    # 形成运动方程
    method.form_eoms()

    # 获取质量矩阵和外力向量
    MM = method.mass_matrix
    forcing = method.forcing
    # 解运动方程得到右手边的结果
    rhs = MM.LUsolve(forcing)

    # 断言第一个方程的简化结果等于给定的表达式
    xd = (-2 * IBxx * alpha * omega * sin(phi) * cos(phi) + 2 * IByy * alpha * omega * sin(phi) *
            cos(phi) - g * lA * mA * sin(theta) - g * lB * mB * sin(theta)) / (IAxx + IBxx *
                sin(phi)**2 + IByy * cos(phi)**2 + lA**2 * mA + lB**2 * mB)
    assert (rhs[0] - xd).simplify() == 0

    # 断言第二个方程的简化结果等于给定的表达式
    xd = (IBxx - IByy) * omega**2 * sin(phi) * cos(phi) / IBzz
    assert (rhs[1] - xd).simplify() == 0
# 定义一个测试函数，用于测试四连杆机构，并包含手动设置的约束条件
def test_four_bar_linkage_with_manual_constraints():
    # 定义动力学符号变量 q1, q2, q3 和其对应的速度变量 u1, u2, u3
    q1, q2, q3, u1, u2, u3 = dynamicsymbols('q1:4, u1:4')
    # 定义长度符号变量 l1, l2, l3, l4 和质量密度 rho
    l1, l2, l3, l4, rho = symbols('l1:5, rho')

    # 创建参考坐标系 N
    N = ReferenceFrame('N')
    # 根据长度符号创建惯性张量列表
    inertias = [inertia(N, 0, 0, rho * l ** 3 / 12) for l in (l1, l2, l3, l4)]

    # 使用警告装饰器，创建四个连杆对象 link1, link2, link3, link4
    with warns_deprecated_sympy():
        link1 = Body('Link1', frame=N, mass=rho * l1,
                     central_inertia=inertias[0])
        link2 = Body('Link2', mass=rho * l2, central_inertia=inertias[1])
        link3 = Body('Link3', mass=rho * l3, central_inertia=inertias[2])
        link4 = Body('Link4', mass=rho * l4, central_inertia=inertias[3])

    # 定义三个销轴关节对象 joint1, joint2, joint3
    joint1 = PinJoint(
        'J1', link1, link2, coordinates=q1, speeds=u1, joint_axis=link1.z,
        parent_point=l1 / 2 * link1.x, child_point=-l2 / 2 * link2.x)
    joint2 = PinJoint(
        'J2', link2, link3, coordinates=q2, speeds=u2, joint_axis=link2.z,
        parent_point=l2 / 2 * link2.x, child_point=-l3 / 2 * link3.x)
    joint3 = PinJoint(
        'J3', link3, link4, coordinates=q3, speeds=u3, joint_axis=link3.z,
        parent_point=l3 / 2 * link3.x, child_point=-l4 / 2 * link4.x)

    # 计算四连杆机构的回路约束
    loop = link4.masscenter.pos_from(link1.masscenter) \
           + l1 / 2 * link1.x + l4 / 2 * link4.x

    # 构建代表回路约束的矩阵 fh
    fh = Matrix([loop.dot(link1.x), loop.dot(link1.y)])

    # 使用警告装饰器，创建连接方法对象 method
    with warns_deprecated_sympy():
        method = JointsMethod(link1, joint1, joint2, joint3)

    # 获取时间变量
    t = dynamicsymbols._t
    # 解决方法中的运动学微分方程，得到广义速度的导数 qdots
    qdots = solve(method.kdes, [q1.diff(t), q2.diff(t), q3.diff(t)])
    # 计算 fh 对时间的导数，并在其中代入 qdots
    fhd = fh.diff(t).subs(qdots)

    # 创建 KanesMethod 对象 kane，进行卡宁方程的建立
    kane = KanesMethod(method.frame, q_ind=[q1], u_ind=[u1],
                       q_dependent=[q2, q3], u_dependent=[u2, u3],
                       kd_eqs=method.kdes, configuration_constraints=fh,
                       velocity_constraints=fhd, forcelist=method.loads,
                       bodies=method.bodies)
    # 求解卡宁方程并断言其结果为零矩阵
    fr, frs = kane.kanes_equations()
    assert fr == zeros(1)

    # 对质量-矩阵和力-矩阵进行数值检查
    p = Matrix([l1, l2, l3, l4, rho])
    q = Matrix([q1, q2, q3])
    u = Matrix([u1, u2, u3])
    eval_m = lambdify((q, p), kane.mass_matrix)
    eval_f = lambdify((q, u, p), kane.forcing)
    eval_fhd = lambdify((q, u, p), fhd)

    # 定义检查的参数值
    p_vals = [0.13, 0.24, 0.21, 0.34, 997]
    q_vals = [2.1, 0.6655470375077588, 2.527408138024188]  # 满足 fh 的值
    u_vals = [0.2, -0.17963733938852067, 0.1309060540601612]  # 满足 fhd 的值
    mass_check = Matrix([[3.452709815256506e+01, 7.003948798374735e+00,
                          -4.939690970641498e+00],
                         [-2.203792703880936e-14, 2.071702479957077e-01,
                          2.842917573033711e-01],
                         [-1.300000000000123e-01, -8.836934896046506e-03,
                          1.864891330060847e-01]])
    forcing_check = Matrix([[-0.031211821321648],
                            [-0.00066022608181],
                            [0.001813559741243]])
    eps = 1e-10
    # 断言所有 fhd 计算结果的绝对值都小于 eps
    assert all(abs(x) < eps for x in eval_fhd(q_vals, u_vals, p_vals))
    # 断言：验证矩阵的每个元素绝对值是否小于给定的误差阈值 eps
    assert all(abs(x) < eps for x in
               (Matrix(eval_m(q_vals, p_vals)) - mass_check))
    
    # 断言：验证矩阵的每个元素绝对值是否小于给定的误差阈值 eps
    assert all(abs(x) < eps for x in
               (Matrix(eval_f(q_vals, u_vals, p_vals)) - forcing_check))
```