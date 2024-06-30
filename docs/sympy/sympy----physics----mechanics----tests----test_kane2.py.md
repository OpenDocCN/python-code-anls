# `D:\src\scipysrc\sympy\sympy\physics\mechanics\tests\test_kane2.py`

```
# 导入所需的符号计算模块和函数
from sympy import cos, Matrix, sin, zeros, tan, pi, symbols
from sympy.simplify.simplify import simplify
from sympy.simplify.trigsimp import trigsimp
from sympy.solvers.solvers import solve
from sympy.physics.mechanics import (cross, dot, dynamicsymbols,
                                     find_dynamicsymbols, KanesMethod, inertia,
                                     inertia_of_point_mass, Point,
                                     ReferenceFrame, RigidBody)

# 定义一个测试函数，用于测试辅助速度和相关依赖
def test_aux_dep():
    # 此测试涉及滚动圆盘的动力学，将KanesMethod得到的结果与手动推导的结果进行比较。
    # 比较了Fr、Fr*和Fr*_steady这几个术语在两种方法中的结果。这里的Fr*_steady指的是静态配置下的广义惯性力。
    # 注意：与test_kane.py中的test_rolling_disc()测试相比，此测试还测试了辅助速度、配置和运动约束，
    # 见于广义依赖坐标q[3]和依赖速度u[3]、u[4]和u[5]。

    # 首先，手动推导Fr、Fr_star、Fr_star_steady。

    # 时间和常数参数的符号表示。
    # 接触力的符号：Fx、Fy、Fz。
    t, r, m, g, I, J = symbols('t r m g I J')
    Fx, Fy, Fz = symbols('Fx Fy Fz')

    # 配置变量及其时间导数：
    # q[0] -- 偏航角
    # q[1] -- 倾斜角
    # q[2] -- 自旋角速度
    # q[3] -- dot(-r*B.z, A.z)，A.z方向上到圆盘中心的地面距离
    # 广义速度及其时间导数：
    # u[0] -- 圆盘角速度分量，圆盘固定x方向
    # u[1] -- 圆盘角速度分量，圆盘固定y方向
    # u[2] -- 圆盘角速度分量，圆盘固定z方向
    # u[3] -- 圆盘速度分量，A.x方向
    # u[4] -- 圆盘速度分量，A.y方向
    # u[5] -- 圆盘速度分量，A.z方向
    # 辅助广义速度：
    # ua[0] -- 接触点辅助广义速度，A.x方向
    # ua[1] -- 接触点辅助广义速度，A.y方向
    # ua[2] -- 接触点辅助广义速度，A.z方向
    q = dynamicsymbols('q:4')
    qd = [qi.diff(t) for qi in q]
    u = dynamicsymbols('u:6')
    ud = [ui.diff(t) for ui in u]
    ud_zero = dict(zip(ud, [0.] * len(ud)))
    ua = dynamicsymbols('ua:3')
    ua_zero = dict(zip(ua, [0.] * len(ua)))  # noqa:F841

    # 参考坐标系：
    # 偏航中间坐标系：A。
    # 倾斜中间坐标系：B。
    # 圆盘固定坐标系：C。
    N = ReferenceFrame('N')
    A = N.orientnew('A', 'Axis', [q[0], N.z])
    B = A.orientnew('B', 'Axis', [q[1], A.x])
    C = B.orientnew('C', 'Axis', [q[2], B.y])

    # 圆盘固定坐标系的角速度和角加速度
    # u[0]、u[1]和u[2]为广义独立速度。
    C.set_ang_vel(N, u[0]*B.x + u[1]*B.y + u[2]*B.z)
    # Calculate angular acceleration of point C in frame N using:
    # angular velocity of C in N differentiated with respect to time in frame B,
    # added to the cross product of angular velocity of B in N and angular velocity of C in N.
    C.set_ang_acc(N, C.ang_vel_in(N).diff(t, B)
                   + cross(B.ang_vel_in(N), C.ang_vel_in(N)))

    # Velocity and acceleration definitions for points:
    # Point P: contact point between disc and ground.
    # Point O: center of the disc, defined relative to P with coordinate q[3].
    # Generalized speeds ua[0], ua[1], ua[2] define the velocity of P in frame N.
    P = Point('P')
    P.set_vel(N, ua[0]*A.x + ua[1]*A.y + ua[2]*A.z)
    O = P.locatenew('O', q[3]*A.z + r*sin(q[1])*A.y)
    O.set_vel(N, u[3]*A.x + u[4]*A.y + u[5]*A.z)
    O.set_acc(N, O.vel(N).diff(t, A) + cross(A.ang_vel_in(N), O.vel(N)))

    # Kinematic differential equations:
    # w_c_n_qd represents the angular velocity of C in N expressed in coordinates of B.
    # v_o_n_qd represents the velocity of O in N, specifically in the A.z direction.
    # kindiffs is a vector of two equations derived from w_c_n_qd and v_o_n_qd,
    # solving for the time derivatives of the generalized coordinates qd in terms of the speeds u.
    w_c_n_qd = qd[0]*A.z + qd[1]*B.x + qd[2]*B.y
    v_o_n_qd = O.pos_from(P).diff(t, A) + cross(A.ang_vel_in(N), O.pos_from(P))
    kindiffs = Matrix([dot(w_c_n_qd - C.ang_vel_in(N), uv) for uv in B] +
                      [dot(v_o_n_qd - O.vel(N), A.z)])
    qd_kd = solve(kindiffs, qd) # noqa:F841

    # Values of generalized speeds during a steady turn for substitution into Fr_star_steady.
    steady_conditions = solve(kindiffs.subs({qd[1] : 0, qd[3] : 0}), u)
    steady_conditions.update({qd[1] : 0, qd[3] : 0})

    # Partial angular velocities and velocities derivatives.
    partial_w_C = [C.ang_vel_in(N).diff(ui, N) for ui in u + ua]
    partial_v_O = [O.vel(N).diff(ui, N) for ui in u + ua]
    partial_v_P = [P.vel(N).diff(ui, N) for ui in u + ua]

    # Configuration constraint f_c: constrains the projection of radius r in the A.z direction to q[3].
    # Velocity constraints f_v: constraints on u3, u4, and u5.
    # Acceleration constraints f_a: constraints on acceleration of O.
    f_c = Matrix([dot(-r*B.z, A.z) - q[3]])
    f_v = Matrix([dot(O.vel(N) - (P.vel(N) + cross(C.ang_vel_in(N),
        O.pos_from(P))), ai).expand() for ai in A])
    v_o_n = cross(C.ang_vel_in(N), O.pos_from(P))
    a_o_n = v_o_n.diff(t, A) + cross(A.ang_vel_in(N), v_o_n)
    f_a = Matrix([dot(O.acc(N) - a_o_n, ai) for ai in A]) # noqa:F841

    # Solve for constraint equations: M_v * [u; ua] = 0
    # M_v is the constraint coefficient matrix, derived from f_v.
    # A_rs is the matrix solving for dependent speeds u_dep based on independent speeds u_i.
    M_v = zeros(3, 9)
    for i in range(3):
        for j, ui in enumerate(u + ua):
            M_v[i, j] = f_v[i].diff(ui)

    M_v_i = M_v[:, :3]
    M_v_d = M_v[:, 3:6]
    M_v_aux = M_v[:, 6:]
    M_v_i_aux = M_v_i.row_join(M_v_aux)
    A_rs = - M_v_d.inv() * M_v_i_aux

    # Calculate dependent speeds u_dep based on A_rs and independent speeds u[:3].
    u_dep = A_rs[:, :3] * Matrix(u[:3])
    # 创建字典 u_dep_dict，将列表 u[3:] 和 u_dep 中的对应元素配对
    u_dep_dict = dict(zip(u[3:], u_dep))

    # 计算活动力 F_O 和 F_P
    F_O = m*g*A.z  # 点 O 上的重力
    F_P = Fx * A.x + Fy * A.y + Fz * A.z  # 在点 P 上的合力

    # 计算广义活动力 Fr_u（无约束情况下）
    Fr_u = Matrix([dot(F_O, pv_o) + dot(F_P, pv_p) for pv_o, pv_p in
            zip(partial_v_O, partial_v_P)])

    # 计算惯性力 R_star_O
    R_star_O = -m*O.acc(N)  # 点 O 上的惯性力

    # 计算盘的惯性 I_C_O
    I_C_O = inertia(B, I, J, I)  # 关于主轴的盘的惯性

    # 计算惯性扭矩 T_star_C
    T_star_C = -(dot(I_C_O, C.ang_acc_in(N)) \
                 + cross(C.ang_vel_in(N), dot(I_C_O, C.ang_vel_in(N))))
    
    # 计算广义惯性力 Fr_star_u（无约束情况下）
    Fr_star_u = Matrix([dot(R_star_O, pv) + dot(T_star_C, pav) for pv, pav in
                        zip(partial_v_O, partial_w_C)])

    # 形成非完整约束力 Fr_c 和非完整约束惯性力 Fr_star_c
    Fr_c = Fr_u[:3, :].col_join(Fr_u[6:, :]) + A_rs.T * Fr_u[3:6, :]
    Fr_star_c = Fr_star_u[:3, :].col_join(Fr_star_u[6:, :]) \
                + A_rs.T * Fr_star_u[3:6, :]

    # 计算稳态转向条件下的非完整约束惯性力 Fr_star_steady
    Fr_star_steady = Fr_star_c.subs(ud_zero).subs(u_dep_dict) \
            .subs(steady_conditions).subs({q[3]: -r*cos(q[1])}).expand()

    # 创建刚体 disc，其惯性为 I_C_O
    iner_tuple = (I_C_O, O)
    disc = RigidBody('disc', O, C, m, iner_tuple)
    bodyList = [disc]

    # 定义广义力 F_o 和辅助力 F_p
    F_o = (O, F_O)
    F_p = (P, F_P)
    forceList = [F_o,  F_p]

    # 创建 KanesMethod 实例 kane，用于力学分析
    kane = KanesMethod(
        N, q_ind= q[:3], u_ind= u[:3], kd_eqs=kindiffs,
        q_dependent=q[3:], configuration_constraints = f_c,
        u_dependent=u[3:], velocity_constraints= f_v,
        u_auxiliary=ua
        )

    # 计算 fr, frstar 和 frstar_steady 以及 kdd（运动微分方程）
    (fr, frstar)= kane.kanes_equations(bodyList, forceList)
    frstar_steady = frstar.subs(ud_zero).subs(u_dep_dict).subs(steady_conditions) \
                    .subs({q[3]: -r*cos(q[1])}).expand()
    kdd = kane.kindiffdict()

    # 断言对比计算得到的力矩和对应的分析结果
    assert Matrix(Fr_c).expand() == fr.expand()
    assert Matrix(Fr_star_c.subs(kdd)).expand() == frstar.expand()
    
    # 使用 SymEngine 运行时，确保对比 Fr_star_steady 的结果为零时使用不同类型的零
    assert (simplify(Matrix(Fr_star_steady).expand()).xreplace({0:0.0}) ==
            simplify(frstar_steady.expand()).xreplace({0:0.0}))

    # 检查动力学表达式中是否包含广义速度符号
    syms_in_forcing = find_dynamicsymbols(kane.forcing)
    for qdi in qd:
        assert qdi not in syms_in_forcing
# 定义一个测试函数，用于验证刚体的非集中惯性的计算
def test_non_central_inertia():
    # This tests that the calculation of Fr* does not depend the point
    # about which the inertia of a rigid body is defined. This test solves
    # exercises 8.12, 8.17 from Kane 1985.

    # 声明符号变量
    q1, q2, q3 = dynamicsymbols('q1:4')
    q1d, q2d, q3d = dynamicsymbols('q1:4', level=1)
    u1, u2, u3, u4, u5 = dynamicsymbols('u1:6')
    u_prime, R, M, g, e, f, theta = symbols('u\' R, M, g, e, f, theta')
    a, b, mA, mB, IA, J, K, t = symbols('a b mA mB IA J K t')
    Q1, Q2, Q3 = symbols('Q1, Q2 Q3')
    IA22, IA23, IA33 = symbols('IA22 IA23 IA33')

    # 参考坐标系
    F = ReferenceFrame('F')
    # 基于F坐标系，定义一个新的P坐标系，其旋转轴为F坐标系的-y轴，角度为-theta
    P = F.orientnew('P', 'axis', [-theta, F.y])
    # 在P坐标系下，定义A坐标系，其旋转轴为P坐标系的x轴，角度为q1
    A = P.orientnew('A', 'axis', [q1, P.x])
    # 设置A坐标系相对于F坐标系的角速度
    A.set_ang_vel(F, u1*A.x + u3*A.z)

    # 定义轮子的坐标系B和C，它们相对于A坐标系绕其z轴旋转角度分别为q2和q3
    B = A.orientnew('B', 'axis', [q2, A.z])
    C = A.orientnew('C', 'axis', [q3, A.z])
    # 设置B坐标系相对于A坐标系的角速度
    B.set_ang_vel(A, u4 * A.z)
    # 设置C坐标系相对于A坐标系的角速度
    C.set_ang_vel(A, u5 * A.z)

    # 在A坐标系下定义点D, S*, Q及其速度
    pD = Point('D')
    # 设置点D在A坐标系中的速度为0
    pD.set_vel(A, 0)
    # 设置点D在F坐标系中的速度，假设轮子仍然滚动但不打滑
    pD.set_vel(F, u2 * A.y)

    # 定义点S*, Q相对于点D在A坐标系中的位置，并计算它们相对于F坐标系的速度
    pS_star = pD.locatenew('S*', e*A.y)
    pQ = pD.locatenew('Q', f*A.y - R*A.x)
    for p in [pS_star, pQ]:
        p.v2pt_theory(pD, F, A)

    # 定义质心A*, B*, C*分别相对于点D在A坐标系中的位置，并计算它们相对于F坐标系的速度
    pA_star = pD.locatenew('A*', a*A.y)
    pB_star = pD.locatenew('B*', b*A.z)
    pC_star = pD.locatenew('C*', -b*A.z)
    for p in [pA_star, pB_star, pC_star]:
        p.v2pt_theory(pD, F, A)

    # 定义轮子B, C接触平面P上的点B^, C^及其相对于F坐标系的速度
    pB_hat = pB_star.locatenew('B^', -R*A.x)
    pC_hat = pC_star.locatenew('C^', -R*A.x)
    pB_hat.v2pt_theory(pB_star, F, B)
    pC_hat.v2pt_theory(pC_star, F, C)

    # 由于假设轮子B, C滚动而不打滑，因此点B^, C^的速度为零
    kde = [q1d - u1, q2d - u4, q3d - u5]
    vc = [dot(p.vel(F), A.y) for p in [pB_hat, pC_hat]]

    # 定义刚体A, B, C的惯性
    # IA22, IA23, IA33在问题陈述中未指定，但是需要定义惯性对象。
    # 虽然IA22, IA23, IA33的值未知，但它们并未出现在一般的惯性项中。
    inertia_A = inertia(A, IA, IA22, IA33, 0, IA23, 0)
    inertia_B = inertia(B, K, K, J)
    inertia_C = inertia(C, K, K, J)

    # 定义刚体A, B, C
    rbA = RigidBody('rbA', pA_star, A, mA, (inertia_A, pA_star))
    rbB = RigidBody('rbB', pB_star, B, mB, (inertia_B, pB_star))
    rbC = RigidBody('rbC', pC_star, C, mB, (inertia_C, pC_star))

    # 初始化KanesMethod对象，设置广义坐标q和速度u，以及速度约束和辅助速度u3
    km = KanesMethod(F, q_ind=[q1, q2, q3], u_ind=[u1, u2], kd_eqs=kde,
                     u_dependent=[u4, u5], velocity_constraints=vc,
                     u_auxiliary=[u3])

    # 定义作用力和刚体列表
    forces = [(pS_star, -M*g*F.x), (pQ, Q1*A.x + Q2*A.y + Q3*A.z)]
    bodies = [rbA, rbB, rbC]
    # 计算广义动力学方程Fr和速度相关的Fr*
    fr, fr_star = km.kanes_equations(bodies, forces)
    # 使用 solve 函数求解 vc_map，将结果保存到 vc_map 变量中
    vc_map = solve(vc, [u4, u5])

    # 计算并定义 KanesMethod 返回的 Fr*，即 Kane1985 中定义的 -Fr
    fr_star_expected = Matrix([
            -(IA + 2*J*b**2/R**2 + 2*K +
              mA*a**2 + 2*mB*b**2) * u1.diff(t) - mA*a*u1*u2,
            -(mA + 2*mB +2*J/R**2) * u2.diff(t) + mA*a*u1**2,
            0])
    
    # 对 fr_star 应用 vc_map 的替换，并将 u3 替换为 0，简化表达式并展开
    t = trigsimp(fr_star.subs(vc_map).subs({u3: 0})).doit().expand()
    
    # 断言 fr_star_expected 减去 t 扩展后等于零向量
    assert ((fr_star_expected - t).expand() == zeros(3, 1))

    # 定义刚体 A、B、C 关于点 D 的惯性
    # I_S/O = I_S/S* + I_S*/O
    bodies2 = []
    for rb, I_star in zip([rbA, rbB, rbC], [inertia_A, inertia_B, inertia_C]):
        # 计算刚体的总惯性，包括质点的惯性
        I = I_star + inertia_of_point_mass(rb.mass,
                                           rb.masscenter.pos_from(pD),
                                           rb.frame)
        # 将刚体信息添加到 bodies2 列表中
        bodies2.append(RigidBody('', rb.masscenter, rb.frame, rb.mass,
                                 (I, pD)))
    
    # 计算 KanesMethod 的方程中的 Fr 和 Fr*
    fr2, fr_star2 = km.kanes_equations(bodies2, forces)

    # 对 fr_star2 应用 vc_map 的替换，并将 u3 替换为 0，简化表达式
    t = trigsimp(fr_star2.subs(vc_map).subs({u3: 0})).doit()
    
    # 断言 fr_star_expected 减去 t 等于零向量
    assert (fr_star_expected - t).expand() == zeros(3, 1)


这段代码主要进行了以下操作：

1. 使用 solve 函数求解并保存 vc_map 变量。
2. 计算并定义了 KanesMethod 返回的 Fr*，即 Kane1985 中定义的负 Fr。
3. 对两个 fr_star 表达式应用 vc_map 的替换，并进行必要的简化和展开。
4. 使用断言确保计算出的表达式与预期的零向量相等，以验证代码的正确性。
def test_sub_qdot():
    # This test function deals with solving exercises 8.12, 8.17 from Kane 1985
    # and defines velocities in terms of q, qdot.

    ## --- Declare symbols ---
    # Declare generalized coordinates q1, q2, q3 and their time derivatives q1d, q2d, q3d
    q1, q2, q3 = dynamicsymbols('q1:4')
    q1d, q2d, q3d = dynamicsymbols('q1:4', level=1)
    # Declare generalized speeds u1, u2, u3
    u1, u2, u3 = dynamicsymbols('u1:4')
    # Declare additional symbols related to the problem
    u_prime, R, M, g, e, f, theta = symbols('u\' R, M, g, e, f, theta')
    a, b, mA, mB, IA, J, K, t = symbols('a b mA mB IA J K t')
    IA22, IA23, IA33 = symbols('IA22 IA23 IA33')
    Q1, Q2, Q3 = symbols('Q1 Q2 Q3')

    # --- Reference Frames ---
    # Define a reference frame F
    F = ReferenceFrame('F')
    # Orient frame P relative to F with a rotation about F's y-axis by -theta radians
    P = F.orientnew('P', 'axis', [-theta, F.y])
    # Orient frame A relative to P with a rotation about P's x-axis by q1 radians
    A = P.orientnew('A', 'axis', [q1, P.x])
    # Set angular velocity of A relative to F
    A.set_ang_vel(F, u1*A.x + u3*A.z)
    # Define frames B and C relative to A with rotations about A's z-axis by q2 and q3 radians respectively
    B = A.orientnew('B', 'axis', [q2, A.z])
    C = A.orientnew('C', 'axis', [q3, A.z])

    ## --- define points D, S*, Q on frame A and their velocities ---
    # Define point D and set its velocity in frame A and F
    pD = Point('D')
    pD.set_vel(A, 0)
    pD.set_vel(F, u2 * A.y)  # Velocity of D in frame F

    # Define points S*, Q, A*, B*, C* relative to point D in frame A
    pS_star = pD.locatenew('S*', e*A.y)
    pQ = pD.locatenew('Q', f*A.y - R*A.x)
    pA_star = pD.locatenew('A*', a*A.y)
    pB_star = pD.locatenew('B*', b*A.z)
    pC_star = pD.locatenew('C*', -b*A.z)
    # Relate velocities of these points to frame F
    for p in [pS_star, pQ, pA_star, pB_star, pC_star]:
        p.v2pt_theory(pD, F, A)

    # Define points B^, C^ touching the plane P and their velocities in frame F
    pB_hat = pB_star.locatenew('B^', -R*A.x)
    pC_hat = pC_star.locatenew('C^', -R*A.x)
    pB_hat.v2pt_theory(pB_star, F, B)
    pC_hat.v2pt_theory(pC_star, F, C)

    # --- relate qdot, u ---
    # Formulate kinematic differential equations (kde) involving velocities of B^, C^ and u1
    kde = [dot(p.vel(F), A.y) for p in [pB_hat, pC_hat]]
    kde += [u1 - q1d]  # Additional kinematic differential equation
    # Solve for qdot (q1d, q2d, q3d) in terms of u1, u2, u3
    kde_map = solve(kde, [q1d, q2d, q3d])
    # Extend the solution map to include time derivatives
    for k, v in list(kde_map.items()):
        kde_map[k.diff(t)] = v.diff(t)

    # Inertias of bodies A, B, C defined using given symbols
    inertia_A = inertia(A, IA, IA22, IA33, 0, IA23, 0)
    inertia_B = inertia(B, K, K, J)
    inertia_C = inertia(C, K, K, J)

    # Define rigid bodies rbA, rbB, rbC with their respective properties
    rbA = RigidBody('rbA', pA_star, A, mA, (inertia_A, pA_star))
    rbB = RigidBody('rbB', pB_star, B, mB, (inertia_B, pB_star))
    rbC = RigidBody('rbC', pC_star, C, mB, (inertia_C, pC_star))

    ## --- use Kane's method ---
    # Initialize Kane's Method object km with frame F, generalized coordinates q1, q2, q3,
    # generalized speeds u1, u2, kinematic differential equations kde, auxiliary speed u3
    km = KanesMethod(F, [q1, q2, q3], [u1, u2], kd_eqs=kde, u_auxiliary=[u3])

    # Define external forces and moments on points pS_star and pQ
    forces = [(pS_star, -M*g*F.x), (pQ, Q1*A.x + Q2*A.y + Q3*A.z)]
    bodies = [rbA, rbB, rbC]

    # Equations related to forces and moments acting on the system
    # Q2 = -u_prime * u2 * Q1 / sqrt(u2**2 + f**2 * u1**2)
    # -u_prime * R * u2 / sqrt(u2**2 + f**2 * u1**2) = R / Q1 * Q2
    # 定义预期的广义力矢量 fr_expected
    fr_expected = Matrix([
            # 第一行：计算第一行的每个项
            f*Q3 + M*g*e*sin(theta)*cos(q1),
            # 第二行：计算第二行的每个项
            Q2 + M*g*sin(theta)*sin(q1),
            # 第三行：计算第三行的每个项
            e*M*g*cos(theta) - Q1*f - Q2*R])
             #Q1 * (f - u_prime * R * u2 / sqrt(u2**2 + f**2 * u1**2)))])
    # 定义预期的广义速度矢量 fr_star_expected
    fr_star_expected = Matrix([
            # 第一行：计算第一行的每个项
            -(IA + 2*J*b**2/R**2 + 2*K +
              mA*a**2 + 2*mB*b**2) * u1.diff(t) - mA*a*u1*u2,
            # 第二行：计算第二行的每个项
            -(mA + 2*mB +2*J/R**2) * u2.diff(t) + mA*a*u1**2,
            # 第三行：零向量
            0])

    # 计算系统的广义力和广义速度
    fr, fr_star = km.kanes_equations(bodies, forces)
    # 断言：广义力矢量等于预期值
    assert (fr.expand() == fr_expected.expand())
    # 断言：广义速度矢量经过简化后等于预期值的零向量
    assert ((fr_star_expected - trigsimp(fr_star)).expand() == zeros(3, 1))
def test_sub_qdot2():
    # This test solves exercises 8.3 from Kane 1985 and defines
    # all velocities in terms of q, qdot. We check that the generalized active
    # forces are correctly computed if u terms are only defined in the
    # kinematic differential equations.
    #
    # This functionality was added in PR 8948. Without qdot/u substitution, the
    # KanesMethod constructor will fail during the constraint initialization as
    # the B matrix will be poorly formed and inversion of the dependent part
    # will fail.

    g, m, Px, Py, Pz, R, t = symbols('g m Px Py Pz R t')
    q = dynamicsymbols('q:5')  # Define generalized coordinates q1 to q5
    qd = dynamicsymbols('q:5', level=1)  # Define their first time derivatives qdot1 to qdot5
    u = dynamicsymbols('u:5')  # Define symbolic variables u1 to u5

    ## Define inertial, intermediate, and rigid body reference frames
    A = ReferenceFrame('A')  # Inertial frame A
    B_prime = A.orientnew('B_prime', 'Axis', [q[0], A.z])  # Intermediate frame B' rotated by q0 around A's z-axis
    B = B_prime.orientnew('B', 'Axis', [pi/2 - q[1], B_prime.x])  # Rigid body frame B rotated by (pi/2 - q1) around B'x
    C = B.orientnew('C', 'Axis', [q[2], B.z])  # Rigid body frame C rotated by q2 around B's z-axis

    ## Define points of interest and their velocities
    pO = Point('O')  # Point O
    pO.set_vel(A, 0)  # Velocity of O in frame A is zero

    # R is the point in plane H that comes into contact with disk C.
    pR = pO.locatenew('R', q[3]*A.x + q[4]*A.y)  # Point R located relative to O
    pR.set_vel(A, pR.pos_from(pO).diff(t, A))  # Velocity of R in frame A
    pR.set_vel(B, 0)  # Velocity of R in frame B is zero

    # C^ is the point in disk C that comes into contact with plane H.
    pC_hat = pR.locatenew('C^', 0)  # Point C^ located relative to R
    pC_hat.set_vel(C, 0)  # Velocity of C^ in frame C is zero

    # C* is the point at the center of disk C.
    pCs = pC_hat.locatenew('C*', R*B.y)  # Point C* located relative to C^
    pCs.set_vel(C, 0)  # Velocity of C* in frame C is zero
    pCs.set_vel(B, 0)  # Velocity of C* in frame B is zero

    # calculate velocities of points C* and C^ in frame A
    pCs.v2pt_theory(pR, A, B)  # Velocity of C* relative to A using velocity of R relative to A in B
    pC_hat.v2pt_theory(pCs, A, C)  # Velocity of C^ relative to A using velocity of C* relative to A in C

    ## Define forces on each point of the system
    R_C_hat = Px*A.x + Py*A.y + Pz*A.z  # Force on point C^
    R_Cs = -m*g*A.z  #
```