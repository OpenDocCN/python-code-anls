# `D:\src\scipysrc\sympy\sympy\physics\mechanics\tests\test_linearize.py`

```
# 导入 sympy 库中的各个模块和函数
from sympy import symbols, Matrix, cos, sin, atan, sqrt, Rational
from sympy.core.sympify import sympify
from sympy.simplify.simplify import simplify
from sympy.solvers.solvers import solve
from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame, Point,\
    dot, cross, inertia, KanesMethod, Particle, RigidBody, Lagrangian,\
    LagrangesMethod
from sympy.testing.pytest import slow

# 标记该测试函数为“慢速”测试
@slow
def test_linearize_rolling_disc_kane():
    # 时间和常量参数的符号表示
    t, r, m, g, v = symbols('t r m g v')

    # 位置变量及其时间导数
    q1, q2, q3, q4, q5, q6 = q = dynamicsymbols('q1:7')
    q1d, q2d, q3d, q4d, q5d, q6d = qd = [qi.diff(t) for qi in q]

    # 广义速度及其时间导数
    u = dynamicsymbols('u:6')
    u1, u2, u3, u4, u5, u6 = u = dynamicsymbols('u1:7')
    u1d, u2d, u3d, u4d, u5d, u6d = [ui.diff(t) for ui in u]

    # 参考框架
    N = ReferenceFrame('N')                   # 惯性参考框架
    NO = Point('NO')                          # 惯性参考系原点
    A = N.orientnew('A', 'Axis', [q1, N.z])   # 偏航中间框架
    B = A.orientnew('B', 'Axis', [q2, A.x])   # 倾斜中间框架
    C = B.orientnew('C', 'Axis', [q3, B.y])   # 固定在盘上的框架
    CO = NO.locatenew('CO', q4*N.x + q5*N.y + q6*N.z)      # 盘的中心

    # 盘在 N 参考框架中的角速度，使用坐标时间导数
    w_c_n_qd = C.ang_vel_in(N)
    w_b_n_qd = B.ang_vel_in(N)

    # 盘上固定框架的惯性角速度和角加速度
    C.set_ang_vel(N, u1*B.x + u2*B.y + u3*B.z)

    # 盘心在 N 参考框架中的速度，使用坐标时间导数
    v_co_n_qd = CO.pos_from(NO).dt(N)

    # 盘心在 N 参考框架中的速度，使用广义速度
    CO.set_vel(N, u4*C.x + u5*C.y + u6*C.z)

    # 盘上地面接触点
    P = CO.locatenew('P', r*B.z)
    P.v2pt_theory(CO, N, C)

    # 位置约束
    f_c = Matrix([q6 - dot(CO.pos_from(P), N.z)])

    # 速度约束
    f_v = Matrix([dot(P.vel(N), uv) for uv in C])

    # 运动微分方程
    kindiffs = Matrix([dot(w_c_n_qd - C.ang_vel_in(N), uv) for uv in B] +
                        [dot(v_co_n_qd - CO.vel(N), uv) for uv in N])
    qdots = solve(kindiffs, qd)

    # 设置剩余框架的角速度
    B.set_ang_vel(N, w_b_n_qd.subs(qdots))
    C.set_ang_acc(N, C.ang_vel_in(N).dt(B) + cross(B.ang_vel_in(N), C.ang_vel_in(N)))

    # 主动力
    F_CO = m*g*A.z

    # 创建盘 C 关于点 CO 的惯性 dyadic
    I = (m * r**2) / 4
    J = (m * r**2) / 2
    I_C_CO = inertia(C, I, J, I)

    # 创建刚体 Disc
    Disc = RigidBody('Disc', CO, C, m, (I_C_CO, CO))
    BL = [Disc]
    FL = [(CO, F_CO)]

    # 创建 Kanes 方法对象
    KM = KanesMethod(N, [q1, q2, q3, q4, q5], [u1, u2, u3], kd_eqs=kindiffs,
            q_dependent=[q6], configuration_constraints=f_c,
            u_dependent=[u4, u5, u6], velocity_constraints=f_v)
    (fr, fr_star) = KM.kanes_equations(BL, FL)
    # 使用Kane方法计算Kane方程，将结果分别赋给fr和fr_star变量
    
    # Test generalized form equations
    linearizer = KM.to_linearizer()
    # 使用Kane方法的线性化工具转换为线性化器对象
    
    assert linearizer.f_c == f_c
    # 断言线性化器的f_c属性与f_c变量相等
    assert linearizer.f_v == f_v
    # 断言线性化器的f_v属性与f_v变量相等
    assert linearizer.f_a == f_v.diff(t).subs(KM.kindiffdict())
    # 断言线性化器的f_a属性与f_v关于时间的导数经过KM.kindiffdict()替换后相等
    sol = solve(linearizer.f_0 + linearizer.f_1, qd)
    # 解线性化器的f_0 + f_1等式，求解出qd变量的解
    for qi in qdots.keys():
        assert sol[qi] == qdots[qi]
    # 对于每个qi在qdots的键中，断言其在sol解中的值与qdots中对应的值相等
    assert simplify(linearizer.f_2 + linearizer.f_3 - fr - fr_star) == Matrix([0, 0, 0])
    # 断言线性化器的f_2 + f_3与fr和fr_star之差经过简化后等于零矩阵
    
    # Perform the linearization
    # 执行线性化过程
    
    # Precomputed operating point
    q_op = {q6: -r*cos(q2)}
    # 预先计算的工作点，定义q_op字典
    u_op = {u1: 0,
            u2: sin(q2)*q1d + q3d,
            u3: cos(q2)*q1d,
            u4: -r*(sin(q2)*q1d + q3d)*cos(q3),
            u5: 0,
            u6: -r*(sin(q2)*q1d + q3d)*sin(q3)}
    # 预先计算的控制输入工作点，定义u_op字典
    qd_op = {q2d: 0,
             q4d: -r*(sin(q2)*q1d + q3d)*cos(q1),
             q5d: -r*(sin(q2)*q1d + q3d)*sin(q1),
             q6d: 0}
    # 预先计算的广义速度工作点，定义qd_op字典
    ud_op = {u1d: 4*g*sin(q2)/(5*r) + sin(2*q2)*q1d**2/2 + 6*cos(q2)*q1d*q3d/5,
             u2d: 0,
             u3d: 0,
             u4d: r*(sin(q2)*sin(q3)*q1d*q3d + sin(q3)*q3d**2),
             u5d: r*(4*g*sin(q2)/(5*r) + sin(2*q2)*q1d**2/2 + 6*cos(q2)*q1d*q3d/5),
             u6d: -r*(sin(q2)*cos(q3)*q1d*q3d + cos(q3)*q3d**2)}
    # 预先计算的控制输入速度工作点，定义ud_op字典
    
    A, B = linearizer.linearize(op_point=[q_op, u_op, qd_op, ud_op], A_and_B=True, simplify=True)
    # 使用预先计算的工作点进行线性化，计算系统的状态空间矩阵A和输入矩阵B，并进行简化处理
    
    upright_nominal = {q1d: 0, q2: 0, m: 1, r: 1, g: 1}
    # 直立工作点的定义，包括零速度、特定参数值
    
    # Precomputed solution
    A_sol = Matrix([[0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                    [sin(q1)*q3d, 0, 0, 0, 0, -sin(q1), -cos(q1), 0],
                    [-cos(q1)*q3d, 0, 0, 0, 0, cos(q1), -sin(q1), 0],
                    [0, Rational(4, 5), 0, 0, 0, 0, 0, 6*q3d/5],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, -2*q3d, 0, 0]])
    # 预先计算的状态空间矩阵A的解
    
    B_sol = Matrix([])
    # 空的预先计算的输入矩阵B的解
    
    # Check that linearization is correct
    # 检查线性化是否正确
    assert A.subs(upright_nominal) == A_sol
    # 断言使用直立工作点替换后的状态空间矩阵A等于预先计算的A_sol
    assert B.subs(upright_nominal) == B_sol
    # 断言使用直立工作点替换后的输入矩阵B等于预先计算的B_sol
    
    # Check eigenvalues at critical speed are all zero:
    # 检查临界速度时的特征值是否全部为零
    assert sympify(A.subs(upright_nominal).subs(q3d, 1/sqrt(3))).eigenvals() == {0: 8}
    # 断言将q3d替换为1/sqrt(3)后的状态空间矩阵A的特征值为{0: 8}
    
    # Check whether alternative solvers work
    # 检查替代求解器是否工作
    # symengine不支持method='GJ'
    linearizer = KM.to_linearizer(linear_solver='GJ')
    # 使用Kane方法的线性化工具转换为线性化器对象，使用Gauss-Jordan法求解器
    A, B = linearizer.linearize(op_point=[q_op, u_op, qd_op, ud_op],
                                A_and_B=True, simplify=True)
    # 使用预先计算的工作点进行线性化，计算系统的状态空间矩阵A和输入矩阵B，并进行简化处理
    assert A.subs(upright_nominal) == A_sol
    # 断言使用直立工作点替换后的状态空间矩阵A等于预先计算的A_sol
    assert B.subs(upright_nominal) == B_sol
    # 断言使用直立工作点替换后的输入矩阵B等于预先计算的B_sol
def test_linearize_pendulum_kane_minimal():
    q1 = dynamicsymbols('q1')                     # 定义摆角度
    u1 = dynamicsymbols('u1')                     # 定义角速度
    q1d = dynamicsymbols('q1', 1)                 # 定义角速度的一阶导数
    L, m, t = symbols('L, m, t')                   # 定义长度 L、质量 m 和时间 t
    g = 9.8                                         # 设置重力加速度

    # 构建世界坐标系 N
    N = ReferenceFrame('N')
    pN = Point('N*')
    pN.set_vel(N, 0)                               # 设置 N 的速度为 0

    # 设定 A.x 沿着摆的方向
    A = N.orientnew('A', 'axis', [q1, N.z])
    A.set_ang_vel(N, u1*N.z)                        # 设置 A 相对于 N 的角速度

    # 相对于原点 N* 定位点 P
    P = pN.locatenew('P', L*A.x)
    P.v2pt_theory(pN, N, A)                         # 计算点 P 的速度
    pP = Particle('pP', P, m)                       # 创建质点 pP

    # 创建运动学微分方程
    kde = Matrix([q1d - u1])

    # 在点 P 输入力结果
    R = m*g*N.x

    # 使用 Kanes 方法求解动力学方程
    KM = KanesMethod(N, q_ind=[q1], u_ind=[u1], kd_eqs=kde)
    (fr, frstar) = KM.kanes_equations([pP], [(P, R)])

    # 线性化
    A, B, inp_vec = KM.linearize(A_and_B=True, simplify=True)

    assert A == Matrix([[0, 1], [-9.8*cos(q1)/L, 0]])
    assert B == Matrix([])


def test_linearize_pendulum_kane_nonminimal():
    # 创建非最小实现的广义坐标和速度
    # q1, q2 = 摆的 N.x 和 N.y 坐标
    # u1, u2 = 摆的 N.x 和 N.y 速度
    q1, q2 = dynamicsymbols('q1:3')
    q1d, q2d = dynamicsymbols('q1:3', level=1)
    u1, u2 = dynamicsymbols('u1:3')
    u1d, u2d = dynamicsymbols('u1:3', level=1)
    L, m, t = symbols('L, m, t')
    g = 9.8

    # 构建世界坐标系 N
    N = ReferenceFrame('N')
    pN = Point('N*')
    pN.set_vel(N, 0)                               # 设置 N 的速度为 0

    # 设定 A.x 沿着摆的方向
    theta1 = atan(q2/q1)
    A = N.orientnew('A', 'axis', [theta1, N.z])

    # 定位质量为 P 的摆
    P = pN.locatenew('P1', q1*N.x + q2*N.y)
    pP = Particle('pP', P, m)

    # 计算运动学微分方程
    kde = Matrix([q1d - u1,
                  q2d - u2])
    dq_dict = solve(kde, [q1d, q2d])

    # 设置点 P 的速度
    P.set_vel(N, P.pos_from(pN).dt(N).subs(dq_dict))

    # 配置约束是摆长为常数 L
    f_c = Matrix([P.pos_from(pN).magnitude() - L])

    # 速度约束是 A.x 方向上的速度始终为零（摆不会变长）
    f_v = Matrix([P.vel(N).express(A).dot(A.x)])
    f_v.simplify()

    # 加速度约束是速度约束的时间导数
    f_a = f_v.diff(t)
    f_a.simplify()

    # 在点 P 输入力结果
    R = m*g*N.x

    # 使用 Kanes 方法推导运动方程
    KM = KanesMethod(N, q_ind=[q2], u_ind=[u2], q_dependent=[q1],
            u_dependent=[u1], configuration_constraints=f_c,
            velocity_constraints=f_v, acceleration_constraints=f_a, kd_eqs=kde)
    (fr, frstar) = KM.kanes_equations([pP], [(P, R)])

    # 设置操作点为竖直向下，并且不动
    q_op = {q1: L, q2: 0}
    u_op = {u1: 0, u2: 0}
    # 初始化一个字典，其中包含两个键值对，每个键对应的值都初始化为0
    ud_op = {u1d: 0, u2d: 0}
    
    # 使用 KM 对象的 linearize 方法，对给定操作点进行线性化处理，并返回线性化后的矩阵 A 和 B，以及输入向量 inp_vec
    # op_point 参数为包含 q_op, u_op, ud_op 的列表，A_and_B=True 表示返回 A 和 B，simplify=True 表示简化结果
    A, B, inp_vec = KM.linearize(op_point=[q_op, u_op, ud_op], A_and_B=True,
                                 simplify=True)
    
    # 断言：验证 A 的展开形式与预期的矩阵相同
    assert A.expand() == Matrix([[0, 1], [-9.8/L, 0]])
    # 断言：验证 B 是一个空矩阵
    assert B == Matrix([])
    
    
    # 使用 KM 对象的 linearize 方法进行线性化处理，此处指定 linear_solver='GJ'，但提醒 symengine 不支持 'GJ' 方法
    A, B, inp_vec = KM.linearize(op_point=[q_op, u_op, ud_op], A_and_B=True,
                                 simplify=True, linear_solver='GJ')
    
    # 断言：验证 A 的展开形式与预期的矩阵相同
    assert A.expand() == Matrix([[0, 1], [-9.8/L, 0]])
    # 断言：验证 B 是一个空矩阵
    assert B == Matrix([])
    
    # 使用 KM 对象的 linearize 方法进行线性化处理，使用自定义的线性求解器，lambda 函数接受参数 A 和 b 并返回 A.LUsolve(b)
    A, B, inp_vec = KM.linearize(op_point=[q_op, u_op, ud_op],
                                 A_and_B=True,
                                 simplify=True,
                                 linear_solver=lambda A, b: A.LUsolve(b))
    
    # 断言：验证 A 的展开形式与预期的矩阵相同
    assert A.expand() == Matrix([[0, 1], [-9.8/L, 0]])
    # 断言：验证 B 是一个空矩阵
    assert B == Matrix([])
# 定义测试函数：线性化滚动圆盘的Lagrange方法
def test_linearize_rolling_disc_lagrange():
    # 定义动力学符号变量
    q1, q2, q3 = q = dynamicsymbols('q1 q2 q3')
    q1d, q2d, q3d = qd = dynamicsymbols('q1 q2 q3', 1)
    r, m, g = symbols('r m g')

    # 创建参考坐标系 N
    N = ReferenceFrame('N')
    # 使用轴角法创建旋转坐标系 Y，旋转轴为 N.z，旋转角度为 q1
    Y = N.orientnew('Y', 'Axis', [q1, N.z])



    # 此处应有更多代码，但省略了一部分



    # 继续函数内的代码，但省略了一部分
    # 定义向量 L，它是 Y 轴绕 q2 轴旋转的结果
    L = Y.orientnew('L', 'Axis', [q2, Y.x])
    # 定义向量 R，它是 L 轴绕 q3 轴旋转的结果
    R = L.orientnew('R', 'Axis', [q3, L.y])

    # 创建点 C，并设置其在惯性系 N 中的速度为零
    C = Point('C')
    C.set_vel(N, 0)
    # 定义点 Dmc，它相对于点 C 是 r * L.z 的位移，同时计算其速度
    Dmc = C.locatenew('Dmc', r * L.z)
    Dmc.v2pt_theory(C, N, R)

    # 计算物体的惯性，使用惯性函数 inertia(L, Ixx, Iyy, Izz)，此处计算关于轴 L 的惯性
    I = inertia(L, m / 4 * r**2, m / 2 * r**2, m / 4 * r**2)
    # 创建刚体 BodyD，使用点 Dmc、轴 R、质量 m 和惯性 I 进行定义
    BodyD = RigidBody('BodyD', Dmc, R, m, (I, Dmc))
    # 设置刚体的势能，这里是在竖直方向上的势能表达式
    BodyD.potential_energy = - m * g * r * cos(q2)

    # 创建拉格朗日函数 Lag，使用惯性系 N 和刚体 BodyD
    Lag = Lagrangian(N, BodyD)
    # 使用拉格朗日方法定义 l，计算拉格朗日方程
    l = LagrangesMethod(Lag, q)
    l.form_lagranges_equations()

    # 在稳态竖直滚动状态线性化系统
    op_point = {q1: 0, q2: 0, q3: 0,
                q1d: 0, q2d: 0,
                q1d.diff(): 0, q2d.diff(): 0, q3d.diff(): 0}
    # 使用 l.linearize() 方法线性化系统，返回状态空间方程组的 A 矩阵
    A = l.linearize(q_ind=q, qd_ind=qd, op_point=op_point, A_and_B=True)[0]
    # 断言 A 矩阵与预期的线性化矩阵 sol 相等
    sol = Matrix([[0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, -6*q3d, 0],
                  [0, -4*g/(5*r), 0, 6*q3d/5, 0, 0],
                  [0, 0, 0, 0, 0, 0]])
    assert A == sol  # 断言确保线性化的 A 矩阵与预期的 sol 矩阵相等
```