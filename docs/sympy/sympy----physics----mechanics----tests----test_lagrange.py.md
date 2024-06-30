# `D:\src\scipysrc\sympy\sympy\physics\mechanics\tests\test_lagrange.py`

```
# 引入必要的库和模块，包括符号计算、物理力学、以及测试相关的功能
from sympy.physics.mechanics import (dynamicsymbols, ReferenceFrame, Point,
                                    RigidBody, LagrangesMethod, Particle,
                                    inertia, Lagrangian)
from sympy.core.function import (Derivative, Function)
from sympy.core.numbers import pi
from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import (cos, sin, tan)
from sympy.matrices.dense import Matrix
from sympy.simplify.simplify import simplify
from sympy.testing.pytest import raises


def test_invalid_coordinates():
    # 简单的单摆系统，但使用符号（symbol）而非动力学符号（dynamicsymbol）
    l, m, g = symbols('l m g')
    q = symbols('q')  # 广义坐标
    N, O = ReferenceFrame('N'), Point('O')
    O.set_vel(N, 0)
    P = Particle('P', Point('P'), m)
    P.point.set_pos(O, l * (sin(q) * N.x - cos(q) * N.y))
    P.potential_energy = m * g * P.point.pos_from(O).dot(N.y)
    L = Lagrangian(N, P)
    raises(ValueError, lambda: LagrangesMethod(L, [q], bodies=P))


def test_disc_on_an_incline_plane():
    # 圆盘在斜面上滚动
    # 首先创建广义坐标。圆盘的质心由广义坐标 'y' 确定，圆盘的姿态由角度 'theta' 确定。
    # 圆盘的质量为 'm'，半径为 'R'。斜面的长度为 'l'，倾角为 'alpha'。'g' 是重力常数。
    y, theta = dynamicsymbols('y theta')
    yd, thetad = dynamicsymbols('y theta', 1)
    m, g, R, l, alpha = symbols('m g R l alpha')

    # 接下来，创建惯性参考系 'N'。参考系 'A' 与斜面相连。最后创建一个与圆盘相连的参考系。
    N = ReferenceFrame('N')
    A = N.orientnew('A', 'Axis', [pi/2 - alpha, N.z])
    B = A.orientnew('B', 'Axis', [-theta, A.z])

    # 创建圆盘 'D'；创建代表圆盘质心的点，并设置其速度。创建圆盘的惯性张量。最后创建圆盘。
    Do = Point('Do')
    Do.set_vel(N, yd * A.x)
    I = m * R**2/2 * B.z | B.z
    D = RigidBody('D', Do, B, m, (I, Do))

    # 构建圆盘的拉格朗日函数 'L'，确定其动能 T 和势能 U，拉格朗日函数 L 定义为 T 减去 U。
    D.potential_energy = m * g * (l - y) * sin(alpha)
    L = Lagrangian(N, D)

    # 创建广义坐标列表和约束方程。约束源于圆盘在斜面上无滑动滚动。然后调用 'LagrangesMethod' 类，
    # 并提供必要的参数以生成运动方程。'rhs' 方法解出 q_double_dots（即二阶导数）。
    # 定义广义坐标 q 和拉格朗日乘子 hol_coneqs，用于描述系统的运动
    q = [y, theta]
    # 定义拉格朗日乘法方法的实例，传入系统的拉格朗日函数 L、广义坐标 q 和保守约束条件 hol_coneqs
    m = LagrangesMethod(L, q, hol_coneqs=hol_coneqs)
    # 计算拉格朗日方程
    m.form_lagranges_equations()
    # 获取右手边的表达式
    rhs = m.rhs()
    # 简化右手边的表达式
    rhs.simplify()
    # 断言第三个元素（索引为2）的右手边等于给定的表达式 2*g*sin(alpha)/3
    assert rhs[2] == 2*g*sin(alpha)/3
def test_dub_pen():

    # The system considered is the double pendulum. Like in the
    # test of the simple pendulum above, we begin by creating the generalized
    # coordinates and the simple generalized speeds and accelerations which
    # define the motion of the system.
    q1, q2 = dynamicsymbols('q1:3')
    q1d, q2d = dynamicsymbols('q1:3', level=1)

    # Define parameters: length of the rods and gravitational constant
    L, m, t = symbols('L, m, t')
    g = 9.8

    # Compose World Frame
    N = ReferenceFrame('N')
    pN = Point('N*')
    pN.set_vel(N, 0)

    # Create point P, the pendulum mass
    P = pN.locatenew('P1', q1*N.x + q2*N.y)
    P.set_vel(N, P.pos_from(pN).dt(N))
    pP = Particle('pP', P, m)

    # Constraint Equations: the length constraint of the double pendulum
    f_c = Matrix([q1**2 + q2**2 - L**2])

    # Calculate the lagrangian, and form the equations of motion
    Lag = Lagrangian(N, pP)
    LM = LagrangesMethod(Lag, [q1, q2], hol_coneqs=f_c,
                        forcelist=[(P, m*g*N.x)], frame=N)
    LM.form_lagranges_equations()

    # Check solution of equations of motion
    lam1 = LM.lam_vec[0, 0]
    eom_sol = Matrix([[m*Derivative(q1, t, t) - 9.8*m + 2*lam1*q1],
                      [m*Derivative(q2, t, t) + 2*lam1*q2]])
    assert LM.eom == eom_sol

    # Check solution of multipliers (Lagrange multipliers)
    lam_sol = Matrix([(19.6*q1 + 2*q1d**2 + 2*q2d**2)/(4*q1**2/m + 4*q2**2/m)])
    assert simplify(LM.solve_multipliers(sol_type='Matrix')) == simplify(lam_sol)
    # 定义动力学符号 q1, q2 以及它们的一阶和二阶导数 q1d, q2d, q1dd, q2dd
    q1, q2 = dynamicsymbols('q1 q2')
    q1d, q2d = dynamicsymbols('q1 q2', 1)
    q1dd, q2dd = dynamicsymbols('q1 q2', 2)
    
    # 定义速度符号 u1, u2 以及它们的一阶导数 u1d, u2d
    u1, u2 = dynamicsymbols('u1 u2')
    u1d, u2d = dynamicsymbols('u1 u2', 1)
    
    # 定义长度 l，质量 m，重力加速度 g 的符号
    l, m, g = symbols('l m g')
    
    # 创建一个惯性参考框架 N
    N = ReferenceFrame('N')
    
    # 在 N 参考框架下创建 A 和 B 参考框架，分别绕 z 轴旋转 q1 和 q2
    A = N.orientnew('A', 'Axis', [q1, N.z])
    B = N.orientnew('B', 'Axis', [q2, N.z])
    
    # 设置 A 和 B 参考框架的角速度
    A.set_ang_vel(N, q1d * A.z)
    B.set_ang_vel(N, q2d * A.z)
    
    # 创建点 O，P 和 R，定义它们之间的几何关系
    O = Point('O')
    P = O.locatenew('P', l * A.x)
    R = P.locatenew('R', l * B.x)
    
    # 设置点 O 的速度为零
    O.set_vel(N, 0)
    
    # 计算点 P 和 R 相对于参考框架 N 的速度
    P.v2pt_theory(O, N, A)
    R.v2pt_theory(P, N, B)
    
    # 创建质点 ParP 和 ParR，分别位于点 P 和 R，质量均为 m
    ParP = Particle('ParP', P, m)
    ParR = Particle('ParR', R, m)
    
    # 设置质点 ParP 和 ParR 的势能
    ParP.potential_energy = - m * g * l * cos(q1)
    ParR.potential_energy = - m * g * l * cos(q1) - m * g * l * cos(q2)
    
    # 创建拉格朗日函数 L
    L = Lagrangian(N, ParP, ParR)
    
    # 创建拉格朗日方程组的生成对象 lm
    lm = LagrangesMethod(L, [q1, q2], bodies=[ParP, ParR])
    
    # 计算拉格朗日方程组
    lm.form_lagranges_equations()
    
    # 断言验证拉格朗日方程的形式，应该为零
    assert simplify(l*m*(2*g*sin(q1) + l*sin(q1)*sin(q2)*q2dd
        + l*sin(q1)*cos(q2)*q2d**2 - l*sin(q2)*cos(q1)*q2d**2
        + l*cos(q1)*cos(q2)*q2dd + 2*l*q1dd) - lm.eom[0]) == 0
    
    assert simplify(l*m*(g*sin(q2) + l*sin(q1)*sin(q2)*q1dd
        - l*sin(q1)*cos(q2)*q1d**2 + l*sin(q2)*cos(q1)*q1d**2
        + l*cos(q1)*cos(q2)*q1dd + l*q2dd) - lm.eom[1]) == 0
    
    # 断言验证 lm 对象中的质点列表应该包含 ParP 和 ParR
    assert lm.bodies == [ParP, ParR]
def test_rolling_disc():
    # Rolling Disc Example
    # Here the rolling disc is formed from the contact point up, removing the
    # need to introduce generalized speeds. Only 3 configuration and 3
    # speed variables are need to describe this system, along with the
    # disc's mass and radius, and the local gravity.

    # 定义动力学符号变量 q1, q2, q3 和它们的一阶导数 q1d, q2d, q3d
    q1, q2, q3 = dynamicsymbols('q1 q2 q3')
    q1d, q2d, q3d = dynamicsymbols('q1 q2 q3', 1)
    r, m, g = symbols('r m g')

    # 创建参考坐标系 N
    N = ReferenceFrame('N')
    # 在 N 坐标系下创建 Y 坐标系，绕 z 轴旋转 q1
    Y = N.orientnew('Y', 'Axis', [q1, N.z])
    # 在 Y 坐标系下创建 L 坐标系，绕 x 轴旋转 q2
    L = Y.orientnew('L', 'Axis', [q2, Y.x])
    # 在 L 坐标系下创建 R 坐标系，绕 y 轴旋转 q3
    R = L.orientnew('R', 'Axis', [q3, L.y])

    # 创建点 C，表示接触点，并在 N 坐标系中使其静止
    C = Point('C')
    C.set_vel(N, 0)
    # 创建点 Dmc，表示质心位置，位于距接触点 r * L.z 处
    Dmc = C.locatenew('Dmc', r * L.z)
    # 计算 Dmc 相对于 C 的速度和加速度
    Dmc.v2pt_theory(C, N, R)

    # 创建惯性二阶张量
    I = inertia(L, m/4 * r**2, m/2 * r**2, m/4 * r**2)
    # 创建刚体 BodyD，质心在 Dmc，定向为 R，质量为 m，惯性张量为 I
    BodyD = RigidBody('BodyD', Dmc, R, m, (I, Dmc))

    # 设置 BodyD 的势能
    BodyD.potential_energy = - m * g * r * cos(q2)
    # 创建拉格朗日函数
    Lag = Lagrangian(N, BodyD)
    # 定义广义坐标 q
    q = [q1, q2, q3]
    q1 = Function('q1')
    q2 = Function('q2')
    q3 = Function('q3')
    # 创建拉格朗日方程的方法
    l = LagrangesMethod(Lag, q)
    # 形成拉格朗日方程
    l.form_lagranges_equations()
    # 获取右手边的表达式
    RHS = l.rhs()
    # 简化右手边的表达式
    RHS.simplify()
    t = symbols('t')

    # 断言质量矩阵的部分值
    assert (l.mass_matrix[3:6] == [0, 5*m*r**2/4, 0])
    # 断言右手边的第四个分量的简化结果
    assert RHS[4].simplify() == (
        (-8*g*sin(q2(t)) + r*(5*sin(2*q2(t))*Derivative(q1(t), t) +
        12*cos(q2(t))*Derivative(q3(t), t))*Derivative(q1(t), t))/(10*r))
    # 断言右手边的第五个分量
    assert RHS[5] == (-5*cos(q2(t))*Derivative(q1(t), t) + 6*tan(q2(t)
        )*Derivative(q3(t), t) + 4*Derivative(q1(t), t)/cos(q2(t))
        )*Derivative(q2(t), t)
```