# `D:\src\scipysrc\sympy\sympy\physics\mechanics\tests\test_kane.py`

```
from sympy import solve
from sympy import (cos, expand, Matrix, sin, symbols, tan, sqrt, S,
                                zeros, eye)
from sympy.simplify.simplify import simplify
from sympy.physics.mechanics import (dynamicsymbols, ReferenceFrame, Point,
                                     RigidBody, KanesMethod, inertia, Particle,
                                     dot, find_dynamicsymbols)
from sympy.testing.pytest import raises

def test_invalid_coordinates():
    # 定义测试函数，验证在使用符号而不是动态符号时的简单单摆情况
    l, m, g = symbols('l m g')
    q, u = symbols('q u')  # 广义坐标
    kd = [q.diff(dynamicsymbols._t) - u]  # 定义运动方程
    N, O = ReferenceFrame('N'), Point('O')  # 定义参考系和点
    O.set_vel(N, 0)  # 设置点的速度
    P = Particle('P', Point('P'), m)  # 创建质点对象
    P.point.set_pos(O, l * (sin(q) * N.x - cos(q) * N.y))  # 设置质点位置
    F = (P.point, -m * g * N.y)  # 定义作用力
    raises(ValueError, lambda: KanesMethod(N, [q], [u], kd, bodies=[P], forcelist=[F]))  # 预期引发值错误异常

def test_one_dof():
    # 定义测试函数，验证单自由度弹簧-质量-阻尼器系统的情况
    q, u = dynamicsymbols('q u')  # 动态符号表示广义坐标和速度
    qd, ud = dynamicsymbols('q u', 1)  # 一阶导数的动态符号
    m, c, k = symbols('m c k')  # 定义质量、阻尼系数和弹簧刚度
    N = ReferenceFrame('N')  # 创建惯性参考系
    P = Point('P')  # 创建一个点对象
    P.set_vel(N, u * N.x)  # 设置点的速度

    kd = [qd - u]  # 运动方程
    FL = [(P, (-k * q - c * u) * N.x)]  # 作用力列表
    pa = Particle('pa', P, m)  # 创建质点对象
    BL = [pa]  # 质点列表

    KM = KanesMethod(N, [q], [u], kd)  # 创建KanesMethod对象
    KM.kanes_equations(BL, FL)  # 计算kanes方程

    assert KM.bodies == BL  # 断言质点列表正确
    assert KM.loads == FL  # 断言作用力列表正确

    MM = KM.mass_matrix  # 质量矩阵
    forcing = KM.forcing  # 强迫项
    rhs = MM.inv() * forcing  # 右手边项
    assert expand(rhs[0]) == expand(-(q * k + u * c) / m)  # 断言右手边项的展开式

    assert simplify(KM.rhs() - KM.mass_matrix_full.LUsolve(KM.forcing_full)) == zeros(2, 1)  # 断言简化kanes方程

    assert (KM.linearize(A_and_B=True)[0] == Matrix([[0, 1], [-k/m, -c/m]]))  # 断言线性化结果

def test_two_dof():
    # 定义测试函数，验证两自由度弹簧-质量-阻尼器系统的情况
    # 第一个坐标是第一个质点的位移，第二个坐标是第一个和第二个质点之间的相对位移
    q1, q2, u1, u2 = dynamicsymbols('q1 q2 u1 u2')  # 动态符号表示广义坐标和速度
    q1d, q2d, u1d, u2d = dynamicsymbols('q1 q2 u1 u2', 1)  # 一阶导数的动态符号
    m, c1, c2, k1, k2 = symbols('m c1 c2 k1 k2')  # 定义质量、阻尼系数和弹簧刚度
    N = ReferenceFrame('N')  # 创建惯性参考系
    P1 = Point('P1')  # 创建点对象P1
    P2 = Point('P2')  # 创建点对象P2
    P1.set_vel(N, u1 * N.x)  # 设置P1点的速度
    P2.set_vel(N, (u1 + u2) * N.x)  # 设置P2点的速度
    # 注意我们通过一个任意因子来乘以运动方程，以测试隐式和显式运动学属性
    kd = [q1d/2 - u1/2, 2*q2d - 2*u2]  # 运动方程

    # 创建作用力列表
    FL = [(P1, (-k1 * q1 - c1 * u1 + k2 * q2 + c2 * u2) * N.x),
          (P2, (-k2 * q2 - c2 * u2) * N.x)]
    pa1 = Particle('pa1', P1, m)  # 创建质点对象pa1
    pa2 = Particle('pa2', P2, m)  # 创建质点对象pa2
    BL = [pa1, pa2]  # 质点列表
    # 创建 KanesMethod 对象，指定惯性参考系 N，广义坐标 q1, q2 和速度 u1, u2，
    # 并传递相关信息，形成 Fr 和 Fr*。然后计算质量矩阵和驱动项，最终求解 udots。
    KM = KanesMethod(N, q_ind=[q1, q2], u_ind=[u1, u2], kd_eqs=kd)
    
    # 使用 KanesMethod 对象计算卡恩方程，并指定惯性载荷 BL 和非惯性载荷 FL
    KM.kanes_equations(BL, FL)
    
    # 获取计算得到的质量矩阵
    MM = KM.mass_matrix
    
    # 获取计算得到的驱动项
    forcing = KM.forcing
    
    # 解线性方程组 MM * udots = forcing，得到 udots
    rhs = MM.inv() * forcing
    
    # 断言验证第一个广义坐标的运动方程
    assert expand(rhs[0]) == expand((-k1 * q1 - c1 * u1 + k2 * q2 + c2 * u2)/m)
    
    # 断言验证第二个广义坐标的运动方程
    assert expand(rhs[1]) == expand((k1 * q1 + c1 * u1 - 2 * k2 * q2 - 2 * c2 * u2) / m)

    # 检查是否使用了显式运动学，默认情况下质量矩阵是单位矩阵
    assert KM.explicit_kinematics
    assert KM.mass_matrix_kin == eye(2)
    
    # 当使用隐式运动学时，检查质量矩阵不是单位矩阵
    KM.explicit_kinematics = False
    assert KM.mass_matrix_kin == Matrix([[S(1)/2, 0], [0, 2]])
    
    # 检查隐式或显式运动学下，右手边的方程与矩阵形式一致
    for explicit_kinematics in [False, True]:
        KM.explicit_kinematics = explicit_kinematics
        assert simplify(KM.rhs() - KM.mass_matrix_full.LUsolve(KM.forcing_full)) == zeros(4, 1)
    
    # 确保如果提供了非线性运动学微分方程，会引发错误
    kd = [q1d - u1**2, sin(q2d) - cos(u2)]
    raises(ValueError, lambda: KanesMethod(N, q_ind=[q1, q2], u_ind=[u1, u2], kd_eqs=kd))
def test_rolling_disc():
    # Rolling Disc Example
    # Here the rolling disc is formed from the contact point up, removing the
    # need to introduce generalized speeds. Only 3 configuration and three
    # speed variables are need to describe this system, along with the disc's
    # mass and radius, and the local gravity (note that mass will drop out).
    
    # 定义动力学符号变量
    q1, q2, q3, u1, u2, u3 = dynamicsymbols('q1 q2 q3 u1 u2 u3')
    q1d, q2d, q3d, u1d, u2d, u3d = dynamicsymbols('q1 q2 q3 u1 u2 u3', 1)
    r, m, g = symbols('r m g')

    # 创建参考框架 N
    N = ReferenceFrame('N')

    # 定义 Y 框架，以及其相对于 N 框架的转动
    Y = N.orientnew('Y', 'Axis', [q1, N.z])

    # 定义 L 框架，以及其相对于 Y 框架的转动
    L = Y.orientnew('L', 'Axis', [q2, Y.x])

    # 定义 R 框架，以及其相对于 L 框架的转动
    R = L.orientnew('R', 'Axis', [q3, L.y])

    # 计算 R 相对于 N 框架的角速度
    w_R_N_qd = R.ang_vel_in(N)

    # 设置 R 框架相对于 N 框架的角速度
    R.set_ang_vel(N, u1 * L.x + u2 * L.y + u3 * L.z)

    # 创建一个点 C，并在 N 框架下设置其速度为零
    C = Point('C')
    C.set_vel(N, 0)

    # 创建点 Dmc，表示盘中心的位置，相对于点 C 在 L 框架下的位置
    Dmc = C.locatenew('Dmc', r * L.z)

    # 使用点 C 和 N 框架下的 R 框架计算 Dmc 的速度
    Dmc.v2pt_theory(C, N, R)

    # 创建惯性张量
    I = inertia(L, m / 4 * r**2, m / 2 * r**2, m / 4 * r**2)

    # 创建运动方程的动力学约束
    kd = [dot(R.ang_vel_in(N) - w_R_N_qd, uv) for uv in L]

    # 创建力列表，这里是盘质心处的重力
    ForceList = [(Dmc, - m * g * Y.z)]

    # 创建刚体，即盘的整体
    BodyD = RigidBody('BodyD', Dmc, R, m, (I, Dmc))

    # 创建刚体列表
    BodyList = [BodyD]

    # 最后形成运动方程，使用相同的步骤
    # 创建 KanesMethod 对象，指定惯性参考系 N，广义坐标 q1, q2, q3 和广义速度 u1, u2, u3
    # kd 是描述运动方程的字典，包含了运动方程的表达式
    KM = KanesMethod(N, q_ind=[q1, q2, q3], u_ind=[u1, u2, u3], kd_eqs=kd)
    
    # 使用 KanesMethod 对象计算 Kanes 方程，输入力列表 ForceList 和主体列表 BodyList
    KM.kanes_equations(BodyList, ForceList)
    
    # 获取 KanesMethod 对象的质量矩阵
    MM = KM.mass_matrix
    
    # 获取 KanesMethod 对象的强迫项
    forcing = KM.forcing
    
    # 解线性方程 MM * u_dot = forcing，求出广义速度的时间导数 u_dot
    rhs = MM.inv() * forcing
    
    # 获取运动微分方程字典
    kdd = KM.kindiffdict()
    
    # 用运动微分方程字典替换 rhs 中的符号
    rhs = rhs.subs(kdd)
    
    # 简化 rhs
    rhs.simplify()
    
    # 验证 rhs 展开后是否等于给定的矩阵
    assert rhs.expand() == Matrix([(6*u2*u3*r - u3**2*r*tan(q2) + 4*g*sin(q2))/(5*r), -2*u1*u3/3, u1*(-2*u2 + u3*tan(q2))]).expand()
    
    # 验证 KanesMethod 对象中的 rhs 是否等于质量矩阵求逆后与强迫项相乘的结果
    assert simplify(KM.rhs() - KM.mass_matrix_full.LUsolve(KM.forcing_full)) == zeros(6, 1)

    # 进行线性化处理，获取线性化后的系统矩阵 A
    A = KM.linearize(A_and_B=True)[0]
    
    # 将参数 r=g=m=1，广义坐标和速度设置为零，得到竖直位置（upright）的系统矩阵 A_upright
    A_upright = A.subs({r: 1, g: 1, m: 1}).subs({q1: 0, q2: 0, q3: 0, u1: 0, u3: 0})
    
    # 导入 sympy 模块，验证 A_upright 在 u2 = 1 / sqrt(3) 时的特征值是否为 {0}
    assert sympy.sympify(A_upright.subs({u2: 1 / sqrt(3)})).eigenvals() == {S.Zero: 6}
def test_aux():
    # Same as above, except we have 2 auxiliary speeds for the ground contact
    # point, which is known to be zero. In one case, we go through then
    # substitute the aux. speeds in at the end (they are zero, as well as their
    # derivative), in the other case, we use the built-in auxiliary speed part
    # of KanesMethod. The equations from each should be the same.

    # 声明系统的广义坐标和速度符号
    q1, q2, q3, u1, u2, u3 = dynamicsymbols('q1 q2 q3 u1 u2 u3')
    q1d, q2d, q3d, u1d, u2d, u3d = dynamicsymbols('q1 q2 q3 u1 u2 u3', 1)
    u4, u5, f1, f2 = dynamicsymbols('u4, u5, f1, f2')
    u4d, u5d = dynamicsymbols('u4, u5', 1)
    r, m, g = symbols('r m g')

    # 创建一个惯性参考系
    N = ReferenceFrame('N')
    # 建立 Y 参考系，使用 z 轴旋转 q1 角度
    Y = N.orientnew('Y', 'Axis', [q1, N.z])
    # 建立 L 参考系，使用 x 轴旋转 q2 角度
    L = Y.orientnew('L', 'Axis', [q2, Y.x])
    # 建立 R 参考系，使用 y 轴旋转 q3 角度
    R = L.orientnew('R', 'Axis', [q3, L.y])
    # 计算 R 相对于 N 的角速度
    w_R_N_qd = R.ang_vel_in(N)
    # 设置 R 相对于 N 的角速度为 u1*L.x + u2*L.y + u3*L.z
    R.set_ang_vel(N, u1 * L.x + u2 * L.y + u3 * L.z)

    # 定义一个运动点 C
    C = Point('C')
    # 设置点 C 在参考系 N 中的速度
    C.set_vel(N, u4 * L.x + u5 * (Y.z ^ L.x))
    # 在点 C 处创建点 Dmc，并将其定位在 r*L.z 的位置
    Dmc = C.locatenew('Dmc', r * L.z)
    # 使用理论方法计算 Dmc 相对于 N 的速度
    Dmc.v2pt_theory(C, N, R)
    # 使用理论方法计算 Dmc 相对于 N 的加速度
    Dmc.a2pt_theory(C, N, R)

    # 计算质量分布的惯性矩阵
    I = inertia(L, m / 4 * r**2, m / 2 * r**2, m / 4 * r**2)

    # 计算广义速度的等式
    kd = [dot(R.ang_vel_in(N) - w_R_N_qd, uv) for uv in L]

    # 定义作用力列表
    ForceList = [(Dmc, - m * g * Y.z), (C, f1 * L.x + f2 * (Y.z ^ L.x))]
    # 创建刚体 D
    BodyD = RigidBody('BodyD', Dmc, R, m, (I, Dmc))
    BodyList = [BodyD]

    # 创建 KanesMethod 对象
    KM = KanesMethod(N, q_ind=[q1, q2, q3], u_ind=[u1, u2, u3, u4, u5],
                     kd_eqs=kd)
    # 计算 Kanes 方程
    (fr, frstar) = KM.kanes_equations(BodyList, ForceList)
    # 替换掉 u4, u5 及其导数为零，并简化方程 fr 和 frstar
    fr = fr.subs({u4d: 0, u5d: 0}).subs({u4: 0, u5: 0})
    frstar = frstar.subs({u4d: 0, u5d: 0}).subs({u4: 0, u5: 0})

    # 创建另一个 KanesMethod 对象，使用内置的辅助速度部分 u4, u5
    KM2 = KanesMethod(N, q_ind=[q1, q2, q3], u_ind=[u1, u2, u3], kd_eqs=kd,
                      u_auxiliary=[u4, u5])
    # 计算 Kanes 方程
    (fr2, frstar2) = KM2.kanes_equations(BodyList, ForceList)
    # 替换掉 u4, u5 及其导数为零，并简化方程 fr2 和 frstar2
    fr2 = fr2.subs({u4d: 0, u5d: 0}).subs({u4: 0, u5: 0})
    frstar2 = frstar2.subs({u4d: 0, u5d: 0}).subs({u4: 0, u5: 0})

    # 简化 frstar 和 frstar2
    frstar.simplify()
    frstar2.simplify()

    # 断言两组方程的差为零向量
    assert (fr - fr2).expand() == Matrix([0, 0, 0, 0, 0])
    assert (frstar - frstar2).expand() == Matrix([0, 0, 0, 0, 0])


def test_parallel_axis():
    # This is for a 2 dof inverted pendulum on a cart.
    # This tests the parallel axis code in KanesMethod. The inertia of the
    # pendulum is defined about the hinge, not about the center of mass.

    # Defining the constants and knowns of the system
    gravity = symbols('g')
    k, ls = symbols('k ls')
    a, mA, mC = symbols('a mA mC')
    F = dynamicsymbols('F')
    Ix, Iy, Iz = symbols('Ix Iy Iz')

    # Declaring the Generalized coordinates and speeds
    q1, q2 = dynamicsymbols('q1 q2')
    q1d, q2d = dynamicsymbols('q1 q2', 1)
    u1, u2 = dynamicsymbols('u1 u2')
    u1d, u2d = dynamicsymbols('u1 u2', 1)

    # Creating reference frames
    N = ReferenceFrame('N')
    A = ReferenceFrame('A')

    # Orienting A frame relative to N frame with rotation about N.z axis by -q2
    A.orient(N, 'Axis', [-q2, N.z])
    # Setting angular velocity of A frame relative to N frame along -N.z axis
    A.set_ang_vel(N, -u2 * N.z)

    # Origin of Newtonian reference frame
    O = Point('O')
    # 创建并定位小车C和摆的质心A的位置
    C = O.locatenew('C', q1 * N.x)
    Ao = C.locatenew('Ao', a * A.y)

    # 定义各点的速度
    O.set_vel(N, 0)                 # 设置点O在参考系N中的速度为零
    C.set_vel(N, u1 * N.x)          # 设置点C在参考系N中的速度为u1*N.x
    Ao.v2pt_theory(C, N, A)         # 根据点C的速度推导出点Ao的速度
    Cart = Particle('Cart', C, mC)  # 创建小车粒子对象Cart，质量为mC，位置在点C处
    Pendulum = RigidBody('Pendulum', Ao, A, mA, (inertia(A, Ix, Iy, Iz), C))
                                    # 创建摆的刚体对象Pendulum，质心在点Ao处，质量为mA，惯性张量为(inertia(A, Ix, Iy, Iz), C)

    # 运动微分方程

    kindiffs = [q1d - u1, q2d - u2]  # 定义运动学微分方程

    bodyList = [Cart, Pendulum]     # 将小车和摆加入物体列表

    forceList = [(Ao, -N.y * gravity * mA),     # 定义作用在点Ao的重力力和
                 (C, -N.y * gravity * mC),      # 定义作用在点C的重力力和
                 (C, -N.x * k * (q1 - ls)),     # 定义作用在点C的弹簧力
                 (C, N.x * F)]                  # 定义作用在点C的水平力F

    km = KanesMethod(N, [q1, q2], [u1, u2], kindiffs)  # 创建KanesMethod对象，用于系统动力学分析
    (fr, frstar) = km.kanes_equations(bodyList, forceList)  # 计算Kane方程
    mm = km.mass_matrix_full         # 获得完整的质量矩阵
    assert mm[3, 3] == Iz            # 断言质量矩阵的(3, 3)元素等于惯性矩Iz
def test_input_format():
    # 定义两个广义坐标符号 q 和 u
    q, u = dynamicsymbols('q u')
    # 定义这两个符号的一阶导数 qd 和 ud
    qd, ud = dynamicsymbols('q u', 1)
    # 定义质量、阻尼和弹簧常数 m, c, k
    m, c, k = symbols('m c k')
    # 创建一个惯性参考系 N
    N = ReferenceFrame('N')
    # 创建一个点 P
    P = Point('P')
    # 设置点 P 在参考系 N 中的速度
    P.set_vel(N, u * N.x)

    # 定义一个运动约束 kd
    kd = [qd - u]
    # 定义一个力列表 FL
    FL = [(P, (-k * q - c * u) * N.x)]
    # 创建一个粒子 pa，位于点 P 上，质量为 m
    pa = Particle('pa', P, m)
    # 将粒子 pa 放入体列表 BL
    BL = [pa]

    # 创建 KanesMethod 对象 KM，使用参考系 N、广义坐标 q 和 u，运动约束为 kd
    KM = KanesMethod(N, [q], [u], kd)
    # 测试 kane.kanes_equations((body1, body2, particle1)) 的输入格式
    assert KM.kanes_equations(BL)[0] == Matrix([0])
    # 测试 kane.kanes_equations(bodies=(body1, body 2), loads=(load1,load2)) 的输入格式
    assert KM.kanes_equations(bodies=BL, loads=None)[0] == Matrix([0])
    # 测试 kane.kanes_equations(bodies=(body1, body 2), loads=None) 的输入格式
    assert KM.kanes_equations(BL, loads=None)[0] == Matrix([0])
    # 测试 kane.kanes_equations(bodies=(body1, body 2)) 的输入格式
    assert KM.kanes_equations(BL)[0] == Matrix([0])
    # 测试 kane.kanes_equations(bodies=(body1, body2), loads=[]) 的输入格式
    assert KM.kanes_equations(BL, [])[0] == Matrix([0])
    # 测试当提供错误的力列表（在这里是一个字符串）时是否引发 ValueError 异常
    raises(ValueError, lambda: KM._form_fr('bad input'))

    # 使用 BL 和 FL 实例化 KanesMethod 对象 KM，创建一个广义坐标为 q 和 u 的系统
    KM = KanesMethod(N, [q], [u], kd, bodies=BL, forcelist=FL)
    # 测试 kane.kanes_equations() 的输出结果
    assert KM.kanes_equations()[0] == Matrix([-c*u - k*q])

    # 定义另外两个广义坐标符号 q1, q2 和其一阶导数 u1, u2
    q1, q2, u1, u2 = dynamicsymbols('q1 q2 u1 u2')
    # 定义这两个符号的一阶导数 q1d, q2d, u1d, u2d
    q1d, q2d, u1d, u2d = dynamicsymbols('q1 q2 u1 u2', 1)
    # 定义质量 m 和阻尼、弹簧常数 c1, c2, k1, k2
    m, c1, c2, k1, k2 = symbols('m c1 c2 k1 k2')
    # 创建一个惯性参考系 N
    N = ReferenceFrame('N')
    # 创建两个点 P1, P2
    P1 = Point('P1')
    P2 = Point('P2')
    # 设置点 P1 在参考系 N 中的速度
    P1.set_vel(N, u1 * N.x)
    # 设置点 P2 在参考系 N 中的速度
    P2.set_vel(N, (u1 + u2) * N.x)
    # 定义运动约束 kd
    kd = [q1d - u1, q2d - u2]

    # 定义力列表 FL
    FL = ((P1, (-k1 * q1 - c1 * u1 + k2 * q2 + c2 * u2) * N.x), (P2, (-k2 *
        q2 - c2 * u2) * N.x))
    # 创建两个粒子 pa1, pa2，位于点 P1, P2 上，质量为 m
    pa1 = Particle('pa1', P1, m)
    pa2 = Particle('pa2', P2, m)
    # 将粒子 pa1, pa2 放入体列表 BL
    BL = (pa1, pa2)

    # 创建 KanesMethod 对象 KM，使用广义坐标 q1, q2 和速度 u1, u2，运动约束为 kd
    KM = KanesMethod(N, q_ind=[q1, q2], u_ind=[u1, u2], kd_eqs=kd)
    # 测试 kane.kanes_equations((body1, body2), (load1, load2)) 的输入格式
    KM.kanes_equations(BL, FL)
    # 获取 KM 的质量矩阵 MM 和外力矩阵 forcing
    MM = KM.mass_matrix
    forcing = KM.forcing
    # 计算方程组的右手边 rhs
    rhs = MM.inv() * forcing
    # 检查第一个方程的右侧是否展开正确
    assert expand(rhs[0]) == expand((-k1 * q1 - c1 * u1 + k2 * q2 + c2 * u2)/m)
    # 检查第二个方程的右侧是否展开正确
    assert expand(rhs[1]) == expand((k1 * q1 + c1 * u1 - 2 * k2 * q2 - 2 *
                                    c2 * u2) / m)


def test_implicit_kinematics():
    # 测试隐式运动学是否能处理复杂方程，显式形式难以处理
    # 参考：https://github.com/sympy/sympy/issues/22626

    # 惯性参考系 NED
    NED = ReferenceFrame('NED')
    # 创建点 NED_o，速度设为 0
    NED_o = Point('NED_o')
    NED_o.set_vel(NED, 0)

    # 航向角的广义坐标 q_att
    q_att = dynamicsymbols('lambda_0:4', real=True)
    # 使用四元数 q_att 定义参考系 B
    B = NED.orientnew('B', 'Quaternion', q_att)

    # 广义坐标 q_pos，代表位置坐标 B_x, B_y, B_z
    q_pos = dynamicsymbols('B_x:z')
    # 定义质心B_cm的位置向量相对于NED参考系
    B_cm = NED_o.locatenew('B_cm', q_pos[0]*B.x + q_pos[1]*B.y + q_pos[2]*B.z)

    # 将姿态变量q_att和位置变量q_pos合并为广义坐标q_ind和q_dep
    q_ind = q_att[1:] + q_pos
    q_dep = [q_att[0]]

    # 初始化运动方程列表
    kinematic_eqs = []

    # 计算B相对于NED的角速度B_ang_vel，并定义广义速度P, Q, R
    B_ang_vel = B.ang_vel_in(NED)
    P, Q, R = dynamicsymbols('P Q R')
    B.set_ang_vel(NED, P*B.x + Q*B.y + R*B.z)

    # 计算B相对于NED的角速度变化率B_ang_vel_kd
    B_ang_vel_kd = (B.ang_vel_in(NED) - B_ang_vel).simplify()

    # 将角速度变化率与各个单位向量B.x, B.y, B.z结合，形成运动方程
    kinematic_eqs += [
        B_ang_vel_kd & B.x,
        B_ang_vel_kd & B.y,
        B_ang_vel_kd & B.z
    ]

    # 计算B_cm相对于NED的速度B_cm_vel，并定义广义速度U, V, W
    B_cm_vel = B_cm.vel(NED)
    U, V, W = dynamicsymbols('U V W')
    B_cm.set_vel(NED, U*B.x + V*B.y + W*B.z)

    # 计算参考速度变化率B_ref_vel_kd
    B_ref_vel_kd = (B_cm.vel(NED) - B_cm_vel)

    # 用B.x, B.y, B.z单位向量点乘参考速度变化率，得到运动方程
    kinematic_eqs += [
        B_ref_vel_kd & B.x,
        B_ref_vel_kd & B.y,
        B_ref_vel_kd & B.z,
    ]

    # 定义广义速度u_ind，包括U, V, W, P, Q, R
    u_ind = [U, V, W, P, Q, R]

    # 定义姿态向量q_att_vec，配置约束为单位范数
    q_att_vec = Matrix(q_att)
    config_cons = [(q_att_vec.T*q_att_vec)[0] - 1]  # unit norm

    # 将姿态变化率添加到运动方程列表
    kinematic_eqs += [(q_att_vec.T * q_att_vec.diff())[0]]

    try:
        # 初始化KanesMethod对象，用于系统动力学分析
        KM = KanesMethod(NED, q_ind, u_ind,
          q_dependent=q_dep,
          kd_eqs=kinematic_eqs,
          configuration_constraints=config_cons,
          velocity_constraints=[],
          u_dependent=[],  # 没有依赖速度
          u_auxiliary=[],  # 没有辅助速度
          explicit_kinematics=False  # 使用隐式运动方程
        )
    except Exception as e:
        raise e

    # 定义质量M_B和相对于质心B_cm的惯性张量J_B
    M_B = symbols('M_B')
    J_B = inertia(B, *[S(f'J_B_{ax}')*(1 if ax[0] == ax[1] else -1)
            for ax in ['xx', 'yy', 'zz', 'xy', 'yz', 'xz']])
    J_B = J_B.subs({S('J_B_xy'): 0, S('J_B_yz'): 0})

    # 定义刚体RB，包括质心B_cm、刚体B、质量M_B和惯性张量J_B
    RB = RigidBody('RB', B_cm, B, M_B, (J_B, B_cm))

    # 定义刚体列表，只包含一个RB刚体
    rigid_bodies = [RB]

    # 定义力列表，包括重力、输入在体坐标系B中的力和扭矩
    force_list = [
        (RB.masscenter, RB.mass*S('g')*NED.z),  # 重力指向下
        (RB.frame, dynamicsymbols('T_z')*B.z),   # 体坐标系中的通用力矩
        (RB.masscenter, dynamicsymbols('F_z')*B.z)  # 体坐标系中的通用力
    ]

    # 计算Kane方程
    KM.kanes_equations(rigid_bodies, force_list)

    # 估计隐式形式的运动学矩阵计算量占总运算量的百分比不超过5%
    n_ops_implicit = sum(
        [x.count_ops() for x in KM.forcing_full] +
        [x.count_ops() for x in KM.mass_matrix_full]
    )

    # 保存隐式运动学矩阵以备后用
    mass_matrix_kin_implicit = KM.mass_matrix_kin
    forcing_kin_implicit = KM.forcing_kin

    # 设置KanesMethod对象使用显式运动学方程
    KM.explicit_kinematics = True
    # 计算显式操作数的数量，包括强制操作和质量矩阵操作的操作数总和
    n_ops_explicit = sum(
        [x.count_ops() for x in KM.forcing_full] +
        [x.count_ops() for x in KM.mass_matrix_full]
    )
    # 获取显式强制运动学方程
    forcing_kin_explicit = KM.forcing_kin

    # 断言隐式操作数与显式操作数的比值小于0.05，用于验证隐式方程的复杂性
    assert n_ops_implicit / n_ops_explicit < .05

    # 理想情况下，我们会检查隐式方程和显式方程在结果上是否一致，类似于 test_one_dof 中的测试
    # 但隐式方程的存在是为了处理显式形式过于复杂的问题，特别是角度部分（即测试会过于缓慢）
    # 相反，我们使用更基本的测试来验证运动学方程是否正确：
    #
    # (1) 确保我们恢复了提供的运动学方程
    assert (mass_matrix_kin_implicit * KM.q.diff() - forcing_kin_implicit) == Matrix(kinematic_eqs)

    # (2) 确保四元数的变化率与“教科书”解决方案一致
    # 注意，我们对线性速度只使用显式运动学方程，因为它们没有角度部分那么复杂
    qdot_candidate = forcing_kin_explicit

    # 计算四元数变化率的“教科书”解决方案
    quat_dot_textbook = Matrix([
        [0, -P, -Q, -R],
        [P,  0,  R, -Q],
        [Q, -R,  0,  P],
        [R,  Q, -P,  0],
    ]) * q_att_vec / 2

    # 再次强调，如果我们不使用这个“教科书”解决方案
    # sympy 将很难处理与四元数速率相关的项，因为涉及的操作太多
    qdot_candidate[-1] = quat_dot_textbook[0]  # lambda_0，注意 [-1] 是因为 sympy 的 Kane 方法将依赖坐标放在最后
    qdot_candidate[0]  = quat_dot_textbook[1]  # lambda_1
    qdot_candidate[1]  = quat_dot_textbook[2]  # lambda_2
    qdot_candidate[2]  = quat_dot_textbook[3]  # lambda_3

    # 将配置约束代入候选解，并与隐式右手边进行比较
    lambda_0_sol = solve(config_cons[0], q_att_vec[0])[1]
    lhs_candidate = simplify(mass_matrix_kin_implicit * qdot_candidate).subs({q_att_vec[0]: lambda_0_sol})
    assert lhs_candidate == forcing_kin_implicit
# 定义测试函数，用于测试 issue 24887
def test_issue_24887():
    # 定义符号变量 g, l, m, c
    g, l, m, c = symbols('g l m c')
    # 定义动力学符号 q1, q2, q3, u1, u2, u3
    q1, q2, q3, u1, u2, u3 = dynamicsymbols('q1:4 u1:4')
    # 定义参考坐标系 N
    N = ReferenceFrame('N')
    # 定义参考坐标系 A
    A = ReferenceFrame('A')
    # 在参考坐标系 N 中用 (q1, q2, q3) 和 'zxy' 方式设置参考坐标系 A
    A.orient_body_fixed(N, (q1, q2, q3), 'zxy')
    # 计算 A 相对于 N 的角速度
    N_w_A = A.ang_vel_in(N)
    # 设置 A 相对于 N 的角速度为 u1 * A.x + u2 * A.y + u3 * A.z
    # A.set_ang_vel(N, u1 * A.x + u2 * A.y + u3 * A.z)
    # 定义运动学描述 kdes，为 N_w_A 在 A.x, A.y, A.z 方向上的分量减去对应的速度 u1, u2, u3
    kdes = [N_w_A.dot(A.x) - u1, N_w_A.dot(A.y) - u2, N_w_A.dot(A.z) - u3]
    # 定义点 O
    O = Point('O')
    # 设置点 O 在参考坐标系 N 中的速度为 0
    O.set_vel(N, 0)
    # 在点 O 处定义 Po，位置为 -l * A.y
    Po = O.locatenew('Po', -l * A.y)
    # 设置 Po 在参考坐标系 A 中的速度为 0
    Po.set_vel(A, 0)
    # 定义质点 P，位于 Po 处，质量为 m
    P = Particle('P', Po, m)
    # 定义 Kanes 方法 kane，使用参考坐标系 N，广义坐标为 [q1, q2, q3]，广义速度为 [u1, u2, u3]，运动学描述为 kdes，包含质点 P
    # 外力列表包含 (Po, -m * g * N.y)
    kane = KanesMethod(N, [q1, q2, q3], [u1, u2, u3], kdes, bodies=[P],
                       forcelist=[(Po, -m * g * N.y)])
    # 计算 Kane 方程
    kane.kanes_equations()
    # 期望的质量矩阵 expected_md
    expected_md = m * l ** 2 * Matrix([[1, 0, 0], [0, 0, 0], [0, 0, 1]])
    # 期望的广义力矩阵 expected_fd
    expected_fd = Matrix([
        [l*m*(g*(sin(q1)*sin(q3) - sin(q2)*cos(q1)*cos(q3)) - l*u2*u3)],
        [0], [l*m*(-g*(sin(q1)*cos(q3) + sin(q2)*sin(q3)*cos(q1)) + l*u1*u2)]])
    # 断言：Kane 方程中的广义坐标包含于 {q1, q2, q3, u1, u2, u3} 中
    assert find_dynamicsymbols(kane.forcing).issubset({q1, q2, q3, u1, u2, u3})
    # 断言：简化后的质量矩阵与期望的质量矩阵相等
    assert simplify(kane.mass_matrix - expected_md) == zeros(3, 3)
    # 断言：简化后的广义力矩阵与期望的广义力矩阵相等
    assert simplify(kane.forcing - expected_fd) == zeros(3, 1)
```