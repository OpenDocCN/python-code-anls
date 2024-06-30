# `D:\src\scipysrc\sympy\sympy\physics\mechanics\tests\test_rigidbody.py`

```
from sympy.physics.mechanics import Point, ReferenceFrame, Dyadic, RigidBody
from sympy.physics.mechanics import dynamicsymbols, outer, inertia, Inertia
from sympy.physics.mechanics import inertia_of_point_mass
from sympy import expand, zeros, simplify, symbols
from sympy.testing.pytest import raises, warns_deprecated_sympy

# 定义测试函数，测试默认情况下的刚体属性
def test_rigidbody_default():
    # 创建一个名为'B'的刚体对象
    b = RigidBody('B')
    # 定义该刚体的惯性张量I，使用符号B_ixx, B_iyy等作为参数
    I = inertia(b.frame, *symbols('B_ixx B_iyy B_izz B_ixy B_iyz B_izx'))
    # 断言刚体的名称为'B'
    assert b.name == 'B'
    # 断言刚体的质量为符号'B_mass'
    assert b.mass == symbols('B_mass')
    # 断言刚体质心的名称为'B_masscenter'
    assert b.masscenter.name == 'B_masscenter'
    # 断言刚体的惯性为(I, B_masscenter)
    assert b.inertia == (I, b.masscenter)
    # 断言刚体的中心惯性为I
    assert b.central_inertia == I
    # 断言刚体的参考框架名称为'B_frame'
    assert b.frame.name == 'B_frame'
    # 断言刚体的字符串表示为'B'
    assert b.__str__() == 'B'
    # 断言刚体的正式表示
    assert b.__repr__() == (
        "RigidBody('B', masscenter=B_masscenter, frame=B_frame, mass=B_mass, "
        "inertia=Inertia(dyadic=B_ixx*(B_frame.x|B_frame.x) + "
        "B_ixy*(B_frame.x|B_frame.y) + B_izx*(B_frame.x|B_frame.z) + "
        "B_ixy*(B_frame.y|B_frame.x) + B_iyy*(B_frame.y|B_frame.y) + "
        "B_iyz*(B_frame.y|B_frame.z) + B_izx*(B_frame.z|B_frame.x) + "
        "B_iyz*(B_frame.z|B_frame.y) + B_izz*(B_frame.z|B_frame.z), "
        "point=B_masscenter))")

# 定义测试函数，测试刚体的各种属性和方法
def test_rigidbody():
    # 定义符号变量
    m, m2, v1, v2, v3, omega = symbols('m m2 v1 v2 v3 omega')
    # 创建一个惯性参考框架'A'
    A = ReferenceFrame('A')
    # 创建另一个惯性参考框架'A2'
    A2 = ReferenceFrame('A2')
    # 创建一个质点'P'
    P = Point('P')
    # 创建另一个质点'P2'
    P2 = Point('P2')
    # 创建一个零惯性张量
    I = Dyadic(0)
    # 创建另一个零惯性张量
    I2 = Dyadic(0)
    # 创建刚体'B'，指定质点P，参考框架A，质量m，惯性(I, P)
    B = RigidBody('B', P, A, m, (I, P))
    # 断言刚体'B'的质量为m
    assert B.mass == m
    # 断言刚体'B'的参考框架为A
    assert B.frame == A
    # 断言刚体'B'的质心为P
    assert B.masscenter == P
    # 断言刚体'B'的惯性为(I, P)
    assert B.inertia == (I, B.masscenter)

    # 修改刚体'B'的属性
    B.mass = m2
    B.frame = A2
    B.masscenter = P2
    B.inertia = (I2, B.masscenter)
    
    # 测试错误情况：质心和惯性参数不匹配
    raises(TypeError, lambda: RigidBody(P, P, A, m, (I, P)))
    raises(TypeError, lambda: RigidBody('B', P, P, m, (I, P)))
    raises(TypeError, lambda: RigidBody('B', P, A, m, (P, P)))
    raises(TypeError, lambda: RigidBody('B', P, A, m, (I, I)))
    
    # 断言刚体'B'的字符串表示为'B'
    assert B.__str__() == 'B'
    # 断言刚体'B'的质量为m2
    assert B.mass == m2
    # 断言刚体'B'的参考框架为A2
    assert B.frame == A2
    # 断言刚体'B'的质心为P2
    assert B.masscenter == P2
    # 断言刚体'B'的惯性为(I2, P2)
    assert B.inertia == (I2, B.masscenter)
    # 断言刚体'B'的惯性为Inertia类型
    assert isinstance(B.inertia, Inertia)

    # 测试线性动量函数
    N = ReferenceFrame('N')
    P2.set_vel(N, v1 * N.x + v2 * N.y + v3 * N.z)
    assert B.linear_momentum(N) == m2 * (v1 * N.x + v2 * N.y + v3 * N.z)

# 定义测试函数，测试刚体的角动量和势能
def test_rigidbody2():
    # 定义动力学符号变量
    M, v, r, omega, g, h = dynamicsymbols('M v r omega g h')
    # 创建惯性参考框架N和b
    N = ReferenceFrame('N')
    b = ReferenceFrame('b')
    # 设置参考框架b的角速度
    b.set_ang_vel(N, omega * b.x)
    # 创建一个质点P
    P = Point('P')
    # 创建一个外积
    I = outer(b.x, b.x)
    # 定义惯性元组
    Inertia_tuple = (I, P)
    # 创建刚体B，指定质点P，参考框架b，质量M，惯性元组(Inertia_tuple)
    B = RigidBody('B', P, b, M, Inertia_tuple)
    # 设置质点P的速度
    P.set_vel(N, v * b.x)
    # 断言刚体B关于质点P和参考框架N的角动量
    assert B.angular_momentum(P, N) == omega * b.x
    # 创建一个质点O
    O = Point('O')
    # 设置质点O的速度
    O.set_vel(N, v * b.x)
    # 设置质点P相对于质点O的位置
    P.set_pos(O, r * b.y)
    # 断言刚体B关于质点O和参考框架N的角动量
    assert B.angular_momentum(O, N) == omega * b.x - M*v*r*b.z
    # 设置刚体B的势能
    B.potential_energy = M * g * h
    # 断言刚体B的势能
    assert B.potential_energy == M * g * h
    # 断言：验证以下表达式是否成立
    assert expand(2 * B.kinetic_energy(N)) == omega**2 + M * v**2
def test_rigidbody3():
    # 定义动力学符号
    q1, q2, q3, q4 = dynamicsymbols('q1:5')
    # 定义位置符号
    p1, p2, p3 = symbols('p1:4')
    # 定义质量符号
    m = symbols('m')

    # 定义参考坐标系A
    A = ReferenceFrame('A')
    # 在A坐标系下，用q1绕A.x轴旋转的坐标系B
    B = A.orientnew('B', 'axis', [q1, A.x])
    # 创建一个点O
    O = Point('O')
    # 设置点O在A坐标系下的速度
    O.set_vel(A, q2*A.x + q3*A.y + q4*A.z)
    # 在点O处，相对于A坐标系，通过位置矢量p1*B.x + p2*B.y + p3*B.z定义点P
    P = O.locatenew('P', p1*B.x + p2*B.y + p3*B.z)
    # 计算P相对于O的速度，考虑到A和B之间的运动关系
    P.v2pt_theory(O, A, B)
    # 计算B.x在B坐标系下的外积
    I = outer(B.x, B.x)

    # 创建刚体rb1，表示P点相对于B坐标系的刚体，质量为m，惯性张量为I
    rb1 = RigidBody('rb1', P, B, m, (I, P))
    # 创建刚体rb2，与rb1相同，但使用直接给定的惯性张量
    rb2 = RigidBody('rb2', P, B, m,
                    (I + inertia_of_point_mass(m, P.pos_from(O), B), O))

    # 断言rb1和rb2的中心惯性相等
    assert rb1.central_inertia == rb2.central_inertia
    # 断言rb1和rb2相对于点O在A坐标系下的角动量相等
    assert rb1.angular_momentum(O, A) == rb2.angular_momentum(O, A)


def test_pendulum_angular_momentum():
    """Consider a pendulum of length OA = 2a, of mass m as a rigid body of
    center of mass G (OG = a) which turn around (O,z). The angle between the
    reference frame R and the rod is q.  The inertia of the body is I =
    (G,0,ma^2/3,ma^2/3). """

    # 定义符号m和a
    m, a = symbols('m, a')
    # 定义动态符号q
    q = dynamicsymbols('q')

    # 定义参考坐标系R
    R = ReferenceFrame('R')
    # 在R坐标系下，用q绕R.z轴旋转的坐标系R1
    R1 = R.orientnew('R1', 'Axis', [q, R.z])
    # 设置R1相对于R坐标系的角速度
    R1.set_ang_vel(R, q.diff() * R.z)

    # 定义刚体的惯性张量
    I = inertia(R1, 0, m * a**2 / 3, m * a**2 / 3)

    # 创建点O
    O = Point('O')

    # 在O点处，相对于R坐标系，通过位置矢量2*a*R1.x定义点A
    A = O.locatenew('A', 2*a * R1.x)
    # 在O点处，相对于R坐标系，通过位置矢量a*R1.x定义质心G
    G = O.locatenew('G', a * R1.x)

    # 创建刚体S，表示质心G相对于坐标系R1的刚体，质量为m，惯性张量为I
    S = RigidBody('S', G, R1, m, (I, G))

    # 设置点O在R坐标系下的速度为0
    O.set_vel(R, 0)
    # 计算点A相对于O的速度，考虑到R和R1之间的运动关系
    A.v2pt_theory(O, R, R1)
    # 计算质心G相对于O的速度，考虑到R和R1之间的运动关系
    G.v2pt_theory(O, R, R1)

    # 断言角动量方程成立
    assert (4 * m * a**2 / 3 * q.diff() * R.z -
            S.angular_momentum(O, R).express(R)) == 0


def test_rigidbody_inertia():
    # 定义参考坐标系N
    N = ReferenceFrame('N')
    # 定义符号m, Ix, Iy, Iz, a, b
    m, Ix, Iy, Iz, a, b = symbols('m, I_x, I_y, I_z, a, b')
    # 定义Io为坐标系N下的惯性张量
    Io = inertia(N, Ix, Iy, Iz)
    # 创建点o
    o = Point('o')
    # 在o点处，相对于N坐标系，通过位置矢量a*N.x + b*N.y定义点p
    p = o.locatenew('p', a * N.x + b * N.y)
    # 创建刚体R，表示o点相对于N坐标系的刚体，质量为m，惯性张量为Io
    R = RigidBody('R', o, N, m, (Io, p))
    # 计算Io_check，作为o点相对于N坐标系的中心惯性
    I_check = inertia(N, Ix - b ** 2 * m, Iy - a ** 2 * m,
                      Iz - m * (a ** 2 + b ** 2), m * a * b)
    # 断言R的惯性是Inertia对象
    assert isinstance(R.inertia, Inertia)
    # 断言R的惯性与给定的值Io, p相等
    assert R.inertia == (Io, p)
    # 断言R的中心惯性与计算的I_check相等
    assert R.central_inertia == I_check
    # 修改R的中心惯性为Io
    R.central_inertia = Io
    # 断言R的惯性为Io, o
    assert R.inertia == (Io, o)
    # 断言R的中心惯性为Io
    assert R.central_inertia == Io
    # 修改R的惯性为Io, p
    R.inertia = (Io, p)
    # 断言R的惯性为Io, p
    assert R.inertia == (Io, p)
    # 断言R的中心惯性为I_check
    assert R.central_inertia == I_check
    # 解析Inertia对象
    R.inertia = Inertia(Io, o)
    # 断言R的惯性为Io, o
    assert R.inertia == (Io, o)


def test_parallel_axis():
    # 定义参考坐标系N
    N = ReferenceFrame('N')
    # 定义符号m, Ix, Iy, Iz, a, b
    m, Ix, Iy, Iz, a, b = symbols('m, I_x, I_y, I_z, a, b')
    # 定义Io为坐标系N下的惯性张量
    Io = inertia(N, Ix, Iy, Iz)
    # 创建点o
    o = Point('o')
    # 在o点处，相对于N坐标系，通过位置矢量a*N.x + b*N.y定义点p
    p = o.locatenew('p', a * N.x + b * N.y)
    # 创建刚体R，表示o点相对于N坐标系的刚体，质量为m，惯性张量为Io
    R = RigidBody('R', o, N, m, (Io, o))
    # 计算Ip，表示R相对于点p的平行轴定理的惯性张量
    Ip = R.parallel_axis(p)
    # 计算预期的Ip_expected
    Ip
    # 定义符号变量 m, g, h，并分配给对应的符号对象
    m, g, h = symbols('m g h')
    
    # 创建一个参考坐标系 A
    A = ReferenceFrame('A')
    
    # 创建一个点 P
    P = Point('P')
    
    # 创建一个惯性二阶张量 I，初始为零
    I = Dyadic(0)
    
    # 创建一个刚体 B，命名为 'B'，位于点 P，相对于参考坐标系 A，具有质量 m，和惯性矩阵 (I, P)
    B = RigidBody('B', P, A, m, (I, P))
    
    # 使用 warns_deprecated_sympy() 上下文管理器，设置刚体 B 的势能为 m*g*h
    with warns_deprecated_sympy():
        B.set_potential_energy(m*g*h)
```