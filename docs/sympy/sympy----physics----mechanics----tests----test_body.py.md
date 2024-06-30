# `D:\src\scipysrc\sympy\sympy\physics\mechanics\tests\test_body.py`

```
from sympy import (Symbol, symbols, sin, cos, Matrix, zeros,
                    simplify)
from sympy.physics.vector import Point, ReferenceFrame, dynamicsymbols, Dyadic
from sympy.physics.mechanics import inertia, Body
from sympy.testing.pytest import raises, warns_deprecated_sympy

# 定义测试函数 test_default，用于测试默认情况下的 Body 类
def test_default():
    # 捕获即将弃用的警告
    with warns_deprecated_sympy():
        # 创建一个名为 'body' 的 Body 对象
        body = Body('body')
    # 断言 Body 对象的名称为 'body'
    assert body.name == 'body'
    # 断言 Body 对象的加载为空列表
    assert body.loads == []
    # 创建一个名为 'body_masscenter' 的 Point 对象
    point = Point('body_masscenter')
    # 将 'body_masscenter' 相对于 body.frame 的速度设为 0
    point.set_vel(body.frame, 0)
    # 获取 Body 对象的质心
    com = body.masscenter
    # 获取 Body 对象的参考框架
    frame = body.frame
    # 断言质心 com 相对于参考框架 frame 的速度与 point 的速度相等
    assert com.vel(frame) == point.vel(frame)
    # 断言 Body 对象的质量为 Symbol('body_mass')
    assert body.mass == Symbol('body_mass')
    # 定义惯性张量的符号变量
    ixx, iyy, izz = symbols('body_ixx body_iyy body_izz')
    ixy, iyz, izx = symbols('body_ixy body_iyz body_izx')
    # 断言 Body 对象的惯性张量与给定的符号变量匹配
    assert body.inertia == (inertia(body.frame, ixx, iyy, izz, ixy, iyz, izx),
                            body.masscenter)

# 定义测试函数 test_custom_rigid_body，用于测试自定义刚体的情况
def test_custom_rigid_body():
    # 创建质量中心为 'rigidbody_masscenter' 的 Point 对象
    rigidbody_masscenter = Point('rigidbody_masscenter')
    # 定义刚体的质量符号变量
    rigidbody_mass = Symbol('rigidbody_mass')
    # 创建参考框架为 'rigidbody_frame' 的 ReferenceFrame 对象
    rigidbody_frame = ReferenceFrame('rigidbody_frame')
    # 计算刚体的惯性
    body_inertia = inertia(rigidbody_frame, 1, 0, 0)
    # 捕获即将弃用的警告
    with warns_deprecated_sympy():
        # 创建一个名为 'rigidbody_body' 的刚体 Body 对象
        rigid_body = Body('rigidbody_body', rigidbody_masscenter,
                          rigidbody_mass, rigidbody_frame, body_inertia)
    # 获取刚体的质心
    com = rigid_body.masscenter
    # 获取刚体的参考框架
    frame = rigid_body.frame
    # 断言质心 com 相对于参考框架 frame 的速度与 rigidbody_masscenter 的速度相等
    rigidbody_masscenter.set_vel(rigidbody_frame, 0)
    assert com.vel(frame) == rigidbody_masscenter.vel(frame)
    # 断言质心 com 相对于自身的位置与 rigidbody_masscenter 的位置相等
    assert com.pos_from(com) == rigidbody_masscenter.pos_from(com)

    # 断言刚体的质量等于 rigidbody_mass
    assert rigid_body.mass == rigidbody_mass
    # 断言刚体的惯性张量与预期的 body_inertia 和质心匹配
    assert rigid_body.inertia == (body_inertia, rigidbody_masscenter)

    # 断言刚体为刚体
    assert rigid_body.is_rigidbody

    # 断言刚体对象具有属性：'masscenter'、'mass'、'frame'、'inertia'
    assert hasattr(rigid_body, 'masscenter')
    assert hasattr(rigid_body, 'mass')
    assert hasattr(rigid_body, 'frame')
    assert hasattr(rigid_body, 'inertia')

# 定义测试函数 test_particle_body，用于测试粒子体的情况
def test_particle_body():
    # 创建质量中心为 'particle_masscenter' 的 Point 对象
    particle_masscenter = Point('particle_masscenter')
    # 定义粒子的质量符号变量
    particle_mass = Symbol('particle_mass')
    # 创建参考框架为 'particle_frame' 的 ReferenceFrame 对象
    particle_frame = ReferenceFrame('particle_frame')
    # 捕获即将弃用的警告
    with warns_deprecated_sympy():
        # 创建一个名为 'particle_body' 的粒子 Body 对象
        particle_body = Body('particle_body', particle_masscenter,
                             particle_mass, particle_frame)
    # 获取粒子的质心
    com = particle_body.masscenter
    # 获取粒子的参考框架
    frame = particle_body.frame
    # 断言质心 com 相对于参考框架 frame 的速度与 particle_masscenter 的速度相等
    particle_masscenter.set_vel(particle_frame, 0)
    assert com.vel(frame) == particle_masscenter.vel(frame)
    # 断言质心 com 相对于自身的位置与 particle_masscenter 的位置相等
    assert com.pos_from(com) == particle_masscenter.pos_from(com)

    # 断言粒子的质量等于 particle_mass
    assert particle_body.mass == particle_mass
    # 断言粒子没有惯性张量属性
    assert not hasattr(particle_body, "_inertia")
    # 断言粒子对象具有属性：'frame'、'masscenter'、'mass'
    assert hasattr(particle_body, 'frame')
    assert hasattr(particle_body, 'masscenter')
    assert hasattr(particle_body, 'mass')
    # 断言粒子的惯性张量与预期的 Dyadic(0) 和质心匹配
    assert particle_body.inertia == (Dyadic(0), particle_body.masscenter)
    # 断言粒子的中心惯性张量为 Dyadic(0)
    assert particle_body.central_inertia == Dyadic(0)
    # 断言粒子不是刚体
    assert not particle_body.is_rigidbody

    # 设置粒子的中心惯性张量
    particle_body.central_inertia = inertia(particle_frame, 1, 1, 1)
    # 断言粒子的质心惯性与给定的惯性参数一致
    assert particle_body.central_inertia == inertia(particle_frame, 1, 1, 1)
    # 断言粒子体是刚体
    assert particle_body.is_rigidbody

    # 使用警告函数 warns_deprecated_sympy() 包裹，标记 sympy 的过时用法警告
    with warns_deprecated_sympy():
        # 创建一个名为 'particle_body' 的质点体，设置质量为 particle_mass
        particle_body = Body('particle_body', mass=particle_mass)
    # 断言粒子体不是刚体（因为上面创建时可能被警告影响）
    assert not particle_body.is_rigidbody

    # 在粒子体的质心位置创建一个名为 'point' 的点
    point = particle_body.masscenter.locatenew('point', particle_body.x)
    # 计算质心点的惯性，为质量乘以在给定坐标系中的惯性
    point_inertia = particle_mass * inertia(particle_body.frame, 0, 1, 1)
    # 将质心点的惯性设置为粒子体的惯性属性
    particle_body.inertia = (point_inertia, point)
    # 断言粒子体的惯性与刚刚设置的惯性一致
    assert particle_body.inertia == (point_inertia, point)
    # 断言粒子的中心惯性为零
    assert particle_body.central_inertia == Dyadic(0)
    # 断言粒子体是刚体
    assert particle_body.is_rigidbody
def test_particle_body_add_force():
    # 定义粒子的质心点
    particle_masscenter = Point('particle_masscenter')
    # 定义粒子的质量
    particle_mass = Symbol('particle_mass')
    # 定义粒子的参考系
    particle_frame = ReferenceFrame('particle_frame')
    # 使用已弃用警告环境创建粒子的刚体
    with warns_deprecated_sympy():
        particle_body = Body('particle_body', particle_masscenter,
                             particle_mass, particle_frame)

    # 定义力的大小符号
    a = Symbol('a')
    # 计算力的向量，方向为粒子刚体的x轴
    force_vector = a * particle_body.frame.x
    # 在粒子的质心点施加力
    particle_body.apply_force(force_vector, particle_body.masscenter)
    # 断言负载列表中的长度为1
    assert len(particle_body.loads) == 1
    # 在质心处创建一个新的点
    point = particle_body.masscenter.locatenew(
        particle_body._name + '_point0', 0)
    # 设置新点在粒子参考系下的速度为0
    point.set_vel(particle_body.frame, 0)
    # 获取施加力的点
    force_point = particle_body.loads[0][0]

    # 断言力点在参考系下的速度与新点相同
    frame = particle_body.frame
    assert force_point.vel(frame) == point.vel(frame)
    # 断言力点到自身的位置与新点到力点的位置相同
    assert force_point.pos_from(force_point) == point.pos_from(force_point)

    # 断言负载列表中第一个力的向量与定义的力向量相同
    assert particle_body.loads[0][1] == force_vector


def test_body_add_force():
    # 定义刚体的质心点
    rigidbody_masscenter = Point('rigidbody_masscenter')
    # 定义刚体的质量
    rigidbody_mass = Symbol('rigidbody_mass')
    # 定义刚体的参考系
    rigidbody_frame = ReferenceFrame('rigidbody_frame')
    # 定义刚体的惯性张量
    body_inertia = inertia(rigidbody_frame, 1, 0, 0)
    # 使用已弃用警告环境创建刚体
    with warns_deprecated_sympy():
        rigid_body = Body('rigidbody_body', rigidbody_masscenter,
                          rigidbody_mass, rigidbody_frame, body_inertia)

    # 定义长度符号
    l = Symbol('l')
    # 定义力符号
    Fa = Symbol('Fa')
    # 在刚体的质心点处创建新点
    point = rigid_body.masscenter.locatenew(
        'rigidbody_body_point0',
        l * rigid_body.frame.x)
    # 设置新点在刚体参考系下的速度为0
    point.set_vel(rigid_body.frame, 0)
    # 定义力的向量，方向为刚体参考系的z轴
    force_vector = Fa * rigid_body.frame.z
    # 在指定点施加力
    rigid_body.apply_force(force_vector, point)
    # 断言负载列表中的长度为1
    assert len(rigid_body.loads) == 1
    # 获取施加力的点
    force_point = rigid_body.loads[0][0]
    # 断言力点在参考系下的速度与新点相同
    frame = rigid_body.frame
    assert force_point.vel(frame) == point.vel(frame)
    # 断言力点到自身的位置与新点到力点的位置相同
    assert force_point.pos_from(force_point) == point.pos_from(force_point)
    # 断言负载列表中第一个力的向量与定义的力向量相同
    assert rigid_body.loads[0][1] == force_vector
    # 在不指定点的情况下施加力
    rigid_body.apply_force(force_vector)
    # 断言负载列表中的长度为2
    assert len(rigid_body.loads) == 2
    # 断言负载列表中第二个力的向量与定义的力向量相同
    assert rigid_body.loads[1][1] == force_vector
    # 施加力时传递非点对象引发类型错误
    raises(TypeError, lambda: rigid_body.apply_force(force_vector,  0))
    raises(TypeError, lambda: rigid_body.apply_force(0))


def test_body_add_torque():
    # 使用已弃用警告环境创建刚体
    with warns_deprecated_sympy():
        body = Body('body')
    # 定义力矩的向量，方向为刚体的x轴
    torque_vector = body.frame.x
    # 在刚体上施加力矩
    body.apply_torque(torque_vector)

    # 断言负载列表中的长度为1
    assert len(body.loads) == 1
    # 断言负载列表中的力矩项与定义的力矩向量相同
    assert body.loads[0] == (body.frame, torque_vector)
    # 施加非向量对象引发类型错误
    raises(TypeError, lambda: body.apply_torque(0))


def test_body_masscenter_vel():
    # 使用已弃用警告环境创建刚体A
    with warns_deprecated_sympy():
        A = Body('A')
    # 创建参考系N
    N = ReferenceFrame('N')
    # 使用已弃用警告环境创建刚体B，使用参考系N
    with warns_deprecated_sympy():
        B = Body('B', frame=N)
    # 设置刚体A的质心在参考系N下的速度为N的z轴方向
    A.masscenter.set_vel(N, N.z)
    # 断言刚体A的质心速度相对于刚体B的参考系N为N的z轴方向
    assert A.masscenter_vel(B) == N.z
    # 断言刚体A的质心速度相对于参考系N为N的z轴方向
    assert A.masscenter_vel(N) == N.z


def test_body_ang_vel():
    # 使用已弃用警告环境创建刚体A
    with warns_deprecated_sympy():
        A = Body('A')
    # 创建一个参考框架 'N'
    N = ReferenceFrame('N')
    # 在进入下一个代码块之前，显示即将弃用的警告信息
    with warns_deprecated_sympy():
        # 创建一个名为 'B' 的刚体，其参考框架是 'N'
        B = Body('B', frame=N)
    # 设定对象 'A' 相对于参考框架 'N' 的角速度是沿着 'N' 的 y 轴方向
    A.frame.set_ang_vel(N, N.y)
    # 断言对象 'A' 相对于 'B' 的角速度是 'N' 参考框架中的 y 轴方向
    assert A.ang_vel_in(B) == N.y
    # 断言对象 'B' 相对于 'A' 的角速度是 'N' 参考框架中的负 y 轴方向
    assert B.ang_vel_in(A) == -N.y
    # 断言对象 'A' 相对于 'N' 的角速度是 'N' 参考框架中的 y 轴方向
    assert A.ang_vel_in(N) == N.y
def test_body_dcm():
    with warns_deprecated_sympy():
        A = Body('A')  # 创建名为'A'的刚体对象A
        B = Body('B')  # 创建名为'B'的刚体对象B
    A.frame.orient_axis(B.frame, B.frame.z, 10)  # 用B的z轴对A的坐标系进行轴向定向旋转10度
    assert A.dcm(B) == Matrix([[cos(10), sin(10), 0], [-sin(10), cos(10), 0], [0, 0, 1]])
    assert A.dcm(B.frame) == Matrix([[cos(10), sin(10), 0], [-sin(10), cos(10), 0], [0, 0, 1]])

def test_body_axis():
    N = ReferenceFrame('N')  # 创建一个名为'N'的参考坐标系对象N
    with warns_deprecated_sympy():
        B = Body('B', frame=N)  # 创建一个名为'B'的刚体对象B，并将其框架指定为N
    assert B.x == N.x  # 断言B的x轴与N的x轴相同
    assert B.y == N.y  # 断言B的y轴与N的y轴相同
    assert B.z == N.z  # 断言B的z轴与N的z轴相同

def test_apply_force_multiple_one_point():
    a, b = symbols('a b')
    P = Point('P')  # 创建一个名为'P'的点对象P
    with warns_deprecated_sympy():
        B = Body('B')  # 创建一个名为'B'的刚体对象B
    f1 = a*B.x  # 定义f1为a乘以B的x轴单位向量
    f2 = b*B.y  # 定义f2为b乘以B的y轴单位向量
    B.apply_force(f1, P)  # 在点P上施加力f1到刚体B
    assert B.loads == [(P, f1)]  # 断言B的负载为[(P, f1)]
    B.apply_force(f2, P)  # 在点P上施加力f2到刚体B
    assert B.loads == [(P, f1+f2)]  # 断言B的负载为[(P, f1+f2)]

def test_apply_force():
    f, g = symbols('f g')
    q, x, v1, v2 = dynamicsymbols('q x v1 v2')
    P1 = Point('P1')  # 创建一个名为'P1'的点对象P1
    P2 = Point('P2')  # 创建一个名为'P2'的点对象P2
    with warns_deprecated_sympy():
        B1 = Body('B1')  # 创建一个名为'B1'的刚体对象B1
        B2 = Body('B2')  # 创建一个名为'B2'的刚体对象B2
    N = ReferenceFrame('N')  # 创建一个名为'N'的参考坐标系对象N

    P1.set_vel(B1.frame, v1*B1.x)  # 设置P1在B1框架中的速度为v1乘以B1的x轴单位向量
    P2.set_vel(B2.frame, v2*B2.x)  # 设置P2在B2框架中的速度为v2乘以B2的x轴单位向量
    force = f*q*N.z  # 创建一个时间变化的力force，方向为N的z轴

    B1.apply_force(force, P1, B2, P2)  # 在点P1和P2上施加相等和相反的力到B1和B2上
    assert B1.loads == [(P1, force)]  # 断言B1的负载为[(P1, force)]
    assert B2.loads == [(P2, -force)]  # 断言B2的负载为[(P2, -force)]

    g1 = B1.mass*g*N.y  # 计算B1的质量乘以重力加速度g在N的y轴上的力g1
    g2 = B2.mass*g*N.y  # 计算B2的质量乘以重力加速度g在N的y轴上的力g2

    B1.apply_force(g1)  # 在B1的质心上施加力g1（重力）
    B2.apply_force(g2)  # 在B2的质心上施加力g2（重力）

    assert B1.loads == [(P1, force), (B1.masscenter, g1)]  # 断言B1的负载为[(P1, force), (B1.masscenter, g1)]
    assert B2.loads == [(P2, -force), (B2.masscenter, g2)]  # 断言B2的负载为[(P2, -force), (B2.masscenter, g2)]

    force2 = x*N.x  # 创建一个时间变化的力force2，方向为N的x轴

    B1.apply_force(force2, reaction_body=B2)  # 在B1的质心上施加force2力，同时指定反作用体为B2

    assert B1.loads == [(P1, force), (B1.masscenter, force2+g1)]  # 断言B1的负载为[(P1, force), (B1.masscenter, force2+g1)]
    assert B2.loads == [(P2, -force), (B2.masscenter, -force2+g2)]  # 断言B2的负载为[(P2, -force), (B2.masscenter, -force2+g2)]

def test_apply_torque():
    t = symbols('t')
    q = dynamicsymbols('q')
    with warns_deprecated_sympy():
        B1 = Body('B1')  # 创建一个名为'B1'的刚体对象B1
        B2 = Body('B2')  # 创建一个名为'B2'的刚体对象B2
    N = ReferenceFrame('N')  # 创建一个名为'N'的参考坐标系对象N
    torque = t*q*N.x  # 创建一个时间变化的扭矩torque，方向为N的x轴

    B1.apply_torque(torque, B2)  # 在B1和B2上施加相等和相反的扭矩
    assert B1.loads == [(B1.frame, torque)]  # 断言B1的负载为[(B1.frame, torque)]
    assert B2.loads == [(B2.frame, -torque)]  # 断言B2的负载为[(B2.frame, -torque)]

    torque2 = t*N.y  # 创建一个时间变化的扭矩torque2，方向为N的y轴
    B1.apply_torque(torque2)  # 在B1上施加扭矩torque2
    assert B1.loads == [(B1.frame, torque+torque2)]  # 断言B1的负载为[(B1.frame, torque+torque2)]

def test_clear_load():
    a = symbols('a')
    P = Point('P')  # 创建一个名为'P'的点对象P
    with warns_deprecated_sympy():
        B = Body('B')  # 创建一个名为'B'的刚体对象B
    force = a*B.z  # 创建一个力force，方向为B的z轴
    B.apply_force(force, P)  # 在点P上施加力force到刚体B
    assert B.loads == [(P, force)]  # 断言B的负载为[(P, force)]
    B.clear_loads()  # 清除B的所有负载
    assert B.loads == []  # 断言B的负载为空列表

def test_remove_load():
    P1 = Point('P1')  # 创建一个名为'P1'的点对象P1
    P2 = Point('P2')  # 创建一个名为'P2'的点对象P2
    with warns_deprecated_sympy():
        B = Body('B')  # 创建一个名为'B'的刚体对象B
    f1 = B.x  # 定义f1为B的x轴单位向量
    f2 = B.y  # 定义f2为B
    # 断言语句，用于检查条件是否为真
    assert B.loads == [(P1, f1)]
def test_apply_loads_on_multi_degree_freedom_holonomic_system():
    """Example based on: https://pydy.readthedocs.io/en/latest/examples/multidof-holonomic.html"""
    # 引入警告关于 sympy 废弃特性的上下文管理器
    with warns_deprecated_sympy():
        # 创建名为 W 的刚体，代表墙壁
        W = Body('W') #Wall
        # 创建名为 B 的刚体，代表块体
        B = Body('B') #Block
        # 创建名为 P 的刚体，代表摆
        P = Body('P') #Pendulum
        # 创建名为 b 的刚体，代表摆锤
        b = Body('b') #bob

    # 定义广义坐标 q1 和 q2
    q1, q2 = dynamicsymbols('q1 q2') #generalized coordinates
    # 定义常数 k, c, g, kT
    k, c, g, kT = symbols('k c g kT') #constants
    # 定义动力学符号 F, T
    F, T = dynamicsymbols('F T') #Specified forces

    # 施加力
    B.apply_force(F*W.x)
    # 在 W 上施加弹簧力
    W.apply_force(k*q1*W.x, reaction_body=B) #Spring force
    # 在 W 上施加阻尼力
    W.apply_force(c*q1.diff()*W.x, reaction_body=B) #dampner
    # 在 P 上施加重力
    P.apply_force(P.mass*g*W.y)
    # 在 b 上施加重力
    b.apply_force(b.mass*g*W.y)

    # 施加扭矩
    # 在 P 上施加由 q2 控制的扭矩
    P.apply_torque(kT*q2*W.z, reaction_body=b)
    # 在 P 上施加由 T 控制的扭矩
    P.apply_torque(T*W.z)

    # 验证加载的力是否正确
    assert B.loads == [(B.masscenter, (F - k*q1 - c*q1.diff())*W.x)]
    assert P.loads == [(P.masscenter, P.mass*g*W.y), (P.frame, (T + kT*q2)*W.z)]
    assert b.loads == [(b.masscenter, b.mass*g*W.y), (b.frame, -kT*q2*W.z)]
    assert W.loads == [(W.masscenter, (c*q1.diff() + k*q1)*W.x)]


def test_parallel_axis():
    # 创建参考系 N
    N = ReferenceFrame('N')
    # 定义质量 m 和惯性矩 Ix, Iy, Iz, 以及长度 a, b
    m, Ix, Iy, Iz, a, b = symbols('m, I_x, I_y, I_z, a, b')
    # 计算关于 N 参考系的惯性矩 Io
    Io = inertia(N, Ix, Iy, Iz)
    
    # 测试刚体 R
    o = Point('o')
    p = o.locatenew('p', a * N.x + b * N.y)
    with warns_deprecated_sympy():
        # 创建具有质心 o、参考系 N、质量 m 和中心惯性 Io 的刚体 R
        R = Body('R', masscenter=o, frame=N, mass=m, central_inertia=Io)
    # 计算关于点 p 的平行轴定理
    Ip = R.parallel_axis(p)
    # 期望的平行轴定理结果
    Ip_expected = inertia(N, Ix + m * b**2, Iy + m * a**2,
                          Iz + m * (a**2 + b**2), ixy=-m * a * b)
    assert Ip == Ip_expected
    
    # 平行轴定理不应依赖于观察平行轴的参考系
    A = ReferenceFrame('A')
    A.orient_axis(N, N.z, 1)
    assert simplify(
        (R.parallel_axis(p, A) - Ip_expected).to_matrix(A)) == zeros(3, 3)
    
    # 测试质点 P
    o = Point('o')
    p = o.locatenew('p', a * N.x + b * N.y)
    with warns_deprecated_sympy():
        # 创建质心 o、质量 m 和参考系 N 的粒子 P
        P = Body('P', masscenter=o, mass=m, frame=N)
    # 计算质点 P 关于点 p 的平行轴定理
    Ip = P.parallel_axis(p, N)
    # 期望的质点平行轴定理结果
    Ip_expected = inertia(N, m * b ** 2, m * a ** 2, m * (a ** 2 + b ** 2),
                          ixy=-m * a * b)
    assert not P.is_rigidbody
    assert Ip == Ip_expected
```