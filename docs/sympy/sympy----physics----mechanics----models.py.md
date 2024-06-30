# `D:\src\scipysrc\sympy\sympy\physics\mechanics\models.py`

```
#!/usr/bin/env python
"""This module contains some sample symbolic models used for testing and
examples."""

# Internal imports
from sympy.core import backend as sm  # 导入 sympy 的后端模块作为 sm
import sympy.physics.mechanics as me  # 导入 sympy 的力学模块作为 me


def multi_mass_spring_damper(n=1, apply_gravity=False,
                             apply_external_forces=False):
    r"""Returns a system containing the symbolic equations of motion and
    associated variables for a simple multi-degree of freedom point mass,
    spring, damper system with optional gravitational and external
    specified forces. For example, a two mass system under the influence of
    gravity and external forces looks like:

    ::

        ----------------
         |     |     |   | g
         \    | |    |   V
      k0 /    --- c0 |
         |     |     | x0, v0
        ---------    V
        |  m0   | -----
        ---------    |
         | |   |     |
         \ v  | |    |
      k1 / f0 --- c1 |
         |     |     | x1, v1
        ---------    V
        |  m1   | -----
        ---------
           | f1
           V

    Parameters
    ==========

    n : integer
        The number of masses in the serial chain.
    apply_gravity : boolean
        If true, gravity will be applied to each mass.
    apply_external_forces : boolean
        If true, a time varying external force will be applied to each mass.

    Returns
    =======

    kane : sympy.physics.mechanics.kane.KanesMethod
        A KanesMethod object.

    """

    mass = sm.symbols('m:{}'.format(n))  # 定义质量符号列表，格式为 m0, m1, ..., mn-1
    stiffness = sm.symbols('k:{}'.format(n))  # 定义刚度符号列表，格式为 k0, k1, ..., kn-1
    damping = sm.symbols('c:{}'.format(n))  # 定义阻尼符号列表，格式为 c0, c1, ..., cn-1

    acceleration_due_to_gravity = sm.symbols('g')  # 定义重力加速度符号 g

    coordinates = me.dynamicsymbols('x:{}'.format(n))  # 定义坐标符号列表，格式为 x0, x1, ..., xn-1
    speeds = me.dynamicsymbols('v:{}'.format(n))  # 定义速度符号列表，格式为 v0, v1, ..., vn-1
    specifieds = me.dynamicsymbols('f:{}'.format(n))  # 定义外力符号列表，格式为 f0, f1, ..., fn-1

    ceiling = me.ReferenceFrame('N')  # 创建惯性参考系对象 N
    origin = me.Point('origin')  # 创建参考点对象 origin
    origin.set_vel(ceiling, 0)  # 设置参考点 origin 相对于参考系 N 的速度为零

    points = [origin]  # 将 origin 添加到点列表中
    kinematic_equations = []  # 初始化运动学方程列表
    particles = []  # 初始化粒子列表
    forces = []  # 初始化力列表
    # 循环 n 次，生成多个质点和它们的运动相关参数
    for i in range(n):
        # 创建质点的中心，命名为'center{i}'，位置位于coordinates[i]乘以ceiling.x处
        center = points[-1].locatenew('center{}'.format(i), coordinates[i] * ceiling.x)
        # 设置质点中心的速度，相对于ceiling的速度为points[-1].vel(ceiling)加上speeds[i]乘以ceiling.x
        center.set_vel(ceiling, points[-1].vel(ceiling) + speeds[i] * ceiling.x)
        # 将新创建的中心质点加入points列表
        points.append(center)
    
        # 创建一个名为'block{i}'的质点对象，位置为中心质点center，质量为mass[i]
        block = me.Particle('block{}'.format(i), center, mass[i])
    
        # 添加运动学方程到kinematic_equations列表，表示速度speeds[i]减去坐标coordinates[i]的导数
        kinematic_equations.append(speeds[i] - coordinates[i].diff())
    
        # 计算总力total_force，包括弹簧力和阻尼力
        total_force = (-stiffness[i] * coordinates[i] - damping[i] * speeds[i])
        try:
            # 尝试添加来自下方的弹簧力和阻尼力（如果有）
            total_force += (stiffness[i + 1] * coordinates[i + 1] + damping[i + 1] * speeds[i + 1])
        except IndexError:  # 如果是最后一个质点，则没有来自下方的力
            pass
    
        # 如果应用重力，添加重力对质点的力
        if apply_gravity:
            total_force += mass[i] * acceleration_due_to_gravity
    
        # 如果应用外部力，添加指定的外部力
        if apply_external_forces:
            total_force += specifieds[i]
    
        # 将中心质点及其受力加入forces列表，乘以ceiling.x以保持单位一致性
        forces.append((center, total_force * ceiling.x))
    
        # 将质点对象block加入particles列表
        particles.append(block)
    
    # 使用Kane方法创建kane对象，指定广义坐标为coordinates，广义速度为speeds，动力学方程为kinematic_equations
    kane = me.KanesMethod(ceiling, q_ind=coordinates, u_ind=speeds, kd_eqs=kinematic_equations)
    # 计算系统的卡恩方程
    kane.kanes_equations(particles, forces)
    
    # 返回kane对象，表示整个系统的动力学方程
    return kane
# 定义一个函数，用于生成描述n节摆链和带滑动小车的系统的动力学方程的符号表达式
def n_link_pendulum_on_cart(n=1, cart_force=True, joint_torques=False):
    r"""Returns the system containing the symbolic first order equations of
    motion for a 2D n-link pendulum on a sliding cart under the influence of
    gravity.

    ::

                  |
         o    y   v
          \ 0 ^   g
           \  |
          --\-|----
          |  \|   |
      F-> |   o --|---> x
          |       |
          ---------
           o     o

    Parameters
    ==========

    n : integer
        The number of links in the pendulum.
    cart_force : boolean, default=True
        If true an external specified lateral force is applied to the cart.
    joint_torques : boolean, default=False
        If true joint torques will be added as specified inputs at each
        joint.

    Returns
    =======

    kane : sympy.physics.mechanics.kane.KanesMethod
        A KanesMethod object.

    Notes
    =====

    The degrees of freedom of the system are n + 1, i.e. one for each
    pendulum link and one for the lateral motion of the cart.

    M x' = F, where x = [u0, ..., un+1, q0, ..., qn+1]

    The joint angles are all defined relative to the ground where the x axis
    defines the ground line and the y axis points up. The joint torques are
    applied between each adjacent link and the between the cart and the
    lower link where a positive torque corresponds to positive angle.

    """
    # 检查链节数量是否为正整数，若不是则抛出异常
    if n <= 0:
        raise ValueError('The number of links must be a positive integer.')

    # 定义n个摆链的广义坐标和速度符号
    q = me.dynamicsymbols('q:{}'.format(n + 1))
    u = me.dynamicsymbols('u:{}'.format(n + 1))

    # 如果需要考虑关节扭矩，则定义n个关节扭矩的符号
    if joint_torques is True:
        T = me.dynamicsymbols('T1:{}'.format(n + 1))

    # 定义n个摆链的质量和n-1个链杆长度的符号
    m = sm.symbols('m:{}'.format(n + 1))
    l = sm.symbols('l:{}'.format(n))
    g, t = sm.symbols('g t')

    # 定义惯性参考框架和一个静止点O
    I = me.ReferenceFrame('I')
    O = me.Point('O')
    O.set_vel(I, 0)

    # 定义第一个链节的点P0，设置其位置和速度
    P0 = me.Point('P0')
    P0.set_pos(O, q[0] * I.x)
    P0.set_vel(I, u[0] * I.x)

    # 定义第一个链节作为质点Pa0
    Pa0 = me.Particle('Pa0', P0, m[0])

    # 初始化存储框架、点、质点、力和运动方程的列表
    frames = [I]
    points = [P0]
    particles = [Pa0]
    forces = [(P0, -m[0] * g * I.y)]
    kindiffs = [q[0].diff(t) - u[0]]

    # 如果需要考虑外部侧向力或关节扭矩，则定义一个空的指定输入列表，否则指定输入为空
    if cart_force is True or joint_torques is True:
        specified = []
    else:
        specified = None
    for i in range(n):
        # 创建第i个刚体Bi，其定位轴为(q[i + 1], I.z)，设置角速度为u[i + 1] * I.z
        Bi = I.orientnew('B{}'.format(i), 'Axis', [q[i + 1], I.z])
        Bi.set_ang_vel(I, u[i + 1] * I.z)
        frames.append(Bi)

        # 在最后一个点上创建第i+1个点Pi，位置偏移为l[i] * Bi.y
        Pi = points[-1].locatenew('P{}'.format(i + 1), l[i] * Bi.y)
        # 使用虚位移理论计算Pi的速度
        Pi.v2pt_theory(points[-1], I, Bi)
        points.append(Pi)

        # 创建第i+1个质点Pai，位于Pi处，质量为m[i + 1]
        Pai = me.Particle('Pa' + str(i + 1), Pi, m[i + 1])
        particles.append(Pai)

        # 添加作用在Pi上的重力力矢量，大小为-m[i + 1] * g，方向为-I.y
        forces.append((Pi, -m[i + 1] * g * I.y))

        # 如果需要关节扭矩
        if joint_torques is True:
            # 将关节扭矩T[i]添加到指定力列表中
            specified.append(T[i])

            # 如果i为0，作用在惯性坐标系I上的力矢量，大小为-T[i]，方向为-I.z
            if i == 0:
                forces.append((I, -T[i] * I.z))

            # 如果i为n-1，作用在Bi上的力矢量，大小为T[i]，方向为I.z
            elif i == n - 1:
                forces.append((Bi, T[i] * I.z))

            # 否则作用在Bi上的力矢量，大小为T[i] - T[i + 1]，方向为I.z
            else:
                forces.append((Bi, T[i] * I.z - T[i + 1] * I.z))

        # 将广义速度q[i + 1]的时间导数与u[i + 1]的差值添加到运动微分方程列表kindiffs中
        kindiffs.append(q[i + 1].diff(t) - u[i + 1])

    # 如果有小车作用力
    if cart_force is True:
        # 定义一个新的动力符号F，表示小车作用力
        F = me.dynamicsymbols('F')
        # 将作用在P0上的力矢量，大小为F，方向为I.x，添加到作用力列表中
        forces.append((P0, F * I.x))
        # 将F添加到指定力列表中
        specified.append(F)

    # 创建Kane对象，使用KanesMethod进行动力学分析，传入惯性参考系I、广义坐标q、广义速度u以及运动微分方程列表kindiffs
    kane = me.KanesMethod(I, q_ind=q, u_ind=u, kd_eqs=kindiffs)
    # 计算并返回Kane方程
    kane.kanes_equations(particles, forces)

    # 返回Kane对象，用于后续动力学模拟或分析
    return kane
```