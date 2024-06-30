# `D:\src\scipysrc\sympy\sympy\physics\mechanics\tests\test_kane3.py`

```
from sympy.core.numbers import pi
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import acos, sin, cos
from sympy.matrices.dense import Matrix
from sympy.physics.mechanics import (ReferenceFrame, dynamicsymbols,
                                     KanesMethod, inertia, Point, RigidBody,
                                     dot)
from sympy.testing.pytest import slow

@slow
def test_bicycle():
    # 获取作为参考的研究文献中自行车模型的运动方程的代码

    # 注意，这段代码是从Autolev粗略移植而来的，这解释了一些不寻常的命名约定。目的是尽可能与原始代码保持相似，以便于调试。

    # 声明坐标和速度
    # 简单定义了一些速度符号 - qd = u
    # 速度包括：
    # - u1: 偏航角速率
    # - u2: 翻滚角速率
    # - u3: 后轮角速率（自旋运动）
    # - u4: 框架角速率（俯仰运动）
    # - u5: 转向角速率
    # - u6: 前轮角速率（自旋运动）
    q1, q2, q4, q5 = dynamicsymbols('q1 q2 q4 q5')
    q1d, q2d, q4d, q5d = dynamicsymbols('q1 q2 q4 q5', 1)
    u1, u2, u3, u4, u5, u6 = dynamicsymbols('u1 u2 u3 u4 u5 u6')
    u1d, u2d, u3d, u4d, u5d, u6d = dynamicsymbols('u1 u2 u3 u4 u5 u6', 1)

    # 声明系统参数
    WFrad, WRrad, htangle, forkoffset = symbols('WFrad WRrad htangle forkoffset')
    forklength, framelength, forkcg1 = symbols('forklength framelength forkcg1')
    forkcg3, framecg1, framecg3, Iwr11 = symbols('forkcg3 framecg1 framecg3 Iwr11')
    Iwr22, Iwf11, Iwf22, Iframe11 = symbols('Iwr22 Iwf11 Iwf22 Iframe11')
    Iframe22, Iframe33, Iframe31, Ifork11 = symbols('Iframe22 Iframe33 Iframe31 Ifork11')
    Ifork22, Ifork33, Ifork31, g = symbols('Ifork22 Ifork33 Ifork31 g')
    mframe, mfork, mwf, mwr = symbols('mframe mfork mwf mwr')

    # 设置系统的参考坐标系
    # N - 惯性坐标系
    # Y - 偏航坐标系
    # R - 翻滚坐标系
    # WR - 后轮，其旋转角度是忽略的坐标，因此没有定向
    # Frame - 自行车框架
    # TempFrame - 静态旋转的框架，用于更容易地参考惯性定义惯量
    # Fork - 自行车前叉
    # TempFork - 静态旋转的前叉，用于更容易地参考惯性定义惯量
    # WF - 前轮，同样有一个忽略的坐标
    N = ReferenceFrame('N')
    Y = N.orientnew('Y', 'Axis', [q1, N.z])
    R = Y.orientnew('R', 'Axis', [q2, Y.x])
    Frame = R.orientnew('Frame', 'Axis', [q4 + htangle, R.y])
    WR = ReferenceFrame('WR')
    # 创建一个名为 TempFrame 的参考框架，其方向与 Axis 方向相反，位置参数为 [-htangle, Frame.y]
    TempFrame = Frame.orientnew('TempFrame', 'Axis', [-htangle, Frame.y])
    
    # 创建一个名为 Fork 的参考框架，其方向与 Axis 方向相反，位置参数为 [q5, Frame.x]
    Fork = Frame.orientnew('Fork', 'Axis', [q5, Frame.x])
    
    # 在 Fork 参考框架下创建一个名为 TempFork 的参考框架，其方向与 Axis 方向相反，位置参数为 [-htangle, Fork.y]
    TempFork = Fork.orientnew('TempFork', 'Axis', [-htangle, Fork.y])
    
    # 创建一个名为 WF 的参考框架
    WF = ReferenceFrame('WF')

    # Bicycle Kinematics First block of code is forming the positions of
    # the relevant points
    # rear wheel contact -> rear wheel mass center -> frame mass center +
    # frame/fork connection -> fork mass center + front wheel mass center ->
    # front wheel contact point
    
    # 创建点 WR_cont，表示后轮接触点
    WR_cont = Point('WR_cont')
    
    # 在 WR_cont 点上创建后轮质心 WR_mc，其位置向上 WRrad * R.z
    WR_mc = WR_cont.locatenew('WR_mc', WRrad * R.z)
    
    # 在 WR_mc 点上创建 Steer 点，位置向前 framelength * Frame.z
    Steer = WR_mc.locatenew('Steer', framelength * Frame.z)
    
    # 在 WR_mc 点上创建 Frame_mc 点，位置为 - framecg1 * Frame.x + framecg3 * Frame.z
    Frame_mc = WR_mc.locatenew('Frame_mc', - framecg1 * Frame.x + framecg3 * Frame.z)
    
    # 在 Steer 点上创建 Fork_mc 点，位置为 - forkcg1 * Fork.x + forkcg3 * Fork.z
    Fork_mc = Steer.locatenew('Fork_mc', - forkcg1 * Fork.x + forkcg3 * Fork.z)
    
    # 在 Steer 点上创建 WF_mc 点，位置为 forklength * Fork.x + forkoffset * Fork.z
    WF_mc = Steer.locatenew('WF_mc', forklength * Fork.x + forkoffset * Fork.z)
    
    # 在 WF_mc 点上创建 WF_cont 点，位置为 WFrad * (dot(Fork.y, Y.z) * Fork.y - Y.z).normalize()
    WF_cont = WF_mc.locatenew('WF_cont', WFrad * (dot(Fork.y, Y.z) * Fork.y - Y.z).normalize())

    # Set the angular velocity of each frame.
    # Angular accelerations end up being calculated automatically by
    # differentiating the angular velocities when first needed.
    # u1 is yaw rate
    # u2 is roll rate
    # u3 is rear wheel rate
    # u4 is frame pitch rate
    # u5 is fork steer rate
    # u6 is front wheel rate
    
    # 设置 Y 参考框架相对于 N 参考框架的角速度，角速度大小为 u1 * Y.z
    Y.set_ang_vel(N, u1 * Y.z)
    
    # 设置 R 参考框架相对于 Y 参考框架的角速度，角速度大小为 u2 * R.x
    R.set_ang_vel(Y, u2 * R.x)
    
    # 设置 WR 参考框架相对于 Frame 参考框架的角速度，角速度大小为 u3 * Frame.y
    WR.set_ang_vel(Frame, u3 * Frame.y)
    
    # 设置 Frame 参考框架相对于 R 参考框架的角速度，角速度大小为 u4 * Frame.y
    Frame.set_ang_vel(R, u4 * Frame.y)
    
    # 设置 Fork 参考框架相对于 Frame 参考框架的角速度，角速度大小为 u5 * Fork.x
    Fork.set_ang_vel(Frame, u5 * Fork.x)
    
    # 设置 WF 参考框架相对于 Fork 参考框架的角速度，角速度大小为 u6 * Fork.y
    WF.set_ang_vel(Fork, u6 * Fork.y)

    # Form the velocities of the previously defined points, using the 2 - point
    # theorem (written out by hand here).  Accelerations again are calculated
    # automatically when first needed.
    
    # 设置 WR_cont 点在 N 参考框架中的速度为 0
    WR_cont.set_vel(N, 0)
    
    # 使用两点定理计算 WR_mc 点在 N 参考框架中的速度
    WR_mc.v2pt_theory(WR_cont, N, WR)
    
    # 使用两点定理计算 Steer 点在 N 参考框架中的速度
    Steer.v2pt_theory(WR_mc, N, Frame)
    
    # 使用两点定理计算 Frame_mc 点在 N 参考框架中的速度
    Frame_mc.v2pt_theory(WR_mc, N, Frame)
    
    # 使用两点定理计算 Fork_mc 点在 N 参考框架中的速度
    Fork_mc.v2pt_theory(Steer, N, Fork)
    
    # 使用两点定理计算 WF_mc 点在 N 参考框架中的速度
    WF_mc.v2pt_theory(Steer, N, Fork)
    
    # 使用两点定理计算 WF_cont 点在 N 参考框架中的速度
    WF_cont.v2pt_theory(WF_mc, N, WF)

    # Sets the inertias of each body. Uses the inertia frame to construct the
    # inertia dyadics. Wheel inertias are only defined by principle moments of
    # inertia, and are in fact constant in the frame and fork reference frames;
    # it is for this reason that the orientations of the wheels does not need
    # to be defined. The frame and fork inertias are defined in the 'Temp'
    # frames which are fixed to the appropriate body frames; this is to allow
    # easier input of the reference values of the benchmark paper. Note that
    # due to slightly different orientations, the products of inertia need to
    # have their signs flipped; this is done later when entering the numerical
    # value.
    
    # 设置 Frame_I 为 Frame 框架的惯性，使用 TempFrame 框架构建惯性对角线
    # 惯性参数 Iframe11, Iframe22, Iframe33, Iframe31 在此处为符号常数
    Frame_I = (inertia(TempFrame, Iframe11, Iframe22, Iframe33, 0, 0, Iframe31), Frame_mc)
    Fork_I = (inertia(TempFork, Ifork11, Ifork22, Ifork33, 0, 0, Ifork31), Fork_mc)
    # Calculate inertia properties for the Fork subsystem using specified parameters.

    WR_I = (inertia(Frame, Iwr11, Iwr22, Iwr11), WR_mc)
    # Calculate inertia properties for the Wheel Rear subsystem using specified parameters.

    WF_I = (inertia(Fork, Iwf11, Iwf22, Iwf11), WF_mc)
    # Calculate inertia properties for the Wheel Front subsystem using specified parameters.

    # Declaration of the RigidBody containers. ::
    BodyFrame = RigidBody('BodyFrame', Frame_mc, Frame, mframe, Frame_I)
    # Define a rigid body named 'BodyFrame' with mass center Frame_mc, reference frame Frame,
    # mass mframe, and inertia Frame_I.

    BodyFork = RigidBody('BodyFork', Fork_mc, Fork, mfork, Fork_I)
    # Define a rigid body named 'BodyFork' with mass center Fork_mc, reference frame Fork,
    # mass mfork, and inertia Fork_I.

    BodyWR = RigidBody('BodyWR', WR_mc, WR, mwr, WR_I)
    # Define a rigid body named 'BodyWR' with mass center WR_mc, reference frame WR,
    # mass mwr, and inertia WR_I.

    BodyWF = RigidBody('BodyWF', WF_mc, WF, mwf, WF_I)
    # Define a rigid body named 'BodyWF' with mass center WF_mc, reference frame WF,
    # mass mwf, and inertia WF_I.

    # The kinematic differential equations; they are defined quite simply. Each
    # entry in this list is equal to zero.
    kd = [q1d - u1, q2d - u2, q4d - u4, q5d - u5]
    # Define kinematic differential equations, relating generalized speeds (q1d, q2d, q4d, q5d)
    # to independent speeds (u1, u2, u4, u5).

    # The nonholonomic constraints are the velocity of the front wheel contact
    # point dotted into the X, Y, and Z directions; the yaw frame is used as it
    # is "closer" to the front wheel (1 less DCM connecting them). These
    # constraints force the velocity of the front wheel contact point to be 0
    # in the inertial frame; the X and Y direction constraints enforce a
    # "no-slip" condition, and the Z direction constraint forces the front
    # wheel contact point to not move away from the ground frame, essentially
    # replicating the holonomic constraint which does not allow the frame pitch
    # to change in an invalid fashion.
    conlist_speed = [WF_cont.vel(N) & Y.x, WF_cont.vel(N) & Y.y, WF_cont.vel(N) & Y.z]
    # Define nonholonomic velocity constraints for the front wheel contact point in X, Y, and Z directions.

    # The holonomic constraint is that the position from the rear wheel contact
    # point to the front wheel contact point when dotted into the
    # normal-to-ground plane direction must be zero; effectively that the front
    # and rear wheel contact points are always touching the ground plane. This
    # is actually not part of the dynamic equations, but instead is necessary
    # for the lineraization process.
    conlist_coord = [WF_cont.pos_from(WR_cont) & Y.z]
    # Define holonomic position constraint ensuring contact between front and rear wheel contact points.

    # The force list; each body has the appropriate gravitational force applied
    # at its mass center.
    FL = [(Frame_mc, -mframe * g * Y.z),
        (Fork_mc, -mfork * g * Y.z),
        (WF_mc, -mwf * g * Y.z),
        (WR_mc, -mwr * g * Y.z)]
    # Define gravitational forces applied to each rigid body's mass center.

    BL = [BodyFrame, BodyFork, BodyWR, BodyWF]
    # Define a list of all rigid bodies in the system.

    # The N frame is the inertial frame, coordinates are supplied in the order
    # of independent, dependent coordinates, as are the speeds. The kinematic
    # differential equation are also entered here.  Here the dependent speeds
    # are specified, in the same order they were provided in earlier, along
    # with the non-holonomic constraints.  The dependent coordinate is also
    # provided, with the holonomic constraint.  Again, this is only provided
    # for the linearization process.
    KM = KanesMethod(N, q_ind=[q1, q2, q5],
            q_dependent=[q4], configuration_constraints=conlist_coord,
            u_ind=[u2, u3, u5],
            u_dependent=[u1, u4, u6], velocity_constraints=conlist_speed,
            kd_eqs=kd,
            constraint_solver="CRAMER")
    # Initialize KanesMethod object for dynamic analysis with specified parameters.
    (fr, frstar) = KM.kanes_equations(BL, FL)
    # 使用Kane方程求解得到惯性力和惯性力矩
    
    # 这里开始输入来自基准论文的数值，用于验证线性化方程的特征值是否与参考特征值一致。
    # 参考文献详细说明了这些值的含义。其中一些是中间值，用于将论文中的值转换为本模型使用的坐标系统。
    PaperRadRear                    =  0.3
    PaperRadFront                   =  0.35
    HTA                             =  (pi / 2 - pi / 10).evalf()
    TrailPaper                      =  0.08
    rake                            =  (-(TrailPaper*sin(HTA)-(PaperRadFront*cos(HTA)))).evalf()
    PaperWb                         =  1.02
    PaperFrameCgX                   =  0.3
    PaperFrameCgZ                   =  0.9
    PaperForkCgX                    =  0.9
    PaperForkCgZ                    =  0.7
    FrameLength                     =  (PaperWb*sin(HTA)-(rake-(PaperRadFront-PaperRadRear)*cos(HTA))).evalf()
    FrameCGNorm                     =  ((PaperFrameCgZ - PaperRadRear-(PaperFrameCgX/sin(HTA))*cos(HTA))*sin(HTA)).evalf()
    FrameCGPar                      =  (PaperFrameCgX / sin(HTA) + (PaperFrameCgZ - PaperRadRear - PaperFrameCgX / sin(HTA) * cos(HTA)) * cos(HTA)).evalf()
    tempa                           =  (PaperForkCgZ - PaperRadFront)
    tempb                           =  (PaperWb-PaperForkCgX)
    tempc                           =  (sqrt(tempa**2+tempb**2)).evalf()
    PaperForkL                      =  (PaperWb*cos(HTA)-(PaperRadFront-PaperRadRear)*sin(HTA)).evalf()
    ForkCGNorm                      =  (rake+(tempc * sin(pi/2-HTA-acos(tempa/tempc)))).evalf()
    ForkCGPar                       =  (tempc * cos((pi/2-HTA)-acos(tempa/tempc))-PaperForkL).evalf()
    
    # 这里是最终组装数值的地方。符号'v'代表自行车的前进速度（这个概念只在直立、静态平衡的情况下有意义？）。
    # 这些值被放入一个字典中，稍后将被替换。再次提醒，这里对*惯性矩*的符号进行了反转，因为坐标系的不同方向。
    v = symbols('v')
    # 定义一个字典，包含系统参数的值和变量名的映射关系
    val_dict = {
        WFrad: PaperRadFront,    # 前轮纵摇角
        WRrad: PaperRadRear,     # 后轮纵摇角
        htangle: HTA,            # 头管角
        forkoffset: rake,        # 叉偏移量
        forklength: PaperForkL,  # 前叉长度
        framelength: FrameLength,  # 车架长度
        forkcg1: ForkCGPar,      # 前叉质心位置参数1
        forkcg3: ForkCGNorm,     # 前叉质心位置参数3
        framecg1: FrameCGNorm,   # 车架质心位置参数1
        framecg3: FrameCGPar,    # 车架质心位置参数3
        Iwr11: 0.0603,           # 后轮转动惯量矩阵元素
        Iwr22: 0.12,             # 后轮转动惯量矩阵元素
        Iwf11: 0.1405,           # 前轮转动惯量矩阵元素
        Iwf22: 0.28,             # 前轮转动惯量矩阵元素
        Ifork11: 0.05892,        # 前叉转动惯量矩阵元素
        Ifork22: 0.06,           # 前叉转动惯量矩阵元素
        Ifork33: 0.00708,        # 前叉转动惯量矩阵元素
        Ifork31: 0.00756,        # 前叉转动惯量矩阵元素
        Iframe11: 9.2,           # 车架转动惯量矩阵元素
        Iframe22: 11,            # 车架转动惯量矩阵元素
        Iframe33: 2.8,           # 车架转动惯量矩阵元素
        Iframe31: -2.4,          # 车架转动惯量矩阵元素
        mfork: 4,                # 前叉质量
        mframe: 85,              # 车架质量
        mwf: 3,                  # 前轮质量
        mwr: 2,                  # 后轮质量
        g: 9.81,                 # 重力加速度
        q1: 0,                   # 广义坐标q1
        q2: 0,                   # 广义坐标q2
        q4: 0,                   # 广义坐标q4
        q5: 0,                   # 广义坐标q5
        u1: 0,                   # 广义速度u1
        u2: 0,                   # 广义速度u2
        u3: v / PaperRadRear,    # 广义速度u3，前轮半径前侧
        u4: 0,                   # 广义速度u4
        u5: 0,                   # 广义速度u5
        u6: v / PaperRadFront    # 广义速度u6，前轮半径后侧
    }
    
    # 调用 KM.linearize 方法进行系统线性化处理，得到系数矩阵 A 和 B
    A, B, _ = KM.linearize(
        A_and_B=True,
        op_point={
            # 设置操作点，用于加速度的线性化，消除系数矩阵中的 u' 项
            u1.diff(): 0,
            u2.diff(): 0,
            u3.diff(): 0,
            u4.diff(): 0,
            u5.diff(): 0,
            u6.diff(): 0,
            u1: 0,
            u2: 0,
            u3: v / PaperRadRear,
            u4: 0,
            u5: 0,
            u6: v / PaperRadFront,
            q1: 0,
            q2: 0,
            q4: 0,
            q5: 0,
        },
        linear_solver="CRAMER",
    )
    
    # 将系数矩阵 A 和 B 中的符号变量替换为实际值，得到数值化的 A_s 和 B_s
    A_s = A.xreplace(val_dict)
    B_s = B.xreplace(val_dict)
    
    # 对数值化后的系数矩阵 A_s 和 B_s 进行数值求解
    A_s = A_s.evalf()
    B_s = B_s.evalf()
    
    # 构建状态方程的矩阵 A，虽然在本例中大小有些不一致，该行代码仅提取所需的最小条目以进行特征值计算。
    # 从 A_s 中提取特定行和列，这些行和列对应于 lean、steer、lean rate 和 steer rate。
    A = A_s.extract([1, 2, 3, 5], [1, 2, 3, 5])

    # 预先计算用于比较的矩阵 Res
    Res = Matrix([[               0,                                           0,                  1.0,                    0],
                  [               0,                                           0,                    0,                  1.0],
                  [9.48977444677355, -0.891197738059089*v**2 - 0.571523173729245, -0.105522449805691*v, -0.330515398992311*v],
                  [11.7194768719633,   -1.97171508499972*v**2 + 30.9087533932407,   3.67680523332152*v,  -3.08486552743311*v]])

    # 实际特征值比较的容差阈值
    eps = 1.e-12
    # 对于每一个 v 的取值，进行 Res 和 A 的差值计算，并进行断言检查
    for i in range(6):
        error = Res.subs(v, i) - A.subs(v, i)
        # 断言所有误差项的绝对值都小于容差阈值 eps
        assert all(abs(x) < eps for x in error)
```