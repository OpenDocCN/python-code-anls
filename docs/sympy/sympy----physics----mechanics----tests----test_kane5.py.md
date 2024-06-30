# `D:\src\scipysrc\sympy\sympy\physics\mechanics\tests\test_kane5.py`

```
# 导入所需的符号、矩阵、符号操作函数和物理力学相关模块
from sympy import (zeros, Matrix, symbols, lambdify, sqrt, pi,
                    simplify)
from sympy.physics.mechanics import (dynamicsymbols, cross, inertia, RigidBody,
                                     ReferenceFrame, KanesMethod)


def _create_rolling_disc():
    # 定义符号和坐标
    t = dynamicsymbols._t  # 时间符号
    q1, q2, q3, q4, q5, u1, u2, u3, u4, u5 = dynamicsymbols('q1:6 u1:6')  # 定义广义坐标和速度
    g, r, m = symbols('g r m')  # 定义重力加速度、半径和质量符号

    # 定义刚体和参考框架
    ground = RigidBody('ground')  # 地面刚体
    disc = RigidBody('disk', mass=m)  # 圆盘刚体，带有质量m
    disc.inertia = (m * r ** 2 / 4 * inertia(disc.frame, 1, 2, 1),  # 圆盘的惯性张量
                    disc.masscenter)  # 圆盘的质心

    # 设置地面和圆盘的速度
    ground.masscenter.set_vel(ground.frame, 0)  # 地面质心速度为0
    disc.masscenter.set_vel(disc.frame, 0)  # 圆盘质心速度为0

    int_frame = ReferenceFrame('int_frame')  # 定义内部参考框架

    # 定向参考框架
    int_frame.orient_body_fixed(ground.frame, (q1, q2, 0), 'zxy')  # 内部框架相对于地面框架固定
    disc.frame.orient_axis(int_frame, int_frame.y, q3)  # 圆盘框架相对于内部框架绕y轴旋转q3

    g_w_d = disc.frame.ang_vel_in(ground.frame)  # 圆盘框架在地面框架中的角速度
    disc.frame.set_ang_vel(ground.frame,
                           u1 * disc.x + u2 * disc.y + u3 * disc.z)  # 设定圆盘框架相对于地面框架的角速度

    # 定义点
    cp = ground.masscenter.locatenew('contact_point',
                                     q4 * ground.x + q5 * ground.y)  # 接触点的位置
    cp.set_vel(ground.frame, u4 * ground.x + u5 * ground.y)  # 接触点相对于地面框架的速度

    disc.masscenter.set_pos(cp, r * int_frame.z)  # 圆盘质心相对于接触点的位置
    disc.masscenter.set_vel(ground.frame, cross(
        disc.frame.ang_vel_in(ground.frame), disc.masscenter.pos_from(cp)))  # 圆盘质心相对于地面框架的速度

    # 定义运动学微分方程
    kdes = [g_w_d.dot(disc.x) - u1, g_w_d.dot(disc.y) - u2,
            g_w_d.dot(disc.z) - u3, q4.diff(t) - u4, q5.diff(t) - u5]

    # 定义非完整约束
    v0 = cp.vel(ground.frame) + cross(
        disc.frame.ang_vel_in(int_frame), cp.pos_from(disc.masscenter))
    fnh = [v0.dot(ground.x), v0.dot(ground.y)]

    # 定义载荷
    loads = [(disc.masscenter, -disc.mass * g * ground.z)]  # 圆盘质心所受重力载荷

    bodies = [disc]  # 刚体列表只包含圆盘

    return {
        'frame': ground.frame,  # 返回地面框架
        'q_ind': [q1, q2, q3, q4, q5],  # 广义坐标列表
        'u_ind': [u1, u2, u3],  # 广义速度列表
        'u_dep': [u4, u5],  # 依赖速度列表
        'kdes': kdes,  # 运动学微分方程列表
        'fnh': fnh,  # 非完整约束列表
        'bodies': bodies,  # 刚体列表
        'loads': loads  # 载荷列表
    }
    # 设置预期的解向量，用于数值测试
    expected = Matrix([
        0.126603940595934, 0.215942571601660, 1.28736069604936,
        0.319764288376543, 0.0989146857254898, -0.925848952664489,
        -0.0181350656532944, 2.91695398184589, -0.00992793421754526,
        0.0412861634829171])
    # 断言：所有解向量元素的绝对值都应小于指定的精度 eps
    assert all(abs(x) < eps for x in
               (solve_sys(q_vals, u_vals, p_vals) - expected))
    
    # 第二个数值测试
    q_vals = (3.97, -0.28, 8.2, -0.35, 2.27)
    u_vals = [-0.25, -2.2, 0.62]
    # 解决 u 的依赖，将其前两行加入 u_vals
    u_vals.extend(solve_u_dep(q_vals, u_vals, p_vals)[:2, 0])
    # 设置预期的解向量，用于数值测试
    expected = Matrix([
        0.0259159090798597, 0.668041660387416, -2.19283799213811,
        0.385441810852219, 0.420109283790573, 1.45030568179066,
        -0.0110924422400793, -8.35617840186040, -0.154098542632173,
        -0.146102664410010])
    # 断言：所有解向量元素的绝对值都应小于指定的精度 eps
    assert all(abs(x) < eps for x in
               (solve_sys(q_vals, u_vals, p_vals) - expected))
    
    # 如果所有 q_vals 和 u_vals 都为零，则断言解向量为零向量
    if all_zero:
        q_vals = (0, 0, 0, 0, 0)
        u_vals = (0, 0, 0, 0, 0)
        assert solve_sys(q_vals, u_vals, p_vals) == zeros(10, 1)
# 定义一个测试函数，用于验证 KanesMethod 类处理滚动盘的功能
def test_kane_rolling_disc_lu():
    # 调用内部函数 _create_rolling_disc()，返回滚动盘的属性字典
    props = _create_rolling_disc()
    # 创建 KanesMethod 实例，用于处理滚动盘的运动方程
    kane = KanesMethod(
        props['frame'],  # 惯性参考系
        props['q_ind'],  # 广义坐标的索引
        props['u_ind'],  # 广义速度的索引
        props['kdes'],   # 滚动盘的运动方程
        u_dependent=props['u_dep'],  # 是否依赖于广义速度
        velocity_constraints=props['fnh'],  # 速度约束
        bodies=props['bodies'],  # 物体列表
        forcelist=props['loads'],  # 外力列表
        explicit_kinematics=False,  # 是否显式动力学
        constraint_solver='LU'  # 约束求解器使用 LU 分解
    )
    # 计算滚动盘的凯恩斯方程
    kane.kanes_equations()
    # 对计算结果进行数值验证
    _verify_rolling_disc_numerically(kane)


# 定义另一个测试函数，测试 KanesMethod 处理滚动盘且 kdes 是可调用的情况
def test_kane_rolling_disc_kdes_callable():
    # 调用内部函数 _create_rolling_disc()，返回滚动盘的属性字典
    props = _create_rolling_disc()
    # 创建 KanesMethod 实例，处理滚动盘的运动方程，并且 kd_eqs_solver 使用 Lambda 表达式进行 LU 分解的简化
    kane = KanesMethod(
        props['frame'],  # 惯性参考系
        props['q_ind'],  # 广义坐标的索引
        props['u_ind'],  # 广义速度的索引
        props['kdes'],   # 滚动盘的运动方程
        u_dependent=props['u_dep'],  # 是否依赖于广义速度
        velocity_constraints=props['fnh'],  # 速度约束
        bodies=props['bodies'],  # 物体列表
        forcelist=props['loads'],  # 外力列表
        explicit_kinematics=False,  # 是否显式动力学
        kd_eqs_solver=lambda A, b: simplify(A.LUsolve(b))  # 自定义 kd_eqs_solver 使用 Lambda 表达式
    )
    # 定义动力学符号
    q, u, p = dynamicsymbols('q1:6'), dynamicsymbols('u1:6'), symbols('g r m')
    qd = dynamicsymbols('q1:6', 1)
    # 使用 lambdify 将凯恩斯方程字典项变为可调用函数
    eval_kdes = lambdify((q, qd, u, p), tuple(kane.kindiffdict().items()))
    eps = 1e-10
    # 使用全零测试参数进行测试，确保 LU 分解不会产生 nan
    p_vals = (9.81, 0.25, 3.5)
    zero_vals = (0, 0, 0, 0, 0)
    # 断言所有计算值都在允许的误差范围内
    assert all(abs(qdi - fui) < eps for qdi, fui in eval_kdes(zero_vals, zero_vals, zero_vals, p_vals))
    # 使用一些任意值进行测试
    q_vals = tuple(map(float, (pi / 6, pi / 3, pi / 2, 0.42, 0.62)))
    qd_vals = tuple(map(float, (4, 1 / 3, 4 - 2 * sqrt(3), 0.25 * (2 * sqrt(3) - 3), 0.25 * (2 - sqrt(3)))))
    u_vals = tuple(map(float, (-2, 4, 1 / 3, 0.25 * (-3 + 2 * sqrt(3)), 0.25 * (-sqrt(3) + 2))))
    # 断言所有计算值都在允许的误差范围内
    assert all(abs(qdi - fui) < eps for qdi, fui in eval_kdes(q_vals, qd_vals, u_vals, p_vals))
```