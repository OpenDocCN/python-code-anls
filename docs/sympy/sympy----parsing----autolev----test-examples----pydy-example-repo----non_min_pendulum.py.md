# `D:\src\scipysrc\sympy\sympy\parsing\autolev\test-examples\pydy-example-repo\non_min_pendulum.py`

```
import sympy.physics.mechanics as _me  # 导入 sympy 的力学模块
import sympy as _sm  # 导入 sympy 库
import math as m  # 导入 math 库，用于数学运算
import numpy as _np  # 导入 numpy 库，用于数组操作

q1, q2 = _me.dynamicsymbols('q1 q2')  # 定义动力学符号 q1 和 q2
q1_d, q2_d = _me.dynamicsymbols('q1_ q2_', 1)  # 定义 q1 和 q2 的一阶导数
q1_dd, q2_dd = _me.dynamicsymbols('q1_ q2_', 2)  # 定义 q1 和 q2 的二阶导数
l, m, g = _sm.symbols('l m g', real=True)  # 定义符号 l, m, g，并指定为实数

frame_n = _me.ReferenceFrame('n')  # 创建参考系 frame_n
point_pn = _me.Point('pn')  # 创建点 point_pn
point_pn.set_vel(frame_n, 0)  # 设置 point_pn 在 frame_n 中的速度为零

theta1 = _sm.atan(q2/q1)  # 计算角度 theta1，为 q2/q1 的反正切值

frame_a = _me.ReferenceFrame('a')  # 创建参考系 frame_a
frame_a.orient(frame_n, 'Axis', [theta1, frame_n.z])  # 使用 theta1 绕 frame_n.z 轴旋转来定向 frame_a

particle_p = _me.Particle('p', _me.Point('p_pt'), _sm.Symbol('m'))  # 创建粒子 particle_p，具有质点和质量 m
particle_p.point.set_pos(point_pn, q1*frame_n.x + q2*frame_n.y)  # 设置粒子的位置
particle_p.mass = m  # 设置粒子的质量为 m
particle_p.point.set_vel(frame_n, (point_pn.pos_from(particle_p.point)).dt(frame_n))  # 计算粒子在 frame_n 中的速度

f_v = _me.dot((particle_p.point.vel(frame_n)).express(frame_a), frame_a.x)  # 计算速度在 frame_a.x 方向上的投影

force_p = particle_p.mass * (g * frame_n.x)  # 计算粒子所受重力的力

dependent = _sm.Matrix([[0]])  # 创建一个 1x1 的零矩阵
dependent[0] = f_v  # 将 f_v 赋值给 dependent 的第一个元素，用于速度约束

velocity_constraints = [i for i in dependent]  # 创建速度约束列表，包含 dependent 中的元素

u_q1_d = _me.dynamicsymbols('u_q1_d')  # 定义符号 u_q1_d
u_q2_d = _me.dynamicsymbols('u_q2_d')  # 定义符号 u_q2_d

kd_eqs = [q1_d - u_q1_d, q2_d - u_q2_d]  # 创建运动方程列表，表示 q1_d 和 q2_d 与 u_q1_d 和 u_q2_d 之间的关系

forceList = [(particle_p.point, particle_p.mass * (g * frame_n.x))]  # 创建力列表，包含粒子所受的重力

# 创建 KanesMethod 对象，用于分析多体动力学系统
kane = _me.KanesMethod(frame_n, q_ind=[q1, q2], u_ind=[u_q2_d], u_dependent=[u_q1_d],
                       kd_eqs=kd_eqs, velocity_constraints=velocity_constraints)

# 计算广义力和广义动力学方程
fr, frstar = kane.kanes_equations([particle_p], forceList)

zero = fr + frstar  # 构建动力学方程的左侧（广义力的总和）

f_c = point_pn.pos_from(particle_p.point).magnitude() - l  # 计算额外的约束条件

config = _sm.Matrix([[0]])  # 创建一个 1x1 的零矩阵
config[0] = f_c  # 将 f_c 赋值给 config 的第一个元素，用于约束

zero = zero.row_insert(zero.shape[0], _sm.Matrix([[0]]))  # 在 zero 的最后插入一个零行
zero[zero.shape[0] - 1] = config[0]  # 将 config 的值赋给 zero 的最后一个元素
```