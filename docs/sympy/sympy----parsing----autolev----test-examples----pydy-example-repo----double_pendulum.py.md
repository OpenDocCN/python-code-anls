# `D:\src\scipysrc\sympy\sympy\parsing\autolev\test-examples\pydy-example-repo\double_pendulum.py`

```
import sympy.physics.mechanics as _me  # 导入符号力学中的机械模块
import sympy as _sm  # 导入符号计算模块
import math as m  # 导入数学模块，并使用别名 m
import numpy as _np  # 导入 numpy 数值计算模块，并使用别名 _np

q1, q2, u1, u2 = _me.dynamicsymbols('q1 q2 u1 u2')  # 定义动力学符号变量 q1, q2, u1, u2
q1_d, q2_d, u1_d, u2_d = _me.dynamicsymbols('q1_ q2_ u1_ u2_', 1)  # 定义动力学符号变量的一阶导数 q1_d, q2_d, u1_d, u2_d
l, m, g = _sm.symbols('l m g', real=True)  # 定义符号变量 l, m, g，均为实数
frame_n = _me.ReferenceFrame('n')  # 创建参考框架 'n'
frame_a = _me.ReferenceFrame('a')  # 创建参考框架 'a'
frame_b = _me.ReferenceFrame('b')  # 创建参考框架 'b'
frame_a.orient(frame_n, 'Axis', [q1, frame_n.z])  # 设置参考框架 'a' 相对于 'n' 的方向，绕 z 轴旋转 q1 角度
frame_b.orient(frame_n, 'Axis', [q2, frame_n.z])  # 设置参考框架 'b' 相对于 'n' 的方向，绕 z 轴旋转 q2 角度
frame_a.set_ang_vel(frame_n, u1*frame_n.z)  # 设置参考框架 'a' 相对于 'n' 的角速度为 u1 关于 z 轴的分量
frame_b.set_ang_vel(frame_n, u2*frame_n.z)  # 设置参考框架 'b' 相对于 'n' 的角速度为 u2 关于 z 轴的分量
point_o = _me.Point('o')  # 创建点 'o'
particle_p = _me.Particle('p', _me.Point('p_pt'), _sm.Symbol('m'))  # 创建质点 'p'，位于点 'p_pt'，质量为 m
particle_r = _me.Particle('r', _me.Point('r_pt'), _sm.Symbol('m'))  # 创建质点 'r'，位于点 'r_pt'，质量为 m
particle_p.point.set_pos(point_o, l*frame_a.x)  # 设置质点 'p' 相对于点 'o' 的位置向量为 l*frame_a.x
particle_r.point.set_pos(particle_p.point, l*frame_b.x)  # 设置质点 'r' 相对于质点 'p' 的位置向量为 l*frame_b.x
point_o.set_vel(frame_n, 0)  # 设置点 'o' 在参考框架 'n' 中的速度为 0
particle_p.point.v2pt_theory(point_o, frame_n, frame_a)  # 计算质点 'p' 相对于点 'o' 的速度，基于相对运动理论
particle_r.point.v2pt_theory(particle_p.point, frame_n, frame_b)  # 计算质点 'r' 相对于质点 'p' 的速度，基于相对运动理论
particle_p.mass = m  # 设置质点 'p' 的质量为 m
particle_r.mass = m  # 设置质点 'r' 的质量为 m
force_p = particle_p.mass*(g*frame_n.x)  # 计算作用在质点 'p' 上的重力
force_r = particle_r.mass*(g*frame_n.x)  # 计算作用在质点 'r' 上的重力
kd_eqs = [q1_d - u1, q2_d - u2]  # 定义系统的速度-广义速度关系方程
forceList = [(particle_p.point, particle_p.mass*(g*frame_n.x)), (particle_r.point, particle_r.mass*(g*frame_n.x))]  # 定义作用力列表
kane = _me.KanesMethod(frame_n, q_ind=[q1, q2], u_ind=[u1, u2], kd_eqs=kd_eqs)  # 创建 Kanes 方法对象
fr, frstar = kane.kanes_equations([particle_p, particle_r], forceList)  # 计算 Kanade 方程
zero = fr + frstar  # 计算总的广义力为零的表达式
from pydy.system import System  # 从 pydy.system 导入 System 类
sys = System(kane, constants={l: 1, m: 1, g: 9.81},  # 创建 System 对象，设定常数和初始条件
             specifieds={},
             initial_conditions={q1: 0.1, q2: 0.2, u1: 0, u2: 0},
             times=_np.linspace(0.0, 10, 10 / 0.01))  # 设定时间范围

y = sys.integrate()  # 对系统进行数值积分，得到系统的演化情况
```