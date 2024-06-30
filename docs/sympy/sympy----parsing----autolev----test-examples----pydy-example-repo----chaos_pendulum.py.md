# `D:\src\scipysrc\sympy\sympy\parsing\autolev\test-examples\pydy-example-repo\chaos_pendulum.py`

```
# 导入 Sympy 中的力学模块
import sympy.physics.mechanics as _me
# 导入 Sympy 库
import sympy as _sm
# 导入 math 库并重命名为 m
import math as m
# 导入 NumPy 库并重命名为 _np
import numpy as _np

# 定义符号变量 g, lb, w, h，并声明为实数
g, lb, w, h = _sm.symbols('g lb w h', real=True)
# 定义动力学符号变量 theta, phi, omega, alpha
theta, phi, omega, alpha = _me.dynamicsymbols('theta phi omega alpha')
# 定义这些动力学符号的一阶导数
theta_d, phi_d, omega_d, alpha_d = _me.dynamicsymbols('theta_ phi_ omega_ alpha_', 1)
# 定义这些动力学符号的二阶导数
theta_dd, phi_dd = _me.dynamicsymbols('theta_ phi_', 2)

# 创建参考坐标系 'n'
frame_n = _me.ReferenceFrame('n')
# 创建质心点 'a_cm'
body_a_cm = _me.Point('a_cm')
# 设置 'a_cm' 相对于 'n' 的速度为 0
body_a_cm.set_vel(frame_n, 0)
# 创建参考坐标系 'a_f'
body_a_f = _me.ReferenceFrame('a_f')
# 创建质点 'a'，其中质心 'body_a_cm' 的质量为 'm'
body_a = _me.RigidBody('a', body_a_cm, body_a_f, _sm.symbols('m'), (_me.outer(body_a_f.x,body_a_f.x),body_a_cm))

# 创建质心点 'b_cm'
body_b_cm = _me.Point('b_cm')
# 设置 'b_cm' 相对于 'n' 的速度为 0
body_b_cm.set_vel(frame_n, 0)
# 创建参考坐标系 'b_f'
body_b_f = _me.ReferenceFrame('b_f')
# 创建质点 'b'，其中质心 'body_b_cm' 的质量为 'm'
body_b = _me.RigidBody('b', body_b_cm, body_b_f, _sm.symbols('m'), (_me.outer(body_b_f.x,body_b_f.x),body_b_cm))

# 使用 'n' 参考坐标系和角度 'theta' 将 'a_f' 与 'n' 进行定向
body_a_f.orient(frame_n, 'Axis', [theta, frame_n.y])
# 使用 'a_f' 参考坐标系和角度 'phi' 将 'b_f' 与 'a_f' 进行定向
body_b_f.orient(body_a_f, 'Axis', [phi, body_a_f.z])

# 创建点 'o' 作为定点
point_o = _me.Point('o')
# 计算 la 的值
la = (lb-h/2)/2
# 设置 'a_cm' 相对于 'o' 点的位置
body_a_cm.set_pos(point_o, la*body_a_f.z)
# 设置 'b_cm' 相对于 'o' 点的位置
body_b_cm.set_pos(point_o, lb*body_a_f.z)

# 设置 'a_f' 相对于 'n' 的角速度为 'omega' 关于 'n.y' 轴的分量
body_a_f.set_ang_vel(frame_n, omega*frame_n.y)
# 设置 'b_f' 相对于 'a_f' 的角速度为 'alpha' 关于 'a_f.z' 轴的分量
body_b_f.set_ang_vel(body_a_f, alpha*body_a_f.z)

# 设置 'o' 点相对于 'n' 的速度为 0
point_o.set_vel(frame_n, 0)

# 计算 'body_a_cm' 相对于 'o' 点的速度
body_a_cm.v2pt_theory(point_o,frame_n,body_a_f)
# 计算 'body_b_cm' 相对于 'o' 点的速度
body_b_cm.v2pt_theory(point_o,frame_n,body_a_f)

# 定义质量 'ma' 并赋值给 'body_a' 的质量
ma = _sm.symbols('ma')
body_a.mass = ma
# 定义质量 'mb' 并赋值给 'body_b' 的质量
mb = _sm.symbols('mb')
body_b.mass = mb

# 计算质点 'a' 的惯性矩 'iaxx', 'iayy', 'iazz'
iaxx = 1/12*ma*(2*la)**2
iayy = iaxx
iazz = 0
# 计算质点 'b' 的惯性矩 'ibxx', 'ibyy', 'ibzz'
ibxx = 1/12*mb*h**2
ibyy = 1/12*mb*(w**2+h**2)
ibzz = 1/12*mb*w**2
# 设置 'body_a' 的惯性为计算得到的惯性矩
body_a.inertia = (_me.inertia(body_a_f, iaxx, iayy, iazz, 0, 0, 0), body_a_cm)
# 设置 'body_b' 的惯性为计算得到的惯性矩
body_b.inertia = (_me.inertia(body_b_f, ibxx, ibyy, ibzz, 0, 0, 0), body_b_cm)

# 计算质点 'a' 受到的力 'force_a'
force_a = body_a.mass*(g*frame_n.z)
# 计算质点 'b' 受到的力 'force_b'
force_b = body_b.mass*(g*frame_n.z)

# 定义运动方程的约束条件 'kd_eqs'，其中包含角速度的关系
kd_eqs = [theta_d - omega, phi_d - alpha]

# 定义力的列表 'forceList'，包含质点 'a' 和 'b' 受到的力
forceList = [(body_a.masscenter,body_a.mass*(g*frame_n.z)), (body_b.masscenter,body_b.mass*(g*frame_n.z))]

# 使用 KanesMethod 创建运动方程的框架 'kane'
kane = _me.KanesMethod(frame_n, q_ind=[theta,phi], u_ind=[omega, alpha], kd_eqs = kd_eqs)

# 计算约束力 'fr' 和广义力 'frstar'
fr, frstar = kane.kanes_equations([body_a, body_b], forceList)

# 定义一个零向量 'zero'，用于等式的平衡条件
zero = fr+frstar

# 导入 pydy 系统库中的 System
from pydy.system import System

# 创建系统 'sys'，包括 'kane' 的运动方程、常数和初始条件
sys = System(kane, constants = {g:9.81, lb:0.2, w:0.2, h:0.1, ma:0.01, mb:0.1},
             specifieds={},
             initial_conditions={theta:_np.deg2rad(90), phi:_np.deg2rad(0.5), omega:0, alpha:0},
             times = _np.linspace(0.0, 10, 10/0.02))

# 对系统进行数值积分并赋值给 'y'
y=sys.integrate()
```