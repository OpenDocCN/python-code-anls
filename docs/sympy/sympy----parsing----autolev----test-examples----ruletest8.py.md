# `D:\src\scipysrc\sympy\sympy\parsing\autolev\test-examples\ruletest8.py`

```
import sympy.physics.mechanics as _me
import sympy as _sm
import math as m
import numpy as _np

# 创建参考系 frame_a
frame_a = _me.ReferenceFrame('a')

# 定义符号变量 c1, c2, c3，并声明其为实数
c1, c2, c3 = _sm.symbols('c1 c2 c3', real=True)

# 创建惯性张量 a
a = _me.inertia(frame_a, 1, 1, 1)

# 创建粒子 particle_p1 和 particle_p2
particle_p1 = _me.Particle('p1', _me.Point('p1_pt'), _sm.Symbol('m'))
particle_p2 = _me.Particle('p2', _me.Point('p2_pt'), _sm.Symbol('m'))

# 创建质心点 body_r_cm 和参考系 body_r_f
body_r_cm = _me.Point('r_cm')
body_r_f = _me.ReferenceFrame('r_f')

# 创建刚体 body_r
body_r = _me.RigidBody('r', body_r_cm, body_r_f, _sm.symbols('m'), (_me.outer(body_r_f.x,body_r_f.x),body_r_cm))

# 设置 frame_a 相对于 body_r_f 的方向余弦矩阵
frame_a.orient(body_r_f, 'DCM', _sm.Matrix([1,1,1,1,1,0,0,0,1]).reshape(3, 3))

# 创建点 point_o
point_o = _me.Point('o')

# 创建符号变量 m1, m2, mr, i1, i2, i3
m1 = _sm.symbols('m1')
m2 = _sm.symbols('m2')
mr = _sm.symbols('mr')
i1 = _sm.symbols('i1')
i2 = _sm.symbols('i2')
i3 = _sm.symbols('i3')

# 分别将质量赋值给 particle_p1, particle_p2 和 body_r
particle_p1.mass = m1
particle_p2.mass = m2
body_r.mass = mr

# 设置 body_r 的惯性张量
body_r.inertia = (_me.inertia(body_r_f, i1, i2, i3, 0, 0, 0), body_r_cm)

# 设置 point_o 相对于 particle_p1, particle_p2 和 body_r_cm 的位置
point_o.set_pos(particle_p1.point, c1*frame_a.x)
point_o.set_pos(particle_p2.point, c2*frame_a.y)
point_o.set_pos(body_r_cm, c3*frame_a.z)

# 计算 particle_p1 相对于 point_o 的质量惯性
a = _me.inertia_of_point_mass(particle_p1.mass, particle_p1.point.pos_from(point_o), frame_a)

# 计算 particle_p2 相对于 point_o 的质量惯性
a = _me.inertia_of_point_mass(particle_p2.mass, particle_p2.point.pos_from(point_o), frame_a)

# 计算 body_r 相对于 point_o 的质量惯性
a = body_r.inertia[0] + _me.inertia_of_point_mass(body_r.mass, body_r.masscenter.pos_from(point_o), frame_a)

# 计算总的质量惯性
a = _me.inertia_of_point_mass(particle_p1.mass, particle_p1.point.pos_from(point_o), frame_a) + _me.inertia_of_point_mass(particle_p2.mass, particle_p2.point.pos_from(point_o), frame_a) + body_r.inertia[0] + _me.inertia_of_point_mass(body_r.mass, body_r.masscenter.pos_from(point_o), frame_a)

# 计算部分质量惯性
a = _me.inertia_of_point_mass(particle_p1.mass, particle_p1.point.pos_from(point_o), frame_a) + body_r.inertia[0] + _me.inertia_of_point_mass(body_r.mass, body_r.masscenter.pos_from(point_o), frame_a)

# 获取 body_r 的主惯量
a = body_r.inertia[0]

# 设置 particle_p2 相对于 particle_p1 的位置
particle_p2.point.set_pos(particle_p1.point, c1*frame_a.x+c2*frame_a.y)

# 设置 body_r_cm 相对于 particle_p1, particle_p2 的位置
body_r_cm.set_pos(particle_p1.point, c3*frame_a.x)
body_r_cm.set_pos(particle_p2.point, c3*frame_a.y)

# 计算系统的质心
b = _me.functions.center_of_mass(point_o, particle_p1, particle_p2, body_r)
b = _me.functions.center_of_mass(point_o, particle_p1, body_r)
b = _me.functions.center_of_mass(particle_p1.point, particle_p1, particle_p2, body_r)

# 定义速度变量 u1, u2, u3
u1, u2, u3 = _me.dynamicsymbols('u1 u2 u3')

# 定义速度向量 v
v = u1*frame_a.x + u2*frame_a.y + u3*frame_a.z

# 计算单位向量 u
u = (v + c1*frame_a.x).normalize()

# 设置 particle_p1 的速度
particle_p1.point.set_vel(frame_a, u1*frame_a.x)

# 计算 particle_p1 相对于 frame_a 的速度分量
a = particle_p1.point.partial_velocity(frame_a, u1)

# 计算系统的总质量 m
m = particle_p1.mass + body_r.mass

# 设置 particle_p2 的质量
m = particle_p2.mass

# 计算系统的总质量 m
m = particle_p1.mass + particle_p2.mass + body_r.mass
```