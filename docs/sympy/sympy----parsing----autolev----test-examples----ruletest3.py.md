# `D:\src\scipysrc\sympy\sympy\parsing\autolev\test-examples\ruletest3.py`

```
import sympy.physics.mechanics as _me  # 导入 sympy 的力学模块，命名为 _me
import sympy as _sm  # 导入 sympy，命名为 _sm
import math as m  # 导入 math 模块，命名为 m
import numpy as _np  # 导入 numpy 模块，命名为 _np

# 创建参考系 frame_a、frame_b、frame_n，并命名为 'a'、'b'、'n'
frame_a = _me.ReferenceFrame('a')
frame_b = _me.ReferenceFrame('b')
frame_n = _me.ReferenceFrame('n')

# 定义动力学符号 x1, x2, x3
x1, x2, x3 = _me.dynamicsymbols('x1 x2 x3')

# 定义实数符号 l
l = _sm.symbols('l', real=True)

# 创建向量 v1、v2、v3，分别使用 frame_a、frame_b、frame_n 的基向量与动力学符号 x1、x2、x3 的组合
v1 = x1 * frame_a.x + x2 * frame_a.y + x3 * frame_a.z
v2 = x1 * frame_b.x + x2 * frame_b.y + x3 * frame_b.z
v3 = x1 * frame_n.x + x2 * frame_n.y + x3 * frame_n.z

# 向量 v 是 v1、v2、v3 的和
v = v1 + v2 + v3

# 创建四个点 point_c、point_d、point_po1、point_po2、point_po3，并命名为 'c'、'd'、'po1'、'po2'、'po3'
point_c = _me.Point('c')
point_d = _me.Point('d')
point_po1 = _me.Point('po1')
point_po2 = _me.Point('po2')
point_po3 = _me.Point('po3')

# 创建三个粒子 particle_l、particle_p1、particle_p2、particle_p3，每个粒子有质量符号 'm'，分别命名为 'l'、'p1'、'p2'、'p3'
particle_l = _me.Particle('l', _me.Point('l_pt'), _sm.Symbol('m'))
particle_p1 = _me.Particle('p1', _me.Point('p1_pt'), _sm.Symbol('m'))
particle_p2 = _me.Particle('p2', _me.Point('p2_pt'), _sm.Symbol('m'))
particle_p3 = _me.Particle('p3', _me.Point('p3_pt'), _sm.Symbol('m'))

# 创建质心 body_s_cm，命名为 's_cm'，并设置其在参考系 frame_n 中的速度为零
body_s_cm = _me.Point('s_cm')
body_s_cm.set_vel(frame_n, 0)

# 创建参考系 body_s_f，命名为 's_f'
body_s_f = _me.ReferenceFrame('s_f')

# 创建刚体 body_s，命名为 's'，质心在 body_s_cm，参考系为 body_s_f，质量符号为 'm'，惯性张量为 (body_s_f.x, body_s_f.x)，质心为 body_s_cm
body_s = _me.RigidBody('s', body_s_cm, body_s_f, _sm.symbols('m'), (_me.outer(body_s_f.x,body_s_f.x),body_s_cm))

# 创建质心 body_r1_cm，命名为 'r1_cm'，并设置其在参考系 frame_n 中的速度为零
body_r1_cm = _me.Point('r1_cm')
body_r1_cm.set_vel(frame_n, 0)

# 创建参考系 body_r1_f，命名为 'r1_f'
body_r1_f = _me.ReferenceFrame('r1_f')

# 创建刚体 body_r1，命名为 'r1'，质心在 body_r1_cm，参考系为 body_r1_f，质量符号为 'm'，惯性张量为 (body_r1_f.x, body_r1_f.x)，质心为 body_r1_cm
body_r1 = _me.RigidBody('r1', body_r1_cm, body_r1_f, _sm.symbols('m'), (_me.outer(body_r1_f.x,body_r1_f.x),body_r1_cm))

# 创建质心 body_r2_cm，命名为 'r2_cm'，并设置其在参考系 frame_n 中的速度为零
body_r2_cm = _me.Point('r2_cm')
body_r2_cm.set_vel(frame_n, 0)

# 创建参考系 body_r2_f，命名为 'r2_f'
body_r2_f = _me.ReferenceFrame('r2_f')

# 创建刚体 body_r2，命名为 'r2'，质心在 body_r2_cm，参考系为 body_r2_f，质量符号为 'm'，惯性张量为 (body_r2_f.x, body_r2_f.x)，质心为 body_r2_cm
body_r2 = _me.RigidBody('r2', body_r2_cm, body_r2_f, _sm.symbols('m'), (_me.outer(body_r2_f.x,body_r2_f.x),body_r2_cm))

# 创建向量 v4，使用 body_s_f 的基向量与动力学符号 x1、x2、x3 的组合
v4 = x1 * body_s_f.x + x2 * body_s_f.y + x3 * body_s_f.z

# 设置 body_s_cm 的位置，相对于 point_c，沿 frame_n 的 x 方向移动 l 单位
body_s_cm.set_pos(point_c, l * frame_n.x)
```