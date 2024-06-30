# `D:\src\scipysrc\sympy\sympy\parsing\autolev\test-examples\ruletest10.py`

```
# 导入 Sympy 的力学模块，别名为 _me
# 导入 Sympy 符号计算模块，别名为 _sm
# 导入数学模块，别名为 m
# 导入 NumPy 数学计算模块，别名为 _np
import sympy.physics.mechanics as _me
import sympy as _sm
import math as m
import numpy as _np

# 定义动力学符号 x 和 y
x, y = _me.dynamicsymbols('x y')

# 定义符号 a 和 b，并指定为实数
a, b = _sm.symbols('a b', real=True)

# 定义表达式 e
e = a*(b*x+y)**2

# 创建一个 2x1 的矩阵 m，其中每个元素为表达式 e
m = _sm.Matrix([e,e]).reshape(2, 1)

# 将表达式 e 展开
e = e.expand()

# 将矩阵 m 中每个元素展开后重新组成矩阵
m = _sm.Matrix([i.expand() for i in m]).reshape((m).shape[0], (m).shape[1])

# 对表达式 e 关于变量 x 进行因式分解
e = _sm.factor(e, x)

# 对矩阵 m 中每个元素关于变量 x 进行因式分解
m = _sm.Matrix([_sm.factor(i,x) for i in m]).reshape((m).shape[0], (m).shape[1])

# 创建一个形状为 (1,1) 的零矩阵 eqn
eqn = _sm.Matrix([[0]])

# 将方程组 eqn 的第一行设置为表达式 a*x+b*y
eqn[0] = a*x+b*y

# 在方程组 eqn 的末尾插入一个形状为 (1,1) 的零矩阵
eqn = eqn.row_insert(eqn.shape[0], _sm.Matrix([[0]]))

# 将方程组 eqn 的最后一行设置为表达式 2*a*x-3*b*y
eqn[eqn.shape[0]-1] = 2*a*x-3*b*y

# 打印方程组 eqn 关于变量 x 和 y 的解
print(_sm.solve(eqn,x,y))

# 求解方程组 eqn 关于变量 x 和 y 的解中的 y 部分
rhs_y = _sm.solve(eqn,x,y)[y]

# 对表达式 e 关于变量 x 进行收集同类项
e = (x+y)**2+2*x**2
e.collect(x)

# 定义符号 a, b, c，并指定为实数
a, b, c = _sm.symbols('a b c', real=True)

# 创建一个 2x2 的矩阵 m，元素为符号 a, b, c, 0
m = _sm.Matrix([a,b,c,0]).reshape(2, 2)

# 将矩阵 m 中每个元素替换为符号 {a:1, b:2, c:3} 对应的值，再重新组成矩阵
m2 = _sm.Matrix([i.subs({a:1,b:2,c:3}) for i in m]).reshape((m).shape[0], (m).shape[1])

# 计算矩阵 m2 的特征值，并将结果转换为浮点数
eigvalue = _sm.Matrix([i.evalf() for i in (m2).eigenvals().keys()])

# 计算矩阵 m2 的特征向量，并将结果中每个向量的第三个元素取出并转换为浮点数
eigvec = _sm.Matrix([i[2][0].evalf() for i in (m2).eigenvects()]).reshape(m2.shape[0], m2.shape[1])

# 创建一个惯性参考系 frame_n
frame_n = _me.ReferenceFrame('n')

# 创建一个相对于 frame_n 的参考系 frame_a
frame_a = _me.ReferenceFrame('a')

# 将参考系 frame_a 相对于 frame_n 进行坐标轴的定向，使用向量 x 和 frame_n.x
frame_a.orient(frame_n, 'Axis', [x, frame_n.x])

# 将参考系 frame_a 相对于 frame_n 进行坐标轴的定向，使用角度 _sm.pi/2 和 frame_n.x
frame_a.orient(frame_n, 'Axis', [_sm.pi/2, frame_n.x])

# 定义符号 c1, c2, c3，并指定为实数
c1, c2, c3 = _sm.symbols('c1 c2 c3', real=True)

# 创建一个向量 v，其基底为 frame_a 的 x, y, z 轴，并使用符号 c1, c2, c3 进行线性组合
v = c1*frame_a.x+c2*frame_a.y+c3*frame_a.z

# 创建一个名为 point_o 的点
point_o = _me.Point('o')

# 创建一个名为 point_p 的点
point_p = _me.Point('p')

# 将点 point_o 相对于 point_p 的位置设置为向量 c1*frame_a.x
point_o.set_pos(point_p, c1*frame_a.x)

# 将向量 v 表示为相对于参考系 frame_n 的向量
v = (v).express(frame_n)

# 将点 point_o 相对于 point_p 的位置设置为相对位置 (point_o.pos_from(point_p))，并表示为相对于参考系 frame_n 的向量
point_o.set_pos(point_p, (point_o.pos_from(point_p)).express(frame_n))

# 将参考系 frame_a 相对于 frame_n 的角速度设置为 c3*frame_a.z
frame_a.set_ang_vel(frame_n, c3*frame_a.z)

# 打印参考系 frame_n 在参考系 frame_a 中的角速度
print(frame_n.ang_vel_in(frame_a))

# 计算点 point_p 相对于 point_o 的速度，并考虑参考系 frame_n 和 frame_a
point_p.v2pt_theory(point_o,frame_n,frame_a)

# 创建一个名为 particle_p1 的质点，质量为符号 'm'，位于名为 'p1_pt' 的点上
particle_p1 = _me.Particle('p1', _me.Point('p1_pt'), _sm.Symbol('m'))

# 创建一个名为 particle_p2 的质点，质量为符号 'm'，位于名为 'p2_pt' 的点上
particle_p2 = _me.Particle('p2', _me.Point('p2_pt'), _sm.Symbol('m'))

# 计算 particle_p2 的质点上一点的速度，相对于 particle_p1 的质点上一点，考虑参考系 frame_n 和 frame_a
particle_p2.point.v2pt_theory(particle_p1.point,frame_n,frame_a)

# 计算点 point_p 相对于 particle_p1 的质点上一点的加速度，考虑参考系 frame_n 和 frame_a
point_p.a2pt_theory(particle_p1.point,frame_n,frame_a)

# 创建一个名为 body_b1_cm 的质心点 'b1_cm'
body_b1_cm = _me.Point('b1_cm')

# 将质心点 body_b1_cm 相对于参考系 frame_n 的速度设置为 0
body_b1_cm.set_vel(frame_n, 0)

# 创建一个名为 body_b1_f 的相对于 frame_n 的参考系 'b1_f'
body_b1_f = _me.ReferenceFrame('b1_f')

# 创建一个质量为符号 'm' 的名为 'b1' 的刚体 body_b1，其质心为 body_b1_cm，参考系为 body_b1_f
body_b1 = _me.RigidBody('b1', body_b1_cm, body_b1_f, _sm.symbols('m'), (_me.outer(body_b1_f.x,body_b1_f.x),body_b1_cm))

# 创建一个名为 body_b2_cm 的质心点 'b2_cm'
body_b2_cm = _me.Point('b2_cm')

# 将质心点 body_b2_cm 相对于参考系 frame_n 的速度设置为 0
body_b2_cm.set_vel(frame_n, 0)
```