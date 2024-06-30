# `D:\src\scipysrc\sympy\sympy\parsing\autolev\test-examples\ruletest5.py`

```
import sympy.physics.mechanics as _me
import sympy as _sm
import math as m
import numpy as _np

# 定义两个动力学符号变量 x 和 y
x, y = _me.dynamicsymbols('x y')

# 定义两个带有时间导数的动力学符号变量 x_d 和 y_d
x_d, y_d = _me.dynamicsymbols('x_ y_', 1)

# 定义表达式 e1, e2, e3
e1 = (x+y)**2 + (x-y)**3
e2 = (x-y)**2
e3 = x**2 + y**2 + 2*x*y

# 创建矩阵 m1，包含 e1 和 e2
m1 = _sm.Matrix([e1, e2]).reshape(2, 1)

# 创建矩阵 m2，包含 (x+y)^2 和 (x-y)^2
m2 = _sm.Matrix([(x+y)**2, (x-y)**2]).reshape(1, 2)

# 创建矩阵 m3，将 m1 和 [x, y] 向量相加
m3 = m1 + _sm.Matrix([x, y]).reshape(2, 1)

# 对矩阵 m1 的每个元素进行展开，形成矩阵 am
am = _sm.Matrix([i.expand() for i in m1]).reshape(m1.shape[0], m1.shape[1])

# 对包含 (x+y)^2 和 (x-y)^2 的矩阵进行展开，形成矩阵 cm
cm = _sm.Matrix([i.expand() for i in _sm.Matrix([(x+y)**2, (x-y)**2]).reshape(1, 2)]).reshape(_sm.Matrix([(x+y)**2, (x-y)**2]).reshape(1, 2).shape[0], _sm.Matrix([(x+y)**2, (x-y)**2]).reshape(1, 2).shape[1])

# 对 m1 和 [x, y] 向量相加的结果进行展开，形成矩阵 em
em = _sm.Matrix([i.expand() for i in m1 + _sm.Matrix([x, y]).reshape(2, 1)]).reshape((m1 + _sm.Matrix([x, y]).reshape(2, 1)).shape[0], (m1 + _sm.Matrix([x, y]).reshape(2, 1)).shape[1])

# 对表达式 e1 进行展开，形成 f
f = e1.expand()

# 对表达式 e2 进行展开，形成 g
g = e2.expand()

# 对表达式 e3 关于 x 进行因式分解，形成 a
a = _sm.factor(e3, x)

# 对矩阵 m1 中的每个元素关于 x 进行因式分解，形成矩阵 bm
bm = _sm.Matrix([_sm.factor(i, x) for i in m1]).reshape(m1.shape[0], m1.shape[1])

# 对 m1 和 [x, y] 向量相加的结果中的每个元素关于 x 进行因式分解，形成矩阵 cm
cm = _sm.Matrix([_sm.factor(i, x) for i in m1 + _sm.Matrix([x, y]).reshape(2, 1)]).reshape((m1 + _sm.Matrix([x, y]).reshape(2, 1)).shape[0], (m1 + _sm.Matrix([x, y]).reshape(2, 1)).shape[1])

# 对表达式 e3 关于 x 和 y 分别求偏导数，形成变量 a 和 b
a = e3.diff(x)
b = e3.diff(y)

# 对 m2 中的每个元素关于 x 求偏导数，形成矩阵 cm
cm = _sm.Matrix([i.diff(x) for i in m2]).reshape(m2.shape[0], m2.shape[1])

# 对 m1 和 [x, y] 向量相加的结果中的每个元素关于 x 求偏导数，形成矩阵 dm
dm = _sm.Matrix([i.diff(x) for i in m1 + _sm.Matrix([x, y]).reshape(2, 1)]).reshape((m1 + _sm.Matrix([x, y]).reshape(2, 1)).shape[0], (m1 + _sm.Matrix([x, y]).reshape(2, 1)).shape[1])

# 创建参考系 frame_a 和 frame_b
frame_a = _me.ReferenceFrame('a')
frame_b = _me.ReferenceFrame('b')

# 将 frame_b 相对于 frame_a 进行 DCM 方式的方向余弦矩阵设置
frame_b.orient(frame_a, 'DCM', _sm.Matrix([1, 0, 0, 1, 0, 0, 1, 0, 0]).reshape(3, 3))

# 定义向量 v1
v1 = x * frame_a.x + y * frame_a.y + x * y * frame_a.z

# 计算 v1 关于 x 在 frame_b 中的时间导数，形成变量 e
e = v1.diff(x, frame_b)

# 对矩阵 m1 中的每个元素关于时间 t 求导数，形成矩阵 fm
fm = _sm.Matrix([i.diff(_sm.Symbol('t')) for i in m1]).reshape(m1.shape[0], m1.shape[1])

# 对包含 (x+y)^2 和 (x-y)^2 的矩阵中的每个元素关于时间 t 求导数，形成矩阵 gm
gm = _sm.Matrix([i.diff(_sm.Symbol('t')) for i in _sm.Matrix([(x+y)**2, (x-y)**2]).reshape(1, 2)]).reshape(_sm.Matrix([(x+y)**2, (x-y)**2]).reshape(1, 2).shape[0], _sm.Matrix([(x+y)**2, (x-y)**2]).reshape(1, 2).shape[1])

# 计算向量 v1 在 frame_b 中的时间导数，形成变量 h
h = v1.dt(frame_b)
```