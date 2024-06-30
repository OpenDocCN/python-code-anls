# `D:\src\scipysrc\sympy\sympy\parsing\autolev\test-examples\ruletest7.py`

```
# 导入 sympy.physics.mechanics 模块，并使用别名 _me
import sympy.physics.mechanics as _me
# 导入 sympy 模块，并使用别名 _sm
import sympy as _sm
# 导入 math 模块，并使用别名 m
import math as m
# 导入 numpy 模块，并使用别名 _np
import numpy as _np

# 创建动力学符号 x 和 y
x, y = _me.dynamicsymbols('x y')
# 创建一阶导数动力学符号 x_d 和 y_d
x_d, y_d = _me.dynamicsymbols('x_ y_', 1)

# 定义表达式 e，包含多个数学函数
e = _sm.cos(x) + _sm.sin(x) + _sm.tan(x) + _sm.cosh(x) + _sm.sinh(x) + _sm.tanh(x) + _sm.acos(x) + \
    _sm.asin(x) + _sm.atan(x) + _sm.log(x) + _sm.exp(x) + _sm.sqrt(x) + _sm.factorial(x) + \
    _sm.ceiling(x) + _sm.floor(x) + _sm.sign(x)

# 重新赋值 e，包含 x 的平方和以 10 为底的对数
e = (x)**2 + _sm.log(x, 10)

# 计算 a 的值，包括绝对值、整数化和四舍五入
a = _sm.Abs(-1*1) + int(1.5) + round(1.9)

# 定义表达式 e1 和 e2
e1 = 2*x + 3*y
e2 = x + y

# 构建系数矩阵 am，展开后取出 x 和 y 的系数，然后重新形状为 2x2 矩阵
am = _sm.Matrix([e1.expand().coeff(x), e1.expand().coeff(y), e2.expand().coeff(x), e2.expand().coeff(y)]).reshape(2, 2)

# 计算 b 和 c 的值，分别为 e1 和 e2 中展开后 x 和 y 的系数
b = (e1).expand().coeff(x)
c = (e2).expand().coeff(y)

# 计算 d1 和 d2，分别为 e1 中 x 的零次和一次系数
d1 = (e1).collect(x).coeff(x, 0)
d2 = (e1).collect(x).coeff(x, 1)

# 构建系数矩阵 fm，对 e1 和 e2 进行收集 x 的操作，并重新形状为原始形状
fm = _sm.Matrix([i.collect(x) for i in _sm.Matrix([e1, e2]).reshape(1, 2)]).reshape((_sm.Matrix([e1, e2]).reshape(1, 2)).shape[0], (_sm.Matrix([e1, e2]).reshape(1, 2)).shape[1])

# 计算 f 的值，为 e1 中关于 y 的系数
f = (e1).collect(y)

# 计算 g 的值，为将 e1 中 x 替换为 2*x 后的结果
g = (e1).subs({x: 2*x})

# 构建系数矩阵 gm，将 e1 和 e2 中 x 替换为 3 后，重新形状为 2x1 矩阵
gm = _sm.Matrix([i.subs({x: 3}) for i in _sm.Matrix([e1, e2]).reshape(2, 1)]).reshape((_sm.Matrix([e1, e2]).reshape(2, 1)).shape[0], (_sm.Matrix([e1, e2]).reshape(2, 1)).shape[1])

# 创建参考框架 frame_a 和 frame_b
frame_a = _me.ReferenceFrame('a')
frame_b = _me.ReferenceFrame('b')

# 创建动力学符号 theta
theta = _me.dynamicsymbols('theta')

# frame_b 相对于 frame_a 使用轴 'Axis' 和 [theta, frame_a.z] 进行定向
frame_b.orient(frame_a, 'Axis', [theta, frame_a.z])

# 创建向量 v1 和 v2
v1 = 2*frame_a.x - 3*frame_a.y + frame_a.z
v2 = frame_b.x + frame_b.y + frame_b.z

# 计算 a 的值，为 v1 和 v2 的点积
a = _me.dot(v1, v2)

# 构建系数矩阵 bm，包括 v1 和 2*v2 的点积，然后重新形状为 2x1 矩阵
bm = _sm.Matrix([_me.dot(v1, v2), _me.dot(v1, 2*v2)]).reshape(2, 1)

# 计算 c 的值，为 v1 和 v2 的叉积
c = _me.cross(v1, v2)

# 计算 d 的值，为 v1 的 2 倍和 3 倍长度之和
d = 2*v1.magnitude() + 3*v1.magnitude()

# 创建 dyadic 张量，并进行向量外积操作
dyadic = _me.outer(3*frame_a.x, frame_a.x) + _me.outer(frame_a.y, frame_a.y) + _me.outer(2*frame_a.z, frame_a.z)

# 将 dyadic 张量相对于 frame_b 转换为矩阵形式，并重新赋值给 am
am = (dyadic).to_matrix(frame_b)

# 创建向量 m
m = _sm.Matrix([1, 2, 3]).reshape(3, 1)

# 创建向量 v，为 m 中元素与 frame_a 的基向量的线性组合
v = m[0]*frame_a.x + m[1]*frame_a.y + m[2]*frame_a.z
```