# `D:\src\scipysrc\sympy\sympy\parsing\autolev\test-examples\ruletest6.py`

```
# 导入 Sympy 的物理力学模块作为 _me，并重命名为 _me
import sympy.physics.mechanics as _me
# 导入 Sympy 库并重命名为 _sm
import sympy as _sm
# 导入 math 库并重命名为 m
import math as m
# 导入 NumPy 库并重命名为 _np
import numpy as _np

# 定义动力学符号 q1 和 q2
q1, q2 = _me.dynamicsymbols('q1 q2')
# 定义动力学符号 x, y, z
x, y, z = _me.dynamicsymbols('x y z')

# 计算 e 为 q1 + q2
e = q1 + q2
# 计算 a 为 e 的表达式，将 q1 替换为 x^2 + y^2，将 q2 替换为 x - y
a = e.subs({q1: x**2 + y**2, q2: x - y})

# 计算 e2 为 cos(x)
e2 = _sm.cos(x)
# 计算 e3 为 cos(x*y)
e3 = _sm.cos(x*y)

# 对 e2 在 x = 0 处展开到二阶，并移除高阶项
a = e2.series(x, 0, 2).removeO()
# 对 e3 在 x = 0 和 y = 0 处展开到二阶，并移除高阶项
b = e3.series(x, 0, 2).removeO().series(y, 0, 2).removeO()

# 计算 e 为 (x + y)^2 的展开式
e = ((x + y)**2).expand()
# 计算 a 为 e 的表达式，将 q1 替换为 x^2 + y^2，将 q2 替换为 x - y，并将 x 替换为 1，y 替换为 z
a = e.subs({q1: x**2 + y**2, q2: x - y}).subs({x: 1, y: z})

# 构造一个 2x1 的 SymPy 矩阵 bm，其中每个元素为 (e, 2*e) 在 x = 1, y = z 处的值
bm = _sm.Matrix([i.subs({x: 1, y: z}) for i in _sm.Matrix([e, 2*e]).reshape(2, 1)]).reshape((_sm.Matrix([e, 2*e]).reshape(2, 1)).shape[0], (_sm.Matrix([e, 2*e]).reshape(2, 1)).shape[1])

# 计算 e 为 q1 + q2
e = q1 + q2
# 计算 a 为 e 的表达式，将 q1 替换为 x^2 + y^2，将 q2 替换为 x - y，并将 x 替换为 2，y 替换为 z^2
a = e.subs({q1: x**2 + y**2, q2: x - y}).subs({x: 2, y: z**2})

# 定义实数符号 j, k, l
j, k, l = _sm.symbols('j k l', real=True)
# 构造一个关于 x 的多项式 p1，其系数为 j, k, l
p1 = _sm.Poly(_sm.Matrix([j, k, l]).reshape(1, 3), x)
# 构造一个关于 x 的多项式 p2，其系数为 j, k
p2 = _sm.Poly(j*x + k, x)

# 求解多项式 p1 关于 x 的根，并将结果转化为浮点数列表
root1 = [i.evalf() for i in _sm.solve(p1, x)]
# 求解一个关于 x 的矩阵形式多项式的根，并将结果转化为浮点数列表
root2 = [i.evalf() for i in _sm.solve(_sm.Poly(_sm.Matrix([1, 2, 3]).reshape(3, 1), x), x)]

# 构造一个 4x4 的 SymPy 矩阵 m
m = _sm.Matrix([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]).reshape(4, 4)
# 计算 am 为 m 的转置加上自身
am = (m).T + m

# 计算矩阵 m 的特征值，并将其转化为浮点数列表
bm = _sm.Matrix([i.evalf() for i in (m).eigenvals().keys()])

# 构造一个对角线为 (1, 1, 1, 1) 的 SymPy 对角矩阵 c1
c1 = _sm.diag(1, 1, 1, 1)
# 构造一个 3x4 的 SymPy 矩阵 c2，其中对角线上的元素为 2
c2 = _sm.Matrix([2 if i == j else 0 for i in range(3) for j in range(4)]).reshape(3, 4)

# 计算 dm 为 (m + c1) 的逆矩阵
dm = (m + c1)**(-1)

# 计算 e 为 (m + c1) 的行列式加上 2x2 单位矩阵的迹
e = (m + c1).det() + (_sm.Matrix([1, 0, 0, 1]).reshape(2, 2)).trace()

# 计算 f 为 m 矩阵的第 1 行，第 2 列的元素
f = (m)[1, 2]

# 计算 a 为矩阵 m 的列数
a = (m).cols

# 计算 bm 为矩阵 m 的第 0 列
bm = (m).col(0)

# 构造一个 5x4 的 SymPy 矩阵 cm，其中每行是矩阵 m 转置后的行
cm = _sm.Matrix([(m).T.row(0), (m).T.row(1), (m).T.row(2), (m).T.row(3), (m).T.row(2)])

# 计算 dm 为矩阵 m 的第 0 行
dm = (m).row(0)

# 构造一个 5x4 的 SymPy 矩阵 em，其中每行是矩阵 m 的行
em = _sm.Matrix([(m).row(0), (m).row(1), (m).row(2), (m).row(3), (m).row(2)])
```