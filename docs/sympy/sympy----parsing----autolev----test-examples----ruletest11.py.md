# `D:\src\scipysrc\sympy\sympy\parsing\autolev\test-examples\ruletest11.py`

```
# 导入符号力学模块中的力学部分作为 _me，并将其重命名为 _me
# 导入符号计算模块作为 _sm
# 导入数学模块作为 m
# 导入 NumPy 数学计算库并重命名为 _np
import sympy.physics.mechanics as _me
import sympy as _sm
import math as m
import numpy as _np

# 定义动力学符号变量 x, y
x, y = _me.dynamicsymbols('x y')

# 定义实数符号变量 a11, a12, a21, a22, b1, b2
a11, a12, a21, a22, b1, b2 = _sm.symbols('a11 a12 a21 a22 b1 b2', real=True)

# 创建一个 1x1 的符号矩阵 eqn，元素为 0
eqn = _sm.Matrix([[0]])

# 修改矩阵 eqn 的第一个元素为 a11*x + a12*y - b1
eqn[0] = a11*x + a12*y - b1

# 在矩阵 eqn 的末尾插入一个 1x1 的符号矩阵，元素为 0
eqn = eqn.row_insert(eqn.shape[0], _sm.Matrix([[0]]))

# 修改矩阵 eqn 的最后一个元素为 a21*x + a22*y - b2
eqn[eqn.shape[0] - 1] = a21*x + a22*y - b2

# 创建一个空列表 eqn_list
eqn_list = []

# 遍历矩阵 eqn 的每个元素 i，将其用给定的符号值替换后添加到 eqn_list 中
for i in eqn:
    eqn_list.append(i.subs({a11: 2, a12: 5, a21: 3, a22: 4, b1: 7, b2: 6}))

# 打印使用线性求解函数 linsolve 求解 eqn_list 中的线性方程组，解为 x 和 y 的值
print(_sm.linsolve(eqn_list, x, y))
```