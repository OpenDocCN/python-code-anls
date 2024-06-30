# `D:\src\scipysrc\sympy\sympy\parsing\autolev\test-examples\ruletest12.py`

```
# 导入符号力学模块中的力学类别，并重命名为_me
import sympy.physics.mechanics as _me
# 导入 sympy 符号计算模块，并重命名为_sm
import sympy as _sm
# 导入数学函数模块，并重命名为m
import math as m
# 导入 numpy 数组计算模块，并重命名为_np
import numpy as _np

# 创建动力学符号变量 x 和 y
x, y = _me.dynamicsymbols('x y')
# 定义符号变量 a, b, r，均为实数
a, b, r = _sm.symbols('a b r', real=True)

# 创建一个 1x1 的零矩阵 eqn
eqn = _sm.Matrix([[0]])
# 设置矩阵 eqn 的第一个元素为表达式 a*x**3 + b*y**2 - r
eqn[0] = a*x**3 + b*y**2 - r

# 在矩阵 eqn 的末尾插入一个新的 1x1 零矩阵
eqn = eqn.row_insert(eqn.shape[0], _sm.Matrix([[0]]))
# 设置插入矩阵中的第一个元素为表达式 a*sin(x)**2 + b*cos(2*y) - r**2
eqn[eqn.shape[0] - 1] = a*_sm.sin(x)**2 + b*_sm.cos(2*y) - r**2

# 创建一个空列表 matrix_list
matrix_list = []
# 遍历矩阵 eqn 中的每个元素，将其代入参数 a=2.0, b=3.0, r=1.0，然后加入到 matrix_list 中
for i in eqn:
    matrix_list.append(i.subs({a: 2.0, b: 3.0, r: 1.0}))

# 使用数值求解函数 _sm.nsolve 解方程组 matrix_list，求解变量 (x, y)，初始猜测角度为 30 度对应的弧度，y=3.14
print(_sm.nsolve(matrix_list, (x, y), (_np.deg2rad(30), 3.14)))
```