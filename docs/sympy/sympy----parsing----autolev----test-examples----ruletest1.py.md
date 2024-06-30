# `D:\src\scipysrc\sympy\sympy\parsing\autolev\test-examples\ruletest1.py`

```
import sympy.physics.mechanics as _me
import sympy as _sm
import math as m
import numpy as _np

# 定义符号变量 f 和 g，分别赋值为 3 和 9.81
f = _sm.S(3)
g = _sm.S(9.81)

# 定义符号变量 a 和 b，均为实数
a, b = _sm.symbols('a b', real=True)

# 定义符号变量 s 和 s1，均为实数
s, s1 = _sm.symbols('s s1', real=True)

# 定义符号变量 s2 和 s3，s2 为非负实数，s3 为实数
s2, s3 = _sm.symbols('s2 s3', real=True, nonnegative=True)

# 定义符号变量 s4，为非正实数
s4 = _sm.symbols('s4', real=True, nonpositive=True)

# 定义一系列符号变量，均为实数
k1, k2, k3, k4, l1, l2, l3, p11, p12, p13, p21, p22, p23 = _sm.symbols('k1 k2 k3 k4 l1 l2 l3 p11 p12 p13 p21 p22 p23', real=True)

# 定义一系列符号变量，均为实数
c11, c12, c13, c21, c22, c23 = _sm.symbols('c11 c12 c13 c21 c22 c23', real=True)

# 定义方程 e1，表示为 a*f + s2 - g
e1 = a*f+s2-g

# 定义方程 e2，表示为 f**2 + k3*k2*g
e2 = f**2+k3*k2*g
```