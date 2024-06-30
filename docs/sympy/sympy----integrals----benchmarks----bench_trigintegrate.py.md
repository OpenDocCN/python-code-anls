# `D:\src\scipysrc\sympy\sympy\integrals\benchmarks\bench_trigintegrate.py`

```
# 从 sympy 库中导入 Symbol 类，用于创建符号变量
from sympy.core.symbol import Symbol
# 从 sympy 库中导入 sin 函数，用于计算正弦值
from sympy.functions.elementary.trigonometric import sin
# 从 sympy 库中导入 trigintegrate 函数，用于进行三角函数的积分计算
from sympy.integrals.trigonometry import trigintegrate

# 创建符号变量 x
x = Symbol('x')

# 定义函数 timeit_trigintegrate_sin3x，用于计算 sin(x)^3 的积分
def timeit_trigintegrate_sin3x():
    trigintegrate(sin(x)**3, x)

# 定义函数 timeit_trigintegrate_x2，用于计算 x^2 的积分
def timeit_trigintegrate_x2():
    trigintegrate(x**2, x)  # -> None
```