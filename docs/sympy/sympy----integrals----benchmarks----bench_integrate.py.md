# `D:\src\scipysrc\sympy\sympy\integrals\benchmarks\bench_integrate.py`

```
# 从 sympy 库中导入 Symbol 类，用于创建符号变量 x
from sympy.core.symbol import Symbol
# 从 sympy 库中导入 sin 函数，用于计算正弦值
from sympy.functions.elementary.trigonometric import sin
# 从 sympy 库中导入 integrate 函数，用于执行积分运算
from sympy.integrals.integrals import integrate

# 创建符号变量 x
x = Symbol('x')

# 定义函数 bench_integrate_sin，用于对 sin(x) 进行积分
def bench_integrate_sin():
    # 调用 sympy 的 integrate 函数，对 sin(x) 进行积分
    integrate(sin(x), x)

# 定义函数 bench_integrate_x1sin，用于对 x*sin(x) 进行积分
def bench_integrate_x1sin():
    # 调用 sympy 的 integrate 函数，对 x*sin(x) 进行积分
    integrate(x**1*sin(x), x)

# 定义函数 bench_integrate_x2sin，用于对 x^2*sin(x) 进行积分
def bench_integrate_x2sin():
    # 调用 sympy 的 integrate 函数，对 x^2*sin(x) 进行积分
    integrate(x**2*sin(x), x)

# 定义函数 bench_integrate_x3sin，用于对 x^3*sin(x) 进行积分
def bench_integrate_x3sin():
    # 调用 sympy 的 integrate 函数，对 x^3*sin(x) 进行积分
    integrate(x**3*sin(x), x)
```