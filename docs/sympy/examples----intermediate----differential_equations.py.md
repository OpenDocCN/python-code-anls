# `D:\src\scipysrc\sympy\examples\intermediate\differential_equations.py`

```
#!/usr/bin/env python

"""Differential equations example

Demonstrates solving 1st and 2nd degree linear ordinary differential
equations.
"""

# 导入必要的库函数
from sympy import dsolve, Eq, Function, sin, Symbol

# 主函数定义
def main():
    # 符号定义：定义符号变量 x
    x = Symbol("x")
    # 函数定义：定义函数 f(x)
    f = Function("f")

    # 第一个微分方程：f'(x) = f(x)
    eq = Eq(f(x).diff(x), f(x))
    # 输出第一个微分方程的解
    print("Solution for ", eq, " : ", dsolve(eq, f(x)))

    # 第二个微分方程：f''(x) = -f(x)
    eq = Eq(f(x).diff(x, 2), -f(x))
    # 输出第二个微分方程的解
    print("Solution for ", eq, " : ", dsolve(eq, f(x)))

    # 第三个微分方程：x^2 * f'(x) = -3 * x * f(x) + sin(x) / x
    eq = Eq(x**2 * f(x).diff(x), -3 * x * f(x) + sin(x) / x)
    # 输出第三个微分方程的解
    print("Solution for ", eq, " : ", dsolve(eq, f(x)))


if __name__ == "__main__":
    main()
```