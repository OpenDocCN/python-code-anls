# `D:\src\scipysrc\sympy\examples\intermediate\partial_differential_eqs.py`

```
#!/usr/bin/env python

"""Partial Differential Equations example

Demonstrates various ways to solve partial differential equations
"""

# 导入需要的符号和函数
from sympy import symbols, Eq, Function, pde_separate, pprint, sin, cos
from sympy import Derivative as D

# 主函数
def main():
    # 定义符号变量
    r, phi, theta = symbols("r,phi,theta")
    # 定义函数 Xi
    Xi = Function('Xi')
    # 定义其他函数和常数
    R, Phi, Theta, u = map(Function, ['R', 'Phi', 'Theta', 'u'])
    C1, C2 = symbols('C1,C2')

    # 打印 Laplace 方程在球坐标系中的表达式
    pprint("Separation of variables in Laplace equation in spherical coordinates")
    pprint("Laplace equation in spherical coordinates:")
    # 构建 Laplace 方程
    eq = Eq(D(Xi(r, phi, theta), r, 2) + 2/r * D(Xi(r, phi, theta), r) +
            1/(r**2 * sin(phi)**2) * D(Xi(r, phi, theta), theta, 2) +
            cos(phi)/(r**2 * sin(phi)) * D(Xi(r, phi, theta), phi) +
            1/r**2 * D(Xi(r, phi, theta), phi, 2), 0)
    pprint(eq)

    # 分离变量，首先针对变量 r
    pprint("We can either separate this equation in regards with variable r:")
    res_r = pde_separate(eq, Xi(r, phi, theta), [R(r), u(phi, theta)])
    pprint(res_r)

    # 然后针对变量 theta
    pprint("Or separate it in regards of theta:")
    res_theta = pde_separate(eq, Xi(r, phi, theta), [Theta(theta), u(r, phi)])
    pprint(res_theta)

    # 尝试针对变量 phi 分离，但不成功
    res_phi = pde_separate(eq, Xi(r, phi, theta), [Phi(phi), u(r, theta)])
    pprint("But we cannot separate it in regards of variable phi: ")
    pprint("Result: %s" % res_phi)

    # 将 theta 的部分设为 -C1
    pprint("\n\nSo let's make theta dependent part equal with -C1:")
    eq_theta = Eq(res_theta[0], -C1)
    pprint(eq_theta)

    # 第二部分也等于 -C1
    pprint("\nThis also means that second part is also equal to -C1:")
    eq_left = Eq(res_theta[1], -C1)
    pprint(eq_left)

    # 再次尝试分离 phi 变量
    pprint("\nLets try to separate phi again :)")
    res_theta = pde_separate(eq_left, u(r, phi), [Phi(phi), R(r)])
    pprint("\nThis time it is successful:")
    pprint(res_theta)

    # 打印最终分离后的方程组
    pprint("\n\nSo our final equations with separated variables are:")
    pprint(eq_theta)
    pprint(Eq(res_theta[0], C2))
    pprint(Eq(res_theta[1], C2))


if __name__ == "__main__":
    main()
```