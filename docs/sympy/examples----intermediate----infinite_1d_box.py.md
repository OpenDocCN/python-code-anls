# `D:\src\scipysrc\sympy\examples\intermediate\infinite_1d_box.py`

```
#!/usr/bin/env python

"""
Applying perturbation theory to calculate the ground state energy
of the infinite 1D box of width ``a`` with a perturbation
which is linear in ``x``, up to second order in perturbation
"""

from sympy.core import pi                    # 导入 sympy 的 pi 常数
from sympy import Integral, var, S           # 导入 sympy 的积分、变量、符号化功能
from sympy.functions import sin, sqrt        # 导入 sympy 的正弦函数和平方根函数


def X_n(n, a, x):
    """
    Returns the wavefunction X_{n} for an infinite 1D box

    ``n``
        the "principal" quantum number. Corresponds to the number of nodes in
        the wavefunction.  n >= 0
    ``a``
        width of the well. a > 0
    ``x``
        x coordinate.
    """
    n, a, x = map(S, [n, a, x])                # 将 n, a, x 转换为 sympy 的符号类型
    C = sqrt(2 / a)                           # 计算归一化常数 C
    return C * sin(pi * n * x / a)            # 返回波函数 X_n


def E_n(n, a, mass):
    """
    Returns the Energy psi_{n} for a 1d potential hole with infinity borders

    ``n``
        the "principal" quantum number. Corresponds to the number of nodes in
        the wavefunction.  n >= 0
    ``a``
        width of the well. a > 0
    ``mass``
        mass.
    """
    return ((n * pi / a)**2) / mass           # 返回能量 E_n


def energy_corrections(perturbation, n, *, a=10, mass=0.5):
    """
    Calculating first two order corrections due to perturbation theory and
    returns tuple where zero element is unperturbated energy, and two second
    is corrections

    ``n``
        the "nodal" quantum number. Corresponds to the number of nodes in the
        wavefunction.  n >= 0
    ``a``
        width of the well. a > 0
    ``mass``
        mass.

    """
    x, _a = var("x _a")                       # 定义 sympy 变量 x 和 _a

    # 定义积分表达式 Vnm
    Vnm = lambda n, m, a: Integral(X_n(n, a, x) * X_n(m, a, x)
        * perturbation.subs({_a: a}), (x, 0, a)).n()

    # 根据理论，计算能量修正项
    return (E_n(n, a, mass).evalf(),

            Vnm(n, n, a).evalf(),

            (Vnm(n, n - 1, a)**2/(E_n(n, a, mass) - E_n(n - 1, a, mass))
           + Vnm(n, n + 1, a)**2/(E_n(n, a, mass) - E_n(n + 1, a, mass))).evalf())


def main():
    print()
    print("Applying perturbation theory to calculate the ground state energy")
    print("of the infinite 1D box of width ``a`` with a perturbation")
    print("which is linear in ``x``, up to second order in perturbation.")
    print()

    x, _a = var("x _a")                       # 定义 sympy 变量 x 和 _a
    perturbation = .1 * x / _a                # 定义扰动项 perturbation

    E1 = energy_corrections(perturbation, 1)  # 计算第一项能量修正
    print("Energy for first term (n=1):")
    print("E_1^{(0)} = ", E1[0])              # 输出零阶能量修正
    print("E_1^{(1)} = ", E1[1])              # 输出一阶能量修正
    print("E_1^{(2)} = ", E1[2])              # 输出二阶能量修正
    print()

    E2 = energy_corrections(perturbation, 2)  # 计算第二项能量修正
    print("Energy for second term (n=2):")
    print("E_2^{(0)} = ", E2[0])              # 输出零阶能量修正
    print("E_2^{(1)} = ", E2[1])              # 输出一阶能量修正
    print("E_2^{(2)} = ", E2[2])              # 输出二阶能量修正
    print()

    E3 = energy_corrections(perturbation, 3)  # 计算第三项能量修正
    print("Energy for third term (n=3):")
    print("E_3^{(0)} = ", E3[0])              # 输出零阶能量修正
    print("E_3^{(1)} = ", E3[1])              # 输出一阶能量修正
    print("E_3^{(2)} = ", E3[2])              # 输出二阶能量修正
    print()


if __name__ == "__main__":
    main()
```