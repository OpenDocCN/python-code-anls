# `D:\src\scipysrc\sympy\sympy\physics\tests\test_pring.py`

```
# 导入波函数和能量相关的模块
from sympy.physics.pring import wavefunction, energy
# 导入复数单位和圆周率
from sympy.core.numbers import (I, pi)
# 导入指数函数
from sympy.functions.elementary.exponential import exp
# 导入平方根函数
from sympy.functions.elementary.miscellaneous import sqrt
# 导入积分函数
from sympy.integrals.integrals import integrate
# 导入化简函数
from sympy.simplify.simplify import simplify
# 导入符号变量 m, x, r
from sympy.abc import m, x, r
# 导入量子力学常数 hbar
from sympy.physics.quantum.constants import hbar


def test_wavefunction():
    # 定义波函数字典 Psi，包含不同量子数对应的波函数表达式
    Psi = {
        0: (1/sqrt(2 * pi)),
        1: (1/sqrt(2 * pi)) * exp(I * x),
        2: (1/sqrt(2 * pi)) * exp(2 * I * x),
        3: (1/sqrt(2 * pi)) * exp(3 * I * x)
    }
    # 遍历 Psi 中的每个量子数 n，并检查计算得到的波函数与 Psi[n] 的化简结果是否为零
    for n in Psi:
        assert simplify(wavefunction(n, x) - Psi[n]) == 0


def test_norm(n=1):
    # 测试波函数正交性的最大量子数 "n"
    for i in range(n + 1):
        # 断言波函数在区间 [0, 2 * pi] 的归一化积分结果为 1
        assert integrate(
            wavefunction(i, x) * wavefunction(-i, x), (x, 0, 2 * pi)) == 1


def test_orthogonality(n=1):
    # 测试波函数正交性和相互正交性的最大量子数 "n"
    for i in range(n + 1):
        for j in range(i+1, n+1):
            # 断言波函数在区间 [0, 2 * pi] 的积分结果为 0，表明不同量子数的波函数正交
            assert integrate(
                wavefunction(i, x) * wavefunction(j, x), (x, 0, 2 * pi)) == 0


def test_energy(n=1):
    # 测试波函数对应的能量公式的最大量子数 "n"
    for i in range(n+1):
        # 断言能量公式计算结果与理论预期值的化简结果是否为零
        assert simplify(
            energy(i, m, r) - ((i**2 * hbar**2) / (2 * m * r**2))) == 0
```