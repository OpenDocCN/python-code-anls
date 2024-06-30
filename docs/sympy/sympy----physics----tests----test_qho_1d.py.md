# `D:\src\scipysrc\sympy\sympy\physics\tests\test_qho_1d.py`

```
# 导入所需的符号、常数和函数库
from sympy.core.numbers import (Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.integrals.integrals import integrate
from sympy.simplify.simplify import simplify
from sympy.abc import omega, m, x
from sympy.physics.qho_1d import psi_n, E_n, coherent_state
from sympy.physics.quantum.constants import hbar

# 计算频率符号 nu
nu = m * omega / hbar

# 定义测试波函数的函数
def test_wavefunction():
    # 定义波函数 Psi，包含四个量子数对应的波函数表达式
    Psi = {
        0: (nu/pi)**Rational(1, 4) * exp(-nu * x**2 /2),
        1: (nu/pi)**Rational(1, 4) * sqrt(2*nu) * x * exp(-nu * x**2 /2),
        2: (nu/pi)**Rational(1, 4) * (2 * nu * x**2 - 1)/sqrt(2) * exp(-nu * x**2 /2),
        3: (nu/pi)**Rational(1, 4) * sqrt(nu/3) * (2 * nu * x**3 - 3 * x) * exp(-nu * x**2 /2)
    }
    # 断言每个波函数与理论波函数的简化结果为零
    for n in Psi:
        assert simplify(psi_n(n, x, m, omega) - Psi[n]) == 0

# 定义测试波函数正交性的函数
def test_norm(n=1):
    # 对最大测试的量子数进行循环
    for i in range(n + 1):
        # 断言波函数的平方积分为1，验证波函数的归一性
        assert integrate(psi_n(i, x, 1, 1)**2, (x, -oo, oo)) == 1

# 定义测试波函数正交性的函数
def test_orthogonality(n=1):
    # 对最大测试的量子数进行循环
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            # 断言不同量子数波函数的内积为零，验证它们的正交性
            assert integrate(
                psi_n(i, x, 1, 1)*psi_n(j, x, 1, 1), (x, -oo, oo)) == 0

# 定义测试能量本征值的函数
def test_energies(n=1):
    # 对最大测试的量子数进行循环
    for i in range(n + 1):
        # 断言能量本征值公式的计算结果与理论值相等
        assert E_n(i, omega) == hbar * omega * (i + S.Half)

# 定义测试相干态的函数
def test_coherent_state(n=10):
    # 对最大测试的量子数进行循环
    # 测试相干态是否是湮灭算符的本征态
    alpha = Symbol("alpha")
    for i in range(n + 1):
        # 断言相干态的简化结果是否满足湮灭算符的本征态关系
        assert simplify(sqrt(n + 1) * coherent_state(n + 1, alpha)) == simplify(alpha * coherent_state(n, alpha))
```