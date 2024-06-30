# `D:\src\scipysrc\sympy\sympy\physics\tests\test_hydrogen.py`

```
# 从 sympy 库中导入多个模块和函数
from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.integrals.integrals import integrate
from sympy.simplify.simplify import simplify
from sympy.physics.hydrogen import R_nl, E_nl, E_nl_dirac, Psi_nlm
from sympy.testing.pytest import raises

# 定义符号变量 n, r, Z
n, r, Z = symbols('n r Z')

# 定义一个函数 feq 用于比较两个数的相等性，可以指定相对误差和绝对误差
def feq(a, b, max_relative_error=1e-12, max_absolute_error=1e-12):
    # 将 a, b 转换为浮点数
    a = float(a)
    b = float(b)
    # 如果两个数的绝对误差足够小，则认为它们相等
    if abs(a - b) < max_absolute_error:
        return True
    # 否则，如果它们的相对误差很小，则也认为它们相等
    if abs(b) > abs(a):
        relative_error = abs((a - b)/b)
    else:
        relative_error = abs((a - b)/a)
    return relative_error <= max_relative_error

# 定义一个测试函数 test_wavefunction
def test_wavefunction():
    # 计算 a 的值
    a = 1/Z
    # 定义 R 字典，存储不同 (n, l) 对应的波函数表达式
    R = {
        (1, 0): 2*sqrt(1/a**3) * exp(-r/a),
        (2, 0): sqrt(1/(2*a**3)) * exp(-r/(2*a)) * (1 - r/(2*a)),
        (2, 1): S.Half * sqrt(1/(6*a**3)) * exp(-r/(2*a)) * r/a,
        (3, 0): Rational(2, 3) * sqrt(1/(3*a**3)) * exp(-r/(3*a)) *
        (1 - 2*r/(3*a) + Rational(2, 27) * (r/a)**2),
        (3, 1): Rational(4, 27) * sqrt(2/(3*a**3)) * exp(-r/(3*a)) *
        (1 - r/(6*a)) * r/a,
        (3, 2): Rational(2, 81) * sqrt(2/(15*a**3)) * exp(-r/(3*a)) * (r/a)**2,
        (4, 0): Rational(1, 4) * sqrt(1/a**3) * exp(-r/(4*a)) *
        (1 - 3*r/(4*a) + Rational(1, 8) * (r/a)**2 - Rational(1, 192) * (r/a)**3),
        (4, 1): Rational(1, 16) * sqrt(5/(3*a**3)) * exp(-r/(4*a)) *
        (1 - r/(4*a) + Rational(1, 80) * (r/a)**2) * (r/a),
        (4, 2): Rational(1, 64) * sqrt(1/(5*a**3)) * exp(-r/(4*a)) *
        (1 - r/(12*a)) * (r/a)**2,
        (4, 3): Rational(1, 768) * sqrt(1/(35*a**3)) * exp(-r/(4*a)) * (r/a)**3,
    }
    # 对 R 中的每个 (n, l) 进行断言，检验 R_nl 函数的输出是否等于预期的 R[(n, l)]
    for n, l in R:
        assert simplify(R_nl(n, l, r, Z) - R[(n, l)]) == 0

# 定义一个测试函数 test_norm
def test_norm():
    # 最大被测试的 n 值
    n_max = 2  # it works, but is slow, for n_max > 2
    # 对每个 n 和 l 进行断言，检验积分结果是否等于 1
    for n in range(n_max + 1):
        for l in range(n):
            assert integrate(R_nl(n, l, r)**2 * r**2, (r, 0, oo)) == 1

# 定义一个测试函数 test_psi_nlm
def test_psi_nlm():
    # 定义符号变量 r, phi, theta
    r = S('r')
    phi = S('phi')
    theta = S('theta')
    # 对几个 Psi_nlm 函数进行断言，检验它们的输出是否等于预期值
    assert (Psi_nlm(1, 0, 0, r, phi, theta) == exp(-r) / sqrt(pi))
    assert (Psi_nlm(2, 1, -1, r, phi, theta)) == S.Half * exp(-r / (2)) * r \
        * (sin(theta) * exp(-I * phi) / (4 * sqrt(pi)))
    assert (Psi_nlm(3, 2, 1, r, phi, theta, 2) == -sqrt(2) * sin(theta) \
         * exp(I * phi) * cos(theta) / (4 * sqrt(pi)) * S(2) / 81 \
        * sqrt(2 * 2 ** 3) * exp(-2 * r / (3)) * (r * 2) ** 2)

# 定义一个测试函数 test_hydrogen_energies
def test_hydrogen_energies():
    # 对氢原子能级公式进行断言，检验其输出是否等于预期值
    assert E_nl(n, Z) == -Z**2/(2*n**2)
    assert E_nl(n) == -1/(2*n**2)

    assert E_nl(1, 47) == -S(47)**2/(2*1**2)
    assert E_nl(2, 47) == -S(47)**2/(2*2**2)
    # 断言：计算和验证量子能级 E_nl(n) 的计算结果是否符合预期值
    assert E_nl(1) == -S.One/(2*1**2)
    # 断言：计算和验证量子能级 E_nl(n) 的计算结果是否符合预期值
    assert E_nl(2) == -S.One/(2*2**2)
    # 断言：计算和验证量子能级 E_nl(n) 的计算结果是否符合预期值
    assert E_nl(3) == -S.One/(2*3**2)
    # 断言：计算和验证量子能级 E_nl(n) 的计算结果是否符合预期值
    assert E_nl(4) == -S.One/(2*4**2)
    # 断言：计算和验证量子能级 E_nl(n) 的计算结果是否符合预期值
    assert E_nl(100) == -S.One/(2*100**2)
    
    # 断言：验证 E_nl(n) 在 n=0 时会引发 ValueError 异常
    raises(ValueError, lambda: E_nl(0))
# 定义一个测试函数，用于测试相对论性氢原子能级的计算

def test_hydrogen_energies_relat():
    # 测试小的"c"值时的精确公式，以获得简洁的表达式：
    assert E_nl_dirac(2, 0, Z=1, c=1) == 1/sqrt(2) - 1

    # 测试"c=2"时的能级，使用简化的公式进行断言
    assert simplify(E_nl_dirac(2, 0, Z=1, c=2) - ((8*sqrt(3) + 16) /
                    sqrt(16*sqrt(3) + 32) - 4)) == 0

    # 测试"c=3"时的能级，使用简化的公式进行断言
    assert simplify(E_nl_dirac(2, 0, Z=1, c=3) - ((54*sqrt(2) + 81) /
                    sqrt(108*sqrt(2) + 162) - 9)) == 0

    # 使用几乎正确的光速（非浮点数）测试能级：
    assert simplify(E_nl_dirac(2, 0, Z=1, c=137) - ((352275361 + 10285412 *
                    sqrt(1173)) / sqrt(704550722 + 20570824 * sqrt(1173)) - 18769)) == 0

    # 使用几乎正确的光速（非浮点数），但是对于Z=82的情况进行断言
    assert simplify(E_nl_dirac(2, 0, Z=82, c=137) - ((352275361 + 2571353 *
                    sqrt(12045)) / sqrt(704550722 + 5142706*sqrt(12045)) - 18769)) == 0

    # 使用精确的光速进行测试，并与非相对论情况下的能级进行比较：
    for n in range(1, 5):
        for l in range(n):
            assert feq(E_nl_dirac(n, l), E_nl(n), 1e-5, 1e-5)
            if l > 0:
                assert feq(E_nl_dirac(n, l, False), E_nl(n), 1e-5, 1e-5)

    # 测试具有特定原子序数Z的情况：
    Z = 2
    for n in range(1, 5):
        for l in range(n):
            assert feq(E_nl_dirac(n, l, Z=Z), E_nl(n, Z), 1e-4, 1e-4)
            if l > 0:
                assert feq(E_nl_dirac(n, l, False, Z), E_nl(n, Z), 1e-4, 1e-4)

    Z = 3
    for n in range(1, 5):
        for l in range(n):
            assert feq(E_nl_dirac(n, l, Z=Z), E_nl(n, Z), 1e-3, 1e-3)
            if l > 0:
                assert feq(E_nl_dirac(n, l, False, Z), E_nl(n, Z), 1e-3, 1e-3)

    # 测试异常情况：
    raises(ValueError, lambda: E_nl_dirac(0, 0))
    raises(ValueError, lambda: E_nl_dirac(1, -1))
    raises(ValueError, lambda: E_nl_dirac(1, 0, False))
```