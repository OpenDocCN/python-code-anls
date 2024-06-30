# `D:\src\scipysrc\sympy\sympy\physics\qho_1d.py`

```
# 从 sympy 库中导入 S, pi, Rational 等核心功能
from sympy.core import S, pi, Rational
# 从 sympy 库中导入 hermite, sqrt, exp, factorial, Abs 等函数
from sympy.functions import hermite, sqrt, exp, factorial, Abs
# 从 sympy.physics.quantum.constants 中导入 hbar 常量
from sympy.physics.quantum.constants import hbar

# 定义函数 psi_n，返回一维谐振子的波函数 psi_n
def psi_n(n, x, m, omega):
    """
    Returns the wavefunction psi_{n} for the One-dimensional harmonic oscillator.

    Parameters
    ==========

    n :
        the "nodal" quantum number.  Corresponds to the number of nodes in the
        wavefunction.  ``n >= 0``
    x :
        x coordinate.
    m :
        Mass of the particle.
    omega :
        Angular frequency of the oscillator.

    Examples
    ========

    >>> from sympy.physics.qho_1d import psi_n
    >>> from sympy.abc import m, x, omega
    >>> psi_n(0, x, m, omega)
    (m*omega)**(1/4)*exp(-m*omega*x**2/(2*hbar))/(hbar**(1/4)*pi**(1/4))

    """

    # 将 n, x, m, omega 转换为 sympy 的符号对象
    n, x, m, omega = map(S, [n, x, m, omega])
    # 计算 nu，即 m * omega / hbar
    nu = m * omega / hbar
    # 计算归一化系数 C
    C = (nu/pi)**Rational(1, 4) * sqrt(1/(2**n*factorial(n)))

    # 返回波函数表达式
    return C * exp(-nu* x**2 /2) * hermite(n, sqrt(nu)*x)


# 定义函数 E_n，返回一维谐振子的能量
def E_n(n, omega):
    """
    Returns the Energy of the One-dimensional harmonic oscillator.

    Parameters
    ==========

    n :
        The "nodal" quantum number.
    omega :
        The harmonic oscillator angular frequency.

    Notes
    =====

    The unit of the returned value matches the unit of hw, since the energy is
    calculated as:

        E_n = hbar * omega*(n + 1/2)

    Examples
    ========

    >>> from sympy.physics.qho_1d import E_n
    >>> from sympy.abc import x, omega
    >>> E_n(x, omega)
    hbar*omega*(x + 1/2)
    """

    # 返回能量公式
    return hbar * omega * (n + S.Half)


# 定义函数 coherent_state，返回一维谐振子的相干态 <n|alpha>
def coherent_state(n, alpha):
    """
    Returns <n|alpha> for the coherent states of 1D harmonic oscillator.
    See https://en.wikipedia.org/wiki/Coherent_states

    Parameters
    ==========

    n :
        The "nodal" quantum number.
    alpha :
        The eigen value of annihilation operator.
    """

    # 返回相干态的表达式
    return exp(- Abs(alpha)**2/2)*(alpha**n)/sqrt(factorial(n))
```