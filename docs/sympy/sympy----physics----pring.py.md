# `D:\src\scipysrc\sympy\sympy\physics\pring.py`

```
# 从 sympy 库中导入特定模块和函数
from sympy.core.numbers import (I, pi)
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.physics.quantum.constants import hbar

# 定义一个函数 wavefunction，用于计算环上粒子的波函数
def wavefunction(n, x):
    """
    Returns the wavefunction for particle on ring.

    Parameters
    ==========

    n : The quantum number.
        Here ``n`` can be positive as well as negative
        which can be used to describe the direction of motion of particle.
        量子数，可以是正数也可以是负数，用于描述粒子运动的方向。
    x : Symbol or expression
        The angle.
        角度。

    Examples
    ========

    >>> from sympy.physics.pring import wavefunction
    >>> from sympy import Symbol, integrate, pi
    >>> x=Symbol("x")
    >>> wavefunction(1, x)
    sqrt(2)*exp(I*x)/(2*sqrt(pi))
    >>> wavefunction(2, x)
    sqrt(2)*exp(2*I*x)/(2*sqrt(pi))
    >>> wavefunction(3, x)
    sqrt(2)*exp(3*I*x)/(2*sqrt(pi))

    The normalization of the wavefunction is:

    >>> integrate(wavefunction(2, x)*wavefunction(-2, x), (x, 0, 2*pi))
    1
    >>> integrate(wavefunction(4, x)*wavefunction(-4, x), (x, 0, 2*pi))
    1

    References
    ==========

    .. [1] Atkins, Peter W.; Friedman, Ronald (2005). Molecular Quantum
           Mechanics (4th ed.).  Pages 71-73.

    """
    # sympify arguments
    n, x = S(n), S(x)
    # 返回环上粒子的波函数
    return exp(n * I * x) / sqrt(2 * pi)


# 定义一个函数 energy，用于计算对应于量子数 ``n`` 的态的能量
def energy(n, m, r):
    """
    Returns the energy of the state corresponding to quantum number ``n``.

    E=(n**2 * (hcross)**2) / (2 * m * r**2)

    Parameters
    ==========

    n : Integer
        The quantum number.
        量子数。
    m : Symbol or expression
        Mass of the particle.
        粒子的质量。
    r : Symbol or expression
        Radius of circle.
        圆的半径。

    Examples
    ========

    >>> from sympy.physics.pring import energy
    >>> from sympy import Symbol
    >>> m=Symbol("m")
    >>> r=Symbol("r")
    >>> energy(1, m, r)
    hbar**2/(2*m*r**2)
    >>> energy(2, m, r)
    2*hbar**2/(m*r**2)
    >>> energy(-2, 2.0, 3.0)
    0.111111111111111*hbar**2

    References
    ==========

    .. [1] Atkins, Peter W.; Friedman, Ronald (2005). Molecular Quantum
           Mechanics (4th ed.).  Pages 71-73.

    """
    n, m, r = S(n), S(m), S(r)
    # 如果 n 是整数，计算并返回态的能量
    if n.is_integer:
        return (n**2 * hbar**2) / (2 * m * r**2)
    else:
        # 如果 n 不是整数，抛出 ValueError
        raise ValueError("'n' must be integer")
```