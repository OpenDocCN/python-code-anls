# `D:\src\scipysrc\sympy\sympy\functions\elementary\_trigonometric_special.py`

```
from __future__ import annotations
# 导入 annotations 模块，支持类型提示中的类型自引用

from typing import Callable
# 导入 Callable 类型，用于函数类型提示

from functools import reduce
# 导入 reduce 函数，用于对可迭代对象应用函数，累积处理结果

from sympy.core.expr import Expr
# 导入 Expr 类，表示 sympy 表达式的基类

from sympy.core.singleton import S
# 导入 S 单例，表示 sympy 单例的特殊对象

from sympy.core.intfunc import igcdex
# 导入 igcdex 函数，用于计算多个整数的扩展最大公约数

from sympy.core.numbers import Integer
# 导入 Integer 类，表示 sympy 中的整数类型

from sympy.functions.elementary.miscellaneous import sqrt
# 导入 sqrt 函数，表示 sympy 中的平方根函数

from sympy.core.cache import cacheit
# 导入 cacheit 函数，用于缓存函数调用结果


def migcdex(*x: int) -> tuple[tuple[int, ...], int]:
    r"""Compute extended gcd for multiple integers.

    Explanation
    ===========

    Given the integers $x_1, \cdots, x_n$ and
    an extended gcd for multiple arguments are defined as a solution
    $(y_1, \cdots, y_n), g$ for the diophantine equation
    $x_1 y_1 + \cdots + x_n y_n = g$ such that
    $g = \gcd(x_1, \cdots, x_n)$.

    Examples
    ========

    >>> from sympy.functions.elementary._trigonometric_special import migcdex
    >>> migcdex()
    ((), 0)
    >>> migcdex(4)
    ((1,), 4)
    >>> migcdex(4, 6)
    ((-1, 1), 2)
    >>> migcdex(6, 10, 15)
    ((1, 1, -1), 1)
    """
    if not x:
        return (), 0
    # 若参数为空，则返回空元组和 0

    if len(x) == 1:
        return (1,), x[0]
    # 若参数个数为 1，则返回 (1,) 和该参数值

    if len(x) == 2:
        u, v, h = igcdex(x[0], x[1])
        return (u, v), h
    # 若参数个数为 2，则调用 igcdex 函数计算最大公约数和扩展最大公约数

    y, g = migcdex(*x[1:])
    u, v, h = igcdex(x[0], g)
    return (u, *(v * i for i in y)), h
# 对多个整数计算扩展最大公约数的函数定义

def ipartfrac(*denoms: int) -> tuple[int, ...]:
    r"""Compute the partial fraction decomposition.

    Explanation
    ===========

    Given a rational number $\frac{1}{q_1 \cdots q_n}$ where all
    $q_1, \cdots, q_n$ are pairwise coprime,

    A partial fraction decomposition is defined as

    .. math::
        \frac{1}{q_1 \cdots q_n} = \frac{p_1}{q_1} + \cdots + \frac{p_n}{q_n}

    And it can be derived from solving the following diophantine equation for
    the $p_1, \cdots, p_n$

    .. math::
        1 = p_1 \prod_{i \ne 1}q_i + \cdots + p_n \prod_{i \ne n}q_i

    Where $q_1, \cdots, q_n$ being pairwise coprime implies
    $\gcd(\prod_{i \ne 1}q_i, \cdots, \prod_{i \ne n}q_i) = 1$,
    """
    pass
# 计算部分分解的函数定义，但未实现具体代码
    # 如果输入的denoms为空集，则返回空元组，因为无法进行分数分解
    if not denoms:
        return ()

    # 定义一个辅助函数mul，用于计算两个整数的乘积
    def mul(x: int, y: int) -> int:
        return x * y

    # 使用reduce函数计算denoms中所有元素的乘积，得到分母denom
    denom = reduce(mul, denoms)
    
    # 计算列表a，其中每个元素是denom除以denoms中对应元素的商
    a = [denom // x for x in denoms]
    
    # 调用migcdex函数，获取其返回值中的第一个元素h，这是分数分解的结果
    h, _ = migcdex(*a)
    
    # 返回分数分解的结果h
    return h
# 定义一个函数，用于判断整数 n 是否可以分解为 Fermat 素数，每个素数的重数为 1
def fermat_coords(n: int) -> list[int] | None:
    """If n can be factored in terms of Fermat primes with
    multiplicity of each being 1, return those primes, else
    None
    """
    primes = []  # 初始化一个空列表，用于存储符合条件的素数
    for p in [3, 5, 17, 257, 65537]:  # 遍历 Fermat 素数列表
        quotient, remainder = divmod(n, p)  # 计算 n 除以当前素数 p 的商和余数
        if remainder == 0:  # 如果余数为 0，说明 n 可以整除当前素数 p
            n = quotient  # 更新 n 的值为商
            primes.append(p)  # 将当前素数 p 添加到素数列表中
            if n == 1:  # 如果 n 变为 1，表示成功分解为符合条件的素数
                return primes  # 返回素数列表
    return None  # 如果无法分解为符合条件的素数，则返回 None


# 装饰器函数，用于缓存 cos_3 函数的计算结果
@cacheit
def cos_3() -> Expr:
    r"""Computes $\cos \frac{\pi}{3}$ in square roots"""
    return S.Half  # 返回 $\cos \frac{\pi}{3}$ 的值，此处为 1/2


# 装饰器函数，用于缓存 cos_5 函数的计算结果
@cacheit
def cos_5() -> Expr:
    r"""Computes $\cos \frac{\pi}{5}$ in square roots"""
    return (sqrt(5) + 1) / 4  # 返回 $\cos \frac{\pi}{5}$ 的值，使用平方根表达式


# 装饰器函数，用于缓存 cos_17 函数的计算结果
@cacheit
def cos_17() -> Expr:
    r"""Computes $\cos \frac{\pi}{17}$ in square roots"""
    return sqrt(
        (15 + sqrt(17)) / 32 + sqrt(2) * (sqrt(17 - sqrt(17)) +
        sqrt(sqrt(2) * (-8 * sqrt(17 + sqrt(17)) - (1 - sqrt(17))
        * sqrt(17 - sqrt(17))) + 6 * sqrt(17) + 34)) / 32)
    # 返回 $\cos \frac{\pi}{17}$ 的值，使用复杂的平方根表达式计算


# 装饰器函数，用于缓存 cos_257 函数的计算结果
@cacheit
def cos_257() -> Expr:
    r"""Computes $\cos \frac{\pi}{257}$ in square roots

    References
    ==========

    .. [*] https://math.stackexchange.com/questions/516142/how-does-cos2-pi-257-look-like-in-real-radicals
    .. [*] https://r-knott.surrey.ac.uk/Fibonacci/simpleTrig.html
    """
    def f1(a: Expr, b: Expr) -> tuple[Expr, Expr]:
        return (a + sqrt(a**2 + b)) / 2, (a - sqrt(a**2 + b)) / 2

    def f2(a: Expr, b: Expr) -> Expr:
        return (a - sqrt(a**2 + b))/2

    t1, t2 = f1(S.NegativeOne, Integer(256))
    z1, z3 = f1(t1, Integer(64))
    z2, z4 = f1(t2, Integer(64))
    y1, y5 = f1(z1, 4*(5 + t1 + 2*z1))
    y6, y2 = f1(z2, 4*(5 + t2 + 2*z2))
    y3, y7 = f1(z3, 4*(5 + t1 + 2*z3))
    y8, y4 = f1(z4, 4*(5 + t2 + 2*z4))
    x1, x9 = f1(y1, -4*(t1 + y1 + y3 + 2*y6))
    x2, x10 = f1(y2, -4*(t2 + y2 + y4 + 2*y7))
    x3, x11 = f1(y3, -4*(t1 + y3 + y5 + 2*y8))
    x4, x12 = f1(y4, -4*(t2 + y4 + y6 + 2*y1))
    x5, x13 = f1(y5, -4*(t1 + y5 + y7 + 2*y2))
    x6, x14 = f1(y6, -4*(t2 + y6 + y8 + 2*y3))
    x15, x7 = f1(y7, -4*(t1 + y7 + y1 + 2*y4))
    x8, x16 = f1(y8, -4*(t2 + y8 + y2 + 2*y5))
    v1 = f2(x1, -4*(x1 + x2 + x3 + x6))
    v2 = f2(x2, -4*(x2 + x3 + x4 + x7))
    v3 = f2(x8, -4*(x8 + x9 + x10 + x13))
    v4 = f2(x9, -4*(x9 + x10 + x11 + x14))
    v5 = f2(x10, -4*(x10 + x11 + x12 + x15))
    v6 = f2(x16, -4*(x16 + x1 + x2 + x5))
    u1 = -f2(-v1, -4*(v2 + v3))
    u2 = -f2(-v4, -4*(v5 + v6))
    w1 = -2*f2(-u1, -4*u2)
    return sqrt(sqrt(2)*sqrt(w1 + 4)/8 + S.Half)
    # 返回 $\cos \frac{\pi}{257}$ 的值，使用复杂的平方根表达式计算


# 返回一个字典，包含计算 $\cos \frac{\pi}{n}$ 的函数，其中 n 是指定的 Fermat 素数
def cos_table() -> dict[int, Callable[[], Expr]]:
    r"""Lazily evaluated table for $\cos \frac{\pi}{n}$ in square roots for
    $n \in \{3, 5, 17, 257, 65537\}$.

    Notes
    =====

    65537 is the only other known Fermat prime and it is nearly impossible to
    build in the current SymPy due to performance issues.

    References
    ==========

    https://r-knott.surrey.ac.uk/Fibonacci/simpleTrig.html
    """
    return {
        3: cos_3,   # 返回计算 $\cos \frac{\pi}{3}$ 的函数 cos_3
        5: cos_5,   # 返回计算 $\cos \frac{\pi}{5}$ 的函数 cos_5
        17: cos_17,  # 返回计算 $\cos \frac{\pi}{17}$ 的函数 cos_17
        257: cos_257  # 返回计算 $\cos \frac{\pi}{257}$ 的函数 cos_257
    }
    }



    }
```