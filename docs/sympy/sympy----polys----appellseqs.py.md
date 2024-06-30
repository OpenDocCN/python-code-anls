# `D:\src\scipysrc\sympy\sympy\polys\appellseqs.py`

```
"""
Efficient functions for generating Appell sequences.

An Appell sequence is a zero-indexed sequence of polynomials `p_i(x)`
satisfying `p_{i+1}'(x)=(i+1)p_i(x)` for all `i`. This definition leads
to the following iterative algorithm:

.. math :: p_0(x) = c_0,\ p_i(x) = i \int_0^x p_{i-1}(t)\,dt + c_i

The constant coefficients `c_i` are usually determined from the
just-evaluated integral and `i`.

Appell sequences satisfy the following identity from umbral calculus:

.. math :: p_n(x+y) = \sum_{k=0}^n \binom{n}{k} p_k(x) y^{n-k}

References
==========

.. [1] https://en.wikipedia.org/wiki/Appell_sequence
.. [2] Peter Luschny, "An introduction to the Bernoulli function",
       https://arxiv.org/abs/2009.06743
"""
# 导入 sympy 库中的相关模块和函数
from sympy.polys.densearith import dup_mul_ground, dup_sub_ground, dup_quo_ground
from sympy.polys.densetools import dup_eval, dup_integrate
from sympy.polys.domains import ZZ, QQ
from sympy.polys.polytools import named_poly
from sympy.utilities import public

# 定义函数 dup_bernoulli，实现 Bernoulli 多项式的低级别计算
def dup_bernoulli(n, K):
    """Low-level implementation of Bernoulli polynomials."""
    # 若 n 小于 1，返回 [K.one] 列表
    if n < 1:
        return [K.one]
    # 初始化 Bernoulli 多项式列表，包括 B_0 和 B_1
    p = [K.one, K(-1,2)]
    # 从 i=2 到 n 循环计算 Bernoulli 多项式
    for i in range(2, n+1):
        # 计算积分 i * ∫(0 to x) p_{i-1}(t) dt + c_i
        p = dup_integrate(dup_mul_ground(p, K(i), K), 1, K)
        # 若 i 是偶数，执行以下操作
        if i % 2 == 0:
            # 计算 Bernoulli 多项式的特定调整
            p = dup_sub_ground(p, dup_eval(p, K(1,2), K) * K(1<<(i-1), (1<<i)-1), K)
    # 返回计算得到的 Bernoulli 多项式列表
    return p

# 声明函数 bernoulli_poly，生成 Bernoulli 多项式 B_n(x)
@public
def bernoulli_poly(n, x=None, polys=False):
    r"""Generates the Bernoulli polynomial `\operatorname{B}_n(x)`.

    `\operatorname{B}_n(x)` is the unique polynomial satisfying

    .. math :: \int_{x}^{x+1} \operatorname{B}_n(t) \,dt = x^n.

    Based on this, we have for nonnegative integer `s` and integer
    `a` and `b`

    .. math :: \sum_{k=a}^{b} k^s = \frac{\operatorname{B}_{s+1}(b+1) -
            \operatorname{B}_{s+1}(a)}{s+1}

    which is related to Jakob Bernoulli's original motivation for introducing
    the Bernoulli numbers, the values of these polynomials at `x = 1`.

    Examples
    ========

    >>> from sympy import summation
    >>> from sympy.abc import x
    >>> from sympy.polys import bernoulli_poly
    >>> bernoulli_poly(5, x)
    x**5 - 5*x**4/2 + 5*x**3/3 - x/6

    >>> def psum(p, a, b):
    ...     return (bernoulli_poly(p+1,b+1) - bernoulli_poly(p+1,a)) / (p+1)
    >>> psum(4, -6, 27)
    3144337
    >>> summation(x**4, (x, -6, 27))
    3144337

    >>> psum(1, 1, x).factor()
    x*(x + 1)/2
    >>> psum(2, 1, x).factor()
    x*(x + 1)*(2*x + 1)/6
    >>> psum(3, 1, x).factor()
    x**2*(x + 1)**2/4

    Parameters
    ==========

    n : int
        Degree of the polynomial.
    x : optional
    polys : bool, optional
        If True, return a Poly, otherwise (default) return an expression.

    See Also
    ========

    sympy.functions.combinatorial.numbers.bernoulli

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Bernoulli_polynomials
    """
    # 调用 named_poly 函数生成 Bernoulli 多项式
    return named_poly(n, dup_bernoulli, QQ, "Bernoulli polynomial", (x,), polys)

# 声明函数 dup_bernoulli_c，未完整的实现，需进一步完善
def dup_bernoulli_c(n, K):
    """Low-level implementation of central Bernoulli polynomials."""
    # 初始化多项式 p，起始值为 K.one（可能是多项式库中的单位元素）
    p = [K.one]
    # 循环计算中心伯努利多项式，从 i = 1 到 n
    for i in range(1, n+1):
        # 对 p 进行积分操作，将当前 p 与 i 相乘后积分，并更新 p
        p = dup_integrate(dup_mul_ground(p, K(i), K), 1, K)
        # 如果 i 是偶数，则执行如下操作
        if i % 2 == 0:
            # 计算 p 在点 K.one 处的值，然后用其乘以 K((1<<(i-1))-1, (1<<i)-1) 的结果从 p 中减去，并更新 p
            p = dup_sub_ground(p, dup_eval(p, K.one, K) * K((1<<(i-1))-1, (1<<i)-1), K)
    # 返回计算结果多项式 p
    return p
# 定义一个公共函数，生成中心伯努利多项式 `\operatorname{B}_n^c(x)`。
# 这些多项式是普通伯努利多项式的缩放和移位版本，使得对于偶数或奇数的 `n`，`\operatorname{B}_n^c(x)`分别是偶函数或奇函数。
# 其定义如下：
# .. math :: \operatorname{B}_n^c(x) = 2^n \operatorname{B}_n \left(\frac{x+1}{2}\right)
@public
def bernoulli_c_poly(n, x=None, polys=False):
    return named_poly(n, dup_bernoulli_c, QQ, "central Bernoulli polynomial", (x,), polys)

# 生成 Genocchi 多项式 `\operatorname{G}_n(x)` 的函数。
# `\operatorname{G}_n(x)` 是普通和中心伯努利多项式之间的差的两倍，因此其次数为 `n-1`。
# 这个多项式的定义中整数系数。
# 可选参数 `polys` 控制是否返回 Poly 对象，默认返回表达式。
@public
def genocchi_poly(n, x=None, polys=False):
    return named_poly(n, dup_genocchi, ZZ, "Genocchi polynomial", (x,), polys)

# 生成 Euler 多项式 `\operatorname{E}_n(x)` 的函数。
# 这些多项式是 Genocchi 多项式的缩放和重新索引版本。
# 其定义如下：
# .. math :: \operatorname{E}_n(x) = -\frac{\operatorname{G}_{n+1}(x)}{n+1}
# 可选参数 `polys` 控制是否返回 Poly 对象，默认返回表达式。
@public
def euler_poly(n, x=None, polys=False):
    return named_poly(n, dup_euler, QQ, "Euler polynomial", (x,), polys)
    r"""Generates the Andre polynomial `\mathcal{A}_n(x)`.
    
    This is the Appell sequence where the constant coefficients form the sequence
    of Euler numbers ``euler(n)``. As such they have integer coefficients
    and parities matching the parity of `n`.
    
    Luschny calls these the *Swiss-knife polynomials* because their values
    at 0 and 1 can be simply transformed into both the Bernoulli and Euler
    numbers. Here they are called the Andre polynomials because
    `|\mathcal{A}_n(n\bmod 2)|` for `n \ge 0` generates what Luschny calls
    the *Andre numbers*, A000111 in the OEIS.
    
    Examples
    ========
    
    >>> from sympy import bernoulli, euler, genocchi
    >>> from sympy.abc import x
    >>> from sympy.polys import andre_poly
    >>> andre_poly(9, x)
    x**9 - 36*x**7 + 630*x**5 - 5124*x**3 + 12465*x
    
    >>> [andre_poly(n, 0) for n in range(11)]
    [1, 0, -1, 0, 5, 0, -61, 0, 1385, 0, -50521]
    >>> [euler(n) for n in range(11)]
    [1, 0, -1, 0, 5, 0, -61, 0, 1385, 0, -50521]
    >>> [andre_poly(n-1, 1) * n / (4**n - 2**n) for n in range(1, 11)]
    [1/2, 1/6, 0, -1/30, 0, 1/42, 0, -1/30, 0, 5/66]
    >>> [bernoulli(n) for n in range(1, 11)]
    [1/2, 1/6, 0, -1/30, 0, 1/42, 0, -1/30, 0, 5/66]
    >>> [-andre_poly(n-1, -1) * n / (-2)**(n-1) for n in range(1, 11)]
    [-1, -1, 0, 1, 0, -3, 0, 17, 0, -155]
    >>> [genocchi(n) for n in range(1, 11)]
    [-1, -1, 0, 1, 0, -3, 0, 17, 0, -155]
    
    >>> [abs(andre_poly(n, n%2)) for n in range(11)]
    [1, 1, 1, 2, 5, 16, 61, 272, 1385, 7936, 50521]
    
    Parameters
    ==========
    
    n : int
        Degree of the polynomial.
    x : optional
    polys : bool, optional
        If True, return a Poly, otherwise (default) return an expression.
    
    See Also
    ========
    
    sympy.functions.combinatorial.numbers.andre
    
    References
    ==========
    
    .. [1] Peter Luschny, "An introduction to the Bernoulli function",
           https://arxiv.org/abs/2009.06743
    """
    # 调用 named_poly 函数生成 Andre 多项式，并返回结果
    return named_poly(n, dup_andre, ZZ, "Andre polynomial", (x,), polys)
```