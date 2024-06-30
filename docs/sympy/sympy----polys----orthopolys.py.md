# `D:\src\scipysrc\sympy\sympy\polys\orthopolys.py`

```
"""Efficient functions for generating orthogonal polynomials."""
# 导入必要的模块和函数
from sympy.core.symbol import Dummy
from sympy.polys.densearith import (dup_mul, dup_mul_ground,
    dup_lshift, dup_sub, dup_add, dup_sub_term, dup_sub_ground, dup_sqr)
from sympy.polys.domains import ZZ, QQ
from sympy.polys.polytools import named_poly
from sympy.utilities import public

# 定义 Jacobi 多项式的低级实现
def dup_jacobi(n, a, b, K):
    """Low-level implementation of Jacobi polynomials."""
    # 如果 n 小于 1，则返回常数多项式 [1]
    if n < 1:
        return [K.one]
    # 初始化 Jacobi 多项式的前两项
    m2, m1 = [K.one], [(a+b)/K(2) + K.one, (a-b)/K(2)]
    # 递推计算 Jacobi 多项式的系数
    for i in range(2, n+1):
        den = K(i)*(a + b + i)*(a + b + K(2)*i - K(2))
        f0 = (a + b + K(2)*i - K.one) * (a*a - b*b) / (K(2)*den)
        f1 = (a + b + K(2)*i - K.one) * (a + b + K(2)*i - K(2)) * (a + b + K(2)*i) / (K(2)*den)
        f2 = (a + i - K.one)*(b + i - K.one)*(a + b + K(2)*i) / den
        p0 = dup_mul_ground(m1, f0, K)
        p1 = dup_mul_ground(dup_lshift(m1, 1, K), f1, K)
        p2 = dup_mul_ground(m2, f2, K)
        m2, m1 = m1, dup_sub(dup_add(p0, p1, K), p2, K)
    return m1

@public
def jacobi_poly(n, a, b, x=None, polys=False):
    r"""Generates the Jacobi polynomial `P_n^{(a,b)}(x)`.

    Parameters
    ==========

    n : int
        Degree of the polynomial.
    a
        Lower limit of minimal domain for the list of coefficients.
    b
        Upper limit of minimal domain for the list of coefficients.
    x : optional
    polys : bool, optional
        If True, return a Poly, otherwise (default) return an expression.
    """
    # 使用 named_poly 函数生成 Jacobi 多项式
    return named_poly(n, dup_jacobi, None, "Jacobi polynomial", (x, a, b), polys)

# 定义 Gegenbauer 多项式的低级实现
def dup_gegenbauer(n, a, K):
    """Low-level implementation of Gegenbauer polynomials."""
    # 如果 n 小于 1，则返回常数多项式 [1]
    if n < 1:
        return [K.one]
    # 初始化 Gegenbauer 多项式的前两项
    m2, m1 = [K.one], [K(2)*a, K.zero]
    # 递推计算 Gegenbauer 多项式的系数
    for i in range(2, n+1):
        p1 = dup_mul_ground(dup_lshift(m1, 1, K), K(2)*(a-K.one)/K(i) + K(2), K)
        p2 = dup_mul_ground(m2, K(2)*(a-K.one)/K(i) + K.one, K)
        m2, m1 = m1, dup_sub(p1, p2, K)
    return m1

# 定义生成 Gegenbauer 多项式的函数
def gegenbauer_poly(n, a, x=None, polys=False):
    r"""Generates the Gegenbauer polynomial `C_n^{(a)}(x)`.

    Parameters
    ==========

    n : int
        Degree of the polynomial.
    x : optional
    a
        Decides minimal domain for the list of coefficients.
    polys : bool, optional
        If True, return a Poly, otherwise (default) return an expression.
    """
    # 使用 named_poly 函数生成 Gegenbauer 多项式
    return named_poly(n, dup_gegenbauer, None, "Gegenbauer polynomial", (x, a), polys)

# 定义 Chebyshev 第一类多项式的低级实现
def dup_chebyshevt(n, K):
    """Low-level implementation of Chebyshev polynomials of the first kind."""
    # 如果 n 小于 1，则返回常数多项式 [1]
    if n < 1:
        return [K.one]
    # 当 n 较小时，直接计算递推关系更快
    if n < 64: # 该阈值作为一个启发式的策略
        return _dup_chebyshevt_rec(n, K)
    # 否则使用产品形式计算 Chebyshev 多项式
    return _dup_chebyshevt_prod(n, K)

# 使用递推关系计算 Chebyshev 第一类多项式的低级实现
def _dup_chebyshevt_rec(n, K):
    r""" Chebyshev polynomials of the first kind using recurrence.

    Explanation
    ===========
    This function computes Chebyshev polynomials of the first kind
    using the recurrence relation.
    """
    Chebyshev polynomials of the first kind are defined by the recurrence
    relation:

    .. math::
        T_0(x) &= 1\\
        T_1(x) &= x\\
        T_n(x) &= 2xT_{n-1}(x) - T_{n-2}(x)

    This function calculates the Chebyshev polynomial of the first kind using
    the above recurrence relation.

    Parameters
    ==========

    n : int
        n is a nonnegative integer.
    K : domain


    m2, m1 = [K.one], [K.one, K.zero]


    # 初始化两个列表m2和m1，分别存储多项式系数
    # 初始值为T_0(x) = 1 和 T_1(x) = x
    for _ in range(n - 1):


        # 更新m2和m1，计算T_{n}(x) = 2x * T_{n-1}(x) - T_{n-2}(x)
        m2, m1 = m1, dup_sub(dup_mul_ground(dup_lshift(m1, 1, K), K(2), K), m2, K)


    # 返回计算得到的Chebyshev多项式系数列表m1
    return m1
def _dup_chebyshevt_prod(n, K):
    r""" Chebyshev polynomials of the first kind using recursive products.

    Explanation
    ===========

    Computes Chebyshev polynomials of the first kind using

    .. math::
        T_{2n}(x) &= 2T_n^2(x) - 1\\
        T_{2n+1}(x) &= 2T_{n+1}(x)T_n(x) - x

    This is faster than ``_dup_chebyshevt_rec`` for large ``n``.

    Parameters
    ==========

    n : int
        n is a nonnegative integer.
    K : domain

    """
    # 初始化多项式系数，T_0(x) 和 T_1(x)
    m2, m1 = [K.one, K.zero], [K(2), K.zero, -K.one]
    # 从二进制表示的 n 中的第三位开始迭代处理
    for i in bin(n)[3:]:
        # 计算递推关系中的下一个多项式系数
        c = dup_sub_term(dup_mul_ground(dup_mul(m1, m2, K), K(2), K), K.one, 1, K)
        if i == '1':
            # 如果当前位为 '1'，更新 m2 和 m1
            m2, m1 = c, dup_sub_ground(dup_mul_ground(dup_sqr(m1, K), K(2), K), K.one, K)
        else:
            # 如果当前位为 '0'，更新 m2 和 m1
            m2, m1 = dup_sub_ground(dup_mul_ground(dup_sqr(m2, K), K(2), K), K.one, K), c
    # 返回计算得到的 Chebyshev 多项式系数
    return m2

def dup_chebyshevu(n, K):
    """Low-level implementation of Chebyshev polynomials of the second kind."""
    if n < 1:
        return [K.one]
    # 初始化多项式系数，U_0(x) 和 U_1(x)
    m2, m1 = [K.one], [K(2), K.zero]
    # 从 n=2 开始迭代计算到 n
    for i in range(2, n+1):
        # 计算递推关系得到下一个多项式系数
        m2, m1 = m1, dup_sub(dup_mul_ground(dup_lshift(m1, 1, K), K(2), K), m2, K)
    # 返回计算得到的 Chebyshev 多项式系数
    return m1

@public
def chebyshevt_poly(n, x=None, polys=False):
    r"""Generates the Chebyshev polynomial of the first kind `T_n(x)`.

    Parameters
    ==========

    n : int
        Degree of the polynomial.
    x : optional
    polys : bool, optional
        If True, return a Poly, otherwise (default) return an expression.
    """
    # 调用通用命名多项式生成函数，生成 Chebyshev 多项式 T_n(x) 的表达式或多项式对象
    return named_poly(n, dup_chebyshevt, ZZ,
            "Chebyshev polynomial of the first kind", (x,), polys)

@public
def chebyshevu_poly(n, x=None, polys=False):
    r"""Generates the Chebyshev polynomial of the second kind `U_n(x)`.

    Parameters
    ==========

    n : int
        Degree of the polynomial.
    x : optional
    polys : bool, optional
        If True, return a Poly, otherwise (default) return an expression.
    """
    # 调用通用命名多项式生成函数，生成 Chebyshev 多项式 U_n(x) 的表达式或多项式对象
    return named_poly(n, dup_chebyshevu, ZZ,
            "Chebyshev polynomial of the second kind", (x,), polys)

def dup_hermite(n, K):
    """Low-level implementation of Hermite polynomials."""
    if n < 1:
        return [K.one]
    # 初始化 Hermite 多项式的系数，H_0(x) 和 H_1(x)
    m2, m1 = [K.one], [K(2), K.zero]
    # 从 n=2 开始迭代计算到 n
    for i in range(2, n+1):
        # 计算递推关系得到下一个多项式系数
        a = dup_lshift(m1, 1, K)
        b = dup_mul_ground(m2, K(i-1), K)
        m2, m1 = m1, dup_mul_ground(dup_sub(a, b, K), K(2), K)
    # 返回计算得到的 Hermite 多项式系数
    return m1

def dup_hermite_prob(n, K):
    """Low-level implementation of probabilist's Hermite polynomials."""
    if n < 1:
        return [K.one]
    # 初始化 Hermite 多项式的系数，H_0(x) 和 H_1(x)
    m2, m1 = [K.one], [K.one, K.zero]
    # 从 n=2 开始迭代计算到 n
    for i in range(2, n+1):
        # 计算递推关系得到下一个多项式系数
        a = dup_lshift(m1, 1, K)
        b = dup_mul_ground(m2, K(i-1), K)
        m2, m1 = m1, dup_sub(a, b, K)
    # 返回计算得到的 Hermite 多项式系数
    return m1

@public
def hermite_poly(n, x=None, polys=False):
    r"""Generates the Hermite polynomial `H_n(x)`.

    Parameters
    ==========

    n : int
        Degree of the polynomial.
    x : optional
    # polys : bool, optional
    # 定义函数参数 polys，类型为布尔值，可选参数（optional）
    # 如果为 True，则返回一个多项式（Poly），否则（默认）返回一个表达式。
    """
    调用 named_poly 函数，生成指定名称的 Hermite 多项式。
    使用参数 n、dup_hermite、ZZ，以及指定的描述字符串 "Hermite polynomial" 和元组 (x,)。
    如果 polys 参数为 True，则返回一个多项式（Poly），否则返回一个表达式。
    """
    return named_poly(n, dup_hermite, ZZ, "Hermite polynomial", (x,), polys)
# 定义一个公共函数装饰器，使函数被声明为公共函数
@public
def hermite_prob_poly(n, x=None, polys=False):
    r"""Generates the probabilist's Hermite polynomial `He_n(x)`.

    Parameters
    ==========

    n : int
        Degree of the polynomial.
    x : optional
    polys : bool, optional
        If True, return a Poly, otherwise (default) return an expression.
    """
    # 调用命名多项式函数，生成概率埃尔米特多项式
    return named_poly(n, dup_hermite_prob, ZZ,
            "probabilist's Hermite polynomial", (x,), polys)

# 低级实现的勒让德多项式
def dup_legendre(n, K):
    """Low-level implementation of Legendre polynomials."""
    # 如果 n < 1，返回常数项 1
    if n < 1:
        return [K.one]
    # 初始化 m2 = [1], m1 = [1, 0]
    m2, m1 = [K.one], [K.one, K.zero]
    # 计算 Legendre 多项式的递推关系
    for i in range(2, n+1):
        # 计算 a = 2*i-1 的倍数乘以 m1
        a = dup_mul_ground(dup_lshift(m1, 1, K), K(2*i-1, i), K)
        # 计算 b = (i-1)/i 的倍数乘以 m2
        b = dup_mul_ground(m2, K(i-1, i), K)
        # 更新 m2, m1
        m2, m1 = m1, dup_sub(a, b, K)
    # 返回计算结果 m1，即 Legendre 多项式
    return m1

# 定义一个公共函数装饰器，使函数被声明为公共函数
@public
def legendre_poly(n, x=None, polys=False):
    r"""Generates the Legendre polynomial `P_n(x)`.

    Parameters
    ==========

    n : int
        Degree of the polynomial.
    x : optional
    polys : bool, optional
        If True, return a Poly, otherwise (default) return an expression.
    """
    # 调用命名多项式函数，生成勒让德多项式
    return named_poly(n, dup_legendre, QQ, "Legendre polynomial", (x,), polys)

# 低级实现的拉盖尔多项式
def dup_laguerre(n, alpha, K):
    """Low-level implementation of Laguerre polynomials."""
    # 初始化 m2 = [0], m1 = [1]
    m2, m1 = [K.zero], [K.one]
    # 计算 Laguerre 多项式的递推关系
    for i in range(1, n+1):
        # 计算 a = -(i-1)/i 和 (alpha-1)/i + 2 的乘积，乘以 m1
        a = dup_mul(m1, [-K.one/K(i), (alpha-K.one)/K(i) + K(2)], K)
        # 计算 b = (alpha-1)/i + 1 的倍数乘以 m2
        b = dup_mul_ground(m2, (alpha-K.one)/K(i) + K.one, K)
        # 更新 m2, m1
        m2, m1 = m1, dup_sub(a, b, K)
    # 返回计算结果 m1，即 Laguerre 多项式
    return m1

# 定义一个公共函数装饰器，使函数被声明为公共函数
@public
def laguerre_poly(n, x=None, alpha=0, polys=False):
    r"""Generates the Laguerre polynomial `L_n^{(\alpha)}(x)`.

    Parameters
    ==========

    n : int
        Degree of the polynomial.
    x : optional
    alpha : optional
        Decides minimal domain for the list of coefficients.
    polys : bool, optional
        If True, return a Poly, otherwise (default) return an expression.
    """
    # 调用命名多项式函数，生成拉盖尔多项式
    return named_poly(n, dup_laguerre, None, "Laguerre polynomial", (x, alpha), polys)

# 低级实现的球贝塞尔函数 fn(n, x)
def dup_spherical_bessel_fn(n, K):
    """Low-level implementation of fn(n, x)."""
    # 如果 n < 1，返回 [1, 0]
    if n < 1:
        return [K.one, K.zero]
    # 初始化 m2 = [1], m1 = [1, 0]
    m2, m1 = [K.one], [K.one, K.zero]
    # 计算球贝塞尔函数 fn(n, x) 的递推关系
    for i in range(2, n+1):
        # 计算 a = (2*i-1) 的倍数乘以 m1，并左移一位
        a = dup_mul_ground(dup_lshift(m1, 1, K), K(2*i-1), K)
        # 计算 b = m2，并减去 m1
        b = dup_sub(a, m2, K)
        # 更新 m2, m1
        m2, m1 = m1, b
    # 左移 m1 一位，返回结果
    return dup_lshift(m1, 1, K)

# 低级实现的球贝塞尔函数 fn(-n, x)
def dup_spherical_bessel_fn_minus(n, K):
    """Low-level implementation of fn(-n, x)."""
    # 初始化 m2 = [1, 0], m1 = [0]
    m2, m1 = [K.one, K.zero], [K.zero]
    # 计算球贝塞尔函数 fn(-n, x) 的递推关系
    for i in range(2, n+1):
        # 计算 a = (3-2*i) 的倍数乘以 m1，并左移一位
        a = dup_mul_ground(dup_lshift(m1, 1, K), K(3-2*i), K)
        # 计算 b = m2，并减去 m1
        b = dup_sub(a, m2, K)
        # 更新 m2, m1
        m2, m1 = m1, b
    # 返回结果 m1
    return m1

# 定义一个公共函数装饰器，使函数被声明为公共函数
@public
def spherical_bessel_fn(n, x=None, polys=False):
    """
    Coefficients for the spherical Bessel functions.

    These are only needed in the jn() function.

    The coefficients are calculated from:

    fn(0, z) = 1/z
    fn(1, z) = 1/z**2
    fn(n-1, z) + fn(n+1, z) == (2*n+1)/z * fn(n, z)

    Parameters
    ==========

    n : int
        Degree of the polynomial.
    x : optional
    """
    # 函数实现了球贝塞尔函数的系数计算，但并未给出具体实现代码
    # polys : bool, optional
    # 如果为True，返回一个多项式（Poly），否则（默认）返回一个表达式。

    Examples
    ========

    >>> from sympy.polys.orthopolys import spherical_bessel_fn as fn
    >>> from sympy import Symbol
    >>> z = Symbol("z")
    >>> fn(1, z)
    z**(-2)
    >>> fn(2, z)
    -1/z + 3/z**3
    >>> fn(3, z)
    -6/z**2 + 15/z**4
    >>> fn(4, z)
    1/z - 45/z**3 + 105/z**5

    """
    # 如果参数 x 为 None，则创建一个虚拟符号 x
    if x is None:
        x = Dummy("x")
    # 根据 n 的正负选择不同的函数 f
    f = dup_spherical_bessel_fn_minus if n < 0 else dup_spherical_bessel_fn
    # 调用 named_poly 函数生成一个命名多项式对象，并返回
    return named_poly(abs(n), f, ZZ, "", (QQ(1)/x,), polys)
```