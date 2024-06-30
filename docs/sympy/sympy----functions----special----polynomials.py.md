# `D:\src\scipysrc\sympy\sympy\functions\special\polynomials.py`

```
"""
This module mainly implements special orthogonal polynomials.

See also functions.combinatorial.numbers which contains some
combinatorial polynomials.
"""

# 从 sympy 库中导入所需的模块和函数
from sympy.core import Rational
from sympy.core.function import Function, ArgumentIndexError
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.functions.combinatorial.factorials import binomial, factorial, RisingFactorial
from sympy.functions.elementary.complexes import re
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import cos, sec
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import hyper
from sympy.polys.orthopolys import (chebyshevt_poly, chebyshevu_poly,
                                    gegenbauer_poly, hermite_poly, hermite_prob_poly,
                                    jacobi_poly, laguerre_poly, legendre_poly)

# 创建一个虚拟符号 _x
_x = Dummy('x')

# 定义一个基类 OrthogonalPolynomial，继承自 Function 类
class OrthogonalPolynomial(Function):
    """Base class for orthogonal polynomials.
    """

    @classmethod
    def _eval_at_order(cls, n, x):
        # 如果 n 是整数且大于等于 0，则返回对应阶数的正交多项式在 x 处的值
        if n.is_integer and n >= 0:
            return cls._ortho_poly(int(n), _x).subs(_x, x)

    # 返回正交多项式的共轭
    def _eval_conjugate(self):
        return self.func(self.args[0], self.args[1].conjugate())

#----------------------------------------------------------------------------
# Jacobi polynomials
#

# 定义 Jacobi 类，继承自 OrthogonalPolynomial 基类
class jacobi(OrthogonalPolynomial):
    r"""
    Jacobi polynomial $P_n^{\left(\alpha, \beta\right)}(x)$.

    Explanation
    ===========

    ``jacobi(n, alpha, beta, x)`` gives the $n$th Jacobi polynomial
    in $x$, $P_n^{\left(\alpha, \beta\right)}(x)$.

    The Jacobi polynomials are orthogonal on $[-1, 1]$ with respect
    to the weight $\left(1-x\right)^\alpha \left(1+x\right)^\beta$.

    Examples
    ========

    >>> from sympy import jacobi, S, conjugate, diff
    >>> from sympy.abc import a, b, n, x

    >>> jacobi(0, a, b, x)
    1
    >>> jacobi(1, a, b, x)
    a/2 - b/2 + x*(a/2 + b/2 + 1)
    >>> jacobi(2, a, b, x)
    a**2/8 - a*b/4 - a/8 + b**2/8 - b/8 + x**2*(a**2/8 + a*b/4 + 7*a/8 + b**2/8 + 7*b/8 + 3/2) + x*(a**2/4 + 3*a/4 - b**2/4 - 3*b/4) - 1/2

    >>> jacobi(n, a, b, x)
    jacobi(n, a, b, x)

    >>> jacobi(n, a, a, x)
    RisingFactorial(a + 1, n)*gegenbauer(n,
        a + 1/2, x)/RisingFactorial(2*a + 1, n)

    >>> jacobi(n, 0, 0, x)
    legendre(n, x)

    >>> jacobi(n, S(1)/2, S(1)/2, x)
    RisingFactorial(3/2, n)*chebyshevu(n, x)/factorial(n + 1)

    >>> jacobi(n, -S(1)/2, -S(1)/2, x)
    RisingFactorial(1/2, n)*chebyshevt(n, x)/factorial(n)

    >>> jacobi(n, a, b, -x)
    (-1)**n*jacobi(n, b, a, x)

    >>> jacobi(n, a, b, 0)
    gamma(a + n + 1)*hyper((-n, -b - n), (a + 1,), -1)/(2**n*factorial(n)*gamma(a + 1))
    >>> jacobi(n, a, b, 1)
    RisingFactorial(a + 1, n)/factorial(n)

    >>> conjugate(jacobi(n, a, b, x))

    """

    # 该类实现 Jacobi 多项式的计算和特性描述
    def __init__(self, n, alpha, beta, x):
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.x = x

    # 定义 Jacobi 多项式的计算方法
    def jacobi_poly(self, n, alpha, beta, x):
        return jacobi_poly(n, alpha, beta, x)

    # 通过改变参数 x 的符号来得到 Jacobi 多项式的性质
    def jacobi_neg_x(self, n, alpha, beta, x):
        return (-1)**n * jacobi(n, beta, alpha, x)

    # 计算 Jacobi 多项式在 x = 0 和 x = 1 处的值
    def jacobi_x_0_1(self, n, alpha, beta, x):
        return gamma(alpha + n + 1) * hyper((-n, -beta - n), (alpha + 1,), -1) / (2 ** n * factorial(n) * gamma(alpha + 1))
        return RisingFactorial(alpha + 1, n) / factorial(n)

    # 求 Jacobi 多项式的复共轭
    def conjugate_jacobi(self, n, alpha, beta, x):
        return conjugate(jacobi(n, alpha, beta, x))
    # 调用 Jacobi 多项式函数，计算参数为 n, conjugate(a), conjugate(b), conjugate(x) 的值
    jacobi(n, conjugate(a), conjugate(b), conjugate(x))

    # 对 Jacobi 多项式函数关于 x 的求导
    >>> diff(jacobi(n,a,b,x), x)
    # 返回 Jacobi 多项式函数关于 x 的导数结果
    (a/2 + b/2 + n/2 + 1/2)*jacobi(n - 1, a + 1, b + 1, x)

    # 查看相关函数
    # 查看 Jacobi 多项式的相关函数，包括 Gegenbauer、Chebyshev、Legendre 等
    See Also
    ========

    gegenbauer,
    chebyshevt_root, chebyshevu, chebyshevu_root,
    legendre, assoc_legendre,
    hermite, hermite_prob,
    laguerre, assoc_laguerre,
    sympy.polys.orthopolys.jacobi_poly,
    sympy.polys.orthopolys.gegenbauer_poly
    sympy.polys.orthopolys.chebyshevt_poly
    sympy.polys.orthopolys.chebyshevu_poly
    sympy.polys.orthopolys.hermite_poly
    sympy.polys.orthopolys.legendre_poly
    sympy.polys.orthopolys.laguerre_poly

    # 参考文献
    References
    ==========

    # 参考文献列表，提供关于 Jacobi 多项式的进一步阅读资源
    .. [1] https://en.wikipedia.org/wiki/Jacobi_polynomials
    .. [2] https://mathworld.wolfram.com/JacobiPolynomial.html
    .. [3] https://functions.wolfram.com/Polynomials/JacobiP/

    """

    @classmethod
    def eval(cls, n, a, b, x):
        # 简化为其他多项式
        # P^{a, a}_n(x)
        if a == b:
            # 如果 a 等于 b
            if a == Rational(-1, 2):
                # 如果 a 是 -1/2
                return RisingFactorial(S.Half, n) / factorial(n) * chebyshevt(n, x)
            elif a.is_zero:
                # 如果 a 是 0
                return legendre(n, x)
            elif a == S.Half:
                # 如果 a 是 1/2
                return RisingFactorial(3*S.Half, n) / factorial(n + 1) * chebyshevu(n, x)
            else:
                # 对应其他情况
                return RisingFactorial(a + 1, n) / RisingFactorial(2*a + 1, n) * gegenbauer(n, a + S.Half, x)
        elif b == -a:
            # 如果 b 是 -a
            # P^{a, -a}_n(x)
            return gamma(n + a + 1) / gamma(n + 1) * (1 + x)**(a/2) / (1 - x)**(a/2) * assoc_legendre(n, -a, x)

        # 如果 n 不是数值
        if not n.is_Number:
            # 符号结果 P^{a,b}_n(x)
            # P^{a,b}_n(-x)  --->  (-1)**n * P^{b,a}_n(-x)
            if x.could_extract_minus_sign():
                return S.NegativeOne**n * jacobi(n, b, a, -x)
            # 可以在一些特定值的情况下进行求值
            if x.is_zero:
                return (2**(-n) * gamma(a + n + 1) / (gamma(a + 1) * factorial(n)) *
                        hyper([-b - n, -n], [a + 1], -1))
            if x == S.One:
                return RisingFactorial(a + 1, n) / factorial(n)
            elif x is S.Infinity:
                # 如果 x 是无穷大且 n 是正数
                # 确保 a+b+2*n 不是整数
                if (a + b + 2*n).is_integer:
                    raise ValueError("Error. a + b + 2*n should not be an integer.")
                return RisingFactorial(a + b + n + 1, n) * S.Infinity
        else:
            # n 是给定的固定整数，转换为多项式进行求解
            return jacobi_poly(n, a, b, x)
    def fdiff(self, argindex=4):
        # 导入 SymPy 中的 Sum 类
        from sympy.concrete.summations import Sum
        # 根据参数 argindex 的不同情况进行处理
        if argindex == 1:
            # 如果 argindex 为 1，则抛出参数索引错误
            raise ArgumentIndexError(self, argindex)
        elif argindex == 2:
            # 如果 argindex 为 2，则按照 n, a, b, x 的值计算差分
            n, a, b, x = self.args
            # 创建一个虚拟变量 k
            k = Dummy("k")
            # 计算 f1 和 f2 的值
            f1 = 1 / (a + b + n + k + 1)
            f2 = ((a + b + 2*k + 1) * RisingFactorial(b + k + 1, n - k) /
                  ((n - k) * RisingFactorial(a + b + k + 1, n - k)))
            # 返回 Sum 对象，表示差分表达式的求和
            return Sum(f1 * (jacobi(n, a, b, x) + f2*jacobi(k, a, b, x)), (k, 0, n - 1))
        elif argindex == 3:
            # 如果 argindex 为 3，则按照 n, a, b, x 的值计算差分
            n, a, b, x = self.args
            # 创建一个虚拟变量 k
            k = Dummy("k")
            # 计算 f1 和 f2 的值
            f1 = 1 / (a + b + n + k + 1)
            f2 = (-1)**(n - k) * ((a + b + 2*k + 1) * RisingFactorial(a + k + 1, n - k) /
                  ((n - k) * RisingFactorial(a + b + k + 1, n - k)))
            # 返回 Sum 对象，表示差分表达式的求和
            return Sum(f1 * (jacobi(n, a, b, x) + f2*jacobi(k, a, b, x)), (k, 0, n - 1))
        elif argindex == 4:
            # 如果 argindex 为 4，则按照 n, a, b, x 的值计算差分
            n, a, b, x = self.args
            # 返回具体的差分表达式
            return S.Half * (a + b + n + 1) * jacobi(n - 1, a + 1, b + 1, x)
        else:
            # 如果 argindex 不在 1 到 4 的范围内，则抛出参数索引错误
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Sum(self, n, a, b, x, **kwargs):
        # 导入 SymPy 中的 Sum 类
        from sympy.concrete.summations import Sum
        # 确保 n 是非负整数
        if n.is_negative or n.is_integer is False:
            raise ValueError("Error: n should be a non-negative integer.")
        # 创建一个虚拟变量 k
        k = Dummy("k")
        # 构建核函数
        kern = (RisingFactorial(-n, k) * RisingFactorial(a + b + n + 1, k) * RisingFactorial(a + k + 1, n - k) /
                factorial(k) * ((1 - x)/2)**k)
        # 返回表示多项式重写的 Sum 对象
        return 1 / factorial(n) * Sum(kern, (k, 0, n))

    def _eval_rewrite_as_polynomial(self, n, a, b, x, **kwargs):
        # 此函数仅用于向后兼容，不建议使用
        return self._eval_rewrite_as_Sum(n, a, b, x, **kwargs)

    def _eval_conjugate(self):
        # 获取参数 n, a, b, x 的共轭值，并返回结果
        n, a, b, x = self.args
        return self.func(n, a.conjugate(), b.conjugate(), x.conjugate())
def jacobi_normalized(n, a, b, x):
    r"""
    Jacobi polynomial $P_n^{\left(\alpha, \beta\right)}(x)$.

    Explanation
    ===========
    
    ``jacobi_normalized(n, alpha, beta, x)`` gives the $n$th
    Jacobi polynomial in $x$, $P_n^{\left(\alpha, \beta\right)}(x)$.

    The Jacobi polynomials are orthogonal on $[-1, 1]$ with respect
    to the weight $\left(1-x\right)^\alpha \left(1+x\right)^\beta$.

    This functions returns the polynomials normilzed:

    .. math::

        \int_{-1}^{1}
          P_m^{\left(\alpha, \beta\right)}(x)
          P_n^{\left(\alpha, \beta\right)}(x)
          (1-x)^{\alpha} (1+x)^{\beta} \mathrm{d}x
        = \delta_{m,n}

    Examples
    ========

    >>> from sympy import jacobi_normalized
    >>> from sympy.abc import n,a,b,x

    >>> jacobi_normalized(n, a, b, x)
    jacobi(n, a, b, x)/sqrt(2**(a + b + 1)*gamma(a + n + 1)*gamma(b + n + 1)/((a + b + 2*n + 1)*factorial(n)*gamma(a + b + n + 1)))

    Parameters
    ==========

    n : integer degree of polynomial

    a : alpha value

    b : beta value

    x : symbol

    See Also
    ========

    gegenbauer,
    chebyshevt_root, chebyshevu, chebyshevu_root,
    legendre, assoc_legendre,
    hermite, hermite_prob,
    laguerre, assoc_laguerre,
    sympy.polys.orthopolys.jacobi_poly,
    sympy.polys.orthopolys.gegenbauer_poly
    sympy.polys.orthopolys.chebyshevt_poly
    sympy.polys.orthopolys.chebyshevu_poly
    sympy.polys.orthopolys.hermite_poly
    sympy.polys.orthopolys.legendre_poly
    sympy.polys.orthopolys.laguerre_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Jacobi_polynomials
    .. [2] https://mathworld.wolfram.com/JacobiPolynomial.html
    .. [3] https://functions.wolfram.com/Polynomials/JacobiP/

    """
    # Calculate the normalization factor for the Jacobi polynomial
    nfactor = (S(2)**(a + b + 1) * (gamma(n + a + 1) * gamma(n + b + 1))
               / (2*n + a + b + 1) / (factorial(n) * gamma(n + a + b + 1)))
    
    # Return the normalized Jacobi polynomial evaluated at x
    return jacobi(n, a, b, x) / sqrt(nfactor)


#----------------------------------------------------------------------------
# Gegenbauer polynomials
#


class gegenbauer(OrthogonalPolynomial):
    r"""
    Gegenbauer polynomial $C_n^{\left(\alpha\right)}(x)$.

    Explanation
    ===========

    ``gegenbauer(n, alpha, x)`` gives the $n$th Gegenbauer polynomial
    in $x$, $C_n^{\left(\alpha\right)}(x)$.

    The Gegenbauer polynomials are orthogonal on $[-1, 1]$ with
    respect to the weight $\left(1-x^2\right)^{\alpha-\frac{1}{2}}$.

    Examples
    ========

    >>> from sympy import gegenbauer, conjugate, diff
    >>> from sympy.abc import n,a,x
    >>> gegenbauer(0, a, x)
    1
    >>> gegenbauer(1, a, x)
    2*a*x
    >>> gegenbauer(2, a, x)
    -a + x**2*(2*a**2 + 2*a)
    >>> gegenbauer(3, a, x)
    x**3*(4*a**3/3 + 4*a**2 + 8*a/3) + x*(-2*a**2 - 2*a)

    >>> gegenbauer(n, a, x)
    gegenbauer(n, a, x)
    >>> gegenbauer(n, a, -x)
    (-1)**n*gegenbauer(n, a, x)

    >>> gegenbauer(n, a, 0)
    """

    # Gegenbauer polynomials class implementation continues...
    # 计算 Gegenbauer 多项式的系数
    2**n*sqrt(pi)*gamma(a + n/2)/(gamma(a)*gamma(1/2 - n/2)*gamma(n + 1))

    # 返回 Gegenbauer 多项式的特定值
    >>> gegenbauer(n, a, 1)

    # 返回 Gegenbauer 多项式的共轭
    >>> conjugate(gegenbauer(n, a, x))

    # 返回 Gegenbauer 多项式对 x 的偏导数
    >>> diff(gegenbauer(n, a, x), x)

    # 参见
    # ========

    # Jacobian 多项式
    jacobi,
    # Chebyshev 多项式的根
    chebyshevt_root, chebyshevu, chebyshevu_root,
    # Legendre 多项式
    legendre, assoc_legendre,
    # Hermite 多项式
    hermite, hermite_prob,
    # Laguerre 多项式
    laguerre, assoc_laguerre,
    # sympy.polys.orthopolys 中的多项式函数
    sympy.polys.orthopolys.jacobi_poly
    sympy.polys.orthopolys.gegenbauer_poly
    sympy.polys.orthopolys.chebyshevt_poly
    sympy.polys.orthopolys.chebyshevu_poly
    sympy.polys.orthopolys.hermite_poly
    sympy.polys.orthopolys.hermite_prob_poly
    sympy.polys.orthopolys.legendre_poly
    sympy.polys.orthopolys.laguerre_poly

    # 参考文献
    # ==========

    # [1] https://en.wikipedia.org/wiki/Gegenbauer_polynomials
    # [2] https://mathworld.wolfram.com/GegenbauerPolynomial.html
    # [3] https://functions.wolfram.com/Polynomials/GegenbauerC3/
    """

    @classmethod
    def eval(cls, n, a, x):
        # 对于负 n，Gegenbauer 多项式为零
        # 参见 https://functions.wolfram.com/Polynomials/GegenbauerC3/03/01/03/0012/
        if n.is_negative:
            return S.Zero

        # 对于特定的 a，返回一些特殊值
        if a == S.Half:
            return legendre(n, x)
        elif a == S.One:
            return chebyshevu(n, x)
        elif a == S.NegativeOne:
            return S.Zero

        # 如果 n 不是一个数字
        if not n.is_Number:
            # 在一般的符号提取规则之前处理这种情况
            if x == S.NegativeOne:
                if (re(a) > S.Half) == True:
                    return S.ComplexInfinity
                else:
                    return (cos(S.Pi*(a+n)) * sec(S.Pi*a) * gamma(2*a+n) /
                                (gamma(2*a) * gamma(n+1)))

            # 符号结果 C^a_n(x)
            # C^a_n(-x)  --->  (-1)**n * C^a_n(x)
            if x.could_extract_minus_sign():
                return S.NegativeOne**n * gegenbauer(n, a, -x)
            # 对一些特殊值的 x 进行评估
            if x.is_zero:
                return (2**n * sqrt(S.Pi) * gamma(a + S.Half*n) /
                        (gamma((1 - n)/2) * gamma(n + 1) * gamma(a)) )
            if x == S.One:
                return gamma(2*a + n) / (gamma(2*a) * gamma(n + 1))
            elif x is S.Infinity:
                if n.is_positive:
                    return RisingFactorial(a, n) * S.Infinity
        else:
            # n 是一个给定的固定整数，评估为多项式
            return gegenbauer_poly(n, a, x)
    def fdiff(self, argindex=3):
        # 导入 SymPy 中的 Sum 类
        from sympy.concrete.summations import Sum
        # 如果 argindex 等于 1，则抛出 ArgumentIndexError 异常
        if argindex == 1:
            raise ArgumentIndexError(self, argindex)
        # 如果 argindex 等于 2
        elif argindex == 2:
            # 对于参数 a, n, x
            n, a, x = self.args
            # 创建一个虚拟变量 k
            k = Dummy("k")
            # 计算 factor1
            factor1 = 2 * (1 + (-1)**(n - k)) * (k + a) / ((k +
                           n + 2*a) * (n - k))
            # 计算 factor2
            factor2 = 2*(k + 1) / ((k + 2*a) * (2*k + 2*a + 1)) + \
                2 / (k + n + 2*a)
            # 计算核函数 kern
            kern = factor1*gegenbauer(k, a, x) + factor2*gegenbauer(n, a, x)
            # 返回求和对象 Sum
            return Sum(kern, (k, 0, n - 1))
        # 如果 argindex 等于 3
        elif argindex == 3:
            # 对于参数 n, a, x
            n, a, x = self.args
            # 返回特定的表达式
            return 2*a*gegenbauer(n - 1, a + 1, x)
        # 如果 argindex 不在已知范围内
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Sum(self, n, a, x, **kwargs):
        # 导入 SymPy 中的 Sum 类
        from sympy.concrete.summations import Sum
        # 创建虚拟变量 k
        k = Dummy("k")
        # 计算核函数 kern
        kern = ((-1)**k * RisingFactorial(a, n - k) * (2*x)**(n - 2*k) /
                (factorial(k) * factorial(n - 2*k)))
        # 返回求和对象 Sum
        return Sum(kern, (k, 0, floor(n/2)))

    def _eval_rewrite_as_polynomial(self, n, a, x, **kwargs):
        # 此函数仅用于向后兼容性，不应使用
        return self._eval_rewrite_as_Sum(n, a, x, **kwargs)

    def _eval_conjugate(self):
        # 对于参数 n, a, x
        n, a, x = self.args
        # 返回共轭函数的实例化
        return self.func(n, a.conjugate(), x.conjugate())
#----------------------------------------------------------------------------
# Chebyshev polynomials of first and second kind
#

# 定义一个类 chebyshevt，继承自 OrthogonalPolynomial
class chebyshevt(OrthogonalPolynomial):
    r"""
    Chebyshev polynomial of the first kind, $T_n(x)$.

    Explanation
    ===========

    ``chebyshevt(n, x)`` gives the $n$th Chebyshev polynomial (of the first
    kind) in $x$, $T_n(x)$.

    The Chebyshev polynomials of the first kind are orthogonal on
    $[-1, 1]$ with respect to the weight $\frac{1}{\sqrt{1-x^2}}$.

    Examples
    ========

    >>> from sympy import chebyshevt, diff
    >>> from sympy.abc import n,x
    >>> chebyshevt(0, x)
    1
    >>> chebyshevt(1, x)
    x
    >>> chebyshevt(2, x)
    2*x**2 - 1

    >>> chebyshevt(n, x)
    chebyshevt(n, x)
    >>> chebyshevt(n, -x)
    (-1)**n*chebyshevt(n, x)
    >>> chebyshevt(-n, x)
    chebyshevt(n, x)

    >>> chebyshevt(n, 0)
    cos(pi*n/2)
    >>> chebyshevt(n, -1)
    (-1)**n

    >>> diff(chebyshevt(n, x), x)
    n*chebyshevu(n - 1, x)

    See Also
    ========

    jacobi, gegenbauer,
    chebyshevt_root, chebyshevu, chebyshevu_root,
    legendre, assoc_legendre,
    hermite, hermite_prob,
    laguerre, assoc_laguerre,
    sympy.polys.orthopolys.jacobi_poly
    sympy.polys.orthopolys.gegenbauer_poly
    sympy.polys.orthopolys.chebyshevt_poly
    sympy.polys.orthopolys.chebyshevu_poly
    sympy.polys.orthopolys.hermite_poly
    sympy.polys.orthopolys.hermite_prob_poly
    sympy.polys.orthopolys.legendre_poly
    sympy.polys.orthopolys.laguerre_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Chebyshev_polynomial
    .. [2] https://mathworld.wolfram.com/ChebyshevPolynomialoftheFirstKind.html
    .. [3] https://mathworld.wolfram.com/ChebyshevPolynomialoftheSecondKind.html
    .. [4] https://functions.wolfram.com/Polynomials/ChebyshevT/
    .. [5] https://functions.wolfram.com/Polynomials/ChebyshevU/

    """

    # 静态方法，用于定义 Chebyshev 多项式的计算
    _ortho_poly = staticmethod(chebyshevt_poly)

    # 类方法，用于计算特定阶数的 Chebyshev 多项式在给定点 x 处的值
    @classmethod
    def eval(cls, n, x):
        # 如果 n 不是一个数值
        if not n.is_Number:
            # 对于符号表达式 T_n(x)
            # 若 x 可以提取负号，返回 (-1)**n * T_n(-x)
            if x.could_extract_minus_sign():
                return S.NegativeOne**n * chebyshevt(n, -x)
            # 若 n 可以提取负号，返回 T_n(x)
            if n.could_extract_minus_sign():
                return chebyshevt(-n, x)
            # 对于特定的 x 值，可以直接计算结果
            if x.is_zero:
                return cos(S.Half * S.Pi * n)
            if x == S.One:
                return S.One
            elif x is S.Infinity:
                return S.Infinity
        else:
            # 如果 n 是一个给定的固定整数，直接计算多项式值
            if n.is_negative:
                # T_{-n}(x) == T_n(x)
                return cls._eval_at_order(-n, x)
            else:
                return cls._eval_at_order(n, x)
    # 定义一个方法 fdiff，用于计算对象的微分
    def fdiff(self, argindex=2):
        # 如果参数索引为 1，抛出参数索引错误
        if argindex == 1:
            raise ArgumentIndexError(self, argindex)
        # 如果参数索引为 2
        elif argindex == 2:
            # 解包 self.args 得到 n 和 x
            n, x = self.args
            # 返回 n 乘以 chebyshevu(n - 1, x) 的结果
            return n * chebyshevu(n - 1, x)
        else:
            # 如果参数索引不是 1 或 2，抛出参数索引错误
            raise ArgumentIndexError(self, argindex)

    # 以 Sum 形式重写对象的表达式
    def _eval_rewrite_as_Sum(self, n, x, **kwargs):
        # 导入 Sum 对象
        from sympy.concrete.summations import Sum
        # 创建一个虚拟变量 k
        k = Dummy("k")
        # 计算核函数 kern
        kern = binomial(n, 2*k) * (x**2 - 1)**k * x**(n - 2*k)
        # 返回 Sum 对象，对核函数 kern 进行求和，范围是 k 从 0 到 floor(n/2)
        return Sum(kern, (k, 0, floor(n/2)))

    # 以多项式形式重写对象的表达式
    def _eval_rewrite_as_polynomial(self, n, x, **kwargs):
        # 这个函数仅保留以保证向后兼容性，不建议使用
        return self._eval_rewrite_as_Sum(n, x, **kwargs)
# 定义一个类 `chebyshevu`，继承自 `OrthogonalPolynomial` 类
class chebyshevu(OrthogonalPolynomial):
    """
    Chebyshev polynomial of the second kind, $U_n(x)$.

    Explanation
    ===========

    `chebyshevu(n, x)` 返回 $n$ 次第二类切比雪夫多项式 $U_n(x)$。

    The Chebyshev polynomials of the second kind are orthogonal on
    $[-1, 1]$ with respect to the weight $\sqrt{1-x^2}$.

    Examples
    ========

    >>> from sympy import chebyshevu, diff
    >>> from sympy.abc import n,x
    >>> chebyshevu(0, x)
    1
    >>> chebyshevu(1, x)
    2*x
    >>> chebyshevu(2, x)
    4*x**2 - 1

    >>> chebyshevu(n, x)
    chebyshevu(n, x)
    >>> chebyshevu(n, -x)
    (-1)**n*chebyshevu(n, x)
    >>> chebyshevu(-n, x)
    -chebyshevu(n - 2, x)

    >>> chebyshevu(n, 0)
    cos(pi*n/2)
    >>> chebyshevu(n, 1)
    n + 1

    >>> diff(chebyshevu(n, x), x)
    (-x*chebyshevu(n, x) + (n + 1)*chebyshevt(n + 1, x))/(x**2 - 1)

    See Also
    ========

    jacobi, gegenbauer,
    chebyshevt, chebyshevt_root, chebyshevu_root,
    legendre, assoc_legendre,
    hermite, hermite_prob,
    laguerre, assoc_laguerre,
    sympy.polys.orthopolys.jacobi_poly
    sympy.polys.orthopolys.gegenbauer_poly
    sympy.polys.orthopolys.chebyshevt_poly
    sympy.polys.orthopolys.chebyshevu_poly
    sympy.polys.orthopolys.hermite_poly
    sympy.polys.orthopolys.hermite_prob_poly
    sympy.polys.orthopolys.legendre_poly
    sympy.polys.orthopolys.laguerre_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Chebyshev_polynomial
    .. [2] https://mathworld.wolfram.com/ChebyshevPolynomialoftheFirstKind.html
    .. [3] https://mathworld.wolfram.com/ChebyshevPolynomialoftheSecondKind.html
    .. [4] https://functions.wolfram.com/Polynomials/ChebyshevT/
    .. [5] https://functions.wolfram.com/Polynomials/ChebyshevU/

    """

    # 定义一个静态方法 `_ortho_poly`，其实现为 `chebyshevu_poly`
    _ortho_poly = staticmethod(chebyshevu_poly)

    # 定义一个类方法
    @classmethod
    def eval(cls, n, x):
        # 如果 n 不是一个数值类型
        if not n.is_Number:
            # 符号计算结果 U_n(x)
            # 如果 x 可以提取负号，则返回 (-1)**n * U_n(x)
            if x.could_extract_minus_sign():
                return S.NegativeOne**n * chebyshevu(n, -x)
            # 如果 n 可以提取负号
            if n.could_extract_minus_sign():
                # 当 n 为 -1 时，返回零
                if n == S.NegativeOne:
                    return S.Zero
                # 否则返回 -U_{n-2}(x)
                elif not (-n - 2).could_extract_minus_sign():
                    return -chebyshevu(-n - 2, x)
            # 可以针对一些特殊值的 x 进行评估
            if x.is_zero:
                return cos(S.Half * S.Pi * n)
            if x == S.One:
                return S.One + n
            elif x is S.Infinity:
                return S.Infinity
        else:
            # n 是给定的固定整数，转换为多项式求值
            if n.is_negative:
                # 当 n 为负数时，返回 -U_{n-2}(x)
                if n == S.NegativeOne:
                    return S.Zero
                else:
                    return -cls._eval_at_order(-n - 2, x)
            else:
                # 否则调用类方法 _eval_at_order(n, x)
                return cls._eval_at_order(n, x)

    def fdiff(self, argindex=2):
        # 如果参数索引为 1，抛出参数索引错误
        if argindex == 1:
            raise ArgumentIndexError(self, argindex)
        elif argindex == 2:
            # 如果参数索引为 2，计算对 x 的导数
            n, x = self.args
            return ((n + 1) * chebyshevt(n + 1, x) - x * chebyshevu(n, x)) / (x**2 - 1)
        else:
            # 其他参数索引抛出参数索引错误
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Sum(self, n, x, **kwargs):
        # 导入求和函数
        from sympy.concrete.summations import Sum
        # 定义虚拟变量 k
        k = Dummy("k")
        # 计算表达式
        kern = S.NegativeOne**k * factorial(
            n - k) * (2*x)**(n - 2*k) / (factorial(k) * factorial(n - 2*k))
        # 返回求和结果
        return Sum(kern, (k, 0, floor(n/2)))

    def _eval_rewrite_as_polynomial(self, n, x, **kwargs):
        # 此函数仅用于向后兼容，不建议使用
        return self._eval_rewrite_as_Sum(n, x, **kwargs)
# 定义一个名为 chebyshevt_root 的类，继承自 Function 类
class chebyshevt_root(Function):
    r"""
    ``chebyshev_root(n, k)`` returns the $k$th root (indexed from zero) of
    the $n$th Chebyshev polynomial of the first kind; that is, if
    $0 \le k < n$, ``chebyshevt(n, chebyshevt_root(n, k)) == 0``.

    Examples
    ========

    >>> from sympy import chebyshevt, chebyshevt_root
    >>> chebyshevt_root(3, 2)
    -sqrt(3)/2
    >>> chebyshevt(3, chebyshevt_root(3, 2))
    0

    See Also
    ========

    jacobi, gegenbauer,
    chebyshevt, chebyshevu, chebyshevu_root,
    legendre, assoc_legendre,
    hermite, hermite_prob,
    laguerre, assoc_laguerre,
    sympy.polys.orthopolys.jacobi_poly
    sympy.polys.orthopolys.gegenbauer_poly
    sympy.polys.orthopolys.chebyshevt_poly
    sympy.polys.orthopolys.chebyshevu_poly
    sympy.polys.orthopolys.hermite_poly
    sympy.polys.orthopolys.hermite_prob_poly
    sympy.polys.orthopolys.legendre_poly
    sympy.polys.orthopolys.laguerre_poly
    """

    @classmethod
    # 类方法 eval，用于计算第一类 Chebyshev 多项式的根
    def eval(cls, n, k):
        # 检查 k 和 n 的取值范围
        if not ((0 <= k) and (k < n)):
            raise ValueError("must have 0 <= k < n, "
                "got k = %s and n = %s" % (k, n))
        # 返回第一类 Chebyshev 多项式的根的余弦值
        return cos(S.Pi*(2*k + 1)/(2*n))


class chebyshevu_root(Function):
    r"""
    ``chebyshevu_root(n, k)`` returns the $k$th root (indexed from zero) of the
    $n$th Chebyshev polynomial of the second kind; that is, if $0 \le k < n$,
    ``chebyshevu(n, chebyshevu_root(n, k)) == 0``.

    Examples
    ========

    >>> from sympy import chebyshevu, chebyshevu_root
    >>> chebyshevu_root(3, 2)
    -sqrt(2)/2
    >>> chebyshevu(3, chebyshevu_root(3, 2))
    0

    See Also
    ========

    chebyshevt, chebyshevt_root, chebyshevu,
    legendre, assoc_legendre,
    hermite, hermite_prob,
    laguerre, assoc_laguerre,
    sympy.polys.orthopolys.jacobi_poly
    sympy.polys.orthopolys.gegenbauer_poly
    sympy.polys.orthopolys.chebyshevt_poly
    sympy.polys.orthopolys.chebyshevu_poly
    sympy.polys.orthopolys.hermite_poly
    sympy.polys.orthopolys.hermite_prob_poly
    sympy.polys.orthopolys.legendre_poly
    sympy.polys.orthopolys.laguerre_poly
    """

    @classmethod
    # 类方法 eval，用于计算第二类 Chebyshev 多项式的根
    def eval(cls, n, k):
        # 检查 k 和 n 的取值范围
        if not ((0 <= k) and (k < n)):
            raise ValueError("must have 0 <= k < n, "
                "got k = %s and n = %s" % (k, n))
        # 返回第二类 Chebyshev 多项式的根的余弦值
        return cos(S.Pi*(k + 1)/(n + 1))

#----------------------------------------------------------------------------
# Legendre polynomials and Associated Legendre polynomials
#

# 定义一个名为 legendre 的类，继承自 OrthogonalPolynomial 类
class legendre(OrthogonalPolynomial):
    r"""
    ``legendre(n, x)`` gives the $n$th Legendre polynomial of $x$, $P_n(x)$

    Explanation
    ===========

    The Legendre polynomials are orthogonal on $[-1, 1]$ with respect to
    the constant weight 1. They satisfy $P_n(1) = 1$ for all $n$; further,
    $P_n$ is odd for odd $n$ and even for even $n$.

    Examples
    ========

    >>> from sympy import legendre, diff
    >>> from sympy.abc import x, n
    >>> legendre(0, x)
    1
    >>> legendre(1, x)
    x
    # 定义变量 x

    >>> legendre(2, x)
    # 调用 legendre 函数计算 n=2 时的 Legendre 多项式值，其中 x 是变量
    3*x**2/2 - 1/2

    >>> legendre(n, x)
    # 返回未求解的 Legendre 多项式表达式，其中 n 和 x 是变量

    >>> diff(legendre(n,x), x)
    # 计算 Legendre 多项式在变量 x 处的导数
    n*(x*legendre(n, x) - legendre(n - 1, x))/(x**2 - 1)

    See Also
    ========

    jacobi, gegenbauer,
    chebyshevt, chebyshevt_root, chebyshevu, chebyshevu_root,
    assoc_legendre,
    hermite, hermite_prob,
    laguerre, assoc_laguerre,
    sympy.polys.orthopolys.jacobi_poly
    sympy.polys.orthopolys.gegenbauer_poly
    sympy.polys.orthopolys.chebyshevt_poly
    sympy.polys.orthopolys.chebyshevu_poly
    sympy.polys.orthopolys.hermite_poly
    sympy.polys.orthopolys.hermite_prob_poly
    sympy.polys.orthopolys.legendre_poly
    sympy.polys.orthopolys.laguerre_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Legendre_polynomial
    .. [2] https://mathworld.wolfram.com/LegendrePolynomial.html
    .. [3] https://functions.wolfram.com/Polynomials/LegendreP/
    .. [4] https://functions.wolfram.com/Polynomials/LegendreP2/

    """

    _ortho_poly = staticmethod(legendre_poly)

    @classmethod
    def eval(cls, n, x):
        if not n.is_Number:
            # Symbolic result L_n(x)
            # L_n(-x)  --->  (-1)**n * L_n(x)
            if x.could_extract_minus_sign():
                return S.NegativeOne**n * legendre(n, -x)
            # L_{-n}(x)  --->  L_{n-1}(x)
            if n.could_extract_minus_sign() and not(-n - 1).could_extract_minus_sign():
                return legendre(-n - S.One, x)
            # We can evaluate for some special values of x
            if x.is_zero:
                return sqrt(S.Pi)/(gamma(S.Half - n/2)*gamma(S.One + n/2))
            elif x == S.One:
                return S.One
            elif x is S.Infinity:
                return S.Infinity
        else:
            # n is a given fixed integer, evaluate into polynomial;
            # L_{-n}(x)  --->  L_{n-1}(x)
            if n.is_negative:
                n = -n - S.One
            return cls._eval_at_order(n, x)

    def fdiff(self, argindex=2):
        if argindex == 1:
            # Diff wrt n
            raise ArgumentIndexError(self, argindex)
        elif argindex == 2:
            # Diff wrt x
            # Find better formula, this is unsuitable for x = +/-1
            # https://www.autodiff.org/ad16/Oral/Buecker_Legendre.pdf says
            # at x = 1:
            #    n*(n + 1)/2            , m = 0
            #    oo                     , m = 1
            #    -(n-1)*n*(n+1)*(n+2)/4 , m = 2
            #    0                      , m = 3, 4, ..., n
            #
            # at x = -1
            #    (-1)**(n+1)*n*(n + 1)/2       , m = 0
            #    (-1)**n*oo                    , m = 1
            #    (-1)**n*(n-1)*n*(n+1)*(n+2)/4 , m = 2
            #    0                             , m = 3, 4, ..., n
            n, x = self.args
            return n/(x**2 - 1)*(x*legendre(n, x) - legendre(n - 1, x))
        else:
            raise ArgumentIndexError(self, argindex)
    # 将 SymPy 中的 Sum 导入，用于表示求和表达式
    def _eval_rewrite_as_Sum(self, n, x, **kwargs):
        # 导入 SymPy 中的 Dummy 符号，创建一个虚拟变量 k
        from sympy.core.symbol import Dummy
        k = Dummy("k")
        # 定义求和式的核心表达式，包括二项式系数和指数项
        kern = S.NegativeOne**k * binomial(n, k)**2 * ((1 + x) / 2)**(n - k) * ((1 - x) / 2)**k
        # 返回求和表达式，范围从 k=0 到 k=n
        return Sum(kern, (k, 0, n))

    # 这个函数仅为了向后兼容而保留，实际不应使用
    def _eval_rewrite_as_polynomial(self, n, x, **kwargs):
        # 直接调用 _eval_rewrite_as_Sum 函数来获取重写的多项式形式
        return self._eval_rewrite_as_Sum(n, x, **kwargs)
class assoc_legendre(Function):
    r"""
    ``assoc_legendre(n, m, x)`` gives $P_n^m(x)$, where $n$ and $m$ are
    the degree and order or an expression which is related to the nth
    order Legendre polynomial, $P_n(x)$ in the following manner:

    .. math::
        P_n^m(x) = (-1)^m (1 - x^2)^{\frac{m}{2}}
                   \frac{\mathrm{d}^m P_n(x)}{\mathrm{d} x^m}

    Explanation
    ===========

    Associated Legendre polynomials are orthogonal on $[-1, 1]$ with:

    - weight $= 1$            for the same $m$ and different $n$.
    - weight $= \frac{1}{1-x^2}$   for the same $n$ and different $m$.

    Examples
    ========

    >>> from sympy import assoc_legendre
    >>> from sympy.abc import x, m, n
    >>> assoc_legendre(0,0, x)
    1
    >>> assoc_legendre(1,0, x)
    x
    >>> assoc_legendre(1,1, x)
    -sqrt(1 - x**2)
    >>> assoc_legendre(n,m,x)
    assoc_legendre(n, m, x)

    See Also
    ========

    jacobi, gegenbauer,
    chebyshevt, chebyshevt_root, chebyshevu, chebyshevu_root,
    legendre,
    hermite, hermite_prob,
    laguerre, assoc_laguerre,
    sympy.polys.orthopolys.jacobi_poly
    sympy.polys.orthopolys.gegenbauer_poly
    sympy.polys.orthopolys.chebyshevt_poly
    sympy.polys.orthopolys.chebyshevu_poly
    sympy.polys.orthopolys.hermite_poly
    sympy.polys.orthopolys.hermite_prob_poly
    sympy.polys.orthopolys.legendre_poly
    sympy.polys.orthopolys.laguerre_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Associated_Legendre_polynomials
    .. [2] https://mathworld.wolfram.com/LegendrePolynomial.html
    .. [3] https://functions.wolfram.com/Polynomials/LegendreP/
    .. [4] https://functions.wolfram.com/Polynomials/LegendreP2/

    """

    @classmethod
    def _eval_at_order(cls, n, m):
        # Compute the associated Legendre polynomial of order m and degree n
        P = legendre_poly(n, _x, polys=True).diff((_x, m))
        # Return the evaluated expression for P_n^m(x)
        return S.NegativeOne**m * (1 - _x**2)**Rational(m, 2) * P.as_expr()

    @classmethod
    def eval(cls, n, m, x):
        # Handle the evaluation of associated Legendre polynomial for given n, m, x
        if m.could_extract_minus_sign():
            # Recursive relation for negative m using symmetry properties
            return S.NegativeOne**(-m) * (factorial(m + n)/factorial(n - m)) * assoc_legendre(n, -m, x)
        if m == 0:
            # Special case for m=0, return the Legendre polynomial of degree n
            return legendre(n, x)
        if x == 0:
            # Special case for x=0, calculate the associated Legendre polynomial value
            return 2**m*sqrt(S.Pi) / (gamma((1 - m - n)/2)*gamma(1 - (m - n)/2))
        if n.is_Number and m.is_Number and n.is_integer and m.is_integer:
            # Check for valid indices and evaluate the associated Legendre polynomial
            if n.is_negative:
                raise ValueError("%s : 1st index must be nonnegative integer (got %r)" % (cls, n))
            if abs(m) > n:
                raise ValueError("%s : abs('2nd index') must be <= '1st index' (got %r, %r)" % (cls, n, m))
            # Evaluate and return the associated Legendre polynomial
            return cls._eval_at_order(int(n), abs(int(m))).subs(_x, x)
    def fdiff(self, argindex=3):
        # 如果参数索引为1，抛出参数索引错误，针对 n 的差分
        if argindex == 1:
            raise ArgumentIndexError(self, argindex)
        # 如果参数索引为2，抛出参数索引错误，针对 m 的差分
        elif argindex == 2:
            raise ArgumentIndexError(self, argindex)
        # 如果参数索引为3，计算关于 x 的差分
        elif argindex == 3:
            # 获取参数 n, m, x
            n, m, x = self.args
            # 计算并返回对 x 的差分公式
            return 1/(x**2 - 1)*(x*n*assoc_legendre(n, m, x) - (m + n)*assoc_legendre(n - 1, m, x))
        else:
            # 对于其他参数索引，抛出参数索引错误
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Sum(self, n, m, x, **kwargs):
        # 导入求和函数 Sum
        from sympy.concrete.summations import Sum
        # 创建虚拟变量 k
        k = Dummy("k")
        # 定义核函数 kern
        kern = factorial(2*n - 2*k)/(2**n*factorial(n - k)*factorial(
            k)*factorial(n - 2*k - m))*S.NegativeOne**k*x**(n - m - 2*k)
        # 返回重写为求和形式的表达式
        return (1 - x**2)**(m/2) * Sum(kern, (k, 0, floor((n - m)*S.Half)))

    def _eval_rewrite_as_polynomial(self, n, m, x, **kwargs):
        # 此函数仅保留用于向后兼容性，不建议使用
        # 直接调用 _eval_rewrite_as_Sum 转换为多项式形式
        return self._eval_rewrite_as_Sum(n, m, x, **kwargs)

    def _eval_conjugate(self):
        # 获取参数 n, m, x
        n, m, x = self.args
        # 返回关于 m 和 x 共轭的对象
        return self.func(n, m.conjugate(), x.conjugate())
#----------------------------------------------------------------------------
# Hermite polynomials
#


class hermite(OrthogonalPolynomial):
    r"""
    ``hermite(n, x)`` gives the $n$th Hermite polynomial in $x$, $H_n(x)$.

    Explanation
    ===========

    The Hermite polynomials are orthogonal on $(-\infty, \infty)$
    with respect to the weight $\exp\left(-x^2\right)$.

    Examples
    ========

    >>> from sympy import hermite, diff
    >>> from sympy.abc import x, n
    >>> hermite(0, x)
    1
    >>> hermite(1, x)
    2*x
    >>> hermite(2, x)
    4*x**2 - 2
    >>> hermite(n, x)
    hermite(n, x)
    >>> diff(hermite(n,x), x)
    2*n*hermite(n - 1, x)
    >>> hermite(n, -x)
    (-1)**n*hermite(n, x)

    See Also
    ========

    jacobi, gegenbauer,
    chebyshevt, chebyshevt_root, chebyshevu, chebyshevu_root,
    legendre, assoc_legendre,
    hermite_prob,
    laguerre, assoc_laguerre,
    sympy.polys.orthopolys.jacobi_poly
    sympy.polys.orthopolys.gegenbauer_poly
    sympy.polys.orthopolys.chebyshevt_poly
    sympy.polys.orthopolys.chebyshevu_poly
    sympy.polys.orthopolys.hermite_poly
    sympy.polys.orthopolys.hermite_prob_poly
    sympy.polys.orthopolys.legendre_poly
    sympy.polys.orthopolys.laguerre_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hermite_polynomial
    .. [2] https://mathworld.wolfram.com/HermitePolynomial.html
    .. [3] https://functions.wolfram.com/Polynomials/HermiteH/

    """

    # 设置静态方法 _ortho_poly，用于指定 Hermite 多项式的计算方法
    _ortho_poly = staticmethod(hermite_poly)

    # 类方法，计算 Hermite 多项式在给定 n 和 x 下的值
    @classmethod
    def eval(cls, n, x):
        # 如果 n 不是一个数值类型
        if not n.is_Number:
            # 返回符号结果 H_n(x)
            # H_n(-x)  --->  (-1)**n * H_n(x)
            if x.could_extract_minus_sign():
                return S.NegativeOne**n * hermite(n, -x)
            # 对于一些特定的 x 值，我们可以进行计算
            if x.is_zero:
                return 2**n * sqrt(S.Pi) / gamma((S.One - n)/2)
            elif x is S.Infinity:
                return S.Infinity
        else:
            # 如果 n 是给定的固定整数，直接计算多项式值
            if n.is_negative:
                raise ValueError(
                    "The index n must be nonnegative integer (got %r)" % n)
            else:
                return cls._eval_at_order(n, x)

    # 求导函数，指定参数索引进行计算
    def fdiff(self, argindex=2):
        if argindex == 1:
            # 对 n 求导
            raise ArgumentIndexError(self, argindex)
        elif argindex == 2:
            # 对 x 求导
            n, x = self.args
            return 2*n*hermite(n - 1, x)
        else:
            raise ArgumentIndexError(self, argindex)

    # 将 Hermite 多项式重写为求和形式的私有方法
    def _eval_rewrite_as_Sum(self, n, x, **kwargs):
        from sympy.concrete.summations import Sum
        k = Dummy("k")
        kern = S.NegativeOne**k / (factorial(k)*factorial(n - 2*k)) * (2*x)**(n - 2*k)
        return factorial(n)*Sum(kern, (k, 0, floor(n/2)))
    # 用于将表达式重写为多项式形式，仅用于向后兼容性但不建议使用
    def _eval_rewrite_as_polynomial(self, n, x, **kwargs):
        # 调用另一个方法将表达式重写为求和形式
        return self._eval_rewrite_as_Sum(n, x, **kwargs)

    # 用于将表达式重写为Hermite多项式的形式
    def _eval_rewrite_as_hermite_prob(self, n, x, **kwargs):
        # 返回计算结果，使用Hermite概率多项式进行重写
        return sqrt(2)**n * hermite_prob(n, x*sqrt(2)))
class hermite_prob(OrthogonalPolynomial):
    r"""
    ``hermite_prob(n, x)`` gives the $n$th probabilist's Hermite polynomial
    in $x$, $He_n(x)$.

    Explanation
    ===========

    The probabilist's Hermite polynomials are orthogonal on $(-\infty, \infty)$
    with respect to the weight $\exp\left(-\frac{x^2}{2}\right)$. They are monic
    polynomials, related to the plain Hermite polynomials (:py:class:`~.hermite`) by

    .. math :: He_n(x) = 2^{-n/2} H_n(x/\sqrt{2})

    Examples
    ========

    >>> from sympy import hermite_prob, diff, I
    >>> from sympy.abc import x, n
    >>> hermite_prob(1, x)
    x
    >>> hermite_prob(5, x)
    x**5 - 10*x**3 + 15*x
    >>> diff(hermite_prob(n,x), x)
    n*hermite_prob(n - 1, x)
    >>> hermite_prob(n, -x)
    (-1)**n*hermite_prob(n, x)

    The sum of absolute values of coefficients of $He_n(x)$ is the number of
    matchings in the complete graph $K_n$ or telephone number, A000085 in the OEIS:

    >>> [hermite_prob(n,I) / I**n for n in range(11)]
    [1, 1, 2, 4, 10, 26, 76, 232, 764, 2620, 9496]

    See Also
    ========

    jacobi, gegenbauer,
    chebyshevt, chebyshevt_root, chebyshevu, chebyshevu_root,
    legendre, assoc_legendre,
    hermite,
    laguerre, assoc_laguerre,
    sympy.polys.orthopolys.jacobi_poly
    sympy.polys.orthopolys.gegenbauer_poly
    sympy.polys.orthopolys.chebyshevt_poly
    sympy.polys.orthopolys.chebyshevu_poly
    sympy.polys.orthopolys.hermite_poly
    sympy.polys.orthopolys.hermite_prob_poly
    sympy.polys.orthopolys.legendre_poly
    sympy.polys.orthopolys.laguerre_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hermite_polynomial
    .. [2] https://mathworld.wolfram.com/HermitePolynomial.html
    """

    # 设定_ortho_poly属性为hermite_prob_poly的静态方法
    _ortho_poly = staticmethod(hermite_prob_poly)

    @classmethod
    def eval(cls, n, x):
        # 如果n不是数字，则根据x的正负性返回相应的Hermite多项式
        if not n.is_Number:
            if x.could_extract_minus_sign():
                return S.NegativeOne**n * hermite_prob(n, -x)
            # 如果x为零，返回sqrt(Pi) / gamma((1-n) / 2)
            if x.is_zero:
                return sqrt(S.Pi) / gamma((S.One-n) / 2)
            # 如果x为正无穷，返回正无穷
            elif x is S.Infinity:
                return S.Infinity
        else:
            # 如果n为负数，引发值错误异常
            if n.is_negative:
                ValueError("n must be a nonnegative integer, not %r" % n)
            else:
                # 否则，调用_eval_at_order方法计算结果
                return cls._eval_at_order(n, x)

    def fdiff(self, argindex=2):
        # 如果argindex为2，返回Hermite多项式的导数
        if argindex == 2:
            n, x = self.args
            return n*hermite_prob(n-1, x)
        else:
            # 否则，引发参数索引错误异常
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Sum(self, n, x, **kwargs):
        # 导入Sum类
        from sympy.concrete.summations import Sum
        # 创建虚拟变量k
        k = Dummy("k")
        # 定义内核
        kern = (-S.Half)**k * x**(n-2*k) / (factorial(k) * factorial(n-2*k))
        # 返回求和表达式
        return factorial(n)*Sum(kern, (k, 0, floor(n/2)))

    def _eval_rewrite_as_polynomial(self, n, x, **kwargs):
        # 此函数仅用于向后兼容性，不建议使用
        # 直接调用_eval_rewrite_as_Sum方法返回结果
        return self._eval_rewrite_as_Sum(n, x, **kwargs)
    # 定义一个方法 `_eval_rewrite_as_hermite`，用于将表达式重写为 Hermite 多项式的形式。
    def _eval_rewrite_as_hermite(self, n, x, **kwargs):
        # 返回 Hermite 多项式的重写形式，根据公式 sqrt(2)^(-n) * H_n(x / sqrt(2))
        return sqrt(2)**(-n) * hermite(n, x/sqrt(2))
#----------------------------------------------------------------------------
# Laguerre polynomials
#

# 定义一个 Laguerre 多项式类，继承自 OrthogonalPolynomial
class laguerre(OrthogonalPolynomial):
    r"""
    Returns the $n$th Laguerre polynomial in $x$, $L_n(x)$.

    Examples
    ========

    >>> from sympy import laguerre, diff
    >>> from sympy.abc import x, n
    >>> laguerre(0, x)
    1
    >>> laguerre(1, x)
    1 - x
    >>> laguerre(2, x)
    x**2/2 - 2*x + 1
    >>> laguerre(3, x)
    -x**3/6 + 3*x**2/2 - 3*x + 1

    >>> laguerre(n, x)
    laguerre(n, x)

    >>> diff(laguerre(n, x), x)
    -assoc_laguerre(n - 1, 1, x)

    Parameters
    ==========

    n : int
        Degree of Laguerre polynomial. Must be `n \ge 0`.

    See Also
    ========

    jacobi, gegenbauer,
    chebyshevt, chebyshevt_root, chebyshevu, chebyshevu_root,
    legendre, assoc_legendre,
    hermite, hermite_prob,
    assoc_laguerre,
    sympy.polys.orthopolys.jacobi_poly
    sympy.polys.orthopolys.gegenbauer_poly
    sympy.polys.orthopolys.chebyshevt_poly
    sympy.polys.orthopolys.chebyshevu_poly
    sympy.polys.orthopolys.hermite_poly
    sympy.polys.orthopolys.hermite_prob_poly
    sympy.polys.orthopolys.legendre_poly
    sympy.polys.orthopolys.laguerre_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Laguerre_polynomial
    .. [2] https://mathworld.wolfram.com/LaguerrePolynomial.html
    .. [3] https://functions.wolfram.com/Polynomials/LaguerreL/
    .. [4] https://functions.wolfram.com/Polynomials/LaguerreL3/

    """

    # 使用静态方法 laguerre_poly 来计算 Laguerre 多项式
    _ortho_poly = staticmethod(laguerre_poly)

    # 类方法，用于计算 Laguerre 多项式在给定阶数和变量 x 的值
    @classmethod
    def eval(cls, n, x):
        if n.is_integer is False:
            raise ValueError("Error: n should be an integer.")
        if not n.is_Number:
            # 如果 n 是符号表达式，返回符号结果 L_n(x)
            # L_{n}(-x)  --->  exp(-x) * L_{-n-1}(x)
            # L_{-n}(x)  --->  exp(x) * L_{n-1}(-x)
            if n.could_extract_minus_sign() and not(-n - 1).could_extract_minus_sign():
                return exp(x)*laguerre(-n - 1, -x)
            # 对于特定的 x 值，可以进行求解
            if x.is_zero:
                return S.One
            elif x is S.NegativeInfinity:
                return S.Infinity
            elif x is S.Infinity:
                return S.NegativeOne**n * S.Infinity
        else:
            # 如果 n 是负数，返回 exp(x)*laguerre(-n - 1, -x)
            # 否则调用 _eval_at_order 方法计算
            if n.is_negative:
                return exp(x)*laguerre(-n - 1, -x)
            else:
                return cls._eval_at_order(n, x)

    # 求偏导数的方法
    def fdiff(self, argindex=2):
        if argindex == 1:
            # 对 n 求偏导数时抛出错误
            raise ArgumentIndexError(self, argindex)
        elif argindex == 2:
            # 对 x 求偏导数时，返回 -assoc_laguerre(n - 1, 1, x)
            n, x = self.args
            return -assoc_laguerre(n - 1, 1, x)
        else:
            # 其他参数索引抛出错误
            raise ArgumentIndexError(self, argindex)
    # 将表达式重写为求和形式
    def _eval_rewrite_as_Sum(self, n, x, **kwargs):
        # 导入 Sum 类从 sympy.concrete.summations 模块
        from sympy.concrete.summations import Sum
        # 确保 n 属于非负整数集合 N_0
        if n.is_negative:
            # 若 n 是负数，则使用指数函数的性质进行递归重写
            return exp(x) * self._eval_rewrite_as_Sum(-n - 1, -x, **kwargs)
        # 若 n 不是整数，则抛出 ValueError 异常
        if n.is_integer is False:
            raise ValueError("Error: n should be an integer.")
        # 创建一个虚拟符号 k
        k = Dummy("k")
        # 定义核函数，为升幂阶乘除以阶乘的平方乘以 x 的 k 次幂
        kern = RisingFactorial(-n, k) / factorial(k)**2 * x**k
        # 返回求和表达式，求和变量为 k，范围从 0 到 n
        return Sum(kern, (k, 0, n))

    # 将表达式重写为多项式形式（此函数仅用于向后兼容，不建议使用）
    def _eval_rewrite_as_polynomial(self, n, x, **kwargs):
        # 此函数仅保留用于向后兼容性，不应被使用
        return self._eval_rewrite_as_Sum(n, x, **kwargs)
class assoc_laguerre(OrthogonalPolynomial):
    r"""
    Returns the $n$th generalized Laguerre polynomial in $x$, $L_n(x)$.

    Examples
    ========

    >>> from sympy import assoc_laguerre, diff
    >>> from sympy.abc import x, n, a
    >>> assoc_laguerre(0, a, x)
    1
    >>> assoc_laguerre(1, a, x)
    a - x + 1
    >>> assoc_laguerre(2, a, x)
    a**2/2 + 3*a/2 + x**2/2 + x*(-a - 2) + 1
    >>> assoc_laguerre(3, a, x)
    a**3/6 + a**2 + 11*a/6 - x**3/6 + x**2*(a/2 + 3/2) +
        x*(-a**2/2 - 5*a/2 - 3) + 1

    >>> assoc_laguerre(n, a, 0)
    binomial(a + n, a)

    >>> assoc_laguerre(n, a, x)
    assoc_laguerre(n, a, x)

    >>> assoc_laguerre(n, 0, x)
    laguerre(n, x)

    >>> diff(assoc_laguerre(n, a, x), x)
    -assoc_laguerre(n - 1, a + 1, x)

    >>> diff(assoc_laguerre(n, a, x), a)
    Sum(assoc_laguerre(_k, a, x)/(-a + n), (_k, 0, n - 1))

    Parameters
    ==========

    n : int
        Degree of Laguerre polynomial. Must be `n \ge 0`.

    alpha : Expr
        Arbitrary expression. For ``alpha=0`` regular Laguerre
        polynomials will be generated.

    See Also
    ========

    jacobi, gegenbauer,
    chebyshevt, chebyshevt_root, chebyshevu, chebyshevu_root,
    legendre, assoc_legendre,
    hermite, hermite_prob,
    laguerre,
    sympy.polys.orthopolys.jacobi_poly
    sympy.polys.orthopolys.gegenbauer_poly
    sympy.polys.orthopolys.chebyshevt_poly
    sympy.polys.orthopolys.chebyshevu_poly
    sympy.polys.orthopolys.hermite_poly
    sympy.polys.orthopolys.hermite_prob_poly
    sympy.polys.orthopolys.legendre_poly
    sympy.polys.orthopolys.laguerre_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Laguerre_polynomial#Generalized_Laguerre_polynomials
    .. [2] https://mathworld.wolfram.com/AssociatedLaguerrePolynomial.html
    .. [3] https://functions.wolfram.com/Polynomials/LaguerreL/
    .. [4] https://functions.wolfram.com/Polynomials/LaguerreL3/

    """

    @classmethod
    def eval(cls, n, alpha, x):
        # Evaluate the nth generalized Laguerre polynomial L_n(x)
        # If alpha is zero, return the regular Laguerre polynomial
        if alpha.is_zero:
            return laguerre(n, x)

        # If n is not a number, handle special cases
        if not n.is_Number:
            # Special case for certain values of x
            if x.is_zero:
                return binomial(n + alpha, alpha)
            elif x is S.Infinity and n > 0:
                return S.NegativeOne**n * S.Infinity
            elif x is S.NegativeInfinity and n > 0:
                return S.Infinity
        else:
            # If n is a fixed integer, evaluate the polynomial
            if n.is_negative:
                raise ValueError(
                    "The index n must be nonnegative integer (got %r)" % n)
            else:
                return laguerre_poly(n, x, alpha)
    def fdiff(self, argindex=3):
        # 导数函数，根据参数索引进行不同的导数计算
        from sympy.concrete.summations import Sum
        if argindex == 1:
            # 对 n 求导数
            raise ArgumentIndexError(self, argindex)
        elif argindex == 2:
            # 对 alpha 求导数
            n, alpha, x = self.args
            k = Dummy("k")
            # 返回关于 alpha 的导数表达式
            return Sum(assoc_laguerre(k, alpha, x) / (n - alpha), (k, 0, n - 1))
        elif argindex == 3:
            # 对 x 求导数
            n, alpha, x = self.args
            # 返回关于 x 的导数表达式
            return -assoc_laguerre(n - 1, alpha + 1, x)
        else:
            # 处理非法参数索引情况
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Sum(self, n, alpha, x, **kwargs):
        from sympy.concrete.summations import Sum
        # 确保 n 是非负整数
        if n.is_negative or n.is_integer is False:
            raise ValueError("Error: n should be a non-negative integer.")
        k = Dummy("k")
        kern = RisingFactorial(-n, k) / (gamma(k + alpha + 1) * factorial(k)) * x**k
        # 返回用 Sum 表示的表达式
        return gamma(n + alpha + 1) / factorial(n) * Sum(kern, (k, 0, n))

    def _eval_rewrite_as_polynomial(self, n, alpha, x, **kwargs):
        # 此函数仅为向后兼容性而保留，不建议使用
        return self._eval_rewrite_as_Sum(n, alpha, x, **kwargs)

    def _eval_conjugate(self):
        n, alpha, x = self.args
        # 返回共轭函数的表达式
        return self.func(n, alpha.conjugate(), x.conjugate())
```