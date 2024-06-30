# `D:\src\scipysrc\sympy\sympy\polys\polyroots.py`

```
"""Algorithms for computing symbolic roots of polynomials. """


import math
from functools import reduce

from sympy.core import S, I, pi
from sympy.core.exprtools import factor_terms
from sympy.core.function import _mexpand
from sympy.core.logic import fuzzy_not
from sympy.core.mul import expand_2arg, Mul
from sympy.core.intfunc import igcd
from sympy.core.numbers import Rational, comp
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy, Symbol, symbols
from sympy.core.sympify import sympify
from sympy.functions import exp, im, cos, acos, Piecewise
from sympy.functions.elementary.miscellaneous import root, sqrt
from sympy.ntheory import divisors, isprime, nextprime
from sympy.polys.domains import EX
from sympy.polys.polyerrors import (PolynomialError, GeneratorsNeeded,
    DomainError, UnsolvableFactorError)
from sympy.polys.polyquinticconst import PolyQuintic
from sympy.polys.polytools import Poly, cancel, factor, gcd_list, discriminant
from sympy.polys.rationaltools import together
from sympy.polys.specialpolys import cyclotomic_poly
from sympy.utilities import public
from sympy.utilities.misc import filldedent



z = Symbol('z')  # 定义符号 z


def roots_linear(f):
    """Returns a list of roots of a linear polynomial."""
    r = -f.nth(0)/f.nth(1)  # 计算线性多项式的根
    dom = f.get_domain()  # 获取多项式的定义域

    if not dom.is_Numerical:
        if dom.is_Composite:
            r = factor(r)  # 对复合定义域进行因式分解
        else:
            from sympy.simplify.simplify import simplify
            r = simplify(r)  # 简化根表达式

    return [r]  # 返回根的列表


def roots_quadratic(f):
    """Returns a list of roots of a quadratic polynomial. If the domain is ZZ
    then the roots will be sorted with negatives coming before positives.
    The ordering will be the same for any numerical coefficients as long as
    the assumptions tested are correct, otherwise the ordering will not be
    sorted (but will be canonical).
    """

    a, b, c = f.all_coeffs()  # 获取二次多项式的所有系数
    dom = f.get_domain()  # 获取多项式的定义域

    def _sqrt(d):
        # 从平方根中去除平方项，因为结果中会包含两种形式；类似的过程在 roots() 中已处理，
        # 但必须在这里重复，因为不是所有的二次多项式都是二项式
        co = []
        other = []
        for di in Mul.make_args(d):
            if di.is_Pow and di.exp.is_Integer and di.exp % 2 == 0:
                co.append(Pow(di.base, di.exp//2))
            else:
                other.append(di)
        if co:
            d = Mul(*other)
            co = Mul(*co)
            return co*sqrt(d)
        return sqrt(d)

    def _simplify(expr):
        if dom.is_Composite:
            return factor(expr)  # 对复合定义域进行因式分解
        else:
            from sympy.simplify.simplify import simplify
            return simplify(expr)  # 简化表达式
    # 如果常数项 c 是零
    if c is S.Zero:
        # 计算两个根的情况：r0 = 0, r1 = -b/a
        r0, r1 = S.Zero, -b/a

        # 如果定义域不是数值类型
        if not dom.is_Numerical:
            # 对 r1 进行简化处理
            r1 = _simplify(r1)
        # 如果 r1 是负数
        elif r1.is_negative:
            # 调整 r0 和 r1 的顺序
            r0, r1 = r1, r0

    # 如果一次项 b 是零
    elif b is S.Zero:
        # 计算单根的情况：r = -c/a
        r = -c/a
        # 如果定义域不是数值类型
        if not dom.is_Numerical:
            # 对 r 进行简化处理
            r = _simplify(r)

        # 计算 r 的平方根
        R = _sqrt(r)
        # 计算两个根：r0 = -R, r1 = R
        r0 = -R
        r1 = R

    # 如果常数项和一次项均不为零
    else:
        # 计算判别式 d = b^2 - 4ac
        d = b**2 - 4*a*c
        # 计算系数 A = 2a
        A = 2*a
        # 计算系数 B = -b/A
        B = -b/A

        # 如果定义域不是数值类型
        if not dom.is_Numerical:
            # 对 d 和 B 进行简化处理
            d = _simplify(d)
            B = _simplify(B)

        # 计算 D = sqrt(d) / A，并进行因式分解
        D = factor_terms(_sqrt(d)/A)
        # 计算两个根：r0 = B - D, r1 = B + D
        r0 = B - D
        r1 = B + D
        # 如果二次项 a 是负数，则交换 r0 和 r1 的顺序
        if a.is_negative:
            r0, r1 = r1, r0
        # 如果定义域不是数值类型
        elif not dom.is_Numerical:
            # 对 r0 和 r1 进行扩展处理
            r0, r1 = [expand_2arg(i) for i in (r0, r1)]

    # 返回两个根组成的列表
    return [r0, r1]
def roots_cubic(f, trig=False):
    """Returns a list of roots of a cubic polynomial.

    References
    ==========
    [1] https://en.wikipedia.org/wiki/Cubic_function, General formula for roots,
    (accessed November 17, 2014).
    """
    # 如果 trig 为真，则使用三角函数解法
    if trig:
        # 获取多项式的系数 a, b, c, d
        a, b, c, d = f.all_coeffs()
        # 计算 p 和 q
        p = (3*a*c - b**2)/(3*a**2)
        q = (2*b**3 - 9*a*b*c + 27*a**2*d)/(27*a**3)
        # 计算判别式 D
        D = 18*a*b*c*d - 4*b**3*d + b**2*c**2 - 4*a*c**3 - 27*a**2*d**2
        # 如果 D 大于 0，则使用三角函数计算根
        if (D > 0) == True:
            rv = []
            for k in range(3):
                # 计算每个根的实部
                rv.append(2*sqrt(-p/3)*cos(acos(q/p*sqrt(-3/p)*Rational(3, 2))/3 - k*pi*Rational(2, 3)))
            # 返回三个根减去 b/(3*a) 的结果
            return [i - b/3/a for i in rv]

    # 将多项式转化为以 x**3 开头的形式
    _, a, b, c = f.monic().all_coeffs()

    # 如果常数项 c 是零
    if c is S.Zero:
        # 求解简化的方程
        x1, x2 = roots([1, a, b], multiple=True)
        return [x1, S.Zero, x2]

    # 将多项式转化为 u**3 + p*u + q 的形式
    p = b - a**2/3
    q = c - a*b/3 + 2*a**3/27

    pon3 = p/3
    aon3 = a/3

    u1 = None
    if p is S.Zero:
        if q is S.Zero:
            return [-aon3]*3
        # 求解 u1
        u1 = -root(q, 3) if q.is_positive else root(-q, 3)
    elif q is S.Zero:
        # 求解 y1 和 y2
        y1, y2 = roots([1, 0, p], multiple=True)
        return [tmp - aon3 for tmp in [y1, S.Zero, y2]]
    elif q.is_real and q.is_negative:
        # 求解 u1
        u1 = -root(-q/2 + sqrt(q**2/4 + pon3**3), 3)

    coeff = I*sqrt(3)/2
    if u1 is None:
        u1 = S.One
        u2 = Rational(-1, 2) + coeff
        u3 = Rational(-1, 2) - coeff
        b, c, d = a, b, c  # a, b, c, d = S.One, a, b, c
        # 计算 D0 和 D1
        D0 = b**2 - 3*c  # b**2 - 3*a*c
        D1 = 2*b**3 - 9*b*c + 27*d  # 2*b**3 - 9*a*b*c + 27*a**2*d
        # 计算 C
        C = root((D1 + sqrt(D1**2 - 4*D0**3))/2, 3)
        # 返回三个根的结果
        return [-(b + uk*C + D0/C/uk)/3 for uk in [u1, u2, u3]]  # -(b + uk*C + D0/C/uk)/3/a

    u2 = u1*(Rational(-1, 2) + coeff)
    u3 = u1*(Rational(-1, 2) - coeff)

    if p is S.Zero:
        return [u1 - aon3, u2 - aon3, u3 - aon3]

    # 计算解
    soln = [
        -u1 + pon3/u1 - aon3,
        -u2 + pon3/u2 - aon3,
        -u3 + pon3/u3 - aon3
    ]

    return soln


def _roots_quartic_euler(p, q, r, a):
    """
    Descartes-Euler solution of the quartic equation

    Parameters
    ==========

    p, q, r: coefficients of ``x**4 + p*x**2 + q*x + r``
    a: shift of the roots

    Notes
    =====

    This is a helper function for ``roots_quartic``.

    Look for solutions of the form ::

      ``x1 = sqrt(R) - sqrt(A + B*sqrt(R))``
      ``x2 = -sqrt(R) - sqrt(A - B*sqrt(R))``
      ``x3 = -sqrt(R) + sqrt(A - B*sqrt(R))``
      ``x4 = sqrt(R) + sqrt(A + B*sqrt(R))``

    To satisfy the quartic equation one must have
    ``p = -2*(R + A); q = -4*B*R; r = (R - A)**2 - B**2*R``
    so that ``R`` must satisfy the Descartes-Euler resolvent equation
    ``64*R**3 + 32*p*R**2 + (4*p**2 - 16*r)*R - q**2 = 0``

    If the resolvent does not have a rational solution, return None;
    in that case it is likely that the Ferrari method gives a simpler
    solution.
    """
    # 解决四次方程的相关计算，返回四个实根之一的表达式
    x = Dummy('x')
    # 定义四次方程的解析式
    eq = 64*x**3 + 32*p*x**2 + (4*p**2 - 16*r)*x - q**2
    # 计算四次方程的根，并转换为列表形式
    xsols = list(roots(Poly(eq, x), cubics=False).keys())
    # 过滤出有理数且非零的解
    xsols = [sol for sol in xsols if sol.is_rational and sol.is_nonzero]
    # 如果没有有效的解，则返回 None
    if not xsols:
        return None
    # 选择最大的有理数解作为 R
    R = max(xsols)
    # 计算 c1
    c1 = sqrt(R)
    # 计算 B
    B = -q*c1/(4*R)
    # 计算 A
    A = -R - p/2
    # 计算 c2 和 c3
    c2 = sqrt(A + B)
    c3 = sqrt(A - B)
    # 返回四个根的列表
    return [c1 - c2 - a, -c1 - c3 - a, -c1 + c3 - a, c1 + c2 - a]
def roots_quartic(f):
    """
    Returns a list of roots of a quartic polynomial.

    There are many references for solving quartic expressions available [1-5].
    This reviewer has found that many of them require one to select from among
    2 or more possible sets of solutions and that some solutions work when one
    is searching for real roots but do not work when searching for complex roots
    (though this is not always stated clearly). The following routine has been
    tested and found to be correct for 0, 2 or 4 complex roots.

    The quasisymmetric case solution [6] looks for quartics that have the form
    `x**4 + A*x**3 + B*x**2 + C*x + D = 0` where `(C/A)**2 = D`.

    Although no general solution that is always applicable for all
    coefficients is known to this reviewer, certain conditions are tested
    to determine the simplest 4 expressions that can be returned:

      1) `f = c + a*(a**2/8 - b/2) == 0`
      2) `g = d - a*(a*(3*a**2/256 - b/16) + c/4) = 0`
      3) if `f != 0` and `g != 0` and `p = -d + a*c/4 - b**2/12` then
        a) `p == 0`
        b) `p != 0`

    Examples
    ========

        >>> from sympy import Poly
        >>> from sympy.polys.polyroots import roots_quartic

        >>> r = roots_quartic(Poly('x**4-6*x**3+17*x**2-26*x+20'))

        >>> # 4 complex roots: 1+-I*sqrt(3), 2+-I
        >>> sorted(str(tmp.evalf(n=2)) for tmp in r)
        ['1.0 + 1.7*I', '1.0 - 1.7*I', '2.0 + 1.0*I', '2.0 - 1.0*I']

    References
    ==========

    1. http://mathforum.org/dr.math/faq/faq.cubic.equations.html
    2. https://en.wikipedia.org/wiki/Quartic_function#Summary_of_Ferrari.27s_method
    3. https://planetmath.org/encyclopedia/GaloisTheoreticDerivationOfTheQuarticFormula.html
    4. https://people.bath.ac.uk/masjhd/JHD-CA.pdf
    5. http://www.albmath.org/files/Math_5713.pdf
    6. https://web.archive.org/web/20171002081448/http://www.statemaster.com/encyclopedia/Quartic-equation
    7. https://eqworld.ipmnet.ru/en/solutions/ae/ae0108.pdf
    """

    # Extract coefficients of the polynomial, assuming it is monic (leading coefficient is 1)
    _, a, b, c, d = f.monic().all_coeffs()

    # Check special cases based on coefficients
    if not d:
        # If d == 0, one root is 0, and solve the cubic polynomial x^3 + a*x^2 + b*x + c
        return [S.Zero] + roots([1, a, b, c], multiple=True)
    elif (c/a)**2 == d:
        x, m = f.gen, c/a

        # Construct a quadratic polynomial g(x) = x^2 + a*x + b - 2*m
        g = Poly(x**2 + a*x + b - 2*m, x)

        # Find roots of the quadratic equation g(x)
        z1, z2 = roots_quadratic(g)

        # Construct two new quadratic polynomials h1(x) and h2(x)
        h1 = Poly(x**2 - z1*x + m, x)
        h2 = Poly(x**2 - z2*x + m, x)

        # Find roots of h1(x) and h2(x) (each should have two roots)
        r1 = roots_quadratic(h1)
        r2 = roots_quadratic(h2)

        # Concatenate all roots and return them
        return r1 + r2
    else:
        # 计算 a 的平方
        a2 = a**2
        # 计算 e
        e = b - 3*a2/8
        # 计算 f
        f = _mexpand(c + a*(a2/8 - b/2))
        # 计算 a/4
        aon4 = a/4
        # 计算 g
        g = _mexpand(d - aon4*(a*(3*a2/64 - b/4) + c))

        # 如果 f 等于零
        if f.is_zero:
            # 计算平方根 y1, y2
            y1, y2 = [sqrt(tmp) for tmp in roots([1, e, g], multiple=True)]
            # 返回根据公式计算得到的结果列表
            return [tmp - aon4 for tmp in [-y1, -y2, y1, y2]]
        
        # 如果 g 等于零
        if g.is_zero:
            # 计算 y
            y = [S.Zero] + roots([1, 0, e, f], multiple=True)
            # 返回根据公式计算得到的结果列表
            return [tmp - aon4 for tmp in y]
        
        else:
            # 使用 Descartes-Euler 方法求解四次方程
            sols = _roots_quartic_euler(e, f, g, aon4)
            # 如果有解，则返回解列表
            if sols:
                return sols
            # 使用 Ferrari 方法求解
            p = -e**2/12 - g
            q = -e**3/108 + e*g/3 - f**2/8
            TH = Rational(1, 3)

            def _ans(y):
                # 计算 w
                w = sqrt(e + 2*y)
                arg1 = 3*e + 2*y
                arg2 = 2*f/w
                ans = []
                # 对于每个 s 和 t 的组合，计算解
                for s in [-1, 1]:
                    root = sqrt(-(arg1 + s*arg2))
                    for t in [-1, 1]:
                        ans.append((s*w - t*root)/2 - aon4)
                return ans

            # 当 p 不等于零时，可能返回 Piecewise
            # 尝试简化 p
            p = _mexpand(p)

            # 当 p 等于零时，计算 y1
            y1 = e*Rational(-5, 6) - q**TH
            if p.is_zero:
                return _ans(y1)

            # 当 p 不等于零时，计算 y2
            root = sqrt(q**2/4 + p**3/27)
            r = -q/2 + root  # or -q/2 - root
            u = r**TH  # solve(x**3 - r, x) 的主要根
            y2 = e*Rational(-5, 6) + u - p/u/3
            if fuzzy_not(p.is_zero):
                return _ans(y2)

            # 一旦知道系数的值，根据条件返回 Piecewise 列表
            return [Piecewise((a1, Eq(p, 0)), (a2, True))
                    for a1, a2 in zip(_ans(y1), _ans(y2))]
# 计算二项式多项式的根列表。如果定义域为 ZZ，则根将按照负值在正值之前的顺序排序。
# 只要测试的假设正确，任何数值系数都会得到相同的排序，否则排序将不会被排序（但会是规范的）。
def roots_binomial(f):
    """Returns a list of roots of a binomial polynomial. If the domain is ZZ
    then the roots will be sorted with negatives coming before positives.
    The ordering will be the same for any numerical coefficients as long as
    the assumptions tested are correct, otherwise the ordering will not be
    sorted (but will be canonical).
    """
    # 获取多项式的次数
    n = f.degree()

    # 提取多项式的首项系数和常数项系数
    a, b = f.nth(n), f.nth(0)

    # 计算基础根 -b/a，进行约分处理
    base = -cancel(b/a)

    # 计算基础根的 n 次根 alpha
    alpha = root(base, n)

    # 如果 alpha 是一个数值，将其扩展为复数形式
    if alpha.is_number:
        alpha = alpha.expand(complex=True)

    # 定义一些参数，用于确定根的排序方式
    # 如果定义域为 ZZ，则保证返回的根是按照实数在非实数之前排序，非实数按照实部和虚部排序，例如 -1, 1, -1 + I, 2 - I
    neg = base.is_negative
    even = n % 2 == 0
    if neg:
        if even == True and (base + 1).is_positive:
            big = True
        else:
            big = False

    # 获取正确顺序的索引，以确保在定义域为 ZZ 时计算的根是有序的
    ks = []
    imax = n//2
    if even:
        ks.append(imax)
        imax -= 1
    if not neg:
        ks.append(0)
    for i in range(imax, 0, -1):
        if neg:
            ks.extend([i, -i])
        else:
            ks.extend([-i, i])
    if neg:
        ks.append(0)
        if big:
            for i in range(0, len(ks), 2):
                pair = ks[i: i + 2]
                pair = list(reversed(pair))

    # 计算根
    roots, d = [], 2*I*pi/n
    for k in ks:
        zeta = exp(k*d).expand(complex=True)
        roots.append((alpha*zeta).expand(power_base=False))

    return roots


def _inv_totient_estimate(m):
    """
    Find ``(L, U)`` such that ``L <= phi^-1(m) <= U``.

    Examples
    ========

    >>> from sympy.polys.polyroots import _inv_totient_estimate

    >>> _inv_totient_estimate(192)
    (192, 840)
    >>> _inv_totient_estimate(400)
    (400, 1750)

    """
    # 计算逆欧拉函数的估计范围 ``(L, U)``，使得 ``L <= phi^-1(m) <= U``。

    # 找到 m 的所有因子中素数 d + 1 的列表 primes
    primes = [ d + 1 for d in divisors(m) if isprime(d + 1) ]

    a, b = 1, 1

    # 计算 Euler 函数的乘积 a 和 b
    for p in primes:
        a *= p
        b *= p - 1

    L = m
    U = int(math.ceil(m*(float(a)/b)))

    P = p = 2
    primes = []

    # 计算素数 P 的乘积，直到大于 U
    while P <= U:
        p = nextprime(p)
        primes.append(p)
        P *= p

    P //= p
    b = 1

    # 计算除最后一个素数外的其余素数的乘积 b
    for p in primes[:-1]:
        b *= p - 1

    U = int(math.ceil(m*(float(P)/b)))

    return L, U


def roots_cyclotomic(f, factor=False):
    """Compute roots of cyclotomic polynomials. """
    # 计算旋轮多项式的根。
    L, U = _inv_totient_estimate(f.degree())

    for n in range(L, U + 1):
        g = cyclotomic_poly(n, f.gen, polys=True)

        if f.expr == g.expr:
            break
    else:  # pragma: no cover
        raise RuntimeError("failed to find index of a cyclotomic polynomial")

    roots = []

    # 这里未完全的函数需要补充注释
    # 如果 factor 为假（例如 None 或空值），执行以下操作：
    # 获取正确顺序的索引，以便计算出的根是排序过的
    h = n // 2
    # 使用 igcd 函数检查每个 i 和 n 是否互质，筛选出符合条件的索引列表
    ks = [i for i in range(1, n + 1) if igcd(i, n) == 1]
    # 根据 lambda 函数对 ks 列表进行排序，lambda 函数根据不同的情况返回不同的元组来排序
    ks.sort(key=lambda x: (x, -1) if x <= h else (abs(x - n), 1))
    # 计算复数环境下的常数 d
    d = 2*I*pi/n
    # 对 ks 列表进行逆序迭代，并计算每个 k 对应的指数值，将结果添加到 roots 列表中
    for k in reversed(ks):
        roots.append(exp(k*d).expand(complex=True))
    # 如果 factor 不为空，执行以下操作：
    # 使用 Poly 类基于给定的 f 和 n 创建多项式 g
    g = Poly(f, extension=root(-1, n))

    # 遍历 g 的因式列表中的每个项 (h, _)，将 -h 的主系数添加到 roots 列表中
    for h, _ in ordered(g.factor_list()[1]):
        roots.append(-h.TC())

    # 返回计算得到的 roots 列表
    return roots
def roots_quintic(f):
    """
    Calculate exact roots of a solvable irreducible quintic with rational coefficients.
    Return an empty list if the quintic is reducible or not solvable.
    """
    result = []  # 初始化一个空列表用于存储计算结果

    coeff_5, coeff_4, p_, q_, r_, s_ = f.all_coeffs()  # 获取多项式 f 的所有系数

    if not all(coeff.is_Rational for coeff in (coeff_5, coeff_4, p_, q_, r_, s_)):
        return result  # 如果任何一个系数不是有理数，则返回空列表

    if coeff_5 != 1:
        f = Poly(f / coeff_5)  # 如果最高次项系数不为 1，则将 f 规范化
        _, coeff_4, p_, q_, r_, s_ = f.all_coeffs()

    # 取消 coeff_4，形成 x^5 + px^3 + qx^2 + rx + s 的形式
    if coeff_4:
        p = p_ - 2*coeff_4*coeff_4/5
        q = q_ - 3*coeff_4*p_/5 + 4*coeff_4**3/25
        r = r_ - 2*coeff_4*q_/5 + 3*coeff_4**2*p_/25 - 3*coeff_4**4/125
        s = s_ - coeff_4*r_/5 + coeff_4**2*q_/25 - coeff_4**3*p_/125 + 4*coeff_4**5/3125
        x = f.gen
        f = Poly(x**5 + p*x**3 + q*x**2 + r*x + s)
    else:
        p, q, r, s = p_, q_, r_, s_

    quintic = PolyQuintic(f)  # 创建一个 PolyQuintic 对象来处理 f

    if not f.is_irreducible:
        return result  # 如果 f 不是不可约的，则返回空列表
    f20 = quintic.f20  # 获取 PolyQuintic 对象的 f20 属性

    if f20.is_irreducible:
        return result  # 如果 f20 是不可约的，则返回空列表

    # 现在我们知道 f 是可解的
    for _factor in f20.factor_list()[1]:  # 遍历 f20 的线性因子列表
        if _factor[0].is_linear:
            theta = _factor[0].root(0)  # 找到线性因子的根 theta
            break

    d = discriminant(f)  # 计算多项式的判别式
    delta = sqrt(d)  # 计算判别式的平方根

    zeta1, zeta2, zeta3, zeta4 = quintic.zeta  # 获取 PolyQuintic 对象的 zeta 属性
    T = quintic.T(theta, d)  # 计算 T 矩阵
    tol = S(1e-10)  # 设置一个容差值

    alpha = T[1] + T[2]*delta
    alpha_bar = T[1] - T[2]*delta
    beta = T[3] + T[4]*delta
    beta_bar = T[3] - T[4]*delta

    disc = alpha**2 - 4*beta
    disc_bar = alpha_bar**2 - 4*beta_bar

    l0 = quintic.l0(theta)  # 计算 l0
    Stwo = S(2)
    l1 = _quintic_simplify((-alpha + sqrt(disc)) / Stwo)  # 简化计算得到的表达式
    l4 = _quintic_simplify((-alpha - sqrt(disc)) / Stwo)

    l2 = _quintic_simplify((-alpha_bar + sqrt(disc_bar)) / Stwo)
    l3 = _quintic_simplify((-alpha_bar - sqrt(disc_bar)) / Stwo)

    order = quintic.order(theta, d)  # 计算序数
    test = (order*delta.n()) - ((l1.n() - l4.n())*(l2.n() - l3.n()))  # 计算测试值，用于浮点数比较

    if not comp(test, 0, tol):  # 如果测试值与零比较不相等
        l2, l3 = l3, l2  # 交换 l2 和 l3 的值

    # 现在 l 的顺序是正确的
    R1 = l0 + l1*zeta1 + l2*zeta2 + l3*zeta3 + l4*zeta4
    R2 = l0 + l3*zeta1 + l1*zeta2 + l4*zeta3 + l2*zeta4
    R3 = l0 + l2*zeta1 + l4*zeta2 + l1*zeta3 + l3*zeta4
    R4 = l0 + l4*zeta1 + l3*zeta2 + l2*zeta3 + l1*zeta4

    Res = [None, [None]*5, [None]*5, [None]*5, [None]*5]  # 初始化结果矩阵
    Res_n = [None, [None]*5, [None]*5, [None]*5, [None]*5]  # 初始化简化后的结果矩阵

    R1 = _quintic_simplify(R1)  # 对 R1 进行简化，提高精确性和性能
    R2 = _quintic_simplify(R2)
    R3 = _quintic_simplify(R3)
    R4 = _quintic_simplify(R4)

    # 以下是为 [factor(i) for i in _vsolve(x**5 - a - I*b, x)] 硬编码的结果
    x0 = z**(S(1)/5)
    x1 = sqrt(2)
    x2 = sqrt(5)
    x3 = sqrt(5 - x2)
    x4 = I*x2
    x5 = x4 + I
    x6 = I*x0/4
    x7 = x1*sqrt(x2 + 5)
    sol = [x0, -x6*(x1*x3 - x5), x6*(x1*x3 + x5), -x6*(x4 + x7 - I), x6*(-x4 + x7 + I)]
    # 定义方程的解表达式列表，包括常数项和线性项的组合

    R1 = R1.as_real_imag()
    R2 = R2.as_real_imag()
    R3 = R3.as_real_imag()
    R4 = R4.as_real_imag()
    # 将 R1, R2, R3, R4 转换为实部和虚部的元组形式

    for i, s in enumerate(sol):
        # 遍历解表达式列表
        Res[1][i] = _quintic_simplify(s.xreplace({z: R1[0] + I*R1[1]}))
        # 计算解表达式 sol[i] 中的 z 代入 R1 的值后的简化结果，并存储在 Res[1][i] 中
        Res[2][i] = _quintic_simplify(s.xreplace({z: R2[0] + I*R2[1]}))
        # 计算解表达式 sol[i] 中的 z 代入 R2 的值后的简化结果，并存储在 Res[2][i] 中
        Res[3][i] = _quintic_simplify(s.xreplace({z: R3[0] + I*R3[1]}))
        # 计算解表达式 sol[i] 中的 z 代入 R3 的值后的简化结果，并存储在 Res[3][i] 中
        Res[4][i] = _quintic_simplify(s.xreplace({z: R4[0] + I*R4[1]}))
        # 计算解表达式 sol[i] 中的 z 代入 R4 的值后的简化结果，并存储在 Res[4][i] 中

    for i in range(1, 5):
        for j in range(5):
            Res_n[i][j] = Res[i][j].n()
            # 将 Res[i][j] 中的表达式转换为数值，并存储在 Res_n[i][j] 中
            Res[i][j] = _quintic_simplify(Res[i][j])
            # 对 Res[i][j] 中的表达式进行简化处理，更新 Res[i][j]

    r1 = Res[1][0]
    r1_n = Res_n[1][0]
    # 获取 Res 中第一行第一列的表达式及其数值化结果

    for i in range(5):
        if comp(im(r1_n*Res_n[4][i]), 0, tol):
            # 检查 r1_n 乘以 Res_n[4][i] 的虚部是否为 0
            r4 = Res[4][i]
            # 如果条件满足，将 r4 设置为 Res 中第五行第 i 列的表达式
            break

    # 现在我们有了各种 Res 的值。每个 Res 都是包含五个值的列表。我们需要从中选择一个 r 值
    u, v = quintic.uv(theta, d)
    # 计算 quintic 类中的 uv 方法返回的 u 和 v 值
    testplus = (u + v*delta*sqrt(5)).n()
    testminus = (u - v*delta*sqrt(5)).n()
    # 计算两个测试值 testplus 和 testminus 的数值

    # 使用数值化的 r4_n 进行计算
    r4_n = r4.n()
    r2 = r3 = None

    for i in range(5):
        r2temp_n = Res_n[2][i]
        # 获取 Res_n 中第二行第 i 列的数值化表达式
        for j in range(5):
            r3temp_n = Res_n[3][j]
            # 获取 Res_n 中第三行第 j 列的数值化表达式
            if (comp((r1_n*r2temp_n**2 + r4_n*r3temp_n**2 - testplus).n(), 0, tol) and
                comp((r3temp_n*r1_n**2 + r2temp_n*r4_n**2 - testminus).n(), 0, tol)):
                # 检查一对方程是否满足条件
                r2 = Res[2][i]
                r3 = Res[3][j]
                # 如果满足条件，则分别将 r2 和 r3 设置为 Res 中对应的表达式
                break
        if r2 is not None:
            break
    else:
        return []  # 如果没有找到满足条件的 r2 和 r3，返回空列表作为回退解

    # 现在，我们有了 r 值，可以计算根
    x1 = (r1 + r2 + r3 + r4)/5
    x2 = (r1*zeta4 + r2*zeta3 + r3*zeta2 + r4*zeta1)/5
    x3 = (r1*zeta3 + r2*zeta1 + r3*zeta4 + r4*zeta2)/5
    x4 = (r1*zeta2 + r2*zeta4 + r3*zeta1 + r4*zeta3)/5
    x5 = (r1*zeta1 + r2*zeta2 + r3*zeta3 + r4*zeta4)/5
    # 计算五个根的平均值

    result = [x1, x2, x3, x4, x5]

    # 现在检查解是否唯一
    saw = set()
    for r in result:
        r = r.n(2)
        # 将根 r 转换为浮点数，保留两位小数
        if r in saw:
            # 如果根重复，则返回空列表，并回退到常规解法
            return []
        saw.add(r)

    # 恢复到原始方程，其中 coeff_4 不为零
    if coeff_4:
        result = [x - coeff_4 / 5 for x in result]
        # 如果 coeff_4 不为零，则从结果中减去 coeff_4 的五分之一

    return result
    # 返回最终的根结果
# 定义一个函数，用于对给定表达式进行五次化简
def _quintic_simplify(expr):
    # 导入 sympy 中的 powsimp 函数，用于化简表达式中的指数项
    from sympy.simplify.simplify import powsimp
    # 对表达式进行指数项化简
    expr = powsimp(expr)
    # 调用未定义的 cancel 函数，对表达式进一步化简
    expr = cancel(expr)
    # 返回合并同类项后的表达式
    return together(expr)


def _integer_basis(poly):
    """计算整数多项式的系数基础。

    返回整数 ``div``，使得将 ``x = div*y`` 代入后，``p(x) = m*q(y)``，
    其中 ``q`` 的系数小于 ``p`` 的系数。

    例如，对于 ``x**5 + 512*x + 1024 = 0``，
    当 ``div = 4`` 时，变为 ``y**5 + 2*y + 1 = 0``。

    若无法进行缩放，则返回 ``None``。

    Examples
    ========

    >>> from sympy.polys import Poly
    >>> from sympy.abc import x
    >>> from sympy.polys.polyroots import _integer_basis
    >>> p = Poly(x**5 + 512*x + 1024, x, domain='ZZ')
    >>> _integer_basis(p)
    4
    """
    # 将多项式的项分解为单项式和系数
    monoms, coeffs = list(zip(*poly.terms()))

    # 提取单项式列表
    monoms, = list(zip(*monoms))
    # 取系数的绝对值
    coeffs = list(map(abs, coeffs))

    # 如果首项系数小于末项系数，则反转列表
    if coeffs[0] < coeffs[-1]:
        coeffs = list(reversed(coeffs))
        n = monoms[0]
        monoms = [n - i for i in reversed(monoms)]
    else:
        return None

    # 去除最后一项
    monoms = monoms[:-1]
    coeffs = coeffs[:-1]

    # 对于两项多项式的特殊情况
    if len(monoms) == 1:
        r = Pow(coeffs[0], S.One/monoms[0])
        # 如果结果是整数，则返回其整数值，否则返回 None
        if r.is_Integer:
            return int(r)
        else:
            return None

    # 从最大公约数列表中逆序获取除数
    divs = reversed(divisors(gcd_list(coeffs))[1:])

    try:
        div = next(divs)
    except StopIteration:
        return None

    while True:
        # 检查每个单项式和系数对是否符合给定的除数
        for monom, coeff in zip(monoms, coeffs):
            if coeff % div**monom != 0:
                try:
                    div = next(divs)
                except StopIteration:
                    return None
                else:
                    break
        else:
            return div


def preprocess_roots(poly):
    """尝试消除多项式 ``poly`` 中的符号系数。"""
    # 初始化系数为 1
    coeff = S.One

    # 获取多项式的函数和多项式本身
    poly_func = poly.func
    try:
        # 清除多项式的分母并转换，若有域错误则返回当前系数和多项式
        _, poly = poly.clear_denoms(convert=True)
    except DomainError:
        return coeff, poly

    # 对多项式进行原始形式
    poly = poly.primitive()[1]
    # 收缩多项式
    poly = poly.retract()

    # TODO: 这段代码很脆弱。找出如何使其独立于 construct_domain()。
    # 如果多项式的域为多项式并且所有系数均为项
    if poly.get_domain().is_Poly and all(c.is_term for c in poly.rep.coeffs()):
        # 对多项式进行注入
        poly = poly.inject()
    
        # 获取多项式的单项式生成器和生成器列表
        strips = list(zip(*poly.monoms()))
        gens = list(poly.gens[1:])
    
        # 将第一个单项式作为基础，其余单项式作为strips列表
        base, strips = strips[0], strips[1:]
    
        # 对每个生成器和对应的strip进行迭代
        for gen, strip in zip(list(gens), strips):
            reverse = False
    
            # 如果strip的第一个元素小于最后一个元素，反转strip
            if strip[0] < strip[-1]:
                strip = reversed(strip)
                reverse = True
    
            ratio = None
    
            # 对基础和strip中的元素进行迭代
            for a, b in zip(base, strip):
                if not a and not b:
                    continue
                elif not a or not b:
                    break
                elif b % a != 0:
                    break
                else:
                    _ratio = b // a
    
                    if ratio is None:
                        ratio = _ratio
                    elif ratio != _ratio:
                        break
            else:
                # 如果reverse为True，则反转比率
                if reverse:
                    ratio = -ratio
    
                # 对多项式进行gen的求值，并更新系数
                poly = poly.eval(gen, 1)
                coeff *= gen**(-ratio)
                # 从生成器列表中移除当前生成器
                gens.remove(gen)
    
        # 如果还有未处理的生成器，则从多项式中弹出这些生成器
        if gens:
            poly = poly.eject(*gens)
    
    # 如果多项式是一元的并且其域为整数环
    if poly.is_univariate and poly.get_domain().is_ZZ:
        # 使用整数基础计算多项式的基础
        basis = _integer_basis(poly)
    
        # 如果基础不为None
        if basis is not None:
            n = poly.degree()
    
            # 定义一个函数，对给定的k和系数进行操作
            def func(k, coeff):
                return coeff // basis**(n - k[0])
    
            # 对多项式进行termwise操作
            poly = poly.termwise(func)
            # 更新系数
            coeff *= basis
    
    # 如果多项式不是poly_func的实例，则将其转换为poly_func的实例
    if not isinstance(poly, poly_func):
        poly = poly_func(poly)
    
    # 返回更新后的系数和多项式
    return coeff, poly
# 定义一个公共函数 roots，用于计算单变量多项式的符号根
@public
def roots(f, *gens,
        auto=True,
        cubics=True,
        trig=False,
        quartics=True,
        quintics=False,
        multiple=False,
        filter=None,
        predicate=None,
        strict=False,
        **flags):
    """
    Computes symbolic roots of a univariate polynomial.

    给定一个具有符号系数（或多项式系数列表）的单变量多项式 f，返回一个字典，
    包含其根及其重数。

    Only roots expressible via radicals will be returned.  To get
    a complete set of roots use RootOf class or numerical methods
    instead. By default cubic and quartic formulas are used in
    the algorithm. To disable them because of unreadable output
    set ``cubics=False`` or ``quartics=False`` respectively. If cubic
    roots are real but are expressed in terms of complex numbers
    (casus irreducibilis [1]) the ``trig`` flag can be set to True to
    have the solutions returned in terms of cosine and inverse cosine
    functions.

    To get roots from a specific domain set the ``filter`` flag with
    one of the following specifiers: Z, Q, R, I, C. By default all
    roots are returned (this is equivalent to setting ``filter='C'``).

    By default a dictionary is returned giving a compact result in
    case of multiple roots.  However to get a list containing all
    those roots set the ``multiple`` flag to True; the list will
    have identical roots appearing next to each other in the result.
    (For a given Poly, the all_roots method will give the roots in
    sorted numerical order.)

    If the ``strict`` flag is True, ``UnsolvableFactorError`` will be
    raised if the roots found are known to be incomplete (because
    some roots are not expressible in radicals).

    Examples
    ========

    >>> from sympy import Poly, roots, degree
    >>> from sympy.abc import x, y

    >>> roots(x**2 - 1, x)
    {-1: 1, 1: 1}

    >>> p = Poly(x**2-1, x)
    >>> roots(p)
    {-1: 1, 1: 1}

    >>> p = Poly(x**2-y, x, y)

    >>> roots(Poly(p, x))
    {-sqrt(y): 1, sqrt(y): 1}

    >>> roots(x**2 - y, x)
    {-sqrt(y): 1, sqrt(y): 1}

    >>> roots([1, 0, -1])
    {-1: 1, 1: 1}

    ``roots`` will only return roots expressible in radicals. If
    the given polynomial has some or all of its roots inexpressible in
    radicals, the result of ``roots`` will be incomplete or empty
    respectively.

    Example where result is incomplete:

    >>> roots((x-1)*(x**5-x+1), x)
    {1: 1}

    In this case, the polynomial has an unsolvable quintic factor
    whose roots cannot be expressed by radicals. The polynomial has a
    rational root (due to the factor `(x-1)`), which is returned since
    ``roots`` always finds all rational roots.

    Example where result is empty:

    >>> roots(x**7-3*x**2+1, x)
    {}

    Here, the polynomial has no roots expressible in radicals, so
    ``roots`` returns an empty dictionary.
    """
    """
    The result produced by ``roots`` is complete if and only if the
    sum of the multiplicity of each root is equal to the degree of
    the polynomial. If strict=True, UnsolvableFactorError will be
    raised if the result is incomplete.

    The result can be checked for completeness as follows:

    >>> f = x**3-2*x**2+1
    >>> sum(roots(f, x).values()) == degree(f, x)
    True
    >>> f = (x-1)*(x**5-x+1)
    >>> sum(roots(f, x).values()) == degree(f, x)
    False

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Cubic_equation#Trigonometric_and_hyperbolic_solutions

    """

    # 导入 sympy.polys.polytools 模块中的 to_rational_coeffs 函数
    from sympy.polys.polytools import to_rational_coeffs

    # 复制 flags 参数的副本，确保不影响原始参数
    flags = dict(flags)

    # 如果输入的多项式 f 是一个列表
    if isinstance(f, list):
        # 如果 gens 非空，抛出 ValueError 异常，指示存在多余的生成器
        if gens:
            raise ValueError('redundant generators given')

        # 创建一个虚拟变量 x
        x = Dummy('x')

        # 初始化一个空字典 poly 和一个指数 i，从 f 的长度减 1 开始递减
        poly, i = {}, len(f) - 1

        # 遍历 f 中的每个系数 coeff
        for coeff in f:
            # 将系数 coeff 转换为 SymPy 的表达式，并存入字典 poly
            poly[i], i = sympify(coeff), i - 1

        # 使用 poly 字典和虚拟变量 x 创建一个多项式对象 Poly，设定其为域 field=True
        f = Poly(poly, x, field=True)
    
    else:
        try:
            # 尝试将 f 转换为 Poly 类型，如果成功，则检查其生成器的类型和正确性
            F = Poly(f, *gens, **flags)
            if not isinstance(f, Poly) and not F.gen.is_Symbol:
                raise PolynomialError("generator must be a Symbol")
            f = F
        except GeneratorsNeeded:
            # 如果 GeneratorsNeeded 异常被抛出，根据 multiple 参数返回空列表或空字典
            if multiple:
                return []
            else:
                return {}
        else:
            # 计算多项式 f 的次数 n
            n = f.degree()

            # 如果多项式 f 的长度为 2 且次数 n 大于 2
            if f.length() == 2 and n > 2:
                # 检查是否存在形如 foo**n 的常数，如果 dep 是 c*gen**m 的形式
                con, dep = f.as_expr().as_independent(*f.gens)
                
                # 将负号从 con 中移除并进行因式分解
                fcon = -(-con).factor()

                # 如果分解后的结果 fcon 不等于原始 con
                if fcon != con:
                    con = fcon
                    bases = []

                    # 遍历 con 的乘法因子
                    for i in Mul.make_args(con):
                        if i.is_Pow:
                            b, e = i.as_base_exp()
                            # 如果指数 e 是整数且基数 b 是加法表达式
                            if e.is_Integer and b.is_Add:
                                # 将基数 b 和一个带有正常量的虚拟变量添加到 bases 列表中
                                bases.append((b, Dummy(positive=True)))
                    
                    # 如果存在符合条件的 bases 列表
                    if bases:
                        # 使用 bases 替换 dep+con 中的变量，并调用 roots 函数计算
                        rv = roots(Poly((dep + con).xreplace(dict(bases)),
                                       *F.gens),
                                   *F.gens,
                                   auto=auto,
                                   cubics=cubics,
                                   trig=trig,
                                   quartics=quartics,
                                   quintics=quintics,
                                   multiple=multiple,
                                   filter=filter,
                                   predicate=predicate,
                                   **flags)
                        # 返回一个字典，其中因子经过项的处理后成为键，值为其对应的根
                        return {factor_terms(k.xreplace({v: k for k, v in bases})): v for k, v in rv.items()}

        # 如果多项式 f 是多变量多项式，抛出异常 PolynomialError
        if f.is_multivariate:
            raise PolynomialError('multivariate polynomials are not supported')
    def _update_dict(result, zeros, currentroot, k):
        # 更新结果字典：如果当前根为零，则将其加入 zeros 字典中；否则将其加入 result 字典中
        if currentroot == S.Zero:
            # 如果当前根为零，并且 zeros 字典中已存在零根，则增加其对应的值；否则将其添加到 zeros 字典中
            if S.Zero in zeros:
                zeros[S.Zero] += k
            else:
                zeros[S.Zero] = k
        # 如果当前根已经存在于 result 字典中，则增加其对应的值；否则将当前根及其对应的值添加到 result 字典中
        if currentroot in result:
            result[currentroot] += k
        else:
            result[currentroot] = k

    def _try_decompose(f):
        """Find roots using functional decomposition. """
        # 将多项式 f 进行分解，得到因子和根的列表
        factors, roots = f.decompose(), []

        # 对于第一个因子的每一个根，尝试使用启发式方法获取根
        for currentroot in _try_heuristics(factors[0]):
            roots.append(currentroot)

        # 对于剩余的因子进行处理
        for currentfactor in factors[1:]:
            previous, roots = list(roots), []

            # 对于上一步得到的每一个根，计算当前因子的 g(x) 并尝试获取其根
            for currentroot in previous:
                g = currentfactor - Poly(currentroot, f.gen)

                for currentroot in _try_heuristics(g):
                    roots.append(currentroot)

        return roots

    def _try_heuristics(f):
        """Find roots using formulas and some tricks. """
        # 如果多项式是常数项，则返回空列表
        if f.is_ground:
            return []
        # 如果多项式是单项式，则返回相应个数的零根列表
        if f.is_monomial:
            return [S.Zero]*f.degree()

        # 如果多项式长度为2
        if f.length() == 2:
            if f.degree() == 1:
                # 对于线性多项式，返回其根的列表
                return list(map(cancel, roots_linear(f)))
            else:
                # 对于二项式，使用二项式根的方法获取其根
                return roots_binomial(f)

        result = []

        # 对于 -1 和 1 进行遍历
        for i in [-1, 1]:
            # 如果在 i 处多项式的值为零
            if not f.eval(i):
                # 将多项式 f 除以 (f.gen - i) 并记录根 i
                f = f.quo(Poly(f.gen - i, f.gen))
                result.append(i)
                break

        n = f.degree()

        # 根据多项式的次数选择适当的方法来获取其根
        if n == 1:
            result += list(map(cancel, roots_linear(f)))
        elif n == 2:
            result += list(map(cancel, roots_quadratic(f)))
        elif f.is_cyclotomic:
            result += roots_cyclotomic(f)
        elif n == 3 and cubics:
            result += roots_cubic(f, trig=trig)
        elif n == 4 and quartics:
            result += roots_quartic(f)
        elif n == 5 and quintics:
            result += roots_quintic(f)

        return result

    # 将生成器转换为符号变量
    dumgens = symbols('x:%d' % len(f.gens), cls=Dummy)
    # 将多项式 f 转换为其表示，并返回其首项的 GCD 和余项
    f = f.per(f.rep, dumgens)

    # 获取首项的 GCD 和余项
    (k,), f = f.terms_gcd()

    # 如果首项的 GCD 为零，则初始化 zeros 字典为空；否则将首项的 GCD 初始化为 S.Zero
    if not k:
        zeros = {}
    else:
        zeros = {S.Zero: k}

    # 预处理多项式 f 的根及其系数
    coeff, f = preprocess_roots(f)

    # 如果自动选项开启且多项式 f 的域是环，则将其转换为域
    if auto and f.get_domain().is_Ring:
        f = f.to_field()

    # 如果多项式 f 的域是 QQ_I，则将其转换为 EX
    if f.get_domain().is_QQ_I:
        f = f.per(f.rep.convert(EX))

    rescale_x = None
    translate_x = None

    # 初始化结果字典为空字典
    result = {}
    # 如果多项式 f 不是地面多项式（即不是具体的数值多项式）
    if not f.is_ground:
        # 获取多项式 f 的定义域
        dom = f.get_domain()
        # 如果定义域不是精确数值，但是是数值类型
        if not dom.is_Exact and dom.is_Numerical:
            # 对于 f 的所有数根，更新结果字典 result 中对应根的计数加 1
            for r in f.nroots():
                _update_dict(result, zeros, r, 1)
        # 如果多项式 f 的次数为 1
        elif f.degree() == 1:
            # 获取一次多项式的根，并将其加入结果字典中
            _update_dict(result, zeros, roots_linear(f)[0], 1)
        # 如果多项式 f 的长度为 2
        elif f.length() == 2:
            # 根据多项式的次数选择使用二次方程或二项式求根函数
            roots_fun = roots_quadratic if f.degree() == 2 else roots_binomial
            # 对于所选函数返回的所有根，更新结果字典 result 中对应根的计数加 1
            for r in roots_fun(f):
                _update_dict(result, zeros, r, 1)
        else:
            # 将多项式 f 转化为表达式并进行因式分解，获取系数和因子列表
            _, factors = Poly(f.as_expr()).factor_list()
            # 如果因式列表只有一个因子且多项式的次数为 2
            if len(factors) == 1 and f.degree() == 2:
                # 对二次多项式求根，并更新结果字典 result 中对应根的计数加 1
                for r in roots_quadratic(f):
                    _update_dict(result, zeros, r, 1)
            else:
                # 如果因式列表只有一个因子且其次数为 1
                if len(factors) == 1 and factors[0][1] == 1:
                    # 如果多项式的定义域是 EX 类型
                    if f.get_domain().is_EX:
                        # 将多项式转化为有理系数形式
                        res = to_rational_coeffs(f)
                        # 如果转化成功
                        if res:
                            # 如果第一个元素为空，表示需要平移 x 轴
                            if res[0] is None:
                                translate_x, f = res[2:]
                            # 否则，需要重新调整 x 的比例
                            else:
                                rescale_x, f = res[1], res[-1]
                            # 获取多项式 f 的根，并更新结果字典 result 中对应根的计数加 1
                            result = roots(f)
                            # 如果没有找到根，尝试分解多项式 f
                            if not result:
                                for currentroot in _try_decompose(f):
                                    _update_dict(result, zeros, currentroot, 1)
                        # 如果无法转化为有理系数形式，尝试启发式算法分解多项式 f
                        else:
                            for r in _try_heuristics(f):
                                _update_dict(result, zeros, r, 1)
                    # 如果多项式的定义域不是 EX 类型，尝试分解多项式 f
                    else:
                        for currentroot in _try_decompose(f):
                            _update_dict(result, zeros, currentroot, 1)
                # 如果因式列表不止一个因子或其次数不为 1，尝试启发式算法分解多项式 f
                else:
                    for currentfactor, k in factors:
                        for r in _try_heuristics(Poly(currentfactor, f.gen, field=True)):
                            _update_dict(result, zeros, r, k)

    # 如果系数 coeff 不等于 1
    if coeff is not S.One:
        # 复制结果字典 _result 到 result，并初始化 result 为空字典
        _result, result, = result, {}

        # 对于复制的结果字典中的每个根 currentroot 和其对应的计数 k
        for currentroot, k in _result.items():
            # 将系数乘以根，并更新结果字典 result 中对应的根
            result[coeff*currentroot] = k

    # 如果 filter 不是 None 或 'C'
    if filter not in [None, 'C']:
        # 定义处理器字典，根据 filter 类型选择不同的处理器函数
        handlers = {
            'Z': lambda r: r.is_Integer,
            'Q': lambda r: r.is_Rational,
            'R': lambda r: all(a.is_real for a in r.as_numer_denom()),
            'I': lambda r: r.is_imaginary,
        }

        try:
            # 根据 filter 类型选择相应的处理器函数
            query = handlers[filter]
        except KeyError:
            # 抛出无效 filter 的错误异常
            raise ValueError("Invalid filter: %s" % filter)

        # 对于结果字典中的每个零点 zero
        for zero in dict(result).keys():
            # 如果零点不满足所选的过滤器函数，从结果字典中删除该零点
            if not query(zero):
                del result[zero]

    # 如果 predicate 不是 None
    if predicate is not None:
        # 对于结果字典中的每个零点 zero
        for zero in dict(result).keys():
            # 如果零点不满足给定的谓词函数，从结果字典中删除该零点
            if not predicate(zero):
                del result[zero]

    # 如果需要重新调整 x 的比例
    if rescale_x:
        result1 = {}
        # 对于结果字典中的每对键值对 k, v
        for k, v in result.items():
            # 将键 k 乘以调整系数 rescale_x，并存储到 result1 中
            result1[k*rescale_x] = v
        # 更新结果字典 result 为调整后的结果字典 result1
        result = result1

    # 如果需要平移 x 轴
    if translate_x:
        result1 = {}
        # 对于结果字典中的每对键值对 k, v
        for k, v in result.items():
            # 将键 k 加上平移量 translate_x，并存储到 result1 中
            result1[k + translate_x] = v
        # 更新结果字典 result 为平移后的结果字典 result1
        result = result1
    # 将零根添加到结果字典，这些根是在转换非平凡根之后添加的
    result.update(zeros)

    # 如果 strict 为真，并且结果字典中的根的数量小于多项式的次数，则抛出异常
    if strict and sum(result.values()) < f.degree():
        raise UnsolvableFactorError(filldedent('''
            严格模式：某些因子无法用根式解出，因此无法返回完整的解的列表。
            调用 roots(strict=False) 来获取可以用根式表示的解（如果有的话）。
            '''))

    # 如果不需要多重根，则直接返回结果字典
    if not multiple:
        return result
    else:
        # 否则，准备一个空列表来存放多重根
        zeros = []

        # 遍历结果字典中的根，并按照其顺序重复添加到 zeros 列表中
        for zero in ordered(result):
            zeros.extend([zero]*result[zero])

        # 返回包含多重根的列表
        return zeros
# 定义一个函数，用于计算多项式的根因子（即多项式的所有因子）

def root_factors(f, *gens, filter=None, **args):
    """
    Returns all factors of a univariate polynomial.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy.polys.polyroots import root_factors

    >>> root_factors(x**2 - y, x)
    [x - sqrt(y), x + sqrt(y)]

    """

    # 将关键字参数转换为字典形式
    args = dict(args)

    # 将输入的多项式 f 转换为多项式对象 F
    F = Poly(f, *gens, **args)

    # 如果 F 不是多项式对象，直接返回原始多项式 f 的列表形式
    if not F.is_Poly:
        return [f]

    # 如果 F 是多变量多项式，抛出错误，因为该函数不支持多变量多项式
    if F.is_multivariate:
        raise ValueError('multivariate polynomials are not supported')

    # 获取多项式的首个变量
    x = F.gens[0]

    # 计算多项式的根
    zeros = roots(F, filter=filter)

    # 如果根为空集，将 F 作为唯一的因子返回
    if not zeros:
        factors = [F]
    else:
        factors, N = [], 0

        # 根据根的顺序，生成根因子
        for r, n in ordered(zeros.items()):
            factors, N = factors + [Poly(x - r, x)] * n, N + n

        # 如果生成的因子数量小于多项式的次数，则继续处理
        if N < F.degree():
            # 使用 reduce 函数计算已有因子的乘积
            G = reduce(lambda p, q: p * q, factors)
            # 将 F 除以乘积 G，得到剩余的因子并加入列表中
            factors.append(F.quo(G))

    # 如果输入的 f 不是多项式对象，将所有因子转换为表达式形式
    if not isinstance(f, Poly):
        factors = [f.as_expr() for f in factors]

    # 返回计算得到的所有因子列表
    return factors
```