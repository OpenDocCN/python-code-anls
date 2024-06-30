# `D:\src\scipysrc\sympy\sympy\functions\combinatorial\numbers.py`

```
"""
This module implements some special functions that commonly appear in
combinatorial contexts (e.g. in power series); in particular,
sequences of rational numbers such as Bernoulli and Fibonacci numbers.

Factorials, binomial coefficients and related functions are located in
the separate 'factorials' module.
"""

# 导入必要的数学函数和数据结构
from math import prod
from collections import defaultdict
from typing import Tuple as tTuple

# 导入 Sympy 中的核心类和函数
from sympy.core import S, Symbol, Add, Dummy
from sympy.core.cache import cacheit
from sympy.core.containers import Dict
from sympy.core.expr import Expr
from sympy.core.function import ArgumentIndexError, Function, expand_mul
from sympy.core.logic import fuzzy_not
from sympy.core.mul import Mul
from sympy.core.numbers import E, I, pi, oo, Rational, Integer
from sympy.core.relational import Eq, is_le, is_gt, is_lt
from sympy.external.gmpy import SYMPY_INTS, remove, lcm, legendre, jacobi, kronecker

# 导入组合数学相关的函数
from sympy.functions.combinatorial.factorials import (
    binomial, factorial, subfactorial
)

# 导入指数函数和分段函数
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.piecewise import Piecewise

# 导入数论相关的函数
from sympy.ntheory.factor_ import (
    factorint, _divisor_sigma, is_carmichael,
    find_carmichael_numbers_in_range, find_first_n_carmichaels
)
from sympy.ntheory.generate import _primepi
from sympy.ntheory.partitions_ import _partition, _partition_rec
from sympy.ntheory.primetest import isprime, is_square

# 导入多项式相关的函数
from sympy.polys.appellseqs import bernoulli_poly, euler_poly, genocchi_poly
from sympy.polys.polytools import cancel

# 导入工具类和异常处理相关的函数
from sympy.utilities.enumerative import MultisetPartitionTraverser
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import multiset, multiset_derangements, iterable
from sympy.utilities.memoization import recurrence_memo
from sympy.utilities.misc import as_int

# 导入 mpmath 库中的函数
from mpmath import mp, workprec
from mpmath.libmp import ifib as _ifib


# 函数用于计算区间 [a, b] 内整数的乘积
def _product(a, b):
    return prod(range(a, b + 1))


# 符号 x 的虚拟符号，用于计算多项式序列
_sym = Symbol('x')


#----------------------------------------------------------------------------#
#                                                                            #
#                           Carmichael numbers                               #
#                                                                            #
#----------------------------------------------------------------------------#

# Carmichael 类，表示 Carmichael 数
class carmichael(Function):
    r"""
    Carmichael Numbers:

    Certain cryptographic algorithms make use of big prime numbers.
    However, checking whether a big number is prime is not so easy.
    Randomized prime number checking tests exist that offer a high degree of
    confidence of accurate determination at low cost, such as the Fermat test.

    Let 'a' be a random number between $2$ and $n - 1$, where $n$ is the
    number whose primality we are testing. Then, $n$ is probably prime if it
    # 静态方法：判断给定整数是否为完全平方数
    @staticmethod
    def is_perfect_square(n):
        # 调用 sympy 库的警告函数，提示该方法已被弃用
        sympy_deprecation_warning(
    @staticmethod
    def is_perfect_square(n):
        sympy_deprecation_warning(
        """
        is_perfect_square is just a wrapper around sympy.ntheory.primetest.is_square
        so use that directly instead.
        """,
        deprecated_since_version="1.11",
        active_deprecations_target='deprecated-carmichael-static-methods',
        )
        return is_square(n)

    @staticmethod
    def divides(p, n):
        sympy_deprecation_warning(
        """
        divides can be replaced by directly testing n % p == 0.
        """,
        deprecated_since_version="1.11",
        active_deprecations_target='deprecated-carmichael-static-methods',
        )
        return n % p == 0

    @staticmethod
    def is_prime(n):
        sympy_deprecation_warning(
        """
        is_prime is just a wrapper around sympy.ntheory.primetest.isprime so use that
        directly instead.
        """,
        deprecated_since_version="1.11",
        active_deprecations_target='deprecated-carmichael-static-methods',
        )
        return isprime(n)

    @staticmethod
    def is_carmichael(n):
        sympy_deprecation_warning(
        """
        is_carmichael is just a wrapper around sympy.ntheory.factor_.is_carmichael so use that
        directly instead.
        """,
        deprecated_since_version="1.13",
        active_deprecations_target='deprecated-ntheory-symbolic-functions',
        )
        return is_carmichael(n)

    @staticmethod
    def find_carmichael_numbers_in_range(x, y):
        sympy_deprecation_warning(
        """
        find_carmichael_numbers_in_range is just a wrapper around sympy.ntheory.factor_.find_carmichael_numbers_in_range so use that
        directly instead.
        """,
        deprecated_since_version="1.13",
        active_deprecations_target='deprecated-ntheory-symbolic-functions',
        )
        return find_carmichael_numbers_in_range(x, y)

    @staticmethod
    def find_first_n_carmichaels(n):
        sympy_deprecation_warning(
        """
        find_first_n_carmichaels is just a wrapper around sympy.ntheory.factor_.find_first_n_carmichaels so use that
        directly instead.
        """,
        deprecated_since_version="1.13",
        active_deprecations_target='deprecated-ntheory-symbolic-functions',
        )
        return find_first_n_carmichaels(n)



#----------------------------------------------------------------------------#
#                                                                            #
#                           Fibonacci numbers                                #
#                                                                            #
#----------------------------------------------------------------------------#

class fibonacci(Function):
    r"""
    Fibonacci numbers / Fibonacci polynomials

    The Fibonacci numbers are the integer sequence defined by the
    initial terms `F_0 = 0`, `F_1 = 1` and the two-term recurrence
    relation `F_n = F_{n-1} + F_{n-2}`.  This definition
    extended to arbitrary real and complex arguments using
    the formula

    .. math :: F_z = \frac{\phi^z - \cos(\pi z) \phi^{-z}}{\sqrt 5}


注释：
这部分代码定义了一个名为`fibonacci`的类，继承自`Function`。它实现了斐波那契数列和斐波那契多项式的计算。斐波那契数列是一个整数序列，起始于 `F_0 = 0` 和 `F_1 = 1`，后续项通过递推关系 `F_n = F_{n-1} + F_{n-2}` 计算。这个定义扩展到任意实数和复数参数使用公式 `F_z = \frac{\phi^z - \cos(\pi z) \phi^{-z}}{\sqrt 5}`。
    """
    The Fibonacci polynomials are defined by `F_1(x) = 1`,
    `F_2(x) = x`, and `F_n(x) = x*F_{n-1}(x) + F_{n-2}(x)` for `n > 2`.
    For all positive integers `n`, `F_n(1) = F_n`.

    * ``fibonacci(n)`` gives the `n^{th}` Fibonacci number, `F_n`
    * ``fibonacci(n, x)`` gives the `n^{th}` Fibonacci polynomial in `x`, `F_n(x)`

    Examples
    ========

    >>> from sympy import fibonacci, Symbol

    >>> [fibonacci(x) for x in range(11)]
    [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    >>> fibonacci(5, Symbol('t'))
    t**4 + 3*t**2 + 1

    See Also
    ========

    bell, bernoulli, catalan, euler, harmonic, lucas, genocchi, partition, tribonacci

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Fibonacci_number
    .. [2] https://mathworld.wolfram.com/FibonacciNumber.html

    """

    @staticmethod
    # 静态方法，用于计算第n个Fibonacci数
    def _fib(n):
        return _ifib(n)

    @staticmethod
    @recurrence_memo([None, S.One, _sym])
    # 静态方法，使用递归记忆化技术计算第n个Fibonacci多项式
    def _fibpoly(n, prev):
        return (prev[-2] + _sym*prev[-1]).expand()

    @classmethod
    # 类方法，计算第n个Fibonacci数或者Fibonacci多项式，根据是否传入符号参数决定
    def eval(cls, n, sym=None):
        if n is S.Infinity:
            return S.Infinity

        if n.is_Integer:
            if sym is None:
                n = int(n)
                if n < 0:
                    return S.NegativeOne**(n + 1) * fibonacci(-n)
                else:
                    return Integer(cls._fib(n))
            else:
                if n < 1:
                    raise ValueError("Fibonacci polynomials are defined "
                       "only for positive integer indices.")
                return cls._fibpoly(n).subs(_sym, sym)

    # 将表达式重写为更容易处理的形式，使用黄金比例和平方根函数
    def _eval_rewrite_as_tractable(self, n, **kwargs):
        from sympy.functions import sqrt, cos
        return (S.GoldenRatio**n - cos(S.Pi*n)/S.GoldenRatio**n)/sqrt(5)

    # 将表达式重写为使用平方根函数的形式
    def _eval_rewrite_as_sqrt(self, n, **kwargs):
        from sympy.functions.elementary.miscellaneous import sqrt
        return 2**(-n)*sqrt(5)*((1 + sqrt(5))**n - (-sqrt(5) + 1)**n) / 5

    # 将表达式重写为使用黄金比例的形式
    def _eval_rewrite_as_GoldenRatio(self,n, **kwargs):
        return (S.GoldenRatio**n - 1/(-S.GoldenRatio)**n)/(2*S.GoldenRatio-1)
class lucas(Function):
    """
    Lucas numbers

    Lucas numbers satisfy a recurrence relation similar to that of
    the Fibonacci sequence, in which each term is the sum of the
    preceding two. They are generated by choosing the initial
    values `L_0 = 2` and `L_1 = 1`.

    * ``lucas(n)`` gives the `n^{th}` Lucas number

    Examples
    ========

    >>> from sympy import lucas

    >>> [lucas(x) for x in range(11)]
    [2, 1, 3, 4, 7, 11, 18, 29, 47, 76, 123]

    See Also
    ========

    bell, bernoulli, catalan, euler, fibonacci, harmonic, genocchi, partition, tribonacci

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Lucas_number
    .. [2] https://mathworld.wolfram.com/LucasNumber.html

    """

    @classmethod
    def eval(cls, n):
        # 如果 n 是无穷大，则返回无穷大
        if n is S.Infinity:
            return S.Infinity

        # 如果 n 是整数，计算第 n 个 Lucas 数
        if n.is_Integer:
            return fibonacci(n + 1) + fibonacci(n - 1)

    def _eval_rewrite_as_sqrt(self, n, **kwargs):
        # 导入平方根函数
        from sympy.functions.elementary.miscellaneous import sqrt
        # 返回重写的表达式，使用平方根和指数运算
        return 2**(-n)*((1 + sqrt(5))**n + (-sqrt(5) + 1)**n)



class tribonacci(Function):
    r"""
    Tribonacci numbers / Tribonacci polynomials

    The Tribonacci numbers are the integer sequence defined by the
    initial terms `T_0 = 0`, `T_1 = 1`, `T_2 = 1` and the three-term
    recurrence relation `T_n = T_{n-1} + T_{n-2} + T_{n-3}`.

    The Tribonacci polynomials are defined by `T_0(x) = 0`, `T_1(x) = 1`,
    `T_2(x) = x^2`, and `T_n(x) = x^2 T_{n-1}(x) + x T_{n-2}(x) + T_{n-3}(x)`
    for `n > 2`.  For all positive integers `n`, `T_n(1) = T_n`.

    * ``tribonacci(n)`` gives the `n^{th}` Tribonacci number, `T_n`
    * ``tribonacci(n, x)`` gives the `n^{th}` Tribonacci polynomial in `x`, `T_n(x)`

    Examples
    ========

    >>> from sympy import tribonacci, Symbol

    >>> [tribonacci(x) for x in range(11)]
    [0, 1, 1, 2, 4, 7, 13, 24, 44, 81, 149]
    >>> tribonacci(5, Symbol('t'))
    t**8 + 3*t**5 + 3*t**2

    See Also
    ========

    bell, bernoulli, catalan, euler, fibonacci, harmonic, lucas, genocchi, partition

    References
    ==========

    """
    """
    通过静态方法计算第 n 个 Tribonacci 数，使用递归记忆化技术优化性能
    """
    @staticmethod
    @recurrence_memo([S.Zero, S.One, S.One])
    def _trib(n, prev):
        return (prev[-3] + prev[-2] + prev[-1])

    """
    通过静态方法计算第 n 个 Tribonacci 多项式，使用递归记忆化技术扩展
    """
    @staticmethod
    @recurrence_memo([S.Zero, S.One, _sym**2])
    def _tribpoly(n, prev):
        return (prev[-3] + _sym*prev[-2] + _sym**2*prev[-1]).expand()

    """
    计算 Tribonacci 数或 Tribonacci 多项式的值
    如果 n 是无穷大，返回无穷大
    如果 n 是整数，根据是否定义了符号变量 sym 进行计算
    """
    @classmethod
    def eval(cls, n, sym=None):
        if n is S.Infinity:
            return S.Infinity

        if n.is_Integer:
            n = int(n)
            if n < 0:
                raise ValueError("Tribonacci polynomials are defined "
                       "only for non-negative integer indices.")
            if sym is None:
                return Integer(cls._trib(n))
            else:
                return cls._tribpoly(n).subs(_sym, sym)

    """
    以 sqrt 形式重写 Tribonacci 数的表达式
    """
    def _eval_rewrite_as_sqrt(self, n, **kwargs):
        from sympy.functions.elementary.miscellaneous import cbrt, sqrt
        w = (-1 + S.ImaginaryUnit * sqrt(3)) / 2
        a = (1 + cbrt(19 + 3*sqrt(33)) + cbrt(19 - 3*sqrt(33))) / 3
        b = (1 + w*cbrt(19 + 3*sqrt(33)) + w**2*cbrt(19 - 3*sqrt(33))) / 3
        c = (1 + w**2*cbrt(19 + 3*sqrt(33)) + w*cbrt(19 - 3*sqrt(33))) / 3
        Tn = (a**(n + 1)/((a - b)*(a - c))
            + b**(n + 1)/((b - a)*(b - c))
            + c**(n + 1)/((c - a)*(c - b)))
        return Tn

    """
    以 Tribonacci 常数形式重写 Tribonacci 数的表达式
    """
    def _eval_rewrite_as_TribonacciConstant(self, n, **kwargs):
        from sympy.functions.elementary.integers import floor
        from sympy.functions.elementary.miscellaneous import cbrt, sqrt
        b = cbrt(586 + 102*sqrt(33))
        Tn = 3 * b * S.TribonacciConstant**n / (b**2 - 2*b + 4)
        return floor(Tn + S.Half)
# 定义一个自定义函数 `bernoulli`，继承于 `Function` 类
class bernoulli(Function):
    r"""
    Bernoulli numbers / Bernoulli polynomials / Bernoulli function
    
    Bernoulli 数 / Bernoulli 多项式 / Bernoulli 函数

    The Bernoulli numbers are a sequence of rational numbers
    defined by `B_0 = 1` and the recursive relation (`n > 0`):
    
    Bernoulli 数是一系列有理数，由 `B_0 = 1` 和递归关系式 (`n > 0`) 定义：
    
    .. math :: n+1 = \sum_{k=0}^n \binom{n+1}{k} B_k
    
    They are also commonly defined by their exponential generating
    function, which is `\frac{x}{1 - e^{-x}}`. For odd indices > 1,
    the Bernoulli numbers are zero.
    
    通常也可以用它们的指数生成函数来定义，即 `\frac{x}{1 - e^{-x}}`。对于奇数索引大于 1 的情况，Bernoulli 数为零。
    
    The Bernoulli polynomials satisfy the analogous formula:
    
    Bernoulli 多项式满足类似的公式：
    
    .. math :: B_n(x) = \sum_{k=0}^n (-1)^k \binom{n}{k} B_k x^{n-k}
    
    Bernoulli numbers and Bernoulli polynomials are related as
    `B_n(1) = B_n`.
    
    Bernoulli 数和 Bernoulli 多项式有如下关系：`B_n(1) = B_n`。
    
    The generalized Bernoulli function `\operatorname{B}(s, a)`
    is defined for any complex `s` and `a`, except where `a` is a
    nonpositive integer and `s` is not a nonnegative integer. It is
    an entire function of `s` for fixed `a`, related to the Hurwitz
    zeta function by
    
    广义 Bernoulli 函数 `\operatorname{B}(s, a)` 定义于任意复数 `s` 和 `a`，除非 `a` 是非正整数且 `s` 不是非负整数。它是 `s` 的整函数，对于固定的 `a`，与 Hurwitz zeta 函数有关：
    
    .. math:: \operatorname{B}(s, a) = \begin{cases}
              -s \zeta(1-s, a) & s \ne 0 \\ 1 & s = 0 \end{cases}
    
    When `s` is a nonnegative integer this function reduces to the
    Bernoulli polynomials: `\operatorname{B}(n, x) = B_n(x)`. When
    `a` is omitted it is assumed to be 1, yielding the (ordinary)
    Bernoulli function which interpolates the Bernoulli numbers and is
    related to the Riemann zeta function.
    
    当 `s` 是非负整数时，此函数退化为 Bernoulli 多项式：`\operatorname{B}(n, x) = B_n(x)`。当省略 `a` 时，默认为 1，得到插值 Bernoulli 数的 (普通) Bernoulli 函数，并与 Riemann zeta 函数有关。
    
    We compute Bernoulli numbers using Ramanujan's formula:
    
    使用 Ramanujan 公式计算 Bernoulli 数：
    
    .. math :: B_n = \frac{A(n) - S(n)}{\binom{n+3}{n}}
    
    where:
    
    其中：
    
    .. math :: A(n) = \begin{cases} \frac{n+3}{3} &
        n \equiv 0\ \text{or}\ 2 \pmod{6} \\
        -\frac{n+3}{6} & n \equiv 4 \pmod{6} \end{cases}
    
    and:
    
    和：
    
    .. math :: S(n) = \sum_{k=1}^{[n/6]} \binom{n+3}{n-6k} B_{n-6k}
    
    This formula is similar to the sum given in the definition, but
    cuts `\frac{2}{3}` of the terms. For Bernoulli polynomials, we use
    Appell sequences.
    
    此公式类似于定义中给出的总和，但削减了 `\frac{2}{3}` 的项。对于 Bernoulli 多项式，我们使用 Appell 序列。
    
    For `n` a nonnegative integer and `s`, `a`, `x` arbitrary complex numbers,
    
    当 `n` 是非负整数，而 `s`, `a`, `x` 是任意复数时，
    
    * ``bernoulli(n)`` gives the nth Bernoulli number, `B_n`
    * ``bernoulli(s)`` gives the Bernoulli function `\operatorname{B}(s)`
    * ``bernoulli(n, x)`` gives the nth Bernoulli polynomial in `x`, `B_n(x)`
    * ``bernoulli(s, a)`` gives the generalized Bernoulli function
      `\operatorname{B}(s, a)`
    * ``bernoulli(n)`` 返回第 n 个 Bernoulli 数 `B_n`
    * ``bernoulli(s)`` 返回 Bernoulli 函数 `\operatorname{B}(s)`
    * ``bernoulli(n, x)`` 返回 x 中的第 n 个 Bernoulli 多项式 `B_n(x)`
    * ``bernoulli(s, a)`` 返回广义 Bernoulli 函数 `\operatorname{B}(s, a)`
    args: tTuple[Integer]

    # 计算正偶数 n 对应的伯努利数 B_n
    @staticmethod
    def _calc_bernoulli(n):
        # 初始化求和变量
        s = 0
        # 初始 binomial(n + 3, n - 6) 作为系数 a
        a = int(binomial(n + 3, n - 6))
        # 遍历范围为 1 到 n//6，每次增加 6
        for j in range(1, n//6 + 1):
            # 将 a 乘以 bernoulli(n - 6*j)，累加到 s
            s += a * bernoulli(n - 6*j)
            # 避免每次从头计算二项式系数
            a *= _product(n - 6 - 6*j + 1, n - 6*j)
            a //= _product(6*j + 4, 6*j + 9)
        # 根据 n % 6 的余数来调整最终的 s
        if n % 6 == 4:
            s = -Rational(n + 3, 6) - s
        else:
            s = Rational(n + 3, 3) - s
        # 返回伯努利数 B_n，通过 binomial(n + 3, n) 进行归一化
        return s / binomial(n + 3, n)

    # 我们实现了一个专门的记忆化方案，以分别处理每个模 6 的情况
    _cache = {0: S.One, 1: Rational(1, 2), 2: Rational(1, 6), 4: Rational(-1, 30)}
    _highest = {0: 0, 1: 1, 2: 2, 4: 4}

    @classmethod
    # 定义一个类方法 eval，用于计算伯努利数或伯努利多项式
    def eval(cls, n, x=None):
        # 如果 x 是 S.One，返回类的实例化对象
        if x is S.One:
            return cls(n)
        # 如果 n 是零，返回 S.One
        elif n.is_zero:
            return S.One
        # 如果 n 不是整数或非负数，则根据 x 的情况返回 S.NaN 或者空值
        elif n.is_integer is False or n.is_nonnegative is False:
            if x is not None and x.is_Integer and x.is_nonpositive:
                return S.NaN
            return
        # 计算伯努利数
        elif x is None:
            if n is S.One:
                return S.Half
            elif n.is_odd and (n-1).is_positive:
                return S.Zero
            elif n.is_Number:
                n = int(n)
                # 对于大的伯努利数，使用 mpmath 计算
                if n > 500:
                    p, q = mp.bernfrac(n)
                    return Rational(int(p), int(q))
                case = n % 6
                highest_cached = cls._highest[case]
                if n <= highest_cached:
                    return cls._cache[n]
                # 避免过多递归，计算并缓存整个序列的伯努利数
                for i in range(highest_cached + 6, n + 6, 6):
                    b = cls._calc_bernoulli(i)
                    cls._cache[i] = b
                    cls._highest[case] = i
                return b
        # 计算伯努利多项式
        elif n.is_Number:
            return bernoulli_poly(n, x)

    # 重写函数，将其表达为黎曼 zeta 函数的形式
    def _eval_rewrite_as_zeta(self, n, x=1, **kwargs):
        from sympy.functions.special.zeta_functions import zeta
        return Piecewise((1, Eq(n, 0)), (-n * zeta(1-n, x), True))

    # 对象的 evalf 方法，将其数值化为给定精度
    def _eval_evalf(self, prec):
        # 如果所有参数都是数字
        if not all(x.is_number for x in self.args):
            return
        # 将第一个参数 n 转换为 mpmath 的精确表示
        n = self.args[0]._to_mpmath(prec)
        # 将第二个参数 x 转换为 mpmath 的精确表示，如果没有则为 S.One
        x = (self.args[1] if len(self.args) > 1 else S.One)._to_mpmath(prec)
        with workprec(prec):
            # 根据 n 的值计算相应的数值
            if n == 0:
                res = mp.mpf(1)
            elif n == 1:
                res = x - mp.mpf(0.5)
            elif mp.isint(n) and n >= 0:
                res = mp.bernoulli(n) if x == 1 else mp.bernpoly(n, x)
            else:
                res = -n * mp.zeta(1-n, x)
        # 返回 mpmath 结果的 sympy 表达式
        return Expr._from_mpmath(res, prec)
# 定义一个类 `bell`，继承自 `Function` 类，用于计算贝尔数和贝尔多项式
class bell(Function):
    r"""
    Bell numbers / Bell polynomials

    The Bell numbers satisfy `B_0 = 1` and

    .. math:: B_n = \sum_{k=0}^{n-1} \binom{n-1}{k} B_k.

    They are also given by:

    .. math:: B_n = \frac{1}{e} \sum_{k=0}^{\infty} \frac{k^n}{k!}.

    The Bell polynomials are given by `B_0(x) = 1` and

    .. math:: B_n(x) = x \sum_{k=1}^{n-1} \binom{n-1}{k-1} B_{k-1}(x).

    The second kind of Bell polynomials (are sometimes called "partial" Bell
    polynomials or incomplete Bell polynomials) are defined as

    .. math:: B_{n,k}(x_1, x_2,\dotsc x_{n-k+1}) =
            \sum_{j_1+j_2+j_2+\dotsb=k \atop j_1+2j_2+3j_2+\dotsb=n}
                \frac{n!}{j_1!j_2!\dotsb j_{n-k+1}!}
                \left(\frac{x_1}{1!} \right)^{j_1}
                \left(\frac{x_2}{2!} \right)^{j_2} \dotsb
                \left(\frac{x_{n-k+1}}{(n-k+1)!} \right) ^{j_{n-k+1}}.

    * ``bell(n)`` gives the `n^{th}` Bell number, `B_n`.
    * ``bell(n, x)`` gives the `n^{th}` Bell polynomial, `B_n(x)`.
    * ``bell(n, k, (x1, x2, ...))`` gives Bell polynomials of the second kind,
      `B_{n,k}(x_1, x_2, \dotsc, x_{n-k+1})`.

    Notes
    =====

    Not to be confused with Bernoulli numbers and Bernoulli polynomials,
    which use the same notation.

    Examples
    ========

    >>> from sympy import bell, Symbol, symbols

    >>> [bell(n) for n in range(11)]
    [1, 1, 2, 5, 15, 52, 203, 877, 4140, 21147, 115975]
    >>> bell(30)
    846749014511809332450147
    >>> bell(4, Symbol('t'))
    t**4 + 6*t**3 + 7*t**2 + t
    >>> bell(6, 2, symbols('x:6')[1:])
    6*x1*x5 + 15*x2*x4 + 10*x3**2

    See Also
    ========

    bernoulli, catalan, euler, fibonacci, harmonic, lucas, genocchi, partition, tribonacci

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Bell_number
    .. [2] https://mathworld.wolfram.com/BellNumber.html
    .. [3] https://mathworld.wolfram.com/BellPolynomial.html

    """

    @staticmethod
    @recurrence_memo([1, 1])
    # 静态方法 `_bell` 计算第 `n` 个贝尔数，使用了递归记忆化技术
    def _bell(n, prev):
        s = 1
        a = 1
        for k in range(1, n):
            a = a * (n - k) // k
            s += a * prev[k]
        return s

    @staticmethod
    @recurrence_memo([S.One, _sym])
    # 静态方法 `_bell_poly` 计算第 `n` 个贝尔多项式，使用了递归记忆化技术
    def _bell_poly(n, prev):
        s = 1
        a = 1
        for k in range(2, n + 1):
            a = a * (n - k + 1) // (k - 1)
            s += a * prev[k - 1]
        return expand_mul(_sym * s)

    @staticmethod
    def _bell_incomplete_poly(n, k, symbols):
        r"""
        The second kind of Bell polynomials (incomplete Bell polynomials).

        Calculated by recurrence formula:

        .. math:: B_{n,k}(x_1, x_2, \dotsc, x_{n-k+1}) =
                \sum_{m=1}^{n-k+1}
                \x_m \binom{n-1}{m-1} B_{n-m,k-1}(x_1, x_2, \dotsc, x_{n-m-k})

        where
            `B_{0,0} = 1;`
            `B_{n,0} = 0; for n \ge 1`
            `B_{0,k} = 0; for k \ge 1`

        """
        # 处理特殊情况：当 n 和 k 均为零时，返回 Bell 多项式的单位值
        if (n == 0) and (k == 0):
            return S.One
        # 处理特殊情况：当 n 或 k 为零时，返回 Bell 多项式的零值
        elif (n == 0) or (k == 0):
            return S.Zero
        # 初始化求和变量 s 和系数 a
        s = S.Zero
        a = S.One
        # 循环计算 Bell 多项式的递归表达式
        for m in range(1, n - k + 2):
            s += a * bell._bell_incomplete_poly(
                n - m, k - 1, symbols) * symbols[m - 1]
            a = a * (n - m) / m
        # 展开和式并返回结果
        return expand_mul(s)

    @classmethod
    def eval(cls, n, k_sym=None, symbols=None):
        # 处理 n 为正无穷的情况
        if n is S.Infinity:
            if k_sym is None:
                return S.Infinity
            else:
                raise ValueError("Bell polynomial is not defined")

        # 检查 n 是否为负数或者非整数
        if n.is_negative or n.is_integer is False:
            raise ValueError("a non-negative integer expected")

        # 当 n 为整数且非负时，根据参数的不同情况返回对应的 Bell 多项式结果
        if n.is_Integer and n.is_nonnegative:
            if k_sym is None:
                return Integer(cls._bell(int(n)))
            elif symbols is None:
                return cls._bell_poly(int(n)).subs(_sym, k_sym)
            else:
                r = cls._bell_incomplete_poly(int(n), int(k_sym), symbols)
                return r

    def _eval_rewrite_as_Sum(self, n, k_sym=None, symbols=None, **kwargs):
        from sympy.concrete.summations import Sum
        # 当 k_sym 或 symbols 不为 None 时，不进行重写
        if (k_sym is not None) or (symbols is not None):
            return self

        # 使用 Dobinski's formula 将 Bell 多项式重写为级数形式
        if not n.is_nonnegative:
            return self
        # 定义一个虚拟变量 k，返回 Bell 多项式的级数表示
        k = Dummy('k', integer=True, nonnegative=True)
        return 1 / E * Sum(k**n / factorial(k), (k, 0, S.Infinity))
# 定义一个类 `harmonic`，继承自 `Function`
class harmonic(Function):
    r"""
    Harmonic numbers

    The nth harmonic number is given by `\operatorname{H}_{n} =
    1 + \frac{1}{2} + \frac{1}{3} + \ldots + \frac{1}{n}`.

    More generally:

    .. math:: \operatorname{H}_{n,m} = \sum_{k=1}^{n} \frac{1}{k^m}

    As `n \rightarrow \infty`, `\operatorname{H}_{n,m} \rightarrow \zeta(m)`,
    the Riemann zeta function.

    * ``harmonic(n)`` gives the nth harmonic number, `\operatorname{H}_n`

    * ``harmonic(n, m)`` gives the nth generalized harmonic number
      of order `m`, `\operatorname{H}_{n,m}`, where
      ``harmonic(n) == harmonic(n, 1)``

    This function can be extended to complex `n` and `m` where `n` is not a
    negative integer or `m` is a nonpositive integer as

    .. math:: \operatorname{H}_{n,m} = \begin{cases} \zeta(m) - \zeta(m, n+1)
            & m \ne 1 \\ \psi(n+1) + \gamma & m = 1 \end{cases}

    Examples
    ========

    >>> from sympy import harmonic, oo

    >>> [harmonic(n) for n in range(6)]
    [0, 1, 3/2, 11/6, 25/12, 137/60]
    >>> [harmonic(n, 2) for n in range(6)]
    [0, 1, 5/4, 49/36, 205/144, 5269/3600]
    >>> harmonic(oo, 2)
    pi**2/6

    >>> from sympy import Symbol, Sum
    >>> n = Symbol("n")

    >>> harmonic(n).rewrite(Sum)
    Sum(1/_k, (_k, 1, n))

    We can evaluate harmonic numbers for all integral and positive
    rational arguments:

    >>> from sympy import S, expand_func, simplify
    >>> harmonic(8)
    761/280
    >>> harmonic(11)
    83711/27720

    >>> H = harmonic(1/S(3))
    >>> H
    harmonic(1/3)
    >>> He = expand_func(H)
    >>> He
    -log(6) - sqrt(3)*pi/6 + 2*Sum(log(sin(_k*pi/3))*cos(2*_k*pi/3), (_k, 1, 1))
                           + 3*Sum(1/(3*_k + 1), (_k, 0, 0))
    >>> He.doit()
    -log(6) - sqrt(3)*pi/6 - log(sqrt(3)/2) + 3
    >>> H = harmonic(25/S(7))
    >>> He = simplify(expand_func(H).doit())
    >>> He
    log(sin(2*pi/7)**(2*cos(16*pi/7))/(14*sin(pi/7)**(2*cos(pi/7))*cos(pi/14)**(2*sin(pi/14)))) + pi*tan(pi/14)/2 + 30247/9900
    >>> He.n(40)
    1.983697455232980674869851942390639915940
    >>> harmonic(25/S(7)).n(40)
    1.983697455232980674869851942390639915940

    We can rewrite harmonic numbers in terms of polygamma functions:

    >>> from sympy import digamma, polygamma
    >>> m = Symbol("m", integer=True, positive=True)

    >>> harmonic(n).rewrite(digamma)
    polygamma(0, n + 1) + EulerGamma

    >>> harmonic(n).rewrite(polygamma)
    polygamma(0, n + 1) + EulerGamma

    >>> harmonic(n,3).rewrite(polygamma)
    polygamma(2, n + 1)/2 + zeta(3)
    ```
    # 对 `harmonic(n, m)` 进行多项式伽玛函数的重写，得到简化的表达式
    simplify(harmonic(n,m).rewrite(polygamma))

    # 在参数中可以拉出整数偏移量：
    from sympy import expand_func

    # 将 `harmonic(n+4)` 展开为和的形式
    expand_func(harmonic(n+4))

    # 将 `harmonic(n-4)` 展开为和的形式
    expand_func(harmonic(n-4))

    # 计算一些极限：
    from sympy import limit, oo

    # 当 `n` 趋向于无穷大时，调和数 `harmonic(n)` 的极限
    limit(harmonic(n), n, oo)

    # 当 `n` 趋向于无穷大时，带有指数 `2` 的调和数 `harmonic(n, 2)` 的极限
    limit(harmonic(n, 2), n, oo)

    # 当 `n` 趋向于无穷大时，带有指数 `3` 的调和数 `harmonic(n, 3)` 的极限
    limit(harmonic(n, 3), n, oo)

    # 对于 `m > 1`，当 `n` 趋向于无穷大时，调和数 `harmonic(n, m+1)` 趋向于黎曼ζ函数的极限
    m = Symbol("m", positive=True)
    limit(harmonic(n, m+1), n, oo)

    # 参见
    # bell, bernoulli, catalan, euler, fibonacci, lucas, genocchi, partition, tribonacci

    # 参考文献
    # [1] https://en.wikipedia.org/wiki/Harmonic_number
    # [2] https://functions.wolfram.com/GammaBetaErf/HarmonicNumber/
    # [3] https://functions.wolfram.com/GammaBetaErf/HarmonicNumber2/

    @classmethod
    # 类方法，用于计算调和数的特定形式
    def eval(cls, n, m=None):
        from sympy.functions.special.zeta_functions import zeta
        # 当 m 为 1 时，返回调和数 `cls(n)`
        if m is S.One:
            return cls(n)
        # 如果 m 为 None，则设为 1
        if m is None:
            m = S.One
        # 当 n 为零时，返回 0
        if n.is_zero:
            return S.Zero
        # 当 m 为零时，返回 n
        elif m.is_zero:
            return n
        # 当 n 趋向于无穷大时，根据 m 的情况返回不同的值
        elif n is S.Infinity:
            if m.is_negative:
                return S.NaN
            elif is_le(m, S.One):
                return S.Infinity
            elif is_gt(m, S.One):
                return zeta(m)
        # 当 m 为整数且不大于零时，根据 Bernoulli 数计算
        elif m.is_Integer and m.is_nonpositive:
            return (bernoulli(1-m, n+1) - bernoulli(1-m)) / (1-m)
        # 当 n 为整数时，根据不同情况返回调和数的表达式
        elif n.is_Integer:
            if n.is_negative and (m.is_integer is False or m.is_nonpositive is False):
                return S.ComplexInfinity if m is S.One else S.NaN
            if n.is_nonnegative:
                return Add(*(k**(-m) for k in range(1, int(n)+1)))

    # 将调和数重写为多项式伽玛函数的形式
    def _eval_rewrite_as_polygamma(self, n, m=S.One, **kwargs):
        from sympy.functions.special.gamma_functions import gamma, polygamma
        if m.is_integer and m.is_positive:
            return Piecewise((polygamma(0, n+1) + S.EulerGamma, Eq(m, 1)),
                    (S.NegativeOne**m * (polygamma(m-1, 1) - polygamma(m-1, n+1)) /
                    gamma(m), True))

    # 将调和数重写为对数伽玛函数的形式
    def _eval_rewrite_as_digamma(self, n, m=1, **kwargs):
        from sympy.functions.special.gamma_functions import polygamma
        return self.rewrite(polygamma)

    # 将调和数重写为三次伽玛函数的形式
    def _eval_rewrite_as_trigamma(self, n, m=1, **kwargs):
        from sympy.functions.special.gamma_functions import polygamma
        return self.rewrite(polygamma)
    # 将当前对象表示为一个求和表达式的重写形式
    def _eval_rewrite_as_Sum(self, n, m=None, **kwargs):
        # 导入求和相关模块
        from sympy.concrete.summations import Sum
        # 创建一个整数虚拟变量 k
        k = Dummy("k", integer=True)
        # 如果未提供 m 参数，则默认为 1
        if m is None:
            m = S.One
        # 返回一个 k**(-m) 的求和表达式，范围从 k=1 到 k=n
        return Sum(k**(-m), (k, 1, n))

    # 将当前对象表示为黎曼 zeta 函数的重写形式
    def _eval_rewrite_as_zeta(self, n, m=S.One, **kwargs):
        # 导入 zeta 函数和 digamma 函数
        from sympy.functions.special.zeta_functions import zeta
        from sympy.functions.special.gamma_functions import digamma
        # 根据条件返回不同的表达式：当 m=1 时返回 digamma(n + 1) + EulerGamma，否则返回 zeta(m) - zeta(m, n+1)
        return Piecewise((digamma(n + 1) + S.EulerGamma, Eq(m, 1)),
                         (zeta(m) - zeta(m, n+1), True))

    # 将当前对象表示为一种展开的函数形式
    def _eval_expand_func(self, **hints):
        # 导入求和相关模块
        from sympy.concrete.summations import Sum
        # 获取函数的参数
        n = self.args[0]
        m = self.args[1] if len(self.args) == 2 else 1

        # 处理 m=1 的情况
        if m == S.One:
            # 当 n 是加法表达式时
            if n.is_Add:
                # 将第一个参数作为偏移量 off
                off = n.args[0]
                # 计算新的 nnew
                nnew = n - off
                # 如果 off 是正整数
                if off.is_Integer and off.is_positive:
                    # 构造一个列表，包含一系列表达式
                    result = [S.One/(nnew + i) for i in range(off, 0, -1)] + [harmonic(nnew)]
                    return Add(*result)
                # 如果 off 是负整数
                elif off.is_Integer and off.is_negative:
                    # 构造另一个列表，包含一系列表达式
                    result = [-S.One/(nnew + i) for i in range(0, off, -1)] + [harmonic(nnew)]
                    return Add(*result)

            # 当 n 是有理数时
            if n.is_Rational:
                # 处理一般有理参数下的谐和数展开
                # 将 n 分解为 u + p/q 形式，其中 p < q
                p, q = n.as_numer_denom()
                u = p // q
                p = p - u * q
                # 如果 u 是非负整数，p 和 q 是正整数，并且 p < q
                if u.is_nonnegative and p.is_positive and q.is_positive and p < q:
                    # 导入相关函数
                    from sympy.functions.elementary.exponential import log
                    from sympy.functions.elementary.integers import floor
                    from sympy.functions.elementary.trigonometric import sin, cos, cot
                    # 创建虚拟变量 k
                    k = Dummy("k")
                    # 构造展开表达式的各个部分
                    t1 = q * Sum(1 / (q * k + p), (k, 0, u))
                    t2 = 2 * Sum(cos((2 * pi * p * k) / S(q)) *
                                   log(sin((pi * k) / S(q))),
                                   (k, 1, floor((q - 1) / S(2))))
                    t3 = (pi / 2) * cot((pi * p) / q) + log(2 * q)
                    # 返回这些部分的和
                    return t1 + t2 - t3

        # 默认返回对象本身
        return self

    # 将当前对象表示为一种 tractable 形式的重写形式
    def _eval_rewrite_as_tractable(self, n, m=1, limitvar=None, **kwargs):
        # 导入相关函数
        from sympy.functions.special.zeta_functions import zeta
        from sympy.functions.special.gamma_functions import polygamma
        # 将对象重写为 polygamma 函数的形式
        pg = self.rewrite(polygamma)
        # 如果不是 harmonic 类型的对象，则返回其 tractable 形式
        if not isinstance(pg, harmonic):
            return pg.rewrite("tractable", deep=True)
        # 计算 m-1
        arg = m - S.One
        # 如果 arg 不为零，则返回 zeta(m) - zeta(m, n+1) 的 tractable 形式
        if arg.is_nonzero:
            return (zeta(m) - zeta(m, n+1)).rewrite("tractable", deep=True)
    # 定义一个方法用于计算表达式的数值评估，返回精度为 `prec` 的计算结果
    def _eval_evalf(self, prec):
        # 检查所有参数是否均为数值类型
        if not all(x.is_number for x in self.args):
            return  # 如果有非数值参数，直接返回
        # 将第一个参数转换为指定精度的 mpmath 数字
        n = self.args[0]._to_mpmath(prec)
        # 如果参数列表长度大于1，将第二个参数赋值给 m；否则，默认为 S.One，并转换为 mpmath 数字
        m = (self.args[1] if len(self.args) > 1 else S.One)._to_mpmath(prec)
        # 如果 n 是负整数，返回 S.NaN（符号未定的数学表达式）
        if mp.isint(n) and n < 0:
            return S.NaN
        # 在指定精度下进行计算
        with workprec(prec):
            if m == 1:
                res = mp.harmonic(n)  # 计算调和数 harmonic(n)
            else:
                res = mp.zeta(m) - mp.zeta(m, n+1)  # 计算 Riemann zeta 函数的差
        # 将 mpmath 计算结果转换为 SymPy 表达式并返回
        return Expr._from_mpmath(res, prec)
    
    # 定义一个方法用于计算偏导数
    def fdiff(self, argindex=1):
        # 导入特殊函数库中的 zeta 函数
        from sympy.functions.special.zeta_functions import zeta
        # 根据参数个数确定 n 和 m 的值
        if len(self.args) == 2:
            n, m = self.args
        else:
            n, m = self.args + (1,)  # 如果参数不足两个，则将默认值 1 分配给 m
        # 如果 argindex 为 1，计算指定形式的偏导数
        if argindex == 1:
            return m * zeta(m+1, n+1)
        else:
            raise ArgumentIndexError  # 抛出参数索引错误异常
# 定义 Euler 类，继承自 Function 类
class euler(Function):
    r"""
    Euler numbers / Euler polynomials / Euler function

    The Euler numbers are given by:

    .. math:: E_{2n} = I \sum_{k=1}^{2n+1} \sum_{j=0}^k \binom{k}{j}
        \frac{(-1)^j (k-2j)^{2n+1}}{2^k I^k k}

    .. math:: E_{2n+1} = 0

    Euler numbers and Euler polynomials are related by

    .. math:: E_n = 2^n E_n\left(\frac{1}{2}\right).

    We compute symbolic Euler polynomials using Appell sequences,
    but numerical evaluation of the Euler polynomial is computed
    more efficiently (and more accurately) using the mpmath library.

    The Euler polynomials are special cases of the generalized Euler function,
    related to the Genocchi function as

    .. math:: \operatorname{E}(s, a) = -\frac{\operatorname{G}(s+1, a)}{s+1}

    with the limit of `\psi\left(\frac{a+1}{2}\right) - \psi\left(\frac{a}{2}\right)`
    being taken when `s = -1`. The (ordinary) Euler function interpolating
    the Euler numbers is then obtained as
    `\operatorname{E}(s) = 2^s \operatorname{E}\left(s, \frac{1}{2}\right)`.

    * ``euler(n)`` gives the nth Euler number `E_n`.
    * ``euler(s)`` gives the Euler function `\operatorname{E}(s)`.
    * ``euler(n, x)`` gives the nth Euler polynomial `E_n(x)`.
    * ``euler(s, a)`` gives the generalized Euler function `\operatorname{E}(s, a)`.

    Examples
    ========

    >>> from sympy import euler, Symbol, S
    >>> [euler(n) for n in range(10)]
    [1, 0, -1, 0, 5, 0, -61, 0, 1385, 0]
    >>> [2**n*euler(n,1) for n in range(10)]
    [1, 1, 0, -2, 0, 16, 0, -272, 0, 7936]
    >>> n = Symbol("n")
    >>> euler(n + 2*n)
    euler(3*n)

    >>> x = Symbol("x")
    >>> euler(n, x)
    euler(n, x)

    >>> euler(0, x)
    1
    >>> euler(1, x)
    x - 1/2
    >>> euler(2, x)
    x**2 - x
    >>> euler(3, x)
    x**3 - 3*x**2/2 + 1/4
    >>> euler(4, x)
    x**4 - 2*x**3 + x

    >>> euler(12, S.Half)
    2702765/4096
    >>> euler(12)
    2702765

    See Also
    ========

    andre, bell, bernoulli, catalan, fibonacci, harmonic, lucas, genocchi,
    partition, tribonacci, sympy.polys.appellseqs.euler_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Euler_numbers
    .. [2] https://mathworld.wolfram.com/EulerNumber.html
    .. [3] https://en.wikipedia.org/wiki/Alternating_permutation
    .. [4] https://mathworld.wolfram.com/AlternatingPermutation.html

    """

    @classmethod
    # 定义类方法 eval，用于计算特定函数在给定参数下的值
    def eval(cls, n, x=None):
        # 如果 n 是零，返回常数 1
        if n.is_zero:
            return S.One
        # 如果 n 是 -1
        elif n is S.NegativeOne:
            # 如果 x 为 None，返回 π/2
            if x is None:
                return S.Pi/2
            # 导入 gamma 函数模块中的 digamma 函数，并计算 digamma((x+1)/2) - digamma(x/2) 的值
            from sympy.functions.special.gamma_functions import digamma
            return digamma((x+1)/2) - digamma(x/2)
        # 如果 n 不是整数或者不是非负数，返回空值
        elif n.is_integer is False or n.is_nonnegative is False:
            return
        # 如果 n 是整数且非负数
        # 欧拉数
        elif x is None:
            # 如果 n 是奇数且为正数，返回常数 0
            if n.is_odd and n.is_positive:
                return S.Zero
            # 如果 n 是数字，从 mpmath 导入模块 mp，计算欧拉数并返回整数结果
            elif n.is_Number:
                from mpmath import mp
                n = n._to_mpmath(mp.prec)
                res = mp.eulernum(n, exact=True)
                return Integer(res)
        # 欧拉多项式
        elif n.is_Number:
            # 调用 euler_poly 函数计算欧拉多项式的值并返回
            return euler_poly(n, x)

    # 定义内部方法 _eval_rewrite_as_Sum，将表达式重写为 Sum 的形式
    def _eval_rewrite_as_Sum(self, n, x=None, **kwargs):
        # 导入 Sum 类
        from sympy.concrete.summations import Sum
        # 如果 x 为 None 并且 n 是偶数
        if x is None and n.is_even:
            # 定义两个虚拟变量 k 和 j
            k = Dummy("k", integer=True)
            j = Dummy("j", integer=True)
            # 将 n 除以 2
            n = n / 2
            # 计算 Em 的表达式
            Em = (S.ImaginaryUnit * Sum(Sum(binomial(k, j) * (S.NegativeOne**j *
                  (k - 2*j)**(2*n + 1)) /
                  (2**k*S.ImaginaryUnit**k * k), (j, 0, k)), (k, 1, 2*n + 1)))
            return Em
        # 如果 x 存在
        if x:
            # 定义虚拟变量 k
            k = Dummy("k", integer=True)
            # 返回 Sum 的表达式
            return Sum(binomial(n, k)*euler(k)/2**k*(x - S.Half)**(n - k), (k, 0, n))

    # 定义内部方法 _eval_rewrite_as_genocchi，将表达式重写为 genocchi 函数的形式
    def _eval_rewrite_as_genocchi(self, n, x=None, **kwargs):
        # 如果 x 为 None
        if x is None:
            # 返回分段函数 Piecewise 的表达式
            return Piecewise((S.Pi/2, Eq(n, -1)),
                             (-2**n * genocchi(n+1, S.Half) / (n+1), True))
        # 导入 digamma 函数
        from sympy.functions.special.gamma_functions import digamma
        # 返回分段函数 Piecewise 的表达式
        return Piecewise((digamma((x+1)/2) - digamma(x/2), Eq(n, -1)),
                         (-genocchi(n+1, x) / (n+1), True))

    # 定义内部方法 _eval_evalf，用于数值计算
    def _eval_evalf(self, prec):
        # 如果不是所有参数都是数字，返回空值
        if not all(i.is_number for i in self.args):
            return
        # 导入 mpmath 模块中的 mp
        from mpmath import mp
        # 获取参数列表中的第一个参数和第二个参数（如果存在）
        m, x = (self.args[0], None) if len(self.args) == 1 else self.args
        # 将 m 转换为 mpmath 中的精度表示
        m = m._to_mpmath(prec)
        # 如果 x 存在，则将 x 转换为 mpmath 中的精度表示
        if x is not None:
            x = x._to_mpmath(prec)
        # 设置工作精度为 prec
        with workprec(prec):
            # 如果 m 是整数且大于等于 0
            if mp.isint(m) and m >= 0:
                # 如果 x 为 None，计算欧拉数，否则计算欧拉多项式
                res = mp.eulernum(m) if x is None else mp.eulerpoly(m, x)
            else:
                # 如果 m 等于 -1
                if m == -1:
                    # 如果 x 为 None，返回 π，否则计算 digamma((x+1)/2) - digamma(x/2)
                    res = mp.pi if x is None else mp.digamma((x+1)/2) - mp.digamma(x/2)
                else:
                    # 否则，设置 y 为 0.5（如果 x 为 None），否则为 x
                    y = 0.5 if x is None else x
                    # 计算特定表达式的值
                    res = 2 * (mp.zeta(-m, y) - 2**(m+1) * mp.zeta(-m, (y+1)/2))
                    # 如果 x 为 None，乘以 2 的 m 次方
                    if x is None:
                        res *= 2**m
        # 从 mpmath 中返回 Expr 类的结果
        return Expr._from_mpmath(res, prec)
# 定义一个名为 catalan 的类，继承自 Function 类
class catalan(Function):
    """
    Catalan numbers

    The `n^{th}` catalan number is given by:

    .. math :: C_n = \frac{1}{n+1} \binom{2n}{n}

    * ``catalan(n)`` gives the `n^{th}` Catalan number, `C_n`

    Examples
    ========

    >>> from sympy import (Symbol, binomial, gamma, hyper,
    ...     catalan, diff, combsimp, Rational, I)

    >>> [catalan(i) for i in range(1,10)]
    [1, 2, 5, 14, 42, 132, 429, 1430, 4862]

    >>> n = Symbol("n", integer=True)

    >>> catalan(n)
    catalan(n)

    Catalan numbers can be transformed into several other, identical
    expressions involving other mathematical functions

    >>> catalan(n).rewrite(binomial)
    binomial(2*n, n)/(n + 1)

    >>> catalan(n).rewrite(gamma)
    4**n*gamma(n + 1/2)/(sqrt(pi)*gamma(n + 2))

    >>> catalan(n).rewrite(hyper)
    hyper((-n, 1 - n), (2,), 1)

    For some non-integer values of n we can get closed form
    expressions by rewriting in terms of gamma functions:

    >>> catalan(Rational(1, 2)).rewrite(gamma)
    8/(3*pi)

    We can differentiate the Catalan numbers C(n) interpreted as a
    continuous real function in n:

    >>> diff(catalan(n), n)
    (polygamma(0, n + 1/2) - polygamma(0, n + 2) + log(4))*catalan(n)

    As a more advanced example consider the following ratio
    between consecutive numbers:

    >>> combsimp((catalan(n + 1)/catalan(n)).rewrite(binomial))
    2*(2*n + 1)/(n + 2)

    The Catalan numbers can be generalized to complex numbers:

    >>> catalan(I).rewrite(gamma)
    4**I*gamma(1/2 + I)/(sqrt(pi)*gamma(2 + I))

    and evaluated with arbitrary precision:

    >>> catalan(I).evalf(20)
    0.39764993382373624267 - 0.020884341620842555705*I

    See Also
    ========

    andre, bell, bernoulli, euler, fibonacci, harmonic, lucas, genocchi,
    partition, tribonacci, sympy.functions.combinatorial.factorials.binomial

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Catalan_number
    .. [2] https://mathworld.wolfram.com/CatalanNumber.html
    .. [3] https://functions.wolfram.com/GammaBetaErf/CatalanNumber/
    .. [4] http://geometer.org/mathcircles/catalan.pdf

    """

    @classmethod
    # 定义一个类方法 eval，用于计算某个特定数值 n 的表达式值
    def eval(cls, n):
        # 导入 gamma 函数用于处理特定的数值情况
        from sympy.functions.special.gamma_functions import gamma
        # 检查 n 是否为整数且非负，或者为非整数且为负数
        if (n.is_Integer and n.is_nonnegative) or \
           (n.is_noninteger and n.is_negative):
            # 返回特定表达式的计算结果
            return 4**n*gamma(n + S.Half)/(gamma(S.Half)*gamma(n + 2))

        # 检查 n 是否为整数且为负数
        if (n.is_integer and n.is_negative):
            # 如果 n+1 是负数，返回 0
            if (n + 1).is_negative:
                return S.Zero
            # 如果 n+1 是零，返回 -1/2
            if (n + 1).is_zero:
                return Rational(-1, 2)

    # 定义一个方法 fdiff，用于对对象的第一个参数进行特定操作
    def fdiff(self, argindex=1):
        # 导入 log 和 polygamma 函数
        from sympy.functions.elementary.exponential import log
        from sympy.functions.special.gamma_functions import polygamma
        # 获取对象的第一个参数 n
        n = self.args[0]
        # 返回特定表达式的计算结果，涉及到 catalan 函数和 polygamma 函数的组合
        return catalan(n)*(polygamma(0, n + S.Half) - polygamma(0, n + 2) + log(4))

    # 定义一个方法 _eval_rewrite_as_binomial，将对象重写为二项式系数形式
    def _eval_rewrite_as_binomial(self, n, **kwargs):
        # 返回对象表达式重写为二项式系数形式的结果
        return binomial(2*n, n)/(n + 1)

    # 定义一个方法 _eval_rewrite_as_factorial，将对象重写为阶乘形式
    def _eval_rewrite_as_factorial(self, n, **kwargs):
        # 返回对象表达式重写为阶乘形式的结果
        return factorial(2*n) / (factorial(n+1) * factorial(n))

    # 定义一个方法 _eval_rewrite_as_gamma，将对象重写为 gamma 函数形式
    def _eval_rewrite_as_gamma(self, n, piecewise=True, **kwargs):
        # 导入 gamma 函数
        from sympy.functions.special.gamma_functions import gamma
        # 返回对象表达式重写为 gamma 函数形式的结果
        return 4**n*gamma(n + S.Half)/(gamma(S.Half)*gamma(n + 2))

    # 定义一个方法 _eval_rewrite_as_hyper，将对象重写为超几何函数形式
    def _eval_rewrite_as_hyper(self, n, **kwargs):
        # 导入超几何函数
        from sympy.functions.special.hyper import hyper
        # 返回对象表达式重写为超几何函数形式的结果
        return hyper([1 - n, -n], [2], 1)

    # 定义一个方法 _eval_rewrite_as_Product，将对象重写为乘积形式
    def _eval_rewrite_as_Product(self, n, **kwargs):
        # 导入 Product 类和 Dummy 变量
        from sympy.concrete.products import Product
        # 如果 n 不是整数或者是负数，返回对象本身
        if not (n.is_integer and n.is_nonnegative):
            return self
        # 创建一个整数和正数限定的 Dummy 变量 k
        k = Dummy('k', integer=True, positive=True)
        # 返回对象表达式重写为乘积形式的结果
        return Product((n + k) / k, (k, 2, n))

    # 定义一个方法 _eval_is_integer，用于判断对象是否为整数
    def _eval_is_integer(self):
        # 如果对象的第一个参数是整数且非负，返回 True
        if self.args[0].is_integer and self.args[0].is_nonnegative:
            return True

    # 定义一个方法 _eval_is_positive，用于判断对象是否为正数
    def _eval_is_positive(self):
        # 如果对象的第一个参数是非负数，返回 True
        if self.args[0].is_nonnegative:
            return True

    # 定义一个方法 _eval_is_composite，用于判断对象是否为合数
    def _eval_is_composite(self):
        # 如果对象的第一个参数是整数且减去 3 后是正数，返回 True
        if self.args[0].is_integer and (self.args[0] - 3).is_positive:
            return True

    # 定义一个方法 _eval_evalf，用于对对象进行数值估算
    def _eval_evalf(self, prec):
        # 导入 gamma 函数
        from sympy.functions.special.gamma_functions import gamma
        # 如果对象的第一个参数是数值，返回对象调用 gamma 重写后的数值估算结果
        if self.args[0].is_number:
            return self.rewrite(gamma)._eval_evalf(prec)
#----------------------------------------------------------------------------#
#                                                                            #
#                           Genocchi numbers                                 #
#                                                                            #
#----------------------------------------------------------------------------#


class genocchi(Function):
    r"""
    Genocchi numbers / Genocchi polynomials / Genocchi function

    The Genocchi numbers are a sequence of integers `G_n` that satisfy the
    relation:

    .. math:: \frac{-2t}{1 + e^{-t}} = \sum_{n=0}^\infty \frac{G_n t^n}{n!}

    They are related to the Bernoulli numbers by

    .. math:: G_n = 2 (1 - 2^n) B_n

    and generalize like the Bernoulli numbers to the Genocchi polynomials and
    function as

    .. math:: \operatorname{G}(s, a) = 2 \left(\operatorname{B}(s, a) -
              2^s \operatorname{B}\left(s, \frac{a+1}{2}\right)\right)

    .. versionchanged:: 1.12
        ``genocchi(1)`` gives `-1` instead of `1`.

    Examples
    ========

    >>> from sympy import genocchi, Symbol
    >>> [genocchi(n) for n in range(9)]
    [0, -1, -1, 0, 1, 0, -3, 0, 17]
    >>> n = Symbol('n', integer=True, positive=True)
    >>> genocchi(2*n + 1)
    0
    >>> x = Symbol('x')
    >>> genocchi(4, x)
    -4*x**3 + 6*x**2 - 1

    See Also
    ========

    bell, bernoulli, catalan, euler, fibonacci, harmonic, lucas, partition, tribonacci
    sympy.polys.appellseqs.genocchi_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Genocchi_number
    .. [2] https://mathworld.wolfram.com/GenocchiNumber.html
    .. [3] Peter Luschny, "An introduction to the Bernoulli function",
           https://arxiv.org/abs/2009.06743

    """

    @classmethod
    def eval(cls, n, x=None):
        # 如果 x 是单位元素 S.One，则返回 genocchi(n)
        if x is S.One:
            return cls(n)
        # 如果 n 不是整数或者不是非负数，则返回空
        elif n.is_integer is False or n.is_nonnegative is False:
            return
        # Genocchi numbers
        elif x is None:
            # 如果 n 是奇数且 n-1 是正数，则返回零
            if n.is_odd and (n-1).is_positive:
                return S.Zero
            # 如果 n 是数字，则返回 2 * (1 - 2**n) * bernoulli(n)
            elif n.is_Number:
                return 2 * (1-S(2)**n) * bernoulli(n)
        # Genocchi polynomials
        elif n.is_Number:
            return genocchi_poly(n, x)

    # 将当前对象重写为 Bernoulli 函数的表达式
    def _eval_rewrite_as_bernoulli(self, n, x=1, **kwargs):
        if x == 1 and n.is_integer and n.is_nonnegative:
            return 2 * (1-S(2)**n) * bernoulli(n)
        return 2 * (bernoulli(n, x) - 2**n * bernoulli(n, (x+1) / 2))

    # 将当前对象重写为 Dirichlet eta 函数的表达式
    def _eval_rewrite_as_dirichlet_eta(self, n, x=1, **kwargs):
        from sympy.functions.special.zeta_functions import dirichlet_eta
        return -2*n * dirichlet_eta(1-n, x)

    # 判断当前对象是否是整数
    def _eval_is_integer(self):
        if len(self.args) > 1 and self.args[1] != 1:
            return
        n = self.args[0]
        if n.is_integer and n.is_nonnegative:
            return True
    # 检查表达式是否为负数
    def _eval_is_negative(self):
        # 如果参数数量大于1且第二个参数不等于1，则返回空（不处理）
        if len(self.args) > 1 and self.args[1] != 1:
            return
        # 获取第一个参数
        n = self.args[0]
        # 如果参数是整数且非负
        if n.is_integer and n.is_nonnegative:
            # 如果参数是奇数
            if n.is_odd:
                # 返回 (n-1) 是否为正数的模糊否定
                return fuzzy_not((n-1).is_positive)
            # 返回 n/2 是否为奇数
            return (n/2).is_odd

    # 检查表达式是否为正数
    def _eval_is_positive(self):
        # 如果参数数量大于1且第二个参数不等于1，则返回空（不处理）
        if len(self.args) > 1 and self.args[1] != 1:
            return
        # 获取第一个参数
        n = self.args[0]
        # 如果参数是整数且非负
        if n.is_integer and n.is_nonnegative:
            # 如果参数是零或者偶数
            if n.is_zero or n.is_odd:
                # 返回 False
                return False
            # 返回 n/2 是否为偶数
            return (n/2).is_even

    # 检查表达式是否为偶数
    def _eval_is_even(self):
        # 如果参数数量大于1且第二个参数不等于1，则返回空（不处理）
        if len(self.args) > 1 and self.args[1] != 1:
            return
        # 获取第一个参数
        n = self.args[0]
        # 如果参数是整数且非负
        if n.is_integer and n.is_nonnegative:
            # 如果参数是偶数
            if n.is_even:
                # 返回 n 是否为零
                return n.is_zero
            # 返回 (n-1) 是否为正数
            return (n-1).is_positive

    # 检查表达式是否为奇数
    def _eval_is_odd(self):
        # 如果参数数量大于1且第二个参数不等于1，则返回空（不处理）
        if len(self.args) > 1 and self.args[1] != 1:
            return
        # 获取第一个参数
        n = self.args[0]
        # 如果参数是整数且非负
        if n.is_integer and n.is_nonnegative:
            # 如果参数是偶数
            if n.is_even:
                # 返回 (n-1) 是否为正数的模糊否定
                return fuzzy_not(n.is_zero)
            # 返回 (n-1) 是否为正数的模糊否定
            return fuzzy_not((n-1).is_positive)

    # 检查表达式是否为质数
    def _eval_is_prime(self):
        # 如果参数数量大于1且第二个参数不等于1，则返回空（不处理）
        if len(self.args) > 1 and self.args[1] != 1:
            return
        # 获取第一个参数
        n = self.args[0]
        # 返回 (n-8) 是否为零
        # 由于 SymPy 不将负数视为质数，因此仅测试 n=8 的情况
        return (n-8).is_zero

    # 对表达式进行数值评估
    def _eval_evalf(self, prec):
        # 如果所有参数都是数字
        if all(i.is_number for i in self.args):
            # 重写为伯努利数，并进行数值评估
            return self.rewrite(bernoulli)._eval_evalf(prec)
# 定义一个名为 andre 的类，继承自 Function 类
class andre(Function):
    r"""
    Andre numbers / Andre function

    The Andre number `\mathcal{A}_n` is Luschny's name for half the number of
    *alternating permutations* on `n` elements, where a permutation is alternating
    if adjacent elements alternately compare "greater" and "smaller" going from
    left to right. For example, `2 < 3 > 1 < 4` is an alternating permutation.

    This sequence is A000111 in the OEIS, which assigns the names *up/down numbers*
    and *Euler zigzag numbers*. It satisfies a recurrence relation similar to that
    for the Catalan numbers, with `\mathcal{A}_0 = 1` and

    .. math:: 2 \mathcal{A}_{n+1} = \sum_{k=0}^n \binom{n}{k} \mathcal{A}_k \mathcal{A}_{n-k}

    The Bernoulli and Euler numbers are signed transformations of the odd- and
    even-indexed elements of this sequence respectively:

    .. math :: \operatorname{B}_{2k} = \frac{2k \mathcal{A}_{2k-1}}{(-4)^k - (-16)^k}

    .. math :: \operatorname{E}_{2k} = (-1)^k \mathcal{A}_{2k}

    Like the Bernoulli and Euler numbers, the Andre numbers are interpolated by the
    entire Andre function:

    .. math :: \mathcal{A}(s) = (-i)^{s+1} \operatorname{Li}_{-s}(i) +
            i^{s+1} \operatorname{Li}_{-s}(-i) = \\ \frac{2 \Gamma(s+1)}{(2\pi)^{s+1}}
            (\zeta(s+1, 1/4) - \zeta(s+1, 3/4) \cos{\pi s})

    Examples
    ========

    >>> from sympy import andre, euler, bernoulli
    >>> [andre(n) for n in range(11)]
    [1, 1, 1, 2, 5, 16, 61, 272, 1385, 7936, 50521]
    >>> [(-1)**k * andre(2*k) for k in range(7)]
    [1, -1, 5, -61, 1385, -50521, 2702765]
    >>> [euler(2*k) for k in range(7)]
    [1, -1, 5, -61, 1385, -50521, 2702765]
    >>> [andre(2*k-1) * (2*k) / ((-4)**k - (-16)**k) for k in range(1, 8)]
    [1/6, -1/30, 1/42, -1/30, 5/66, -691/2730, 7/6]
    >>> [bernoulli(2*k) for k in range(1, 8)]
    [1/6, -1/30, 1/42, -1/30, 5/66, -691/2730, 7/6]

    See Also
    ========

    bernoulli, catalan, euler, sympy.polys.appellseqs.andre_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Alternating_permutation
    .. [2] https://mathworld.wolfram.com/EulerZigzagNumber.html
    .. [3] Peter Luschny, "An introduction to the Bernoulli function",
           https://arxiv.org/abs/2009.06743
    """

    # 类方法，用于计算 Andre 数
    @classmethod
    def eval(cls, n):
        # 如果 n 是符号 S.NaN，则返回 S.NaN
        if n is S.NaN:
            return S.NaN
        # 如果 n 是符号 S.Infinity，则返回 S.Infinity
        elif n is S.Infinity:
            return S.Infinity
        # 如果 n 是零
        if n.is_zero:
            # 返回符号 S.One
            return S.One
        # 如果 n 等于 -1
        elif n == -1:
            # 返回 -log(2)
            return -log(2)
        # 如果 n 等于 -2
        elif n == -2:
            # 返回 -2*S.Catalan
            return -2*S.Catalan
        # 如果 n 是整数
        elif n.is_Integer:
            # 如果 n 是非负偶数
            if n.is_nonnegative and n.is_even:
                # 返回 euler(n) 的绝对值
                return abs(euler(n))
            # 如果 n 是奇数
            elif n.is_odd:
                # 导入 zeta 函数
                from sympy.functions.special.zeta_functions import zeta
                # 计算 m = -n-1
                m = -n-1
                # 返回 I**m * Rational(1-2**m, 4**m) * zeta(-n)
                return I**m * Rational(1-2**m, 4**m) * zeta(-n)

    def _eval_rewrite_as_zeta(self, s, **kwargs):
        # 导入需要的函数
        from sympy.functions.elementary.trigonometric import cos
        from sympy.functions.special.gamma_functions import gamma
        from sympy.functions.special.zeta_functions import zeta
        # 返回重写后的表达式
        return 2 * gamma(s+1) / (2*pi)**(s+1) * \
                (zeta(s+1, S.One/4) - cos(pi*s) * zeta(s+1, S(3)/4))

    def _eval_rewrite_as_polylog(self, s, **kwargs):
        # 导入 polylog 函数
        from sympy.functions.special.zeta_functions import polylog
        # 返回重写后的表达式
        return (-I)**(s+1) * polylog(-s, I) + I**(s+1) * polylog(-s, -I)

    def _eval_is_integer(self):
        # 获取参数 n
        n = self.args[0]
        # 如果 n 是整数且非负
        if n.is_integer and n.is_nonnegative:
            # 返回 True
            return True

    def _eval_is_positive(self):
        # 获取参数的第一个元素
        if self.args[0].is_nonnegative:
            # 返回 True
            return True

    def _eval_evalf(self, prec):
        # 如果参数不是数字，则返回空
        if not self.args[0].is_number:
            return
        # 将参数转换为 mpmath 的精度格式
        s = self.args[0]._to_mpmath(prec+12)
        # 设置工作精度
        with workprec(prec+12):
            # 计算 sinpi(s/2) 和 cospi(s/2)
            sp, cp = mp.sinpi(s/2), mp.cospi(s/2)
            # 计算结果
            res = 2*mp.dirichlet(-s, (-sp, cp, sp, -cp))
        # 返回 Expr._from_mpmath 格式化后的结果
        return Expr._from_mpmath(res, prec)
#----------------------------------------------------------------------------#
#                                                                            #
#                           Partition numbers                                #
#                                                                            #
#----------------------------------------------------------------------------#

# 定义一个继承自 Function 的类 partition，表示分区数函数
class partition(Function):
    r"""
    Partition numbers

    The Partition numbers are a sequence of integers `p_n` that represent the
    number of distinct ways of representing `n` as a sum of natural numbers
    (with order irrelevant). The generating function for `p_n` is given by:

    .. math:: \sum_{n=0}^\infty p_n x^n = \prod_{k=1}^\infty (1 - x^k)^{-1}

    Examples
    ========

    >>> from sympy import partition, Symbol
    >>> [partition(n) for n in range(9)]
    [1, 1, 2, 3, 5, 7, 11, 15, 22]
    >>> n = Symbol('n', integer=True, negative=True)
    >>> partition(n)
    0

    See Also
    ========

    bell, bernoulli, catalan, euler, fibonacci, harmonic, lucas, genocchi, tribonacci

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Partition_(number_theory%29
    .. [2] https://en.wikipedia.org/wiki/Pentagonal_number_theorem

    """
    # 表明该函数返回的结果是整数
    is_integer = True
    # 表明该函数的结果是非负数
    is_nonnegative = True

    # 类方法，用来计算分区数函数的值
    @classmethod
    def eval(cls, n):
        # 如果 n 不是整数，则抛出 TypeError 异常
        if n.is_integer is False:
            raise TypeError("n should be an integer")
        # 如果 n 是负数，则返回 0
        if n.is_negative is True:
            return S.Zero
        # 如果 n 是零或者是 1，则返回 1
        if n.is_zero is True or n is S.One:
            return S.One
        # 如果 n 是整数，则返回分区数函数的值
        if n.is_Integer is True:
            return S(_partition(as_int(n)))

    # 私有方法，用来评估函数是否是正数
    def _eval_is_positive(self):
        # 如果参数的非负标志为真，则返回真
        if self.args[0].is_nonnegative is True:
            return True


# 定义一个继承自 Function 的类 divisor_sigma，表示约数函数
class divisor_sigma(Function):
    r"""
    Calculate the divisor function `\sigma_k(n)` for positive integer n

    ``divisor_sigma(n, k)`` is equal to ``sum([x**k for x in divisors(n)])``

    If n's prime factorization is:

    .. math ::
        n = \prod_{i=1}^\omega p_i^{m_i},

    then

    .. math ::
        \sigma_k(n) = \prod_{i=1}^\omega (1+p_i^k+p_i^{2k}+\cdots
        + p_i^{m_ik}).

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import divisor_sigma
    >>> divisor_sigma(18, 0)
    6
    >>> divisor_sigma(39, 1)
    56
    >>> divisor_sigma(12, 2)
    210
    >>> divisor_sigma(37)
    38

    See Also
    ========

    sympy.ntheory.factor_.divisor_count, totient, sympy.ntheory.factor_.divisors, sympy.ntheory.factor_.factorint

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Divisor_function

    """
    # 表明该函数返回的结果是整数
    is_integer = True
    # 表明该函数的结果是正数
    is_positive = True

    # 类方法，用来计算约数函数的值
    @classmethod
    def eval(cls, n, k=1):
        pass  # 此处省略具体实现，需进一步编写

    # 剩余的类定义和方法实现需要根据具体需求继续完善
    # 定义一个类方法 eval，用于计算特定数学函数的结果，参数包括 n 和 k，默认 k 为 S.One
    def eval(cls, n, k=S.One):
        # 检查 n 是否为整数，如果不是则抛出 TypeError 异常
        if n.is_integer is False:
            raise TypeError("n should be an integer")
        # 检查 n 是否为正整数，如果不是则抛出 ValueError 异常
        if n.is_positive is False:
            raise ValueError("n should be a positive integer")
        # 检查 k 是否为整数，如果不是则抛出 TypeError 异常
        if k.is_integer is False:
            raise TypeError("k should be an integer")
        # 检查 k 是否为非负整数，如果不是则抛出 ValueError 异常
        if k.is_nonnegative is False:
            raise ValueError("k should be a nonnegative integer")
        # 如果 n 是素数，则返回 1 + n 的 k 次方
        if n.is_prime is True:
            return 1 + n**k
        # 如果 n 等于 S.One，则直接返回 S.One
        if n is S.One:
            return S.One
        # 如果 n 是整数，则进一步判断 k 的情况
        if n.is_Integer is True:
            # 如果 k 是零，则返回 n 的所有素因子次方加一的乘积
            if k.is_zero is True:
                return Mul(*[e + 1 for e in factorint(n).values()])
            # 如果 k 是整数，则返回特定函数 _divisor_sigma 的结果
            if k.is_Integer is True:
                return S(_divisor_sigma(as_int(n), as_int(k)))
            # 如果 k 不为零，则计算 n 的所有素因子次方的和的分式化简乘积
            if k.is_zero is False:
                return Mul(*[cancel((p**(k*(e + 1)) - 1) / (p**k - 1)) for p, e in factorint(n).items()])
class udivisor_sigma(Function):
    r"""
    Calculate the unitary divisor function `\sigma_k^*(n)` for positive integer n

    ``udivisor_sigma(n, k)`` is equal to ``sum([x**k for x in udivisors(n)])``

    If n's prime factorization is:

    .. math ::
        n = \prod_{i=1}^\omega p_i^{m_i},

    then

    .. math ::
        \sigma_k^*(n) = \prod_{i=1}^\omega (1+ p_i^{m_ik}).

    Parameters
    ==========

    k : power of divisors in the sum

        for k = 0, 1:
        ``udivisor_sigma(n, 0)`` is equal to ``udivisor_count(n)``
        ``udivisor_sigma(n, 1)`` is equal to ``sum(udivisors(n))``

        Default for k is 1.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import udivisor_sigma
    >>> udivisor_sigma(18, 0)
    4
    >>> udivisor_sigma(74, 1)
    114
    >>> udivisor_sigma(36, 3)
    47450
    >>> udivisor_sigma(111)
    152

    See Also
    ========

    sympy.ntheory.factor_.divisor_count, totient, sympy.ntheory.factor_.divisors,
    sympy.ntheory.factor_.udivisors, sympy.ntheory.factor_.udivisor_count, divisor_sigma,
    sympy.ntheory.factor_.factorint

    References
    ==========

    .. [1] https://mathworld.wolfram.com/UnitaryDivisorFunction.html

    """
    is_integer = True  # 类变量，指示该函数的参数是否为整数
    is_positive = True  # 类变量，指示该函数的参数是否为正整数

    @classmethod
    def eval(cls, n, k=S.One):
        # 检查 n 是否为整数，若不是则引发异常
        if n.is_integer is False:
            raise TypeError("n should be an integer")
        # 检查 n 是否为正整数，若不是则引发异常
        if n.is_positive is False:
            raise ValueError("n should be a positive integer")
        # 检查 k 是否为整数，若不是则引发异常
        if k.is_integer is False:
            raise TypeError("k should be an integer")
        # 检查 k 是否为非负整数，若不是则引发异常
        if k.is_nonnegative is False:
            raise ValueError("k should be a nonnegative integer")
        # 若 n 为素数，则返回特定值
        if n.is_prime is True:
            return 1 + n**k
        # 若 n 为整数，则计算并返回 \sigma_k^*(n) 的值
        if n.is_Integer:
            return Mul(*[1+p**(k*e) for p, e in factorint(n).items()])


class legendre_symbol(Function):
    r"""
    Returns the Legendre symbol `(a / p)`.

    For an integer ``a`` and an odd prime ``p``, the Legendre symbol is
    defined as

    .. math ::
        \genfrac(){}{}{a}{p} = \begin{cases}
             0 & \text{if } p \text{ divides } a\\
             1 & \text{if } a \text{ is a quadratic residue modulo } p\\
            -1 & \text{if } a \text{ is a quadratic nonresidue modulo } p
        \end{cases}

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import legendre_symbol
    >>> [legendre_symbol(i, 7) for i in range(7)]
    [0, 1, 1, -1, 1, -1, -1]
    >>> sorted(set([i**2 % 7 for i in range(7)]))
    [0, 1, 2, 4]

    See Also
    ========

    sympy.ntheory.residue_ntheory.is_quad_residue, jacobi_symbol

    """
    is_integer = True  # 类变量，指示该函数的参数是否为整数
    is_prime = False  # 类变量，指示该函数的参数是否为素数

    @classmethod
    # 类方法，用于计算 Legendre 符号
    # 定义一个类方法 `eval`，接受参数 `cls`, `a`, `p`
    def eval(cls, a, p):
        # 如果 `a` 不是整数，则抛出类型错误异常
        if a.is_integer is False:
            raise TypeError("a should be an integer")
        # 如果 `p` 不是整数，则抛出类型错误异常
        if p.is_integer is False:
            raise TypeError("p should be an integer")
        # 如果 `p` 不是奇素数，则抛出值错误异常
        if p.is_prime is False or p.is_odd is False:
            raise ValueError("p should be an odd prime integer")
        # 如果 `a` 对 `p` 取模为零，则返回零
        if (a % p).is_zero is True:
            return S.Zero
        # 如果 `a` 等于一，则返回一
        if a is S.One:
            return S.One
        # 如果 `a` 和 `p` 都是整数，则返回 `legendre` 函数的结果
        if a.is_Integer is True and p.is_Integer is True:
            return S(legendre(as_int(a), as_int(p)))
class jacobi_symbol(Function):
    r"""
    Returns the Jacobi symbol `(m / n)`.

    For any integer ``m`` and any positive odd integer ``n`` the Jacobi symbol
    is defined as the product of the Legendre symbols corresponding to the
    prime factors of ``n``:

    .. math ::
        \genfrac(){}{}{m}{n} =
            \genfrac(){}{}{m}{p^{1}}^{\alpha_1}
            \genfrac(){}{}{m}{p^{2}}^{\alpha_2}
            ...
            \genfrac(){}{}{m}{p^{k}}^{\alpha_k}
            \text{ where } n =
                p_1^{\alpha_1}
                p_2^{\alpha_2}
                ...
                p_k^{\alpha_k}

    Like the Legendre symbol, if the Jacobi symbol `\genfrac(){}{}{m}{n} = -1`
    then ``m`` is a quadratic nonresidue modulo ``n``.

    But, unlike the Legendre symbol, if the Jacobi symbol
    `\genfrac(){}{}{m}{n} = 1` then ``m`` may or may not be a quadratic residue
    modulo ``n``.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import jacobi_symbol, legendre_symbol
    >>> from sympy import S
    >>> jacobi_symbol(45, 77)
    -1
    >>> jacobi_symbol(60, 121)
    1

    The relationship between the ``jacobi_symbol`` and ``legendre_symbol`` can
    be demonstrated as follows:

    >>> L = legendre_symbol
    >>> S(45).factors()
    {3: 2, 5: 1}
    >>> jacobi_symbol(7, 45) == L(7, 3)**2 * L(7, 5)**1
    True

    See Also
    ========

    sympy.ntheory.residue_ntheory.is_quad_residue, legendre_symbol

    """
    is_integer = True        # 类变量，表示该函数处理整数参数
    is_prime = False         # 类变量，表示该函数处理的不是素数参数

    @classmethod
    def eval(cls, m, n):
        if m.is_integer is False:   # 如果 m 不是整数，抛出类型错误异常
            raise TypeError("m should be an integer")
        if n.is_integer is False:   # 如果 n 不是整数，抛出类型错误异常
            raise TypeError("n should be an integer")
        if n.is_positive is False or n.is_odd is False:   # 如果 n 不是正奇数，抛出值错误异常
            raise ValueError("n should be an odd positive integer")
        if m is S.One or n is S.One:   # 如果 m 或 n 等于 1，返回 1
            return S.One
        if (m % n).is_zero is True:    # 如果 m 除以 n 的余数为 0，返回 0
            return S.Zero
        if m.is_Integer is True and n.is_Integer is True:   # 如果 m 和 n 都是整数，返回 Jacobi 符号的计算结果
            return S(jacobi(as_int(m), as_int(n)))


class kronecker_symbol(Function):
    r"""
    Returns the Kronecker symbol `(a / n)`.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import kronecker_symbol
    >>> kronecker_symbol(45, 77)
    -1
    >>> kronecker_symbol(13, -120)
    1

    See Also
    ========

    jacobi_symbol, legendre_symbol

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Kronecker_symbol

    """
    is_integer = True    # 类变量，表示该函数处理整数参数
    is_prime = False     # 类变量，表示该函数处理的不是素数参数

    @classmethod
    def eval(cls, a, n):
        if a.is_integer is False:   # 如果 a 不是整数，抛出类型错误异常
            raise TypeError("a should be an integer")
        if n.is_integer is False:   # 如果 n 不是整数，抛出类型错误异常
            raise TypeError("n should be an integer")
        if a is S.One or n is S.One:   # 如果 a 或 n 等于 1，返回 1
            return S.One
        if a.is_Integer is True and n.is_Integer is True:   # 如果 a 和 n 都是整数，返回 Kronecker 符号的计算结果
            return S(kronecker(as_int(a), as_int(n)))


class mobius(Function):
    # 这里是一个空的函数定义，用于计算莫比乌斯函数，暂无代码实现
    """
    Mobius function maps natural number to {-1, 0, 1}

    It is defined as follows:
        1) `1` if `n = 1`.
        2) `0` if `n` has a squared prime factor.
        3) `(-1)^k` if `n` is a square-free positive integer with `k`
           number of prime factors.

    It is an important multiplicative function in number theory
    and combinatorics.  It has applications in mathematical series,
    algebraic number theory and also physics (Fermion operator has very
    concrete realization with Mobius Function model).

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import mobius
    >>> mobius(13*7)
    1
    >>> mobius(1)
    1
    >>> mobius(13*7*5)
    -1
    >>> mobius(13**2)
    0

    Even in the case of a symbol, if it clearly contains a squared prime factor, it will be zero.

    >>> from sympy import Symbol
    >>> n = Symbol("n", integer=True, positive=True)
    >>> mobius(4*n)
    0
    >>> mobius(n**2)
    0

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/M%C3%B6bius_function
    .. [2] Thomas Koshy "Elementary Number Theory with Applications"
    .. [3] https://oeis.org/A008683

    """
    # 初始化变量，用于存储检查条件的布尔值
    is_integer = True  # 是否为整数
    is_prime = False   # 是否为质数

    # 类方法，计算 Mobius 函数的值
    @classmethod
    def eval(cls, n):
        # 检查输入参数 n 是否为整数
        if n.is_integer is False:
            raise TypeError("n should be an integer")
        # 检查输入参数 n 是否为正整数
        if n.is_positive is False:
            raise ValueError("n should be a positive integer")
        # 如果 n 是质数，返回 -1
        if n.is_prime is True:
            return S.NegativeOne
        # 如果 n 等于 1，返回 1
        if n is S.One:
            return S.One
        result = None  # 初始化结果变量
        # 对 n 进行因式分解，并遍历每一个基数和指数对
        for m, e in (_.as_base_exp() for _ in Mul.make_args(n)):
            # 检查基数 m 和指数 e 是否为整数且为正数
            if m.is_integer is True and m.is_positive is True and \
               e.is_integer is True and e.is_positive is True:
                lt = is_lt(S.One, e)  # 检查 1 是否小于 e
                # 如果 1 < e，则结果为 0
                if lt is True:
                    result = S.Zero
                # 如果 m 是整数，继续判断其因数是否有大于 1 的值
                elif m.is_Integer is True:
                    factors = factorint(m)
                    # 如果存在大于 1 的因数，则结果为 0
                    if any(v > 1 for v in factors.values()):
                        result = S.Zero
                    # 否则根据因数数量的奇偶性确定结果为 1 或 -1
                    elif lt is False:
                        s = S.NegativeOne if len(factors) % 2 else S.One
                        if result is None:
                            result = s
                        else:
                            result *= s
            else:
                return  # 如果 m 或 e 不符合条件，则返回 None
        return result  # 返回计算得到的 Mobius 函数值
class primenu(Function):
    r"""
    Calculate the number of distinct prime factors for a positive integer n.

    If n's prime factorization is:

    .. math ::
        n = \prod_{i=1}^k p_i^{m_i},

    then ``primenu(n)`` or `\nu(n)` is:

    .. math ::
        \nu(n) = k.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import primenu
    >>> primenu(1)
    0
    >>> primenu(30)
    3

    See Also
    ========

    sympy.ntheory.factor_.factorint

    References
    ==========

    .. [1] https://mathworld.wolfram.com/PrimeFactor.html
    .. [2] https://oeis.org/A001221

    """
    is_integer = True  # 类属性：标识该函数接受整数参数
    is_nonnegative = True  # 类属性：标识该函数返回非负值

    @classmethod
    def eval(cls, n):
        if n.is_integer is False:  # 检查参数是否为整数
            raise TypeError("n should be an integer")
        if n.is_positive is False:  # 检查参数是否为正整数
            raise ValueError("n should be a positive integer")
        if n.is_prime is True:  # 如果参数是素数，直接返回 1
            return S.One
        if n is S.One:  # 如果参数是 1，直接返回 0
            return S.Zero
        if n.is_Integer is True:  # 对正整数 n 计算其不同素因子的个数
            return S(len(factorint(n)))  # 返回不同素因子的个数


class primeomega(Function):
    r"""
    Calculate the number of prime factors counting multiplicities for a
    positive integer n.

    If n's prime factorization is:

    .. math ::
        n = \prod_{i=1}^k p_i^{m_i},

    then ``primeomega(n)``  or `\Omega(n)` is:

    .. math ::
        \Omega(n) = \sum_{i=1}^k m_i.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import primeomega
    >>> primeomega(1)
    0
    >>> primeomega(20)
    3

    See Also
    ========

    sympy.ntheory.factor_.factorint

    References
    ==========

    .. [1] https://mathworld.wolfram.com/PrimeFactor.html
    .. [2] https://oeis.org/A001222

    """
    is_integer = True  # 类属性：标识该函数接受整数参数
    is_nonnegative = True  # 类属性：标识该函数返回非负值

    @classmethod
    def eval(cls, n):
        if n.is_integer is False:  # 检查参数是否为整数
            raise TypeError("n should be an integer")
        if n.is_positive is False:  # 检查参数是否为正整数
            raise ValueError("n should be a positive integer")
        if n.is_prime is True:  # 如果参数是素数，直接返回 1
            return S.One
        if n is S.One:  # 如果参数是 1，直接返回 0
            return S.Zero
        if n.is_Integer is True:  # 对正整数 n 计算其所有素因子的指数之和
            return S(sum(factorint(n).values()))  # 返回所有素因子指数之和


class totient(Function):
    r"""
    Calculate the Euler totient function phi(n)

    ``totient(n)`` or `\phi(n)` is the number of positive integers `\leq` n
    that are relatively prime to n.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import totient
    >>> totient(1)
    1
    >>> totient(25)
    20
    >>> totient(45) == totient(5)*totient(9)
    True

    See Also
    ========

    sympy.ntheory.factor_.divisor_count

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Euler%27s_totient_function
    .. [2] https://mathworld.wolfram.com/TotientFunction.html
    .. [3] https://oeis.org/A000010

    """
    is_integer = True  # 类属性：标识该函数接受整数参数
    is_positive = True  # 类属性：标识该函数返回正值

    @classmethod
    # 定义一个类方法 eval，用于计算特定数值 n 的结果
    def eval(cls, n):
        # 检查 n 是否为整数，若不是则抛出类型错误异常
        if n.is_integer is False:
            raise TypeError("n should be an integer")
        # 检查 n 是否为正整数，若不是则抛出数值错误异常
        if n.is_positive is False:
            raise ValueError("n should be a positive integer")
        # 若 n 等于 S.One，则直接返回 S.One
        if n is S.One:
            return S.One
        # 检查 n 是否为质数，若是则返回 n - 1
        if n.is_prime is True:
            return n - 1
        # 若 n 是字典类型（Dict），计算其作为 Euler 函数参数的结果并返回
        if isinstance(n, Dict):
            return S(prod(p**(k-1)*(p-1) for p, k in n.items()))
        # 若 n 是整数类型，计算其作为 Euler 函数参数的结果并返回
        if n.is_Integer is True:
            return S(prod(p**(k-1)*(p-1) for p, k in factorint(n).items()))
class reduced_totient(Function):
    r"""
    Calculate the Carmichael reduced totient function lambda(n)

    ``reduced_totient(n)`` or `\lambda(n)` is the smallest m > 0 such that
    `k^m \equiv 1 \mod n` for all k relatively prime to n.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import reduced_totient
    >>> reduced_totient(1)
    1
    >>> reduced_totient(8)
    2
    >>> reduced_totient(30)
    4

    See Also
    ========

    totient

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Carmichael_function
    .. [2] https://mathworld.wolfram.com/CarmichaelFunction.html
    .. [3] https://oeis.org/A002322

    """
    # 声明这是一个整数函数
    is_integer = True
    # 声明这是一个正数函数
    is_positive = True

    # 类方法，用于计算 lambda(n) 的值
    @classmethod
    def eval(cls, n):
        # 如果 n 不是整数，则抛出类型错误
        if n.is_integer is False:
            raise TypeError("n should be an integer")
        # 如果 n 不是正整数，则抛出数值错误
        if n.is_positive is False:
            raise ValueError("n should be a positive integer")
        # 如果 n 等于 1，则返回 1
        if n is S.One:
            return S.One
        # 如果 n 是质数，则返回 n - 1
        if n.is_prime is True:
            return n - 1
        # 如果 n 是字典类型，则进行特定处理
        if isinstance(n, Dict):
            t = 1
            # 如果字典中包含键为 2，则计算 t 的值
            if 2 in n:
                t = (1 << (n[2] - 2)) if 2 < n[2] else n[2]
            # 返回使用 lcm 计算的结果
            return S(lcm(int(t), *(int(p-1)*int(p)**int(k-1) for p, k in n.items() if p != 2)))
        # 如果 n 是整数类型，则进行因数分解后计算
        if n.is_Integer is True:
            n, t = remove(int(n), 2)
            if not t:
                t = 1
            elif 2 < t:
                t = 1 << (t - 2)
            # 返回使用 lcm 计算的结果
            return S(lcm(t, *((p-1)*p**(k-1) for p, k in factorint(n).items())))


class primepi(Function):
    r""" Represents the prime counting function pi(n) = the number
    of prime numbers less than or equal to n.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import primepi
    >>> from sympy import prime, prevprime, isprime
    >>> primepi(25)
    9

    So there are 9 primes less than or equal to 25. Is 25 prime?

    >>> isprime(25)
    False

    It is not. So the first prime less than 25 must be the
    9th prime:

    >>> prevprime(25) == prime(9)
    True

    See Also
    ========

    sympy.ntheory.primetest.isprime : Test if n is prime
    sympy.ntheory.generate.primerange : Generate all primes in a given range
    sympy.ntheory.generate.prime : Return the nth prime

    References
    ==========

    .. [1] https://oeis.org/A000720

    """
    # 声明这是一个整数函数
    is_integer = True
    # 声明这是一个非负整数函数
    is_nonnegative = True

    # 类方法，用于计算 pi(n) 的值
    @classmethod
    def eval(cls, n):
        # 如果 n 是正无穷，则返回正无穷
        if n is S.Infinity:
            return S.Infinity
        # 如果 n 是负无穷，则返回 0
        if n is S.NegativeInfinity:
            return S.Zero
        # 如果 n 不是实数，则抛出类型错误
        if n.is_real is False:
            raise TypeError("n should be a real")
        # 如果 n 小于 2，则返回 0
        if is_lt(n, S(2)) is True:
            return S.Zero
        try:
            n = int(n)
        except TypeError:
            return
        # 返回调用 _primepi 函数计算的结果
        return S(_primepi(n))


#######################################################################
###
### Functions for enumerating partitions, permutations and combinations
###
#######################################################################

class _MultisetHistogram(tuple):
    pass
    # 定义一个特殊的元组子类用于存储多重集直方图的数据结构

_N = -1
_ITEMS = -2
_M = slice(None, _ITEMS)
# 定义常量_N为-1，_ITEMS为-2，_M为slice(None, _ITEMS)，用于在代码中引用特定的切片和索引位置

def _multiset_histogram(n):
    """Return tuple used in permutation and combination counting. Input
    is a dictionary giving items with counts as values or a sequence of
    items (which need not be sorted).

    The data is stored in a class deriving from tuple so it is easily
    recognized and so it can be converted easily to a list.
    """
    if isinstance(n, dict):  # item: count
        # 如果输入n是字典形式（元素:计数）
        if not all(isinstance(v, int) and v >= 0 for v in n.values()):
            raise ValueError
        # 如果所有计数值为非负整数，计算总计数和不为零的元素数量
        tot = sum(n.values())
        items = sum(1 for k in n if n[k] > 0)
        # 返回一个_MultisetHistogram类的实例，将字典中每个元素的计数和不为零的元素数量添加到元组中
        return _MultisetHistogram([n[k] for k in n if n[k] > 0] + [items, tot])
    else:
        # 如果输入n是序列形式
        n = list(n)
        s = set(n)
        lens = len(s)
        lenn = len(n)
        # 计算序列中独特元素的数量和总元素的数量
        if lens == lenn:
            # 如果所有元素都是唯一的，创建一个_MultisetHistogram类的实例，元组包含所有元素为1的列表以及序列长度两次
            n = [1]*lenn + [lenn, lenn]
            return _MultisetHistogram(n)
        m = dict(zip(s, range(lens)))
        d = dict(zip(range(lens), (0,)*lens))
        # 计算序列中每个元素的出现次数，创建_MultisetHistogram类的实例返回
        for i in n:
            d[m[i]] += 1
        return _multiset_histogram(d)
        # 返回一个_MultisetHistogram类的实例，其中包含每个独特元素的计数和总元素数量

def nP(n, k=None, replacement=False):
    """Return the number of permutations of ``n`` items taken ``k`` at a time.

    Possible values for ``n``:

        integer - set of length ``n``

        sequence - converted to a multiset internally

        multiset - {element: multiplicity}

    If ``k`` is None then the total of all permutations of length 0
    through the number of items represented by ``n`` will be returned.

    If ``replacement`` is True then a given item can appear more than once
    in the ``k`` items. (For example, for 'ab' permutations of 2 would
    include 'aa', 'ab', 'ba' and 'bb'.) The multiplicity of elements in
    ``n`` is ignored when ``replacement`` is True but the total number
    of elements is considered since no element can appear more times than
    the number of elements in ``n``.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import nP
    >>> from sympy.utilities.iterables import multiset_permutations, multiset
    >>> nP(3, 2)
    6
    >>> nP('abc', 2) == nP(multiset('abc'), 2) == 6
    True
    >>> nP('aab', 2)
    3
    >>> nP([1, 2, 2], 2)
    3
    >>> [nP(3, i) for i in range(4)]
    [1, 3, 6, 6]
    >>> nP(3) == sum(_)
    True

    When ``replacement`` is True, each item can have multiplicity
    equal to the length represented by ``n``:

    >>> nP('aabc', replacement=True)
    121
    >>> [len(list(multiset_permutations('aaaabbbbcccc', i))) for i in range(5)]
    [1, 3, 9, 27, 81]
    >>> sum(_)
    121

    See Also
    ========
    sympy.utilities.iterables.multiset_permutations

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Permutation

    """
    try:
        n = as_int(n)
    except ValueError:
        return Integer(_nP(_multiset_histogram(n), k, replacement))
    # 如果输入的n不是整数，调用_as_int函数转换为整数再进行计算，并返回结果
    # 调用 _nP 函数计算排列数，并将结果封装为 Integer 对象后返回
    return Integer(_nP(n, k, replacement))
# 使用装饰器 @cacheit 对 _nP 函数进行缓存，以提高性能
@cacheit
# 定义 _nP 函数，计算排列数或多重集的排列数
def _nP(n, k=None, replacement=False):

    # 如果 k 为 0，返回 1，表示空集的排列数为 1
    if k == 0:
        return 1
    # 如果 n 是 SYMPY_INTS 类型的整数，表示不同的项
    if isinstance(n, SYMPY_INTS):  # n different items
        # 如果 k 为 None，则返回所有可能的排列数之和
        if k is None:
            return sum(_nP(n, i, replacement) for i in range(n + 1))
        # 如果启用替换，返回排列数 n^k
        elif replacement:
            return n**k
        # 如果 k 大于 n，返回 0，因为无法形成 k 个不同项的排列
        elif k > n:
            return 0
        # 如果 k 等于 n，返回 n 的阶乘，即 n 个不同项的排列数
        elif k == n:
            return factorial(k)
        # 如果 k 等于 1，返回 n，表示只选择其中一个项的排列数
        elif k == 1:
            return n
        else:
            # 否则，返回 n - k + 1 到 n 的乘积，表示选择 k 个项的排列数
            return _product(n - k + 1, n)
    # 如果 n 是 _MultisetHistogram 类型的对象，表示多重集直方图
    elif isinstance(n, _MultisetHistogram):
        # 如果 k 为 None，则返回所有可能的排列数之和
        if k is None:
            return sum(_nP(n, i, replacement) for i in range(n[_N] + 1))
        # 如果启用替换，返回多重集的元素的 k 次幂的排列数
        elif replacement:
            return n[_ITEMS]**k
        # 如果 k 等于多重集的元素总数，返回根据重复元素计算的排列数
        elif k == n[_N]:
            return factorial(k)/prod([factorial(i) for i in n[_M] if i > 1])
        # 如果 k 大于多重集的元素总数，返回 0
        elif k > n[_N]:
            return 0
        # 如果 k 等于 1，返回多重集的元素总数
        elif k == 1:
            return n[_ITEMS]
        else:
            # 否则，递归计算多重集中选择 k 个元素的排列数
            tot = 0
            n = list(n)
            for i in range(len(n[_M])):
                if not n[i]:
                    continue
                n[_N] -= 1
                if n[i] == 1:
                    n[i] = 0
                    n[_ITEMS] -= 1
                    tot += _nP(_MultisetHistogram(n), k - 1)
                    n[_ITEMS] += 1
                    n[i] = 1
                else:
                    n[i] -= 1
                    tot += _nP(_MultisetHistogram(n), k - 1)
                    n[i] += 1
                n[_N] += 1
            return tot


# 使用装饰器 @cacheit 对 _AOP_product 函数进行缓存，以提高性能
@cacheit
# 定义 _AOP_product 函数，计算多个 all-one 多项式的乘积的系数
def _AOP_product(n):
    """for n = (m1, m2, .., mk) return the coefficients of the polynomial,
    prod(sum(x**i for i in range(nj + 1)) for nj in n); i.e. the coefficients
    of the product of AOPs (all-one polynomials) or order given in n.  The
    resulting coefficient corresponding to x**r is the number of r-length
    combinations of sum(n) elements with multiplicities given in n.
    The coefficients are given as a default dictionary (so if a query is made
    for a key that is not present, 0 will be returned).

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import _AOP_product
    >>> from sympy.abc import x
    >>> n = (2, 2, 3)  # e.g. aabbccc
    >>> prod = ((x**2 + x + 1)*(x**2 + x + 1)*(x**3 + x**2 + x + 1)).expand()
    >>> c = _AOP_product(n); dict(c)
    {0: 1, 1: 3, 2: 6, 3: 8, 4: 8, 5: 6, 6: 3, 7: 1}
    >>> [c[i] for i in range(8)] == [prod.coeff(x, i) for i in range(8)]
    True

    The generating poly used here is the same as that listed in
    https://tinyurl.com/cep849r, but in a refactored form.

    """

    # 将 n 转换为列表形式
    n = list(n)
    # 计算 n 中所有元素的总和
    ord = sum(n)
    # 需要的长度为 (总和 + 2) // 2，用于初始化系数列表
    need = (ord + 2)//2
    rv = [1]*(n.pop() + 1)  # 初始化系数列表，长度为最后一个元素值加一
    rv.extend((0,) * (need - len(rv)))  # 将列表扩展到所需的长度
    rv = rv[:need]  # 取需要长度的子列表
    while n:
        ni = n.pop()
        N = ni + 1
        was = rv[:]
        for i in range(1, min(N, len(rv))):
            rv[i] += rv[i - 1]
        for i in range(N, need):
            rv[i] += rv[i - 1] - was[i - N]
    # 将列表 rv 反转，并转换为列表类型
    rev = list(reversed(rv))
    # 如果 ord 除以 2 的余数不为 0（即 ord 是奇数）
    if ord % 2:
        # 将反转后的列表 rev 追加到原始列表 rv 的末尾
        rv = rv + rev
    else:
        # 如果 ord 是偶数，用反转后的列表 rev 替换原始列表 rv 的最后一个元素
        rv[-1:] = rev
    # 创建一个默认值为整数类型的字典 d
    d = defaultdict(int)
    # 遍历列表 rv 的索引和元素，并将索引作为键，元素作为值存入字典 d
    for i, r in enumerate(rv):
        d[i] = r
    # 返回结果字典 d
    return d
# 定义函数 nC，计算从 n 个项目中取 k 个的组合数。
# n 可以是整数，表示长度为 n 的集合；也可以是序列，内部将被转换为多重集；还可以是多重集，格式为 {元素: 重复次数}。
def nC(n, k=None, replacement=False):
    # 如果 n 是 SYMPY_INTS 类型（即 SymPy 中的整数类型）
    if isinstance(n, SYMPY_INTS):
        # 如果 k 为 None，则返回所有长度从 0 到 n 的组合总数
        if k is None:
            # 如果不允许重复，返回 2 的 n 次方
            if not replacement:
                return 2**n
            # 允许重复时，返回所有长度从 0 到 n 的组合数之和
            return sum(nC(n, i, replacement) for i in range(n + 1))
        # 如果 k 小于 0，则抛出 ValueError
        if k < 0:
            raise ValueError("k cannot be negative")
        # 如果允许重复，则使用二项式系数计算组合数
        if replacement:
            return binomial(n + k - 1, k)
        # 否则，使用二项式系数计算组合数
        return binomial(n, k)
    # 如果 n 是 _MultisetHistogram 类型
    if isinstance(n, _MultisetHistogram):
        # 获取 n 中元素总数 N
        N = n[_N]
        # 如果 k 为 None，则返回所有长度从 0 到 N 的组合总数
        if k is None:
            # 如果不允许重复，返回 n 中各元素的 (m+1) 的乘积
            if not replacement:
                return prod(m + 1 for m in n[_M])
            # 允许重复时，返回所有长度从 0 到 N 的组合数之和
            return sum(nC(n, i, replacement) for i in range(N + 1))
        # 如果允许重复，则继续递归调用 nC 函数
        elif replacement:
            return nC(n[_ITEMS], k, replacement)
        # 如果 k 等于 1 或 N-1，则返回 n 中的元素
        elif k in (1, N - 1):
            return n[_ITEMS]
        # 如果 k 等于 0 或 N，则返回 1
        elif k in (0, N):
            return 1
        # 否则，使用 AOP_product 函数计算 k 的组合数
        return _AOP_product(tuple(n[_M]))[k]
    # 如果 n 是其他类型，则将其转换为多重集直方图后再调用 nC 函数
    else:
        return nC(_multiset_histogram(n), k, replacement)


# 定义一个函数 _eval_stirling1，计算第一类 Stirling 数 S(n, k)
def _eval_stirling1(n, k):
    # 当 n 和 k 都为 0 时，返回 SymPy 中的 S.One
    if n == k == 0:
        return S.One
    # 当 n 或 k 之一为 0 时，返回 SymPy 中的 S.Zero
    if 0 in (n, k):
        return S.Zero
    # 检查一些特殊的情况和数值
    if n == k:
        # 如果 k 等于 n，返回 S.One
        return S.One
    elif k == n - 1:
        # 如果 k 等于 n - 1，返回二项式系数 binomial(n, 2)
        return binomial(n, 2)
    elif k == n - 2:
        # 如果 k 等于 n - 2，返回 (3*n - 1)*binomial(n, 3)/4
        return (3*n - 1)*binomial(n, 3)/4
    elif k == n - 3:
        # 如果 k 等于 n - 3，返回 binomial(n, 2)*binomial(n, 4)
        return binomial(n, 2)*binomial(n, 4)

    # 如果上述情况都不满足，则调用 _stirling1(n, k) 函数处理
    return _stirling1(n, k)
# 使用装饰器 @cacheit 对 _stirling1 函数进行缓存，以优化重复调用时的性能
@cacheit
def _stirling1(n, k):
    # 初始化第一行的斯特林数为 [0, 1, 0, ..., 0]，对应 n = 1 的情况
    row = [0, 1]+[0]*(k-1) # for n = 1
    # 从 n = 2 开始计算斯特林数
    for i in range(2, n+1):
        # 从后向前更新斯特林数的每一项
        for j in range(min(k,i), 0, -1):
            # 根据斯特林数的递推公式更新当前行的每一项
            row[j] = (i-1) * row[j] + row[j-1]
    # 返回斯特林数 S(n, k)
    return Integer(row[k])


# 计算第二类斯特林数的函数，用于 _eval_stirling2 和 stirling 函数
def _eval_stirling2(n, k):
    # 处理一些特殊情况
    if n == k == 0:
        return S.One
    if 0 in (n, k):
        return S.Zero

    # 处理一些特殊值
    if n == k:
        return S.One
    elif k == n - 1:
        return binomial(n, 2)
    elif k == 1:
        return S.One
    elif k == 2:
        return Integer(2**(n - 1) - 1)

    # 调用 _stirling2 函数计算第二类斯特林数 S(n, k)
    return _stirling2(n, k)


# 使用装饰器 @cacheit 对 _stirling2 函数进行缓存，以优化重复调用时的性能
@cacheit
def _stirling2(n, k):
    # 初始化第一行的斯特林数为 [0, 1, 0, ..., 0]，对应 n = 1 的情况
    row = [0, 1]+[0]*(k-1) # for n = 1
    # 从 n = 2 开始计算斯特林数
    for i in range(2, n+1):
        # 从后向前更新斯特林数的每一项
        for j in range(min(k,i), 0, -1):
            # 根据斯特林数的递推公式更新当前行的每一项
            row[j] = j * row[j] + row[j-1]
    # 返回斯特林数 S(n, k)
    return Integer(row[k])


# 计算斯特林数 S(n, k) 的函数，根据给定的参数返回第一或第二类斯特林数
def stirling(n, k, d=None, kind=2, signed=False):
    r"""Return Stirling number $S(n, k)$ of the first or second (default) kind.

    The sum of all Stirling numbers of the second kind for $k = 1$
    through $n$ is ``bell(n)``. The recurrence relationship for these numbers
    is:

    .. math :: {0 \brace 0} = 1; {n \brace 0} = {0 \brace k} = 0;

    .. math :: {{n+1} \brace k} = j {n \brace k} + {n \brace {k-1}}

    where $j$ is:
        $n$ for Stirling numbers of the first kind,
        $-n$ for signed Stirling numbers of the first kind,
        $k$ for Stirling numbers of the second kind.

    The first kind of Stirling number counts the number of permutations of
    ``n`` distinct items that have ``k`` cycles; the second kind counts the
    ways in which ``n`` distinct items can be partitioned into ``k`` parts.
    If ``d`` is given, the "reduced Stirling number of the second kind" is
    returned: $S^{d}(n, k) = S(n - d + 1, k - d + 1)$ with $n \ge k \ge d$.
    (This counts the ways to partition $n$ consecutive integers into $k$
    groups with no pairwise difference less than $d$. See example below.)

    To obtain the signed Stirling numbers of the first kind, use keyword
    ``signed=True``. Using this keyword automatically sets ``kind`` to 1.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import stirling, bell
    >>> from sympy.combinatorics import Permutation
    >>> from sympy.utilities.iterables import multiset_partitions, permutations

    First kind (unsigned by default):

    >>> [stirling(6, i, kind=1) for i in range(7)]
    [0, 120, 274, 225, 85, 15, 1]
    >>> perms = list(permutations(range(4)))
    >>> [sum(Permutation(p).cycles == i for p in perms) for i in range(5)]
    [0, 6, 11, 6, 1]
    >>> [stirling(4, i, kind=1) for i in range(5)]
    [0, 6, 11, 6, 1]

    First kind (signed):

    >>> [stirling(4, i, signed=True) for i in range(5)]
    [0, -6, 11, -6, 1]

    Second kind:

    >>> [stirling(10, i) for i in range(12)]
    [0, 1, 511, 9330, 34105, 42525, 22827, 5880, 750, 45, 1, 0]
    >>> sum(_) == bell(10)
    True
    >>> len(list(multiset_partitions(range(4), 2))) == stirling(4, 2)
    True

    Reduced second kind:
    # 导入 sympy 库中的 subsets 和 oo
    >>> from sympy import subsets, oo
    # 定义一个函数 delta，计算给定集合中任意两元素之间的最小差值
    >>> def delta(p):
        # 如果集合 p 的长度为 1，返回无穷大 oo
        ...    if len(p) == 1:
        ...        return oo
        # 返回集合 p 中任意两元素之间差值的最小值
        ...    return min(abs(i[0] - i[1]) for i in subsets(p, 2))
    # 使用 multiset_partitions 函数对 range(5) 进行多重集合划分，划分成 3 个部分
    >>> parts = multiset_partitions(range(5), 3)
    # 初始化变量 d 为 2
    >>> d = 2
    # 统计满足条件的部分 p 的个数，其中所有部分中的任意两元素之差均大于等于 d
    >>> sum(1 for p in parts if all(delta(i) >= d for i in p))
    7
    # 计算第一类 Stirling 数 S(5, 3)，参数为 (n=5, k=3, kind=2)
    >>> stirling(5, 3, 2)
    7
    
    # 参见
    # ========
    # sympy.utilities.iterables.multiset_partitions
    
    # 参考文献
    # ==========
    # [1] https://en.wikipedia.org/wiki/Stirling_numbers_of_the_first_kind
    # [2] https://en.wikipedia.org/wiki/Stirling_numbers_of_the_second_kind
    
    """
    # TODO: make this a class like bell()
    
    # 将 n 和 k 转换为整数
    n = as_int(n)
    k = as_int(k)
    # 如果 n 小于 0，抛出 ValueError 异常
    if n < 0:
        raise ValueError('n must be nonnegative')
    # 如果 k 大于 n，返回零 S.Zero
    if k > n:
        return S.Zero
    # 如果 d 非零
    if d:
        # kind 被忽略 -- 只支持 kind=2
        # 返回第二类 Stirling 数 _eval_stirling2(n - d + 1, k - d + 1)
        return _eval_stirling2(n - d + 1, k - d + 1)
    # 如果 signed 为真
    elif signed:
        # kind 被忽略 -- 只支持 kind=1
        # 返回 (-1)^(n - k) * 第一类 Stirling 数 _eval_stirling1(n, k)
        return S.NegativeOne**(n - k)*_eval_stirling1(n, k)
    
    # 如果 kind 为 1
    if kind == 1:
        # 返回第一类 Stirling 数 _eval_stirling1(n, k)
        return _eval_stirling1(n, k)
    # 如果 kind 为 2
    elif kind == 2:
        # 返回第二类 Stirling 数 _eval_stirling2(n, k)
        return _eval_stirling2(n, k)
    # 否则抛出 ValueError 异常，kind 必须是 1 或 2
    else:
        raise ValueError('kind must be 1 or 2, not %s' % k)
# 缓存装饰器，用于优化函数调用，避免重复计算
@cacheit
# 计算将 ``n`` 个项目分成 ``k`` 部分的分区数目。此函数在 ``n`` 为整数时被 ``nT`` 使用。
def _nT(n, k):
    """Return the partitions of ``n`` items into ``k`` parts. This
    is used by ``nT`` for the case when ``n`` is an integer."""
    # 真正的快速退出
    if k > n or k < 0:
        return 0
    if k in (1, n):
        return 1
    if k == 0:
        return 0
    # 可以在下面完成的退出，但这样更快
    if k == 2:
        return n//2
    d = n - k
    if d <= 3:
        return d
    # 快速退出
    if 3*k >= n:  # 或者等效地，2*k >= d
        # 在这种情况下，所有需要的信息都将在缓存中，以计算 partition(d)，所以...
        # 更新缓存
        tot = _partition_rec(d)
        # 并且修正不需要的值
        if d - k > 0:
            tot -= sum(_partition_rec.fetch_item(slice(d - k)))
        return tot
    # 常规退出
    # nT(n, k) = Sum(nT(n - k, m), (m, 1, k));
    # 计算需要的 nT(i, j) 值
    p = [1]*d
    for i in range(2, k + 1):
        for m in range(i + 1, d):
            p[m] += p[m - i]
        d -= 1
    # 如果 p[0] 被追加到 p 的末尾，那么 p 的最后 k 个值是逆序的 nT(n, j) 值，对于 0 < j < k。
    # p[-1] = nT(n, 1), p[-2] = nT(n, 2)，等等... 然而，不将 p[0] 中的 1 放在这里，而是简单地添加到下面的和中，对于 1 < k <= n//2 是有效的
    return (1 + sum(p[1 - k:]))

# 返回 ``n`` 项中大小为 ``k`` 的分区的数量。
def nT(n, k=None):
    """Return the number of ``k``-sized partitions of ``n`` items.

    Possible values for ``n``:

        integer - ``n`` identical items

        sequence - converted to a multiset internally

        multiset - {element: multiplicity}

    Note: the convention for ``nT`` is different than that of ``nC`` and
    ``nP`` in that
    here an integer indicates ``n`` *identical* items instead of a set of
    length ``n``; this is in keeping with the ``partitions`` function which
    treats its integer-``n`` input like a list of ``n`` 1s. One can use
    ``range(n)`` for ``n`` to indicate ``n`` distinct items.

    If ``k`` is None then the total number of ways to partition the elements
    represented in ``n`` will be returned.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import nT

    Partitions of the given multiset:

    >>> [nT('aabbc', i) for i in range(1, 7)]
    [1, 8, 11, 5, 1, 0]
    >>> nT('aabbc') == sum(_)
    True

    >>> [nT("mississippi", i) for i in range(1, 12)]
    [1, 74, 609, 1521, 1768, 1224, 579, 197, 50, 9, 1]

    Partitions when all items are identical:

    >>> [nT(5, i) for i in range(1, 6)]
    [1, 2, 2, 1, 1]
    >>> nT('1'*5) == sum(_)
    True

    When all items are different:

    >>> [nT(range(5), i) for i in range(1, 6)]
    [1, 15, 25, 10, 1]
    >>> nT(range(5)) == sum(_)
    True

    Partitions of an integer expressed as a sum of positive integers:

    >>> from sympy import partition
    >>> partition(4)
    5
    >>> nT(4, 1) + nT(4, 2) + nT(4, 3) + nT(4, 4)
    5
    """
    """
    Calculate the number of partitions of a multiset.

    Parameters
    ==========
    n : int or _MultisetHistogram
        The total number of items or the multiset histogram.
    k : int or None
        The number of parts in the partition.

    Returns
    =======
    int
        The number of partitions of the multiset.

    See Also
    ========
    sympy.utilities.iterables.partitions
    sympy.utilities.iterables.multiset_partitions
    sympy.functions.combinatorial.numbers.partition

    References
    ==========
    .. [1] https://web.archive.org/web/20210507012732/https://teaching.csse.uwa.edu.au/units/CITS7209/partition.pdf

    """

    # Check if n is an instance of SYMPY_INTS (integer-like objects)
    if isinstance(n, SYMPY_INTS):
        # Handle the case where n is a number of identical items
        if k is None:
            return partition(n)
        
        # Handle the case where both n and k are integers
        if isinstance(k, SYMPY_INTS):
            n = as_int(n)
            k = as_int(k)
            return Integer(_nT(n, k))

    # Check if n is not an instance of _MultisetHistogram
    if not isinstance(n, _MultisetHistogram):
        try:
            # Attempt to handle n as a collection of hashable items
            u = len(set(n))
            if u <= 1:
                return nT(len(n), k)  # Return the number of partitions
            elif u == len(n):
                n = range(u)
            raise TypeError
        except TypeError:
            n = _multiset_histogram(n)  # Convert n into _MultisetHistogram

    # Extract the total count N from the multiset histogram n
    N = n[_N]

    # Handle special cases based on the values of k and N
    if k is None and N == 1:
        return 1
    if k in (1, N):
        return 1
    if k == 2 or N == 2 and k is None:
        m, r = divmod(N, 2)
        rv = sum(nC(n, i) for i in range(1, m + 1))
        if not r:
            rv -= nC(n, m)//2
        if k is None:
            rv += 1  # Adjust for the case where k == 1
        return rv

    # Handle cases where all items in n are distinct
    if N == n[_ITEMS]:
        if k is None:
            return bell(N)  # Calculate Bell number for N
        return stirling(N, k)  # Calculate Stirling number of the second kind

    # Initialize a MultisetPartitionTraverser object
    m = MultisetPartitionTraverser()

    # Handle case where k is None
    if k is None:
        return m.count_partitions(n[_M])  # Count partitions of multiset

    # Count partitions using enumeration since MultisetPartitionTraverser
    # does not directly support range-limited counting
    tot = 0
    for discard in m.enum_range(n[_M], k-1, k):
        tot += 1
    return tot  # Return the total count of partitions
#-----------------------------------------------------------------------------#
#                                                                             #
#                          Motzkin numbers                                    #
#                                                                             #
#-----------------------------------------------------------------------------#

class motzkin(Function):
    """
    The nth Motzkin number is the number
    of ways of drawing non-intersecting chords
    between n points on a circle (not necessarily touching
    every point by a chord). The Motzkin numbers are named
    after Theodore Motzkin and have diverse applications
    in geometry, combinatorics and number theory.

    Motzkin numbers are the integer sequence defined by the
    initial terms `M_0 = 1`, `M_1 = 1` and the two-term recurrence relation
    `M_n = \frac{2*n + 1}{n + 2} * M_{n-1} + \frac{3n - 3}{n + 2} * M_{n-2}`.


    Examples
    ========

    >>> from sympy import motzkin

    >>> motzkin.is_motzkin(5)
    False
    >>> motzkin.find_motzkin_numbers_in_range(2,300)
    [2, 4, 9, 21, 51, 127]
    >>> motzkin.find_motzkin_numbers_in_range(2,900)
    [2, 4, 9, 21, 51, 127, 323, 835]
    >>> motzkin.find_first_n_motzkins(10)
    [1, 1, 2, 4, 9, 21, 51, 127, 323, 835]


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Motzkin_number
    .. [2] https://mathworld.wolfram.com/MotzkinNumber.html

    """

    @staticmethod
    def is_motzkin(n):
        # 将输入转换为整数，若无法转换则返回 False
        try:
            n = as_int(n)
        except ValueError:
            return False
        
        # 检查输入是否大于零
        if n > 0:
            # 对于 n 等于 1 或 2 的情况，直接返回 True
            if n in (1, 2):
                return True
            
            # 初始化 M_1 和 M_2
            tn1 = 1
            tn = 2
            i = 3
            # 使用 Motzkin 数的递推关系计算直到找到等于 n 的情况或超过 n
            while tn < n:
                a = ((2*i + 1)*tn + (3*i - 3)*tn1)/(i + 2)
                i += 1
                tn1 = tn
                tn = a
            
            # 如果计算得到的 tn 等于 n，则返回 True，否则返回 False
            if tn == n:
                return True
            else:
                return False
        else:
            return False

    @staticmethod
    def find_motzkin_numbers_in_range(x, y):
        # 检查输入的范围是否有效，若无效则抛出 ValueError 异常
        if 0 <= x <= y:
            motzkins = []
            # 如果 x 到 y 范围包含 Motzkin 数 1，则添加到列表中
            if x <= 1 <= y:
                motzkins.append(1)
            
            # 初始化 M_1 和 M_2
            tn1 = 1
            tn = 2
            i = 3
            # 使用 Motzkin 数的递推关系计算范围内的 Motzkin 数
            while tn <= y:
                if tn >= x:
                    motzkins.append(tn)
                a = ((2*i + 1)*tn + (3*i - 3)*tn1)/(i + 2)
                i += 1
                tn1 = tn
                tn = int(a)
            
            return motzkins
        else:
            # 若输入范围无效，则抛出 ValueError 异常
            raise ValueError('The provided range is not valid. This condition should satisfy x <= y')
    # 定义一个静态方法，用于计算 Motzkin 数列的前 n 项
    def find_first_n_motzkins(n):
        try:
            # 尝试将输入转换为整数
            n = as_int(n)
        except ValueError:
            # 如果转换失败，抛出异常，要求提供一个正整数
            raise ValueError('The provided number must be a positive integer')
        # 如果 n 小于 0，抛出异常，要求提供一个正整数
        if n < 0:
            raise ValueError('The provided number must be a positive integer')
        
        # 初始化 Motzkin 数列，初始项为 1
        motzkins = [1]
        # 如果 n 大于等于 1，添加第二项 1 到 Motzkin 数列中
        if n >= 1:
            motzkins.append(1)
        
        # 初始化 Motzkin 数列的递推关系的初始值
        tn1 = 1
        tn = 2
        i = 3
        # 循环计算 Motzkin 数列直到第 n 项
        while i <= n:
            # 计算当前项的值 a，根据 Motzkin 数列的递推关系
            a = ((2*i + 1)*tn + (3*i - 3)*tn1)/(i + 2)
            # 将当前项加入 Motzkin 数列
            motzkins.append(tn)
            # 更新下一轮循环所需的值
            i += 1
            tn1 = tn
            tn = int(a)
        
        # 返回计算得到的 Motzkin 数列的前 n 项
        return motzkins

    @staticmethod
    @recurrence_memo([S.One, S.One])
    # 定义一个静态方法，使用递推公式计算 Motzkin 数列的第 n 项
    def _motzkin(n, prev):
        return ((2*n + 1)*prev[-1] + (3*n - 3)*prev[-2]) // (n + 2)

    @classmethod
    # 定义一个类方法，计算 Motzkin 数列的第 n 项
    def eval(cls, n):
        try:
            # 尝试将输入转换为整数
            n = as_int(n)
        except ValueError:
            # 如果转换失败，抛出异常，要求提供一个正整数
            raise ValueError('The provided number must be a positive integer')
        # 如果 n 小于 0，抛出异常，要求提供一个正整数
        if n < 0:
            raise ValueError('The provided number must be a positive integer')
        
        # 调用静态方法 _motzkin 计算 Motzkin 数列的第 n 项，并返回结果
        return Integer(cls._motzkin(n - 1))
# 定义一个函数 nD，用于计算排列中的错位排列数
def nD(i=None, brute=None, *, n=None, m=None):
    """return the number of derangements for: ``n`` unique items, ``i``
    items (as a sequence or multiset), or multiplicities, ``m`` given
    as a sequence or multiset.

    Examples
    ========

    >>> from sympy.utilities.iterables import generate_derangements as enum
    >>> from sympy.functions.combinatorial.numbers import nD

    A derangement ``d`` of sequence ``s`` has all ``d[i] != s[i]``:

    >>> set([''.join(i) for i in enum('abc')])
    {'bca', 'cab'}
    >>> nD('abc')
    2

    Input as iterable or dictionary (multiset form) is accepted:

    >>> assert nD([1, 2, 2, 3, 3, 3]) == nD({1: 1, 2: 2, 3: 3})

    By default, a brute-force enumeration and count of multiset permutations
    is only done if there are fewer than 9 elements. There may be cases when
    there is high multiplicity with few unique elements that will benefit
    from a brute-force enumeration, too. For this reason, the `brute`
    keyword (default None) is provided. When False, the brute-force
    enumeration will never be used. When True, it will always be used.

    >>> nD('1111222233', brute=True)
    44

    For convenience, one may specify ``n`` distinct items using the
    ``n`` keyword:

    >>> assert nD(n=3) == nD('abc') == 2

    Since the number of derangments depends on the multiplicity of the
    elements and not the elements themselves, it may be more convenient
    to give a list or multiset of multiplicities using keyword ``m``:

    >>> assert nD('abc') == nD(m=(1,1,1)) == nD(m={1:3}) == 2

    """
    # 导入需要的函数和变量
    from sympy.integrals.integrals import integrate
    from sympy.functions.special.polynomials import laguerre
    from sympy.abc import x
    
    # 定义一个函数 ok，用于验证参数是否为整数
    def ok(x):
        if not isinstance(x, SYMPY_INTS):  # 如果 x 不是整数
            raise TypeError('expecting integer values')  # 抛出类型错误异常
        if x < 0:  # 如果 x 小于 0
            raise ValueError('value must not be negative')  # 抛出数值错误异常
        return True  # 返回 True 表示验证通过
    
    # 如果 i, n, m 中不止一个参数是非 None，则抛出数值错误异常
    if (i, n, m).count(None) != 2:
        raise ValueError('enter only 1 of i, n, or m')
    
    # 如果指定了 i 参数
    if i is not None:
        if isinstance(i, SYMPY_INTS):  # 如果 i 是整数
            raise TypeError('items must be a list or dictionary')  # 抛出类型错误异常
        if not i:  # 如果 i 是空的
            return S.Zero  # 返回 SymPy 中的零值
        if type(i) is not dict:  # 如果 i 不是字典类型
            s = list(i)  # 将 i 转换为列表 s
            ms = multiset(s)  # 使用 multiset 函数创建多重集合 ms
        elif type(i) is dict:  # 如果 i 是字典类型
            all(ok(_) for _ in i.values())  # 验证字典 i 的值是否都是整数
            ms = {k: v for k, v in i.items() if v}  # 创建非空值的字典 ms
            s = None  # 设置 s 为 None
        if not ms:  # 如果 ms 是空的
            return S.Zero  # 返回 SymPy 中的零值
        N = sum(ms.values())  # 计算多重集合中元素的总数 N
        counts = multiset(ms.values())  # 创建多重集合 counts，用于计算各元素的个数
        nkey = len(ms)  # 计算多重集合 ms 的长度
    # 如果指定了 n 参数
    elif n is not None:
        ok(n)  # 验证 n 是否为整数
        if not n:  # 如果 n 是空的
            return S.Zero  # 返回 SymPy 中的零值
        return subfactorial(n)  # 返回 n 的错位排列数
    elif m is not None:
        # 如果 m 不为 None，则进入条件判断
        if isinstance(m, dict):
            # 如果 m 是字典类型
            all(ok(i) and ok(j) for i, j in m.items())
            # 对字典 m 中的每对键值进行检查
            counts = {k: v for k, v in m.items() if k*v}
            # 构建 counts 字典，包含满足条件 k*v 的键值对
        elif iterable(m) or isinstance(m, str):
            # 如果 m 是可迭代对象或字符串类型
            m = list(m)
            # 将 m 转换为列表
            all(ok(i) for i in m)
            # 对 m 中的每个元素进行检查
            counts = multiset([i for i in m if i])
            # 使用 multiset 创建一个多重集，包含 m 中非空元素的计数
        else:
            # 如果 m 类型不符合预期，抛出类型错误异常
            raise TypeError('expecting iterable')
        if not counts:
            # 如果 counts 为空集合
            return S.Zero
        N = sum(k*v for k, v in counts.items())
        # 计算 N，即 counts 中所有键值乘积的总和
        nkey = sum(counts.values())
        # 计算 nkey，即 counts 中所有值的总和
        s = None
    big = int(max(counts))
    # 计算 counts 中最大值并转换为整数
    if big == 1:  # no repetition
        # 如果最大值为 1，即无重复
        return subfactorial(nkey)
        # 返回 nkey 的子阶乘值
    nval = len(counts)
    # 计算 counts 的长度，即键值对的数量
    if big*2 > N:
        # 如果 counts 中最大值的两倍大于 N
        return S.Zero
        # 返回零
    if big*2 == N:
        # 如果 counts 中最大值的两倍等于 N
        if nkey == 2 and nval == 1:
            return S.One  # aaabbb
            # 返回一，表示类似 aaabbb 的情况
        if nkey - 1 == big:  # one element repeated
            return factorial(big)  # e.g. abc part of abcddd
            # 返回 big 的阶乘，例如 abc 中的 abcddd 的一部分
    if N < 9 and brute is None or brute:
        # 如果 N 小于 9 并且 brute 为 None 或者 True
        # 对于所有可能性，发现这种方式更快
        if s is None:
            s = []
            i = 0
            for m, v in counts.items():
                for j in range(v):
                    s.extend([i]*m)
                    i += 1
            # 构建 s 列表，包含所有可能的排列组合
        return Integer(sum(1 for i in multiset_derangements(s)))
        # 返回 s 的 derangement 数的总和
    from sympy.functions.elementary.exponential import exp
    return Integer(abs(integrate(exp(-x)*Mul(*[
        laguerre(i, x)**m for i, m in counts.items()]), (x, 0, oo))))
    # 返回积分结果的绝对值，计算特定函数的复杂表达式
```