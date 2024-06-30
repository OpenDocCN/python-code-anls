# `D:\src\scipysrc\sympy\sympy\core\intfunc.py`

```
"""
The routines here were removed from numbers.py, power.py,
digits.py and factor_.py so they could be imported into core
without raising circular import errors.

Although the name 'intfunc' was chosen to represent functions that
work with integers, it can also be thought of as containing
internal/core functions that are needed by the classes of the core.
"""

# 导入所需的模块
import math
import sys
from functools import lru_cache

# 导入自定义的模块
from .sympify import sympify
from .singleton import S
from sympy.external.gmpy import (gcd as number_gcd, lcm as number_lcm, sqrt,
                                 iroot, bit_scan1, gcdext)
from sympy.utilities.misc import as_int, filldedent

# 定义函数：计算在给定进制下表示整数 n 所需的位数
def num_digits(n, base=10):
    """Return the number of digits needed to express n in give base.

    Examples
    ========

    >>> from sympy.core.intfunc import num_digits
    >>> num_digits(10)
    2
    >>> num_digits(10, 2)  # 1010 -> 4 digits
    4
    >>> num_digits(-100, 16)  # -64 -> 2 digits
    2


    Parameters
    ==========

    n: integer
        The number whose digits are counted.

    b: integer
        The base in which digits are computed.

    See Also
    ========
    sympy.ntheory.digits.digits, sympy.ntheory.digits.count_digits
    """
    # 如果进制小于 0，则引发错误
    if base < 0:
        raise ValueError('base must be int greater than 1')
    # 如果 n 为 0，则返回 1，因为 0 的任何进制都只有一个位
    if not n:
        return 1
    # 计算 n 的绝对值在给定进制下所需的位数
    e, t = integer_log(abs(n), base)
    return 1 + e


# 定义函数：计算整数 n 在基数 b 下的对数
def integer_log(n, b):
    r"""
    Returns ``(e, bool)`` where e is the largest nonnegative integer
    such that :math:`|n| \geq |b^e|` and ``bool`` is True if $n = b^e$.

    Examples
    ========

    >>> from sympy import integer_log
    >>> integer_log(125, 5)
    (3, True)
    >>> integer_log(17, 9)
    (1, False)

    If the base is positive and the number negative the
    return value will always be the same except for 2:

    >>> integer_log(-4, 2)
    (2, False)
    >>> integer_log(-16, 4)
    (0, False)

    When the base is negative, the returned value
    will only be True if the parity of the exponent is
    correct for the sign of the base:

    >>> integer_log(4, -2)
    (2, True)
    >>> integer_log(8, -2)
    (3, False)
    >>> integer_log(-8, -2)
    (3, True)
    >>> integer_log(-4, -2)
    (2, False)

    See Also
    ========
    integer_nthroot
    sympy.ntheory.primetest.is_square
    sympy.ntheory.factor_.multiplicity
    sympy.ntheory.factor_.perfect_power
    """
    # 强制将 n 和 b 转换为整数类型
    n = as_int(n)
    b = as_int(b)

    # 如果基数 b 小于 0，则取绝对值重新计算
    if b < 0:
        e, t = integer_log(abs(n), -b)
        # 当基数为负数时，处理幂次的奇偶性以及 n 的正负关系
        t = t and e % 2 == (n < 0)
        return e, t
    # 如果基数为 1 或者小于等于 1，引发错误
    if b <= 1:
        raise ValueError('base must be 2 or more')
    # 如果 n 小于 0，且基数不为 2，返回 (0, False)
    if n < 0:
        if b != 2:
            return 0, False
        # 如果基数为 2，计算 -n 的对数，并返回 (e, False)
        e, t = integer_log(-n, b)
        return e, False
    # 如果 n 等于 0，引发错误
    if n == 0:
        raise ValueError('n cannot be 0')

    # 如果 n 小于基数 b，返回 (0, n == 1)
    if n < b:
        return 0, n == 1
    # 如果基数为 2，计算 n 的位数减一，并检查是否满足条件
    if b == 2:
        e = n.bit_length() - 1
        return e, trailing(n) == e
    # 计算基数的末尾零的数量
    t = trailing(b)
    # 如果 2 的 t 次方等于 b
    if 2**t == b:
        # 计算 e，e 是 n 二进制长度减 1 除以 t 的整数部分
        e = int(n.bit_length() - 1) // t
        # 计算 n_，n_ 是 t*e 的 2 的幂次方
        n_ = 1 << (t*e)
        # 返回 e 和 n_ 是否等于 n 的布尔值
        return e, n_ == n
    
    # 计算 d，d 是 n 的以 b 为底的对数的整数部分
    d = math.floor(math.log10(n) / math.log10(b))
    # 计算 n_，n_ 是 b 的 d 次方
    n_ = b ** d
    
    # 当 n_ 小于等于 n 时循环，这个循环最多迭代 0、1 或 2 次
    while n_ <= n:
        # 增加 d 的值
        d += 1
        # 计算 n_，n_ 是 b 的 d 次方
        n_ *= b
    
    # 返回 d 减去 (n_ 大于 n 的布尔值)，以及 n_ 是否等于 n 或者 n_ 除以 b 是否等于 n 的布尔值
    return d - (n_ > n), (n_ == n or n_ // b == n)
# 定义一个函数，用于计算整数 n 的二进制表示中末尾连续零的个数，
# 即确定能整除 n 的最大的2的幂次数。
def trailing(n):
    """Count the number of trailing zero digits in the binary
    representation of n, i.e. determine the largest power of 2
    that divides n.

    Examples
    ========

    >>> from sympy import trailing
    >>> trailing(128)
    7
    >>> trailing(63)
    0

    See Also
    ========
    sympy.ntheory.factor_.multiplicity

    """
    # 如果 n 是 0，则直接返回 0，因为二进制中没有末尾的零
    if not n:
        return 0
    # 调用 bit_scan1 函数来计算 n 的末尾连续零的个数
    return bit_scan1(int(n))


# igcd 函数使用了 lru_cache 装饰器，提供了一个缓存机制，用于计算多个整数的最大公约数
@lru_cache(1024)
def igcd(*args):
    """Computes nonnegative integer greatest common divisor.

    Explanation
    ===========

    The algorithm is based on the well known Euclid's algorithm [1]_. To
    improve speed, ``igcd()`` has its own caching mechanism.
    If you do not need the cache mechanism, using ``sympy.external.gmpy.gcd``.

    Examples
    ========

    >>> from sympy import igcd
    >>> igcd(2, 4)
    2
    >>> igcd(5, 10, 15)
    5

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Euclidean_algorithm

    """
    # 如果传入的参数少于 2 个，则抛出 TypeError 异常
    if len(args) < 2:
        raise TypeError("igcd() takes at least 2 arguments (%s given)" % len(args))
    # 调用 number_gcd 函数计算 args 中整数的最大公约数，并转换为整数返回
    return int(number_gcd(*map(as_int, args)))


# igcd2 函数使用了 math 模块中的 gcd 函数，直接计算两个整数的最大公约数
igcd2 = math.gcd


def igcd_lehmer(a, b):
    r"""Computes greatest common divisor of two integers.

    Explanation
    ===========

    Euclid's algorithm for the computation of the greatest
    common divisor ``gcd(a, b)``  of two (positive) integers
    $a$ and $b$ is based on the division identity
    $$ a = q \times b + r$$,
    where the quotient  $q$  and the remainder  $r$  are integers
    and  $0 \le r < b$. Then each common divisor of  $a$  and  $b$
    divides  $r$, and it follows that  ``gcd(a, b) == gcd(b, r)``.
    The algorithm works by constructing the sequence
    r0, r1, r2, ..., where  r0 = a, r1 = b,  and each  rn
    is the remainder from the division of the two preceding
    elements.

    In Python, ``q = a // b``  and  ``r = a % b``  are obtained by the
    floor division and the remainder operations, respectively.
    These are the most expensive arithmetic operations, especially
    for large  a  and  b.

    Lehmer's algorithm [1]_ is based on the observation that the quotients
    ``qn = r(n-1) // rn``  are in general small integers even
    when  a  and  b  are very large. Hence the quotients can be
    usually determined from a relatively small number of most
    significant bits.

    The efficiency of the algorithm is further enhanced by not
    computing each long remainder in Euclid's sequence. The remainders
    are linear combinations of  a  and  b  with integer coefficients
    derived from the quotients. The coefficients can be computed
    as far as the quotients can be determined from the chosen
    most significant parts of  a  and  b. Only then a new pair of
    consecutive remainders is computed and the algorithm starts
    anew with this pair.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Lehmer%27s_GCD_algorithm

    """
    # 将 a 和 b 转换为非负整数
    a, b = abs(as_int(a)), abs(as_int(b))
    # 如果 a 小于 b，则交换它们的值，确保 a 总是大于或等于 b
    if a < b:
        a, b = b, a

    # 算法通过使用一位或两位数的除法来尽可能快地进行计算。
    # 外部循环将替换 (a, b) 这一对数值，用欧几里德最大公约数序列中
    # 较短的连续元素对替代 a 和 b，直到 a 和 b 可以在两个 Python（长）整数位数内表示。
    nbits = 2 * sys.int_info.bits_per_digit

    # 使用较小的除数。最终采用标准算法完成。
    while b:
        # 更新 a 和 b，使得 b 成为 a 除以 b 的余数
        a, b = b, a % b

    # 返回最大公约数
    return a
# 定义一个函数，计算多个整数的最小公倍数
def ilcm(*args):
    """Computes integer least common multiple.

    Examples
    ========

    >>> from sympy import ilcm
    >>> ilcm(5, 10)
    10
    >>> ilcm(7, 3)
    21
    >>> ilcm(5, 10, 15)
    30

    """
    # 如果传入的参数少于两个，抛出类型错误异常
    if len(args) < 2:
        raise TypeError("ilcm() takes at least 2 arguments (%s given)" % len(args))
    # 调用 number_lcm 函数计算参数的最小公倍数，并返回整数结果
    return int(number_lcm(*map(as_int, args)))


# 定义一个函数，返回对于给定的两个整数 a 和 b，满足 gcd(a, b) = x*a + y*b 的 x, y, g
def igcdex(a, b):
    """Returns x, y, g such that g = x*a + y*b = gcd(a, b).

    Examples
    ========

    >>> from sympy.core.intfunc import igcdex
    >>> igcdex(2, 3)
    (-1, 1, 1)
    >>> igcdex(10, 12)
    (-1, 1, 2)

    >>> x, y, g = igcdex(100, 2004)
    >>> x, y, g
    (-20, 1, 4)
    >>> x*100 + y*2004
    4

    """
    # 如果 a 和 b 都为 0，则返回 (0, 1, 0)
    if (not a) and (not b):
        return (0, 1, 0)
    # 调用 gcdext 函数计算 a 和 b 的扩展欧几里得算法结果，返回 x, y, g
    g, x, y = gcdext(int(a), int(b))
    return x, y, g


# 定义一个函数，返回满足 a * c ≡ 1 (mod m) 的 c 值，若不存在则抛出 ValueError 异常
def mod_inverse(a, m):
    r"""
    Return the number $c$ such that, $a \times c = 1 \pmod{m}$
    where $c$ has the same sign as $m$. If no such value exists,
    a ValueError is raised.

    Examples
    ========

    >>> from sympy import mod_inverse, S

    Suppose we wish to find multiplicative inverse $x$ of
    3 modulo 11. This is the same as finding $x$ such
    that $3x = 1 \pmod{11}$. One value of x that satisfies
    this congruence is 4. Because $3 \times 4 = 12$ and $12 = 1 \pmod{11}$.
    This is the value returned by ``mod_inverse``:

    >>> mod_inverse(3, 11)
    4
    >>> mod_inverse(-3, 11)
    7

    When there is a common factor between the numerators of
    `a` and `m` the inverse does not exist:

    >>> mod_inverse(2, 4)
    Traceback (most recent call last):
    ...
    ValueError: inverse of 2 mod 4 does not exist

    >>> mod_inverse(S(2)/7, S(5)/2)
    7/2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Modular_multiplicative_inverse
    .. [2] https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm
    """
    # 初始化 c 为 None
    c = None
    try:
        # 尝试将 a 和 m 转换为整数
        a, m = as_int(a), as_int(m)
        # 如果 m 不等于 1 且不等于 -1
        if m != 1 and m != -1:
            # 调用 igcdex 函数计算 a 和 m 的扩展欧几里得算法结果，得到 x, y, g
            x, _, g = igcdex(a, m)
            # 如果 g 等于 1，计算 x 对 m 取模的结果作为 c
            if g == 1:
                c = x % m
    except ValueError:
        # 如果转换失败，尝试使用符号表达式进行处理
        a, m = sympify(a), sympify(m)
        # 如果 a 和 m 不是数值类型，则抛出类型错误异常
        if not (a.is_number and m.is_number):
            raise TypeError(
                filldedent(
                    """
                Expected numbers for arguments; symbolic `mod_inverse`
                is not implemented
                but symbolic expressions can be handled with the
                similar function,
                sympy.polys.polytools.invert"""
                )
            )
        # 检查 m 是否大于 1，若不是则抛出值错误异常
        big = m > 1
        if big not in (S.true, S.false):
            raise ValueError("m > 1 did not evaluate; try to simplify %s" % m)
        # 若 m 大于 1，计算 1/a 的结果作为 c
        elif big:
            c = 1 / a
    # 如果 c 为 None，抛出值错误异常
    if c is None:
        raise ValueError("inverse of %s (mod %s) does not exist" % (a, m))
    # 返回计算得到的 c 值
    return c


# 定义一个函数，返回不大于 sqrt(n) 的最大整数
def isqrt(n):
    r""" Return the largest integer less than or equal to `\sqrt{n}`.

    Parameters
    ==========

    n : non-negative integer

    Returns
    =======
    int : `\left\lfloor\sqrt{n}\right\rfloor`

    # 这行指定函数返回类型为整数，表示对输入 n 开平方后向下取整的结果

    Raises
    ======

    ValueError
        If n is negative.
    TypeError
        If n is of a type that cannot be compared to ``int``.
        Therefore, a TypeError is raised for ``str``, but not for ``float``.

    # 定义函数可能引发的异常情况及其描述：

    # 如果 n 为负数，则引发 ValueError 异常。
    # 如果 n 的类型无法与整数比较，则引发 TypeError 异常。因此，对于字符串（str），会引发 TypeError 异常，但对于浮点数（float），不会引发。

    Examples
    ========

    >>> from sympy.core.intfunc import isqrt
    >>> isqrt(0)
    0
    >>> isqrt(9)
    3
    >>> isqrt(10)
    3
    >>> isqrt("30")
    Traceback (most recent call last):
        ...
    TypeError: '<' not supported between instances of 'str' and 'int'
    >>> from sympy.core.numbers import Rational
    >>> isqrt(Rational(-1, 2))
    Traceback (most recent call last):
        ...
    ValueError: n must be nonnegative

    """

    # 以下是函数 isqrt 的使用示例及其预期输出：

    # 调用 isqrt(0) 应返回 0
    # 调用 isqrt(9) 应返回 3
    # 调用 isqrt(10) 应返回 3
    # 调用 isqrt("30") 应引发 TypeError 异常，因为字符串与整数无法比较
    # 调用 isqrt(Rational(-1, 2)) 应引发 ValueError 异常，因为 n 必须是非负数

    """
    if n < 0:
        raise ValueError("n must be nonnegative")
    return int(sqrt(int(n)))

    # 如果输入 n 小于 0，则抛出 ValueError 异常，说明 n 必须是非负数。
    # 否则，计算 n 的平方根并将结果转换为整数后返回。
# 计算整数 y 的 n 次根的整数部分 x 和一个布尔值，指示结果是否精确（即 x**n == y）。

def integer_nthroot(y, n):
    """
    Return a tuple containing x = floor(y**(1/n))
    and a boolean indicating whether the result is exact (that is,
    whether x**n == y).

    Examples
    ========

    >>> from sympy import integer_nthroot
    >>> integer_nthroot(16, 2)
    (4, True)
    >>> integer_nthroot(26, 2)
    (5, False)

    To simply determine if a number is a perfect square, the is_square
    function should be used:

    >>> from sympy.ntheory.primetest import is_square
    >>> is_square(26)
    False

    See Also
    ========
    sympy.ntheory.primetest.is_square
    integer_log
    """

    # 使用 iroot 函数计算 y 的 n 次根，as_int 函数确保输入为整数
    x, b = iroot(as_int(y), as_int(n))
    # 返回 x 的整数部分和布尔值 b
    return int(x), b
```