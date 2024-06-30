# `D:\src\scipysrc\sympy\sympy\ntheory\primetest.py`

```
"""
Primality testing

"""

# 导入必要的模块和函数
from itertools import count  # 导入 count 函数，用于生成连续整数序列

from sympy.core.sympify import sympify  # 导入 sympify 函数，用于将输入转换为 SymPy 表达式
from sympy.external.gmpy import (gmpy as _gmpy, gcd, jacobi,  # 导入 gmpy 模块中的多个函数和常量
                                 is_square as gmpy_is_square,
                                 bit_scan1, is_fermat_prp, is_euler_prp,
                                 is_selfridge_prp, is_strong_selfridge_prp,
                                 is_strong_bpsw_prp)
from sympy.external.ntheory import _lucas_sequence  # 导入 _lucas_sequence 函数，用于计算 Lucas 序列
from sympy.utilities.misc import as_int, filldedent  # 导入 as_int 和 filldedent 函数，用于处理整数和文本格式化

# Note: This list should be updated whenever new Mersenne primes are found.
# Refer: https://www.mersenne.org/
# 质数指数列表，包含已知的 Mersenne 质数指数
MERSENNE_PRIME_EXPONENTS = (2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127, 521, 607, 1279, 2203,
 2281, 3217, 4253, 4423, 9689, 9941, 11213, 19937, 21701, 23209, 44497, 86243, 110503, 132049,
 216091, 756839, 859433, 1257787, 1398269, 2976221, 3021377, 6972593, 13466917, 20996011, 24036583,
 25964951, 30402457, 32582657, 37156667, 42643801, 43112609, 57885161, 74207281, 77232917, 82589933)


def is_fermat_pseudoprime(n, a):
    r"""Returns True if ``n`` is prime or is an odd composite integer that
    is coprime to ``a`` and satisfy the modular arithmetic congruence relation:

    .. math ::
        a^{n-1} \equiv 1 \pmod{n}

    (where mod refers to the modulo operation).

    Parameters
    ==========

    n : Integer
        ``n`` is a positive integer.
    a : Integer
        ``a`` is a positive integer.
        ``a`` and ``n`` should be relatively prime.

    Returns
    =======

    bool : If ``n`` is prime, it always returns ``True``.
           The composite number that returns ``True`` is called an Fermat pseudoprime.

    Examples
    ========

    >>> from sympy.ntheory.primetest import is_fermat_pseudoprime
    >>> from sympy.ntheory.factor_ import isprime
    >>> for n in range(1, 1000):
    ...     if is_fermat_pseudoprime(n, 2) and not isprime(n):
    ...         print(n)
    341
    561
    645

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Fermat_pseudoprime
    """
    n, a = as_int(n), as_int(a)  # 将 n 和 a 转换为整数类型
    if a == 1:
        return n == 2 or bool(n % 2)  # 如果 a == 1，检查 n 是否为奇数
    return is_fermat_prp(n, a)  # 使用 SymPy 提供的 is_fermat_prp 函数判断是否为 Fermat 伪素数


def is_euler_pseudoprime(n, a):
    r"""Returns True if ``n`` is prime or is an odd composite integer that
    is coprime to ``a`` and satisfy the modular arithmetic congruence relation:

    .. math ::
        a^{(n-1)/2} \equiv \pm 1 \pmod{n}

    (where mod refers to the modulo operation).

    Parameters
    ==========

    n : Integer
        ``n`` is a positive integer.
    a : Integer
        ``a`` is a positive integer.
        ``a`` and ``n`` should be relatively prime.

    Returns
    =======

    bool : If ``n`` is prime, it always returns ``True``.
           The composite number that returns ``True`` is called an Euler pseudoprime.

    Examples
    ========

    >>> from sympy.ntheory.primetest import is_euler_pseudoprime
    >>> from sympy.ntheory.factor_ import isprime
    # 对给定的整数 n 和底数 a 进行欧拉伪素数测试
    n, a = as_int(n), as_int(a)
    
    # 如果底数 a 小于 1，则抛出数值错误异常
    if a < 1:
        raise ValueError("a should be an integer greater than 0")
    
    # 如果整数 n 小于 1，则抛出数值错误异常
    if n < 1:
        raise ValueError("n should be an integer greater than 0")
    
    # 如果 n 等于 1，则返回 False，因为 1 不是素数也不是伪素数
    if n == 1:
        return False
    
    # 如果底数 a 等于 1，则返回判断表达式 n == 2 or bool(n % 2)
    # 其中 n == 2 表示 n 是素数，或者 bool(n % 2) 表示 n 是奇数合数
    if a == 1:
        return n == 2 or bool(n % 2)
    
    # 如果 n 是偶数，则返回判断表达式 n == 2，因为偶数只有 2 是素数
    if n % 2 == 0:
        return n == 2
    
    # 如果 n 和 a 的最大公约数不等于 1，则抛出数值错误异常
    if gcd(n, a) != 1:
        raise ValueError("The two numbers should be relatively prime")
    
    # 使用快速幂算法判断 a^((n-1)/2) 对 n 取模的结果是否在 [1, n-1] 中
    return pow(a, (n - 1) // 2, n) in [1, n - 1]
def is_euler_jacobi_pseudoprime(n, a):
    r"""Returns True if ``n`` is prime or is an odd composite integer that
    is coprime to ``a`` and satisfy the modular arithmetic congruence relation:

    .. math ::
        a^{(n-1)/2} \equiv \left(\frac{a}{n}\right) \pmod{n}

    (where mod refers to the modulo operation).

    Parameters
    ==========

    n : Integer
        ``n`` is a positive integer.
    a : Integer
        ``a`` is a positive integer.
        ``a`` and ``n`` should be relatively prime.

    Returns
    =======

    bool : If ``n`` is prime, it always returns ``True``.
           The composite number that returns ``True`` is called an Euler-Jacobi pseudoprime.

    Examples
    ========

    >>> from sympy.ntheory.primetest import is_euler_jacobi_pseudoprime
    >>> from sympy.ntheory.factor_ import isprime
    >>> for n in range(1, 1000):
    ...     if is_euler_jacobi_pseudoprime(n, 2) and not isprime(n):
    ...         print(n)
    561

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Euler%E2%80%93Jacobi_pseudoprime
    """
    # 将 n 和 a 转换为整数类型
    n, a = as_int(n), as_int(a)
    # 若 a 等于 1，则返回 n 是否为 2 或奇数
    if a == 1:
        return n == 2 or bool(n % 2)
    # 否则调用 Euler 伪素数测试函数 is_euler_prp
    return is_euler_prp(n, a)


def is_square(n, prep=True):
    """Return True if n == a * a for some integer a, else False.
    If n is suspected of *not* being a square then this is a
    quick method of confirming that it is not.

    Examples
    ========

    >>> from sympy.ntheory.primetest import is_square
    >>> is_square(25)
    True
    >>> is_square(2)
    False

    References
    ==========

    .. [1]  https://mersenneforum.org/showpost.php?p=110896

    See Also
    ========
    sympy.core.intfunc.isqrt
    """
    # 如果 prep 为 True，将 n 转换为整数类型
    if prep:
        n = as_int(n)
        # 若 n 小于 0，则返回 False
        if n < 0:
            return False
        # 若 n 为 0 或 1，则返回 True
        if n in (0, 1):
            return True
    # 否则调用 gmpy_is_square 函数进行平方数判断
    return gmpy_is_square(n)


def _test(n, base, s, t):
    """Miller-Rabin strong pseudoprime test for one base.
    Return False if n is definitely composite, True if n is
    probably prime, with a probability greater than 3/4.

    """
    # 进行费马测试
    b = pow(base, t, n)
    # 若 b 等于 1 或 b 等于 n - 1，则返回 True
    if b == 1 or b == n - 1:
        return True
    # 进行 s - 1 次平方检查
    for _ in range(s - 1):
        b = pow(b, 2, n)
        # 若 b 等于 n - 1，则返回 True
        if b == n - 1:
            return True
        # 参考 Niven 等人的书籍，如果 b 等于 1，则返回 False
        if b == 1:
            return False
    # 最终返回 False，表示 n 明确是合数
    return False


def mr(n, bases):
    """Perform a Miller-Rabin strong pseudoprime test on n using a
    given list of bases/witnesses.

    References
    ==========

    .. [1] Richard Crandall & Carl Pomerance (2005), "Prime Numbers:
           A Computational Perspective", Springer, 2nd edition, 135-138

    A list of thresholds and the bases they require are here:
    https://en.wikipedia.org/wiki/Miller%E2%80%93Rabin_primality_test#Deterministic_variants

    Examples
    ========

    >>> from sympy.ntheory.primetest import mr
    >>> mr(1373651, [2, 3])
    False
    """
    # 使用给定的一组基/witnesses 执行 Miller-Rabin 强伪素数测试
    # 返回 False 表示 n 明确是合数
    # 参考文献 [1] 中描述了阈值和它们所需的基
    pass
    >>> mr(479001599, [31, 73])
    True


# 调用Miller-Rabin素性测试函数，检查479001599是否为质数，使用基础[31, 73]进行测试
"""
from sympy.polys.domains import ZZ

# 将n转换为整数
n = as_int(n)
# 如果n小于2，则直接返回False，因为小于2的数不是质数
if n < 2:
    return False
# 计算n-1中的最高位1的位置，即s，同时计算t = n // 2**s
s = bit_scan1(n - 1)
t = n >> s
for base in bases:
    # 如果base大于等于n，则取其模n的余数，以确保base在有效范围内
    if base >= n:
        base %= n
    # 如果base大于等于2，则将其转换为整数对象
    if base >= 2:
        base = ZZ(base)
        # 使用Miller-Rabin素性测试进行检查，如果不通过则返回False
        if not _test(n, base, s, t):
            return False
# 如果所有基础都通过了Miller-Rabin测试，则n很可能是一个质数，返回True
return True
# 计算给定正奇数 n 的“额外强大”参数（D, P, Q）。

def _lucas_extrastrong_params(n):
    # 使用 itertools.count 生成从 3 开始的整数序列 P
    for P in count(3):
        # 计算 D = P^2 - 4
        D = P**2 - 4
        # 计算 D 对 n 的雅可比符号
        j = jacobi(D, n)
        # 如果雅可比符号为 -1，返回 (D, P, 1)
        if j == -1:
            return (D, P, 1)
        # 如果雅可比符号为 0 且 D 对 n 不为零，则返回 (0, 0, 0)
        elif j == 0 and D % n:
            return (0, 0, 0)


# 标准的 Lucas 复合性测试，使用 Selfridge 参数。如果 n 明确为合数，则返回 False，如果 n 是 Lucas 可能素数，则返回 True。
def is_lucas_prp(n):
    # 将 n 转换为整数
    n = as_int(n)
    # 如果 n 小于 2，则返回 False
    if n < 2:
        return False
    # 调用 is_selfridge_prp 函数，进行 Selfridge 测试
    return is_selfridge_prp(n)


# 强 Lucas 复合性测试，使用 Selfridge 参数。如果 n 明确为合数，则返回 False，如果 n 是强 Lucas 可能素数，则返回 True。
def is_strong_lucas_prp(n):
    # 将 n 转换为整数
    n = as_int(n)
    # 如果 n 小于 2，则返回 False
    if n < 2:
        return False
    # 调用 is_selfridge_prp 函数，进行 Selfridge 测试
    return is_selfridge_prp(n)
    # 将输入的参数 n 转换为整数（如果可能的话）
    n = as_int(n)
    # 如果 n 小于 2，则 n 不是素数，返回 False
    if n < 2:
        return False
    # 调用函数 is_strong_selfridge_prp(n)，检查 n 是否是强 Selfridge PRP（概率素性测试）
    return is_strong_selfridge_prp(n)
def is_extra_strong_lucas_prp(n):
    """Extra Strong Lucas compositeness test.  Returns False if n is
    definitely composite, and True if n is an "extra strong" Lucas probable
    prime.

    The parameters are selected using P = 3, Q = 1, then incrementing P until
    (D|n) == -1.  The test itself is as defined in [1]_, from the
    Mo and Jones preprint.  The parameter selection and test are the same as
    used in OEIS A217719, Perl's Math::Prime::Util, and the Lucas pseudoprime
    page on Wikipedia.

    It is 20-50% faster than the strong test.

    Because of the different parameters selected, there is no relationship
    between the strong Lucas pseudoprimes and extra strong Lucas pseudoprimes.
    In particular, one is not a subset of the other.

    References
    ==========
    .. [1] Jon Grantham, Frobenius Pseudoprimes,
           Math. Comp. Vol 70, Number 234 (2001), pp. 873-891,
           https://doi.org/10.1090%2FS0025-5718-00-01197-2
    .. [2] OEIS A217719: Extra Strong Lucas Pseudoprimes
           https://oeis.org/A217719
    .. [3] https://en.wikipedia.org/wiki/Lucas_pseudoprime

    Examples
    ========

    >>> from sympy.ntheory.primetest import isprime, is_extra_strong_lucas_prp
    >>> for i in range(20000):
    ...     if is_extra_strong_lucas_prp(i) and not isprime(i):
    ...        print(i)
    989
    3239
    5777
    10877
    """
    # Implementation notes:
    #   1) the parameters differ from Thomas R. Nicely's.  His parameter
    #      selection leads to pseudoprimes that overlap M-R tests, and
    #      contradict Baillie and Wagstaff's suggestion of (D|n) = -1.
    #   2) The MathWorld page as of June 2013 specifies Q=-1.  The Lucas
    #      sequence must have Q=1.  See Grantham theorem 2.3, any of the
    #      references on the MathWorld page, or run it and see Q=-1 is wrong.
    
    # Convert n to an integer if it's not already
    n = as_int(n)
    # Return True immediately if n is 2 (the only even prime number)
    if n == 2:
        return True
    # Return False if n is less than 2 or even
    if n < 2 or (n % 2) == 0:
        return False
    # Return False if n is a perfect square
    if gmpy_is_square(n):
        return False

    # Get parameters D, P, Q for the extra strong Lucas test
    D, P, Q = _lucas_extrastrong_params(n)
    # If D is zero, return False
    if D == 0:
        return False

    # Compute s (number of trailing zeros in n+1) and k
    s = bit_scan1(n + 1)
    k = (n + 1) >> s

    # Compute the U, V values using the Lucas sequence function
    U, V, _ = _lucas_sequence(n, P, Q, k)

    # Check if n is an extra strong Lucas pseudoprime
    if U == 0 and (V == 2 or V == n - 2):
        return True
    for _ in range(1, s):
        if V == 0:
            return True
        V = (V*V - 2) % n
    return False


def proth_test(n):
    r""" Test if the Proth number `n = k2^m + 1` is prime. where k is a positive odd number and `2^m > k`.

    Parameters
    ==========

    n : Integer
        ``n`` is Proth number

    Returns
    =======

    bool : If ``True``, then ``n`` is the Proth prime

    Raises
    ======

    ValueError
        If ``n`` is not Proth number.

    Examples
    ========

    >>> from sympy.ntheory.primetest import proth_test
    >>> proth_test(41)
    True
    >>> proth_test(57)
    False

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Proth_prime
    """
    """
    # 将输入的参数 `n` 转换为整数
    n = as_int(n)
    # 如果 `n` 小于 3，则抛出数值错误异常，表示 `n` 不是Proth数
    if n < 3:
        raise ValueError("n is not Proth number")
    # 计算 `m`，其值为 `(n - 1)` 的二进制中最右边的1的位置
    m = bit_scan1(n - 1)
    # 计算 `k`，其值为 `n` 右移 `m` 位后的结果
    k = n >> m
    # 如果 `m` 小于 `k` 的二进制长度，则抛出数值错误异常，表示 `n` 不是Proth数
    if m < k.bit_length():
        raise ValueError("n is not Proth number")
    # 如果 `n` 能被 3 整除，则返回 `n` 是否等于 3
    if n % 3 == 0:
        return n == 3
    # 如果 `k` 除以 3 的余数不为 0（即 `n % 12 == 5`），则判断条件成立
    if k % 3:
        # 返回判断条件 `pow(3, n >> 1, n) == n - 1` 的结果
        return pow(3, n >> 1, n) == n - 1
    # 如果 `n` 是一个平方数，则返回 `False`
    if gmpy_is_square(n):
        return False
    # 在此处选择 `a` 为范围从 5 到 `n` 的数值
    # 寻找使得 `jacobi(a, n) = -1` 的 `a`
    for a in range(5, n):
        j = jacobi(a, n)
        # 如果 `jacobi(a, n) = -1`，则返回判断条件 `pow(a, n >> 1, n) == n - 1` 的结果
        if j == -1:
            return pow(a, n >> 1, n) == n - 1
        # 如果 `jacobi(a, n) = 0`，则返回 `False`
        if j == 0:
            return False
# 判断给定的奇素数 p 是否为 Mersenne 数 `M_p = 2^p - 1` 的素数
def _lucas_lehmer_primality_test(p):
    v = 4  # 初始化 v 为 4
    m = 2**p - 1  # 计算 Mersenne 数 M_p = 2^p - 1
    # 进行 p - 2 次迭代
    for _ in range(p - 2):
        v = pow(v, 2, m) - 2  # 使用 Lucas-Lehmer 公式更新 v
    return v == 0  # 返回是否 v 等于 0，判断 M_p 是否为素数


# 判断给定的整数 n 是否为 Mersenne 素数
def is_mersenne_prime(n):
    """Returns True if ``n`` is a Mersenne prime, else False.

    A Mersenne prime is a prime number having the form `2^i - 1`.

    Examples
    ========

    >>> from sympy.ntheory.factor_ import is_mersenne_prime
    >>> is_mersenne_prime(6)
    False
    >>> is_mersenne_prime(127)
    True

    References
    ==========

    .. [1] https://mathworld.wolfram.com/MersennePrime.html

    """
    n = as_int(n)  # 将 n 转换为整数
    if n < 1:
        return False  # 如果 n 小于 1，则不是 Mersenne 数
    if n & (n + 1):
        # 如果 n 不是 Mersenne 数，即 n 和 n+1 不同为 0 的位
        return False
    p = n.bit_length()  # 计算 n 的比特位长度
    if p in MERSENNE_PRIME_EXPONENTS:  # 如果 p 在已知的 Mersenne 素数指数集合中
        return True
    if p < 65_000_000 or not isprime(p):
        # 如果 p 小于 6500万或者 p 不是素数，则 n=2^p-1 不是素数
        # 根据 GIMPS 的数据，截至 2023 年 9 月 19 日已完成对小于 6500 万的 p 的验证
        return False
    result = _lucas_lehmer_primality_test(p)  # 使用 Lucas-Lehmer 测试判断 Mersenne 素数
    if result:
        raise ValueError(filldedent('''
            This Mersenne Prime, 2^%s - 1, should
            be added to SymPy's known values.''' % p))
    return result


# 判断给定的整数 n 是否为素数
def isprime(n):
    """
    Test if n is a prime number (True) or not (False). For n < 2^64 the
    answer is definitive; larger n values have a small probability of actually
    being pseudoprimes.

    Negative numbers (e.g. -2) are not considered prime.

    The first step is looking for trivial factors, which if found enables
    a quick return.  Next, if the sieve is large enough, use bisection search
    on the sieve.  For small numbers, a set of deterministic Miller-Rabin
    tests are performed with bases that are known to have no counterexamples
    in their range.  Finally if the number is larger than 2^64, a strong
    BPSW test is performed.  While this is a probable prime test and we
    believe counterexamples exist, there are no known counterexamples.

    Examples
    ========

    >>> from sympy.ntheory import isprime
    >>> isprime(13)
    True
    >>> isprime(15)
    False

    Notes
    =====

    This routine is intended only for integer input, not numerical
    expressions which may represent numbers. Floats are also
    """
    # 将输入的参数转换为整数
    n = as_int(n)
    
    # Step 1, do quick composite testing via trial division.  The individual
    # modulo tests benchmark faster than one or two primorial igcds for me.
    # The point here is just to speedily handle small numbers and many
    # composites.  Step 2 only requires that n <= 2 get handled here.
    # 第一步，通过试除法进行快速的合数测试。对于我来说，单独的取模测试比一个或两个素数阶乘的最大公约数测试更快。
    # 这里的目的是快速处理小数字和大量合数。第二步仅要求处理 n <= 2 的情况。
    if n in [2, 3, 5]:
        return True
    if n < 2 or (n % 2) == 0 or (n % 3) == 0 or (n % 5) == 0:
        return False
    if n < 49:
        return True
    if (n %  7) == 0 or (n % 11) == 0 or (n % 13) == 0 or (n % 17) == 0 or \
       (n % 19) == 0 or (n % 23) == 0 or (n % 29) == 0 or (n % 31) == 0 or \
       (n % 37) == 0 or (n % 41) == 0 or (n % 43) == 0 or (n % 47) == 0:
        return False
    if n < 2809:
        return True
    if n < 65077:
        # There are only five Euler pseudoprimes with a least prime factor greater than 47
        return pow(2, n >> 1, n) in [1, n - 1] and n not in [8321, 31621, 42799, 49141, 49981]
    
    # bisection search on the sieve if the sieve is large enough
    from sympy.ntheory.generate import sieve as s
    if n <= s._list[-1]:
        l, u = s.search(n)
        return l == u
    
    # If we have GMPY2, skip straight to step 3 and do a strong BPSW test.
    # This should be a bit faster than our step 2, and for large values will
    # be a lot faster than our step 3 (C+GMP vs. Python).
    if _gmpy is not None:
        return is_strong_bpsw_prp(n)
    
    
    # Step 2: deterministic Miller-Rabin testing for numbers < 2^64.  See:
    #    https://miller-rabin.appspot.com/
    # for lists.  We have made sure the M-R routine will successfully handle
    # bases larger than n, so we can use the minimal set.
    # 如果 n 小于 341531，则使用 Miller-Rabin 算法进行素数测试，使用固定的基数 [9345883071009581737]
    if n < 341531:
        return mr(n, [9345883071009581737])
    
    # 如果 n 小于 885594169，则使用 Miller-Rabin 算法进行素数测试，使用固定的基数 [725270293939359937, 3569819667048198375]
    if n < 885594169:
        return mr(n, [725270293939359937, 3569819667048198375])
    
    # 如果 n 小于 350269456337，则使用 Miller-Rabin 算法进行素数测试，使用固定的基数 [4230279247111683200, 14694767155120705706, 16641139526367750375]
    if n < 350269456337:
        return mr(n, [4230279247111683200, 14694767155120705706, 16641139526367750375])
    
    # 如果 n 小于 55245642489451，则使用 Miller-Rabin 算法进行素数测试，使用固定的基数 [2, 141889084524735, 1199124725622454117, 11096072698276303650]
    if n < 55245642489451:
        return mr(n, [2, 141889084524735, 1199124725622454117, 11096072698276303650])
    
    # 如果 n 小于 7999252175582851，则使用 Miller-Rabin 算法进行素数测试，使用固定的基数 [2, 4130806001517, 149795463772692060, 186635894390467037, 3967304179347715805]
    if n < 7999252175582851:
        return mr(n, [2, 4130806001517, 149795463772692060, 186635894390467037, 3967304179347715805])
    
    # 如果 n 小于 585226005592931977，则使用 Miller-Rabin 算法进行素数测试，使用固定的基数 [2, 123635709730000, 9233062284813009, 43835965440333360, 761179012939631437, 1263739024124850375]
    if n < 585226005592931977:
        return mr(n, [2, 123635709730000, 9233062284813009, 43835965440333360, 761179012939631437, 1263739024124850375])
    
    # 如果 n 小于 18446744073709551616，则使用 Miller-Rabin 算法进行素数测试，使用固定的基数 [2, 325, 9375, 28178, 450775, 9780504, 1795265022]
    if n < 18446744073709551616:
        return mr(n, [2, 325, 9375, 28178, 450775, 9780504, 1795265022])
    
    # 如果 n 小于 318665857834031151167461，则使用 Miller-Rabin 算法进行素数测试，使用固定的基数 [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    if n < 318665857834031151167461:
        return mr(n, [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37])
    
    # 如果 n 小于 3317044064679887385961981，则使用 Miller-Rabin 算法进行素数测试，使用固定的基数 [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
    if n < 3317044064679887385961981:
        return mr(n, [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41])
    
    # Step 3: BPSW.
    #
    #  Time for isprime(10**2000 + 4561), no gmpy or gmpy2 installed
    #     44.0s   old isprime using 46 bases
    #      5.3s   strong BPSW + one random base
    #      4.3s   extra strong BPSW + one random base
    #      4.1s   strong BPSW
    #      3.2s   extra strong BPSW

    # 使用经典的 BPSW 算法进行素数测试，详情见论文第1401页。也可以使用下面的替代想法。
    return is_strong_bpsw_prp(n)

    # 使用额外强的测试，这可能会更快一些
    #return mr(n, [2]) and is_extra_strong_lucas_prp(n)

    # 添加一个随机的 Miller-Rabin 基数
    #import random
    #return mr(n, [2, random.randint(3, n-1)]) and is_strong_lucas_prp(n)
def is_gaussian_prime(num):
    r"""Test if num is a Gaussian prime number.

    References
    ==========

    .. [1] https://oeis.org/wiki/Gaussian_primes
    """

    # 使用 sympify 将 num 转换为 SymPy 的表达式
    num = sympify(num)
    # 将 num 分解为实部和虚部 a, b
    a, b = num.as_real_imag()
    # 将 a 和 b 转换为整数（如果可能）
    a = as_int(a, strict=False)
    b = as_int(b, strict=False)

    # 如果实部 a 为 0
    if a == 0:
        # 取虚部的绝对值作为整数 b
        b = abs(b)
        # 检查 b 是否为素数，并且 b 对 4 取模为 3
        return isprime(b) and b % 4 == 3
    # 如果虚部 b 为 0
    elif b == 0:
        # 取实部的绝对值作为整数 a
        a = abs(a)
        # 检查 a 是否为素数，并且 a 对 4 取模为 3
        return isprime(a) and a % 4 == 3

    # 对于其他情况，检查 a^2 + b^2 是否为素数，判断是否为 Gaussian prime
    return isprime(a**2 + b**2)
```