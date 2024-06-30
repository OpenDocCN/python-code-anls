# `D:\src\scipysrc\sympy\sympy\ntheory\partitions_.py`

```
# 导入所需的函数和类
from mpmath.libmp import (fzero, from_int, from_rational,
    fone, fhalf, bitcount, to_int, mpf_mul, mpf_div, mpf_sub,
    mpf_add, mpf_sqrt, mpf_pi, mpf_cosh_sinh, mpf_cos, mpf_sin)
from sympy.external.gmpy import gcd, legendre, jacobi
from .residue_ntheory import _sqrt_mod_prime_power, is_quad_residue
from sympy.utilities.decorator import deprecated
from sympy.utilities.memoization import recurrence_memo

import math
from itertools import count

def _pre():
    # 设置最大数值
    maxn = 10**5
    # 声明全局变量 _factor 和 _totient
    global _factor
    global _totient
    # 初始化 _factor 和 _totient 列表
    _factor = [0]*maxn
    _totient = [1]*maxn
    # 计算平方根限制
    lim = int(maxn**0.5) + 5
    # 埃氏筛法求解素因子
    for i in range(2, lim):
        if _factor[i] == 0:
            for j in range(i*i, maxn, i):
                if _factor[j] == 0:
                    _factor[j] = i
    # 计算每个数的欧拉函数值
    for i in range(2, maxn):
        if _factor[i] == 0:
            _factor[i] = i
            _totient[i] = i-1
            continue
        x = _factor[i]
        y = i//x
        if y % x == 0:
            _totient[i] = _totient[y]*x
        else:
            _totient[i] = _totient[y]*(x - 1)

def _a(n, k, prec):
    """ Compute the inner sum in HRR formula [1]_

    Parameters
    ----------
    n : int
        Integer parameter for the calculation.
    k : int
        Integer parameter for the calculation.
    prec : int
        Precision parameter for the calculation.

    Returns
    -------
    mpf
        The computed result based on the formula.

    Notes
    -----
    This function computes the inner sum of the HRR formula [1]_,
    utilizing various mathematical functions and constants.

    References
    ----------
    .. [1] https://msp.org/pjm/1956/6-1/pjm-v6-n1-p18-p.pdf

    """
    # 如果 k 等于 1，直接返回 1.0
    if k == 1:
        return fone

    k1 = k
    e = 0
    p = _factor[k]
    # 求解 k 的最大素因子及其指数
    while k1 % p == 0:
        k1 //= p
        e += 1
    k2 = k//k1 # k2 = p^e
    # 计算 v = 1 - 24*n
    v = 1 - 24*n
    # 计算圆周率的值
    pi = mpf_pi(prec)
    # 如果 k1 等于 1，执行以下逻辑
    if k1 == 1:
        # 当 p 等于 2 时的特殊情况处理
        if p == 2:
            # 计算 mod = 8 * k
            mod = 8 * k
            # 计算 v + v % mod
            v = mod + v % mod
            # 计算 v * (9^(k-1) % mod) % mod 的平方根模 p^e + 3
            m = _sqrt_mod_prime_power(v, 2, e + 3)[0]
            # 计算参数 arg = 4*m*pi / mod
            arg = mpf_div(mpf_mul(
                from_int(4 * m), pi, prec), from_int(mod), prec)
            # 返回结果：(-1)^e * jacobi(m-1, m) * sqrt(k) * sin(arg)
            return mpf_mul(mpf_mul(
                from_int((-1)**e * jacobi(m - 1, m)),
                mpf_sqrt(from_int(k), prec), prec),
                mpf_sin(arg, prec), prec)
        
        # 当 p 等于 3 时的特殊情况处理
        if p == 3:
            # 计算 mod = 3 * k
            mod = 3 * k
            # 计算 v + v % mod
            v = mod + v % mod
            # 当 e 大于 1 时，计算 v * (64^(k//3 - 1) % mod) % mod
            if e > 1:
                v = (v * pow(64, k // 3 - 1, mod)) % mod
            # 计算 v 的平方根模 p^e + 1
            m = _sqrt_mod_prime_power(v, 3, e + 1)[0]
            # 计算参数 arg = 4*m*pi / mod
            arg = mpf_div(mpf_mul(from_int(4 * m), pi, prec),
                from_int(mod), prec)
            # 返回结果：2 * (-1)^(e + 1) * legendre(m, 3) * sqrt(k//3) * sin(arg)
            return mpf_mul(mpf_mul(
                from_int(2 * (-1)**(e + 1) * legendre(m, 3)),
                mpf_sqrt(from_int(k // 3), prec), prec),
                mpf_sin(arg, prec), prec)
        
        # 处理一般情况，v = k + v % k
        v = k + v % k
        # 如果 v 能被 p 整除，且 e 等于 1
        if v % p == 0:
            if e == 1:
                # 返回结果：jacobi(3, k) * sqrt(k)
                return mpf_mul(
                    from_int(jacobi(3, k)),
                    mpf_sqrt(from_int(k), prec), prec)
            # 返回 0
            return fzero
        # 如果 v 不是 p 的平方剩余
        if not is_quad_residue(v, p):
            # 返回 0
            return fzero
        # 计算 _phi = p^(e-1) * (p-1)
        _phi = p**(e - 1) * (p - 1)
        # 计算 v * (576^(_phi - 1) % k)
        v = (v * pow(576, _phi - 1, k))
        # 计算 v 的平方根模 p^e
        m = _sqrt_mod_prime_power(v, p, e)[0]
        # 计算参数 arg = 4*m*pi / k
        arg = mpf_div(
            mpf_mul(from_int(4 * m), pi, prec),
            from_int(k), prec)
        # 返回结果：2 * jacobi(3, k) * sqrt(k) * cos(arg)
        return mpf_mul(mpf_mul(
            from_int(2 * jacobi(3, k)),
            mpf_sqrt(from_int(k), prec), prec),
            mpf_cos(arg, prec), prec)

    # 如果 p 不等于 2 或 e 大于等于 3
    if p != 2 or e >= 3:
        # 计算 gcd(k1, 24) 和 gcd(k2, 24)
        d1, d2 = gcd(k1, 24), gcd(k2, 24)
        # 计算 e = 24 // (d1 * d2)
        e = 24 // (d1 * d2)
        # 计算 n1 = ((d2 * e * n + (k2^2 - 1) // d1) * (e * k2^2 * d2)^(φ(k1) - 1 % k1)) % k1
        n1 = ((d2 * e * n + (k2**2 - 1) // d1) *
            pow(e * k2**2 * d2, _totient[k1] - 1, k1)) % k1
        # 计算 n2 = ((d1 * e * n + (k1^2 - 1) // d2) * (e * k1^2 * d1)^(φ(k2) - 1 % k2)) % k2
        n2 = ((d1 * e * n + (k1**2 - 1) // d2) *
            pow(e * k1**2 * d1, _totient[k2] - 1, k2)) % k2
        # 返回结果：_a(n1, k1, prec) * _a(n2, k2, prec)
        return mpf_mul(_a(n1, k1, prec), _a(n2, k2, prec), prec)
    
    # 如果 e 等于 2
    if e == 2:
        # 计算 n1 = ((8 * n + 5) * (128^(φ(k1) - 1 % k1)) % k1
        n1 = ((8 * n + 5) * pow(128, _totient[k1] - 1, k1)) % k1
        # 计算 n2 = (4 + ((n - 2 - (k1^2 - 1) // 8) * k1^2) % 4) % 4
        n2 = (4 + ((n - 2 - (k1**2 - 1) // 8) * k1**2) % 4) % 4
        # 返回结果：-1 * _a(n1, k1, prec) * _a(n2, k2, prec)
        return mpf_mul(mpf_mul(
            from_int(-1),
            _a(n1, k1, prec), prec),
            _a(n2, k2, prec))
    
    # 计算 n1 = ((8 * n + 1) * (32^(φ(k1) - 1 % k1)) % k1
    n1 = ((8 * n + 1) * pow(32, _totient[k1] - 1, k1)) % k1
    # 计算 n2 = (2 + (n - (k1^2 - 1) // 8) % 2) % 2
    n2 = (2 + (n - (k1**2 - 1) // 8) % 2) % 2
    # 返回结果：_a(n1, k1, prec) * _a(n2, k2, prec)
    return mpf_mul(_a(n1, k1, prec), _a(n2, k2, prec), prec)
def _d(n, j, prec, sq23pi, sqrt8):
    """
    Compute the sinh term in the outer sum of the HRR formula.
    The constants sqrt(2/3*pi) and sqrt(8) must be precomputed.
    """
    j = from_int(j)  # Convert j to a multi-precision floating-point number
    pi = mpf_pi(prec)  # Compute pi with the given precision
    a = mpf_div(sq23pi, j, prec)  # Compute sq23pi / j with the given precision
    b = mpf_sub(from_int(n), from_rational(1, 24, prec), prec)  # Compute n - 1/24 with the given precision
    c = mpf_sqrt(b, prec)  # Compute the square root of b with the given precision
    ch, sh = mpf_cosh_sinh(mpf_mul(a, c), prec)  # Compute hyperbolic cosine and sine of a * c with the given precision
    D = mpf_div(
        mpf_sqrt(j, prec),
        mpf_mul(mpf_mul(sqrt8, b), pi), prec)  # Compute D using sqrt(j) / (sqrt(8) * b * pi) with the given precision
    E = mpf_sub(mpf_mul(a, ch), mpf_div(sh, c, prec), prec)  # Compute E as a * ch - sh / c with the given precision
    return mpf_mul(D, E)  # Return D * E with the given precision


@recurrence_memo([1, 1])
def _partition_rec(n: int, prev) -> int:
    """ Calculate the partition function P(n)

    Parameters
    ==========

    n : int
        nonnegative integer

    """
    v = 0  # Initialize the result accumulator
    penta = 0  # Initialize the pentagonal number: 1, 5, 12, ...
    for i in count():  # Loop indefinitely
        penta += 3*i + 1  # Increment the pentagonal number
        np = n - penta  # Calculate n - pentagonal number
        if np < 0:  # If np is negative, break the loop
            break
        s = prev[np]  # Get the value from the previous results for np
        np -= i + 1  # Update np to account for generalized pentagonal numbers
        if 0 <= np:  # If np is non-negative
            s += prev[np]  # Add the value from previous results for updated np
        v += -s if i % 2 else s  # Add or subtract s based on the parity of i
    return v  # Return the computed partition function value


def _partition(n: int) -> int:
    """ Calculate the partition function P(n)

    Parameters
    ==========

    n : int

    """
    if n < 0:
        return 0  # Return 0 if n is negative
    if (n <= 200_000 and n - _partition_rec.cache_length() < 70 or
            _partition_rec.cache_length() == 2 and n < 14_400):
        # Check conditions to decide whether to use precomputed results
        return _partition_rec(n)  # Return the result using precomputed recursion if conditions are met
    if '_factor' not in globals():
        _pre()  # Initialize global variables if not already done
    pbits = int((
        math.pi*(2*n/3.)**0.5 -
        math.log(4*n))/math.log(10) + 1) * \
        math.log2(10)  # Estimate number of bits in p(n)
    prec = p = int(pbits*1.1 + 100)  # Calculate precision based on estimated bits

    # Calculate the number of terms needed for accurate sum using Rademacher's bound
    c1 = 44*math.pi**2/(225*math.sqrt(3))
    c2 = math.pi*math.sqrt(2)/75
    c3 = math.pi*math.sqrt(2/3)
    def _M(n, N):
        sqrt = math.sqrt
        return c1/sqrt(N) + c2*sqrt(N/(n - 1))*math.sinh(c3*sqrt(n)/N)
    big = max(9, math.ceil(n**0.5))  # Calculate a value that should be sufficiently large for n > 65
    # 断言条件：如果 _M(n, big) 小于 0.5，则满足条件；否则将 big 值翻倍直至太大。
    assert _M(n, big) < 0.5  # else double big until too large

    # 当 big 大于 40 且 _M(n, big) 小于 0.5 时，将 big 逐步除以 2，直至不再满足条件。
    while big > 40 and _M(n, big) < 0.5:
        big //= 2

    # 将 small 设为当前的 big 值，然后将 big 设置为 small 的两倍。
    small = big
    big = small * 2

    # 当 big 和 small 之差大于 1 时，执行二分查找。
    while big - small > 1:
        # 计算中间值 N，并根据 _M(n, N) 的结果调整 big 或 small。
        N = (big + small) // 2
        if (er := _M(n, N)) < 0.5:
            big = N
        elif er >= 0.5:
            small = N

    # 将最终确定的 big 值赋给变量 M，表示函数 M 的计算完成，现在有了最终的值。
    M = big  # done with function M; now have value

    # 对答案的预期大小进行健全性检查。
    if M > 10**5:  # i.e. M > maxn
        raise ValueError("Input too big")  # i.e. n > 149832547102

    # 计算结果 s 的初始值为 fzero。
    s = fzero

    # 计算 sq23pi 和 sqrt8 的乘积，这些是计算中使用的常数。
    sq23pi = mpf_mul(mpf_sqrt(from_rational(2, 3, p), p), mpf_pi(p), p)
    sqrt8 = mpf_sqrt(from_int(8), p)

    # 迭代计算和 s，循环次数为 M-1。
    for q in range(1, M):
        # 计算 _a(n, q, p) 和 _d(n, q, p, sq23pi, sqrt8) 的乘积，并加到 s 上。
        a = _a(n, q, p)
        d = _d(n, q, p, sq23pi, sqrt8)
        s = mpf_add(s, mpf_mul(a, d), prec)

        # 平均来说，这些项的大小迅速减少。
        # 动态减少精度极大地提升了性能。
        p = bitcount(abs(to_int(d))) + 50

    # 返回 s 加上 fhalf 并转换为整数的结果。
    return int(to_int(mpf_add(s, fhalf, prec)))
# 使用 @deprecated 装饰器标记函数为废弃状态，并提供相关的说明文档和版本信息
@deprecated("""\
The `sympy.ntheory.partitions_.npartitions` has been moved to `sympy.functions.combinatorial.numbers.partition`.""",
            deprecated_since_version="1.13",
            active_deprecations_target='deprecated-ntheory-symbolic-functions')
# 定义函数 npartitions，计算将 n 分解为正整数之和的方式数 P(n)
def npartitions(n, verbose=False):
    """
    Calculate the partition function P(n), i.e. the number of ways that
    n can be written as a sum of positive integers.

    .. deprecated:: 1.13

        The ``npartitions`` function is deprecated. Use :class:`sympy.functions.combinatorial.numbers.partition`
        instead. See its documentation for more information. See
        :ref:`deprecated-ntheory-symbolic-functions` for details.

    P(n) is computed using the Hardy-Ramanujan-Rademacher formula [1]_.


    The correctness of this implementation has been tested through $10^{10}$.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import partition
    >>> partition(25)
    1958

    References
    ==========

    .. [1] https://mathworld.wolfram.com/PartitionFunctionP.html

    """
    # 导入 sympy 库中的 partition 函数，并使用其计算 P(n)
    from sympy.functions.combinatorial.numbers import partition as func_partition
    # 返回计算得到的 P(n) 值
    return func_partition(n)
```