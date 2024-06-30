# `D:\src\scipysrc\sympy\sympy\ntheory\residue_ntheory.py`

```
# 从未来导入注释，使得在Python 2中使用Python 3的一些特性
from __future__ import annotations

# 从sympy.external.gmpy模块中导入以下函数：gcd, lcm, invert, sqrt, jacobi,
# bit_scan1, remove
from sympy.external.gmpy import (gcd, lcm, invert, sqrt, jacobi,
                                 bit_scan1, remove)

# 从sympy.polys模块中导入Poly类
from sympy.polys import Poly

# 从sympy.polys.domains模块中导入ZZ对象
from sympy.polys.domains import ZZ

# 从sympy.polys.galoistools模块中导入gf_crt1, gf_crt2, linear_congruence, gf_csolve函数
from sympy.polys.galoistools import gf_crt1, gf_crt2, linear_congruence, gf_csolve

# 从当前目录的primetest模块中导入isprime函数
from .primetest import isprime

# 从当前目录的generate模块中导入primerange函数
from .generate import primerange

# 从当前目录的factor_模块中导入factorint, _perfect_power函数
from .factor_ import factorint, _perfect_power

# 从当前目录的modular模块中导入crt函数
from .modular import crt

# 从sympy.utilities.decorator模块中导入deprecated装饰器
from sympy.utilities.decorator import deprecated

# 从sympy.utilities.memoization模块中导入recurrence_memo函数
from sympy.utilities.memoization import recurrence_memo

# 从sympy.utilities.misc模块中导入as_int函数
from sympy.utilities.misc import as_int

# 从sympy.utilities.iterables模块中导入iproduct函数
from sympy.utilities.iterables import iproduct

# 从sympy.core.random模块中导入_randint, randint函数
from sympy.core.random import _randint, randint

# 从itertools模块中导入product函数
from itertools import product


def n_order(a, n):
    r""" Returns the order of ``a`` modulo ``n``.

    Explanation
    ===========

    The order of ``a`` modulo ``n`` is the smallest integer
    ``k`` such that `a^k` leaves a remainder of 1 with ``n``.

    Parameters
    ==========

    a : integer
    n : integer, n > 1. a and n should be relatively prime

    Returns
    =======

    int : the order of ``a`` modulo ``n``

    Raises
    ======

    ValueError
        If `n \le 1` or `\gcd(a, n) \neq 1`.
        If ``a`` or ``n`` is not an integer.

    Examples
    ========

    >>> from sympy.ntheory import n_order
    >>> n_order(3, 7)
    6
    >>> n_order(4, 7)
    3

    See Also
    ========

    is_primitive_root
        We say that ``a`` is a primitive root of ``n``
        when the order of ``a`` modulo ``n`` equals ``totient(n)``

    """
    a, n = as_int(a), as_int(n)
    if n <= 1:
        raise ValueError("n should be an integer greater than 1")
    a = a % n
    # Trivial
    if a == 1:
        return 1
    if gcd(a, n) != 1:
        raise ValueError("The two numbers should be relatively prime")
    a_order = 1
    for p, e in factorint(n).items():
        pe = p**e
        pe_order = (p - 1) * p**(e - 1)
        factors = factorint(p - 1)
        if e > 1:
            factors[p] = e - 1
        order = 1
        for px, ex in factors.items():
            x = pow(a, pe_order // px**ex, pe)
            while x != 1:
                x = pow(x, px, pe)
                order *= px
        a_order = lcm(a_order, order)
    return int(a_order)


def _primitive_root_prime_iter(p):
    r""" Generates the primitive roots for a prime ``p``.

    Explanation
    ===========

    The primitive roots generated are not necessarily sorted.
    However, the first one is the smallest primitive root.

    Find the element whose order is ``p-1`` from the smaller one.
    If we can find the first primitive root ``g``, we can use the following theorem.

    .. math ::
        \operatorname{ord}(g^k) = \frac{\operatorname{ord}(g)}{\gcd(\operatorname{ord}(g), k)}

    From the assumption that `\operatorname{ord}(g)=p-1`,
    it is a necessary and sufficient condition for
    `\operatorname{ord}(g^k)=p-1` that `\gcd(p-1, k)=1`.

    Parameters
    ==========

    p : odd prime

    """
    # 此函数未完整，缺少实际实现部分
    pass
    Yields
    ======

    int
        返回``p``的原根

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import _primitive_root_prime_iter
    >>> sorted(_primitive_root_prime_iter(19))
    [2, 3, 10, 13, 14, 15]

    References
    ==========

    .. [1] W. Stein "Elementary Number Theory" (2011), page 44

    """
    # 如果 p 等于 3，则直接产生 2 作为原根，然后返回
    if p == 3:
        yield 2
        return
    # 设定最小的可能原根 g_min，如果 p 除以 8 的余数为 1 或 7，则 g_min 为 3，否则为 2
    g_min = 3 if p % 8 in [1, 7] else 2
    # 如果 p 小于 41，则特殊处理
    if p < 41:
        g = 5 if p == 23 else g_min
    else:
        # 计算 p-1 的所有因子
        v = [(p - 1) // i for i in factorint(p - 1).keys()]
        # 从 g_min 开始循环，找到 p 的一个原根 g
        for g in range(g_min, p):
            # 如果 g 的所有 v 中的幂次方模 p 都不等于 1，则 g 是 p 的原根
            if all(pow(g, pw, p) != 1 for pw in v):
                break
    # 产生找到的原根 g
    yield g
    # 对于从 3 到 p 之间的奇数 k，如果 gcd(p-1, k) = 1，则 g**k 是 p 的原根
    for k in range(3, p, 2):
        if gcd(p - 1, k) == 1:
            yield pow(g, k, p)
def _primitive_root_prime_power_iter(p, e):
    r""" Generates the primitive roots of `p^e`.

    Explanation
    ===========

    Let ``g`` be the primitive root of ``p``.
    If `g^{p-1} \not\equiv 1 \pmod{p^2}`, then ``g`` is primitive root of `p^e`.
    Thus, if we find a primitive root ``g`` of ``p``,
    then `g, g+p, g+2p, \ldots, g+(p-1)p` are primitive roots of `p^2` except one.
    That one satisfies `\hat{g}^{p-1} \equiv 1 \pmod{p^2}`.
    If ``h`` is the primitive root of `p^2`,
    then `h, h+p^2, h+2p^2, \ldots, h+(p^{e-2}-1)p^e` are primitive roots of `p^e`.

    Parameters
    ==========

    p : odd prime
        The base prime number for which primitive roots are generated.
    e : positive integer
        The exponent to which the prime `p` is raised.

    Yields
    ======

    int
        Yields integers which are primitive roots of `p^e`.

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import _primitive_root_prime_power_iter
    >>> sorted(_primitive_root_prime_power_iter(5, 2))
    [2, 3, 8, 12, 13, 17, 22, 23]

    """
    if e == 1:
        # If e is 1, yield primitive roots of p using _primitive_root_prime_iter
        yield from _primitive_root_prime_iter(p)
    else:
        p2 = p**2
        # Iterate over primitive roots of p
        for g in _primitive_root_prime_iter(p):
            # Calculate t as per the condition to find primitive roots of p^e
            t = (g - pow(g, 2 - p, p2)) % p2
            # Iterate over potential primitive roots of p^e
            for k in range(0, p2, p):
                if k != t:
                    # Yield each primitive root of p^e
                    yield from (g + k + m for m in range(0, p**e, p2))


def _primitive_root_prime_power2_iter(p, e):
    r""" Generates the primitive roots of `2p^e`.

    Explanation
    ===========

    If ``g`` is the primitive root of ``p**e``,
    then the odd one of ``g`` and ``g+p**e`` is the primitive root of ``2*p**e``.

    Parameters
    ==========

    p : odd prime
        The base prime number multiplied by 2 for which primitive roots are generated.
    e : positive integer
        The exponent to which the prime `p` is raised.

    Yields
    ======

    int
        Yields integers which are primitive roots of `2p^e`.

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import _primitive_root_prime_power2_iter
    >>> sorted(_primitive_root_prime_power2_iter(5, 2))
    [3, 13, 17, 23, 27, 33, 37, 47]

    """
    for g in _primitive_root_prime_power_iter(p, e):
        # Check if g is odd
        if g % 2 == 1:
            yield g
        else:
            # Yield the odd primitive root of 2*p^e
            yield g + p**e


def primitive_root(p, smallest=True):
    r""" Returns a primitive root of ``p`` or None.

    Explanation
    ===========

    For the definition of primitive root,
    see the explanation of ``is_primitive_root``.

    The primitive root of ``p`` exist only for
    `p = 2, 4, q^e, 2q^e` (``q`` is an odd prime).
    Now, if we know the primitive root of ``q``,
    we can calculate the primitive root of `q^e`,
    and if we know the primitive root of `q^e`,
    we can calculate the primitive root of `2q^e`.
    When there is no need to find the smallest primitive root,
    this property can be used to obtain a fast primitive root.
    On the other hand, when we want the smallest primitive root,
    we naively determine whether it is a primitive root or not.

    Parameters
    ==========

    p : integer, p > 1
        The number for which the primitive root is to be found.
    smallest : bool, optional
        If True, returns the smallest primitive root.

    Returns
    =======

    int or None
        Returns the smallest primitive root of p if smallest=True; otherwise, returns any primitive root of p, or None if no primitive root exists.

    """
    # 如果存在原根，返回整数 p 的原根；否则返回 None。
    p = as_int(p)  # 将 p 转换为整数
    if p <= 1:
        raise ValueError("p should be an integer greater than 1")  # 如果 p 小于等于 1，抛出数值错误异常
    if p <= 4:
        return p - 1  # 如果 p 小于等于 4，则返回 p - 1

    p_even = p % 2 == 0  # 检查 p 是否为偶数
    if not p_even:
        q = p  # 如果 p 是奇数，则 q 等于 p
    elif p % 4:
        q = p//2  # 如果 p 有一个因子为 2，则 q 等于 p 的一半
    else:
        return None  # 如果 p 有多于一个因子为 2，则返回 None

    if isprime(q):
        e = 1  # 如果 q 是素数，设置 e 为 1
    else:
        m = _perfect_power(q, 3)  # 检查 q 是否是完美立方数
        if not m:
            return None  # 如果不是完美立方数，返回 None
        q, e = m  # 获取完美立方数的底数和指数
        if not isprime(q):
            return None  # 如果底数不是素数，返回 None

    if not smallest:
        if p_even:
            return next(_primitive_root_prime_power2_iter(q, e))  # 返回 q 的 2 次幂的素数次幂的原根迭代器的下一个值
        return next(_primitive_root_prime_power_iter(q, e))  # 返回 q 的素数次幂的原根迭代器的下一个值

    if p_even:
        for i in range(3, p, 2):
            if i % q and is_primitive_root(i, p):
                return i  # 在给定的范围内查找原根并返回找到的第一个值

    g = next(_primitive_root_prime_iter(q))  # 返回 q 的原根迭代器的下一个值
    if e == 1 or pow(g, q - 1, q**2) != 1:
        return g  # 如果 e 等于 1 或 g 的 q-1 次幂除以 q^2 不等于 1，返回 g

    for i in range(g + 1, p):
        if i % q and is_primitive_root(i, p):
            return i  # 在给定的范围内查找原根并返回找到的第一个值
def is_primitive_root(a, p):
    r""" Returns True if ``a`` is a primitive root of ``p``.

    Explanation
    ===========

    ``a`` is said to be the primitive root of ``p`` if `\gcd(a, p) = 1` and
    `\phi(p)` is the smallest positive number s.t.

        `a^{\phi(p)} \equiv 1 \pmod{p}`.

    where `\phi(p)` is Euler's totient function.

    The primitive root of ``p`` exists only for
    `p = 2, 4, q^e, 2q^e` (``q`` is an odd prime).
    Hence, if it is not such a ``p``, it returns False.
    To determine the primitive root, we need to know
    the prime factorization of ``q-1``.
    The hardness of the determination depends on this complexity.

    Parameters
    ==========

    a : integer
    p : integer, ``p`` > 1. ``a`` and ``p`` should be relatively prime

    Returns
    =======

    bool : If True, ``a`` is the primitive root of ``p``.

    Raises
    ======

    ValueError
        If `p \le 1` or `\gcd(a, p) \neq 1`.
        If ``a`` or ``p`` is not an integer.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import totient
    >>> from sympy.ntheory import is_primitive_root, n_order
    >>> is_primitive_root(3, 10)
    True
    >>> is_primitive_root(9, 10)
    False
    >>> n_order(3, 10) == totient(10)
    True
    >>> n_order(9, 10) == totient(10)
    False

    See Also
    ========

    primitive_root

    """
    # 将输入的a和p转换为整数
    a, p = as_int(a), as_int(p)
    # 如果p小于等于1，抛出值错误异常
    if p <= 1:
        raise ValueError("p should be an integer greater than 1")
    # 对a取模p，确保a在0到p-1之间
    a = a % p
    # 如果a和p的最大公约数不等于1，抛出值错误异常
    if gcd(a, p) != 1:
        raise ValueError("The two numbers should be relatively prime")
    # p = 2或4时，原根只有p-1
    if p <= 4:
        # 原根是p-1
        return a == p - 1
    # 如果p是奇数，则q为p
    if p % 2:
        q = p  # p是奇数
    elif p % 4:
        q = p//2  # p有一个因子2
    else:
        return False  # p有多于一个因子2
    # 如果q是质数
    if isprime(q):
        # 群的阶数是q-1
        group_order = q - 1
        # 获取q-1的质因数
        factors = factorint(q - 1).keys()
    else:
        # 检查是否是完美立方
        m = _perfect_power(q, 3)
        if not m:
            return False
        q, e = m
        # 如果q不是质数，返回False
        if not isprime(q):
            return False
        # 群的阶数是q**(e-1)*(q-1)
        group_order = q**(e - 1)*(q - 1)
        # 获取q-1的质因数集合
        factors = set(factorint(q - 1).keys())
        factors.add(q)
    # 对于群阶数的每个质数，验证是否都不等于1
    return all(pow(a, group_order // prime, p) != 1 for prime in factors)
    """
    s = bit_scan1(p - 1)
    t = p >> s
    # 根据 Carl Pomerance 和 Richard Crandall 的书籍 [1]，第二版第101页，找到素数的非平方剩余
    # 根据 p % 12 的余数来选择 d 的值
    if p % 12 == 5:
        # 如果 p 除以 12 的余数是 5，则选择 d = 3
        d = 3
    elif p % 5 in [2, 3]:
        # 如果 p 除以 5 的余数是 2 或 3，则选择 d = 5
        d = 5
    else:
        # 否则随机选择一个 6 到 p-1 之间的数 d，确保 jacobi 符号 jacobi(d, p) 等于 -1
        while 1:
            d = randint(6, p - 1)
            if jacobi(d, p) == -1:
                break
    # 计算 A 和 D 的值
    A = pow(a, t, p)
    D = pow(d, t, p)
    m = 0
    # 计算 m 的值，满足一定条件
    for i in range(s):
        # 计算 adm 的值
        adm = A*pow(D, m, p) % p
        # 计算 adm 的平方
        adm = pow(adm, 2**(s - 1 - i), p)
        if adm % p == p - 1:
            m += 2**i
    # 计算 x 的值并返回
    x = pow(a, (t + 1)//2, p)*pow(D, m//2, p) % p
    return x
    ```
def sqrt_mod(a, p, all_roots=False):
    """
    Find a root of ``x**2 = a mod p``.

    Parameters
    ==========

    a : integer
        The integer `a` in the equation `x**2 = a mod p`.
    p : positive integer
        The modulus `p` for the equation.
    all_roots : bool, optional
        If True, returns a list of all roots; otherwise, returns a single root.

    Returns
    =======

    integer or list
        Depending on `all_roots`:
        - If `all_roots` is False, returns a single root `r`.
        - If `all_roots` is True, returns a sorted list of all roots.

    Notes
    =====

    - If no root exists, returns None.
    - The returned root is guaranteed to be less than or equal to `p // 2`.
    - If `p // 2` is the only root, it is returned.
    - Use `all_roots` cautiously as it may consume significant memory.

    Examples
    ========

    >>> from sympy.ntheory import sqrt_mod
    >>> sqrt_mod(11, 43)
    21
    >>> sqrt_mod(17, 32, True)
    [7, 9, 23, 25]
    """
    if all_roots:
        # Return a sorted list of all roots
        return sorted(sqrt_mod_iter(a, p))
    
    # Ensure p is positive
    p = abs(as_int(p))
    halfp = p // 2
    x = None
    for r in sqrt_mod_iter(a, p):
        if r < halfp:
            return r
        elif r > halfp:
            return p - r
        else:
            x = r
    return x


def sqrt_mod_iter(a, p, domain=int):
    """
    Iterate over solutions to ``x**2 = a mod p``.

    Parameters
    ==========

    a : integer
        The integer `a` in the equation `x**2 = a mod p`.
    p : positive integer
        The modulus `p` for the equation.
    domain : type, optional
        The domain type for returned solutions (`int`, `ZZ`, or `Integer`).

    Yields
    ======

    integer
        Solutions to the equation `x**2 = a mod p`.

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import sqrt_mod_iter
    >>> list(sqrt_mod_iter(11, 43))
    [21, 22]

    See Also
    ========

    sqrt_mod : Provides a sorted list or single solution for `x**2 = a mod p`.
    """
    # Ensure a and p are integers and p is positive
    a, p = as_int(a), abs(as_int(p))
    v = []
    pv = []
    _product = product
    
    # Factorize p and iterate over factors
    for px, ex in factorint(p).items():
        if a % px:
            # Find solutions for each prime power factor
            rx = _sqrt_mod_prime_power(a, px, ex)
        else:
            # Handle case where a is divisible by px
            rx = _sqrt_mod1(a, px, ex)
            _product = iproduct
        
        # If no solutions found, return None
        if not rx:
            return
        
        # Store solutions and respective powers
        v.append(rx)
        pv.append(px**ex)
    
    # If only one set of solutions, yield each as domain type
    if len(v) == 1:
        yield from map(domain, v[0])
    else:
        # Use Chinese Remainder Theorem to combine solutions
        mm, e, s = gf_crt1(pv, ZZ)
        for vx in _product(*v):
            yield domain(gf_crt2(vx, pv, mm, e, s, ZZ))


def _sqrt_mod_prime_power(a, p, k):
    """
    Find the solutions to ``x**2 = a mod p**k`` when ``a % p != 0``.

    Parameters
    ==========

    a : integer
        The integer `a` in the equation `x**2 = a mod p**k`.
    p : prime number
        The prime number `p` for the equation.
    k : positive integer
        The exponent `k` of `p` in the modulus `p**k`.

    Returns
    =======

    list
        List of solutions to the equation `x**2 = a mod p**k`.

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import _sqrt_mod_prime_power
    >>> _sqrt_mod_prime_power(11, 43, 1)
    [21, 22]

    References
    ==========

    .. [1] P. Hackman "Elementary Number Theory" (2009), page 160
    .. [2] http://www.numbertheory.org/php/squareroot.html
    .. [3] [Gathen99]_
    """
    pk = p**k
    a = a % pk
    
    # Implementation details omitted for brevity in this comment
    if p == 2:
        # 如果 p 等于 2，则执行以下操作
        if a % 8 != 1:
            # 如果 a 对 8 取余不等于 1，则返回 None
            return None
        # 如果 k 小于等于 3，则返回从 1 到 pk-1 的奇数列表
        if k <= 3:
            return list(range(1, pk, 2))
        r = 1
        # r 是方程 x**2 - a ≡ 0 (mod 2**3) 的一个解
        # 使用 Hensel 提升这些解到 x**2 - a ≡ 0 (mod 2**k)
        # 如果 r**2 - a ≡ 0 (mod 2**nx) 但不是 (mod 2**(nx+1))
        # 则 r + 2**(nx - 1) 是模 2**(nx+1) 的一个根
        for nx in range(3, k):
            if ((r**2 - a) >> nx) % 2:
                r += 1 << (nx - 1)
        # r 是方程 x**2 - a ≡ 0 (mod 2**k) 的一个解，并且存在其他解 -r, r+h, -(r+h)，这些都是解
        h = 1 << (k - 1)
        return sorted([r, pk - r, (r + h) % pk, -(r + h) % pk])

    # 如果 Legendre 符号 (a/p) 不等于 1，则不存在解
    if jacobi(a, p) != 1:
        return None
    if p % 4 == 3:
        # 如果 p mod 4 等于 3，则计算 a 的 (p+1)//4 次方模 p 的结果
        res = pow(a, (p + 1) // 4, p)
    elif p % 8 == 5:
        # 如果 p mod 8 等于 5，则计算 a 的 (p+3)//8 次方模 p 的结果
        res = pow(a, (p + 3) // 8, p)
        # 如果 res 的平方模 p 不等于 a mod p，则进行修正
        if pow(res, 2, p) != a % p:
            res = res * pow(2, (p - 1) // 4, p) % p
    else:
        # 否则使用 Tonelli-Shanks 方法计算模 p 的平方根
        res = _sqrt_mod_tonelli_shanks(a, p)
    if k > 1:
        # 使用 Hensel 提升和牛顿迭代，参见 Ref.[3] 第 9 章
        # 其中 f(x) = x**2 - a；对于 p != 2，有 f'(a) ≠ 0 (mod p)
        px = p
        for _ in range(k.bit_length() - 1):
            px = px**2
            frinv = invert(2*res, px)
            res = (res - (res**2 - a)*frinv) % px
        if k & (k - 1): # 如果 k 不是 2 的幂
            frinv = invert(2*res, pk)
            res = (res - (res**2 - a)*frinv) % pk
    return sorted([res, pk - res])
def _sqrt_mod1(a, p, n):
    """
    Find solution to ``x**2 == a mod p**n`` when ``a % p == 0``.
    If no solution exists, return ``None``.

    Parameters
    ==========

    a : integer
        An integer such that ``a % p == 0``.
    p : prime number, p must divide a
        A prime number that divides `a`.
    n : positive integer
        A positive integer specifying the power of `p`.

    References
    ==========

    .. [1] http://www.numbertheory.org/php/squareroot.html
    """
    pn = p**n  # 计算 p 的 n 次幂
    a = a % pn  # 计算 a 对 p**n 取模
    if a == 0:
        # 当 gcd(a, p**k) = p**n 时的情况
        return range(0, pn, p**((n + 1) // 2))
    # 当 gcd(a, p**k) = p**r, r < n 时的情况
    a, r = remove(a, p)
    if r % 2 == 1:
        return None
    res = _sqrt_mod_prime_power(a, p, n - r)  # 调用内部函数求解特定情况下的平方根模 p**n
    if res is None:
        return None
    m = r // 2
    return (x for rx in res for x in range(rx*p**m, pn, p**(n - m)))


def is_quad_residue(a, p):
    """
    Returns True if ``a`` (mod ``p``) is in the set of squares mod ``p``,
    i.e a % p in set([i**2 % p for i in range(p)]).

    Parameters
    ==========

    a : integer
        An integer for which we check quadratic residue.
    p : positive integer
        A prime modulus.

    Returns
    =======

    bool
        True if ``a`` is a quadratic residue modulo ``p``, False otherwise.

    Raises
    ======

    ValueError
        If ``a``, ``p`` is not integer.
        If ``p`` is not positive.

    Examples
    ========

    >>> from sympy.ntheory import is_quad_residue
    >>> is_quad_residue(21, 100)
    True

    Indeed, ``pow(39, 2, 100)`` would be 21.

    >>> is_quad_residue(21, 120)
    False

    That is, for any integer ``x``, ``pow(x, 2, 120)`` is not 21.

    If ``p`` is an odd
    prime, an iterative method is used to make the determination:

    >>> from sympy.ntheory import is_quad_residue
    >>> sorted(set([i**2 % 7 for i in range(7)]))
    [0, 1, 2, 4]
    >>> [j for j in range(7) if is_quad_residue(j, 7)]
    [0, 1, 2, 4]

    See Also
    ========

    legendre_symbol, jacobi_symbol, sqrt_mod
    """
    a, p = as_int(a), as_int(p)
    if p < 1:
        raise ValueError('p must be > 0')
    a %= p
    if a < 2 or p < 3:
        return True
    t = bit_scan1(p)
    if t:
        a_ = a % (1 << t)
        if a_:
            r = bit_scan1(a_)
            if r % 2 or (a_ >> r) & 6:
                return False
        p >>= t
        a %= p
        if a < 2 or p < 3:
            return True
    j = jacobi(a, p)
    if j == -1 or isprime(p):
        return j == 1
    for px, ex in factorint(p).items():
        if a % px:
            if jacobi(a, px) != 1:
                return False
        else:
            a_ = a % px**ex
            if a_ == 0:
                continue
            a_, r = remove(a_, px)
            if r % 2 or jacobi(a_, px) != 1:
                return False
    # 返回布尔值 True，表示函数执行成功或满足条件
    return True
def is_nthpow_residue(a, n, m):
    """
    Returns True if ``x**n == a (mod m)`` has solutions.

    References
    ==========

    .. [1] P. Hackman "Elementary Number Theory" (2009), page 76

    """
    # Reduce `a` modulo `m` to ensure it falls within valid range
    a = a % m
    # Convert `a`, `n`, `m` to integers if they are not already
    a, n, m = as_int(a), as_int(n), as_int(m)
    # Check validity of parameters `m` and `n`
    if m <= 0:
        raise ValueError('m must be > 0')
    if n < 0:
        raise ValueError('n must be >= 0')
    # Special cases where solution is trivial
    if n == 0:
        if m == 1:
            return False
        return a == 1
    if a == 0:
        return True
    if n == 1:
        return True
    if n == 2:
        # Check if `a` is a quadratic residue modulo `m`
        return is_quad_residue(a, m)
    # For `n > 2`, check residue conditions for each prime factor of `m`
    return all(_is_nthpow_residue_bign_prime_power(a, n, p, e)
               for p, e in factorint(m).items())


def _is_nthpow_residue_bign_prime_power(a, n, p, k):
    r"""
    Returns True if `x^n = a \pmod{p^k}` has solutions for `n > 2`.

    Parameters
    ==========

    a : positive integer
    n : integer, n > 2
    p : prime number
    k : positive integer

    """
    # Reduce `a` modulo `p^k` repeatedly while possible
    while a % p == 0:
        a %= pow(p, k)
        if not a:
            return True
        # Use remove function to extract multiples of `p` from `a`
        a, mu = remove(a, p)
        # Check if conditions for residue are met
        if mu % n:
            return False
        k -= mu
    # For `p != 2`, use properties of totient function and gcd to check residue
    if p != 2:
        f = p**(k - 1)*(p - 1)  # `f` is totient function of `p^k`
        return pow(a, f // gcd(f, n), pow(p, k)) == 1
    # Special case for `p = 2` and `n` odd, directly return True
    if n & 1:
        return True
    # Determine exponent `c` and check residue condition
    c = min(bit_scan1(n) + 2, k)
    return a % pow(2, c) == 1


def _nthroot_mod1(s, q, p, all_roots):
    """
    Root of ``x**q = s mod p``, ``p`` prime and ``q`` divides ``p - 1``.
    Assume that the root exists.

    Parameters
    ==========

    s : integer
    q : integer, n > 2. ``q`` divides ``p - 1``.
    p : prime number
    all_roots : if False returns the smallest root, else the list of roots

    Returns
    =======

    list[int] | int :
        Root of ``x**q = s mod p``. If ``all_roots == True``,
        returned ascending list. otherwise, returned an int.

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import _nthroot_mod1
    >>> _nthroot_mod1(5, 3, 13, False)
    7
    >>> _nthroot_mod1(13, 4, 17, True)
    [3, 5, 12, 14]

    References
    ==========

    .. [1] A. M. Johnston, A Generalized qth Root Algorithm,
           ACM-SIAM Symposium on Discrete Algorithms (1999), pp. 929-930

    """
    # Obtain a primitive root `g` modulo `p`
    g = next(_primitive_root_prime_iter(p))
    r = s
    # Factor `q` and process each prime factor
    for qx, ex in factorint(q).items():
        f = (p - 1) // qx**ex
        while f % qx == 0:
            f //= qx
        z = f*invert(-f, qx)
        x = (1 + z) // qx
        t = discrete_log(p, pow(r, f, p), pow(g, f*qx, p))
        # Iterate and adjust `r` based on calculations
        for _ in range(ex):
            # assert t == discrete_log(p, pow(r, f, p), pow(g, f*qx, p))
            r = pow(r, x, p)*pow(g, -z*t % (p - 1), p) % p
            t //= qx
    res = [r]
    h = pow(g, (p - 1) // q, p)
    # assert pow(h, q, p) == 1
    hx = r
    # Generate all roots if `all_roots` is True, otherwise return the smallest one
    for _ in range(q - 1):
        hx = (hx*h) % p
        res.append(hx)
    if all_roots:
        res.sort()
        return res
    return min(res)


def _nthroot_mod_prime_power(a, n, p, k):
    """
    Computes the nth root of `a` modulo `p^k` where `p` is prime.

    Parameters
    ==========

    a : integer
    n : integer, n > 2
    p : prime number
    k : positive integer

    Returns
    =======

    integer :
        The nth root of `a` modulo `p^k`.

    Examples
    ========

    >>> _nthroot_mod_prime_power(5, 3, 13, 2)
    5
    >>> _nthroot_mod_prime_power(13, 4, 17, 3)
    13

    """
    # Placeholder for future implementation
    pass
    # 定义函数，计算“x**n = a mod p**k”的根
    
    Parameters
    ==========
    a : 整数
    n : 整数，n > 2
    p : 质数
    k : 正整数
    
    Returns
    =======
    list[int] :
        升序列表，包含“x**n = a mod p**k”的所有根。
        如果没有解，返回空列表“[]”。
    
    """
    if not _is_nthpow_residue_bign_prime_power(a, n, p, k):
        # 如果“a”不是“n”次幂的p**k模剩余，则返回空列表
        return []
    
    # 计算a对p取模
    a_mod_p = a % p
    
    # 如果a对p取模等于0，根为0
    if a_mod_p == 0:
        base_roots = [0]
    # 如果(p - 1) % n == 0，使用特定函数计算根
    elif (p - 1) % n == 0:
        base_roots = _nthroot_mod1(a_mod_p, n, p, all_roots=True)
    else:
        # 计算“x**n - a = 0 (mod p)”的根是“gcd(x**n - a, x**(p - 1) - 1) = 0 (mod p)”的根
        pa = n
        pb = p - 1
        b = 1
    
        # 确保pa >= pb，否则交换变量
        if pa < pb:
            a_mod_p, pa, b, pb = b, pb, a_mod_p, pa
    
        # 使用扩展欧几里得算法计算gcd，得到根
        while pb:
            q, pc = divmod(pa, pb)
            c = pow(b, -q, p) * a_mod_p % p
            pa, pb = pb, pc
            a_mod_p, b = b, c
    
        # 根据pa的值选择不同的计算方法得到根
        if pa == 1:
            base_roots = [a_mod_p]
        elif pa == 2:
            base_roots = sqrt_mod(a_mod_p, p, all_roots=True)
        else:
            base_roots = _nthroot_mod1(a_mod_p, pa, p, all_roots=True)
    
    # 如果k == 1，直接返回基本根列表
    if k == 1:
        return base_roots
    
    # 否则，处理p**k的情况
    a %= p**k
    tot_roots = set()
    
    # 对基本根进行处理，计算所有满足条件的根
    for root in base_roots:
        diff = pow(root, n - 1, p) * n % p
        new_base = p
    
        # 如果diff不等于0，使用逆元进行计算根
        if diff != 0:
            m_inv = invert(diff, p)
            for _ in range(k - 1):
                new_base *= p
                tmp = pow(root, n, new_base) - a
                tmp *= m_inv
                root = (root - tmp) % new_base
            tot_roots.add(root)
        else:
            roots_in_base = {root}
            for _ in range(k - 1):
                new_base *= p
                new_roots = set()
    
                # 计算满足条件的所有根
                for k_ in roots_in_base:
                    if pow(k_, n, new_base) != a % new_base:
                        continue
                    while k_ not in new_roots:
                        new_roots.add(k_)
                        k_ = (k_ + (new_base // p)) % new_base
                roots_in_base = new_roots
    
            tot_roots = tot_roots | roots_in_base
    
    # 返回排序后的所有根
    return sorted(tot_roots)
# 计算给定的整数 `a` 取模 `p` 后的余数
a = a % p
# 确保 `a`, `n`, `p` 都是整数
a, n, p = as_int(a), as_int(n), as_int(p)

# 如果 `n` 不是正整数，抛出数值错误
if n < 1:
    raise ValueError("n should be positive")
# 如果 `p` 不是正整数，抛出数值错误
if p < 1:
    raise ValueError("p should be positive")

# 如果 `n` 等于 `1`，返回 `a` 的列表形式（或者单个值），根据 `all_roots` 的值确定
if n == 1:
    return [a] if all_roots else a
# 如果 `n` 等于 `2`，调用 `sqrt_mod` 函数计算 `a` 在模 `p` 下的平方根，根据 `all_roots` 的值返回结果
if n == 2:
    return sqrt_mod(a, p, all_roots)

# 初始化空列表 `base` 和 `prime_power`
base = []
prime_power = []

# 使用 `factorint` 函数将 `p` 分解为质因数的幂次，并遍历每个质因数 `q` 及其幂次 `e`
for q, e in factorint(p).items():
    # 调用 `_nthroot_mod_prime_power` 函数计算在模 `q^e` 下 `a` 的 `n` 次根
    tot_roots = _nthroot_mod_prime_power(a, n, q, e)
    # 如果没有找到根，则根据 `all_roots` 的值返回空列表或者 `None`
    if not tot_roots:
        return [] if all_roots else None
    # 将 `q^e` 添加到 `prime_power` 列表，将计算出的根按序添加到 `base` 列表
    prime_power.append(q**e)
    base.append(sorted(tot_roots))

# 使用 `gf_crt1` 函数计算 `prime_power` 的公共剩余类，得到 `P`, `E`, `S`
P, E, S = gf_crt1(prime_power, ZZ)

# 使用 `product` 函数遍历 `base` 中的根组合，使用 `gf_crt2` 函数计算其同余类
ret = sorted(map(int, {gf_crt2(c, prime_power, P, E, S, ZZ) for c in product(*base)}))

# 如果 `all_roots` 为真，则返回所有根的排序列表，否则返回第一个根
if all_roots:
    return ret
if ret:
    return ret[0]


``````python
def nthroot_mod(a, n, p, all_roots=False):
    """
    Find the solutions to ``x**n = a mod p``.

    Parameters
    ==========

    a : integer
        The base integer.
    n : positive integer
        The exponent integer (must be positive).
    p : positive integer
        The modulus integer (must be positive).
    all_roots : bool, optional
        If False, returns the smallest root. If True, returns all roots.

    Returns
    =======

    list[int] | int | None :
        solutions to ``x**n = a mod p``.
        The table of the output type is:

        ========== ========== ==========
        all_roots  has roots  Returns
        ========== ========== ==========
        True       Yes        list[int]
        True       No         []
        False      Yes        int
        False      No         None
        ========== ========== ==========

    Raises
    ======

    ValueError
        If ``a``, ``n`` or ``p`` is not integer.
        If ``n`` or ``p`` is not positive.

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import nthroot_mod
    >>> nthroot_mod(11, 4, 19)
    8
    >>> nthroot_mod(11, 4, 19, True)
    [8, 11]
    >>> nthroot_mod(68, 3, 109)
    23

    References
    ==========

    .. [1] P. Hackman "Elementary Number Theory" (2009), page 76

    """
    # 计算 `a` 对 `p` 取模后的余数
    a = a % p
    # 确保 `a`, `n`, `p` 都是整数
    a, n, p = as_int(a), as_int(n), as_int(p)

    # 如果 `n` 小于 1，抛出数值错误
    if n < 1:
        raise ValueError("n should be positive")
    # 如果 `p` 小于 1，抛出数值错误
    if p < 1:
        raise ValueError("p should be positive")
    # 如果 `n` 等于 1，返回 `a` 的列表形式（或者单个值），根据 `all_roots` 的值确定
    if n == 1:
        return [a] if all_roots else a
    # 如果 `n` 等于 2，调用 `sqrt_mod` 函数计算 `a` 在模 `p` 下的平方根，根据 `all_roots` 的值返回结果
    if n == 2:
        return sqrt_mod(a, p, all_roots)

    # 初始化空列表 `base` 和 `prime_power`
    base = []
    prime_power = []

    # 使用 `factorint` 函数将 `p` 分解为质因数的幂次，并遍历每个质因数 `q` 及其幂次 `e`
    for q, e in factorint(p).items():
        # 调用 `_nthroot_mod_prime_power` 函数计算在模 `q^e` 下 `a` 的 `n` 次根
        tot_roots = _nthroot_mod_prime_power(a, n, q, e)
        # 如果没有找到根，则根据 `all_roots` 的值返回空列表或者 `None`
        if not tot_roots:
            return [] if all_roots else None
        # 将 `q^e` 添加到 `prime_power` 列表，将计算出的根按序添加到 `base` 列表
        prime_power.append(q**e)
        base.append(sorted(tot_roots))

    # 使用 `gf_crt1` 函数计算 `prime_power` 的公共剩余类，得到 `P`, `E`, `S`
    P, E, S = gf_crt1(prime_power, ZZ)

    # 使用 `product` 函数遍历 `base` 中的根组合，使用 `gf_crt2` 函数计算其同余类
    ret = sorted(map(int, {gf_crt2(c, prime_power, P, E, S, ZZ) for c in product(*base)}))

    # 如果 `all_roots` 为真，则返回所有根的排序列表，否则返回第一个根
    if all_roots:
        return ret
    if ret:
        return ret[0]
    # 从 sympy.functions.combinatorial.numbers 导入 legendre_symbol 函数，并将其命名为 _legendre_symbol
    from sympy.functions.combinatorial.numbers import legendre_symbol as _legendre_symbol
    # 调用 _legendre_symbol 函数计算 Legendre 符号，返回结果
    return _legendre_symbol(a, p)
# 声明一个装饰器函数，用于标记被弃用的函数 `jacobi_symbol`
@deprecated("""\
The `sympy.ntheory.residue_ntheory.jacobi_symbol` has been moved to `sympy.functions.combinatorial.numbers.jacobi_symbol`.""",
            deprecated_since_version="1.13",
            active_deprecations_target='deprecated-ntheory-symbolic-functions')
# 定义 `jacobi_symbol` 函数，返回 Jacobi 符号 `(m / n)`
def jacobi_symbol(m, n):
    """
    Returns the Jacobi symbol `(m / n)`.

    .. deprecated:: 1.13

        The ``jacobi_symbol`` function is deprecated. Use :class:`sympy.functions.combinatorial.numbers.jacobi_symbol`
        instead. See its documentation for more information. See
        :ref:`deprecated-ntheory-symbolic-functions` for details.

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

    Parameters
    ==========

    m : integer
    n : odd positive integer

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

    is_quad_residue, legendre_symbol
    """
    # 导入 `jacobi_symbol` 函数并返回其结果
    from sympy.functions.combinatorial.numbers import jacobi_symbol as _jacobi_symbol
    return _jacobi_symbol(m, n)


# 声明一个装饰器函数，用于标记被弃用的函数 `mobius`
@deprecated("""\
The `sympy.ntheory.residue_ntheory.mobius` has been moved to `sympy.functions.combinatorial.numbers.mobius`.""",
            deprecated_since_version="1.13",
            active_deprecations_target='deprecated-ntheory-symbolic-functions')
# 定义 `mobius` 函数，返回莫比乌斯函数值
def mobius(n):
    """
    Mobius function maps natural number to {-1, 0, 1}

    .. deprecated:: 1.13

        The ``mobius`` function is deprecated. Use :class:`sympy.functions.combinatorial.numbers.mobius`
        instead. See its documentation for more information. See
        :ref:`deprecated-ntheory-symbolic-functions` for details.

    It is defined as follows:
        1) `1` if `n = 1`.
        2) `0` if `n` has a squared prime factor.
        3) `(-1)^k` if `n` is a square-free positive integer with `k`
           number of prime factors.
    """
    # Import the 'mobius' function from the sympy library for combinatorial numbers
    from sympy.functions.combinatorial.numbers import mobius as _mobius
    # Call the Mobius function to calculate its value for the given positive integer n and return the result
    return _mobius(n)
# 使用试乘法算法计算模 ``n`` 下以 ``b`` 为底 ``a`` 的离散对数。

# 如果未指定 ``order``，则将其设为模数 ``n``。
order = n

# 初始化 ``x`` 为 1，作为试乘法算法的起始值。
x = 1

# 使用循环尝试所有可能的离散对数，范围为 ``order``。
for i in range(order):
    # 如果找到离散对数，即 ``x == a``，则返回当前循环索引 ``i``。
    if x == a:
        return i
    # 否则，更新 ``x`` 为 ``x * b % n``，继续下一轮循环。
    x = x * b % n

# 如果未找到离散对数，抛出值错误异常。
raise ValueError("Log does not exist")


# 使用 Baby-step giant-step 算法计算模 ``n`` 下以 ``b`` 为底 ``a`` 的离散对数。

# 如果未指定 ``order``，则使用函数 ``n_order(b, n)`` 计算模 ``n`` 下 ``b`` 的阶数。
m = sqrt(order) + 1

# 初始化字典 ``T``，用于存储 Baby-step 的结果。
T = {}

# 初始化 ``x`` 为 1，作为 Baby-step 算法的起始值。
x = 1

# 计算 Baby-step 阶段的结果。
for i in range(m):
    # 将当前 ``x`` 的值映射到字典 ``T`` 中，键为 ``x``，值为当前循环索引 ``i``。
    T[x] = i
    # 更新 ``x`` 为 ``x * b % n``，继续下一轮循环。
    x = x * b % n

# 计算 Giant-step 的步长。
z = pow(b, -m, n)

# 初始化 ``x`` 为 ``a``，作为 Giant-step 算法的起始值。
x = a

# 计算 Giant-step 阶段的结果。
for i in range(m):
    # 如果 ``x`` 存在于字典 ``T`` 中，则找到离散对数，返回相应的结果。
    if x in T:
        return i * m + T[x]
    # 更新 ``x`` 为 ``x * z % n``，继续下一轮循环。
    x = x * z % n

# 如果未找到离散对数，抛出值错误异常。
raise ValueError("Log does not exist")


# 使用 Pollard's Rho 算法计算模 ``n`` 下以 ``b`` 为底 ``a`` 的离散对数。

# 如果未指定 ``order``，则使用函数 ``n_order(b, n)`` 计算模 ``n`` 下 ``b`` 的阶数。
if order is None:
    order = n_order(b, n)

# 初始化随机数生成器 ``randint``，用于生成随机种子。
randint = _randint(rseed)
    # 对于给定的重试次数范围内进行迭代
    for i in range(retries):
        # 随机生成两个整数 aa 和 ba，范围在 1 到 order-1 之间
        aa = randint(1, order - 1)
        ba = randint(1, order - 1)
        # 计算 xa = (b^aa * a^ba) % n
        xa = pow(b, aa, n) * pow(a, ba, n) % n

        # 计算 xa 对 3 取模的结果
        c = xa % 3
        # 根据 c 的值执行不同的操作分支
        if c == 0:
            # 当 c 等于 0 时，计算 xb = (a * xa) % n，并更新 ab 和 bb 的值
            xb = a * xa % n
            ab = aa
            bb = (ba + 1) % order
        elif c == 1:
            # 当 c 等于 1 时，计算 xb = (xa * xa) % n，并更新 ab 和 bb 的值
            xb = xa * xa % n
            ab = (aa + aa) % order
            bb = (ba + ba) % order
        else:
            # 当 c 不等于 0 或 1 时，计算 xb = (b * xa) % n，并更新 ab 的值
            xb = b * xa % n
            ab = (aa + 1) % order
            bb = ba

        # 对于 order 范围内的每个 j 进行迭代
        for j in range(order):
            # 计算 xa 对 3 取模的结果
            c = xa % 3
            # 根据 c 的值执行不同的操作分支
            if c == 0:
                # 当 c 等于 0 时，计算 xa = (a * xa) % n，并更新 ba 的值
                xa = a * xa % n
                ba = (ba + 1) % order
            elif c == 1:
                # 当 c 等于 1 时，计算 xa = (xa * xa) % n，并更新 aa 和 ba 的值
                xa = xa * xa % n
                aa = (aa + aa) % order
                ba = (ba + ba) % order
            else:
                # 当 c 不等于 0 或 1 时，计算 xa = (b * xa) % n，并更新 aa 的值
                xa = b * xa % n
                aa = (aa + 1) % order

            # 计算 xb 对 3 取模的结果
            c = xb % 3
            # 根据 c 的值执行不同的操作分支
            if c == 0:
                # 当 c 等于 0 时，计算 xb = (a * xb) % n，并更新 bb 的值
                xb = a * xb % n
                bb = (bb + 1) % order
            elif c == 1:
                # 当 c 等于 1 时，计算 xb = (xb * xb) % n，并更新 ab 和 bb 的值
                xb = xb * xb % n
                ab = (ab + ab) % order
                bb = (bb + bb) % order
            else:
                # 当 c 不等于 0 或 1 时，计算 xb = (b * xb) % n，并更新 ab 的值
                xb = b * xb % n
                ab = (ab + 1) % order

            # 计算 xb 对 3 取模的结果
            c = xb % 3
            # 根据 c 的值执行不同的操作分支
            if c == 0:
                # 当 c 等于 0 时，计算 xb = (a * xb) % n，并更新 bb 的值
                xb = a * xb % n
                bb = (bb + 1) % order
            elif c == 1:
                # 当 c 等于 1 时，计算 xb = (xb * xb) % n，并更新 ab 和 bb 的值
                xb = xb * xb % n
                ab = (ab + ab) % order
                bb = (bb + bb) % order
            else:
                # 当 c 不等于 0 或 1 时，计算 xb = (b * xb) % n，并更新 ab 的值
                xb = b * xb % n
                ab = (ab + 1) % order

            # 如果 xa 等于 xb，则计算 r = (ba - bb) % order
            if xa == xb:
                r = (ba - bb) % order
                try:
                    # 尝试计算 e = (r^(-1) * (ab - aa)) % order
                    e = invert(r, order) * (ab - aa) % order
                    # 如果满足条件 pow(b, e, n) % n == a，则返回 e
                    if (pow(b, e, n) - a) % n == 0:
                        return e
                except ZeroDivisionError:
                    pass
                # 找到符合条件的 e 后退出循环
                break
    # 如果所有的重试都没有找到符合条件的 e，则抛出 ValueError 异常
    raise ValueError("Pollard's Rho failed to find logarithm")
# 尝试使用给定的因子基因素分解 n
# 如果成功，返回相对于因子基的指数列表；否则返回 None
def _discrete_log_is_smooth(n: int, factorbase: list):
    factors = [0]*len(factorbase)  # 初始化一个长度为因子基长度的指数列表
    for i, p in enumerate(factorbase):
        while n % p == 0:  # 尽可能多地除以 p
            factors[i] += 1
            n = n // p
    if n != 1:
        return None  # 如果在结束时还有剩余，表示无法完全分解
    return factors


# 使用指数对数算法计算模 n 下基数为 b 的 a 的离散对数
# 需要给出并确保群的阶数是素数。不适用于小阶数，此时算法可能无法找到解决方案。
def _discrete_log_index_calculus(n, a, b, order, rseed=None):
    """
    Index Calculus algorithm for computing the discrete logarithm of ``a`` to
    the base ``b`` modulo ``n``.

    The group order must be given and prime. It is not suitable for small orders
    and the algorithm might fail to find a solution in such situations.

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import _discrete_log_index_calculus
    >>> _discrete_log_index_calculus(24570203447, 23859756228, 2, 12285101723)
    4519867240

    See Also
    ========

    discrete_log

    References
    ==========

    .. [1] "Handbook of applied cryptography", Menezes, A. J., Van, O. P. C., &
        Vanstone, S. A. (1997).
    """
    randint = _randint(rseed)  # 使用给定的种子生成随机数
    from math import sqrt, exp, log
    a %= n  # 取模运算，确保 a 在模 n 下
    b %= n  # 取模运算，确保 b 在模 n 下
    B = int(exp(0.5 * sqrt(log(n) * log(log(n))) * (1 + 1/log(log(n)))))  # 计算因子基的启发式界限 B
    max = 5 * B * B  # 预期的关系寻找尝试次数上限
    factorbase = list(primerange(B))  # 计算因子基
    lf = len(factorbase)  # 因子基的长度
    ordermo = order - 1  # 群的阶数减一
    abx = a
    for x in range(order):
        if abx == 1:
            return (order - x) % order  # 如果 abx 等于 1，返回 x 对于 order 的模
        relationa = _discrete_log_is_smooth(abx, factorbase)  # 检查 abx 是否是光滑数
        if relationa:
            relationa = [r % order for r in relationa] + [x]  # 将关系转换为模 order 的形式，并添加 x
            break
        abx = abx * b % n  # 计算下一个 abx，即 abx = a*pow(b, x, n) % n

    else:
        raise ValueError("Index Calculus failed")  # 如果循环结束仍未找到解决方案，则引发 ValueError

    relations = [None] * lf  # 初始化关系列表
    k = 1  # 找到的关系数量
    kk = 0
    while k < 3 * lf and kk < max:  # 对于我们因子基中的所有素数查找关系
        x = randint(1, ordermo)
        # 计算 b^x mod n 的离散对数，并检查是否平滑
        relation = _discrete_log_is_smooth(pow(b, x, n), factorbase)
        if relation is None:
            kk += 1
            continue
        k += 1
        kk = 0
        relation += [x]
        index = lf  # 确定第一个非零条目的索引
        for i in range(lf):
            ri = relation[i] % order
            if ri > 0 and relations[i] is not None:  # 如果可能，使此条目为零
                for j in range(lf + 1):
                    relation[j] = (relation[j] - ri * relations[i][j]) % order
            else:
                relation[i] = ri
            if relation[i] > 0 and index == lf:  # 是否是第一个非零条目的索引？
                index = i
        if index == lf or relations[index] is not None:  # 关系没有新信息
            continue
        # 关系包含新信息
        rinv = pow(relation[index], -1, order)  # 规范化第一个非零条目
        for j in range(index, lf + 1):
            relation[j] = rinv * relation[j] % order
        relations[index] = relation
        for i in range(lf):  # 从关于 a 的一个关系中减去新关系
            if relationa[i] > 0 and relations[i] is not None:
                rbi = relationa[i]
                for j in range(lf + 1):
                    relationa[j] = (relationa[j] - rbi * relations[i][j]) % order
            if relationa[i] > 0:  # 第一个非零条目的索引
                break  # 在这一点上我们不需要进一步减少
        else:  # 所有未知数已消失
            #print(f"成功在 {k} 个关系中找到 {lf} 个")
            x = (order - relationa[lf]) % order
            if pow(b, x, n) == a:
                return x
            raise ValueError("指数算法失败")
    raise ValueError("指数算法失败")
# 计算 Pohlig-Hellman 离散对数算法，用于计算模 n 下以 b 为底 a 的离散对数。
def _discrete_log_pohlig_hellman(n, a, b, order=None, order_factors=None):
    """
    Pohlig-Hellman algorithm for computing the discrete logarithm of ``a`` to
    the base ``b`` modulo ``n``.

    In order to compute the discrete logarithm, the algorithm takes advantage
    of the factorization of the group order. It is more efficient when the
    group order factors into many small primes.

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import _discrete_log_pohlig_hellman
    >>> _discrete_log_pohlig_hellman(251, 210, 71)
    197

    See Also
    ========

    discrete_log

    References
    ==========

    .. [1] "Handbook of applied cryptography", Menezes, A. J., Van, O. P. C., &
        Vanstone, S. A. (1997).
    """
    # 导入中国剩余定理模块
    from .modular import crt
    # 取模运算，确保 a 和 b 在模 n 下的值
    a %= n
    b %= n

    # 如果未提供 order 参数，则计算 b 在模 n 下的阶
    if order is None:
        order = n_order(b, n)
    # 如果未提供 order_factors 参数，则分解阶 order 的因数
    if order_factors is None:
        order_factors = factorint(order)
    # 初始化 l 列表，用于存放每个因数的计算结果
    l = [0] * len(order_factors)

    # 遍历每个因数及其指数
    for i, (pi, ri) in enumerate(order_factors.items()):
        # 对于每个因数的每个指数，执行 Pohlig-Hellman 算法步骤
        for j in range(ri):
            # 计算 aj，这里涉及幂运算和离散对数计算
            aj = pow(a * pow(b, -l[i], n), order // pi**(j + 1), n)
            # 计算 bj
            bj = pow(b, order // pi, n)
            # 计算 cj，使用递归调用的 discrete_log 函数来计算
            cj = discrete_log(n, aj, bj, pi, True)
            # 更新 l[i]，按照 Pohlig-Hellman 算法中的公式
            l[i] += cj * pi**j

    # 使用中国剩余定理计算最终的离散对数结果
    d, _ = crt([pi**ri for pi, ri in order_factors.items()], l)
    return d


# 计算离散对数的通用函数，支持不同的算法和参数选项
def discrete_log(n, a, b, order=None, prime_order=None):
    """
    Compute the discrete logarithm of ``a`` to the base ``b`` modulo ``n``.

    This is a recursive function to reduce the discrete logarithm problem in
    cyclic groups of composite order to the problem in cyclic groups of prime
    order.

    It employs different algorithms depending on the problem (subgroup order
    size, prime order or not):

        * Trial multiplication
        * Baby-step giant-step
        * Pollard's Rho
        * Index Calculus
        * Pohlig-Hellman

    Examples
    ========

    >>> from sympy.ntheory import discrete_log
    >>> discrete_log(41, 15, 7)
    3

    References
    ==========

    .. [1] https://mathworld.wolfram.com/DiscreteLogarithm.html
    .. [2] "Handbook of applied cryptography", Menezes, A. J., Van, O. P. C., &
        Vanstone, S. A. (1997).

    """
    # 导入数学模块中的一些函数
    from math import sqrt, log
    # 强制转换为整数类型
    n, a, b = as_int(n), as_int(a), as_int(b)
    if order is None:
        # 如果没有提供 order 参数，则计算它及其因数
        factors = {}
        # 对 n 进行因数分解，得到它的素因子及其指数
        for px, kx in factorint(n).items():
            if kx > 1:
                # 处理素因子的指数大于1的情况
                if px in factors:
                    factors[px] += kx - 1
                else:
                    factors[px] = kx - 1
            # 对 px - 1 进行因数分解，得到它的素因子及其指数
            for py, ky in factorint(px - 1).items():
                if py in factors:
                    factors[py] += ky
                else:
                    factors[py] = ky
        # 计算 order 的值
        order = 1
        for px, kx in factors.items():
            order *= px**kx
        # 现在 order 是群的阶数，factors 是 order 的因数分解

        # 计算 order 的因数
        order_factors = {}
        for p, e in factors.items():
            i = 0
            for _ in range(e):
                # 判断 b 的阶数是否能整除群的阶数
                if pow(b, order // p, n) == 1:
                   order //= p
                   i += 1
                else:
                    break
            if i < e:
                order_factors[p] = e - i

    if prime_order is None:
        # 如果 prime_order 参数未提供，则检查 order 是否为素数
        prime_order = isprime(order)

    if order < 1000:
        # 如果 order 小于 1000，使用试乘法和试除法求离散对数
        return _discrete_log_trial_mul(n, a, b, order)
    elif prime_order:
        # 如果 order 是素数，根据不同的运行时间选择合适的算法
        if 4*sqrt(log(n)*log(log(n))) < log(order) - 10:  # 10 是经验确定的值
            # 如果指数对数小于特定值，使用指数对数法求离散对数
            return _discrete_log_index_calculus(n, a, b, order)
        elif order < 1000000000000:
            # 否则如果 order 小于一定值，使用 Shanks 算法求离散对数
            return _discrete_log_shanks_steps(n, a, b, order)
        # 其他情况使用 Pollard rho 算法求离散对数
        return _discrete_log_pollard_rho(n, a, b, order)

    # 如果 order 不是素数或者大于等于 1000，则使用 Pohlig-Hellman 算法求离散对数
    return _discrete_log_pohlig_hellman(n, a, b, order, order_factors)
# 寻找满足 `a x^2 + b x + c \equiv 0 \pmod{n}` 的解
def quadratic_congruence(a, b, c, n):
    # 确保输入的参数为整数
    a = as_int(a)
    b = as_int(b)
    c = as_int(c)
    n = as_int(n)
    # 如果 n 小于等于 1，抛出值错误
    if n <= 1:
        raise ValueError("n should be an integer greater than 1")
    # 对 a, b, c 取模 n
    a %= n
    b %= n
    c %= n

    # 如果 a 等于 0，则转为一元线性同余方程求解
    if a == 0:
        return linear_congruence(b, -c, n)
    
    # 如果 n 等于 2，特殊情况处理
    if n == 2:
        roots = []
        if c == 0:
            roots.append(0)
        if (b + c) % 2:
            roots.append(1)
        return roots
    
    # 如果 gcd(2*a, n) == 1，则使用二次剩余求解
    if gcd(2*a, n) == 1:
        inv_a = invert(a, n)
        b *= inv_a
        c *= inv_a
        if b % 2:
            b += n
        b >>= 1
        return sorted((i - b) % n for i in sqrt_mod_iter(b**2 - c, n))
    
    # 一般情况下，使用高阶剩余求解
    res = set()
    for i in sqrt_mod_iter(b**2 - 4*a*c, 4*a*n):
        res.update(j % n for j in linear_congruence(2*a, i - b, 4*a*n))
    return sorted(res)


# 检查表达式是否为整数系数的一元多项式，如果是则返回系数，否则抛出 ValueError
def _valid_expr(expr):
    if not expr.is_polynomial():
        raise ValueError("The expression should be a polynomial")
    polynomial = Poly(expr)
    if not polynomial.is_univariate:
        raise ValueError("The expression should be univariate")
    if not polynomial.domain == ZZ:
        raise ValueError("The expression should should have integer coefficients")
    return polynomial.all_coeffs()


# 求解多项式同余方程 `expr \equiv 0 \pmod{m}`
def polynomial_congruence(expr, m):
    # 获取表达式的系数，并对其取模 m
    coefficients = _valid_expr(expr)
    coefficients = [num % m for num in coefficients]
    rank = len(coefficients)
    
    # 根据系数的长度，选择相应的求解方式
    if rank == 3:
        return quadratic_congruence(*coefficients, m)
    if rank == 2:
        return quadratic_congruence(0, *coefficients, m)
    if coefficients[0] == 1 and 1 + coefficients[-1] == sum(coefficients):
        return nthroot_mod(-coefficients[-1], rank - 1, m, True)
    return gf_csolve(coefficients, m)
def binomial_mod(n, m, k):
    """Compute ``binomial(n, m) % k``.

    Explanation
    ===========

    Returns ``binomial(n, m) % k`` using a generalization of Lucas'
    Theorem for prime powers given by Granville [1]_, in conjunction with
    the Chinese Remainder Theorem.  The residue for each prime power
    is calculated in time O(log^2(n) + q^4*log(n)log(p) + q^4*p*log^3(p)).

    Parameters
    ==========

    n : an integer
        The first parameter of the binomial coefficient.
    m : an integer
        The second parameter of the binomial coefficient.
    k : a positive integer
        Modulus for which the binomial coefficient is computed.

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import binomial_mod
    >>> binomial_mod(10, 2, 6)  # binomial(10, 2) = 45
    3
    >>> binomial_mod(17, 9, 10)  # binomial(17, 9) = 24310
    0

    References
    ==========

    .. [1] Binomial coefficients modulo prime powers, Andrew Granville,
        Available: https://web.archive.org/web/20170202003812/http://www.dms.umontreal.ca/~andrew/PDF/BinCoeff.pdf
    """
    if k < 1: raise ValueError('k is required to be positive')
    # We decompose k into a product of prime powers and apply
    # the generalization of Lucas' Theorem given by Granville
    # to obtain binomial(n, m) mod p^e, and then use the Chinese
    # Remainder Theorem to obtain the result mod k
    if n < 0 or m < 0 or m > n: return 0
    factorisation = factorint(k)
    # Compute residues for each prime power using _binomial_mod_prime_power function
    residues = [_binomial_mod_prime_power(n, m, p, e) for p, e in factorisation.items()]
    # Apply the Chinese Remainder Theorem to combine residues
    return crt([p**pw for p, pw in factorisation.items()], residues, check=False)[0]


def _binomial_mod_prime_power(n, m, p, q):
    """Compute ``binomial(n, m) % p**q`` for a prime ``p``.

    Parameters
    ==========

    n : positive integer
        The first parameter of the binomial coefficient.
    m : nonnegative integer
        The second parameter of the binomial coefficient.
    p : prime
        The prime base for the modulus.
    q : positive integer
        The exponent of the prime modulus.

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import _binomial_mod_prime_power
    >>> _binomial_mod_prime_power(10, 2, 3, 2)  # binomial(10, 2) = 45
    0
    >>> _binomial_mod_prime_power(17, 9, 2, 4)  # binomial(17, 9) = 24310
    6

    References
    ==========

    .. [1] Binomial coefficients modulo prime powers, Andrew Granville,
        Available: https://web.archive.org/web/20170202003812/http://www.dms.umontreal.ca/~andrew/PDF/BinCoeff.pdf
    """
    # Calculate the modulus p^q for binomial coefficient computation
    modulo = pow(p, q)
    def up_factorial(u):
        """Compute (u*p)!_p modulo p^q."""
        r = q // 2  # 计算 q 的一半
        fac = prod = 1  # 初始化阶乘和乘积为 1
        if r == 1 and p == 2 or 2*r + 1 in (p, p*p):
            if q % 2 == 1: r += 1  # 如果 q 是奇数，增加 r 的值
            modulo, div = pow(p, 2*r), pow(p, 2*r - q)  # 计算模数和除数
        else:
            modulo, div = pow(p, 2*r + 1), pow(p, (2*r + 1) - q)  # 计算模数和除数
        for j in range(1, r + 1):
            for mul in range((j - 1)*p + 1, j*p):  # 遍历乘积范围
                fac *= mul  # 计算阶乘
                fac %= modulo  # 取模运算
            bj_ = bj(u, j, r)  # 计算 bj(u, j, r)
            prod *= pow(fac, bj_, modulo)  # 计算乘积
            prod %= modulo  # 取模运算
        if p == 2:
            sm = u // 2  # 计算 u 除以 2 的商
            for j in range(1, r + 1): sm += j//2 * bj(u, j, r)  # 计算累加值
            if sm % 2 == 1: prod *= -1  # 如果累加值为奇数，乘以 -1
        prod %= modulo//div  # 取模运算
        return prod % modulo  # 返回最终结果

    def bj(u, j, r):
        """Compute the exponent of (j*p)!_p in the calculation of (u*p)!_p."""
        prod = u  # 初始值为 u
        for i in range(1, r + 1):
            if i != j: prod *= u*u - i*i  # 计算乘积
        for i in range(1, r + 1):
            if i != j: prod //= j*j - i*i  # 计算除法
        return prod // j  # 返回结果

    def up_plus_v_binom(u, v):
        """Compute binomial(u*p + v, v)_p modulo p^q."""
        prod = 1  # 初始化乘积为 1
        div = invert(factorial(v), modulo)  # 计算阶乘的逆元
        for j in range(1, q):
            b = div  # 初始化 b 为阶乘的逆元
            for v_ in range(j*p + 1, j*p + v + 1):
                b *= v_  # 计算乘积
                b %= modulo  # 取模运算
            aj = u  # 初始值为 u
            for i in range(1, q):
                if i != j: aj *= u - i  # 计算乘积
            for i in range(1, q):
                if i != j: aj //= j - i  # 计算除法
            aj //= j  # 计算除法
            prod *= pow(b, aj, modulo)  # 计算乘积
            prod %= modulo  # 取模运算
        return prod  # 返回最终结果

    @recurrence_memo([1])
    def factorial(v, prev):
        """Compute v! modulo p^q."""
        return v*prev[-1] % modulo  # 计算阶乘

    def factorial_p(n):
        """Compute n!_p modulo p^q."""
        u, v = divmod(n, p)  # 计算商和余数
        return (factorial(v) * up_factorial(u) * up_plus_v_binom(u, v)) % modulo  # 返回最终结果

    prod = 1  # 初始化乘积为 1
    Nj, Mj, Rj = n, m, n - m  # 初始化变量
    # e0 will be the p-adic valuation of binomial(n, m) at p
    e0 = carry = eq_1 = j = 0  # 初始化 e0、carry、eq_1 和 j
    while Nj:
        numerator = factorial_p(Nj % modulo)  # 计算分子
        denominator = factorial_p(Mj % modulo) * factorial_p(Rj % modulo) % modulo  # 计算分母
        Nj, (Mj, mj), (Rj, rj) = Nj//p, divmod(Mj, p), divmod(Rj, p)  # 更新 Nj、Mj 和 Rj
        carry = (mj + rj + carry) // p  # 计算进位值
        e0 += carry  # 更新 e0
        if j >= q - 1: eq_1 += carry  # 更新 eq_1
        prod *= numerator * invert(denominator, modulo)  # 计算乘积
        prod %= modulo  # 取模运算
        j += 1  # 更新 j

    mul = pow(1 if p == 2 and q >= 3 else -1, eq_1, modulo)  # 计算乘法结果
    return (pow(p, e0, modulo) * mul * prod) % modulo  # 返回最终结果
```