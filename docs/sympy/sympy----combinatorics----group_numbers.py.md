# `D:\src\scipysrc\sympy\sympy\combinatorics\group_numbers.py`

```
from itertools import chain, combinations  # 导入 itertools 库中的 chain 和 combinations 函数

from sympy.external.gmpy import gcd  # 从 sympy.external.gmpy 中导入 gcd 函数
from sympy.ntheory.factor_ import factorint  # 从 sympy.ntheory.factor_ 中导入 factorint 函数
from sympy.utilities.misc import as_int  # 从 sympy.utilities.misc 中导入 as_int 函数


def _is_nilpotent_number(factors: dict) -> bool:
    """ Check whether `n` is a nilpotent number.
    Note that ``factors`` is a prime factorization of `n`.

    This is a low-level helper for ``is_nilpotent_number``, for internal use.
    """
    for p in factors.keys():  # 遍历 factors 字典中的键 p
        for q, e in factors.items():  # 遍历 factors 字典中的键值对 (q, e)
            # We want to calculate
            # any(pow(q, k, p) == 1 for k in range(1, e + 1))
            m = 1  # 初始化 m 为 1
            for _ in range(e):  # 循环 e 次
                m = m*q % p  # 计算 m = (m * q) % p
                if m == 1:  # 如果 m 等于 1
                    return False  # 返回 False
    return True  # 若未找到 m == 1 的情况，则返回 True


def is_nilpotent_number(n) -> bool:
    """
    Check whether `n` is a nilpotent number. A number `n` is said to be
    nilpotent if and only if every finite group of order `n` is nilpotent.
    For more information see [1]_.

    Examples
    ========

    >>> from sympy.combinatorics.group_numbers import is_nilpotent_number
    >>> from sympy import randprime
    >>> is_nilpotent_number(21)
    False
    >>> is_nilpotent_number(randprime(1, 30)**12)
    True

    References
    ==========

    .. [1] Pakianathan, J., Shankar, K., Nilpotent Numbers,
           The American Mathematical Monthly, 107(7), 631-634.
    .. [2] https://oeis.org/A056867

    """
    n = as_int(n)  # 将 n 转换为整数
    if n <= 0:  # 如果 n 小于等于 0
        raise ValueError("n must be a positive integer, not %i" % n)  # 抛出数值错误异常
    return _is_nilpotent_number(factorint(n))  # 返回 _is_nilpotent_number 函数的结果


def is_abelian_number(n) -> bool:
    """
    Check whether `n` is an abelian number. A number `n` is said to be abelian
    if and only if every finite group of order `n` is abelian. For more
    information see [1]_.

    Examples
    ========

    >>> from sympy.combinatorics.group_numbers import is_abelian_number
    >>> from sympy import randprime
    >>> is_abelian_number(4)
    True
    >>> is_abelian_number(randprime(1, 2000)**2)
    True
    >>> is_abelian_number(60)
    False

    References
    ==========

    .. [1] Pakianathan, J., Shankar, K., Nilpotent Numbers,
           The American Mathematical Monthly, 107(7), 631-634.
    .. [2] https://oeis.org/A051532

    """
    n = as_int(n)  # 将 n 转换为整数
    if n <= 0:  # 如果 n 小于等于 0
        raise ValueError("n must be a positive integer, not %i" % n)  # 抛出数值错误异常
    factors = factorint(n)  # 计算 n 的质因数分解
    return all(e < 3 for e in factors.values()) and _is_nilpotent_number(factors)  # 返回判断条件的结果


def is_cyclic_number(n) -> bool:
    """
    Check whether `n` is a cyclic number. A number `n` is said to be cyclic
    if and only if every finite group of order `n` is cyclic. For more
    information see [1]_.

    Examples
    ========

    >>> from sympy.combinatorics.group_numbers import is_cyclic_number
    >>> from sympy import randprime
    >>> is_cyclic_number(15)
    True
    >>> is_cyclic_number(randprime(1, 2000)**2)
    False
    >>> is_cyclic_number(4)
    False

    References
    ==========

    .. [1] Pakianathan, J., Shankar, K., Nilpotent Numbers,
           The American Mathematical Monthly, 107(7), 631-634.
    .. [2] https://oeis.org/A056867

    """
    n = as_int(n)  # 将 n 转换为整数
    if n <= 0:  # 如果 n 小于等于 0
        raise ValueError("n must be a positive integer, not %i" % n)  # 抛出数值错误异常
    factors = factorint(n)  # 计算 n 的质因数分解
    return all(e < 3 for e in factors.values()) and _is_nilpotent_number(factors)  # 返回判断条件的结果
    """
    n = as_int(n)  # 将输入的 n 转换为整数（如果可能）
    if n <= 0:  # 如果 n 小于等于 0
        raise ValueError("n must be a positive integer, not %i" % n)  # 抛出数值错误，要求 n 必须是正整数，而不是当前值 n
    factors = factorint(n)  # 使用 factorint 函数计算 n 的质因数分解
    return all(e == 1 for e in factors.values()) and _is_nilpotent_number(factors)  # 返回条件：所有质因数的指数均为 1，并且调用 _is_nilpotent_number 函数判断是否为 nilpotent 数
def _holder_formula(prime_factors):
    r""" Number of groups of order `n`.
    where `n` is squarefree and its prime factors are ``prime_factors``.
    i.e., ``n == math.prod(prime_factors)``

    Explanation
    ===========

    When `n` is squarefree, the number of groups of order `n` is expressed by

    .. math ::
        \sum_{d \mid n} \prod_p \frac{p^{c(p, d)} - 1}{p - 1}

    where `n=de`, `p` is the prime factor of `e`,
    and `c(p, d)` is the number of prime factors `q` of `d` such that `q \equiv 1 \pmod{p}` [2]_.

    The formula is elegant, but can be improved when implemented as an algorithm.
    Since `n` is assumed to be squarefree, the divisor `d` of `n` can be identified with the power set of prime factors.
    We let `N` be the set of prime factors of `n`.
    `F = \{p \in N : \forall q \in N, q \not\equiv 1 \pmod{p} \}, M = N \setminus F`, we have the following.

    .. math ::
        \sum_{d \in 2^{M}} \prod_{p \in M \setminus d} \frac{p^{c(p, F \cup d)} - 1}{p - 1}

    Practically, many prime factors are expected to be members of `F`, thus reducing computation time.

    Parameters
    ==========

    prime_factors : set
        The set of prime factors of ``n``. where `n` is squarefree.

    Returns
    =======

    int : Number of groups of order ``n``

    Examples
    ========

    >>> from sympy.combinatorics.group_numbers import _holder_formula
    >>> _holder_formula({2}) # n = 2
    1
    >>> _holder_formula({2, 3}) # n = 2*3 = 6
    2

    See Also
    ========

    groups_count

    References
    ==========

    .. [1] Otto Holder, Die Gruppen der Ordnungen p^3, pq^2, pqr, p^4,
           Math. Ann. 43 pp. 301-412 (1893).
           http://dx.doi.org/10.1007/BF01443651
    .. [2] John H. Conway, Heiko Dietrich and E.A. O'Brien,
           Counting groups: gnus, moas and other exotica
           The Mathematical Intelligencer 30, 6-15 (2008)
           https://doi.org/10.1007/BF02985731

    """
    # Determine the set F of prime factors p such that q % p != 1 for all q in prime_factors
    F = {p for p in prime_factors if all(q % p != 1 for q in prime_factors)}
    # Determine set M of prime factors that are not in F
    M = prime_factors - F

    # Initialize the sum to calculate the number of groups of order n
    s = 0
    # Generate the power set of M using combinations
    powerset = chain.from_iterable(combinations(M, r) for r in range(len(M)+1))
    # Iterate through each subset ps in the powerset
    for ps in powerset:
        ps = set(ps)
        prod = 1
        # Iterate through each prime factor p in M minus ps
        for p in M - ps:
            # Calculate c(p, F ∪ ps), the count of prime factors q in F ∪ ps such that q % p == 1
            c = len([q for q in F | ps if q % p == 1])
            # Update prod with (p^c - 1) / (p - 1)
            prod *= (p**c - 1) // (p - 1)
            # If prod becomes zero, break out of the loop
            if not prod:
                break
        # Add prod to the sum s
        s += prod
    # Return the calculated number of groups of order n
    return s


def groups_count(n):
    r""" Number of groups of order `n`.
    In [1]_, ``gnu(n)`` is given, so we follow this notation here as well.

    Parameters
    ==========

    n : Integer
        ``n`` is a positive integer

    Returns
    =======

    int : ``gnu(n)``

    Raises
    ======

    ValueError
        Number of groups of order ``n`` is unknown or not implemented.
        For example, gnu(`2^{11}`) is not yet known.
        On the other hand, gnu(12) is known to be 5,
        but this has not yet been implemented in this function.

    Examples
    ========

    ```
    # 从 sympy.combinatorics.group_numbers 模块导入 groups_count 函数
    >>> from sympy.combinatorics.group_numbers import groups_count
    # 计算阶数为 3 的循环群的数量，结果为 1
    >>> groups_count(3) # There is only one cyclic group of order 3
    1
    # 计算阶数为 10 的群的数量，结果为 2，包括循环群和二面角群
    >>> # There are two groups of order 10: the cyclic group and the dihedral group
    >>> groups_count(10)
    2

    # 参见
    # ========

    # 如果 n 是循环数，则 `n` 是循环数
    # `n` is cyclic iff gnu(n) = 1

    # 参考资料
    # ==========

    # [1] John H. Conway, Heiko Dietrich and E.A. O'Brien,
    # Counting groups: gnus, moas and other exotica
    # The Mathematical Intelligencer 30, 6-15 (2008)
    # https://doi.org/10.1007/BF02985731
    # [2] https://oeis.org/A000001

    """
    将输入参数 n 转换为整数
    n = as_int(n)
    # 如果 n 小于等于 0，则抛出 ValueError 异常
    if n <= 0:
        raise ValueError("n must be a positive integer, not %i" % n)
    # 获取 n 的所有质因数及其指数
    factors = factorint(n)
    # 如果只有一个质因数
    if len(factors) == 1:
        # 取出质因数和其指数
        (p, e) = list(factors.items())[0]
        # 如果质因数是 2
        if p == 2:
            # 预定义 2 的幂次的循环群数量列表 A000679
            A000679 = [1, 1, 2, 5, 14, 51, 267, 2328, 56092, 10494213, 49487367289]
            # 如果指数 e 小于 A000679 的长度，返回相应的循环群数量
            if e < len(A000679):
                return A000679[e]
        # 如果质因数是 3
        if p == 3:
            # 预定义 3 的幂次的循环群数量列表 A090091
            A090091 = [1, 1, 2, 5, 15, 67, 504, 9310, 1396077, 5937876645]
            # 如果指数 e 小于 A090091 的长度，返回相应的循环群数量
            if e < len(A090091):
                return A090091[e]
        # 如果指数 e 小于等于 2，返回对应的循环群数量
        if e <= 2: # gnu(p) = 1, gnu(p**2) = 2
            return e
        # 如果指数 e 等于 3，返回对应的循环群数量
        if e == 3: # gnu(p**3) = 5
            return 5
        # 如果指数 e 等于 4，且 p 是奇质数，返回对应的循环群数量
        if e == 4: # if p is an odd prime, gnu(p**4) = 15
            return 15
        # 如果指数 e 等于 5，返回对应的循环群数量
        if e == 5: # if p >= 5, gnu(p**5) is expressed by the following equation
            return 61 + 2*p + 2*gcd(p-1, 3) + gcd(p-1, 4)
        # 如果指数 e 等于 6，返回对应的循环群数量
        if e == 6: # if p >= 6, gnu(p**6) is expressed by the following equation
            return 3*p**2 + 39*p + 344 +\
                  24*gcd(p-1, 3) + 11*gcd(p-1, 4) + 2*gcd(p-1, 5)
        # 如果指数 e 等于 7，返回对应的循环群数量
        if e == 7: # if p >= 7, gnu(p**7) is expressed by the following equation
            if p == 5:
                return 34297
            return 3*p**5 + 12*p**4 + 44*p**3 + 170*p**2 + 707*p + 2455 +\
                  (4*p**2 + 44*p + 291)*gcd(p-1, 3) + (p**2 + 19*p + 135)*gcd(p-1, 4) + \
                  (3*p + 31)*gcd(p-1, 5) + 4*gcd(p-1, 7) + 5*gcd(p-1, 8) + gcd(p-1, 9)
    # 如果有任何指数大于 1 的质因数，表示 n 不是平方自由数
    if any(e > 1 for e in factors.values()): # n is not squarefree
        # 对于一些已知的 n 值，它们有多个因子且不是平方自由数，返回预定义的结果
        small = {12: 5, 18: 5, 20: 5, 24: 15, 28: 4, 36: 14, 40: 14, 44: 4, 45: 2, 48: 52,
                50: 5, 52: 5, 54: 15, 56: 13, 60: 13, 63: 4, 68: 5, 72: 50, 75: 3, 76: 4,
                80: 52, 84: 15, 88: 12, 90: 10, 92: 4}
        # 如果 n 在 small 中，返回对应的值
        if n in small:
            return small[n]
        # 否则抛出异常，表示该阶数的群的数量未知或未实现
        raise ValueError("Number of groups of order n is unknown or not implemented")
    # 如果 n 有两个质因数，表示 n 是平方自由的半素数
    if len(factors) == 2: # n is squarefree semiprime
        p, q = list(factors.keys())
        # 如果 p 大于 q，交换它们的位置
        if p > q:
            p, q = q, p
        # 如果 q 除以 p 的余数为 1，返回 2，否则返回 1
        return 2 if q % p == 1 else 1
    # 使用 _holder_formula 处理其他情况
    return _holder_formula(set(factors.keys()))
```