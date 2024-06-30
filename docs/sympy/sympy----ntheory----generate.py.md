# `D:\src\scipysrc\sympy\sympy\ntheory\generate.py`

```
"""
Generating and counting primes.

"""

from bisect import bisect, bisect_left
from itertools import count
# Using arrays for sieving instead of lists greatly reduces
# memory consumption
from array import array as _array

from sympy.core.random import randint
from sympy.external.gmpy import sqrt
from .primetest import isprime
from sympy.utilities.decorator import deprecated
from sympy.utilities.misc import as_int


def _as_int_ceiling(a):
    """ Wrapping ceiling in as_int will raise an error if there was a problem
        determining whether the expression was exactly an integer or not."""
    from sympy.functions.elementary.integers import ceiling
    return as_int(ceiling(a))


class Sieve:
    """A list of prime numbers, implemented as a dynamically
    growing sieve of Eratosthenes. When a lookup is requested involving
    an odd number that has not been sieved, the sieve is automatically
    extended up to that number. Implementation details limit the number of
    primes to ``2^32-1``.

    Examples
    ========

    >>> from sympy import sieve
    >>> sieve._reset() # this line for doctest only
    >>> 25 in sieve
    False
    >>> sieve._list
    array('L', [2, 3, 5, 7, 11, 13, 17, 19, 23])
    """

    # data shared (and updated) by all Sieve instances
    def __init__(self, sieve_interval=1_000_000):
        """ Initial parameters for the Sieve class.

        Parameters
        ==========

        sieve_interval (int): Amount of memory to be used

        Raises
        ======

        ValueError
            If ``sieve_interval`` is not positive.

        """
        # Initialize starting values for prime, totient, and mobius lists
        self._n = 6
        self._list = _array('L', [2, 3, 5, 7, 11, 13]) # primes
        self._tlist = _array('L', [0, 1, 1, 2, 2, 4]) # totient
        self._mlist = _array('i', [0, 1, -1, -1, 0, -1]) # mobius
        # Validate sieve_interval parameter
        if sieve_interval <= 0:
            raise ValueError("sieve_interval should be a positive integer")
        self.sieve_interval = sieve_interval
        # Assert that initial arrays have correct lengths
        assert all(len(i) == self._n for i in (self._list, self._tlist, self._mlist))

    def __repr__(self):
        """Returns a string representation of the Sieve object."""
        return ("<%s sieve (%i): %i, %i, %i, ... %i, %i\n"
             "%s sieve (%i): %i, %i, %i, ... %i, %i\n"
             "%s sieve (%i): %i, %i, %i, ... %i, %i>") % (
             'prime', len(self._list),
                 self._list[0], self._list[1], self._list[2],
                 self._list[-2], self._list[-1],
             'totient', len(self._tlist),
                 self._tlist[0], self._tlist[1],
                 self._tlist[2], self._tlist[-2], self._tlist[-1],
             'mobius', len(self._mlist),
                 self._mlist[0], self._mlist[1],
                 self._mlist[2], self._mlist[-2], self._mlist[-1])
    def _reset(self, prime=None, totient=None, mobius=None):
        """重置所有缓存（默认）。若要重置其中一个或多个，请将相应关键字设置为True。"""
        # 如果三个参数都为None，则默认全部重置
        if all(i is None for i in (prime, totient, mobius)):
            prime = totient = mobius = True
        # 如果需要重置素数缓存
        if prime:
            # 保留前self._n个元素，其余丢弃，相当于重置
            self._list = self._list[:self._n]
        # 如果需要重置欧拉函数缓存
        if totient:
            self._tlist = self._tlist[:self._n]
        # 如果需要重置莫比乌斯函数缓存
        if mobius:
            self._mlist = self._mlist[:self._n]

    def extend(self, n):
        """扩展筛法以覆盖所有不大于n的素数。

        Examples
        ========

        >>> from sympy import sieve
        >>> sieve._reset() # 仅为doctest准备的行
        >>> sieve.extend(30)
        >>> sieve[10] == 29
        True
        """
        n = int(n)
        # `num`始终为偶数，这满足`self._primerange`函数的要求。
        num = self._list[-1] + 1
        if n < num:
            return
        num2 = num**2
        while num2 <= n:
            # 将素数范围[num, num2)内的素数添加到self._list中
            self._list += _array('L', self._primerange(num, num2))
            num, num2 = num2, num2**2
        # 合并筛法结果，将[num, n+1)内的素数添加到self._list中
        self._list += _array('L', self._primerange(num, n + 1))

    def _primerange(self, a, b):
        """生成范围(a, b)内的所有素数。

        Parameters
        ==========

        a, b : 正整数，满足以下条件
                * a是偶数
                * 2 < self._list[-1] < a < b < nextprime(self._list[-1])**2

        Yields
        ======

        p (int): 范围内的素数，满足``a < p < b``

        Examples
        ========

        >>> from sympy.ntheory.generate import Sieve
        >>> s = Sieve()
        >>> s._list[-1]
        13
        >>> list(s._primerange(18, 31))
        [19, 23, 29]

        """
        if b % 2:
            b -= 1
        while a < b:
            block_size = min(self.sieve_interval, (b - a) // 2)
            # 创建列表，使得block[x]对应的(a + 2x + 1)是素数。注意这里不考虑偶数。
            block = [True] * block_size
            for p in self._list[1:bisect(self._list, sqrt(a + 2 * block_size + 1))]:
                for t in range((-(a + 1 + p) // 2) % p, block_size, p):
                    block[t] = False
            for idx, p in enumerate(block):
                if p:
                    yield a + 2 * idx + 1
            a += 2 * block_size
    def extend_to_no(self, i):
        """Extend to include the ith prime number.

        Parameters
        ==========

        i : integer
            The index of the prime number to extend the list to.

        Examples
        ========

        >>> from sympy import sieve
        >>> sieve._reset() # this line for doctest only
        >>> sieve.extend_to_no(9)
        >>> sieve._list
        array('L', [2, 3, 5, 7, 11, 13, 17, 19, 23])

        Notes
        =====

        The list is extended by 50% if it is too short, so it is
        likely that it will be longer than requested.
        """
        i = as_int(i)  # Ensure i is an integer
        while len(self._list) < i:
            self.extend(int(self._list[-1] * 1.5))  # Extend the list by 1.5 times the last element

    def primerange(self, a, b=None):
        """Generate all prime numbers in the range [2, a) or [a, b).

        Examples
        ========

        >>> from sympy import sieve, prime

        All primes less than 19:

        >>> print([i for i in sieve.primerange(19)])
        [2, 3, 5, 7, 11, 13, 17]

        All primes greater than or equal to 7 and less than 19:

        >>> print([i for i in sieve.primerange(7, 19)])
        [7, 11, 13, 17]

        All primes through the 10th prime

        >>> list(sieve.primerange(prime(10) + 1))
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

        """
        if b is None:
            b = _as_int_ceiling(a)  # Convert a to integer and set b to ceiling of a
            a = 2
        else:
            a = max(2, _as_int_ceiling(a))  # Ensure a is at least 2 and convert a to integer
            b = _as_int_ceiling(b)  # Convert b to integer
        if a >= b:
            return  # If a is greater than or equal to b, return nothing
        self.extend(b)  # Extend the list of primes up to b
        yield from self._list[bisect_left(self._list, a):  # Yield primes in range [a, b)
                              bisect_left(self._list, b)]

    def totientrange(self, a, b):
        """Generate all totient numbers for the range [a, b).

        Examples
        ========

        >>> from sympy import sieve
        >>> print([i for i in sieve.totientrange(7, 18)])
        [6, 4, 6, 4, 10, 4, 12, 6, 8, 8, 16]
        """
        a = max(1, _as_int_ceiling(a))  # Ensure a is at least 1 and convert a to integer
        b = _as_int_ceiling(b)  # Convert b to integer
        n = len(self._tlist)
        if a >= b:
            return  # If a is greater than or equal to b, return nothing
        elif b <= n:
            for i in range(a, b):
                yield self._tlist[i]  # Yield totient values from _tlist for range [a, b)
        else:
            self._tlist += _array('L', range(n, b))  # Extend _tlist up to b
            for i in range(1, n):
                ti = self._tlist[i]
                if ti == i - 1:
                    startindex = (n + i - 1) // i * i
                    for j in range(startindex, b, i):
                        self._tlist[j] -= self._tlist[j] // i
                if i >= a:
                    yield ti

            for i in range(n, b):
                ti = self._tlist[i]
                if ti == i:
                    for j in range(i, b, i):
                        self._tlist[j] -= self._tlist[j] // i
                if i >= a:
                    yield self._tlist[i]
    def mobiusrange(self, a, b):
        """Generate all mobius numbers for the range [a, b).

        Parameters
        ==========

        a : integer
            First number in range

        b : integer
            First number outside of range

        Examples
        ========

        >>> from sympy import sieve
        >>> print([i for i in sieve.mobiusrange(7, 18)])
        [-1, 0, 0, 1, -1, 0, -1, 1, 1, 0, -1]
        """
        # 将 a 设定为大于等于 1 的整数
        a = max(1, _as_int_ceiling(a))
        # 将 b 设定为大于等于 b 的整数
        b = _as_int_ceiling(b)
        # 获取当前 mobius 列表的长度
        n = len(self._mlist)
        
        # 如果 a 大于等于 b，则直接返回
        if a >= b:
            return
        # 如果 b 小于等于当前 mobius 列表长度 n，则在现有列表中迭代生成结果
        elif b <= n:
            for i in range(a, b):
                yield self._mlist[i]
        # 如果 b 大于当前 mobius 列表长度 n，则扩展列表长度并生成结果
        else:
            self._mlist += _array('i', [0]*(b - n))
            for i in range(1, n):
                mi = self._mlist[i]
                startindex = (n + i - 1) // i * i
                for j in range(startindex, b, i):
                    self._mlist[j] -= mi
                if i >= a:
                    yield mi

            for i in range(n, b):
                mi = self._mlist[i]
                for j in range(2 * i, b, i):
                    self._mlist[j] -= mi
                if i >= a:
                    yield mi

    def search(self, n):
        """Return the indices i, j of the primes that bound n.

        If n is prime then i == j.

        Although n can be an expression, if ceiling cannot convert
        it to an integer then an n error will be raised.

        Examples
        ========

        >>> from sympy import sieve
        >>> sieve.search(25)
        (9, 10)
        >>> sieve.search(23)
        (9, 9)
        """
        # 将 n 设为大于等于 n 的整数
        test = _as_int_ceiling(n)
        n = as_int(n)
        # 如果 n 小于 2，则引发 ValueError 异常
        if n < 2:
            raise ValueError("n should be >= 2 but got: %s" % n)
        # 如果 n 大于当前列表中的最后一个素数，则扩展列表
        if n > self._list[-1]:
            self.extend(n)
        # 使用二分查找确定 n 在列表中的位置
        b = bisect(self._list, n)
        # 如果列表中的第 b-1 个元素等于 test，则返回 (b, b)
        if self._list[b - 1] == test:
            return b, b
        # 否则返回 (b, b+1)
        else:
            return b, b + 1

    def __contains__(self, n):
        try:
            n = as_int(n)
            assert n >= 2
        except (ValueError, AssertionError):
            return False
        # 如果 n 是偶数，则只有当 n 等于 2 时才返回 True
        if n % 2 == 0:
            return n == 2
        # 否则调用 search 方法，判断返回的两个索引是否相等
        a, b = self.search(n)
        return a == b

    def __iter__(self):
        # 使用 count 从 1 开始迭代
        for n in count(1):
            # 生成迭代器，调用 mobiusrange 方法，返回 mobius 数
            yield self[n]
    def __getitem__(self, n):
        """Return the nth prime number"""
        # 如果 n 是切片对象
        if isinstance(n, slice):
            # 扩展筛选列表以包含 n.stop 之前的所有素数
            self.extend_to_no(n.stop)
            # Python 2.7 中，切片的起始默认为 0 而不是 None，所以不能默认为 1。
            start = n.start if n.start is not None else 0
            if start < 1:
                # 如果起始索引小于 1，则抛出索引错误
                raise IndexError("Sieve indices start at 1.")
            # 返回切片结果列表，注意索引要减去 1
            return self._list[start - 1:n.stop - 1:n.step]
        else:
            # 如果 n 小于 1，则抛出索引错误
            if n < 1:
                raise IndexError("Sieve indices start at 1.")
            # 将 n 转换为整数
            n = as_int(n)
            # 扩展筛选列表以包含第 n 个素数
            self.extend_to_no(n)
            # 返回第 n 个素数，注意索引要减去 1
            return self._list[n - 1]
# 生成一个全局对象，用于在试除等操作中重复使用
sieve = Sieve()

# 计算第n个素数，其中素数从prime(1) = 2开始索引，prime(2) = 3，依此类推...
# 第n个素数大约是$n\log(n)$
# 对于$x$的对数积分是对小于等于$x$的素数数量的一个很好的近似
# 即li(x) ~ pi(x)
# 实际上，对于我们关心的数字（x<1e11），li(x) - pi(x) < 50000
# 同样地，对于此函数可以处理的数字，我们可以安全地假设li(x) > pi(x)

# 在这里，我们使用二分搜索找到最小的整数m，使得li(m) > n
# 现在有pi(m-1) < li(m-1) <= n
# 我们使用primepi函数找到pi(m - 1)

# 从m开始，我们需要找到n - pi(m-1)个额外的素数
# 对于此实现可以处理的输入，我们最多需要测试约10**5个数以获取我们的答案

def prime(nth):
    n = as_int(nth)
    if n < 1:
        raise ValueError("nth must be a positive integer; prime(1) == 2")
    if n <= len(sieve._list):
        return sieve[n]

    from sympy.functions.elementary.exponential import log
    from sympy.functions.special.error_functions import li
    a = 2 # 二分搜索的下界
    # 二分搜索的上界
    b = int(n*(log(n) + log(log(n))))

    while a < b:
        mid = (a + b) >> 1
        if li(mid) > n:
            b = mid
        else:
            a = mid + 1
    n_primes = _primepi(a - 1)
    while n_primes < n:
        if isprime(a):
            n_primes += 1
        a += 1
    return a - 1

# primepi函数的弃用声明
@deprecated("""\
The `sympy.ntheory.generate.primepi` has been moved to `sympy.functions.combinatorial.numbers.primepi`.""",
deprecated_since_version="1.13",
active_deprecations_target='deprecated-ntheory-symbolic-functions')
def primepi(n):
    # 导入 sympy 库中的 primepi 函数，用于计算不大于 n 的素数个数
    from sympy.functions.combinatorial.numbers import primepi as func_primepi
    # 调用 func_primepi 函数计算不大于 n 的素数个数，并返回结果
    return func_primepi(n)
def _primepi(n:int) -> int:
    r""" Represents the prime counting function pi(n) = the number
    of prime numbers less than or equal to n.

    Explanation
    ===========

    In sieve method, we remove all multiples of prime p
    except p itself.

    Let phi(i,j) be the number of integers 2 <= k <= i
    which remain after sieving from primes less than
    or equal to j.
    Clearly, pi(n) = phi(n, sqrt(n))

    If j is not a prime,
    phi(i,j) = phi(i, j - 1)

    if j is a prime,
    We remove all numbers(except j) whose
    smallest prime factor is j.

    Let $x= j \times a$ be such a number, where $2 \le a \le i / j$
    Now, after sieving from primes $\le j - 1$,
    a must remain
    (because x, and hence a has no prime factor $\le j - 1$)
    Clearly, there are phi(i / j, j - 1) such a
    which remain on sieving from primes $\le j - 1$

    Now, if a is a prime less than equal to j - 1,
    $x= j \times a$ has smallest prime factor = a, and
    has already been removed(by sieving from a).
    So, we do not need to remove it again.
    (Note: there will be pi(j - 1) such x)

    Thus, number of x, that will be removed are:
    phi(i / j, j - 1) - phi(j - 1, j - 1)
    (Note that pi(j - 1) = phi(j - 1, j - 1))

    $\Rightarrow$ phi(i,j) = phi(i, j - 1) - phi(i / j, j - 1) + phi(j - 1, j - 1)

    So,following recursion is used and implemented as dp:

    phi(a, b) = phi(a, b - 1), if b is not a prime
    phi(a, b) = phi(a, b-1)-phi(a / b, b-1) + phi(b-1, b-1), if b is prime

    Clearly a is always of the form floor(n / k),
    which can take at most $2\sqrt{n}$ values.
    Two arrays arr1,arr2 are maintained
    arr1[i] = phi(i, j),
    arr2[i] = phi(n // i, j)

    Finally the answer is arr2[1]

    Parameters
    ==========

    n : int

    """
    # 如果 n 小于 2，则返回 0，因为没有素数小于 2
    if n < 2:
        return 0
    # 如果 n 小于等于预先筛选素数表中最大的素数，则使用线性搜索得出结果
    if n <= sieve._list[-1]:
        return sieve.search(n)[0]
    # 计算 sqrt(n)
    lim = sqrt(n)
    # 初始化 arr1 数组，arr1[i] = phi(i, j)，其中 j <= lim
    arr1 = [(i + 1) >> 1 for i in range(lim + 1)]
    # 初始化 arr2 数组，arr2[i] = phi(n // i, j)，其中 j <= lim
    arr2 = [0] + [(n//i + 1) >> 1 for i in range(1, lim + 1)]
    # 初始化跳过数组，用于标记非素数
    skip = [False] * (lim + 1)
    # 对于每一个奇数 i 从 3 到 lim，步长为 2
    for i in range(3, lim + 1, 2):
        # 现在，arr1[k]=phi(k,i - 1),
        # arr2[k] = phi(n // k,i - 1) # 不是所有的 k 都会执行这个
        # 如果 skip[i] 为 True，跳过当前循环，因为 i 是一个合数
        if skip[i]:
            continue
        # 将 arr1[i - 1] 的值赋给 p
        p = arr1[i - 1]
        # 将 i 的倍数 j 标记为 True，表示它们是合数
        for j in range(i, lim + 1, i):
            skip[j] = True
        # 更新 arr2
        # 计算 phi(n/j, i) = phi(n/j, i-1) - phi(n/(i*j), i-1) + phi(i-1, i-1)
        for j in range(1, min(n // (i * i), lim) + 1, 2):
            # 如果 skip[j] 为 True，跳过当前循环
            if skip[j]:
                continue
            # 计算 st = i * j
            st = i * j
            if st <= lim:
                # 更新 arr2[j] 的值
                arr2[j] -= arr2[st] - p
            else:
                arr2[j] -= arr1[n // st] - p
        # 更新 arr1
        # 计算 phi(j, i) = phi(j, i-1) - phi(j/i, i-1) + phi(i-1, i-1)
        # 对于范围在 i**2 以下的 j，这部分是固定的，不需要重新计算
        for j in range(lim, min(lim, i*i - 1), -1):
            arr1[j] -= arr1[j // i] - p
    # 返回 arr2[1] 的值作为结果
    return arr2[1]
# 返回大于 n 的第 ith 个素数
def nextprime(n, ith=1):
    n = int(n)  # 将 n 转换为整数
    i = as_int(ith)  # 将 ith 转换为整数
    if i <= 0:
        raise ValueError("ith should be positive")  # 如果 ith 不为正整数，抛出错误
    if n < 2:
        n = 2
        i -= 1
    if n <= sieve._list[-2]:  # 如果 n 小于等于筛选器的倒数第二个元素
        l, _ = sieve.search(n)  # 在筛选器中搜索 n
        if l + i - 1 < len(sieve._list):  # 如果 l + i - 1 小于筛选器列表的长度
            return sieve._list[l + i - 1]  # 返回筛选器中的第 l + i - 1 个素数
        return nextprime(sieve._list[-1], l + i - len(sieve._list))  # 否则递归调用 nextprime 函数
    if 1 < i:
        for _ in range(i):  # 循环 i 次
            n = nextprime(n)  # 调用 nextprime 函数
        return n
    nn = 6*(n//6)  # 取 n 的最接近的 6 的倍数
    if nn == n:  # 如果 nn 等于 n
        n += 1
        if isprime(n):  # 如果 n 是素数
            return n
        n += 4
    elif n - nn == 5:
        n += 2
        if isprime(n):  # 如果 n 是素数
            return n
        n += 4
    else:
        n = nn + 5
    while 1:
        if isprime(n):  # 如果 n 是素数
            return n
        n += 2
        if isprime(n):  # 如果 n 是素数
            return n
        n += 4


# 返回小于 n 的最大素数
def prevprime(n):
    n = _as_int_ceiling(n)  # 向上取整 n
    if n < 3:
        raise ValueError("no preceding primes")  # 如果没有前面的素数，抛出错误
    if n < 8:
        return {3: 2, 4: 3, 5: 3, 6: 5, 7: 5}[n]  # 返回 n 对应的素数
    if n <= sieve._list[-1]:  # 如果 n 小于等于筛选器的最后一个元素
        l, u = sieve.search(n)  # 在筛选器中搜索 n
        if l == u:
            return sieve[l-1]
        else:
            return sieve[l]
    nn = 6*(n//6)  # 取 n 的最接近的 6 的倍数
    if n - nn <= 1:
        n = nn - 1
        if isprime(n):  # 如果 n 是素数
            return n
        n -= 4
    else:
        n = nn + 1
    while 1:
        if isprime(n):  # 如果 n 是素数
            return n
        n -= 2
        if isprime(n):  # 如果 n 是素数
            return n
        n -= 4
    """ Generate a list of all prime numbers in the range [2, a),
        or [a, b).

        If the range exists in the default sieve, the values will
        be returned from there; otherwise values will be returned
        but will not modify the sieve.

        Examples
        ========

        >>> from sympy import primerange, prime

        All primes less than 19:

        >>> list(primerange(19))
        [2, 3, 5, 7, 11, 13, 17]

        All primes greater than or equal to 7 and less than 19:

        >>> list(primerange(7, 19))
        [7, 11, 13, 17]

        All primes through the 10th prime

        >>> list(primerange(prime(10) + 1))
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

        The Sieve method, primerange, is generally faster but it will
        occupy more memory as the sieve stores values. The default
        instance of Sieve, named sieve, can be used:

        >>> from sympy import sieve
        >>> list(sieve.primerange(1, 30))
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

        Notes
        =====

        Some famous conjectures about the occurrence of primes in a given
        range are [1]:

        - Twin primes: though often not, the following will give 2 primes
                    an infinite number of times:
                        primerange(6*n - 1, 6*n + 2)
        - Legendre's: the following always yields at least one prime
                        primerange(n**2, (n+1)**2+1)
        - Bertrand's (proven): there is always a prime in the range
                        primerange(n, 2*n)
        - Brocard's: there are at least four primes in the range
                        primerange(prime(n)**2, prime(n+1)**2)

        The average gap between primes is log(n) [2]; the gap between
        primes can be arbitrarily large since sequences of composite
        numbers are arbitrarily large, e.g. the numbers in the sequence
        n! + 2, n! + 3 ... n! + n are all composite.

        See Also
        ========

        prime : Return the nth prime
        nextprime : Return the ith prime greater than n
        prevprime : Return the largest prime smaller than n
        randprime : Returns a random prime in a given range
        primorial : Returns the product of primes based on condition
        Sieve.primerange : return range from already computed primes
                           or extend the sieve to contain the requested
                           range.

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Prime_number
        .. [2] https://primes.utm.edu/notes/gaps.html
    """
    # 如果 b 为 None，则设定默认范围 [2, a)
    if b is None:
        a, b = 2, a
    # 如果 a 大于等于 b，则直接返回
    if a >= b:
        return
    # 如果已有默认筛选器中包含了所需范围，直接返回该范围的素数
    largest_known_prime = sieve._list[-1]
    if b <= largest_known_prime:
        yield from sieve.primerange(a, b)
        return
    # 如果部分范围已知，则返回已知部分的素数
    # 如果 a 小于等于已知的最大素数 largest_known_prime
    if a <= largest_known_prime:
        # 使用生成器从筛选器对象的列表中生成大于等于 a 的所有素数
        yield from sieve._list[bisect_left(sieve._list, a):]
        # 更新 a，使其大于已知的最大素数 largest_known_prime
        a = largest_known_prime + 1
    # 如果 a 不小于 largest_known_prime 且 a 是奇数
    elif a % 2:
        # 减小 a，使其变为偶数
        a -= 1
    # 计算 b 和 (largest_known_prime)**2 中较小的值，作为尾部值 tail
    tail = min(b, (largest_known_prime)**2)
    # 如果 a 小于尾部值 tail
    if a < tail:
        # 使用生成器从筛选器对象中生成 a 到 tail 范围内的所有素数
        yield from sieve._primerange(a, tail)
        # 更新 a，使其等于 tail
        a = tail
    # 如果 b 小于等于 a，则直接返回
    if b <= a:
        return
    # 否则，计算并生成大于 a 的素数，直到达到 b
    while 1:
        # 获取大于当前 a 的下一个素数
        a = nextprime(a)
        # 如果当前素数 a 小于 b，则生成该素数
        if a < b:
            yield a
        # 否则，结束生成器
        else:
            return
# 定义一个函数，用于生成一个指定范围内的随机素数
def randprime(a, b):
    """ Return a random prime number in the range [a, b).

        Bertrand's postulate assures that
        randprime(a, 2*a) will always succeed for a > 1.

        Note that due to implementation difficulties,
        the prime numbers chosen are not uniformly random.
        For example, there are two primes in the range [112, 128),
        ``113`` and ``127``, but ``randprime(112, 128)`` returns ``127``
        with a probability of 15/17.

        Examples
        ========

        >>> from sympy import randprime, isprime
        >>> randprime(1, 30) #doctest: +SKIP
        13
        >>> isprime(randprime(1, 30))
        True

        See Also
        ========

        primerange : Generate all primes in a given range

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Bertrand's_postulate

    """
    # 如果a大于等于b，返回空（无效范围）
    if a >= b:
        return
    # 将a和b转换为整数
    a, b = map(int, (a, b))
    # 在[a-1, b)范围内生成一个随机整数n
    n = randint(a - 1, b)
    # 找到大于等于n的下一个素数p
    p = nextprime(n)
    # 如果找到的素数p超出了范围[b, +∞)，则使用小于等于b的前一个素数
    if p >= b:
        p = prevprime(b)
    # 如果找到的素数p小于a，则说明指定范围内没有素数，抛出值错误
    if p < a:
        raise ValueError("no primes exist in the specified range")
    # 返回找到的素数p
    return p


def primorial(n, nth=True):
    """
    Returns the product of the first n primes (default) or
    the primes less than or equal to n (when ``nth=False``).

    Examples
    ========

    >>> from sympy.ntheory.generate import primorial, primerange
    >>> from sympy import factorint, Mul, primefactors, sqrt
    >>> primorial(4) # the first 4 primes are 2, 3, 5, 7
    210
    >>> primorial(4, nth=False) # primes <= 4 are 2 and 3
    6
    >>> primorial(1)
    2
    >>> primorial(1, nth=False)
    1
    >>> primorial(sqrt(101), nth=False)
    210

    One can argue that the primes are infinite since if you take
    a set of primes and multiply them together (e.g. the primorial) and
    then add or subtract 1, the result cannot be divided by any of the
    original factors, hence either 1 or more new primes must divide this
    product of primes.

    In this case, the number itself is a new prime:

    >>> factorint(primorial(4) + 1)
    {211: 1}

    In this case two new primes are the factors:

    >>> factorint(primorial(4) - 1)
    {11: 1, 19: 1}

    Here, some primes smaller and larger than the primes multiplied together
    are obtained:

    >>> p = list(primerange(10, 20))
    >>> sorted(set(primefactors(Mul(*p) + 1)).difference(set(p)))
    [2, 5, 31, 149]

    See Also
    ========

    primerange : Generate all primes in a given range

    """
    # 如果nth为True，将n转换为整数，否则转换为int类型
    if nth:
        n = as_int(n)
    else:
        n = int(n)
    # 如果n小于1，抛出值错误
    if n < 1:
        raise ValueError("primorial argument must be >= 1")
    # 初始化p为1
    p = 1
    # 如果nth为True，计算前n个素数的乘积
    if nth:
        for i in range(1, n + 1):
            p *= prime(i)
    else:
        # 如果nth为False，计算小于等于n的所有素数的乘积
        for i in primerange(2, n + 1):
            p *= i
    # 返回计算得到的乘积
    return p


def cycle_length(f, x0, nmax=None, values=False):
    """For a given iterated sequence, return a generator that gives
    the length of the iterated cycle (lambda) and the length of terms

    """
    # 函数未完整，后续代码未提供
    # 将 nmax 转换为整数，如果为 None 则设为 0
    nmax = int(nmax or 0)

    # 主循环阶段：搜索连续的二的幂次方
    power = lam = 1
    tortoise, hare = x0, f(x0)  # tortoise 和 hare 分别为起始点和其下一个元素/节点
    i = 1
    if values:
        yield tortoise  # 如果 values 为真，则生成 tortoise
    while tortoise != hare and (not nmax or i < nmax):
        i += 1
        if power == lam:   # 是时候开始一个新的二的幂次方了？
            tortoise = hare
            power *= 2
            lam = 0
        if values:
            yield hare  # 如果 values 为真，则生成 hare
        hare = f(hare)  # 更新 hare
        lam += 1
    if nmax and i == nmax:
        if values:
            return  # 如果达到 nmax 并且 values 为真，则返回
        else:
            yield nmax, None  # 如果达到 nmax 并且 values 不为真，则生成 nmax 和 None
            return
    if not values:
        # 寻找长度为 lambda 的第一个重复出现位置
        mu = 0
        tortoise = hare = x0
        for i in range(lam):
            hare = f(hare)
        while tortoise != hare:
            tortoise = f(tortoise)
            hare = f(hare)
            mu += 1
        yield lam, mu  # 生成 lambda 和 mu
# 返回第 n 个复合数，其中复合数按顺序索引，如 composite(1) = 4, composite(2) = 6，依此类推。
def composite(nth):
    # 将 nth 转换为整数类型
    n = as_int(nth)
    # 如果 nth 小于 1，则抛出值错误
    if n < 1:
        raise ValueError("nth must be a positive integer; composite(1) == 4")
    
    # 初始已知的前10个复合数
    composite_arr = [4, 6, 8, 9, 10, 12, 14, 15, 16, 18]
    # 如果 nth 小于等于 10，直接返回已知的第 nth 个复合数
    if n <= 10:
        return composite_arr[n - 1]

    # 设定初始搜索范围
    a, b = 4, sieve._list[-1]
    # 如果 nth 小于等于 b - _primepi(b) - 1，使用二分查找找到第 nth 个复合数
    if n <= b - _primepi(b) - 1:
        while a < b - 1:
            mid = (a + b) >> 1  # 取中间值
            if mid - _primepi(mid) - 1 > n:
                b = mid
            else:
                a = mid
        if isprime(a):
            a -= 1
        return a

    # 导入所需函数
    from sympy.functions.elementary.exponential import log
    from sympy.functions.special.error_functions import li
    # 设定二分查找的初始上下界
    a = 4  # 下界
    b = int(n*(log(n) + log(log(n))))  # 上界

    # 二分查找找到第 nth 个复合数
    while a < b:
        mid = (a + b) >> 1  # 取中间值
        if mid - li(mid) - 1 > n:
            b = mid
        else:
            a = mid + 1

    # 计算小于等于 a 的复合数个数
    n_composites = a - _primepi(a) - 1
    # 找到第 nth 个复合数
    while n_composites > n:
        if not isprime(a):
            n_composites -= 1
        a -= 1
    if isprime(a):
        a -= 1
    return a


# 返回小于等于 n 的正整数中的复合数个数
def compositepi(n):
    # 将 n 转换为整数类型
    n = int(n)
    # 如果 n 小于 4，直接返回 0
    if n < 4:
        return 0
    # 返回小于等于 n 的复合数个数
    return n - _primepi(n) - 1
```