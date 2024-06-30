# `D:\src\scipysrc\sympy\sympy\external\ntheory.py`

```
# sympy.external.ntheory
#
# This module provides pure Python implementations of some number theory
# functions that are alternately used from gmpy2 if it is installed.

# 导入系统模块
import sys
# 导入数学模块
import math

# 导入 mpmath 库中的 libmp 模块
import mpmath.libmp as mlib

# 用于快速查找最小的设置位数
_small_trailing = [0] * 256
for j in range(1, 8):
    _small_trailing[1 << j :: 1 << (j + 1)] = [j] * (1 << (7 - j))

# 在整数 x 中查找最低位的 1 的位置
def bit_scan1(x, n=0):
    if not x:
        return
    x = abs(x >> n)
    low_byte = x & 0xFF
    if low_byte:
        return _small_trailing[low_byte] + n

    t = 8 + n
    x >>= 8
    # 对于大部分情况，直接计算 2**m
    z = x.bit_length() - 1
    if x == 1 << z:
        return z + t

    if z < 300:
        # 固定 8 字节减少
        while not x & 0xFF:
            x >>= 8
            t += 8
    else:
        # 当可能有大量尾随 0 时，二进制减少是重要的
        p = z >> 1
        while not x & 0xFF:
            while x & ((1 << p) - 1):
                p >>= 1
            x >>= p
            t += p
    return t + _small_trailing[x & 0xFF]

# 在整数 x 中查找最低位的 0 的位置
def bit_scan0(x, n=0):
    return bit_scan1(x + (1 << n), n)

# 从 x 中移除因子 f
def remove(x, f):
    if f < 2:
        raise ValueError("factor must be > 1")
    if x == 0:
        return 0, 0
    if f == 2:
        b = bit_scan1(x)
        return x >> b, b
    m = 0
    y, rem = divmod(x, f)
    while not rem:
        x = y
        m += 1
        if m > 5:
            pow_list = [f**2]
            while pow_list:
                _f = pow_list[-1]
                y, rem = divmod(x, _f)
                if not rem:
                    m += 1 << len(pow_list)
                    x = y
                    pow_list.append(_f**2)
                else:
                    pow_list.pop()
        y, rem = divmod(x, f)
    return x, m

# 计算 x 的阶乘
def factorial(x):
    """Return x!."""
    return int(mlib.ifac(int(x)))

# 计算 x 的整数平方根
def sqrt(x):
    """Integer square root of x."""
    return int(mlib.isqrt(int(x)))

# 计算 x 的整数平方根和余数
def sqrtrem(x):
    """Integer square root of x and remainder."""
    s, r = mlib.sqrtrem(int(x))
    return (int(s), int(r))

# 根据 Python 版本选择 gcd 和 lcm 函数
if sys.version_info[:2] >= (3, 9):
    # 在 Python 3.9 及以上版本，可以接受多个参数
    gcd = math.gcd
    lcm = math.lcm

else:
    # 在 Python 3.8 及以下版本，使用 functools 中的 reduce 函数
    from functools import reduce

    # 计算多个整数的最大公约数
    def gcd(*args):
        """gcd of multiple integers."""
        return reduce(math.gcd, args, 0)

    # 计算多个整数的最小公倍数
    def lcm(*args):
        """lcm of multiple integers."""
        if 0 in args:
            return 0
        return reduce(lambda x, y: x*y//math.gcd(x, y), args, 1)

# 辅助函数，返回整数 n 的符号和绝对值
def _sign(n):
    if n < 0:
        return -1, -n
    return 1, n

# 计算 a 和 b 的最大公约数，并返回扩展欧几里得算法的结果
def gcdext(a, b):
    if not a or not b:
        g = abs(a) or abs(b)
        if not g:
            return (0, 0, 0)
        return (g, a // g, b // g)

    x_sign, a = _sign(a)
    y_sign, b = _sign(b)
    x, r = 1, 0
    y, s = 0, 1

    while b:
        q, c = divmod(a, b)
        a, b = b, c
        x, r = r, x - q*r
        y, s = s, y - q*s
    # 返回一个元组，包含变量 a 和两个经过符号处理后的数值 x 和 y
    return (a, x * x_sign, y * y_sign)
def is_square(x):
    """Return True if x is a square number."""
    if x < 0:
        return False

    # Note that the possible values of y**2 % n for a given n are limited.
    # For example, when n=4, y**2 % n can only take 0 or 1.
    # In other words, if x % 4 is 2 or 3, then x is not a square number.
    # Mathematically, it determines if it belongs to the set {y**2 % n},
    # but implementationally, it can be realized as a logical conjunction
    # with an n-bit integer.
    # see https://mersenneforum.org/showpost.php?p=110896
    # def magic(n):
    #     s = {y**2 % n for y in range(n)}
    #     s = set(range(n)) - s
    #     return sum(1 << bit for bit in s)
    # >>> print(hex(magic(128)))
    # 0xfdfdfdedfdfdfdecfdfdfdedfdfcfdec
    # >>> print(hex(magic(99)))
    # 0x5f6f9ffb6fb7ddfcb75befdec
    # >>> print(hex(magic(91)))
    # 0x6fd1bfcfed5f3679d3ebdec
    # >>> print(hex(magic(85)))
    # 0xdef9ae771ffe3b9d67dec
    if 0xfdfdfdedfdfdfdecfdfdfdedfdfcfdec & (1 << (x & 127)):
        return False  # Checks if x modulo operation is not a square number
    m = x % 765765  # 765765 = 99 * 91 * 85
    if 0x5f6f9ffb6fb7ddfcb75befdec & (1 << (m % 99)):
        return False  # Checks if m modulo operation is not a square number
    if 0x6fd1bfcfed5f3679d3ebdec & (1 << (m % 91)):
        return False  # Checks if m modulo operation is not a square number
    if 0xdef9ae771ffe3b9d67dec & (1 << (m % 85)):
        return False  # Checks if m modulo operation is not a square number
    return mlib.sqrtrem(int(x))[1] == 0  # Checks if the remainder of x squared is zero


def invert(x, m):
    """Modular inverse of x modulo m.

    Returns y such that x*y == 1 mod m.

    Uses ``math.pow`` but reproduces the behaviour of ``gmpy2.invert``
    which raises ZeroDivisionError if no inverse exists.
    """
    try:
        return pow(x, -1, m)  # Calculates modular inverse of x modulo m
    except ValueError:
        raise ZeroDivisionError("invert() no inverse exists")


def legendre(x, y):
    """Legendre symbol (x / y).

    Following the implementation of gmpy2,
    the error is raised only when y is an even number.
    """
    if y <= 0 or not y % 2:
        raise ValueError("y should be an odd prime")  # Raises error if y is not an odd prime
    x %= y
    if not x:
        return 0  # Returns 0 if x is zero modulo y
    if pow(x, (y - 1) // 2, y) == 1:
        return 1  # Returns 1 if x is a quadratic residue modulo y
    return -1  # Returns -1 if x is a quadratic non-residue modulo y


def jacobi(x, y):
    """Jacobi symbol (x / y)."""
    if y <= 0 or not y % 2:
        raise ValueError("y should be an odd positive integer")  # Raises error if y is not an odd positive integer
    x %= y
    if not x:
        return int(y == 1)  # Returns 1 if x is 0 and y is 1, otherwise 0
    if y == 1 or x == 1:
        return 1  # Returns 1 if either x or y is 1
    if gcd(x, y) != 1:
        return 0  # Returns 0 if x and y are not coprime
    j = 1
    while x != 0:
        while x % 2 == 0 and x > 0:
            x >>= 1
            if y % 8 in [3, 5]:
                j = -j
        x, y = y, x
        if x % 4 == y % 4 == 3:
            j = -j
        x %= y
    return j  # Returns the Jacobi symbol of x and y


def kronecker(x, y):
    """Kronecker symbol (x / y)."""
    if gcd(x, y) != 1:
        return 0  # Returns 0 if x and y are not coprime
    if y == 0:
        return 1  # Returns 1 if y is 0
    sign = -1 if y < 0 and x < 0 else 1
    y = abs(y)
    s = bit_scan1(y)
    y >>= s
    if s % 2 and x % 8 in [3, 5]:
        sign = -sign
    return sign * jacobi(x, y)  # Returns the Kronecker symbol of x and y


def iroot(y, n):
    # 检查 y 是否为负数，如果是则抛出值错误异常
    if y < 0:
        raise ValueError("y must be nonnegative")
    # 检查 n 是否小于 1，如果是则抛出值错误异常
    if n < 1:
        raise ValueError("n must be positive")
    # 如果 y 为 0 或 1，则直接返回 y 和 True
    if y in (0, 1):
        return y, True
    # 如果 n 等于 1，则直接返回 y 和 True
    if n == 1:
        return y, True
    # 如果 n 等于 2，则使用 mlib.sqrtrem 函数计算平方根和余数，并返回结果
    if n == 2:
        x, rem = mlib.sqrtrem(y)
        return int(x), not rem
    # 如果 n 大于等于 y 的二进制位数，则返回 1 和 False
    if n >= y.bit_length():
        return 1, False

    # 使用 Newton 法的初始估计值。需要小心处理以避免溢出
    try:
        guess = int(y**(1./n) + 0.5)
    except OverflowError:
        # 如果溢出，则计算对数并根据情况调整初始猜测值
        exp = math.log2(y)/n
        if exp > 53:
            shift = int(exp - 53)
            guess = int(2.0**(exp - shift) + 1) << shift
        else:
            guess = int(2.0**exp)
    # 如果猜测值大于 2 的 50 次方，则进行牛顿迭代法
    if guess > 2**50:
        xprev, x = -1, guess
        # 进行牛顿迭代直到收敛
        while 1:
            t = x**(n - 1)
            xprev, x = x, ((n - 1)*x + y//t)//n
            if abs(x - xprev) < 2:
                break
    else:
        x = guess
    
    # 补偿阶段，确保 x 的 n 次方接近 y
    t = x**n
    while t < y:
        x += 1
        t = x**n
    while t > y:
        x -= 1
        t = x**n
    
    # 返回计算结果，以及判断是否恰好等于 y
    return x, t == y
# 检查费马伪素数的函数
def is_fermat_prp(n, a):
    # 如果 a 小于 2，则抛出数值错误
    if a < 2:
        raise ValueError("is_fermat_prp() requires 'a' greater than or equal to 2")
    # 如果 n 小于 1，则抛出数值错误
    if n < 1:
        raise ValueError("is_fermat_prp() requires 'n' be greater than 0")
    # 如果 n 等于 1，则返回 False
    if n == 1:
        return False
    # 如果 n 是偶数，则只有当 n 等于 2 时返回 True
    if n % 2 == 0:
        return n == 2
    # 取模操作，确保 a 在 n 范围内
    a %= n
    # 如果 gcd(n, a) 不等于 1，则抛出数值错误
    if gcd(n, a) != 1:
        raise ValueError("is_fermat_prp() requires gcd(n,a) == 1")
    # 检查费马条件是否满足
    return pow(a, n - 1, n) == 1


# 检查欧拉伪素数的函数
def is_euler_prp(n, a):
    # 如果 a 小于 2，则抛出数值错误
    if a < 2:
        raise ValueError("is_euler_prp() requires 'a' greater than or equal to 2")
    # 如果 n 小于 1，则抛出数值错误
    if n < 1:
        raise ValueError("is_euler_prp() requires 'n' be greater than 0")
    # 如果 n 等于 1，则返回 False
    if n == 1:
        return False
    # 如果 n 是偶数，则只有当 n 等于 2 时返回 True
    if n % 2 == 0:
        return n == 2
    # 取模操作，确保 a 在 n 范围内
    a %= n
    # 如果 gcd(n, a) 不等于 1，则抛出数值错误
    if gcd(n, a) != 1:
        raise ValueError("is_euler_prp() requires gcd(n,a) == 1")
    # 检查欧拉条件是否满足
    return pow(a, n >> 1, n) == jacobi(a, n) % n


# 辅助函数：检查强伪素数的内部函数
def _is_strong_prp(n, a):
    # 计算 s，表示 n-1 的二进制中末尾连续 0 的数量
    s = bit_scan1(n - 1)
    # 计算 a^((n-1)/2) % n
    a = pow(a, n >> s, n)
    # 检查是否符合强伪素数条件
    if a == 1 or a == n - 1:
        return True
    for _ in range(s - 1):
        # 计算 a^2 % n
        a = pow(a, 2, n)
        if a == n - 1:
            return True
        if a == 1:
            return False
    return False


# 检查强伪素数的函数
def is_strong_prp(n, a):
    # 如果 a 小于 2，则抛出数值错误
    if a < 2:
        raise ValueError("is_strong_prp() requires 'a' greater than or equal to 2")
    # 如果 n 小于 1，则抛出数值错误
    if n < 1:
        raise ValueError("is_strong_prp() requires 'n' be greater than 0")
    # 如果 n 等于 1，则返回 False
    if n == 1:
        return False
    # 如果 n 是偶数，则只有当 n 等于 2 时返回 True
    if n % 2 == 0:
        return n == 2
    # 取模操作，确保 a 在 n 范围内
    a %= n
    # 如果 gcd(n, a) 不等于 1，则抛出数值错误
    if gcd(n, a) != 1:
        raise ValueError("is_strong_prp() requires gcd(n,a) == 1")
    # 调用内部函数检查强伪素数条件是否满足
    return _is_strong_prp(n, a)


# Lucas 序列计算函数
def _lucas_sequence(n, P, Q, k):
    r"""Return the modular Lucas sequence (U_k, V_k, Q_k).

    Explanation
    ===========

    Given a Lucas sequence defined by P, Q, returns the kth values for
    U and V, along with Q^k, all modulo n. This is intended for use with
    possibly very large values of n and k, where the combinatorial functions
    would be completely unusable.

    .. math ::
        U_k = \begin{cases}
             0 & \text{if } k = 0\\
             1 & \text{if } k = 1\\
             PU_{k-1} - QU_{k-2} & \text{if } k > 1
        \end{cases}\\
        V_k = \begin{cases}
             2 & \text{if } k = 0\\
             P & \text{if } k = 1\\
             PV_{k-1} - QV_{k-2} & \text{if } k > 1
        \end{cases}

    The modular Lucas sequences are used in numerous places in number theory,
    especially in the Lucas compositeness tests and the various n + 1 proofs.

    Parameters
    ==========

    n : int
        n is an odd number greater than or equal to 3
    P : int
    Q : int
        D determined by D = P**2 - 4*Q is non-zero
    k : int
        k is a nonnegative integer

    Returns
    =======

    U, V, Qk : (int, int, int)
        `(U_k \bmod{n}, V_k \bmod{n}, Q^k \bmod{n})`

    Examples
    ========

    >>> from sympy.external.ntheory import _lucas_sequence
    >>> N = 10**2000 + 4561
    >>> sol = U, V, Qk = _lucas_sequence(N, 3, 1, N//2); sol
    (0, 2, 1)
    # 根据给定的 Lucas 序列参数计算第 k 个 Lucas 数
    def lucas_sequence(k, P, Q, n):
        if k == 0:
            # 初始情况，返回 Lucas 序列的前三个元素
            return (0, 2, 1)
        
        # 计算判别式 D
        D = P**2 - 4 * Q
        U = 1
        V = P
        Qk = Q % n
        
        if Q == 1:
            # 优化，用于额外强的测试
            for b in bin(k)[3:]:
                U = (U * V) % n
                V = (V * V - 2) % n
                if b == "1":
                    U, V = U * P + V, V * P + U * D
                    if U & 1:
                        U += n
                    if V & 1:
                        V += n
                    U, V = U >> 1, V >> 1
        elif P == 1 and Q == -1:
            # 对于50%的 Selfridge 参数的小优化
            for b in bin(k)[3:]:
                U = (U * V) % n
                if Qk == 1:
                    V = (V * V - 2) % n
                else:
                    V = (V * V + 2) % n
                    Qk = 1
                if b == "1":
                    U, V = U + V, U << 1
                    if U & 1:
                        U += n
                    U >>= 1
                    V += U
                    Qk = -1
            Qk %= n
        elif P == 1:
            for b in bin(k)[3:]:
                U = (U * V) % n
                V = (V * V - 2 * Qk) % n
                Qk *= Qk
                if b == "1":
                    U, V = U + V, (Q * U) << 1
                    if U & 1:
                        U += n
                    U >>= 1
                    V = U - V
                    Qk *= Q
                Qk %= n
        else:
            # 一般情况下的计算
            for b in bin(k)[3:]:
                U = (U * V) % n
                V = (V * V - 2 * Qk) % n
                Qk *= Qk
                if b == "1":
                    U, V = U * P + V, V * P + U * D
                    if U & 1:
                        U += n
                    if V & 1:
                        V += n
                    U, V = U >> 1, V >> 1
                    Qk *= Q
                Qk %= n
        
        # 返回计算结果的模 n 后的值
        return (U % n, V % n, Qk)
# 判断一个数是否满足 Fibonacci 式的概率素数 (PRP) 条件，使用给定的参数 p 和 q
def is_fibonacci_prp(n, p, q):
    # 计算判别式 d
    d = p**2 - 4*q
    # 检查判别式 d 是否为零，或者 p 小于等于零，或者 q 不在 [1, -1] 中
    if d == 0 or p <= 0 or q not in [1, -1]:
        raise ValueError("invalid values for p,q in is_fibonacci_prp()")
    # 检查 n 是否小于 1
    if n < 1:
        raise ValueError("is_fibonacci_prp() requires 'n' be greater than 0")
    # 如果 n 等于 1，则返回 False
    if n == 1:
        return False
    # 如果 n 是偶数，返回是否等于 2
    if n % 2 == 0:
        return n == 2
    # 调用 _lucas_sequence 函数来计算 Lucas 序列，检查是否满足条件
    return _lucas_sequence(n, p, q, n)[1] == p % n


# 判断一个数是否满足 Lucas 式的概率素数 (PRP) 条件，使用给定的参数 p 和 q
def is_lucas_prp(n, p, q):
    # 计算判别式 d
    d = p**2 - 4*q
    # 检查判别式 d 是否为零
    if d == 0:
        raise ValueError("invalid values for p,q in is_lucas_prp()")
    # 检查 n 是否小于 1
    if n < 1:
        raise ValueError("is_lucas_prp() requires 'n' be greater than 0")
    # 如果 n 等于 1，则返回 False
    if n == 1:
        return False
    # 如果 n 是偶数，返回是否等于 2
    if n % 2 == 0:
        return n == 2
    # 检查 gcd(n, q*d) 是否在 [1, n] 中，如果不是则抛出异常
    if gcd(n, q*d) not in [1, n]:
        raise ValueError("is_lucas_prp() requires gcd(n,2*q*D) == 1")
    # 调用 _lucas_sequence 函数来计算 Lucas 序列，检查是否满足条件
    return _lucas_sequence(n, p, q, n - jacobi(d, n))[0] == 0


# 使用 Selfridge 方法检查一个数是否满足概率素数 (PRP) 条件
def _is_selfridge_prp(n):
    """Lucas compositeness test with the Selfridge parameters for n.

    Explanation
    ===========

    The Lucas compositeness test checks whether n is a prime number.
    The test can be run with arbitrary parameters ``P`` and ``Q``, which also change the performance of the test.
    So, which parameters are most effective for running the Lucas compositeness test?
    As an algorithm for determining ``P`` and ``Q``, Selfridge proposed method A [1]_ page 1401
    (Since two methods were proposed, referred to simply as A and B in the paper,
    we will refer to one of them as "method A").

    method A fixes ``P = 1``. Then, ``D`` defined by ``D = P**2 - 4Q`` is varied from 5, -7, 9, -11, 13, and so on,
    with the first ``D`` being ``jacobi(D, n) == -1``. Once ``D`` is determined,
    ``Q`` is determined to be ``(P**2 - D)//4``.

    References
    ==========

    .. [1] Robert Baillie, Samuel S. Wagstaff, Lucas Pseudoprimes,
           Math. Comp. Vol 35, Number 152 (1980), pp. 1391-1417,
           https://doi.org/10.1090%2FS0025-5718-1980-0583518-6
           http://mpqs.free.fr/LucasPseudoprimes.pdf

    """
    # 使用 Selfridge 方法 A 来计算适合的 D 值
    for D in range(5, 1_000_000, 2):
        # 如果 D % 4 == 3
        if D & 2: # if D % 4 == 3
            D = -D
        # 计算 Jacobi 符号
        j = jacobi(D, n)
        # 如果 Jacobi 符号为 -1，则进行 Lucas 序列测试
        if j == -1:
            return _lucas_sequence(n, 1, (1-D) // 4, n + 1)[0] == 0
        # 如果 Jacobi 符号为 0，并且 D % n 不为零，则返回 False
        if j == 0 and D % n:
            return False
        # 当 Jacobi 符号为 -1 难以找到时，怀疑是一个平方数
        if D == 13 and is_square(n):
            return False
    # 如果找不到适合的 D 值，则抛出异常
    raise ValueError("appropriate value for D cannot be found in is_selfridge_prp()")


# 判断一个数是否满足 Selfridge 方法的概率素数 (PRP) 条件
def is_selfridge_prp(n):
    # 检查 n 是否小于 1
    if n < 1:
        raise ValueError("is_selfridge_prp() requires 'n' be greater than 0")
    # 如果 n 等于 1，则返回 False
    if n == 1:
        return False
    # 如果 n 是偶数，返回是否等于 2
    if n % 2 == 0:
        return n == 2
    # 调用 _is_selfridge_prp 函数来检查是否满足 Selfridge 方法的条件
    return _is_selfridge_prp(n)


# 判断一个数是否满足 Strong Lucas 式的概率素数 (PRP) 条件，使用给定的参数 p 和 q
def is_strong_lucas_prp(n, p, q):
    # 计算判别式 D
    D = p**2 - 4*q
    # 检查 D 是否为零
    if D == 0:
        raise ValueError("invalid values for p,q in is_strong_lucas_prp()")
    # 检查 n 是否小于 1
    if n < 1:
        raise ValueError("is_selfridge_prp() requires 'n' be greater than 0")
    # 如果 n 等于 1，则返回 False
    if n == 1:
        return False
    # 如果 n 是偶数，则判断 n 是否等于 2，返回结果
    if n % 2 == 0:
        return n == 2
    # 如果 gcd(n, q*D) 不在 [1, n] 中，则抛出数值错误异常，要求 gcd(n, 2*q*D) == 1
    if gcd(n, q*D) not in [1, n]:
        raise ValueError("is_strong_lucas_prp() requires gcd(n,2*q*D) == 1")
    # 计算 Jacobi 符号 j = jacobi(D, n)
    j = jacobi(D, n)
    # 计算 s，其中 s 是 (n - j) 的最低有效位数
    s = bit_scan1(n - j)
    # 调用 _lucas_sequence 函数获取 Lucas 序列的 U, V, Qk 值
    U, V, Qk = _lucas_sequence(n, p, q, (n - j) >> s)
    # 如果 U 或者 V 等于 0，则返回 True
    if U == 0 or V == 0:
        return True
    # 进行 s - 1 次迭代
    for _ in range(s - 1):
        # 更新 V = (V*V - 2*Qk) % n
        V = (V*V - 2*Qk) % n
        # 如果 V 等于 0，则返回 True
        if V == 0:
            return True
        # 更新 Qk = Qk^2 % n
        Qk = pow(Qk, 2, n)
    # 若以上条件都不满足，则返回 False
    return False
# 检查一个数是否为强素数（Strong PRP），使用 Selfridge 方法
def _is_strong_selfridge_prp(n):
    # 在特定范围内选择不同的 D 值进行检查
    for D in range(5, 1_000_000, 2):
        # 如果 D % 4 == 3，则将 D 变为负数
        if D & 2:  # if D % 4 == 3
            D = -D
        # 计算雅可比符号 J(D, n)
        j = jacobi(D, n)
        # 如果 J(D, n) == -1，则执行以下步骤
        if j == -1:
            # 计算 n+1 的最低有效位（最后一个 1 的位置）
            s = bit_scan1(n + 1)
            # 计算 Lucas 序列的参数 U, V, Q_k
            U, V, Qk = _lucas_sequence(n, 1, (1-D) // 4, (n + 1) >> s)
            # 如果 U 或者 V 等于 0，则 n 可能是强素数
            if U == 0 or V == 0:
                return True
            # 通过循环计算 Lucas 序列的后续值，并检查是否为 0
            for _ in range(s - 1):
                V = (V*V - 2*Qk) % n
                if V == 0:
                    return True
                Qk = pow(Qk, 2, n)
            return False
        # 如果 J(D, n) == 0 且 D % n != 0，则 n 不是强素数
        if j == 0 and D % n:
            return False
        # 当难以找到 J(D, n) == -1 时，怀疑 n 是一个平方数
        if D == 13 and is_square(n):
            return False
    # 如果找不到适合的 D 值，则抛出异常
    raise ValueError("appropriate value for D cannot be found in is_strong_selfridge_prp()")


# 检查一个数是否为 Selfridge 强素数（Strong PRP）
def is_strong_selfridge_prp(n):
    # 如果 n 小于 1，则抛出异常
    if n < 1:
        raise ValueError("is_strong_selfridge_prp() requires 'n' be greater than 0")
    # 如果 n 等于 1，则返回 False
    if n == 1:
        return False
    # 如果 n 是偶数，则 n 是素数当且仅当 n == 2
    if n % 2 == 0:
        return n == 2
    # 否则，调用 _is_strong_selfridge_prp 函数进行检查
    return _is_strong_selfridge_prp(n)


# 检查一个数是否为 BPSW 强素数（Strong PRP）
def is_bpsw_prp(n):
    # 如果 n 小于 1，则抛出异常
    if n < 1:
        raise ValueError("is_bpsw_prp() requires 'n' be greater than 0")
    # 如果 n 等于 1，则返回 False
    if n == 1:
        return False
    # 如果 n 是偶数，则 n 是素数当且仅当 n == 2
    if n % 2 == 0:
        return n == 2
    # 否则，同时满足 _is_strong_prp(n, 2) 和 _is_selfridge_prp(n) 的条件
    return _is_strong_prp(n, 2) and _is_selfridge_prp(n)


# 检查一个数是否为 Strong BPSW 强素数（Strong PRP）
def is_strong_bpsw_prp(n):
    # 如果 n 小于 1，则抛出异常
    if n < 1:
        raise ValueError("is_strong_bpsw_prp() requires 'n' be greater than 0")
    # 如果 n 等于 1，则返回 False
    if n == 1:
        return False
    # 如果 n 是偶数，则 n 是素数当且仅当 n == 2
    if n % 2 == 0:
        return n == 2
    # 否则，同时满足 _is_strong_prp(n, 2) 和 _is_strong_selfridge_prp(n) 的条件
    return _is_strong_prp(n, 2) and _is_strong_selfridge_prp(n)
```