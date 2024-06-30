# `D:\src\scipysrc\sympy\sympy\ntheory\ecm.py`

```
# 导入数学模块中的对数函数
from math import log

# 导入随机整数生成函数
from sympy.core.random import _randint

# 导入外部库 gmpy 中的最大公约数、求逆和平方根函数
from sympy.external.gmpy import gcd, invert, sqrt

# 导入实用工具中的整数化函数
from sympy.utilities.misc import as_int

# 导入当前目录下的 generate 模块中的筛法和质数范围生成函数
from .generate import sieve, primerange

# 导入当前目录下的 primetest 模块中的质数判断函数
from .primetest import isprime


#----------------------------------------------------------------------------#
#                                                                            #
#                   Lenstra's Elliptic Curve Factorization                   #
#                                                                            #
#----------------------------------------------------------------------------#


class Point:
    """Montgomery form of Points in an elliptic curve.
    In this form, the addition and doubling of points
    does not need any y-coordinate information thus
    decreasing the number of operations.
    Using Montgomery form we try to perform point addition
    and doubling in least amount of multiplications.

    The elliptic curve used here is of the form
    (E : b*y**2*z = x**3 + a*x**2*z + x*z**2).
    The a_24 parameter is equal to (a + 2)/4.

    References
    ==========

    .. [1] Kris Gaj, Soonhak Kwon, Patrick Baier, Paul Kohlbrenner, Hoang Le, Mohammed Khaleeluddin, Ramakrishna Bachimanchi,
           Implementing the Elliptic Curve Method of Factoring in Reconfigurable Hardware,
           Cryptographic Hardware and Embedded Systems - CHES 2006 (2006), pp. 119-133,
           https://doi.org/10.1007/11894063_10
           https://www.hyperelliptic.org/tanja/SHARCS/talks06/Gaj.pdf

    """

    def __init__(self, x_cord, z_cord, a_24, mod):
        """
        Initial parameters for the Point class.

        Parameters
        ==========

        x_cord : X coordinate of the Point
        z_cord : Z coordinate of the Point
        a_24 : Parameter of the elliptic curve in Montgomery form
        mod : modulus
        """
        self.x_cord = x_cord  # 设置点的 X 坐标
        self.z_cord = z_cord  # 设置点的 Z 坐标
        self.a_24 = a_24  # 设置椭圆曲线的参数 a_24（Montgomery form）
        self.mod = mod  # 设置模数

    def __eq__(self, other):
        """Two points are equal if X/Z of both points are equal
        """
        if self.a_24 != other.a_24 or self.mod != other.mod:
            return False
        return self.x_cord * other.z_cord % self.mod ==\
            other.x_cord * self.z_cord % self.mod
    def add(self, Q, diff):
        """
        Add two points self and Q where diff = self - Q. Moreover the assumption
        is self.x_cord*Q.x_cord*(self.x_cord - Q.x_cord) != 0. This algorithm
        requires 6 multiplications. Here the difference between the points
        is already known and using this algorithm speeds up the addition
        by reducing the number of multiplication required. Also in the
        mont_ladder algorithm is constructed in a way so that the difference
        between intermediate points is always equal to the initial point.
        So, we always know what the difference between the point is.

        Parameters
        ==========

        Q : point on the curve in Montgomery form
        diff : self - Q

        Examples
        ========

        >>> from sympy.ntheory.ecm import Point
        >>> p1 = Point(11, 16, 7, 29)
        >>> p2 = Point(13, 10, 7, 29)
        >>> p3 = p2.add(p1, p1)
        >>> p3.x_cord
        23
        >>> p3.z_cord
        17
        """
        # Calculate u and v based on Montgomery addition formulas
        u = (self.x_cord - self.z_cord)*(Q.x_cord + Q.z_cord)
        v = (self.x_cord + self.z_cord)*(Q.x_cord - Q.z_cord)
        add, subt = u + v, u - v
        # Compute new x_cord and z_cord using Montgomery reduction
        x_cord = diff.z_cord * add * add % self.mod
        z_cord = diff.x_cord * subt * subt % self.mod
        # Return a new Point object representing the result of the addition
        return Point(x_cord, z_cord, self.a_24, self.mod)

    def double(self):
        """
        Doubles a point in an elliptic curve in Montgomery form.
        This algorithm requires 5 multiplications.

        Examples
        ========

        >>> from sympy.ntheory.ecm import Point
        >>> p1 = Point(11, 16, 7, 29)
        >>> p2 = p1.double()
        >>> p2.x_cord
        13
        >>> p2.z_cord
        10
        """
        # Compute u and v based on Montgomery doubling formulas
        u = pow(self.x_cord + self.z_cord, 2, self.mod)
        v = pow(self.x_cord - self.z_cord, 2, self.mod)
        diff = u - v
        # Calculate new x_cord and z_cord using Montgomery reduction
        x_cord = u * v % self.mod
        z_cord = diff * (v + self.a_24 * diff) % self.mod
        # Return a new Point object representing the result of the doubling
        return Point(x_cord, z_cord, self.a_24, self.mod)

    def mont_ladder(self, k):
        """
        Scalar multiplication of a point in Montgomery form
        using Montgomery Ladder Algorithm.
        A total of 11 multiplications are required in each step of this
        algorithm.

        Parameters
        ==========

        k : The positive integer multiplier

        Examples
        ========

        >>> from sympy.ntheory.ecm import Point
        >>> p1 = Point(11, 16, 7, 29)
        >>> p3 = p1.mont_ladder(3)
        >>> p3.x_cord
        23
        >>> p3.z_cord
        17
        """
        # Initialize points Q and R for Montgomery ladder
        Q = self
        R = self.double()
        # Iterate through the binary representation of k
        for i in bin(k)[3:]:
            if i == '1':
                # If the current bit is 1, perform addition and doubling
                Q = R.add(Q, self)
                R = R.double()
            else:
                # If the current bit is 0, perform doubling and addition
                R = Q.add(R, self)
                Q = Q.double()
        # Return the final result after scalar multiplication
        return Q
    """Returns one factor of n using
    Lenstra's 2 Stage Elliptic curve Factorization
    with Suyama's Parameterization. Here Montgomery
    arithmetic is used for fast computation of addition
    and doubling of points in elliptic curve.

    Explanation
    ===========

    This ECM method considers elliptic curves in Montgomery
    form (E : b*y**2*z = x**3 + a*x**2*z + x*z**2) and involves
    elliptic curve operations (mod N), where the elements in
    Z are reduced (mod N). Since N is not a prime, E over FF(N)
    is not really an elliptic curve but we can still do point additions
    and doubling as if FF(N) was a field.

    Stage 1 : The basic algorithm involves taking a random point (P) on an
    elliptic curve in FF(N). The compute k*P using Montgomery ladder algorithm.
    Let q be an unknown factor of N. Then the order of the curve E, |E(FF(q))|,
    might be a smooth number that divides k. Then we have k = l * |E(FF(q))|
    for some l. For any point belonging to the curve E, |E(FF(q))|*P = O,
    hence k*P = l*|E(FF(q))|*P. Thus kP.z_cord = 0 (mod q), and the unknown
    factor of N (q) can be recovered by taking gcd(kP.z_cord, N).

    Stage 2 : This is a continuation of Stage 1 if k*P != O. The idea utilizes
    the fact that even if kP != 0, the value of k might miss just one large
    prime divisor of |E(FF(q))|. In this case we only need to compute the
    scalar multiplication by p to get p*k*P = O. Here a second bound B2
    restricts the size of possible values of p.

    Parameters
    ==========

    n : Number to be Factored
    B1 : Stage 1 Bound. Must be an even number.
    B2 : Stage 2 Bound. Must be an even number.
    max_curve : Maximum number of curves generated

    Returns
    =======

    integer | None : ``n`` (if it is prime) else a non-trivial divisor of ``n``. ``None`` if not found

    References
    ==========

    .. [1] Carl Pomerance, Richard Crandall, Prime Numbers: A Computational Perspective,
           2nd Edition (2005), page 344, ISBN:978-0387252827
    """
    # Generate a random integer based on the given seed
    randint = _randint(seed)
    # Check if n is prime; if true, return n as it cannot be factored further
    if isprime(n):
        return n

    # Determine the value of D for Stage 1 based on B1 and B2
    D = min(sqrt(B2), B1 // 2 - 1)
    # Extend the sieve with the calculated value of D
    sieve.extend(D)
    # Initialize beta and S lists with zeros of size D
    beta = [0] * D
    S = [0] * D
    # Initialize k to 1; calculate the product of primes up to B1
    k = 1
    for p in primerange(2, B1 + 1):
        k *= pow(p, int(log(B1, p)))

    # Pre-calculate the deltas_list for Stage 2
    deltas_list = []
    for r in range(B1 + 2*D, B2 + 2*D, 4*D):
        # Compute deltas based on prime numbers within the range
        deltas = {abs(q - r) >> 1 for q in primerange(r - 2*D, r + 2*D)}
        deltas_list.append(list(deltas))
    for _ in range(max_curve):
        # 在每个曲线尝试中，使用 Suyama's 参数化生成参数
        sigma = randint(6, n - 1)
        u = (sigma**2 - 5) % n
        v = (4*sigma) % n
        u_3 = pow(u, 3, n)

        try:
            # 计算椭圆曲线参数 a24，使用 elliptic curve y**2 = x**3 + a*x**2 + x
            # 这里 a = pow(v - u, 3, n)*(3*u + v)*invert(4*u_3*v, n) - 2
            # 但是为了方便，使用 a24 = (a + 2)*invert(4, n) 进行计算
            a24 = pow(v - u, 3, n)*(3*u + v)*invert(16*u_3*v, n) % n
        except ZeroDivisionError:
            # 处理除零错误，如果 invert(16*u_3*v, n) 不存在 (即 g != 1)
            g = gcd(2*u_3*v, n)
            # 如果 g = n，则尝试另一条曲线
            if g == n:
                continue
            return g

        # 创建椭圆曲线上的点 Q
        Q = Point(u_3, pow(v, 3, n), a24, n)
        # 使用 Montgomery Ladder 方法计算 Q 的倍数 k
        Q = Q.mont_ladder(k)
        g = gcd(Q.z_cord, n)

        # 第一阶段因子分解
        if g != 1 and g != n:
            return g
        # 第一阶段失败，Q.z = 0，尝试另一条曲线
        elif g == n:
            continue

        # 第二阶段 - 改进的标准继续法
        S[0] = Q
        Q2 = Q.double()
        S[1] = Q2.add(Q, Q)
        beta[0] = (S[0].x_cord*S[0].z_cord) % n
        beta[1] = (S[1].x_cord*S[1].z_cord) % n
        for d in range(2, D):
            S[d] = S[d - 1].add(Q2, S[d - 2])
            beta[d] = (S[d].x_cord*S[d].z_cord) % n
        # 即，S[i] = Q.mont_ladder(2*i + 1)

        g = 1
        W = Q.mont_ladder(4*D)
        T = Q.mont_ladder(B1 - 2*D)
        R = Q.mont_ladder(B1 + 2*D)
        for deltas in deltas_list:
            # R = Q.mont_ladder(r)，其中 r 在范围 (B1 + 2*D, B2 + 2*D, 4*D) 内
            alpha = (R.x_cord*R.z_cord) % n
            for delta in deltas:
                # 计算 f = R.x_cord * S[delta].z_cord - S[delta].x_cord * R.z_cord
                f = (R.x_cord - S[delta].x_cord)*\
                    (R.z_cord + S[delta].z_cord) - alpha + beta[delta]
                g = (g*f) % n
            T, R = R, R.add(W, T)
        g = gcd(n, g)

        # 第二阶段找到因子
        if g != 1 and g != n:
            return g
# 使用 Lenstra 的椭圆曲线方法进行因式分解
def ecm(n, B1=10000, B2=100000, max_curve=200, seed=1234):
    """Performs factorization using Lenstra's Elliptic curve method.

    This function repeatedly calls ``_ecm_one_factor`` to compute the factors
    of n. First all the small factors are taken out using trial division.
    Then ``_ecm_one_factor`` is used to compute one factor at a time.

    Parameters
    ==========

    n : Number to be Factored
    B1 : Stage 1 Bound. Must be an even number.
    B2 : Stage 2 Bound. Must be an even number.
    max_curve : Maximum number of curves generated
    seed : Initialize pseudorandom generator

    Examples
    ========

    >>> from sympy.ntheory import ecm
    >>> ecm(25645121643901801)
    {5394769, 4753701529}
    >>> ecm(9804659461513846513)
    {4641991, 2112166839943}
    """
    n = as_int(n)  # 将 n 转换为整数类型
    if B1 % 2 != 0 or B2 % 2 != 0:
        raise ValueError("both bounds must be even")  # 如果 B1 或 B2 不是偶数，抛出异常
    _factors = set()  # 初始化一个集合用于存放因子
    for prime in sieve.primerange(1, 100000):  # 遍历从1到100000的质数
        if n % prime == 0:  # 如果 n 能被当前质数整除
            _factors.add(prime)  # 将该质数添加到因子集合中
            while(n % prime == 0):
                n //= prime  # 反复除以该质数，直到不能整除为止
    while(n > 1):  # 当 n 大于1时，继续分解
        factor = _ecm_one_factor(n, B1, B2, max_curve, seed)  # 调用 _ecm_one_factor 计算一个因子
        if factor is None:
            raise ValueError("Increase the bounds")  # 如果无法找到因子，抛出异常
        _factors.add(factor)  # 将找到的因子添加到集合中
        n //= factor  # 对 n 进行因子除法

    factors = set()  # 初始化结果因子集合
    for factor in _factors:  # 遍历所有已找到的因子
        if isprime(factor):  # 如果因子是质数
            factors.add(factor)  # 将其添加到结果集合中
            continue
        factors |= ecm(factor)  # 否则递归调用 ecm 函数继续分解该因子
    return factors  # 返回最终找到的所有质因子集合
```