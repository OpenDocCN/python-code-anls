# `D:\src\scipysrc\sympy\sympy\ntheory\factor_.py`

```
"""
Integer factorization
"""

# 导入必要的模块
from collections import defaultdict
import math

# 导入 SymPy 相关模块和函数
from sympy.core.containers import Dict
from sympy.core.mul import Mul
from sympy.core.numbers import Rational, Integer
from sympy.core.intfunc import num_digits
from sympy.core.power import Pow
from sympy.core.random import _randint
from sympy.core.singleton import S
from sympy.external.gmpy import (SYMPY_INTS, gcd, sqrt as isqrt,
                                 sqrtrem, iroot, bit_scan1, remove)
# 导入本地模块和函数
from .primetest import isprime, MERSENNE_PRIME_EXPONENTS, is_mersenne_prime
from .generate import sieve, primerange, nextprime
from .digits import digits
from sympy.utilities.decorator import deprecated
from sympy.utilities.iterables import flatten
from sympy.utilities.misc import as_int, filldedent
from .ecm import _ecm_one_factor

# smoothness 函数：计算整数 n 的平滑度和平滑度指数
def smoothness(n):
    """
    Return the B-smooth and B-power smooth values of n.

    The smoothness of n is the largest prime factor of n; the power-
    smoothness is the largest divisor raised to its multiplicity.

    Examples
    ========

    >>> from sympy.ntheory.factor_ import smoothness
    >>> smoothness(2**7*3**2)
    (3, 128)
    >>> smoothness(2**4*13)
    (13, 16)
    >>> smoothness(2)
    (2, 2)

    See Also
    ========

    factorint, smoothness_p
    """

    # 如果 n 等于 1，返回 (1, 1)，避免特殊情况的麻烦
    if n == 1:
        return (1, 1)
    
    # 使用 factorint 函数计算 n 的因子分解结果
    facs = factorint(n)
    # 返回最大的素因子和最大的幂次乘积
    return max(facs), max(m**facs[m] for m in facs)


# smoothness_p 函数：计算整数 n 的平滑度和平滑度指数列表
def smoothness_p(n, m=-1, power=0, visual=None):
    """
    Return a list of [m, (p, (M, sm(p + m), psm(p + m)))...]
    where:

    1. p**M is the base-p divisor of n
    2. sm(p + m) is the smoothness of p + m (m = -1 by default)
    3. psm(p + m) is the power smoothness of p + m

    The list is sorted according to smoothness (default) or by power smoothness
    if power=1.

    The smoothness of the numbers to the left (m = -1) or right (m = 1) of a
    factor govern the results that are obtained from the p +/- 1 type factoring
    methods.

        >>> from sympy.ntheory.factor_ import smoothness_p, factorint
        >>> smoothness_p(10431, m=1)
        (1, [(3, (2, 2, 4)), (19, (1, 5, 5)), (61, (1, 31, 31))])
        >>> smoothness_p(10431)
        (-1, [(3, (2, 2, 2)), (19, (1, 3, 9)), (61, (1, 5, 5))])
        >>> smoothness_p(10431, power=1)
        (-1, [(3, (2, 2, 2)), (61, (1, 5, 5)), (19, (1, 3, 9))])

    If visual=True then an annotated string will be returned:

        >>> print(smoothness_p(21477639576571, visual=1))
        p**i=4410317**1 has p-1 B=1787, B-pow=1787
        p**i=4869863**1 has p-1 B=2434931, B-pow=2434931

    This string can also be generated directly from a factorization dictionary
    and vice versa:

        >>> factorint(17*9)
        {3: 2, 17: 1}
        >>> smoothness_p(_)
        'p**i=3**2 has p-1 B=2, B-pow=2\\np**i=17**1 has p-1 B=2, B-pow=16'
        >>> smoothness_p(_)
        {3: 2, 17: 1}
    """

    # 函数的实现略
    pass
    """
    The function handles various input types (`n`) and a visual flag (`visual`), generating formatted output based on conditions.

    Parameters:
    ----------
    n : str or tuple or int or Mul
        Input parameter determining behavior of the function.
    visual : bool or None
        Flag controlling visual output format. Must be True, False, or None.

    Returns:
    -------
    str or dict or tuple
        Formatted output based on input type and visual flag.

    Notes:
    ------
    - `visual` must be True, False, or None (stored as None if not).
    - If `n` is a string, it may split and parse data into a dictionary.
    - If `n` is not a tuple, it calculates factors using `factorint`.
    - Depending on conditions, it returns various formatted outputs.

    See Also:
    --------
    factorint, smoothness
    """

    # Check if visual is 1 or 0 and convert to boolean
    if visual in (1, 0):
        visual = bool(visual)
    # If visual is not explicitly True or False, set it to None
    elif visual not in (True, False):
        visual = None

    # If n is a string
    if isinstance(n, str):
        # Return the string itself if visual is True
        if visual:
            return n
        d = {}
        # Parse lines in the string, extracting key-value pairs
        for li in n.splitlines():
            k, v = [int(i) for i in li.split('has')[0].split('=')[1].split('**')]
            d[k] = v
        # If visual is not explicitly True or False, return the parsed dictionary
        if visual is not True and visual is not False:
            return d
        # Otherwise, return the result of smoothness_p function with visual set to False
        return smoothness_p(d, visual=False)
    # If n is not a tuple, calculate its factors without visual output
    elif not isinstance(n, tuple):
        facs = factorint(n, visual=False)

    # Set k based on the value of power flag
    if power:
        k = -1
    else:
        k = 1
    # Determine the return value based on the type of n
    if isinstance(n, tuple):
        rv = n
    else:
        rv = (m, sorted([(f,
                          tuple([M] + list(smoothness(f + m))))
                         for f, M in list(facs.items())],
                        key=lambda x: (x[1][k], x[0])))

    # Return rv if visual is explicitly False or if n is of type int or Mul
    if visual is False or (visual is not True) and (type(n) in [int, Mul]):
        return rv

    # Generate formatted lines based on rv[1] and return as a joined string
    lines = []
    for dat in rv[1]:
        dat = flatten(dat)
        dat.insert(2, m)
        lines.append('p**i=%i**%i has p%+i B=%i, B-pow=%i' % tuple(dat))
    return '\n'.join(lines)
# 计算给定质数 p 对整数 n 的最大幂次数，使得 p 的这个幂次能整除 n
def multiplicity(p, n):
    try:
        # 尝试将 p 和 n 转换为整数
        p, n = as_int(p), as_int(n)
    except ValueError:
        # 处理值错误异常，如果 p 或 n 是整数或有理数，调用相应的处理方法
        from sympy.functions.combinatorial.factorials import factorial
        if all(isinstance(i, (SYMPY_INTS, Rational)) for i in (p, n)):
            # 如果 p 和 n 都是整数或有理数，将它们转换为有理数
            p = Rational(p)
            n = Rational(n)
            if p.q == 1:
                if n.p == 1:
                    # 当 p 是整数且 n 是整数时，返回 p 对 n/q 的负的幂次
                    return -multiplicity(p.p, n.q)
                # 否则返回 p 对 n/p 的幂次减去 p 对 n/q 的幂次
                return multiplicity(p.p, n.p) - multiplicity(p.p, n.q)
            elif p.p == 1:
                # 当 p 是整数且 p == 1 时，返回 p/q 对 n/q 的幂次
                return multiplicity(p.q, n.q)
            else:
                # 返回 p.p 对 n.p 的幂次和 p.q 对 n.q 的幂次的最小值
                like = min(
                    multiplicity(p.p, n.p),
                    multiplicity(p.q, n.q))
                # 返回 p.q 对 n.p 和 p.p 对 n.q 的幂次的最小值
                cross = min(
                    multiplicity(p.q, n.p),
                    multiplicity(p.p, n.q))
                return like - cross
        elif (isinstance(p, (SYMPY_INTS, Integer)) and
                isinstance(n, factorial) and
                isinstance(n.args[0], Integer) and
                n.args[0] >= 0):
            # 如果 p 是整数，n 是阶乘对象且 n 的参数大于等于 0，调用多项式阶乘中的 multiplicity_in_factorial 方法
            return multiplicity_in_factorial(p, n.args[0])
        # 抛出值错误异常，显示期望整数或分数，但得到了 p 和 n
        raise ValueError('expecting ints or fractions, got %s and %s' % (p, n))

    if n == 0:
        # 如果 n 等于 0，抛出值错误异常，显示没有这样的整数存在：多重性为 n 的定义未定义
        raise ValueError('no such integer exists: multiplicity of %s is not-defined' %(n))
    # 返回从 n 中移除 p 后的结果
    return remove(n, p)[1]


# 返回一个最大整数 m，使得 p**m 能整除 n!，而不计算 n 的阶乘
def multiplicity_in_factorial(p, n):
    """
    Parameters
    ==========

    p : Integer
        正整数
    n : Integer
        非负整数

    Examples
    ========

    >>> from sympy.ntheory import multiplicity_in_factorial
    >>> from sympy import factorial

    >>> multiplicity_in_factorial(2, 3)
    1

    An instructive use of this is to tell how many trailing zeros
    a given factorial has. For example, there are 6 in 25!:

    >>> factorial(25)
    15511210043330985984000000
    >>> multiplicity_in_factorial(10, 25)
    6

    For large factorials, it is much faster/feasible to use
    this function rather than computing the actual factorial:

    >>> multiplicity_in_factorial(factorial(25), 2**100)
    52818775009509558395695966887
    """
    # 返回 p 的最大幂次数 m，使得 p**m 能整除 n!
    pass  # 这里是一个占位符，实际函数体未提供
    # 将输入参数 p 和 n 转换为整数类型
    p, n = as_int(p), as_int(n)

    # 如果 p 小于等于 0，则抛出值错误异常
    if p <= 0:
        raise ValueError('expecting positive integer got %s' % p )

    # 如果 n 小于 0，则抛出值错误异常
    if n < 0:
        raise ValueError('expecting non-negative integer got %s' % n )

    # 使用 defaultdict 创建一个空字典 f，值的默认类型为 int
    # 将 p 分解质因数后，对于每个质因数的幂次，保留最大的底数
    f = defaultdict(int)
    for k, v in factorint(p).items():
        f[v] = max(k, f[v])
    
    # 计算返回值，这里是一个生成器表达式
    # 计算每个幂次 v 对应的 (n + k - sum(digits(n, k)))//(k - 1)//v 的最小值
    return min((n + k - sum(digits(n, k)))//(k - 1)//v for v, k in f.items())
def _perfect_power(n, next_p=2):
    """ Return integers ``(b, e)`` such that ``n == b**e`` if ``n`` is a unique
    perfect power with ``e > 1``, else ``False`` (e.g. 1 is not a perfect power).

    Explanation
    ===========

    This is a low-level helper for ``perfect_power``, for internal use.

    Parameters
    ==========

    n : int
        assume that n is a nonnegative integer
    next_p : int
        Assume that n has no factor less than next_p.
        i.e., all(n % p for p in range(2, next_p)) is True

    Examples
    ========
    >>> from sympy.ntheory.factor_ import _perfect_power
    >>> _perfect_power(16)
    (2, 4)
    >>> _perfect_power(17)
    False

    """
    # 如果 n 小于等于 3，则不可能是完全幂
    if n <= 3:
        return False

    factors = {}  # 用于存储 n 的质因数及其指数的字典
    g = 0  # 初始 gcd 值为 0
    multi = 1  # 初始指数倍增系数为 1

    def done(n, factors, g, multi):
        # 计算所有质因数的最大公约数
        g = gcd(g, multi)
        if g == 1:
            return False
        factors[n] = multi
        # 返回 b 和 e，其中 b 是所有质因数的积的部分幂次根，e 是所有质因数指数的最大公约数
        return math.prod(p**(e//g) for p, e in factors.items()), g

    # 如果 n 较小，则只进行试除法因式分解更快
    if n <= 1_000_000:
        # 使用小型因式分解函数尝试将 n 分解为质因数的乘积
        n = _factorint_small(factors, n, 1_000, 1_000, next_p)[0]
        if n > 1:
            return False
        # 计算所有质因数的最大公约数
        g = gcd(*factors.values())
        if g == 1:
            return False
        # 返回 b 和 e，其中 b 是所有质因数的积的部分幂次根，e 是所有质因数指数的最大公约数
        return math.prod(p**(e//g) for p, e in factors.items()), g

    # 如果 next_p 小于 3，则试除法因式分解中包含 2 的部分处理
    if next_p < 3:
        # 通过位运算找到 n 的二进制表示中最低位的 1 的位置，作为因子 2 的指数
        g = bit_scan1(n)
        if g:
            if g == 1:
                return False
            # 将 n 右移 g 位，相当于除以 2 的 g 次方
            n >>= g
            factors[2] = g  # 记录因子 2 的指数
            if n == 1:
                return 2, g
            else:
                # 如果 `m**g`，则找到了完全幂
                # 否则，特别是如果 g 是素数，就没有完全幂的可能性
                m, _exact = iroot(n, g)
                if _exact:
                    return 2*m, g
                elif isprime(g):
                    return False
        next_p = 3

    # 如果 n 是平方数，继续检查是否是完全幂
    while n & 7 == 1:  # 等效于 n % 8 == 1，检查 n 的二进制表示的最后三位是否为 001
        m, _exact = iroot(n, 2)
        if _exact:
            n = m
            multi <<= 1  # 指数倍增
        else:
            break

    # 如果 n 小于 next_p 的立方，进行最后的因式分解
    if n < next_p**3:
        return done(n, factors, g, multi)

    # 试除法因式分解
    # 由于指数的最大值为 `log_{next_p}(n)`，可以通过试除法因式分解来减少检查的指数数量
    # `tf_max` 的值需要更多的考虑
    tf_max = n.bit_length()//27 + 24
    # 如果下一个素数小于 tf_max，则进入循环
    if next_p < tf_max:
        # 遍历从 next_p 到 tf_max 的素数
        for p in primerange(next_p, tf_max):
            # 调用 remove 函数，获取移除 n 中的素数 p 后的结果 m 和 t
            m, t = remove(n, p)
            # 如果 t 非零
            if t:
                # 更新 n 为 m，同时 t 增大为 multi 的倍数
                n = m
                t *= multi
                # 计算 t 和 g 的最大公约数
                _g = gcd(g, t)
                # 如果最大公约数为 1，则找到了新的互质因子，返回 False
                if _g == 1:
                    return False
                # 记录素数 p 对应的因子 t
                factors[p] = t
                # 如果 n 等于 1，则返回因子分解结果
                if n == 1:
                    return math.prod(p**(e//_g)
                                        for p, e in factors.items()), _g
                # 如果 g 为 0 或者 _g 比 g 小，则更新 g
                elif g == 0 or _g < g: # 如果更新了 g
                    g = _g
                    # 计算 n 的 multi 次根，并检查是否为整数
                    m, _exact = iroot(n**multi, g)
                    if _exact:
                        # 返回根据新的 g 计算得到的结果
                        return m * math.prod(p**(e//g)
                                            for p, e in factors.items()), g
                    elif isprime(g):
                        return False
        # 更新下一个素数为 tf_max
        next_p = tf_max
    
    # 如果 n 小于 next_p 的立方
    if n < next_p**3:
        return done(n, factors, g, multi)

    # 检查 iroot 函数
    if g:
        # 如果 g 非零，则指数是 g 的一个因子
        # 2 可以省略，因为已经检查过
        prime_iter = sorted(factorint(g >> bit_scan1(g)).keys())
    else:
        # 指数的最大可能值为 log_{next_p}(n)
        # 为了补偿计算误差，加上 2
        prime_iter = primerange(3, int(math.log(n, next_p)) + 2)
    # 计算 n 的对数
    logn = math.log2(n)
    # 计算直接计算的阈值
    threshold = logn / 40 # 直接计算的阈值
    # 遍历素数迭代器 prime_iter
    for p in prime_iter:
        # 如果阈值小于素数 p
        if threshold < p:
            # 如果 p 很大，则直接找到 p 次方根而不使用 `iroot`
            while True:
                b = pow(2, logn / p)
                rb = int(b + 0.5)
                # 如果计算的根 rb 满足条件，则更新 n 和 multi
                if abs(rb - b) < 0.01 and rb**p == n:
                    n = rb
                    multi *= p
                    logn = math.log2(n)
                else:
                    break
        else:
            # 否则使用 `iroot` 函数找到 n 的 p 次方根
            while True:
                m, _exact = iroot(n, p)
                if _exact:
                    n = m
                    multi *= p
                    logn = math.log2(n)
                else:
                    break
        # 如果 n 小于 next_p 的 (p + 2) 次方，则跳出循环
        if n < next_p**(p + 2):
            break
    # 返回完成的结果
    return done(n, factors, g, multi)
# 定义函数 perfect_power，用于确定整数 n 是否为唯一的完全幂（即能表示为 b**e，其中 e > 1），否则返回 False
def perfect_power(n, candidates=None, big=True, factor=True):
    """
    Return ``(b, e)`` such that ``n`` == ``b**e`` if ``n`` is a unique
    perfect power with ``e > 1``, else ``False`` (e.g. 1 is not a
    perfect power). A ValueError is raised if ``n`` is not Rational.

    By default, the base is recursively decomposed and the exponents
    collected so the largest possible ``e`` is sought. If ``big=False``
    then the smallest possible ``e`` (thus prime) will be chosen.

    If ``factor=True`` then simultaneous factorization of ``n`` is
    attempted since finding a factor indicates the only possible root
    for ``n``. This is True by default since only a few small factors will
    be tested in the course of searching for the perfect power.

    The use of ``candidates`` is primarily for internal use; if provided,
    False will be returned if ``n`` cannot be written as a power with one
    of the candidates as an exponent and factoring (beyond testing for
    a factor of 2) will not be attempted.

    Examples
    ========

    >>> from sympy import perfect_power, Rational
    >>> perfect_power(16)
    (2, 4)
    >>> perfect_power(16, big=False)
    (4, 2)

    Negative numbers can only have odd perfect powers:

    >>> perfect_power(-4)
    False
    >>> perfect_power(-8)
    (-2, 3)

    Rationals are also recognized:

    >>> perfect_power(Rational(1, 2)**3)
    (1/2, 3)
    >>> perfect_power(Rational(-3, 2)**3)
    (-3/2, 3)

    Notes
    =====

    To know whether an integer is a perfect power of 2 use

        >>> is2pow = lambda n: bool(n and not n & (n - 1))
        >>> [(i, is2pow(i)) for i in range(5)]
        [(0, False), (1, True), (2, True), (3, False), (4, True)]

    It is not necessary to provide ``candidates``. When provided
    it will be assumed that they are ints. The first one that is
    larger than the computed maximum possible exponent will signal
    failure for the routine.

        >>> perfect_power(3**8, [9])
        False
        >>> perfect_power(3**8, [2, 4, 8])
        (3, 8)
        >>> perfect_power(3**8, [4, 8], big=False)
        (9, 4)

    See Also
    ========
    sympy.core.intfunc.integer_nthroot
    sympy.ntheory.primetest.is_square
    """
    
    # 若 n 是有理数且不是整数，则分子和分母分别处理
    if isinstance(n, Rational) and not n.is_Integer:
        p, q = n.as_numer_denom()
        # 若分子为 1，则递归检查分母
        if p is S.One:
            pp = perfect_power(q)
            if pp:
                pp = (n.func(1, pp[0]), pp[1])
        else:
            # 否则，先检查分子，再尝试检查分母
            pp = perfect_power(p)
            if pp:
                num, e = pp
                pq = perfect_power(q, [e])
                if pq:
                    den, _ = pq
                    pp = n.func(num, den), e
        return pp
    
    # 将 n 转换为整数
    n = as_int(n)
    # 如果 n 小于 0，则检查其绝对值是否为完全幂
    if n < 0:
        pp = perfect_power(-n)
        if pp:
            b, e = pp
            # 如果指数 e 是奇数，则返回负数的 b 和 e
            if e % 2:
                return -b, e
        return False
    
    # 若没有提供 candidates 并且 big 为 True，则调用内部函数 _perfect_power 进行处理
    if candidates is None and big:
        return _perfect_power(n)
    # 如果 n 小于等于 3，则直接返回 False，因为对于 0 和 1 没有唯一的指数，2 和 3 的指数为 1
    if n <= 3:
        # 没有唯一的指数适用于 0 和 1
        # 2 和 3 的指数为 1
        return False
    
    # 计算 n 的对数
    logn = math.log2(n)
    
    # 设置最大可能的指数值，只检查比这个值小的候选值
    max_possible = int(logn) + 2
    
    # 检查 n 是否为平方数，因为平方数的末尾数字不能是 2、3、7、8
    not_square = n % 10 in [2, 3, 7, 8]
    
    # 计算最小可能的候选值
    min_possible = 2 + not_square
    
    # 如果没有提供候选值，则生成范围在 min_possible 和 max_possible 之间的质数候选值
    if not candidates:
        candidates = primerange(min_possible, max_possible)
    else:
        # 如果提供了候选值，则从中筛选出范围在 min_possible 和 max_possible 之间的候选值，并排序
        candidates = sorted([i for i in candidates if min_possible <= i < max_possible])
        
        # 如果 n 是偶数，计算其二进制表示中最低位的 1 的位置
        if n % 2 == 0:
            e = bit_scan1(n)
            # 仅保留能整除 e 的候选值
            candidates = [i for i in candidates if e % i == 0]
        
        # 如果 big 标志为真，反转候选值的顺序
        if big:
            candidates = reversed(candidates)
        
        # 遍历候选值
        for e in candidates:
            # 计算 n 的 e 次根
            r, ok = iroot(n, e)
            if ok:
                # 如果计算成功，则返回结果 r 和指数 e
                return int(r), e
        # 如果没有找到合适的 e，则返回 False
        return False

    # 定义内部函数 _factors，生成可能的因子
    def _factors():
        rv = 2 + n % 2
        while True:
            yield rv
            rv = nextprime(rv)

    # 遍历可能的因子和之前计算得到的候选值
    for fac, e in zip(_factors(), candidates):
        # 检查是否存在某个因子 fac 能整除 n
        if factor and n % fac == 0:
            # 计算去除因子 fac 后的剩余部分的指数
            e = remove(n, fac)[1]
            
            # 如果指数 e 等于 1，表示 fac 是 n 的一个唯一因子，返回 False
            if e == 1:
                return False
            
            # 尝试计算 n 的 e 次根
            r, exact = iroot(n, e)
            
            # 如果计算成功，则继续处理
            if not exact:
                # 如果 fac 是一个因子，那么 e 是根 n 的最大可能值
                # 如果 n = fac**e*m 能写成一个完美的幂，则检查 m 是否能写成 r**E 的形式
                # 其中 gcd(e, E) != 1，这样 n = (fac**(e//E)*r)**E
                m = n // fac**e
                rE = perfect_power(m, candidates=divisors(e, generator=True))
                if not rE:
                    return False
                else:
                    r, E = rE
                    r, e = fac**(e // E) * r, E
            
            # 如果不是 big 标志，则继续处理
            if not big:
                e0 = primefactors(e)
                if e0[0] != e:
                    r, e = r**(e // e0[0]), e0[0]
            # 返回计算结果 r 和指数 e
            return int(r), e
        
        # 排除明显不可能的候选值
        if logn / e < 40:
            b = 2.0**(logn / e)
            if abs(int(b + 0.5) - b) > 0.01:
                continue
        
        # 再次尝试计算 n 的 e 次根
        r, exact = iroot(n, e)
        if exact:
            # 如果计算成功，则继续处理
            if big:
                # 计算 r 的完美幂
                m = perfect_power(r, big=big, factor=factor)
                if m:
                    r, e = m[0], e * m[1]
            # 返回计算结果 r 和指数 e
            return int(r), e
    
    # 如果所有条件都不满足，则返回 False
    return False
# 使用 Pollard rho 方法尝试提取 n 的一个非平凡因子
def pollard_rho(n, s=2, a=1, retries=5, seed=1234, max_steps=None, F=None):
    # 如果没有提供 F 函数，则使用默认的函数 x**2 + a
    # F 函数生成伪随机的 x 值，并将其替换为 F(x)
    r"""
    Use Pollard's rho method to try to extract a nontrivial factor
    of ``n``. The returned factor may be a composite number. If no
    factor is found, ``None`` is returned.

    The algorithm generates pseudo-random values of x with a generator
    function, replacing x with F(x). If F is not supplied then the
    function x**2 + ``a`` is used. The first value supplied to F(x) is ``s``.
    Upon failure (if ``retries`` is > 0) a new ``a`` and ``s`` will be
    supplied; the ``a`` will be ignored if F was supplied.

    The sequence of numbers generated by such functions generally have a
    a lead-up to some number and then loop around back to that number and
    begin to repeat the sequence, e.g. 1, 2, 3, 4, 5, 3, 4, 5 -- this leader
    and loop look a bit like the Greek letter rho, and thus the name, 'rho'.

    For a given function, very different leader-loop values can be obtained
    so it is a good idea to allow for retries:

    >>> from sympy.ntheory.generate import cycle_length
    >>> n = 16843009
    >>> F = lambda x:(2048*pow(x, 2, n) + 32767) % n
    >>> for s in range(5):
    ...     print('loop length = %4i; leader length = %3i' % next(cycle_length(F, s)))
    ...
    loop length = 2489; leader length =  43
    loop length =   78; leader length = 121
    loop length = 1482; leader length = 100
    loop length = 1482; leader length = 286
    loop length = 1482; leader length = 101

    Here is an explicit example where there is a three element leadup to
    a sequence of 3 numbers (11, 14, 4) that then repeat:

    >>> x=2
    >>> for i in range(9):
    ...     print(x)
    ...     x=(x**2+12)%17
    ...
    2
    16
    13
    11
    14
    4
    11
    14
    4
    >>> next(cycle_length(lambda x: (x**2+12)%17, 2))
    (3, 3)
    >>> list(cycle_length(lambda x: (x**2+12)%17, 2, values=True))
    [2, 16, 13, 11, 14, 4]

    Instead of checking the differences of all generated values for a gcd
    with n, only the kth and 2*kth numbers are checked, e.g. 1st and 2nd,
    2nd and 4th, 3rd and 6th until it has been detected that the loop has been
    traversed. Loops may be many thousands of steps long before rho finds a
    factor or reports failure. If ``max_steps`` is specified, the iteration
    is cancelled with a failure after the specified number of steps.

    Examples
    ========

    >>> from sympy import pollard_rho
    >>> n=16843009
    >>> F=lambda x:(2048*pow(x,2,n) + 32767) % n
    >>> pollard_rho(n, F=F)
    257

    Use the default setting with a bad value of ``a`` and no retries:

    >>> pollard_rho(n, a=n-2, retries=0)

    If retries is > 0 then perhaps the problem will correct itself when
    new values are generated for a:

    >>> pollard_rho(n, a=n-2, retries=1)
    257

    References
    ==========
    """
    # 如果 max_steps 超过了指定步数，则取消迭代并返回失败
    # 如果 F 函数不存在，重新生成新的 a 和 s 值，并重新尝试
    # 返回找到的因子，如果失败则返回 None
    .. [1] Richard Crandall & Carl Pomerance (2005), "Prime Numbers:
           A Computational Perspective", Springer, 2nd edition, 229-231

    """
    # 将输入参数 n 转换为整数类型
    n = int(n)
    # 如果 n 小于 5，抛出数值错误异常，要求 n 必须大于 4
    if n < 5:
        raise ValueError('pollard_rho should receive n > 4')
    
    # 使用 seed 和 retries 生成一个随机整数生成器
    randint = _randint(seed + retries)
    
    # 初始化 V 为输入参数 s
    V = s
    
    # 执行 retries + 1 次迭代
    for i in range(retries + 1):
        # 将 U 设为当前的 V
        U = V
        
        # 如果 F 未定义，则定义 F 为 lambda 函数，计算 x^2 + a % n
        if not F:
            F = lambda x: (pow(x, 2, n) + a) % n
        
        # 初始化 j 为 0，进入内循环
        j = 0
        while 1:
            # 如果设定了最大步数并且 j 超过了最大步数，则跳出内循环
            if max_steps and (j > max_steps):
                break
            
            # 增加 j 的计数
            j += 1
            
            # 计算 U 的下一个值
            U = F(U)
            # 计算 V 的下一个值，V 是 U 的两倍步长
            V = F(F(V))  # V is 2x further along than U
            
            # 计算 U-V 和 n 的最大公约数
            g = gcd(U - V, n)
            
            # 如果最大公约数为 1，则继续内循环
            if g == 1:
                continue
            
            # 如果最大公约数等于 n，则退出循环
            if g == n:
                break
            
            # 返回最大公约数 g 的整数值
            return int(g)
        
        # 重新生成 V 为 0 到 n-1 之间的随机整数
        V = randint(0, n - 1)
        # 生成 a 为 1 到 n-3 之间的随机整数，确保 a % n 不为 0 或 -2
        a = randint(1, n - 3)  # for x**2 + a, a%n should not be 0 or -2
        
        # 将 F 设为 None，以便下一轮重新定义
        F = None
    
    # 若未找到因子，则返回 None
    return None
# 使用 Pollard's p-1 方法尝试提取给定整数 n 的非平凡因子
# 返回一个除数（可能是复合的）或者 None

def pollard_pm1(n, B=10, a=2, retries=0, seed=1234):
    """
    Use Pollard's p-1 method to try to extract a nontrivial factor
    of ``n``. Either a divisor (perhaps composite) or ``None`` is returned.

    The value of ``a`` is the base that is used in the test gcd(a**M - 1, n).
    The default is 2.  If ``retries`` > 0 then if no factor is found after the
    first attempt, a new ``a`` will be generated randomly (using the ``seed``)
    and the process repeated.

    Note: the value of M is lcm(1..B) = reduce(ilcm, range(2, B + 1)).

    A search is made for factors next to even numbers having a power smoothness
    less than ``B``. Choosing a larger B increases the likelihood of finding a
    larger factor but takes longer. Whether a factor of n is found or not
    depends on ``a`` and the power smoothness of the even number just less than
    the factor p (hence the name p - 1).

    Although some discussion of what constitutes a good ``a`` some
    descriptions are hard to interpret. At the modular.math site referenced
    below it is stated that if gcd(a**M - 1, n) = N then a**M % q**r is 1
    for every prime power divisor of N. But consider the following:

        >>> from sympy.ntheory.factor_ import smoothness_p, pollard_pm1
        >>> n=257*1009
        >>> smoothness_p(n)
        (-1, [(257, (1, 2, 256)), (1009, (1, 7, 16))])

    So we should (and can) find a root with B=16:

        >>> pollard_pm1(n, B=16, a=3)
        1009

    If we attempt to increase B to 256 we find that it does not work:

        >>> pollard_pm1(n, B=256)
        >>>

    But if the value of ``a`` is changed we find that only multiples of
    257 work, e.g.:

        >>> pollard_pm1(n, B=256, a=257)
        1009

    Checking different ``a`` values shows that all the ones that did not
    work had a gcd value not equal to ``n`` but equal to one of the
    factors:

        >>> from sympy import ilcm, igcd, factorint, Pow
        >>> M = 1
        >>> for i in range(2, 256):
        ...     M = ilcm(M, i)
        ...
        >>> set([igcd(pow(a, M, n) - 1, n) for a in range(2, 256) if
        ...      igcd(pow(a, M, n) - 1, n) != n])
        {1009}

    But does aM % d for every divisor of n give 1?

        >>> aM = pow(255, M, n)
        >>> [(d, aM%Pow(*d.args)) for d in factorint(n, visual=True).args]
        [(257**1, 1), (1009**1, 1)]

    No, only one of them. So perhaps the principle is that a root will
    be found for a given value of B provided that:

    1) the power smoothness of the p - 1 value next to the root
       does not exceed B
    2) a**M % p != 1 for any of the divisors of n.

    By trying more than one ``a`` it is possible that one of them
    will yield a factor.

    Examples
    ========

    With the default smoothness bound, this number cannot be cracked:

        >>> from sympy.ntheory import pollard_pm1
        >>> pollard_pm1(21477639576571)
    """
    # 计算 M 的值，M 是 1 到 B 的最小公倍数
    M = 1
    for i in range(2, B + 1):
        M = ilcm(M, i)

    # 重试机制，如果 retries 大于 0，允许使用不同的随机数种子生成新的 a 值
    for _ in range(retries + 1):
        # 计算 a 的 M 次幂对 n 取模后减一的最大公约数
        g = igcd(pow(a, M, n) - 1, n)
        # 如果最大公约数不等于 n，则返回找到的因子
        if 1 < g < n:
            return g
        # 如果最大公约数等于 n，则尝试新的随机生成的 a
        a = rand(seed)

    # 如果没有找到因子，则返回 None
    return None
    # 将输入的 n 转换为整数类型
    n = int(n)
    # 如果 n 小于 4 或者 B 小于 3，则抛出数值错误异常
    if n < 4 or B < 3:
        raise ValueError('pollard_pm1 should receive n > 3 and B > 2')
    # 根据给定的种子值和 B 计算一个随机整数
    randint = _randint(seed + B)

    # 计算 a 的 lcm(1, 2, 3, ..., B) 模 n 的值，其中 B > 2
    # 这看起来很奇怪，但这是正确的：质数范围是 [2, B]
    # 直到循环完成后，答案才会是正确的。
    # 尝试重试次数加一次（包括第一次）
    for i in range(retries + 1):
        # 备份当前的 a 到 aM
        aM = a
        # 使用筛法获取所有小于等于 B 的素数，并对每个素数进行处理
        for p in sieve.primerange(2, B + 1):
            # 计算 p 的最大指数 e，使得 p^e <= B
            e = int(math.log(B, p))
            # 计算 aM^(p^e) % n，利用模幂运算
            aM = pow(aM, pow(p, e), n)
        # 计算 aM - 1 与 n 的最大公约数
        g = gcd(aM - 1, n)
        # 如果 g 在 1 和 n 之间，则找到了 n 的一个非平凡因子
        if 1 < g < n:
            # 返回找到的因子 g
            return int(g)

        # 获取一个新的随机数 a：
        # 由于指数 lcm(1..B) 是偶数，如果允许 'a' 等于 'n-1'，则 (n - 1)**even % n 将为 1，
        # 这将导致 g 为 0 和 1 也会给出零，因此我们设置范围为 [2, n-2]。
        # 一些参考资料表明 'a' 应该与 n 互质，但任意值都能检测因子。
        a = randint(2, n - 2)
def _trial(factors, n, candidates, verbose=False):
    """
    Helper function for integer factorization. Trial factors `n`
    against all integers given in the sequence `candidates`
    and updates the dict `factors` in-place. Returns the reduced
    value of `n` and a flag indicating whether any factors were found.
    """
    # If verbose mode is enabled, capture current keys in factors
    if verbose:
        factors0 = list(factors.keys())
    
    # Record the initial number of factors in factors dictionary
    nfactors = len(factors)
    
    # Iterate through each candidate divisor d
    for d in candidates:
        # Check if n is divisible by d
        if n % d == 0:
            # Perform division and update factors dictionary with multiplicity
            n, m = remove(n // d, d)
            factors[d] = m + 1
    
    # If verbose mode is enabled, print messages for newly found factors
    if verbose:
        # Print messages for each newly added factor
        for k in sorted(set(factors).difference(set(factors0))):
            print(factor_msg % (k, factors[k]))
    
    # Return the reduced value of n and whether any new factors were found
    return int(n), len(factors) != nfactors


def _check_termination(factors, n, limit, use_trial, use_rho, use_pm1,
                       verbose, next_p):
    """
    Helper function for integer factorization. Checks if `n`
    is a prime or a perfect power, and in those cases updates the factorization.
    """
    # If verbose mode is enabled, print a message indicating termination check
    if verbose:
        print('Check for termination')
    
    # Check if n equals 1, if so, finalize factorization and return True
    if n == 1:
        if verbose:
            print(complete_msg)
        return True
    
    # Check if n is less than the square of next_p or if n is prime
    if n < next_p**2 or isprime(n):
        # Record n as a factor with multiplicity 1 and print completion message if verbose
        factors[int(n)] = 1
        if verbose:
            print(complete_msg)
        return True
    
    # Check for perfect powers and update factorization accordingly
    p = _perfect_power(n, next_p)
    if not p:
        return False
    base, exp = p
    
    # Further check conditions for base in perfect power and update factors
    if base < next_p**2 or isprime(base):
        factors[base] = exp
    else:
        # Factor base using external function and update factors dictionary
        facs = factorint(base, limit, use_trial, use_rho, use_pm1,
                         verbose=False)
        for b, e in facs.items():
            if verbose:
                print(factor_msg % (b, e))
            # int() can be removed when https://github.com/flintlib/python-flint/issues/92 is resolved
            factors[b] = int(exp*e)
    
    # Print completion message if verbose mode is enabled
    if verbose:
        print(complete_msg)
    
    # Return True indicating factorization is complete
    return True


trial_int_msg = "Trial division with ints [%i ... %i] and fail_max=%i"
trial_msg = "Trial division with primes [%i ... %i]"
rho_msg = "Pollard's rho with retries %i, max_steps %i and seed %i"
pm1_msg = "Pollard's p-1 with smoothness bound %i and seed %i"
ecm_msg = "Elliptic Curve with B1 bound %i, B2 bound %i, num_curves %i"
factor_msg = '\t%i ** %i'  # Message format for printing factors
fermat_msg = 'Close factors satisying Fermat condition found.'
complete_msg = 'Factorization is complete.'


def _factorint_small(factors, n, limit, fail_max, next_p=2):
    """
    Return the value of n and either a 0 (indicating that factorization up
    to the limit was complete) or else the next near-prime that would have
    been tested.

    Factoring stops if there are fail_max unsuccessful tests in a row.

    If factors of n were found they will be in the factors dictionary as
    {factor: multiplicity} and the returned value of n will have had those
    factors removed. The factors dictionary is modified in-place.

    """
    def done(n, d):
        """返回 n 和 d，如果 sqrt(n) 尚未达到，则返回 n, d；否则返回 n, 0，表示因子分解完成。"""
        if d*d <= n:
            return n, d
        return n, 0

    # 计算 limit 的平方
    limit2 = limit**2
    # 计算阈值的平方，取 n 和 limit2 中较小的一个作为阈值的平方
    threshold2 = min(n, limit2)

    if next_p < 3:
        # 如果 next_p 小于 3，即为 2 或更小的数
        if not n & 1:
            # 如果 n 是偶数，找出其二进制表示中最低位的 1 的位置
            m = bit_scan1(n)
            factors[2] = m
            n >>= m  # 将 n 右移 m 位，相当于除以 2 的 m 次方
            threshold2 = min(n, limit2)
        next_p = 3
        if threshold2 < 9:  # 如果阈值的平方小于 9（即 next_p**2 = 9）
            return done(n, next_p)

    if next_p < 5:
        # 如果 next_p 小于 5，即为 3 或更小的数
        if not n % 3:
            # 如果 n 是 3 的倍数，不断除以 3 直到不能整除为止
            n //= 3
            m = 1
            while not n % 3:
                n //= 3
                m += 1
                if m == 20:
                    n, mm = remove(n, 3)
                    m += mm
                    break
            factors[3] = m
            threshold2 = min(n, limit2)
        next_p = 5
        if threshold2 < 25:  # 如果阈值的平方小于 25（即 next_p**2 = 25）
            return done(n, next_p)

    # 由于检查顺序，从 min_p = 6k+5 开始，会导致无用的检查。
    # 我们希望计算 next_p += [-1, -2, 3, 2, 1, 0][next_p % 6]
    p6 = next_p % 6
    next_p += (-1 if p6 < 2 else 5) - p6

    fails = 0
    while fails < fail_max:
        if n % next_p:
            fails += 1
        else:
            # 如果 n 能被 next_p 整除，不断除以 next_p 直到不能整除为止
            n //= next_p
            m = 1
            while not n % next_p:
                n //= next_p
                m += 1
                if m == 20:
                    n, mm = remove(n, next_p)
                    m += mm
                    break
            factors[next_p] = m
            fails = 0
            threshold2 = min(n, limit2)
        next_p += 2
        if threshold2 < next_p**2:
            return done(n, next_p)

        if n % next_p:
            fails += 1
        else:
            # 如果 n 能被 next_p 整除，不断除以 next_p 直到不能整除为止
            n //= next_p
            m = 1
            while not n % next_p:
                n //= next_p
                m += 1
                if m == 20:
                    n, mm = remove(n, next_p)
                    m += mm
                    break
            factors[next_p] = m
            fails = 0
            threshold2 = min(n, limit2)
        next_p += 4
        if threshold2 < next_p**2:
            return done(n, next_p)
    return done(n, next_p)
# 定义一个函数 factorint，用于分解正整数 n 的质因数，并返回一个字典，键为质因数，值为其重数
def factorint(n, limit=None, use_trial=True, use_rho=True, use_pm1=True,
              use_ecm=True, verbose=False, visual=None, multiple=False):
    r"""
    给定一个正整数 ``n``，``factorint(n)`` 返回一个包含 ``n`` 的质因数作为键及其重数作为值的字典。例如：

    >>> from sympy.ntheory import factorint
    >>> factorint(2000)    # 2000 = (2**4) * (5**3)
    {2: 4, 5: 3}
    >>> factorint(65537)   # 这个数是质数
    {65537: 1}

    对于小于 2 的输入，factorint 的行为如下：

        - ``factorint(1)`` 返回空的因式分解， ``{}``
        - ``factorint(0)`` 返回 ``{0:1}``
        - ``factorint(-n)`` 在因子中添加 ``-1:1``，然后对 ``n`` 进行因式分解

    部分因式分解：

    如果指定了 ``limit``（大于 3），则在执行试除法直到（包括）限制（或者执行相应数量的 rho/p-1 步骤）后停止搜索。如果有一个大数，并且只想找到小因子（如果有的话），这很有用。注意，设置限制并不阻止更大的因子早期被发现；它只是意味着最大的因子可能是复合的。由于检查是否完美幂比较便宜，因此不管限制设置如何，都会执行此操作。

    例如，这个数有两个小因子和一个难以简化的巨大半质数因子：

    >>> from sympy.ntheory import isprime
    >>> a = 1407633717262338957430697921446883
    >>> f = factorint(a, limit=10000)
    >>> f == {991: 1, int(202916782076162456022877024859): 1, 7: 1}
    True
    >>> isprime(max(f))
    False

    这个数有一个小因子和一个基数大于限制的残余完美幂：

    >>> factorint(3*101**7, limit=5)
    {3: 1, 101: 7}

    因子列表：

    如果将 ``multiple`` 设置为 ``True``，则返回包含质因数及其重数的列表。

    >>> factorint(24, multiple=True)
    [2, 2, 2, 3]

    可视化因子分解：

    如果将 ``visual`` 设置为 ``True``，则返回整数的可视化因子分解。例如：

    >>> from sympy import pprint
    >>> pprint(factorint(4200, visual=True))
     3  1  2  1
    2 *3 *5 *7

    注意，这通过在 Mul 和 Pow 中使用 evaluate=False 标志来实现。如果您对带有 evaluate=False 的表达式进行其他操作，它可能会计算。因此，如果要对因子执行操作，则应仅在需要可视化时使用 visual 选项，并在不需要可视化时使用 visual=False 返回的普通字典。

    您可以通过将它们重新发送到 factorint 轻松地在两种形式之间切换：

    >>> from sympy import Mul
    >>> regular = factorint(1764); regular
    {2: 2, 3: 2, 7: 2}
    >>> pprint(factorint(regular))
     2  2  2
    2 *3 *7

    >>> visual = factorint(1764, visual=True); pprint(visual)
     2  2  2
    """
    2 *3 *7
    >>> print(factorint(visual))
    {2: 2, 3: 2, 7: 2}

    If you want to send a number to be factored in a partially factored form
    you can do so with a dictionary or unevaluated expression:

    >>> factorint(factorint({4: 2, 12: 3})) # twice to toggle to dict form
    {2: 10, 3: 3}
    >>> factorint(Mul(4, 12, evaluate=False))
    {2: 4, 3: 1}

    The table of the output logic is:

        ====== ====== ======= =======
                       Visual
        ------ ----------------------
        Input  True   False   other
        ====== ====== ======= =======
        dict    mul    dict    mul
        n       mul    dict    dict
        mul     mul    dict    dict
        ====== ====== ======= =======

    Notes
    =====

    Algorithm:

    The function switches between multiple algorithms. Trial division
    quickly finds small factors (of the order 1-5 digits), and finds
    all large factors if given enough time. The Pollard rho and p-1
    algorithms are used to find large factors ahead of time; they
    will often find factors of the order of 10 digits within a few
    seconds:

    >>> factors = factorint(12345678910111213141516)
    >>> for base, exp in sorted(factors.items()):
    ...     print('%s %s' % (base, exp))
    ...
    2 2
    2507191691 1
    1231026625769 1

    Any of these methods can optionally be disabled with the following
    boolean parameters:

        - ``use_trial``: Toggle use of trial division
        - ``use_rho``: Toggle use of Pollard's rho method
        - ``use_pm1``: Toggle use of Pollard's p-1 method

    ``factorint`` also periodically checks if the remaining part is
    a prime number or a perfect power, and in those cases stops.

    For unevaluated factorial, it uses Legendre's formula (theorem).

    If ``verbose`` is set to ``True``, detailed progress is printed.

    See Also
    ========

    smoothness, smoothness_p, divisors

    """
    # 如果 n 是字典类型，则转换为标准字典类型
    if isinstance(n, Dict):
        n = dict(n)
    # 如果 multiple 为真，则进行多重因数分解
    if multiple:
        # 调用 factorint 函数进行因数分解，返回因数列表
        fac = factorint(n, limit=limit, use_trial=use_trial,
                           use_rho=use_rho, use_pm1=use_pm1,
                           verbose=verbose, visual=False, multiple=False)
        # 组装因数列表
        factorlist = sum(([p] * fac[p] if fac[p] > 0 else [S.One/p]*(-fac[p])
                               for p in sorted(fac)), [])
        return factorlist

    # 初始化一个空字典来存储因数分解结果
    factordict = {}
    # 如果 visual 为真且 n 不是 Mul 类型或字典类型，则进行因数分解
    if visual and not isinstance(n, (Mul, dict)):
        factordict = factorint(n, limit=limit, use_trial=use_trial,
                               use_rho=use_rho, use_pm1=use_pm1,
                               verbose=verbose, visual=False)
    # 如果 n 是 Mul 类型，则转换为字典形式存储因数
    elif isinstance(n, Mul):
        factordict = {int(k): int(v) for k, v in
            n.as_powers_dict().items()}
    # 如果 n 是字典类型，则直接使用该字典作为因数分解结果
    elif isinstance(n, dict):
        factordict = n
    # 如果 factordict 不为空且 n 是 Mul 类型或者 dict 类型，则进入条件判断
    if factordict and isinstance(n, (Mul, dict)):
        # 检查 factordict 中的每个键是否为素数，如果是则跳过
        for key in list(factordict.keys()):
            if isprime(key):
                continue
            # 将非素数键对应的值从 factordict 中取出，并进行更深层次的因子分解
            e = factordict.pop(key)
            d = factorint(key, limit=limit, use_trial=use_trial, use_rho=use_rho,
                          use_pm1=use_pm1, verbose=verbose, visual=False)
            # 将新得到的因子分解结果与原来的 factordict 合并
            for k, v in d.items():
                if k in factordict:
                    factordict[k] += v*e
                else:
                    factordict[k] = v*e

    # 如果 visual 为真，或者 n 是字典或 Mul 类型且 visual 不为 True 且不为 False，则进入条件判断
    if visual or (type(n) is dict and
                  visual is not True and
                  visual is not False):
        # 如果 factordict 是空字典，则返回 S.One
        if factordict == {}:
            return S.One
        # 如果 factordict 中包含 -1，则将其移除并初始化 args 列表
        if -1 in factordict:
            factordict.pop(-1)
            args = [S.NegativeOne]
        else:
            args = []
        # 将 factordict 中的键值对转换为 Pow 对象，按键排序后作为参数传递给 Mul 构造函数
        args.extend([Pow(*i, evaluate=False)
                     for i in sorted(factordict.items())])
        return Mul(*args, evaluate=False)
    # 如果 n 是字典或 Mul 类型，则直接返回 factordict
    elif isinstance(n, (dict, Mul)):
        return factordict

    # 断言至少有一种方法（use_trial, use_rho, use_pm1, use_ecm）被启用
    assert use_trial or use_rho or use_pm1 or use_ecm

    # 导入阶乘函数
    from sympy.functions.combinatorial.factorials import factorial
    # 如果 n 是 factorial 类型
    if isinstance(n, factorial):
        # 获取 n 的参数并转换为整数
        x = as_int(n.args[0])
        # 如果 x 大于等于 20
        if x >= 20:
            factors = {}
            m = 2  # 用于初始化下面的 if 条件
            # 使用素数筛选器计算 x 的所有质因子的幂次和
            for p in sieve.primerange(2, x + 1):
                if m > 1:
                    m, q = 0, x // p
                    while q != 0:
                        m += q
                        q //= p
                factors[p] = m
            # 如果 factors 不为空且 verbose 为真，则打印每个质因子及其幂次和
            if factors and verbose:
                for k in sorted(factors):
                    print(factor_msg % (k, factors[k]))
            # 如果 verbose 为真，则打印完整消息
            if verbose:
                print(complete_msg)
            return factors
        else:
            # 如果 x 小于 20，直接使用查找表计算阶乘的质因子
            n = n.func(x)

    # 将 n 转换为整数
    n = as_int(n)
    # 如果 limit 存在，则转换为整数并关闭 use_ecm
    if limit:
        limit = int(limit)
        use_ecm = False

    # 特殊情况处理
    # 如果 n 小于 0，计算其负数的因子分解，并将 -1 添加到结果中
    if n < 0:
        factors = factorint(
            -n, limit=limit, use_trial=use_trial, use_rho=use_rho,
            use_pm1=use_pm1, verbose=verbose, visual=False)
        factors[-1] = 1
        return factors

    # 如果 limit 存在且小于 2
    if limit and limit < 2:
        # 如果 n 等于 1，返回空字典；否则返回 {n: 1}
        if n == 1:
            return {}
        return {n: 1}
    # 如果 n 小于 10
    elif n < 10:
        # 返回预定义的列表中的第 n 个元素，该列表中包含了小于 10 的整数的因子分解结果
        return [{0: 1}, {}, {2: 1}, {3: 1}, {2: 2}, {5: 1},
                {2: 1, 3: 1}, {7: 1}, {2: 3}, {3: 2}][n]

    # 初始化 factors 字典，用于存储因子分解结果
    factors = {}

    # 进行简单的因子分解
    if verbose:
        # 打印正在进行因子分解的数值 n
        sn = str(n)
        if len(sn) > 50:
            print('Factoring %s' % sn[:5] + \
                  '..(%i other digits)..' % (len(sn) - 10) + sn[-5:])
        else:
            print('Factoring', n)

    # 进行小因子的初步因子分解
    # 我们希望保证没有小质因子存在，
    # 这一步骤用于确保没有小质因子，
    # 设置一个较小的初始值，用于简单试除法
    small = 2**15
    # 失败的最大尝试次数
    fail_max = 600
    # 如果指定了限制条件，则选择限制值和初始值中的较小者作为 small 的值
    small = min(small, limit or small)
    # 如果 verbose 为 True，则打印试除法的信息
    if verbose:
        print(trial_int_msg % (2, small, fail_max))
    # 调用函数 _factorint_small 进行简单试除法的因式分解
    n, next_p = _factorint_small(factors, n, small, fail_max)
    # 如果 factors 不为空且 verbose 为 True，则打印因子的信息
    if factors and verbose:
        for k in sorted(factors):
            print(factor_msg % (k, factors[k]))
    # 如果 next_p 为 0，则说明简单试除法已完成
    if next_p == 0:
        # 如果 n 大于 1，则将 n 添加为因子
        if n > 1:
            factors[int(n)] = 1
        # 如果 verbose 为 True，则打印完成的信息
        if verbose:
            print(complete_msg)
        return factors
    # 如果限制条件 limit 存在且 next_p 超过了 limit
    if limit and next_p > limit:
        # 如果 verbose 为 True，则打印超出限制的信息
        if verbose:
            print('Exceeded limit:', limit)
        # 检查是否满足终止条件，如果满足则返回 factors
        if _check_termination(factors, n, limit, use_trial,
                              use_rho, use_pm1, verbose, next_p):
            return factors
        # 如果 n 大于 1，则将 n 添加为因子
        if n > 1:
            factors[int(n)] = 1
        return factors
    # 检查是否满足终止条件，如果满足则返回 factors
    if _check_termination(factors, n, limit, use_trial,
                          use_rho, use_pm1, verbose, next_p):
        return factors

    # 继续使用更高级的因式分解方法

    # 计算 n 的平方根
    sqrt_n = isqrt(n)
    # 设定 a 的初始值
    a = sqrt_n + 1
    # 如果 n % 4 == 1，则 a 必须是奇数，以使 a**2 - n 是一个平方数
    if (n % 4 == 1) ^ (a & 1):
        a += 1
    # 计算 a 的平方
    a2 = a**2
    # 计算 b**2 = a**2 - n
    b2 = a2 - n
    # 进行三次费马测试
    for _ in range(3):
        # 计算 b 和 fermat，其中 b 是平方根，fermat 是是否通过费马测试
        b, fermat = sqrtrem(b2)
        # 如果费马测试失败，且 verbose 为 True，则打印费马测试失败的信息
        if not fermat:
            if verbose:
                print(fermat_msg)
            # 对 r = a - b 和 r = a + b 进行因式分解
            for r in [a - b, a + b]:
                facs = factorint(r, limit=limit, use_trial=use_trial,
                                 use_rho=use_rho, use_pm1=use_pm1,
                                 verbose=verbose)
                # 将分解得到的因子添加到 factors 中
                for k, v in facs.items():
                    factors[k] = factors.get(k, 0) + v
            # 如果 verbose 为 True，则打印完成的信息
            if verbose:
                print(complete_msg)
            return factors
        # 更新 b2，相当于 (a + 2)**2 - n
        b2 += (a + 1) << 2  # equiv to (a + 2)**2 - n
        # 更新 a，增加 2
        a += 2

    # 设定简单试除法的限制范围
    low, high = next_p, 2*next_p

    # 增加 1 以确保在 primerange 调用中达到限制
    _limit = (limit or sqrt_n) + 1
    # 初始化迭代次数
    iteration = 0
    # 设定 B1, B2 和 num_curves 的初始值
    B1 = 10000
    B2 = 100*B1
    num_curves = 50
    # 进入无限循环，直到手动退出
    while(1):
        # 如果设置了详细输出，打印当前 ECM 计算的信息
        if verbose:
            print(ecm_msg % (B1, B2, num_curves))
        # 调用 _ecm_one_factor 函数，尝试找到 n 的一个因子
        factor = _ecm_one_factor(n, B1, B2, num_curves, seed=B1)
        # 如果找到了因子
        if factor:
            # 如果找到的因子小于下一个素数的平方或者本身是素数
            if factor < next_p**2 or isprime(factor):
                # 将该因子加入到因子列表中
                ps = [factor]
            else:
                # 对找到的非素数因子进行分解为素因子
                ps = factorint(factor, limit=limit,
                               use_trial=use_trial,
                               use_rho=use_rho,
                               use_pm1=use_pm1,
                               use_ecm=use_ecm,
                               verbose=verbose)
            # 使用找到的素因子更新 n 的分解结果
            n, _ = _trial(factors, n, ps, verbose=False)
            # 检查是否满足终止条件，如果满足则返回当前的因子列表
            if _check_termination(factors, n, limit, use_trial,
                                  use_rho, use_pm1, verbose, next_p):
                return factors
        # 更新 B1, B2 和 num_curves 的值，为下一轮 ECM 计算做准备
        B1 *= 5
        B2 = 100 * B1
        num_curves *= 4
# 给定有理数 `rat`，计算其素因子分解，返回一个字典，键为素因子，值为其对应的幂次数。
def factorrat(rat, limit=None, use_trial=True, use_rho=True, use_pm1=True,
              verbose=False, visual=None, multiple=False):
    r"""
    给定有理数 `r`，`factorrat(r)` 返回一个字典，包含 `r` 的素因子作为键，它们的重数作为值。例如：

    >>> from sympy import factorrat, S
    >>> factorrat(S(8)/9)    # 8/9 = (2**3) * (3**-2)
    {2: 3, 3: -2}
    >>> factorrat(S(-1)/987)    # -1/789 = -1 * (3**-1) * (7**-1) * (47**-1)
    {-1: 1, 3: -1, 7: -1, 47: -1}

    详细解释和示例请参阅 `factorint` 的文档字符串，了解以下关键字的用法：

        - `limit`：试除法的整数限制
        - `use_trial`：启用试除法
        - `use_rho`：启用波拉德 rho 方法
        - `use_pm1`：启用波拉德 p-1 方法
        - `verbose`：详细输出进度信息
        - `multiple`：返回因子列表或字典的开关
        - `visual`：输出的乘积形式的开关
    """
    if multiple:
        # 如果 multiple 为 True，则返回因子列表
        fac = factorrat(rat, limit=limit, use_trial=use_trial,
                  use_rho=use_rho, use_pm1=use_pm1,
                  verbose=verbose, visual=False, multiple=False)
        # 根据因子的幂次数构建因子列表
        factorlist = sum(([p] * fac[p] if fac[p] > 0 else [S.One/p]*(-fac[p])
                               for p, _ in sorted(fac.items(),
                                                        key=lambda elem: elem[0]
                                                        if elem[1] > 0
                                                        else 1/elem[0])), [])
        return factorlist

    # 分解 `rat` 的分子部分
    f = factorint(rat.p, limit=limit, use_trial=use_trial,
                  use_rho=use_rho, use_pm1=use_pm1,
                  verbose=verbose).copy()
    f = defaultdict(int, f)
    # 分解 `rat` 的分母部分，并将幂次数加上负号
    for p, e in factorint(rat.q, limit=limit,
                          use_trial=use_trial,
                          use_rho=use_rho,
                          use_pm1=use_pm1,
                          verbose=verbose).items():
        f[p] += -e

    # 如果结果字典长度大于1且包含键1，则删除键1
    if len(f) > 1 and 1 in f:
        del f[1]
    # 如果 visual 为 False，则返回标准字典形式的结果
    if not visual:
        return dict(f)
    else:
        # 如果 visual 为 True，则返回乘积形式的结果
        if -1 in f:
            f.pop(-1)
            args = [S.NegativeOne]
        else:
            args = []
        # 根据因子及其幂次数，构建乘积表达式
        args.extend([Pow(*i, evaluate=False)
                     for i in sorted(f.items())])
        return Mul(*args, evaluate=False)


# 返回整数 `n` 的素因子列表，按升序排列，忽略重复因子和限制条件下无法完全因子分解的合成因子
def primefactors(n, limit=None, verbose=False, **kwargs):
    """Return a sorted list of n's prime factors, ignoring multiplicity
    and any composite factor that remains if the limit was set too low
    for complete factorization. Unlike factorint(), primefactors() does
    not return -1 or 0.

    Parameters
    ==========

    n : integer
    # 将输入的参数 `n` 转换为整数
    n = int(n)
    
    # 更新关键字参数 `kwargs`，添加额外的参数 "visual": None, "multiple": False,
    # "limit": limit, "verbose": verbose
    kwargs.update({"visual": None, "multiple": False,
                   "limit": limit, "verbose": verbose})
    
    # 对整数 `n` 进行因数分解，并按键（因子）排序，存储在 `factors` 列表中
    factors = sorted(factorint(n=n, **kwargs).keys())
    
    # 从排序后的因子列表 `factors` 中筛选出素数（大于1且只有两个正因子：1 和自身）
    # 这里 `factors[:-1]` 排除最后一个元素，因为它可能是一个大的非素数因子
    s = [f for f in factors[:-1] if f not in [-1, 0, 1]]
    
    # 检查最后一个因子是否为素数，如果是，则加入到结果列表 `s` 中
    if factors and isprime(factors[-1]):
        s += [factors[-1]]
    
    # 返回筛选后的素数列表 `s`
    return s
# 定义一个函数 _divisors，用于生成给定整数 n 的所有约数
def _divisors(n, proper=False):
    """Helper function for divisors which generates the divisors.

    Parameters
    ==========

    n : int
        a nonnegative integer
    proper: bool
        If `True`, returns the generator that outputs only the proper divisor (i.e., excluding n).

    """
    # 如果 n 小于等于 1
    if n <= 1:
        # 如果 proper 是 False 且 n 不为 0，则生成 1
        if not proper and n:
            yield 1
        return

    # 使用 factorint 函数获取 n 的质因数分解结果，返回字典
    factordict = factorint(n)
    # 对质因数进行排序
    ps = sorted(factordict.keys())

    # 定义一个递归生成器函数 rec_gen
    def rec_gen(n=0):
        # 如果 n 等于质因数列表 ps 的长度，则生成 1
        if n == len(ps):
            yield 1
        else:
            # 初始化质因数 ps[n] 的幂次列表
            pows = [1]
            # 对质因数 ps[n] 进行迭代，生成其幂次列表
            for _ in range(factordict[ps[n]]):
                pows.append(pows[-1] * ps[n])
            # 递归生成余下质因数的乘积
            yield from (p * q for q in rec_gen(n + 1) for p in pows)

    # 如果 proper 是 True，则生成不包含 n 自身的约数
    if proper:
        yield from (p for p in rec_gen() if p != n)
    else:
        # 否则生成所有约数
        yield from rec_gen()


# 定义函数 divisors，返回给定整数 n 的所有约数，可以选择返回生成器
def divisors(n, generator=False, proper=False):
    r"""
    Return all divisors of n sorted from 1..n by default.
    If generator is ``True`` an unordered generator is returned.

    The number of divisors of n can be quite large if there are many
    prime factors (counting repeated factors). If only the number of
    factors is desired use divisor_count(n).

    Examples
    ========

    >>> from sympy import divisors, divisor_count
    >>> divisors(24)
    [1, 2, 3, 4, 6, 8, 12, 24]
    >>> divisor_count(24)
    8

    >>> list(divisors(120, generator=True))
    [1, 2, 4, 8, 3, 6, 12, 24, 5, 10, 20, 40, 15, 30, 60, 120]

    Notes
    =====

    This is a slightly modified version of Tim Peters referenced at:
    https://stackoverflow.com/questions/1010381/python-factorization

    See Also
    ========

    primefactors, factorint, divisor_count
    """
    # 调用 _divisors 函数获取 n 的所有约数
    rv = _divisors(as_int(abs(n)), proper)
    # 如果 generator 是 True，则返回生成器；否则返回排序后的列表
    return rv if generator else sorted(rv)


# 定义函数 divisor_count，返回给定整数 n 的约数个数，可以选择按模数统计或排除自身
def divisor_count(n, modulus=1, proper=False):
    """
    Return the number of divisors of ``n``. If ``modulus`` is not 1 then only
    those that are divisible by ``modulus`` are counted. If ``proper`` is True
    then the divisor of ``n`` will not be counted.

    Examples
    ========

    >>> from sympy import divisor_count
    >>> divisor_count(6)
    4
    >>> divisor_count(6, 2)
    2
    >>> divisor_count(6, proper=True)
    3

    See Also
    ========

    factorint, divisors, totient, proper_divisor_count

    """
    # 如果 modulus 为 0，则返回 0
    if not modulus:
        return 0
    elif modulus != 1:
        # 对 n 进行 modulus 的模运算，如果有余数则返回 0
        n, r = divmod(n, modulus)
        if r:
            return 0
    # 如果 n 为 0，则返回 0
    if n == 0:
        return 0
    # 计算 n 的所有约数个数，包括重复因子
    n = Mul(*[v + 1 for k, v in factorint(n).items() if k > 1])
    # 如果 proper 是 True，则减去自身的约数 1
    if n and proper:
        n -= 1
    return n


# 定义函数 proper_divisors，返回给定整数 n 的所有真约数（不包含自身），可以选择返回生成器
def proper_divisors(n, generator=False):
    """
    Return all divisors of n except n, sorted by default.
    If generator is ``True`` an unordered generator is returned.

    Examples
    ========

    >>> from sympy import proper_divisors, proper_divisor_count
    >>> proper_divisors(24)
    [1, 2, 3, 4, 6, 8, 12]
    >>> proper_divisor_count(24)
    7
    >>> list(proper_divisors(120, generator=True))

    """
    [1, 2, 4, 8, 3, 6, 12, 24, 5, 10, 20, 40, 15, 30, 60]


这是一个包含整数的列表。


    See Also
    ========

    factorint, divisors, proper_divisor_count


这是一个文档字符串的一部分，列出了相关的函数或概念，供参考。


    """
    return divisors(n, generator=generator, proper=True)
    """


这是一个函数定义内的注释部分，指示函数的返回值是调用 `divisors` 函数的结果，其中包括参数 `n`、`generator` 和 `proper`。
# 返回整数 n 的真约数的数量
def proper_divisor_count(n, modulus=1):
    return divisor_count(n, modulus=modulus, proper=True)


# 辅助函数，用于生成整数 n 的单位约数
def _udivisors(n):
    """Helper function for udivisors which generates the unitary divisors.

    Parameters
    ==========

    n : int
        a nonnegative integer

    """
    if n <= 1:
        if n == 1:
            yield 1
        return

    # 计算 n 的所有质因数的幂次方列表
    factorpows = [p**e for p, e in factorint(n).items()]
    # 生成所有可能的乘积子集，作为单位约数
    for i in range(2**len(factorpows)):
        d = 1
        for k in range(i.bit_length()):
            if i & 1:
                d *= factorpows[k]
            i >>= 1
        yield d


# 返回整数 n 的单位约数列表，按照默认顺序排序
def udivisors(n, generator=False):
    r"""
    Return all unitary divisors of n sorted from 1..n by default.
    If generator is ``True`` an unordered generator is returned.

    The number of unitary divisors of n can be quite large if there are many
    prime factors. If only the number of unitary divisors is desired use
    udivisor_count(n).

    Examples
    ========

    >>> from sympy.ntheory.factor_ import udivisors, udivisor_count
    >>> udivisors(15)
    [1, 3, 5, 15]
    >>> udivisor_count(15)
    4

    >>> sorted(udivisors(120, generator=True))
    [1, 3, 5, 8, 15, 24, 40, 120]

    See Also
    ========

    primefactors, factorint, divisors, divisor_count, udivisor_count

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Unitary_divisor
    .. [2] https://mathworld.wolfram.com/UnitaryDivisor.html

    """
    # 调用 _udivisors 函数生成单位约数的生成器，根据 generator 参数返回结果
    rv = _udivisors(as_int(abs(n)))
    return rv if generator else sorted(rv)


# 返回整数 n 的单位约数的数量
def udivisor_count(n):
    """
    Return the number of unitary divisors of ``n``.

    Parameters
    ==========

    n : integer

    Examples
    ========

    >>> from sympy.ntheory.factor_ import udivisor_count
    >>> udivisor_count(120)
    8

    See Also
    ========

    factorint, divisors, udivisors, divisor_count, totient

    References
    ==========

    .. [1] https://mathworld.wolfram.com/UnitaryDivisorFunction.html

    """
    if n == 0:
        return 0
    # 计算 n 的所有质因数中大于 1 的个数，返回 2 的这个数量次方
    return 2**len([p for p in factorint(n) if p > 1])


# 辅助函数，用于生成整数 n 的反约数
def _antidivisors(n):
    """Helper function for antidivisors which generates the antidivisors.

    Parameters
    ==========

    n : int
        a nonnegative integer

    """
    if n <= 2:
        return
    # 生成满足条件的反约数
    for d in _divisors(n):
        y = 2*d
        if n > y and n % y:
            yield y
    for d in _divisors(2*n-1):
        if n > d >= 2 and n % d:
            yield d
    for d in _divisors(2*n+1):
        if n > d >= 2 and n % d:
            yield d


# 返回整数 n 的反约数列表，按照默认顺序排序
def antidivisors(n, generator=False):
    r"""
    Return all antidivisors of n.

    Parameters
    ==========

    n : integer

    generator : bool, optional
        If True, return an unordered generator; if False, return a sorted list.

    Examples
    ========

    >>> from sympy.ntheory.factor_ import antidivisors
    >>> antidivisors(10)
    [4, 5, 6, 7, 8, 9]

    See Also
    ========

    _antidivisors

    """
    # 根据输入的整数 n 计算其反因子（antidivisors），默认按照 1 到 n 排序返回结果。

    # 反因子 [1]_ 是指不以最大可能的余数整除 n 的数字。如果 generator 参数为 True，则返回一个无序的生成器。

    # 示例
    # ========

    # >>> from sympy.ntheory.factor_ import antidivisors
    # >>> antidivisors(24)
    # [7, 16]

    # >>> sorted(antidivisors(128, generator=True))
    # [3, 5, 15, 17, 51, 85]

    # 参见
    # ========

    # primefactors, factorint, divisors, divisor_count, antidivisor_count

    # 参考文献
    # ==========

    # .. [1] 反因子的定义请参考 https://oeis.org/A066272/a066272a.html

    """
    rv = _antidivisors(as_int(abs(n)))  # 调用 _antidivisors 函数计算 n 的反因子，参数为 n 的绝对值的整数形式
    return rv if generator else sorted(rv)  # 如果 generator 为 True，则返回未排序的生成器 rv；否则返回排序后的列表 rv
# 定义一个函数，计算整数 n 的 antidivisors 的数量
def antidivisor_count(n):
    """
    Return the number of antidivisors [1]_ of ``n``.

    Parameters
    ==========

    n : integer
        输入的整数 n

    Examples
    ========

    >>> from sympy.ntheory.factor_ import antidivisor_count
    >>> antidivisor_count(13)
    4
    >>> antidivisor_count(27)
    5

    See Also
    ========

    factorint, divisors, antidivisors, divisor_count, totient

    References
    ==========

    .. [1] formula from https://oeis.org/A066272

    """

    # 将 n 转换为其绝对值的整数
    n = as_int(abs(n))
    # 如果 n 小于等于 2，则直接返回 0
    if n <= 2:
        return 0
    # 计算 antidivisor 数量的具体计算公式
    return divisor_count(2*n - 1) + divisor_count(2*n + 1) + \
        divisor_count(n) - divisor_count(n, 2) - 5


@deprecated("""\
The `sympy.ntheory.factor_.totient` has been moved to `sympy.functions.combinatorial.numbers.totient`.""",
            deprecated_since_version="1.13",
            active_deprecations_target='deprecated-ntheory-symbolic-functions')
# 标记该函数已弃用，并提供替代的 totient 函数的信息
def totient(n):
    r"""
    Calculate the Euler totient function phi(n)

    .. deprecated:: 1.13

        The ``totient`` function is deprecated. Use :class:`sympy.functions.combinatorial.numbers.totient`
        instead. See its documentation for more information. See
        :ref:`deprecated-ntheory-symbolic-functions` for details.

    ``totient(n)`` or `\phi(n)` is the number of positive integers `\leq` n
    that are relatively prime to n.

    Parameters
    ==========

    n : integer
        输入的整数 n

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

    divisor_count

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Euler%27s_totient_function
    .. [2] https://mathworld.wolfram.com/TotientFunction.html

    """
    # 从 sympy 的新位置导入 totient 函数，并调用它返回结果
    from sympy.functions.combinatorial.numbers import totient as _totient
    return _totient(n)


@deprecated("""\
The `sympy.ntheory.factor_.reduced_totient` has been moved to `sympy.functions.combinatorial.numbers.reduced_totient`.""",
            deprecated_since_version="1.13",
            active_deprecations_target='deprecated-ntheory-symbolic-functions')
# 标记该函数已弃用，并提供替代的 reduced_totient 函数的信息
def reduced_totient(n):
    r"""
    Calculate the Carmichael reduced totient function lambda(n)

    .. deprecated:: 1.13

        The ``reduced_totient`` function is deprecated. Use :class:`sympy.functions.combinatorial.numbers.reduced_totient`
        instead. See its documentation for more information. See
        :ref:`deprecated-ntheory-symbolic-functions` for details.

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

    """
    # 从 sympy 的新位置导入 reduced_totient 函数，并调用它返回结果
    from sympy.functions.combinatorial.numbers import reduced_totient as _reduced_totient
    return _reduced_totient(n)
    # 导入 SymPy 库中的 reduced_totient 函数，用于计算 n 的约化欧拉函数
    from sympy.functions.combinatorial.numbers import reduced_totient as _reduced_totient
    # 返回调用 reduced_totient 函数计算得到的 n 的约化欧拉函数值
    return _reduced_totient(n)
@deprecated("""\
The `sympy.ntheory.factor_.divisor_sigma` has been moved to `sympy.functions.combinatorial.numbers.divisor_sigma`.""",
deprecated_since_version="1.13",
active_deprecations_target='deprecated-ntheory-symbolic-functions')
# 使用装饰器 @deprecated 标记该函数为已废弃，提供了废弃信息和版本信息
def divisor_sigma(n, k=1):
    r"""
    Calculate the divisor function `\sigma_k(n)` for positive integer n

    .. deprecated:: 1.13

        The ``divisor_sigma`` function is deprecated. Use :class:`sympy.functions.combinatorial.numbers.divisor_sigma`
        instead. See its documentation for more information. See
        :ref:`deprecated-ntheory-symbolic-functions` for details.

    ``divisor_sigma(n, k)`` is equal to ``sum([x**k for x in divisors(n)])``

    If n's prime factorization is:

    .. math ::
        n = \prod_{i=1}^\omega p_i^{m_i},

    then

    .. math ::
        \sigma_k(n) = \prod_{i=1}^\omega (1+p_i^k+p_i^{2k}+\cdots
        + p_i^{m_ik}).

    Parameters
    ==========

    n : integer

    k : integer, optional
        power of divisors in the sum

        for k = 0, 1:
        ``divisor_sigma(n, 0)`` is equal to ``divisor_count(n)``
        ``divisor_sigma(n, 1)`` is equal to ``sum(divisors(n))``

        Default for k is 1.

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

    divisor_count, totient, divisors, factorint

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Divisor_function

    """
    from sympy.functions.combinatorial.numbers import divisor_sigma as func_divisor_sigma
    # 使用导入的新函数计算结果并返回
    return func_divisor_sigma(n, k)


def _divisor_sigma(n:int, k:int=1) -> int:
    r""" Calculate the divisor function `\sigma_k(n)` for positive integer n

    Parameters
    ==========

    n : int
        positive integer
    k : int
        nonnegative integer

    See Also
    ========

    sympy.functions.combinatorial.numbers.divisor_sigma

    """
    # 根据给定的 k 值计算 `\sigma_k(n)`，对于 k = 0 时使用 `factorint` 函数
    if k == 0:
        return math.prod(e + 1 for e in factorint(n).values())
    # 对于 k > 0 使用数论公式计算 `\sigma_k(n)` 的值
    return math.prod((p**(k*(e + 1)) - 1)//(p**k - 1) for p, e in factorint(n).items())


def core(n, t=2):
    r"""
    Calculate core(n, t) = `core_t(n)` of a positive integer n

    ``core_2(n)`` is equal to the squarefree part of n

    If n's prime factorization is:

    .. math ::
        n = \prod_{i=1}^\omega p_i^{m_i},

    then

    .. math ::
        core_t(n) = \prod_{i=1}^\omega p_i^{m_i \mod t}.

    Parameters
    ==========

    n : integer

    t : integer
        core(n, t) calculates the t-th power free part of n

        ``core(n, 2)`` is the squarefree part of ``n``
        ``core(n, 3)`` is the cubefree part of ``n``

        Default for t is 2.

    Examples
    ========

    >>> from sympy.ntheory.factor_ import core
    >>> core(24, 2)
    6
    >>> core(9424, 3)
    1178
    >>> core(379238)
    379238

    """
    # 计算正整数 n 的 t-th 幂次自由部分 core_t(n)
    # 对每个质因数应用模运算以得到结果
    if t == 2:
        return math.prod(p**(e % 2) for p, e in factorint(n).items())
    elif t == 3:
        return math.prod(p**(e % 3) for p, e in factorint(n).items())
    else:
        return math.prod(p**(e % t) for p, e in factorint(n).items())
    # 调用 as_int 函数将 n 转换为整数类型
    n = as_int(n)
    # 调用 as_int 函数将 t 转换为整数类型
    t = as_int(t)
    # 如果 n 小于等于 0，则抛出数值错误异常
    if n <= 0:
        raise ValueError("n must be a positive integer")
    # 如果 t 小于等于 1，则抛出数值错误异常
    elif t <= 1:
        raise ValueError("t must be >= 2")
    else:
        # 初始化 y 为 1
        y = 1
        # 使用 factorint 函数获取 n 的质因数分解结果，并遍历其中的质因数 p 及其指数 e
        for p, e in factorint(n).items():
            # 将每个质因数 p 的指数 e 对 t 取模，然后计算 p 的该模值次幂，并乘到 y 上
            y *= p**(e % t)
        # 返回计算得到的 y
        return y
# 使用 @deprecated 装饰器标记函数为已废弃，提供替代方法和相关信息
@deprecated("""
The `sympy.ntheory.factor_.udivisor_sigma` has been moved to `sympy.functions.combinatorial.numbers.udivisor_sigma`.
""",
deprecated_since_version="1.13",
active_deprecations_target='deprecated-ntheory-symbolic-functions')
# 定义函数 udivisor_sigma(n, k)，计算正整数 n 的单位因子和函数 σ_k*(n)
def udivisor_sigma(n, k=1):
    r"""
    计算正整数 n 的单位因子和函数 `\sigma_k^*(n)`

    .. deprecated:: 1.13

        函数 ``udivisor_sigma`` 已被废弃。请使用 :class:`sympy.functions.combinatorial.numbers.udivisor_sigma`
        替代。更多信息请参见其文档。参见
        :ref:`deprecated-ntheory-symbolic-functions` 获取详细信息。

    ``udivisor_sigma(n, k)`` 等于 ``sum([x**k for x in udivisors(n)])``

    如果 n 的质因数分解为:

    .. math ::
        n = \prod_{i=1}^\omega p_i^{m_i},

    则

    .. math ::
        \sigma_k^*(n) = \prod_{i=1}^\omega (1+ p_i^{m_ik}).

    Parameters
    ==========

    k : 求和中因子的幂次

        对于 k = 0, 1:
        ``udivisor_sigma(n, 0)`` 等于 ``udivisor_count(n)``
        ``udivisor_sigma(n, 1)`` 等于 ``sum(udivisors(n))``

        默认值为 k = 1.

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

    divisor_count, totient, divisors, udivisors, udivisor_count, divisor_sigma,
    factorint

    References
    ==========

    .. [1] https://mathworld.wolfram.com/UnitaryDivisorFunction.html

    """
    # 导入新位置的 udivisor_sigma 函数并返回其计算结果
    from sympy.functions.combinatorial.numbers import udivisor_sigma as _udivisor_sigma
    return _udivisor_sigma(n, k)


@deprecated("""
The `sympy.ntheory.factor_.primenu` has been moved to `sympy.functions.combinatorial.numbers.primenu`.
""",
deprecated_since_version="1.13",
active_deprecations_target='deprecated-ntheory-symbolic-functions')
# 定义函数 primenu(n)，计算正整数 n 的不同质因数数量
def primenu(n):
    r"""
    计算正整数 n 的不同质因数数量。

    .. deprecated:: 1.13

        函数 ``primenu`` 已被废弃。请使用 :class:`sympy.functions.combinatorial.numbers.primenu`
        替代。更多信息请参见其文档。参见
        :ref:`deprecated-ntheory-symbolic-functions` 获取详细信息。

    如果 n 的质因数分解为:

    .. math ::
        n = \prod_{i=1}^k p_i^{m_i},

    则 ``primenu(n)`` 或 `\nu(n)` 为:

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

    factorint

    References
    ==========

    .. [1] https://mathworld.wolfram.com/PrimeFactor.html

    """
    # 导入新位置的 primenu 函数并返回其计算结果
    from sympy.functions.combinatorial.numbers import primenu as _primenu
    return _primenu(n)
# 定义了一个被弃用的函数 `primeomega`，用于计算正整数 n 的素因子个数（考虑重复计数）
# 被弃用声明，指出自版本 1.13 开始已不推荐使用该函数，建议使用 `sympy.functions.combinatorial.numbers.primeomega`
# 更多信息请参见相关文档和 `deprecated-ntheory-symbolic-functions` 部分。
def primeomega(n):
    # 导入并重命名 `sympy.functions.combinatorial.numbers.primeomega` 为 `_primeomega`
    from sympy.functions.combinatorial.numbers import primeomega as _primeomega
    # 调用重命名后的函数 `_primeomega` 计算并返回结果
    return _primeomega(n)


def mersenne_prime_exponent(nth):
    """Returns the exponent ``i`` for the nth Mersenne prime (which
    has the form `2^i - 1`).

    Examples
    ========

    >>> from sympy.ntheory.factor_ import mersenne_prime_exponent
    >>> mersenne_prime_exponent(1)
    2
    >>> mersenne_prime_exponent(20)
    4423
    """
    # 将输入 `nth` 转换为整数 `n`
    n = as_int(nth)
    # 如果 `n` 小于 1，引发 ValueError
    if n < 1:
        raise ValueError("nth must be a positive integer; mersenne_prime_exponent(1) == 2")
    # 如果 `n` 大于 51，引发 ValueError
    if n > 51:
        raise ValueError("There are only 51 perfect numbers; nth must be less than or equal to 51")
    # 返回预定义列表 `MERSENNE_PRIME_EXPONENTS` 中第 `n-1` 个元素（实际上是第 `n` 个 Mersenne 素数的指数 `i`）
    return MERSENNE_PRIME_EXPONENTS[n - 1]


def is_perfect(n):
    """Returns True if ``n`` is a perfect number, else False.

    A perfect number is equal to the sum of its positive, proper divisors.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import divisor_sigma
    >>> from sympy.ntheory.factor_ import is_perfect, divisors
    >>> is_perfect(20)
    False
    >>> is_perfect(6)
    True
    >>> 6 == divisor_sigma(6) - 6 == sum(divisors(6)[:-1])
    True

    References
    ==========

    .. [1] https://mathworld.wolfram.com/PerfectNumber.html
    .. [2] https://en.wikipedia.org/wiki/Perfect_number

    """
    # 将输入 `n` 转换为整数 `n`
    n = as_int(n)
    # 如果 `n` 小于 1，返回 False
    if n < 1:
        return False
    # 如果 `n` 是偶数
    if n % 2 == 0:
        # 计算 `m` 的值
        m = (n.bit_length() + 1) >> 1
        # 如果不满足偶完全数的形式条件，返回 False
        if (1 << (m - 1)) * ((1 << m) - 1) != n:
            return False
        # 检查 `m` 是否在预定义的 `MERSENNE_PRIME_EXPONENTS` 列表中或者是 Mersenne 素数的指数
        return m in MERSENNE_PRIME_EXPONENTS or is_mersenne_prime(2**m - 1)

    # 当 `n` 是奇数时的处理逻辑
    # 如果 `n` 小于 10^2000，返回 False
    if n < 10**2000:  # https://www.lirmm.fr/~ochem/opn/
        return False
    # 如果 `n` 是 105 的倍数，返回 False
    if n % 105 == 0:  # not divis by 105
        return False
    # 检查是否满足一组特定的条件，这些条件与因子结构有关
    # 如果要测试其结构，则必须对其进行因式分解；一旦获取了因子，
    # 就可以检查它是否是完美数。因此，跳过结构检查，直接进行最终的测试。
    result = abundance(n) == 0
    # 如果结果为真，则抛出值错误，附带一段引用自Sylvester的言论和n的因子分解结果
    if result:
        raise ValueError(filldedent('''In 1888, Sylvester stated: "
            ...a prolonged meditation on the subject has satisfied
            me that the existence of any one such [odd perfect number]
            -- its escape, so to say, from the complex web of conditions
            which hem it in on all sides -- would be little short of a
            miracle." I guess SymPy just found that miracle and it
            factors like this: %s''' % factorint(n)))
    # 返回最终的结果，即布尔值result
    return result
# 返回一个数的丰富度（abundance），即其正真因子之和与自身的差值
def abundance(n):
    return _divisor_sigma(n) - 2 * n


# 返回 True 如果 n 是丰富数（abundant number），否则返回 False
# 丰富数是其所有正真因子之和大于自身的数
def is_abundant(n):
    n = as_int(n)
    # 如果 n 是完全数（perfect number），则不是丰富数
    if is_perfect(n):
        return False
    # n 是丰富数的条件之一是其丰富度大于 0
    return n % 6 == 0 or bool(abundance(n) > 0)


# 返回 True 如果 n 是不足数（deficient number），否则返回 False
# 不足数是其所有正真因子之和小于自身的数
def is_deficient(n):
    n = as_int(n)
    # 如果 n 是完全数（perfect number），则不是不足数
    if is_perfect(n):
        return False
    # n 是不足数的条件之一是其丰富度小于 0
    return bool(abundance(n) < 0)


# 返回 True 如果 m 和 n 是亲和数（amicable numbers），否则返回 False
# 亲和数是一对数，彼此的正真因子之和都等于另一个数本身
def is_amicable(m, n):
    return m != n and m + n == _divisor_sigma(m) == _divisor_sigma(n)


# 返回 True 如果 n 是卡迈克尔数（Carmichael number），否则返回 False
# 卡迈克尔数是大于 561 且不是素数，并且满足一定条件的数
def is_carmichael(n):
    if n < 561:
        return False
    return n % 2 and not isprime(n) and \
           all(e == 1 and (n - 1) % (p - 1) == 0 for p, e in factorint(n).items())


# 返回在指定范围内的所有卡迈克尔数的列表
def find_carmichael_numbers_in_range(x, y):
    if 0 <= x <= y:
        if x % 2 == 0:
            return [i for i in range(x + 1, y, 2) if is_carmichael(i)]
        else:
            return [i for i in range(x, y, 2) if is_carmichael(i)]
    else:
        # 如果条件不满足，则抛出值错误异常，提示给定范围无效，要求 x 和 y 必须是非负整数，并且 x <= y
        raise ValueError('The provided range is not valid. x and y must be non-negative integers and x <= y')
# 返回前 n 个 Carmichael 数字的列表
def find_first_n_carmichaels(n):
    # 初始化 i 为第一个已知的 Carmichael 数字
    i = 561
    # 创建空列表，用于存储找到的 Carmichael 数字
    carmichaels = []

    # 当找到的 Carmichael 数字个数小于 n 时执行循环
    while len(carmichaels) < n:
        # 如果当前的 i 是 Carmichael 数字，则将其添加到列表中
        if is_carmichael(i):
            carmichaels.append(i)
        # 跳过偶数，因为已知 Carmichael 数字都是奇数
        i += 2

    # 返回找到的前 n 个 Carmichael 数字的列表
    return carmichaels


# 返回数字 n 在给定基数 b 下的加法数字根
def dra(n, b):
    """
    Parameters
    ==========

    n : Integer
        自然数 n

    b : Integer
        进制 b

    Returns
    =======

    Integer
        加法数字根

    Raises
    ======

    ValueError
        如果基数 b 不大于 1

    """
    # 获取 n 的绝对值
    num = abs(as_int(n))
    # 将 b 转换为整数
    b = as_int(b)
    # 如果基数 b 小于等于 1，则抛出 ValueError 异常
    if b <= 1:
        raise ValueError("Base should be an integer greater than 1")

    # 如果 n 为 0，则直接返回 0
    if num == 0:
        return 0

    # 计算并返回 n 在基数 b 下的加法数字根
    return (1 + (num - 1) % (b - 1))


# 返回数字 n 在给定基数 b 下的乘法数字根
def drm(n, b):
    """
    Parameters
    ==========

    n : Integer
        自然数 n

    b : Integer
        进制 b

    Returns
    =======

    Integer
        乘法数字根

    Raises
    ======

    ValueError
        如果基数 b 不大于 1

    """
    # 获取 n 的绝对值
    n = abs(as_int(n))
    # 将 b 转换为整数
    b = as_int(b)
    # 如果基数 b 小于等于 1，则抛出 ValueError 异常
    if b <= 1:
        raise ValueError("Base should be an integer greater than 1")

    # 当 n 大于基数 b 时执行循环
    while n > b:
        mul = 1
        # 将 n 拆解为每一位数字，并计算它们的乘积
        while n > 1:
            n, r = divmod(n, b)
            # 如果余数为 0，则直接返回 0
            if r == 0:
                return 0
            mul *= r
        # 将结果赋给 n，继续下一轮迭代
        n = mul

    # 返回 n 在基数 b 下的乘法数字根
    return n
```