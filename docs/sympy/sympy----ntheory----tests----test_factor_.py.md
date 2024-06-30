# `D:\src\scipysrc\sympy\sympy\ntheory\tests\test_factor_.py`

```
from sympy.core.containers import Dict
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.functions.combinatorial.factorials import factorial as fac
from sympy.core.numbers import Integer, Rational
from sympy.external.gmpy import gcd

from sympy.ntheory import (totient,
    factorint, primefactors, divisors, nextprime,
    pollard_rho, perfect_power, multiplicity, multiplicity_in_factorial,
    divisor_count, primorial, pollard_pm1, divisor_sigma,
    factorrat, reduced_totient)
from sympy.ntheory.factor_ import (smoothness, smoothness_p, proper_divisors,
    antidivisors, antidivisor_count, _divisor_sigma, core, udivisors, udivisor_sigma,
    udivisor_count, proper_divisor_count, primenu, primeomega,
    mersenne_prime_exponent, is_perfect, is_abundant,
    is_deficient, is_amicable, is_carmichael, find_carmichael_numbers_in_range,
    find_first_n_carmichaels, dra, drm, _perfect_power)

from sympy.testing.pytest import raises, slow

from sympy.utilities.iterables import capture


def fac_multiplicity(n, p):
    """Return the power of the prime number p in the
    factorization of n!"""
    if p > n:
        return 0
    if p > n//2:
        return 1
    q, m = n, 0
    while q >= p:
        q //= p
        m += q
    return m


def multiproduct(seq=(), start=1):
    """
    Return the product of a sequence of factors with multiplicities,
    times the value of the parameter ``start``. The input may be a
    sequence of (factor, exponent) pairs or a dict of such pairs.

        >>> multiproduct({3:7, 2:5}, 4) # = 3**7 * 2**5 * 4
        279936

    """
    if not seq:
        return start
    if isinstance(seq, dict):
        seq = iter(seq.items())
    units = start
    multi = []
    for base, exp in seq:
        if not exp:
            continue
        elif exp == 1:
            units *= base
        else:
            if exp % 2:
                units *= base
            multi.append((base, exp//2))
    return units * multiproduct(multi)**2


def test_multiplicity():
    for b in range(2, 20):
        for i in range(100):
            assert multiplicity(b, b**i) == i
            assert multiplicity(b, (b**i) * 23) == i
            assert multiplicity(b, (b**i) * 1000249) == i
    # Should be fast
    assert multiplicity(10, 10**10023) == 10023
    # Should exit quickly
    assert multiplicity(10**10, 10**10) == 1
    # Should raise errors for bad input
    raises(ValueError, lambda: multiplicity(1, 1))
    raises(ValueError, lambda: multiplicity(1, 2))
    raises(ValueError, lambda: multiplicity(1.3, 2))
    raises(ValueError, lambda: multiplicity(2, 0))
    raises(ValueError, lambda: multiplicity(1.3, 0))

    # handles Rationals
    assert multiplicity(10, Rational(30, 7)) == 1
    assert multiplicity(Rational(2, 7), Rational(4, 7)) == 1
    assert multiplicity(Rational(1, 7), Rational(3, 49)) == 2
    assert multiplicity(Rational(2, 7), Rational(7, 2)) == -1
    # 使用 assert 断言来验证 multiplicity 函数的预期行为
    assert multiplicity(3, Rational(1, 9)) == -2
# 定义测试函数，用于验证 `multiplicity` 函数在计算阶乘中的多重性
def test_multiplicity_in_factorial():
    # 计算阶乘 1000 的值
    n = fac(1000)
    # 遍历一系列整数，验证 `multiplicity` 函数的结果与 `multiplicity_in_factorial` 函数在阶乘中的计算一致
    for i in (2, 4, 6, 12, 30, 36, 48, 60, 72, 96):
        assert multiplicity(i, n) == multiplicity_in_factorial(i, 1000)


# 定义测试函数，用于验证 `_perfect_power` 函数的私有方法
def test_private_perfect_power():
    # 验证 `_perfect_power` 函数在一些边界情况下的返回值
    assert _perfect_power(0) is False
    assert _perfect_power(1) is False
    assert _perfect_power(2) is False
    assert _perfect_power(3) is False
    # 遍历一系列整数及其幂指数，验证 `_perfect_power` 函数的正确性
    for x in [2, 3, 5, 6, 7, 12, 15, 105, 100003]:
        for y in range(2, 100):
            assert _perfect_power(x**y) == (x, y)
            # 对于奇数 x，验证使用不同的下一个素数对 `_perfect_power` 的影响
            if x & 1:
                assert _perfect_power(x**y, next_p=3) == (x, y)
            # 对于特定的 x，验证使用不同的下一个素数对 `_perfect_power` 的影响
            if x == 100003:
                assert _perfect_power(x**y, next_p=100003) == (x, y)
            # 验证 `_perfect_power` 在一些数值的情况下返回 False，例如 Catalan 猜想
            assert _perfect_power(101*x**y) == False
            if x**y not in [8, 9]:
                assert _perfect_power(x**y + 1) == False  # Catalan's conjecture
                assert _perfect_power(x**y - 1) == False
    # 遍历一系列整数对，验证 `_perfect_power` 函数对于一些组合的正确性
    for x in range(1, 10):
        for y in range(1, 10):
            # 计算 x 和 y 的最大公约数
            g = gcd(x, y)
            # 根据最大公约数的不同，验证 `_perfect_power` 函数的返回值
            if g == 1:
                assert _perfect_power(5**x * 101**y) == False
            else:
                assert _perfect_power(5**x * 101**y) == (5**(x//g) * 101**(y//g), g)


# 定义测试函数，用于验证 `perfect_power` 函数
def test_perfect_power():
    # 验证 `perfect_power` 函数在特定参数下会引发 ValueError
    raises(ValueError, lambda: perfect_power(0.1))
    # 验证 `perfect_power` 函数在一些边界情况下的返回值
    assert perfect_power(0) is False
    assert perfect_power(1) is False
    assert perfect_power(2) is False
    assert perfect_power(3) is False
    assert perfect_power(4) == (2, 2)
    assert perfect_power(14) is False
    assert perfect_power(25) == (5, 2)
    assert perfect_power(22) is False
    assert perfect_power(22, [2]) is False
    assert perfect_power(137**(3*5*13)) == (137, 3*5*13)
    assert perfect_power(137**(3*5*13) + 1) is False
    assert perfect_power(137**(3*5*13) - 1) is False
    assert perfect_power(103005006004**7) == (103005006004, 7)
    assert perfect_power(103005006004**7 + 1) is False
    assert perfect_power(103005006004**7 - 1) is False
    assert perfect_power(103005006004**12) == (103005006004, 12)
    assert perfect_power(103005006004**12 + 1) is False
    assert perfect_power(103005006004**12 - 1) is False
    assert perfect_power(2**10007) == (2, 10007)
    assert perfect_power(2**10007 + 1) is False
    assert perfect_power(2**10007 - 1) is False
    assert perfect_power((9**99 + 1)**60) == (9**99 + 1, 60)
    assert perfect_power((9**99 + 1)**60 + 1) is False
    assert perfect_power((9**99 + 1)**60 - 1) is False
    assert perfect_power((10**40000)**2, big=False) == (10**40000, 2)
    assert perfect_power(10**100000) == (10, 100000)
    assert perfect_power(10**100001) == (10, 100001)
    assert perfect_power(13**4, [3, 5]) is False
    assert perfect_power(3**4, [3, 10], factor=0) is False
    assert perfect_power(3**3*5**3) == (15, 3)
    assert perfect_power(2**3*5**5) is False
    assert perfect_power(2*13**4) is False
    assert perfect_power(2**5*3**3) is False
    # 定义变量并赋值
    t = 2**24
    # 对于数值 24 的所有因子，依次执行以下操作：
    for d in divisors(24):
        # 计算 t * 3^d 是否为完全幂
        m = perfect_power(t*3**d)
        # 断言条件：m 存在且 m 的指数部分等于 d，或者 d 等于 1
        assert m and m[1] == d or d == 1
        # 再次计算 t * 3^d 是否为完全幂，但此处不考虑大整数的情况
        m = perfect_power(t*3**d, big=False)
        # 断言条件：m 存在且 m 的指数部分等于 2，或者 d 等于 1，或者 d 等于 3，如果断言失败则输出 (d, m)
        assert m and m[1] == 2 or d == 1 or d == 3, (d, m)

    # 断言条件：-4 不是完全幂
    assert perfect_power(-4) is False
    # 断言条件：-8 的完全幂形式为 (-2, 3)
    assert perfect_power(-8) == (-2, 3)
    # 断言条件：(1/2)^3 的完全幂形式为 (S.Half, 3)
    assert perfect_power(Rational(1, 2)**3) == (S.Half, 3)
    # 断言条件：(-3/2)^3 的完全幂形式为 (-3*S.Half, 3)
    assert perfect_power(Rational(-3, 2)**3) == (-3*S.Half, 3)
# 声明一个装饰器，用于标记该函数执行缓慢，需要更长的时间来完成
@slow
# 定义一个测试函数，用于测试 factorint 函数的功能
def test_factorint():
    # 断言对于输入 123456，primefactors 函数返回 [2, 3, 643]
    assert primefactors(123456) == [2, 3, 643]
    # 断言对于输入 0，factorint 函数返回 {0: 1}
    assert factorint(0) == {0: 1}
    # 断言对于输入 1，factorint 函数返回空字典 {}
    assert factorint(1) == {}
    # 断言对于输入 -1，factorint 函数返回 {-1: 1}
    assert factorint(-1) == {-1: 1}
    # 断言对于输入 -2，factorint 函数返回 {-1: 1, 2: 1}
    assert factorint(-2) == {-1: 1, 2: 1}
    # 断言对于输入 -16，factorint 函数返回 {-1: 1, 2: 4}
    assert factorint(-16) == {-1: 1, 2: 4}
    # 断言对于输入 2，factorint 函数返回 {2: 1}
    assert factorint(2) == {2: 1}
    # 断言对于输入 126，factorint 函数返回 {2: 1, 3: 2, 7: 1}
    assert factorint(126) == {2: 1, 3: 2, 7: 1}
    # 断言对于输入 123456，factorint 函数返回 {2: 6, 3: 1, 643: 1}
    assert factorint(123456) == {2: 6, 3: 1, 643: 1}
    # 断言对于输入 5951757，factorint 函数返回 {3: 1, 7: 1, 29: 2, 337: 1}
    assert factorint(5951757) == {3: 1, 7: 1, 29: 2, 337: 1}
    # 断言对于输入 64015937，factorint 函数返回 {7993: 1, 8009: 1}
    assert factorint(64015937) == {7993: 1, 8009: 1}
    # 断言对于输入 2**(2**6) + 1，factorint 函数返回 {274177: 1, 67280421310721: 1}
    assert factorint(2**(2**6) + 1) == {274177: 1, 67280421310721: 1}
    # issue 19683
    # 断言对于输入 10**38 - 1，factorint 函数返回 {3: 2, 11: 1, 909090909090909091: 1, 1111111111111111111: 1}
    assert factorint(10**38 - 1) == {3: 2, 11: 1, 909090909090909091: 1, 1111111111111111111: 1}
    # issue 17676
    # 断言对于输入 28300421052393658575，factorint 函数返回 {3: 1, 5: 2, 11: 2, 43: 1, 2063: 2, 4127: 1, 4129: 1}
    assert factorint(28300421052393658575) == {3: 1, 5: 2, 11: 2, 43: 1, 2063: 2, 4127: 1, 4129: 1}
    # 断言对于输入 2063**2 * 4127**1 * 4129**1，factorint 函数返回 {2063: 2, 4127: 1, 4129: 1}
    assert factorint(2063**2 * 4127**1 * 4129**1) == {2063: 2, 4127: 1, 4129: 1}
    # 断言对于输入 2347**2 * 7039**1 * 7043**1，factorint 函数返回 {2347: 2, 7039: 1, 7043: 1}

    # 断言对于输入 0，multiple=True 参数时，factorint 函数返回 [0]
    assert factorint(0, multiple=True) == [0]
    # 断言对于输入 1，multiple=True 参数时，factorint 函数返回空列表 []
    assert factorint(1, multiple=True) == []
    # 断言对于输入 -1，multiple=True 参数时，factorint 函数返回 [-1]
    assert factorint(-1, multiple=True) == [-1]
    # 断言对于输入 -2，multiple=True 参数时，factorint 函数返回 [-1, 2]
    assert factorint(-2, multiple=True) == [-1, 2]
    # 断言对于输入 -16，multiple=True 参数时，factorint 函数返回 [-1, 2, 2, 2, 2]
    assert factorint(-16, multiple=True) == [-1, 2, 2, 2, 2]
    # 断言对于输入 2，multiple=True 参数时，factorint 函数返回 [2]
    assert factorint(2, multiple=True) == [2]
    # 断言对于输入 24，multiple=True 参数时，factorint 函数返回 [2, 2, 2, 3]
    assert factorint(24, multiple=True) == [2, 2, 2, 3]
    # 断言对于输入 126，multiple=True 参数时，factorint 函数返回 [2, 3, 3, 7]
    assert factorint(126, multiple=True) == [2, 3, 3, 7]
    # 断言对于输入 123456，multiple=True 参数时，factorint 函数返回 [2, 2, 2, 2, 2, 2, 3, 643]
    assert factorint(123456, multiple=True) == [2, 2, 2, 2, 2, 2, 3, 643]
    # 断言对于输入 5951757，multiple=True 参数时，factorint 函数返回 [3, 7, 29, 29, 337]
    assert factorint(5951757, multiple=True) == [3, 7, 29, 29, 337]
    # 断言对于输入 64015937，multiple=True 参数时，factorint 函数返回 [7993, 8009]
    assert factorint(64015937, multiple=True) == [7993, 8009]
    # 断言对于输入 2**(2**6) + 1，multiple=True 参数时，factorint 函数返回 [274177, 67280421310721]
    assert factorint(2**(2**6) + 1, multiple=True) == [274177, 67280421310721]

    # 断言对于输入 fac(1, evaluate=False)，factorint 函数返回空字典 {}
    assert factorint(fac(1, evaluate=False)) == {}
    # 断言对于输入 fac(7, evaluate=False)，factorint 函数返回 {2: 4, 3: 2, 5: 1, 7: 1}
    assert factorint(fac(7, evaluate=False)) == {2: 4, 3: 2, 5: 1, 7: 1}
    # 断言对于输入 fac(15, evaluate=False)，factorint 函数返回 {2: 11, 3: 6, 5: 3, 7: 2, 11: 1, 13: 1}
    assert factorint(fac(15, evaluate=False)) == {2: 11, 3: 6, 5: 3, 7: 2, 11: 1, 13: 1}
    # 断言对于输入 fac(20, evaluate=False)，factorint 函数返回 {2: 18, 3: 8, 5: 4, 7: 2, 11: 1, 13: 1, 17: 1, 19: 1}
    assert factorint(fac(20, evaluate=False)) == {2: 18, 3: 8, 5: 4, 7: 2, 11: 1, 13: 1, 17: 1
    # 断言：验证使用 Pollard Rho 算法计算 2^64 + 1 的因数分解结果与直接计算的结果相同
    assert factorint(2**64 + 1, use_trial=False) == factorint(2**64 + 1)
    
    # 循环：对于范围在 [0, 60000) 的每个整数 n，验证其质因数分解后再还原是否等于原始值 n
    for n in range(60000):
        assert multiproduct(factorint(n)) == n
    
    # 断言：验证使用 Pollard Rho 算法计算 2^64 + 1 的一个特定种子(seed=1)的因数分解结果是否等于 274177
    assert pollard_rho(2**64 + 1, seed=1) == 274177
    
    # 断言：验证对于质数 19，使用 Pollard Rho 算法的因数分解结果是否为 None
    assert pollard_rho(19, seed=1) is None
    
    # 断言：验证使用 factorint 函数对 3 进行质因数分解，限制最多返回 2 个因子
    assert factorint(3, limit=2) == {3: 1}
    
    # 断言：验证使用 factorint 函数对 12345 进行质因数分解
    assert factorint(12345) == {3: 1, 5: 1, 823: 1}
    
    # 断言：验证使用 factorint 函数对 12345 进行质因数分解，限制最多返回 3 个因子
    assert factorint(12345, limit=3) == {4115: 1, 3: 1}  # the 5 is greater than the limit
    
    # 断言：验证使用 factorint 函数对 1 进行质因数分解，限制最多返回 1 个因子
    assert factorint(1, limit=1) == {}
    
    # 断言：验证使用 factorint 函数对 0 进行质因数分解，返回结果中包含 0 的一个因子
    assert factorint(0, 3) == {0: 1}
    
    # 断言：验证使用 factorint 函数对 12 进行质因数分解，限制最多返回 1 个因子
    assert factorint(12, limit=1) == {12: 1}
    
    # 断言：验证使用 factorint 函数对 30 进行质因数分解，限制最多返回 2 个因子
    assert factorint(30, limit=2) == {2: 1, 15: 1}
    
    # 断言：验证使用 factorint 函数对 16 进行质因数分解，限制最多返回 2 个因子
    assert factorint(16, limit=2) == {2: 4}
    
    # 断言：验证使用 factorint 函数对 124 进行质因数分解，限制最多返回 3 个因子
    assert factorint(124, limit=3) == {2: 2, 31: 1}
    
    # 断言：验证使用 factorint 函数对 4*31^2 进行质因数分解，限制最多返回 3 个因子
    assert factorint(4*31**2, limit=3) == {2: 2, 31: 2}
    
    # 计算质数 p1、p2、p3，并验证它们的乘积的质因数分解结果
    p1 = nextprime(2**32)
    p2 = nextprime(2**16)
    p3 = nextprime(p2)
    assert factorint(p1*p2*p3) == {p1: 1, p2: 1, p3: 1}
    
    # 断言：验证使用 factorint 函数对 13*17*19 进行质因数分解，限制最多返回 15 个因子
    assert factorint(13*17*19, limit=15) == {13: 1, 17*19: 1}
    
    # 断言：验证使用 factorint 函数对 1951*15013*15053 进行质因数分解，限制最多返回 2000 个因子
    assert factorint(1951*15013*15053, limit=2000) == {225990689: 1, 1951: 1}
    
    # 断言：验证使用 factorint 函数对 primorial(17) + 1 进行质因数分解，不使用 Pollard PM1 算法
    assert factorint(primorial(17) + 1, use_pm1=0) == {int(19026377261): 1, 3467: 1, 277: 1, 105229: 1}
    
    # 注释：对于接近 78 位数的质数 a，计算下一个质数
    a = nextprime(2**2**8)  # 78 digits
    
    # 注释：计算比质数 a 大 2^(2^4) 的下一个质数，并将其存储在变量 b 中
    b = nextprime(a + 2**2**4)
    
    # 断言：验证在质数 a 和 b 的乘积上使用 capture 函数执行 factorint 函数，结果中包含 'Fermat'
    assert 'Fermat' in capture(lambda: factorint(a*b, verbose=1))
    
    # 断言：验证 pollard_rho 函数对非整数输入值引发 ValueError 异常
    raises(ValueError, lambda: pollard_rho(4))
    
    # 断言：验证 pollard_pm1 函数对非整数输入值引发 ValueError 异常
    raises(ValueError, lambda: pollard_pm1(3))
    
    # 断言：验证 pollard_pm1 函数在 B=2 的情况下对 10 进行因数分解时引发 ValueError 异常
    raises(ValueError, lambda: pollard_pm1(10, B=2))
    
    # 注释：计算三个接近质数的乘积，并存储在变量 n 中
    n = nextprime(2**16)*nextprime(2**17)*nextprime(1901)
    
    # 断言：验证在变量 n 上使用 capture 函数执行 factorint 函数，结果中包含 'with primes'
    assert 'with primes' in capture(lambda: factorint(n, verbose=1))
    
    # 注释：在计算的基础上，使用 capture 函数执行 factorint 函数，结果中包含 'with primes'
    capture(lambda: factorint(nextprime(2**16)*1012, verbose=1))
    
    # 注释：计算质数 n 的立方数，并使用 capture 函数执行 factorint 函数，观察完美幂次结束情况
    n = nextprime(2**17)
    capture(lambda: factorint(n**3, verbose=1))  # perfect power termination
    
    # 注释：在计算的基础上，使用 capture 函数执行 factorint 函数，观察完整因数分解消息
    capture(lambda: factorint(2*n, verbose=1))  # factoring complete msg
    
    # 注释：计算连续多个质数的乘积 n，并在 limit=1000 的情况下使用 capture 函数执行 factorint 函数
    n = nextprime(2**17)
    n *= nextprime(n)
    assert '1000' in capture(lambda: factorint(n, limit=1000, verbose=1))
    
    # 断言：验证计算的乘积 n 的质因数分解结果长度为 3
    assert len(factorint(n)) == 3
    
    # 断言：验证在限制最大因子为 p1 的情况下，计算的乘积 n 的质因数分解结果长度为 3
    assert len(factorint(n, limit=p1)) == 3
    
    # 注释：计算连续多个质数的乘积 n，并在 limit=2000 的情况下使用 capture 函数执行 factorint 函数
    n *= nextprime(2*n)
    
    # 断言：验证在限制最大因子为 2001 的情况下，计算的乘积 n 的
# 定义测试函数，用于测试 divisors 函数的各种输入情况
def test_divisors_and_divisor_count():
    # 断言当输入为 -1 时，divisors 返回 [1]
    assert divisors(-1) == [1]
    # 断言当输入为 0 时，divisors 返回空列表 []
    assert divisors(0) == []
    # 断言当输入为 1 时，divisors 返回 [1]
    assert divisors(1) == [1]
    # 断言当输入为 2 时，divisors 返回 [1, 2]
    assert divisors(2) == [1, 2]
    # 断言当输入为 3 时，divisors 返回 [1, 3]
    assert divisors(3) == [1, 3]
    # 断言当输入为 17 时，divisors 返回 [1, 17]
    assert divisors(17) == [1, 17]
    # 断言当输入为 10 时，divisors 返回 [1, 2, 5, 10]
    assert divisors(10) == [1, 2, 5, 10]
    # 断言当输入为 100 时，divisors 返回 [1, 2, 4, 5, 10, 20, 25, 50, 100]
    assert divisors(100) == [1, 2, 4, 5, 10, 20, 25, 50, 100]
    # 断言当输入为 101 时，divisors 返回 [1, 101]
    assert divisors(101) == [1, 101]
    # 断言当输入为 2 且 generator 参数为 True 时，divisors 返回的类型不是列表
    assert type(divisors(2, generator=True)) is not list

    # 断言当输入为 0 时，divisor_count 返回 0
    assert divisor_count(0) == 0
    # 断言当输入为 -1 时，divisor_count 返回 1
    assert divisor_count(-1) == 1
    # 断言当输入为 1 时，divisor_count 返回 1
    assert divisor_count(1) == 1
    # 断言当输入为 6 时，divisor_count 返回 4
    assert divisor_count(6) == 4
    # 断言当输入为 12 时，divisor_count 返回 6
    assert divisor_count(12) == 6

    # 断言 divisor_count(180, 3) 等于 divisor_count(180//3)
    assert divisor_count(180, 3) == divisor_count(180//3)
    # 断言 divisor_count(2*3*5, 7) 等于 0
    assert divisor_count(2*3*5, 7) == 0


# 定义测试函数，用于测试 proper_divisors 函数的各种输入情况
def test_proper_divisors_and_proper_divisor_count():
    # 断言当输入为 -1 时，proper_divisors 返回空列表 []
    assert proper_divisors(-1) == []
    # 断言当输入为 0 时，proper_divisors 返回空列表 []
    assert proper_divisors(0) == []
    # 断言当输入为 1 时，proper_divisors 返回空列表 []
    assert proper_divisors(1) == []
    # 断言当输入为 2 时，proper_divisors 返回 [1]
    assert proper_divisors(2) == [1]
    # 断言当输入为 3 时，proper_divisors 返回 [1]
    assert proper_divisors(3) == [1]
    # 断言当输入为 17 时，proper_divisors 返回 [1]
    assert proper_divisors(17) == [1]
    # 断言当输入为 10 时，proper_divisors 返回 [1, 2, 5]
    assert proper_divisors(10) == [1, 2, 5]
    # 断言当输入为 100 时，proper_divisors 返回 [1, 2, 4, 5, 10, 20, 25, 50]
    assert proper_divisors(100) == [1, 2, 4, 5, 10, 20, 25, 50]
    # 断言当输入为 1000000007 时，proper_divisors 返回 [1]
    assert proper_divisors(1000000007) == [1]
    # 断言当输入为 2 且 generator 参数为 True 时，proper_divisors 返回的类型不是列表
    assert type(proper_divisors(2, generator=True)) is not list

    # 断言当输入为 0 时，proper_divisor_count 返回 0
    assert proper_divisor_count(0) == 0
    # 断言当输入为 -1 时，proper_divisor_count 返回 0
    assert proper_divisor_count(-1) == 0
    # 断言当输入为 1 时，proper_divisor_count 返回 0
    assert proper_divisor_count(1) == 0
    # 断言当输入为 36 时，proper_divisor_count 返回 8
    assert proper_divisor_count(36) == 8
    # 断言当输入为 2*3*5 时，proper_divisor_count 返回 7
    assert proper_divisor_count(2*3*5) == 7


# 定义测试函数，用于测试 udivisors 函数的各种输入情况
def test_udivisors_and_udivisor_count():
    # 断言当输入为 -1 时，udivisors 返回 [1]
    assert udivisors(-1) == [1]
    # 断言当输入为 0 时，udivisors 返回空列表 []
    assert udivisors(0) == []
    # 断言当输入为 1 时，udivisors 返回 [1]
    assert udivisors(1) == [1]
    # 断言当输入为 2 时，udivisors 返回 [1, 2]
    assert udivisors(2) == [1, 2]
    # 断言当输入为 3 时，udivisors 返回 [1, 3]
    assert udivisors(3) == [1, 3]
    # 断言当输入为 17 时，udivisors 返回 [1, 17]
    assert udivisors(17) == [1, 17]
    # 断言当输入为 10 时，udivisors 返回 [1, 2, 5, 10]
    assert udivisors(10) == [1, 2, 5, 10]
    # 断言当输入为 100 时，udivisors 返回 [1, 4, 25, 100]
    assert udivisors(100) == [1, 4, 25, 100]
    # 断言当输入为 101 时，udivisors 返回 [1, 101]
    assert udivisors(101) == [1, 101]
    # 断言当输入为 1000 时，udivisors 返回 [1, 8, 125, 1000]
    assert udivisors(1000) == [1, 8, 125, 1000]
    # 断言当输入为 2 且 generator 参数为 True 时，udivisors 返回的类型不是列表
    assert type(udivisors(2, generator=True)) is not list

    # 断言当输入为 0 时，udivisor_count 返回 0
    assert udivisor_count(0) == 0
    # 断言当输入为 -1 时，udivisor_count 返回 1
    assert udivisor_count(-1) == 1
    # 断言当输入为 1 时，udivisor_count 返回 1
    assert udivisor_count(1) == 1
    # 断言当输入为 6 时，udivisor_count 返回 4
    assert udivisor_count(6) == 4
    # 断言当输入为 12 时，udivisor_count 返回 4
    assert udivisor_count(12) == 4

    # 断言 udivisor_count(180) 等于 8
    assert udivisor_count(180) == 8
    # 断言 udivisor_count(2*3*5*7) 等于 16
    assert udivisor_count(2*3*5*7) == 16


# 定义测试函数，用于测试 issue_6981 中的内容
def test_issue_6981():
    # 创建集合 S，包含 divisors(4) 和 div
    # 断言，检查调用 antidivisors 函数返回的结果是否符合预期
    assert antidivisors(14) == [3, 4, 9]
    
    # 断言，检查调用 antidivisors 函数返回的结果是否符合预期
    assert antidivisors(237) == [2, 5, 6, 11, 19, 25, 43, 95, 158]
    
    # 断言，检查调用 antidivisors 函数返回的结果是否符合预期
    assert antidivisors(12345) == [2, 6, 7, 10, 30, 1646, 3527, 4938, 8230]
    
    # 断言，检查调用 antidivisors 函数返回的结果是否符合预期
    assert antidivisors(393216) == [262144]
    
    # 断言，检查调用 antidivisors 函数返回的结果是否符合预期
    assert sorted(x for x in antidivisors(3*5*7, 1)) == [2, 6, 10, 11, 14, 19, 30, 42, 70]
    
    # 断言，检查调用 antidivisors 函数返回的结果是否符合预期
    assert antidivisors(1) == []
    
    # 断言，检查调用 antidivisors 函数的返回类型是否不是列表
    assert type(antidivisors(2, generator=True)) is not list
# 定义函数 `test_antidivisor_count`，用于测试 `antidivisor_count` 函数的多个断言
def test_antidivisor_count():
    # 断言当输入为 0 时，返回值为 0
    assert antidivisor_count(0) == 0
    # 断言当输入为 -1 时，返回值为 0
    assert antidivisor_count(-1) == 0
    # 断言当输入为 -4 时，返回值为 1
    assert antidivisor_count(-4) == 1
    # 断言当输入为 20 时，返回值为 3
    assert antidivisor_count(20) == 3
    # 断言当输入为 25 时，返回值为 5
    assert antidivisor_count(25) == 5
    # 断言当输入为 38 时，返回值为 7
    assert antidivisor_count(38) == 7
    # 断言当输入为 180 时，返回值为 6
    assert antidivisor_count(180) == 6
    # 断言当输入为 2*3*5 时，返回值为 3
    assert antidivisor_count(2*3*5) == 3


# 定义函数 `test_smoothness_and_smoothness_p`，用于测试 `smoothness` 和 `smoothness_p` 函数的多个断言
def test_smoothness_and_smoothness_p():
    # 断言 smoothness(1) 返回 (1, 1)
    assert smoothness(1) == (1, 1)
    # 断言 smoothness(2**4*3**2) 返回 (3, 16)
    assert smoothness(2**4*3**2) == (3, 16)

    # 断言 smoothness_p(10431, m=1) 返回特定的元组
    assert smoothness_p(10431, m=1) == \
        (1, [(3, (2, 2, 4)), (19, (1, 5, 5)), (61, (1, 31, 31))])
    # 断言 smoothness_p(10431) 返回特定的元组
    assert smoothness_p(10431) == \
        (-1, [(3, (2, 2, 2)), (19, (1, 3, 9)), (61, (1, 5, 5))])
    # 断言 smoothness_p(10431, power=1) 返回特定的元组
    assert smoothness_p(10431, power=1) == \
        (-1, [(3, (2, 2, 2)), (61, (1, 5, 5)), (19, (1, 3, 9))])
    # 断言 smoothness_p(21477639576571, visual=1) 返回特定的字符串
    assert smoothness_p(21477639576571, visual=1) == \
        'p**i=4410317**1 has p-1 B=1787, B-pow=1787\n' + \
        'p**i=4869863**1 has p-1 B=2434931, B-pow=2434931'


# 定义函数 `test_visual_factorint`，用于测试 `factorint` 函数的多个断言
def test_visual_factorint():
    # 断言 factorint(1, visual=1) 返回 1
    assert factorint(1, visual=1) == 1
    # 断言 factorint(42, visual=True) 返回类型为 Mul 的对象
    forty2 = factorint(42, visual=True)
    assert type(forty2) == Mul
    # 断言 factorint(42, visual=True) 的字符串表示符合预期
    assert str(forty2) == '2**1*3**1*7**1'
    # 断言 factorint(1, visual=True) 返回 S.One
    assert factorint(1, visual=True) is S.One
    # 断言 factorint(42**2, visual=True) 返回特定的 Mul 对象
    no = {"evaluate": False}
    assert factorint(42**2, visual=True) == Mul(Pow(2, 2, **no),
                                                Pow(3, 2, **no),
                                                Pow(7, 2, **no), **no)
    # 断言 factorint(-42, visual=True) 的 args 中包含 -1
    assert -1 in factorint(-42, visual=True).args


# 定义函数 `test_factorrat`，用于测试 `factorrat` 函数的多个断言
def test_factorrat():
    # 断言 factorrat(S(12)/1, visual=True) 返回特定的字符串
    assert str(factorrat(S(12)/1, visual=True)) == '2**2*3**1'
    # 断言 factorrat(Rational(1, 1), visual=True) 返回 '1'
    assert str(factorrat(Rational(1, 1), visual=True)) == '1'
    # 断言 factorrat(S(25)/14, visual=True) 返回特定的字符串
    assert str(factorrat(S(25)/14, visual=True)) == '5**2/(2*7)'
    # 断言 factorrat(Rational(25, 14), visual=True) 返回特定的字符串
    assert str(factorrat(Rational(25, 14), visual=True)) == '5**2/(2*7)'
    # 断言 factorrat(S(-25)/14/9, visual=True) 返回特定的字符串
    assert str(factorrat(S(-25)/14/9, visual=True)) == '-1*5**2/(2*3**2*7)'

    # 断言 factorrat(S(12)/1, multiple=True) 返回特定的列表
    assert factorrat(S(12)/1, multiple=True) == [2, 2, 3]
    # 断言 factorrat(Rational(1, 1), multiple=True) 返回空列表
    assert factorrat(Rational(1, 1), multiple=True) == []
    # 断言 factorrat(S(25)/14, multiple=True) 返回特定的列表
    assert factorrat(S(25)/14, multiple=True) == [Rational(1, 7), S.Half, 5, 5]
    # 断言 factorrat(Rational(25, 14), multiple=True) 返回特定的列表
    assert factorrat(Rational(25, 14), multiple=True) == [Rational(1, 7), S.Half, 5, 5]
    # 断言 factorrat(Rational(12, 1), multiple=True) 返回特定的列表
    assert factorrat(Rational(12, 1), multiple=True) == [2, 2, 3]
    # 断言 factorrat(S(-25)/14/9, multiple=True) 返回特定的列表
    assert factorrat(S(-25)/14/9, multiple=True) == \
        [-1, Rational(1, 7), Rational(1, 3), Rational(1, 3), S.Half, 5, 5]


# 定义函数 `test_visual_io`，用于测试 smoothness_p 和 factorint 函数在 visual 模式下的行为
def test_visual_io():
    # 定义别名 sm 和 fi 分别代表 smoothness_p 和 factorint 函数
    sm = smoothness_p
    fi = factorint

    # 使用 smoothness_p 函数的不同 visual 参数进行测试
    n = 124
    d = fi(n)
    m = fi(d, visual=True)
    t = sm(n)
    s = sm(t)
    for th in [d, s, t, n, m]:
        # 断言 smoothness_p(th, visual=True) 返回预期的 s
        assert sm(th, visual=True) == s
        # 断言 smoothness_p(th, visual=1) 返回预期的 s
        assert sm(th, visual=1) == s
    for th in [d, s, t, n, m]:
        # 断言 smoothness_p(th, visual=False) 返回预期的 t
        assert sm(th, visual=False) == t
    # 断言 smoothness_p(th, visual=None) 对于列表中的每个 th 返回预期的结果
    assert [sm(th, visual=None) for th in [d, s, t, n, m]] == [s, d, s, t, t]
    # 断言 smoothness_p(th, visual=2) 对于列表中的每个 th 返回预期的结果
    assert [sm(th, visual=2) for th in [d, s, t, n, m]] == [s, d, s, t, t]

    # 使用 factorint 函数的 visual 参数进行测试
    for th in [d, m, n]:
        #
    # 断言：验证调用 fi 函数时参数 visual=None，分别传入 d, m, n 时的返回结果是否为 [m, d, d]
    assert [fi(th, visual=None) for th in [d, m, n]] == [m, d, d]
    
    # 断言：验证调用 fi 函数时参数 visual=0，分别传入 d, m, n 时的返回结果是否为 [m, d, d]
    assert [fi(th, visual=0) for th in [d, m, n]] == [m, d, d]

    # 测试重新评估
    no = {"evaluate": False}
    
    # 断言：验证调用 sm 函数时传入 {4: 2}, visual=False 的返回结果是否等于 sm(16)
    assert sm({4: 2}, visual=False) == sm(16)
    
    # 构造参数为 Mul(*[Pow(k, v, **no) for k, v in {4: 2, 2: 6}.items()], **no)，visual=False，验证其返回结果是否等于 sm(2**10)
    assert sm(Mul(*[Pow(k, v, **no) for k, v in {4: 2, 2: 6}.items()], **no),
              visual=False) == sm(2**10)

    # 断言：验证调用 fi 函数时传入 {4: 2}, visual=False 的返回结果是否等于 fi(16)
    assert fi({4: 2}, visual=False) == fi(16)
    
    # 构造参数为 Mul(*[Pow(k, v, **no) for k, v in {4: 2, 2: 6}.items()], **no)，visual=False，验证其返回结果是否等于 fi(2**10)
    assert fi(Mul(*[Pow(k, v, **no) for k, v in {4: 2, 2: 6}.items()], **no),
              visual=False) == fi(2**10)
# 定义一个测试函数，用于验证核心函数的返回结果是否符合预期
def test_core():
    # 验证核心函数在给定的两个参数下返回的结果是否为预期值
    assert core(35**13, 10) == 42875
    # 验证核心函数在只有一个参数时返回的结果是否为预期值
    assert core(210**2) == 1
    # 验证核心函数在两个参数下返回的结果是否为预期值
    assert core(7776, 3) == 36
    # 验证核心函数在两个参数下返回的结果是否为预期值
    assert core(10**27, 22) == 10**5
    # 验证核心函数在只有一个参数时返回的结果是否为预期值
    assert core(537824) == 14
    # 验证核心函数在只有一个参数时返回的结果是否为预期值
    assert core(1, 6) == 1


# 定义一个测试函数，用于验证 _divisor_sigma 函数的不同参数组合下返回的结果是否符合预期
def test__divisor_sigma():
    # 验证 _divisor_sigma 函数在指定参数下返回的结果是否为预期值
    assert _divisor_sigma(23450) == 50592
    # 验证 _divisor_sigma 函数在指定参数下返回的结果是否为预期值
    assert _divisor_sigma(23450, 0) == 24
    # 验证 _divisor_sigma 函数在指定参数下返回的结果是否为预期值
    assert _divisor_sigma(23450, 1) == 50592
    # 验证 _divisor_sigma 函数在指定参数下返回的结果是否为预期值
    assert _divisor_sigma(23450, 2) == 730747500
    # 验证 _divisor_sigma 函数在指定参数下返回的结果是否为预期值
    assert _divisor_sigma(23450, 3) == 14666785333344
    # 创建一个列表，包含已知数列 A000005 的前几项
    A000005 = [1, 2, 2, 3, 2, 4, 2, 4, 3, 4, 2, 6, 2, 4, 4, 5, 2, 6, 2, 6, 4,
               4, 2, 8, 3, 4, 4, 6, 2, 8, 2, 6, 4, 4, 4, 9, 2, 4, 4, 8, 2, 8]
    # 遍历 A000005 列表，并验证 _divisor_sigma 函数在不同参数下返回的结果是否与预期一致
    for n, val in enumerate(A000005, 1):
        assert _divisor_sigma(n, 0) == val
    # 创建一个列表，包含已知数列 A000203 的前几项
    A000203 = [1, 3, 4, 7, 6, 12, 8, 15, 13, 18, 12, 28, 14, 24, 24, 31, 18,
               39, 20, 42, 32, 36, 24, 60, 31, 42, 40, 56, 30, 72, 32, 63, 48]
    # 遍历 A000203 列表，并验证 _divisor_sigma 函数在不同参数下返回的结果是否与预期一致
    for n, val in enumerate(A000203, 1):
        assert _divisor_sigma(n, 1) == val
    # 创建一个列表，包含已知数列 A001157 的前几项
    A001157 = [1, 5, 10, 21, 26, 50, 50, 85, 91, 130, 122, 210, 170, 250, 260,
               341, 290, 455, 362, 546, 500, 610, 530, 850, 651, 850, 820, 1050]
    # 遍历 A001157 列表，并验证 _divisor_sigma 函数在不同参数下返回的结果是否与预期一致
    for n, val in enumerate(A001157, 1):
        assert _divisor_sigma(n, 2) == val


# 定义一个测试函数，用于验证 mersenne_prime_exponent 函数的返回结果是否符合预期
def test_mersenne_prime_exponent():
    # 验证 mersenne_prime_exponent 函数在给定参数下返回的结果是否为预期值
    assert mersenne_prime_exponent(1) == 2
    # 验证 mersenne_prime_exponent 函数在给定参数下返回的结果是否为预期值
    assert mersenne_prime_exponent(4) == 7
    # 验证 mersenne_prime_exponent 函数在给定参数下返回的结果是否为预期值
    assert mersenne_prime_exponent(10) == 89
    # 验证 mersenne_prime_exponent 函数在给定参数下返回的结果是否为预期值
    assert mersenne_prime_exponent(25) == 21701
    # 验证 mersenne_prime_exponent 函数在给定参数下是否引发 ValueError 异常
    raises(ValueError, lambda: mersenne_prime_exponent(52))
    # 验证 mersenne_prime_exponent 函数在给定参数下是否引发 ValueError 异常
    raises(ValueError, lambda: mersenne_prime_exponent(0))


# 定义一个测试函数，用于验证 is_perfect 函数在不同参数下返回的结果是否符合预期
def test_is_perfect():
    # 验证 is_perfect 函数在给定参数下返回的结果是否为预期值
    assert is_perfect(-6) is False
    # 验证 is_perfect 函数在给定参数下返回的结果是否为预期值
    assert is_perfect(6) is True
    # 验证 is_perfect 函数在给定参数下返回的结果是否为预期值
    assert is_perfect(15) is False
    # 验证 is_perfect 函数在给定参数下返回的结果是否为预期值
    assert is_perfect(28) is True
    # 验证 is_perfect 函数在给定参数下返回的结果是否为预期值
    assert is_perfect(400) is False
    # 验证 is_perfect 函数在给定参数下返回的结果是否为预期值
    assert is_perfect(496) is True
    # 验证 is_perfect 函数在给定参数下返回的结果是否为预期值
    assert is_perfect(8128) is True
    # 验证 is_perfect 函数在给定参数下返回的结果是否为预期值
    assert is_perfect(10000) is False


# 定义一个测试函数，用于验证 is_abundant 函数在不同参数下返回的结果是否符合预期
def test_is_abundant():
    # 验证 is_abundant 函数在给定参数下返回的结果是否为预期值
    assert is_abundant(10) is False
    # 验证 is_abundant 函数在给定参数下返回的结果是否为预期值
    assert is_abundant(12) is True
    # 验证 is_abundant 函数在给定参数下返回的结果是否为预期值
    assert is_abundant(18) is True
    # 验证 is_abundant 函数在给定参数下返回的结果是否为预期值
    assert is_abundant(21) is False
    # 验证 is_abundant 函数在给定参数下返回的结果是否为预期值
    assert is_abundant(945) is True


# 定义一个测试函数，用于验证 is_deficient 函数在不同参数下返回的结果是否符合预期
def test_is_deficient():
    # 验证 is_deficient 函数在给定参数下返回的结果是否为预期值
    assert is_deficient(10) is True
    # 验证 is_deficient 函数在给定参数下返回的结果是否为预期值
    assert is_deficient(22) is True
    # 验证 is_deficient 函数在给定参数下返回的结果是否为预期值
    # 使用 pytest 的 raises 函数来测试函数 find_carmichael_numbers_in_range
    # 对于输入范围为 (-2, 2)，预期会引发 ValueError 异常
    raises(ValueError, lambda: find_carmichael_numbers_in_range(-2, 2))
    # 对于输入范围为 (22, 2)，预期会引发 ValueError 异常
    raises(ValueError, lambda: find_carmichael_numbers_in_range(22, 2))
# 测试寻找卡迈克尔数函数的结果是否符合预期
def test_find_first_n_carmichaels():
    # 测试当 n = 0 时的情况，预期返回空列表
    assert find_first_n_carmichaels(0) == []
    # 测试当 n = 1 时的情况，预期返回 [561]
    assert find_first_n_carmichaels(1) == [561]
    # 测试当 n = 2 时的情况，预期返回 [561, 1105]
    assert find_first_n_carmichaels(2) == [561, 1105]

# 测试数字重排算法 (Digital Root Algorithm, DRA) 的正确性
def test_dra():
    # 测试 dra(19, 12)，预期结果为 8
    assert dra(19, 12) == 8
    # 测试 dra(2718, 10)，预期结果为 9
    assert dra(2718, 10) == 9
    # 测试 dra(0, 22)，预期结果为 0
    assert dra(0, 22) == 0
    # 测试 dra(23456789, 10)，预期结果为 8
    assert dra(23456789, 10) == 8
    # 测试非法输入，应引发 ValueError 异常，输入 (24, -2)
    raises(ValueError, lambda: dra(24, -2))
    # 测试非法输入，应引发 ValueError 异常，输入 (24.2, 5)
    raises(ValueError, lambda: dra(24.2, 5))

# 测试数字重排模 (Digital Root Modulo, DRM) 的正确性
def test_drm():
    # 测试 drm(19, 12)，预期结果为 7
    assert drm(19, 12) == 7
    # 测试 drm(2718, 10)，预期结果为 2
    assert drm(2718, 10) == 2
    # 测试 drm(0, 15)，预期结果为 0
    assert drm(0, 15) == 0
    # 测试 drm(234161, 10)，预期结果为 6
    assert drm(234161, 10) == 6
    # 测试非法输入，应引发 ValueError 异常，输入 (24, -2)
    raises(ValueError, lambda: drm(24, -2))
    # 测试非法输入，应引发 ValueError 异常，输入 (11.6, 9)
    raises(ValueError, lambda: drm(11.6, 9))

# 测试 sympy 库中已弃用的数论符号函数的正确性
def test_deprecated_ntheory_symbolic_functions():
    from sympy.testing.pytest import warns_deprecated_sympy
    
    with warns_deprecated_sympy():
        # 测试 primenu(3)，应返回 1
        assert primenu(3) == 1
    with warns_deprecated_sympy():
        # 测试 primeomega(3)，应返回 1
        assert primeomega(3) == 1
    with warns_deprecated_sympy():
        # 测试 totient(3)，应返回 2
        assert totient(3) == 2
    with warns_deprecated_sympy():
        # 测试 reduced_totient(3)，应返回 2
        assert reduced_totient(3) == 2
    with warns_deprecated_sympy():
        # 测试 divisor_sigma(3)，应返回 4
        assert divisor_sigma(3) == 4
    with warns_deprecated_sympy():
        # 测试 udivisor_sigma(3)，应返回 4
        assert udivisor_sigma(3) == 4
```