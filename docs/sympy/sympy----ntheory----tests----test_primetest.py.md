# `D:\src\scipysrc\sympy\sympy\ntheory\tests\test_primetest.py`

```
# 导入 math 模块中的 gcd 函数
from math import gcd

# 导入 sympy.ntheory.generate 模块中的 Sieve 和 sieve 函数
# 导入 sympy.ntheory.primetest 模块中的多个函数
from sympy.ntheory.generate import Sieve, sieve
from sympy.ntheory.primetest import (mr, _lucas_extrastrong_params, is_lucas_prp, is_square,
                                     is_strong_lucas_prp, is_extra_strong_lucas_prp,
                                     proth_test, isprime, is_euler_pseudoprime,
                                     is_gaussian_prime, is_fermat_pseudoprime, is_euler_jacobi_pseudoprime,
                                     MERSENNE_PRIME_EXPONENTS, _lucas_lehmer_primality_test,
                                     is_mersenne_prime)

# 导入 sympy.testing.pytest 模块中的 slow 和 raises 函数
# 导入 sympy.core.numbers 模块中的 I 和 Float 类
from sympy.testing.pytest import slow, raises
from sympy.core.numbers import I, Float


# 测试函数：检查 Fermat 伪素数判断函数的正确性
def test_is_fermat_pseudoprime():
    assert is_fermat_pseudoprime(5, 1)
    assert is_fermat_pseudoprime(9, 1)


# 测试函数：检查 Euler 伪素数判断函数的正确性
def test_euler_pseudoprimes():
    assert is_euler_pseudoprime(13, 1)
    assert is_euler_pseudoprime(15, 1)
    assert is_euler_pseudoprime(17, 6)
    assert is_euler_pseudoprime(101, 7)
    assert is_euler_pseudoprime(1009, 10)
    assert is_euler_pseudoprime(11287, 41)

    # 检查参数错误时是否能够引发 ValueError
    raises(ValueError, lambda: is_euler_pseudoprime(0, 4))
    raises(ValueError, lambda: is_euler_pseudoprime(3, 0))
    raises(ValueError, lambda: is_euler_pseudoprime(15, 6))

    # 检查一系列已知的 Euler 伪素数
    euler_prp = [341, 561, 1105, 1729, 1905, 2047, 2465, 3277,
                 4033, 4681, 5461, 6601, 8321, 8481, 10261, 10585]
    for p in euler_prp:
        assert is_euler_pseudoprime(p, 2)

    euler_prp = [121, 703, 1729, 1891, 2821, 3281, 7381, 8401, 8911, 10585,
                 12403, 15457, 15841, 16531, 18721, 19345, 23521, 24661, 28009]
    for p in euler_prp:
        assert is_euler_pseudoprime(p, 3)

    absolute_euler_prp = [1729, 2465, 15841, 41041, 46657, 75361,
                          162401, 172081, 399001, 449065, 488881]
    for p in absolute_euler_prp:
        for a in range(2, p):
            if gcd(a, p) != 1:
                continue
            assert is_euler_pseudoprime(p, a)


# 测试函数：检查 Euler-Jacobi 伪素数判断函数的正确性
def test_is_euler_jacobi_pseudoprime():
    assert is_euler_jacobi_pseudoprime(11, 1)
    assert is_euler_jacobi_pseudoprime(15, 1)


# 测试函数：检查 Lucas 强伪素数生成参数的正确性
def test_lucas_extrastrong_params():
    assert _lucas_extrastrong_params(3) == (5, 3, 1)
    assert _lucas_extrastrong_params(5) == (12, 4, 1)
    assert _lucas_extrastrong_params(7) == (5, 3, 1)
    assert _lucas_extrastrong_params(9) == (0, 0, 0)
    assert _lucas_extrastrong_params(11) == (21, 5, 1)
    assert _lucas_extrastrong_params(59) == (32, 6, 1)
    assert _lucas_extrastrong_params(479) == (117, 11, 1)


# 测试函数：检查额外强 Lucas 伪素数判断函数的正确性
def test_is_extra_strong_lucas_prp():
    assert is_extra_strong_lucas_prp(4) == False
    assert is_extra_strong_lucas_prp(989) == True
    assert is_extra_strong_lucas_prp(10877) == True
    assert is_extra_strong_lucas_prp(9) == False
    assert is_extra_strong_lucas_prp(16) == False
    assert is_extra_strong_lucas_prp(169) == False

# 标记为 slow，测试一般目的函数的正确性
@slow
def test_prps():
    # 生成一个列表，包含所有从 1 到 10^5 范围内的奇合数（即除以2余1且不是素数的数）
    oddcomposites = [n for n in range(1, 10**5) if
        n % 2 and not isprime(n)]

    # 断言：奇合数列表的所有元素之和应等于 2045603465
    assert sum(oddcomposites) == 2045603465

    # 断言：使用 Miller-Rabin 素性测试中的基数为 [2]，得到的奇合数列表
    assert [n for n in oddcomposites if mr(n, [2])] == [
        2047, 3277, 4033, 4681, 8321, 15841, 29341, 42799, 49141,
        52633, 65281, 74665, 80581, 85489, 88357, 90751]

    # 断言：使用 Miller-Rabin 素性测试中的基数为 [3]，得到的奇合数列表
    assert [n for n in oddcomposites if mr(n, [3])] == [
        121, 703, 1891, 3281, 8401, 8911, 10585, 12403, 16531,
        18721, 19345, 23521, 31621, 44287, 47197, 55969, 63139,
        74593, 79003, 82513, 87913, 88573, 97567]

    # 断言：使用 Miller-Rabin 素性测试中的基数为 [325]，得到的奇合数列表
    assert [n for n in oddcomposites if mr(n, [325])] == [
        9, 25, 27, 49, 65, 81, 325, 341, 343, 697, 1141, 2059,
        2149, 3097, 3537, 4033, 4681, 4941, 5833, 6517, 7987, 8911,
        12403, 12913, 15043, 16021, 20017, 22261, 23221, 24649,
        24929, 31841, 35371, 38503, 43213, 44173, 47197, 50041,
        55909, 56033, 58969, 59089, 61337, 65441, 68823, 72641,
        76793, 78409, 85879]

    # 断言：所有奇合数中，不存在可以通过大整数素性测试（基数为 9345883071009581737）为素数的数
    assert not any(mr(n, [9345883071009581737]) for n in oddcomposites)

    # 断言：所有奇合数中，满足 Lucas 强伪素数条件的数的列表
    assert [n for n in oddcomposites if is_lucas_prp(n)] == [
        323, 377, 1159, 1829, 3827, 5459, 5777, 9071, 9179, 10877,
        11419, 11663, 13919, 14839, 16109, 16211, 18407, 18971,
        19043, 22499, 23407, 24569, 25199, 25877, 26069, 27323,
        32759, 34943, 35207, 39059, 39203, 39689, 40309, 44099,
        46979, 47879, 50183, 51983, 53663, 56279, 58519, 60377,
        63881, 69509, 72389, 73919, 75077, 77219, 79547, 79799,
        82983, 84419, 86063, 90287, 94667, 97019, 97439]

    # 断言：所有奇合数中，满足 Lucas 强伪素数条件的数的列表
    assert [n for n in oddcomposites if is_strong_lucas_prp(n)] == [
        5459, 5777, 10877, 16109, 18971, 22499, 24569, 25199, 40309,
        58519, 75077, 97439]

    # 断言：所有奇合数中，满足额外强 Lucas 式伪素数条件的数的列表
    assert [n for n in oddcomposites if is_extra_strong_lucas_prp(n)
            ] == [
        989, 3239, 5777, 10877, 27971, 29681, 30739, 31631, 39059,
        72389, 73919, 75077]
def test_proth_test():
    # Proth number
    A080075 = [3, 5, 9, 13, 17, 25, 33, 41, 49, 57, 65,
               81, 97, 113, 129, 145, 161, 177, 193]
    # Proth prime
    A080076 = [3, 5, 13, 17, 41, 97, 113, 193]

    for n in range(200):
        # 如果 n 在 Proth number 列表中，则检查 proth_test(n) 的返回结果是否与 n 是否在 Proth prime 列表中相符
        if n in A080075:
            assert proth_test(n) == (n in A080076)
        else:
            # 否则，断言调用 proth_test(n) 会引发 ValueError 异常
            raises(ValueError, lambda: proth_test(n))


def test_lucas_lehmer_primality_test():
    # 对于范围在 3 到 100 内的素数 p，检查其是否在 MERSENNE_PRIME_EXPONENTS 列表中
    for p in sieve.primerange(3, 100):
        assert _lucas_lehmer_primality_test(p) == (p in MERSENNE_PRIME_EXPONENTS)


def test_is_mersenne_prime():
    # 检查一些特定数是否为 Mersenne 素数
    assert is_mersenne_prime(-3) is False
    assert is_mersenne_prime(3) is True
    assert is_mersenne_prime(10) is False
    assert is_mersenne_prime(127) is True
    assert is_mersenne_prime(511) is False
    assert is_mersenne_prime(131071) is True
    assert is_mersenne_prime(2147483647) is True


def test_isprime():
    s = Sieve()
    s.extend(100000)
    ps = set(s.primerange(2, 100001))
    for n in range(100001):
        # 检查 n 是否在素数集合 ps 中，与 isprime(n) 的结果一致
        assert (n in ps) == isprime(n)
    # 检查一些大整数是否为素数
    assert isprime(179424673)
    assert isprime(20678048681)
    assert isprime(1968188556461)
    assert isprime(2614941710599)
    assert isprime(65635624165761929287)
    assert isprime(1162566711635022452267983)
    assert isprime(77123077103005189615466924501)
    assert isprime(3991617775553178702574451996736229)
    assert isprime(273952953553395851092382714516720001799)
    assert isprime(int('''
531137992816767098689588206552468627329593117727031923199444138200403\
559860852242739162502265229285668889329486246501015346579337652707239\
409519978766587351943831270835393219031728127'''))

    # 检查一些 Mersenne 素数
    assert isprime(2**61 - 1)
    assert isprime(2**89 - 1)
    assert isprime(2**607 - 1)
    # 但不是所有 Mersenne 数都是素数
    assert not isprime(2**601 - 1)

    # 检查几个伪素数
    #-------------
    # 对于某些小基数
    assert not isprime(2152302898747)
    assert not isprime(3474749660383)
    assert not isprime(341550071728321)
    assert not isprime(3825123056546413051)
    # 在基数集合 [2, 3, 7, 61, 24251] 下通过测试
    assert not isprime(9188353522314541)
    # 一些大例子
    assert not isprime(877777777777777777777777)
    # 推测的 psi_12，来源于 http://mathworld.wolfram.com/StrongPseudoprime.html
    assert not isprime(318665857834031151167461)
    # 推测的 psi_17，来源于 http://mathworld.wolfram.com/StrongPseudoprime.html
    assert not isprime(564132928021909221014087501701)
    # Arnault's 1993 数字；其因子如下
    # 400958216639499605418306452084546853005188166041132508774506\
    # 204738003217070119624271622319159721973358216316508535816696\
    # 9145233813917169287527980445796800452592031836601
    assert not isprime(int('''
803837457453639491257079614341942108138837688287558145837488917522297\
427376533365218650233616396004545791504202360320876656996676098728404\
# 定义一个名为`test_is_prime`的测试函数，用于测试`is_prime`函数的不同输入情况
def test_is_prime():
    # 该数是贝尔纳·阿尔诺（Bernard Arnault）1995年的数值，可以分解为特定形式的素数乘积
    # 具体分解方式为 p1*(313*(p1 - 1) + 1)*(353*(p1 - 1) + 1)，其中 p1 是以下素数的一个
    # 296744956686855105501541746429053327307719917998530433509950\
    # 755312768387531717701995942385964281211880336647542183455624\
    # 93168782883
    assert not is_prime(int('''
396540823292873879185086916685732826776177102938969773947016708230428\
687109997439976544144845341155872450633409279022275296229414984230688\
1685404326457534018329786111298960644845216191652872597534901'''))
    
    # 将一个数加入素数筛选列表，扩展至 3000
    sieve.extend(3000)
    
    # 验证一个素数
    assert is_prime(2819)
    
    # 验证一个非素数
    assert not is_prime(2931)
    
    # 使用 lambda 表达式验证一个浮点数是否引发 ValueError 异常
    raises(ValueError, lambda: is_prime(2.0))
    
    # 使用 lambda 表达式验证一个 SymPy 的 Float 类型是否引发 ValueError 异常
    raises(ValueError, lambda: is_prime(Float(2)))


# 定义一个名为`test_is_square`的测试函数，用于测试`is_square`函数的不同输入情况
def test_is_square():
    # 验证 0 到 24 中的数值中哪些是平方数，应该是 [0, 1, 4, 9, 16]
    assert [i for i in range(25) if is_square(i)] == [0, 1, 4, 9, 16]

    # 验证非平方数的情况
    assert not is_square(60 ** 3)
    assert not is_square(60 ** 5)
    assert not is_square(84 ** 7)
    assert not is_square(105 ** 9)
    assert not is_square(120 ** 3)


# 定义一个名为`test_is_gaussianprime`的测试函数，用于测试`is_gaussian_prime`函数的不同输入情况
def test_is_gaussianprime():
    # 验证高斯整数是否为素数的情况
    assert is_gaussian_prime(7*I)
    assert is_gaussian_prime(7)
    assert is_gaussian_prime(2 + 3*I)
    
    # 验证高斯整数不是素数的情况
    assert not is_gaussian_prime(2 + 2*I)
```