# `D:\src\scipysrc\sympy\sympy\external\tests\test_ntheory.py`

```
# 从 itertools 模块导入 permutations 函数
# 从 sympy.external.ntheory 模块导入多个函数
# 从 sympy.testing.pytest 模块导入 raises 函数
from itertools import permutations

from sympy.external.ntheory import (bit_scan1, remove, bit_scan0, is_fermat_prp,
                                    is_euler_prp, is_strong_prp, gcdext, _lucas_sequence,
                                    is_fibonacci_prp, is_lucas_prp, is_selfridge_prp,
                                    is_strong_lucas_prp, is_strong_selfridge_prp,
                                    is_bpsw_prp, is_strong_bpsw_prp)
from sympy.testing.pytest import raises


# 定义测试函数 test_bit_scan1，测试 bit_scan1 函数的各种情况
def test_bit_scan1():
    assert bit_scan1(0) is None  # 当输入为 0 时，应返回 None
    assert bit_scan1(1) == 0  # 当输入为 1 时，应返回 0
    assert bit_scan1(-1) == 0  # 当输入为 -1 时，应返回 0
    assert bit_scan1(2) == 1  # 当输入为 2 时，应返回 1
    assert bit_scan1(7) == 0  # 当输入为 7 时，应返回 0
    assert bit_scan1(-7) == 0  # 当输入为 -7 时，应返回 0
    # 对于每一个 i 在范围 [0, 100) 中，验证 bit_scan1 函数对于 (1 << i) 的返回值是否为 i
    for i in range(100):
        assert bit_scan1(1 << i) == i
        assert bit_scan1((1 << i) * 31337) == i
    # 对于每一个 i 在范围 [0, 500) 中，构造 n = (1 << 500) + (1 << i)，验证 bit_scan1 函数的返回值是否为 i
    for i in range(500):
        n = (1 << 500) + (1 << i)
        assert bit_scan1(n) == i
    assert bit_scan1(1 << 1000001) == 1000001  # 当输入为 2^1000001 时，应返回 1000001
    assert bit_scan1((1 << 273956)*7**37) == 273956  # 对于较大的输入，验证 bit_scan1 函数的返回值
    # issue 12709：验证 bit_scan1 函数在输入为大数的负数时的返回值是否正确
    for i in range(1, 10):
        big = 1 << i
        assert bit_scan1(-big) == bit_scan1(big)


# 定义测试函数 test_bit_scan0，测试 bit_scan0 函数的各种情况
def test_bit_scan0():
    assert bit_scan0(-1) is None  # 当输入为 -1 时，应返回 None
    assert bit_scan0(0) == 0  # 当输入为 0 时，应返回 0
    assert bit_scan0(1) == 1  # 当输入为 1 时，应返回 1
    assert bit_scan0(-2) == 0  # 当输入为 -2 时，应返回 0


# 定义测试函数 test_remove，测试 remove 函数的各种情况
def test_remove():
    raises(ValueError, lambda: remove(1, 1))  # 当输入不合法时，应抛出 ValueError 异常
    assert remove(0, 3) == (0, 0)  # 当输入合法时，验证 remove 函数的返回值是否正确
    # 对于每一个 f 在范围 [2, 10) 中，每一个 y 在范围 [2, 1000) 中，每一个 z 在列表 [1, 17, 101, 1009] 中，
    # 验证 remove 函数的返回值是否符合预期
    for f in range(2, 10):
        for y in range(2, 1000):
            for z in [1, 17, 101, 1009]:
                assert remove(z*f**y, f) == (z, y)


# 定义测试函数 test_gcdext，测试 gcdext 函数的各种情况
def test_gcdext():
    assert gcdext(0, 0) == (0, 0, 0)  # 当输入为 (0, 0) 时，应返回 (0, 0, 0)
    assert gcdext(3, 0) == (3, 1, 0)  # 当输入为 (3, 0) 时，应返回 (3, 1, 0)
    assert gcdext(0, 4) == (4, 0, 1)  # 当输入为 (0, 4) 时，应返回 (4, 0, 1)
    
    # 对于每一个 n 在范围 [1, 10) 中，验证 gcdext 函数的返回值是否符合预期
    for n in range(1, 10):
        assert gcdext(n, 1) == gcdext(-n, 1) == (1, 0, 1)
        assert gcdext(n, -1) == gcdext(-n, -1) == (1, 0, -1)
        assert gcdext(n, n) == gcdext(-n, n) == (n, 0, 1)
        assert gcdext(n, -n) == gcdext(-n, -n) == (n, 0, -1)
    
    # 对于每一个 n 在范围 [2, 10) 中，验证 gcdext 函数的返回值是否符合预期
    for n in range(2, 10):
        assert gcdext(1, n) == gcdext(1, -n) == (1, 1, 0)
        assert gcdext(-1, n) == gcdext(-1, -n) == (1, -1, 0)
    
    # 对于排列组合的每一对 a, b，验证 gcdext 函数的返回值是否符合扩展欧几里得算法的定义
    for a, b in permutations([2**5, 3, 5, 7**2, 11], 2):
        g, x, y = gcdext(a, b)
        assert g == a*x + b*y == 1


# 定义测试函数 test_is_fermat_prp，测试 is_fermat_prp 函数的各种情况
def test_is_fermat_prp():
    # 非法输入：验证 is_fermat_prp 函数在输入为非法值时是否抛出 ValueError 异常
    raises(ValueError, lambda: is_fermat_prp(0, 10))
    raises(ValueError, lambda: is_fermat_prp(5, 1))

    # n = 1 时，应返回 False
    assert not is_fermat_prp(1, 3)

    # n 是素数时，应返回 True
    assert is_fermat_prp(2, 4)
    assert is_fermat_prp(3, 2)
    assert is_fermat_prp(11, 3)
    assert is_fermat_prp(2**31-1, 5)

    # A001567 中的伪素数，应返回 True
    pseudorpime = [341, 561, 645, 1105, 1387, 1729, 1905, 2047,
                   2465, 2701, 2821, 3277, 4033, 4369, 4371, 4681]
    for n in pseudorpime:
        assert is_fermat_prp(n, 2)

    # A020136 中的伪素数，应返回 True
    pseudorpime = [15, 85, 91, 341, 435, 451, 561, 645, 703, 1105,
                   1247, 1271, 1387, 1581, 1695, 1729, 1891, 1905]
    for n in pseudorpime:
        assert is_fermat_prp(n, 4)


# 定义测试函数 test_is_euler_prp，暂未完整实现
def test_is_euler_prp():
    # 在后续的实现中，会对 is_euler_prp 函数进行详细测试
    pass
    # 调用 is_euler_prp 函数，验证是否会引发 ValueError 异常，预期参数 (0, 10)
    raises(ValueError, lambda: is_euler_prp(0, 10))
    # 调用 is_euler_prp 函数，验证是否会引发 ValueError 异常，预期参数 (5, 1)
    raises(ValueError, lambda: is_euler_prp(5, 1))

    # 对于 n = 1，预期返回 False
    assert not is_euler_prp(1, 3)

    # 当 n 是素数时，预期返回 True
    assert is_euler_prp(2, 4)
    assert is_euler_prp(3, 2)
    assert is_euler_prp(11, 3)
    assert is_euler_prp(2**31-1, 5)

    # 对于 A047713 中的每个伪素数，预期调用 is_euler_prp 函数返回 True，k = 2
    pseudorpime = [561, 1105, 1729, 1905, 2047, 2465, 3277, 4033,
                   4681, 6601, 8321, 8481, 10585, 12801, 15841]
    for n in pseudorpime:
        assert is_euler_prp(n, 2)

    # 对于 A048950 中的每个伪素数，预期调用 is_euler_prp 函数返回 True，k = 3
    pseudorpime = [121, 703, 1729, 1891, 2821, 3281, 7381, 8401,
                   8911, 10585, 12403, 15457, 15841, 16531, 18721]
    for n in pseudorpime:
        assert is_euler_prp(n, 3)
def test_is_strong_prp():
    # 检查无效输入情况
    raises(ValueError, lambda: is_strong_prp(0, 10))  # 期望引发值错误异常
    raises(ValueError, lambda: is_strong_prp(5, 1))   # 期望引发值错误异常

    # n = 1 的情况
    assert not is_strong_prp(1, 3)  # 断言 n=1 时返回 False

    # n 是素数的情况
    assert is_strong_prp(2, 4)  # 断言 n=2, a=4 是强伪素数
    assert is_strong_prp(3, 2)  # 断言 n=3, a=2 是强伪素数
    assert is_strong_prp(11, 3)  # 断言 n=11, a=3 是强伪素数
    assert is_strong_prp(2**31-1, 5)  # 断言 n=2^31-1, a=5 是强伪素数

    # A001262 序列的测试
    pseudorpime = [2047, 3277, 4033, 4681, 8321, 15841, 29341,
                   42799, 49141, 52633, 65281, 74665, 80581]
    for n in pseudorpime:
        assert is_strong_prp(n, 2)  # 断言每个数 n 在 a=2 时是强伪素数

    # A020229 序列的测试
    pseudorpime = [121, 703, 1891, 3281, 8401, 8911, 10585, 12403,
                   16531, 18721, 19345, 23521, 31621, 44287, 47197]
    for n in pseudorpime:
        assert is_strong_prp(n, 3)  # 断言每个数 n 在 a=3 时是强伪素数


def test_lucas_sequence():
    def lucas_u(P, Q, length):
        # 初始化 lucas_u 数组
        array = [0] * length
        array[1] = 1
        for k in range(2, length):
            array[k] = P * array[k - 1] - Q * array[k - 2]
        return array

    def lucas_v(P, Q, length):
        # 初始化 lucas_v 数组
        array = [0] * length
        array[0] = 2
        array[1] = P
        for k in range(2, length):
            array[k] = P * array[k - 1] - Q * array[k - 2]
        return array

    length = 20
    for P in range(-10, 10):
        for Q in range(-10, 10):
            D = P**2 - 4*Q
            if D == 0:
                continue
            us = lucas_u(P, Q, length)
            vs = lucas_v(P, Q, length)
            for n in range(3, 100, 2):
                for k in range(length):
                    U, V, Qk = _lucas_sequence(n, P, Q, k)
                    assert U == us[k] % n  # 断言 U 和 lucas_u 生成的数组对 n 取模相同
                    assert V == vs[k] % n  # 断言 V 和 lucas_v 生成的数组对 n 取模相同
                    assert pow(Q, k, n) == Qk  # 断言 pow(Q, k, n) 等于 _lucas_sequence 返回的 Qk


def test_is_fibonacci_prp():
    # 检查无效输入情况
    raises(ValueError, lambda: is_fibonacci_prp(3, 2, 1))  # 期望引发值错误异常
    raises(ValueError, lambda: is_fibonacci_prp(3, -5, 1))  # 期望引发值错误异常
    raises(ValueError, lambda: is_fibonacci_prp(3, 5, 2))  # 期望引发值错误异常
    raises(ValueError, lambda: is_fibonacci_prp(0, 5, -1))  # 期望引发值错误异常

    # n = 1 的情况
    assert not is_fibonacci_prp(1, 3, 1)  # 断言 n=1, P=3, Q=1 返回 False

    # n 是素数的情况
    assert is_fibonacci_prp(2, 5, 1)  # 断言 n=2, P=5, Q=1 是 Fibonacci 强伪素数
    assert is_fibonacci_prp(3, 6, -1)  # 断言 n=3, P=6, Q=-1 是 Fibonacci 强伪素数
    assert is_fibonacci_prp(11, 7, 1)  # 断言 n=11, P=7, Q=1 是 Fibonacci 强伪素数
    assert is_fibonacci_prp(2**31-1, 8, -1)  # 断言 n=2^31-1, P=8, Q=-1 是 Fibonacci 强伪素数

    # A005845 序列的测试
    pseudorpime = [705, 2465, 2737, 3745, 4181, 5777, 6721,
                   10877, 13201, 15251, 24465, 29281, 34561]
    for n in pseudorpime:
        assert is_fibonacci_prp(n, 1, -1)  # 断言每个数 n 在 P=1, Q=-1 时是 Fibonacci 强伪素数


def test_is_lucas_prp():
    # 检查无效输入情况
    raises(ValueError, lambda: is_lucas_prp(3, 2, 1))  # 期望引发值错误异常
    raises(ValueError, lambda: is_lucas_prp(0, 5, -1))  # 期望引发值错误异常
    raises(ValueError, lambda: is_lucas_prp(15, 3, 1))  # 期望引发值错误异常

    # n = 1 的情况
    assert not is_lucas_prp(1, 3, 1)  # 断言 n=1, P=3, Q=1 返回 False

    # n 是素数的情况
    assert is_lucas_prp(2, 5, 2)  # 断言 n=2, P=5, Q=2 是 Lucas 强伪素数
    assert is_lucas_prp(3, 6, -1)  # 断言 n=3, P=6, Q=-1 是 Lucas 强伪素数
    assert is_lucas_prp(11, 7, 5)  # 断言 n=11, P=7, Q=5 是 Lucas 强伪素数
    assert is_lucas_prp(2**31-1, 8, -3)  # 断言 n=2^31-1, P=8, Q=-3 是 Lucas 强伪素数

    # A081264 序列的测试
    pseudorpime = [323, 377, 1891, 3827, 4181, 5777, 6601, 6721,
                   8149, 10877, 11663, 13201, 13981, 15251, 17119]
    for n in pseudorpime:
        assert is_lucas_prp(n, 3, 1)  # 断言每个数 n 在 P=3, Q=1 时是 Lucas 强伪素数
    # 遍历 pseudorpime 列表中的每个元素 n
    for n in pseudorpime:
        # 断言 n 是 Lucas 强伪素数，使用参数 1 和 -1 进行检查
        assert is_lucas_prp(n, 1, -1)
def test_is_selfridge_prp():
    # 验证无效输入是否会引发 ValueError 异常
    raises(ValueError, lambda: is_selfridge_prp(0))

    # 对于 n = 1，应该返回 False
    assert not is_selfridge_prp(1)

    # 对于素数 n，应该返回 True
    assert is_selfridge_prp(2)
    assert is_selfridge_prp(3)
    assert is_selfridge_prp(11)
    assert is_selfridge_prp(2**31-1)

    # A217120 序列中的伪素数
    pseudorpime = [323, 377, 1159, 1829, 3827, 5459, 5777, 9071,
                   9179, 10877, 11419, 11663, 13919, 14839, 16109]
    for n in pseudorpime:
        # 验证 A217120 序列中的每个伪素数是否满足 is_selfridge_prp 函数
        assert is_selfridge_prp(n)


def test_is_strong_lucas_prp():
    # 验证无效输入是否会引发 ValueError 异常
    raises(ValueError, lambda: is_strong_lucas_prp(3, 2, 1))
    raises(ValueError, lambda: is_strong_lucas_prp(0, 5, -1))
    raises(ValueError, lambda: is_strong_lucas_prp(15, 3, 1))

    # 对于 n = 1，应该返回 False
    assert not is_strong_lucas_prp(1, 3, 1)

    # 对于素数 n，应该返回 True
    assert is_strong_lucas_prp(2, 5, 2)
    assert is_strong_lucas_prp(3, 6, -1)
    assert is_strong_lucas_prp(11, 7, 5)
    assert is_strong_lucas_prp(2**31-1, 8, -3)


def test_is_strong_selfridge_prp():
    # 验证无效输入是否会引发 ValueError 异常
    raises(ValueError, lambda: is_strong_selfridge_prp(0))

    # 对于 n = 1，应该返回 False
    assert not is_strong_selfridge_prp(1)

    # 对于素数 n，应该返回 True
    assert is_strong_selfridge_prp(2)
    assert is_strong_selfridge_prp(3)
    assert is_strong_selfridge_prp(11)
    assert is_strong_selfridge_prp(2**31-1)

    # A217255 序列中的伪素数
    pseudorpime = [5459, 5777, 10877, 16109, 18971, 22499, 24569,
                   25199, 40309, 58519, 75077, 97439, 100127, 113573]
    for n in pseudorpime:
        # 验证 A217255 序列中的每个伪素数是否满足 is_strong_selfridge_prp 函数
        assert is_strong_selfridge_prp(n)


def test_is_bpsw_prp():
    # 验证无效输入是否会引发 ValueError 异常
    raises(ValueError, lambda: is_bpsw_prp(0))

    # 对于 n = 1，应该返回 False
    assert not is_bpsw_prp(1)

    # 对于素数 n，应该返回 True
    assert is_bpsw_prp(2)
    assert is_bpsw_prp(3)
    assert is_bpsw_prp(11)
    assert is_bpsw_prp(2**31-1)


def test_is_strong_bpsw_prp():
    # 验证无效输入是否会引发 ValueError 异常
    raises(ValueError, lambda: is_strong_bpsw_prp(0))

    # 对于 n = 1，应该返回 False
    assert not is_strong_bpsw_prp(1)

    # 对于素数 n，应该返回 True
    assert is_strong_bpsw_prp(2)
    assert is_strong_bpsw_prp(3)
    assert is_strong_bpsw_prp(11)
    assert is_strong_bpsw_prp(2**31-1)
```