# `D:\src\scipysrc\sympy\sympy\ntheory\tests\test_generate.py`

```
# 从 bisect 模块中导入 bisect 和 bisect_left 函数
from bisect import bisect, bisect_left

# 从 sympy 函数模块中导入 mobius 和 totient 函数
from sympy.functions.combinatorial.numbers import mobius, totient

# 从 sympy.ntheory.generate 模块中导入 sieve 和 Sieve 类
from sympy.ntheory.generate import sieve, Sieve

# 从 sympy.ntheory 模块中导入 isprime, randprime, nextprime, prevprime 函数等
from sympy.ntheory import isprime, randprime, nextprime, prevprime, \
    primerange, primepi, prime, primorial, composite, compositepi

# 从 sympy.ntheory.generate 模块中导入 cycle_length 和 _primepi 函数
from sympy.ntheory.generate import cycle_length, _primepi

# 从 sympy.ntheory.primetest 模块中导入 mr 函数
from sympy.ntheory.primetest import mr

# 从 sympy.testing.pytest 模块中导入 raises 函数
from sympy.testing.pytest import raises


# 定义测试函数 test_prime，用于测试 prime 函数
def test_prime():
    assert prime(1) == 2
    assert prime(2) == 3
    assert prime(5) == 11
    assert prime(11) == 31
    assert prime(57) == 269
    assert prime(296) == 1949
    assert prime(559) == 4051
    assert prime(3000) == 27449
    assert prime(4096) == 38873
    assert prime(9096) == 94321
    assert prime(25023) == 287341
    assert prime(10000000) == 179424673  # 检测边界情况，参见问题 #20951
    assert prime(99999999) == 2038074739
    raises(ValueError, lambda: prime(0))  # 检测异常情况，应抛出 ValueError 异常
    sieve.extend(3000)  # 扩展筛选器，增加到 3000
    assert prime(401) == 2749
    raises(ValueError, lambda: prime(-1))  # 检测异常情况，应抛出 ValueError 异常


# 定义测试函数 test__primepi，用于测试 _primepi 函数
def test__primepi():
    assert _primepi(-1) == 0
    assert _primepi(1) == 0
    assert _primepi(2) == 1
    assert _primepi(5) == 3
    assert _primepi(11) == 5
    assert _primepi(57) == 16
    assert _primepi(296) == 62
    assert _primepi(559) == 102
    assert _primepi(3000) == 430
    assert _primepi(4096) == 564
    assert _primepi(9096) == 1128
    assert _primepi(25023) == 2763
    assert _primepi(10**8) == 5761455
    assert _primepi(253425253) == 13856396
    assert _primepi(8769575643) == 401464322
    sieve.extend(3000)  # 扩展筛选器，增加到 3000
    assert _primepi(2000) == 303  # 检测边界情况


# 定义测试函数 test_composite，用于测试 composite 函数
def test_composite():
    from sympy.ntheory.generate import sieve  # 从 sympy.ntheory.generate 模块中导入 sieve
    sieve._reset()  # 重置筛选器状态
    assert composite(1) == 4
    assert composite(2) == 6
    assert composite(5) == 10
    assert composite(11) == 20
    assert composite(41) == 58
    assert composite(57) == 80
    assert composite(296) == 370
    assert composite(559) == 684
    assert composite(3000) == 3488
    assert composite(4096) == 4736
    assert composite(9096) == 10368
    assert composite(25023) == 28088
    sieve.extend(3000)  # 扩展筛选器，增加到 3000
    assert composite(1957) == 2300
    assert composite(2568) == 2998
    raises(ValueError, lambda: composite(0))  # 检测异常情况，应抛出 ValueError 异常


# 定义测试函数 test_compositepi，用于测试 compositepi 函数
def test_compositepi():
    assert compositepi(1) == 0
    assert compositepi(2) == 0
    assert compositepi(5) == 1
    assert compositepi(11) == 5
    assert compositepi(57) == 40
    assert compositepi(296) == 233
    assert compositepi(559) == 456
    assert compositepi(3000) == 2569
    assert compositepi(4096) == 3531
    assert compositepi(9096) == 7967
    assert compositepi(25023) == 22259
    assert compositepi(10**8) == 94238544
    assert compositepi(253425253) == 239568856
    assert compositepi(8769575643) == 8368111320
    sieve.extend(3000)  # 扩展筛选器，增加到 3000
    assert compositepi(2321) == 1976  # 检测边界情况


# 定义测试函数 test_generate，用于测试 nextprime 函数
def test_generate():
    from sympy.ntheory.generate import sieve  # 从 sympy.ntheory.generate 模块中导入 sieve
    sieve._reset()  # 重置筛选器状态
    assert nextprime(-4) == 2
    assert nextprime(2) == 3
    assert nextprime(5) == 7
    # 确保 nextprime 函数能够正确返回大于给定数的下一个质数
    assert nextprime(12) == 13
    # 确保 prevprime 函数能够正确返回小于给定数的前一个质数
    assert prevprime(3) == 2
    assert prevprime(7) == 5
    assert prevprime(13) == 11
    assert prevprime(19) == 17
    assert prevprime(20) == 19

    # 将筛选器扩展到包含至少给定数量的质数
    sieve.extend_to_no(9)
    # 确保筛选器的最后一个元素是预期的质数
    assert sieve._list[-1] == 23

    # 确保筛选器的最后一个元素小于31
    assert sieve._list[-1] < 31
    # 确保31在筛选器中
    assert 31 in sieve

    # 确保 nextprime 函数能够正确返回大于给定数的下一个质数
    assert nextprime(90) == 97
    # 确保 nextprime 函数能够处理非常大的输入
    assert nextprime(10**40) == (10**40 + 121)
    
    # 预定义的质数列表
    primelist = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31,
                 37, 41, 43, 47, 53, 59, 61, 67, 71, 73,
                 79, 83, 89, 97, 101, 103, 107, 109, 113,
                 127, 131, 137, 139, 149, 151, 157, 163,
                 167, 173, 179, 181, 191, 193, 197, 199,
                 211, 223, 227, 229, 233, 239, 241, 251,
                 257, 263, 269, 271, 277, 281, 283, 293]
    
    # 遍历质数列表的组合情况，确保 nextprime 函数能够按预期返回下一个质数
    for i in range(len(primelist) - 2):
        for j in range(2, len(primelist) - i):
            assert nextprime(primelist[i], j) == primelist[i + j]
            # 在特定条件下，确保 nextprime 函数能够正确处理减去1的情况
            if 3 < i:
                assert nextprime(primelist[i] - 1, j) == primelist[i + j - 1]
    
    # 确保在非法参数输入时，nextprime 函数能够引发 ValueError 异常
    raises(ValueError, lambda: nextprime(2, 0))
    raises(ValueError, lambda: nextprime(2, -1))
    
    # 确保 prevprime 函数能够正确返回小于给定数的前一个质数
    assert prevprime(97) == 89
    # 确保 prevprime 函数能够处理非常大的输入
    assert prevprime(10**40) == (10**40 - 17)

    # 确保在非法参数输入时，Sieve 类的初始化能够引发 ValueError 异常
    raises(ValueError, lambda: Sieve(0))
    raises(ValueError, lambda: Sieve(-1))
    
    # 遍历筛选器初始化的不同间隔，确保筛选器功能正常
    for sieve_interval in [1, 10, 11, 1_000_000]:
        s = Sieve(sieve_interval=sieve_interval)
        # 遍历特定范围内的素数，确保筛选器生成的素数列表与预期相符
        for head in range(s._list[-1] + 1, (s._list[-1] + 1)**2, 2):
            for tail in range(head + 1, (s._list[-1] + 1)**2):
                A = list(s._primerange(head, tail))
                B = primelist[bisect(primelist, head):bisect_left(primelist, tail)]
                assert A == B
        # 遍历特定值，确保在给定的值扩展筛选器后，其生成的素数列表与预期相符
        for k in range(s._list[-1], primelist[-1] - 1, 2):
            s = Sieve(sieve_interval=sieve_interval)
            s.extend(k)
            assert list(s._list) == primelist[:bisect(primelist, k)]
            s.extend(primelist[-1])
            assert list(s._list) == primelist
    
    # 确保在给定的范围内，primerange 函数能够返回预期的质数列表
    assert list(sieve.primerange(10, 1)) == []
    assert list(sieve.primerange(5, 9)) == [5, 7]
    sieve._reset(prime=True)
    assert list(sieve.primerange(2, 13)) == [2, 3, 5, 7, 11]
    assert list(sieve.primerange(13)) == [2, 3, 5, 7, 11]
    assert list(sieve.primerange(8)) == [2, 3, 5, 7]
    assert list(sieve.primerange(-2)) == []
    assert list(sieve.primerange(29)) == [2, 3, 5, 7, 11, 13, 17, 19, 23]
    assert list(sieve.primerange(34)) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

    # 确保在给定的范围内，totientrange 函数能够返回预期的欧拉函数值列表
    assert list(sieve.totientrange(5, 15)) == [4, 2, 6, 4, 6, 4, 10, 4, 12, 6]
    sieve._reset(totient=True)
    assert list(sieve.totientrange(3, 13)) == [2, 2, 4, 2, 6, 4, 6, 4, 10, 4]
    assert list(sieve.totientrange(900, 1000)) == [totient(x) for x in range(900, 1000)]
    assert list(sieve.totientrange(0, 1)) == []
    assert list(sieve.totientrange(1, 2)) == [1]

    # 确保在给定的范围内，mobiusrange 函数能够返回预期的莫比乌斯函数值列表
    assert list(sieve.mobiusrange(5, 15)) == [-1, 1, -1, 0, 0, 1, -1, 0, -1, 1]
    sieve._reset(mobius=True)
    # 确定素数筛选对象的 mobiusrange 方法返回的结果是否与预期列表匹配
    assert list(sieve.mobiusrange(3, 13)) == [-1, 0, -1, 1, -1, 0, 0, 1, -1, 0]
    # 确定素数筛选对象的 mobiusrange 方法返回的结果是否与指定范围内的 mobius 函数结果列表匹配
    assert list(sieve.mobiusrange(1050, 1100)) == [mobius(x) for x in range(1050, 1100)]
    # 确定素数筛选对象的 mobiusrange 方法在范围为 [0, 1) 时是否返回空列表
    assert list(sieve.mobiusrange(0, 1)) == []
    # 确定素数筛选对象的 mobiusrange 方法在范围为 [1, 2) 时是否返回 [1]
    assert list(sieve.mobiusrange(1, 2)) == [1]

    # 确定素数范围对象的 primerange 方法在逆序范围 [10, 1) 时是否返回空列表
    assert list(primerange(10, 1)) == []
    # 确定素数范围对象的 primerange 方法在范围 [2, 7) 时是否返回 [2, 3, 5]
    assert list(primerange(2, 7)) == [2, 3, 5]
    # 确定素数范围对象的 primerange 方法在范围 [2, 10) 时是否返回 [2, 3, 5, 7]
    assert list(primerange(2, 10)) == [2, 3, 5, 7]
    # 确定素数范围对象的 primerange 方法在范围 [1050, 1100) 时是否返回指定列表
    assert list(primerange(1050, 1100)) == [1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097]

    # 创建素数筛选对象
    s = Sieve()
    # 使用循环验证自定义范围内的素数生成是否一致
    for i in range(30, 2350, 376):
        for j in range(2, 5096, 1139):
            A = list(s.primerange(i, i + j))
            B = list(primerange(i, i + j))
            assert A == B

    # 创建素数筛选对象
    s = Sieve()
    # 重置素数筛选对象，并扩展到第一个素数为 13
    sieve._reset(prime=True)
    sieve.extend(13)
    # 使用循环验证不同范围内的素数生成是否一致
    for i in range(200):
        for j in range(i, 200):
            A = list(s.primerange(i, j))
            B = list(primerange(i, j))
            assert A == B
    # 继续扩展素数筛选对象，使其包含前 1000 个素数
    sieve.extend(1000)
    # 使用多组范围验证不同边界条件下的素数生成是否一致
    for a, b in [(901, 1103),   # a < 1000 < b < 1000**2
                 (806, 1002007),  # a < 1000 < 1000**2 < b
                 (2000, 30001),   # 1000 < a < b < 1000**2
                 (100005, 1010001),  # 1000 < a < 1000**2 < b
                 (1003003, 1005000),  # 1000**2 < a < b
                 ]:
        assert list(primerange(a, b)) == list(s.primerange(a, b))

    # 重置素数筛选对象，继续扩展到第一个素数为 100000
    sieve._reset(prime=True)
    sieve.extend(100000)
    # 确认扩展后的素数列表中无重复元素
    assert len(sieve._list) == len(set(sieve._list))
    # 创建素数筛选对象，并验证索引为 10 的素数是否为 29
    s = Sieve()
    assert s[10] == 29

    # 确认大于 2 的下一个素数为 5
    assert nextprime(2, 2) == 5

    # 验证计算欧拉函数 totient(0) 是否引发 ValueError
    raises(ValueError, lambda: totient(0))

    # 验证计算 primorial(0) 是否引发 ValueError
    raises(ValueError, lambda: primorial(0))

    # 确认使用指定参数时的循环长度函数计算结果
    assert mr(1, [2]) is False

    # 创建函数 func，并验证循环长度函数在指定初始值为 4 时的首个结果是否为 (6, 3)
    func = lambda i: (i**2 + 1) % 51
    assert next(cycle_length(func, 4)) == (6, 3)
    # 验证循环长度函数在指定初始值为 4 时的所有计算结果列表是否与预期一致
    assert list(cycle_length(func, 4, values=True)) == \
        [4, 17, 35, 2, 5, 26, 14, 44, 50, 2, 5, 26, 14]
    # 继续验证循环长度函数在指定初始值为 4 且最大迭代次数为 5 时的首个结果是否为 (5, None)
    assert next(cycle_length(func, 4, nmax=5)) == (5, None)
    # 验证循环长度函数在指定初始值为 4 且最大迭代次数为 5 时的所有计算结果列表是否与预期一致
    assert list(cycle_length(func, 4, nmax=5, values=True)) == \
        [4, 17, 35, 2, 5]

    # 继续扩展素数筛选对象，使其包含前 3000 个素数
    sieve.extend(3000)
    # 确认大于 2968 的下一个素数为 2969
    assert nextprime(2968) == 2969
    # 确认小于 2930 的前一个素数为 2927
    assert prevprime(2930) == 2927
    # 验证 prevprime 函数在参数为 1 时是否引发 ValueError
    raises(ValueError, lambda: prevprime(1))
    # 验证 prevprime 函数在参数为 -4 时是否引发 ValueError
    raises(ValueError, lambda: prevprime(-4))
# 定义测试函数 test_randprime，用于测试 randprime 函数的各种情况
def test_randprime():
    # 确保当上限小于下限时返回 None
    assert randprime(10, 1) is None
    # 确保当下限为负数时返回 None
    assert randprime(3, -3) is None
    # 确保在范围内正确返回最小质数
    assert randprime(2, 3) == 2
    # 确保在范围内正确返回最小质数
    assert randprime(1, 3) == 2
    # 确保在范围内正确返回质数
    assert randprime(3, 5) == 3
    # 确保抛出异常，当下限大于上限时
    raises(ValueError, lambda: randprime(-12, -2))
    # 确保抛出异常，当下限为非正数时
    raises(ValueError, lambda: randprime(-10, 0))
    # 确保抛出异常，当范围内无质数时
    raises(ValueError, lambda: randprime(20, 22))
    # 确保抛出异常，当范围为非正数时
    raises(ValueError, lambda: randprime(0, 2))
    # 确保抛出异常，当范围为单一数时
    raises(ValueError, lambda: randprime(1, 2))
    # 循环测试一系列较大的范围，确保返回的质数在指定范围内且为质数
    for a in [100, 300, 500, 250000]:
        for b in [100, 300, 500, 250000]:
            p = randprime(a, a + b)
            assert a <= p < (a + b) and isprime(p)


# 定义测试函数 test_primorial，用于测试 primorial 函数的各种情况
def test_primorial():
    # 确保计算第一个素数积的正确性
    assert primorial(1) == 2
    # 确保计算第一个素数积的正确性
    assert primorial(1, nth=0) == 1
    # 确保计算前两个素数积的正确性
    assert primorial(2) == 6
    # 确保计算前两个素数积的正确性
    assert primorial(2, nth=0) == 2
    # 确保计算前四个素数积的正确性
    assert primorial(4, nth=0) == 6


# 定义测试函数 test_search，用于测试 sieve 对象的搜索功能
def test_search():
    # 确保数字 2 存在于素数筛中
    assert 2 in sieve
    # 确保浮点数 2.1 不存在于素数筛中
    assert 2.1 not in sieve
    # 确保数字 1 不存在于素数筛中
    assert 1 not in sieve
    # 确保超大数 2 的 1000 次方不存在于素数筛中
    assert 2**1000 not in sieve
    # 确保调用 search 方法时抛出异常，因为 search 方法已弃用
    raises(ValueError, lambda: sieve.search(1))


# 定义测试函数 test_sieve_slice，用于测试 sieve 对象的切片功能
def test_sieve_slice():
    # 确保索引为 5 的素数正确
    assert sieve[5] == 11
    # 确保切片操作返回预期的素数列表
    assert list(sieve[5:10]) == [sieve[x] for x in range(5, 10)]
    # 确保带步长的切片操作返回预期的素数列表
    assert list(sieve[5:10:2]) == [sieve[x] for x in range(5, 10, 2)]
    # 确保索引为 1 到 5 的素数列表正确
    assert list(sieve[1:5]) == [2, 3, 5, 7]
    # 确保切片到开头的操作抛出异常
    raises(IndexError, lambda: sieve[:5])
    # 确保索引为 0 的操作抛出异常
    raises(IndexError, lambda: sieve[0])
    # 确保索引为 0 到 5 的切片操作抛出异常
    raises(IndexError, lambda: sieve[0:5])


# 定义测试函数 test_sieve_iter，用于测试 sieve 对象的迭代功能
def test_sieve_iter():
    values = []
    # 迭代素数筛，将小于 7 的素数收集到 values 中
    for value in sieve:
        if value > 7:
            break
        values.append(value)
    # 确保收集到的素数列表与预期的一致
    assert values == list(sieve[1:5])


# 定义测试函数 test_sieve_repr，用于测试 sieve 对象的字符串表示功能
def test_sieve_repr():
    # 确保在字符串表示中包含关键字 "sieve"
    assert "sieve" in repr(sieve)
    # 确保在字符串表示中包含关键字 "prime"
    assert "prime" in repr(sieve)


# 定义测试函数 test_deprecated_ntheory_symbolic_functions，用于测试弃用的数论符号函数的行为
def test_deprecated_ntheory_symbolic_functions():
    from sympy.testing.pytest import warns_deprecated_sympy

    with warns_deprecated_sympy():
        # 确保调用已弃用的 primepi 函数时抛出警告并返回正确结果
        assert primepi(0) == 0
```