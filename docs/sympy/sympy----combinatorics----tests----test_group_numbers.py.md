# `D:\src\scipysrc\sympy\sympy\combinatorics\tests\test_group_numbers.py`

```
# 导入必要的函数和类从 sympy 库
from sympy.combinatorics.group_numbers import (is_nilpotent_number,
    is_abelian_number, is_cyclic_number, _holder_formula, groups_count)
# 导入 sympy 库中的数论相关函数
from sympy.ntheory.factor_ import factorint
from sympy.ntheory.generate import prime
# 导入 raises 函数用于测试异常
from sympy.testing.pytest import raises
# 导入 randprime 函数从 sympy 库
from sympy import randprime

# 定义测试函数 test_is_nilpotent_number()
def test_is_nilpotent_number():
    # 断言对于给定的数值，is_nilpotent_number 函数应返回 False
    assert is_nilpotent_number(21) == False
    # 断言对于随机素数的 12 次方，is_nilpotent_number 函数应返回 True
    assert is_nilpotent_number(randprime(1, 30)**12) == True
    # 断言对于负数调用 is_nilpotent_number 函数应引发 ValueError 异常
    raises(ValueError, lambda: is_nilpotent_number(-5))

    # A056867 序列中包含的数值列表
    A056867    = [1, 2, 3, 4, 5, 7, 8, 9, 11, 13, 15, 16, 17, 19,
               23, 25, 27, 29, 31, 32, 33, 35, 37, 41, 43, 45,
               47, 49, 51, 53, 59, 61, 64, 65, 67, 69, 71, 73,
               77, 79, 81, 83, 85, 87, 89, 91, 95, 97, 99]
    # 遍历范围为 1 到 100 的每个数值 n
    for n in range(1, 100):
        # 断言 is_nilpotent_number(n) 的返回值与 n 是否在 A056867 序列中匹配
        assert is_nilpotent_number(n) == (n in A056867)

# 定义测试函数 test_is_abelian_number()
def test_is_abelian_number():
    # 断言对于给定的数值，is_abelian_number 函数应返回 True
    assert is_abelian_number(4) == True
    # 断言对于随机素数的平方，is_abelian_number 函数应返回 True
    assert is_abelian_number(randprime(1, 2000)**2) == True
    # 断言对于位于 1000 到 100000 之间的随机素数，is_abelian_number 函数应返回 True
    assert is_abelian_number(randprime(1000, 100000)) == True
    # 断言对于特定数值，is_abelian_number 函数应返回 False
    assert is_abelian_number(60) == False
    assert is_abelian_number(24) == False
    # 断言对于负数调用 is_abelian_number 函数应引发 ValueError 异常
    raises(ValueError, lambda: is_abelian_number(-5))

    # A051532 序列中包含的数值列表
    A051532 = [1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 19, 23, 25,
               29, 31, 33, 35, 37, 41, 43, 45, 47, 49, 51, 53,
               59, 61, 65, 67, 69, 71, 73, 77, 79, 83, 85, 87,
               89, 91, 95, 97, 99]
    # 遍历范围为 1 到 100 的每个数值 n
    for n in range(1, 100):
        # 断言 is_abelian_number(n) 的返回值与 n 是否在 A051532 序列中匹配
        assert is_abelian_number(n) == (n in A051532)

# A003277 序列中包含的数值列表
A003277 = [1, 2, 3, 5, 7, 11, 13, 15, 17, 19, 23, 29,
           31, 33, 35, 37, 41, 43, 47, 51, 53, 59, 61,
           65, 67, 69, 71, 73, 77, 79, 83, 85, 87, 89,
           91, 95, 97]

# 定义测试函数 test_is_cyclic_number()
def test_is_cyclic_number():
    # 断言对于给定的数值，is_cyclic_number 函数应返回 True
    assert is_cyclic_number(15) == True
    # 断言对于随机素数的平方，is_cyclic_number 函数应返回 False
    assert is_cyclic_number(randprime(1, 2000)**2) == False
    # 断言对于位于 1000 到 100000 之间的随机素数，is_cyclic_number 函数应返回 True
    assert is_cyclic_number(randprime(1000, 100000)) == True
    # 断言对于特定数值，is_cyclic_number 函数应返回 False
    assert is_cyclic_number(4) == False
    # 断言对于负数调用 is_cyclic_number 函数应引发 ValueError 异常
    raises(ValueError, lambda: is_cyclic_number(-5))

    # 遍历范围为 1 到 100 的每个数值 n
    for n in range(1, 100):
        # 断言 is_cyclic_number(n) 的返回值与 n 是否在 A003277 序列中匹配
        assert is_cyclic_number(n) == (n in A003277)

# 定义测试函数 test_holder_formula()
def test_holder_formula():
    # 断言对于半素数 {3, 5}，_holder_formula 函数应返回 1
    assert _holder_formula({3, 5}) == 1
    # 断言对于半素数 {5, 11}，_holder_formula 函数应返回 2
    assert _holder_formula({5, 11}) == 2
    # 对于 A003277 中的每个数值 n，_holder_formula 函数应始终返回 1
    for n in A003277:
        assert _holder_formula(set(factorint(n).keys())) == 1
    # 对于其他情况，_holder_formula 函数应返回 12
    assert _holder_formula({2, 3, 5, 7}) == 12

# 定义测试函数 test_groups_count()
def test_groups_count():
    # A000001 序列中包含的数值列表
    A000001 = [0, 1, 1, 1, 2, 1, 2, 1, 5, 2, 2, 1, 5, 1,
               2, 1, 14, 1, 5, 1, 5, 2, 2, 1, 15, 2, 2,
               5, 4, 1, 4, 1, 51, 1, 2, 1, 14, 1, 2, 2,
               14, 1, 6, 1, 4, 2, 2, 1, 52, 2, 5, 1, 5,
               1, 15, 2, 13, 2, 2, 1, 13, 1, 2, 4, 267,
               1, 4, 1, 5, 1, 4, 1, 50, 1, 2, 3, 4, 1,
               6, 1, 52, 15, 2, 1, 15, 1, 2, 1, 12, 1,
               10, 1, 4, 2]
    # 遍历范围为 1 到 A000001 列表的长度的每个数值 n
    for n in range(1, len(A000001)):
        try:
            # 断言 groups_count(n) 的返回值与 A000001[n
    # 对于序列 A000679 中的每个指数 e，验证 groups_count 函数计算出的结果是否等于 A000679[e]
    for e in range(1, len(A000679)):
        assert groups_count(2**e) == A000679[e]

    # 定义序列 A090091，并验证 groups_count 函数计算出的结果是否等于 A090091[e]
    A090091 = [1, 1, 2, 5, 15, 67, 504, 9310, 1396077, 5937876645]
    for e in range(1, len(A090091)):
        assert groups_count(3**e) == A090091[e]

    # 定义序列 A090130，并验证 groups_count 函数计算出的结果是否等于 A090130[e]
    A090130 = [1, 1, 2, 5, 15, 77, 684, 34297]
    for e in range(1, len(A090130)):
        assert groups_count(5**e) == A090130[e]

    # 定义序列 A090140，并验证 groups_count 函数计算出的结果是否等于 A090140[e]
    A090140 = [1, 1, 2, 5, 15, 83, 860, 113147]
    for e in range(1, len(A090140)):
        assert groups_count(7**e) == A090140[e]

    # 定义序列 A232105，并验证 groups_count 函数计算出的结果是否等于 A232105[i]
    A232105 = [51, 67, 77, 83, 87, 97, 101, 107, 111, 125, 131,
               145, 149, 155, 159, 173, 183, 193, 203, 207, 217]
    for i in range(len(A232105)):
        assert groups_count(prime(i+1)**5) == A232105[i]

    # 定义序列 A232106，并验证 groups_count 函数计算出的结果是否等于 A232106[i]
    A232106 = [267, 504, 684, 860, 1192, 1476, 1944, 2264, 2876,
               4068, 4540, 6012, 7064, 7664, 8852, 10908, 13136]
    for i in range(len(A232106)):
        assert groups_count(prime(i+1)**6) == A232106[i]

    # 定义序列 A232107，并验证 groups_count 函数计算出的结果是否等于 A232107[i]
    A232107 = [2328, 9310, 34297, 113147, 750735, 1600573,
               5546909, 9380741, 23316851, 71271069, 98488755]
    for i in range(len(A232107)):
        assert groups_count(prime(i+1)**7) == A232107[i]
```