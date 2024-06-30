# `D:\src\scipysrc\sympy\sympy\ntheory\tests\test_egyptian_fraction.py`

```
# 导入必要的函数和类
from sympy.core.numbers import Rational
from sympy.ntheory.egyptian_fraction import egyptian_fraction
from sympy.core.add import Add
from sympy.testing.pytest import raises
from sympy.core.random import random_complex_number

# 定义测试函数 test_egyptian_fraction
def test_egyptian_fraction():
    # 定义内部辅助函数 test_equality，用于测试分数 r 是否等于其埃及分数表示
    def test_equality(r, alg="Greedy"):
        return r == Add(*[Rational(1, i) for i in egyptian_fraction(r, alg)])
    
    # 生成一个随机复数有理数 r，确保其分数形式
    r = random_complex_number(a=0, c=1, b=0, d=0, rational=True)
    # 断言 test_equality 返回 True
    assert test_equality(r)
    
    # 以下是一系列断言语句，验证特定有理数的埃及分数表示是否正确
    assert egyptian_fraction(Rational(4, 17)) == [5, 29, 1233, 3039345]
    assert egyptian_fraction(Rational(7, 13), "Greedy") == [2, 26]
    assert egyptian_fraction(Rational(23, 101), "Greedy") == \
        [5, 37, 1438, 2985448, 40108045937720]
    assert egyptian_fraction(Rational(18, 23), "Takenouchi") == \
        [2, 6, 12, 35, 276, 2415]
    assert egyptian_fraction(Rational(5, 6), "Graham Jewett") == \
        [6, 7, 8, 9, 10, 42, 43, 44, 45, 56, 57, 58, 72, 73, 90, 1806, 1807,
         1808, 1892, 1893, 1980, 3192, 3193, 3306, 5256, 3263442, 3263443,
         3267056, 3581556, 10192056, 10650056950806]
    assert egyptian_fraction(Rational(5, 6), "Golomb") == [2, 6, 12, 20, 30]
    assert egyptian_fraction(Rational(5, 121), "Golomb") == [25, 1225, 3577, 7081, 11737]
    raises(ValueError, lambda: egyptian_fraction(Rational(-4, 9)))
    assert egyptian_fraction(Rational(8, 3), "Golomb") == [1, 2, 3, 4, 5, 6, 7,
                                                           14, 574, 2788, 6460,
                                                           11590, 33062, 113820]
    assert egyptian_fraction(Rational(355, 113)) == [1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                     10, 11, 12, 27, 744, 893588,
                                                     1251493536607,
                                                     20361068938197002344405230]

# 定义测试函数 test_input
def test_input():
    # 定义测试输入数据集合 r
    r = (2,3), Rational(2, 3), (Rational(2), Rational(3))
    # 遍历每种算法类型 m 和每个输入数据集合 i 进行测试
    for m in ["Greedy", "Graham Jewett", "Takenouchi", "Golomb"]:
        for i in r:
            # 获取输入数据集合 i 的埃及分数表示 d
            d = egyptian_fraction(i, m)
            # 断言 d 中所有元素都是整数
            assert all(i.is_Integer for i in d)
            # 如果 m 是 "Graham Jewett"，则验证 d 是否等于特定值 [3, 4, 12]
            if m == "Graham Jewett":
                assert d == [3, 4, 12]
            else:
                assert d == [2, 6]
    # 验证特定有理数的埃及分数表示是否正确，同时确保所有元素都是整数
    d = egyptian_fraction(Rational(5, 3))
    assert d == [1, 2, 6] and all(i.is_Integer for i in d)
```