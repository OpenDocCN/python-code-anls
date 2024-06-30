# `D:\src\scipysrc\sympy\sympy\ntheory\tests\test_digits.py`

```
# 导入必要的函数和模块：count_digits、digits、is_palindromic 来自 sympy.ntheory 模块；num_digits 来自 sympy.core.intfunc 模块；raises 来自 sympy.testing.pytest 模块
from sympy.ntheory import count_digits, digits, is_palindromic
from sympy.core.intfunc import num_digits
from sympy.testing.pytest import raises


# 定义测试函数 test_num_digits，用于测试 num_digits 函数
def test_num_digits():
    # 检查 num_digits 函数在不同参数下的返回值是否符合预期
    assert num_digits(2, 2) == 2
    assert num_digits(2**48 - 1, 2) == 48
    assert num_digits(1000, 10) == 4
    assert num_digits(125, 5) == 4
    assert num_digits(100, 16) == 2
    assert num_digits(-1000, 10) == 4
    # 对于一定范围内的 base 和 e 值，检查 num_digits 函数的返回是否正确
    for base in range(2, 100):
        for e in range(1, 100):
            n = base**e
            assert num_digits(n, base) == e + 1
            assert num_digits(n + 1, base) == e + 1
            assert num_digits(n - 1, base) == e


# 定义测试函数 test_digits，用于测试 digits 函数
def test_digits():
    # 检查 digits 函数在不同基数下的返回是否与 format 函数生成的结果一致
    assert all(digits(n, 2)[1:] == [int(d) for d in format(n, 'b')]
                for n in range(20))
    assert all(digits(n, 8)[1:] == [int(d) for d in format(n, 'o')]
                for n in range(20))
    assert all(digits(n, 16)[1:] == [int(d, 16) for d in format(n, 'x')]
                for n in range(20))
    # 检查在特定情况下 digits 函数的返回是否正确
    assert digits(2345, 34) == [34, 2, 0, 33]
    assert digits(384753, 71) == [71, 1, 5, 23, 4]
    assert digits(93409, 10) == [10, 9, 3, 4, 0, 9]
    assert digits(-92838, 11) == [-11, 6, 3, 8, 2, 9]
    assert digits(35, 10) == [10, 3, 5]
    assert digits(35, 10, 3) == [10, 0, 3, 5]
    assert digits(-35, 10, 4) == [-10, 0, 0, 3, 5]
    # 检查 digits 函数在不合法输入时是否引发 ValueError 异常
    raises(ValueError, lambda: digits(2, 2, 1))


# 定义测试函数 test_count_digits，用于测试 count_digits 函数
def test_count_digits():
    # 检查 count_digits 函数在不同基数下的返回值是否符合预期
    assert count_digits(55, 2) == {1: 5, 0: 1}
    assert count_digits(55, 10) == {5: 2}
    # 检查 count_digits 函数返回的字典类型及其值是否符合预期
    n = count_digits(123)
    assert n[4] == 0 and type(n[4]) is int


# 定义测试函数 test_is_palindromic，用于测试 is_palindromic 函数
def test_is_palindromic():
    # 检查 is_palindromic 函数在不同参数下的返回值是否符合预期
    assert is_palindromic(-11)
    assert is_palindromic(11)
    assert is_palindromic(0o121, 8)
    assert not is_palindromic(123)
```