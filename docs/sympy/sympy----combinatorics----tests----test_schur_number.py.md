# `D:\src\scipysrc\sympy\sympy\combinatorics\tests\test_schur_number.py`

```
# 导入必要的库和模块
from sympy.core import S, Rational  # 导入SymPy库中的S和Rational
from sympy.combinatorics.schur_number import schur_partition, SchurNumber  # 导入Schur分割和Schur数相关的函数和类
from sympy.core.random import _randint  # 导入SymPy库中的随机整数生成函数_randint
from sympy.testing.pytest import raises  # 导入SymPy库中的raises函数用于测试异常
from sympy.core.symbol import symbols  # 导入SymPy库中的symbols函数用于符号操作


def _sum_free_test(subset):
    """
    检查子集是否是无和子集（即不存在x, y, z使得x + y = z）
    """
    for i in subset:
        for j in subset:
            assert (i + j in subset) is False


def test_schur_partition():
    raises(ValueError, lambda: schur_partition(S.Infinity))  # 检查当输入为无穷大时是否引发值错误
    raises(ValueError, lambda: schur_partition(-1))  # 检查当输入为负数时是否引发值错误
    raises(ValueError, lambda: schur_partition(0))  # 检查当输入为0时是否引发值错误
    assert schur_partition(2) == [[1, 2]]  # 检查Schur分割是否正确返回[[1, 2]]

    random_number_generator = _randint(1000)  # 使用_randint函数创建一个随机数生成器
    for _ in range(5):
        n = random_number_generator(1, 1000)  # 生成一个1到1000之间的随机数n
        result = schur_partition(n)  # 计算n的Schur分割
        t = 0
        numbers = []
        for item in result:
            _sum_free_test(item)  # 检查每个Schur分割是否是无和子集
            """
            检查所有数字的出现次数是否恰好为一
            """
            t += len(item)  # 更新数字总数
            for l in item:
                assert (l in numbers) is False  # 检查数字是否在已有列表中出现过
                numbers.append(l)  # 将数字添加到列表中
        assert n == t  # 检查生成的数字总数是否等于n

    x = symbols("x")
    raises(ValueError, lambda: schur_partition(x))  # 检查当输入为符号x时是否引发值错误


def test_schur_number():
    first_known_schur_numbers = {1: 1, 2: 4, 3: 13, 4: 44, 5: 160}  # 已知的Schur数的前几个值
    for k in first_known_schur_numbers:
        assert SchurNumber(k) == first_known_schur_numbers[k]  # 检查SchurNumber类计算得到的值是否与预期相符

    assert SchurNumber(S.Infinity) == S.Infinity  # 检查当输入为无穷大时SchurNumber类的返回值
    assert SchurNumber(0) == 0  # 检查当输入为0时SchurNumber类的返回值
    raises(ValueError, lambda: SchurNumber(0.5))  # 检查当输入为浮点数时是否引发值错误

    n = symbols("n")
    assert SchurNumber(n).lower_bound() == 3**n/2 - Rational(1, 2)  # 检查SchurNumber类计算的下界是否正确
    assert SchurNumber(8).lower_bound() == 5039  # 检查SchurNumber类计算n=8时的下界是否正确
```