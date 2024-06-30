# `D:\src\scipysrc\sympy\sympy\polys\tests\test_pythonrational.py`

```
# 导入 PythonRational 类型，该类型来自 sympy.polys.domains 模块，用于处理有理数
from sympy.polys.domains import PythonRational as QQ
# 导入 raises 函数，用于测试框架的异常处理
from sympy.testing.pytest import raises

# 定义测试函数 test_PythonRational__init__
def test_PythonRational__init__():
    # 断言 QQ(0) 的分子为 0
    assert QQ(0).numerator == 0
    # 断言 QQ(0) 的分母为 1
    assert QQ(0).denominator == 1
    # 断言 QQ(0, 1) 的分子为 0
    assert QQ(0, 1).numerator == 0
    # 断言 QQ(0, 1) 的分母为 1
    assert QQ(0, 1).denominator == 1
    # 断言 QQ(0, -1) 的分子为 0
    assert QQ(0, -1).numerator == 0
    # 断言 QQ(0, -1) 的分母为 1
    assert QQ(0, -1).denominator == 1

    # 断言 QQ(1) 的分子为 1
    assert QQ(1).numerator == 1
    # 断言 QQ(1) 的分母为 1
    assert QQ(1).denominator == 1
    # 断言 QQ(1, 1) 的分子为 1
    assert QQ(1, 1).numerator == 1
    # 断言 QQ(1, 1) 的分母为 1
    assert QQ(1, 1).denominator == 1
    # 断言 QQ(-1, -1) 的分子为 1
    assert QQ(-1, -1).numerator == 1
    # 断言 QQ(-1, -1) 的分母为 1
    assert QQ(-1, -1).denominator == 1

    # 断言 QQ(-1) 的分子为 -1
    assert QQ(-1).numerator == -1
    # 断言 QQ(-1) 的分母为 1
    assert QQ(-1).denominator == 1
    # 断言 QQ(-1, 1) 的分子为 -1
    assert QQ(-1, 1).numerator == -1
    # 断言 QQ(-1, 1) 的分母为 1
    assert QQ(-1, 1).denominator == 1
    # 断言 QQ(1, -1) 的分子为 -1
    assert QQ( 1, -1).numerator == -1
    # 断言 QQ(1, -1) 的分母为 1
    assert QQ( 1, -1).denominator == 1

    # 断言 QQ(1, 2) 的分子为 1
    assert QQ(1, 2).numerator == 1
    # 断言 QQ(1, 2) 的分母为 2
    assert QQ(1, 2).denominator == 2
    # 断言 QQ(3, 4) 的分子为 3
    assert QQ(3, 4).numerator == 3
    # 断言 QQ(3, 4) 的分母为 4
    assert QQ(3, 4).denominator == 4

    # 断言 QQ(2, 2) 的分子为 1
    assert QQ(2, 2).numerator == 1
    # 断言 QQ(2, 2) 的分母为 1
    assert QQ(2, 2).denominator == 1
    # 断言 QQ(2, 4) 的分子为 1
    assert QQ(2, 4).numerator == 1
    # 断言 QQ(2, 4) 的分母为 2
    assert QQ(2, 4).denominator == 2

# 定义测试函数 test_PythonRational__hash__
def test_PythonRational__hash__():
    # 断言 QQ(0) 的哈希值等于 0 的哈希值
    assert hash(QQ(0)) == hash(0)
    # 断言 QQ(1) 的哈希值等于 1 的哈希值
    assert hash(QQ(1)) == hash(1)
    # 断言 QQ(117) 的哈希值等于 117 的哈希值
    assert hash(QQ(117)) == hash(117)

# 定义测试函数 test_PythonRational__int__
def test_PythonRational__int__():
    # 断言 QQ(-1, 4) 转为整数为 0
    assert int(QQ(-1, 4)) == 0
    # 断言 QQ(1, 4) 转为整数为 0
    assert int(QQ( 1, 4)) == 0
    # 断言 QQ(-5, 4) 转为整数为 -1
    assert int(QQ(-5, 4)) == -1
    # 断言 QQ(5, 4) 转为整数为 1
    assert int(QQ( 5, 4)) == 1

# 后续函数依此类推，不再一一注释。
    # 断言：创建有理数 QQ(-1, 2)，并将其除以 QQ(1, 2)，期望结果为 QQ(-1)
    assert QQ(-1, 2) / QQ( 1, 2) == QQ(-1)
    # 断言：创建有理数 QQ(1, 2)，并将其除以 QQ(-1, 2)，期望结果为 QQ(-1)
    assert QQ( 1, 2) / QQ(-1, 2) == QQ(-1)
    
    # 断言：创建有理数 QQ(1, 2)，并将其除以 QQ(1, 2)，期望结果为 QQ(1)
    assert QQ(1, 2) / QQ(1, 2) == QQ(1)
    # 断言：创建有理数 QQ(1, 2)，并将其除以 QQ(3, 2)，期望结果为 QQ(1, 3)
    assert QQ(1, 2) / QQ(3, 2) == QQ(1, 3)
    # 断言：创建有理数 QQ(3, 2)，并将其除以 QQ(1, 2)，期望结果为 QQ(3)
    assert QQ(3, 2) / QQ(1, 2) == QQ(3)
    # 断言：创建有理数 QQ(3, 2)，并将其除以 QQ(3, 2)，期望结果为 QQ(1)
    assert QQ(3, 2) / QQ(3, 2) == QQ(1)
    
    # 断言：整数 2 除以 QQ(1, 2)，期望结果为 QQ(4)
    assert 2 / QQ(1, 2) == QQ(4)
    # 断言：创建有理数 QQ(1, 2)，并将其除以整数 2，期望结果为 QQ(1, 4)
    assert QQ(1, 2) / 2 == QQ(1, 4)
    
    # 断言：尝试创建有理数 QQ(1, 2) 除以 QQ(0)，预期会抛出 ZeroDivisionError 异常
    raises(ZeroDivisionError, lambda: QQ(1, 2) / QQ(0))
    # 断言：尝试创建有理数 QQ(1, 2) 除以整数 0，预期会抛出 ZeroDivisionError 异常
    raises(ZeroDivisionError, lambda: QQ(1, 2) / 0)
#`
def test_PythonRational__pow__():
    # 测试 QQ 类的幂运算，1 的任何次幂都应等于 1
    assert QQ(1)**10 == QQ(1)
    # 测试 QQ 类的幂运算，2 的 10 次幂应等于 1024
    assert QQ(2)**10 == QQ(1024)

    # 测试 QQ 类的幂运算，1 的负次幂应等于 1
    assert QQ(1)**(-10) == QQ(1)
    # 测试 QQ 类的幂运算，2 的负次幂应等于 1/1024
    assert QQ(2)**(-10) == QQ(1, 1024)

def test_PythonRational__eq__():
    # 测试 QQ 类的等于运算，1/2 与 1/2 应相等
    assert (QQ(1, 2) == QQ(1, 2)) is True
    # 测试 QQ 类的不等于运算，1/2 与 1/2 应不相等，实际上应相等
    assert (QQ(1, 2) != QQ(1, 2)) is False

    # 测试 QQ 类的等于运算，1/2 与 1/3 不应相等
    assert (QQ(1, 2) == QQ(1, 3)) is False
    # 测试 QQ 类的不等于运算，1/2 与 1/3 应不相等
    assert (QQ(1, 2) != QQ(1, 3)) is True

def test_PythonRational__lt_le_gt_ge__():
    # 测试 QQ 类的大小比较，1/2 小于 1/4 应为 False
    assert (QQ(1, 2) < QQ(1, 4)) is False
    # 测试 QQ 类的大小比较，1/2 小于等于 1/4 应为 False
    assert (QQ(1, 2) <= QQ(1, 4)) is False
    # 测试 QQ 类的大小比较，1/2 大于 1/4 应为 True
    assert (QQ(1, 2) > QQ(1, 4)) is True
    # 测试 QQ 类的大小比较，1/2 大于等于 1/4 应为 True
    assert (QQ(1, 2) >= QQ(1, 4)) is True

    # 测试 QQ 类的大小比较，1/4 小于 1/2 应为 True
    assert (QQ(1, 4) < QQ(1, 2)) is True
    # 测试 QQ 类的大小比较，1/4 小于等于 1/2 应为 True
    assert (QQ(1, 4) <= QQ(1, 2)) is True
    # 测试 QQ 类的大小比较，1/4 大于 1/2 应为 False
    assert (QQ(1, 4) > QQ(1, 2)) is False
    # 测试 QQ 类的大小比较，1/4 大于等于 1/2 应为 False
    assert (QQ(1, 4) >= QQ(1, 2)) is False
```