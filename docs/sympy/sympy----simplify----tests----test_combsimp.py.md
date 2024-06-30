# `D:\src\scipysrc\sympy\sympy\simplify\tests\test_combsimp.py`

```
# 导入所需的符号和函数模块
from sympy.core.numbers import Rational
from sympy.core.symbol import symbols
from sympy.functions.combinatorial.factorials import (FallingFactorial, RisingFactorial, binomial, factorial)
from sympy.functions.special.gamma_functions import gamma
from sympy.simplify.combsimp import combsimp
from sympy.abc import x

# 定义测试函数 test_combsimp
def test_combsimp():
    # 声明符号变量 k, m, n，均为整数
    k, m, n = symbols('k m n', integer=True)

    # 断言：对阶乘函数使用 combsimp 应返回其自身
    assert combsimp(factorial(n)) == factorial(n)
    # 断言：对二项式系数函数使用 combsimp 应返回其自身
    assert combsimp(binomial(n, k)) == binomial(n, k)

    # 断言：对阶乘与阶乘差的商应简化为阶乘展开式
    assert combsimp(factorial(n)/factorial(n - 3)) == n*(-1 + n)*(-2 + n)
    # 断言：对二项式系数与其差商的简化应为相应的比例
    assert combsimp(binomial(n + 1, k + 1)/binomial(n, k)) == (1 + n)/(1 + k)

    # 断言：对二项式系数的复杂比例应简化为特定形式
    assert combsimp(binomial(3*n + 4, n + 1)/binomial(3*n + 1, n)) == \
        Rational(3, 2)*((3*n + 2)*(3*n + 4)/((n + 1)*(2*n + 3)))

    # 断言：对阶乘平方与阶乘差的商的简化应为阶乘展开式的乘积
    assert combsimp(factorial(n)**2/factorial(n - 3)) == \
        factorial(n)*n*(-1 + n)*(-2 + n)
    # 断言：对阶乘与二项式系数与其差商的简化应为相应的比例
    assert combsimp(factorial(n)*binomial(n + 1, k + 1)/binomial(n, k)) == \
        factorial(n + 1)/(1 + k)

    # 断言：对伽玛函数的简化应为相应的阶乘
    assert combsimp(gamma(n + 3)) == factorial(n + 2)

    # 断言：对阶乘与伽玛函数的简化应为相应的伽玛函数
    assert combsimp(factorial(x)) == gamma(x + 1)

    # issue 9699 的问题处理
    assert combsimp((n + 1)*factorial(n)) == factorial(n + 1)
    assert combsimp(factorial(n)/n) == factorial(n - 1)

    # issue 6658 的问题处理
    assert combsimp(binomial(n, n - k)) == binomial(n, k)

    # issue 6341, 7135 的问题处理
    assert combsimp(factorial(n)/(factorial(k)*factorial(n - k))) == \
        binomial(n, k)
    assert combsimp(factorial(k)*factorial(n - k)/factorial(n)) == \
        1/binomial(n, k)
    assert combsimp(factorial(2*n)/factorial(n)**2) == binomial(2*n, n)
    assert combsimp(factorial(2*n)*factorial(k)*factorial(n - k)/
        factorial(n)**3) == binomial(2*n, n)/binomial(n, k)

    # 断言：对给定表达式的简化应为常数 1
    assert combsimp(factorial(n*(1 + n) - n**2 - n)) == 1

    # 断言：对 FallingFactorial 与阶乘的比例的简化应为特定形式
    assert combsimp(6*FallingFactorial(-4, n)/factorial(n)) == \
        (-1)**n*(n + 1)*(n + 2)*(n + 3)
    assert combsimp(6*FallingFactorial(-4, n - 1)/factorial(n - 1)) == \
        (-1)**(n - 1)*n*(n + 1)*(n + 2)
    assert combsimp(6*FallingFactorial(-4, n - 3)/factorial(n - 3)) == \
        (-1)**(n - 3)*n*(n - 1)*(n - 2)
    assert combsimp(6*FallingFactorial(-4, -n - 1)/factorial(-n - 1)) == \
        -(-1)**(-n - 1)*n*(n - 1)*(n - 2)

    # 断言：对 RisingFactorial 与阶乘的比例的简化应为特定形式
    assert combsimp(6*RisingFactorial(4, n)/factorial(n)) == \
        (n + 1)*(n + 2)*(n + 3)
    assert combsimp(6*RisingFactorial(4, n - 1)/factorial(n - 1)) == \
        n*(n + 1)*(n + 2)
    assert combsimp(6*RisingFactorial(4, n - 3)/factorial(n - 3)) == \
        n*(n - 1)*(n - 2)
    assert combsimp(6*RisingFactorial(4, -n - 1)/factorial(-n - 1)) == \
        -n*(n - 1)*(n - 2)


# 定义问题解决测试函数 test_issue_6878
def test_issue_6878():
    # 声明符号变量 n，为整数
    n = symbols('n', integer=True)
    # 断言：对特定 RisingFactorial 的简化应为特定形式
    assert combsimp(RisingFactorial(-10, n)) == 3628800*(-1)**n/factorial(10 - n)


# 定义问题解决测试函数 test_issue_14528
def test_issue_14528():
    # 声明符号变量 p，为正整数
    p = symbols("p", integer=True, positive=True)
    # 断言：对特定二项式系数的简化应为其分解形式的倒数
    assert combsimp(binomial(1,p)) == 1/(factorial(p)*factorial(1-p))
    # 断言：对特定阶乘的简化应为其自身
    assert combsimp(factorial(2-p)) == factorial(2-p)
```