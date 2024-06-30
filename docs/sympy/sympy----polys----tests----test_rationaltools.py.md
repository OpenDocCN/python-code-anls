# `D:\src\scipysrc\sympy\sympy\polys\tests\test_rationaltools.py`

```
"""Tests for tools for manipulation of rational expressions. """

from sympy.polys.rationaltools import together

from sympy.core.mul import Mul          # 导入乘法运算符
from sympy.core.numbers import Rational  # 导入有理数
from sympy.core.relational import Eq    # 导入等式
from sympy.core.singleton import S      # 导入单例
from sympy.core.symbol import symbols   # 导入符号
from sympy.functions.elementary.exponential import exp  # 导入指数函数
from sympy.functions.elementary.trigonometric import sin  # 导入正弦函数
from sympy.integrals.integrals import Integral  # 导入积分
from sympy.abc import x, y, z  # 导入符号变量 x, y, z

A, B = symbols('A,B', commutative=False)  # 定义非交换符号 A 和 B


def test_together():
    assert together(0) == 0  # 测试：合并常数 0
    assert together(1) == 1  # 测试：合并常数 1

    assert together(x*y*z) == x*y*z  # 测试：合并 x*y*z
    assert together(x + y) == x + y  # 测试：合并 x + y

    assert together(1/x) == 1/x  # 测试：合并 1/x

    assert together(1/x + 1) == (x + 1)/x  # 测试：合并 1/x + 1
    assert together(1/x + 3) == (3*x + 1)/x  # 测试：合并 1/x + 3
    assert together(1/x + x) == (x**2 + 1)/x  # 测试：合并 1/x + x

    assert together(1/x + S.Half) == (x + 2)/(2*x)  # 测试：合并 1/x + S.Half
    assert together(S.Half + x/2) == Mul(S.Half, x + 1, evaluate=False)  # 测试：合并 S.Half + x/2

    assert together(1/x + 2/y) == (2*x + y)/(y*x)  # 测试：合并 1/x + 2/y
    assert together(1/(1 + 1/x)) == x/(1 + x)  # 测试：合并 1/(1 + 1/x)
    assert together(x/(1 + 1/x)) == x**2/(1 + x)  # 测试：合并 x/(1 + 1/x)

    assert together(1/x + 1/y + 1/z) == (x*y + x*z + y*z)/(x*y*z)  # 测试：合并 1/x + 1/y + 1/z
    assert together(1/(1 + x + 1/y + 1/z)) == y*z/(y + z + y*z + x*y*z)  # 测试：合并 1/(1 + x + 1/y + 1/z)

    assert together(1/(x*y) + 1/(x*y)**2) == y**(-2)*x**(-2)*(1 + x*y)  # 测试：合并 1/(x*y) + 1/(x*y)**2
    assert together(1/(x*y) + 1/(x*y)**4) == y**(-4)*x**(-4)*(1 + x**3*y**3)  # 测试：合并 1/(x*y) + 1/(x*y)**4
    assert together(1/(x**7*y) + 1/(x*y)**4) == y**(-4)*x**(-7)*(x**3 + y**3)  # 测试：合并 1/(x**7*y) + 1/(x*y)**4

    assert together(5/(2 + 6/(3 + 7/(4 + 8/(5 + 9/x))))) == \
        Rational(5, 2)*((171 + 119*x)/(279 + 203*x))  # 测试：合并复杂表达式

    assert together(1 + 1/(x + 1)**2) == (1 + (x + 1)**2)/(x + 1)**2  # 测试：合并 1 + 1/(x + 1)**2
    assert together(1 + 1/(x*(1 + x))) == (1 + x*(1 + x))/(x*(1 + x))  # 测试：合并 1 + 1/(x*(1 + x))
    assert together(
        1/(x*(x + 1)) + 1/(x*(x + 2))) == (3 + 2*x)/(x*(1 + x)*(2 + x))  # 测试：合并 1/(x*(x + 1)) + 1/(x*(x + 2))
    assert together(1 + 1/(2*x + 2)**2) == (4*(x + 1)**2 + 1)/(4*(x + 1)**2)  # 测试：合并 1 + 1/(2*x + 2)**2

    assert together(sin(1/x + 1/y)) == sin(1/x + 1/y)  # 测试：合并 sin(1/x + 1/y)
    assert together(sin(1/x + 1/y), deep=True) == sin((x + y)/(x*y))  # 测试：深度合并 sin(1/x + 1/y)

    assert together(1/exp(x) + 1/(x*exp(x))) == (1 + x)/(x*exp(x))  # 测试：合并 1/exp(x) + 1/(x*exp(x))
    assert together(1/exp(2*x) + 1/(x*exp(3*x))) == (1 + exp(x)*x)/(x*exp(3*x))  # 测试：合并 1/exp(2*x) + 1/(x*exp(3*x))

    assert together(Integral(1/x + 1/y, x)) == Integral((x + y)/(x*y), x)  # 测试：合并积分表达式
    assert together(Eq(1/x + 1/y, 1 + 1/z)) == Eq((x + y)/(x*y), (z + 1)/z)  # 测试：合并等式表达式

    assert together((A*B)**-1 + (B*A)**-1) == (A*B)**-1 + (B*A)**-1  # 测试：合并非交换符号的逆
```