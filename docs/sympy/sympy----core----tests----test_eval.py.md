# `D:\src\scipysrc\sympy\sympy\core\tests\test_eval.py`

```
from sympy.core.function import Function
from sympy.core.numbers import (I, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, tan)
from sympy.testing.pytest import XFAIL

# 定义一个测试函数，用于测试加法表达式的求值
def test_add_eval():
    # 创建符号变量 a 和 b
    a = Symbol("a")
    b = Symbol("b")
    # 创建有理数常量 1 和 5
    c = Rational(1)
    p = Rational(5)
    # 断言：a*b + c + p 的值等于 a*b + 6
    assert a*b + c + p == a*b + 6
    # 断言：c + a + p 的值等于 a + 6
    assert c + a + p == a + 6
    # 断言：c + a - p 的值等于 a - 4
    assert c + a - p == a + (-4)
    # 断言：a + a 的值等于 2*a
    assert a + a == 2*a
    # 断言：a + p + a 的值等于 2*a + 5
    assert a + p + a == 2*a + 5
    # 断言：c + p 的值等于 6
    assert c + p == Rational(6)
    # 断言：b + a - b 的值等于 a
    assert b + a - b == a

# 定义一个测试函数，用于测试加法和乘法混合表达式的求值
def test_addmul_eval():
    # 创建符号变量 a 和 b
    a = Symbol("a")
    b = Symbol("b")
    # 创建有理数常量 1 和 5
    c = Rational(1)
    p = Rational(5)
    # 断言：c + a + b*c + a - p 的值等于 2*a + b - 4
    assert c + a + b*c + a - p == 2*a + b + (-4)
    # 断言：a*2 + p + a 的值等于 3*a + 5
    assert a*2 + p + a == 3*a + 5
    # 断言：a*2 + a 的值等于 3*a
    assert a*2 + a == 3*a

# 定义一个测试函数，用于测试幂运算的求值
def test_pow_eval():
    # XXX Pow does not fully support conversion of negative numbers
    #     to their complex equivalent

    # 断言：对负数进行开平方得到虚数单位 I
    assert sqrt(-1) == I

    # 断言：对 -4 进行开平方得到 2*I
    assert sqrt(-4) == 2*I
    # 断言：对 4 进行开平方得到 2
    assert sqrt(4) == 2
    # 断言：8 的 1/3 次方等于 2
    assert (8)**Rational(1, 3) == 2
    # 断言：-8 的 1/3 次方等于 2*(-1)**(1/3)
    assert (-8)**Rational(1, 3) == 2*((-1)**Rational(1, 3))

    # 断言：对 -2 进行开平方得到 I*sqrt(2)
    assert sqrt(-2) == I*sqrt(2)
    # 断言：(-1)**(1/3) 不等于 I
    assert (-1)**Rational(1, 3) != I
    # 断言：(-10)**(1/3) 不等于 I*((10)**(1/3))
    assert (-10)**Rational(1, 3) != I*((10)**Rational(1, 3))
    # 断言：(-2)**(1/4) 不等于 (2)**(1/4)
    assert (-2)**Rational(1, 4) != (2)**Rational(1, 4)

    # 断言：64 的 1/3 次方等于 4
    assert 64**Rational(1, 3) == 4
    # 断言：64 的 2/3 次方等于 16
    assert 64**Rational(2, 3) == 16
    # 断言：24 除以 64 的开平方等于 3
    assert 24/sqrt(64) == 3
    # 断言：(-27)**(1/3) 等于 3*(-1)**(1/3)
    assert (-27)**Rational(1, 3) == 3*(-1)**Rational(1, 3)

    # 断言：(cos(2) / tan(2)) 的平方等于 (cos(2) / tan(2)) 的平方
    assert (cos(2) / tan(2))**2 == (cos(2) / tan(2))**2

# 标记为 XFAIL 的测试函数，用于测试幂运算的特定情况
@XFAIL
def test_pow_eval_X1():
    assert (-1)**Rational(1, 3) == S.Half + S.Half*I*sqrt(3)

# 定义一个测试函数，用于测试乘法和幂运算混合表达式的求值
def test_mulpow_eval():
    x = Symbol('x')
    # 断言：sqrt(50)/(sqrt(2)*x) 的值等于 5/x
    assert sqrt(50)/(sqrt(2)*x) == 5/x
    # 断言：sqrt(27)/sqrt(3) 的值等于 3
    assert sqrt(27)/sqrt(3) == 3

# 定义一个测试函数，用于测试幂运算求值的 bug
def test_evalpow_bug():
    x = Symbol("x")
    # 断言：1 / (1/x) 的值等于 x
    assert 1/(1/x) == x
    # 断言：1 / (-1/x) 的值等于 -x
    assert 1/(-1/x) == -x

# 定义一个测试函数，用于测试符号展开
def test_symbol_expand():
    x = Symbol('x')
    y = Symbol('y')

    # 创建一个符号表达式 f
    f = x**4*y**4
    # 断言：f 等于 x^4 * y^4
    assert f == x**4*y**4
    # 断言：f 等于 f 的展开形式
    assert f == f.expand()

    # 创建一个符号表达式 g
    g = (x*y)**4
    # 断言：g 等于 f
    assert g == f
    # 断言：g 的展开形式等于 f
    assert g.expand() == f
    # 断言：g 的展开形式等于 g 的二次展开形式
    assert g.expand() == g.expand().expand()

# 定义一个测试函数，用于测试函数的应用
def test_function():
    # 创建函数符号 f 和 l
    f, l = map(Function, 'fl')
    x = Symbol('x')
    # 断言：exp(l(x)) * l(x) / exp(l(x)) 的值等于 l(x)
    assert exp(l(x))*l(x)/exp(l(x)) == l(x)
    # 断言：exp(f(x)) * f(x) / exp(f(x)) 的值等于 f(x)
    assert exp(f(x))*f(x)/exp(f(x)) == f(x)
```