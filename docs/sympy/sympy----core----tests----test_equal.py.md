# `D:\src\scipysrc\sympy\sympy\core\tests\test_equal.py`

```
# 导入 Rational 类和 Symbol 类，以及 exp 函数
from sympy.core.numbers import Rational
from sympy.core.symbol import (Dummy, Symbol)
from sympy.functions.elementary.exponential import exp


# 定义一个测试函数 test_equal，用于比较数学表达式的相等性
def test_equal():
    # 创建两个符号变量 b 和 a
    b = Symbol("b")
    a = Symbol("a")
    # 创建数学表达式 e1 到 e4
    e1 = a + b
    e2 = 2*a*b
    e3 = a**3*b**2
    e4 = a*b + b*a
    
    # 进行表达式相等性断言
    assert not e1 == e2
    assert not e1 == e2  # 重复的断言，应该删除
    assert e1 != e2
    assert e2 == e4
    assert e2 != e3
    assert not e2 == e3

    # 创建一个符号变量 x
    x = Symbol("x")
    # 创建表达式 e1 和 e2，并进行相等性断言
    e1 = exp(x + 1/x)
    y = Symbol("x")  # 可以考虑删除，因为没有使用到这个 y
    e2 = exp(y + 1/y)
    assert e1 == e2
    assert not e1 != e2

    # 重新赋值符号变量 y，并创建新的表达式 e2，并进行相等性断言
    y = Symbol("y")
    e2 = exp(y + 1/y)
    assert not e1 == e2
    assert e1 != e2

    # 创建表达式 e5，并进行相等性断言
    e5 = Rational(3) + 2*x - x - x
    assert e5 == 3
    assert 3 == e5
    assert e5 != 4
    assert 4 != e5
    assert e5 != 3 + x
    assert 3 + x != e5


# 定义测试函数 test_expevalbug，用于检查指数函数的求值问题
def test_expevalbug():
    # 创建符号变量 x，并创建两个表达式 e1 和 e3，并进行相等性断言
    x = Symbol("x")
    e1 = exp(1*x)
    e3 = exp(x)
    assert e1 == e3


# 定义测试函数 test_cmp_bug1，用于检查自定义类与符号变量比较的问题
def test_cmp_bug1():
    # 定义一个简单的类 T
    class T:
        pass

    # 创建类的一个实例 t 和符号变量 x，并进行相等性断言
    t = T()
    x = Symbol("x")
    assert not (x == t)
    assert (x != t)


# 定义测试函数 test_cmp_bug2，用于检查自定义类与 Symbol 类比较的问题
def test_cmp_bug2():
    # 定义一个简单的类 T
    class T:
        pass

    # 创建类的一个实例 t，并进行相等性断言
    t = T()
    assert not (Symbol == t)
    assert (Symbol != t)


# 定义测试函数 test_cmp_issue_4357，用于检查 Basic 子类与可符号化对象比较的问题
def test_cmp_issue_4357():
    # 进行几个断言来验证 Basic 子类与 sympifiable 对象的比较问题
    assert not (Symbol == 1)
    assert (Symbol != 1)
    assert not (Symbol == 'x')
    assert (Symbol != 'x')


# 定义测试函数 test_dummy_eq，用于检查 Dummy 对象的比较问题
def test_dummy_eq():
    # 创建符号变量 x 和 y，以及 Dummy 对象 u
    x = Symbol('x')
    y = Symbol('y')
    u = Dummy('u')

    # 进行几个 Dummy 对象的比较断言
    assert (u**2 + 1).dummy_eq(x**2 + 1) is True
    assert ((u**2 + 1) == (x**2 + 1)) is False

    assert (u**2 + y).dummy_eq(x**2 + y, x) is True
    assert (u**2 + y).dummy_eq(x**2 + y, y) is False
```