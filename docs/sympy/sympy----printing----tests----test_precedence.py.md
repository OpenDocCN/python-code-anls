# `D:\src\scipysrc\sympy\sympy\printing\tests\test_precedence.py`

```
# 导入必要的 SymPy 模块和类
from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
from sympy.core.function import Derivative
from sympy.core.numbers import Integer, Rational, Float, oo
from sympy.core.relational import Rel
from sympy.core.symbol import symbols
from sympy.functions import sin
from sympy.integrals.integrals import Integral
from sympy.series.order import Order

# 导入打印相关的函数和变量
from sympy.printing.precedence import precedence, PRECEDENCE

# 定义符号变量 x 和 y
x, y = symbols("x,y")

# 定义测试函数 test_Add
def test_Add():
    # 断言 x + y 的运算优先级等于 "Add" 的预设值
    assert precedence(x + y) == PRECEDENCE["Add"]
    # 断言 x*y + 1 的运算优先级等于 "Add" 的预设值
    assert precedence(x*y + 1) == PRECEDENCE["Add"]

# 定义测试函数 test_Function
def test_Function():
    # 断言 sin(x) 的运算优先级等于 "Func" 的预设值
    assert precedence(sin(x)) == PRECEDENCE["Func"]

# 定义测试函数 test_Derivative
def test_Derivative():
    # 断言 Derivative(x, y) 的运算优先级等于 "Atom" 的预设值
    assert precedence(Derivative(x, y)) == PRECEDENCE["Atom"]

# 定义测试函数 test_Integral
def test_Integral():
    # 断言 Integral(x, y) 的运算优先级等于 "Atom" 的预设值
    assert precedence(Integral(x, y)) == PRECEDENCE["Atom"]

# 定义测试函数 test_Mul
def test_Mul():
    # 断言 x*y 的运算优先级等于 "Mul" 的预设值
    assert precedence(x*y) == PRECEDENCE["Mul"]
    # 断言 -x*y 的运算优先级等于 "Add" 的预设值
    assert precedence(-x*y) == PRECEDENCE["Add"]

# 定义测试函数 test_Number
def test_Number():
    # 不同整数值的运算优先级断言
    assert precedence(Integer(0)) == PRECEDENCE["Atom"]
    assert precedence(Integer(1)) == PRECEDENCE["Atom"]
    assert precedence(Integer(-1)) == PRECEDENCE["Add"]
    assert precedence(Integer(10)) == PRECEDENCE["Atom"]
    # 不同有理数值的运算优先级断言
    assert precedence(Rational(5, 2)) == PRECEDENCE["Mul"]
    assert precedence(Rational(-5, 2)) == PRECEDENCE["Add"]
    # 不同浮点数值的运算优先级断言
    assert precedence(Float(5)) == PRECEDENCE["Atom"]
    assert precedence(Float(-5)) == PRECEDENCE["Add"]
    # 无穷大数值的运算优先级断言
    assert precedence(oo) == PRECEDENCE["Atom"]
    assert precedence(-oo) == PRECEDENCE["Add"]

# 定义测试函数 test_Order
def test_Order():
    # 断言 Order(x) 的运算优先级等于 "Atom" 的预设值
    assert precedence(Order(x)) == PRECEDENCE["Atom"]

# 定义测试函数 test_Pow
def test_Pow():
    # 不同指数幂的运算优先级断言
    assert precedence(x**y) == PRECEDENCE["Pow"]
    assert precedence(-x**y) == PRECEDENCE["Add"]
    assert precedence(x**-y) == PRECEDENCE["Pow"]

# 定义测试函数 test_Product
def test_Product():
    # 断言 Product(x, (x, y, y + 1)) 的运算优先级等于 "Atom" 的预设值
    assert precedence(Product(x, (x, y, y + 1))) == PRECEDENCE["Atom"]

# 定义测试函数 test_Relational
def test_Relational():
    # 断言 Rel(x + y, y, "<") 的运算优先级等于 "Relational" 的预设值
    assert precedence(Rel(x + y, y, "<")) == PRECEDENCE["Relational"]

# 定义测试函数 test_Sum
def test_Sum():
    # 断言 Sum(x, (x, y, y + 1)) 的运算优先级等于 "Atom" 的预设值
    assert precedence(Sum(x, (x, y, y + 1))) == PRECEDENCE["Atom"]

# 定义测试函数 test_Symbol
def test_Symbol():
    # 断言符号变量 x 的运算优先级等于 "Atom" 的预设值
    assert precedence(x) == PRECEDENCE["Atom"]

# 定义测试函数 test_And_Or
def test_And_Or():
    # 逻辑运算符的优先级关系断言
    assert precedence(x & y) > precedence(x | y)
    assert precedence(~y) > precedence(x & y)
    # 与其他运算符的优先级关系断言（类似于其他编程语言）
    assert precedence(x + y) > precedence(x | y)
    assert precedence(x + y) > precedence(x & y)
    assert precedence(x*y) > precedence(x | y)
    assert precedence(x*y) > precedence(x & y)
    assert precedence(~y) > precedence(x*y)
    assert precedence(~y) > precedence(x - y)
    # 进行双重检查
    assert precedence(x & y) == PRECEDENCE["And"]
    assert precedence(x | y) == PRECEDENCE["Or"]
    assert precedence(~y) == PRECEDENCE["Not"]
```