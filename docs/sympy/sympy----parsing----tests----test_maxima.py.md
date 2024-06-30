# `D:\src\scipysrc\sympy\sympy\parsing\tests\test_maxima.py`

```
# 导入从Maxima语言解析器的模块
from sympy.parsing.maxima import parse_maxima
# 导入Euler数、有理数、无穷大等数学常数
from sympy.core.numbers import (E, Rational, oo)
# 导入符号变量相关模块
from sympy.core.symbol import Symbol
# 导入阶乘函数
from sympy.functions.combinatorial.factorials import factorial
# 导入绝对值函数
from sympy.functions.elementary.complexes import Abs
# 导入对数函数
from sympy.functions.elementary.exponential import log
# 导入三角函数
from sympy.functions.elementary.trigonometric import (cos, sin)
# 从 sympy.abc 模块导入符号变量 x
from sympy.abc import x

# 创建一个整数符号变量 n
n = Symbol('n', integer=True)

# 定义一个测试函数，用于验证解析器的正确性
def test_parser():
    # 断言解析Maxima表达式 "float(1/3)" 的结果与期望值接近
    assert Abs(parse_maxima('float(1/3)') - 0.333333333) < 10**(-5)
    # 断言解析Maxima表达式 "13^26" 的结果与期望值相等
    assert parse_maxima('13^26') == 91733330193268616658399616009
    # 断言解析Maxima表达式 "sin(%pi/2) + cos(%pi/3)" 的结果与有理数 3/2 相等
    assert parse_maxima('sin(%pi/2) + cos(%pi/3)') == Rational(3, 2)
    # 断言解析Maxima表达式 "log(%e)" 的结果为数值 1
    assert parse_maxima('log(%e)') == 1

# 定义一个测试函数，用于验证在全局命名空间中创建的变量
def test_injection():
    # 在全局命名空间中执行解析Maxima表达式 "c: x+1"，并将结果赋给全局变量 c
    parse_maxima('c: x+1', globals=globals())
    # 断言全局变量 c 的值与表达式 x+1 相等
    assert c == x + 1 # noqa:F821  # 忽略未定义变量的警告

    # 在全局命名空间中执行解析Maxima表达式 "g: sqrt(81)"，并将结果赋给全局变量 g
    parse_maxima('g: sqrt(81)', globals=globals())
    # 断言全局变量 g 的值与数值 9 相等
    assert g == 9 # noqa:F821  # 忽略未定义变量的警告

# 定义一个测试函数，用于验证Maxima函数的正确解析
def test_maxima_functions():
    # 断言解析Maxima表达式 "expand( (x+1)^2)" 的结果与 x**2 + 2*x + 1 相等
    assert parse_maxima('expand( (x+1)^2)') == x**2 + 2*x + 1
    # 断言解析Maxima表达式 "factor( x**2 + 2*x + 1)" 的结果与 (x + 1)**2 相等
    assert parse_maxima('factor( x**2 + 2*x + 1)') == (x + 1)**2
    # 断言解析Maxima表达式 "2*cos(x)^2 + sin(x)^2" 的结果与 2*cos(x)**2 + sin(x)**2 相等
    assert parse_maxima('2*cos(x)^2 + sin(x)^2') == 2*cos(x)**2 + sin(x)**2
    # 断言解析Maxima表达式 "trigexpand(sin(2*x)+cos(2*x))" 的结果与 -1 + 2*cos(x)**2 + 2*cos(x)*sin(x) 相等
    assert parse_maxima('trigexpand(sin(2*x)+cos(2*x))') == -1 + 2*cos(x)**2 + 2*cos(x)*sin(x)
    # 断言解析Maxima表达式 "solve(x^2-4,x)" 的结果与列表 [-2, 2] 相等
    assert parse_maxima('solve(x^2-4,x)') == [-2, 2]
    # 断言解析Maxima表达式 "limit((1+1/x)^x,x,inf)" 的结果与常数 E 相等
    assert parse_maxima('limit((1+1/x)^x,x,inf)') == E
    # 断言解析Maxima表达式 "limit(sqrt(-x)/x,x,0,minus)" 的结果与负无穷大相等
    assert parse_maxima('limit(sqrt(-x)/x,x,0,minus)') is -oo
    # 断言解析Maxima表达式 "diff(x^x, x)" 的结果与 x**x*(1 + log(x)) 相等
    assert parse_maxima('diff(x^x, x)') == x**x*(1 + log(x))
    # 断言解析Maxima表达式 "sum(k, k, 1, n)" 的结果与 (n**2 + n)/2 相等，使用自定义符号变量字典
    assert parse_maxima('sum(k, k, 1, n)', name_dict={
        "n": Symbol('n', integer=True),
        "k": Symbol('k', integer=True)
    }) == (n**2 + n)/2
    # 断言解析Maxima表达式 "product(k, k, 1, n)" 的结果与 n! 相等，使用自定义符号变量字典
    assert parse_maxima('product(k, k, 1, n)', name_dict={
        "n": Symbol('n', integer=True),
        "k": Symbol('k', integer=True)
    }) == factorial(n)
    # 断言解析Maxima表达式 "ratsimp((x^2-1)/(x+1))" 的结果与 x - 1 相等
    assert parse_maxima('ratsimp((x^2-1)/(x+1))') == x - 1
    # 断言解析Maxima表达式 "float(sec(%pi/3) + csc(%pi/3))" 的结果与 3.154700538379252 相差小于 10**(-5)
    assert Abs( parse_maxima('float(sec(%pi/3) + csc(%pi/3))') - 3.154700538379252) < 10**(-5)
```