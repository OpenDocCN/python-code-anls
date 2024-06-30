# `D:\src\scipysrc\sympy\sympy\series\tests\test_demidovich.py`

```
# 导入需要的符号和函数库
from sympy.core.numbers import (Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.trigonometric import (asin, cos, sin, tan)
from sympy.polys.rationaltools import together
from sympy.series.limits import limit

# 定义符号变量 x
x = Symbol("x")

# 测试 leadterm 方法
def test_leadterm():
    assert (3 + 2*x**(log(3)/log(2) - 1)).leadterm(x) == (3, 0)

# 定义计算立方根的函数
def root3(x):
    return root(x, 3)

# 定义计算四次方根的函数
def root4(x):
    return root(x, 4)

# 测试极限计算的函数

# 测试极限计算，结果应为 3
def test_Limits_simple_0():
    assert limit((2**(x + 1) + 3**(x + 1))/(2**x + 3**x), x, oo) == 3  # 175

# 测试极限计算，包含多个断言
def test_Limits_simple_1():
    assert limit((x + 1)*(x + 2)*(x + 3)/x**3, x, oo) == 1  # 172
    assert limit(sqrt(x + 1) - sqrt(x), x, oo) == 0  # 179
    assert limit((2*x - 3)*(3*x + 5)*(4*x - 6)/(3*x**3 + x - 1), x, oo) == 8  # Primjer 1
    assert limit(x/root3(x**3 + 10), x, oo) == 1  # Primjer 2
    assert limit((x + 1)**2/(x**2 + 1), x, oo) == 1  # 181

# 测试极限计算，包含多个断言
def test_Limits_simple_2():
    assert limit(1000*x/(x**2 - 1), x, oo) == 0  # 182
    assert limit((x**2 - 5*x + 1)/(3*x + 7), x, oo) is oo  # 183
    assert limit((2*x**2 - x + 3)/(x**3 - 8*x + 5), x, oo) == 0  # 184
    assert limit((2*x**2 - 3*x - 4)/sqrt(x**4 + 1), x, oo) == 2  # 186
    assert limit((2*x + 3)/(x + root3(x)), x, oo) == 2  # 187
    assert limit(x**2/(10 + x*sqrt(x)), x, oo) is oo  # 188
    assert limit(root3(x**2 + 1)/(x + 1), x, oo) == 0  # 189
    assert limit(sqrt(x)/sqrt(x + sqrt(x + sqrt(x))), x, oo) == 1  # 190

# 测试极限计算，包含多个断言
def test_Limits_simple_3a():
    a = Symbol('a')
    assert together(limit((x**2 - (a + 1)*x + a)/(x**3 - a**3), x, a)) == \
        (a - 1)/(3*a**2)  # 196

# 测试极限计算，包含多个断言
def test_Limits_simple_3b():
    h = Symbol("h")
    assert limit(((x + h)**3 - x**3)/h, h, 0) == 3*x**2  # 197
    assert limit((1/(1 - x) - 3/(1 - x**3)), x, 1) == -1  # 198
    assert limit((sqrt(1 + x) - 1)/(root3(1 + x) - 1), x, 0) == Rational(3)/2  # Primer 4
    assert limit((sqrt(x) - 1)/(x - 1), x, 1) == Rational(1)/2  # 199
    assert limit((sqrt(x) - 8)/(root3(x) - 4), x, 64) == 3  # 200
    assert limit((root3(x) - 1)/(root4(x) - 1), x, 1) == Rational(4)/3  # 201
    assert limit(
        (root3(x**2) - 2*root3(x) + 1)/(x - 1)**2, x, 1) == Rational(1)/9  # 202

# 测试极限计算，包含多个断言
def test_Limits_simple_4a():
    a = Symbol('a')
    assert limit((sqrt(x) - sqrt(a))/(x - a), x, a) == 1/(2*sqrt(a))  # Primer 5
    assert limit((sqrt(x) - 1)/(root3(x) - 1), x, 1) == Rational(3, 2)  # 205
    assert limit((sqrt(1 + x) - sqrt(1 - x))/x, x, 0) == 1  # 207
    assert limit(sqrt(x**2 - 5*x + 6) - x, x, oo) == Rational(-5, 2)  # 213

# 测试极限计算，单个断言
def test_limits_simple_4aa():
    assert limit(x*(sqrt(x**2 + 1) - x), x, oo) == Rational(1)/2  # 214

# 测试极限计算，单个断言
def test_Limits_simple_4b():
    #issue 3511
    # 断言测试，验证 limit(x - root3(x**3 - 1), x, oo) 的返回值是否等于 0
    assert limit(x - root3(x**3 - 1), x, oo) == 0  # 215
def test_Limits_simple_4c():
    assert limit(log(1 + exp(x))/x, x, -oo) == 0  # 求极限 lim(log(1 + exp(x))/x, x -> -∞)，结果应为 0，编号 267a
    assert limit(log(1 + exp(x))/x, x, oo) == 1  # 求极限 lim(log(1 + exp(x))/x, x -> ∞)，结果应为 1，编号 267b


def test_bounded():
    assert limit(sin(x)/x, x, oo) == 0  # 求极限 lim(sin(x)/x, x -> ∞)，结果应为 0，编号 216b
    assert limit(x*sin(1/x), x, 0) == 0  # 求极限 lim(x*sin(1/x), x -> 0)，结果应为 0，编号 227a


def test_f1a():
    # issue 3508:
    assert limit((sin(2*x)/x)**(1 + x), x, 0) == 2  # 求极限 lim((sin(2*x)/x)**(1 + x), x -> 0)，结果应为 2，编号 Primer 7


def test_f1a2():
    # issue 3509:
    assert limit(((x - 1)/(x + 1))**x, x, oo) == exp(-2)  # 求极限 lim(((x - 1)/(x + 1))**x, x -> ∞)，结果应为 exp(-2)，编号 Primer 9


def test_f1b():
    m = Symbol("m")
    n = Symbol("n")
    h = Symbol("h")
    a = Symbol("a")
    assert limit(sin(x)/x, x, 2) == sin(2)/2  # 求极限 lim(sin(x)/x, x -> 2)，结果应为 sin(2)/2，编号 216a
    assert limit(sin(3*x)/x, x, 0) == 3  # 求极限 lim(sin(3*x)/x, x -> 0)，结果应为 3，编号 217
    assert limit(sin(5*x)/sin(2*x), x, 0) == Rational(5, 2)  # 求极限 lim(sin(5*x)/sin(2*x), x -> 0)，结果应为 5/2，编号 218
    assert limit(sin(pi*x)/sin(3*pi*x), x, 0) == Rational(1, 3)  # 求极限 lim(sin(pi*x)/sin(3*pi*x), x -> 0)，结果应为 1/3，编号 219
    assert limit(x*sin(pi/x), x, oo) == pi  # 求极限 lim(x*sin(pi/x), x -> ∞)，结果应为 π，编号 220
    assert limit((1 - cos(x))/x**2, x, 0) == S.Half  # 求极限 lim((1 - cos(x))/x**2, x -> 0)，结果应为 1/2，编号 221
    assert limit(x*sin(1/x), x, oo) == 1  # 求极限 lim(x*sin(1/x), x -> ∞)，结果应为 1，编号 227b
    assert limit((cos(m*x) - cos(n*x))/x**2, x, 0) == -m**2/2 + n**2/2  # 求极限 lim((cos(m*x) - cos(n*x))/x**2, x -> 0)，结果应为 -m**2/2 + n**2/2，编号 232
    assert limit((tan(x) - sin(x))/x**3, x, 0) == S.Half  # 求极限 lim((tan(x) - sin(x))/x**3, x -> 0)，结果应为 1/2，编号 233
    assert limit((x - sin(2*x))/(x + sin(3*x)), x, 0) == -Rational(1, 4)  # 求极限 lim((x - sin(2*x))/(x + sin(3*x)), x -> 0)，结果应为 -1/4，编号 237
    assert limit((1 - sqrt(cos(x)))/x**2, x, 0) == Rational(1, 4)  # 求极限 lim((1 - sqrt(cos(x)))/x**2, x -> 0)，结果应为 1/4，编号 239
    assert limit((sqrt(1 + sin(x)) - sqrt(1 - sin(x)))/x, x, 0) == 1  # 求极限 lim((sqrt(1 + sin(x)) - sqrt(1 - sin(x)))/x, x -> 0)，结果应为 1，编号 240

    assert limit((1 + h/x)**x, x, oo) == exp(h)  # 求极限 lim((1 + h/x)**x, x -> ∞)，结果应为 exp(h)，编号 Primer 9
    assert limit((sin(x) - sin(a))/(x - a), x, a) == cos(a)  # 求极限 lim((sin(x) - sin(a))/(x - a), x -> a)，结果应为 cos(a)，编号 222, *176
    assert limit((cos(x) - cos(a))/(x - a), x, a) == -sin(a)  # 求极限 lim((cos(x) - cos(a))/(x - a), x -> a)，结果应为 -sin(a)，编号 223
    assert limit((sin(x + h) - sin(x))/h, h, 0) == cos(x)  # 求极限 lim((sin(x + h) - sin(x))/h, h -> 0)，结果应为 cos(x)，编号 225


def test_f2a():
    assert limit(((x + 1)/(2*x + 1))**(x**2), x, oo) == 0  # 求极限 lim(((x + 1)/(2*x + 1))**(x**2), x -> ∞)，结果应为 0，编号 Primer 8


def test_f2():
    assert limit((sqrt(
        cos(x)) - root3(cos(x)))/(sin(x)**2), x, 0) == -Rational(1, 12)  # 求极限 lim((sqrt(cos(x)) - root3(cos(x)))/(sin(x)**2), x -> 0)，结果应为 -1/12，编号 *184


def test_f3():
    a = Symbol('a')
    # issue 3504
    assert limit(asin(a*x)/x, x, 0) == a  # 求极限 lim(asin(a*x)/x, x -> 0)，结果应为 a，编号 不提供
```