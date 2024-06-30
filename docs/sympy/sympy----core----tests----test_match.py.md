# `D:\src\scipysrc\sympy\sympy\core\tests\test_match.py`

```
# 导入符号计算库中的符号变量 'abc'
from sympy import abc
# 导入具体的求和、加法、导数、函数等功能
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.function import (Derivative, Function, diff)
from sympy.core.mul import Mul
from sympy.core.numbers import (Float, I, Integer, Rational, oo, pi)
from sympy.core.singleton import S
# 导入符号变量、通配符等
from sympy.core.symbol import (Symbol, Wild, symbols)
# 导入指数、对数、平方根等数学函数
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
# 导入三角函数及其它特殊函数
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.hyper import meijerg
# 导入多项式处理相关工具
from sympy.polys.polytools import Poly
# 导入简化表达式的函数
from sympy.simplify.radsimp import collect
from sympy.simplify.simplify import signsimp

# 导入用于测试的 XFAIL 工具
from sympy.testing.pytest import XFAIL

# 定义符号测试函数 test_symbol
def test_symbol():
    # 创建符号变量 x
    x = Symbol('x')
    # 创建通配符变量 a, b, c, p, q
    a, b, c, p, q = map(Wild, 'abcpq')

    # 测试匹配符号 x
    e = x
    assert e.match(x) == {}
    assert e.matches(x) == {}
    assert e.match(a) == {a: x}

    # 测试匹配有理数 5
    e = Rational(5)
    assert e.match(c) == {c: 5}
    assert e.match(e) == {}
    assert e.match(e + 1) is None


# 定义加法测试函数 test_add
def test_add():
    # 创建符号变量 x, y, a, b, c
    x, y, a, b, c = map(Symbol, 'xyabc')
    # 创建通配符变量 p, q, r
    p, q, r = map(Wild, 'pqr')

    # 测试加法表达式 a + b
    e = a + b
    assert e.match(p + b) == {p: a}
    assert e.match(p + a) == {p: b}

    # 测试加法表达式 1 + b
    e = 1 + b
    assert e.match(p + b) == {p: 1}

    # 测试加法表达式 a + b + c
    e = a + b + c
    assert e.match(a + p + c) == {p: b}
    assert e.match(b + p + c) == {p: a}

    # 测试加法表达式 a + b + c + x
    e = a + b + c + x
    assert e.match(a + p + x + c) == {p: b}
    assert e.match(b + p + c + x) == {p: a}
    assert e.match(b) is None
    assert e.match(b + p) == {p: a + c + x}
    assert e.match(a + p + c) == {p: b + x}
    assert e.match(b + p + c) == {p: a + x}

    # 测试加法表达式 4*x + 5
    e = 4*x + 5
    assert e.match(4*x + p) == {p: 5}
    assert e.match(3*x + p) == {p: x + 5}
    assert e.match(p*x + 5) == {p: 4}


# 定义乘方测试函数 test_power
def test_power():
    # 创建符号变量 x, y, a, b, c
    x, y, a, b, c = map(Symbol, 'xyabc')
    # 创建通配符变量 p, q, r
    p, q, r = map(Wild, 'pqr')

    # 测试乘方表达式 (x + y)**a
    e = (x + y)**a
    assert e.match(p**q) == {p: x + y, q: a}
    assert e.match(p**p) is None

    # 测试乘方表达式 (x + y)**(x + y)
    e = (x + y)**(x + y)
    assert e.match(p**p) == {p: x + y}
    assert e.match(p**q) == {p: x + y, q: x + y}

    # 测试乘方表达式 (2*x)**2
    e = (2*x)**2
    assert e.match(p*q**r) == {p: 4, q: x, r: 2}

    # 测试乘方表达式 1
    e = Integer(1)
    assert e.match(x**p) == {p: 0}


# 定义匹配排除测试函数 test_match_exclude
def test_match_exclude():
    # 创建符号变量 x, y
    x = Symbol('x')
    y = Symbol('y')
    # 创建通配符变量 p, q, r
    p = Wild("p")
    q = Wild("q")
    r = Wild("r")

    # 测试匹配排除有理数 6
    e = Rational(6)
    assert e.match(2*p) == {p: 3}

    # 测试匹配排除表达式 3/(4*x + 5)
    e = 3/(4*x + 5)
    assert e.match(3/(p*x + q)) == {p: 4, q: 5}

    # 测试匹配排除表达式 3/(4*x + 5)
    e = 3/(4*x + 5)
    assert e.match(p/(q*x + r)) == {p: 3, q: 4, r: 5}

    # 测试匹配排除表达式 2/(x + 1)
    e = 2/(x + 1)
    assert e.match(p/(q*x + r)) == {p: 2, q: 1, r: 1}

    # 测试匹配排除表达式 1/(x + 1)
    e = 1/(x + 1)
    assert e.match(p/(q*x + r)) == {p: 1, q: 1, r: 1}

    # 测试匹配排除表达式 4*x + 5
    e = 4*x + 5
    assert e.match(p*x + q) == {p: 4, q: 5}

    # 测试匹配排除表达式 4*x + 5*y + 6
    e = 4*x + 5*y + 6
    assert e.match(p*x + q*y + r) == {p: 4, q: 5, r: 6}

    # 创建带排除的通配符变量 a
    a = Wild('a', exclude=[x])

    # 测试匹配排除表达式 3*x
    e = 3*x
    assert e.match(p*x) == {p: 3}
    assert e.match(a*x) == {a: 3}

    # 测试匹配排除表达式 3*x**2
    e = 3*x**2
    assert e.match(p*x) == {p: 3*x}
    # 断言检查表达式 e.match(a*x) 是否为 None
    assert e.match(a*x) is None
    
    # 定义表达式 e，表示为 3*x + 3 + 6/x
    e = 3*x + 3 + 6/x
    # 断言检查表达式 e 是否匹配模式 p*x**2 + p*x + 2*p，并返回匹配的字典 {p: 3/x}
    assert e.match(p*x**2 + p*x + 2*p) == {p: 3/x}
    # 断言检查表达式 e 是否匹配模式 a*x**2 + a*x + 2*a，并期望返回 None
    assert e.match(a*x**2 + a*x + 2*a) is None
# 定义一个函数 test_mul，用于测试符号匹配的功能
def test_mul():
    # 使用 Symbol 类将 'x', 'y', 'a', 'b', 'c' 映射为符号对象
    x, y, a, b, c = map(Symbol, 'xyabc')
    # 使用 Wild 类将 'p', 'q' 映射为通配符对象
    p, q = map(Wild, 'pq')

    # 创建表达式 e = 4*x
    e = 4*x
    # 断言 e 匹配模式 p*x，并且返回匹配到的字典 {p: 4}
    assert e.match(p*x) == {p: 4}
    # 断言 e 不匹配模式 p*y，返回 None
    assert e.match(p*y) is None
    # 断言 e 匹配模式 e + p*y，并且返回匹配到的字典 {p: 0}
    assert e.match(e + p*y) == {p: 0}

    # 创建表达式 e = a*x*b*c
    e = a*x*b*c
    # 断言 e 匹配模式 p*x，并且返回匹配到的字典 {p: a*b*c}
    assert e.match(p*x) == {p: a*b*c}
    # 断言 e 匹配模式 c*p*x，并且返回匹配到的字典 {p: a*b}
    assert e.match(c*p*x) == {p: a*b}

    # 创建表达式 e = (a + b)*(a + c)
    e = (a + b)*(a + c)
    # 断言 e 匹配模式 (p + b)*(p + c)，并且返回匹配到的字典 {p: a}
    assert e.match((p + b)*(p + c)) == {p: a}

    # 创建表达式 e = x
    e = x
    # 断言 e 匹配模式 p*x，并且返回匹配到的字典 {p: 1}
    assert e.match(p*x) == {p: 1}

    # 创建表达式 e = exp(x)
    e = exp(x)
    # 断言 e 匹配模式 x**p*exp(x*q)，并且返回匹配到的字典 {p: 0, q: 1}
    assert e.match(x**p*exp(x*q)) == {p: 0, q: 1}

    # 创建表达式 e = I*Poly(x, x)
    e = I*Poly(x, x)
    # 断言 e 匹配模式 I*p，并且返回匹配到的字典 {p: x}
    assert e.match(I*p) == {p: x}


# 定义一个函数 test_mul_noncommutative，用于测试非交换符号的匹配功能
def test_mul_noncommutative():
    # 使用 symbols 函数创建符号对象 x, y
    x, y = symbols('x y')
    # 使用 symbols 函数创建非交换符号对象 A, B, C
    A, B, C = symbols('A B C', commutative=False)
    # 使用 Wild 类创建通配符对象 u, v
    u, v = symbols('u v', cls=Wild)
    # 使用 Wild 类创建非交换通配符对象 w, z
    w, z = symbols('w z', cls=Wild, commutative=False)

    # 断言 (u*v).matches(x) 返回的结果在 ({v: x, u: 1}, {u: x, v: 1}) 中
    assert (u*v).matches(x) in ({v: x, u: 1}, {u: x, v: 1})
    # 断言 (u*v).matches(x*y) 返回的结果在 ({v: y, u: x}, {u: y, v: x}) 中
    assert (u*v).matches(x*y) in ({v: y, u: x}, {u: y, v: x})
    # 断言 (u*v).matches(A) 返回 None，即不匹配 A
    assert (u*v).matches(A) is None
    # 断言 (u*v).matches(A*B) 返回 None，即不匹配 A*B
    assert (u*v).matches(A*B) is None
    # 断言 (u*v).matches(x*A) 返回 None，即不匹配 x*A
    assert (u*v).matches(x*A) is None
    # 断言 (u*v).matches(x*y*A) 返回 None，即不匹配 x*y*A
    assert (u*v).matches(x*y*A) is None
    # 断言 (u*v).matches(x*A*B) 返回 None，即不匹配 x*A*B
    assert (u*v).matches(x*A*B) is None
    # 断言 (u*v).matches(x*y*A*B) 返回 None，即不匹配 x*y*A*B
    assert (u*v).matches(x*y*A*B) is None

    # 断言 (v*w).matches(x) 返回 None，即不匹配 x
    assert (v*w).matches(x) is None
    # 断言 (v*w).matches(x*y) 返回 None，即不匹配 x*y
    assert (v*w).matches(x*y) is None
    # 断言 (v*w).matches(A) 返回的字典为 {w: A, v: 1}
    assert (v*w).matches(A) == {w: A, v: 1}
    # 断言 (v*w).matches(A*B) 返回的字典为 {w: A*B, v: 1}
    assert (v*w).matches(A*B) == {w: A*B, v: 1}
    # 断言 (v*w).matches(x*A) 返回的字典为 {w: A, v: x}
    assert (v*w).matches(x*A) == {w: A, v: x}
    # 断言 (v*w).matches(x*y*A) 返回的字典为 {w: A, v: x*y}
    assert (v*w).matches(x*y*A) == {w: A, v: x*y}
    # 断言 (v*w).matches(x*A*B) 返回的字典为 {w: A*B, v: x}
    assert (v*w).matches(x*A*B) == {w: A*B, v: x}
    # 断言 (v*w).matches(x*y*A*B) 返回的字典为 {w: A*B, v: x*y}
    assert (v*w).matches(x*y*A*B) == {w: A*B, v: x*y}

    # 断言 (v*w).matches(-x) 返回 None，即不匹配 -x
    assert (v*w).matches(-x) is None
    # 断言 (v*w).matches(-x*y) 返回 None，即不匹配 -x*y
    assert (v*w).matches(-x*y) is None
    # 断言 (v*w).matches(-A) 返回的字典为 {w: A, v: -1}
    assert (v*w).matches(-A) == {w: A, v: -1}
    # 断言 (v*w).matches(-A*B) 返回的字典为 {w: A*B, v: -1}
    assert (v*w).matches(-A*B) == {w: A*B, v: -1}
    # 断言 (v*w).matches(-x*A) 返回的字典为 {w: A, v: -x}
    assert (v*w).matches(-x*A) == {w: A, v: -x}
    # 断言 (v*w).matches(-x*y*A) 返回的字典为 {w: A, v: -x*y}
    assert (v*w).matches(-x*y*A) == {w: A, v: -x*y}
    # 断言 (v*w).matches(-x*A*B) 返回的字典为 {w: A*B, v: -x}
    assert (v*w).matches(-x*A*B) == {w: A*B, v: -x}
    # 断言 (v*w).matches(-x*y*A*B) 返回的字典为 {w: A*B, v: -x*y}
    assert (v*w).matches(-x*y*A*B) == {w: A*B, v: -x*y}

    # 断言 (w*z).matches(x) 返回 None，即不匹配 x
    assert (w*z).matches(x) is None
    # 断言 (w*z).matches(x*y) 返回 None，即不匹配 x*y
    assert (w*z).matches(x*y) is None
    # 断言 (w
    # 断言：验证是否匹配条件，如果不匹配则触发异常
    assert (u*B).matches(x*A) is None
    # 断言：验证是否匹配条件，如果匹配则返回匹配的映射关系
    assert (u*A*B).matches(x*A*B) == {u: x}
    # 断言：验证是否匹配条件，如果不匹配则触发异常
    assert (u*A*B).matches(x*B*A) is None
    # 断言：验证是否匹配条件，如果不匹配则触发异常
    assert (u*A*B).matches(x*A) is None
    
    # 断言：验证是否匹配条件，如果不匹配则触发异常
    assert (u*w*A).matches(x*A*B) is None
    # 断言：验证是否匹配条件，如果匹配则返回匹配的映射关系
    assert (u*w*B).matches(x*A*B) == {u: x, w: A}
    
    # 断言：验证是否匹配条件，如果匹配则返回匹配的映射关系，有两种可能性
    assert (u*v*A*B).matches(x*A*B) in [{u: x, v: 1}, {v: x, u: 1}]
    # 断言：验证是否匹配条件，如果不匹配则触发异常
    assert (u*v*A*B).matches(x*B*A) is None
    # 断言：验证是否匹配条件，如果不匹配则触发异常
    assert (u*v*A*B).matches(u*v*A*C) is None
# 定义一个测试函数，用于验证非交换乘法的 Wild 符号匹配功能
def test_mul_noncommutative_mismatch():
    # 创建三个非交换符号 A, B, C
    A, B, C = symbols('A B C', commutative=False)
    # 创建一个 Wild 符号 w，指定为非交换符号
    w = symbols('w', cls=Wild, commutative=False)

    # 断言语句：验证 w*B*w 是否匹配 A*B*A，并返回匹配结果 {w: A}
    assert (w*B*w).matches(A*B*A) == {w: A}
    # 断言语句：验证 w*B*w 是否匹配 A*C*B*A*C，并返回匹配结果 {w: A*C}
    assert (w*B*w).matches(A*C*B*A*C) == {w: A*C}
    # 断言语句：验证 w*B*w 是否匹配 A*C*B*A*B，预期结果为 None
    assert (w*B*w).matches(A*C*B*A*B) is None
    # 断言语句：验证 w*B*w 是否匹配 A*B*C，预期结果为 None
    assert (w*B*w).matches(A*B*C) is None
    # 断言语句：验证 w*w*C 是否匹配 A*B*C，预期结果为 None
    assert (w*w*C).matches(A*B*C) is None


# 定义一个测试函数，用于验证非交换乘法中的幂次符号匹配功能
def test_mul_noncommutative_pow():
    # 创建三个非交换符号 A, B, C
    A, B, C = symbols('A B C', commutative=False)
    # 创建一个 Wild 符号 w，指定为非交换符号
    w = symbols('w', cls=Wild, commutative=False)

    # 断言语句：验证 A*B*w 是否匹配 A*B**2，并返回匹配结果 {w: B}
    assert (A*B*w).matches(A*B**2) == {w: B}
    # 断言语句：验证 A*(B**2)*w*(B**3) 是否匹配 A*B**8，并返回匹配结果 {w: B**3}
    assert (A*(B**2)*w*(B**3)).matches(A*B**8) == {w: B**3}
    # 断言语句：验证 A*B*w*C 是否匹配 A*(B**4)*C，并返回匹配结果 {w: B**3}
    assert (A*B*w*C).matches(A*(B**4)*C) == {w: B**3}

    # 断言语句：验证 A*B*(w**(-1)) 是否匹配 A*B*(C**(-1))，并返回匹配结果 {w: C}
    assert (A*B*(w**(-1))).matches(A*B*(C**(-1))) == {w: C}
    # 断言语句：验证 A*(B*w)**(-1)*C 是否匹配 A*(B*C)**(-1)*C，并返回匹配结果 {w: C}
    assert (A*(B*w)**(-1)*C).matches(A*(B*C)**(-1)*C) == {w: C}

    # 断言语句：验证 (w**2)*B*C 是否匹配 (A**2)*B*C，并返回匹配结果 {w: A}
    assert ((w**2)*B*C).matches((A**2)*B*C) == {w: A}
    # 断言语句：验证 (w**2)*B*(w**3) 是否匹配 (A**2)*B*(A**3)，并返回匹配结果 {w: A}
    assert ((w**2)*B*(w**3)).matches((A**2)*B*(A**3)) == {w: A}
    # 断言语句：验证 (w**2)*B*(w**4) 是否匹配 (A**2)*B*(A**2)，预期结果为 None
    assert ((w**2)*B*(w**4)).matches((A**2)*B*(A**2)) is None


# 定义一个测试函数，用于验证复杂的符号匹配
def test_complex():
    # 创建三个符号 a, b, c
    a, b, c = map(Symbol, 'abc')
    # 创建两个 Wild 符号 x, y
    x, y = map(Wild, 'xy')

    # 断言语句：验证 (1 + I) 是否匹配 x + I，并返回匹配结果 {x: 1}
    assert (1 + I).match(x + I) == {x: 1}
    # 断言语句：验证 (a + I) 是否匹配 x + I，并返回匹配结果 {x: a}
    assert (a + I).match(x + I) == {x: a}
    # 断言语句：验证 (2*I) 是否匹配 x*I，并返回匹配结果 {x: 2}
    assert (2*I).match(x*I) == {x: 2}
    # 断言语句：验证 (a*I) 是否匹配 x*I，并返回匹配结果 {x: a}
    assert (a*I).match(x*I) == {x: a}
    # 断言语句：验证 (a*I) 是否匹配 x*y，并返回匹配结果 {x: I, y: a}
    assert (a*I).match(x*y) == {x: I, y: a}
    # 断言语句：验证 (2*I) 是否匹配 x*y，并返回匹配结果 {x: 2, y: I}
    assert (2*I).match(x*y) == {x: 2, y: I}
    # 断言语句：验证 (a + b*I) 是否匹配 x + y*I，并返回匹配结果 {x: a, y: b}


def test_functions():
    # 导入 WildFunction 类
    from sympy.core.function import WildFunction
    # 创建一个符号 x
    x = Symbol('x')
    # 创建一个 WildFunction g
    g = WildFunction('g')
    # 创建两个 Wild 符号 p, q
    p = Wild('p')
    q = Wild('q')

    # 创建 cos(5*x) 的表达式
    f = cos(5*x)
    # 创建 x 的符号表达式
    notf = x

    # 断言语句：验证 f.match(p*cos(q*x)) 是否匹配 {p: 1, q: 5}
    assert f.match(p*cos(q*x)) == {p: 1, q: 5}
    # 断言语句：验证 f.match(p*g) 是否匹配 {p: 1, g: cos(5*x)}
    assert f.match(p*g) == {p: 1, g: cos(5*x)}
    # 断言语句：验证 notf.match(g) 是否为 None
    assert notf.match(g) is None


@XFAIL
def test_functions_X1():
    # 导入 WildFunction 类
    from sympy.core.function import WildFunction
    # 创建一个符号 x
    x = Symbol('x')
    # 创建一个 WildFunction g
    g = WildFunction('g')
    # 创建两个 Wild 符号 p, q
    p = Wild('p')
    q = Wild('q')

    # 创建 cos(5*x) 的表达式
    f = cos(5*x)
    # 断言语句：验证 f.match(p*g(q*x)) 是否匹配 {p: 1, g: cos, q: 5}


def test_interface():
    # 创建两个符号 x, y
    x, y = map(Symbol, 'xy')
    # 创建两个 Wild 符号 p, q
    p, q = map(Wild, 'pq')

    # 断言语句：验证 (x + 1) 是否匹配 p + 1，并返回匹配结果 {p: x}
    assert (x + 1).match(p + 1) == {p: x}
    # 断言语句：验证 (x*3) 是否匹配 p*3，并返回匹配结果 {p: x}
    assert (x*3).match(p*3) == {p: x}
    # 断言语句：验证 (x**3) 是否匹配 p**3，并返回匹配结果 {p: x}
    assert (x**3).match(p**3) == {p: x}
    # 断言语句：验证 (x*cos(y)) 是否匹配 p*cos(q)，并返回
    # 断言语句，用于确保变量 d2 的值为 None
    assert d2 is None
def test_derivative2():
    # 创建函数符号和变量符号
    f = Function("f")
    x = Symbol("x")
    # 创建排除指定符号的通配符
    a = Wild("a", exclude=[f, x])
    b = Wild("b", exclude=[f])
    # 创建 f(x) 对 x 的一阶导数表达式
    e = Derivative(f(x), x)
    # 断言 Derivative(f(x), x) 不匹配空字典
    assert e.match(Derivative(f(x), x)) == {}
    # 断言 Derivative(f(x), x, x) 为 None
    assert e.match(Derivative(f(x), x, x)) is None
    # 创建 f(x) 对 x 的二阶导数表达式
    e = Derivative(f(x), x, x)
    # 断言 Derivative(f(x), x) 为 None
    assert e.match(Derivative(f(x), x)) is None
    # 断言 Derivative(f(x), x, x) 不匹配空字典
    assert e.match(Derivative(f(x), x, x)) == {}
    # 创建 f(x) 对 x 的一阶导数与 x**2 相加的表达式
    e = Derivative(f(x), x) + x**2
    # 断言 a*Derivative(f(x), x) + b 的匹配结果为 {a: 1, b: x**2}
    assert e.match(a*Derivative(f(x), x) + b) == {a: 1, b: x**2}
    # 断言 a*Derivative(f(x), x, x) + b 为 None
    assert e.match(a*Derivative(f(x), x, x) + b) is None
    # 创建 f(x) 对 x 的二阶导数与 x**2 相加的表达式
    e = Derivative(f(x), x, x) + x**2
    # 断言 a*Derivative(f(x), x) + b 为 None
    assert e.match(a*Derivative(f(x), x) + b) is None
    # 断言 a*Derivative(f(x), x, x) + b 的匹配结果为 {a: 1, b: x**2}


def test_match_deriv_bug1():
    # 创建函数符号
    n = Function('n')
    l = Function('l')

    # 创建变量符号和通配符
    x = Symbol('x')
    p = Wild('p')

    # 创建复杂表达式
    e = diff(l(x), x)/x - diff(diff(n(x), x), x)/2 - \
        diff(n(x), x)**2/4 + diff(n(x), x)*diff(l(x), x)/4
    # 对 n(x) 进行替换和化简
    e = e.subs(n(x), -l(x)).doit()
    # 创建简单函数表达式
    t = x*exp(-l(x))
    # 计算 t 对 x 的二阶导数与 t 相除的表达式
    t2 = t.diff(x, x)/t
    # 断言 e 与 (p*t2).expand() 的匹配结果为 {p: Rational(-1, 2)}
    assert e.match( (p*t2).expand() ) == {p: Rational(-1, 2)}


def test_match_bug2():
    # 创建变量符号和通配符
    x, y = map(Symbol, 'xy')
    p, q, r = map(Wild, 'pqr')
    # 对 (x + y) 进行模式匹配
    res = (x + y).match(p + q + r)
    # 断言 (p + q + r).subs(res) 等于 x + y
    assert (p + q + r).subs(res) == x + y


def test_match_bug3():
    # 创建变量符号和通配符
    x, a, b = map(Symbol, 'xab')
    p = Wild('p')
    # 断言 (b*x*exp(a*x)) 不匹配 x*exp(p*x)
    assert (b*x*exp(a*x)).match(x*exp(p*x)) is None


def test_match_bug4():
    # 创建变量符号和通配符
    x = Symbol('x')
    p = Wild('p')
    # 创建表达式 e = x
    e = x
    # 断言 e 与 -p*x 的匹配结果为 {p: -1}
    assert e.match(-p*x) == {p: -1}


def test_match_bug5():
    # 创建变量符号和通配符
    x = Symbol('x')
    p = Wild('p')
    # 创建表达式 e = -x
    e = -x
    # 断言 e 与 -p*x 的匹配结果为 {p: 1}
    assert e.match(-p*x) == {p: 1}


def test_match_bug6():
    # 创建变量符号和通配符
    x = Symbol('x')
    p = Wild('p')
    # 创建表达式 e = x
    e = x
    # 断言 e 与 3*p*x 的匹配结果为 {p: Rational(1)/3}
    assert e.match(3*p*x) == {p: Rational(1)/3}


def test_match_polynomial():
    # 创建变量符号和排除特定符号的通配符
    x = Symbol('x')
    a = Wild('a', exclude=[x])
    b = Wild('b', exclude=[x])
    c = Wild('c', exclude=[x])
    d = Wild('d', exclude=[x])

    # 创建多项式表达式和模式
    eq = 4*x**3 + 3*x**2 + 2*x + 1
    pattern = a*x**3 + b*x**2 + c*x + d
    # 断言 eq 与 pattern 的匹配结果为 {a: 4, b: 3, c: 2, d: 1}
    assert eq.match(pattern) == {a: 4, b: 3, c: 2, d: 1}
    # 断言 (eq - 3*x**2) 与 pattern 的匹配结果为 {a: 4, b: 0, c: 2, d: 1}
    assert (eq - 3*x**2).match(pattern) == {a: 4, b: 0, c: 2, d: 1}
    # 断言 (x + sqrt(2) + 3) 与 a + b*x + c*x**2 的匹配结果为 {b: 1, a: sqrt(2) + 3, c: 0}
    assert (x + sqrt(2) + 3).match(a + b*x + c*x**2) == \
        {b: 1, a: sqrt(2) + 3, c: 0}


def test_exclude():
    # 创建变量符号和排除特定符号的通配符
    x, y, a = map(Symbol, 'xya')
    p = Wild('p', exclude=[1, x])
    q = Wild('q')
    r = Wild('r', exclude=[sin, y])

    # 断言 sin(x) 不匹配 r
    assert sin(x).match(r) is None
    # 断言 cos(y) 不匹配 r
    assert cos(y).match(r) is None

    # 创建表达式 e = 3*x**2 + y*x + a
    e = 3*x**2 + y*x + a
    # 断言 e 与 p*x**2 + q*x + r 的匹配结果为 {p: 3, q: y, r: a}

    assert e.match(p*x**2 + q*x + r) == {p: 3, q: y, r: a}

    # 创建表达式 e = x + 1
    e = x + 1
    # 断言 e 不与 x + p 匹配
    assert e.match(x + p) is None
    # 断言 e 不与 p + 1 匹配
    assert e.match(p + 1) is None

assert e.match(x + 1 + p) == {p: 0}

e = cos(x) + 5*sin(y)
assert e.match(r) is None
assert e.match(cos(y) + r) is None
assert e.match(r + p*sin(q)) == {r: cos(x), p: 5, q: y}
    # 创建一个名为"a"的通配符对象，并排除 f(x) 的匹配结果
    a = Wild("a", exclude=[f(x)])
    
    # 创建一个名为"b"的通配符对象，并排除 f(x) 的匹配结果
    b = Wild("b", exclude=[f(x)])
    
    # 计算 f(x) 对 x 的导数
    eq = f(x).diff(x)
    
    # 使用通配符a和b对表达式进行模式匹配，并断言其匹配结果等于 {a: 1, b: 0}
    assert eq.match(a*Derivative(f(x), x) + b) == {a: 1, b: 0}
def test_match_wild_wild():
    # 定义三个 Wild 对象 p, q, r
    p = Wild('p')
    q = Wild('q')
    r = Wild('r')

    # 断言 p 匹配 q + r 的结果在指定的两个字典中的一个
    assert p.match(q + r) in [ {q: p, r: 0}, {q: 0, r: p} ]
    # 断言 p 匹配 q*r 的结果在指定的两个字典中的一个
    assert p.match(q*r) in [ {q: p, r: 1}, {q: 1, r: p} ]

    # 重新定义 p, q, r 对象
    p = Wild('p')
    q = Wild('q', exclude=[p])
    r = Wild('r')

    # 断言 p 匹配 q + r 的结果为 {q: 0, r: p}
    assert p.match(q + r) == {q: 0, r: p}
    # 断言 p 匹配 q*r 的结果为 {q: 1, r: p}
    assert p.match(q*r) == {q: 1, r: p}

    # 重新定义 p, q, r 对象
    p = Wild('p')
    q = Wild('q', exclude=[p])
    r = Wild('r', exclude=[p])

    # 断言 p 匹配 q + r 的结果为 None
    assert p.match(q + r) is None
    # 断言 p 匹配 q*r 的结果为 None
    assert p.match(q*r) is None


def test__combine_inverse():
    # 定义符号变量 x, y
    x, y = symbols("x y")
    # 断言 Mul 类的 _combine_inverse 方法对 x*I*y, x*I 的计算结果为 y
    assert Mul._combine_inverse(x*I*y, x*I) == y
    # 断言 Mul 类的 _combine_inverse 方法对 x*x**(1 + y), x**(1 + y) 的计算结果为 x
    assert Mul._combine_inverse(x*x**(1 + y), x**(1 + y)) == x
    # 断言 Mul 类的 _combine_inverse 方法对 x*I*y, y*I 的计算结果为 x
    assert Mul._combine_inverse(x*I*y, y*I) == x
    # 断言 Mul 类的 _combine_inverse 方法对 oo*I*y, y*I 的计算结果为 oo
    assert Mul._combine_inverse(oo*I*y, y*I) is oo
    # 断言 Mul 类的 _combine_inverse 方法对 oo*I*y, oo*I 的计算结果为 y
    assert Mul._combine_inverse(oo*I*y, oo*I) == y
    # 断言 Mul 类的 _combine_inverse 方法对 oo*y, -oo 的计算结果为 -y
    assert Mul._combine_inverse(oo*y, -oo) == -y
    # 断言 Mul 类的 _combine_inverse 方法对 -oo*y, oo 的计算结果为 -y
    assert Mul._combine_inverse(-oo*y, oo) == -y
    # 断言 Mul 类的 _combine_inverse 方法对 (1-exp(x/y)), (exp(x/y)-1) 的计算结果为 -1
    assert Mul._combine_inverse((1-exp(x/y)),(exp(x/y)-1)) == -1
    # 断言 Add 类的 _combine_inverse 方法对 oo, oo 的计算结果为 S.Zero
    assert Add._combine_inverse(oo, oo) is S.Zero
    # 断言 Add 类的 _combine_inverse 方法对 oo*I, oo*I 的计算结果为 S.Zero
    assert Add._combine_inverse(oo*I, oo*I) is S.Zero
    # 断言 Add 类的 _combine_inverse 方法对 x*oo, x*oo 的计算结果为 S.Zero
    assert Add._combine_inverse(x*oo, x*oo) is S.Zero
    # 断言 Add 类的 _combine_inverse 方法对 -x*oo, -x*oo 的计算结果为 S.Zero
    assert Add._combine_inverse(-x*oo, -x*oo) is S.Zero
    # 断言 Add 类的 _combine_inverse 方法对 (x - oo)*(x + oo), -oo 的计算结果为 None
    assert Add._combine_inverse((x - oo)*(x + oo), -oo)


def test_issue_3773():
    # 定义符号变量 x
    x = symbols('x')
    # 定义符号变量 z, phi, r
    z, phi, r = symbols('z phi r')
    # 定义符号变量 c, A, B, N，并使用 Wild 类
    c, A, B, N = symbols('c A B N', cls=Wild)
    # 定义 Wild 类对象 l，并排除值为 0
    l = Wild('l', exclude=(0,))

    # 定义方程 eq
    eq = z * sin(2*phi) * r**7
    # 定义匹配对象 matcher
    matcher = c * sin(phi*N)**l * r**A * log(r)**B

    # 断言 eq 匹配 matcher 的结果为指定的字典
    assert eq.match(matcher) == {c: z, l: 1, N: 2, A: 7, B: 0}
    # 断言 (-eq) 匹配 matcher 的结果为指定的字典
    assert (-eq).match(matcher) == {c: -z, l: 1, N: 2, A: 7, B: 0}
    # 断言 (x*eq) 匹配 matcher 的结果为指定的字典
    assert (x*eq).match(matcher) == {c: x*z, l: 1, N: 2, A: 7, B: 0}
    # 断言 (-7*x*eq) 匹配 matcher 的结果为指定的字典
    assert (-7*x*eq).match(matcher) == {c: -7*x*z, l: 1, N: 2, A: 7, B: 0}

    # 重新定义 matcher 对象
    matcher = c*sin(phi*N)**l * r**A

    # 断言 eq 匹配 matcher 的结果为指定的字典
    assert eq.match(matcher) == {c: z, l: 1, N: 2, A: 7}
    # 断言 (-eq) 匹配 matcher 的结果为指定的字典
    assert (-eq).match(matcher) == {c: -z, l: 1, N: 2, A: 7}
    # 断言 (x*eq) 匹配 matcher 的结果为指定的字典
    assert (x*eq).match(matcher) == {c: x*z, l: 1, N: 2, A: 7}
    # 断言 (-7*x*eq) 匹配 matcher 的结果为指定的字典
    assert (-7*x*eq).match(matcher) == {c: -7*x*z, l: 1, N: 2, A: 7}


def test_issue_3883():
    # 导入符号变量 gamma, mu, x
    from sympy.abc import gamma, mu, x
    # 定义表达式 f
    f = (-gamma * (x - mu)**2 - log(gamma) + log(2*pi))/2
    # 定义 Wild 类对象 a, b, c，并排除变量 gamma
    a, b, c = symbols('a b c', cls=Wild, exclude=(gamma,))

    # 断言 f 匹配 a * log(gamma) + b * gamma + c 的结果为指定的字典
    assert f.match(a * log(gamma) + b * gamma + c) == \
        {a: Rational(-1, 2), b: -(-mu + x)**2/2, c: log(2*pi)/2}
    # 断言 f 展开后再进行 gamma 收集，匹配 a * log(gamma) + b * gamma + c 的结果为指定的字典
    assert f.expand().collect(gamma).match(a * log(gamma) + b * gamma + c) == \
        {a: Rational(-1, 2), b: (-(x - mu)**2/2).expand(), c: (log(2*pi)/2).expand()}
    # 定义三个 Wild 类对象 g1, g2, g3，并排除变量 gamma
    g1 = Wild('g1', exclude=[gamma])
    g2 = Wild('g2', exclude=[gamma])
    g3 = Wild('g3', exclude=[gamma])

    # 断言 f 展开后匹配 g1 * log(gamma)
    # 计算 g(x)*f'(x) 的导数，并对结果进行求导
    eq = diff(g(x)*f(x).diff(x), x)

    # 使用模式匹配来验证等式，确保等式左边匹配到右边的表达式
    assert eq.match(
        g(x).diff(x)*f(x).diff(x) + g(x)*f(x).diff(x, x) + c) == {c: 0}

    # 使用模式匹配来验证等式，确保等式左边匹配到右边的表达式，并确定其中的系数
    assert eq.match(a*g(x).diff(
        x)*f(x).diff(x) + b*g(x)*f(x).diff(x, x) + c) == {a: 1, b: 1, c: 0}
def test_issue_4700():
    # 创建函数符号 f(x)
    f = Function('f')
    # 创建符号变量 x
    x = Symbol('x')
    # 创建两个野字符号 a 和 b，排除 f(x) 符号
    a, b = symbols('a b', cls=Wild, exclude=(f(x),))

    # 创建表达式 p = a*f(x) + b
    p = a*f(x) + b
    # 创建不同的表达式 eq1, eq2, eq3, eq4
    eq1 = sin(x)
    eq2 = f(x) + sin(x)
    eq3 = f(x) + x + sin(x)
    eq4 = x + sin(x)

    # 断言各表达式与 p 的匹配结果
    assert eq1.match(p) == {a: 0, b: sin(x)}
    assert eq2.match(p) == {a: 1, b: sin(x)}
    assert eq3.match(p) == {a: 1, b: x + sin(x)}
    assert eq4.match(p) == {a: 0, b: x + sin(x)}


def test_issue_5168():
    # 创建三个野字符号 a, b, c
    a, b, c = symbols('a b c', cls=Wild)
    # 创建符号变量 x
    x = Symbol('x')
    # 创建函数符号 f(x)
    f = Function('f')

    # 断言 x 与不同模式的匹配结果
    assert x.match(a) == {a: x}
    assert x.match(a*f(x)**c) == {a: x, c: 0}
    assert x.match(a*b) == {a: 1, b: x}
    assert x.match(a*b*f(x)**c) == {a: 1, b: x, c: 0}

    # 断言 -x 与不同模式的匹配结果
    assert (-x).match(a) == {a: -x}
    assert (-x).match(a*f(x)**c) == {a: -x, c: 0}
    assert (-x).match(a*b) == {a: -1, b: x}
    assert (-x).match(a*b*f(x)**c) == {a: -1, b: x, c: 0}

    # 断言 2*x 与不同模式的匹配结果
    assert (2*x).match(a) == {a: 2*x}
    assert (2*x).match(a*f(x)**c) == {a: 2*x, c: 0}
    assert (2*x).match(a*b) == {a: 2, b: x}
    assert (2*x).match(a*b*f(x)**c) == {a: 2, b: x, c: 0}

    # 断言 -2*x 与不同模式的匹配结果
    assert (-2*x).match(a) == {a: -2*x}
    assert (-2*x).match(a*f(x)**c) == {a: -2*x, c: 0}
    assert (-2*x).match(a*b) == {a: -2, b: x}
    assert (-2*x).match(a*b*f(x)**c) == {a: -2, b: x, c: 0}


def test_issue_4559():
    # 创建符号变量 x 和 e
    x = Symbol('x')
    e = Symbol('e')
    # 创建两个野字符号 w 和 y，排除 x 符号
    w = Wild('w', exclude=[x])
    y = Wild('y')

    # 断言不同表达式与模式的匹配结果
    assert (3/x).match(w/y) == {w: 3, y: x}
    assert (3*x).match(w*y) == {w: 3, y: x}
    assert (x/3).match(y/w) == {w: 3, y: x}
    assert (3*x).match(y/w) == {w: S.One/3, y: x}
    assert (3*x).match(y/w) == {w: Rational(1, 3), y: x}

    # 断言更多的匹配结果
    assert (x/3).match(w/y) == {w: S.One/3, y: 1/x}
    assert (3*x).match(w/y) == {w: 3, y: 1/x}
    assert (3/x).match(w*y) == {w: 3, y: 1/x}

    # 使用有理数符号 r
    r = Symbol('r', rational=True)
    assert (x**r).match(y**2) == {y: x**(r/2)}
    assert (x**e).match(y**2) == {y: sqrt(x**e)}

    # 断言更多的匹配结果
    a = Wild('a')
    e = S.Zero
    assert e.match(a) == {a: e}
    assert e.match(1/a) is None
    assert e.match(a**.3) is None

    e = S(3)
    assert e.match(1/a) == {a: 1/e}
    assert e.match(1/a**2) == {a: 1/sqrt(e)}
    e = pi
    assert e.match(1/a) == {a: 1/e}
    assert e.match(1/a**2) == {a: 1/sqrt(e)}
    assert (-e).match(sqrt(a)) is None
    assert (-e).match(a**2) == {a: I*sqrt(pi)}
    # 根据给定的列表生成平方列表，其中每个元素是 x-a 和 a-x 的平方
    p = [i**2 for i in (x - a, a - x)]
    
    # 遍历列表 e 中的每个元素 eq
    for eq in e:
        # 遍历列表 p 中的每个元素 pat
        for pat in p:
            # 断言表达式 eq.match(pat) 返回的结果为 {a: 2}
            assert eq.match(pat) == {a: 2}
def test_issue_4319():
    x, y = symbols('x y')  # 定义符号变量 x 和 y

    p = -x*(S.One/8 - y)  # 设置表达式 p
    ans = {S.Zero, y - S.One/8}  # 定义预期结果集 ans

    def ok(pat):
        assert set(p.match(pat).values()) == ans  # 使用 p 匹配传入的模式 pat

    ok(Wild("coeff", exclude=[x])*x + Wild("rest"))  # 测试匹配模式1
    ok(Wild("w", exclude=[x])*x + Wild("rest"))  # 测试匹配模式2
    ok(Wild("coeff", exclude=[x])*x + Wild("rest"))  # 测试匹配模式3
    ok(Wild("w", exclude=[x])*x + Wild("rest"))  # 测试匹配模式4
    ok(Wild("e", exclude=[x])*x + Wild("rest"))  # 测试匹配模式5
    ok(Wild("ress", exclude=[x])*x + Wild("rest"))  # 测试匹配模式6
    ok(Wild("resu", exclude=[x])*x + Wild("rest"))  # 测试匹配模式7


def test_issue_3778():
    p, c, q = symbols('p c q', cls=Wild)  # 定义符号变量 p, c, q
    x = Symbol('x')  # 定义符号变量 x

    assert (sin(x)**2).match(sin(p)*sin(q)*c) == {q: x, c: 1, p: x}  # 断言匹配结果1
    assert (2*sin(x)).match(sin(p) + sin(q) + c) == {q: x, c: 0, p: x}  # 断言匹配结果2


def test_issue_6103():
    x = Symbol('x')  # 定义符号变量 x
    a = Wild('a')  # 定义通配符 a

    assert (-I*x*oo).match(I*a*oo) == {a: -x}  # 断言匹配结果


def test_issue_3539():
    a = Wild('a')  # 定义通配符 a
    x = Symbol('x')  # 定义符号变量 x

    assert (x - 2).match(a - x) is None  # 断言不匹配结果1
    assert (6/x).match(a*x) is None  # 断言不匹配结果2
    assert (6/x**2).match(a/x) == {a: 6/x}  # 断言匹配结果


def test_gh_issue_2711():
    x = Symbol('x')  # 定义符号变量 x
    f = meijerg(((), ()), ((0,), ()), x)  # 定义特殊函数 f
    a = Wild('a')  # 定义通配符 a
    b = Wild('b')  # 定义通配符 b

    assert f.find(a) == {(S.Zero,), ((), ()), ((S.Zero,), ()), x, S.Zero,
                             (), meijerg(((), ()), ((S.Zero,), ()), x)}  # 断言查找结果1
    assert f.find(a + b) == \
        {meijerg(((), ()), ((S.Zero,), ()), x), x, S.Zero}  # 断言查找结果2
    assert f.find(a**2) == {meijerg(((), ()), ((S.Zero,), ()), x), x}  # 断言查找结果3


def test_issue_17354():
    from sympy.core.symbol import (Wild, symbols)  # 导入通配符和符号变量
    x, y = symbols("x y", real=True)  # 定义实数域符号变量 x 和 y
    a, b = symbols("a b", cls=Wild)  # 定义通配符 a 和 b

    assert ((0 <= x).reversed | (y <= x)).match((1/a <= b) | (a <= b)) is None  # 断言不匹配结果


def test_match_issue_17397():
    f = Function("f")  # 定义函数 f
    x = Symbol("x")  # 定义符号变量 x
    a3 = Wild('a3', exclude=[f(x), f(x).diff(x), f(x).diff(x, 2)])  # 定义通配符 a3
    b3 = Wild('b3', exclude=[f(x), f(x).diff(x), f(x).diff(x, 2)])  # 定义通配符 b3
    c3 = Wild('c3', exclude=[f(x), f(x).diff(x), f(x).diff(x, 2)])  # 定义通配符 c3
    deq = a3*(f(x).diff(x, 2)) + b3*f(x).diff(x) + c3*f(x)  # 定义微分方程 deq

    eq = (x-2)**2*(f(x).diff(x, 2)) + (x-2)*(f(x).diff(x)) + ((x-2)**2 - 4)*f(x)  # 定义等式 eq
    r = collect(eq, [f(x).diff(x, 2), f(x).diff(x), f(x)]).match(deq)  # 对 eq 使用 collect 函数匹配 deq
    assert r == {a3: (x - 2)**2, c3: (x - 2)**2 - 4, b3: x - 2}  # 断言匹配结果1

    eq = x*f(x) + x*Derivative(f(x), (x, 2)) - 4*f(x) + Derivative(f(x), x) \
        - 4*Derivative(f(x), (x, 2)) - 2*Derivative(f(x), x)/x + 4*Derivative(f(x), (x, 2))/x  # 定义新的等式 eq
    r = collect(eq, [f(x).diff(x, 2), f(x).diff(x), f(x)]).match(deq)  # 对 eq 使用 collect 函数匹配 deq
    assert r == {a3: x - 4 + 4/x, b3: 1 - 2/x, c3: x - 4}  # 断言匹配结果2


def test_match_issue_21942():
    a, r, w = symbols('a, r, w', nonnegative=True)  # 定义非负符号变量 a, r, w
    p = symbols('p', positive=True)  # 定义正数符号变量 p
    g_ = Wild('g')  # 定义通配符 g
    pattern = g_ ** (1 / (1 - p))  # 定义模式 pattern
    eq = (a * r ** (1 - p) + w ** (1 - p) * (1 - a)) ** (1 / (1 - p))  # 定义等式 eq
    m = {g_: a * r ** (1 - p) + w ** (1 - p) * (1 - a)}  # 定义预期匹配结果集 m

    assert pattern.matches(eq) == m  # 断言匹配结果1
    assert (-pattern).matches(-eq) == m  # 断言匹配结果2
    assert pattern.matches(signsimp(eq)) is None  # 断言不匹配结果
    # 使用 SymPy 库中的 Wild 函数，创建两个通配符变量 X 和 Y，分别代表通配符表达式中的 X 和 Y
    X, Y = map(Wild, "XY")
    
    # 使用 SymPy 库中的 symbols 函数，创建符号变量 x, y, z，分别代表数学符号表达式中的 x, y, z
    x, y, z = symbols('x y z')
    
    # 断言：验证数学表达式 (5*y - x) 是否匹配模式 (5*X - Y)，并且返回匹配结果的字典
    assert (5*y - x).match(5*X - Y) == {X: y, Y: x}
    # 15907
    
    # 断言：验证数学表达式 (x + (y - 1)*z) 是否匹配模式 (x + X*z)，并且返回匹配结果的字典
    assert (x + (y - 1)*z).match(x + X*z) == {X: y - 1}
    # 20747
    
    # 断言：验证数学表达式 (x - log(x/y)*(1-exp(x/y))) 是否匹配模式 (x - log(X/y)*(1-exp(x/y)))，并且返回匹配结果的字典
    assert (x - log(x/y)*(1-exp(x/y))).match(x - log(X/y)*(1-exp(x/y))) == {X: x}
# 定义一个测试函数，用于测试符号匹配功能
def test_match_bound():
    # 创建通配符对象 V 和 W，并赋值给 V 和 W 变量
    V, W = map(Wild, "VW")
    # 创建符号变量 x 和 y
    x, y = symbols('x y')
    # 断言：对 Sum(x, (x, 1, 2)) 进行模式匹配，期望匹配到 Sum(y, (y, 1, W))，结果应为 {W: 2}
    assert Sum(x, (x, 1, 2)).match(Sum(y, (y, 1, W))) == {W: 2}
    # 断言：对 Sum(x, (x, 1, 2)) 进行模式匹配，期望匹配到 Sum(V, (V, 1, W))，结果应为 {W: 2, V: x}
    assert Sum(x, (x, 1, 2)).match(Sum(V, (V, 1, W))) == {W: 2, V: x}
    # 断言：对 Sum(x, (x, 1, 2)) 进行模式匹配，期望匹配到 Sum(V, (V, 1, 2))，结果应为 {V: x}
    assert Sum(x, (x, 1, 2)).match(Sum(V, (V, 1, 2))) == {V: x}


# 定义一个测试函数，用于测试 issue 22462 的问题
def test_issue_22462():
    # 创建符号变量 x 和 f(x)
    x, f = symbols('x'), Function('f')
    # 创建通配符变量 n 和 Q
    n, Q = symbols('n Q', cls=Wild)
    # 创建模式表达式 -Q*f(x)**n
    pattern = -Q*f(x)**n
    # 创建等式表达式 5*f(x)**2
    eq = 5*f(x)**2
    # 断言：对模式表达式 pattern 进行匹配，期望匹配到等式表达式 eq，结果应为 {n: 2, Q: -5}
    assert pattern.matches(eq) == {n: 2, Q: -5}
```