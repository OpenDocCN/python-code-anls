# `D:\src\scipysrc\sympy\sympy\solvers\tests\test_constantsimp.py`

```
"""
如果issue 4435中的任意常量类被实现，这些测试用例将作为一组测试案例。
"""

# 从 sympy 库导入不同的模块和函数
from sympy.core.function import Function
from sympy.core.numbers import I
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (cosh, sinh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, cos, sin)
from sympy.integrals.integrals import Integral
from sympy.solvers.ode.ode import constantsimp, constant_renumber
from sympy.testing.pytest import XFAIL

# 定义多个符号变量
x = Symbol('x')
y = Symbol('y')
z = Symbol('z')
u2 = Symbol('u2')
_a = Symbol('_a')
C1 = Symbol('C1')
C2 = Symbol('C2')
C3 = Symbol('C3')
f = Function('f')

# 定义测试函数 test_constant_mul()
def test_constant_mul():
    # 测试常数重编号功能，确保常数 C1 只吸收 y，但不吸收 x
    assert constant_renumber(constantsimp(y*C1, [C1])) == C1*y
    assert constant_renumber(constantsimp(C1*y, [C1])) == C1*y
    assert constant_renumber(constantsimp(x*C1, [C1])) == x*C1
    assert constant_renumber(constantsimp(C1*x, [C1])) == x*C1
    assert constant_renumber(constantsimp(2*C1, [C1])) == C1
    assert constant_renumber(constantsimp(C1*2, [C1])) == C1
    assert constant_renumber(constantsimp(y*C1*x, [C1, y])) == C1*x
    assert constant_renumber(constantsimp(x*y*C1, [C1, y])) == x*C1
    assert constant_renumber(constantsimp(y*x*C1, [C1, y])) == x*C1
    assert constant_renumber(constantsimp(C1*x*y, [C1, y])) == C1*x
    assert constant_renumber(constantsimp(x*C1*y, [C1, y])) == x*C1
    assert constant_renumber(constantsimp(C1*y*(y + 1), [C1])) == C1*y*(y+1)
    assert constant_renumber(constantsimp(y*C1*(y + 1), [C1])) == C1*y*(y+1)
    assert constant_renumber(constantsimp(x*(y*C1), [C1])) == x*y*C1
    assert constant_renumber(constantsimp(x*(C1*y), [C1])) == x*y*C1
    assert constant_renumber(constantsimp(C1*(x*y), [C1, y])) == C1*x
    assert constant_renumber(constantsimp((x*y)*C1, [C1, y])) == x*C1
    assert constant_renumber(constantsimp((y*x)*C1, [C1, y])) == x*C1
    assert constant_renumber(constantsimp(y*(y + 1)*C1, [C1, y])) == C1
    assert constant_renumber(constantsimp((C1*x)*y, [C1, y])) == C1*x
    assert constant_renumber(constantsimp(y*(x*C1), [C1, y])) == x*C1
    assert constant_renumber(constantsimp((x*C1)*y, [C1, y])) == x*C1
    assert constant_renumber(constantsimp(C1*x*y*x*y*2, [C1, y])) == C1*x**2
    assert constant_renumber(constantsimp(C1*x*y*z, [C1, y, z])) == C1*x
    assert constant_renumber(constantsimp(C1*x*y**2*sin(z), [C1, y, z])) == C1*x
    assert constant_renumber(constantsimp(C1*C1, [C1])) == C1
    assert constant_renumber(constantsimp(C1*C2, [C1, C2])) == C1
    assert constant_renumber(constantsimp(C2*C2, [C1, C2])) == C1
    assert constant_renumber(constantsimp(C1*C1*C2, [C1, C2])) == C1
    # 断言语句，用于检查常数重编号后的简化结果是否等于原始表达式
    assert constant_renumber(constantsimp(C1*x*2**x, [C1])) == C1*x*2**x
# 定义测试函数 `test_constant_add()`，用于测试常量加法表达式的简化结果是否正确
def test_constant_add():
    # 断言常量简化后，两个相同的常量相加结果为该常量本身
    assert constant_renumber(constantsimp(C1 + C1, [C1])) == C1
    # 断言常量简化后，常量加上数字的结果为该常量本身
    assert constant_renumber(constantsimp(C1 + 2, [C1])) == C1
    # 断言常量简化后，数字加上常量的结果为该常量本身
    assert constant_renumber(constantsimp(2 + C1, [C1])) == C1
    # 断言常量简化后，常量加上变量的结果为该常量本身
    assert constant_renumber(constantsimp(C1 + y, [C1, y])) == C1
    # 断言常量简化后，常量加上变量的结果为常量加上该变量
    assert constant_renumber(constantsimp(C1 + x, [C1])) == C1 + x
    # 再次验证常量简化后，两个相同的常量相加结果为该常量本身
    assert constant_renumber(constantsimp(C1 + C1, [C1])) == C1
    # 断言常量简化后，两个不同的常量相加结果为第一个常量
    assert constant_renumber(constantsimp(C1 + C2, [C1, C2])) == C1
    # 断言常量简化后，两个不同的常量相加顺序无关
    assert constant_renumber(constantsimp(C2 + C1, [C1, C2])) == C1
    # 断言常量简化后，常量加上两个不同的常量的结果为第一个常量
    assert constant_renumber(constantsimp(C1 + C2 + C1, [C1, C2])) == C1


# 定义测试函数 `test_constant_power_as_base()`，用于测试常量作为幂底数的情况下的简化结果是否正确
def test_constant_power_as_base():
    # 断言常量简化后，常量的幂次幂的结果为该常量本身
    assert constant_renumber(constantsimp(C1**C1, [C1])) == C1
    # 断言常量简化后，幂函数的底数是常量时的简化结果为该常量本身
    assert constant_renumber(constantsimp(Pow(C1, C1), [C1])) == C1
    # 再次验证常量简化后，常量的幂次幂的结果为该常量本身
    assert constant_renumber(constantsimp(C1**C1, [C1])) == C1
    # 断言常量简化后，常量的幂次幂的结果为该常量本身
    assert constant_renumber(constantsimp(C1**C2, [C1, C2])) == C1
    # 断言常量简化后，常量的幂次幂的结果为第一个常量
    assert constant_renumber(constantsimp(C2**C1, [C1, C2])) == C1
    # 断言常量简化后，常量的幂次幂的结果为第一个常量
    assert constant_renumber(constantsimp(C2**C2, [C1, C2])) == C1
    # 断言常量简化后，常量的幂次幂的结果为该常量本身
    assert constant_renumber(constantsimp(C1**y, [C1, y])) == C1
    # 断言常量简化后，常量的幂次幂的结果为常量的幂次幂
    assert constant_renumber(constantsimp(C1**x, [C1])) == C1**x
    # 断言常量简化后，常量的幂次幂的结果为该常量本身
    assert constant_renumber(constantsimp(C1**2, [C1])) == C1
    # 断言常量简化后，常量的幂次幂的结果为常量的幂次幂
    assert constant_renumber(constantsimp(C1**(x*y), [C1])) == C1**(x*y)


# 定义测试函数 `test_constant_power_as_exp()`，用于测试常量作为幂指数的情况下的简化结果是否正确
def test_constant_power_as_exp():
    # 断言常量简化后，变量的幂函数以常量为指数的结果为变量的幂函数
    assert constant_renumber(constantsimp(x**C1, [C1])) == x**C1
    # 断言常量简化后，变量的幂函数以常量为指数的结果为该常量本身
    assert constant_renumber(constantsimp(y**C1, [C1, y])) == C1
    # 断言常量简化后，变量的幂函数以变量和常量为指数的结果为变量的幂函数
    assert constant_renumber(constantsimp(x**y**C1, [C1, y])) == x**C1
    # 断言常量简化后，变量的幂函数以变量的幂函数为基数和常量为指数的结果为变量的幂函数
    assert constant_renumber(constantsimp((x**y)**C1, [C1])) == (x**y)**C1
    # 断言常量简化后，变量的幂函数以变量的幂函数为基数和常量为指数的结果为变量的幂函数
    assert constant_renumber(constantsimp(x**(y**C1), [C1, y])) == x**C1
    # 断言常量简化后，变量的幂函数以常量和变量为指数的结果为变量的幂函数
    assert constant_renumber(constantsimp(x**C1**y, [C1, y])) == x**C1
    # 断言常量简化后，变量的幂函数以常量的幂函数为指数的结果为变量的幂函数
    assert constant_renumber(constantsimp(x**(C1**y), [C1, y])) == x**C1
    # 断言常量简化后，变量的幂函数以常量为指数的结果为变量的幂函数
    assert constant_renumber(constantsimp((x**C1)**y, [C1])) == (x**C1)**y
    # 断言常量简化后，常数的幂函数以常量为指数的结果为该常量本身
    assert constant_renumber(constantsimp(2**C1, [C1])) == C1
    # 断言常量简化后，有理数的幂函数以常量为指数的结果为该常量本身
    assert constant_renumber(constantsimp(S(2)**C1, [C1])) == C1
    # 断言常量简化后，指数函数以常量为指数的结果为该常量本身
    assert constant_renumber(constantsimp(exp(C1), [C1])) == C1
    # 断言常量简化后，指数函数以常量和变量相加为指数的结果为常量乘以指数函数
    assert constant_renumber(constantsimp(exp(C1 + x), [C1])) == C1*exp(x)
    # 断言常量简化后，指数函数以常量为指数的结果为常量的指数函数
    assert constant_renumber(constantsimp(Pow(2, C1), [C1])) == C1


# 定义测试函数 `test_constant_function()`，用于测试常量作为函数参数的情况下的简化结果是否正确
def test_constant_function():
    # 断言常量简化后，正弦函数的常量参数的结果为该常量本身
    assert constant_renumber(constantsimp(sin(C1), [C1])) == C1
    # 断言常量简化后，函数的常量参数的结果为该常量本身
    assert constant_renumber(constantsimp(f(C1), [C1])) == C1
    # 断言常量简化后，函数的两个相同的常量参数的结果为该常量本身
    assert constant_renumber(constantsimp(f(C1, C1), [C1])) == C1
    # 断言常量简化后，函数
    # 断言语句：验证常数重编号后的结果是否等于常数简化后的 f(C1, y, C2)。
    assert constant_renumber(constantsimp(f(C1, y, C2), [C1, C2, y])) == C1
# 定义测试函数，测试常数重编号函数 constant_renumber 的多个用例
def test_constant_function_multiple():
    # 断言常数重编号后的结果与原始表达式相同
    assert constant_renumber(
        constantsimp(f(C1, C1, x), [C1])) == f(C1, C1, x)


# 定义测试函数，测试常数重编号函数 constant_renumber 的多个用例
def test_constant_multiple():
    # 断言常数重编号后的结果与预期的常数 C1 相同
    assert constant_renumber(constantsimp(C1*2 + 2, [C1])) == C1
    # 断言常数重编号后的结果与预期的表达式 C1*x 相同
    assert constant_renumber(constantsimp(x*2/C1, [C1])) == C1*x
    # 断言常数重编号后的结果与预期的常数 C1 相同
    assert constant_renumber(constantsimp(C1**2*2 + 2, [C1])) == C1
    # 断言常数重编号后的结果与预期的表达式 C1 + x 相同
    assert constant_renumber(
        constantsimp(sin(2*C1) + x + sqrt(2), [C1])) == C1 + x
    # 断言常数重编号后的结果与预期的常数 C1 相同
    assert constant_renumber(constantsimp(2*C1 + C2, [C1, C2])) == C1


# 定义测试函数，测试常数重编号函数 constant_renumber 的一个用例
def test_constant_repeated():
    # 断言常数重编号后的结果与原始表达式相同
    assert C1 + C1*x == constant_renumber( C1 + C1*x)


# 定义测试函数，测试常数重编号函数 constant_renumber 的多个用例
def test_ode_solutions():
    # 断言常数重编号后的结果与预期的表达式相同
    assert constant_renumber(constantsimp(C1*exp(2*x) + exp(x)*(C2 + C3), [C1, C2, C3])) == \
        constant_renumber(C1*exp(x) + C2*exp(2*x))
    # 断言常数重编号后的结果与预期的方程相同
    assert constant_renumber(
        constantsimp(Eq(f(x), I*C1*sinh(x/3) + C2*cosh(x/3)), [C1, C2])
        ) == constant_renumber(Eq(f(x), C1*sinh(x/3) + C2*cosh(x/3)))
    # 断言常数重编号后的结果与预期的方程相同
    assert constant_renumber(constantsimp(Eq(f(x), acos((-C1)/cos(x))), [C1])) == \
        Eq(f(x), acos(C1/cos(x)))
    # 断言常数重编号后的结果与预期的方程相同
    assert constant_renumber(
        constantsimp(Eq(log(f(x)/C1) + 2*exp(x/f(x)), 0), [C1])
        ) == Eq(log(C1*f(x)) + 2*exp(x/f(x)), 0)
    # 断言常数重编号后的结果与预期的方程相同
    assert constant_renumber(constantsimp(Eq(log(x*sqrt(2)*sqrt(1/x)*sqrt(f(x))
        /C1) + x**2/(2*f(x)**2), 0), [C1])) == \
        Eq(log(C1*sqrt(x)*sqrt(f(x))) + x**2/(2*f(x)**2), 0)
    # 断言常数重编号后的结果与预期的方程相同
    assert constant_renumber(constantsimp(Eq(-exp(-f(x)/x)*sin(f(x)/x)/2 + log(x/C1) -
        cos(f(x)/x)*exp(-f(x)/x)/2, 0), [C1])) == \
        Eq(-exp(-f(x)/x)*sin(f(x)/x)/2 + log(C1*x) - cos(f(x)/x)*
           exp(-f(x)/x)/2, 0)
    # 断言常数重编号后的结果与预期的方程相同
    assert constant_renumber(constantsimp(Eq(-Integral(-1/(sqrt(1 - u2**2)*u2),
        (u2, _a, x/f(x))) + log(f(x)/C1), 0), [C1])) == \
        Eq(-Integral(-1/(u2*sqrt(1 - u2**2)), (u2, _a, x/f(x))) +
        log(C1*f(x)), 0)
    # 断言常数重编号后的结果与预期的列表中的方程相同
    assert [constantsimp(i, [C1]) for i in [Eq(f(x), sqrt(-C1*x + x**2)), Eq(f(x), -sqrt(-C1*x + x**2))]] == \
        [Eq(f(x), sqrt(x*(C1 + x))), Eq(f(x), -sqrt(x*(C1 + x)))]

# 定义标记为 XFAIL 的测试函数，测试在非局部简化的情况下的常数重编号函数 constant_renumber 的用例
@XFAIL
def test_nonlocal_simplification():
    # 断言常数重编号后的结果与预期的表达式相同
    assert constantsimp(C1 + C2+x*C2, [C1, C2]) == C1 + C2*x


# 定义测试函数，测试常数重编号函数 constant_renumber 的多个用例
def test_constant_Eq():
    # 断言常数重编号后的结果与预期的方程相同
    assert constantsimp(Eq(C1, 3 + f(x)*x), [C1]) == Eq(x*f(x), C1)
    # 断言常数重编号后的结果与预期的方程相同
    assert constantsimp(Eq(C1, 3 * f(x)*x), [C1]) == Eq(f(x)*x, C1)
```