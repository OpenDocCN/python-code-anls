# `D:\src\scipysrc\sympy\sympy\parsing\tests\test_latex_lark.py`

```
from sympy.testing.pytest import XFAIL  # 导入 XFAIL，用于标记预期失败的测试用例
from sympy.parsing.latex.lark import parse_latex_lark  # 导入 LaTeX 解析器的具体函数
from sympy.external import import_module  # 导入用于动态导入模块的函数

from sympy.concrete.products import Product  # 导入乘积相关的类和函数
from sympy.concrete.summations import Sum  # 导入求和相关的类和函数
from sympy.core.function import Derivative, Function  # 导入导数和函数相关的类
from sympy.core.numbers import E, oo, Rational  # 导入常数 e、无穷大和有理数
from sympy.core.power import Pow  # 导入幂运算相关的类和函数
from sympy.core.parameters import evaluate  # 导入参数评估相关的函数
from sympy.core.relational import GreaterThan, LessThan, StrictGreaterThan, StrictLessThan, Unequality  # 导入不同类型的比较关系类
from sympy.core.symbol import Symbol  # 导入符号变量类
from sympy.functions.combinatorial.factorials import binomial, factorial  # 导入组合数和阶乘函数
from sympy.functions.elementary.complexes import Abs, conjugate  # 导入复数相关函数
from sympy.functions.elementary.exponential import exp, log  # 导入指数和对数函数
from sympy.functions.elementary.integers import ceiling, floor  # 导入取天花板和取地板函数
from sympy.functions.elementary.miscellaneous import root, sqrt, Min, Max  # 导入根号、平方根和最小最大值函数
from sympy.functions.elementary.trigonometric import asin, cos, csc, sec, sin, tan  # 导入三角函数
from sympy.integrals.integrals import Integral  # 导入积分相关类
from sympy.series.limits import Limit  # 导入极限相关类

from sympy.core.relational import Eq, Ne, Lt, Le, Gt, Ge  # 导入等于、不等于、小于、小于等于、大于、大于等于关系类
from sympy.physics.quantum import Bra, Ket, InnerProduct  # 导入量子力学相关类
from sympy.abc import x, y, z, a, b, c, d, t, k, n  # 导入符号变量

from .test_latex import theta, f, _Add, _Mul, _Pow, _Sqrt, _Conjugate, _Abs, _factorial, _exp, _binomial  # 导入测试所需的函数和变量

lark = import_module("lark")  # 尝试导入名为 "lark" 的模块

# 如果没有成功导入 lark 模块，则禁用相关的测试
disabled = lark is None

# 定义一些仅用于 Lark LaTeX 解析器的简写函数

def _Min(*args):
    return Min(*args, evaluate=False)

def _Max(*args):
    return Max(*args, evaluate=False)

def _log(a, b=E):
    if b == E:
        return log(a, evaluate=False)
    else:
        return log(a, b, evaluate=False)

# 这些 LaTeX 字符串应该解析为相应的 SymPy 表达式的对
SYMBOL_EXPRESSION_PAIRS = [
    (r"x_0", Symbol('x_{0}')),
    (r"x_{1}", Symbol('x_{1}')),
    (r"x_a", Symbol('x_{a}')),
    (r"x_{b}", Symbol('x_{b}')),
    (r"h_\theta", Symbol('h_{theta}')),
    (r"h_{\theta}", Symbol('h_{theta}')),
    (r"y''_1", Symbol("y_{1}''")),
    (r"y_1''", Symbol("y_{1}''")),
    (r"\mathit{x}", Symbol('x')),
    (r"\mathit{test}", Symbol('test')),
    (r"\mathit{TEST}", Symbol('TEST')),
    (r"\mathit{HELLO world}", Symbol('HELLO world'))
]

# 这些未评估的简单表达式对应的 SymPy 表达式
UNEVALUATED_SIMPLE_EXPRESSION_PAIRS = [
    (r"0", 0),
    (r"1", 1),
    (r"-3.14", -3.14),
    (r"(-7.13)(1.5)", _Mul(-7.13, 1.5)),
    (r"1+1", _Add(1, 1)),
    (r"0+1", _Add(0, 1)),
    (r"1*2", _Mul(1, 2)),
    (r"0*1", _Mul(0, 1)),
    (r"x", x),
    (r"2x", 2 * x),
    (r"3x - 1", _Add(_Mul(3, x), -1)),
    (r"-c", -c),
    (r"\infty", oo),
    (r"a \cdot b", a * b),
    (r"1 \times 2 ", _Mul(1, 2)),
    (r"a / b", a / b),
    (r"a \div b", a / b),
    (r"a + b", a + b),
    (r"a + b - a", _Add(a + b, -a)),
    (r"(x + y) z", _Mul(_Add(x, y), z)),
    (r"a'b+ab'", _Add(_Mul(Symbol("a'"), b), _Mul(a, Symbol("b'"))))
]

# 这些已评估的简单表达式对应的 SymPy 表达式
EVALUATED_SIMPLE_EXPRESSION_PAIRS = [
    (r"(-7.13)(1.5)", -10.695),
    (r"1+1", 2),
    `
    # 定义一组测试用例，每个元组包含一个正则表达式和一个期望的结果
    (r"0+1", 1),             # 正则表达式匹配 '0+1'，预期结果为 1
    (r"1*2", 2),             # 正则表达式匹配 '1*2'，预期结果为 2
    (r"0*1", 0),             # 正则表达式匹配 '0*1'，预期结果为 0
    (r"2x", 2 * x),           # 正则表达式匹配 '2x'，预期结果为 2 * x
    (r"3x - 1", 3 * x - 1),   # 正则表达式匹配 '3x - 1'，预期结果为 3 * x - 1
    (r"-c", -c),              # 正则表达式匹配 '-c'，预期结果为 -c
    (r"a \cdot b", a * b),     # 正则表达式匹配 'a · b'，预期结果为 a * b
    (r"1 \times 2 ", 2),       # 正则表达式匹配 '1 × 2'，预期结果为 2
    (r"a / b", a / b),         # 正则表达式匹配 'a / b'，预期结果为 a / b
    (r"a \div b", a / b),      # 正则表达式匹配 'a ÷ b'，预期结果为 a / b
    (r"a + b", a + b),         # 正则表达式匹配 'a + b'，预期结果为 a + b
    (r"a + b - a", b),         # 正则表达式匹配 'a + b - a'，预期结果为 b
    (r"(x + y) z", (x + y) * z),# 正则表达式匹配 '(x + y) z'，预期结果为 (x + y) * z
# 未评估的分数表达式对，包含 LaTeX 格式的分数表达式及其对应的未简化数学表达式
UNEVALUATED_FRACTION_EXPRESSION_PAIRS = [
    (r"\frac{a}{b}", a / b),                 # a 除以 b
    (r"\dfrac{a}{b}", a / b),                # a 除以 b
    (r"\tfrac{a}{b}", a / b),                # a 除以 b
    (r"\frac12", _Mul(1, _Pow(2, -1))),      # 1 乘以 2 的负1次方，即 1/2
    (r"\frac12y", _Mul(_Mul(1, _Pow(2, -1)), y)),  # (1/2) * y
    (r"\frac1234", _Mul(_Mul(1, _Pow(2, -1)), 34)),  # (1/2) * 34
    (r"\frac2{3}", _Mul(2, _Pow(3, -1))),    # 2 除以 3
    (r"\frac{a + b}{c}", _Mul(a + b, _Pow(c, -1))),  # (a + b) 除以 c
    (r"\frac{7}{3}", _Mul(7, _Pow(3, -1)))   # 7 除以 3
]

# 已评估的分数表达式对，包含 LaTeX 格式的分数表达式及其对应的简化数学表达式
EVALUATED_FRACTION_EXPRESSION_PAIRS = [
    (r"\frac{a}{b}", a / b),                 # a 除以 b
    (r"\dfrac{a}{b}", a / b),                # a 除以 b
    (r"\tfrac{a}{b}", a / b),                # a 除以 b
    (r"\frac12", Rational(1, 2)),            # 分数形式的 1/2
    (r"\frac12y", y / 2),                    # y 的一半
    (r"\frac1234", 17),                      # 简化的分数表达式为 17
    (r"\frac2{3}", Rational(2, 3)),          # 分数形式的 2/3
    (r"\frac{a + b}{c}", (a + b) / c),       # 简化的分数表达式 (a + b) / c
    (r"\frac{7}{3}", Rational(7, 3))         # 分数形式的 7/3
]

# 关系表达式对，包含 LaTeX 格式的关系表达式及其对应的 SymPy 关系表达式
RELATION_EXPRESSION_PAIRS = [
    (r"x = y", Eq(x, y)),                     # x 等于 y
    (r"x \neq y", Ne(x, y)),                  # x 不等于 y
    (r"x < y", Lt(x, y)),                     # x 小于 y
    (r"x > y", Gt(x, y)),                     # x 大于 y
    (r"x \leq y", Le(x, y)),                  # x 小于等于 y
    (r"x \geq y", Ge(x, y)),                  # x 大于等于 y
    (r"x \le y", Le(x, y)),                   # x 小于等于 y
    (r"x \ge y", Ge(x, y)),                   # x 大于等于 y
    (r"x < y", StrictLessThan(x, y)),         # x 严格小于 y
    (r"x \leq y", LessThan(x, y)),            # x 小于等于 y
    (r"x > y", StrictGreaterThan(x, y)),      # x 严格大于 y
    (r"x \geq y", GreaterThan(x, y)),         # x 大于等于 y
    (r"x \neq y", Unequality(x, y)),          # x 不等于 y，与列表中第二个表达式相同
    (r"a^2 + b^2 = c^2", Eq(a**2 + b**2, c**2))  # a平方加b平方等于c平方
]

# 未评估的幂表达式对，包含 LaTeX 格式的幂表达式及其对应的未简化数学表达式
UNEVALUATED_POWER_EXPRESSION_PAIRS = [
    (r"x^2", x ** 2),                        # x 的平方
    (r"x^\frac{1}{2}", _Pow(x, _Mul(1, _Pow(2, -1)))),  # x 的 1/2 次方
    (r"x^{3 + 1}", x ** _Add(3, 1)),         # x 的 4 次方
    (r"\pi^{|xy|}", Symbol('pi') ** _Abs(x * y)),  # π 的 |xy| 次方
    (r"5^0 - 4^0", _Add(_Pow(5, 0), _Mul(-1, _Pow(4, 0))))  # 5 的 0 次方减去 4 的 0 次方
]

# 已评估的幂表达式对，包含 LaTeX 格式的幂表达式及其对应的简化数学表达式
EVALUATED_POWER_EXPRESSION_PAIRS = [
    (r"x^2", x ** 2),                        # x 的平方
    (r"x^\frac{1}{2}", sqrt(x)),             # x 的平方根
    (r"x^{3 + 1}", x ** 4),                  # x 的 4 次方
    (r"\pi^{|xy|}", Symbol('pi') ** _Abs(x * y)),  # π 的 |xy| 次方
    (r"5^0 - 4^0", 0)                        # 5 的 0 次方减去 4 的 0 次方等于 0
]

# 未评估的积分表达式对，包含 LaTeX 格式的积分表达式及其对应的未简化数学表达式
UNEVALUATED_INTEGRAL_EXPRESSION_PAIRS = [
    (r"\int x dx", Integral(_Mul(1, x), x)),  # 对 x 进行积分，积分变量为 x
    (r"\int x \, dx", Integral(_Mul(1, x), x)),  # 同上，但有附加的空格
    (r"\int x d\theta", Integral(_Mul(1, x), theta)),  # 对 x 进行积分，积分变量为 theta
    (r"\int (x^2 - y)dx", Integral(_Mul(1, x ** 2 - y), x)),  # 对 x^2 - y 进行积分，积分变量为 x
    (r"\int x + a dx", Integral(_Mul(1, _Add(x, a)), x)),  # 对 x + a 进行积分，积分变量为 x
    (r"\int da", Integral(_Mul(1, 1), a)),  # 对 1 进行积分，积分变量为 a
    (r"\int_0^7 dx", Integral(_Mul(1, 1), (x, 0, 7))),  # 对 1 进行积分，积分变量为 x，积分范围为 0 到 7
    (r"\int\limits_{0}^{1} x dx", Integral(_Mul(1, x), (x, 0, 1))),  # 对 x 进行积分，积分变量为 x，积分范围为 0 到 1
    (r"\int_a^b x dx", Integral(_Mul(1, x), (x, a, b))),  # 对 x 进行积分，积分变量为 x，积分范围为 a 到 b
    (r"\int^b_a x dx", Integral(_Mul(1, x), (x, a, b))),  # 同上
    (r"\int_{a}^b x dx", Integral(_Mul(1, x), (x, a, b))),  # 同上
    (r"\int^{b}_a x dx", Integral(_Mul(1, x), (x, a, b))),  # 同上
    # 创建包含两个元组的元组，每个元组包含一个数学表达式和其对应的 SymPy 积分表达式
    (
        # 第一个元组包含数学表达式 "\int \frac{1}{a} + \frac{1}{b} dx" 和其对应的 SymPy 积分表达式
        r"\int \frac{1}{a} + \frac{1}{b} dx",
        Integral(_Mul(1, _Add(_Mul(1, _Pow(a, -1)), _Mul(1, Pow(b, -1)))), x)
    ),
    (
        # 第二个元组包含数学表达式 "\int \frac{1}{x} + 1 dx" 和其对应的 SymPy 积分表达式
        r"\int \frac{1}{x} + 1 dx",
        Integral(_Mul(1, _Add(_Mul(1, _Pow(x, -1)), 1)), x)
    )
# 未被评估的积分表达式对
UNEVALUATED_INTEGRAL_EXPRESSION_PAIRS = [
    # 表达式 r"\int x dx" 对应积分 Integral(x, x)
    (r"\int x dx", Integral(x, x)),
    # 表达式 r"\int x \, dx" 对应积分 Integral(x, x)
    (r"\int x \, dx", Integral(x, x)),
    # 表达式 r"\int x d\theta" 对应积分 Integral(x, theta)
    (r"\int x d\theta", Integral(x, theta)),
    # 表达式 r"\int (x^2 - y)dx" 对应积分 Integral(x ** 2 - y, x)
    (r"\int (x^2 - y)dx", Integral(x ** 2 - y, x)),
    # 表达式 r"\int x + a dx" 对应积分 Integral(x + a, x)
    (r"\int x + a dx", Integral(x + a, x)),
    # 表达式 r"\int da" 对应积分 Integral(1, a)
    (r"\int da", Integral(1, a)),
    # 表达式 r"\int_0^7 dx" 对应积分 Integral(1, (x, 0, 7))
    (r"\int_0^7 dx", Integral(1, (x, 0, 7))),
    # 表达式 r"\int\limits_{0}^{1} x dx" 对应积分 Integral(x, (x, 0, 1))
    (r"\int\limits_{0}^{1} x dx", Integral(x, (x, 0, 1))),
    # 表达式 r"\int_a^b x dx" 对应积分 Integral(x, (x, a, b))
    (r"\int_a^b x dx", Integral(x, (x, a, b))),
    # 表达式 r"\int^b_a x dx" 对应积分 Integral(x, (x, a, b))
    (r"\int^b_a x dx", Integral(x, (x, a, b))),
    # 表达式 r"\int_{a}^b x dx" 对应积分 Integral(x, (x, a, b))
    (r"\int_{a}^b x dx", Integral(x, (x, a, b))),
    # 表达式 r"\int^{b}_a x dx" 对应积分 Integral(x, (x, a, b))
    (r"\int^{b}_a x dx", Integral(x, (x, a, b))),
    # 表达式 r"\int_{a}^{b} x dx" 对应积分 Integral(x, (x, a, b))
    (r"\int_{a}^{b} x dx", Integral(x, (x, a, b))),
    # 表达式 r"\int^{b}_{a} x dx" 对应积分 Integral(x, (x, a, b))
    (r"\int^{b}_{a} x dx", Integral(x, (x, a, b))),
    # 表达式 r"\int_{f(a)}^{f(b)} f(z) dz" 对应积分 Integral(f(z), (z, f(a), f(b)))
    (r"\int_{f(a)}^{f(b)} f(z) dz", Integral(f(z), (z, f(a), f(b)))),
    # 表达式 r"\int a + b + c dx" 对应积分 Integral(a + b + c, x)
    (r"\int a + b + c dx", Integral(a + b + c, x)),
    # 表达式 r"\int \frac{dz}{z}" 对应积分 Integral(Pow(z, -1), z)
    (r"\int \frac{dz}{z}", Integral(Pow(z, -1), z)),
    # 表达式 r"\int \frac{3 dz}{z}" 对应积分 Integral(3 * Pow(z, -1), z)
    (r"\int \frac{3 dz}{z}", Integral(3 * Pow(z, -1), z)),
    # 表达式 r"\int \frac{1}{x} dx" 对应积分 Integral(1 / x, x)
    (r"\int \frac{1}{x} dx", Integral(1 / x, x)),
    # 表达式 r"\int \frac{1}{a} + \frac{1}{b} dx" 对应积分 Integral(1 / a + 1 / b, x)
    (r"\int \frac{1}{a} + \frac{1}{b} dx", Integral(1 / a + 1 / b, x)),
    # 表达式 r"\int \frac{1}{x} + 1 dx" 对应积分 Integral(1 / x + 1, x)
    (r"\int \frac{1}{x} + 1 dx", Integral(1 / x + 1, x))
]

# 未被评估的导数表达式对
DERIVATIVE_EXPRESSION_PAIRS = [
    # 表达式 r"\frac{d}{dx} x" 对应导数 Derivative(x, x)
    (r"\frac{d}{dx} x", Derivative(x, x)),
    # 表达式 r"\frac{d}{dt} x" 对应导数 Derivative(x, t)
    (r"\frac{d}{dt} x", Derivative(x, t)),
    # 表达式 r"\frac{d}{dx} ( \tan x )" 对应导数 Derivative(tan(x), x)
    (r"\frac{d}{dx} ( \tan x )", Derivative(tan(x), x)),
    # 表达式 r"\frac{d f(x)}{dx}" 对应导数 Derivative(f(x), x)
    (r"\frac{d f(x)}{dx}", Derivative(f(x), x)),
    # 表达式 r"\frac{d\theta(x)}{dx}" 对应导数 Derivative(Function('theta')(x), x)
    (r"\frac{d\theta(x)}{dx}", Derivative(Function('theta')(x), x))
]

# 三角函数表达式对
TRIGONOMETRIC_EXPRESSION_PAIRS = [
    # 表达式 r"\sin \theta" 对应 sin(theta)
    (r"\sin \theta", sin(theta)),
    # 表达式 r"\sin(\theta)" 对应 sin(theta)
    (r"\sin(\theta)", sin(theta)),
    # 表达式 r"\sin^{-1} a" 对应 asin(a)
    (r"\sin^{-1} a", asin(a)),
    # 表达式 r"\sin a \cos b" 对应 sin(a) * cos(b)
    (r"\sin a \cos b", _Mul(sin(a), cos(b))),
    # 表达式 r"\sin \cos \theta" 对应 sin(cos(theta))
    (r"\sin \cos \theta", sin(cos(theta))),
    # 表达式 r"\sin(\cos \theta)" 对应 sin(cos(theta))
    (r"\sin(\cos \theta)", sin(cos(theta))),
    # 表达式 r"(\csc x)(\sec y)" 对应 csc(x) * sec(y)
    (r"(\csc x)(\sec y)", csc(x) * sec(y)),
    # 表达式 r"\frac{\sin{x}}2" 对应 sin(x) * (2 ** -1)
    (r"\frac{\sin{x}}2", _Mul(sin(x), _Pow(2, -1)))
]

# 未被评估的极限表达式对
UNEVALUATED_LIMIT_EXPRESSION_PAIRS = [
    # 表达式 r"\lim_{x \to 3} a" 对应极限 Limit(a, x, 3, dir="+-")
    (r"\lim_{x \to 3} a", Limit(a, x, 3, dir="+-")),
    # 表达式 r"\lim_{x \rightarrow 3} a" 对应极限 Limit(a, x, 3, dir="+-")
    (r"\lim_{x \rightarrow 3} a", Limit(a, x, 3, dir="+-")),
    # 表达式 r"\lim_{x \Rightarrow 3} a" 对应极限 Limit(a, x, 3, dir="+-")
    (r"\lim_{x \Rightarrow 3} a", Limit(a, x, 3, dir="+-")),
    # 表达式 r"\lim_{x \longrightarrow 3} a" 对应极限 Limit(a, x, 3, dir="+-"))
    (r"\lim_{x \longrightarrow 3} a", Limit(a, x, 3, dir="+-")),
    # 表达式 r"\lim_{x \Longrightarrow 3} a" 对应极限 Limit(a, x,
EVALUATED_SQRT_EXPRESSION_PAIRS = [
    # 表示平方根表达式 \sqrt{x} 的对应求值表达式 sqrt(x)
    (r"\sqrt{x}", sqrt(x)),
    # 表示平方根表达式 \sqrt{x + b} 的对应求值表达式 sqrt(x + b)
    (r"\sqrt{x + b}", sqrt(x + b)),
    # 表示立方根表达式 \sqrt[3]{\sin x} 的对应求值表达式 root(sin(x), 3)
    (r"\sqrt[3]{\sin x}", root(sin(x), 3)),
    # 表示 y 次根表达式 \sqrt[y]{\sin x} 的对应求值表达式 root(sin(x), y)
    (r"\sqrt[y]{\sin x}", root(sin(x), y)),
    # 表示 \theta 次根表达式 \sqrt[\theta]{\sin x} 的对应求值表达式 root(sin(x), theta)
    (r"\sqrt[\theta]{\sin x}", root(sin(x), theta)),
    # 表示简单的平方根表达式 \sqrt{\frac{12}{6}} 的对应求值表达式 sqrt(2)
    (r"\sqrt{\frac{12}{6}}", sqrt(2))
]

UNEVALUATED_FACTORIAL_EXPRESSION_PAIRS = [
    # 表示未求值的阶乘表达式 x! 的对应表达式 _factorial(x)
    (r"x!", _factorial(x)),
    # 表示未求值的阶乘表达式 100! 的对应表达式 _factorial(100)
    (r"100!", _factorial(100)),
    # 表示未求值的阶乘表达式 \theta! 的对应表达式 _factorial(theta)
    (r"\theta!", _factorial(theta)),
    # 表示未求值的阶乘表达式 (x + 1)! 的对应表达式 _factorial(_Add(x, 1))
    (r"(x + 1)!", _factorial(_Add(x, 1))),
    # 表示未求值的阶乘表达式 (x!)! 的对应表达式 _factorial(_factorial(x))
    (r"(x!)!", _factorial(_factorial(x))),
    # 表示未求值的阶乘表达式 x!!! 的对应表达式 _factorial(_factorial(_factorial(x)))
    (r"x!!!", _factorial(_factorial(_factorial(x)))),
    # 表示未求值的阶乘表达式 5!7! 的对应表达式 _Mul(_factorial(5), _factorial(7))
    (r"5!7!", _Mul(_factorial(5), _factorial(7)))
]

EVALUATED_FACTORIAL_EXPRESSION_PAIRS = [
    # 表示已求值的阶乘表达式 x! 的对应表达式 factorial(x)
    (r"x!", factorial(x)),
    # 表示已求值的阶乘表达式 100! 的对应表达式 factorial(100)
    (r"100!", factorial(100)),
    # 表示已求值的阶乘表达式 \theta! 的对应表达式 factorial(theta)
    (r"\theta!", factorial(theta)),
    # 表示已求值的阶乘表达式 (x + 1)! 的对应表达式 factorial(x + 1)
    (r"(x + 1)!", factorial(x + 1)),
    # 表示已求值的阶乘表达式 (x!)! 的对应表达式 factorial(factorial(x))
    (r"(x!)!", factorial(factorial(x))),
    # 表示已求值的阶乘表达式 x!!! 的对应表达式 factorial(factorial(factorial(x)))
    (r"x!!!", factorial(factorial(factorial(x)))),
    # 表示已求值的阶乘表达式 5!7! 的对应表达式 factorial(5) * factorial(7)
    (r"5!7!", factorial(5) * factorial(7))
]

UNEVALUATED_SUM_EXPRESSION_PAIRS = [
    # 表示未求值的求和表达式 \sum_{k = 1}^{3} c 的对应表达式 Sum(_Mul(1, c), (k, 1, 3))
    (r"\sum_{k = 1}^{3} c", Sum(_Mul(1, c), (k, 1, 3))),
    # 表示未求值的求和表达式 \sum_{k = 1}^3 c 的对应表达式 Sum(_Mul(1, c), (k, 1, 3))
    (r"\sum_{k = 1}^3 c", Sum(_Mul(1, c), (k, 1, 3))),
    # 表示未求值的求和表达式 \sum^{3}_{k = 1} c 的对应表达式 Sum(_Mul(1, c), (k, 1, 3))
    (r"\sum^{3}_{k = 1} c", Sum(_Mul(1, c), (k, 1, 3))),
    # 表示未求值的求和表达式 \sum^3_{k = 1} c 的对应表达式 Sum(_Mul(1, c), (k, 1, 3))
    (r"\sum^3_{k = 1} c", Sum(_Mul(1, c), (k, 1, 3))),
    # 表示未求值的求和表达式 \sum_{k = 1}^{10} k^2 的对应表达式 Sum(_Mul(1, k ** 2), (k, 1, 10))
    (r"\sum_{k = 1}^{10} k^2", Sum(_Mul(1, k ** 2), (k, 1, 10))),
    # 表示未求值的求和表达式 \sum_{n = 0}^{\infty} \frac{1}{n!} 的对应表达式 Sum(_Mul(1, _Mul(1, _Pow(_factorial(n), -1))), (n, 0, oo))
    (r"\sum_{n = 0}^{\infty} \frac{1}{n!}", Sum(_Mul(1, _Mul(1, _Pow(_factorial(n), -1))), (n, 0, oo)))
]

EVALUATED_SUM_EXPRESSION_PAIRS = [
    # 表示已求值的求和表达式 \sum_{k = 1}^{3} c 的对应表达式 Sum(c, (k, 1, 3))
    (r"\sum_{k = 1}^{3} c", Sum(c, (k, 1, 3))),
    # 表示已求值的求和表达式 \sum_{k = 1}^3 c 的对应表达式 Sum(c, (k, 1, 3))
    (r"\sum_{k = 1}^3 c", Sum(c, (k, 1, 3))),
    # 表示已求值的求和表达式 \sum^{3}_{k = 1} c 的对应表达式 Sum(c, (k, 1, 3))
    (r"\sum^{3}_{k = 1} c", Sum(c, (k, 1, 3))),
    # 表示已求值的求和表达式 \sum^3_{k = 1} c 的对应表达式 Sum(c, (k, 1, 3))
    (r"\sum^3_{k = 1} c", Sum(c, (k, 1, 3))),
    # 表示已求值的求和表达式 \sum_{k = 1}^{10} k^2 的对应表达式 Sum(k ** 2, (k, 1, 10))
    (r"\sum_{k = 1}^{10} k^2", Sum(k ** 2, (k, 1, 10))),
    # 表示已求值的求和表达式 \sum_{n = 0}^{\infty} \frac{1}{n!} 的对应表达式 Sum(1 / factorial(n), (n, 0, oo))
    (r"\sum_{n = 0}^{\infty} \frac{1}{n!}", Sum(1 / factorial(n), (n, 0, oo)))
]

UNEVALUATED_PRODUCT_EXPRESSION_PAIRS = [
    # 表示未求值的乘积表达式 \prod_{a = b}^{c} x 的对应表达式
    # 创建一个元组，包含 LaTeX 格式的数学表达式和对应的表达式对象
    (r"\overline{\overline{z}}", _Conjugate(_Conjugate(z))),
    # 创建一个元组，包含 LaTeX 格式的数学表达式和对应的复共轭对象
    (r"\overline{x + y}", _Conjugate(_Add(x, y))),
    # 创建一个元组，包含 LaTeX 格式的数学表达式和对应的表达式对象，对 x 和 y 求复共轭后相加
    (r"\overline{x} + \overline{y}", _Conjugate(x) + _Conjugate(y)),
    # 创建一个元组，包含 LaTeX 格式的数学表达式和对应的最小值函数对象，计算 a 和 b 的最小值
    (r"\min(a, b)", _Min(a, b)),
    # 创建一个元组，包含 LaTeX 格式的数学表达式和对应的最小值函数对象，计算 a、b、c-d 和 x*y 的最小值
    (r"\min(a, b, c - d, xy)", _Min(a, b, c - d, x * y)),
    # 创建一个元组，包含 LaTeX 格式的数学表达式和对应的最大值函数对象，计算 a 和 b 的最大值
    (r"\max(a, b)", _Max(a, b)),
    # 创建一个元组，包含 LaTeX 格式的数学表达式和对应的最大值函数对象，计算 a、b、c-d 和 x*y 的最大值
    (r"\max(a, b, c - d, xy)", _Max(a, b, c - d, x * y)),
    # 创建一个元组，包含 LaTeX 格式的数学表达式和对应的 Bra 对象，表示量子力学中的 Bra 向量
    (r"\langle x |", Bra('x')),
    # 创建一个元组，包含 LaTeX 格式的数学表达式和对应的 Ket 对象，表示量子力学中的 Ket 向量
    (r"| x \rangle", Ket('x')),
    # 创建一个元组，包含 LaTeX 格式的数学表达式和对应的 InnerProduct 对象，表示量子力学中的内积
    (r"\langle x | y \rangle", InnerProduct(Bra('x'), Ket('y'))),
# 定义符号表达式对的列表，每个元素包含 LaTeX 字符串和相应的 SymPy 表达式
SYMBOL_EXPRESSION_PAIRS = [
    (r"|x|", Abs(x)),
    (r"||x||", Abs(Abs(x))),
    (r"|x||y|", Abs(x) * Abs(y)),
    (r"||x||y||", Abs(Abs(x) * Abs(y))),
    (r"\lfloor x \rfloor", floor(x)),
    (r"\lceil x \rceil", ceiling(x)),
    (r"\exp x", exp(x)),
    (r"\exp(x)", exp(x)),
    (r"\lg x", log(x, 10)),
    (r"\ln x", log(x)),
    (r"\ln xy", log(x * y)),
    (r"\log x", log(x)),
    (r"\log xy", log(x * y)),
    (r"\log_{2} x", log(x, 2)),
    (r"\log_{a} x", log(x, a)),
    (r"\log_{11} x", log(x, 11)),
    (r"\log_{a^2} x", log(x, _Pow(a, 2))),
    (r"\log_2 x", log(x, 2)),
    (r"\log_a x", log(x, a)),
    (r"\overline{z}", conjugate(z)),
    (r"\overline{\overline{z}}", conjugate(conjugate(z))),
    (r"\overline{x + y}", conjugate(x + y)),
    (r"\overline{x} + \overline{y}", conjugate(x) + conjugate(y)),
    (r"\min(a, b)", Min(a, b)),
    (r"\min(a, b, c - d, xy)", Min(a, b, c - d, x * y)),
    (r"\max(a, b)", Max(a, b)),
    (r"\max(a, b, c - d, xy)", Max(a, b, c - d, x * y)),
    (r"\langle x |", Bra('x')),
    (r"| x \rangle", Ket('x')),
    (r"\langle x | y \rangle", InnerProduct(Bra('x'), Ket('y'))),
]

# 定义与间距相关的表达式对的列表，每个元素包含 LaTeX 字符串和相应的 SymPy 表达式
SPACING_RELATED_EXPRESSION_PAIRS = [
    (r"a \, b", _Mul(a, b)),
    (r"a \thinspace b", _Mul(a, b)),
    (r"a \: b", _Mul(a, b)),
    (r"a \medspace b", _Mul(a, b)),
    (r"a \; b", _Mul(a, b)),
    (r"a \thickspace b", _Mul(a, b)),
    (r"a \quad b", _Mul(a, b)),
    (r"a \qquad b", _Mul(a, b)),
    (r"a \! b", _Mul(a, b)),
    (r"a \negthinspace b", _Mul(a, b)),
    (r"a \negmedspace b", _Mul(a, b)),
    (r"a \negthickspace b", _Mul(a, b))
]

# 定义未评估的二项式表达式对的列表，每个元素包含 LaTeX 字符串和相应的 SymPy 表达式
UNEVALUATED_BINOMIAL_EXPRESSION_PAIRS = [
    (r"\binom{n}{k}", _binomial(n, k)),
    (r"\tbinom{n}{k}", _binomial(n, k)),
    (r"\dbinom{n}{k}", _binomial(n, k)),
    (r"\binom{n}{0}", _binomial(n, 0)),
    (r"x^\binom{n}{k}", _Pow(x, _binomial(n, k)))
]

# 定义评估过的二项式表达式对的列表，每个元素包含 LaTeX 字符串和相应的 SymPy 表达式
EVALUATED_BINOMIAL_EXPRESSION_PAIRS = [
    (r"\binom{n}{k}", binomial(n, k)),
    (r"\tbinom{n}{k}", binomial(n, k)),
    (r"\dbinom{n}{k}", binomial(n, k)),
    (r"\binom{n}{0}", binomial(n, 0)),
    (r"x^\binom{n}{k}", x ** binomial(n, k))
]

# 定义杂项表达式对的列表，每个元素包含 LaTeX 字符串和相应的 SymPy 表达式
MISCELLANEOUS_EXPRESSION_PAIRS = [
    (r"\left(x + y\right) z", _Mul(_Add(x, y), z)),
    (r"\left( x + y\right ) z", _Mul(_Add(x, y), z)),
    (r"\left(  x + y\right ) z", _Mul(_Add(x, y), z)),
]

def test_symbol_expressions():
    # 预期失败的测试用例索引集合
    expected_failures = {6, 7}
    # 遍历符号表达式对列表的索引和元素
    for i, (latex_str, sympy_expr) in enumerate(SYMBOL_EXPRESSION_PAIRS):
        # 如果当前索引在预期失败的集合中，则跳过
        if i in expected_failures:
            continue
        # 使用禁止评估上下文环境
        with evaluate(False):
            # 断言解析 LaTeX 字符串后的表达式等于预期的 SymPy 表达式
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str

def test_simple_expressions():
    # 预期失败的测试用例索引集合
    expected_failures = {20}
    # 遍历简单表达式对列表的索引和元素
    for i, (latex_str, sympy_expr) in enumerate(UNEVALUATED_SIMPLE_EXPRESSION_PAIRS):
        # 如果当前索引在预期失败的集合中，则跳过
        if i in expected_failures:
            continue
        # 使用禁止评估上下文环境
        with evaluate(False):
            # 断言解析 LaTeX 字符串后的表达式等于预期的 SymPy 表达式
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str
    # 遍历 EVALUATED_SIMPLE_EXPRESSION_PAIRS 列表的元素，每个元素是一个元组 (latex_str, sympy_expr)，并且用 i 记录索引值
    for i, (latex_str, sympy_expr) in enumerate(EVALUATED_SIMPLE_EXPRESSION_PAIRS):
        # 如果当前索引 i 出现在 expected_failures 集合中，则跳过本次循环，不执行后续断言操作
        if i in expected_failures:
            continue
        # 断言：解析 latex_str 后得到的结果应当与 sympy_expr 相等，用于验证解析的正确性
        assert parse_latex_lark(latex_str) == sympy_expr, latex_str
# 测试分数表达式的函数
def test_fraction_expressions():
    # 遍历未求值的分数表达式对
    for latex_str, sympy_expr in UNEVALUATED_FRACTION_EXPRESSION_PAIRS:
        # 关闭 Sympy 的求值模式
        with evaluate(False):
            # 断言解析 LaTeX 字符串后得到的 Sympy 表达式与预期的 Sympy 表达式相等
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str

    # 遍历已求值的分数表达式对
    for latex_str, sympy_expr in EVALUATED_FRACTION_EXPRESSION_PAIRS:
        # 断言解析 LaTeX 字符串后得到的 Sympy 表达式与预期的 Sympy 表达式相等
        assert parse_latex_lark(latex_str) == sympy_expr, latex_str


# 测试关系表达式的函数
def test_relation_expressions():
    # 遍历关系表达式对
    for latex_str, sympy_expr in RELATION_EXPRESSION_PAIRS:
        # 关闭 Sympy 的求值模式
        with evaluate(False):
            # 断言解析 LaTeX 字符串后得到的 Sympy 表达式与预期的 Sympy 表达式相等
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str


# 测试幂次表达式的函数
def test_power_expressions():
    expected_failures = {3}
    # 遍历未求值的幂次表达式对
    for i, (latex_str, sympy_expr) in enumerate(UNEVALUATED_POWER_EXPRESSION_PAIRS):
        if i in expected_failures:
            continue
        # 关闭 Sympy 的求值模式
        with evaluate(False):
            # 断言解析 LaTeX 字符串后得到的 Sympy 表达式与预期的 Sympy 表达式相等
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str

    # 遍历已求值的幂次表达式对
    for i, (latex_str, sympy_expr) in enumerate(EVALUATED_POWER_EXPRESSION_PAIRS):
        if i in expected_failures:
            continue
        # 断言解析 LaTeX 字符串后得到的 Sympy 表达式与预期的 Sympy 表达式相等
        assert parse_latex_lark(latex_str) == sympy_expr, latex_str


# 测试积分表达式的函数
def test_integral_expressions():
    expected_failures = {14}
    # 遍历未求值的积分表达式对
    for i, (latex_str, sympy_expr) in enumerate(UNEVALUATED_INTEGRAL_EXPRESSION_PAIRS):
        if i in expected_failures:
            continue
        # 关闭 Sympy 的求值模式
        with evaluate(False):
            # 断言解析 LaTeX 字符串后得到的 Sympy 表达式与预期的 Sympy 表达式相等
            assert parse_latex_lark(latex_str) == sympy_expr, i

    # 遍历已求值的积分表达式对
    for i, (latex_str, sympy_expr) in enumerate(EVALUATED_INTEGRAL_EXPRESSION_PAIRS):
        if i in expected_failures:
            continue
        # 断言解析 LaTeX 字符串后得到的 Sympy 表达式与预期的 Sympy 表达式相等
        assert parse_latex_lark(latex_str) == sympy_expr, latex_str


# 测试导数表达式的函数
def test_derivative_expressions():
    expected_failures = {3, 4}
    # 遍历导数表达式对
    for i, (latex_str, sympy_expr) in enumerate(DERIVATIVE_EXPRESSION_PAIRS):
        if i in expected_failures:
            continue
        # 关闭 Sympy 的求值模式
        with evaluate(False):
            # 断言解析 LaTeX 字符串后得到的 Sympy 表达式与预期的 Sympy 表达式相等
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str

    # 再次遍历导数表达式对，进行不求值的断言
    for i, (latex_str, sympy_expr) in enumerate(DERIVATIVE_EXPRESSION_PAIRS):
        if i in expected_failures:
            continue
        # 断言解析 LaTeX 字符串后得到的 Sympy 表达式与预期的 Sympy 表达式相等
        assert parse_latex_lark(latex_str) == sympy_expr, latex_str


# 测试三角函数表达式的函数
def test_trigonometric_expressions():
    expected_failures = {3}
    # 遍历三角函数表达式对
    for i, (latex_str, sympy_expr) in enumerate(TRIGONOMETRIC_EXPRESSION_PAIRS):
        if i in expected_failures:
            continue
        # 关闭 Sympy 的求值模式
        with evaluate(False):
            # 断言解析 LaTeX 字符串后得到的 Sympy 表达式与预期的 Sympy 表达式相等
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str


# 测试极限表达式的函数
def test_limit_expressions():
    # 遍历未求值的极限表达式对
    for latex_str, sympy_expr in UNEVALUATED_LIMIT_EXPRESSION_PAIRS:
        # 关闭 Sympy 的求值模式
        with evaluate(False):
            # 断言解析 LaTeX 字符串后得到的 Sympy 表达式与预期的 Sympy 表达式相等
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str


# 测试平方根表达式的函数
def test_square_root_expressions():
    # 遍历未求值的平方根表达式对
    for latex_str, sympy_expr in UNEVALUATED_SQRT_EXPRESSION_PAIRS:
        # 关闭 Sympy 的求值模式
        with evaluate(False):
            # 断言解析 LaTeX 字符串后得到的 Sympy 表达式与预期的 Sympy 表达式相等
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str

    # 遍历已求值的平方根表达式对
    for latex_str, sympy_expr in EVALUATED_SQRT_EXPRESSION_PAIRS:
        # 断言解析 LaTeX 字符串后得到的 Sympy 表达式与预期的 Sympy 表达式相等
        assert parse_latex_lark(latex_str) == sympy_expr, latex_str


# 测试阶乘表达式的函数
def test_factorial_expressions():
    # 该函数暂未给出完整的实现
    pass
    # 遍历未评估的阶乘表达式对列表
    for latex_str, sympy_expr in UNEVALUATED_FACTORIAL_EXPRESSION_PAIRS:
        # 在禁用评估的上下文中，解析 LaTeX 字符串并断言其与 Sympy 表达式相等
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str
    
    # 遍历已评估的阶乘表达式对列表
    for latex_str, sympy_expr in EVALUATED_FACTORIAL_EXPRESSION_PAIRS:
        # 解析 LaTeX 字符串并断言其与 Sympy 表达式相等
        assert parse_latex_lark(latex_str) == sympy_expr, latex_str
# 测试和验证未求值的求和表达式
def test_sum_expressions():
    for latex_str, sympy_expr in UNEVALUATED_SUM_EXPRESSION_PAIRS:
        # 禁用求值上下文，验证解析的 LaTeX 字符串与 SymPy 表达式是否相等
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str

    # 验证已求值的求和表达式
    for latex_str, sympy_expr in EVALUATED_SUM_EXPRESSION_PAIRS:
        assert parse_latex_lark(latex_str) == sympy_expr, latex_str


# 测试和验证未求值的乘积表达式
def test_product_expressions():
    for latex_str, sympy_expr in UNEVALUATED_PRODUCT_EXPRESSION_PAIRS:
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str


# 标记为期望失败的应用函数表达式测试
@XFAIL
def test_applied_function_expressions():
    expected_failures = {0, 3, 4}  # 0 是模糊的，其它的需要尚未添加的功能
    # 不确定为什么 1 和 2 会失败
    for i, (latex_str, sympy_expr) in enumerate(APPLIED_FUNCTION_EXPRESSION_PAIRS):
        if i in expected_failures:
            continue
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str


# 测试和验证未求值的常见函数表达式
def test_common_function_expressions():
    for latex_str, sympy_expr in UNEVALUATED_COMMON_FUNCTION_EXPRESSION_PAIRS:
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str

    # 验证已求值的常见函数表达式
    for latex_str, sympy_expr in EVALUATED_COMMON_FUNCTION_EXPRESSION_PAIRS:
        assert parse_latex_lark(latex_str) == sympy_expr, latex_str


# 标记为未处理的 bug 导致失败的空格相关表达式测试
@XFAIL
def test_spacing():
    for latex_str, sympy_expr in SPACING_RELATED_EXPRESSION_PAIRS:
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str


# 测试和验证未求值的二项式表达式
def test_binomial_expressions():
    for latex_str, sympy_expr in UNEVALUATED_BINOMIAL_EXPRESSION_PAIRS:
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str

    # 验证已求值的二项式表达式
    for latex_str, sympy_expr in EVALUATED_BINOMIAL_EXPRESSION_PAIRS:
        assert parse_latex_lark(latex_str) == sympy_expr, latex_str


# 测试和验证各种杂项表达式
def test_miscellaneous_expressions():
    for latex_str, sympy_expr in MISCELLANEOUS_EXPRESSION_PAIRS:
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str
```