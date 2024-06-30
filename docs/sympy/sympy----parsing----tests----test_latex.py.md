# `D:\src\scipysrc\sympy\sympy\parsing\tests\test_latex.py`

```
from sympy.testing.pytest import raises, XFAIL
from sympy.external import import_module

from sympy.concrete.products import Product  # 导入 SymPy 中的 Product 类
from sympy.concrete.summations import Sum  # 导入 SymPy 中的 Sum 类
from sympy.core.add import Add  # 导入 SymPy 中的 Add 类
from sympy.core.function import (Derivative, Function)  # 导入 SymPy 中的 Derivative 和 Function 类
from sympy.core.mul import Mul  # 导入 SymPy 中的 Mul 类
from sympy.core.numbers import (E, oo)  # 导入 SymPy 中的 E（自然对数的底）和 oo（无穷大）
from sympy.core.power import Pow  # 导入 SymPy 中的 Pow 类
from sympy.core.relational import (GreaterThan, LessThan, StrictGreaterThan, StrictLessThan, Unequality)  # 导入 SymPy 中的比较运算符类
from sympy.core.symbol import Symbol  # 导入 SymPy 中的 Symbol 类
from sympy.functions.combinatorial.factorials import (binomial, factorial)  # 导入 SymPy 中的组合数学函数 binomial 和 factorial
from sympy.functions.elementary.complexes import (Abs, conjugate)  # 导入 SymPy 中的复数函数 Abs 和 conjugate
from sympy.functions.elementary.exponential import (exp, log)  # 导入 SymPy 中的指数和对数函数 exp 和 log
from sympy.functions.elementary.integers import (ceiling, floor)  # 导入 SymPy 中的整数函数 ceiling 和 floor
from sympy.functions.elementary.miscellaneous import (root, sqrt)  # 导入 SymPy 中的其他函数 root 和 sqrt
from sympy.functions.elementary.trigonometric import (asin, cos, csc, sec, sin, tan)  # 导入 SymPy 中的三角函数 asin, cos, csc, sec, sin, tan
from sympy.integrals.integrals import Integral  # 导入 SymPy 中的积分 Integral 类
from sympy.series.limits import Limit  # 导入 SymPy 中的极限 Limit 类

from sympy.core.relational import Eq, Ne, Lt, Le, Gt, Ge  # 导入 SymPy 中的等式和不等式类 Eq, Ne, Lt, Le, Gt, Ge
from sympy.physics.quantum.state import Bra, Ket  # 导入 SymPy 中的量子力学 Bra 和 Ket 类
from sympy.abc import x, y, z, a, b, c, t, k, n  # 导入 SymPy 中的常用符号

antlr4 = import_module("antlr4")  # 尝试导入 antlr4 模块，用于 LaTeX 解析

# 如果未成功导入 antlr4 模块，则将 disabled 设置为 True
disabled = antlr4 is None

theta = Symbol('theta')  # 创建 SymPy 符号对象 theta
f = Function('f')  # 创建 SymPy 函数对象 f

# 定义几个简便函数，用于创建 SymPy 对象，不进行自动求值
def _Add(a, b):
    return Add(a, b, evaluate=False)

def _Mul(a, b):
    return Mul(a, b, evaluate=False)

def _Pow(a, b):
    return Pow(a, b, evaluate=False)

def _Sqrt(a):
    return sqrt(a, evaluate=False)

def _Conjugate(a):
    return conjugate(a, evaluate=False)

def _Abs(a):
    return Abs(a, evaluate=False)

def _factorial(a):
    return factorial(a, evaluate=False)

def _exp(a):
    return exp(a, evaluate=False)

def _log(a, b):
    return log(a, b, evaluate=False)

def _binomial(n, k):
    return binomial(n, k, evaluate=False)

# 定义一个测试函数 test_import，用于测试 LaTeX 解析模块的导入情况
def test_import():
    from sympy.parsing.latex._build_latex_antlr import (
        build_parser,  # 导入 LaTeX 解析器的构建函数 build_parser
        check_antlr_version,  # 导入检查 antlr 版本的函数 check_antlr_version
        dir_latex_antlr  # 导入 LaTeX antlr 目录函数 dir_latex_antlr
    )
    # XXX: It would be better to come up with a test for these...
    del build_parser, check_antlr_version, dir_latex_antlr  # 删除这些导入，不进行测试

# 这些 LaTeX 字符串应该解析为对应的 SymPy 表达式
GOOD_PAIRS = [
    (r"0", 0),
    (r"1", 1),
    (r"-3.14", -3.14),
    (r"(-7.13)(1.5)", _Mul(-7.13, 1.5)),
    (r"x", x),
    (r"2x", 2*x),
    (r"x^2", x**2),
    (r"x^\frac{1}{2}", _Pow(x, _Pow(2, -1))),
    (r"x^{3 + 1}", x**_Add(3, 1)),
    (r"-c", -c),
    (r"a \cdot b", a * b),
    (r"a / b", a / b),
    (r"a \div b", a / b),
    (r"a + b", a + b),
    (r"a + b - a", _Add(a+b, -a)),
    (r"a^2 + b^2 = c^2", Eq(a**2 + b**2, c**2)),
    (r"(x + y) z", _Mul(_Add(x, y), z)),
    (r"a'b+ab'", _Add(_Mul(Symbol("a'"), b), _Mul(a, Symbol("b'")))),
    (r"y''_1", Symbol("y_{1}''")),
    (r"y_1''", Symbol("y_{1}''")),
    (r"\left(x + y\right) z", _Mul(_Add(x, y), z)),
    (r"\left( x + y\right ) z", _Mul(_Add(x, y), z)),
]
    (r"\left(  x + y\right ) z", _Mul(_Add(x, y), z)),
    # 解析 LaTeX 格式的表达式 \left( x + y \right ) z，转换为 Python 表达式 _Mul(_Add(x, y), z)

    (r"\left[x + y\right] z", _Mul(_Add(x, y), z)),
    # 解析 LaTeX 格式的表达式 \left[x + y\right] z，转换为 Python 表达式 _Mul(_Add(x, y), z)

    (r"\left\{x + y\right\} z", _Mul(_Add(x, y), z)),
    # 解析 LaTeX 格式的表达式 \left\{x + y\right\} z，转换为 Python 表达式 _Mul(_Add(x, y), z)

    (r"1+1", _Add(1, 1)),
    # 解析数学表达式 1+1，转换为 Python 表达式 _Add(1, 1)

    (r"0+1", _Add(0, 1)),
    # 解析数学表达式 0+1，转换为 Python 表达式 _Add(0, 1)

    (r"1*2", _Mul(1, 2)),
    # 解析数学表达式 1*2，转换为 Python 表达式 _Mul(1, 2)

    (r"0*1", _Mul(0, 1)),
    # 解析数学表达式 0*1，转换为 Python 表达式 _Mul(0, 1)

    (r"1 \times 2 ", _Mul(1, 2)),
    # 解析数学表达式 1 \times 2，转换为 Python 表达式 _Mul(1, 2)

    (r"x = y", Eq(x, y)),
    # 解析数学表达式 x = y，转换为 Python 表达式 Eq(x, y)

    (r"x \neq y", Ne(x, y)),
    # 解析数学表达式 x \neq y，转换为 Python 表达式 Ne(x, y)

    (r"x < y", Lt(x, y)),
    # 解析数学表达式 x < y，转换为 Python 表达式 Lt(x, y)

    (r"x > y", Gt(x, y)),
    # 解析数学表达式 x > y，转换为 Python 表达式 Gt(x, y)

    (r"x \leq y", Le(x, y)),
    # 解析数学表达式 x \leq y，转换为 Python 表达式 Le(x, y)

    (r"x \geq y", Ge(x, y)),
    # 解析数学表达式 x \geq y，转换为 Python 表达式 Ge(x, y)

    (r"x \le y", Le(x, y)),
    # 解析数学表达式 x \le y，转换为 Python 表达式 Le(x, y)

    (r"x \ge y", Ge(x, y)),
    # 解析数学表达式 x \ge y，转换为 Python 表达式 Ge(x, y)

    (r"\lfloor x \rfloor", floor(x)),
    # 解析数学表达式 \lfloor x \rfloor，转换为 Python 函数调用 floor(x)

    (r"\lceil x \rceil", ceiling(x)),
    # 解析数学表达式 \lceil x \rceil，转换为 Python 函数调用 ceiling(x)

    (r"\langle x |", Bra('x')),
    # 解析数学表达式 \langle x |，转换为 Python 函数调用 Bra('x')

    (r"| x \rangle", Ket('x')),
    # 解析数学表达式 | x \rangle，转换为 Python 函数调用 Ket('x')

    (r"\sin \theta", sin(theta)),
    # 解析数学表达式 \sin \theta，转换为 Python 函数调用 sin(theta)

    (r"\sin(\theta)", sin(theta)),
    # 解析数学表达式 \sin(\theta)，转换为 Python 函数调用 sin(theta)

    (r"\sin^{-1} a", asin(a)),
    # 解析数学表达式 \sin^{-1} a，转换为 Python 函数调用 asin(a)

    (r"\sin a \cos b", _Mul(sin(a), cos(b))),
    # 解析数学表达式 \sin a \cos b，转换为 Python 表达式 _Mul(sin(a), cos(b))

    (r"\sin \cos \theta", sin(cos(theta))),
    # 解析数学表达式 \sin \cos \theta，转换为 Python 函数调用 sin(cos(theta)))

    (r"\sin(\cos \theta)", sin(cos(theta))),
    # 解析数学表达式 \sin(\cos \theta)，转换为 Python 函数调用 sin(cos(theta)))

    (r"\frac{a}{b}", a / b),
    # 解析数学表达式 \frac{a}{b}，转换为 Python 表达式 a / b

    (r"\dfrac{a}{b}", a / b),
    # 解析数学表达式 \dfrac{a}{b}，转换为 Python 表达式 a / b

    (r"\tfrac{a}{b}", a / b),
    # 解析数学表达式 \tfrac{a}{b}，转换为 Python 表达式 a / b

    (r"\frac12", _Pow(2, -1)),
    # 解析数学表达式 \frac12，转换为 Python 表达式 _Pow(2, -1)

    (r"\frac12y", _Mul(_Pow(2, -1), y)),
    # 解析数学表达式 \frac12y，转换为 Python 表达式 _Mul(_Pow(2, -1), y)

    (r"\frac1234", _Mul(_Pow(2, -1), 34)),
    # 解析数学表达式 \frac1234，转换为 Python 表达式 _Mul(_Pow(2, -1), 34)

    (r"\frac2{3}", _Mul(2, _Pow(3, -1))),
    # 解析数学表达式 \frac2{3}，转换为 Python 表达式 _Mul(2, _Pow(3, -1))

    (r"\frac{\sin{x}}2", _Mul(sin(x), _Pow(2, -1))),
    # 解析数学表达式 \frac{\sin{x}}2，转换为 Python 表达式 _Mul(sin(x), _Pow(2, -1))

    (r"\frac{a + b}{c}", _Mul(a + b, _Pow(c, -1))),
    # 解析数学表达式 \frac{a + b}{c}，转换为 Python 表达式 _Mul(a + b, _Pow(c, -1))

    (r"\frac{7}{3}", _Mul(7, _Pow(3, -1))),
    # 解析数学表达式 \frac{7}{3}，转换为 Python 表达式 _Mul(7, _Pow(3, -1))

    (r"(\csc x)(\sec y)", csc(x)*sec(y)),
    # 解析数学表达式 (\csc x)(\sec y)，转换为 Python 表达式 csc(x)*sec(y)

    (r"\lim_{x \to 3} a", Limit(a, x, 3, dir='+-')),
    # 解析数学表达式 \lim_{x \to 3} a，转换为 Python 函数调用 Limit(a, x, 3, dir='+-')

    (r"\lim_{x \rightarrow 3} a", Limit(a, x, 3, dir='+-')),
    # 解析数学表达式 \lim_{x \rightarrow 3} a，转换为 Python 函数调用 Limit(a, x, 3, dir='+-')

    (r"\lim_{x \Rightarrow 3} a", Limit(a, x, 3, dir='+-')),
    # 解析数学表达式 \lim_{x \Rightarrow 3} a，转换为 Python 函数调用 Limit(a, x, 3, dir='+-')

    (r"\lim_{x \longrightarrow 3} a", Limit(a, x, 3, dir='+-')),
    # 解析数学表达式 \lim_{x \longrightarrow 3} a，转换为 Python 函数调用 Limit(a, x, 3, dir='+-')

    (r"\lim_{x \Longrightarrow 3} a", Limit(a, x, 3, dir='+-')),
    # 解析数学
    # 创建 LaTeX 表示的积分符号 x dx 对应的积分对象
    (r"\int^{b}_a x dx", Integral(x, (x, a, b))),
    # 创建 LaTeX 表示的积分符号 x dx 对应的积分对象
    (r"\int_{a}^{b} x dx", Integral(x, (x, a, b))),
    # 创建 LaTeX 表示的积分符号 x dx 对应的积分对象
    (r"\int^{b}_{a} x dx", Integral(x, (x, a, b))),
    # 创建 LaTeX 表示的积分符号 f(z) dz 对应的积分对象
    (r"\int_{f(a)}^{f(b)} f(z) dz", Integral(f(z), (z, f(a), f(b)))),
    # 创建 LaTeX 表示的积分符号 (x+a) 对应的积分对象
    (r"\int (x+a)", Integral(_Add(x, a), x)),
    # 创建 LaTeX 表示的积分符号 a + b + c dx 对应的积分对象
    (r"\int a + b + c dx", Integral(_Add(_Add(a, b), c), x)),
    # 创建 LaTeX 表示的积分符号 \frac{dz}{z} 对应的积分对象
    (r"\int \frac{dz}{z}", Integral(Pow(z, -1), z)),
    # 创建 LaTeX 表示的积分符号 \frac{3 dz}{z} 对应的积分对象
    (r"\int \frac{3 dz}{z}", Integral(3*Pow(z, -1), z)),
    # 创建 LaTeX 表示的积分符号 \frac{1}{x} dx 对应的积分对象
    (r"\int \frac{1}{x} dx", Integral(Pow(x, -1), x)),
    # 创建 LaTeX 表示的积分符号 \frac{1}{a} + \frac{1}{b} dx 对应的积分对象
    (r"\int \frac{1}{a} + \frac{1}{b} dx",
     Integral(_Add(_Pow(a, -1), Pow(b, -1)), x)),
    # 创建 LaTeX 表示的积分符号 \frac{3 \cdot d\theta}{\theta} 对应的积分对象
    (r"\int \frac{3 \cdot d\theta}{\theta}",
     Integral(3*_Pow(theta, -1), theta)),
    # 创建 LaTeX 表示的积分符号 \frac{1}{x} + 1 dx 对应的积分对象
    (r"\int \frac{1}{x} + 1 dx", Integral(_Add(_Pow(x, -1), 1), x)),
    # 创建 LaTeX 表示的符号 x_0 对应的符号对象
    (r"x_0", Symbol('x_{0}')),
    # 创建 LaTeX 表示的符号 x_{1} 对应的符号对象
    (r"x_{1}", Symbol('x_{1}')),
    # 创建 LaTeX 表示的符号 x_a 对应的符号对象
    (r"x_a", Symbol('x_{a}')),
    # 创建 LaTeX 表示的符号 x_{b} 对应的符号对象
    (r"x_{b}", Symbol('x_{b}')),
    # 创建 LaTeX 表示的符号 h_\theta 对应的符号对象
    (r"h_\theta", Symbol('h_{theta}')),
    # 创建 LaTeX 表示的符号 h_{\theta} 对应的符号对象
    (r"h_{\theta}", Symbol('h_{theta}')),
    # 创建 LaTeX 表示的函数 h_{\theta}(x_0, x_1) 对应的函数对象
    (r"h_{\theta}(x_0, x_1)",
     Function('h_{theta}')(Symbol('x_{0}'), Symbol('x_{1}'))),
    # 创建 LaTeX 表示的符号 x! 对应的阶乘对象
    (r"x!", _factorial(x)),
    # 创建 LaTeX 表示的符号 100! 对应的阶乘对象
    (r"100!", _factorial(100)),
    # 创建 LaTeX 表示的符号 \theta! 对应的阶乘对象
    (r"\theta!", _factorial(theta)),
    # 创建 LaTeX 表示的符号 (x + 1)! 对应的阶乘对象
    (r"(x + 1)!", _factorial(_Add(x, 1))),
    # 创建 LaTeX 表示的符号 (x!)! 对应的阶乘对象
    (r"(x!)!", _factorial(_factorial(x))),
    # 创建 LaTeX 表示的符号 x!!! 对应的多重阶乘对象
    (r"x!!!", _factorial(_factorial(_factorial(x)))),
    # 创建 LaTeX 表示的符号 5!7! 对应的乘法对象
    (r"5!7!", _Mul(_factorial(5), _factorial(7))),
    # 创建 LaTeX 表示的符号 \sqrt{x} 对应的平方根对象
    (r"\sqrt{x}", sqrt(x)),
    # 创建 LaTeX 表示的符号 \sqrt{x + b} 对应的平方根对象
    (r"\sqrt{x + b}", sqrt(_Add(x, b))),
    # 创建 LaTeX 表示的符号 \sqrt[3]{\sin x} 对应的三次方根对象
    (r"\sqrt[3]{\sin x}", root(sin(x), 3)),
    # 创建 LaTeX 表示的符号 \sqrt[y]{\sin x} 对应的 y 次方根对象
    (r"\sqrt[y]{\sin x}", root(sin(x), y)),
    # 创建 LaTeX 表示的符号 \sqrt[\theta]{\sin x} 对应的 theta 次方根对象
    (r"\sqrt[\theta]{\sin x}", root(sin(x), theta)),
    # 创建 LaTeX 表示的符号 \sqrt{\frac{12}{6}} 对应的平方根对象
    (r"\sqrt{\frac{12}{6}}", _Sqrt(_Mul(12, _Pow(6, -1)))),
    # 创建 LaTeX 表示的符号 \overline{z} 对应的复共轭对象
    (r"\overline{z}", _Conjugate(z)),
    # 创建 LaTeX 表示的符号 \overline{\overline{z}} 对应的两次复共轭对象
    (r"\overline{\overline{z}}", _Conjugate(_Conjugate(z))),
    # 创建 LaTeX 表示的符号 \overline{x + y} 对应的复共轭加法对象
    (r"\overline{x + y}", _Conjugate(_Add(x, y))),
    # 创建 LaTeX 表示的符号 \overline{x} + \overline{y} 对应的复共轭和对象
    (r"\overline{x} + \overline{y}", _Conjugate(x) + _Conjugate(y)),
    # 创建 LaTeX 表示的符号 x < y 对应的严格小于对象
    (r"x < y", StrictLessThan(x, y)),
    # 创建 LaTeX 表示的符号 x \leq y 对应的小于等于对象
    (r"x \leq y", LessThan(x, y)),
    # 创建 LaTeX 表示的符号 x > y 对应的严格大于对象
    (r"x > y", StrictGreaterThan(x, y)),
    # 创建 LaTeX 表示的符号 x \geq y 对应的大于等于对象
    (r"x \geq y", GreaterThan(x, y)),
    # 创建 LaTeX 表示的符号 \mathit{x} 对应的符号对象
    (r"\mathit{x}", Symbol('x')),
    # 创建 LaTeX 表示的符号 \mathit{test} 对应的符号对象
    (r"\mathit{test}", Symbol('test')),
    # 创建 LaTeX 表示的符号 \mathit{TEST} 对应的符号对象
    (r"\mathit{TEST}", Symbol('TEST')),
    # 创建 LaTeX 表示的符号 \mathit{HELLO world} 对应的符号对象
    (r"\mathit{HELLO world}", Symbol('HELLO world')),
    # 创建 LaTeX 表示的符号 \sum_{k = 1}^{3} c 对应的求和对象
    (r"\sum_{k = 1}^{3} c", Sum(c, (k, 1, 3))),
    # 创建 LaTeX 表示的符号 \sum_{k =
    (r"\log_{a^2} x", _log(x, _Pow(a, 2))),
    # 创建对数表达式 \log_{a^2} x，其中 _log 是对数函数，x 是底数，_Pow(a, 2) 是指数

    (r"[x]", x),
    # 返回变量 x 的带方括号的字符串表达式

    (r"[a + b]", _Add(a, b)),
    # 返回表达式 a + b 的带方括号的字符串形式，_Add 是加法函数

    (r"\frac{d}{dx} [ \tan x ]", Derivative(tan(x), x)),
    # 返回导数表达式 \frac{d}{dx} [ \tan x ]，其中 Derivative 是求导函数，tan(x) 是被求导函数

    (r"\binom{n}{k}", _binomial(n, k)),
    # 返回二项式系数 \binom{n}{k}，_binomial 是计算二项式系数的函数

    (r"\tbinom{n}{k}", _binomial(n, k)),
    # 返回二项式系数 \tbinom{n}{k}，_binomial 是计算二项式系数的函数

    (r"\dbinom{n}{k}", _binomial(n, k)),
    # 返回二项式系数 \dbinom{n}{k}，_binomial 是计算二项式系数的函数

    (r"\binom{n}{0}", _binomial(n, 0)),
    # 返回二项式系数 \binom{n}{0}，_binomial 是计算二项式系数的函数

    (r"x^\binom{n}{k}", _Pow(x, _binomial(n, k))),
    # 返回幂指数表达式 x^\binom{n}{k}，其中 _Pow 是幂函数，_binomial 是计算二项式系数的函数

    (r"a \, b", _Mul(a, b)),
    # 返回乘法表达式 a \, b，其中 _Mul 是乘法函数

    (r"a \thinspace b", _Mul(a, b)),
    # 返回乘法表达式 a \thinspace b，其中 _Mul 是乘法函数

    (r"a \: b", _Mul(a, b)),
    # 返回乘法表达式 a \: b，其中 _Mul 是乘法函数

    (r"a \medspace b", _Mul(a, b)),
    # 返回乘法表达式 a \medspace b，其中 _Mul 是乘法函数

    (r"a \; b", _Mul(a, b)),
    # 返回乘法表达式 a \; b，其中 _Mul 是乘法函数

    (r"a \thickspace b", _Mul(a, b)),
    # 返回乘法表达式 a \thickspace b，其中 _Mul 是乘法函数

    (r"a \quad b", _Mul(a, b)),
    # 返回乘法表达式 a \quad b，其中 _Mul 是乘法函数

    (r"a \qquad b", _Mul(a, b)),
    # 返回乘法表达式 a \qquad b，其中 _Mul 是乘法函数

    (r"a \! b", _Mul(a, b)),
    # 返回乘法表达式 a \! b，其中 _Mul 是乘法函数

    (r"a \negthinspace b", _Mul(a, b)),
    # 返回乘法表达式 a \negthinspace b，其中 _Mul 是乘法函数

    (r"a \negmedspace b", _Mul(a, b)),
    # 返回乘法表达式 a \negmedspace b，其中 _Mul 是乘法函数

    (r"a \negthickspace b", _Mul(a, b)),
    # 返回乘法表达式 a \negthickspace b，其中 _Mul 是乘法函数

    (r"\int x \, dx", Integral(x, x)),
    # 返回积分表达式 \int x \, dx，其中 Integral 是积分函数，x 是被积函数

    (r"\log_2 x", _log(x, 2)),
    # 返回以 2 为底的对数表达式 \log_2 x，其中 _log 是对数函数

    (r"\log_a x", _log(x, a)),
    # 返回以 a 为底的对数表达式 \log_a x，其中 _log 是对数函数

    (r"5^0 - 4^0", _Add(_Pow(5, 0), _Mul(-1, _Pow(4, 0)))),
    # 返回表达式 5^0 - 4^0，其中 _Add 是加法函数，_Pow 是幂函数，_Mul 是乘法函数

    (r"3x - 1", _Add(_Mul(3, x), -1))
    # 返回表达式 3x - 1，其中 _Add 是加法函数，_Mul 是乘法函数
# 导入 parse_latex 函数从 sympy.parsing.latex 模块中
def test_parseable():
    from sympy.parsing.latex import parse_latex
    # 遍历 GOOD_PAIRS 中的每一个元组 (latex_str, sympy_expr)
    for latex_str, sympy_expr in GOOD_PAIRS:
        # 断言 parse_latex 函数对当前 latex_str 的输出与对应的 sympy_expr 相等，如果不等则抛出异常
        assert parse_latex(latex_str) == sympy_expr, latex_str

# 这些错误的 LaTeX 字符串在解析时应该引发 LaTeXParsingError 异常
BAD_STRINGS = [
    r"(",
    r")",
    r"\frac{d}{dx}",
    r"(\frac{d}{dx})",
    r"\sqrt{}",
    r"\sqrt",
    r"\overline{}",
    r"\overline",
    r"{",
    r"}",
    r"\mathit{x + y}",
    r"\mathit{21}",
    r"\frac{2}{}",
    r"\frac{}{2}",
    r"\int",
    r"!",
    r"!0",
    r"_",
    r"^",
    r"|",
    r"||x|",
    r"()",
    r"((((((((((((((((()))))))))))))))))",
    r"-",
    r"\frac{d}{dx} + \frac{d}{dt}",
    r"f(x,,y)",
    r"f(x,y,",
    r"\sin^x",
    r"\cos^2",
    r"@",
    r"#",
    r"$",
    r"%",
    r"&",
    r"*",
    r"" "\\",
    r"~",
    r"\frac{(2 + x}{1 - x)}",
]

# 测试无法解析的 LaTeX 字符串是否引发 LaTeXParsingError 异常
def test_not_parseable():
    from sympy.parsing.latex import parse_latex, LaTeXParsingError
    for latex_str in BAD_STRINGS:
        # 使用 pytest 的 raises 方法检查 parse_latex 是否能正确抛出 LaTeXParsingError 异常
        with raises(LaTeXParsingError):
            parse_latex(latex_str)

# 在从 latex2sympy 迁移时，这些失败的字符串应该失败但实际上并没有
FAILING_BAD_STRINGS = [
    r"\cos 1 \cos",
    r"f(,",
    r"f()",
    r"a \div \div b",
    r"a \cdot \cdot b",
    r"a // b",
    r"a +",
    r"1.1.1",
    r"1 +",
    r"a / b /",
]

# 使用 XFAIL 装饰器标记的测试函数，测试无法解析的 LaTeX 字符串是否引发 LaTeXParsingError 异常
@XFAIL
def test_failing_not_parseable():
    from sympy.parsing.latex import parse_latex, LaTeXParsingError
    for latex_str in FAILING_BAD_STRINGS:
        # 使用 pytest 的 raises 方法检查 parse_latex 是否能正确抛出 LaTeXParsingError 异常
        with raises(LaTeXParsingError):
            parse_latex(latex_str)

# 在严格模式下，这些失败的字符串应该失败
def test_strict_mode():
    from sympy.parsing.latex import parse_latex, LaTeXParsingError
    for latex_str in FAILING_BAD_STRINGS:
        # 使用 pytest 的 raises 方法检查 parse_latex 是否能在严格模式下正确抛出 LaTeXParsingError 异常
        with raises(LaTeXParsingError):
            parse_latex(latex_str, strict=True)
```