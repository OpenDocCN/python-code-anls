# `D:\src\scipysrc\sympy\sympy\functions\elementary\tests\test_exponential.py`

```
# 导入 SymPy 中的 refine 函数，用于表达式的精化
from sympy.assumptions.refine import refine
# 导入 SymPy 中的 AccumBounds 类，处理累积边界
from sympy.calculus.accumulationbounds import AccumBounds
# 导入 SymPy 中的 Product 类，用于处理连乘积
from sympy.concrete.products import Product
# 导入 SymPy 中的 Sum 类，用于处理求和表达式
from sympy.concrete.summations import Sum
# 导入 SymPy 中的 expand_log 函数，用于展开对数表达式
from sympy.core.function import expand_log
# 导入 SymPy 中的特定常数和数学常数，如 E、pi、oo（无穷大）、nan（非数值）、zoo（复数无穷大）
from sympy.core.numbers import (E, Float, I, Rational, nan, oo, pi, zoo)
# 导入 SymPy 中的 Pow 类，用于表示幂运算
from sympy.core.power import Pow
# 导入 SymPy 中的 S 单例，用于表示 SymPy 中的常量
from sympy.core.singleton import S
# 导入 SymPy 中的 Symbol 和 symbols 函数，用于创建符号变量
from sympy.core.symbol import (Symbol, symbols)
# 导入 SymPy 中的复数运算函数，如 adjoint（共轭转置）、conjugate（共轭）、re（实部）、sign（符号函数）、transpose（转置）
from sympy.functions.elementary.complexes import (adjoint, conjugate, re, sign, transpose)
# 导入 SymPy 中的指数函数，如 LambertW（Lambert W 函数）、exp（指数函数）、exp_polar（极坐标指数函数）、log（对数函数）
from sympy.functions.elementary.exponential import (LambertW, exp, exp_polar, log)
# 导入 SymPy 中的双曲三角函数，如 cosh（双曲余弦）、sinh（双曲正弦）、tanh（双曲正切）
from sympy.functions.elementary.hyperbolic import (cosh, sinh, tanh)
# 导入 SymPy 中的平方根函数 sqrt
from sympy.functions.elementary.miscellaneous import sqrt
# 导入 SymPy 中的三角函数，如 cos（余弦）、sin（正弦）、tan（正切）
from sympy.functions.elementary.trigonometric import (cos, sin, tan)
# 导入 SymPy 中的 MatrixSymbol 类，用于表示矩阵符号表达式
from sympy.matrices.expressions.matexpr import MatrixSymbol
# 导入 SymPy 中的多项式工具函数 gcd
from sympy.polys.polytools import gcd
# 导入 SymPy 中的 O 类，用于表示 O 符号（阶符号）
from sympy.series.order import O
# 导入 SymPy 中的 simplify 函数，用于简化表达式
from sympy.simplify.simplify import simplify
# 导入 SymPy 中的全局参数 global_parameters
from sympy.core.parameters import global_parameters
# 导入 SymPy 中的 match_real_imag 函数，用于匹配实部和虚部
from sympy.functions.elementary.exponential import match_real_imag
# 导入 SymPy 中的预定义符号 x, y, z
from sympy.abc import x, y, z
# 导入 SymPy 中的 unchanged 类，表示不变的表达式
from sympy.core.expr import unchanged
# 导入 SymPy 中的 ArgumentIndexError 类，表示参数索引错误
from sympy.core.function import ArgumentIndexError
# 导入 SymPy 中的测试工具函数 raises, XFAIL, _both_exp_pow
from sympy.testing.pytest import raises, XFAIL, _both_exp_pow

# 使用装饰器 @_both_exp_pow 定义测试函数 test_exp_values
@_both_exp_pow
def test_exp_values():
    # 检查全局参数 exp_is_pow，确定 exp(x) 的类型是 Pow 类还是 exp 类
    if global_parameters.exp_is_pow:
        assert type(exp(x)) is Pow
    else:
        assert type(exp(x)) is exp

    # 创建整数符号 k
    k = Symbol('k', integer=True)

    # 检查 exp(nan) 结果是否为 nan
    assert exp(nan) is nan

    # 检查 exp(oo) 结果是否为 oo
    assert exp(oo) is oo
    # 检查 exp(-oo) 结果是否为 0
    assert exp(-oo) == 0

    # 检查 exp(0) 结果是否为 1
    assert exp(0) == 1
    # 检查 exp(1) 结果是否为自然常数 e
    assert exp(1) == E
    # 检查 exp(-1 + x) 的基本指数表示是否为 (S.Exp1, x - 1)
    assert exp(-1 + x).as_base_exp() == (S.Exp1, x - 1)
    # 检查 exp(1 + x) 的基本指数表示是否为 (S.Exp1, x + 1)
    assert exp(1 + x).as_base_exp() == (S.Exp1, x + 1)

    # 检查 exp(pi*I/2) 结果是否为虚数单位 i
    assert exp(pi*I/2) == I
    # 检查 exp(pi*I) 结果是否为 -1
    assert exp(pi*I) == -1
    # 检查 exp(pi*I*3/2) 结果是否为 -i
    assert exp(pi*I*Rational(3, 2)) == -I
    # 检查 exp(2*pi*I) 结果是否为 1
    assert exp(2*pi*I) == 1

    # 利用 refine 函数精化 exp(pi*I*2*k)，验证结果是否为 1
    assert refine(exp(pi*I*2*k)) == 1
    # 利用 refine 函数精化 exp(pi*I*2*(k + S.Half))，验证结果是否为 -1
    assert refine(exp(pi*I*2*(k + S.Half))) == -1
    # 利用 refine 函数精化 exp(pi*I*2*(k + Rational(1, 4)))，验证结果是否为 i
    assert refine(exp(pi*I*2*(k + Rational(1, 4)))) == I
    # 利用 refine 函数精化 exp(pi*I*2*(k + Rational(3, 4)))，验证结果是否为 -i
    assert refine(exp(pi*I*2*(k + Rational(3, 4)))) == -I

    # 检查 exp(log(x)) 结果是否为 x
    assert exp(log(x)) == x
    # 检查 exp(2*log(x)) 结果是否为 x^2
    assert exp(2*log(x)) == x**2
    # 检查 exp(pi*log(x)) 结果是否为 x^pi
    assert exp(pi*log(x)) == x**pi

    # 检查 exp(17*log(x) + E*log(y)) 结果是否为 x^17 * y^E
    assert exp(17*log(x) + E*log(y)) == x**17 * y**E

    # 检查 exp(x*log(x)) 结果是否不等于 x^x
    assert exp(x*log(x)) != x**x
    # 检查 exp(sin(x)*log(x)) 结果是否不等于 x
    assert exp(sin(x)*log(x)) != x

    # 检查 exp(3*log(x) + oo*x) 结果是否等于 exp(oo*x) * x^3
    assert exp(3*log(x) + oo*x) == exp(oo*x) * x**3
    # 检查 exp(4*log(x)*log(y) + 3*log(x)) 结果是否等于 x^3 * exp(4*log(x)*log(y))
    assert exp(4*log(x)*log(y) + 3*log(x)) == x**3 * exp(4*log(x)*log(y))

    # 检查 exp(-oo, evaluate=False) 的无限性属性是否为 True
    assert exp(-oo, evaluate=False).is_finite is True
    # 检查 exp(oo, evaluate=False) 的无限性属性是否为 False
    assert exp(oo, evaluate=False).is_finite is False


# 使用装饰器 @_both_exp_pow 定义测试函数 test_exp_period
@_both_exp_pow
def test_exp_period():
    # 检查 exp(I*pi*9/4) 结果
    # 检查复数指数函数的等式是否成立，左侧和右侧分别为指数函数的值
    assert exp(2 - I*pi*Rational(17, 5)) == exp(2 + I*pi*Rational(3, 5))
    
    # 检查复数对数函数与指数函数乘积的等式是否成立，左侧和右侧分别为乘积的值
    assert exp(log(3) + I*pi*Rational(29, 9)) == 3 * exp(I*pi*Rational(-7, 9))
    
    # 定义整数符号变量 n
    n = Symbol('n', integer=True)
    # 定义偶数符号变量 e
    e = Symbol('e', even=True)
    
    # 检查复数指数函数的值是否为 1
    assert exp(e*I*pi) == 1
    # 检查复数指数函数的值是否为 -1
    assert exp((e + 1)*I*pi) == -1
    # 检查复数指数函数的值是否为纯虚数 I
    assert exp((1 + 4*n)*I*pi/2) == I
    # 检查复数指数函数的值是否为负纯虚数 -I
    assert exp((-1 + 4*n)*I*pi/2) == -I
@_both_exp_pow
def test_exp_log():
    # 定义一个符号变量 x，限定其为实数
    x = Symbol("x", real=True)
    # 断言 log(exp(x)) 等于 x
    assert log(exp(x)) == x
    # 断言 exp(log(x)) 等于 x
    assert exp(log(x)) == x

    # 如果全局参数中 exp_is_pow 为假
    if not global_parameters.exp_is_pow:
        # 断言 log(x) 的逆操作等于 exp
        assert log(x).inverse() == exp
        # 断言 exp(x) 的逆操作等于 log
        assert exp(x).inverse() == log

    # 定义一个符号变量 y，限定其为极坐标形式
    y = Symbol("y", polar=True)
    # 断言 log(exp_polar(z)) 等于 z
    assert log(exp_polar(z)) == z
    # 断言 exp(log(y)) 等于 y
    assert exp(log(y)) == y


@_both_exp_pow
def test_exp_expand():
    # 定义 e 表达式，展开后应等于 2
    e = exp(log(Rational(2))*(1 + x) - log(Rational(2))*x)
    assert e.expand() == 2
    # 断言 exp(x + y) 不等于 exp(x)*exp(y)
    assert exp(x + y) != exp(x)*exp(y)
    # 断言 exp(x + y) 展开后等于 exp(x)*exp(y)
    assert exp(x + y).expand() == exp(x)*exp(y)


@_both_exp_pow
def test_exp__as_base_exp():
    # 断言 exp(x) 的基数和指数应分别为 E 和 x
    assert exp(x).as_base_exp() == (E, x)
    # 断言 exp(2*x) 的基数和指数应分别为 E 和 2*x
    assert exp(2*x).as_base_exp() == (E, 2*x)
    # 断言 exp(x*y) 的基数和指数应分别为 E 和 x*y
    assert exp(x*y).as_base_exp() == (E, x*y)
    # 断言 exp(-x) 的基数和指数应分别为 E 和 -x
    assert exp(-x).as_base_exp() == (E, -x)

    # Pow( *expr.as_base_exp() ) == expr 应该成立
    assert E**x == exp(x)
    assert E**(2*x) == exp(2*x)
    assert E**(x*y) == exp(x*y)

    # 断言 exp(x) 的基数为 S.Exp1
    assert exp(x).base is S.Exp1
    # 断言 exp(x) 的指数为 x
    assert exp(x).exp == x


@_both_exp_pow
def test_exp_infinity():
    # 断言 exp(I*y) 不等于 nan
    assert exp(I*y) != nan
    # 断言 refine(exp(I*oo)) 结果为 nan
    assert refine(exp(I*oo)) is nan
    # 断言 refine(exp(-I*oo)) 结果为 nan
    assert refine(exp(-I*oo)) is nan
    # 断言 exp(y*I*oo) 不等于 nan
    assert exp(y*I*oo) != nan
    # 断言 exp(zoo) 结果为 nan
    assert exp(zoo) is nan
    # 定义一个扩展实数变量 x，有限性为 False
    x = Symbol('x', extended_real=True, finite=False)
    # 断言 exp(x).is_complex 结果为 None
    assert exp(x).is_complex is None


@_both_exp_pow
def test_exp_subs():
    # 定义一个符号变量 x
    x = Symbol('x')
    # 定义 e 表达式，未评估为 x**3
    e = (exp(3*log(x), evaluate=False))  # evaluates to x**3
    # 断言 e 在用 x**3 替换时等于自身
    assert e.subs(x**3, y**3) == e
    # 断言 e 在用 x**2 替换时等于自身
    assert e.subs(x**2, 5) == e
    # 断言 x**3 在用 x**2 替换时不等于 y**Rational(3, 2)
    assert (x**3).subs(x**2, y) != y**Rational(3, 2)
    # 断言 exp(exp(x) + exp(x**2)) 在用 exp(exp(x)) 替换为 y 时等于 y * exp(exp(x**2))
    assert exp(exp(x) + exp(x**2)).subs(exp(exp(x)), y) == y * exp(exp(x**2))
    # 断言 exp(x) 在用 E 替换为 y 时等于 y**x
    assert exp(x).subs(E, y) == y**x
    # 重新定义一个实数变量 x
    x = symbols('x', real=True)
    # 断言 exp(5*x) 在用 exp(7*x) 替换为 y 时等于 y**(5/7)
    assert exp(5*x).subs(exp(7*x), y) == y**Rational(5, 7)
    # 断言 exp(2*x + 7) 在用 exp(3*x) 替换为 y 时等于 y**(2/3) * exp(7)
    assert exp(2*x + 7).subs(exp(3*x), y) == y**Rational(2, 3) * exp(7)
    # 重新定义一个正数变量 x
    x = symbols('x', positive=True)
    # 断言 exp(3*log(x)) 在用 x**2 替换为 y 时等于 y**(3/2)
    assert exp(3*log(x)).subs(x**2, y) == y**Rational(3, 2)
    # 区分 E 和 exp
    # 断言 exp(exp(x + E)) 在用 exp 替换为 3 时等于 3**(3**(x + E))
    assert exp(exp(x + E)).subs(exp, 3) == 3**(3**(x + E))
    # 断言 exp(exp(x + E)) 在用 exp 替换为 sin 时等于 sin(sin(x + E))
    assert exp(exp(x + E)).subs(exp, sin) == sin(sin(x + E))
    # 断言 exp(exp(x + E)) 在用 E 替换为 3 时等于 3**(3**(x + 3))
    assert exp(exp(x + E)).subs(E, 3) == 3**(3**(x + 3))
    # 断言 exp(3) 在用 E 替换为 sin 时等于 sin(3)
    assert exp(3).subs(E, sin) == sin(3)


def test_exp_adjoint():
    # 断言 exp(x) 的伴随操作等于 exp(x) 的伴随
    assert adjoint(exp(x)) == exp(adjoint(x))


def test_exp_conjugate():
    # 断言 exp(x) 的共轭等于 exp(x) 的共轭
    assert conjugate(exp(x)) == exp(conjugate(x))


@_both_exp_pow
def test_exp_transpose():
    # 断言 exp(x) 的转置等于 exp(x) 的转置
    assert transpose(exp(x)) == exp(transpose(x))


@_both_exp_pow
def test_exp_rewrite():
    # 断言 exp(x) 重写为 sinh(x) + cosh(x)
    assert exp(x).rewrite(sin) == sinh(x) + cosh(x)
    # 断言 exp(x*I) 重写为 cos(x) + I*sin(x)
    assert exp(x*I).rewrite(cos) == cos(x) + I*sin(x)
    # 断言 exp(1) 重写为 cosh(1) + sinh(1)
    assert exp(1).rewrite(cos) == sinh(1) + cosh(1)
    # 断言 exp(1) 重写为 sinh(1) + cosh(1)
    assert exp(1).rewrite(sin) == sinh(1) + cosh(1)
    # 断言 exp(x) 重写为 (1 + tanh(x/2))/(1 - tanh(x/2))
    assert exp(x).rewrite(tanh) == (1 + tanh(x/2))/(1 - tanh(x/2))
    # 断言 exp(pi*I/4) 重写为 sqrt(2)/2 + sqrt(2)*I/2
    assert exp(pi*I/4).rewrite(sqrt) == sqrt(2)/2 + sqrt(2)*I/2
    # 断言 exp(pi*I/3) 重写
    # 如果全局参数中的 exp_is_pow 不为真，则执行以下断言
    if not global_parameters.exp_is_pow:
        # 断言验证 exp(x*log(y)) 重写为 y**x
        assert exp(x*log(y)).rewrite(Pow) == y**x
        # 断言验证 exp(log(x)*log(y)) 重写为 x**log(y) 或者 y**log(x) 中的一种
        assert exp(log(x)*log(y)).rewrite(Pow) in [x**log(y), y**log(x)]
        # 断言验证 exp(log(log(x))*y) 重写为 log(x)**y
        assert exp(log(log(x))*y).rewrite(Pow) == log(x)**y

    # 创建一个整数符号 n
    n = Symbol('n', integer=True)

    # 断言验证级数求和 Sum((exp(pi*I/2)/2)**n, (n, 0, oo)) 重写为 sqrt，并计算结果
    assert Sum((exp(pi*I/2)/2)**n, (n, 0, oo)).rewrite(sqrt).doit() == Rational(4, 5) + I*2/5
    # 断言验证级数求和 Sum((exp(pi*I/4)/2)**n, (n, 0, oo)) 重写为 sqrt，并计算结果
    assert Sum((exp(pi*I/4)/2)**n, (n, 0, oo)).rewrite(sqrt).doit() == 1/(1 - sqrt(2)*(1 + I)/4)
    # 断言验证级数求和 Sum((exp(pi*I/3)/2)**n, (n, 0, oo)) 重写为 sqrt，并取消简化后的结果
    assert (Sum((exp(pi*I/3)/2)**n, (n, 0, oo)).rewrite(sqrt).doit().cancel()
            == 4*I/(sqrt(3) + 3*I))


这段代码中包含了几个数学表达式的断言，用来验证特定数学恒等式在某些条件下的正确性。
@_both_exp_pow
def test_exp_leading_term():
    # 测试 exp(x) 的主导项是否为 1
    assert exp(x).as_leading_term(x) == 1
    # 测试 exp(2 + x) 的主导项是否为 exp(2)
    assert exp(2 + x).as_leading_term(x) == exp(2)
    # 测试 exp((2*x + 3) / (x+1)) 的主导项是否为 exp(3)
    assert exp((2*x + 3) / (x+1)).as_leading_term(x) == exp(3)

    # 以下测试被注释掉，因为 SymPy 现在在级数展开的主导项不存在时返回原始函数。
    # raises(NotImplementedError, lambda: exp(1/x).as_leading_term(x))
    # raises(NotImplementedError, lambda: exp((x + 1) / x**2).as_leading_term(x))
    # raises(NotImplementedError, lambda: exp(x + 1/x).as_leading_term(x))


@_both_exp_pow
def test_exp_taylor_term():
    x = symbols('x')
    # 测试 exp(x) 的泰勒展开项(n=1)是否为 x
    assert exp(x).taylor_term(1, x) == x
    # 测试 exp(x) 的泰勒展开项(n=3)是否为 x**3/6
    assert exp(x).taylor_term(3, x) == x**3/6
    # 测试 exp(x) 的泰勒展开项(n=4)是否为 x**4/24
    assert exp(x).taylor_term(4, x) == x**4/24
    # 测试 exp(x) 的泰勒展开项(n=-1)是否为 0
    assert exp(x).taylor_term(-1, x) is S.Zero


def test_exp_MatrixSymbol():
    A = MatrixSymbol("A", 2, 2)
    # 测试 exp(A) 是否包含 exp 函数
    assert exp(A).has(exp)


def test_exp_fdiff():
    x = Symbol('x')
    # 测试 exp(x) 的二阶导数是否会引发 ArgumentIndexError 异常
    raises(ArgumentIndexError, lambda: exp(x).fdiff(2))


def test_log_values():
    # 测试 log 函数对各种特殊值的处理情况

    assert log(nan) is nan

    assert log(oo) is oo
    assert log(-oo) is oo

    assert log(zoo) is zoo
    assert log(-zoo) is zoo

    assert log(0) is zoo

    assert log(1) == 0
    assert log(-1) == I*pi

    assert log(E) == 1
    assert log(-E).expand() == 1 + I*pi

    assert unchanged(log, pi)
    assert log(-pi).expand() == log(pi) + I*pi

    assert unchanged(log, 17)
    assert log(-17) == log(17) + I*pi

    assert log(I) == I*pi/2
    assert log(-I) == -I*pi/2

    assert log(17*I) == I*pi/2 + log(17)
    assert log(-17*I).expand() == -I*pi/2 + log(17)

    assert log(oo*I) is oo
    assert log(-oo*I) is oo
    assert log(0, 2) is zoo
    assert log(0, 5) is zoo

    assert exp(-log(3))**(-1) == 3

    assert log(S.Half) == -log(2)
    assert log(2*3).func is log
    assert log(2*3**2).func is log


def test_match_real_imag():
    x, y = symbols('x,y', real=True)
    i = Symbol('i', imaginary=True)
    # 测试 match_real_imag 函数对不同类型数值的处理
    assert match_real_imag(S.One) == (1, 0)
    assert match_real_imag(I) == (0, 1)
    assert match_real_imag(3 - 5*I) == (3, -5)
    assert match_real_imag(-sqrt(3) + S.Half*I) == (-sqrt(3), S.Half)
    assert match_real_imag(x + y*I) == (x, y)
    assert match_real_imag(x*I + y*I) == (0, x + y)
    assert match_real_imag((x + y)*I) == (0, x + y)
    assert match_real_imag(Rational(-2, 3)*i*I) == (None, None)
    assert match_real_imag(1 - 2*i) == (None, None)
    assert match_real_imag(sqrt(2)*(3 - 5*I)) == (None, None)


def test_log_exact():
    # 检查对于 pi/2, pi/3, pi/4, pi/6, pi/8, pi/12; pi/5, pi/10 的 log 函数精确性
    for n in range(-23, 24):
        if gcd(n, 24) != 1:
            assert log(exp(n*I*pi/24).rewrite(sqrt)) == n*I*pi/24
        for n in range(-9, 10):
            assert log(exp(n*I*pi/10).rewrite(sqrt)) == n*I*pi/10

    assert log(S.Half - I*sqrt(3)/2) == -I*pi/3
    assert log(Rational(-1, 2) + I*sqrt(3)/2) == I*pi*Rational(2, 3)
    # 断言：计算 log(-sqrt(2)/2 - I*sqrt(2)/2)，应该等于 -I*pi*Rational(3, 4)
    assert log(-sqrt(2)/2 - I*sqrt(2)/2) == -I*pi*Rational(3, 4)
    
    # 断言：计算 log(-sqrt(3)/2 - I*S.Half)，应该等于 -I*pi*Rational(5, 6)
    assert log(-sqrt(3)/2 - I*S.Half) == -I*pi*Rational(5, 6)
    
    # 断言：计算 log(Rational(-1, 4) + sqrt(5)/4 - I*sqrt(sqrt(5)/8 + Rational(5, 8)))，应该等于 -I*pi*Rational(2, 5)
    assert log(Rational(-1, 4) + sqrt(5)/4 - I*sqrt(sqrt(5)/8 + Rational(5, 8))) == -I*pi*Rational(2, 5)
    
    # 断言：计算 log(sqrt(Rational(5, 8) - sqrt(5)/8) + I*(Rational(1, 4) + sqrt(5)/4))，应该等于 I*pi*Rational(3, 10)
    assert log(sqrt(Rational(5, 8) - sqrt(5)/8) + I*(Rational(1, 4) + sqrt(5)/4)) == I*pi*Rational(3, 10)
    
    # 断言：计算 log(-sqrt(sqrt(2)/4 + S.Half) + I*sqrt(S.Half - sqrt(2)/4))，应该等于 I*pi*Rational(7, 8)
    assert log(-sqrt(sqrt(2)/4 + S.Half) + I*sqrt(S.Half - sqrt(2)/4)) == I*pi*Rational(7, 8)
    
    # 断言：计算 log(-sqrt(6)/4 - sqrt(2)/4 + I*(-sqrt(6)/4 + sqrt(2)/4))，应该等于 -I*pi*Rational(11, 12)
    assert log(-sqrt(6)/4 - sqrt(2)/4 + I*(-sqrt(6)/4 + sqrt(2)/4)) == -I*pi*Rational(11, 12)
    
    # 断言：计算 log(-1 + I*sqrt(3))，应该等于 log(2) + I*pi*Rational(2, 3)
    assert log(-1 + I*sqrt(3)) == log(2) + I*pi*Rational(2, 3)
    
    # 断言：计算 log(5 + 5*I)，应该等于 log(5*sqrt(2)) + I*pi/4
    assert log(5 + 5*I) == log(5*sqrt(2)) + I*pi/4
    
    # 断言：计算 log(sqrt(-12))，应该等于 log(2*sqrt(3)) + I*pi/2
    assert log(sqrt(-12)) == log(2*sqrt(3)) + I*pi/2
    
    # 断言：计算 log(-sqrt(6) + sqrt(2) - I*sqrt(6) - I*sqrt(2))，应该等于 log(4) - I*pi*Rational(7, 12)
    assert log(-sqrt(6) + sqrt(2) - I*sqrt(6) - I*sqrt(2)) == log(4) - I*pi*Rational(7, 12)
    
    # 断言：计算 log(-sqrt(6-3*sqrt(2)) - I*sqrt(6+3*sqrt(2)))，应该等于 log(2*sqrt(3)) - I*pi*Rational(5, 8)
    assert log(-sqrt(6-3*sqrt(2)) - I*sqrt(6+3*sqrt(2))) == log(2*sqrt(3)) - I*pi*Rational(5, 8)
    
    # 断言：计算 log(1 + I*sqrt(2-sqrt(2))/sqrt(2+sqrt(2)))，应该等于 log(2/sqrt(sqrt(2) + 2)) + I*pi/8
    assert log(1 + I*sqrt(2-sqrt(2))/sqrt(2+sqrt(2))) == log(2/sqrt(sqrt(2) + 2)) + I*pi/8
    
    # 断言：计算 log(cos(pi*Rational(7, 12)) + I*sin(pi*Rational(7, 12)))，应该等于 I*pi*Rational(7, 12)
    assert log(cos(pi*Rational(7, 12)) + I*sin(pi*Rational(7, 12))) == I*pi*Rational(7, 12)
    
    # 断言：计算 log(cos(pi*Rational(6, 5)) + I*sin(pi*Rational(6, 5)))，应该等于 I*pi*Rational(-4, 5)
    assert log(cos(pi*Rational(6, 5)) + I*sin(pi*Rational(6, 5))) == I*pi*Rational(-4, 5)
    
    # 断言：计算 log(5*(1 + I)/sqrt(2))，应该等于 log(5) + I*pi/4
    assert log(5*(1 + I)/sqrt(2)) == log(5) + I*pi/4
    
    # 断言：计算 log(sqrt(2)*(-sqrt(3) + 1 - sqrt(3)*I - I))，应该等于 log(4) - I*pi*Rational(7, 12)
    assert log(sqrt(2)*(-sqrt(3) + 1 - sqrt(3)*I - I)) == log(4) - I*pi*Rational(7, 12)
    
    # 断言：计算 log(-sqrt(2)*(1 - I*sqrt(3)))，应该等于 log(2*sqrt(2)) + I*pi*Rational(2, 3)
    assert log(-sqrt(2)*(1 - I*sqrt(3))) == log(2*sqrt(2)) + I*pi*Rational(2, 3)
    
    # 断言：计算 log(sqrt(3)*I*(-sqrt(6 - 3*sqrt(2)) - I*sqrt(3*sqrt(2) + 6)))，应该等于 log(6) - I*pi/8
    assert log(sqrt(3)*I*(-sqrt(6 - 3*sqrt(2)) - I*sqrt(3*sqrt(2) + 6))) == log(6) - I*pi/8
    
    # 计算 zero 的值
    zero = (1 + sqrt(2))**2 - 3 - 2*sqrt(2)
    
    # 断言：计算 log(zero - I*sqrt(3))，应该等于 log(sqrt(3)) - I*pi/2
    assert log(zero - I*sqrt(3)) == log(sqrt(3)) - I*pi/2
    
    # 断言：检查 log 函数在 zero + I*zero 上是否保持不变，或者 zero + zero*I 是否是无穷大
    assert unchanged(log, zero + I*zero) or log(zero + zero*I) is zoo
    
    # 断言：快速失败，如果无法进行明显的简化
    assert unchanged(log, (sqrt(2)-1/sqrt(sqrt(3)+I))**1000)
    
    # 断言：注意非实系数
    assert unchanged(log, sqrt(2-sqrt(5))*(1 + I))
# 定义测试函数 test_log_base，用于测试对数函数 log 的基本功能和特性
def test_log_base():
    # 断言 log(1, 2) 等于 0
    assert log(1, 2) == 0
    # 断言 log(2, 2) 等于 1
    assert log(2, 2) == 1
    # 断言 log(3, 2) 等于 log(3) / log(2)
    assert log(3, 2) == log(3) / log(2)
    # 断言 log(6, 2) 等于 1 + log(3) / log(2)
    assert log(6, 2) == 1 + log(3) / log(2)
    # 断言 log(6, 3) 等于 1 + log(2) / log(3)
    assert log(6, 3) == 1 + log(2) / log(3)
    # 断言 log(2**3, 2) 等于 3
    assert log(2**3, 2) == 3
    # 断言 log(3**3, 3) 等于 3
    assert log(3**3, 3) == 3
    # 断言 log(5, 1) 为正无穷
    assert log(5, 1) is zoo
    # 断言 log(1, 1) 为非数
    assert log(1, 1) is nan
    # 断言 log(Rational(2, 3), 10) 等于 log(Rational(2, 3)) / log(10)
    assert log(Rational(2, 3), 10) == log(Rational(2, 3)) / log(10)
    # 断言 log(Rational(2, 3), Rational(1, 3)) 等于 -log(2) / log(3) + 1
    assert log(Rational(2, 3), Rational(1, 3)) == -log(2) / log(3) + 1
    # 断言 log(Rational(2, 3), Rational(2, 5)) 等于 log(Rational(2, 3)) / log(Rational(2, 5))
    assert log(Rational(2, 3), Rational(2, 5)) == \
        log(Rational(2, 3)) / log(Rational(2, 5))
    # issue 17148
    # 断言 log(Rational(8, 3), 2) 等于 -log(3) / log(2) + 3
    assert log(Rational(8, 3), 2) == -log(3) / log(2) + 3


# 定义测试函数 test_log_symbolic，用于测试对数函数 log 在符号表达式中的行为
def test_log_symbolic():
    # 断言 log(x, exp(1)) 等于 log(x)
    assert log(x, exp(1)) == log(x)
    # 断言 log(exp(x)) 不等于 x
    assert log(exp(x)) != x

    # 断言 log(x, exp(1)) 等于 log(x)
    assert log(x, exp(1)) == log(x)
    # 断言 log(x*y) 不等于 log(x) + log(y)
    assert log(x*y) != log(x) + log(y)
    # 断言 log(x/y).expand() 不等于 log(x) - log(y)
    assert log(x/y).expand() != log(x) - log(y)
    # 断言 log(x/y).expand(force=True) 等于 log(x) - log(y)
    assert log(x/y).expand(force=True) == log(x) - log(y)
    # 断言 log(x**y).expand() 不等于 y*log(x)
    assert log(x**y).expand() != y*log(x)
    # 断言 log(x**y).expand(force=True) 等于 y*log(x)
    assert log(x**y).expand(force=True) == y*log(x)

    # 断言 log(x, 2) 等于 log(x) / log(2)
    assert log(x, 2) == log(x) / log(2)
    # 断言 log(E, 2) 等于 1 / log(2)
    assert log(E, 2) == 1 / log(2)

    # 定义符号 p, q，限定其为正数
    p, q = symbols('p,q', positive=True)
    r = Symbol('r', real=True)

    # 断言 log(p**2) 不等于 2*log(p)
    assert log(p**2) != 2*log(p)
    # 断言 log(p**2).expand() 等于 2*log(p)
    assert log(p**2).expand() == 2*log(p)
    # 断言 log(x**2).expand() 不等于 2*log(x)
    assert log(x**2).expand() != 2*log(x)
    # 断言 log(p**q) 不等于 q*log(p)
    assert log(p**q) != q*log(p)
    # 断言 log(exp(p)) 等于 p
    assert log(exp(p)) == p
    # 断言 log(p*q) 不等于 log(p) + log(q)
    assert log(p*q) != log(p) + log(q)
    # 断言 log(p*q).expand() 等于 log(p) + log(q)
    assert log(p*q).expand() == log(p) + log(q)

    # 断言 log(-sqrt(3)) 等于 log(sqrt(3)) + I*pi
    assert log(-sqrt(3)) == log(sqrt(3)) + I*pi
    # 断言 log(-exp(p)) 不等于 p + I*pi
    assert log(-exp(p)) != p + I*pi
    # 断言 log(-exp(x)).expand() 不等于 x + I*pi
    assert log(-exp(x)).expand() != x + I*pi
    # 断言 log(-exp(r)).expand() 等于 r + I*pi
    assert log(-exp(r)).expand() == r + I*pi

    # 断言 log(x**y) 不等于 y*log(x)
    assert log(x**y) != y*log(x)

    # 断言 (log(x**-5)**-1).expand() 不等于 -1/log(x)/5
    assert (log(x**-5)**-1).expand() != -1/log(x)/5
    # 断言 (log(p**-5)**-1).expand() 等于 -1/log(p)/5
    assert (log(p**-5)**-1).expand() == -1/log(p)/5
    # 断言 log(-x).func is log and log(-x).args[0] 等于 -x
    assert log(-x).func is log and log(-x).args[0] == -x
    # 断言 log(-p).func is log and log(-p).args[0] 等于 -p
    assert log(-p).func is log and log(-p).args[0] == -p


# 定义测试函数 test_log_exp，用于测试对数函数 log 与指数函数 exp 的交互作用
def test_log_exp():
    # 断言 log(exp(4*I*pi)) 等于 0，exp 计算结果
    assert log(exp(4*I*pi)) == 0
    # 断言 log(exp(-5*I*pi)) 等于 I*pi，exp 计算结果
    assert log(exp(-5*I*pi)) == I*pi
    # 断言 log(exp(I*pi*Rational(19, 4))) 等于 I*pi*Rational(3, 4)
    assert log(exp(I*pi*Rational(19, 4))) == I*pi*Rational(3, 4)
    # 断言 log(exp(I*pi*Rational(25, 7))) 等于 I*pi*Rational(-3, 7)
    assert log(exp(I*pi*Rational(25, 7))) == I*pi*Rational(-3, 7)
    # 断言 log(exp(-5*I)) 等于 -5*I + 2*I*pi
    assert log(exp(-5*I)) == -5*I + 2*I*pi


# 定义装饰器函数 test_exp_assumptions，用于测试指数函数 exp 的各种假设条件
@_both_exp_pow
def test_exp_assumptions():
    # 定义实数符号 r 和虚数符号 i
    r = Symbol('r', real=True)
    i = Symbol('i', imaginary=True)
    # 遍历 exp 和 exp_polar 函数
    # 创建一个有理数符号对象 'r'，默认为有理数
    r = Symbol('r', rational=True)
    
    # 创建一个有理数非零符号对象 'rn'，要求其为非零有理数
    rn = Symbol('rn', rational=True, nonzero=True)
    
    # 断言：计算指数函数 exp(a) 的代数性质，预期结果为 None
    assert exp(a).is_algebraic is None
    
    # 断言：计算指数函数 exp(an) 的代数性质，预期结果为 False
    assert exp(an).is_algebraic is False
    
    # 断言：计算指数函数 exp(pi*r) 的代数性质，预期结果为 None
    assert exp(pi*r).is_algebraic is None
    
    # 断言：计算指数函数 exp(pi*rn) 的代数性质，预期结果为 False
    assert exp(pi*rn).is_algebraic is False
    
    # 断言：计算指数函数 exp(0) 的代数性质，强制关闭求值，预期结果为 True
    assert exp(0, evaluate=False).is_algebraic is True
    
    # 断言：计算指数函数 exp(I*pi/3) 的代数性质，强制关闭求值，预期结果为 True
    assert exp(I*pi/3, evaluate=False).is_algebraic is True
    
    # 断言：计算指数函数 exp(I*pi*r) 的代数性质，强制关闭求值，预期结果为 True
    assert exp(I*pi*r, evaluate=False).is_algebraic is True
# 定义装饰器函数 @_both_exp_pow，用于测试 AccumBounds 对象的指数函数
@_both_exp_pow
def test_exp_AccumBounds():
    # 断言 exp(AccumBounds(1, 2)) 的结果是否等于 AccumBounds(E, E**2)
    assert exp(AccumBounds(1, 2)) == AccumBounds(E, E**2)


# 测试对数函数 log 的各种属性
def test_log_assumptions():
    # 声明正数符号 p、负数符号 n、零符号 z、正无穷符号 x
    p = symbols('p', positive=True)
    n = symbols('n', negative=True)
    z = symbols('z', zero=True)
    x = symbols('x', infinite=True, extended_positive=True)

    # 断言 log(z) 不是正数
    assert log(z).is_positive is False
    # 断言 log(x) 是扩展正数
    assert log(x).is_extended_positive is True
    # 断言 log(2) 大于 0
    assert log(2) > 0
    # 断言 log(1, evaluate=False) 是零
    assert log(1, evaluate=False).is_zero
    # 断言 log(1 + z) 是零
    assert log(1 + z).is_zero
    # 断言 log(p) 是零
    assert log(p).is_zero is None
    # 断言 log(n) 不是零
    assert log(n).is_zero is False
    # 断言 log(0.5) 是负数
    assert log(0.5).is_negative is True
    # 断言 log(exp(p) + 1) 是正数
    assert log(exp(p) + 1).is_positive

    # 断言 log(1, evaluate=False) 是代数的
    assert log(1, evaluate=False).is_algebraic
    # 断言 log(42, evaluate=False) 不是代数的
    assert log(42, evaluate=False).is_algebraic is False

    # 断言 log(1 + z) 是有理数
    assert log(1 + z).is_rational


# 测试 log 函数的哈希值
def test_log_hashing():
    # 断言 x 不等于 log(log(x))
    assert x != log(log(x))
    # 断言 x 的哈希值不等于 log(log(x)) 的哈希值
    assert hash(x) != hash(log(log(x)))
    # 断言 log(x) 不等于 log(log(log(x)))
    assert log(x) != log(log(log(x)))

    # 创建表达式 e，用于测试 log(log(x) + log(log(x)))
    e = 1/log(log(x) + log(log(x)))
    # 断言 e 的基函数是 log
    assert e.base.func is log
    # 创建另一个表达式 e，用于测试 log(log(x) + log(log(log(x))))
    e = 1/log(log(x) + log(log(log(x))))
    # 断言 e 的基函数是 log
    assert e.base.func is log

    # 创建表达式 e，用于测试 log(log(x))
    e = log(log(x))
    # 断言 e 的函数是 log
    assert e.func is log
    # 断言 x 的函数不是 log
    assert x.func is not log
    # 断言 log(log(log(x))) 的哈希值不等于 x 的哈希值
    assert hash(log(log(x))) != hash(x)
    # 断言 e 不等于 x
    assert e != x


# 测试 log 函数的符号
def test_log_sign():
    # 断言 log(2) 的符号是 1
    assert sign(log(2)) == 1


# 测试复数域下 log 函数的展开
def test_log_expand_complex():
    # 断言 log(1 + I) 在复数域下的展开结果
    assert log(1 + I).expand(complex=True) == log(2)/2 + I*pi/4
    # 断言 log(1 - sqrt(2)) 在复数域下的展开结果
    assert log(1 - sqrt(2)).expand(complex=True) == log(sqrt(2) - 1) + I*pi


# 测试 log 函数的数值应用
def test_log_apply_evalf():
    # 计算 log(3)/log(2) - 1 的数值近似，与指定精确值比较
    value = (log(3)/log(2) - 1).evalf()
    assert value.epsilon_eq(Float("0.58496250072115618145373"))


# 测试 log 函数的主导项
def test_log_leading_term():
    p = Symbol('p')

    # 测试 STEP 3，log(1 + x + x**2) 的主导项
    assert log(1 + x + x**2).as_leading_term(x, cdir=1) == x
    # 测试 STEP 4，log(2*x) 的主导项
    assert log(2*x).as_leading_term(x, cdir=1) == log(x) + log(2)
    assert log(2*x).as_leading_term(x, cdir=-1) == log(x) + log(2)
    assert log(-2*x).as_leading_term(x, cdir=1, logx=p) == p + log(2) + I*pi
    assert log(-2*x).as_leading_term(x, cdir=-1, logx=p) == p + log(2) - I*pi
    # 测试 STEP 5
    assert log(-2*x + (3 - I)*x**2).as_leading_term(x, cdir=1) == log(x) + log(2) - I*pi
    assert log(-2*x + (3 - I)*x**2).as_leading_term(x, cdir=-1) == log(x) + log(2) - I*pi
    assert log(2*x + (3 - I)*x**2).as_leading_term(x, cdir=1) == log(x) + log(2)
    assert log(2*x + (3 - I)*x**2).as_leading_term(x, cdir=-1) == log(x) + log(2) - 2*I*pi
    assert log(-1 + x - I*x**2 + I*x**3).as_leading_term(x, cdir=1) == -I*pi
    assert log(-1 + x - I*x**2 + I*x**3).as_leading_term(x, cdir=-1) == -I*pi
    assert log(-1/(1 - x)).as_leading_term(x, cdir=1) == I*pi
    assert log(-1/(1 - x)).as_leading_term(x, cdir=-1) == I*pi


# 测试 log 函数的数列展开
def test_log_nseries():
    p = Symbol('p')
    # 断言 log(1/x) 在给定条件下的数列展开结果
    assert log(1/x)._eval_nseries(x, 4, logx=-p, cdir=1) == p
    assert log(1/x)._eval_nseries(x, 4, logx=-p, cdir=-1) == p + 2*I*pi
    assert log(x - 1)._eval_nseries(x, 4, None, I) == I*pi - x - x**2/2 - x**3/3 + O(x**4)
    assert log(x - 1)._eval_nseries(x, 4, None, -I) == -I*pi - x - x**2/2 - x**3/3 + O(x**4)
    # 断言：对 log(I*x + I*x**3 - 1) 进行 n 次级数展开，并验证展开结果是否等于指定表达式
    assert log(I*x + I*x**3 - 1)._eval_nseries(x, 3, None, 1) == I*pi - I*x + x**2/2 + O(x**3)
    
    # 断言：对 log(I*x + I*x**3 - 1) 进行 n 次级数展开，并验证展开结果是否等于指定表达式
    assert log(I*x + I*x**3 - 1)._eval_nseries(x, 3, None, -1) == -I*pi - I*x + x**2/2 + O(x**3)
    
    # 断言：对 log(I*x**2 + I*x**3 - 1) 进行 n 次级数展开，并验证展开结果是否等于指定表达式
    assert log(I*x**2 + I*x**3 - 1)._eval_nseries(x, 3, None, 1) == I*pi - I*x**2 + O(x**3)
    
    # 断言：对 log(I*x**2 + I*x**3 - 1) 进行 n 次级数展开，并验证展开结果是否等于指定表达式
    assert log(I*x**2 + I*x**3 - 1)._eval_nseries(x, 3, None, -1) == I*pi - I*x**2 + O(x**3)
    
    # 断言：对 log(2*x + (3 - I)*x**2) 进行 n 次级数展开，并验证展开结果是否等于指定表达式
    assert log(2*x + (3 - I)*x**2)._eval_nseries(x, 3, None, 1) == log(2) + log(x) + \
        x*(S(3)/2 - I/2) + x**2*(-1 + 3*I/4) + O(x**3)
    
    # 断言：对 log(2*x + (3 - I)*x**2) 进行 n 次级数展开，并验证展开结果是否等于指定表达式
    assert log(2*x + (3 - I)*x**2)._eval_nseries(x, 3, None, -1) == -2*I*pi + log(2) + \
        log(x) - x*(-S(3)/2 + I/2) + x**2*(-1 + 3*I/4) + O(x**3)
    
    # 断言：对 log(-2*x + (3 - I)*x**2) 进行 n 次级数展开，并验证展开结果是否等于指定表达式
    assert log(-2*x + (3 - I)*x**2)._eval_nseries(x, 3, None, 1) == -I*pi + log(2) + log(x) + \
        x*(-S(3)/2 + I/2) + x**2*(-1 + 3*I/4) + O(x**3)
    
    # 断言：对 log(-2*x + (3 - I)*x**2) 进行 n 次级数展开，并验证展开结果是否等于指定表达式
    assert log(-2*x + (3 - I)*x**2)._eval_nseries(x, 3, None, -1) == -I*pi + log(2) + log(x) - \
        x*(S(3)/2 - I/2) + x**2*(-1 + 3*I/4) + O(x**3)
    
    # 断言：对 log(sqrt(-I*x**2 - 3)*sqrt(-I*x**2 - 1) - 2) 进行 n 次级数展开，并验证展开结果是否等于指定表达式
    assert log(sqrt(-I*x**2 - 3)*sqrt(-I*x**2 - 1) - 2)._eval_nseries(x, 3, None, 1) == -I*pi + \
        log(sqrt(3) + 2) + 2*sqrt(3)*I*x**2/(3*sqrt(3) + 6) + O(x**3)
    
    # 断言：对 log(-1/(1 - x)) 进行 n 次级数展开，并验证展开结果是否等于指定表达式
    assert log(-1/(1 - x))._eval_nseries(x, 3, None, 1) == I*pi + x + x**2/2 + O(x**3)
    
    # 断言：对 log(-1/(1 - x)) 进行 n 次级数展开，并验证展开结果是否等于指定表达式
    assert log(-1/(1 - x))._eval_nseries(x, 3, None, -1) == I*pi + x + x**2/2 + O(x**3)
# 测试对数函数 log 在不同情况下的展开和简化

def test_log_series():
    # 定义两个对数表达式
    expr1 = log(1 + x)
    expr2 = log(x + sqrt(x**2 + 1))

    # 测试对数在 x 趋向于无穷复数 I*oo 时的级数展开
    assert expr1.series(x, x0=I*oo, n=4) == 1/(3*x**3) - 1/(2*x**2) + 1/x + \
    I*pi/2 - log(I/x) + O(x**(-4), (x, oo*I))
    # 测试对数在 x 趋向于负无穷复数 -I*oo 时的级数展开
    assert expr1.series(x, x0=-I*oo, n=4) == 1/(3*x**3) - 1/(2*x**2) + 1/x - \
    I*pi/2 - log(-I/x) + O(x**(-4), (x, -oo*I))
    # 测试对数在 x 趋向于无穷复数 I*oo 时的级数展开
    assert expr2.series(x, x0=I*oo, n=4) == 1/(4*x**2) + I*pi/2 + log(2) - \
    log(I/x) + O(x**(-4), (x, oo*I))
    # 测试对数在 x 趋向于负无穷复数 -I*oo 时的级数展开
    assert expr2.series(x, x0=-I*oo, n=4) == -1/(4*x**2) - I*pi/2 - log(2) + \
    log(-I/x) + O(x**(-4), (x, -oo*I))


def test_log_expand():
    # 定义符号 w，并计算对数表达式 e
    w = Symbol("w", positive=True)
    e = log(w**(log(5)/log(3)))
    # 测试对数表达式的展开
    assert e.expand() == log(5)/log(3) * log(w)

    # 定义符号 x, y, z，并展开复合对数表达式
    x, y, z = symbols('x,y,z', positive=True)
    assert log(x*(y + z)).expand(mul=False) == log(x) + log(y + z)
    assert log(log(x**2)*log(y*z)).expand() in [log(2*log(x)*log(y) +
        2*log(x)*log(z)), log(log(x)*log(z) + log(y)*log(x)) + log(2),
        log((log(y) + log(z))*log(x)) + log(2)]
    assert log(x**log(x**2)).expand(deep=False) == log(x)*log(x**2)
    assert log(x**log(x**2)).expand() == 2*log(x)**2

    # 再次定义符号 x, y，并展开对数表达式，强制展开
    x, y = symbols('x,y')
    assert log(x*y).expand(force=True) == log(x) + log(y)
    assert log(x**y).expand(force=True) == y*log(x)
    assert log(exp(x)).expand(force=True) == x

    # 提醒：通常不需要展开对数，因为这要求进行因式分解，如果要简化，将对数合并比分开更便宜。
    assert log(2*3**2).expand() != 2*log(3) + log(2)


@XFAIL
def test_log_expand_fail():
    # 定义符号 x, y, z，并测试复合对数表达式的展开，预期失败
    x, y, z = symbols('x,y,z', positive=True)
    assert (log(x*(y + z))*(x + y)).expand(mul=True, log=True) == y*log(
        x) + y*log(y + z) + z*log(x) + z*log(y + z)


def test_log_simplify():
    # 定义符号 x，并测试对数表达式的展开和简化
    x = Symbol("x", positive=True)
    assert log(x**2).expand() == 2*log(x)
    assert expand_log(log(x**(2 + log(2)))) == (2 + log(2))*log(x)

    # 定义符号 z，并测试对数表达式的展开和简化
    z = Symbol('z')
    assert log(sqrt(z)).expand() == log(z)/2
    assert expand_log(log(z**(log(2) - 1))) == (log(2) - 1)*log(z)
    assert log(z**(-1)).expand() != -log(z)
    assert log(z**(x/(x+1))).expand() == x*log(z)/(x + 1)


def test_log_AccumBounds():
    # 测试对数函数应用于 AccumBounds 对象的结果
    assert log(AccumBounds(1, E)) == AccumBounds(0, 1)
    assert log(AccumBounds(0, E)) == AccumBounds(-oo, 1)
    assert log(AccumBounds(-1, E)) == S.NaN
    assert log(AccumBounds(0, oo)) == AccumBounds(-oo, oo)
    assert log(AccumBounds(-oo, 0)) == S.NaN
    assert log(AccumBounds(-oo, oo)) == S.NaN


@_both_exp_pow
def test_lambertw():
    # 定义符号 k，并测试 LambertW 函数的不同情况
    k = Symbol('k')

    assert LambertW(x, 0) == LambertW(x)
    assert LambertW(x, 0, evaluate=False) != LambertW(x)
    assert LambertW(0) == 0
    assert LambertW(E) == 1
    assert LambertW(-1/E) == -1
    assert LambertW(100*log(100)) == log(100)
    # 断言 LambertW(-log(2)/2) 的计算结果等于 -log(2)
    assert LambertW(-log(2)/2) == -log(2)
    # 断言 LambertW(81*log(3)) 的计算结果等于 3*log(3)
    assert LambertW(81*log(3)) == 3*log(3)
    # 断言 LambertW(sqrt(E)/2) 的计算结果等于 S.Half
    assert LambertW(sqrt(E)/2) == S.Half
    # 断言 LambertW(oo) 的计算结果为正无穷
    assert LambertW(oo) is oo
    # 断言 LambertW(0, 1) 的计算结果为负无穷
    assert LambertW(0, 1) is -oo
    # 断言 LambertW(0, 42) 的计算结果为负无穷
    assert LambertW(0, 42) is -oo
    # 断言 LambertW(-pi/2, -1) 的计算结果等于 -I*pi/2
    assert LambertW(-pi/2, -1) == -I*pi/2
    # 断言 LambertW(-1/E, -1) 的计算结果等于 -1
    assert LambertW(-1/E, -1) == -1
    # 断言 LambertW(-2*exp(-2), -1) 的计算结果等于 -2
    assert LambertW(-2*exp(-2), -1) == -2
    # 断言 LambertW(2*log(2)) 的计算结果等于 log(2)
    assert LambertW(2*log(2)) == log(2)
    # 断言 LambertW(-pi/2) 的计算结果等于 I*pi/2
    assert LambertW(-pi/2) == I*pi/2
    # 断言 LambertW(exp(1 + E)) 的计算结果等于 E
    assert LambertW(exp(1 + E)) == E

    # 断言 LambertW(x**2).diff(x) 的计算结果等于 2*LambertW(x**2)/x/(1 + LambertW(x**2))
    assert LambertW(x**2).diff(x) == 2*LambertW(x**2)/x/(1 + LambertW(x**2))
    # 断言 LambertW(x, k).diff(x) 的计算结果等于 LambertW(x, k)/x/(1 + LambertW(x, k))
    assert LambertW(x, k).diff(x) == LambertW(x, k)/x/(1 + LambertW(x, k))

    # 断言 LambertW(sqrt(2)).evalf(30) 的计算结果接近给定的浮点数，精度为 1e-29
    assert LambertW(sqrt(2)).evalf(30).epsilon_eq(
        Float("0.701338383413663009202120278965", 30), 1e-29)
    # 断言 re(LambertW(2, -1)).evalf() 的计算结果接近给定的浮点数
    assert re(LambertW(2, -1)).evalf().epsilon_eq(Float("-0.834310366631110"))

    # 断言 LambertW(-1).is_real 的结果为 False，对应 issue 5215
    assert LambertW(-1).is_real is False
    # 断言 LambertW(2, evaluate=False).is_real 的结果为 True
    assert LambertW(2, evaluate=False).is_real
    # 创建正数符号 p，并断言 LambertW(p, evaluate=False).is_real 的结果为 True
    p = Symbol('p', positive=True)
    assert LambertW(p, evaluate=False).is_real
    # 断言 LambertW(p**(p+1)*log(p)) 的计算结果等于 p*log(p)
    assert LambertW(p**(p+1)*log(p)) == p*log(p)
    # 创建符号 p，并断言 LambertW(p - 1, evaluate=False).is_real 的结果为 None
    assert LambertW(p - 1, evaluate=False).is_real is None
    # 创建符号 p，并断言 LambertW(-p - 2/S.Exp1, evaluate=False).is_real 的结果为 False
    assert LambertW(-p - 2/S.Exp1, evaluate=False).is_real is False
    # 断言 LambertW(S.Half, -1, evaluate=False).is_real 的结果为 False
    assert LambertW(S.Half, -1, evaluate=False).is_real is False
    # 断言 LambertW(Rational(-1, 10), -1, evaluate=False).is_real 的结果为 True
    assert LambertW(Rational(-1, 10), -1, evaluate=False).is_real
    # 断言 LambertW(-10, -1, evaluate=False).is_real 的结果为 False
    assert LambertW(-10, -1, evaluate=False).is_real is False
    # 断言 LambertW(-2, 2, evaluate=False).is_real 的结果为 False
    assert LambertW(-2, 2, evaluate=False).is_real is False

    # 断言 LambertW(0, evaluate=False).is_algebraic 的结果为 True
    assert LambertW(0, evaluate=False).is_algebraic
    # 创建非零代数符号 na，并断言 LambertW(na).is_algebraic 的结果为 False
    na = Symbol('na', nonzero=True, algebraic=True)
    assert LambertW(na).is_algebraic is False
    # 断言 LambertW(p).is_zero 的结果为 False
    assert LambertW(p).is_zero is False
    # 创建负数符号 n，并断言 LambertW(n).is_zero 的结果为 False
    n = Symbol('n', negative=True)
    assert LambertW(n).is_zero is False
# 定义一个名为 test_issue_5673 的测试函数
def test_issue_5673():
    # 使用 LambertW 函数创建一个对象 e，参数为 -1
    e = LambertW(-1)
    # 断言对象 e 的 is_comparable 属性为 False
    assert e.is_comparable is False
    # 断言对象 e 的 is_positive 属性不是 True
    assert e.is_positive is not True
    # 计算 e2，并断言其 is_positive 属性不是 True
    e2 = 1 - 1/(1 - exp(-1000))
    assert e2.is_positive is not True
    # 计算 e3，并断言其 is_nonzero 属性不是 True
    e3 = -2 + exp(exp(LambertW(log(2)))*LambertW(log(2)))
    assert e3.is_nonzero is not True


# 定义一个名为 test_log_fdiff 的测试函数
def test_log_fdiff():
    # 创建一个符号 x
    x = Symbol('x')
    # 断言调用 log(x) 的 fdiff 方法，参数为 2，会抛出 ArgumentIndexError 异常
    raises(ArgumentIndexError, lambda: log(x).fdiff(2))


# 定义一个名为 test_log_taylor_term 的测试函数
def test_log_taylor_term():
    # 创建一个符号 x
    x = symbols('x')
    # 断言 log(x) 的 Taylor 展开项中，阶数为 0 时应为 x
    assert log(x).taylor_term(0, x) == x
    # 断言 log(x) 的 Taylor 展开项中，阶数为 1 时应为 -x**2/2
    assert log(x).taylor_term(1, x) == -x**2/2
    # 断言 log(x) 的 Taylor 展开项中，阶数为 4 时应为 x**5/5
    assert log(x).taylor_term(4, x) == x**5/5
    # 断言 log(x) 的 Taylor 展开项中，阶数为 -1 时应为 S.Zero
    assert log(x).taylor_term(-1, x) is S.Zero


# 定义一个名为 test_exp_expand_NC 的测试函数，修饰器 @_both_exp_pow 表示该函数在两种符号下都执行
@_both_exp_pow
def test_exp_expand_NC():
    # 创建三个非交换符号 A, B, C
    A, B, C = symbols('A,B,C', commutative=False)

    # 断言 exp(A + B) 的展开结果应为 exp(A + B) 本身
    assert exp(A + B).expand() == exp(A + B)
    # 断言 exp(A + B + C) 的展开结果应为 exp(A + B + C) 本身
    assert exp(A + B + C).expand() == exp(A + B + C)
    # 断言 exp(x + y) 的展开结果应为 exp(x)*exp(y)
    assert exp(x + y).expand() == exp(x)*exp(y)
    # 断言 exp(x + y + z) 的展开结果应为 exp(x)*exp(y)*exp(z)
    assert exp(x + y + z).expand() == exp(x)*exp(y)*exp(z)


# 定义一个名为 test_as_numer_denom 的测试函数，修饰器 @_both_exp_pow 表示该函数在两种符号下都执行
@_both_exp_pow
def test_as_numer_denom():
    # 创建一个负数符号 n
    n = symbols('n', negative=True)
    # 断言 exp(x) 的分子分母表示应为 (exp(x), 1)
    assert exp(x).as_numer_denom() == (exp(x), 1)
    # 断言 exp(-x) 的分子分母表示应为 (1, exp(x))
    assert exp(-x).as_numer_denom() == (1, exp(x))
    # 断言 exp(-2*x) 的分子分母表示应为 (1, exp(2*x))
    assert exp(-2*x).as_numer_denom() == (1, exp(2*x))
    # 断言 exp(-2) 的分子分母表示应为 (1, exp(2))
    assert exp(-2).as_numer_denom() == (1, exp(2))
    # 断言 exp(n) 的分子分母表示应为 (1, exp(-n))
    assert exp(n).as_numer_denom() == (1, exp(-n))
    # 断言 exp(-n) 的分子分母表示应为 (exp(-n), 1)
    assert exp(-n).as_numer_denom() == (exp(-n), 1)
    # 断言 exp(-I*x) 的分子分母表示应为 (1, exp(I*x))
    assert exp(-I*x).as_numer_denom() == (1, exp(I*x))
    # 断言 exp(-I*n) 的分子分母表示应为 (1, exp(I*n))
    assert exp(-I*n).as_numer_denom() == (1, exp(I*n))
    # 断言 exp(-n) 的分子分母表示应为 (exp(-n), 1)
    assert exp(-n).as_numer_denom() == (exp(-n), 1)
    # 创建一个非交换符号 a
    a = symbols('a', commutative=False)
    # 断言 exp(-a) 的分子分母表示应为 (exp(-a), 1)
    assert exp(-a).as_numer_denom() == (exp(-a), 1)


# 定义一个名为 test_polar 的测试函数，修饰器 @_both_exp_pow 表示该函数在两种符号下都执行
@_both_exp_pow
def test_polar():
    # 创建两个极坐标符号 x, y
    x, y = symbols('x y', polar=True)

    # 断言绝对值函数 abs 对 exp_polar(I*4) 的结果为 1
    assert abs(exp_polar(I*4)) == 1
    # 断言绝对值函数 abs 对 exp_polar(0) 的结果为 1
    assert abs(exp_polar(0)) == 1
    # 断言绝对值函数 abs 对 exp_polar(2 + 3*I) 的结果为 exp(2)
    assert abs(exp_polar(2 + 3*I)) == exp(2)
    # 断言 exp_polar(I*10).n() 的结果应为 exp_polar(I*10) 本身
    assert exp_polar(I*10).n() == exp_polar(I*10)

    # 断言 log(exp_polar(z)) 的结果应为 z
    assert log(exp_polar(z)) == z
    # 断言 log(x*y) 展开后应为 log(x) + log(y)
    assert log(x*y).expand() == log(x) + log(y)
    # 断言 log(x**z) 展开后应为 z*log(x)
    assert log(x**z).expand() == z*log(x)

    # 断言 exp_polar(3).exp 的结果为 3
    assert exp_polar(3).exp == 3

    # 比较 exp(1.0*pi*I) 的实部和虚部
    assert (exp_polar(1.0*pi*I).n(n=5)).as_real_imag()[1] >= 0

    # 断言 exp_polar(0) 的 is_rational 属性为 True，解决问题 8008
    assert exp_polar(0).is_rational is True


# 定义一个名为 test_exp_summation 的测试函数
def test_exp_summation():
    # 创建符号 w
    w = symbols("w")
    # 创建符号 m, n, i, j
    m, n, i, j = symbols("m n i j")
    # 创建一个表达式 expr，其中包含 Sum 符号的嵌套
    expr = exp(Sum(w*i, (i, 0, n), (j, 0, m)))
    # 断言 expr 的展开结果应为 Product(exp(w*i), (i, 0, n), (j, 0, m))
    assert expr.expand() == Product(exp(w*i), (i, 0, n), (j, 0, m))


# 定义一个名为 test_log_product 的测试函数
def test_log_product():
    # 从 sympy.abc 中导入符号 n, m
    from sympy.abc import n, m

    # 创建正整数符号 i, j 和正实数符号 x, y，实数符号 z 和符号 w
    # 对给定表达式求对数，其中表达式为 exp(z*i) 的乘积，i 从 0 到 n
    expr = log(Product(exp(z*i), (i, 0, n)))
    # 使用展开操作符对表达式进行展开，然后与预期的求和表达式进行断言比较
    assert expr.expand() == Sum(z*i, (i, 0, n))

    # 对给定表达式求对数，其中表达式为 exp(w*i) 的乘积，i 从 0 到 n
    expr = log(Product(exp(w*i), (i, 0, n)))
    # 断言表达式展开后应与自身相等
    assert expr.expand() == expr
    # 强制展开表达式，并与预期的求和表达式进行断言比较
    assert expr.expand(force=True) == Sum(w*i, (i, 0, n))

    # 对给定表达式求对数，其中表达式为 i**2*abs(j) 的乘积，i 从 1 到 n，j 从 1 到 m
    expr = log(Product(i**2*abs(j), (i, 1, n), (j, 1, m)))
    # 使用展开操作符对表达式进行展开，然后与预期的求和表达式进行断言比较
    assert expr.expand() == Sum(2*log(i) + log(j), (i, 1, n), (j, 1, m))
@XFAIL
def test_log_product_simplify_to_sum():
    # 导入符号变量 n, m
    from sympy.abc import n, m
    # 声明符号变量 i, j，限制其为正整数
    i, j = symbols('i,j', positive=True, integer=True)
    # 声明符号变量 x, y，限制其为正数
    x, y = symbols('x,y', positive=True)
    # 断言简化后对数的乘积化简为和的形式
    assert simplify(log(Product(x**i, (i, 1, n)))) == Sum(i*log(x), (i, 1, n))
    assert simplify(log(Product(x**i*y**j, (i, 1, n), (j, 1, m)))) == \
            Sum(i*log(x) + j*log(y), (i, 1, n), (j, 1, m))


def test_issue_8866():
    # 断言带有 evaluate=False 参数的对数化简结果相同
    assert simplify(log(x, 10, evaluate=False)) == simplify(log(x, 10))
    assert expand_log(log(x, 10, evaluate=False)) == expand_log(log(x, 10))

    # 声明符号变量 y 为正数
    y = Symbol('y', positive=True)
    # 创建一系列对数表达式
    l1 = log(exp(y), exp(10))
    b1 = log(exp(y), exp(5))
    l2 = log(exp(y), exp(10), evaluate=False)
    b2 = log(exp(y), exp(5), evaluate=False)
    # 断言带有 evaluate=False 参数的对数操作结果相同
    assert simplify(log(l1, b1)) == simplify(log(l2, b2))
    assert expand_log(log(l1, b1)) == expand_log(log(l2, b2))


def test_log_expand_factor():
    # 断言对数表达式的展开和因式分解操作
    assert (log(18)/log(3) - 2).expand(factor=True) == log(2)/log(3)
    assert (log(12)/log(2)).expand(factor=True) == log(3)/log(2) + 2
    assert (log(15)/log(3)).expand(factor=True) == 1 + log(5)/log(3)
    assert (log(2)/(-log(12) + log(24))).expand(factor=True) == 1

    assert expand_log(log(12), factor=True) == log(3) + 2*log(2)
    assert expand_log(log(21)/log(7), factor=False) == log(3)/log(7) + 1
    assert expand_log(log(45)/log(5) + log(20), factor=False) == \
        1 + 2*log(3)/log(5) + log(20)
    assert expand_log(log(45)/log(5) + log(26), factor=True) == \
        log(2) + log(13) + (log(5) + 2*log(3))/log(5)


def test_issue_9116():
    # 声明符号变量 n 为正整数
    n = Symbol('n', positive=True, integer=True)
    # 断言对数函数是否非负
    assert log(n).is_nonnegative is True


def test_issue_18473():
    # 断言对数函数在一些特定表达式下的导出项
    assert exp(x*log(cos(1/x))).as_leading_term(x) == S.NaN
    assert exp(x*log(tan(1/x))).as_leading_term(x) == S.NaN
    assert log(cos(1/x)).as_leading_term(x) == S.NaN
    assert log(tan(1/x)).as_leading_term(x) == S.NaN
    assert log(cos(1/x) + 2).as_leading_term(x) == AccumBounds(0, log(3))
    assert exp(x*log(cos(1/x) + 2)).as_leading_term(x) == 1
    assert log(cos(1/x) - 2).as_leading_term(x) == S.NaN
    assert exp(x*log(cos(1/x) - 2)).as_leading_term(x) == S.NaN
    assert log(cos(1/x) + 1).as_leading_term(x) == AccumBounds(-oo, log(2))
    assert exp(x*log(cos(1/x) + 1)).as_leading_term(x) == AccumBounds(0, 1)
    assert log(sin(1/x)**2).as_leading_term(x) == AccumBounds(-oo, 0)
    assert exp(x*log(sin(1/x)**2)).as_leading_term(x) == AccumBounds(0, 1)
    assert log(tan(1/x)**2).as_leading_term(x) == AccumBounds(-oo, oo)
    assert exp(2*x*(log(tan(1/x)**2))).as_leading_term(x) == AccumBounds(0, oo)
```