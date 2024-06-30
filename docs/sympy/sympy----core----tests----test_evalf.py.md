# `D:\src\scipysrc\sympy\sympy\core\tests\test_evalf.py`

```
import math
从 math 模块导入基本数学函数和常数

from sympy.concrete.products import (Product, product)
从 sympy.concrete.products 模块导入 Product 类和 product 函数

from sympy.concrete.summations import Sum
从 sympy.concrete.summations 模块导入 Sum 类

from sympy.core.add import Add
从 sympy.core.add 模块导入 Add 类

from sympy.core.evalf import N
从 sympy.core.evalf 模块导入 N 函数

from sympy.core.function import (Function, nfloat)
从 sympy.core.function 模块导入 Function 类和 nfloat 函数

from sympy.core.mul import Mul
从 sympy.core.mul 模块导入 Mul 类

from sympy.core import (GoldenRatio)
从 sympy.core 模块导入 GoldenRatio 常数

from sympy.core.numbers import (AlgebraicNumber, E, Float, I, Rational,
                                oo, zoo, nan, pi)
从 sympy.core.numbers 模块导入各种数学常数和类型，如代数数、Euler数、浮点数、虚数单位、有理数、无穷大 oo、无穷小 zoo、NaN、圆周率 pi

from sympy.core.power import Pow
从 sympy.core.power 模块导入 Pow 类

from sympy.core.relational import Eq
从 sympy.core.relational 模块导入 Eq 类

from sympy.core.singleton import S
从 sympy.core.singleton 模块导入 S 单例

from sympy.core.symbol import Symbol
从 sympy.core.symbol 模块导入 Symbol 类

from sympy.core.sympify import sympify
从 sympy.core.sympify 模块导入 sympify 函数

from sympy.functions.combinatorial.factorials import factorial
从 sympy.functions.combinatorial.factorials 模块导入 factorial 函数

from sympy.functions.combinatorial.numbers import fibonacci
从 sympy.functions.combinatorial.numbers 模块导入 fibonacci 函数

from sympy.functions.elementary.complexes import (Abs, re, im)
从 sympy.functions.elementary.complexes 模块导入复数相关函数，如绝对值 Abs、实部 re、虚部 im

from sympy.functions.elementary.exponential import (exp, log)
从 sympy.functions.elementary.exponential 模块导入指数和对数函数，如 exp、log

from sympy.functions.elementary.hyperbolic import (acosh, cosh)
从 sympy.functions.elementary.hyperbolic 模块导入双曲函数，如反双曲余弦 acosh、双曲余弦 cosh

from sympy.functions.elementary.integers import (ceiling, floor)
从 sympy.functions.elementary.integers 模块导入取整函数，如向上取整 ceiling、向下取整 floor

from sympy.functions.elementary.miscellaneous import (Max, sqrt)
从 sympy.functions.elementary.miscellaneous 模块导入杂项函数，如最大值 Max、平方根 sqrt

from sympy.functions.elementary.trigonometric import (acos, atan, cos, sin, tan)
从 sympy.functions.elementary.trigonometric 模块导入三角函数，如反余弦 acos、反正切 atan、余弦 cos、正弦 sin、正切 tan

from sympy.integrals.integrals import (Integral, integrate)
从 sympy.integrals.integrals 模块导入积分相关类和函数，如积分 Integral、积分计算函数 integrate

from sympy.polys.polytools import factor
从 sympy.polys.polytools 模块导入多项式工具函数 factor

from sympy.polys.rootoftools import CRootOf
从 sympy.polys.rootoftools 模块导入 CRootOf 类

from sympy.polys.specialpolys import cyclotomic_poly
从 sympy.polys.specialpolys 模块导入周期多项式 cyclotomic_poly

from sympy.printing import srepr
从 sympy.printing 模块导入 srepr 函数

from sympy.printing.str import sstr
从 sympy.printing.str 模块导入 sstr 函数

from sympy.simplify.simplify import simplify
从 sympy.simplify.simplify 模块导入 simplify 函数

from sympy.core.numbers import comp
从 sympy.core.numbers 模块导入 comp 类

from sympy.core.evalf import (complex_accuracy, PrecisionExhausted,
                              scaled_zero, get_integer_part, as_mpmath, evalf, _evalf_with_bounded_error)
从 sympy.core.evalf 模块导入数值计算相关函数和异常类

from mpmath import inf, ninf, make_mpc
从 mpmath 模块导入无穷大 inf、无穷小 ninf 和 make_mpc 函数

from mpmath.libmp.libmpf import from_float, fzero
从 mpmath.libmp.libmpf 模块导入 from_float 和 fzero 函数

from sympy.core.expr import unchanged
从 sympy.core.expr 模块导入 unchanged 函数

from sympy.testing.pytest import raises, XFAIL
从 sympy.testing.pytest 模块导入 raises 和 XFAIL 函数

from sympy.abc import n, x, y
从 sympy.abc 模块导入符号变量 n、x、y

def NS(e, n=15, **options):
    使用 sympify 函数将输入的表达式 e 转换为符号表达式，然后调用 evalf 函数进行数值计算，精度为 n
    return sstr(sympify(e).evalf(n, **options), full_prec=True)

def test_evalf_helpers():
    定义测试函数 test_evalf_helpers
    from mpmath.libmp import finf
    导入 mpmath.libmp 模块中的 finf 常数
    assert complex_accuracy((from_float(2.0), None, 35, None)) == 35
    断言：使用 complex_accuracy 函数计算给定参数的复杂精度，预期结果为 35
    assert complex_accuracy((from_float(2.0), from_float(10.0), 35, 100)) == 37
    断言：使用 complex_accuracy 函数计算给定参数的复杂精度，预期结果为 37
    assert complex_accuracy(
        (from_float(2.0), from_float(1000.0), 35, 100)) == 43
    断言：使用 complex_accuracy 函数计算给定参数的复杂精度，预期结果为 43
    assert complex_accuracy((from_float(2.0), from_float(10.0), 100, 35)) == 35
    断言：使用 complex_accuracy 函数计算给定参数的复杂精度，预期结果为 35
    assert complex_accuracy(
        (from_float(2.0), from_float(1000.0), 100, 35)) == 35
    断言：使用 complex_accuracy 函数计算给定参数的复杂精度，预期结果为 35
    assert complex_accuracy(finf) == math.inf
    断言：使用 complex_accuracy 函数计算给定参数的复杂精度，预期结果为 math.inf
    assert complex_accuracy(zoo) == math.inf
    断言：使用 complex_accuracy 函数计算给定参数的复杂精度，预期结果为 math.inf
    raises(ValueError, lambda: get_integer_part(zoo, 1, {}))
    断言：验证调用 get_integer_part 函数时抛出 ValueError 异常，期望捕获到该异常

def test_evalf_basic():
    定义测试函数 test_evalf_basic
    assert NS('pi', 15) == '3.14159265358979'
    断言：调用 NS 函数计算圆周率 pi 的数值表达式，精度为 15，预期结果为 '3.14159265358979'
    assert NS('2/3', 10) == '0.6666666667'
    断言：调用 NS 函数计算分数 2/3 的数值表达式，精度为 10，预期结果为 '0.6666666667'
    assert NS('355/113-pi', 6) == '2.66764e-7'
    断言：调用 NS 函数计算表达式 355/113 - pi 的数值表达式，精度为 6，预期结果为 '
    # 断言：使用 NS 函数测试 pi 的 10 的 20 次方的字符串表示，保留 10 位有效数字，断言结果是否等于指定字符串。
    assert NS('pi**(10**20)', 10) == '1.339148777e+49714987269413385435'
    
    # 断言：使用 NS 函数测试 pi 的 10 的 100 次方的数值表示，保留 10 位有效数字，断言结果是否等于指定数值的字符串表示。
    assert NS(pi**(10**100), 10) == ('4.946362032e+4971498726941338543512682882'
                                      '9089887365167832438044244613405349992494711208'
                                      '95526746555473864642912223')
    
    # 断言：使用 NS 函数测试 2 的 1 除以 10 的 50 次方的数值表示，保留 15 位有效数字，断言结果是否等于指定字符串。
    assert NS('2**(1/10**50)', 15) == '1.00000000000000'
    
    # 断言：使用 NS 函数测试 2 的 1 除以 10 的 50 次方再减去 1 的数值表示，保留 15 位有效数字，断言结果是否等于指定字符串。
    assert NS('2**(1/10**50)-1', 15) == '6.93147180559945e-51'
# Evaluation of Rump's ill-conditioned polynomial

# 定义一个测试函数，用于评估 Rump 的病态多项式求值
def test_evalf_rump():
    # 定义 Rump 的多项式
    a = 1335*y**6/4 + x**2*(11*x**2*y**2 - y**6 - 121*y**4 - 2) + 11*y**8/2 + x/(2*y)
    # 使用 SymPy 的 NS 函数对表达式进行数值求解，指定精度为 15，同时替换 x 和 y 的值
    assert NS(a, 15, subs={x: 77617, y: 33096}) == '-0.827396059946821'


# 测试复数运算

# 测试复数表达式 '2*sqrt(pi)*I' 的数值化结果，精度为 10
assert NS('2*sqrt(pi)*I', 10) == '3.544907702*I'

# 测试复数表达式 '3+3*I' 的数值化结果，精度为 15
assert NS('3+3*I', 15) == '3.00000000000000 + 3.00000000000000*I'

# 测试复数表达式 'E+pi*I' 的数值化结果，精度为 15
assert NS('E+pi*I', 15) == '2.71828182845905 + 3.14159265358979*I'

# 测试复数表达式 'pi * (3+4*I)' 的数值化结果，精度为 15
assert NS('pi * (3+4*I)', 15) == '9.42477796076938 + 12.5663706143592*I'

# 测试复数表达式 'I*(2+I)' 的数值化结果，精度为 15
assert NS('I*(2+I)', 15) == '-1.00000000000000 + 2.00000000000000*I'


# 标记为预期失败的复数运算测试函数

# 定义一个预期失败的测试函数，测试复数表达式 '(pi+E*I)*(E+pi*I)' 的数值化结果
@XFAIL
def test_evalf_complex_bug():
    assert NS('(pi+E*I)*(E+pi*I)', 15) in ('0.e-15 + 17.25866050002*I',
              '0.e-17 + 17.25866050002*I', '-0.e-17 + 17.25866050002*I')


# 测试复数幂运算

# 测试复数表达式 '(E+pi*I)**100000000000000000' 的数值化结果
def test_evalf_complex_powers():
    assert NS('(E+pi*I)**100000000000000000') == \
        '-3.58896782867793e+61850354284995199 + 4.58581754997159e+61850354284995199*I'

    # XXX: 如果 SymPy 引入了 a+a*I 简化，重写此断言
    #assert NS('(pi + pi*I)**2') in ('0.e-15 + 19.7392088021787*I', '0.e-16 + 19.7392088021787*I')

    # 测试复数表达式 '(pi + pi*I)**2' 的数值化结果，开启 chop 参数
    assert NS('(pi + pi*I)**2', chop=True) == '19.7392088021787*I'

    # 测试复数表达式 '(pi + 1/10**8 + pi*I)**2' 的数值化结果
    assert NS('(pi + 1/10**8 + pi*I)**2') == '6.2831853e-8 + 19.7392088650106*I'

    # 测试复数表达式 '(pi + 1/10**12 + pi*I)**2' 的数值化结果
    assert NS('(pi + 1/10**12 + pi*I)**2') == '6.283e-12 + 19.7392088021850*I'

    # 测试复数表达式 '(pi + pi*I)**4' 的数值化结果，开启 chop 参数
    assert NS('(pi + pi*I)**4', chop=True) == '-389.636364136010'

    # 测试复数表达式 '(pi + 1/10**8 + pi*I)**4' 的数值化结果
    assert NS('(pi + 1/10**8 + pi*I)**4') == '-389.636366616512 + 2.4805021e-6*I'

    # 测试复数表达式 '(pi + 1/10**12 + pi*I)**4' 的数值化结果
    assert NS('(pi + 1/10**12 + pi*I)**4') == '-389.636364136258 + 2.481e-10*I'

    # 测试复数表达式 '(10000*pi + 10000*pi*I)**4' 的数值化结果，开启 chop 参数
    assert NS('(10000*pi + 10000*pi*I)**4', chop=True) == '-3.89636364136010e+18'


# 标记为预期失败的复数幂运算测试函数

# 定义一个预期失败的测试函数，测试复数表达式 '(pi + pi*I)**4' 的数值化结果
@XFAIL
def test_evalf_complex_powers_bug():
    assert NS('(pi + pi*I)**4') == '-389.63636413601 + 0.e-14*I'


# 测试指数运算

# 测试表达式 'sqrt(-pi)' 的数值化结果
def test_evalf_exponentiation():
    assert NS(sqrt(-pi)) == '1.77245385090552*I'

    # 测试表达式 'pi**I' 的数值化结果，指定 evaluate=False
    assert NS(Pow(pi*I, Rational(1, 2), evaluate=False)) == '1.25331413731550 + 1.25331413731550*I'

    # 测试表达式 'pi**I' 的数值化结果
    assert NS(pi**I) == '0.413292116101594 + 0.910598499212615*I'

    # 测试表达式 'pi**(E + I/3)' 的数值化结果
    assert NS(pi**(E + I/3)) == '20.8438653991931 + 8.36343473930031*I'

    # 测试表达式 '(pi + I/3)**(E + I/3)' 的数值化结果
    assert NS((pi + I/3)**(E + I/3)) == '17.2442906093590 + 13.6839376767037*I'

    # 测试表达式 'exp(pi)' 的数值化结果
    assert NS(exp(pi)) == '23.1406926327793'

    # 测试表达式 'exp(pi + E*I)' 的数值化结果
    assert NS(exp(pi + E*I)) == '-21.0981542849657 + 9.50576358282422*I'

    # 测试表达式 'pi**pi' 的数值化结果
    assert NS(pi**pi) == '36.4621596072079'

    # 测试表达式 '(-pi)**pi' 的数值化结果
    assert NS((-pi)**pi) == '-32.9138577418939 - 15.6897116534332*I'

    # 测试表达式 '(-pi)**(-pi)' 的数值化结果
    assert NS((-pi)**(-pi)) == '-0.0247567717232697 + 0.0118013091280262*I'


# Smith 的例子，测试复数取消项

# 测试复数表达式的数值化结果
def test_evalf_complex_cancellation():
    A = Rational('63287/100000')
    B = Rational('52498/100000')
    C = Rational('69301/100000')
    D = Rational('83542/100000')
    F = Rational('2231321613/2500000000')

    # XXX: 实部中返回的尾数位数可能因实现方式而变化。重要的是返回的尾数位数是
    # change with the implementation. What matters is that the returned digits are
    # 断言：验证数学表达式的计算结果是否符合预期值
    # 使用 SymPy 的 NS 函数对复数表达式进行数值计算和格式化输出
    assert NS((A + B*I)*(C + D*I), 6) == '6.44710e-6 + 0.892529*I'
    assert NS((A + B*I)*(C + D*I), 10) == '6.447100000e-6 + 0.8925286452*I'
    # 断言：验证复数表达式减去 F*I 的计算结果是否在指定的可能输出范围内
    assert NS((A + B*I)*(C + D*I) - F*I, 5) in ('6.4471e-6 + 0.e-14*I', '6.4471e-6 - 0.e-14*I')
# 定义一个函数用于测试对数函数的精确计算
def test_evalf_logs():
    # 检查对 log(3+pi*I) 进行数值计算的结果是否等于指定的复数值
    assert NS("log(3+pi*I)", 15) == '1.46877619736226 + 0.808448792630022*I'
    # 检查对 log(pi*I) 进行数值计算的结果是否等于指定的复数值
    assert NS("log(pi*I)", 15) == '1.14472988584940 + 1.57079632679490*I'
    # 检查对 log(-1 + 0.00001) 进行数值计算的结果是否等于指定的复数值
    assert NS('log(-1 + 0.00001)', 2) == '-1.0e-5 + 3.1*I'
    # 检查对 log(100, 10, evaluate=False) 进行数值计算的结果是否等于指定的数值
    assert NS('log(100, 10, evaluate=False)', 15) == '2.00000000000000'
    # 检查对 -2*I*log(-(-1)**(S(1)/9)) 进行数值计算的结果是否等于指定的复数值
    assert NS('-2*I*log(-(-1)**(S(1)/9))', 15) == '-5.58505360638185'


# 定义一个函数用于测试三角函数的精确计算
def test_evalf_trig():
    # 检查对 sin(1) 进行数值计算的结果是否等于指定的数值
    assert NS('sin(1)', 15) == '0.841470984807897'
    # 检查对 cos(1) 进行数值计算的结果是否等于指定的数值
    assert NS('cos(1)', 15) == '0.540302305868140'
    # 检查对 sin(10**-6) 进行数值计算的结果是否等于指定的数值
    assert NS('sin(10**-6)', 15) == '9.99999999999833e-7'
    # 检查对 cos(10**-6) 进行数值计算的结果是否等于指定的数值
    assert NS('cos(10**-6)', 15) == '0.999999999999500'
    # 检查对 sin(E*10**100) 进行数值计算的结果是否等于指定的数值
    assert NS('sin(E*10**100)', 15) == '0.409160531722613'
    # 检查对 sin(exp(pi*sqrt(163))*pi) 进行数值计算的结果是否等于指定的数值
    assert NS(sin(exp(pi*sqrt(163))*pi), 15) == '-2.35596641936785e-12'
    # 检查对 sin(pi*10**100 + Rational(7, 10**5), evaluate=False) 进行数值计算的结果是否等于指定的数值
    assert NS(sin(pi*10**100 + Rational(7, 10**5), evaluate=False), 15, maxn=120) == \
        '6.99999999428333e-5'
    # 检查对 sin(Rational(7, 10**5), evaluate=False) 进行数值计算的结果是否等于指定的数值
    assert NS(sin(Rational(7, 10**5), evaluate=False), 15) == \
        '6.99999999428333e-5'


# 定义一个函数用于测试接近整数的数值计算
def test_evalf_near_integers():
    # Binet's formula
    f = lambda n: ((1 + sqrt(5))**n)/(2**n * sqrt(5))
    # 检查 Binet's formula 计算的结果与 Fibonacci 数列的差异是否在指定精度内
    assert NS(f(5000) - fibonacci(5000), 10, maxn=1500) == '5.156009964e-1046'
    # 检查 sin(2017*2**(1/5)) 进行数值计算的结果是否等于指定的数值
    assert NS('sin(2017*2**(1/5))', 15) == '-1.00000000000000'
    # 检查 sin(2017*2**(1/5)) 进行数值计算的结果是否等于指定的数值
    assert NS('sin(2017*2**(1/5))', 20) == '-0.99999999999999997857'
    # 检查 1+sin(2017*2**(1/5)) 进行数值计算的结果是否等于指定的数值
    assert NS('1+sin(2017*2**(1/5))', 15) == '2.14322287389390e-17'
    # 检查 45 - 613*E/37 + 35/991 进行数值计算的结果是否等于指定的数值
    assert NS('45 - 613*E/37 + 35/991', 15) == '6.03764498766326e-11'


# 定义一个函数用于测试与 Ramanujan 相关的数值计算
def test_evalf_ramanujan():
    # 检查 exp(pi*sqrt(163)) - 640320**3 - 744 进行数值计算的结果是否等于指定的数值
    assert NS(exp(pi*sqrt(163)) - 640320**3 - 744, 10) == '-7.499274028e-13'
    # 检查 1 - A - B + C 进行数值计算的结果是否等于指定的数值，其中 A, B, C 是给定的数学表达式
    A = 262537412640768744*exp(-pi*sqrt(163))
    B = 196884*exp(-2*pi*sqrt(163))
    C = 103378831900730205293632*exp(-3*pi*sqrt(163))
    assert NS(1 - A - B + C, 10) == '1.613679005e-59'


# 定义一个函数用于测试已知的问题或 bug 的处理
def test_evalf_bugs():
    # 检查 sin(1) + exp(-10**10) 进行数值计算的结果是否与 sin(1) 的数值计算结果相等
    assert NS(sin(1) + exp(-10**10), 10) == NS(sin(1), 10)
    # 检查 exp(10**10) + sin(1) 进行数值计算的结果是否与 exp(10**10) 的数值计算结果相等
    assert NS(exp(10**10) + sin(1), 10) == NS(exp(10**10), 10)
    # 检查 expand_log(log(1+1/10**50)) 进行数值计算的结果是否等于指定的数值
    assert NS('expand_log(log(1+1/10**50))', 20) == '1.0000000000000000000e-50'
    # 检查 log(10**100,10) 进行数值计算的结果是否等于指定的数值
    assert NS('log(10**100,10)', 10) == '100.0000000'
    # 检查 log(2) 进行数值计算的结果是否等于指定的数值
    assert NS('log(2)', 10) == '0.6931471806'
    # 检查 (sin(x)-x)/x**3 进行数值计算的结果是否等于指定的数值，其中 x 被替换为 1/10**50
    assert NS('(sin(x)-x)/x**3', 15, subs={x: '1/10**50'}) == '-0.166666666666667'
    # 检查 sin(1) + Rational(1, 10**100)*I 进行数值计算的结果是否等于指定的复数值
    assert NS(sin(1) + Rational(1, 10**100)*I, 15) == '0.841470984807897 + 1.00000000000000e-100*I'
    # 检查 x.evalf() 的结果是否等于 x 本身
    # issue 4758 (2/2): With the bug present, this still only fails if the
    # terms are in the order given here. This is not generally the case,
    # because the order depends on the hashes of the terms.
    # 断言：验证表达式是否符合预期结果
    assert NS(20 - 5008329267844*n**25 - 477638700*n**37 - 19*n,
              subs={n: .01}) == '19.8100000000000'
    
    # 断言：验证数值计算结果的精度
    assert NS(((x - 1)*(1 - x)**1000).n()
              ) == '(1.00000000000000 - x)**1000*(x - 1.00000000000000)'
    
    # 断言：验证数值计算结果的精度
    assert NS((-x).n()) == '-x'
    
    # 断言：验证数值计算结果的精度
    assert NS((-2*x).n()) == '-2.00000000000000*x'
    
    # 断言：验证数值计算结果的精度
    assert NS((-2*x*y).n()) == '-2.00000000000000*x*y'
    
    # 断言：验证复数情况下余弦函数的数值计算结果的精度
    assert cos(x).n(subs={x: 1+I}) == cos(x).subs(x, 1+I).n()
    
    # issue 6660. Also NaN != mpmath.nan
    # In this order:
    # 0*nan, 0/nan, 0*inf, 0/inf
    # 0+nan, 0-nan, 0+inf, 0-inf
    # >>> n = Some Number
    # n*nan, n/nan, n*inf, n/inf
    # n+nan, n-nan, n+inf, n-inf
    # 断言：验证数值计算结果的精度
    assert (0*E**(oo)).n() is S.NaN
    
    # 断言：验证数值计算结果的精度
    assert (0/E**(oo)).n() is S.Zero
    
    # 断言：验证数值计算结果的精度
    assert (0+E**(oo)).n() is S.Infinity
    
    # 断言：验证数值计算结果的精度
    assert (0-E**(oo)).n() is S.NegativeInfinity
    
    # 断言：验证数值计算结果的精度
    assert (5*E**(oo)).n() is S.Infinity
    
    # 断言：验证数值计算结果的精度
    assert (5/E**(oo)).n() is S.Zero
    
    # 断言：验证数值计算结果的精度
    assert (5+E**(oo)).n() is S.Infinity
    
    # 断言：验证数值计算结果的精度
    assert (5-E**(oo)).n() is S.NegativeInfinity
    
    # issue 7416
    # 断言：验证数值计算结果的精度
    assert as_mpmath(0.0, 10, {'chop': True}) == 0
    
    # issue 5412
    # 断言：验证数值计算结果的精度
    assert ((oo*I).n() == S.Infinity*I)
    
    # 断言：验证数值计算结果的精度
    assert ((oo+oo*I).n() == S.Infinity + S.Infinity*I)
    
    # issue 11518
    # 断言：验证数值计算结果的精度
    assert NS(2*x**2.5, 5) == '2.0000*x**2.5000'
    
    # issue 13076
    # 断言：验证数值计算结果的精度
    assert NS(Mul(Max(0, y), x, evaluate=False).evalf()) == 'x*Max(0, y)'
    
    # issue 18516
    # 断言：验证数值计算结果的精度
    assert NS(log(S(3273390607896141870013189696827599152216642046043064789483291368096133796404674554883270092325904157150886684127560071009217256545885393053328527589376)/36360291795869936842385267079543319118023385026001623040346035832580600191583895484198508262979388783308179702534403855752855931517013066142992430916562025780021771247847643450125342836565813209972590371590152578728008385990139795377610001).evalf(15, chop=True)) == '-oo'
# 定义一个测试函数，用于测试 evalf_integer_parts 的功能
def test_evalf_integer_parts():
    # 使用 evaluate=False 参数来禁止对数函数的求值
    a = floor(log(8)/log(2) - exp(-1000), evaluate=False)
    # 使用 evaluate=False 参数来禁止对数函数的求值
    b = floor(log(8)/log(2), evaluate=False)
    # 断言 a 的数值求值结果为 3.0
    assert a.evalf() == 3.0
    # 断言 b 的数值求值结果为 3.0
    assert b.evalf() == 3.0
    # 断言 10*(sin(1)**2 + cos(1)**2) 的天花板值为 10
    # 在某些情况下可能会相等，但不能绝对保证相等
    assert ceiling(10*(sin(1)**2 + cos(1)**2)) == 10

    # 断言计算阶乘(50) / 自然常数 e 的地板值，精确到小数点后 70 位
    assert int(floor(factorial(50)/E, evaluate=False).evalf(70)) == \
        int(11188719610782480504630258070757734324011354208865721592720336800)
    # 断言计算阶乘(50) / 自然常数 e 的天花板值，精确到小数点后 70 位
    assert int(ceiling(factorial(50)/E, evaluate=False).evalf(70)) == \
        int(11188719610782480504630258070757734324011354208865721592720336801)
    # 断言计算黄金比例的 999 次幂除以根号 5 的地板值加上 1/2，精确到小数点后 1000 位
    assert int(floor(GoldenRatio**999 / sqrt(5) + S.Half)
               .evalf(1000)) == fibonacci(999)
    # 断言计算黄金比例的 1000 次幂除以根号 5 的地板值加上 1/2，精确到小数点后 1000 位
    assert int(floor(GoldenRatio**1000 / sqrt(5) + S.Half)
               .evalf(1000)) == fibonacci(1000)

    # 断言对变量 x 进行天花板运算，其中 x 被赋值为 3 时的结果为 3.0
    assert ceiling(x).evalf(subs={x: 3}) == 3.0
    # 断言对变量 x 进行天花板运算，其中 x 被赋值为 3*I 时的结果为 3.0*I
    assert ceiling(x).evalf(subs={x: 3*I}) == 3.0*I
    # 断言对变量 x 进行天花板运算，其中 x 被赋值为 2 + 3*I 时的结果为 2.0 + 3.0*I
    assert ceiling(x).evalf(subs={x: 2 + 3*I}) == 2.0 + 3.0*I
    # 断言对变量 x 进行天花板运算，其中 x 被赋值为 3. 时的结果为 3.0
    assert ceiling(x).evalf(subs={x: 3.}) == 3.0
    # 断言对变量 x 进行天花板运算，其中 x 被赋值为 3.*I 时的结果为 3.0*I
    assert ceiling(x).evalf(subs={x: 3.*I}) == 3.0*I
    # 断言对变量 x 进行天花板运算，其中 x 被赋值为 2. + 3*I 时的结果为 2.0 + 3.0*I
    assert ceiling(x).evalf(subs={x: 2. + 3*I}) == 2.0 + 3.0*I

    # 断言计算 floor(1.5, evaluate=False)+1/9 的浮点数结果等于 1 + 1/9
    assert float((floor(1.5, evaluate=False)+1/9).evalf()) == 1 + 1/9
    # 断言计算 floor(0.5, evaluate=False)+20 的浮点数结果等于 20
    assert float((floor(0.5, evaluate=False)+20).evalf()) == 20

    # issue 19991
    # 定义大整数 n 和 r，然后进行一个 floor 运算的断言
    n = 1169809367327212570704813632106852886389036911
    r = 744723773141314414542111064094745678855643068
    assert floor(n / (pi / 2)) == r

    # 断言 floor(80782 * sqrt(2)) 的结果为 114242
    assert floor(80782 * sqrt(2)) == 114242

    # issue 20076
    # 断言 260515 - floor(260515/pi + 1/2) * pi 的结果等于 atan(tan(260515))
    assert 260515 - floor(260515/pi + 1/2) * pi == atan(tan(260515))

    # 断言对变量 x 进行地板运算，其中 x 被赋值为 sqrt(2) 时的结果为 1.0
    assert floor(x).evalf(subs={x: sqrt(2)}) == 1.0


# 定义一个测试函数，用于测试 evalf_trig_zero_detection 的功能
def test_evalf_trig_zero_detection():
    # 计算 sin(160*pi) 的数值，使用 evaluate=False 禁止求值
    a = sin(160*pi, evaluate=False)
    # 对 a 进行数值求值，精度最大为 100
    t = a.evalf(maxn=100)
    # 断言 t 的绝对值小于 1e-100
    assert abs(t) < 1e-100
    # 断言 t 的精度小于 2
    assert t._prec < 2
    # 断言对 a 进行数值求值，启用 chop=True 选项，结果为 0
    assert a.evalf(chop=True) == 0
    # 断言在严格模式下对 a 进行数值求值会引发 PrecisionExhausted 异常
    raises(PrecisionExhausted, lambda: a.evalf(strict=True))


# 定义一个测试函数，用于测试 evalf_sum 的功能
def test_evalf_sum():
    # 断言对 Sum(n,(n,1,2)) 进行数值求值结果为 3.0
    assert Sum(n,(n,1,2)).evalf() == 3.
    # 断言对 Sum(n,(n,1,2)) 进行求值后再进行数值求值结果为 3.0
    assert Sum(n,(n,1,2)).doit().evalf() == 3.
    # 断言对 Sum(1/n,(n,1,2)) 进行数值求值结果为 1.5
    # 注意这是一个收敛的级数求和
    assert Sum(1/n,(n,1,2)).evalf() == 1.5

    # issue 8219
    # 断言对级数 Sum(E/factorial(n), (n, 0, oo)) 进行数值求值结果为 E^2 的数值
    assert Sum(E/factorial(n), (n, 0, oo)).evalf() == (E*E).evalf()
    # issue 8254
    # 断言对级数 Sum(2**n*n/factorial(n), (n, 0, oo)) 进行数值求值结果为 2*E^2 的数值
    assert Sum(2**n*n/factorial(n), (n, 0, oo)).evalf() == (2*E*E).evalf()
    # issue 8411
    # 创建一个级数对象 s，然后断言对 s 进行数值求值和直接求值后的结果相等
    s = Sum(1/x**2, (x, 100, oo))
    assert s.n() == s.doit().n()


# 定义一个测试函数，用于测试 evalf_div
def test_evalf_product():
    # 测试 Product 符号计算的正确性，期望结果为 10 的阶乘
    assert Product(n, (n, 1, 10)).evalf() == 3628800.
    # 测试复合 Product 符号计算结果近似为指定值
    assert comp(Product(1 - S.Half**2/n**2, (n, 1, oo)).n(5), 0.63662)
    # 测试负数范围的 Product 符号计算结果为 0
    assert Product(n, (n, -1, 3)).evalf() == 0


def test_evalf_py_methods():
    # 测试浮点数加法的精确性
    assert abs(float(pi + 1) - 4.1415926535897932) < 1e-10
    # 测试复数加法的精确性
    assert abs(complex(pi + 1) - 4.1415926535897932) < 1e-10
    # 测试复数加法包含虚数部分的精确性
    assert abs(complex(pi + E*I) - (3.1415926535897931 + 2.7182818284590451j)) < 1e-10
    # 测试在变量 x 上使用 float 方法，期望抛出 TypeError
    raises(TypeError, lambda: float(pi + x))


def test_evalf_power_subs_bugs():
    # 测试在 x = 0 时的幂运算结果
    assert (x**2).evalf(subs={x: 0}) == 0
    assert sqrt(x).evalf(subs={x: 0}) == 0
    assert (x**Rational(2, 3)).evalf(subs={x: 0}) == 0
    assert (x**x).evalf(subs={x: 0}) == 1.0
    assert (3**x).evalf(subs={x: 0}) == 1.0
    assert exp(x).evalf(subs={x: 0}) == 1.0
    assert ((2 + I)**x).evalf(subs={x: 0}) == 1.0
    assert (0**x).evalf(subs={x: 0}) == 1.0


def test_evalf_arguments():
    # 测试在 evalf 方法中使用不存在的方法选项，期望抛出 TypeError
    raises(TypeError, lambda: pi.evalf(method="garbage"))


def test_implemented_function_evalf():
    # 测试 implemented_function 对象的基本功能
    from sympy.utilities.lambdify import implemented_function
    f = Function('f')
    f = implemented_function(f, lambda x: x + 1)
    assert str(f(x)) == "f(x)"
    assert str(f(2)) == "f(2)"
    assert f(2).evalf() == 3.0
    assert f(x).evalf() == f(x)
    # 清理 implemented_function 对象的缓存，以避免影响其他测试
    del f._imp_     # XXX: due to caching _imp_ would influence all other tests


def test_evaluate_false():
    # 测试在 evaluate=False 选项下的加法、乘法、乘方运算
    for no in [0, False]:
        assert Add(3, 2, evaluate=no).is_Add
        assert Mul(3, 2, evaluate=no).is_Mul
        assert Pow(3, 2, evaluate=no).is_Pow
    # 测试带 evaluate=True 的幂运算，期望结果为 0
    assert Pow(y, 2, evaluate=True) - Pow(y, 2, evaluate=True) == 0


def test_evalf_relational():
    # 测试在 evalf 后保持不变的关系运算
    assert Eq(x/5, y/10).evalf() == Eq(0.2*x, 0.1*y)
    # 如果第一个断言失败，则使用不会失败的替代断言
    assert unchanged(Eq, (3 - I)**2/2 + I, 0)
    # 测试为假的关系运算
    assert Eq((3 - I)**2/2 + I, 0).n() is S.false
    # 测试 nfloat 对关系运算的结果
    assert nfloat(Eq((3 - I)**2 + I, 0)) == S.false


def test_issue_5486():
    # 测试不是函数的 cos(sqrt(0.5 + I)) 的 evalf 结果是否为 Function
    assert not cos(sqrt(0.5 + I)).n().is_Function


def test_issue_5486_bug():
    # 测试在 Expr 类中通过 mpmath 和 sympy.core.numbers 的处理
    from sympy.core.expr import Expr
    from sympy.core.numbers import I
    assert abs(Expr._from_mpmath(I._to_mpmath(15), 15) - I) < 1.0e-15


def test_bugs():
    # 测试 re((1 + I)**2) 的结果的绝对值是否小于 1e-15
    from sympy.functions.elementary.complexes import (polar_lift, re)
    assert abs(re((1 + I)**2)) < 1e-15
    # 测试 polar_lift(0) 的 evalf 结果是否为 0
    assert abs(polar_lift(0)).n() == 0


def test_subs():
    # 测试通过 NS 函数进行符号表达式求值
    assert NS('besseli(-x, y) - besseli(x, y)', subs={x: 3.5, y: 20.0}) == \
        '-4.92535585957223e-10'
    assert NS('Piecewise((x, x>0)) + Piecewise((1-x, x>0))', subs={x: 0.1}) == \
        '1.00000000000000'
    # 测试在 evalf(subs=(x, 1)) 时抛出 TypeError
    raises(TypeError, lambda: x.evalf(subs=(x, 1)))


def test_issue_4956_5204():
    # issue 4956
    # 测试符号表达式中复杂计算的精确性
    v = S('''(-27*12**(1/3)*sqrt(31)*I +
    27*2**(2/3)*3**(1/3)*sqrt(31)*I)/(-2511*2**(2/3)*3**(1/3) +
    # 定义一个复杂的数学表达式
    (29*18**(1/3) + 9*2**(1/3)*3**(2/3)*sqrt(31)*I +
    87*2**(1/3)*3**(1/6)*I)**2)
    # 断言：对该表达式的数值求精确数值字符串，期望结果是 '0.e-118 - 0.e-118*I'

    # issue 5204
    # 定义另一个复杂的数学表达式，该表达式涉及多个数学运算和常数
    v = S('''-(357587765856 + 18873261792*249**(1/2) + 56619785376*I*83**(1/2) +
    108755765856*I*3**(1/2) + 41281887168*6**(1/3)*(1422 +
    54*249**(1/2))**(1/3) - 1239810624*6**(1/3)*249**(1/2)*(1422 +
    54*249**(1/2))**(1/3) - 3110400000*I*6**(1/3)*83**(1/2)*(1422 +
    54*249**(1/2))**(1/3) + 13478400000*I*3**(1/2)*6**(1/3)*(1422 +
    54*249**(1/2))**(1/3) + 1274950152*6**(2/3)*(1422 +
    54*249**(1/2))**(2/3) + 32347944*6**(2/3)*249**(1/2)*(1422 +
    54*249**(1/2))**(2/3) - 1758790152*I*3**(1/2)*6**(2/3)*(1422 +
    54*249**(1/2))**(2/3) - 304403832*I*6**(2/3)*83**(1/2)*(1422 +
    4*249**(1/2))**(2/3))/(175732658352 + (1106028 + 25596*249**(1/2) +
    76788*I*83**(1/2))**2)''')
    # 断言：对该表达式的数值求保留5位小数的数值字符串，期望结果是 '0.077284 + 1.1104*I'
    assert NS(v, 5) == '0.077284 + 1.1104*I'
    # 断言：对该表达式的数值求保留1位小数的数值字符串，期望结果是 '0.08 + 1.*I'
    assert NS(v, 1) == '0.08 + 1.*I'
def test_old_docstring():
    # 计算表达式 (E + pi*I)*(E - pi*I)
    a = (E + pi*I)*(E - pi*I)
    # 断言 NS(a) 的结果为 '17.2586605000200'
    assert NS(a) == '17.2586605000200'
    # 断言 a.n() 的结果为 17.25866050002001
    assert a.n() == 17.25866050002001


def test_issue_4806():
    # 断言对 atan(x)**2 在区间 (-1, 1) 上进行积分后，结果四舍五入到小数点后一位为 0.5
    assert integrate(atan(x)**2, (x, -1, 1)).evalf().round(1) == Float(0.5, 1)
    # 断言 atan(0) 的数值结果为 0
    assert atan(0, evaluate=False).n() == 0


def test_evalf_mul():
    # SymPy 应该通过 mpmath 逐项处理此乘积，而不是尝试展开它
    assert NS(product(1 + sqrt(n)*I, (n, 1, 500)), 1) == '5.e+567 + 2.e+568*I'


def test_scaled_zero():
    # 测试 scaled_zero 函数的不同参数组合返回的结果
    a, b = (([0], 1, 100, 1), -1)
    assert scaled_zero(100) == (a, b)
    assert scaled_zero(a) == (0, 1, 100, 1)
    a, b = (([1], 1, 100, 1), -1)
    assert scaled_zero(100, -1) == (a, b)
    assert scaled_zero(a) == (1, 1, 100, 1)
    # 测试 scaled_zero 函数在特定条件下是否引发 ValueError
    raises(ValueError, lambda: scaled_zero(scaled_zero(100)))
    raises(ValueError, lambda: scaled_zero(100, 2))
    raises(ValueError, lambda: scaled_zero(100, 0))
    raises(ValueError, lambda: scaled_zero((1, 5, 1, 3)))


def test_chop_value():
    # 对于范围从 -27 到 27 的整数，检查 Pow(10, i)*2 是否在精度为 10**i 时返回真，Pow(10, i) 在相同精度时返回假
    for i in range(-27, 28):
        assert (Pow(10, i)*2).n(chop=10**i) and not (Pow(10, i)).n(chop=10**i)


def test_infinities():
    # 断言正无穷大的数值计算结果为正无穷
    assert oo.evalf(chop=True) == inf
    # 断言负无穷大的数值计算结果为负无穷
    assert (-oo).evalf(chop=True) == ninf


def test_to_mpmath():
    # 断言 sqrt(3) 转换为 mpmath 格式后的结果满足给定条件
    assert sqrt(3)._to_mpmath(20)._mpf_ == (0, int(908093), -19, 20)
    assert S(3.2)._to_mpmath(20)._mpf_ == (0, int(838861), -18, 20)


def test_issue_6632_evalf():
    # 计算表达式 (-100000*sqrt(2500000001) + 5000000001) 的数值，断言结果为 9.999999998e-11
    add = (-100000*sqrt(2500000001) + 5000000001)
    assert add.n() == 9.999999998e-11
    # 断言表达式 (add*add).n() 的数值结果为 9.999999996e-21
    assert (add*add).n() == 9.999999996e-21


def test_issue_4945():
    from sympy.abc import H
    # 断言 (H/0).evalf(subs={H:1}) 的结果为 zoo
    assert (H/0).evalf(subs={H:1}) == zoo


def test_evalf_integral():
    # 断言对于积分 Integral(sin(x), (x, -pi, pi + eps)).n(2) 的计算精度 _prec 为 10
    eps = Rational(1, 1000000)
    assert Integral(sin(x), (x, -pi, pi + eps)).n(2)._prec == 10


def test_issue_8821_highprec_from_str():
    s = str(pi.evalf(128))
    # 断言经过高精度计算后，sin(p) 的绝对值小于 1e-15
    p = N(s)
    assert Abs(sin(p)) < 1e-15
    # 断言经过指定精度计算后，sin(p) 的绝对值小于 1e-64
    p = N(s, 64)
    assert Abs(sin(p)) < 1e-64


def test_issue_8853():
    p = Symbol('x', even=True, positive=True)
    # 断言 floor(-p - S.Half).is_even 为 False
    assert floor(-p - S.Half).is_even == False
    # 断言 floor(-p + S.Half).is_even 为 True
    assert floor(-p + S.Half).is_even == True
    # 断言 ceiling(p - S.Half).is_even 为 True
    assert ceiling(p - S.Half).is_even == True
    # 断言 ceiling(p + S.Half).is_even 为 False
    assert ceiling(p + S.Half).is_even == False

    # 断言 get_integer_part(S.Half, -1, {}, True) 的结果为 (0, 0)
    assert get_integer_part(S.Half, -1, {}, True) == (0, 0)
    # 断言 get_integer_part(S.Half, 1, {}, True) 的结果为 (1, 0)
    assert get_integer_part(S.Half, 1, {}, True) == (1, 0)
    # 断言 get_integer_part(Rational(-1, 2), -1, {}, True) 的结果为 (-1, 0)
    assert get_integer_part(Rational(-1, 2), -1, {}, True) == (-1, 0)
    # 断言 get_integer_part(Rational(-1, 2), 1, {}, True) 的结果为 (0, 0)
    assert get_integer_part(Rational(-1, 2), 1, {}, True) == (0, 0)


def test_issue_17681():
    class identity_func(Function):
        # 重载 _eval_evalf 方法，以便返回参数的数值评估结果
        def _eval_evalf(self, *args, **kwargs):
            return self.args[0].evalf(*args, **kwargs)

    # 断言对于 identity_func(S(0))，floor 的计算结果为 0
    assert floor(identity_func(S(0))) == 0
    # 断言 get_integer_part(S(0), 1, {}, True) 的结果为 (0, 0)
    assert get_integer_part(S(0), 1, {}, True) == (0, 0)


def test_issue_9326():
    from sympy.core.symbol import Dummy
    d1 = Dummy('d')
    d2 = Dummy('d')
    e = d1 + d2
    # 断言对于 e.evalf(subs={d1: 1, d2: 2}) 的结果为 3.0
    assert e.evalf(subs={d1: 1, d2: 2}) == 3.0


def test_issue_10323():
    # 这个测试函数尚未实现
    pass
    # 断言：确保 sqrt(2**30 + 1) 的上整数值等于 2**15 + 1
    assert ceiling(sqrt(2**30 + 1)) == 2**15 + 1
# 测试关联操作函数的函数
def test_AssocOp_Function():
    # 如果Min的第一个参数在虚部不可比较，则引发值错误异常
    raises(ValueError, lambda: S('''
    Min(-sqrt(3)*cos(pi/18)/6 + re(1/((-1/2 - sqrt(3)*I/2)*(1/6 +
    sqrt(3)*I/18)**(1/3)))/3 + sin(pi/18)/2 + 2 + I*(-cos(pi/18)/2 -
    sqrt(3)*sin(pi/18)/6 + im(1/((-1/2 - sqrt(3)*I/2)*(1/6 +
    sqrt(3)*I/18)**(1/3)))/3), re(1/((-1/2 + sqrt(3)*I/2)*(1/6 +
    sqrt(3)*I/18)**(1/3)))/3 - sqrt(3)*cos(pi/18)/6 - sin(pi/18)/2 + 2 +
    I*(im(1/((-1/2 + sqrt(3)*I/2)*(1/6 + sqrt(3)*I/18)**(1/3)))/3 -
    sqrt(3)*sin(pi/18)/6 + cos(pi/18)/2))'''))
    
    # 如果更改导致存在不可比较的参数，则需要更改Min/Max实例化代码以在简化时注意不可比较的参数
    # 下面的测试应该添加进去（其中e是上面符号化的表达式）：
    # raises(ValueError, lambda: e._eval_evalf(2))


def test_issue_10395():
    eq = x*Max(0, y)
    assert nfloat(eq) == eq
    eq = x*Max(y, -1.1)
    assert nfloat(eq) == eq
    assert Max(y, 4).n() == Max(4.0, y)


def test_issue_13098():
    assert floor(log(S('9.'+'9'*20), 10)) == 0
    assert ceiling(log(S('9.'+'9'*20), 10)) == 1
    assert floor(log(20 - S('9.'+'9'*20), 10)) == 1
    assert ceiling(log(20 - S('9.'+'9'*20), 10)) == 2


def test_issue_14601():
    e = 5*x*y/2 - y*(35*(x**3)/2 - 15*x/2)
    subst = {x:0.0, y:0.0}
    e2 = e.evalf(subs=subst)
    assert float(e2) == 0.0
    assert float((x + x*(x**2 + x)).evalf(subs={x: 0.0})) == 0.0


def test_issue_11151():
    z = S.Zero
    e = Sum(z, (x, 1, 2))
    assert e != z  # 它不应该求值
    # 当它确实求值时，应该给出以下结果
    assert evalf(e, 15, {}) == \
        evalf(z, 15, {}) == (None, None, 15, None)
    # 因此，这不应该失败
    assert (e/2).n() == 0
    # 这是问题出现的地方
    expr0 = Sum(x**2 + x, (x, 1, 2))
    expr1 = Sum(0, (x, 1, 2))
    expr2 = expr1/expr0
    assert simplify(factor(expr2) - expr2) == 0


def test_issue_13425():
    assert N('2**.5', 30) == N('sqrt(2)', 30)
    assert N('x - x', 30) == 0
    assert abs((N('pi*.1', 22)*10 - pi).n()) < 1e-22


def test_issue_17421():
    assert N(acos(-I + acosh(cosh(cosh(1) + I)))) == 1.0*I


def test_issue_20291():
    from sympy.sets import EmptySet, Reals
    from sympy.sets.sets import (Complement, FiniteSet, Intersection)
    a = Symbol('a')
    b = Symbol('b')
    A = FiniteSet(a, b)
    assert A.evalf(subs={a: 1, b: 2}) == FiniteSet(1.0, 2.0)
    B = FiniteSet(a-b, 1)
    assert B.evalf(subs={a: 1, b: 2}) == FiniteSet(-1.0, 1.0)

    sol = Complement(Intersection(FiniteSet(-b/2 - sqrt(b**2-4*pi)/2), Reals), FiniteSet(0))
    assert sol.evalf(subs={b: 1}) == EmptySet


def test_evalf_with_zoo():
    assert (1/x).evalf(subs={x: 0}) == zoo  # issue 8242
    assert (-1/x).evalf(subs={x: 0}) == zoo  # PR 16150
    assert (0 ** x).evalf(subs={x: -1}) == zoo  # PR 16150
    # 断言：对于复数域中的运算，0 的 x 次方，带入 x = -1 + i 后结果应为 NaN
    assert (0 ** x).evalf(subs={x: -1 + I}) == nan
    
    # 断言：处理数学表达式 2 * 0^(-1)，这里的 0^(-1) 使用 evaluate=False 表示不进行求值，结果应为无穷大 (zoo)
    assert Mul(2, Pow(0, -1, evaluate=False), evaluate=False).evalf() == zoo  # issue 21147
    
    # 断言：对于数学表达式 x * 1/x，在 x = 0 处计算结果应为 NaN
    assert Mul(x, 1/x, evaluate=False).evalf(subs={x: 0}) == Mul(x, 1/x, evaluate=False).subs(x, 0) == nan
    
    # 断言：对于数学表达式 1/x * 1/x，在 x = 0 处计算结果应为无穷大 (zoo)
    assert Mul(1/x, 1/x, evaluate=False).evalf(subs={x: 0}) == zoo
    
    # 断言：对于数学表达式 1/x * |1/x|，在 x = 0 处计算结果应为无穷大 (zoo)
    assert Mul(1/x, Abs(1/x), evaluate=False).evalf(subs={x: 0}) == zoo
    
    # 断言：对于绝对值函数 Abs(zoo)，在 zoo 处计算结果应为正无穷大 (oo)
    assert Abs(zoo, evaluate=False).evalf() == oo
    
    # 断言：对于实部函数 re(zoo)，在 zoo 处计算结果应为 NaN
    assert re(zoo, evaluate=False).evalf() == nan
    
    # 断言：对于虚部函数 im(zoo)，在 zoo 处计算结果应为 NaN
    assert im(zoo, evaluate=False).evalf() == nan
    
    # 断言：对于加法函数 Add(zoo, zoo)，在 zoo 处计算结果应为 NaN
    assert Add(zoo, zoo, evaluate=False).evalf() == nan
    
    # 断言：对于加法函数 Add(oo, zoo)，在 oo 和 zoo 相加后计算结果应为 NaN
    assert Add(oo, zoo, evaluate=False).evalf() == nan
    
    # 断言：对于幂函数 Pow(zoo, -1)，在 zoo 处计算结果应为 0
    assert Pow(zoo, -1, evaluate=False).evalf() == 0
    
    # 断言：对于幂函数 Pow(zoo, -1/3)，在 zoo 处计算结果应为 0
    assert Pow(zoo, Rational(-1, 3), evaluate=False).evalf() == 0
    
    # 断言：对于幂函数 Pow(zoo, 1/3)，在 zoo 处计算结果应为无穷大 (zoo)
    assert Pow(zoo, Rational(1, 3), evaluate=False).evalf() == zoo
    
    # 断言：对于幂函数 Pow(zoo, 1/2)，在 zoo 处计算结果应为无穷大 (zoo)
    assert Pow(zoo, S.Half, evaluate=False).evalf() == zoo
    
    # 断言：对于幂函数 Pow(zoo, 2)，在 zoo 处计算结果应为无穷大 (zoo)
    assert Pow(zoo, 2, evaluate=False).evalf() == zoo
    
    # 断言：对于幂函数 Pow(0, zoo)，在 0 的 zoo 次方处计算结果应为 NaN
    assert Pow(0, zoo, evaluate=False).evalf() == nan
    
    # 断言：对于对数函数 log(zoo)，在 zoo 处计算结果应为无穷大 (zoo)
    assert log(zoo, evaluate=False).evalf() == zoo
    
    # 断言：在启用截断模式下，zoo 的 evalf() 结果应为无穷大 (zoo)
    assert zoo.evalf(chop=True) == zoo
    
    # 断言：在 x = zoo 处计算 x 的 evalf() 结果应为无穷大 (zoo)
    assert x.evalf(subs={x: zoo}) == zoo
def test_evalf_with_bounded_error():
    cases = [
        # zero
        (Rational(0), None, 1),
        # zero im part
        (pi, None, 10),
        # zero real part
        (pi*I, None, 10),
        # re and im nonzero
        (2-3*I, None, 5),
        # similar tests again, but using eps instead of m
        (Rational(0), Rational(1, 2), None),
        (pi, Rational(1, 1000), None),
        (pi * I, Rational(1, 1000), None),
        (2 - 3 * I, Rational(1, 1000), None),
        # very large eps
        (2 - 3 * I, Rational(1000), None),
        # case where x already small, hence some cancellation in p = m + n - 1
        (Rational(1234, 10**8), Rational(1, 10**12), None),
    ]
    for x0, eps, m in cases:
        a, b, _, _ = evalf(x0, 53, {})  # Evaluate x0 with 53-bit precision
        c, d, _, _ = _evalf_with_bounded_error(x0, eps, m)  # Evaluate x0 with error bounded by eps and m
        if eps is None:
            eps = 2**(-m)  # Set eps to 2^(-m) if eps is None
        z = make_mpc((a or fzero, b or fzero))  # Create complex number z
        w = make_mpc((c or fzero, d or fzero))  # Create complex number w
        assert abs(w - z) < eps  # Assert that the absolute difference between w and z is less than eps

    # eps must be positive
    raises(ValueError, lambda: _evalf_with_bounded_error(pi, Rational(0)))  # Check ValueError for zero eps
    raises(ValueError, lambda: _evalf_with_bounded_error(pi, -pi))  # Check ValueError for negative eps
    raises(ValueError, lambda: _evalf_with_bounded_error(pi, I))  # Check ValueError for complex eps


def test_issue_22849():
    a = -8 + 3 * sqrt(3)
    x = AlgebraicNumber(a)
    assert evalf(a, 1, {}) == evalf(x, 1, {})  # Evaluate a and x with 1-bit precision


def test_evalf_real_alg_num():
    # This test demonstrates why the entry for `AlgebraicNumber` in
    # `sympy.core.evalf._create_evalf_table()` has to use `x.to_root()`,
    # instead of `x.as_expr()`. If the latter is used, then `z` will be
    # a complex number with `0.e-20` for imaginary part, even though `a5`
    # is a real number.
    zeta = Symbol('zeta')
    a5 = AlgebraicNumber(CRootOf(cyclotomic_poly(5), -1), [-1, -1, 0, 0], alias=zeta)
    z = a5.evalf()
    assert isinstance(z, Float)  # Assert that z is an instance of Float
    assert not hasattr(z, '_mpc_')  # Assert that z does not have _mpc_ attribute
    assert hasattr(z, '_mpf_')  # Assert that z has _mpf_ attribute


def test_issue_20733():
    expr = 1/((x - 9)*(x - 8)*(x - 7)*(x - 4)**2*(x - 3)**3*(x - 2))
    assert str(expr.evalf(1, subs={x:1})) == '-4.e-5'  # Assert evaluation of expr with 1-digit precision and substitution
    assert str(expr.evalf(2, subs={x:1})) == '-4.1e-5'  # Assert evaluation of expr with 2-digit precision and substitution
    assert str(expr.evalf(11, subs={x:1})) == '-4.1335978836e-5'  # Assert evaluation of expr with 11-digit precision and substitution
    assert str(expr.evalf(20, subs={x:1})) == '-0.000041335978835978835979'  # Assert evaluation of expr with 20-digit precision and substitution

    expr = Mul(*((x - i) for i in range(2, 1000)))
    assert srepr(expr.evalf(2, subs={x: 1})) == "Float('4.0271e+2561', precision=10)"  # Assert representation of expr evaluation with 2-digit precision and substitution
    assert srepr(expr.evalf(10, subs={x: 1})) == "Float('4.02790050126e+2561', precision=37)"  # Assert representation of expr evaluation with 10-digit precision and substitution
    assert srepr(expr.evalf(53, subs={x: 1})) == "Float('4.0279005012722099453824067459760158730668154575647110393e+2561', precision=179)"  # Assert representation of expr evaluation with 53-digit precision and substitution
```